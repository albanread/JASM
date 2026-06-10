//! Crash dumper for JITed code.
//!
//! Installs a process-wide Vectored Exception Handler (VEH) that runs
//! first for any exception — access violations, illegal instructions,
//! divide-by-zero, AND `int 3` breakpoints. On entry the handler dumps
//! the exception kind, the full register state, and the top of the
//! stack, with symbolic resolution against a (name, addr) table we
//! populate from the JIT.
//!
//! ## No OS unwind info — and why that's the right call
//!
//! We do NOT install `.pdata` / `.xdata` via `RtlAddFunctionTable`, and we never
//! will for the kernel: in this subroutine-threaded Forth, **RSP *is* the Forth
//! return stack** (it holds real `call` return addresses interleaved with `>r`'d
//! values and loop cells), primitives are bare no-prologue leaves, and RBP is
//! the data-stack pointer, not a frame pointer. The Win64 unwind contract
//! (`RtlLookupFunctionEntry` → `UNWIND_INFO` → reconstruct caller) cannot be
//! satisfied, and a blanket leaf `RUNTIME_FUNCTION` would make the OS unwinder
//! produce confidently-wrong frames. See `docs/design/rasm-replace-llvm.md`.
//!
//! ## Forth-centric dump instead
//!
//! STC turns that liability into an asset: a stack qword that points into a
//! registered code range (see [`register_code_range`]) *is* a return address, so
//! a Forth backtrace is a linear classify-and-filter scan — no metadata needed.
//! [`format_forth_dump`] renders, in order: the faulting word, the **data stack**
//! (SP), the **return-stack word trace** (RP) with data cells flagged, the key
//! **user vars**, and only then the raw CPU registers. The host wires the layout
//! via [`set_forth_dump_info`]. Every memory read is page-guarded
//! ([`read_qword`]), so a corrupt SP/DSP/UP degrades to a note instead of
//! faulting the handler. The same renderer backs the GUI's deferred crash view.
//!
//! ## `int 3` is non-fatal
//!
//! `STATUS_BREAKPOINT` (0x80000003) is decoded specially: dump state,
//! advance RIP past the 1-byte `0xCC`, return
//! `EXCEPTION_CONTINUE_EXECUTION`. So a `brk()` macro that emits
//! `int 3` becomes a "dump state and keep going" inspector — drop one
//! in a Forth primitive, run, see registers + stack at that point,
//! keep running.
//!
//! All other exceptions: dump and return `CONTINUE_SEARCH` so a
//! debugger or the default handler also see the failure.
//!
//! Platform: Windows only.

#![cfg(windows)]
#![allow(non_snake_case, non_camel_case_types)]

use std::ffi::c_void;
use std::sync::{OnceLock, RwLock};

// ── Minimal Win32 FFI ────────────────────────────────────────────────

/// x64 `CONTEXT` from `winnt.h`. Only the fields up through `Rip` are
/// declared — Windows writes the rest (XMM, vector state), we don't
/// inspect them. We never construct one; Windows hands us a `*mut`.
///
/// Layout offsets (the part that matters):
///
/// ```text
///   0x000  P1Home..P6Home      6 * u64
///   0x030  ContextFlags        u32
///   0x034  MxCsr               u32
///   0x038  SegCs..SegSs        6 * u16
///   0x044  EFlags              u32
///   0x048  Dr0..Dr3, Dr6, Dr7  6 * u64   ← six debug regs total
///   0x078  Rax                 u64
///   ...
///   0x0F8  Rip                 u64
/// ```
///
/// AMD64 has no Dr4/Dr5 — they alias Dr6/Dr7 on x86 and are absent
/// on x64. The total debug-register block is six qwords, NOT eight.
/// (An earlier version of this struct added a phantom `[u64; 2]`
/// after the array and corrupted every subsequent offset by 16 bytes
/// — RIP modifications landed in the XMM save area, and `int 3`
/// could not be advanced past, producing an infinite handler loop.)
#[repr(C)]
struct CONTEXT {
    _p_home: [u64; 6],
    ContextFlags: u32,
    MxCsr: u32,
    SegCs: u16,
    SegDs: u16,
    SegEs: u16,
    SegFs: u16,
    SegGs: u16,
    SegSs: u16,
    EFlags: u32,
    _dr: [u64; 6],
    Rax: u64,
    Rcx: u64,
    Rdx: u64,
    Rbx: u64,
    Rsp: u64,
    Rbp: u64,
    Rsi: u64,
    Rdi: u64,
    R8: u64,
    R9: u64,
    R10: u64,
    R11: u64,
    R12: u64,
    R13: u64,
    R14: u64,
    R15: u64,
    Rip: u64,
    // (tail omitted — XMM save area, vector state, etc.)
}

// Compile-time check: Rip must be at offset 0xF8. The `int 3` advance writes
// `(*ctx).Rip += 1`, so a wrong offset corrupts an unrelated field (a past bug
// landed it in the XMM save area → infinite handler loop). `offset_of!` makes
// the check exact instead of the old size-only guard.
const _: () = {
    assert!(std::mem::size_of::<CONTEXT>() >= 0x100);
    assert!(core::mem::offset_of!(CONTEXT, Rip) == 0xF8);
};

#[repr(C)]
struct EXCEPTION_RECORD {
    ExceptionCode: u32,
    ExceptionFlags: u32,
    ExceptionRecord: *mut EXCEPTION_RECORD,
    ExceptionAddress: *mut c_void,
    NumberParameters: u32,
    _pad: u32,
    ExceptionInformation: [usize; 15],
}

#[repr(C)]
struct EXCEPTION_POINTERS {
    ExceptionRecord: *mut EXCEPTION_RECORD,
    ContextRecord: *mut CONTEXT,
}

const EXCEPTION_CONTINUE_SEARCH: i32 = 0;
const EXCEPTION_CONTINUE_EXECUTION: i32 = -1;

const STATUS_ACCESS_VIOLATION: u32 = 0xC000_0005;
const STATUS_BREAKPOINT: u32 = 0x8000_0003;
const STATUS_SINGLE_STEP: u32 = 0x8000_0004;
const STATUS_ILLEGAL_INSTRUCTION: u32 = 0xC000_001D;
const STATUS_PRIVILEGED_INSTRUCTION: u32 = 0xC000_0096;
const STATUS_STACK_OVERFLOW: u32 = 0xC000_00FD;
const STATUS_INTEGER_DIVIDE_BY_ZERO: u32 = 0xC000_0094;
const STATUS_INTEGER_OVERFLOW: u32 = 0xC000_0095;

type PVECTORED_EXCEPTION_HANDLER =
    unsafe extern "system" fn(*mut EXCEPTION_POINTERS) -> i32;

/// x64 `MEMORY_BASIC_INFORMATION` (winnt.h). Only used to page-check an address
/// before dereferencing it in the dumper, so a corrupt SP/DSP can't fault the
/// handler recursively (a VEH can't catch its own nested exception).
#[repr(C)]
struct MEMORY_BASIC_INFORMATION {
    BaseAddress: *mut c_void,
    AllocationBase: *mut c_void,
    AllocationProtect: u32,
    __alignment1: u32,
    RegionSize: usize,
    State: u32,
    Protect: u32,
    Type: u32,
    __alignment2: u32,
}

const MEM_COMMIT: u32 = 0x1000;
const PAGE_GUARD: u32 = 0x100;
/// Protections that permit a read (READONLY|READWRITE|WRITECOPY and their
/// EXECUTE_* variants). Execute-only (0x10) and NOACCESS (0x01) do not.
const PAGE_READABLE: u32 = 0x02 | 0x04 | 0x08 | 0x20 | 0x40 | 0x80;

#[link(name = "kernel32")]
extern "system" {
    fn AddVectoredExceptionHandler(First: u32, Handler: PVECTORED_EXCEPTION_HANDLER)
        -> *mut c_void;
    fn VirtualQuery(
        lpAddress: *const c_void,
        lpBuffer: *mut MEMORY_BASIC_INFORMATION,
        dwLength: usize,
    ) -> usize;
}

/// True iff the page containing `addr` is committed and readable.
unsafe fn page_readable(addr: u64) -> bool {
    let mut mbi: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    let n = unsafe {
        VirtualQuery(
            addr as *const c_void,
            &mut mbi,
            std::mem::size_of::<MEMORY_BASIC_INFORMATION>(),
        )
    };
    if n == 0 {
        return false;
    }
    mbi.State == MEM_COMMIT && (mbi.Protect & PAGE_GUARD) == 0 && (mbi.Protect & PAGE_READABLE) != 0
}

/// Read one 8-byte aligned qword, returning `None` if its page isn't readable.
/// An 8-aligned qword never straddles a page boundary, so a single page check
/// is sufficient. Used everywhere the dumper walks a (possibly corrupt) stack.
unsafe fn read_qword(addr: u64) -> Option<u64> {
    if addr & 7 != 0 || !unsafe { page_readable(addr) } {
        return None;
    }
    Some(unsafe { *(addr as *const u64) })
}

// ── Symbol table ─────────────────────────────────────────────────────

/// One registered symbol — start address plus a name. We resolve by
/// "nearest predecessor": for a queried address, find the highest
/// `addr` that is `<= query`, print `name + (query - addr)`. No
/// per-symbol size info is needed for that.
#[derive(Clone)]
struct Symbol {
    addr: u64,
    name: String,
    source: &'static str,
}

struct SymbolTable {
    /// Sorted by `addr` ascending.
    syms: Vec<Symbol>,
    /// True after the first sort — subsequent inserts re-sort lazily.
    dirty: bool,
}

impl SymbolTable {
    fn new() -> Self {
        SymbolTable {
            syms: Vec::new(),
            dirty: false,
        }
    }

    fn add(&mut self, name: String, addr: u64, source: &'static str) {
        self.syms.push(Symbol { addr, name, source });
        self.dirty = true;
    }

    fn ensure_sorted(&mut self) {
        if self.dirty {
            self.syms.sort_by_key(|s| s.addr);
            self.dirty = false;
        }
    }

    /// Nearest predecessor with NO plausibility cap, plus the successor's
    /// address (for tight offset bounding). Used when the caller already knows
    /// `query` is inside a registered code range, so the attribution is sound
    /// no matter how large the offset (a kernel packed with small procs makes
    /// the offset small anyway).
    fn resolve_nearest(&self, query: u64) -> Option<(&Symbol, u64, Option<u64>)> {
        let idx = self.syms.partition_point(|s| s.addr <= query);
        if idx == 0 {
            return None;
        }
        let sym = &self.syms[idx - 1];
        let next = self.syms.get(idx).map(|s| s.addr);
        Some((sym, query - sym.addr, next))
    }

    /// Find the symbol whose `addr` is the highest value `<= query`.
    /// Returns `None` if no such symbol exists or the candidate is
    /// implausibly far away.
    fn resolve(&self, query: u64) -> Option<(&Symbol, u64)> {
        let (sym, offset, next) = self.resolve_nearest(query)?;
        // Reject implausibly large offsets. With "nearest predecessor" and no
        // per-symbol size info, anything well past the typical function size is
        // more likely random stack data that happens to be numerically near a
        // code address. Bound by the successor symbol (tight in a packed kernel)
        // and a 64 KiB ceiling (covers the biggest Win32 routines, ~30 KB).
        let ceiling = next.map(|n| (n - sym.addr).min(64 * 1024)).unwrap_or(64 * 1024);
        if offset >= ceiling {
            return None;
        }
        Some((sym, offset))
    }
}

// ── Code ranges + Forth dump descriptor ──────────────────────────────
//
// The host (WF64) registers the JIT code extents and the Forth runtime layout
// so the dumper can be FORTH-CENTRIC: classify return-stack qwords as Forth
// word addresses vs data, and render the data stack + key user vars before the
// raw CPU state. wfasm is WF64's assembler, so the register convention is baked
// in here (documented per field) rather than abstracted away.

struct CodeRange {
    start: u64,
    end: u64,
    #[allow(dead_code)]
    kind: &'static str,
}

/// One user-area variable to surface in the crash dump (name + byte offset
/// from the user-area base / `UP`).
#[derive(Clone, Copy)]
pub struct ForthVar {
    pub name: &'static str,
    pub offset: u64,
}

/// Everything the dumper needs to render the Forth view. Registered (and
/// overwritten) by the host at each session boot via [`set_forth_dump_info`].
pub struct ForthDumpInfo {
    /// Expected value of `UP` (RBX) while Forth runs; user vars are read
    /// relative to it and only shown when RBX matches.
    pub user_base: u64,
    /// Data stack (DSP = RBP) grows DOWN toward `dstack_top`; live cells are
    /// `[RBP, dstack_top)` with TOS cached in RAX.
    pub dstack_low: u64,
    pub dstack_top: u64,
    /// Return stack (RP = RSP) grows DOWN toward `rstack_top`; live region is
    /// `[RSP, rstack_top)`.
    pub rstack_low: u64,
    pub rstack_top: u64,
    /// Key user vars to print (HERE, LATEST, STATE, …).
    pub vars: Vec<ForthVar>,
}

/// A snapshot of the integer register file at the fault, decoupled from the OS
/// `CONTEXT` so both the inline stderr dumper and the GUI's deferred
/// supervisor-thread formatter can share one renderer ([`format_forth_dump`]).
#[derive(Clone, Copy, Default)]
pub struct CrashRegs {
    pub rax: u64,
    pub rbx: u64,
    pub rcx: u64,
    pub rdx: u64,
    pub rsi: u64,
    pub rdi: u64,
    pub rbp: u64,
    pub rsp: u64,
    pub r8: u64,
    pub r9: u64,
    pub r10: u64,
    pub r11: u64,
    pub r12: u64,
    pub r13: u64,
    pub r14: u64,
    pub r15: u64,
    pub rip: u64,
    pub eflags: u32,
}

static CODE_RANGES: OnceLock<RwLock<Vec<CodeRange>>> = OnceLock::new();
static FORTH_INFO: OnceLock<RwLock<Option<ForthDumpInfo>>> = OnceLock::new();

fn code_ranges() -> &'static RwLock<Vec<CodeRange>> {
    CODE_RANGES.get_or_init(|| RwLock::new(Vec::new()))
}
fn forth_info() -> &'static RwLock<Option<ForthDumpInfo>> {
    FORTH_INFO.get_or_init(|| RwLock::new(None))
}

static SYMBOLS: OnceLock<RwLock<SymbolTable>> = OnceLock::new();
static INSTALLED: OnceLock<()> = OnceLock::new();

fn symbols() -> &'static RwLock<SymbolTable> {
    SYMBOLS.get_or_init(|| RwLock::new(SymbolTable::new()))
}

// ── Public API ───────────────────────────────────────────────────────

/// Install the Vectored Exception Handler. Idempotent — multiple
/// calls are no-ops. Returns `Err` if the OS refuses (extremely rare).
pub fn install() -> Result<(), &'static str> {
    if INSTALLED.get().is_some() {
        return Ok(());
    }
    unsafe {
        let h = AddVectoredExceptionHandler(1, veh_handler);
        if h.is_null() {
            return Err("AddVectoredExceptionHandler returned null");
        }
    }
    let _ = INSTALLED.set(());
    Ok(())
}

/// True if [`install`] has been called.
pub fn is_installed() -> bool {
    INSTALLED.get().is_some()
}

/// Register a (name, address) pair for symbolic crash dumps. Cheap;
/// the symbol table is sorted lazily on first lookup.
///
/// `source` is a short string for the dump's source column —
/// `"win32"`, `"jit_proc"`, `"rust_extern"`, `"user"`, etc.
pub fn register(name: impl Into<String>, addr: u64, source: &'static str) {
    if let Ok(mut t) = symbols().write() {
        t.add(name.into(), addr, source);
    }
}

/// Bulk-register from any iterator of `(name, addr, source)`.
pub fn register_many<I, N>(items: I)
where
    I: IntoIterator<Item = (N, u64, &'static str)>,
    N: Into<String>,
{
    if let Ok(mut t) = symbols().write() {
        for (name, addr, source) in items {
            t.add(name.into(), addr, source);
        }
    }
}

/// Number of registered symbols. Useful for diagnostic prints.
pub fn symbol_count() -> usize {
    symbols().read().map(|t| t.syms.len()).unwrap_or(0)
}

/// Register a JIT code extent `[start, end)`. The dumper treats a stack qword
/// pointing inside any registered range as a **return address** (a Forth frame)
/// rather than data, and resolves its symbol without the off-range plausibility
/// cap. `kind` is a short tag (`"kernel"`, `"user"`).
pub fn register_code_range(start: u64, end: u64, kind: &'static str) {
    if end > start {
        if let Ok(mut r) = code_ranges().write() {
            r.push(CodeRange { start, end, kind });
        }
    }
}

/// Drop all registered code ranges. The host calls this at each session boot
/// before re-registering, so a rebooted session's stale (freed) ranges don't
/// linger and mis-classify stack data as code.
pub fn clear_code_ranges() {
    if let Ok(mut r) = code_ranges().write() {
        r.clear();
    }
}

/// True iff `addr` lies inside a registered code range — i.e. it is plausibly a
/// return address into JIT'd Forth code, not stack data.
pub fn is_code(addr: u64) -> bool {
    code_ranges()
        .read()
        .map(|r| r.iter().any(|c| addr >= c.start && addr < c.end))
        .unwrap_or(false)
}

/// Symbolicate an address: `"<name+0xNN> [source]"`, or `None` if nothing is
/// near. In-code-range addresses resolve without the plausibility cap; others
/// (Win32 externs etc.) use the capped nearest-predecessor.
pub fn symbolize(addr: u64) -> Option<String> {
    // Self-sort: the nearest-predecessor search needs a sorted table, and
    // callers (GUI supervisor, direct lookups) may arrive before any sort.
    // `ensure_sorted` is a no-op once clean, so the per-qword cost is trivial.
    let mut t = symbols().write().ok()?;
    t.ensure_sorted();
    let hit = if is_code(addr) {
        t.resolve_nearest(addr).map(|(s, off, _)| (s, off))
    } else {
        t.resolve(addr)
    };
    let (sym, off) = hit?;
    Some(if off == 0 {
        format!("<{}> [{}]", sym.name, sym.source)
    } else {
        format!("<{}+0x{:x}> [{}]", sym.name, off, sym.source)
    })
}

/// Register (overwrite) the Forth runtime layout so the dumper can render the
/// data stack, return-stack word trace, and key user vars. Called at each
/// session boot; the latest wins.
pub fn set_forth_dump_info(info: ForthDumpInfo) {
    if let Ok(mut g) = forth_info().write() {
        *g = Some(info);
    }
}

// ── VEH handler ──────────────────────────────────────────────────────

unsafe extern "system" fn veh_handler(info: *mut EXCEPTION_POINTERS) -> i32 {
    if info.is_null() {
        return EXCEPTION_CONTINUE_SEARCH;
    }
    let rec = (*info).ExceptionRecord;
    let ctx = (*info).ContextRecord;
    if rec.is_null() || ctx.is_null() {
        return EXCEPTION_CONTINUE_SEARCH;
    }

    let code = (*rec).ExceptionCode;

    // Single-step we ignore (debugger uses it; we don't want to
    // produce noise during single-stepping).
    if code == STATUS_SINGLE_STEP {
        return EXCEPTION_CONTINUE_SEARCH;
    }

    print_dump(rec, ctx);

    // `int 3` (and `int3` mnemonic) is a non-fatal inspect point.
    // Advance past it so execution continues.
    if code == STATUS_BREAKPOINT {
        (*ctx).Rip += 1;
        return EXCEPTION_CONTINUE_EXECUTION;
    }

    // Everything else: let the debugger / default handler also see it.
    EXCEPTION_CONTINUE_SEARCH
}

unsafe fn print_dump(rec: *mut EXCEPTION_RECORD, ctx: *mut CONTEXT) {
    let regs = regs_from_context(ctx);
    let access = access_info(rec);
    let dump = format_forth_dump(&regs, (*rec).ExceptionCode, access);
    eprint!("{dump}");
}

/// Pull the integer register file out of an OS `CONTEXT` into our portable
/// snapshot.
unsafe fn regs_from_context(ctx: *mut CONTEXT) -> CrashRegs {
    CrashRegs {
        rax: (*ctx).Rax, rbx: (*ctx).Rbx, rcx: (*ctx).Rcx, rdx: (*ctx).Rdx,
        rsi: (*ctx).Rsi, rdi: (*ctx).Rdi, rbp: (*ctx).Rbp, rsp: (*ctx).Rsp,
        r8: (*ctx).R8, r9: (*ctx).R9, r10: (*ctx).R10, r11: (*ctx).R11,
        r12: (*ctx).R12, r13: (*ctx).R13, r14: (*ctx).R14, r15: (*ctx).R15,
        rip: (*ctx).Rip, eflags: (*ctx).EFlags,
    }
}

/// `(access_kind, faulting_address)` for an access violation, else `None`.
unsafe fn access_info(rec: *mut EXCEPTION_RECORD) -> Option<(u32, u64)> {
    if (*rec).ExceptionCode == STATUS_ACCESS_VIOLATION && (*rec).NumberParameters >= 2 {
        Some(((*rec).ExceptionInformation[0] as u32, (*rec).ExceptionInformation[1] as u64))
    } else {
        None
    }
}

/// Render the crash as a FORTH-CENTRIC text block: faulting word, then the
/// **data stack** (SP), the **return-stack word trace** (RP), the key **user
/// vars**, and only then the raw CPU registers + stack. Shared by the inline
/// stderr dumper (here) and the GUI's deferred supervisor formatter, so both
/// views are identical. All memory reads are page-guarded, so a corrupt
/// SP/DSP/UP degrades to a note instead of faulting the handler.
pub fn format_forth_dump(regs: &CrashRegs, code: u32, access: Option<(u32, u64)>) -> String {
    use std::fmt::Write;
    let mut s = String::with_capacity(2048);
    let p = |s: &mut String, args: std::fmt::Arguments| {
        let _ = s.write_fmt(args);
        s.push('\n');
    };

    p(&mut s, format_args!("┌─── WF64 Forth crash dump ──────────────────────────────────────────────"));
    p(&mut s, format_args!("│ exception : {:#010X}  {}", code, exception_kind(code)));
    p(&mut s, format_args!(
        "│ at word   : {:016X}  {}",
        regs.rip,
        symbolize(regs.rip).unwrap_or_else(|| "<?>".into())
    ));
    if let Some((kind, addr)) = access {
        let op = match kind { 0 => "read", 1 => "write", 8 => "execute", _ => "?" };
        p(&mut s, format_args!("│ access    : {op} at {:016X}  {}", addr, symbolize(addr).unwrap_or_default()));
    }

    let info = forth_info().read().ok();
    let info = info.as_ref().and_then(|g| g.as_ref());

    if let Some(fi) = info {
        // ── DATA STACK (SP = RBP; TOS cached in RAX) ──
        p(&mut s, format_args!("│"));
        // `rbp` is the internal DSP: TOS is cached in RAX and the in-memory
        // cells (NOS downward) live at `[rbp, dstack_top)`. With cached TOS, an
        // EMPTY stack leaves rbp one cell ABOVE dstack_top, so the valid range
        // extends to `dstack_top + 8`.
        if regs.rbp >= fi.dstack_low && regs.rbp <= fi.dstack_top + 8 {
            let has_tos = regs.rbp <= fi.dstack_top;
            let mem_cells = if has_tos { ((fi.dstack_top - regs.rbp) / 8) as usize } else { 0 };
            let depth = mem_cells + has_tos as usize;
            p(&mut s, format_args!("│ DATA STACK  rbp={:016X}  depth={depth}", regs.rbp));
            if has_tos {
                p(&mut s, format_args!("│   TOS   {:016X}   (rax)", regs.rax));
            }
            let mut shown = 0usize;
            let mut addr = regs.rbp;
            while addr < fi.dstack_top && shown < 16 {
                match unsafe { read_qword(addr) } {
                    Some(v) => p(&mut s, format_args!("│   NOS+{:<2} {:016X}", shown, v)),
                    None => { p(&mut s, format_args!("│   <unreadable @ {:016X}>", addr)); break; }
                }
                shown += 1; addr += 8;
            }
            if mem_cells > shown { p(&mut s, format_args!("│   … {} more", mem_cells - shown)); }
        } else {
            p(&mut s, format_args!("│ DATA STACK  rbp={:016X}  (not in data-stack region)", regs.rbp));
        }

        // ── RETURN STACK word trace (RP = RSP) ──
        p(&mut s, format_args!("│"));
        if regs.rsp >= fi.rstack_low && regs.rsp <= fi.rstack_top {
            p(&mut s, format_args!("│ RETURN STACK  rsp={:016X}  top={:016X}", regs.rsp, fi.rstack_top));
            let mut frame = 0usize;
            let mut shown = 0usize;
            let mut addr = regs.rsp;
            while addr < fi.rstack_top && shown < 64 {
                match unsafe { read_qword(addr) } {
                    Some(v) if is_code(v) => {
                        p(&mut s, format_args!(
                            "│   #{:<2} {:016X}  {}   @ rp+{}",
                            frame, v, symbolize(v).unwrap_or_default(), shown * 8
                        ));
                        frame += 1;
                    }
                    Some(v) => p(&mut s, format_args!("│        {:016X}  (data)        @ rp+{}", v, shown * 8)),
                    None => { p(&mut s, format_args!("│   <unreadable @ {:016X}>", addr)); break; }
                }
                shown += 1; addr += 8;
            }
            if frame == 0 { p(&mut s, format_args!("│   (no Forth return addresses found)")); }
        } else {
            p(&mut s, format_args!("│ RETURN STACK  rsp={:016X}  (not in return-stack region)", regs.rsp));
        }

        // ── KEY USER VARS (relative to UP = RBX) ──
        p(&mut s, format_args!("│"));
        if regs.rbx == fi.user_base {
            p(&mut s, format_args!("│ USER VARS  up={:016X}", regs.rbx));
            for v in &fi.vars {
                match unsafe { read_qword(regs.rbx + v.offset) } {
                    Some(val) => p(&mut s, format_args!("│   {:<10} = {:016X}", v.name, val)),
                    None => p(&mut s, format_args!("│   {:<10} = <unreadable>", v.name)),
                }
            }
        } else {
            p(&mut s, format_args!("│ USER VARS  rbx={:016X}  (≠ user base {:016X}; skipped)", regs.rbx, fi.user_base));
        }
    }

    // ── CPU registers (after the Forth view) ──
    p(&mut s, format_args!("│"));
    p(&mut s, format_args!("│ CPU  rax={:016X} rbx={:016X} rcx={:016X}", regs.rax, regs.rbx, regs.rcx));
    p(&mut s, format_args!("│      rdx={:016X} rsi={:016X} rdi={:016X}", regs.rdx, regs.rsi, regs.rdi));
    p(&mut s, format_args!("│      rbp={:016X} rsp={:016X} rip={:016X}", regs.rbp, regs.rsp, regs.rip));
    p(&mut s, format_args!("│      r8 ={:016X} r9 ={:016X} r10={:016X}", regs.r8, regs.r9, regs.r10));
    p(&mut s, format_args!("│      r11={:016X} r12={:016X} r13={:016X}", regs.r11, regs.r12, regs.r13));
    p(&mut s, format_args!("│      r14={:016X} r15={:016X} flags={:08X}", regs.r14, regs.r15, regs.eflags));

    // ── Raw stack tail (guarded), when no Forth view classified it ──
    if info.is_none() {
        p(&mut s, format_args!("│"));
        p(&mut s, format_args!("│ raw stack (32 qwords from rsp):"));
        for i in 0..32u64 {
            let addr = regs.rsp.wrapping_add(i * 8);
            match unsafe { read_qword(addr) } {
                Some(v) => p(&mut s, format_args!("│   [rsp+{:>3}] {:016X}  {}", i * 8, v, symbolize(v).unwrap_or_default())),
                None => { p(&mut s, format_args!("│   [rsp+{:>3}] <unreadable>", i * 8)); break; }
            }
        }
    }
    p(&mut s, format_args!("└───────────────────────────────────────────────────────────────────────"));
    s
}

fn exception_kind(code: u32) -> &'static str {
    match code {
        STATUS_ACCESS_VIOLATION => "ACCESS_VIOLATION",
        STATUS_BREAKPOINT => "BREAKPOINT (int 3)",
        STATUS_ILLEGAL_INSTRUCTION => "ILLEGAL_INSTRUCTION",
        STATUS_PRIVILEGED_INSTRUCTION => "PRIVILEGED_INSTRUCTION",
        STATUS_STACK_OVERFLOW => "STACK_OVERFLOW",
        STATUS_INTEGER_DIVIDE_BY_ZERO => "INTEGER_DIVIDE_BY_ZERO",
        STATUS_INTEGER_OVERFLOW => "INTEGER_OVERFLOW",
        _ => "(see Windows status codes for hex)",
    }
}

// ── Integration helpers ──────────────────────────────────────────────

/// Register multiple JIT procs by name. Looks each up via
/// `Jit::lookup_addr` and stuffs the result into the dump symbol
/// table. Call once after the JIT is finalized; subsequent calls add
/// more entries.
///
/// For Win32 externs (whose addresses came from `GetProcAddress` via
/// `bind_externs`), the host has the (name, addr) pairs already —
/// feed them to `register_many` directly with source `"win32"`.
pub fn register_jit_procs(
    loader: &mut dyn crate::backend::Loader,
    names: &[&str],
) -> anyhow::Result<()> {
    let mut acc: Vec<(String, u64, &'static str)> = Vec::with_capacity(names.len());
    for &name in names {
        let addr = loader.lookup_addr(name)?;
        acc.push((name.to_string(), addr, "jit_proc"));
    }
    register_many(acc);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nearest_predecessor_resolution() {
        let mut t = SymbolTable::new();
        t.add("foo".into(), 0x1000, "test");
        t.add("bar".into(), 0x2000, "test");
        t.add("baz".into(), 0x3000, "test");
        t.ensure_sorted();

        // Exact hit on foo.
        let (s, off) = t.resolve(0x1000).unwrap();
        assert_eq!(s.name, "foo");
        assert_eq!(off, 0);

        // Mid-foo.
        let (s, off) = t.resolve(0x1500).unwrap();
        assert_eq!(s.name, "foo");
        assert_eq!(off, 0x500);

        // Right at bar.
        let (s, off) = t.resolve(0x2000).unwrap();
        assert_eq!(s.name, "bar");
        assert_eq!(off, 0);

        // Mid-baz.
        let (s, off) = t.resolve(0x3010).unwrap();
        assert_eq!(s.name, "baz");
        assert_eq!(off, 0x10);

        // Below all symbols.
        assert!(t.resolve(0x500).is_none());
    }

    #[test]
    fn resolve_rejects_implausible_distance() {
        let mut t = SymbolTable::new();
        t.add("foo".into(), 0x1000, "test");
        t.ensure_sorted();
        // 128 KiB away — past the 64 KiB plausibility window.
        assert!(t.resolve(0x1000 + 128 * 1024).is_none());
    }

    #[test]
    fn extent_bounds_offset_to_successor() {
        // Two close symbols: a query past `foo`'s extent (i.e. at/after `bar`'s
        // start region but attributed to foo) must be rejected — the successor
        // bounds the plausible offset tighter than the 64 KiB ceiling.
        let mut t = SymbolTable::new();
        t.add("foo".into(), 0x1000, "test");
        t.add("bar".into(), 0x1100, "test"); // 0x100 after foo
        t.ensure_sorted();
        // 0x80 into foo: within [foo, bar) → attributed to foo.
        let (s, off) = t.resolve(0x1080).unwrap();
        assert_eq!((s.name.as_str(), off), ("foo", 0x80));
        // resolve_nearest never rejects (used for in-code-range hits).
        let (s, off, next) = t.resolve_nearest(0x1FFF).unwrap();
        assert_eq!((s.name.as_str(), off, next), ("bar", 0xEFF, None));
    }

    #[test]
    fn code_range_membership_and_uncapped_symbolize() {
        // Use addresses far from any real registered symbol to avoid collisions
        // with the process-global tables.
        let base = 0x5000_0000_0000u64;
        register("__seh_rng_word", base, "test");
        register_code_range(base, base + 0x4000, "test");
        assert!(is_code(base + 0x10));
        assert!(!is_code(base + 0x4000)); // end is exclusive
        // Within range, symbolize attributes even at a large offset (uncapped).
        let s = symbolize(base + 0x3FF0).unwrap();
        assert!(s.contains("__seh_rng_word+0x3ff0"), "got {s}");
    }

    #[test]
    fn forth_dump_renders_sp_rp_trace_and_vars() {
        // Fake in-process buffers stand in for the data stack, return stack,
        // and user area — all heap memory is committed+readable, so the
        // page-guarded reads succeed. Addresses are far from any real symbol.
        let code_base = 0x6000_0000_0000u64;
        register("fake_wordA", code_base, "test");
        register("fake_wordB", code_base + 0x40, "test");
        register_code_range(code_base, code_base + 0x100, "test");

        let dstack: Vec<u64> = vec![0x1111, 0x2222, 0x3333, 0x4444];
        let dlo = dstack.as_ptr() as u64;
        let dtop = dlo + (dstack.len() as u64) * 8;

        // Return stack: a frame, a data cell, another frame.
        let rstack: Vec<u64> = vec![code_base + 0x10, 0xDEAD_BEEF, code_base + 0x50];
        let rlo = rstack.as_ptr() as u64;
        let rtop = rlo + (rstack.len() as u64) * 8;

        let uarea: Vec<u64> = vec![0xAA, 0xBB];
        let ubase = uarea.as_ptr() as u64;

        set_forth_dump_info(ForthDumpInfo {
            user_base: ubase,
            dstack_low: dlo,
            dstack_top: dtop,
            rstack_low: rlo,
            rstack_top: rtop,
            vars: vec![
                ForthVar { name: "V0", offset: 0 },
                ForthVar { name: "V1", offset: 8 },
            ],
        });

        let regs = CrashRegs {
            rax: 0x7705,
            rbx: ubase,
            rbp: dlo,
            rsp: rlo,
            rip: code_base + 0x20,
            ..Default::default()
        };
        let out = format_forth_dump(&regs, STATUS_ACCESS_VIOLATION, Some((1, 0xBAD)));

        // Faulting word symbolicated from RIP.
        assert!(out.contains("fake_wordA+0x20"), "rip sym missing:\n{out}");
        // Data stack: TOS in rax + the four cells.
        assert!(out.contains("DATA STACK"), "{out}");
        assert!(out.contains("7705") && out.contains("0000000000001111"), "{out}");
        // Return-stack trace: two code frames + one data cell.
        assert!(out.contains("#0") && out.contains("fake_wordA+0x10"), "{out}");
        assert!(out.contains("#1") && out.contains("fake_wordB+0x10"), "{out}");
        assert!(out.contains("(data)"), "data cell not flagged:\n{out}");
        // Key user vars (formatter emits uppercase hex).
        assert!(out.contains("V0") && out.contains("00000000000000AA"), "{out}");
        // CPU register block comes after the Forth view.
        let cpu = out.find("CPU").unwrap();
        let ds = out.find("DATA STACK").unwrap();
        assert!(ds < cpu, "Forth view must precede CPU registers");
    }

    #[test]
    fn symbol_count_tracks_registration() {
        // Each test runs in its own process state, but `SYMBOLS` is
        // process-global. We can only test relative changes.
        let before = symbol_count();
        register("__seh_test_alpha", 0xAABBCC00, "test");
        register("__seh_test_beta", 0xAABBCC10, "test");
        let after = symbol_count();
        assert!(after >= before + 2);
    }
}
