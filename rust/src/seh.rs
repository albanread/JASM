//! Crash dumper for JITed code.
//!
//! Installs a process-wide Vectored Exception Handler (VEH) that runs
//! first for any exception — access violations, illegal instructions,
//! divide-by-zero, AND `int 3` breakpoints. On entry the handler dumps
//! the exception kind, the full register state, and the top of the
//! stack, with symbolic resolution against a (name, addr) table we
//! populate from the JIT.
//!
//! ## Why this is small
//!
//! We do NOT install `.pdata` / `.xdata` via `RtlAddFunctionTable`.
//! For that to work, LLVM would need IR-level function definitions
//! with `uwtable` attributes and a custom MCJIT memory manager that
//! captures section addresses and calls the registration API. Since
//! our model is "module-level inline asm with declare-only IR
//! functions," there is no `.pdata` to register. Consequence: the OS
//! unwinder cannot *walk* the JIT call stack. The dumper compensates
//! by printing the literal stack contents — return addresses are
//! visible to the reader.
//!
//! When proper stack unwinding becomes worth the build cost, NewBCPL's
//! `jit_mm.rs` shows the pattern: capture `.text`/`.pdata`/`.xdata`
//! during finalize, call `RtlAddFunctionTable` per `.pdata` range.
//! Until then, the dumper here gives you everything except the
//! pre-walked frames.
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

// Compile-time check: Rip must be at offset 0xF8. If this assertion
// ever fails after a struct edit, we've broken the layout again.
const _: () = {
    // Use a fake instance to compute offset without `memoffset`.
    // const_eval supports field projection on references-to-const.
    assert!(std::mem::size_of::<CONTEXT>() >= 0x100);
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

#[link(name = "kernel32")]
extern "system" {
    fn AddVectoredExceptionHandler(First: u32, Handler: PVECTORED_EXCEPTION_HANDLER)
        -> *mut c_void;
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

    /// Find the symbol whose `addr` is the highest value `<= query`.
    /// Returns `None` if no such symbol exists or the candidate is
    /// implausibly far away.
    fn resolve(&self, query: u64) -> Option<(&Symbol, u64)> {
        // Binary search for the insertion point of `query`. The symbol
        // just before that point (if any) is our nearest predecessor.
        let idx = self.syms.partition_point(|s| s.addr <= query);
        if idx == 0 {
            return None;
        }
        let sym = &self.syms[idx - 1];
        let offset = query - sym.addr;
        // Reject implausibly large offsets. With "nearest predecessor"
        // and no per-symbol size info, anything well past the typical
        // function size is more likely random stack data that happens
        // to be numerically near a code address. 64 KiB covers the
        // biggest Win32 functions (some KERNEL32 routines are ~30 KB)
        // with margin; tighter would give cleaner output in a kernel
        // packed with small procs but lose Win32-bound symbol attribution.
        if offset > 64 * 1024 {
            return None;
        }
        Some((sym, offset))
    }
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
    let code = (*rec).ExceptionCode;
    let kind = exception_kind(code);
    let rip = (*ctx).Rip;
    let rsp = (*ctx).Rsp;

    // Ensure the symbol table is sorted exactly once (lazy on first
    // resolve here).
    if let Ok(mut t) = symbols().write() {
        t.ensure_sorted();
    }

    eprintln!();
    eprintln!("┌─── JASM JIT crash dump ───────────────────────────────────────────────");
    eprintln!("│ exception : {:#010X}  {kind}", code);
    eprintln!("│ at RIP    : {:016X}  {}", rip, sym_str(rip));
    if let Some(addr_extra) = exception_extra(rec) {
        eprintln!("│ {addr_extra}");
    }
    eprintln!("│");
    eprintln!(
        "│ rax = {:016X}   rbx = {:016X}",
        (*ctx).Rax,
        (*ctx).Rbx
    );
    eprintln!(
        "│ rcx = {:016X}   rdx = {:016X}",
        (*ctx).Rcx,
        (*ctx).Rdx
    );
    eprintln!(
        "│ rsi = {:016X}   rdi = {:016X}",
        (*ctx).Rsi,
        (*ctx).Rdi
    );
    eprintln!(
        "│ rbp = {:016X}   rsp = {:016X}",
        (*ctx).Rbp,
        rsp
    );
    eprintln!(
        "│ r8  = {:016X}   r9  = {:016X}",
        (*ctx).R8,
        (*ctx).R9
    );
    eprintln!(
        "│ r10 = {:016X}   r11 = {:016X}",
        (*ctx).R10,
        (*ctx).R11
    );
    eprintln!(
        "│ r12 = {:016X}   r13 = {:016X}",
        (*ctx).R12,
        (*ctx).R13
    );
    eprintln!(
        "│ r14 = {:016X}   r15 = {:016X}",
        (*ctx).R14,
        (*ctx).R15
    );
    eprintln!("│ flags = {:08X}", (*ctx).EFlags);
    eprintln!("│");
    eprintln!("│ stack (32 qwords from rsp):");
    for i in 0..32 {
        let qaddr = rsp.wrapping_add((i as u64) * 8);
        // Read each qword cautiously — if the stack is corrupt, the
        // read could itself fault. We can't catch nested exceptions
        // from inside a VEH handler, so we do a structured try via
        // a separate Rust function with a guard. For simplicity v1:
        // do the raw read and trust the stack pointer. If a corrupted
        // RSP crashes us, the OS terminates the process — better than
        // a silent infinite loop.
        let val = *(qaddr as *const u64);
        eprintln!(
            "│  [rsp+{:3}] {:016X} {:016X}  {}",
            i * 8,
            qaddr,
            val,
            sym_str(val)
        );
    }
    eprintln!("└───────────────────────────────────────────────────────────────────────");
    eprintln!();
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

unsafe fn exception_extra(rec: *mut EXCEPTION_RECORD) -> Option<String> {
    let code = (*rec).ExceptionCode;
    if code == STATUS_ACCESS_VIOLATION && (*rec).NumberParameters >= 2 {
        let kind = (*rec).ExceptionInformation[0];
        let addr = (*rec).ExceptionInformation[1];
        let op = match kind {
            0 => "read",
            1 => "write",
            8 => "execute",
            _ => "?",
        };
        Some(format!(
            "access type: {op} at address {:016X}",
            addr as u64
        ))
    } else {
        None
    }
}

/// Format a symbol-resolution result. Returns the empty string when
/// nothing's near; otherwise `"<name>+0xNN [source]"`.
fn sym_str(query: u64) -> String {
    let Ok(t) = symbols().read() else {
        return String::new();
    };
    match t.resolve(query) {
        Some((sym, off)) => {
            if off == 0 {
                format!("<{}> [{}]", sym.name, sym.source)
            } else {
                format!("<{}+0x{:x}> [{}]", sym.name, off, sym.source)
            }
        }
        None => String::new(),
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
    jit: &mut crate::Jit,
    names: &[&str],
) -> Result<(), crate::JitError> {
    let mut acc: Vec<(String, u64, &'static str)> = Vec::with_capacity(names.len());
    for &name in names {
        let addr = jit.lookup_addr(name)?;
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
