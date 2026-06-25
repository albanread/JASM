//! WF64 — 64-bit ANS Forth kernel driver.
//!
//! Assembles forth/kernel.masm, allocates memory regions, initializes the
//! user area, and transfers control to the Forth interpreter.
//!
//! Register conventions (agreed with the kernel):
//!   RAX = TOS  (top of data stack)
//!   RBP = DSP  (data stack pointer, points at NOS)
//!   RBX = UP   (user area pointer)
//!   RSP = return stack (native x64 stack)
//!   R12 = Win64 RSP-save for callouts
//!
//! Entry point called:
//!   extern "C" fn forth_entry(user_area: u64, data_stack_ptr: u64) -> u64

use std::ffi::c_void;
use std::io::{self, Read, Write};

use anyhow::{Context, Result};
use wfasm::{Assembler, Jit};

// ── Memory region sizes ──────────────────────────────────────────────────────

const DATA_STACK_CELLS: usize = 4096;        // 4 096 cells × 8 bytes = 32 KB
const DICT_SPACE_BYTES: usize = 1024 * 1024; // 1 MB for runtime-defined words
const TIB_SIZE: usize = 1024;                // terminal input buffer

// ── User area offsets (must match forth/user-area.masm) ─────────────────────

const UP_DP: usize = 0;
const UP_BASE: usize = 8;
const UP_STATE: usize = 16;
const UP_LATEST: usize = 24;
const UP_IN: usize = 32;
const UP_SOURCE_ADDR: usize = 40;
const UP_SOURCE_LEN: usize = 48;
const UP_TIB: usize = 56;
const UP_NTIB: usize = 64;
const UP_HLD: usize = 72;
const UP_HANDLER: usize = 80;
const UP_SIZE: usize = 128; // total user area size in bytes

// ── Runtime functions provided to the JIT ───────────────────────────────────

/// EMIT ( c -- )
/// Print one character to stdout. Win64 ABI: arg in RCX, result in RAX.
extern "C" fn rt_emit(ch: u64) -> u64 {
    let _ = io::stdout().write_all(&[ch as u8]);
    let _ = io::stdout().flush();
    0
}

/// KEY ( -- c )
/// Read one character from stdin. Returns the character code in RAX.
/// Returns 0 on EOF or error.
extern "C" fn rt_key() -> u64 {
    let mut buf = [0u8; 1];
    match io::stdin().read_exact(&mut buf) {
        Ok(_) => buf[0] as u64,
        Err(_) => 0,
    }
}

/// ACCEPT ( buf-addr max-len -- actual-len )
/// Read a line (up to max_len bytes, without the newline) into the buffer.
/// Win64 ABI: arg1=buf_ptr in RCX, arg2=max_len in RDX. Returns char count in RAX.
extern "C" fn rt_accept(buf_ptr: u64, max_len: u64) -> u64 {
    // SAFETY: the caller (Forth kernel) passes a valid buffer pointer and length.
    let buf = unsafe { std::slice::from_raw_parts_mut(buf_ptr as *mut u8, max_len as usize) };
    let mut n = 0usize;
    let stdin = io::stdin();
    let mut lock = stdin.lock();
    for b in buf.iter_mut() {
        let mut ch = [0u8];
        match lock.read_exact(&mut ch) {
            Ok(_) => {
                if ch[0] == b'\n' || ch[0] == b'\r' {
                    break;
                }
                *b = ch[0];
                n += 1;
            }
            Err(_) => break,
        }
    }
    n as u64
}

/// BYE ( -- )
/// Terminate the Forth process cleanly.
extern "C" fn rt_bye() -> u64 {
    std::process::exit(0);
}

// ── Runtime function table ───────────────────────────────────────────────────

/// Pairs (name, arg_count, fn_ptr) for every host function the kernel may
/// declare with `@extern`. Computed here (not a static) because `*mut c_void`
/// is not `Sync`.
fn runtime_fns() -> Vec<(&'static str, usize, *mut c_void)> {
    vec![
        ("rt_emit", 1, rt_emit as *mut c_void),
        ("rt_key", 0, rt_key as *mut c_void),
        ("rt_accept", 2, rt_accept as *mut c_void),
        ("rt_bye", 0, rt_bye as *mut c_void),
    ]
}

// ── Entry point ─────────────────────────────────────────────────────────────

fn main() -> Result<()> {
    // ── 1. Assemble the kernel ───────────────────────────────────────────────

    let mut asm = Assembler::new();

    // Inject host-defined constants the kernel source may read via @if / @assign.
    asm.define("cell", 8);

    // Register the `stk` Rust macro (stack-effect DSP adjuster). The kernel
    // declares `@rust_macro stk` and calls `stk(in, out)` in primitives.
    asm.register_macro("stk", wfasm::asm::macros::stk);

    let kernel_path = "forth/kernel.masm";
    let asm_text = asm
        .assemble_file(kernel_path)
        .with_context(|| format!("assembling {kernel_path}"))?;

    if std::env::var_os("WF64_DUMP_ASM").is_some() {
        eprintln!("=== assembled kernel ===\n{asm_text}\n========================");
    }

    // ── 2. Set up the JIT ───────────────────────────────────────────────────

    let mut jit = Jit::new("wf64-kernel").context("Jit::new")?;

    // Snapshot the extern table first: define_extern_fn borrows `jit`
    // mutably, which would conflict with holding an iterator into `asm`.
    let externs: Vec<(String, usize, Option<String>)> = asm
        .externs()
        .map(|(n, d)| (n.to_string(), d.arg_count, d.dll.clone()))
        .collect();

    let fns = runtime_fns();

    for (name, arg_count, dll) in &externs {
        if dll.is_some() {
            // Win32 DLL function — handled by bind_externs below.
            continue;
        }
        let addr = fns
            .iter()
            .find_map(|(n, _a, p)| if *n == name.as_str() { Some(*p) } else { None })
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "kernel declares @extern `{name}` but runtime has no entry for it"
                )
            })?;
        jit.define_extern_fn(name, *arg_count, addr)
            .with_context(|| format!("define_extern_fn for `{name}`"))?;
    }

    // Bind any Win32 DLL externs (if kernel @includes win32/*.masm).
    #[cfg(windows)]
    {
        let _report = wfasm::win32::bind_externs(&asm, &mut jit, |_name| {
            // All non-DLL externs were already registered above; returning
            // None here causes bind_externs to error for any unregistered
            // host function — good: we catch mismatches early.
            None
        })
        .context("bind Win32 externs")?;
    }

    // ── 3. Load the kernel into the JIT ─────────────────────────────────────

    jit.add_asm(&asm_text).context("jit.add_asm")?;

    // ── 4. Declare the entry point so MCJIT exposes it after finalization ────

    // forth_entry(user_area: u64, data_stack_ptr: u64) -> u64
    jit.declare_fn("forth_entry", 2)
        .context("declare forth_entry")?;

    // Also declare forth_last_link so we can read the kernel's LATEST value.
    // This is a .globl label pointing at a .quad that holds the LFA of the
    // last defined word. The kernel must export it.
    jit.declare_fn("forth_last_link", 0)
        .context("declare forth_last_link")?;

    if std::env::var_os("WF64_DUMP_IR").is_some() {
        jit.dump_ir();
    }

    // ── 5. Allocate memory regions ───────────────────────────────────────────

    // Data stack: grows downward. The buffer holds DATA_STACK_CELLS u64 words.
    // Initial DSP = one-past-end of the buffer. The first push decrements DSP
    // by 8 (one cell) before writing, so it lands on the last valid slot.
    let mut data_stack = vec![0u64; DATA_STACK_CELLS];
    // SAFETY: add(DATA_STACK_CELLS) is one-past-end, which is valid for pointer arithmetic.
    let initial_dsp =
        unsafe { data_stack.as_mut_ptr().add(DATA_STACK_CELLS) } as u64;

    // Dictionary space: used for words defined at runtime.
    let mut dict_space = vec![0u8; DICT_SPACE_BYTES];
    let dict_ptr = dict_space.as_mut_ptr() as u64;

    // Terminal input buffer.
    let mut tib = vec![0u8; TIB_SIZE];
    let tib_ptr = tib.as_mut_ptr() as u64;

    // User area (UP_SIZE bytes = 16 cells of 8 bytes each; we allocate as u64s
    // so the slice is naturally aligned).
    let mut user_area = vec![0u64; UP_SIZE / 8];
    let up_ptr = user_area.as_mut_ptr() as u64;

    // ── 6. Read LATEST from the kernel ──────────────────────────────────────

    // forth_last_link is a .globl label in kernel.masm that aliases the
    // address of a .quad holding the LFA of the last dictionary word.
    // lookup_addr forces JIT finalization, so we do this before setup_user_area
    // so we can pass the result in.
    let forth_last_link_addr = jit
        .lookup_addr("forth_last_link")
        .context("lookup forth_last_link — is kernel.masm assembled?")?;

    // Read the 64-bit value stored at that address.
    // SAFETY: forth_last_link_addr points to a .quad in the JIT code segment,
    // which is valid for the lifetime of the jit object.
    let initial_latest = unsafe { *(forth_last_link_addr as *const u64) };

    // ── 7. Initialize the user area ─────────────────────────────────────────

    // SAFETY: user_area is a Vec<u64> we own; the pointer arithmetic stays
    // within the allocated slice (UP_SIZE / 8 = 16 elements, all accessed).
    unsafe {
        let ua = user_area.as_mut_ptr();
        *ua.add(UP_DP / 8) = dict_ptr;       // DP = start of dictionary space
        *ua.add(UP_BASE / 8) = 10;            // BASE = decimal
        *ua.add(UP_STATE / 8) = 0;            // STATE = interpreting
        *ua.add(UP_LATEST / 8) = initial_latest; // LATEST = last kernel word
        *ua.add(UP_IN / 8) = 0;               // >IN = 0
        *ua.add(UP_SOURCE_ADDR / 8) = 0;      // SOURCE = empty
        *ua.add(UP_SOURCE_LEN / 8) = 0;
        *ua.add(UP_TIB / 8) = tib_ptr;        // TIB = address of TIB buffer
        *ua.add(UP_NTIB / 8) = TIB_SIZE as u64; // #TIB = TIB buffer capacity
        *ua.add(UP_HLD / 8) = 0;              // HLD = 0 (no pictured output in progress)
        *ua.add(UP_HANDLER / 8) = 0;          // HANDLER = 0 (no CATCH frame)
    }

    // ── 8. Install crash dumper (Windows only) ───────────────────────────────

    #[cfg(windows)]
    {
        wfasm::seh::install()
            .map_err(|e| anyhow::anyhow!("SEH install failed: {e}"))?;
    }

    // ── 9. Look up and call the kernel entry point ───────────────────────────

    // SAFETY: forth_entry is a symbol in our JIT module whose Forth-side ABI
    // matches `extern "C" fn(u64, u64) -> u64` on Win64 (RCX=user_area,
    // RDX=data_stack_ptr, return in RAX).
    type ForthEntry = extern "C" fn(user_area: u64, data_stack_ptr: u64) -> u64;
    let forth_entry: ForthEntry = unsafe {
        jit.lookup_fn("forth_entry")
            .context("lookup forth_entry — is .globl forth_entry in kernel.masm?")?
    };

    eprintln!(
        "WF64 starting. UP @ {up_ptr:#x}, DSP @ {initial_dsp:#x}, LATEST @ {initial_latest:#x}"
    );

    let result = forth_entry(up_ptr, initial_dsp);

    eprintln!("WF64 exited with {result}");

    // The buffers must remain live for the entire duration of forth_entry.
    // Rust's drop order is last-declared first, so data_stack / dict_space /
    // tib / user_area all live until after the call above. Make it explicit:
    drop(user_area);
    drop(tib);
    drop(dict_space);
    drop(data_stack);

    Ok(())
}
