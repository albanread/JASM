//! End-to-end smoke test for the runtime-call path:
//!
//!   JITed Forth → Win64 ABI → Rust runtime function → back to Forth
//!
//! Pattern:
//!
//! 1. Source declares `@extern rt_print_int(1)` and the user-defined
//!    `win64_call(target)` macro that wraps the call site with RSP
//!    alignment + 32-byte shadow space.
//! 2. The JITed function `forth_main` loads 42 into `rcx` (Win64 arg1)
//!    and `win64_call`s `rt_print_int`.
//! 3. Rust's `rt_print_int` prints `from JIT: 42`.
//! 4. JITed code returns 0 in `rax`.
//! 5. Rust main prints the return value.
//!
//! Expected output:
//!
//!   from JIT: 42
//!   forth_main() = 0
//!
//! This proves: (a) `@extern` records declarations the host can iterate;
//! (b) `Jit::define_extern_fn` plus `LLVMAddGlobalMapping` make Rust
//! functions callable by name from JITed asm; (c) the user-defined
//! Win64 prologue/epilogue actually works end-to-end with a real call.

use std::ffi::c_void;

use anyhow::{Context, Result};
use wfasm::{Assembler, Jit};

const SOURCE: &str = r#"
    .intel_syntax noprefix
    .text

@extern rt_print_int(1)

@macro proc(name)
    @scope &name
    .globl &name
&name:
@endmacro

@macro endp()
    ret
    @endscope
@endmacro

; ── Win64 callout wrapper ──────────────────────────────────────────
; The user defines the prologue. wfasm ships no calling-convention
; assumptions. R12 is callee-saved on Win64, so we use it to park RSP
; across the call (R11 would be unsafe — caller-saved).
@macro win64_call(target)
    mov     r12, rsp
    and     rsp, -16
    sub     rsp, 32
    call    &target
    mov     rsp, r12
@endmacro

; ── The entry point ───────────────────────────────────────────────
proc(forth_main)
    mov     rcx, 42                 ; Win64 arg1
    win64_call(rt_print_int)
    mov     rax, 0                  ; return 0
endp()
"#;

/// The runtime function our JITed asm calls. `extern "C"` on x86-64
/// Windows uses the Win64 ABI by default: first integer arg in RCX,
/// return value in RAX.
extern "C" fn rt_print_int(n: u64) -> u64 {
    println!("from JIT: {n}");
    0
}

/// Map of `@extern` names to their Rust function pointers. Anything
/// the source `@extern`s must appear here, or the smoke test errors
/// before it runs.
///
/// Computed each call (not a static) because `*mut c_void` isn't
/// `Sync` and a `static` in Rust needs to be. For the smoke test the
/// recomputation cost is negligible.
fn runtime_table() -> Vec<(&'static str, *mut c_void)> {
    vec![("rt_print_int", rt_print_int as *mut c_void)]
}

fn main() -> Result<()> {
    // 1. Assemble.
    let mut asm = Assembler::new();
    let asm_text = asm
        .assemble("hello-runtime.masm", SOURCE)
        .context("assemble failed")?;

    if std::env::var_os("WFASM_DUMP_ASM").is_some() {
        eprintln!("=== assembled output ===\n{asm_text}========================");
    }

    // 2. Inspect declared externs, pair each with a runtime pointer.
    // Snapshot first — `define_extern_fn` borrows `jit` mutably, which
    // would conflict with iterating `asm.externs()` while still holding
    // a borrow.
    let externs: Vec<(String, usize, Option<String>)> = asm
        .externs()
        .map(|(n, d)| (n.to_string(), d.arg_count, d.dll.clone()))
        .collect();
    let table = runtime_table();

    let mut jit = Jit::new("hello-runtime").context("Jit::new")?;

    for (name, arg_count, _dll) in &externs {
        let addr = table
            .iter()
            .find_map(|(n, a)| if *n == name.as_str() { Some(*a) } else { None })
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "asm declares @extern `{name}` but the runtime table has no entry"
                )
            })?;
        jit.define_extern_fn(name, *arg_count, addr)
            .with_context(|| format!("define_extern_fn for `{name}`"))?;
    }

    // 3. Add the asm and declare our entry point.
    jit.add_asm(&asm_text).context("add_asm")?;
    jit.declare_fn("forth_main", 0).context("declare_fn")?;

    if std::env::var_os("WFASM_DUMP_IR").is_some() {
        jit.dump_ir();
    }

    // 4. Look up and call.
    type ForthMain = extern "C" fn() -> u64;
    let f: ForthMain = unsafe { jit.lookup_fn("forth_main") }
        .context("lookup forth_main")?;

    let result = f();
    println!("forth_main() = {result}");

    if result != 0 {
        anyhow::bail!("expected 0, got {result}");
    }
    Ok(())
}
