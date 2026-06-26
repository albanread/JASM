//! hello-aarch64 — the native Apple Silicon pipeline end-to-end, no LLVM:
//!
//!   AArch64 `.masm` source → macro front-end (`Assembler`)
//!     → native `A64Encoder` → `MacJit` (MAP_JIT + W^X + icache + veneers)
//!     → `extern "C"` fn pointer → call from Rust, with a host callback.
//!
//! Mirrors `hello-runtime` (the x86/LLVM smoke test) but uses the LLVM-free
//! AArch64 backend. Expected output:
//!
//!   from JIT: 42
//!   forth_main() = 84
//!
//! Proving: (a) the macro engine is arch-neutral (proc/endp/call_rt expand the
//! same way); (b) `A64Encoder` assembles the expanded AArch64 text; (c) `MacJit`
//! places it, routes the `bl rt_print_int` extern through an absolute veneer, and
//! runs it; (d) AAPCS64 host calls work (arg in x0, result in x0).

#[cfg(target_os = "macos")]
fn run() -> anyhow::Result<()> {
    use std::ffi::c_void;
    use wfasm::{Assembler, Loader};

    // AAPCS64 Forth-flavoured source. No `.intel_syntax`; AArch64 has one syntax.
    const SOURCE: &str = r#"
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

; AAPCS64 callout wrapper: save FP/LR, keep SP 16-byte aligned across the call.
@macro call_rt(target)
    stp x29, x30, [sp, #-16]!
    bl  &target
    ldp x29, x30, [sp], #16
@endmacro

; forth_main(): print 42 via a Rust host call, then return 42*2 = 84.
proc(forth_main)
    mov x0, #42                 ; AAPCS64 arg1
    call_rt(rt_print_int)
    mov x0, #42
    add x0, x0, x0             ; return 84
endp()
"#;

    extern "C" fn rt_print_int(n: u64) -> u64 {
        println!("from JIT: {n}");
        0
    }

    let mut asm = Assembler::new();
    let text = asm
        .assemble("hello-a64.masm", SOURCE)
        .map_err(|e| anyhow::anyhow!("assemble failed: {e}"))?;
    if std::env::var_os("WFASM_DUMP_ASM").is_some() {
        eprintln!("=== assembled output ===\n{text}========================");
    }

    let mut jit = wfasm::native_macos::MacJit::new();
    jit.define_extern_fn("rt_print_int", 1, rt_print_int as *const () as *mut c_void)?;
    jit.add_asm(&text)?;

    let f: extern "C" fn() -> u64 = unsafe { jit.lookup_fn("forth_main")? };
    let result = f();
    println!("forth_main() = {result}");
    anyhow::ensure!(result == 84, "expected 84, got {result}");
    Ok(())
}

#[cfg(not(target_os = "macos"))]
fn run() -> anyhow::Result<()> {
    eprintln!("hello-aarch64 is macOS/Apple-Silicon only (uses MAP_JIT).");
    Ok(())
}

fn main() -> anyhow::Result<()> {
    run()
}
