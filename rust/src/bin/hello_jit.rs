//! End-to-end smoke test for the full wfasm pipeline:
//!
//!   source text
//!     → lexer  → tokens
//!     → expander → expanded tokens
//!     → emitter  → MC-flavor assembly string
//!     → MCJIT (`LLVMAppendModuleInlineAsm` + `LLVMGetFunctionAddress`)
//!     → function pointer
//!     → call from Rust, print the result
//!
//! Expected output: `forth_main() = 42`.
//!
//! The Forth-style `proc(name)` macro is defined in source — the
//! assembler ships no Forth conventions, only the primitives the user
//! composes them from.

use anyhow::{Context, Result};
use wfasm::{Assembler, Jit};

const SOURCE: &str = r#"
    .intel_syntax noprefix
    .text

@macro proc(name)
    @scope &name
    .globl &name
&name:
@endmacro

@macro endp()
    ret
    @endscope
@endmacro

proc(forth_main)
    mov rax, 42
endp()
"#;

fn main() -> Result<()> {
    // 1. Assemble: text in, MC-flavor asm out.
    let mut asm = Assembler::new();
    let asm_text = asm
        .assemble("hello.masm", SOURCE)
        .context("assemble failed")?;

    if std::env::var_os("WFASM_DUMP_ASM").is_some() {
        eprintln!("=== assembled output ===\n{asm_text}========================");
    }

    // 2. JIT: asm string in, executable function out.
    let mut jit = Jit::new("hello").context("Jit::new")?;
    jit.add_asm(&asm_text).context("add_asm")?;
    jit.declare_fn("forth_main", 0).context("declare_fn")?;

    if std::env::var_os("WFASM_DUMP_IR").is_some() {
        jit.dump_ir();
    }

    // 3. Look up and call.
    type ForthMain = extern "C" fn() -> u64;
    let f: ForthMain = unsafe { jit.lookup_fn("forth_main") }
        .context("lookup forth_main")?;

    let result = f();
    println!("forth_main() = {result}");

    if result != 42 {
        anyhow::bail!("expected 42, got {result}");
    }
    Ok(())
}
