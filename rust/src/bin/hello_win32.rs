//! End-to-end Win32 demo using the generator-produced bindings.
//!
//! `@include "win32/kernel32.masm"` pulls in all 1,165 KERNEL32
//! function declarations. The host iterates the assembler's `externs`,
//! `LoadLibraryW`s each distinct DLL once, `GetProcAddress`es every
//! function, registers each (name → address) with the JIT. MCJIT/RTDyld
//! then resolves direct `call GetTickCount64` instructions in JITed
//! Forth code to the real Windows function.
//!
//! What this proves:
//!
//! * The generator output is consumable as-is by `Assembler::assemble`.
//! * `bind_externs` handles a kilo-extern module without falling over.
//! * Functions Windows-on-this-machine doesn't export (`bind_externs`
//!   skips them) don't block the build; they only surface as RTDyld
//!   "unresolved symbol" errors if JITed code actually `call`s them.
//! * A user-defined `win64_call` macro is enough — no
//!   marshalling layer required.

use std::ffi::c_void;

use anyhow::{Context, Result};
use wfasm::{Assembler, Jit};

const SOURCE: &str = r#"
    .intel_syntax noprefix
    .text

; Pull in KERNEL32. v2 generator emits @extern declarations AND
; matching invoke-style @macro wrappers. After this, every supported
; KERNEL32 function is a direct call by name with the Win64 ABI
; (shadow space, alignment, register placement) handled by the wrapper.
@include "win32/kernel32.masm"

@macro proc(name)
    @scope &name
    .globl &name
&name:
@endmacro

@macro endp()
    ret
    @endscope
@endmacro

proc(get_ticks)
    GetTickCount64()      ; the wrapper handles everything
endp()
"#;

fn main() -> Result<()> {
    let mut asm = Assembler::new();
    let asm_text = asm
        .assemble("hello-win32.masm", SOURCE)
        .context("assemble failed")?;

    let extern_count = asm.externs().count();
    println!("assembler saw {extern_count} @extern declarations");

    let mut jit = Jit::new("hello-win32").context("Jit::new")?;

    let report = wfasm::win32::bind_externs(&asm, &mut jit, |name| -> Option<*mut c_void> {
        eprintln!("warning: unresolved host extern `{name}`");
        None
    })
    .context("bind_externs failed")?;

    println!(
        "bound {} extern{} ({} unresolved)",
        report.bound,
        if report.bound == 1 { "" } else { "s" },
        report.missing_proc.len(),
    );
    if std::env::var_os("WFASM_DUMP_MISSING").is_some() && !report.missing_proc.is_empty() {
        for (name, dll, err) in &report.missing_proc {
            eprintln!("  missing: {dll}!{name} (GetLastError={err})");
        }
    }

    jit.add_asm(&asm_text).context("add_asm")?;
    jit.declare_fn("get_ticks", 0).context("declare_fn")?;

    type GetTicks = extern "C" fn() -> u64;
    let f: GetTicks = unsafe { jit.lookup_fn("get_ticks") }
        .context("lookup get_ticks")?;

    let t1 = f();
    std::thread::sleep(std::time::Duration::from_millis(20));
    let t2 = f();

    println!("GetTickCount64() #1 = {t1}");
    println!("GetTickCount64() #2 = {t2}");
    println!("delta              = {} ms", t2 - t1);

    if t1 == 0 {
        anyhow::bail!("first call returned 0 — did the JIT bind?");
    }
    if t2 <= t1 {
        anyhow::bail!("second call didn't advance past first");
    }
    Ok(())
}
