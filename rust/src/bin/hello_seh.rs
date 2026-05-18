//! Demo: SEH crash dumper.
//!
//! JITs two procs and exercises both behaviours of the VEH handler:
//!
//! 1. **`brk_demo`** — does an `int 3`, then a few more instructions.
//!    The handler dumps state, advances RIP past the `0xCC`, and lets
//!    execution continue. The function returns normally.
//!
//! 2. **`crash_demo`** — does a deliberate null-pointer deref. The
//!    handler dumps state and returns `CONTINUE_SEARCH`, which lets
//!    the default handler kill the process. We run this LAST.
//!
//! Both dumps show all 16 GPRs, the RIP at fault, and 32 qwords of
//! stack. Symbol resolution turns return-address stack values into
//! `<name+offset>` form so the user can see who called what.

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

; Drop an `int 3` mid-proc. The SEH handler decodes it, dumps state,
; advances past the byte, and lets us continue. Returns 0xCAFE so we
; can confirm the proc finished.
proc(brk_demo)
    mov     rax, 0xDEAD
    int     3                  ; <- VEH catches STATUS_BREAKPOINT
    mov     rax, 0xCAFE        ; <- runs after the handler returns
endp()

; Deliberate access violation. Loads a null pointer and reads through
; it. The handler dumps and returns CONTINUE_SEARCH, so the process
; aborts afterwards.
proc(crash_demo)
    xor     rcx, rcx
    mov     rax, qword ptr [rcx]    ; <- access violation
endp()
"#;

fn main() -> Result<()> {
    // Install the crash dumper. Idempotent; safe to call any number
    // of times. Must happen before any JIT'd code runs.
    wfasm::seh::install().expect("install SEH handler");

    let mut asm = Assembler::new();
    let asm_text = asm
        .assemble("hello-seh.masm", SOURCE)
        .context("assemble failed")?;

    let mut jit = Jit::new("hello-seh").context("Jit::new")?;
    jit.add_asm(&asm_text).context("add_asm")?;
    jit.declare_fn("brk_demo", 0).context("declare brk_demo")?;
    jit.declare_fn("crash_demo", 0).context("declare crash_demo")?;

    // Tell the dumper about our JIT'd procs so RIP and stack values
    // that fall inside them get symbolic names.
    wfasm::seh::register_jit_procs(&mut jit, &["brk_demo", "crash_demo"])
        .context("register jit procs")?;

    println!(
        "SEH installed, {} symbols registered. running brk_demo...",
        wfasm::seh::symbol_count()
    );

    // 1. Trigger the breakpoint path. The dump appears mid-call; the
    //    function still returns normally afterwards.
    type Fn0 = extern "C" fn() -> u64;
    let brk: Fn0 = unsafe { jit.lookup_fn("brk_demo") }.context("lookup brk_demo")?;
    let r = brk();
    println!("brk_demo returned {r:#X} (expected 0xCAFE)");
    if r != 0xCAFE {
        anyhow::bail!("expected 0xCAFE after BRK; got {r:#X}");
    }

    println!();
    println!("now triggering an access violation. dump will print, then process aborts:");
    println!();

    // 2. Trigger the access violation. Process should die after the
    //    dump prints.
    let crash: Fn0 = unsafe { jit.lookup_fn("crash_demo") }.context("lookup crash_demo")?;
    let _ = crash();

    // Unreachable in normal runs.
    println!("unexpectedly survived the crash");
    Ok(())
}
