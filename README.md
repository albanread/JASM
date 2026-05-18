# JASM

A JIT macro-assembler for x86-64 Windows. Source in, function pointer
out.

Brings MASM32-era ergonomics to a modern LLVM-MC + MCJIT pipeline,
exposes the entire Win32 API by name, and ships a crash dumper for
when your hand-written asm goes sideways.

```masm
@include "win32/kernel32.masm"

@macro proc(name)
    @scope &name
    .globl &name
&name:
@endmacro

@macro endp() ret @endscope @endmacro

proc(get_ticks)
    GetTickCount64()      ; a real Win32 call by name
endp()
```

The Rust crate, sources, Python generator, and generated `.masm`
bindings live in [`rust/`](rust/). Start there:

- **[rust/README.md](rust/README.md)** — the project README
- **[rust/USER-GUIDE.md](rust/USER-GUIDE.md)** — the language reference
- **[rust/PLAN-MACROS.md](rust/PLAN-MACROS.md)** — design notes

## Quick start

```
$ cd rust
$ cargo test                       # 138 lib tests
$ cargo run --bin hello-jit        # mov rax, 42; ret  →  prints 42
$ cargo run --bin hello-runtime    # JIT calls Rust, prints "from JIT: 42"
$ cargo run --bin hello-win32      # JIT calls KERNEL32!GetTickCount64
$ cargo run --bin hello-seh        # int 3 dumps and continues; segfault dumps and aborts
```

Prerequisites: Rust 1.95+, LLVM 22.x at `C:\Program Files\LLVM\` (or
set `LLVM_DIR`), and Python 3.x if you want to regenerate Win32
bindings from `windows_api.db`.

## License

MIT — see [LICENSE](LICENSE).
