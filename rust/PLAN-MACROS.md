# wfasm — Macro Assembler Design

A line-oriented x86-64 macro assembler that delegates instruction encoding
to LLVM's integrated assembler (via MCJIT `LLVMAppendModuleInlineAsm`)
and adds, on top, a macro layer with **zero implicit code generation**.

## The hard rule

**Every byte emitted traces to text the user wrote or to a macro the user
defined.** The assembler ships:

- No default prologue. No default epilogue.
- No "calling convention." No register reservations.
- No "function" concept beyond labels.
- No assumed ABI. No automatic shadow-space, stack alignment, or unwind info.

If a project wants Win64 ABI + frame pointer, it writes a `@macro win64_proc`
that emits the prologue and bumps `rsp`. If it doesn't want any of that, it
writes nothing. **The assembler is not a code generator — it is a text → MC
pipeline with a macro front-end.**

## Source grammar (line-oriented)

| Surface form | Meaning |
|---|---|
| `; comment` | comment to EOL |
| `foo:` at col 0 | global label |
| `.foo:` | local label, scoped to enclosing `@scope` (or to file if none) |
| `@directive args` | built-in directive |
| `NAME args` where NAME is a known macro | macro invocation |
| anything else | passed verbatim to LLVM MC at end of expansion |

Lines that aren't directives or macro calls are accumulated into the
output buffer with macro names expanded to their `&param` substitutions
and `@CONTEXT` names expanded to their values. Mnemonics and operand
syntax are LLVM-MC's job.

## Built-in directives

All prefixed `@` to be unconfusable with mnemonics or labels.

### Definitions and constants

| Directive | Effect |
|---|---|
| `@define NAME value` | Simple text define. `NAME` expands to `value` everywhere it appears as an identifier token. |
| `@assign NAME = expr` | Compile-time integer arithmetic. Supports `+ - * / %`, parens, prior `@define`/`@assign` names. |
| `@undef NAME` | Remove a define. |

### Conditionals

| Directive | Effect |
|---|---|
| `@if expr` | Begin conditional block; `expr` is integer (0 = false). |
| `@elif expr` | Else-if. |
| `@else` | Else. |
| `@endif` | End. |
| `@ifdef NAME` / `@ifndef NAME` | Check if a name is defined. |

### Repetition

| Directive | Effect |
|---|---|
| `@rept N` … `@endr` | Repeat block N times. Inside, `@INDEX` is the 0-based index. |

### Macros (text)

| Directive | Effect |
|---|---|
| `@macro NAME(p1, p2, …)` … `@endmacro` | Define a text macro. |
| `@macro NAME(args…)` … `@endmacro` | Variadic; `&args` is the comma-joined tail. |

Inside a macro body, `&p` substitutes the parameter. `&a##b` concatenates
tokens without a separator (for synthesizing names). `&&p` emits a literal
`&` followed by `p` for the rare case you actually wanted `&` in output.

### Macros (Rust)

| Directive | Effect |
|---|---|
| `@rust_macro NAME` | Declare that NAME is a Rust-implemented macro. The host program registers a closure for NAME before assembling. If invoked without a registered handler, assembly aborts with a clear error. |

Rust macros receive parsed arguments and an emitter handle; they can do
anything (arithmetic, file I/O, table lookups). Use sparingly — the
textual ones are easier to debug.

### Scopes

| Directive | Effect |
|---|---|
| `@scope NAME` | Open a scope; sets `@PROC = NAME`; starts a local-label namespace; pushes onto the scope stack. |
| `@endscope` | Close the innermost scope. |

Local labels (`.foo`) inside scope `bar` mangle to `bar$$foo` in the
emitted asm. They're resolvable only from within `bar`. Without a scope,
`.foo` mangles to `__file__$$foo` (file-scoped).

### Labels and branches

#### Label scoping (we mangle)

| Surface form | Where it works | Mangled to |
|---|---|---|
| `foo:` (column 0) | global, whole module | `foo` (unchanged) |
| `.foo:` inside `@scope bar` | only within `bar` | `bar$$foo` |
| `.foo:` inside `@macro NAME` body | only within this one expansion | `NAME$$<n>$$foo` (`<n>` = invocation counter) |
| `.foo:` outside any scope or macro | file-local | `<basename>$$foo` |
| `1:`, `2:` (GAS numeric locals) | pass through to MC | unchanged |

The macro-local mangling is what makes `.done:` inside a macro safe — two
invocations of the same macro produce two different `.done` labels.

#### Branch size selection (LLVM MC does this for us)

We delegate to LLVM MC. Write `jmp foo` or `jz done` and MC's relaxation
pass picks `rel8` if the target's in reach, `rel32` otherwise. `call` on
x64 only has `rel32`. `jcxz`/`jrcxz` only have `rel8` — out-of-reach is an
MC error you'll see at assembly time.

**Important:** relaxation is per MC invocation. Labels in a *different*
`LLVMAppendModuleInlineAsm` chunk are resolved by RTDyld at the link
stage, and that path always uses the long form — no relaxation across
chunk boundaries. The practical rule for a Forth kernel: emit everything
in one chunk so intra-kernel calls relax, and accept that calls to Rust
runtime functions (which are external symbols) stay 5 bytes.



| Directive | Effect |
|---|---|
| `@include "path"` | Inline another source file. Paths are relative to the including file. |
| `@error "msg"` | Halt assembly with a user-defined error. |
| `@warn "msg"` | Print a warning. |
| `@bits N` | Sanity-check the target bit width (always 64 for now). |

### Sections

| Directive | Effect |
|---|---|
| `@section NAME` | Switch sections. Emits `.section NAME` (or platform equivalent) to MC. |
| Predefined: `@code`, `@data`, `@rodata`, `@bss` | Shortcuts. |

### Data emission

| Directive | Effect |
|---|---|
| `@db a, b, c, …` | Emit bytes (forwards to MC `.byte`). |
| `@dw …` | 16-bit words (`.short`). |
| `@dd …` | 32-bit (`.long`). |
| `@dq …` | 64-bit (`.quad`). |
| `@dz "string"` | Zero-terminated string (`.asciz`). |

These are 1:1 with MC directives but use names that don't fight MASM
intuition. (MASM's `db`/`dw`/`dd`/`dq` are reserved for the body; ours have
the `@` prefix to be unmistakable.)

## Built-in context names

Expand to text at point of use. Read-only.

| Name | Value |
|---|---|
| `@PROC` | Name of innermost open `@scope`, or empty string. |
| `@PROC_SIZE` | Bytes emitted in current scope since `@scope NAME`. Resets per scope. |
| `@COUNTER` | Auto-incremented integer, unique per assembly. Useful for synthesizing labels: `.tmp_$@COUNTER`. |
| `@LINE` | 1-based line in current file. |
| `@FILE` | Current file path. |
| `@SECTION` | Name of current section. |
| `@BITS` | Target bit width (64). |

## Worked examples

### Forth STC primitives

```masm
@include "forth-macros.masm"

@code

proc(plus)                    ; ( a b -- a+b )
    stk 2, 1                  ; Rust macro: emits add rbp, cell
    add  TOS, [DSP - cell]
    next
endp

proc(dup)                     ; ( a -- a a )
    stk 1, 2                  ; Rust macro: emits sub rbp, cell
    mov  [DSP], TOS
    next
endp

proc(literal_42)
    pushd 42
    next
endp
```

`forth-macros.masm` (project-supplied):

```masm
@define TOS  rax
@define DSP  rbp
@define UP   rbx
@define cell 8

@macro proc(name)
    @scope &name
    .globl &name
&name:
@endmacro

@macro endp()
    @endscope
@endmacro

@macro next()
    ret
@endmacro

@macro pushd(val)
    sub  DSP, cell
    mov  [DSP], TOS
    mov  TOS, &val
@endmacro

@rust_macro stk    ; registered from Rust to emit stack-effect adjustment
```

### Win64 ABI prologue when (and only when) we want one

```masm
@macro win64_proc(name)
    @scope &name
    .globl &name
&name:
    push   rbp
    mov    rbp, rsp
    sub    rsp, 32              ; shadow space if we're going to call APIs
@endmacro

@macro win64_endp()
    add    rsp, 32
    pop    rbp
    ret
    @endscope
@endmacro

win64_proc(say_hello)
    lea    rcx, [rip + .msg]
    call   puts
win64_endp()
.msg:   @dz "hello"
```

### Unroll

```masm
@assign N = 8
@scope memset_quad
.globl memset_quad
memset_quad:
    @rept N
        mov  qword [rdi + @INDEX * 8], rsi
    @endr
    ret
@endscope
```

### Compile-time switch (host bitness vs target)

```masm
@if cell == 8
    @define SP_ADJ_ONE_PUSH 8
@else
    @define SP_ADJ_ONE_PUSH 4
@endif
```

## What's NOT here

- **Floating-point macro arithmetic.** Integers only. If you need it later,
  add it.
- **Per-macro hygiene.** Identifiers leak. Use `_$@COUNTER` suffixes when
  collisions matter. NASM-compatible escape hatch.
- **Built-in instruction encoding.** That's LLVM MC's job. We never
  re-implement what MC already does correctly.
- **Output to disk by default.** The assembler emits a string of LLVM
  MC-flavored text; the JIT layer feeds it to `LLVMAppendModuleInlineAsm`.
  A `--emit-asm` flag for inspection is easy to add later.

## Implementation map

```
src/
  lib.rs           — re-exports
  llvm.rs          — LLVM-C FFI bindings (done)
  jit.rs           — MCJIT wrapper (done; symbol-lookup edge case being closed)
  asm/
    lex.rs         — tokens, line splitter, comment stripper
    parse.rs       — directive recognizer, macro-call recognizer, label parser
    expr.rs        — integer expression evaluator for @assign/@if/@rept
    macros.rs      — text macro store, parameter substitution
    scope.rs       — scope stack, local-label mangling, @PROC tracking
    rust_macro.rs  — host-side closure registration and dispatch
    emit.rs        — accumulates output text, tracks @PROC_SIZE, @LINE, etc.
    mod.rs         — driver: source path → output string
```

## Implementation order

1. **Lexer** — line tokenizer that's macro-aware (knows to keep `@directive`
   and `.local` and `&param` as single tokens). ~200 LOC.
2. **Expr eval** — integer expressions for `@assign`, `@if`, `@rept`. ~150 LOC.
3. **Macro engine** — `@macro`/`@endmacro` store; `@define`/`@assign`;
   parameter substitution; `@if`/`@elif`/`@else`/`@endif`; `@rept`. ~400 LOC.
4. **Scope** — `@scope`/`@endscope` stack; `.local` → `scope$$local`
   mangling; `@PROC` context. ~100 LOC.
5. **Rust macros** — `Assembler::register_macro("name", |args, emit| { … })`.
   ~50 LOC.
6. **Driver** — `Assembler::assemble(source_path) -> Result<String>` produces
   the final MC-flavor text. ~100 LOC.
7. **Integration** — `Jit::add_asm(assembler.assemble(...))` and prove the
   smoke test still works. The MCJIT symbol-lookup issue gets debugged here
   in context with real content (multi-line proc, .globl + label + body).

Total: ~1000 LOC of Rust, plus tests.

## Open questions parked for later

- **Lazy compilation per proc.** MCJIT is whole-module. For a Forth REPL
  with `:` colon definitions added at runtime we'll want per-definition
  modules added via `LLVMAddModule`. The macro layer doesn't change; the
  JIT layer grows a "fresh module per definition" mode.
- **Debug info.** `@dwarf_proc` macro family to emit `.cfi_*` directives
  for gdb / windbg unwinding. Defer until something cares.
- **PE-style .pdata / .xdata** for Win64 SEH unwind. NewBCPL has the memory-
  manager-side hook (`jit_mm.rs`) — we'd port that when a Forth primitive
  starts calling Rust code that can panic and we want clean unwind. Until
  then, panics escape; document and move on.
