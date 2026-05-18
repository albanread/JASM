# JASM — User Guide

JASM (the `wfasm` crate) is a line-oriented x86-64 macro assembler. It
parses your source, expands macros, mangles local labels, and hands the
result to LLVM's integrated assembler for encoding. The output is a
function pointer you can call from Rust.

Workspace: `E:\JASM\rust\`.

## The prime rule

**Every byte that gets emitted traces to something you wrote or to a macro
you defined.** wfasm ships:

- no default function prologue
- no default function epilogue
- no calling convention
- no register reservations
- no automatic shadow-space, stack alignment, or unwind info

If you want a Win64 frame, write `@macro win64_proc` and use it. If you
don't, your label is just a label and your `ret` is just a `ret`. The
assembler is not a code generator — it is a text-to-machine-code pipeline
with a macro front end.

## Table of contents

1. [At a glance](#at-a-glance)
2. [How a source file is read](#how-a-source-file-is-read)
3. [Labels](#labels)
4. [Branches and label resolution](#branches-and-label-resolution)
5. [Built-in directives](#built-in-directives)
6. [Text macros](#text-macros)
7. [Rust macros](#rust-macros)
8. [Built-in context names](#built-in-context-names)
9. [Conditional assembly and assertions](#conditional-assembly-and-assertions)
10. [Iteration](#iteration)
11. [MASM-style runtime control flow](#masm-style-runtime-control-flow)
12. [Calling external (Rust) functions](#calling-external-rust-functions)
13. [Worked example — Forth STC primitives](#worked-example--forth-stc-primitives)
14. [Worked example — Win64 callout macro](#worked-example--win64-callout-macro)
15. [Worked example — a tight memchr loop](#worked-example--a-tight-memchr-loop)
16. [Gotchas and traps](#gotchas-and-traps)
17. [Integration with the wfasm JIT](#integration-with-the-wfasm-jit)
18. [Calling the Windows API](#calling-the-windows-api)
19. [Crash dumps and breakpoints](#crash-dumps-and-breakpoints)
20. [What this guide doesn't cover (yet)](#what-this-guide-doesnt-cover-yet)

---

## At a glance

`hello.masm`:

```masm
    .intel_syntax noprefix
    .text
    .globl  forth_main

forth_main:
    mov     rax, 42
    ret
```

`main.rs`:

```rust
use wfasm::{Assembler, Jit};

fn main() -> anyhow::Result<()> {
    let asm  = Assembler::new().assemble_file("hello.masm")?;
    let mut jit  = Jit::new("hello")?;
    jit.add_asm(&asm)?;
    jit.declare_fn("forth_main", 0)?;

    type ForthMain = extern "C" fn() -> u64;
    let f: ForthMain = unsafe { jit.lookup_fn("forth_main")? };
    println!("forth_main() = {}", f());
    Ok(())
}
```

Runs, prints `forth_main() = 42`. That's the whole pipe.

---

## How a source file is read

wfasm reads the file line by line. Each line falls into one of four
buckets:

| Surface | Bucket | Treatment |
|---|---|---|
| empty / whitespace only | blank | passed through to MC unchanged |
| `; comment` | comment | stripped; not emitted |
| starts with `@` | directive | parsed by wfasm |
| starts with a known macro name + `(` | macro call | expanded by wfasm |
| `name:` at column 0 | label | mangled by wfasm, then emitted |
| anything else | instruction or directive for MC | emitted unchanged |

wfasm does not parse x86 instructions. Mnemonics, operand syntax, hex
literals (`0FFh` won't work; use `0xFF` — LLVM is GAS-flavored), and all
of MC's own directives (`.intel_syntax`, `.text`, `.globl`, `.byte`,
`.cfi_*`, etc.) are passed through verbatim.

Source files are typically named `.masm`. The extension carries no
meaning; wfasm reads anything.

### Numbers

| Form | Meaning |
|---|---|
| `42` | decimal |
| `0x2A`, `0X2A` | hex |
| `0b101010` | binary |
| `0o52` | octal |
| `'A'` | ASCII byte (65) |
| `'\\n'` | escape: `\n \r \t \\ \' \" \0 \xNN` |

### Strings

Double-quoted, same escapes as character literals. Used for `@include`,
`@error`, `@dz`, and `@warn`. There are no string variables at macro time.

---

## Labels

### Global labels

A bare identifier followed by `:` at the start of a line:

```masm
forth_main:
    ret
```

Emitted to MC as-is. To export, follow your assembler / linker convention
(`.globl forth_main`).

### Scope-local labels

Inside an `@scope NAME` block, any label starting with `.` is mangled to
`NAME$$LABEL` so it can't collide with same-named locals in another
scope:

```masm
@scope plus
.globl plus
plus:
    add  rax, [rbp]
    add  rbp, 8
    jnz  .done       ; resolves to plus$$done
    int  3
.done:
    ret
@endscope
```

The mangling is invisible to you — you write `.done`, you reference
`.done`, and it just works.

### Macro-local labels

Inside a `@macro` body, any `.label` is mangled per **invocation**, not
just per macro. Two calls to the same macro produce two distinct
labels — no `%%` ceremony required.

```masm
@macro retry_if_zero(reg)
    test  &reg, &reg
    jz    .skip
    ; ... work ...
.skip:
@endmacro

retry_if_zero(rax)   ; .skip becomes retry_if_zero$$1$$skip
retry_if_zero(rcx)   ; .skip becomes retry_if_zero$$2$$skip
```

### File-local labels

If a `.label` appears outside any `@scope` and outside any `@macro`, it's
scoped to the file: mangled as `<basename>$$LABEL`. Useful for static
data tables at the top of a file.

### GAS-style numeric labels

LLVM MC accepts `1:`, `2:` etc. with `1b` / `1f` references for backward
/ forward. wfasm passes these through unchanged. Handy in tight loops
where you don't want to invent names:

```masm
1:  cmp  byte [rdi], 0
    je   2f
    inc  rdi
    jmp  1b
2:
```

---

## Branches and label resolution

You don't compute branch displacements. You write `jmp foo` or `jz done`;
LLVM MC's relaxation pass picks `rel8` or `rel32` based on distance.

Rules:

- `jmp` and `jcc` have both short (`rel8`) and near (`rel32`) forms. MC
  picks the smallest that reaches.
- `call` only has `rel32` on x64 — always 5 bytes.
- `jcxz`, `jrcxz` only have `rel8`. Out-of-reach is an assembly error.

**One subtlety:** relaxation works *within one assembler chunk*. If you
emit your kernel as one big blob (one call to `Assembler::assemble_file`,
one call to `Jit::add_asm`), all intra-kernel jumps relax. Calls to
external Rust functions (`call rt_emit`) go through a relocation and
stay `rel32`.

For a Forth kernel this means: keep the whole kernel as one source tree
included into one top-level file. You'll get optimal intra-kernel
encoding "for free."

---

## Built-in directives

All built-in directives start with `@`.

### Definitions

| Directive | Effect |
|---|---|
| `@define NAME value` | Plain text define. Every later occurrence of `NAME` as an identifier expands to `value`. |
| `@assign NAME = expr` | Compile-time integer arithmetic. Result is stored as a number. Supports `+ - * / % & ^ |` and parens. |
| `@undef NAME` | Remove a previously defined name. |

```masm
@define TOS    rax
@define DSP    rbp
@assign cell = 8
@assign frame = cell * 4    ; 32
```

`@define` is plain text; `@assign` evaluates. If you write
`@define foo 1+2`, `foo` expands to the three characters `1+2`. If you
write `@assign foo = 1+2`, `foo` expands to `3`.

### Conditionals

| Directive | Effect |
|---|---|
| `@if expr` | Begin a conditional block. `expr` is integer; non-zero is true. |
| `@elif expr` | Else-if. |
| `@else` | Else. |
| `@endif` | End. |
| `@ifdef NAME` | True if NAME is defined. |
| `@ifndef NAME` | Inverse. |
| `@assert expr, "message"` | Hard fail if `expr` is zero. |

```masm
@assert cell == 8, "wfasm: this kernel requires 64-bit cells"

@if DEBUG
    int  3
@endif
```

### Iteration

| Directive | Effect |
|---|---|
| `@rept N` … `@endr` | Repeat the block N times. Inside, `@INDEX` is 0..N-1. |
| `@for x in args` … `@endfor` | Iterate over a variadic argument or a comma-separated literal list. |

```masm
@rept 4
    push  rax
@endr

@for r in {rax, rbx, rcx}
    push  &r
@endfor
```

### Macros

| Directive | Effect |
|---|---|
| `@macro NAME(p1, p2, ...)` … `@endmacro` | Define a text macro. |
| `@macro NAME(args...)` … `@endmacro` | Variadic. `&args` is the comma-joined tail. |
| `@rust_macro NAME` | Declare that NAME is implemented by a host-registered Rust closure. |
| `@local NAME` | Inside a macro body, declare an identifier visible only inside this expansion. |

See [Text macros](#text-macros) and [Rust macros](#rust-macros) below.

### Scopes

| Directive | Effect |
|---|---|
| `@scope NAME` | Open a named scope. Sets `@PROC = NAME`. Starts a local-label namespace. |
| `@endscope` | Close the innermost scope. |

Scopes nest, though it's unusual to want that in assembly.

### Inclusion

| Directive | Effect |
|---|---|
| `@include "path"` | Inline another source file. Paths are relative to the including file. |

### Diagnostics

| Directive | Effect |
|---|---|
| `@error "msg"` | Halt assembly with the given message. |
| `@warn "msg"` | Print a warning, continue. |
| `@bits N` | Sanity-check the target bit width. (Always 64 for now.) |

### Sections

| Directive | Effect |
|---|---|
| `@section NAME` | Switch to a named section. |
| `@code` / `@data` / `@rodata` / `@bss` | Shortcuts. |

### Data emission

| Directive | Effect |
|---|---|
| `@db a, b, c, ...` | One byte each. Forwards to `.byte`. |
| `@dw …` | 16-bit. |
| `@dd …` | 32-bit. |
| `@dq …` | 64-bit. |
| `@dz "string"` | Zero-terminated ASCII. |

### Externals

| Directive | Effect |
|---|---|
| `@extern NAME(arg_count)` | Declare that NAME is a function provided by the host. Calls to it are emitted as relocations resolved at JIT link time. |

The `arg_count` is required even though, on x86-64, the assembler doesn't
need it for codegen — it's a documentation aid that helps the host
double-check the registration:

```masm
@extern rt_emit(1)

    mov  rcx, rax
    call rt_emit
```

On the host side:

```rust
jit.define_extern_fn("rt_emit", 1, rt_emit as *mut c_void)?;
```

If you `call rt_emit` in asm but forget to register it from Rust, the
JIT raises an unresolved-symbol error pointing at `rt_emit`.

---

## Text macros

### Definition

```masm
@macro pushd(val)
    sub  DSP, cell
    mov  [DSP], TOS
    mov  TOS, &val
@endmacro
```

Body is asm text. Each `&NAME` is a parameter substitution slot. Each
occurrence of `&NAME` in the body is replaced by the corresponding
argument when the macro is invoked.

### Invocation

```masm
pushd(42)
pushd([rbp + cell])    ; argument with brackets is fine
pushd({1, 2, 3})       ; curly-braced argument — commas inside don't split
```

### Parameters

Parameter names follow identifier rules. The `&` prefix is *mandatory*
for substitution. Bare `name` inside a body is **not** a substitution —
it's whatever identifier `name` is in the surrounding context. This
makes shadowing impossible.

### Token pasting

`&a##b` concatenates the substituted `&a` with the substituted `&b` with
no separator and no whitespace. Useful for synthesizing names:

```masm
@macro bin_op(name, op)
    @scope &name
.globl &name
&name:
    &op  TOS, [DSP]
    add  DSP, cell
    ret
@endscope
@endmacro
```

To emit a literal `&`, write `&&`.

### Variadics

```masm
@macro save_regs(regs...)
    @for r in regs
        push  &r
    @endfor
@endmacro

save_regs(rax, rbx, rcx, r8)
```

Inside the macro, `&regs` expands to the verbatim comma-joined argument
list. `@count(regs)` returns the count. `@nth(regs, N)` extracts the
Nth (0-based) element.

### Grouping arguments that contain commas

By default `,` separates arguments at the top level. Wrap an argument in
`{ ... }` to keep its inner commas:

```masm
@macro three_things(a, b, c)
    @dq &a, &b, &c
@endmacro

three_things({1, 2}, {3, 4}, 5)    ; passes 1,2 then 3,4 then 5
```

Inside `{ ... }`, commas don't split. You can nest.

### Local names inside a macro

`@local NAME` declares a name that's visible only inside this expansion.
It's stored in the macro's expansion namespace, separate from outer
defines.

```masm
@macro reload_after(target)
    @local saved
    @assign saved = TOS
    call  &target
    @assign TOS = saved
@endmacro
```

### Hygiene summary

| What | Hygienic? |
|---|---|
| Local labels (`.label` inside a macro body) | Yes. Per invocation. |
| `@local NAME` declarations | Yes. Visible only in expansion. |
| `@define` / `@assign` at file scope | No. They're file-scoped, persist after the macro returns. |
| `@scope` / `@endscope` inside a macro | No. The scope is real, not hygienic. |

If you need a per-invocation `@assign`, prefix the name with
`_$@COUNTER`-style uniqueness explicitly.

### Composition

Macros may invoke other macros (including themselves recursively). After
parameter substitution wfasm rescans the body for further directives and
macro calls. The expansion depth is bounded (default 64) to catch
runaway recursion; depth-exceeded is a hard error pointing at the call
chain.

---

## Rust macros

When a textual macro can't model what you need — typically because the
expansion depends on arithmetic over the arguments — declare a Rust
macro and register a closure from the host.

In the source:

```masm
@rust_macro stk            ; stack-effect adjuster
```

In Rust:

```rust
use wfasm::{Assembler, RustMacroArgs, RustMacroEmit};

let mut asm = Assembler::new();

asm.register_macro("stk", |args: &RustMacroArgs, emit: &mut RustMacroEmit| {
    let in_count:  i64 = args.parse_int(0)?;
    let out_count: i64 = args.parse_int(1)?;
    let cell = emit.lookup_int("cell").unwrap_or(8);
    let delta = (out_count - in_count) * cell;
    match delta.cmp(&0) {
        std::cmp::Ordering::Greater => emit.writeln(&format!("sub rbp, {}", delta))?,
        std::cmp::Ordering::Less    => emit.writeln(&format!("add rbp, {}", -delta))?,
        std::cmp::Ordering::Equal   => {},
    }
    Ok(())
});

let text = asm.assemble_file("kernel.masm")?;
```

Rust macros see parsed arguments (already split, with `{...}` groups
collapsed), can read currently-defined names via `emit.lookup_int(...)`,
can read `@PROC` via `emit.proc_name()`, and emit asm text via
`emit.writeln(...)`. They cannot define `@macro`s or invoke text macros —
they live one layer below.

---

## Built-in context names

| Name | Expands to |
|---|---|
| `@PROC` | Name of innermost open `@scope`, or empty string. |
| `@PROC_SIZE` | Bytes emitted in current scope since `@scope NAME`. Resets per scope. |
| `@COUNTER` | Monotonically increasing integer. Bumped on every read. Useful for synthesizing unique names. |
| `@LINE` | 1-based line number in current file. |
| `@FILE` | Path of current file. |
| `@SECTION` | Name of current section. |
| `@INDEX` | 0-based index inside `@rept` / `@for` loops. |
| `@BITS` | Target bit width. Always `64` for now. |
| `@to_string(expr)` | Evaluate an integer expression and emit its decimal representation. Used for label / identifier synthesis: `.case_$@to_string(N):`. |

All read-only.

---

## Conditional assembly and assertions

### `@if` / `@elif` / `@else` / `@endif`

```masm
@if cell == 8
    @define ADJ 8
@elif cell == 4
    @define ADJ 4
@else
    @error "cell must be 4 or 8"
@endif
```

`@if` evaluates an integer expression. Zero is false; anything else is
true. Operators: `+ - * / %`, `&` `|` `^` `<< >>`, comparison `== != <
<= > >=`, logical `&& || !`, parens.

### `@ifdef` / `@ifndef`

```masm
@ifdef DEBUG
    int  3
@endif
```

### `@assert`

Runs at assembly time. Aborts assembly with your message if the
expression is zero:

```masm
@assert cell == 8, "wf64 kernel: cells must be 8 bytes"
@assert TOS_REG == "rax", "register convention drift"
```

### Host-defined values

The Rust driver can inject defines before assembly starts:

```rust
let mut asm = Assembler::new();
asm.define("DEBUG", 1);
asm.define("BUILD_REV", env!("BUILD_REV").parse().unwrap_or(0));
asm.assemble_file("kernel.masm")?;
```

In the source you see `@if DEBUG`, `@dq BUILD_REV` — same syntax as
in-source `@define` / `@assign`. Build flags don't need a header file.

---

## Iteration

### `@rept N` / `@endr`

Repeat a block N times. `@INDEX` is the 0-based iteration counter:

```masm
@rept 4
    mov  qword [rdi + @INDEX * 8], rsi
@endr
```

If you want a non-trivial limit:

```masm
@assign WORDS = 16
@rept WORDS
    mov  qword [rdi + @INDEX * 8], 0
@endr
```

### `@for x in list` / `@endfor`

Iterate over a list — either a variadic arg or a literal list:

```masm
@macro save_regs(regs...)
    @for r in regs
        push  &r
    @endfor
@endmacro

@for r in {rax, rbx, rcx}
    push  &r
@endfor
```

`@INDEX` is the 0-based iteration index inside the loop body.

---

## MASM-style runtime control flow

Compile-time `@if` decides whether the assembler emits some text at
all. The directives in this section are different: they're an
ergonomic sugar over the `cmp` + `jcc` + label sequences that
appear over and over in hand-written asm. Each directive expands
into a few asm instructions that you'd otherwise type out yourself.
They cost zero runtime — they ARE the runtime; they're just less
fiddly to read and write than naming every skip-label by hand.

The dialect is borrowed from MASM, kept deliberately close to the
syntax MASM32 programmers spent two decades writing.

### `.if` / `.elseif` / `.else` / `.endif`

```masm
.if rax == 0
    mov rbx, 1
.elseif rax == 1
    mov rbx, 2
.else
    mov rbx, 3
.endif
```

Each branch's condition expands to `cmp lhs, rhs` followed by the
inverse `jcc` jumping past the branch body. The `.endif` closes the
chain with the end label every branch falls through to.

**Supported operators:** `==`, `!=`, `<`, `<=`, `>`, `>=`. Signed
semantics (`jl`/`jle`/`jg`/`jge` for the strict-and-non-strict
comparisons). Unsigned variants not yet modelled — write your own
`cmp` + conditional jump if you need them.

**Operands** are arbitrary text MC understands: registers, memory
operands with size prefixes, immediates, labels. Both sides are
pre-expanded through the macro engine, so `@define TOS rax` plus
`.if TOS == 0` works.

### `.while` / `.endw`

Pre-test loop. The body runs while the condition is true.

```masm
.while rcx > 0
    add rax, rcx
    dec rcx
.endw
```

Expands to:

```asm
__while0_top:
    cmp rcx, 0
    jle __while0_bot
    add rax, rcx
    dec rcx
    jmp __while0_top
__while0_bot:
```

### `.repeat` / `.until`

Post-test loop. The body runs at least once, then the condition is
tested at the bottom; loop exits when the condition becomes TRUE
(MASM semantics — the *opposite* of C's `do/while`).

```masm
.repeat
    dec rax
.until rax == 0
```

### `.break` / `.continue`

Inside the innermost open loop:

* `.break` jumps past `.endw` / `.until` — exits the loop.
* `.continue` jumps to the loop's continuation point — to the top
  (re-testing the condition) for `.while`, or to the post-test for
  `.repeat`.

```masm
.while rax > 0
    .if rax == 5
        .break        ; jump out of the .while
    .endif
    dec rax
.endw
```

### Nesting and scope-vs-block interaction

Runtime blocks nest freely with each other and with `@scope`,
`@macro`, `@include`. The block stack is separate from the scope
stack; each `.if`/`.while`/`.repeat` is closed by its matching
terminator regardless of any `@scope`s opened in between.

A `@macro` that opens a runtime block must close it inside the same
macro (otherwise the user's call site is left with an open block in
a confusing state). An unclosed block at end of expansion is a hard
error pointing at the unclosed opener.

### Generated label names

Labels are stamped with a per-block id and an internal prefix
(`__if<n>_end`, `__while<n>_top`, `__rep<n>_cont`, etc.). The user
shouldn't reference them — they're internal — but they show up in
`-S` style dumps if you ever look.

### Worked example — the Hutch tribute

```masm
@macro proc(name)
    @scope &name
    .globl &name
&name:
@endmacro

@macro endp()
    ret
    @endscope
@endmacro

; Return min(rax, rcx).
proc(min2)
    .if rax > rcx
        mov rax, rcx
    .endif
endp()

; Sum 1..rcx into rax. Hutch-style do-while.
proc(sum_up_to)
    xor rax, rax
    .while rcx > 0
        add rax, rcx
        dec rcx
    .endw
endp()
```

The MASM32 community spent years pushing this style — high-level
control-flow syntax over straight asm, where the assembler is just
a typing aid for the compare+branch idiom. Same machine code as a
hand-written version; an order of magnitude less label bookkeeping
for the programmer.

---

## Calling external (Rust) functions

The two halves:

### In source

```masm
@extern rt_emit(1)
@extern rt_key(0)

proc(emit)
    mov  rcx, TOS           ; Win64 arg1
    mov  TOS, [DSP]
    add  DSP, cell
    call rt_emit
    ret
endp
```

### In Rust

```rust
extern "C" fn rt_emit(ch: u64) -> u64 {
    print!("{}", ch as u8 as char);
    0
}

let mut jit = Jit::new("kernel")?;
jit.define_extern_fn("rt_emit", 1, rt_emit as *mut c_void)?;
jit.add_asm(&text)?;
```

The `arg_count` in `@extern` must match `define_extern_fn`. wfasm
double-checks at JIT-bind time and errors clearly if they disagree.

### Win64 ABI for external calls

The user calls a Rust function from Forth-primitive context (where RAX
= TOS, RBP = DSP). Win64 requires:

- args in RCX, RDX, R8, R9 (then stack)
- RSP 16-byte aligned at the `call`
- 32 bytes of shadow space below the call

None of that is automatic. Write a macro:

```masm
@macro win64_call(target)
    mov     r12, rsp        ; R12 is callee-saved; safe parking
    and     rsp, -16        ; align to 16
    sub     rsp, 32         ; shadow space
    call    &target
    mov     rsp, r12        ; restore
@endmacro
```

And use it:

```masm
proc(emit)
    mov     rcx, TOS
    mov     TOS, [DSP]
    add     DSP, cell
    win64_call(rt_emit)
    ret
endp
```

The frame is opt-in. Forth primitives that don't call out don't pay
for it.

---

## Worked example — Forth STC primitives

`forth-macros.masm`:

```masm
@assert cell == 8, "wf64 requires 64-bit cells"

@define TOS  rax
@define DSP  rbp
@define UP   rbx

; ── Begin a Forth STC primitive (no prologue, no frame). ──
@macro proc(name)
    @scope &name
    .globl  &name
&name:
@endmacro

@macro endp()
    @endscope
@endmacro

@macro next()
    ret
@endmacro

; Push an immediate onto the data stack.
@macro pushd(val)
    sub  DSP, cell
    mov  [DSP], TOS
    mov  TOS, &val
@endmacro

; A binary op between TOS and NOS, result in TOS, drop one.
@macro binop(name, op)
    proc(&name)
        &op  TOS, [DSP]
        add  DSP, cell
        next
    endp
@endmacro

; A comparison against zero: TOS → (0 if false, -1 if true).
@macro cmp_to_zero(name, setcc)
    proc(&name)
        test    TOS, TOS
        &setcc  al
        movzx   TOS, al
        neg     TOS
        next
    endp
@endmacro

; Win64 callout wrapper.
@rust_macro stk

@macro win64_call(target)
    mov     r12, rsp
    and     rsp, -16
    sub     rsp, 32
    call    &target
    mov     rsp, r12
@endmacro
```

`kernel.masm`:

```masm
@include "forth-macros.masm"

@code
    .intel_syntax noprefix

; ── Arithmetic ──
binop(plus,  add)
binop(minus, sub)
binop(and_,  and)
binop(or_,   or)
binop(xor_,  xor)

; ── Comparisons ──
cmp_to_zero(zero_equal,   setz)
cmp_to_zero(zero_less,    setl)
cmp_to_zero(zero_greater, setg)

; ── Stack ops ──
proc(dup)
    stk 1, 2
    mov  [DSP], TOS
    next
endp

proc(drop)
    stk 1, 0
    mov  TOS, [DSP]
    next
endp

proc(swap)
    stk 2, 2
    xchg TOS, [DSP]
    next
endp

; ── I/O via Rust ──
@extern rt_emit(1)
@extern rt_key(0)

proc(emit)
    mov     rcx, TOS
    mov     TOS, [DSP]
    add     DSP, cell
    win64_call(rt_emit)
    next
endp

proc(key)
    stk 0, 1
    mov  [DSP], TOS
    win64_call(rt_key)
    mov  TOS, rax    ; rt_key returned in rax; same register, but explicit
    next
endp
```

Driver:

```rust
use std::ffi::c_void;
use wfasm::{Assembler, Jit};

extern "C" fn rt_emit(ch: u64) -> u64 {
    print!("{}", ch as u8 as char);
    0
}
extern "C" fn rt_key() -> u64 {
    let mut buf = [0u8; 1];
    use std::io::Read;
    std::io::stdin().read_exact(&mut buf).ok();
    buf[0] as u64
}

fn main() -> anyhow::Result<()> {
    let mut asm = Assembler::new();
    asm.define("cell", 8);
    asm.register_macro("stk", wfasm::macros::stk_adjust);
    let text = asm.assemble_file("kernel.masm")?;

    let mut jit = Jit::new("kernel")?;
    jit.define_extern_fn("rt_emit", 1, rt_emit as *mut c_void)?;
    jit.define_extern_fn("rt_key", 0, rt_key as *mut c_void)?;
    jit.add_asm(&text)?;

    // Run some Forth — assume an `interpret` primitive exists.
    type Interpret = extern "C" fn();
    let interp: Interpret = unsafe { jit.lookup_fn("interpret")? };
    interp();
    Ok(())
}
```

---

## Worked example — Win64 callout macro

When you do want a Win64 ABI prologue — say you're writing a leaf
function in asm called by a C library:

```masm
@macro win64_proc(name)
    @scope &name
.globl &name
&name:
    push    rbp
    mov     rbp, rsp
    sub     rsp, 32             ; shadow space for callees
@endmacro

@macro win64_endp()
    add     rsp, 32
    pop     rbp
    ret
    @endscope
@endmacro

@extern puts(1)

win64_proc(say_hello)
    lea     rcx, [rip + .msg]
    call    puts
win64_endp()

@rodata
.msg:   @dz "hello from wfasm"
```

Note `.msg` is scope-local to `say_hello` — emitted as `say_hello$$msg`
in the object — but only the scope owner can `lea` it. If you need a
truly global label, write it without the leading dot.

---

## Worked example — a tight memchr loop

To prove it's not Forth-specific:

```masm
@extern none()        ; no-op extern to satisfy our checker

@code
    .intel_syntax noprefix

@macro tight_proc(name)
    @scope &name
.globl &name
&name:
@endmacro

@macro tight_endp()
    @endscope
@endmacro

; uint8_t *wf_memchr(uint8_t *haystack, size_t len, uint8_t needle);
; Win64: rcx=haystack, rdx=len, r8=needle, returns ptr or null in rax.
tight_proc(wf_memchr)
    mov     rax, rcx
    add     rcx, rdx        ; end pointer
1:  cmp     rax, rcx
    je      .not_found
    cmp     byte [rax], r8b
    je      .found
    inc     rax
    jmp     1b
.found:
    ret
.not_found:
    xor     rax, rax
    ret
tight_endp()
```

GAS-style numeric `1:` and `1b` are passed straight through to MC. The
two `.found` / `.not_found` labels are scope-local so they won't collide
with anything else.

---

## Gotchas and traps

### Macro arguments are token sequences, not strings

`pushd(1 + 2)` passes the three tokens `1`, `+`, `2` to the macro body.
The body uses `&val` which substitutes those three tokens. The final
expression is given to MC, which evaluates `1 + 2` with full precedence
and emits `mov rax, 3`. No defensive parenthesization needed.

### Two `.label`s with the same name in the same scope collide

Inside one `@scope NAME` body, `.foo:` defined twice is an error. Inside
a `@macro` body, two separate invocations get distinct mangling — they
do NOT collide. If you need two same-named labels in one scope, use
GAS-style numeric labels (`1:` / `1b` / `1f`) which can repeat.

### `@define foo bar` then redefine `foo` is silently allowed

Later definitions shadow earlier ones. This matches the pragmatic
"defines are config" use case. If you want a "define-once" name, use
`@assign` after checking `@ifdef foo @error ...`.

### `@extern` and `define_extern_fn` arg counts must match

```masm
@extern rt_emit(1)
```
```rust
jit.define_extern_fn("rt_emit", 1, ...)?;
```

A mismatch is a hard error at `add_asm` time. wfasm tracks every
`@extern` declaration and checks against `define_extern_fn` at the
binding step.

### Macros are reentrant but expansion depth is bounded

If you write `@macro a() a() @endmacro` and invoke `a()`, you'll get a
"maximum macro expansion depth (64) exceeded" error. The default 64 can
be raised: `Assembler::with_max_depth(256)`.

### `@scope` must be balanced before assembly ends

If `@scope NAME` is open at end of file with no `@endscope`, that's a
hard error. The error points at the unclosed `@scope`. Same for `@if`,
`@macro`, `@rept`, `@for`.

### Built-in directive names are reserved

`@macro @scope ...` is a hard error — `@scope` is built-in.

### Whitespace in token pasting

`&a##b` is the *paste* operator. `&a ## b` (with spaces) is the same.
`&a&b` (no spaces) is a syntax error: wfasm refuses to guess whether
you meant to paste or to write two adjacent substitutions.

### MASM hex vs MC hex

MASM accepts `0FFh`. LLVM MC does not. Use `0xFF`. (`@define`s and
`@assign`s also use `0x` form.)

---

## Integration with the wfasm JIT

The full pipeline:

```rust
use std::ffi::c_void;
use wfasm::{Assembler, Jit};

fn build_and_call() -> anyhow::Result<()> {
    // 1. Assembler — pure text in, text out. No LLVM contact.
    let mut asm = Assembler::new();
    asm.define("cell", 8);
    asm.register_macro("stk", wfasm::macros::stk_adjust);
    let text = asm.assemble_file("kernel.masm")?;

    // 2. JIT — text in, function pointer out.
    let mut jit = Jit::new("kernel")?;
    jit.define_extern_fn("rt_emit", 1, my_emit as *mut c_void)?;
    jit.add_asm(&text)?;
    jit.declare_fn("forth_main", 0)?;

    // 3. Lookup + call.
    type Main = extern "C" fn();
    let f: Main = unsafe { jit.lookup_fn("forth_main")? };
    f();
    Ok(())
}
```

`Assembler` doesn't depend on LLVM. You can assemble to a string, write
that string to disk, and inspect what MC would see. `Jit` is the LLVM
layer; it takes the string from the assembler.

This separation means an AOT mode is a small addition: instead of
`Jit::add_asm`, you write the asm string to a `.s` file and run
`clang -c` to produce an object. The assembler doesn't change.

---

## Calling the Windows API

JASM ships a generator that turns Microsoft's official Win32 metadata
into JASM bindings. After running the generator, JIT'd code can call
any of ~3,700 Windows API functions by name. About 3,200 of them get
a full **`invoke`-style wrapper macro** so the call site is one line —
no manual register placement, no shadow-space arithmetic. New SDK
from Microsoft → re-run the generator → bindings update.

The MASM32 community spent years writing `invoke MessageBoxW, hwnd,
addr text, addr capt, MB_OK` and getting the right machine code out.
v2 of the generator emits the same shape: `MessageBoxW(hwnd, text,
caption, MB_OK)` expands to the four register moves, the shadow-space
prologue, the call, and the epilogue — generated from the WinMD type
information for that specific function.

### The bitstream pipeline

```
Windows.Win32.winmd               (Microsoft — binary metadata, ECMA-335)
        │                          ships in Microsoft.Windows.SDK.Win32Metadata NuGet
        ▼  (C# importer, run once by the upstream NewM2 project)
E:\windows_api\windows_api.db     SQLite, 18,271 functions, 60,780 params, 37,830 types
        │                          primitives pre-resolved: DWORD→u32, HANDLE→usize, …
        ▼  scripts/win32_gen.py    (reads param types, emits @extern + @macro wrappers)
E:\JASM\rust\win32\*.masm         11 DLLs, 3,757 functions, one file per DLL
        │
        ▼  @include "win32/kernel32.masm"
@extern "KERNEL32.dll" GetTickCount64(0)
@macro GetTickCount64()
    mov r12, rsp
    and rsp, -16
    sub rsp, 32
    call GetTickCount64
    mov rsp, r12
@endmacro
        │
        ▼  wfasm::win32::bind_externs(&asm, &mut jit, host_resolver)
LoadLibraryW + GetProcAddress + LLVMAddGlobalMapping
        │
        ▼  MCJIT + RTDyld
JITed `call GetTickCount64` resolves to the real Windows function.
```

**No marshalling layer.** Args go straight into Win64 register slots
via the generated wrapper. There is no dispatcher, no boxing, no
libffi. The instruction the integrated assembler emits is the same
one a hand-written assembler would produce — direct `call <symbol>`
that RTDyld resolves to the address `GetProcAddress` returned.

### Regenerating bindings

```
python scripts/win32_gen.py                          # default core (12 DLLs)
python scripts/win32_gen.py --dll USER32.dll         # one specific DLL
python scripts/win32_gen.py --all                    # every DLL in the DB
```

Output: one `win32/<dll>.masm` per requested DLL. Idempotent — re-runs
overwrite in place. Each file pairs an `@extern` with an
`invoke`-style wrapper macro per function:

```masm
; Generated from windows_api.db — DLL: USER32.dll
; v2: @extern declarations + invoke-style wrapper macros.

@extern "USER32.dll" MessageBoxW(4)
@macro MessageBoxW(arg0, arg1, arg2, arg3)
    mov     r12, rsp           ; save the caller's rsp (callee-saved on Win64)
    and     rsp, -16           ; align to 16
    sub     rsp, 32            ; shadow space
    mov     rcx, &arg0         ; HWND      — pointer-width
    mov     rdx, &arg1         ; PCWSTR    — pointer
    mov     r8,  &arg2         ; PCWSTR    — pointer
    mov     r9d, &arg3         ; MESSAGEBOX_STYLE — u32 (32-bit alias of r9)
    call    MessageBoxW
    mov     rsp, r12
@endmacro
```

Functions whose signatures can't be modelled (struct-by-value params,
variadics, exotic primitives) get the `@extern` declaration only —
the user calls those by hand with `mov`s + a `win64_call` helper. The
current split: **~3,177 wrapped, ~580 extern-only across 11 DLLs**.

The generator filters out `A`-family duplicates when a `W` variant
exists (modern code only wants `W`), and skips variadic functions.

### Using a binding from source

```masm
    .intel_syntax noprefix
    .text

@include "win32/kernel32.masm"     ; brings @extern + @macro per function

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
    GetTickCount64()        ; <-- the wrapper handles everything:
                            ;     shadow space, alignment, the call,
                            ;     RSP restore. rax holds the result.
endp()
```

That's a complete Forth-style primitive calling a Win32 function. No
manual `mov r12, rsp; and rsp, -16; sub rsp, 32; call …` — the
wrapper macro the generator produced does that. **Calling a Win32
function is one line of source.** This is the `invoke` ergonomics
MASM32 was famous for, reproduced from Microsoft's own metadata.

For functions with arguments, the wrapper places them in the right
register or stack slot per the param's type:

```masm
proc(say_hello)
    ; MessageBoxW(hwnd, text, caption, type)
    ;             rcx   rdx   r8       r9d (u32)
    MessageBoxW(
        0,                          ; hwnd = NULL
        [rip + .text_str],          ; text pointer
        [rip + .caption_str],       ; caption pointer
        0)                          ; MB_OK
endp()

@rodata
.text_str:    @dz "hello from JIT"
.caption_str: @dz "JASM"
```

(`@dz` is planned — for now use `.asciz "..."` which MC accepts directly.)

The `@include` brings every KERNEL32 declaration into scope, but only
functions actually `call`ed in source produce relocations the JIT must
resolve. Including thousands of unused declarations costs a small
amount of MCJIT setup time (registering global mappings) but no
runtime overhead per call.

### Binding from the host

```rust
use std::ffi::c_void;
use wfasm::{Assembler, Jit};

fn main() -> anyhow::Result<()> {
    let mut asm = Assembler::new();
    let asm_text = asm.assemble("kernel.masm", source_text)?;

    let mut jit = Jit::new("kernel")?;

    // For each @extern "DLL.dll" NAME(N), the helper does:
    //   - LoadLibraryW("DLL.dll")           (cached per DLL)
    //   - GetProcAddress("NAME")
    //   - jit.define_extern_fn(name, N, addr) → LLVMAddGlobalMapping
    // The host_resolver closure is consulted only for @externs that
    // have NO DLL string — i.e., the host's own Rust runtime fns.
    let report = wfasm::win32::bind_externs(&asm, &mut jit, |name| -> Option<*mut c_void> {
        match name {
            "rt_emit" => Some(rt_emit as *mut c_void),
            _         => None,
        }
    })?;

    println!("bound {} externs ({} unresolved)",
             report.bound, report.missing_proc.len());

    jit.add_asm(&asm_text)?;
    jit.declare_fn("get_ticks", 0)?;

    type GetTicks = extern "C" fn() -> u64;
    let f: GetTicks = unsafe { jit.lookup_fn("get_ticks")? };
    println!("ticks = {}", f());
    Ok(())
}
```

### Tolerant binding

`bind_externs` does NOT fail when `GetProcAddress` returns null for a
declared function. Different Windows versions export slightly different
sets; the generator emits the union, and the binder skips what isn't
present on the host. Missing names are recorded in `BindReport`:

```rust
let report = wfasm::win32::bind_externs(&asm, &mut jit, host_resolver)?;
for (name, dll, win_error) in &report.missing_proc {
    eprintln!("  not exported: {dll}!{name} (GetLastError={win_error})");
}
```

A `LoadLibraryW` failure IS fatal — if `USER32.dll` itself won't load,
nothing using it will work. A host-resolver returning `None` for a
non-DLL extern is also fatal — the host should know exactly which
Rust functions it provides.

If JITed code actually `call`s a missing function, RTDyld surfaces the
"unresolved symbol" error when the module materializes — not silently
at the call site.

### Worked example — `hello-win32`

The crate ships a binary that exercises the whole pipe:

```
$ cargo run --bin hello-win32
assembler saw 1165 @extern declarations
bound 1158 externs (7 unresolved)
GetTickCount64() #1 = 9998109
GetTickCount64() #2 = 9998125
delta              = 16 ms
```

1,165 `@extern` declarations from `win32/kernel32.masm`, of which 1,158
resolved on this Windows install. The source calls `GetTickCount64()`
as a one-liner — the generated wrapper handled the Win64 ABI; MCJIT
linked the call to the real Windows function via `LoadLibraryW` +
`GetProcAddress` + `LLVMAddGlobalMapping`. Two calls 20 ms apart
returned tick counts 16 ms apart (Windows tick timer granularity).

Source: `src/bin/hello_win32.rs`. The whole Forth-side primitive is:

```masm
proc(get_ticks)
    GetTickCount64()
endp()
```

Three lines.

### File map

```
E:\JASM\rust\
├── scripts/win32_gen.py        ← Python generator (~200 lines)
├── src/win32.rs                ← bind_externs + LoadLibraryW + GetProcAddress
├── win32/                      ← generated bindings (committed, regen via scripts/)
│   ├── kernel32.masm           1,165 @extern declarations
│   ├── user32.masm               625
│   ├── gdi32.masm                384
│   ├── advapi32.masm             467
│   ├── shlwapi.masm              217
│   ├── comctl32.masm             110
│   ├── comdlg32.masm              11
│   ├── ole32.masm                273
│   ├── shell32.masm              210
│   ├── winmm.masm                151
│   └── ws2_32.masm               144      (3,757 functions total)
└── src/bin/hello_win32.rs      ← worked example

E:\windows_api\windows_api.db   ← upstream SQLite, not part of JASM
                                  built by NewM2 from Microsoft's WinMD
```

### Limits in v2

- **Argument register interference is not solved.** If you pass register
  operands in patterns that clobber as the wrapper writes them
  (`MessageBoxW(rdx, rcx, ...)` etc.), the second `mov` reads a
  just-modified register. MASM32's `invoke` had a reorder pass for
  this; we don't yet. In practice Forth code passes memory operands and
  immediates, where it doesn't bite. Workaround: spill to scratch
  registers before invoking.
- **No variadics.** Functions like `wsprintfW` are skipped. Win64 ABI
  for variadics has rules (every "..." arg goes to the stack AND to a
  GPR in the float case) the generator doesn't model.
- **No struct-by-value args.** Functions with by-value struct params
  get the `@extern` declaration but no wrapper. The DB has the
  layouts; a future generator could synthesize `defstruct`-style
  wrappers. Today, pass pointers to structs you allocate yourself.
- **No callbacks.** Functions that take a `WNDPROC`-style callback
  need a JIT-emitted trampoline matching the Win64 ABI for that
  signature. Out of scope for v2.
- **Stack args don't size-extend.** The wrapper emits `mov qword ptr
  [rsp+N], &argi` for stack args regardless of declared type. If the
  user passes a 32-bit value, the upper 32 bits of the stack slot may
  contain garbage. Workaround: zero the slot first or pass a properly
  sized operand.

None of these block the common case of "call a flat Win32 function
that takes scalars and pointers."

---

## Crash dumps and breakpoints

JITed code crashes. Often, while you're writing it. JASM ships a
process-wide crash dumper (Windows-only for now) that catches every
exception, prints registers + stack + symbolic context, and either
continues (for `int 3`) or lets the failure propagate to the
debugger (for everything else).

### Installing

```rust
wfasm::seh::install().expect("install SEH handler");
```

Idempotent. Call once at startup, before any JIT'd code runs. This
adds a Vectored Exception Handler that runs first for any exception
in the process — access violations, illegal instructions,
divide-by-zero, `int 3` breakpoints, the lot.

### Registering symbols

For RIP and stack values to print symbolic names, register what you
know about:

```rust
// JIT-emitted procs: looked up via the engine.
wfasm::seh::register_jit_procs(&mut jit, &["forth_main", "plus", "minus"])?;

// Win32 imports already resolved by bind_externs: feed them in.
let win32_entries: Vec<(String, u64, &'static str)> = my_table
    .iter()
    .map(|(name, addr)| (name.to_string(), *addr as u64, "win32"))
    .collect();
wfasm::seh::register_many(win32_entries);
```

Symbol resolution is "nearest predecessor" — for a queried address,
find the highest registered `addr` that's `<= query`. The dumper
prints `<name+offset>` if the nearest match is within 64 KiB, blank
otherwise. The 64 KiB window is generous enough for big Win32
functions, tight enough to avoid spurious attributions to random
stack data.

### `int 3` is non-fatal

The handler treats `STATUS_BREAKPOINT` specially:

1. Dump state — exception kind, RIP, all 16 GPRs, RFLAGS, 32 qwords
   of stack with symbolic resolution.
2. Advance RIP past the 1-byte `0xCC`.
3. Return `EXCEPTION_CONTINUE_EXECUTION` — execution resumes at the
   next instruction.

Sprinkle `int 3` (or define a `@macro brk() int 3 @endmacro` for
ergonomics) anywhere you want to inspect state. The function keeps
running; you see registers and stack at that exact point.

```masm
proc(diagnose_me)
    mov     rax, 0xDEAD     ; ← a value we want to see
    int     3               ; ← dump fires here, then continues
    mov     rax, 0xCAFE     ; ← still runs
endp()
```

### Everything else aborts (after dumping)

For access violations / illegal instructions / divide-by-zero / etc.,
the handler dumps and returns `EXCEPTION_CONTINUE_SEARCH`, so the
debugger or default handler also sees the failure. The process
typically dies — you get a clean dump first.

```
┌─── JASM JIT crash dump ───────────────────────────────────────────────
│ exception : 0xC0000005  ACCESS_VIOLATION
│ at RIP    : 000001764E840013  <crash_demo+0x3> [jit_proc]
│ access type: read at address 0000000000000000
│
│ rax = 000001764E840010   rbx = 000001764E875AE0
│ rcx = 0000000000000000   rdx = 000001764E840010
│ rsi = 00007FF7DBDC99C8   rdi = 000000716FAFFE50
│ rbp = 000000716FAFF670   rsp = 000000716FAFF5E8
│ r8  = 0000000000000011   r9  = 000000000000000B
│ r10 = 000001764E960000   r11 = 8101010101010100
│ r12 = 0000000000000000   r13 = 0000000000000000
│ r14 = 0000000000000000   r15 = 0000000000000000
│ flags = 00010246
│
│ stack (32 qwords from rsp):
│  [rsp+  0] 000000716FAFF5E8 00007FF7DBD55465
│  [rsp+ 16] 000000716FAFF5F8 000001764E869E80
│  ...
│  [rsp+104] 000000716FAFF650 000001764E840010  <crash_demo> [jit_proc]
│  ...
└───────────────────────────────────────────────────────────────────────
```

The dumper resolves RIP, marks values that look like return
addresses with their proc-and-offset, and lists every GPR. Reading
the stack as a Forth return stack ("which proc called this?") gets
you most of the way to a usable trace without a real stack walker.

### What's not here

- **No frame-by-frame stack walk.** A proper unwinder needs
  `.pdata`/`.xdata` entries and `RtlAddFunctionTable` registration,
  which in turn need LLVM to emit `uwtable` per function — and our
  module-level-inline-asm pattern doesn't have IR functions for LLVM
  to attach `uwtable` to. The literal stack dump compensates: return
  addresses are visible to your eyes, even if the OS unwinder can't
  walk them.
- **No source-line attribution.** RIP becomes `<proc_name+offset>`,
  not `kernel.masm:127`. Would need DWARF or PDB emission; deferred.
- **Single-step is suppressed.** `STATUS_SINGLE_STEP` (0x80000004)
  returns `CONTINUE_SEARCH` without dumping, so it doesn't fight a
  debugger's step machinery.

### `hello-seh` worked example

The crate ships `src/bin/hello_seh.rs`:

```
$ cargo run --bin hello-seh
SEH installed, 2 symbols registered. running brk_demo...

┌─── JASM JIT crash dump ───────────────────────────────────────────────
│ exception : 0x80000003  BREAKPOINT (int 3)
│ at RIP    : 0000028564810007  <brk_demo+0x7> [jit_proc]
│ rax = 000000000000DEAD   ...
└───────────────────────────────────────────────────────────────────────

brk_demo returned 0xCAFE (expected 0xCAFE)

now triggering an access violation. dump will print, then process aborts:

┌─── JASM JIT crash dump ───────────────────────────────────────────────
│ exception : 0xC0000005  ACCESS_VIOLATION
│ at RIP    : 000001764E840013  <crash_demo+0x3> [jit_proc]
│ access type: read at address 0000000000000000
│ ...
└───────────────────────────────────────────────────────────────────────
[process aborts]
```

That's two procs: one with a deliberate `int 3` that continues after
the dump, and one that derefs null. Both produce clean symbolic
dumps; the first proves the "dump and keep going" path, the second
the "dump and let the process die" path.

---

## What this guide doesn't cover (yet)

These are designed but not in v1:

- **DWARF / `.cfi_*`** for stack unwinding in debuggers. Forth STC code
  doesn't have unwind info, period; calling Rust code that might panic
  will tear the process down. Acceptable until something cares.

- **PE-style `.pdata` / `.xdata`** for Win64 SEH. Same issue, different
  ABI. NewBCPL's `jit_mm.rs` shows the pattern when we need it.

- **`@align N`** / **`@times N db value`** for padding and fills. NASM-shaped.

- **`@once`** include guards. `@ifdef` works around it.

- **AOT compilation** — emit a `.o` file directly. The assembler is
  already factored so this is mostly LLVM glue.

- **Cross-architecture targets** — wfasm is x86-64-only. AArch64 would
  mean a different instruction set passes through to MC, plus revised
  register conventions for the macro library. The assembler core
  doesn't care; it's the user's macros that lock x86-64 in.

When any of these stop being acceptable, they become a one-paragraph
addition to this guide.
