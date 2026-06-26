# JASM

A JIT macro-assembler for x86-64 Windows. Source in, function pointer
out. Brings MASM32-era ergonomics to a modern LLVM-MC + MCJIT pipeline,
exposes the entire Win32 API by name, and ships a crash dumper for when
your hand-written asm goes sideways.

```masm
.intel_syntax noprefix
.text

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
    GetTickCount64()      ; <- a real Win32 call. The wrapper macro
                          ;    (generated from Microsoft's WinMD
                          ;    metadata) handles register placement,
                          ;    shadow space, and 16-byte RSP alignment.
endp()
```

```rust
let mut jit  = wfasm::Jit::new("demo")?;
let mut asm  = wfasm::Assembler::new();
let text     = asm.assemble("demo.masm", SOURCE)?;
wfasm::win32::bind_externs(&asm, &mut jit, |_| None)?;
jit.add_asm(&text)?;
let f: extern "C" fn() -> u64 = unsafe { jit.lookup_fn("get_ticks")? };
println!("ticks = {}", f());          // prints a real Windows tick count
```

That's the whole flow.

## What is JASM?

JASM is three things stacked on each other:

1. **A MASM-flavoured macro assembler.** Line-oriented Intel-syntax
   source with a real macro language: text macros with hygienic
   per-invocation labels, Rust-implemented macros for things text
   substitution can't model, `@scope`/`@endscope` for label namespacing,
   `@include` for multi-file projects, `.if`/`.while`/`.repeat` for
   runtime control flow that compiles to plain `cmp` + `jcc` + label.

2. **An LLVM-MC + MCJIT runtime.** Instruction encoding, branch
   relaxation, and instruction-size selection are delegated to LLVM's
   integrated assembler. MCJIT compiles the module in-process; you
   get an `extern "C"` function pointer you can call from Rust. No
   linker, no `.o` file, no `.exe` to ship.

3. **A Win32 bridge.** A Python generator reads Microsoft's official
   `Windows.Win32.winmd` metadata (via the SQLite database the sibling
   NewM2 project already builds) and emits one `.masm` file per DLL.
   3,757 Win32 functions become callable by name. Per-function wrapper
   macros place arguments in Win64 register slots automatically, in
   the spirit of MASM32's `invoke`.

Plus a crash dumper because none of the above is going to work first
time.

## Why JASM?

The original project was a 64-bit Forth — a port of Alex McDonald's
WF32 STC Forth (Also here somewhere) to x64. The first attempt followed WF32's model: use
WF32 itself as the bootstrap host, metacompile through WF32's own
`asmx64-core.fs` to produce `wf64.exe`. That hit two walls. The
assembler had untested 64-bit paths; the metacompiler manipulated
4-byte image offsets that needed widening to 8 bytes everywhere; the
combined surface felt bigger than the original Forth, although that may just have been us.

The pivot: write an assembler in Rust, delegate encoding to LLVM,
JIT instead of metacompile to a PE32+. That replaces three things
with one. But it raised a new question — how do you write Forth-style
assembly in Rust without losing the hand-tooled feel that made the 
project worth porting?

The answer became JASM:

- **Every byte you emit traces to text you wrote or a macro you
  defined.** No implicit prologue, no implicit calling convention, no
  register policy baked into the assembler. The user defines
  `proc(name)` and `endp()` as macros, picks the data-stack register,
  picks the calling convention they want for callouts. The assembler
  is a typing aid; you choose the conventions.

- **Macros where macros help, Rust where they don't.** Text macros
  handle ~95% of the shapes that come up. The rest — stack-effect
  arithmetic, conditional emission based on runtime state, anything
  that needs to compute over arguments — live in Rust closures
  registered against the assembler. Hutch's MASM32 community pushed
  every problem they could into text macros; we have one more tool
  in the box for the cases that strained that limit.

- **Microsoft maintains the type info.** Hand-curated `.inc` files of
  Win32 prototypes were a MASM32 community burden. The WinMD bitstream
  is Microsoft's gift: 18,271 functions with full parameter types, kept
  in sync with the SDK by the people writing the SDK. JASM consumes
  this directly; new Windows release → re-run the generator → bindings
  update.

- **MC encodes, not us.** Modern LLVM's integrated assembler is better
  tested by orders of magnitude than any hand-written x64 encoder.
  Branch relaxation (`jmp short` vs `jmp near`), operand-size selection,
  REX prefix handling — all already correct. We don't reinvent.

If MASM32 + LLVM + Microsoft's WinMD existed in 2005, this is roughly
what I imagine might have come out of Hutch's forum. Although maybe not.

## What you can do with it

**Write x64 instructions, exactly as MASM would have shown them:**

```masm
mov     rax, 42
add     rax, [rbp + 8]
jne     somewhere
```

**Use Forth-style register conventions:**

```masm
@define TOS  rax
@define DSP  rbp
@define cell 8

@macro binop(name, op)
proc(&name)
    &op  TOS, [DSP]
    add  DSP, cell
    next()
endp()
@endmacro

binop(plus,  add)
binop(minus, sub)
binop(and_,  and)
binop(or_,   or)
```

**Write some HLA style runtime control flow:**

```masm
proc(min2)
    .if rax > rcx
        mov rax, rcx
    .endif
endp()

proc(sum_up_to)
    xor rax, rax
    .while rcx > 0
        add rax, rcx
        dec rcx
    .endw
endp()
```

`.if`/`.elseif`/`.else`/`.endif` and `.while`/`.endw` and
`.repeat`/`.until` and `.break`/`.continue` — they compile to plain
`cmp` + `jcc` + label sequences with internal labels. Signed
comparisons (`==`, `!=`, `<`, `<=`, `>`, `>=`). Conditions get
pre-expanded through the macro engine, so `.if TOS == 0` substitutes
defines correctly.

**Call any Win32 function:**

```masm
@include "win32/user32.masm"

proc(say_hello)
    MessageBoxW(
        0,                  ; HWND
        [rip + .text],      ; PCWSTR
        [rip + .capt],      ; PCWSTR
        0)                  ; MESSAGEBOX_STYLE (MB_OK)
endp()

@rodata
.text: .asciz "hello from JIT"
.capt: .asciz "JASM"
```

The generator emits a wrapper macro per function that places args in
the right register or stack slot per Win64 ABI. ~3,177 of 3,757
exposed functions have wrappers; the rest (struct-by-value params,
variadics) get the `@extern` declaration only — call those by hand.

**Call your own Rust functions:**

```masm
@extern rt_emit(1)

proc(emit)
    mov     rcx, rax        ; first Win64 arg = TOS
    mov     rax, [rbp]      ; pop
    add     rbp, 8
    win64_call(rt_emit)
endp()
```

```rust
extern "C" fn rt_emit(ch: u64) -> u64 {
    print!("{}", ch as u8 as char);
    0
}

jit.define_extern_fn("rt_emit", 1, rt_emit as *mut c_void)?;
```

**Drop into the debugger non-fatally:**

```masm
proc(diagnose_me)
    mov     rax, 0xDEAD     ; ← inspect this value
    int     3               ; ← dump fires, then execution continues
    mov     rax, 0xCAFE     ; ← still runs
endp()
```

The SEH handler catches `int 3` as `STATUS_BREAKPOINT`, prints
registers + stack + symbolic context, advances RIP past the byte,
returns `CONTINUE_EXECUTION`. For access violations and other
non-breakpoint faults, the same dump prints and the process aborts so
your debugger gets the exception too.

```
┌─── JASM JIT crash dump ─────────────────────────────────────────────
│ exception : 0x80000003  BREAKPOINT (int 3)
│ at RIP    : 0000028564810007  <diagnose_me+0x7> [jit_proc]
│
│ rax = 000000000000DEAD   rbx = 00000285645E6620
│ rcx = 0000000000000000   rdx = 0000028564810000
│ ...
│ stack (32 qwords from rsp):
│  [rsp+ 208] 000000A8E45DF668 0000028564810000  <diagnose_me> [jit_proc]
│  ...
└─────────────────────────────────────────────────────────────────────
```

## Status

| | |
|---|---|
| Library tests | **139** all passing |
| Smoke binaries | **4** all passing |
| Forth kernel | **in progress** — `forth/` directory; Phase 1 primitives complete |
| Win32 functions exposed | **3,757** across 11 DLLs |
| Win32 functions with `invoke`-style wrappers | **3,177** |
| Target platforms | x86-64 Windows (LLVM 22.x) **and Apple Silicon (macOS arm64, LLVM-free)** |
| Status | Assembler complete on both targets. ANS Forth kernel underway in `forth/`. |

Smoke binaries:

```
$ cargo run --bin hello-jit       # mov rax, 42; ret  →  prints 42  (x86, LLVM)
$ cargo run --bin hello-runtime   # JIT calls a Rust function and back (x86, LLVM)
$ cargo run --bin hello-win32     # JIT calls GetTickCount64 by name  (x86, LLVM)
$ cargo run --bin hello-seh       # int 3 dumps and continues; segfault dumps and aborts
$ cargo run --bin wf64            # ANS Forth interpreter (requires forth/kernel.masm)
$ cargo run --bin hello-aarch64   # AArch64 JIT calls a Rust function (Apple Silicon, no LLVM)
```

## Apple Silicon (macOS arm64)

JASM runs natively on Apple Silicon with **no LLVM dependency at runtime**. The
default build (`cargo build`, the `llvm` feature *off*) compiles a from-scratch
AArch64 assembler + a `MAP_JIT` loader:

```
.masm source → wfasm::asm (macro engine, arch-neutral)
     ▼
wfasm::a64::A64Encoder        ← native AArch64 encoder (text → machine code),
     │                          byte-identical to LLVM-MC, no LLVM linked
     ▼
wfasm::native_macos::MacJit   ← mmap(MAP_JIT) + per-thread W^X toggle
     │                          + sys_icache_invalidate + far-call veneers
     ▼
extern "C" fn pointer         ← call from Rust
```

```
$ cargo build                          # native arm64 build, no LLVM
$ cargo test --lib                     # 182 tests (incl. a 1,181-form corpus replay)
$ cargo run --bin hello-aarch64
from JIT: 42
forth_main() = 84
```

- **`src/a64/`** — the native AArch64 encoder (parser, encoder, two-pass driver
  with conditional-branch relaxation). Covers the practical Apple-Silicon
  user-space ISA: integer (ALU/mul/div/shift/bitfield/csel/ccmp/adc/crc),
  loads/stores (incl. FP/SIMD, pairs, exclusives, LSE atomics), control flow,
  scalar FP, the full NEON Advanced-SIMD set, AES/SHA crypto, and system/barriers.
- **`src/native_macos.rs`** — `MacJit`, the macOS loader (the AArch64 sibling of
  the Windows `NativeJit`). `MAP_JIT` works in a plain `cargo test` binary; the
  hardened-runtime `com.apple.security.cs.allow-jit` entitlement is only needed
  for a signed, distributed binary.
- **The oracle, kept honest.** Every encoded form is gated byte-for-byte against
  LLVM-MC (`aarch64-apple-darwin`) via the `difftest` harness, and frozen into
  `corpus/aarch64.tsv` (1,181 forms). The replay test re-verifies the encoder
  with **no LLVM installed**; regenerate with `cargo run --bin a64-corpus
  --features llvm` after extending coverage.

See [docs/design/aarch64-apple-silicon.md](docs/design/aarch64-apple-silicon.md)
for the full design, phasing, and the verified encoding reference.

## Quick start

```
$ git clone …
$ cd JASM/rust
$ cargo build
$ cargo test               # 139 tests
$ cargo run --bin hello-jit
forth_main() = 42

$ cargo run --bin hello-win32
assembler saw 1165 @extern declarations
bound 1158 externs (7 unresolved)
GetTickCount64() #1 = 9998109
GetTickCount64() #2 = 9998125
delta              = 16 ms
```

To regenerate Win32 bindings from a newer SDK:

```
$ python scripts/win32_gen.py --all   # everything in windows_api.db
$ python scripts/win32_gen.py --dll USER32.dll GDI32.dll
```

## Architecture

```
.masm source
     │ wfasm::asm::lex          ← tokens (file/line/col-stamped)
     │ wfasm::asm::expand       ← macros, scopes, .if/.while, @include resolved
     │ wfasm::asm::emit         ← MC-flavor Intel-syntax asm string
     ▼
LLVM MC integrated assembler   ← encoding, branch relaxation
     │
     ▼
LLVM MCJIT + RTDyld            ← compile, link, allocate, finalize
     │
     │ wfasm::jit::Jit          ← safe wrapper, symbol lookup
     │ wfasm::win32::bind_externs ← LoadLibraryW + GetProcAddress
     │ wfasm::seh::install      ← VEH crash dumper
     ▼
extern "C" fn pointer          ← call from Rust
```

LLVM-C bindings are hand-written (~25 functions, ~150 LOC) against the
shipped `LLVM-C.dll`. No `llvm-sys`, no `inkwell` — LLVM 22 is too new
for the published crate versions and the C API is stable enough that
hand-writing the bindings we need was cheaper than version-pinning a
third-party wrapper.

## Project layout

```
JASM/rust/
├── Cargo.toml
├── build.rs                    finds LLVM-C.lib, copies LLVM-C.dll to target/
├── README.md                   this file
├── USER-GUIDE.md               1,700 lines — the language reference
├── PLAN-MACROS.md              design notes
│
├── src/
│   ├── lib.rs                  module root + re-exports
│   ├── llvm.rs                 hand-written LLVM-C FFI bindings
│   ├── jit.rs                  MCJIT wrapper (Jit, JitError)
│   ├── win32.rs                bind_externs, BindReport (Windows-only)
│   ├── seh.rs                  Vectored Exception Handler + dump (Windows-only)
│   │
│   ├── asm/                    the assembler
│   │   ├── source.rs           SourceMap + FileId
│   │   ├── span.rs             Span (file, line, col, len)
│   │   ├── token.rs            Token, TokenKind, Punct, NumberLit, StringLit
│   │   ├── error.rs            AsmError family
│   │   ├── lex.rs              line-oriented lexer
│   │   ├── expr.rs             integer expression evaluator (for @assign, @if, .if)
│   │   ├── macros.rs           MacroDef, ScopeFrame, MC-directive recognition,
│   │   │                       built-in stk Rust macro
│   │   ├── expand.rs           directives + scopes + macros + .if/.while/.repeat
│   │   │                       + @include + @extern + @rust_macro
│   │   └── emit.rs             token stream → MC-flavor string
│   │
│   └── bin/
│       ├── hello_jit.rs        mov rax, 42; ret  →  prints 42
│       ├── hello_runtime.rs    JIT ↔ Rust runtime fn
│       ├── hello_win32.rs      JIT calls GetTickCount64 by name
│       └── hello_seh.rs        int 3 dump + access violation dump
│
├── scripts/
│   └── win32_gen.py            Python generator: WinMD SQLite → .masm files
│
├── forth/                      ANS Forth kernel source (STC, 64-bit)
│   ├── macros.masm             register aliases, head()/endword() macros, win64_call
│   ├── primitives.masm         ~47 code words: stack, arith, compare, logic
│   ├── memory.masm             @, !, C@, C!, 2@, 2!, +!, FILL, MOVE, CMOVE, CMOVE>
│   ├── rstack.masm             >R, R>, R@, RDROP, 2>R, 2R>, 2R@
│   ├── io.masm                 EMIT, KEY, SPACE, CR, TYPE
│   ├── user-area.masm          UP offsets + BASE, STATE, >IN, HERE, ALLOT, , etc.
│   ├── end-of-kernel.masm      forth_last_link sentinel (include last)
│   └── kernel.masm             top-level @include (TODO: interpreter + compiler)
│
└── win32/                      generated bindings, regen via scripts/win32_gen.py
    ├── kernel32.masm           1,048 wrapped, 117 extern-only
    ├── user32.masm               511 wrapped, 114 extern-only
    ├── gdi32.masm                328 wrapped,  56 extern-only
    ├── advapi32.masm             412 wrapped,  55 extern-only
    ├── shlwapi.masm              212 wrapped,   5 extern-only
    ├── comctl32.masm              46 wrapped,  64 extern-only
    ├── comdlg32.masm              11 wrapped,   0 extern-only
    ├── ole32.masm                241 wrapped,  32 extern-only
    ├── shell32.masm              167 wrapped,  43 extern-only
    ├── winmm.masm                 65 wrapped,  86 extern-only
    └── ws2_32.masm               136 wrapped,   8 extern-only
                                ────────────────────────────────
                                3,177 wrapped, 580 extern-only

External (not in this repo):
E:\windows_api\windows_api.db    29 MB SQLite — the WinMD bitstream
                                  built by NewM2 from Microsoft's
                                  Microsoft.Windows.SDK.Win32Metadata NuGet
```

## Building

Prerequisites:

- **Rust 1.95+** (any recent stable).
- **LLVM 22.x** installed at `C:\Program Files\LLVM\` (or wherever —
  set `LLVM_DIR` to the install root if different). We link against
  `LLVM-C.lib` and run against `LLVM-C.dll`; `build.rs` copies the
  DLL next to your binaries.
- **Python 3.x** with `sqlite3` (stdlib) if you want to regenerate
  Win32 bindings. Not required to build or run.
- **Microsoft Visual C++ Build Tools** for the MSVC linker (default
  Rust toolchain on Windows already needs this).
- **`E:\windows_api\windows_api.db`** if regenerating bindings.
  The committed `win32/*.masm` work without it.

```
$ cargo build           # builds the library + all 4 binaries
$ cargo test            # 138 lib tests
$ cargo run --bin hello-jit
```

## Documentation

- **[USER-GUIDE.md](USER-GUIDE.md)** — the language reference: every
  directive, every built-in, the macro engine, the Windows binding
  pipe, the crash dumper. ~1,700 lines; copy-paste-runnable examples
  throughout.
- **[PLAN-MACROS.md](PLAN-MACROS.md)** — design notes from the macro
  language's first cut. Useful background on why the assembler is
  shaped the way it is.

## Acknowledgments

JASM stands on a stack of generous prior work:

- **Steve Hutchesson (hutch)** and the MASM32 community spent two
  decades proving that asm + Win32 + a good macro layer was a
  complete environment. Every ergonomic choice in JASM points back
  to that lineage — `proc/endp` shape, `invoke`-style wrappers, the
  `.if`/`.while` family, `int 3` as an inspector. The MASM32
  distribution's `.inc` files were the conceptual ancestor of
  `win32/*.masm`.
- **Alex McDonald's WF32** is the Forth this project was originally
  meant to port. Its STC architecture, register conventions
  (RAX=TOS, RBP=DSP, RBX=UP), and stack-effect adjustment idiom
  remain the model for the eventual Forth kernel.
- **NewM2** (the sister Modula-2 compiler project) did the work of
  importing Microsoft's `Windows.Win32.winmd` metadata into
  `windows_api.db`. JASM consumes that database directly; the C#
  importer and the SQLite normalization were already paid for.
- **NewCormanLisp** (also sibling) showed the pattern for consuming
  the WinMD database to generate per-namespace bindings, and the
  custom MCJIT memory manager (`jit_mm.rs`) it ships will be the
  template if JASM ever needs full `RtlAddFunctionTable` SEH unwinding.
- **NewBCPL** (sibling again) was the working reference for the
  MCJIT-with-module-level-inline-asm pattern. Their use of
  `LLVMAppendModuleInlineAsm` + IR `declare` + RTDyld was the proof
  that the symbol-resolution approach JASM uses would work.
- **LLVM** for the integrated assembler, MCJIT, RTDyld, and the C API
  that makes hand-writing minimal FFI bindings practical.
- **Microsoft** for shipping the SDK metadata as a machine-readable
  binary blob in the first place. Without WinMD, the Win32 surface
  in JASM would be hundreds of hand-curated `.inc` files maintained
  by someone/noone.

## Limitations

- **Two backends: x86-64 Windows (LLVM) and Apple Silicon (native).**
  The Windows path uses LLVM-MC + MCJIT and the Win32 bindings; the
  macOS arm64 path is the LLVM-free `a64` encoder + `MacJit` (see the
  [Apple Silicon](#apple-silicon-macos-arm64) section). Linux x86-64
  works as an LLVM oracle target but has no native loader yet. `win32`
  and `seh` are `#[cfg(windows)]`; `native_macos` is `#[cfg(macos)]`.
- **AArch64 gaps (deferred, niche).** Single-structure *lane* load/store
  (`ld1 {v0.s}[2], …`), the SVE/SME vector extensions, and the AArch64
  crash dumper are not yet implemented. Out-of-range conditional branches
  are relaxed (inverted + `b`) — a deliberate extension beyond LLVM-MC,
  which errors on them.
- **No DWARF or PDB.** Crash dumps resolve to `<proc+offset>`, not to
  source lines. Adding `.cfi_*` macros and proper line directives is
  a future feature.
- **No frame-walking unwinder.** The crash dumper prints the literal
  stack contents; the OS unwinder can't walk JIT frames without
  `.pdata`/`.xdata` and `RtlAddFunctionTable` registration, which in
  turn needs IR-level function definitions (not just declares).
  Tractable but deferred.
- **No COM.** Flat exports only. COM's vtable model is a separate
  design pass.
- **No struct-by-value Win32 args** in the wrapper macros. ~580 of
  3,757 functions fall in this bucket and get the `@extern`
  declaration without a wrapper — call them by hand.
- **No callbacks** from Win32 → JIT'd code yet. Needs a per-signature
  JIT-emitted trampoline. Future.
- **No EXE** should have mentioned that first?, just add it if you want it.

None of these block the common case of writing x64 assembly that
calls into the OS or into Rust.

## License

MIT — see the top-level [LICENSE](../LICENSE) file.
