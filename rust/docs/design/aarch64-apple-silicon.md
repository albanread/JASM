# AArch64 / Apple Silicon Port — Master Design

Status: **Phase 0–1 complete and committed** (oracle bring-up); Phase 2+ designed.
Companion to `docs/design/rasm-difftest.md` (the differential harness this reuses)
and `docs/design/rasm-replace-llvm.md` (the native-encoder strategy this mirrors).

This document is kept in sync with HEAD. Where it says **✅ done** the code is on
the `macos-arm64` branch with passing tests; everything else is the plan.

---

## 1. Goal & strategy

Bring JASM (`wfasm`) to Apple Silicon (macOS arm64) by running the *same play*
the project already ran for x86-64:

1. **Stand up the LLVM-MC + MCJIT path as the AArch64 ground truth** — the
   oracle. LLVM cross-assembles AArch64 in-process and we parse the Mach-O it
   emits into the existing `EncodedModule` shape. **✅ done** (§3.1–§3.3).
2. **Build a native, LLVM-free `Aarch64Encoder`**, gated **byte-for-byte** against
   that oracle through the *already arch-neutral* `difftest` harness, closing the
   instruction frontier family-by-family exactly as the x86 `RasmEncoder` was
   (§3.4–§3.5).
3. **Build a macOS native loader** — `mmap(MAP_JIT)` + the per-thread W^X toggle +
   icache invalidation + far-call veneers — replacing the Windows
   `VirtualAlloc2`/`VirtualProtect` loader (§3.6).

The leverage: the author built the seams for this. The macro front-end, the
`Encoder`/`Loader` backend traits, the diff driver, and corpus record/replay are
all architecture-neutral and compile + pass on macOS unchanged.

### Verification baseline (this branch, on an arm64 Mac, LLVM 22.1.2)

| Build | Result |
|---|---|
| `cargo test --lib` (default, **no LLVM**) | **165 passed** — front-end + x86 rasm + corpus replay |
| `cargo test --lib --features llvm` | **172 passed** — adds the oracle path, incl. the full 3393-form x86 differential (cross-assembled) and the new AArch64 oracle tests |

The `--features llvm` run **loads and executes** `libLLVM` at runtime (the oracle
tests call into it), so it proves dyld resolution of the rpath/reexport chain, not
just linking.

---

## 2. What ports unchanged (the arch-neutral core)

| Component | Why it's neutral |
|---|---|
| Front-end macro engine `asm/` (lex, expand, expr, emit) | `emit.rs` is a pass-through token→text serializer; the macro/expr engine never inspects mnemonics. The **lexer** keeps AArch64 *compound tokens* whole (`read_ident_chars` consumes an interior `.` for `b.<cond>`/`v0.8b` and an interior `@` for reloc specifiers `sym@PAGE`/`sym@PAGEOFF`), so the `@scope` mangler and the expr evaluator don't mistake a condition-suffix / arrangement / reloc-suffix for a local label or directive. A `.`/`@` at token start is still a local-label / directive. Only the **macro library** (`forth/*.masm`) is x86 (§3.7). |
| Backend seam `backend.rs` | `Encoder`/`Loader`/`EncodedModule`/`Reloc` are target-agnostic; only `RelocKind` grew variants (§3.3). |
| Diff driver `difftest.rs` | Compares two `EncodedModule`s; `Verdict`/`Report`/`diff_*` name no arch. One change needed: the field-mask becomes per-kind bitmasks (§3.5.2). |
| Corpus record/replay | JSONL/TSV of `{asm, masked-code, relocs}`; a new `corpus/aarch64.tsv` is gated by the *same* replay test with **zero driver changes**. |
| Oracle plumbing `oracle.rs`/`llvm.rs` | `LlvmMcEncoder::with_triple` was already multi-arch; LLVM-MC + `object` are multi-target/multi-format. |

---

## 3. Per-arch / per-OS edges

### 3.1 `build.rs` — LLVM discovery ✅ done

Per-OS dispatch on `CARGO_CFG_TARGET_OS`:

- **macOS/Linux** (`link_unix_dylib`): resolve the LLVM root from `LLVM_DIR` →
  `llvm-config --prefix` → Homebrew default `/opt/homebrew/opt/llvm`; emit
  `cargo:rustc-link-lib=dylib=LLVM-C` + `-rpath <libdir>`.
- **Windows** (`link_windows`): unchanged — link `LLVM-C.lib`, copy `LLVM-C.dll`.

**Mechanism, not a question (per review):** Homebrew's `libLLVM-C.dylib` is an
~89 KB pure *reexport shim* (`otool -L` → `@rpath/libLLVM.dylib (reexport)`); it
exports **zero** `LLVMInitialize*` symbols itself — the real symbols live in the
~160 MB `libLLVM.dylib`. So we link `-lLLVM-C` for the stable C-API surface and
add the rpath so the reexported `libLLVM` resolves at load time. There is no "or
link `LLVM` directly" choice to make.

> Distribution caveat: the embedded rpath is `/opt/homebrew/opt/llvm/lib`. A
> shipped `--features llvm` binary fails to launch with a dyld error if LLVM is
> absent/moved. This only affects the **oracle/test** build; the native shipping
> build (default features) has no LLVM dependency.

### 3.2 `llvm.rs` — AArch64 target init ✅ done

`init_targets_and_mcjit()` registers **both** X86 and AArch64
(`LLVMInitializeAArch64{TargetInfo,Target,TargetMC,AsmParser,AsmPrinter}`) and
links MCJIT, idempotently. `init_x86_mcjit()` is kept as an alias. Both backends
are present in every standard LLVM distribution, so the externs resolve on all
platforms.

### 3.3 `oracle.rs` — Mach-O / AArch64 oracle ✅ done

| Concern | Resolution |
|---|---|
| Target init | calls `init_targets_and_mcjit()` |
| Syntax | `triple_is_x86()` gates `.intel_syntax noprefix`; AArch64 passes GAS text through verbatim (`ensure_trailing_newline`) |
| CPU features | `+avx512*` for x86 only; empty for AArch64 (base ARMv8-A) |
| Text section | `section_by_name(".text")` → `"__text"` → first `SectionKind::Text` |
| Symbol names | **kept verbatim** on every format (see below) |
| Reloc map | Mach-O ARM64 types classified first (§3.3.1) |
| Constructors | `x86_64()` (pins `x86_64-pc-windows-msvc` — host-independent x86 diff) and `aarch64_macos()` (pins `aarch64-apple-darwin`) |

**Symbols are verbatim — do NOT strip leading `_`.** Verified: `.globl w` →
Mach-O symbol `w`; `.globl _w` → `_w`. The leading underscore is a *C-compiler*
convention, not an assembler one, so rasm (verbatim labels) and the oracle agree
without mangling. The only place the asymmetry is real is the **`dlsym`
extern-binding path** at runtime (§3.6.6), where `dlsym("malloc")` resolves the
exported `_malloc` — a loader concern, scoped there, not an object-parse rule.

#### 3.3.1 `RelocKind` extension ✅ done — the minimal, sound set

`backend.rs` adds exactly three variants; `Abs64` is shared:

```rust
Branch26,        // ARM64_RELOC_BRANCH26   — b/bl imm26, PC-rel <<2
AdrpPage21,      // ARM64_RELOC_PAGE21     — adrp immlo[30:29]/immhi[23:5]
AddPageOff12,    // ARM64_RELOC_PAGEOFF12  — add/ldr imm[21:10]
// Abs64       <- ARM64_RELOC_UNSIGNED(len 8) — shared with x86 .quad sym
```

Deliberately **excluded** (the map errors fail-loud on them — correct for
bring-up; revisit when a form needs them):

- **No `scale` field on `AddPageOff12`.** `ARM64_RELOC_PAGEOFF12` carries no
  scale; the *consuming instruction* encodes its access size and the linker reads
  it from the instruction. Scale is an **encoder** concern.
- **No per-kind `addend`.** `Reloc.addend` already exists and is populated from
  `rel.addend()`. `ARM64_RELOC_ADDEND` flows through that single field — no second
  source of truth.
- **No GOT / SUBTRACTOR variants.** `GOT_LOAD_PAGE21/PAGEOFF12`, `SUBTRACTOR`,
  standalone `ADDEND` → `map_reloc` returns `Err`. Our `@PAGE`/`@PAGEOFF` forms
  never emit them; SUBTRACTOR pairs only appear for **cross-section** deltas. When
  needed, add the map arms *and* the paired-reloc lookahead loop together (today's
  one-reloc-per-iteration loop is correct precisely because pairs are rejected
  upstream).

### 3.4 `rasm/` → a new AArch64 encoder

New sibling module (e.g. `src/rasm_a64/` or `src/a64/`) implementing `Encoder`.
The x86 `rasm/` is the template, but AArch64 is *structurally simpler* in encoding
(every instruction is one little-endian 32-bit word — no REX/ModRM/SIB/prefixes/
variable length) and *harder* in branch ranges (fixed widths → veneers).

#### 3.4.1 Parser register & addressing model

- **Registers:** `x0–x30` (+ `sp`/`xzr` aliasing #31 by *context*), `w0–w30`
  (+`wsp`/`wzr`), SIMD/FP `v0–v31` with views `q/d/s/h/b`. Extend `RegClass`.
- **#31 is SP or XZR by operand position**, and that choice changes the encoding
  *family*: `add x0, sp, x1` uses the **extended-register** form, not the
  shifted-register form. The parser records the token; the encoder picks the
  family per position (§3.4.2). This must be in scope from the first ALU phase.
- **Addressing modes:** `[xn]`, `[xn, #imm]`, `[xn, #imm]!` (pre-index),
  `[xn], #imm` (post-index), `[xn, xm{, lsl #s}]`, `[xn, wm, (u|s)xtw {#s}]`.
- **PC-relative symbols:** `adrp xN, sym@PAGE` + `add/ldr ..., sym@PAGEOFF`
  (the AArch64 analogue of x86 `lea [rip+sym]`).

#### 3.4.2 Fixed-width encoder

Per-format word builders (one `u32`, then `to_le_bytes()`): `DPImm` (add/sub/
logical-imm with bitmask-immediate encoding, `movz/movk/movn`), `DPReg`
(shifted- and extended-register), `LoadStore` (scaled `imm12` unsigned-offset and
unscaled `imm9` pre/post), `Branch` (imm26/imm19/imm14), `PCRel` (adr/adrp). No
NOP-padding table needed (`.p2align` in a code section pads with `0xD503201F`).

**Invariant to exploit:** every instruction that writes a W register zeroes the
top 32 bits of the X register — true for *all* 32-bit-result ops, not just loads.
The macro port must not sprinkle defensive `uxtw`.

#### 3.4.3 Two-pass driver: range-check + veneers (replaces rel8→rel32 relaxation)

x86 grows branches rel8→rel32. AArch64 widths are fixed; out-of-range targets are
handled by **rewriting**, under a strict **grow-only** discipline for termination:

- **Reach:** B/BL ±128 MB (imm26), B.cond/CBZ/CBNZ ±1 MB (imm19), TBZ/TBNZ
  ±32 KB (imm14), ADR ±1 MB, ADRP ±4 GB @ 4 KB page.
- **Conditional out of imm19:** invert + skip-over an unconditional:
  `b.<inv> +8 ; b far` (grows the site by 4 bytes).
- **Unconditional out of imm26 / extern:** emit a **veneer** —
  `adrp x16, t@PAGE ; add x16, x16, t@PAGEOFF ; br x16` (**12 bytes**, full
  ±4 GB range) and retarget the branch to it. x16/x17 are the ABI's
  intra-procedure-call scratch registers (IP0/IP1).
- **Termination:** *pessimistic pre-sizing* — any branch that *might* overflow is
  reserved at worst-case size on pass 1, then only ever shrinks (or never
  shrinks). Grow-only converges; optimistic-shrink can oscillate (inverting a
  cond adds 4 bytes, which can push another branch past imm26). Commit to
  grow-only.

> Veneer sizing is **12 bytes (the ADRP form)** everywhere — loader stub
> accounting must use 12, not the 20-byte `movz/movk×4; br` alternative.

#### 3.4.4 Encoder validation (no `iced` analogue)

`iced-x86` is x86-only and the *secondary* check anyway. On AArch64 the **primary**
LLVM-MC oracle is the full gate (`disasm: None`). Optional secondary decode later:
Capstone or `LLVMCreateDisasm` (both multi-target, behind the existing
`Disassembler` trait) — never leak a decoder past that trait.

### 3.5 `difftest/` — `Aarch64Model` + `corpus/aarch64.tsv`

#### 3.5.1 The form generator

`Aarch64Model: IsaModel` with `triple() = "aarch64-apple-darwin"` (**Mach-O,
one triple** — recording the golden under ELF "because reloc names are easier"
would encode the wrong object semantics). Product of small tables: register banks
(x/w with a low + a high-numbered representative to exercise the reg fields,
v/d/s), addressing-mode templates, immediate buckets (incl. the
bitmask-immediate-legal vs -illegal edges for logical-imm; the `movz/movk` shift
buckets), and a mnemonic catalog seeded from the kernel's needs.

#### 3.5.2 Masking: per-kind **bit** masks, not byte ranges

x86 `mask_relocs` zeroes `size` contiguous LE bytes — wrong for AArch64, where the
immediate is *bit-packed inside the 32-bit word* alongside opcode/register bits.
Replace with a per-`RelocKind` **word bitmask** clearing only immediate bits:

| Kind | Cleared bits (within the LE word) |
|---|---|
| `Branch26` | `imm26` = bits [25:0] |
| `AdrpPage21` | `immlo` [30:29] + `immhi` [23:5] |
| `AddPageOff12` | `imm12` [21:10] |

**Key simplification (verified from the emitted object):** LLVM emits reloc sites
with the immediate bits already **zero** (`bl`→`94000000`, `adrp`→`90000000`,
`add@PAGEOFF`→`91000000`). If the encoder likewise leaves placeholders zeroed
(the contract), reloc-bearing forms are **directly byte-comparable** and the
bitmask is a safety net, not a correctness crutch. This also means a naïve raw
byte golden on a *symbolic* form is fine **only because both sides are zero** — but
the corpus must still record through the masked path so an encoder that fills a
field partially is caught, and so opcode bits (e.g. `B` `0x14` vs `BL` `0x94`,
and the placeholder bit in ADRP's high byte) are preserved by the *bit* mask.

#### 3.5.3 Record/replay

`difftest --record --triple aarch64-apple-darwin` (needs `--features llvm`) writes
`corpus/aarch64.tsv`; the existing `corpus_replay_matches_golden`-style `#[test]`
gates rasm against it with **no LLVM** — same code, new data file.

### 3.6 Native loader (`native.rs` → a macOS sibling)

`native.rs` is `#![cfg(windows)]`. Add a `#[cfg(target_os = "macos")]` sibling
(e.g. `native_macos.rs`) behind the same `Loader` impl; re-export per-OS from
`lib.rs`.

#### 3.6.1 Allocation
`mmap(NULL, cap, PROT_READ|PROT_WRITE|PROT_EXEC, MAP_ANON|MAP_JIT, -1, 0)`. A
`MAP_JIT` region is mapped **RWX once** and stays RWX in the page tables;
protection is then controlled *only* by the per-thread W^X toggle (below) — there
is **no per-cycle `mprotect`**. (Note: plain RWX `mmap` *without* `MAP_JIT` is
**denied** under Hardened Runtime + `allow-jit`; `MAP_JIT` is mandatory — see
§3.6.8.)

#### 3.6.2 W^X — per-thread toggle only
Guard with `pthread_jit_write_protect_supported_np()`; if unsupported, the toggle
is a no-op and the platform path must fall back (or refuse). The cycle:

```
pthread_jit_write_protect_np(false);   // this thread: region writable
… copy code + apply relocations …
pthread_jit_write_protect_np(true);    // this thread: region executable
sys_icache_invalidate(base, len);      // DC clean → IC invalidate
```

Constraints (per SDK): the toggle is **per-thread** — the thread that will
*execute* must itself be in exec mode; a second thread cannot flip on the writer's
behalf. No non-local control flow (no `longjmp`/unwind) may cross the write
window. The newer `pthread_jit_write_with_callback_np` API and the
`com.apple.security.cs.jit-write-allowlist` entitlement family are alternatives if
the bare toggle is disallowed by the chosen signing posture (§3.6.8).

#### 3.6.3 Icache invalidation — mandatory
ARM has split I/D caches; freshly written code is **not** guaranteed visible to the
fetch unit. `sys_icache_invalidate(void*, size_t)` is **public and documented**
(`<libkern/OSCacheControl.h>`, available since 10.5) — not "undocumented." It
performs the data-cache clean + instruction-cache invalidate; the executing thread
still needs the implicit `ISB` (the toggle/return provides ordering). Call it after
writes are visible and before execution.

#### 3.6.4 Far-call veneers
The x86 loader routes >2 GB targets through `movabs rax,t; jmp rax`. The AArch64
analogue appended after the code, **one 12-byte stub per distinct target**:
`adrp x16, t@PAGE ; add x16, x16, t@PAGEOFF ; br x16` (data-built as 3 words with
the page/pageoff of `t` computed against the stub address). Stub accounting uses
**12 bytes** (matches §3.4.3).

#### 3.6.5 Relocation patching
The `finalize` reloc match handles `Branch26`/`AdrpPage21`/`AddPageOff12`/`Abs64`
by **inserting bits into the target word** (read word, OR in the field, write
back) — not by writing N LE bytes. Range checks per kind; over-range `Branch26`
→ veneer (§3.6.4); over-range `AdrpPage21` is impossible within ±4 GB of any sane
arena, else error.

#### 3.6.6 Extern binding — `dlopen`/`dlsym`
Replace `LoadLibraryW`/`GetProcAddress` with `dlopen`/`dlsym`. `dlsym` takes the
**C name without underscore** (`dlsym(h,"malloc")` finds `_malloc`) — this is the
one place the underscore convention is handled, at resolution time. Frameworks via
`dlopen("/System/Library/Frameworks/Foo.framework/Foo", RTLD_NOW)`. Externs land
at libSystem/Homebrew addresses **routinely >128 MB** from the arena, so the
far-call/GOT path is the **common** case for externs — not a rarity.

#### 3.6.7 Crash dumper — **deferrable**
`seh.rs` (Windows VEH) → a POSIX `sigaction(SIGTRAP/SIGILL/SIGSEGV/SIGBUS)`
handler or a Mach exception port. `int 3` → `brk #0`. Dump x0–x30/sp/pc, advance
`pc` past the `brk` on SIGTRAP to continue. If it touches JIT pages it must obey
the no-non-local-control-flow rule (§3.6.2). Not on the critical path.

#### 3.6.8 Code arena + entitlements
`arena.rs` (`CodeArena`) must also use `MAP_JIT` (RWX without `MAP_JIT` is denied
under Hardened Runtime + `allow-jit`; arbitrary RWX needs the heavier
`allow-unsigned-executable-memory` we want to avoid). **Open decision** (resolve,
don't defer): adopt `com.apple.security.cs.allow-jit` + `MAP_JIT` + per-thread
toggle (recommended), and codesign test/JIT binaries with that entitlement.

### 3.7 Calling convention (AAPCS64) + Forth macro library

A new `forth-a64/` macro library (the existing `forth/*.masm` is x86: TOS=rax,
DSP=rbp, win64_call, `rep movsb`, `cqo/idiv`, `push/pop`). Sketch:

- **Register roles:** TOS=x19, DSP=x20, UP=x21 (callee-saved x19–x28 so they
  survive AAPCS64 calls); scratch x9–x15; args x0–x7; LR x30; FP x29; SP
  16-byte-aligned.
- **`a64_call(fn)`:** args in x0–x7 (no shadow space, no stack home area unlike
  Win64); SP stays 16-aligned; v0–v7 for FP args; results x0/x1.
- **Idiom rewrites:** `push/pop`→`str/ldr` with `[sp,#-16]!`/`[sp],#16`;
  `cqo;idiv`→`sdiv`+`msub` for the remainder; `rep movsb`→ldr/str loop;
  `mov eax,[mem]` zero-extend → `ldr w0,[..]` (automatic). Per the project's
  philosophy, this lives in *user* macros, not the assembler.

---

## 4. Phased plan

Each phase is independently verifiable. Oracle first; encoder frontier closes
family-by-family against the corpus, exactly like the x86 effort.

| Phase | Scope | Verify | State |
|---|---|---|---|
| 0 | Repo builds + x86 tests pass on macOS | `cargo test --lib` 165 ✅ | **done** |
| 1 | AArch64 oracle bring-up (build.rs, llvm.rs, oracle.rs, RelocKind, x86_64()/aarch64_macos()) | `cargo test --lib --features llvm`; AArch64 byte-identity + BRANCH26 reloc tests | **done** |
| 1b | **Native encoder first slice + direct oracle differential** (`src/a64/`): moves (mov/mvn/movz/movk/movn), ALU (add/sub reg+imm+shifted, and/orr/eor), loads/stores (uoff), branches (b/bl/b.cond/cbz/cbnz internal + extern bl reloc), br/blr | `a64::oracle_diff` — 40+ forms + whole-function loop + extern reloc all byte-identical to oracle | **done** |
| 3 | Encoder: bitmask-immediate logical (`and/orr/eor/ands/tst #imm`, full `processLogicalImmediate` port) + `mov #imm` lowering (movz→movn→orr) | `alu_extended_families_match_oracle` | **done** |
| 4 | Encoder: extended-register `add/sub` for SP, adds/subs + cmp/cmn/tst/neg, mul/madd/msub/mneg/smull/umull, sdiv/udiv, variable+immediate shifts (lslv…/UBFM/SBFM), bitfield (ubfx/sbfx/ubfiz) + extends (sxt*/uxt*), conditional select (csel/cset/…) | `alu_extended_families_match_oracle` | **done** |
| 5 | Encoder: loads/stores — byte/half/signed, pre/post-index imm9, LDUR/STUR, register offset `[Rn,Rm,ext #s]`, `ldp/stp` | `loads_stores_match_oracle` | **done** |
| 5b | Encoder: misc data-processing — `clz`/`cls`/`rbit`/`rev`/`rev16`/`rev32`, `extr`/`ror #imm`, `bfi`/`bfxil` | `misc_dataproc_match_oracle` | **done** |
| 8 | pc-relative: `adrp Xd, sym@PAGE` + `add Xd, Xn, sym@PAGEOFF` → `AdrpPage21`/`AddPageOff12` relocs (always relocated, local + extern); `@`-in-symbol parsing | `pcrel_adrp_add_matches_oracle` (code + reloc list) | **done** |
| 2 + 9 | **`Aarch64Model` (`IsaModel`) generator + committed `corpus/aarch64.tsv` + no-LLVM replay gate** (`src/difftest/aarch64.rs`, `a64-corpus` bin, `corpus_replay_aarch64_matches_golden`) | 1159 forms; `diff_model` 0 mismatch / 0 gap; replay green **without LLVM** | **done** |
| 5c | Encoder: **scalar FP** — arith (fadd/fmul/fdiv/fmadd…), compare (fcmp/fccmp/fcsel), convert (fcvt, fcvtzs/scvtf…), fmov gpr↔fp + FP imm8 | `fp_scalar_match_oracle` | **done** |
| 5d | Encoder: **NEON SIMD** — 3-same int/logical/FP, 2-misc int/FP, across-lane reductions, dup (elem + GP) | `simd_neon_match_oracle` (60+ forms) | **done** |
| 5e | Encoder: **integer tail** — ccmp/ccmn, adc/sbc/ngc, smulh/umulh, crc32/crc32c | `integer_tail_match_oracle` | **done** |
| 5f | Encoder: **system** — barriers (dmb/dsb/isb), hints, svc/brk/hlt, mrs/msr (named sysregs + PSTATE) | `system_match_oracle` | **done** |
| 5g | Encoder: **atomics** — exclusives (ldxr/stxr/ldar/stlr +b/h), LSE (ldadd/ldclr/…/swp/cas with a/l/al + b/h) | `atomics_match_oracle` | **done** |
| 5h | Encoder: **FP/SIMD loads** — ldr/str/ldp/stp for b/h/s/d/q (uoff, pre/post, reg-offset) | `fp_simd_loads_match_oracle` | **done** |
| 6 | **Two-pass relaxation driver** (`a64/mod.rs`): out-of-range/extern conditional branches relax to *inverted-cond + `b`* (grow-only fixpoint). A deliberate extension beyond LLVM-MC, which errors on out-of-range AArch64 cond branches. B/BL beyond ±128 MB still errors (absurd for a single function); extern far calls are veneered by the loader. | `cond_branch_to_extern_relaxes`, `far_conditional_branch_relaxes_to_inverted_plus_b` | **done** |
| 7g | **Crypto** (AES: aese/aesd/aesmc/aesimc; SHA1/SHA256 2-op + 3-op) + **fixed-point `fcvt`** (`fcvtzs/zu`, `scvtf/ucvtf` with `#fbits`) | `crypto_and_fixedpoint_match_oracle` | **done** |
| 7a | SIMD copy (`ins`/`umov`/`smov`) + permute (`zip`/`uzp`/`trn`/`ext`) | `simd_copy_permute_match_oracle` | **done** |
| 7b | SIMD shift-by-immediate (shl/sshr/ushr/ssra/usra/srshr/urshr/srsra/ursra/sli/sri) — immh:immb encoding | `simd_shift_imm_match_oracle` | **done** |
| 7c | Vector FP: 3-same (fabd/fmulx/fmaxnm/fminnm/faddp/fmaxp/fminp/fmaxnmp/fminnmp/fcmge), 2-misc convert/round (scvtf/ucvtf/fcvtzs/fcvtzu/frint*), compare-#0.0 (fcmeq/ge/gt/le/lt) | `simd_fp_vector_match_oracle` | **done** |
| 7d | SIMD modified immediate: movi/mvni/bic-imm/orr-imm (full cmode table, msl, .2d byte-mask, scalar Dd) | `simd_movi_match_oracle` | **done** |
| 7e | SIMD widen/narrow/long (saddl/uaddw/addhn/xtn/sqxtn/smull-vec…) + by-element (mul/fmla/smull by lane, H:L:M index) + fcvtn/fcvtl/fcvtxn | `simd_widen_narrow`/`simd_byelement` | **done** |
| 7f | SIMD `tbl`/`tbx` + structured `ld1`–`ld4`/`st1`–`st4` (register-list `{…}` parsing + post-index) | `simd_copy_permute`/`simd_struct_ldst` | **done** |

**NEON now comprehensively covered** (ARMv8-A AdvSIMD): 3-same int/logical/FP, 2-misc int/FP, across-lane, dup, copy (ins/umov/smov), permute (zip/uzp/trn/ext), tbl/tbx, shift-by-immediate, widen/narrow/long, by-element, vector FP convert/compare/round, modified-immediate, structured load/store.
| 7b | Oracle now enables `+lse,+crc,+fullfp16,+rcpc,+rdm,+dotprod` (Apple Silicon baseline) | — | **done** |
| 8b | pc-relative tail: `adr`, `ldr Xt, [Xn, sym@PAGEOFF]`, GOT (`@GOTPAGE`/`@GOTPAGEOFF`) | masked diff green | |
| 9 | Record + commit `corpus/aarch64.tsv`; full frontier lock | `corpus_replay` green **no-LLVM** | |
| 10 | **macOS native loader `MacJit`** (`src/native_macos.rs`): mmap MAP_JIT, per-thread W^X toggle, icache flush, AArch64 reloc patching (Branch26/AdrpPage21/AddPageOff12/Abs64 by bit-insertion), abs `movz/movk x16; br x16` veneers, `Loader` impl over `a64::assemble` | `leaf_executes`/`internal_call_executes`/`host_callback_executes` — JIT'd AArch64 runs + calls back into Rust | **done** |

> **De-risk confirmed (Phase 10):** `mmap(MAP_JIT)` + `pthread_jit_write_protect_np` + `sys_icache_invalidate` work in a plain `cargo test` binary on Apple Silicon with **no code-signing or entitlements** (ad-hoc-signed, non-hardened-runtime). The `com.apple.security.cs.allow-jit` entitlement is only needed when distributing a hardened-runtime binary. End-to-end execution (encode → place → relocate → run → host callback via veneer) is proven.

| 10b | **`hello-aarch64` bin** (`src/bin/hello_a64.rs`) — macro front-end → `A64Encoder` → `MacJit`, with an AAPCS64 Rust host callback, no LLVM | `cargo run --bin hello-aarch64` prints `from JIT: 42` / `forth_main() = 84` | **done** |
| 11 | AAPCS64 Forth macro library (`forth-a64/`) | a kernel word JITs + runs | |
| 12 | (deferred) crash dumper + codesign/entitlement hardening | brk #0 dumps + continues | |

**Mach-O local-label convention (found in Phase 1b):** LLVM-MC's Mach-O
assembler rejects a *conditional/compare* branch (`b.cond`/`cbz`/`cbnz`) whose
target is not an **assembler-local label** ("requires assembler-local label") —
those branches are never relocated, so the target must resolve at assembly time.
On Mach-O, local labels are `L`-prefixed (and excluded from the symbol table). The
native encoder resolves *any* internal label, but for the oracle/corpus to accept
cond-branch-heavy code — and therefore for the front-end's macro-generated
internal labels (e.g. `.if`/`.while` skip labels) on AArch64 — those labels must
use the `L…` convention. Unconditional `b`/`bl` to a non-local internal label is
fine (resolved in-section). Action item for the macro library (Phase 11): emit
`L`-prefixed internal labels on the Mach-O target.

---

## 5. Verified AArch64 encoding vectors (golden-test seeds)

Bytes are little-endian as they sit in memory, **verified with
`llvm-mc -triple=aarch64-apple-darwin --show-encoding`** unless noted. Use these to
seed Phase 2–3 goldens.

### 5a. Absolute-byte goldens (reloc-free — compare raw)

| asm | bytes |
|---|---|
| `ret` | `c0 03 5f d6` |
| `nop` | `1f 20 03 d5` |
| `brk #0` | `00 00 20 d4` |
| `mov x0, x1` | `e0 03 01 aa` |
| `mov w0, w1` | `e0 03 01 2a` |
| `mvn x0, x1` | `e0 03 21 aa` |
| `mov x0, #42` | `40 05 80 d2` |
| `movz x0, #0x1234` | `80 46 82 d2` |
| `movk x0, #0xabcd, lsl #32` | `a0 79 d5 f2` |
| `movn x0, #0` | `00 00 80 92` |
| `add x0, x1, x2` | `20 00 02 8b` |
| `sub sp, sp, #16` | `ff 43 00 d1` |
| `ldr x0, [x1, #8]` | `20 04 40 f9` |
| `str x0, [sp, #-16]!` | `e0 0f 1f f8` |
| `lsl x0, x1, #1` (alias) | (derive in Phase 7) |
| `lsr x0, x1, x2` | `20 24 c2 9a` |
| `asr x0, x1, x2` | `20 28 c2 9a` |

### 5b. Symbolic forms — **diff through the masked/reloc path, NOT raw bytes**

LLVM emits these with immediate bits **as relocation placeholders**. In the
emitted *object* those bits read as zero (so they compare equal to a
zero-placeholder encoder), but `--show-encoding` shows them as `A`:

| asm | `--show-encoding` | emitted object word | reloc |
|---|---|---|---|
| `b _label` / `bl _label` | `[A,A,A,0b000101AA]`/`…100101AA` | `14000000`/`94000000` | `Branch26` |
| `b.eq _label` | `[0bAAA00000,A,A,0x54]` | `54000000` | (internal→patched; extern→via veneer) |
| `adrp x0, _s@PAGE` | `[A,A,A,0x90'A']` | `90000000` | `AdrpPage21` |
| `add x0, x0, _s@PAGEOFF` | `[0x00,0bAAAAAA00,0b00AAAAAA,0x91]` | `91000000` | `AddPageOff12` |

Note `b #0` is a **literal self-branch and emits no reloc**; only `b _label`
emits `BRANCH26`. The ADRP high byte itself carries an `immhi` placeholder bit
(`0x90'A'`), which is why the §3.5.2 mask is a *bit* mask, not a byte mask.

---

## 6. Risks & open questions

1. **Entitlement/signing posture (Phase 10).** Adopt `allow-jit` + `MAP_JIT` +
   per-thread toggle; codesign JIT/test binaries. Resolve whether CI runs signed.
   `pthread_jit_write_protect_supported_np()` must guard the toggle.
2. **Veneer correctness (Phase 6).** Grow-only relaxation for termination; one
   12-byte ADRP veneer per target; externs always take the far path.
3. **Difftest masking (Phase 2).** Per-kind bit masks; rely on the
   both-sides-zero property but keep the mask as the safety net.
4. **AAPCS64 variadics (Phase 11).** Wrapper-macro arg placement differs from
   Win64 (no shadow space; Apple's variadic ABI passes *all* variadic args on the
   stack — relevant if/when binding `printf`-shaped externs).
5. **MCJIT execution on the Mac (parallel to the native loader).** The oracle path
   only *emits objects*; running JIT'd AArch64 via MCJIT (`jit.rs`) is the same C
   API and host-targets AArch64 here, but its first run will surface the same
   entitlement/W^X realities as the native loader — bring it up alongside Phase 10.

---

## 7. Appendix — file:line touch map (Phase 0–1, committed)

- `build.rs` — per-OS dispatch; `link_unix_dylib`/`link_windows`/`llvm_root`.
- `src/llvm.rs` — AArch64 init externs; `init_targets_and_mcjit()`.
- `src/oracle.rs` — `triple_is_x86`, `ensure_trailing_newline`, `__text`
  fallback, Mach-O ARM64 `map_reloc`, `x86_64()`/`aarch64_macos()`, AArch64 tests.
- `src/backend.rs` — `RelocKind::{Branch26,AdrpPage21,AddPageOff12}`.
- `src/difftest.rs` — `reloc_class`/`kind_str`/`parse_corpus_line` arms for the
  new kinds.
