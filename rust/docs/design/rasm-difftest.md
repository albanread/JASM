# rasm differential oracle — design

Status: design (no code yet). Companion to the rasm replacement work
(`src/rasm/`, the `Encoder`/`Loader` seam in `src/backend.rs`).

## 1. Goal

Turn the LLVM-MC "byte-identity" relationship rasm is already built against
(`encode.rs`: *"chosen to match LLVM-MC … the golden differential gates that"*)
from a **manual, copy-pasted ritual** (hand-pasted `vec![0x48,0x89,…]` literals in
`encode.rs` tests) into an **automated differential harness**: generate instruction
forms, encode each with both rasm and an LLVM oracle, assert byte/reloc equality,
and report the failing `asm_text` on mismatch.

Second goal, designed in from the start: the harness must **generalize to other
target CPUs** (AArch64, RISC-V, …) by swapping plug-ins, not by rewriting the core.
The known portability hazard is `iced-x86` (x86-only); §9 contains it.

### Non-goals
- Generating rasm's encoder *tables* from LLVM (TableGen mining). LLVM is used as a
  **test oracle**, not a code generator — see the parent discussion for why.
- Changing the shipping runtime. The oracle is build/test-time only, behind the
  `llvm` feature. Native remains the default build (`4c77d1b`).

## 2. Core insight: the oracle conforms to the `Encoder` trait

rasm already produces the exact shape we want to diff against:

```rust
// src/backend.rs
pub struct EncodedModule { code: Vec<u8>, symbols: BTreeMap<String,usize>,
                           relocs: Vec<Reloc>, externs: Vec<String> }
pub trait Encoder { fn encode(&self, asm_text: &str) -> Result<EncodedModule>; }
```

Reloc fields are left as **zeroed placeholders** in `code`, with a structured
`Reloc { at, size, kind, target, addend }` list beside them. If we make the LLVM
oracle emit the *same* shape, the differential is `Encoder`-vs-`Encoder`:

```rust
fn diff(rasm: &dyn Encoder, oracle: &dyn Encoder, asm: &str) -> Verdict
```

This is the keystone: the diff driver is **arch-neutral by construction** — it only
ever sees `&dyn Encoder` and `EncodedModule`. Everything CPU-specific is pushed to
the edges (the generator and the reloc-type map).

## 3. Architecture: arch-neutral core, arch plug-ins

| Layer | Arch-neutral? | Component |
|---|---|---|
| Diff driver + report | ✅ neutral | `difftest::driver` — compares two `EncodedModule`s, masks reloc fields, formats mismatches |
| Corpus record/replay | ✅ neutral | `difftest::corpus` — JSONL of `{asm, code_hex, relocs}` goldens |
| Encoding oracle | ✅ neutral *interface* | `LlvmMcEncoder: Encoder` (LLVM-MC is itself multi-target; only the reloc-type map per §5.4 is arch-specific) |
| Form generator | ❌ per-arch | `trait IsaModel` → x86: `X86Model`; later `Aarch64Model` |
| Decode round-trip (secondary) | ❌ per-arch | `trait Disassembler` → x86: `IcedDisasm`; later Capstone / `LLVMCreateDisasm` |
| Reloc-type map | ❌ per-arch | object reloc kind → `RelocKind` |

Two of the three arch-specific seams (generator, reloc-map) are small and
unavoidable for *any* approach. The third (disassembler) is **optional** — see §9.

## 4. Trait seams

```rust
/// (1) The oracle is just an Encoder. RasmEncoder already implements this;
///     LlvmMcEncoder is the new impl (§5).
// trait Encoder — unchanged, from src/backend.rs

/// (2) Per-arch form generator. Yields candidate assembly lines plus enough
///     metadata to bucket/report. No bytes here — generation is pure text.
pub struct Form {
    pub asm: String,        // e.g. "add rax, [rbp - 8]"
    pub family: &'static str, // "alu.rm", "sse.scalar", … for reporting/sharding
    pub mnemonic: String,
}
pub trait IsaModel {
    fn triple(&self) -> &str;          // "x86_64-pc-windows-msvc"
    fn forms(&self) -> Box<dyn Iterator<Item = Form> + '_>;
}

/// (3) Secondary, OPTIONAL self-check: does our blob decode back to the same
///     instruction? Catches "valid bytes, wrong instruction". Arch-pluggable.
pub struct DecodeView { pub text: String, pub len: usize, pub invalid: bool }
pub trait Disassembler {
    fn decode_one(&self, bytes: &[u8]) -> DecodeView;
}

/// (4) Per-arch: map an object-file relocation type to our RelocKind.
pub trait RelocMap { fn map(&self, obj_kind: object::RelocationKind) -> Option<RelocKind>; }
```

The driver is generic over all four and never names a concrete arch:

```rust
pub struct DiffHarness<'a> {
    rasm:   &'a dyn Encoder,
    oracle: &'a dyn Encoder,
    model:  &'a dyn IsaModel,
    disasm: Option<&'a dyn Disassembler>,  // None on arches with no decoder yet
}
```

## 5. Getting ground truth out of LLVM (the hard part)

The current LLVM path (`jit.rs`) is `LLVMAppendModuleInlineAsm` → **MCJIT** →
bytes in *executable memory, already relocated*. That is not directly diffable
against rasm's placeholder+reloc-list form. Three ways to get a clean
`EncodedModule` from LLVM, in recommended order:

### 5.1 Recommended: in-process object emit + `object` crate  *(multi-arch)*
Add a TargetMachine and emit a relocatable object to a memory buffer, then parse
it with the pure-Rust [`object`] crate:

```
LLVMCreateTargetMachine(triple, …)                         // new LLVM-C binding
LLVMTargetMachineEmitToMemoryBuffer(tm, module, ObjectFile) // new LLVM-C binding
object::File::parse(buf) → .text bytes + symbols + relocations
```

Why this is the pick:
- The object **already has placeholder reloc fields + a relocation table** — the
  same model as rasm's `EncodedModule`. No relocation un-applying needed.
- **Multi-arch for free**: TargetMachine supports every LLVM target via `triple`;
  `object` reads COFF *and* ELF *and* Mach-O with one API. The only per-arch piece
  is the small `RelocMap` (§4.4).
- In-process — no external binary. `llvm-mc.exe` is **not installed** with the
  stock LLVM Windows package on this box (only `llvm-mca`/`llvm-objdump` ship), so
  any subprocess oracle is a non-starter here.

Cost: ~4 new `extern` lines in `llvm.rs` + `object` as a dev/feature dependency +
a per-arch reloc map (COFF `IMAGE_REL_AMD64_REL32`→`BranchRel32`, etc.).

### 5.2 Alternative: `llvm-mc --show-encoding` (text)
`llvm-mc -triple=… --show-encoding` prints `# encoding: [0x48,0x89,…]` with
variable fields marked. Cleanest *symbolically*, also multi-arch — but the binary
isn't installed here, so rejected for now. Keep as a portable fallback if a future
CI image has full LLVM tools.

### 5.3 Fallback: MCJIT memory readback  *(x86 near-term only, zero new bindings)*
Reuse `jit.rs` as-is: finalize, read N bytes from `LLVMGetFunctionAddress`. Bytes
are **post-relocation**, so normalize before comparing (§5.4). Fine for the
self-contained reg/imm forms that dominate the current frontier (they have *no*
relocations); awkward for branch/RIP-rel. Use only to bootstrap before 5.1 lands.

### 5.4 Normalization (applies to 5.1 and 5.3)
Before `assert_eq!(code)`, **mask every byte covered by a fixup to 0x00** on *both*
sides, then compare `code`. Separately compare the fixup lists structurally:
`(at, size, kind, target?, addend)`. This makes the comparison invariant to the
actual displacement values (which differ between an object's 0-placeholder and
MCJIT's patched address) while still catching a wrong fixup *width/position/kind*.

## 6. Diff & report

```
Verdict::Match
Verdict::ByteMismatch { asm, rasm: Vec<u8>, oracle: Vec<u8>, first_diff: usize }
Verdict::RelocMismatch { asm, rasm: Vec<Reloc>, oracle: Vec<Reloc> }
Verdict::RasmError { asm, err }      // rasm refused a form the oracle accepts → a gap
Verdict::OracleError { asm, err }    // both refused → generator produced an illegal form; drop it
```

`RasmError` where the oracle succeeds is the **coverage frontier signal** — these
are exactly the rows to implement next. Report groups mismatches by `Form::family`
so "all of `sse.packed` is unimplemented" reads as one line, not 400.

## 7. Record / replay — decouple CI from LLVM *and* from arch

Two run modes over the same generated `Form` set:

- **Record** (`difftest --record`, needs `--features llvm`): run the oracle,
  write `corpus/<triple>.jsonl` lines `{asm, code_hex, relocs}` for every form the
  oracle accepts. This is the only step that touches LLVM.
- **Replay** (a plain `#[test]`, **no llvm feature, no LLVM installed**): load the
  committed corpus, run `RasmEncoder` + the `Disassembler` self-check, assert
  byte-equality against the recorded goldens.

Consequences:
- CI gates rasm with **zero LLVM dependency** — the corpus *is* the frozen oracle.
- The replay test is **100% arch-neutral**: it just replays data. Bringing up
  AArch64 later means `difftest --record --triple aarch64-…` on a machine with that
  LLVM target, committing `corpus/aarch64.jsonl`, and the *same* replay test gates
  it. No new test code.
- The corpus diff in code review shows, byte-for-byte, what an encoder change did.

This supersedes the hand-pasted goldens in `encode.rs` — those literals become
generated corpus rows.

## 8. The x86-64 generator (concrete)

`X86Model::forms()` is a product of small, explicit tables — *templated*, not random,
so the corpus is deterministic and reviewable (no `rand`/`Date`/`proptest` needed):

```
register banks:  R8/R16/R32/R64 × {rax-class, r8-r15 to exercise REX.B/R}, XMM0-15
mem templates:   [base], [base+disp8], [base+disp32], [base+index*scale],
                 [rip+sym], with/without segment, sized (byte/word/dword/qword ptr)
imm buckets:     0, 1, 0x7f, 0x80, 0x7fff_ffff, -1, 0xffff_ffff (sign/zero-extend edges)
mnemonic catalog: per family, the operand shapes that family accepts
```

A family entry expands its shapes against the banks/templates. Example — ALU `r/m, r`
and `r, r/m` for `add`: emit `add <r64>, <r64>`, `add <r/m>, <r64>`, `add <r64>, <r/m>`,
`add <r/m>, <imm>`, across the register/mem/imm tables. Cap or stride large products
(e.g. one representative high register rather than all of r8–r15) and **`log()` what
was strided** so coverage gaps are explicit, never silent.

Seed the catalog from the mnemonics rasm already claims (the `encode.rs` dispatch:
ALU, mov/movzx/movsx/lea/push/pop, shifts, unary, setcc/jcc/cmovcc, string,
xchg/xadd, scalar-double SSE). That set should pass on day one and **lock the
frontier**; then add catalog rows (movss, packed SSE, BMI `bt`, x87, `endbr64`, …)
and watch them surface as `RasmError` until implemented.

## 9. Multi-arch plan — and containing `iced-x86`

What actually changes per new target:

| Seam | Effort | Notes |
|---|---|---|
| `IsaModel` | **large** | register files, addressing modes, imm encodings are genuinely different per ISA. Unavoidable for any generator. |
| `RelocMap` | small | a dozen reloc-type rows (ELF `R_AARCH64_CALL26`→branch, etc.) |
| `LlvmMcEncoder` | **none** | same code; pass a different `triple`. LLVM-MC + `object` are already multi-target. |
| Diff driver / corpus / replay test | **none** | arch-neutral (§3). |
| `Disassembler` | optional | **this is the iced problem** ↓ |

**The `iced-x86` answer.** iced is x86-only and does *not* port. But it is the
**secondary** check (decode round-trip), not the gate. The **primary** oracle is
LLVM-MC, which is already multi-target — so on AArch64/RISC-V you keep the full
differential gate and the byte-exact corpus with **no decoder at all** (`disasm:
None`). The portability hazard the user flagged turns out to sit on the *least*
load-bearing seam. If a secondary decode check is wanted on other arches, two
multi-arch options drop in behind the same `Disassembler` trait:
- **Capstone** (`capstone` crate) — covers AArch64/ARM/RISC-V/MIPS/PPC/…; or
- **`LLVMCreateDisasm`** from the LLVM-C we already link — same dependency, every
  target, no new crate.

So the rule: **never let iced (or any decoder) leak past the `Disassembler` trait**;
the driver, oracle, generator-interface, and corpus must not reference it. Then
new-arch bring-up is "write an `IsaModel` + a `RelocMap`, record a corpus" — the
decoder is icing.

## 10. Phasing

1. **Bytes-out:** ✅ **done.** `LlvmMcEncoder: Encoder` (`src/oracle.rs`) + LLVM-C
   TargetMachine/object-emit bindings (`src/llvm.rs`) + optional `object` dep
   (gated by `llvm`) + x86 COFF `map_reloc`. Empirical findings worth carrying
   into Phase 2:
   - The emitted `.text` is **byte-identical** to rasm with **no trailing
     alignment padding** — direct `assert_eq!(o.code, r.code)` works for
     reloc-free forms (test `matches_rasm_on_reg_imm_alu`).
   - `.globl` symbols, extern relocs, and the `E8 rel32` form all extract
     correctly; bad asm returns `Err` via the diagnostic handler (no abort).
   - **Phase 2 must canonicalize REL32:** COFF emits one reloc type for both
     branch `rel32` and RIP-rel `disp32`, so `map_reloc` returns `BranchRel32`
     for both. The diff must treat `BranchRel32` ≡ `RipRel32` (§5.4) and mask
     reloc-covered bytes before comparing `code`.
2. **Driver + normalization** (§5.4, §6) over a tiny hand-list of forms.
   ✅ **done.** `src/difftest.rs` — `Verdict`, `diff`/`compare`/`diff_all`,
   `Report`, reloc-field masking, and `BranchRel32 ≡ RipRel32` canonicalization.
   Arch/oracle-neutral (depends only on `backend`; compiles without `llvm`, so
   Phase 4 replay can reuse it). The live gate
   (`rasm_matches_llvm_on_current_coverage`, `#[cfg(feature = "llvm")]`) diffs
   **53 forms** spanning rasm's claimed coverage: **53 match, 0 mismatch, 0 gap**.
   - `addend` is excluded from reloc equivalence for now (rasm folds the PC bias
     into `RelocKind`; objects carry a container addend convention). Revisit when
     an addend-bearing form (`lea [rip+sym+N]`) enters the corpus.
   - Finding: the harness flagged `movsx r64, r/m32` — *both* encoders reject it,
     because the canonical 32→64 mnemonic is `movsxd`. A bad form in the list,
     not a rasm bug, but a clean demonstration of the oracle gating mnemonic
     legality, not just bytes.
3. **`X86Model`** seeded from rasm's current mnemonics (§8). ✅ **generator +
   first gap report done.** `src/difftest/x86.rs` (templated catalog), `Form`/
   `IsaModel`/`diff_model`/`ModelReport` in `src/difftest.rs`, and the
   `rasm-gaps` bin (`cargo run --bin rasm-gaps --features llvm`). First sweep of
   the integer + SSE/SSE2 tier: **1593 forms → 900 match, 11 MISMATCH, 682 gaps,
   0 oracle-reject.**
   - **11 mismatches = 2 real rasm bugs** (rasm accepts but mis-encodes):
     1. `cvtsi2sd`/`cvtsi2ss` with a **32-bit source** — rasm forces `REX.W`
        (`cvtsi2sd xmm1, eax` → `f2 48 0f 2a c8`, should be `f2 0f 2a c8`). A
        **correctness** bug: the bytes decode as the 64-bit-source form.
     2. `xchg` with the accumulator — rasm uses generic `87 /r`; LLVM uses the
        `90+r` short form (and `xchg rax,rax` → `90`). Byte-identity divergence.
   - **682 gaps** cluster by family — the to-implement worklist: all float32
     scalar (`addss`…), all packed f32/f64/int (`addps`/`pxor`/`paddd`…), most
     SSE moves (`movss`/`movaps`/`movd`/`movq`/`movdqa`…) and conversions, plus
     integer `bt`/`bts`/`btr`/`btc`, `bswap`, `cmpxchg`, `movbe`, `endbr64`,
     `test r/m,imm`, `push r/m`/`push imm`, and bare (non-`rep`) string ops.
   - Phase 3b: fix the 2 mismatches, add a `generated_forms_never_misencode`
     gate (`mismatches().is_empty()`), then close gaps family by family.

   **Phase 3b ✅ done — integer + SSE/SSE2 tier complete: 1593/1593 forms match,
   0 mismatch, 0 gaps.**
   - Both mismatch bugs fixed in `encode.rs`: `cvtsi2sd` REX.W now keyed off the
     source width (`operand_w(src)`); `xchg` gained the accumulator `90+rd` short
     form (incl. `xchg rax,rax` → `90`).
   - SSE filled in via table-driven helpers in `encode.rs`: `sse_rrm` (scalar-f32,
     packed f32/f64/int, logicals, unpack, ordered compares, xmm→xmm cvt),
     `sse_shuffle` (imm8), `sse_mov` (load/store moves), `sse_movd_q`, `sse_cvt_gpr`.
   - Integer filled in via `try_int_misc`: `bt/bts/btr/btc`, `bswap`, `cmpxchg`,
     `movbe`, `endbr64`, bare string ops, `test r/m,imm`, `push r/m`/`push imm`.
   - Every batch matched LLVM byte-for-byte on the first run — no encoding iteration.
   - The `generated_forms_never_misencode` gate is now green and guards the tier.
   - WF66 unaffected: `rasm-diff` still at its pre-existing 141 (stale-golden
     displacement divergences); the encoder changes are byte-neutral to the kernel.
4. **Record/replay** (§7). ✅ **done.** `record_corpus`/`corpus_line`/
   `parse_corpus_line` in `src/difftest.rs`, the `rasm-corpus` bin, and the
   committed golden `corpus/x86_64.tsv` (1593 forms, TSV: `asm \t hexcode \t
   relocs`). The `corpus_replay_matches_golden` `#[test]` gates rasm against it
   with **no `llvm` feature and no LLVM installed** — verified it fails on a
   single corrupted byte and that the default (no-LLVM) suite runs it green.
   Format note: corpus stores masked code (reloc fields zeroed) + a canonical
   reloc list; replay reconstructs an `EncodedModule` and reuses `compare`.
   - The hand-pasted `encode.rs` goldens are kept (fast LLVM-free spot checks +
     iced round-trip); the corpus is the broad gate, not a replacement for them.
   - Regeneration after closing gaps: `cargo run --bin rasm-corpus --features llvm`.
5. **Push the frontier — AVX/AVX2 (VEX).** ✅ **done. The full chosen tier
   (integer + SSE/SSE2 + AVX/AVX2) is complete: 3393/3393 forms match, 0 mismatch,
   0 gaps.**
   - `Ymm` reg class + `ymm0–15` lexing in `parse.rs`.
   - `emit_mem_rm` refactored to share `emit_modrm_mem`/`mem_xb`/`emit_modrm_reg`
     with the VEX path (verified byte-neutral by the 1593-form corpus replay).
   - VEX encoder in `encode.rs`: `vex` (2-byte C5 / 3-byte C4 selection),
     `emit_vex_rm`, and `try_vex` → `vex_rvm` (3-op), `vex_rm` (2-op), `vex_mov`
     (load/store), `vex_shuffle` (imm8). Generator: `Ymm`/`YmmRm` ops + a VEX
     catalog (both VEX.128 and VEX.256).
   - Replicated LLVM's *shortest-encoding* policy to stay byte-identical: (a)
     commutative-source swap so a high reg lands in `vvvv` not `rm` (packed only —
     scalar upper lanes come from `vvvv`, so they're excluded); (b) reg-reg move
     direction flip (store opcode) for the same reason; (c) both gated on
     `map==0F`, since `vpmulld` (0F38) is always 3-byte and LLVM doesn't swap it.
   - Corpus regenerated to 3393 forms; harness/oracle/driver needed zero changes.
   - WF66 unaffected (builds clean; `rasm-diff` still 141; byte-neutral to kernel).
6. **Prove portability** (when an arch is wanted): add `Aarch64Model` + reloc map,
   record `corpus/aarch64.jsonl`, confirm the driver/replay/corpus code is untouched.
7. **AVX-512 (EVEX), increment 1 — unmasked.** ✅ **done: 5109/5109 forms match,
   0 mismatch, 0 gaps.**
   - `Zmm` reg class + vector registers extended to 0..=31 (`parse.rs`).
   - `emit_evex_rm` (4-byte `62` prefix with R/X/B/R'/V' extension bits) and
     `emit_vec_rm`, which auto-selects VEX vs EVEX (EVEX iff zmm or any reg ≥ 16).
     The VEX form helpers feed `emit_vec_rm` and gate their 2-byte optimizations
     to the VEX case.
   - EVEX `W` is element-size semantic (W1 for double/qword); VEX is WIG, so
     `emit_vec_rm` forces VEX `W=0`. The oracle's TargetMachine now enables
     `+avx512f,+avx512vl,+avx512dq,+avx512bw` so it accepts EVEX forms.
   - Excluded from the unmasked set (they change form under EVEX): `vpand/vpor/…`
     (→ `vpandd/q`), `vpcmpeq*` (mask-register destination).
   - Increment 2 (pending): masking — `{k}{z}` operand decorators, `k0..k7` mask
     registers, `kmov*`. Needs the only invasive piece: a writemask on the
     destination operand in the parser/operand model.

[`object`]: https://crates.io/crates/object
