//! Differential driver: compare rasm against an oracle [`Encoder`], byte-for-byte.
//!
//! Arch- and oracle-neutral by construction — it only ever sees two
//! `&dyn Encoder` and the [`EncodedModule`]s they return, so the same driver
//! gates rasm against [`LlvmMcEncoder`](crate::oracle::LlvmMcEncoder) today and
//! against a recorded corpus (Phase 4) later. No `llvm` feature needed here; the
//! tests below opt in to get a live oracle. See `docs/design/rasm-difftest.md`.
//!
//! Normalization (design §5.4): the two encoders agree on the *encoding* but not
//! on relocated displacement values (rasm leaves a zero placeholder; an object's
//! field carries the container's convention). So before comparing `code` we mask
//! every byte a relocation covers to `0x00` on **both** sides, then compare the
//! relocation lists structurally. Branch `rel32` and RIP-rel `disp32` share one
//! machine relocation on x86-64, so they're compared as a single class.

use std::collections::BTreeMap;

use crate::backend::{Encoder, EncodedModule, Reloc, RelocKind};

/// Per-arch templated form generators (x86-64 today). See [`IsaModel`].
pub mod x86;
pub mod aarch64;

/// The outcome of diffing one assembled form.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Verdict {
    /// Byte-identical (after masking reloc fields) and reloc-equivalent.
    Match,
    /// Code bytes diverge after masking — rasm encoded the wrong bytes.
    ByteMismatch {
        asm: String,
        rasm: Vec<u8>,
        oracle: Vec<u8>,
        /// First differing index in the masked, compared view.
        first_diff: usize,
    },
    /// Code matched but the relocation lists are not equivalent.
    RelocMismatch {
        asm: String,
        rasm: Vec<Reloc>,
        oracle: Vec<Reloc>,
    },
    /// rasm errored on a form the oracle accepted — a coverage gap to implement.
    RasmError { asm: String, err: String },
    /// The oracle errored (illegal/unsupported form) — drop the form.
    OracleError { asm: String, err: String },
}

impl Verdict {
    pub fn is_match(&self) -> bool {
        matches!(self, Verdict::Match)
    }
    /// A real failure: rasm produced wrong bytes or wrong relocations.
    pub fn is_mismatch(&self) -> bool {
        matches!(self, Verdict::ByteMismatch { .. } | Verdict::RelocMismatch { .. })
    }
    pub fn is_gap(&self) -> bool {
        matches!(self, Verdict::RasmError { .. })
    }
}

/// Diff one form: encode with both, classify. Oracle is consulted first so a
/// form the oracle rejects becomes `OracleError` (dropped) rather than masking a
/// rasm gap.
pub fn diff(rasm: &dyn Encoder, oracle: &dyn Encoder, asm: &str) -> Verdict {
    let oracle_mod = match oracle.encode(asm) {
        Ok(m) => m,
        Err(e) => return Verdict::OracleError { asm: asm.to_string(), err: format!("{e:#}") },
    };
    let rasm_mod = match rasm.encode(asm) {
        Ok(m) => m,
        Err(e) => return Verdict::RasmError { asm: asm.to_string(), err: format!("{e:#}") },
    };
    compare(asm, &rasm_mod, &oracle_mod)
}

/// Compare two already-encoded modules. Split out so Phase 4 corpus replay can
/// compare a fresh rasm encode against a recorded oracle module.
pub fn compare(asm: &str, rasm_mod: &EncodedModule, oracle_mod: &EncodedModule) -> Verdict {
    let masked_rasm = mask_relocs(&rasm_mod.code, &rasm_mod.relocs);
    let masked_oracle = mask_relocs(&oracle_mod.code, &oracle_mod.relocs);
    if masked_rasm != masked_oracle {
        return Verdict::ByteMismatch {
            asm: asm.to_string(),
            rasm: rasm_mod.code.clone(),
            oracle: oracle_mod.code.clone(),
            first_diff: first_diff(&masked_rasm, &masked_oracle),
        };
    }
    if !relocs_equiv(&rasm_mod.relocs, &oracle_mod.relocs) {
        return Verdict::RelocMismatch {
            asm: asm.to_string(),
            rasm: rasm_mod.relocs.clone(),
            oracle: oracle_mod.relocs.clone(),
        };
    }
    Verdict::Match
}

/// Return a copy of `code` with every byte covered by a relocation zeroed.
fn mask_relocs(code: &[u8], relocs: &[Reloc]) -> Vec<u8> {
    let mut out = code.to_vec();
    for r in relocs {
        let start = r.at.min(out.len());
        let end = (r.at + r.size as usize).min(out.len());
        for b in &mut out[start..end] {
            *b = 0;
        }
    }
    out
}

/// Collapse `RelocKind` to its machine class. On x86-64 branch `rel32` and
/// RIP-rel `disp32` are the *same* relocation, so they compare equal.
fn reloc_class(k: RelocKind) -> u8 {
    match k {
        RelocKind::BranchRel32 | RelocKind::RipRel32 => 0,
        RelocKind::Abs64 => 1,
        // AArch64: each field shape is its own class (no aliasing analogous to
        // x86's rel32/disp32 collision — the object carries distinct reloc
        // types per the verified Mach-O dump).
        RelocKind::Branch26 => 2,
        RelocKind::AdrpPage21 => 3,
        RelocKind::AddPageOff12 => 4,
    }
}

/// Structural reloc equivalence on `(offset, width, class, target)` as an
/// order-independent multiset.
///
/// `addend` is intentionally excluded: rasm folds the PC bias into the
/// `RelocKind` (always `addend: 0`), whereas an object's reloc carries the
/// container's addend convention. No current form has a non-zero symbolic
/// addend; tighten this when one appears (design §10, Phase 3+).
fn relocs_equiv(a: &[Reloc], b: &[Reloc]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let key = |r: &Reloc| (r.at, r.size, reloc_class(r.kind), r.target.clone());
    let mut ka: Vec<_> = a.iter().map(key).collect();
    let mut kb: Vec<_> = b.iter().map(key).collect();
    ka.sort();
    kb.sort();
    ka == kb
}

fn first_diff(a: &[u8], b: &[u8]) -> usize {
    a.iter().zip(b).position(|(x, y)| x != y).unwrap_or(a.len().min(b.len()))
}

// ── batch driver + report ────────────────────────────────────────────────────

/// Aggregate verdicts over many forms.
#[derive(Debug, Default)]
pub struct Report {
    pub verdicts: Vec<Verdict>,
}

impl Report {
    pub fn matched(&self) -> usize {
        self.verdicts.iter().filter(|v| v.is_match()).count()
    }
    /// Real failures — rasm is wrong here.
    pub fn mismatches(&self) -> Vec<&Verdict> {
        self.verdicts.iter().filter(|v| v.is_mismatch()).collect()
    }
    /// Coverage frontier — forms the oracle accepts but rasm can't yet encode.
    pub fn gaps(&self) -> Vec<&Verdict> {
        self.verdicts.iter().filter(|v| v.is_gap()).collect()
    }
    /// Forms dropped because the oracle itself rejected them.
    pub fn oracle_errors(&self) -> Vec<&Verdict> {
        self.verdicts
            .iter()
            .filter(|v| matches!(v, Verdict::OracleError { .. }))
            .collect()
    }

    pub fn summary(&self) -> String {
        format!(
            "{} forms: {} match, {} MISMATCH, {} gaps, {} dropped(oracle-reject)",
            self.verdicts.len(),
            self.matched(),
            self.mismatches().len(),
            self.gaps().len(),
            self.oracle_errors().len(),
        )
    }
}

/// Diff every form and collect a [`Report`].
pub fn diff_all<'a>(
    rasm: &dyn Encoder,
    oracle: &dyn Encoder,
    forms: impl IntoIterator<Item = &'a str>,
) -> Report {
    Report {
        verdicts: forms.into_iter().map(|f| diff(rasm, oracle, f)).collect(),
    }
}

// ── generator-driven coverage (Phase 3) ──────────────────────────────────────

/// One generated assembly form plus enough metadata to bucket the verdict.
#[derive(Debug, Clone)]
pub struct Form {
    /// The single-instruction assembly text (Intel syntax).
    pub asm: String,
    /// Coarse bucket for grouping the report, e.g. `"sse.packed.int"`.
    pub family: &'static str,
    /// The mnemonic, for per-instruction grouping.
    pub mnemonic: &'static str,
}

/// A per-architecture form generator. The driver stays arch-neutral; all
/// CPU-specific knowledge (register banks, addressing modes, the mnemonic
/// catalog) lives behind this. See [`x86::X86Model`].
pub trait IsaModel {
    fn triple(&self) -> &str;
    fn forms(&self) -> Vec<Form>;
}

/// A [`Verdict`] paired with the [`Form`] that produced it.
#[derive(Debug, Clone)]
pub struct ModelReport {
    pub entries: Vec<(Form, Verdict)>,
}

impl ModelReport {
    pub fn matched(&self) -> usize {
        self.entries.iter().filter(|(_, v)| v.is_match()).count()
    }
    /// Real failures — rasm accepts these forms but encodes them wrong.
    pub fn mismatches(&self) -> Vec<&(Form, Verdict)> {
        self.entries.iter().filter(|(_, v)| v.is_mismatch()).collect()
    }
    /// Coverage frontier — forms the oracle accepts but rasm can't yet encode.
    pub fn gaps(&self) -> Vec<&(Form, Verdict)> {
        self.entries.iter().filter(|(_, v)| v.is_gap()).collect()
    }
    /// Forms the oracle itself rejected (illegal text / sizing artifact).
    pub fn oracle_rejects(&self) -> Vec<&(Form, Verdict)> {
        self.entries
            .iter()
            .filter(|(_, v)| matches!(v, Verdict::OracleError { .. }))
            .collect()
    }

    pub fn summary(&self) -> String {
        format!(
            "{} forms: {} match, {} MISMATCH, {} gaps, {} dropped(oracle-reject)",
            self.entries.len(),
            self.matched(),
            self.mismatches().len(),
            self.gaps().len(),
            self.oracle_rejects().len(),
        )
    }

    /// Human-readable worklist: gaps grouped by family then mnemonic (distinct
    /// mnemonic count + one example), and every mismatch listed individually
    /// (those are bugs, not gaps).
    pub fn worklist(&self) -> String {
        let mut out = String::new();

        let mismatches = self.mismatches();
        if !mismatches.is_empty() {
            out.push_str(&format!("\n=== {} MISMATCH(es) — rasm encodes these WRONG ===\n", mismatches.len()));
            for (_, v) in mismatches {
                out.push_str(&format!("{v}\n"));
            }
        }

        // family -> mnemonic -> (count, example asm, error)
        let mut by_family: BTreeMap<&str, BTreeMap<&str, (usize, String, String)>> = BTreeMap::new();
        for (form, v) in self.gaps() {
            if let Verdict::RasmError { err, .. } = v {
                let e = by_family.entry(form.family).or_default();
                let slot = e.entry(form.mnemonic).or_insert((0, form.asm.clone(), err.clone()));
                slot.0 += 1;
            }
        }
        out.push_str(&format!("\n=== coverage gaps ({} forms) — worklist by family ===\n", self.gaps().len()));
        for (family, mnems) in &by_family {
            out.push_str(&format!("\n[{family}]  ({} mnemonics)\n", mnems.len()));
            for (mnem, (count, example, err)) in mnems {
                out.push_str(&format!("  {mnem:<12} x{count:<3} e.g. `{example}`  ({err})\n"));
            }
        }

        let rejects = self.oracle_rejects();
        if !rejects.is_empty() {
            out.push_str(&format!("\n=== {} oracle-rejected form(s) (generator sizing artifacts; first 10) ===\n", rejects.len()));
            for (form, v) in rejects.iter().take(10) {
                if let Verdict::OracleError { err, .. } = v {
                    out.push_str(&format!("  `{}`  ({err})\n", form.asm));
                }
            }
        }
        out
    }
}

/// Run a model's full form set through the differential.
pub fn diff_model(rasm: &dyn Encoder, oracle: &dyn Encoder, model: &dyn IsaModel) -> ModelReport {
    let entries = model
        .forms()
        .into_iter()
        .map(|f| {
            let v = diff(rasm, oracle, &f.asm);
            (f, v)
        })
        .collect();
    ModelReport { entries }
}

// ── record / replay corpus (Phase 4) ─────────────────────────────────────────
//
// The corpus is the no-LLVM regression gate: oracle goldens for every form rasm
// currently encodes correctly, frozen to a committed TSV the replay test reads
// via `include_str!`. CI then gates rasm byte-for-byte with **no LLVM
// dependency** — the corpus *is* the frozen oracle (design §7). Format, one form
// per line, tab-separated:
//
//     <asm> \t <hex code, reloc fields zeroed> \t <reloc;reloc;...>
//
// where each reloc is `at,size,kind,target` and kind ∈ {rel32, abs64}. `rel32`
// is the canonical class for branch + RIP-rel (they share one machine reloc).

fn kind_str(k: RelocKind) -> &'static str {
    match k {
        RelocKind::Abs64 => "abs64",
        RelocKind::BranchRel32 | RelocKind::RipRel32 => "rel32",
        RelocKind::Branch26 => "branch26",
        RelocKind::AdrpPage21 => "adrp_page21",
        RelocKind::AddPageOff12 => "add_pageoff12",
    }
}

/// Serialize one form's golden module to a corpus line (no trailing newline).
fn corpus_line(asm: &str, m: &EncodedModule) -> String {
    let masked = mask_relocs(&m.code, &m.relocs);
    let code: String = masked.iter().map(|b| format!("{b:02x}")).collect();
    let relocs = m
        .relocs
        .iter()
        .map(|r| format!("{},{},{},{}", r.at, r.size, kind_str(r.kind), r.target))
        .collect::<Vec<_>>()
        .join(";");
    format!("{asm}\t{code}\t{relocs}")
}

fn unhex(s: &str) -> Option<Vec<u8>> {
    if s.len() % 2 != 0 {
        return None;
    }
    (0..s.len()).step_by(2).map(|i| u8::from_str_radix(&s[i..i + 2], 16).ok()).collect()
}

/// Parse a corpus line back into `(asm, golden EncodedModule)`.
fn parse_corpus_line(line: &str) -> Option<(String, EncodedModule)> {
    let mut fields = line.splitn(3, '\t');
    let asm = fields.next()?.to_string();
    let code = unhex(fields.next()?)?;
    let reloc_field = fields.next().unwrap_or("");
    let relocs = if reloc_field.is_empty() {
        Vec::new()
    } else {
        reloc_field
            .split(';')
            .map(|r| {
                let mut p = r.split(',');
                let at = p.next()?.parse().ok()?;
                let size = p.next()?.parse().ok()?;
                let kind = match p.next()? {
                    "rel32" => RelocKind::BranchRel32,
                    "abs64" => RelocKind::Abs64,
                    "branch26" => RelocKind::Branch26,
                    "adrp_page21" => RelocKind::AdrpPage21,
                    "add_pageoff12" => RelocKind::AddPageOff12,
                    _ => return None,
                };
                let target = p.next()?.to_string();
                Some(Reloc { at, size, kind, target, addend: 0 })
            })
            .collect::<Option<Vec<_>>>()?
    };
    Some((asm, EncodedModule { code, relocs, ..Default::default() }))
}

/// Outcome of building a corpus.
pub struct CorpusBuild {
    pub text: String,
    pub recorded: usize,
    pub gaps: usize,
    pub oracle_rejects: usize,
    pub mismatches: usize,
}

/// Build a corpus: oracle goldens for every form rasm currently matches. Forms
/// the oracle rejects, rasm can't encode (gap), or rasm mis-encodes (mismatch)
/// are skipped and counted — so the committed corpus is always a green gate.
pub fn record_corpus(rasm: &dyn Encoder, oracle: &dyn Encoder, model: &dyn IsaModel) -> CorpusBuild {
    let mut b = CorpusBuild { text: String::new(), recorded: 0, gaps: 0, oracle_rejects: 0, mismatches: 0 };
    for form in model.forms() {
        let golden = match oracle.encode(&form.asm) {
            Ok(m) => m,
            Err(_) => {
                b.oracle_rejects += 1;
                continue;
            }
        };
        let rm = match rasm.encode(&form.asm) {
            Ok(m) => m,
            Err(_) => {
                b.gaps += 1;
                continue;
            }
        };
        if !compare(&form.asm, &rm, &golden).is_match() {
            b.mismatches += 1;
            continue;
        }
        b.text.push_str(&corpus_line(&form.asm, &golden));
        b.text.push('\n');
        b.recorded += 1;
    }
    b
}

// ── readable failures ────────────────────────────────────────────────────────

impl std::fmt::Display for Verdict {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Verdict::Match => write!(f, "match"),
            Verdict::ByteMismatch { asm, rasm, oracle, first_diff } => write!(
                f,
                "BYTE MISMATCH `{asm}` @{first_diff}\n  rasm:   {}\n  oracle: {}",
                hex(rasm),
                hex(oracle)
            ),
            Verdict::RelocMismatch { asm, rasm, oracle } => write!(
                f,
                "RELOC MISMATCH `{asm}`\n  rasm:   {rasm:?}\n  oracle: {oracle:?}"
            ),
            Verdict::RasmError { asm, err } => write!(f, "RASM GAP `{asm}`: {err}"),
            Verdict::OracleError { asm, err } => write!(f, "ORACLE REJECT `{asm}`: {err}"),
        }
    }
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect::<Vec<_>>().join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::{Reloc, RelocKind};

    /// The no-LLVM regression gate: rasm must reproduce every committed golden
    /// in `corpus/x86_64.tsv` byte-for-byte. Runs without the `llvm` feature —
    /// the corpus *is* the frozen oracle. Regenerate with `rasm-corpus` after
    /// closing gaps.
    #[test]
    fn corpus_replay_matches_golden() {
        use crate::rasm::RasmEncoder;
        let corpus = include_str!("../corpus/x86_64.tsv");
        let mut n = 0usize;
        for (i, line) in corpus.lines().enumerate() {
            if line.is_empty() {
                continue;
            }
            let (asm, golden) = parse_corpus_line(line)
                .unwrap_or_else(|| panic!("corpus line {} malformed: {line:?}", i + 1));
            let rm = RasmEncoder
                .encode(&asm)
                .unwrap_or_else(|e| panic!("rasm failed on corpus form `{asm}`: {e:#}"));
            let v = compare(&asm, &rm, &golden);
            assert!(v.is_match(), "corpus regression on `{asm}`: {v}");
            n += 1;
        }
        assert!(n > 1000, "corpus suspiciously small ({n} forms) — regenerate with `rasm-corpus`");
    }

    /// The no-LLVM AArch64 regression gate: the native `A64Encoder` must
    /// reproduce every committed golden in `corpus/aarch64.tsv` byte-for-byte.
    /// Runs without the `llvm` feature. Regenerate with `a64-corpus`.
    #[test]
    fn corpus_replay_aarch64_matches_golden() {
        use crate::a64::A64Encoder;
        let corpus = include_str!("../corpus/aarch64.tsv");
        let mut n = 0usize;
        for (i, line) in corpus.lines().enumerate() {
            if line.is_empty() {
                continue;
            }
            let (asm, golden) = parse_corpus_line(line)
                .unwrap_or_else(|| panic!("aarch64 corpus line {} malformed: {line:?}", i + 1));
            let rm = A64Encoder
                .encode(&asm)
                .unwrap_or_else(|e| panic!("a64 failed on corpus form `{asm}`: {e:#}"));
            let v = compare(&asm, &rm, &golden);
            assert!(v.is_match(), "aarch64 corpus regression on `{asm}`: {v}");
            n += 1;
        }
        assert!(n > 1000, "aarch64 corpus suspiciously small ({n} forms) — regenerate with `a64-corpus`");
    }

    // Unit tests for the normalization logic — no oracle needed.
    #[test]
    fn masking_then_compare_ignores_reloc_field_values() {
        let asm = "call ext";
        let r = EncodedModule {
            code: vec![0xE8, 0, 0, 0, 0],
            relocs: vec![Reloc { at: 1, size: 4, kind: RelocKind::BranchRel32, target: "ext".into(), addend: 0 }],
            ..Default::default()
        };
        // Oracle: same encoding, but its REL32 field happens to hold -4.
        let o = EncodedModule {
            code: vec![0xE8, 0xFC, 0xFF, 0xFF, 0xFF],
            relocs: vec![Reloc { at: 1, size: 4, kind: RelocKind::BranchRel32, target: "ext".into(), addend: 0 }],
            ..Default::default()
        };
        assert_eq!(compare(asm, &r, &o), Verdict::Match);
    }

    #[test]
    fn branch_and_riprel_are_one_class() {
        let a = [Reloc { at: 3, size: 4, kind: RelocKind::RipRel32, target: "g".into(), addend: 0 }];
        let b = [Reloc { at: 3, size: 4, kind: RelocKind::BranchRel32, target: "g".into(), addend: 0 }];
        assert!(relocs_equiv(&a, &b));
    }

    #[test]
    fn different_reloc_target_is_reloc_mismatch() {
        // Identical bytes, reloc at the same field, but a different target name →
        // masked code matches, so the divergence must surface as RelocMismatch.
        let asm = "call x";
        let base = EncodedModule { code: vec![0xE8, 0, 0, 0, 0], ..Default::default() };
        let r = EncodedModule {
            relocs: vec![Reloc { at: 1, size: 4, kind: RelocKind::BranchRel32, target: "g".into(), addend: 0 }],
            ..base.clone()
        };
        let o = EncodedModule {
            relocs: vec![Reloc { at: 1, size: 4, kind: RelocKind::BranchRel32, target: "h".into(), addend: 0 }],
            ..base
        };
        assert!(matches!(compare(asm, &r, &o), Verdict::RelocMismatch { .. }));
    }

    #[test]
    fn wrong_reloc_offset_surfaces_as_mismatch() {
        // A reloc at the wrong offset zeroes a different code byte, so it shows
        // up as a ByteMismatch — still caught, never a false Match.
        let asm = "lea rax, [rip + g]";
        let base = EncodedModule { code: vec![0x48, 0x8D, 0x05, 0, 0, 0, 0], ..Default::default() };
        let r = EncodedModule {
            relocs: vec![Reloc { at: 3, size: 4, kind: RelocKind::RipRel32, target: "g".into(), addend: 0 }],
            ..base.clone()
        };
        let o = EncodedModule {
            relocs: vec![Reloc { at: 2, size: 4, kind: RelocKind::RipRel32, target: "g".into(), addend: 0 }],
            ..base
        };
        assert!(compare(asm, &r, &o).is_mismatch());
    }

    #[test]
    fn byte_mismatch_reports_first_diff() {
        let asm = "mov rax, rbx";
        let r = EncodedModule { code: vec![0x48, 0x89, 0xD8], ..Default::default() };
        let o = EncodedModule { code: vec![0x48, 0x8B, 0xC3], ..Default::default() };
        match compare(asm, &r, &o) {
            Verdict::ByteMismatch { first_diff, .. } => assert_eq!(first_diff, 1),
            v => panic!("expected ByteMismatch, got {v}"),
        }
    }

    // Live differential against LLVM-MC — the real Phase-2 gate. Every form here
    // is within rasm's *current* claimed coverage, so this must be all-green and
    // locks the frontier. Adding a form rasm can't encode surfaces as a gap.
    #[cfg(feature = "llvm")]
    #[test]
    fn rasm_matches_llvm_on_current_coverage() {
        use crate::oracle::LlvmMcEncoder;
        use crate::rasm::RasmEncoder;

        const FORMS: &[&str] = &[
            // reg/imm ALU
            "mov rax, 42", "mov rcx, rdx", "add rax, rbx", "sub rsp, 8",
            "and rcx, 0xff", "or r8, r9", "xor rdx, rdx", "cmp rax, rbx", "adc rsi, rdi",
            // memory operands
            "mov [rbp - 8], rax", "add rax, [rbp]", "mov rax, [rsp + 16]",
            "lea rax, [rbp - 32]", "lea r10, [rdi + rsi*4]",
            // sized loads + extends (movsx is 8/16-bit src; 32→64 is movsxd)
            "movzx eax, byte ptr [rbx]", "movzx ecx, word ptr [rdx]",
            "movsx rax, word ptr [rcx]", "movsx r8, byte ptr [rdx]",
            "movsxd rax, dword ptr [rcx]", "movsxd rdx, ecx",
            // shifts / rotates
            "shl rax, 3", "sar rdx, cl", "shr r8, 1", "rol rbx, 7",
            // unary / mul / div
            "neg rax", "not rbx", "inc rcx", "dec r12", "imul rax, rbx", "mul rcx", "idiv rsi",
            // stack
            "push rbx", "pop r15", "push r12", "pop rbp",
            // setcc / cmov
            "setz al", "setne bl", "cmove rax, rbx", "cmovl r8, r9",
            // scalar-double SSE
            "addsd xmm0, xmm1", "mulsd xmm2, xmm3", "divsd xmm4, xmm5", "sqrtsd xmm0, xmm1",
            "cvtsi2sd xmm0, rax", "cvttsd2si rax, xmm0", "addsd xmm0, [rbp]",
            // zero-operand
            "ret", "leave", "cqo", "nop",
            // relocation-bearing forms — exercise masking + REL32 canonicalization
            "call rt_emit", "jmp some_target", "lea rax, [rip + extern_data]",
        ];

        let rasm = RasmEncoder;
        let oracle = LlvmMcEncoder::x86_64();
        let report = diff_all(&rasm, &oracle, FORMS.iter().copied());

        eprintln!("{}", report.summary());
        let bad = report.mismatches();
        let gaps = report.gaps();
        let dropped = report.oracle_errors();
        for v in bad.iter().chain(gaps.iter()).chain(dropped.iter()) {
            eprintln!("{v}");
        }
        assert!(bad.is_empty(), "{} byte/reloc mismatches vs LLVM", bad.len());
        assert!(gaps.is_empty(), "{} forms rasm couldn't encode (expected all in-coverage)", gaps.len());
        // Every form here is legal x86-64, so the oracle must accept them all.
        assert!(dropped.is_empty(), "oracle rejected forms it should assemble");
        assert_eq!(report.matched(), FORMS.len(), "every form should match");
    }

    /// The no-mismatch gate: across the *entire* generated form set, rasm must
    /// never accept a form and encode it differently from LLVM. Gaps (forms rasm
    /// can't encode yet) are allowed and expected; a *mismatch* is a silent
    /// miscompile and is forbidden. This is what makes growing rasm's coverage
    /// safe — no edit can reintroduce a wrong encoding without failing here.
    #[cfg(feature = "llvm")]
    #[test]
    fn generated_forms_never_misencode() {
        use crate::oracle::LlvmMcEncoder;
        use crate::rasm::RasmEncoder;

        let oracle = LlvmMcEncoder::x86_64();
        let report = diff_model(&RasmEncoder, &oracle, &x86::X86Model);
        eprintln!("{}", report.summary());
        let bad = report.mismatches();
        for (_, v) in &bad {
            eprintln!("{v}");
        }
        assert!(
            bad.is_empty(),
            "{} generated form(s) mis-encode vs LLVM (rasm accepts but emits wrong bytes)",
            bad.len()
        );
    }
}
