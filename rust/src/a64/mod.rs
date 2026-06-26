//! `a64` — the native, LLVM-free AArch64 encoder for Apple Silicon.
//!
//! The AArch64 sibling of [`rasm`](crate::rasm): it takes the assembled
//! (post-macro-expansion) AArch64 assembly text the `asm/` front-end produces and
//! emits an [`EncodedModule`](crate::backend::EncodedModule) the native loader
//! places. Tables/logic are derived from — and gated byte-for-byte against — the
//! LLVM-MC oracle ([`LlvmMcEncoder::aarch64_macos`](crate::oracle), behind the
//! `llvm` feature). See docs/design/aarch64-apple-silicon.md.
//!
//! Layering: [`parse`] (text → [`Line`](parse::Line)) → [`encode`] (one
//! instruction → one 32-bit word + fixups) → this module's driver (assign
//! offsets, resolve internal labels, emit relocs for externs).
//!
//! Unlike x86, AArch64 instructions are fixed-width, so there is no
//! rel8→rel32 relaxation: offsets are known after one layout pass. Out-of-range
//! branches will be handled by veneer insertion in a later phase (§3.4.3); the
//! first slice range-checks and errors instead.

pub mod encode;
pub mod parse;

use std::collections::BTreeMap;

use anyhow::{bail, Context, Result};

use crate::backend::{EncodedModule, Reloc, RelocKind};

use encode::{encode, FixupKind};
use parse::{Directive, Line};

/// The native AArch64 [`Encoder`](crate::backend::Encoder).
#[derive(Debug, Default, Clone, Copy)]
pub struct A64Encoder;

impl crate::backend::Encoder for A64Encoder {
    fn encode(&self, asm_text: &str) -> Result<EncodedModule> {
        assemble(asm_text)
    }
}

/// A laid-out item in the text stream.
enum Item {
    /// Fixed bytes with optional branch fixups (offset-within-bytes, kind, target).
    Code { bytes: Vec<u8>, fixups: Vec<(usize, FixupKind, String)> },
    Label(String),
    Globl(String),
    /// Pad to a 2^n boundary.
    AlignP2(u32),
    /// A relaxable conditional/compare branch (`b.cond`/`cbz`/`cbnz`/`tbz`/`tbnz`).
    /// `word` is the instruction with a zero immediate. When the target is out of
    /// the imm19/imm14 range, or is an extern, it relaxes to the **long** form —
    /// the condition inverted to skip an unconditional `b` to the target (which
    /// has ±128 MB reach, and may itself become a `Branch26` relocation/veneer).
    /// This is a deliberate extension beyond LLVM-MC, which errors on out-of-range
    /// conditional branches. See docs/design/aarch64-apple-silicon.md §3.4.3.
    CondBr { word: u32, target: String, is_long: bool },
}

impl Item {
    fn size_at(&self, off: usize) -> usize {
        match self {
            Item::Code { bytes, .. } => bytes.len(),
            Item::Label(_) | Item::Globl(_) => 0,
            Item::CondBr { is_long, .. } => {
                if *is_long {
                    8
                } else {
                    4
                }
            }
            Item::AlignP2(n) => {
                let align = 1usize << *n;
                (align - (off % align)) % align
            }
        }
    }
}

/// Invert a conditional/compare branch word: flip the `b.cond` condition, or the
/// `cbz`/`cbnz`/`tbz`/`tbnz` op bit (which swaps zero ↔ non-zero).
fn invert_cond(word: u32) -> u32 {
    if (word >> 24) & 0xFF == 0x54 {
        word ^ 1 // b.cond: invert the 4-bit condition (low bit)
    } else {
        word ^ (1 << 24) // cbz↔cbnz / tbz↔tbnz: flip op bit24
    }
}

/// Assemble a module's AArch64 text into an [`EncodedModule`].
pub fn assemble(text: &str) -> Result<EncodedModule> {
    // ── Pass 1: parse into items ────────────────────────────────────────────
    let mut items: Vec<Item> = Vec::new();
    for (lineno, raw) in text.lines().enumerate() {
        let clean = parse::strip_comment(raw);
        let (label, rest) = parse::split_leading_label(clean);
        if let Some(name) = label {
            items.push(Item::Label(name.to_string()));
            if rest.is_empty() {
                continue;
            }
        }
        let body = if label.is_some() { rest } else { clean };
        let line = parse::parse_line(body).with_context(|| format!("line {}: `{raw}`", lineno + 1))?;
        match line {
            Line::Empty => {}
            Line::Label(name) => items.push(Item::Label(name)),
            Line::Directive(d) => push_directive(&mut items, d)?,
            Line::Insn { mnemonic, ops } => {
                let enc = encode(&mnemonic, &ops)
                    .with_context(|| format!("line {}: encode `{raw}`", lineno + 1))?;
                // A conditional/compare branch (single Branch19 fixup) becomes a
                // relaxable CondBr item; everything else is fixed Code.
                if enc.fixups.len() == 1 && enc.fixups[0].kind == FixupKind::Branch19 {
                    let word = u32::from_le_bytes(enc.bytes[..4].try_into().unwrap());
                    items.push(Item::CondBr { word, target: enc.fixups[0].target.clone(), is_long: false });
                } else {
                    let fixups = enc.fixups.into_iter().map(|f| (f.at, f.kind, f.target)).collect();
                    items.push(Item::Code { bytes: enc.bytes, fixups });
                }
            }
        }
    }

    // ── Relaxation: grow conditional branches to the long form when their
    // target is out of imm19 range or is an extern. Grow-only ⇒ converges. ────
    loop {
        let (offsets, labels) = layout(&items);
        let mut changed = false;
        for (idx, it) in items.iter_mut().enumerate() {
            if let Item::CondBr { target, is_long, .. } = it {
                if *is_long {
                    continue;
                }
                let must_long = match labels.get(target) {
                    None => true, // extern → must use the long (b reloc) form
                    Some(&tgt) => {
                        let disp = tgt as i64 - offsets[idx] as i64;
                        !(-(1 << 20)..(1 << 20)).contains(&disp)
                    }
                };
                if must_long {
                    *is_long = true;
                    changed = true;
                }
            }
        }
        if !changed {
            break;
        }
    }

    // ── Layout: byte offset of each item + every label ──────────────────────
    let (offsets, labels) = layout(&items);

    // ── Emit: place bytes, patch internal branches, emit relocs for externs ─
    let mut code: Vec<u8> = Vec::new();
    let mut symbols: BTreeMap<String, usize> = BTreeMap::new();
    let mut relocs: Vec<Reloc> = Vec::new();
    let mut externs: Vec<String> = Vec::new();
    let mut globls: std::collections::HashSet<String> = std::collections::HashSet::new();

    for (idx, it) in items.iter().enumerate() {
        let off = offsets[idx];
        debug_assert_eq!(off, code.len());
        match it {
            Item::Label(n) => {
                symbols.insert(n.clone(), code.len());
            }
            Item::Globl(n) => {
                globls.insert(n.clone());
            }
            Item::AlignP2(n) => {
                let align = 1usize << *n;
                let pad = (align - (code.len() % align)) % align;
                write_nop_padding(&mut code, pad);
            }
            Item::Code { bytes, fixups } => {
                let base = code.len();
                code.extend_from_slice(bytes);
                for (at, kind, target) in fixups {
                    let field = base + at;
                    match kind {
                        // PC-relative page/page-offset are *always* relocated —
                        // the page address is only known at load time, so even a
                        // local symbol carries the reloc (matches LLVM-MC).
                        FixupKind::AdrpPage21 | FixupKind::AddPageOff12 => {
                            let rk = if *kind == FixupKind::AdrpPage21 {
                                RelocKind::AdrpPage21
                            } else {
                                RelocKind::AddPageOff12
                            };
                            relocs.push(Reloc { at: field, size: 4, kind: rk, target: target.clone(), addend: 0 });
                            if !labels.contains_key(target) {
                                externs.push(target.clone());
                            }
                        }
                        // Branches: patch internal targets, relocate externs.
                        FixupKind::Branch26 | FixupKind::Branch19 => {
                            if let Some(&tgt) = labels.get(target) {
                                patch_branch(&mut code, field, *kind, tgt)?;
                            } else if *kind == FixupKind::Branch26 {
                                relocs.push(Reloc {
                                    at: field,
                                    size: 4,
                                    kind: RelocKind::Branch26,
                                    target: target.clone(),
                                    addend: 0,
                                });
                                externs.push(target.clone());
                            } else {
                                bail!("conditional/compare branch to extern `{target}` needs a veneer (not in the first slice)");
                            }
                        }
                    }
                }
            }
            Item::CondBr { word, target, is_long } => {
                let base = code.len();
                if !*is_long {
                    // Short form: patch imm19 in place.
                    code.extend_from_slice(&word.to_le_bytes());
                    let tgt = *labels.get(target).expect("short cond-branch to extern is impossible");
                    let disp = tgt as i64 - base as i64;
                    let mut w = *word;
                    w |= (((disp >> 2) as u32) & 0x7_FFFF) << 5;
                    code[base..base + 4].copy_from_slice(&w.to_le_bytes());
                } else {
                    // Long form: `<inverted cond> +8` then `b target`.
                    let inv = invert_cond(*word) | (2u32 << 5); // imm19 = 2 → skip the b
                    code.extend_from_slice(&inv.to_le_bytes());
                    let b_at = code.len();
                    code.extend_from_slice(&0x1400_0000u32.to_le_bytes()); // b, imm26=0
                    if let Some(&tgt) = labels.get(target) {
                        let disp = tgt as i64 - b_at as i64;
                        if !(-(1 << 27)..(1 << 27)).contains(&disp) {
                            bail!("relaxed conditional branch target out of ±128MB range");
                        }
                        let w = 0x1400_0000u32 | (((disp >> 2) as u32) & 0x03FF_FFFF);
                        code[b_at..b_at + 4].copy_from_slice(&w.to_le_bytes());
                    } else {
                        // Conditional branch to an extern: the `b` is a Branch26
                        // relocation (the loader veneers it if it ends up far).
                        relocs.push(Reloc { at: b_at, size: 4, kind: RelocKind::Branch26, target: target.clone(), addend: 0 });
                        externs.push(target.clone());
                    }
                }
            }
        }
    }

    symbols.retain(|name, _| globls.contains(name));
    externs.sort();
    externs.dedup();

    Ok(EncodedModule { code, symbols, relocs, externs })
}

/// Patch a PC-relative branch immediate into the 4-byte word at `field`.
fn patch_branch(code: &mut [u8], field: usize, kind: FixupKind, target: usize) -> Result<()> {
    let site = field as i64;
    let disp = target as i64 - site;
    if disp % 4 != 0 {
        bail!("branch target not 4-byte aligned (disp {disp})");
    }
    let imm = disp >> 2;
    let mut w = u32::from_le_bytes(code[field..field + 4].try_into().unwrap());
    match kind {
        FixupKind::Branch26 => {
            if !(-(1 << 25)..(1 << 25)).contains(&imm) {
                bail!("b/bl target out of ±128MB range (needs a veneer)");
            }
            w |= (imm as u32) & 0x03FF_FFFF;
        }
        FixupKind::Branch19 => {
            if !(-(1 << 18)..(1 << 18)).contains(&imm) {
                bail!("conditional/compare branch out of ±1MB range (needs inversion+veneer)");
            }
            w |= ((imm as u32) & 0x7_FFFF) << 5;
        }
        FixupKind::AdrpPage21 | FixupKind::AddPageOff12 => {
            bail!("pc-relative fixups are resolved as relocations, not patched in-section");
        }
    }
    code[field..field + 4].copy_from_slice(&w.to_le_bytes());
    Ok(())
}

/// Pad with AArch64 `NOP` words (`0xD503201F`) for whole words, zero bytes for
/// any sub-word remainder (a pure instruction stream is always word-aligned).
fn write_nop_padding(code: &mut Vec<u8>, mut pad: usize) {
    const NOP: [u8; 4] = [0x1f, 0x20, 0x03, 0xd5];
    while pad >= 4 {
        code.extend_from_slice(&NOP);
        pad -= 4;
    }
    for _ in 0..pad {
        code.push(0);
    }
}

fn layout(items: &[Item]) -> (Vec<usize>, BTreeMap<String, usize>) {
    let mut offsets = Vec::with_capacity(items.len());
    let mut labels = BTreeMap::new();
    let mut off = 0usize;
    for it in items {
        offsets.push(off);
        if let Item::Label(n) = it {
            labels.insert(n.clone(), off);
        }
        off += it.size_at(off);
    }
    (offsets, labels)
}

fn push_directive(items: &mut Vec<Item>, d: Directive) -> Result<()> {
    match d {
        Directive::Text | Directive::Other(_) => {}
        Directive::Globl(n) => items.push(Item::Globl(n)),
        Directive::P2align(n) => items.push(Item::AlignP2(n)),
        Directive::Quad(vs) => items.push(Item::Code {
            bytes: vs.iter().flat_map(|v| v.to_le_bytes()).collect(),
            fixups: vec![],
        }),
        Directive::Byte(b) => items.push(Item::Code { bytes: vec![b], fixups: vec![] }),
        Directive::Zero(n) => items.push(Item::Code { bytes: vec![0u8; n], fixups: vec![] }),
        Directive::Ascii(bytes, nul) => {
            let mut v = bytes;
            if nul {
                v.push(0);
            }
            items.push(Item::Code { bytes: v, fixups: vec![] });
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn internal_branch_resolves_no_reloc() {
        // A countdown loop: cbz exits, b loops. Both internal → patched, no reloc.
        let src = "\
.globl countdown
countdown:
loop:
cbz x0, done
sub x0, x0, #1
b loop
done:
ret
";
        let m = assemble(src).unwrap();
        assert!(m.symbols.contains_key("countdown"));
        assert!(!m.symbols.contains_key("loop"), "local label not exported");
        assert!(m.relocs.is_empty(), "internal branches must not relocate: {:?}", m.relocs);
        assert!(m.externs.is_empty());
        // `b loop` is the 3rd word (offset 8): branch back to offset 0 → disp -8.
        // 0x17FFFFFE = b #-8.
        let b = u32::from_le_bytes(m.code[8..12].try_into().unwrap());
        assert_eq!(b, 0x17FF_FFFE, "b loop encodes as {b:#010x}");
    }

    #[test]
    fn cond_branch_to_extern_relaxes() {
        // `cbz x0, ext` → `cbnz x0, #8 ; b ext(reloc) ; ret`.
        let m = assemble(".globl w\nw:\ncbz x0, ext\nret\n").unwrap();
        assert_eq!(m.code.len(), 12, "long cond branch = 2 words + ret");
        assert_eq!(&m.code[0..4], &0xB500_0040u32.to_le_bytes(), "cbnz x0, #8");
        assert_eq!(&m.code[4..8], &0x1400_0000u32.to_le_bytes(), "b ext (imm26=0, reloc)");
        assert_eq!(m.relocs.len(), 1);
        assert_eq!(m.relocs[0].kind, RelocKind::Branch26);
        assert_eq!(m.relocs[0].target, "ext");
        assert_eq!(m.relocs[0].at, 4);
        assert_eq!(m.externs, vec!["ext".to_string()]);
    }

    #[test]
    fn far_conditional_branch_relaxes_to_inverted_plus_b() {
        // `cbz x0, far` with `far` > 1MB away → inverted cond + unconditional b.
        let src = ".globl w\nw:\ncbz x0, far\n.space 0x100000\nfar:\nret\n";
        let m = assemble(src).unwrap();
        assert!(m.relocs.is_empty(), "internal target → no relocs");
        // cbz relaxed to 8 bytes; `far` now sits at 8 + 0x100000.
        assert_eq!(m.code.len(), 0x10000C);
        assert_eq!(&m.code[0..4], &0xB500_0040u32.to_le_bytes(), "cbnz x0, #8");
        // b at offset 4 → far (0x100008): disp 0x100004, imm26 0x40001.
        assert_eq!(&m.code[4..8], &0x1404_0001u32.to_le_bytes(), "b far");
    }

    #[test]
    fn extern_bl_becomes_branch26_reloc() {
        let src = ".globl w\nw:\nbl rt_emit\nret\n";
        let m = assemble(src).unwrap();
        assert_eq!(m.externs, vec!["rt_emit".to_string()]);
        assert_eq!(m.relocs.len(), 1);
        assert_eq!(m.relocs[0].kind, RelocKind::Branch26);
        assert_eq!(m.relocs[0].target, "rt_emit");
        assert_eq!(m.relocs[0].at, 0);
        // bl placeholder = 0x94000000 (imm26 = 0).
        assert_eq!(&m.code[0..4], &[0x00, 0x00, 0x00, 0x94]);
    }
}

/// The gate: the native [`A64Encoder`] must be byte-identical to the LLVM-MC
/// oracle ([`LlvmMcEncoder::aarch64_macos`](crate::oracle::LlvmMcEncoder)) across
/// the implemented instruction frontier — the AArch64 analogue of the x86
/// `rasm`-vs-LLVM differential. Build/run with `--features llvm`.
#[cfg(all(test, feature = "llvm"))]
mod oracle_diff {
    use super::*;
    use crate::backend::Encoder;
    use crate::oracle::LlvmMcEncoder;

    /// Diff one form's `code` against the oracle, reporting both on mismatch.
    fn diff_code(asm: &str) {
        let o = LlvmMcEncoder::aarch64_macos().encode(asm).expect("oracle");
        let r = A64Encoder.encode(asm).expect("a64 encode");
        assert_eq!(
            r.code, o.code,
            "\n  asm:    {asm}\n  a64:    {:02x?}\n  oracle: {:02x?}",
            r.code, o.code
        );
    }

    #[test]
    fn first_slice_matches_oracle() {
        #[rustfmt::skip]
        let forms = [
            // system / no-operand
            "ret", "nop", "brk #0", "brk #1",
            // moves (incl. high registers to exercise the reg fields)
            "mov x0, x1", "mov x5, x9", "mov x28, x27", "mov w3, w4", "mov w0, wzr",
            "mvn x0, x1", "mvn w7, w8",
            "movz x0, #0x1234", "movz x9, #0xffff, lsl #48", "movz w2, #0x42",
            "movk x0, #0xabcd, lsl #32", "movn x0, #0", "movn x2, #0xffff, lsl #16",
            // add/sub — register, shifted register, immediate, lsl #12
            "add x0, x1, x2", "add w0, w1, w2", "sub x3, x4, x5", "sub w6, w7, w8",
            "add x0, x1, x2, lsl #3", "sub x9, x10, x11, lsl #1",
            "add x0, x1, #42", "sub sp, sp, #16", "add sp, sp, #16",
            "add x0, x1, #1, lsl #12", "add x28, x29, #4095",
            // logical (register + shifted register)
            "and x0, x1, x2", "orr x3, x4, x5", "eor x0, x1, x2",
            "and w0, w1, w2", "orr x0, x1, x2, lsl #4", "eor x7, x8, x9, lsr #2",
            // loads / stores (unsigned scaled offset)
            "ldr x0, [x1, #8]", "str x0, [x1]", "ldr w0, [x2, #16]",
            "ldr x5, [sp, #0]", "str x9, [x10, #4088]", "str w11, [x12, #4092]",
            // indirect branches
            "br x16", "blr x17",
        ];
        for asm in forms {
            diff_code(asm);
        }
    }

    #[test]
    fn alu_extended_families_match_oracle() {
        #[rustfmt::skip]
        let forms = [
            // adds/subs + cmp/cmn/tst/neg aliases
            "adds x0, x1, x2", "subs w3, w4, w5", "adds x0, x1, #42", "subs x6, x7, #4095",
            "cmp x1, x2", "cmp x1, #42", "cmn x1, x2", "cmp w1, w2", "cmp sp, #16",
            "tst x1, x2", "tst x0, #0xff", "neg x0, x1", "negs x0, x1", "neg x0, x1, lsl #2",
            // extended-register add/sub for SP
            "add x0, sp, x1", "add sp, x0, x1", "sub sp, sp, x1", "add x0, sp, x1, lsl #3",
            "mov sp, x9", "mov x9, sp",
            // multiply / divide
            "mul x0, x1, x2", "mul w0, w1, w2", "madd x0, x1, x2, x3", "msub x0, x1, x2, x3",
            "mneg x0, x1, x2", "smull x0, w1, w2", "umull x0, w1, w2",
            "sdiv x0, x1, x2", "udiv x0, x1, x2", "sdiv w0, w1, w2",
            // variable + immediate shifts
            "lsl x0, x1, x2", "lsr x0, x1, x2", "asr x0, x1, x2", "ror x0, x1, x2",
            "lsl x0, x1, #3", "lsr x0, x1, #3", "asr x0, x1, #3", "lsl w0, w1, #3", "lsr x0, x1, #63",
            // bitfield / extends
            "ubfx x0, x1, #4, #8", "sbfx x0, x1, #4, #8", "ubfiz x0, x1, #4, #8",
            "sxtw x0, w1", "sxtb x0, w1", "sxth x0, w1", "uxtb w0, w1", "uxth w0, w1",
            // bitmask-immediate logicals + mov #imm lowering
            "and x0, x1, #0xff", "orr x0, x1, #0xf", "eor x0, x1, #0x1",
            "and w0, w1, #0xff", "ands x0, x1, #0xff", "orr x0, xzr, #0x1",
            "mov x0, #0x1234", "mov x0, #-1", "mov w0, #0xffff0000", "mov x0, #0x10000",
            "mov x0, #0xffff", "mov x0, #0x5555555555555555", "mov x0, #0xffffffffffff0000",
            // conditional select family
            "csel x0, x1, x2, eq", "csinc x0, x1, x2, ne", "csinv x0, x1, x2, lt",
            "csneg x0, x1, x2, ge", "cset x0, eq", "cset w0, ne", "csetm x0, lt", "cinc x0, x1, eq",
        ];
        for asm in forms {
            diff_code(asm);
        }
    }

    #[test]
    fn loads_stores_match_oracle() {
        #[rustfmt::skip]
        let forms = [
            // byte / halfword / signed loads (unsigned offset)
            "ldrb w0, [x1, #4]", "strb w0, [x1]", "ldrh w0, [x1, #8]", "strh w0, [x1, #2]",
            "ldrsw x0, [x1, #4]", "ldrsb x0, [x1]", "ldrsh x0, [x1, #2]", "ldrsb w0, [x1]",
            // pre/post-index (imm9, incl. negative)
            "ldr x0, [x1, #-8]!", "str x0, [x1], #8", "ldr w0, [x2, #-4]!", "ldrb w0, [x1], #1",
            "ldr x0, [x1, #-8]", "ldur x0, [x1, #7]",
            // register offset
            "ldr x0, [x1, x2]", "ldr x0, [x1, x2, lsl #3]", "str x0, [x1, x2, lsl #3]",
            "ldr w0, [x1, x2, lsl #2]", "ldr x0, [x1, w2, sxtw]", "ldr x0, [x1, w2, sxtw #3]",
            // pairs
            "ldp x0, x1, [x2]", "ldp x0, x1, [x2, #16]", "stp x0, x1, [sp, #-16]!",
            "ldp x29, x30, [sp], #16", "ldp w0, w1, [x2, #8]",
        ];
        for asm in forms {
            diff_code(asm);
        }
    }

    #[test]
    fn fp_scalar_match_oracle() {
        #[rustfmt::skip]
        let forms = [
            // 2-source (S and D)
            "fadd s0, s1, s2", "fadd d0, d1, d2", "fsub s3, s4, s5", "fmul d6, d7, d8",
            "fdiv s0, s1, s2", "fnmul d0, d1, d2", "fmax s0, s1, s2", "fmin d0, d1, d2",
            "fmaxnm s0, s1, s2", "fminnm d0, d1, d2",
            // 1-source
            "fabs s0, s1", "fneg d0, d1", "fsqrt s0, s1", "fmov d0, d1",
            "frinta s0, s1", "frintn d0, d1", "frintm s0, s1", "frintz d0, d1", "frintp s0, s1",
            // convert between sizes
            "fcvt d0, s1", "fcvt s0, d1",
            // 3-source
            "fmadd s0, s1, s2, s3", "fmsub d0, d1, d2, d3",
            "fnmadd s0, s1, s2, s3", "fnmsub d0, d1, d2, d3",
            // compare / conditional
            "fcmp s0, s1", "fcmp d0, d1", "fcmp s0, #0.0", "fcmpe d0, d1", "fcmpe s0, #0.0",
            "fccmp s0, s1, #0, eq", "fccmpe d0, d1, #15, ne",
            "fcsel s0, s1, s2, eq", "fcsel d0, d1, d2, ne",
            // convert FP↔int
            "fcvtzs w0, s0", "fcvtzs x0, d0", "fcvtzu w0, d0", "fcvtzu x0, s0",
            "fcvtas w0, s0", "fcvtau x0, d0", "fcvtms w0, s0", "fcvtps x0, d0", "fcvtns w0, s0",
            "scvtf s0, w0", "scvtf d0, x0", "ucvtf d0, w0", "ucvtf s0, x0",
            // fmov gpr↔fp + immediate
            "fmov w0, s1", "fmov s0, w1", "fmov x0, d1", "fmov d0, x1",
            "fmov d0, #1.0", "fmov s0, #2.0", "fmov d0, #0.5", "fmov s0, #-1.0", "fmov d0, #1.9375",
        ];
        for asm in forms {
            diff_code(asm);
        }
    }

    #[test]
    fn simd_neon_match_oracle() {
        #[rustfmt::skip]
        let forms = [
            // 3-same integer (arrangements exercise Q + size)
            "add v0.16b, v1.16b, v2.16b", "add v0.4s, v1.4s, v2.4s", "add v0.2d, v1.2d, v2.2d",
            "sub v0.8h, v1.8h, v2.8h", "mul v0.4s, v1.4s, v2.4s", "mla v0.4s, v1.4s, v2.4s",
            "mls v0.8b, v1.8b, v2.8b", "smax v0.4s, v1.4s, v2.4s", "smin v0.4s, v1.4s, v2.4s",
            "umax v0.8h, v1.8h, v2.8h", "umin v0.8h, v1.8h, v2.8h",
            "sshl v0.4s, v1.4s, v2.4s", "ushl v0.2d, v1.2d, v2.2d",
            "sqadd v0.4s, v1.4s, v2.4s", "uqadd v0.8h, v1.8h, v2.8h",
            "sqsub v0.4s, v1.4s, v2.4s", "uqsub v0.8h, v1.8h, v2.8h",
            "cmeq v0.4s, v1.4s, v2.4s", "cmgt v0.8h, v1.8h, v2.8h", "cmge v0.4s, v1.4s, v2.4s",
            "cmhi v0.8h, v1.8h, v2.8h", "cmhs v0.16b, v1.16b, v2.16b", "cmtst v0.4s, v1.4s, v2.4s",
            "addp v0.4s, v1.4s, v2.4s",
            // 3-same logical (.8b/.16b)
            "and v0.16b, v1.16b, v2.16b", "orr v0.16b, v1.16b, v2.16b", "eor v0.16b, v1.16b, v2.16b",
            "bic v0.8b, v1.8b, v2.8b", "orn v0.16b, v1.16b, v2.16b", "bsl v0.16b, v1.16b, v2.16b",
            // 3-same FP
            "fadd v0.4s, v1.4s, v2.4s", "fsub v0.4s, v1.4s, v2.4s", "fmul v0.2d, v1.2d, v2.2d",
            "fdiv v0.2d, v1.2d, v2.2d", "fmla v0.4s, v1.4s, v2.4s", "fmls v0.4s, v1.4s, v2.4s",
            "fmax v0.4s, v1.4s, v2.4s", "fmin v0.2d, v1.2d, v2.2d",
            "fcmeq v0.4s, v1.4s, v2.4s", "fcmgt v0.4s, v1.4s, v2.4s",
            // 2-misc integer
            "neg v0.4s, v1.4s", "abs v0.2d, v1.2d", "not v0.16b, v1.16b", "cnt v0.8b, v1.8b",
            "clz v0.4s, v1.4s", "cls v0.8h, v1.8h", "rev64 v0.4s, v1.4s", "rev32 v0.8b, v1.8b",
            "rev16 v0.16b, v1.16b",
            // 2-misc FP
            "fneg v0.4s, v1.4s", "fabs v0.2d, v1.2d", "fsqrt v0.4s, v1.4s",
            // across-lane reductions
            "addv s0, v1.4s", "addv b0, v1.8b", "smaxv s0, v1.4s", "uminv h0, v1.8h",
            "saddlv h0, v1.8b",
            // dup (element + GP)
            "dup v0.4s, v1.s[1]", "dup v0.16b, w1", "dup v0.2d, x1", "dup v0.8h, v1.h[3]",
        ];
        for asm in forms {
            diff_code(asm);
        }
    }

    #[test]
    fn simd_struct_ldst_match_oracle() {
        #[rustfmt::skip]
        let forms = [
            "ld1 {v0.16b}, [x0]", "ld1 {v0.16b, v1.16b}, [x0]", "ld1 {v0.4s}, [x1]",
            "ld1 {v0.2d, v1.2d}, [x1]", "st1 {v0.16b}, [x0]", "st1 {v0.4s, v1.4s}, [x2]",
            "ld2 {v0.4s, v1.4s}, [x0]", "st2 {v0.8h, v1.8h}, [x0]",
            "ld3 {v0.8h, v1.8h, v2.8h}, [x0]", "ld4 {v0.4s, v1.4s, v2.4s, v3.4s}, [x0]",
            // post-index (immediate + register)
            "ld1 {v0.16b}, [x0], #16", "ld1 {v0.16b, v1.16b}, [x0], #32",
            "ld1 {v0.16b}, [x0], x2", "st1 {v0.4s}, [x0], #16",
        ];
        for asm in forms {
            diff_code(asm);
        }
    }

    #[test]
    fn simd_byelement_match_oracle() {
        #[rustfmt::skip]
        let forms = [
            "mul v0.4s, v1.4s, v2.s[1]", "mul v0.8h, v1.8h, v2.h[7]",
            "mla v0.4s, v1.4s, v2.s[3]", "mls v0.8h, v1.8h, v2.h[2]",
            "fmul v0.4s, v1.4s, v2.s[2]", "fmul v0.2d, v1.2d, v2.d[1]",
            "fmla v0.4s, v1.4s, v2.s[0]", "fmls v0.2d, v1.2d, v2.d[0]", "fmulx v0.4s, v1.4s, v2.s[1]",
            "sqdmulh v0.4s, v1.4s, v2.s[1]", "sqrdmulh v0.8h, v1.8h, v2.h[5]",
            "smull v0.4s, v1.4h, v2.h[3]", "smull2 v0.4s, v1.8h, v2.h[7]",
            "umull v0.2d, v1.2s, v2.s[1]", "smlal v0.4s, v1.4h, v2.h[0]", "umlal v0.2d, v1.2s, v2.s[3]",
        ];
        for asm in forms {
            diff_code(asm);
        }
    }

    #[test]
    fn simd_widen_narrow_match_oracle() {
        #[rustfmt::skip]
        let forms = [
            // long (3-diff): size+Q from the narrow source Vn
            "saddl v0.8h, v1.8b, v2.8b", "saddl2 v0.8h, v1.16b, v2.16b", "uaddl v0.4s, v1.4h, v2.4h",
            "ssubl v0.8h, v1.8b, v2.8b", "usubl v0.2d, v1.2s, v2.2s",
            "smull v0.8h, v1.8b, v2.8b", "umull2 v0.4s, v1.8h, v2.8h",
            "smlal v0.8h, v1.8b, v2.8b", "umlal v0.4s, v1.4h, v2.4h", "smlsl v0.2d, v1.2s, v2.2s",
            // widening (saddw): size+Q from the narrow Vm
            "saddw v0.8h, v1.8h, v2.8b", "uaddw2 v0.8h, v1.8h, v2.16b", "ssubw v0.4s, v1.4s, v2.4h",
            // narrowing (3-op): size+Q from the narrow Vd
            "addhn v0.8b, v1.8h, v2.8h", "addhn2 v0.16b, v1.8h, v2.8h", "subhn v0.4h, v1.4s, v2.4s",
            "raddhn v0.8b, v1.8h, v2.8h", "rsubhn v0.2s, v1.2d, v2.2d",
            // narrowing (2-op): size+Q from Vd
            "xtn v0.8b, v1.8h", "xtn2 v0.16b, v1.8h", "sqxtn v0.4h, v1.4s",
            "uqxtn v0.8b, v1.8h", "sqxtun v0.2s, v1.2d",
        ];
        for asm in forms {
            diff_code(asm);
        }
    }

    #[test]
    fn simd_movi_match_oracle() {
        #[rustfmt::skip]
        let forms = [
            "movi v0.16b, #0xab", "movi v0.8b, #0xab",
            "movi v0.4s, #0xab", "movi v0.4s, #0xab, lsl #8", "movi v0.4s, #0xab, lsl #16",
            "movi v0.4s, #0xab, lsl #24", "movi v0.4s, #0xab, msl #8", "movi v0.4s, #0xab, msl #16",
            "movi v0.8h, #0xab", "movi v0.8h, #0xab, lsl #8",
            "movi v0.2d, #0xff00ff00ff00ff00", "movi v0.2d, #0xffffffffffffffff",
            "movi d0, #0xff00ff00ff00ff00",
            "mvni v0.4s, #0xab", "mvni v0.4s, #0xab, lsl #24", "mvni v0.4s, #0xab, msl #8",
            "mvni v0.8h, #0xab, lsl #8",
            "bic v0.4s, #0xab", "bic v0.4s, #0xab, lsl #8", "bic v0.8h, #0xab, lsl #8",
            "orr v0.4s, #0xab", "orr v0.4s, #0xab, lsl #16", "orr v0.8h, #0xab",
        ];
        for asm in forms {
            diff_code(asm);
        }
    }

    #[test]
    fn simd_fp_vector_match_oracle() {
        #[rustfmt::skip]
        let forms = [
            // 3-same FP additions
            "fabd v0.4s, v1.4s, v2.4s", "fmulx v0.4s, v1.4s, v2.4s",
            "fmaxnm v0.4s, v1.4s, v2.4s", "fminnm v0.4s, v1.4s, v2.4s",
            "faddp v0.4s, v1.4s, v2.4s", "fmaxp v0.4s, v1.4s, v2.4s", "fminp v0.4s, v1.4s, v2.4s",
            "fmaxnmp v0.4s, v1.4s, v2.4s", "fminnmp v0.2d, v1.2d, v2.2d", "fcmge v0.4s, v1.4s, v2.4s",
            // 2-misc FP convert / round (vector)
            "scvtf v0.4s, v1.4s", "ucvtf v0.2d, v1.2d", "fcvtzs v0.4s, v1.4s", "fcvtzu v0.2d, v1.2d",
            "frintn v0.4s, v1.4s", "frintp v0.2d, v1.2d", "frintm v0.4s, v1.4s", "frintz v0.4s, v1.4s",
            // FP compare against zero (vector)
            "fcmeq v0.4s, v1.4s, #0.0", "fcmge v0.4s, v1.4s, #0.0", "fcmgt v0.2d, v1.2d, #0.0",
            "fcmle v0.4s, v1.4s, #0.0", "fcmlt v0.4s, v1.4s, #0.0",
            // FP narrow / long convert
            "fcvtn v0.4h, v1.4s", "fcvtn2 v0.8h, v1.4s", "fcvtn v0.2s, v1.2d",
            "fcvtl v0.4s, v1.4h", "fcvtl2 v0.4s, v1.8h", "fcvtl v0.2d, v1.2s", "fcvtxn v0.2s, v1.2d",
        ];
        for asm in forms {
            diff_code(asm);
        }
    }

    #[test]
    fn simd_shift_imm_match_oracle() {
        #[rustfmt::skip]
        let forms = [
            "shl v0.4s, v1.4s, #3", "shl v0.16b, v1.16b, #1", "shl v0.2d, v1.2d, #20",
            "sshr v0.4s, v1.4s, #3", "ushr v0.8h, v1.8h, #2", "sshr v0.2d, v1.2d, #40",
            "ssra v0.4s, v1.4s, #3", "usra v0.4s, v1.4s, #3",
            "srshr v0.4s, v1.4s, #3", "urshr v0.8h, v1.8h, #2",
            "srsra v0.4s, v1.4s, #3", "ursra v0.8h, v1.8h, #2",
            "sli v0.4s, v1.4s, #3", "sri v0.8h, v1.8h, #2",
        ];
        for asm in forms {
            diff_code(asm);
        }
    }

    #[test]
    fn simd_copy_permute_match_oracle() {
        #[rustfmt::skip]
        let forms = [
            // permute
            "zip1 v0.4s, v1.4s, v2.4s", "zip2 v0.16b, v1.16b, v2.16b",
            "uzp1 v0.4s, v1.4s, v2.4s", "uzp2 v0.8h, v1.8h, v2.8h",
            "trn1 v0.4s, v1.4s, v2.4s", "trn2 v0.2d, v1.2d, v2.2d",
            "ext v0.16b, v1.16b, v2.16b, #4", "ext v0.8b, v1.8b, v2.8b, #3",
            // ins (element ← element / GP)
            "ins v0.s[1], v1.s[2]", "ins v0.b[3], v1.b[5]", "ins v0.d[1], v1.d[0]",
            "ins v0.s[1], w2", "ins v0.d[1], x2",
            // umov / smov
            "umov w0, v1.s[2]", "umov x0, v1.d[1]", "umov w0, v1.b[3]",
            "smov w0, v1.b[3]", "smov x0, v1.h[2]",
            // tbl / tbx (1-4 table registers)
            "tbl v0.8b, {v1.16b}, v2.8b", "tbl v0.16b, {v1.16b}, v2.16b",
            "tbl v0.16b, {v1.16b, v2.16b}, v3.16b", "tbl v0.16b, {v1.16b, v2.16b, v3.16b}, v4.16b",
            "tbl v0.16b, {v1.16b, v2.16b, v3.16b, v4.16b}, v5.16b",
            "tbx v0.16b, {v1.16b, v2.16b}, v3.16b",
        ];
        for asm in forms {
            diff_code(asm);
        }
    }

    #[test]
    fn crypto_and_fixedpoint_match_oracle() {
        #[rustfmt::skip]
        let forms = [
            // AES
            "aese v0.16b, v1.16b", "aesd v0.16b, v1.16b", "aesmc v0.16b, v1.16b", "aesimc v0.16b, v1.16b",
            // SHA 2-op + 3-op
            "sha1h s0, s1", "sha1su1 v0.4s, v1.4s", "sha256su0 v0.4s, v1.4s",
            "sha1c q0, s1, v2.4s", "sha1p q0, s1, v2.4s", "sha1m q0, s1, v2.4s",
            "sha1su0 v0.4s, v1.4s, v2.4s", "sha256h q0, q1, v2.4s", "sha256h2 q0, q1, v2.4s",
            "sha256su1 v0.4s, v1.4s, v2.4s",
            // fixed-point FP↔int
            "fcvtzs w0, s0, #4", "fcvtzu x0, d0, #8", "scvtf s0, w0, #4", "ucvtf d0, x0, #16",
            "fcvtzs x0, d0, #1", "scvtf d0, x0, #32",
        ];
        for asm in forms {
            diff_code(asm);
        }
    }

    #[test]
    fn integer_tail_match_oracle() {
        #[rustfmt::skip]
        let forms = [
            "ccmp x0, x1, #0, eq", "ccmp w0, #31, #15, ne", "ccmn x0, x1, #4, ge",
            "adc x0, x1, x2", "sbc w0, w1, w2", "adcs x0, x1, x2", "sbcs x0, x1, x2",
            "ngc x0, x1", "ngcs w0, w1",
            "smulh x0, x1, x2", "umulh x0, x1, x2",
            "crc32b w0, w1, w2", "crc32h w0, w1, w2", "crc32w w0, w1, w2", "crc32x w0, w1, x2",
            "crc32cb w0, w1, w2", "crc32cx w0, w1, x2",
        ];
        for asm in forms {
            diff_code(asm);
        }
    }

    #[test]
    fn system_match_oracle() {
        #[rustfmt::skip]
        let forms = [
            "nop", "yield", "wfe", "wfi", "sev", "sevl", "hint #7",
            "svc #0", "brk #0", "hlt #0",
            "dmb ish", "dmb sy", "dsb sy", "dsb ishst", "isb",
            "msr daifset, #2", "msr daifclr, #15", "msr spsel, #1",
            "mrs x0, nzcv", "msr nzcv, x0", "mrs x1, fpcr", "msr fpsr, x2",
            "mrs x0, tpidr_el0", "mrs x3, cntvct_el0", "mrs x0, ctr_el0",
        ];
        for asm in forms {
            diff_code(asm);
        }
    }

    #[test]
    fn atomics_match_oracle() {
        #[rustfmt::skip]
        let forms = [
            // exclusives
            "ldxr x0, [x1]", "ldxr w0, [x1]", "stxr w0, x1, [x2]", "ldaxr x0, [x1]",
            "stlxr w0, x1, [x2]", "ldar x0, [x1]", "stlr x0, [x1]", "ldarb w0, [x1]",
            "ldxrb w0, [x1]", "stxrh w0, w1, [x2]",
            // LSE atomics
            "ldadd x0, x1, [x2]", "ldadda x0, x1, [x2]", "ldaddl x0, x1, [x2]", "ldaddal x0, x1, [x2]",
            "ldaddb w0, w1, [x2]", "ldadd w0, w1, [x2]",
            "ldclr x0, x1, [x2]", "ldeor x0, x1, [x2]", "ldset x0, x1, [x2]",
            "ldsmax x0, x1, [x2]", "ldumin x0, x1, [x2]",
            "swp x0, x1, [x2]", "swpal x0, x1, [x2]", "swpb w0, w1, [x2]",
            "cas x0, x1, [x2]", "casal x0, x1, [x2]", "casb w0, w1, [x2]",
        ];
        for asm in forms {
            diff_code(asm);
        }
    }

    #[test]
    fn fp_simd_loads_match_oracle() {
        #[rustfmt::skip]
        let forms = [
            "ldr q0, [x1, #16]", "str q0, [x1]", "ldr d0, [x1, #8]", "ldr s0, [x1, #4]",
            "str d0, [x1, #-8]!", "ldr s0, [x2], #4", "ldr b0, [x1]", "ldr h0, [x1, #2]",
            "ldr q0, [x1, x2, lsl #4]", "ldr d0, [x1, x2, lsl #3]",
            "ldp q0, q1, [x2]", "ldp q0, q1, [x2, #32]", "stp d0, d1, [sp, #-16]!",
            "ldp s0, s1, [x2, #8]", "ldp d29, d30, [sp], #16",
        ];
        for asm in forms {
            diff_code(asm);
        }
    }

    #[test]
    fn misc_dataproc_match_oracle() {
        #[rustfmt::skip]
        let forms = [
            // count / reverse (1-source)
            "clz x0, x1", "cls x0, x1", "rbit x0, x1", "rev x0, x1", "rev16 x0, x1",
            "rev32 x0, x1", "rev w0, w1", "clz w3, w4",
            // extract / rotate-immediate
            "extr x0, x1, x2, #8", "ror x0, x1, #8", "extr w0, w1, w2, #3",
            // bitfield insert
            "bfi x0, x1, #4, #8", "bfxil x0, x1, #4, #8",
        ];
        for asm in forms {
            diff_code(asm);
        }
    }

    /// A whole function with an internal countdown loop — exercises label layout
    /// and internal branch patching (cbz/b), which the oracle resolves in-section
    /// too, so the modules must be byte-identical and reloc-free on both sides.
    ///
    /// Note the `L`-prefixed locals: LLVM-MC's Mach-O assembler rejects a
    /// conditional/compare branch to a *non-local* label ("requires
    /// assembler-local label"), since those branches are never relocated. The
    /// native encoder resolves any internal label, but the oracle (and therefore
    /// the front-end's macro-generated internal labels on AArch64/Mach-O) must
    /// use the `L…` local convention. See docs/design/aarch64-apple-silicon.md.
    #[test]
    fn function_with_internal_loop_matches_oracle() {
        let asm = "\
.globl countdown
countdown:
Lloop:
cbz x0, Ldone
sub x0, x0, #1
b Lloop
Ldone:
ret
";
        let o = LlvmMcEncoder::aarch64_macos().encode(asm).expect("oracle");
        let r = A64Encoder.encode(asm).expect("a64 encode");
        assert_eq!(r.code, o.code, "\n  a64:    {:02x?}\n  oracle: {:02x?}", r.code, o.code);
        assert!(r.relocs.is_empty() && o.relocs.is_empty(), "internal branches: no relocs");
        assert_eq!(r.symbols.get("countdown"), o.symbols.get("countdown"));
    }

    /// pc-relative address materialization: `adrp Xd, sym@PAGE` +
    /// `add Xd, Xn, sym@PAGEOFF`. Both are *always* relocated (the page address
    /// is load-time), so the modules must agree on code bytes *and* the
    /// `AdrpPage21`/`AddPageOff12` reloc list — for an extern and a local symbol.
    #[test]
    fn pcrel_adrp_add_matches_oracle() {
        for (asm, undefined) in [
            (".globl w\nw:\nadrp x0, ext_g@PAGE\nadd x0, x0, ext_g@PAGEOFF\nret\n", true),
            (".globl w\nw:\nadrp x1, w@PAGE\nadd x1, x1, w@PAGEOFF\nret\n", false),
        ] {
            let o = LlvmMcEncoder::aarch64_macos().encode(asm).expect("oracle");
            let r = A64Encoder.encode(asm).expect("a64 encode");
            assert_eq!(r.code, o.code, "\n  asm: {asm}\n  a64:    {:02x?}\n  oracle: {:02x?}", r.code, o.code);
            // Same reloc kinds/targets/offsets (order-independent).
            let key = |m: &EncodedModule| {
                let mut v: Vec<_> =
                    m.relocs.iter().map(|x| (x.at, format!("{:?}", x.kind), x.target.clone())).collect();
                v.sort();
                v
            };
            assert_eq!(key(&r), key(&o), "reloc mismatch\n  a64:    {:?}\n  oracle: {:?}", r.relocs, o.relocs);
            assert_eq!(r.relocs.len(), 2, "adrp + add → two relocs");
            if undefined {
                assert!(r.externs.contains(&"ext_g".to_string()));
            }
        }
    }

    /// An external `bl` — both sides emit a zero-placeholder word + a `Branch26`
    /// reloc to the same target at the same offset.
    #[test]
    fn extern_bl_reloc_matches_oracle() {
        let asm = ".globl w\nw:\nbl rt_emit\nret\n";
        let o = LlvmMcEncoder::aarch64_macos().encode(asm).expect("oracle");
        let r = A64Encoder.encode(asm).expect("a64 encode");
        assert_eq!(r.code, o.code, "\n  a64:    {:02x?}\n  oracle: {:02x?}", r.code, o.code);
        assert_eq!(r.relocs.len(), 1);
        assert_eq!(o.relocs.len(), 1);
        assert_eq!(r.relocs[0].kind, RelocKind::Branch26);
        assert_eq!(o.relocs[0].kind, RelocKind::Branch26);
        assert_eq!(r.relocs[0].target, o.relocs[0].target);
        assert_eq!(r.relocs[0].at, o.relocs[0].at);
    }
}
