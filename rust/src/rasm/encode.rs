//! Encode one parsed [`Insn`](super::Line) into machine-code bytes + fixups.
//!
//! The encoding machinery (REX / ModRM / SIB / displacement / immediate) is the
//! reusable core; per-mnemonic logic sits on top. Branch/call/RIP-rel targets
//! become [`Fixup`]s the two-pass driver later resolves (internal label →
//! patch, extern → `Reloc`).
//!
//! Form choices (e.g. ALU `r/m,r` vs `r,r/m`, disp8 vs disp32) are chosen to
//! match LLVM-MC so output is byte-identical; the golden differential gates that.

use anyhow::{bail, Result};

use super::parse::{Mem, MemSize, Operand, Reg, RegClass};

/// A symbolic reference left in the encoded bytes for the driver to resolve.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Fixup {
    /// Offset of the field within this instruction's bytes.
    pub at: usize,
    pub kind: FixupKind,
    pub target: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FixupKind {
    /// 4-byte branch displacement (call/jmp/jcc rel32).
    Rel32,
    /// 4-byte RIP-relative displacement (`lea`/SSE `[rip+sym]`).
    RipRel32,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Encoded {
    pub bytes: Vec<u8>,
    pub fixups: Vec<Fixup>,
}

impl Encoded {
    fn b(&mut self, byte: u8) {
        self.bytes.push(byte);
    }
    fn ext(&mut self, bytes: &[u8]) {
        self.bytes.extend_from_slice(bytes);
    }
    fn len(&self) -> usize {
        self.bytes.len()
    }
}

fn rex_byte(w: bool, r: bool, x: bool, b: bool) -> Option<u8> {
    if w || r || x || b {
        Some(0x40 | ((w as u8) << 3) | ((r as u8) << 2) | ((x as u8) << 1) | (b as u8))
    } else {
        None
    }
}

fn is64(r: Reg) -> bool {
    r.class == RegClass::R64
}

/// Emit `mandatory` prefixes, REX (if needed), `opcode`, then a ModRM for a
/// register r/m (`mod = 11`). `reg_field` and `rm` are full 0..15 numbers.
fn emit_reg_rm(e: &mut Encoded, rex_w: bool, mandatory: &[u8], opcode: &[u8], reg_field: u8, rm: u8) {
    e.ext(mandatory);
    if let Some(r) = rex_byte(rex_w, reg_field >= 8, false, rm >= 8) {
        e.b(r);
    }
    e.ext(opcode);
    e.b(0xC0 | ((reg_field & 7) << 3) | (rm & 7));
}

/// Emit `mandatory`, REX, `opcode`, then ModRM+SIB+disp for a memory r/m.
/// Records a RIP-rel fixup when `mem.rip_sym` is set. Matches MC's disp sizing.
fn emit_mem_rm(
    e: &mut Encoded,
    rex_w: bool,
    mandatory: &[u8],
    opcode: &[u8],
    reg_field: u8,
    mem: &Mem,
) -> Result<()> {
    e.ext(mandatory);

    // RIP-relative: mod=00, rm=101, disp32 (fixup).
    if let Some(sym) = &mem.rip_sym {
        if let Some(r) = rex_byte(rex_w, reg_field >= 8, false, false) {
            e.b(r);
        }
        e.ext(opcode);
        e.b(0x00 | ((reg_field & 7) << 3) | 0b101);
        let at = e.len();
        e.ext(&[0, 0, 0, 0]);
        e.fixups.push(Fixup { at, kind: FixupKind::RipRel32, target: sym.clone() });
        return Ok(());
    }

    let base = mem.base;
    let index = mem.index;
    let rex_x = index.map(|r| r.num >= 8).unwrap_or(false);
    let rex_b = base.map(|r| r.num >= 8).unwrap_or(false);
    if let Some(r) = rex_byte(rex_w, reg_field >= 8, rex_x, rex_b) {
        e.b(r);
    }
    e.ext(opcode);

    let reg3 = (reg_field & 7) << 3;

    // Decide mod + whether a SIB is needed.
    let needs_sib = index.is_some() || matches!(base.map(|b| b.num & 7), Some(0b100)); // rsp/r12 base

    let base_low3 = base.map(|b| b.num & 7);
    // [rbp]/[r13] (low3==5) with no disp can't use mod=00 (that's rip/disp32);
    // force disp8=0.
    let force_disp8 = matches!(base_low3, Some(0b101)) && mem.disp == 0 && !needs_sib;

    let (md, disp_bytes): (u8, &'static [u8]) = if base.is_none() {
        // [disp32] / [index*scale + disp32] — mod=00 with SIB.base=101 form.
        (0b00, &[])
    } else if force_disp8 {
        (0b01, &[])
    } else if mem.disp == 0 && !force_disp8 {
        (0b00, &[])
    } else if (-128..=127).contains(&mem.disp) {
        (0b01, &[])
    } else {
        (0b10, &[])
    };
    let _ = disp_bytes;

    if needs_sib {
        let rm = 0b100u8; // SIB follows
        e.b((md << 6) | reg3 | rm);
        let scale_bits = match mem.scale {
            1 => 0b00,
            2 => 0b01,
            4 => 0b10,
            8 => 0b11,
            other => bail!("bad scale {other}"),
        };
        let index_bits = match index {
            Some(r) => r.num & 7,
            None => 0b100, // no index
        };
        let base_bits = match base_low3 {
            Some(b) => b,
            None => 0b101, // no base (disp32)
        };
        e.b((scale_bits << 6) | (index_bits << 3) | base_bits);
    } else {
        let rm = base_low3.unwrap_or(0b101);
        e.b((md << 6) | reg3 | rm);
    }

    // Displacement.
    match md {
        0b01 => e.b(mem.disp as i8 as u8),
        0b10 => e.ext(&(mem.disp as i32).to_le_bytes()),
        0b00 if base.is_none() => e.ext(&(mem.disp as i32).to_le_bytes()),
        _ => {}
    }
    Ok(())
}

/// Emit a register or memory r/m with the given reg field.
fn emit_rm(
    e: &mut Encoded,
    rex_w: bool,
    mandatory: &[u8],
    opcode: &[u8],
    reg_field: u8,
    rm: &Operand,
) -> Result<()> {
    match rm {
        Operand::Reg(r) => {
            emit_reg_rm(e, rex_w, mandatory, opcode, reg_field, r.num);
            Ok(())
        }
        Operand::Mem(m) => emit_mem_rm(e, rex_w, mandatory, opcode, reg_field, m),
        other => bail!("expected reg/mem r/m, got {other:?}"),
    }
}

// ── group-1 ALU (add/or/adc/sbb/and/sub/xor/cmp) ────────────────────────────
// reg/reg + mem,reg use the `r/m, r` form (opcode base 0x01); reg,mem uses the
// `r, r/m` form (base 0x03); r/m,imm uses the 0x81/0x83 group with /digit.

struct Alu {
    /// /digit for the 0x81/0x83 imm group.
    ext: u8,
    /// base opcode for the `r/m, r` form (0x01 family).
    rm_r: u8,
}

fn alu(mnem: &str) -> Option<Alu> {
    Some(match mnem {
        "add" => Alu { ext: 0, rm_r: 0x01 },
        "or" => Alu { ext: 1, rm_r: 0x09 },
        "adc" => Alu { ext: 2, rm_r: 0x11 },
        "sbb" => Alu { ext: 3, rm_r: 0x19 },
        "and" => Alu { ext: 4, rm_r: 0x21 },
        "sub" => Alu { ext: 5, rm_r: 0x29 },
        "xor" => Alu { ext: 6, rm_r: 0x31 },
        "cmp" => Alu { ext: 7, rm_r: 0x39 },
        _ => return None,
    })
}

/// Encode a single instruction. `ops` are the parsed operands.
pub fn encode(mnemonic: &str, ops: &[Operand]) -> Result<Encoded> {
    if let Some(r) = try_sse(mnemonic, ops) {
        return r;
    }
    if let Some(r) = try_unary(mnemonic, ops) {
        return r;
    }
    if let Some(r) = try_shift(mnemonic, ops) {
        return r;
    }
    if let Some(r) = try_cc(mnemonic, ops) {
        return r;
    }
    if let Some(r) = try_misc(mnemonic, ops) {
        return r;
    }
    let mut e = Encoded::default();
    match (mnemonic, ops) {
        ("ret", []) => e.b(0xC3),
        ("nop", []) => e.b(0x90),
        ("cqo", []) => {
            e.b(0x48);
            e.b(0x99);
        }
        ("leave", []) => e.b(0xC9),
        ("std", []) => e.b(0xFD),
        ("cld", []) => e.b(0xFC),

        // mov
        ("mov", [dst, src]) => encode_mov(&mut e, dst, src)?,
        // lea reg, mem
        ("lea", [Operand::Reg(d), Operand::Mem(m)]) => {
            emit_mem_rm(&mut e, is64(*d), &[], &[0x8D], d.num, m)?;
        }
        // push/pop reg64
        ("push", [Operand::Reg(r)]) if r.class == RegClass::R64 => push_pop(&mut e, 0x50, r.num),
        ("pop", [Operand::Reg(r)]) if r.class == RegClass::R64 => push_pop(&mut e, 0x58, r.num),

        // call/jmp/jcc target
        ("call", [Operand::Sym(s)]) => {
            e.b(0xE8);
            rel32_fixup(&mut e, s);
        }
        ("jmp", [Operand::Sym(s)]) => {
            e.b(0xE9);
            rel32_fixup(&mut e, s);
        }
        (m, [Operand::Sym(s)]) if jcc_code(m).is_some() => {
            e.b(0x0F);
            e.b(0x80 | jcc_code(m).unwrap());
            rel32_fixup(&mut e, s);
        }

        // group-1 ALU
        (m, [dst, src]) if alu(m).is_some() => encode_alu(&mut e, alu(m).unwrap(), dst, src)?,

        // test
        ("test", [rm, Operand::Reg(r)]) => {
            // test r/m, r : 85 /r (r/m,r form)
            emit_rm(&mut e, is64(*r), &[], &[0x85], r.num, rm)?;
        }

        _ => bail!("rasm: unsupported instruction `{mnemonic}` with {} operand(s)", ops.len()),
    }
    Ok(e)
}

fn push_pop(e: &mut Encoded, base: u8, num: u8) {
    if num >= 8 {
        e.b(0x41); // REX.B
    }
    e.b(base + (num & 7));
}

fn rel32_fixup(e: &mut Encoded, sym: &str) {
    let at = e.len();
    e.ext(&[0, 0, 0, 0]);
    e.fixups.push(Fixup { at, kind: FixupKind::Rel32, target: sym.to_string() });
}

/// Condition-code nibble for a bare condition suffix (shared by jcc/setcc/cmovcc).
fn cc_code(cc: &str) -> Option<u8> {
    Some(match cc {
        "o" => 0x0,
        "no" => 0x1,
        "b" | "c" | "nae" => 0x2,
        "ae" | "nb" | "nc" => 0x3,
        "e" | "z" => 0x4,
        "ne" | "nz" => 0x5,
        "be" | "na" => 0x6,
        "a" | "nbe" => 0x7,
        "s" => 0x8,
        "ns" => 0x9,
        "p" | "pe" => 0xA,
        "np" | "po" => 0xB,
        "l" | "nge" => 0xC,
        "ge" | "nl" => 0xD,
        "le" | "ng" => 0xE,
        "g" | "nle" => 0xF,
        _ => return None,
    })
}

fn jcc_code(m: &str) -> Option<u8> {
    m.strip_prefix('j').and_then(cc_code)
}

/// The condition nibble for a `jcc` mnemonic (`jz`→4, …) — used by the
/// two-pass driver to build the short/long branch forms. `None` for `jmp`.
pub(crate) fn jcc_nibble(m: &str) -> Option<u8> {
    jcc_code(m)
}

fn src_size_word(src: &Operand) -> Option<bool> {
    match src {
        Operand::Mem(m) => match m.size {
            Some(MemSize::Byte) => Some(false),
            Some(MemSize::Word) => Some(true),
            _ => None,
        },
        Operand::Reg(r) => match r.class {
            RegClass::R8 => Some(false),
            RegClass::R16 => Some(true),
            _ => None,
        },
        _ => None,
    }
}

/// `(opcode, needs REX.W)` for a `rep`-prefixed string op.
fn string_op(s: &str) -> Option<(u8, bool)> {
    Some(match s {
        "movsb" => (0xA4, false),
        "movsq" => (0xA5, true),
        "stosb" => (0xAA, false),
        "stosq" => (0xAB, true),
        "cmpsb" => (0xA6, false),
        "cmpsq" => (0xA7, true),
        "scasb" => (0xAE, false),
        "scasq" => (0xAF, true),
        "lodsb" => (0xAC, false),
        "lodsq" => (0xAD, true),
        _ => return None,
    })
}

/// setcc r/m8 (0F 90+cc /0) and cmovcc r, r/m (0F 40+cc /r).
fn try_cc(mnemonic: &str, ops: &[Operand]) -> Option<Result<Encoded>> {
    let mut e = Encoded::default();
    if let Some(cc) = mnemonic.strip_prefix("set").and_then(cc_code) {
        let [rm] = ops else {
            return Some(Err(anyhow::anyhow!("setcc needs 1 operand")));
        };
        return Some(emit_rm(&mut e, false, &[], &[0x0F, 0x90 | cc], 0, rm).map(|()| e));
    }
    if let Some(cc) = mnemonic.strip_prefix("cmov").and_then(cc_code) {
        if let [Operand::Reg(d), src] = ops {
            return Some(emit_rm(&mut e, is64(*d), &[], &[0x0F, 0x40 | cc], d.num, src).map(|()| e));
        }
    }
    None
}

/// movzx/movsx/movsxd, 2-operand imul, xchg, xadd, and rep-string ops.
fn try_misc(mnemonic: &str, ops: &[Operand]) -> Option<Result<Encoded>> {
    let mut e = Encoded::default();
    let r = (|| -> Result<bool> {
        match (mnemonic, ops) {
            ("movzx" | "movsx", [Operand::Reg(d), src]) => {
                let word = src_size_word(src)
                    .ok_or_else(|| anyhow::anyhow!("movzx/movsx needs a sized source: {src:?}"))?;
                let op = match (mnemonic, word) {
                    ("movzx", false) => 0xB6,
                    ("movzx", true) => 0xB7,
                    ("movsx", false) => 0xBE,
                    ("movsx", true) => 0xBF,
                    _ => unreachable!(),
                };
                emit_rm(&mut e, is64(*d), &[], &[0x0F, op], d.num, src)?;
            }
            ("movsxd", [Operand::Reg(d), src]) => {
                // movsxd r64, r/m32 : REX.W 63 /r
                emit_rm(&mut e, true, &[], &[0x63], d.num, src)?;
            }
            ("imul", [Operand::Reg(d), src]) => {
                // 2-operand imul r, r/m : 0F AF /r
                emit_rm(&mut e, is64(*d), &[], &[0x0F, 0xAF], d.num, src)?;
            }
            ("xchg", [dst, Operand::Reg(s)]) => {
                // xchg r/m, r : 87 /r
                emit_rm(&mut e, is64(*s), &[], &[0x87], s.num, dst)?;
            }
            ("xadd", [dst, Operand::Reg(s)]) => {
                // xadd r/m, r : 0F C1 /r
                emit_rm(&mut e, is64(*s), &[], &[0x0F, 0xC1], s.num, dst)?;
            }
            ("rep" | "repe" | "repz" | "repne" | "repnz", [Operand::Sym(strop)]) => {
                let pfx = if mnemonic.starts_with("repn") { 0xF2u8 } else { 0xF3 };
                let (opc, w) =
                    string_op(strop).ok_or_else(|| anyhow::anyhow!("unknown string op `{strop}`"))?;
                e.b(pfx);
                if w {
                    e.b(0x48); // REX.W (after the F3/F2 prefix, before the opcode)
                }
                e.b(opc);
            }
            _ => return Ok(false),
        }
        Ok(true)
    })();
    match r {
        Ok(true) => Some(Ok(e)),
        Ok(false) => None,
        Err(err) => Some(Err(err)),
    }
}

fn encode_mov(e: &mut Encoded, dst: &Operand, src: &Operand) -> Result<()> {
    match (dst, src) {
        // mov r/m, r  (89 /r)
        (rm, Operand::Reg(r)) => emit_rm(e, is64(*r), &[], &[0x89], r.num, rm),
        // mov r, r/m  (8B /r) — when dst is reg and src is mem
        (Operand::Reg(d), Operand::Mem(m)) => emit_mem_rm(e, is64(*d), &[], &[0x8B], d.num, m),
        // mov r/m64, imm32 (sign-extended): C7 /0 id ; or mov r64, imm64: B8+r
        (Operand::Reg(d), Operand::Imm(v)) if d.class == RegClass::R64 => {
            if i32::try_from(*v).is_ok() {
                // C7 /0 id (canonical for values fitting i32 — matches MC)
                emit_reg_rm(e, true, &[], &[0xC7], 0, d.num);
                e.ext(&(*v as i32).to_le_bytes());
            } else {
                // movabs r64, imm64: REX.W B8+rd io
                if let Some(r) = rex_byte(true, false, false, d.num >= 8) {
                    e.b(r);
                }
                e.b(0xB8 + (d.num & 7));
                e.ext(&(*v as u64).to_le_bytes());
            }
            Ok(())
        }
        (Operand::Mem(m), Operand::Imm(v)) => {
            // mov r/m64, imm32: C7 /0 id (size from mem, default qword here)
            let w = mem_is_qword(m);
            emit_mem_rm(e, w, &[], &[0xC7], 0, m)?;
            e.ext(&(*v as i32).to_le_bytes());
            Ok(())
        }
        _ => bail!("rasm: unsupported mov form {dst:?} <- {src:?}"),
    }
}

fn mem_is_qword(m: &Mem) -> bool {
    matches!(m.size, Some(MemSize::Qword) | None)
}

fn encode_alu(e: &mut Encoded, a: Alu, dst: &Operand, src: &Operand) -> Result<()> {
    match (dst, src) {
        // r/m, r : (rm_r) /r
        (rm, Operand::Reg(r)) => emit_rm(e, is64(*r), &[], &[a.rm_r], r.num, rm),
        // r, r/m (mem) : (rm_r + 2) /r
        (Operand::Reg(d), Operand::Mem(m)) => emit_mem_rm(e, is64(*d), &[], &[a.rm_r + 2], d.num, m),
        // r/m, imm : 83 /ext ib (imm8) or 81 /ext id (imm32)
        (rm, Operand::Imm(v)) => {
            let w = match rm {
                Operand::Reg(r) => is64(*r),
                Operand::Mem(m) => mem_is_qword(m),
                _ => true,
            };
            if (-128..=127).contains(v) {
                emit_rm(e, w, &[], &[0x83], a.ext, rm)?;
                e.b(*v as i8 as u8);
            } else {
                emit_rm(e, w, &[], &[0x81], a.ext, rm)?;
                e.ext(&(*v as i32).to_le_bytes());
            }
            Ok(())
        }
        _ => bail!("rasm: unsupported alu form {dst:?}, {src:?}"),
    }
}

fn operand_w(op: &Operand) -> bool {
    match op {
        Operand::Reg(r) => is64(*r),
        Operand::Mem(m) => mem_is_qword(m),
        _ => true,
    }
}

/// SSE2 scalar-double + the xmm move/convert family (the FTOS/REX.R island).
fn try_sse(mnemonic: &str, ops: &[Operand]) -> Option<Result<Encoded>> {
    let mut e = Encoded::default();
    let r = (|| -> Result<bool> {
        match (mnemonic, ops) {
            // movsd: load (xmm <- r/m) F2 0F 10 ; store (m <- xmm) F2 0F 11
            ("movsd", [Operand::Reg(d), src]) if d.class == RegClass::Xmm => {
                emit_rm(&mut e, false, &[0xF2], &[0x0F, 0x10], d.num, src)?;
            }
            ("movsd", [Operand::Mem(m), Operand::Reg(s)]) if s.class == RegClass::Xmm => {
                emit_mem_rm(&mut e, false, &[0xF2], &[0x0F, 0x11], s.num, m)?;
            }
            // movups: load 0F 10 ; store 0F 11 (no mandatory prefix)
            ("movups", [Operand::Reg(d), src]) if d.class == RegClass::Xmm => {
                emit_rm(&mut e, false, &[], &[0x0F, 0x10], d.num, src)?;
            }
            ("movups", [Operand::Mem(m), Operand::Reg(s)]) if s.class == RegClass::Xmm => {
                emit_mem_rm(&mut e, false, &[], &[0x0F, 0x11], s.num, m)?;
            }
            // arithmetic: F2 0F <op> /r, dst xmm = reg field
            ("addsd" | "subsd" | "mulsd" | "divsd", [Operand::Reg(d), src])
                if d.class == RegClass::Xmm =>
            {
                let op = match mnemonic {
                    "addsd" => 0x58,
                    "subsd" => 0x5C,
                    "mulsd" => 0x59,
                    "divsd" => 0x5E,
                    _ => unreachable!(),
                };
                emit_rm(&mut e, false, &[0xF2], &[0x0F, op], d.num, src)?;
            }
            ("ucomisd", [Operand::Reg(d), src]) if d.class == RegClass::Xmm => {
                emit_rm(&mut e, false, &[0x66], &[0x0F, 0x2E], d.num, src)?;
            }
            ("xorpd", [Operand::Reg(d), src]) if d.class == RegClass::Xmm => {
                emit_rm(&mut e, false, &[0x66], &[0x0F, 0x57], d.num, src)?;
            }
            // movq xmm, r64 : 66 REX.W 0F 6E /r ; movq r64, xmm : 66 REX.W 0F 7E /r
            ("movq", [Operand::Reg(d), Operand::Reg(s)])
                if d.class == RegClass::Xmm && s.class == RegClass::R64 =>
            {
                emit_reg_rm(&mut e, true, &[0x66], &[0x0F, 0x6E], d.num, s.num);
            }
            ("movq", [Operand::Reg(d), Operand::Reg(s)])
                if d.class == RegClass::R64 && s.class == RegClass::Xmm =>
            {
                emit_reg_rm(&mut e, true, &[0x66], &[0x0F, 0x7E], s.num, d.num);
            }
            // cvtsi2sd xmm, r/m64 : F2 REX.W 0F 2A /r
            ("cvtsi2sd", [Operand::Reg(d), src]) if d.class == RegClass::Xmm => {
                emit_rm(&mut e, true, &[0xF2], &[0x0F, 0x2A], d.num, src)?;
            }
            // cvttsd2si r64, xmm/m : F2 REX.W 0F 2C /r (reg field = GPR)
            ("cvttsd2si", [Operand::Reg(d), src]) if d.class == RegClass::R64 => {
                emit_rm(&mut e, true, &[0xF2], &[0x0F, 0x2C], d.num, src)?;
            }
            _ => return Ok(false),
        }
        Ok(true)
    })();
    match r {
        Ok(true) => Some(Ok(e)),
        Ok(false) => None,
        Err(err) => Some(Err(err)),
    }
}

/// One-operand F7/FF group: neg/not/mul/imul/div/idiv (F7 /ext) and inc/dec
/// (FF /ext).
fn try_unary(mnemonic: &str, ops: &[Operand]) -> Option<Result<Encoded>> {
    let (opcode, ext) = match mnemonic {
        "not" => (0xF7u8, 2u8),
        "neg" => (0xF7, 3),
        "mul" => (0xF7, 4),
        "imul" if ops.len() == 1 => (0xF7, 5),
        "div" => (0xF7, 6),
        "idiv" => (0xF7, 7),
        "inc" => (0xFF, 0),
        "dec" => (0xFF, 1),
        _ => return None,
    };
    let [rm] = ops else { return None };
    let mut e = Encoded::default();
    match emit_rm(&mut e, operand_w(rm), &[], &[opcode], ext, rm) {
        Ok(()) => Some(Ok(e)),
        Err(err) => Some(Err(err)),
    }
}

/// Shift group: shl/sal/shr/sar/rol/ror by 1 (D1), imm8 (C1), or cl (D3).
fn try_shift(mnemonic: &str, ops: &[Operand]) -> Option<Result<Encoded>> {
    let ext = match mnemonic {
        "rol" => 0u8,
        "ror" => 1,
        "shl" | "sal" => 4,
        "shr" => 5,
        "sar" => 7,
        _ => return None,
    };
    let [rm, count] = ops else { return None };
    let mut e = Encoded::default();
    let w = operand_w(rm);
    let r = match count {
        Operand::Imm(1) => emit_rm(&mut e, w, &[], &[0xD1], ext, rm),
        Operand::Imm(n) => emit_rm(&mut e, w, &[], &[0xC1], ext, rm).map(|()| e.b(*n as u8)),
        Operand::Reg(r) if r.class == RegClass::R8 && r.num == 1 => {
            emit_rm(&mut e, w, &[], &[0xD3], ext, rm) // shift by cl
        }
        other => Err(anyhow::anyhow!("bad shift count {other:?}")),
    };
    match r {
        Ok(()) => Some(Ok(e)),
        Err(err) => Some(Err(err)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rasm::parse::parse_line;
    use crate::rasm::Line;
    use iced_x86::{Decoder, DecoderOptions, Formatter, Instruction, IntelFormatter};

    fn enc(line: &str) -> Encoded {
        match parse_line(line).unwrap() {
            Line::Insn { mnemonic, ops } => encode(&mnemonic, &ops).unwrap(),
            other => panic!("not an insn: {other:?}"),
        }
    }

    fn bytes(line: &str) -> Vec<u8> {
        enc(line).bytes
    }

    /// Decode our bytes with iced and format back to Intel syntax — the
    /// round-trip oracle (catches "valid but wrong instruction").
    fn roundtrip(line: &str) -> String {
        let b = bytes(line);
        let mut dec = Decoder::with_ip(64, &b, 0x1000, DecoderOptions::NONE);
        let insn: Instruction = dec.decode();
        assert!(!insn.is_invalid(), "iced could not decode `{line}` -> {b:02x?}");
        assert_eq!(dec.position(), b.len(), "trailing bytes after `{line}`: {b:02x?}");
        let mut f = IntelFormatter::new();
        let mut s = String::new();
        f.format(&insn, &mut s);
        s
    }

    #[test]
    fn exact_bytes_match_golden_leaves() {
        // dup_ body: mov [rbp-8], rax ; sub rbp, 8 ; ret  (golden 48 89 45 f8 / 48 83 ed 08 / c3)
        assert_eq!(bytes("mov [rbp - 8], rax"), vec![0x48, 0x89, 0x45, 0xF8]);
        assert_eq!(bytes("sub rbp, 8"), vec![0x48, 0x83, 0xED, 0x08]);
        assert_eq!(bytes("ret"), vec![0xC3]);
        // plus body: add rax, [rbp] ; add rbp, 8 ; ret  (golden 48 03 45 00 / 48 83 c5 08 / c3)
        assert_eq!(bytes("add rax, [rbp]"), vec![0x48, 0x03, 0x45, 0x00]);
        assert_eq!(bytes("add rbp, 8"), vec![0x48, 0x83, 0xC5, 0x08]);
    }

    #[test]
    fn mov_forms() {
        assert_eq!(roundtrip("mov rax, rcx"), "mov rax,rcx");
        assert_eq!(roundtrip("mov rcx, [rbx + 4632]"), "mov rcx,[rbx+1218h]");
        assert_eq!(roundtrip("mov [rbx + 4632], rcx"), "mov [rbx+1218h],rcx");
        assert_eq!(roundtrip("mov rax, 0xDEADBEEF"), "mov rax,0DEADBEEFh");
        assert_eq!(roundtrip("mov r8, [rsp]"), "mov r8,[rsp]");
        assert_eq!(roundtrip("mov rax, 0x4010000000000000"), "mov rax,4010000000000000h");
    }

    #[test]
    fn alu_forms() {
        assert_eq!(roundtrip("add rax, rcx"), "add rax,rcx");
        assert_eq!(roundtrip("sub rbp, 8"), "sub rbp,8");
        assert_eq!(roundtrip("cmp rax, [rbp]"), "cmp rax,[rbp]");
        assert_eq!(roundtrip("and rax, 0x1FF"), "and rax,1FFh");
        assert_eq!(roundtrip("xor r10, r11"), "xor r10,r11");
        assert_eq!(roundtrip("add qword ptr [rsp], 8"), "add qword ptr [rsp],8");
    }

    #[test]
    fn push_pop_lea_test() {
        assert_eq!(bytes("push rbx"), vec![0x53]);
        assert_eq!(bytes("push r15"), vec![0x41, 0x57]);
        assert_eq!(bytes("pop rbp"), vec![0x5D]);
        assert_eq!(roundtrip("lea r8, [rax + rax*1]"), "lea r8,[rax+rax]");
        assert_eq!(roundtrip("test rax, rax"), "test rax,rax");
    }

    #[test]
    fn sse_ftos_xmm15_island() {
        // f_plus golden: addsd xmm15,[rcx] = f2 44 0f 58 39 (REX.R for xmm15)
        assert_eq!(bytes("addsd xmm15, qword ptr [rcx]"), vec![0xF2, 0x44, 0x0F, 0x58, 0x39]);
        // f_fetch golden fragment: movsd qword ptr [rdx-8], xmm15 (store, REX.R)
        assert_eq!(bytes("movsd qword ptr [rcx - 8], xmm15"), vec![0xF2, 0x44, 0x0F, 0x11, 0x79, 0xF8]);
        // movsd xmm15, [rcx] (load, REX.R)
        assert_eq!(bytes("movsd xmm15, qword ptr [rcx]"), vec![0xF2, 0x44, 0x0F, 0x10, 0x39]);
        // round-trips for the rest of the island
        assert_eq!(roundtrip("subsd xmm15, xmm14"), "subsd xmm15,xmm14");
        assert_eq!(roundtrip("mulsd xmm0, xmm1"), "mulsd xmm0,xmm1");
        assert_eq!(roundtrip("divsd xmm6, qword ptr [rbx]"), "divsd xmm6,[rbx]");
        assert_eq!(roundtrip("ucomisd xmm15, xmm0"), "ucomisd xmm15,xmm0");
        assert_eq!(roundtrip("xorpd xmm0, xmm0"), "xorpd xmm0,xmm0");
        assert_eq!(roundtrip("movups xmm8, [rsp]"), "movups xmm8,[rsp]");
        assert_eq!(roundtrip("movups [rsp + 32], xmm8"), "movups [rsp+20h],xmm8");
        assert_eq!(roundtrip("movq xmm15, rdx"), "movq xmm15,rdx");
        assert_eq!(roundtrip("movq r10, xmm15"), "movq r10,xmm15");
        assert_eq!(roundtrip("cvtsi2sd xmm0, rcx"), "cvtsi2sd xmm0,rcx");
        assert_eq!(roundtrip("cvttsd2si rcx, xmm15"), "cvttsd2si rcx,xmm15");
    }

    #[test]
    fn unary_and_shift() {
        assert_eq!(roundtrip("neg rcx"), "neg rcx");
        assert_eq!(roundtrip("not rax"), "not rax");
        assert_eq!(roundtrip("idiv rcx"), "idiv rcx");
        assert_eq!(roundtrip("inc qword ptr [rbx]"), "inc qword ptr [rbx]");
        assert_eq!(roundtrip("dec rax"), "dec rax");
        // shifts: by-1 → D1 (not C1 imm=1), to match MC
        assert_eq!(bytes("shl rax, 1"), vec![0x48, 0xD1, 0xE0]);
        assert_eq!(roundtrip("shl rax, 3"), "shl rax,3");
        assert_eq!(roundtrip("sar rdx, 63"), "sar rdx,3Fh");
        assert_eq!(roundtrip("shr r9, cl"), "shr r9,cl");
    }

    #[test]
    fn extend_setcc_cmov_imul_xchg_string() {
        // movzx/movsx with sized source (kernel forms)
        assert_eq!(roundtrip("movzx rax, byte ptr [rax]"), "movzx rax,byte ptr [rax]");
        assert_eq!(roundtrip("movsx rax, word ptr [rax]"), "movsx rax,word ptr [rax]");
        assert_eq!(roundtrip("movsxd rdx, edx"), "movsxd rdx,edx");
        // setcc / cmovcc
        assert_eq!(bytes("sete cl"), vec![0x0F, 0x94, 0xC1]);
        assert_eq!(bytes("setb cl"), vec![0x0F, 0x92, 0xC1]);
        assert_eq!(roundtrip("cmovl rax, rcx"), "cmovl rax,rcx");
        assert_eq!(roundtrip("cmovg rax, rcx"), "cmovg rax,rcx");
        // imul: 1-operand (F7 /5) vs 2-operand (0F AF)
        assert_eq!(roundtrip("imul rax, [rbp]"), "imul rax,[rbp]");
        assert_eq!(roundtrip("imul qword ptr [rbp]"), "imul qword ptr [rbp]");
        // xchg / xadd
        assert_eq!(roundtrip("xchg rbp, rsp"), "xchg rbp,rsp");
        assert_eq!(roundtrip("xadd [rcx], rax"), "xadd [rcx],rax");
        // rep string ops
        assert_eq!(bytes("rep movsq"), vec![0xF3, 0x48, 0xA5]);
        assert_eq!(bytes("rep stosb"), vec![0xF3, 0xAA]);
    }

    #[test]
    fn rbp_r13_rsp_r12_modrm_traps() {
        // [rbp] forces disp8=0 (mod=01).
        assert_eq!(bytes("mov rax, [rbp]"), vec![0x48, 0x8B, 0x45, 0x00]);
        // [rsp] forces a SIB byte.
        assert_eq!(bytes("mov rax, [rsp]"), vec![0x48, 0x8B, 0x04, 0x24]);
        // [r13] forces disp8=0 + REX.B.
        assert_eq!(bytes("mov rax, [r13]"), vec![0x49, 0x8B, 0x45, 0x00]);
        // [r12] forces SIB + REX.B.
        assert_eq!(bytes("mov rax, [r12]"), vec![0x49, 0x8B, 0x04, 0x24]);
    }
}
