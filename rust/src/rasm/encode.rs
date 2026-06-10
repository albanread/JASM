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

fn jcc_code(m: &str) -> Option<u8> {
    Some(match m {
        "jo" => 0x0,
        "jno" => 0x1,
        "jb" | "jc" | "jnae" => 0x2,
        "jae" | "jnb" | "jnc" => 0x3,
        "je" | "jz" => 0x4,
        "jne" | "jnz" => 0x5,
        "jbe" | "jna" => 0x6,
        "ja" | "jnbe" => 0x7,
        "js" => 0x8,
        "jns" => 0x9,
        "jp" | "jpe" => 0xA,
        "jnp" | "jpo" => 0xB,
        "jl" | "jnge" => 0xC,
        "jge" | "jnl" => 0xD,
        "jle" | "jng" => 0xE,
        "jg" | "jnle" => 0xF,
        _ => return None,
    })
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
