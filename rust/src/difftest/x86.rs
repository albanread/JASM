//! `X86Model` — templated form generator for the integer + SSE/SSE2 tier.
//!
//! Produces single-instruction forms as the Cartesian product of small,
//! representative operand banks against a mnemonic catalog. Deterministic (no
//! RNG) so the generated set — and any corpus recorded from it — is reviewable.
//! VEX/AVX is a later tier (design §10, Phase 5) and lives in its own catalog.
//!
//! Operand banks are intentionally small: each register slot offers the
//! accumulator (to hit the `al/ax/eax/rax` short-opcode forms) plus one high
//! register (to exercise REX.B/REX.R). Each r/m slot adds two memory templates —
//! a plain `[reg]` and a `[base + index*scale]` SIB — to exercise the addressing
//! encoders. Breadth over depth: the goal is to surface *which instructions and
//! forms* rasm lacks, not to fuzz every operand combination.

use super::{Form, IsaModel};

/// Operand width.
#[derive(Clone, Copy)]
enum W {
    B,
    Wd,
    D,
    Q,
}

impl W {
    fn ptr(self) -> &'static str {
        match self {
            W::B => "byte",
            W::Wd => "word",
            W::D => "dword",
            W::Q => "qword",
        }
    }
    fn acc(self) -> &'static str {
        match self {
            W::B => "al",
            W::Wd => "ax",
            W::D => "eax",
            W::Q => "rax",
        }
    }
    fn hi(self) -> &'static str {
        match self {
            W::B => "r9b",
            W::Wd => "r10w",
            W::D => "r11d",
            W::Q => "r12",
        }
    }
    fn imm(self) -> &'static str {
        match self {
            W::B => "0x7f",
            W::Wd => "0x1234",
            W::D | W::Q => "0x7fffffff", // ALU imm is imm32, sign-extended at Q
        }
    }
}

/// A generator operand slot.
#[derive(Clone, Copy)]
enum Op {
    /// A register of the given width.
    Reg(W),
    /// Register-or-memory of the given width (memory templates get a size ptr).
    Rm(W),
    /// Immediate sized to the width.
    Imm(W),
    /// The `cl` register (shift counts).
    Cl,
    /// Literal `1` (shift-by-one short form).
    One,
    /// An xmm register.
    Xmm,
    /// xmm-or-memory (memory is bare; the mnemonic fixes the size).
    XmmRm,
    /// A ymm register (AVX 256-bit).
    Ymm,
    /// ymm-or-memory.
    YmmRm,
    /// A zmm register (AVX-512 512-bit) — low + extended (16..31) to exercise EVEX.
    Zmm,
    /// zmm-or-memory.
    ZmmRm,
    /// A bare memory operand (no size ptr) — for `lea`/`movbe`.
    Mem,
    /// A symbol reference (branch/call target).
    Sym,
}

const MEMS: &[&str] = &["[rax]", "[r13 + r14*2]"];

fn op_choices(op: Op) -> Vec<String> {
    match op {
        Op::Reg(w) => vec![w.acc().into(), w.hi().into()],
        Op::Rm(w) => {
            let mut v = vec![w.acc().to_string(), w.hi().to_string()];
            for m in MEMS {
                v.push(format!("{} ptr {m}", w.ptr()));
            }
            v
        }
        Op::Imm(w) => vec![w.imm().into()],
        Op::Cl => vec!["cl".into()],
        Op::One => vec!["1".into()],
        Op::Xmm => vec!["xmm1".into(), "xmm12".into()],
        Op::XmmRm => {
            let mut v = vec!["xmm1".to_string(), "xmm12".to_string()];
            for m in MEMS {
                v.push((*m).to_string());
            }
            v
        }
        Op::Ymm => vec!["ymm1".into(), "ymm12".into()],
        Op::YmmRm => {
            let mut v = vec!["ymm1".to_string(), "ymm12".to_string()];
            for m in MEMS {
                v.push((*m).to_string());
            }
            v
        }
        // zmm1 (all-low), zmm12 (REX bit3 set), zmm18 (EVEX bit4 → R'/X/V').
        Op::Zmm => vec!["zmm1".into(), "zmm12".into(), "zmm18".into()],
        Op::ZmmRm => {
            let mut v = vec!["zmm1".to_string(), "zmm12".to_string(), "zmm18".to_string()];
            for m in MEMS {
                v.push((*m).to_string());
            }
            v
        }
        Op::Mem => {
            let mut v: Vec<String> = MEMS.iter().map(|m| m.to_string()).collect();
            v.push("[rip + sym0]".into());
            v
        }
        Op::Sym => vec!["sym0".into()],
    }
}

/// Cartesian product of each slot's candidate strings.
fn product(ops: &[Op]) -> Vec<Vec<String>> {
    let mut acc: Vec<Vec<String>> = vec![vec![]];
    for &op in ops {
        let choices = op_choices(op);
        let mut next = Vec::with_capacity(acc.len() * choices.len());
        for prefix in &acc {
            for c in &choices {
                let mut row = prefix.clone();
                row.push(c.clone());
                next.push(row);
            }
        }
        acc = next;
    }
    acc
}

fn emit(out: &mut Vec<Form>, mnemonic: &'static str, family: &'static str, ops: &[Op]) {
    for combo in product(ops) {
        let joined = combo.join(", ");
        let asm = if joined.is_empty() {
            mnemonic.to_string()
        } else {
            format!("{mnemonic} {joined}")
        };
        out.push(Form { asm, family, mnemonic });
    }
}

/// The integer + SSE/SSE2 form generator.
pub struct X86Model;

impl IsaModel for X86Model {
    fn triple(&self) -> &str {
        "x86_64-pc-windows-msvc"
    }

    fn forms(&self) -> Vec<Form> {
        use Op::*;
        use W::*;
        let mut o = Vec::new();

        // ── integer ALU group-1 ───────────────────────────────────────────────
        for m in ["add", "or", "adc", "sbb", "and", "sub", "xor", "cmp"] {
            emit(&mut o, m, "int.alu", &[Rm(Q), Reg(Q)]);
            emit(&mut o, m, "int.alu", &[Reg(Q), Rm(Q)]);
            emit(&mut o, m, "int.alu", &[Rm(Q), Imm(D)]);
            emit(&mut o, m, "int.alu", &[Rm(Q), Imm(B)]); // sign-extended imm8 short form
            emit(&mut o, m, "int.alu", &[Reg(D), Rm(D)]);
            emit(&mut o, m, "int.alu", &[Rm(B), Reg(B)]);
        }

        // ── mov / lea / extends ───────────────────────────────────────────────
        emit(&mut o, "mov", "int.mov", &[Rm(Q), Reg(Q)]);
        emit(&mut o, "mov", "int.mov", &[Reg(Q), Rm(Q)]);
        emit(&mut o, "mov", "int.mov", &[Rm(Q), Imm(D)]);
        emit(&mut o, "mov", "int.mov", &[Reg(D), Rm(D)]);
        emit(&mut o, "mov", "int.mov", &[Rm(B), Reg(B)]);
        emit(&mut o, "movabs", "int.mov", &[Reg(Q), Imm(D)]); // forces imm64 B8 form
        emit(&mut o, "lea", "int.lea", &[Reg(Q), Mem]);
        emit(&mut o, "movzx", "int.ext", &[Reg(D), Rm(B)]);
        emit(&mut o, "movzx", "int.ext", &[Reg(D), Rm(Wd)]);
        emit(&mut o, "movzx", "int.ext", &[Reg(Q), Rm(B)]);
        emit(&mut o, "movsx", "int.ext", &[Reg(D), Rm(B)]);
        emit(&mut o, "movsx", "int.ext", &[Reg(Q), Rm(Wd)]);
        emit(&mut o, "movsxd", "int.ext", &[Reg(Q), Rm(D)]);

        // ── shifts / rotates ──────────────────────────────────────────────────
        for m in ["shl", "shr", "sar", "sal", "rol", "ror", "rcl", "rcr"] {
            emit(&mut o, m, "int.shift", &[Rm(Q), One]);
            emit(&mut o, m, "int.shift", &[Rm(Q), Cl]);
            emit(&mut o, m, "int.shift", &[Rm(Q), Imm(B)]);
        }

        // ── unary / mul / div ─────────────────────────────────────────────────
        for m in ["inc", "dec", "neg", "not", "mul", "imul", "div", "idiv"] {
            emit(&mut o, m, "int.unary", &[Rm(Q)]);
        }
        emit(&mut o, "imul", "int.mul", &[Reg(Q), Rm(Q)]);
        emit(&mut o, "imul", "int.mul", &[Reg(Q), Rm(Q), Imm(B)]);
        emit(&mut o, "imul", "int.mul", &[Reg(Q), Rm(Q), Imm(D)]);
        emit(&mut o, "test", "int.test", &[Rm(Q), Reg(Q)]);
        emit(&mut o, "test", "int.test", &[Rm(Q), Imm(D)]);

        // ── stack / xchg / atomics ────────────────────────────────────────────
        emit(&mut o, "push", "int.stack", &[Reg(Q)]);
        emit(&mut o, "pop", "int.stack", &[Reg(Q)]);
        emit(&mut o, "push", "int.stack", &[Rm(Q)]);
        emit(&mut o, "push", "int.stack", &[Imm(D)]);
        emit(&mut o, "xchg", "int.atomic", &[Rm(Q), Reg(Q)]);
        emit(&mut o, "xadd", "int.atomic", &[Rm(Q), Reg(Q)]);
        emit(&mut o, "cmpxchg", "int.atomic", &[Rm(Q), Reg(Q)]);

        // ── bit instructions ──────────────────────────────────────────────────
        for m in ["bt", "bts", "btr", "btc"] {
            emit(&mut o, m, "int.bit", &[Rm(Q), Reg(Q)]);
            emit(&mut o, m, "int.bit", &[Rm(Q), Imm(B)]);
        }
        for m in ["bsf", "bsr", "popcnt", "lzcnt", "tzcnt"] {
            emit(&mut o, m, "int.bit", &[Reg(Q), Rm(Q)]);
        }
        emit(&mut o, "bswap", "int.bit", &[Reg(Q)]);
        emit(&mut o, "bswap", "int.bit", &[Reg(D)]);

        // ── movbe (load/store byte-swapped) ───────────────────────────────────
        emit(&mut o, "movbe", "int.movbe", &[Reg(Q), Mem]);
        emit(&mut o, "movbe", "int.movbe", &[Mem, Reg(Q)]);

        // ── setcc / cmovcc / branches ─────────────────────────────────────────
        for m in ["setz", "setne", "setl", "setg", "seta", "setb", "sets", "seto"] {
            emit(&mut o, m, "int.setcc", &[Rm(B)]);
        }
        for m in ["cmove", "cmovne", "cmovl", "cmovg", "cmova", "cmovb"] {
            emit(&mut o, m, "int.cmovcc", &[Reg(Q), Rm(Q)]);
        }
        for m in ["jmp", "call"] {
            emit(&mut o, m, "int.branch", &[Sym]);
            emit(&mut o, m, "int.branch", &[Rm(Q)]); // indirect
        }
        for m in ["je", "jne", "jl", "jg", "ja", "jb", "js", "jo"] {
            emit(&mut o, m, "int.branch", &[Sym]);
        }

        // ── string ops & zero-operand misc ────────────────────────────────────
        for m in ["movsb", "movsq", "stosb", "stosq", "lodsb", "lodsq", "scasb", "scasq", "cmpsb", "cmpsq"] {
            emit(&mut o, m, "int.string", &[]);
        }
        for m in [
            "ret", "leave", "nop", "cqo", "cdqe", "cwde", "cdq", "syscall", "cpuid", "rdtsc",
            "pause", "int3", "endbr64", "sahf", "lahf", "clc", "stc", "cmc", "cld", "std",
        ] {
            emit(&mut o, m, "int.misc0", &[]);
        }

        // ── SSE/SSE2 scalar double ────────────────────────────────────────────
        for m in ["addsd", "subsd", "mulsd", "divsd", "minsd", "maxsd", "sqrtsd", "comisd", "ucomisd"] {
            emit(&mut o, m, "sse.scalar.f64", &[Xmm, XmmRm]);
        }
        // ── SSE scalar single ─────────────────────────────────────────────────
        for m in ["addss", "subss", "mulss", "divss", "minss", "maxss", "sqrtss", "comiss", "ucomiss"] {
            emit(&mut o, m, "sse.scalar.f32", &[Xmm, XmmRm]);
        }
        // ── packed single / double ────────────────────────────────────────────
        for m in ["addps", "subps", "mulps", "divps", "minps", "maxps", "andps", "orps", "xorps", "andnps", "sqrtps", "unpcklps", "unpckhps"] {
            emit(&mut o, m, "sse.packed.f32", &[Xmm, XmmRm]);
        }
        for m in ["addpd", "subpd", "mulpd", "divpd", "andpd", "orpd", "xorpd", "andnpd"] {
            emit(&mut o, m, "sse.packed.f64", &[Xmm, XmmRm]);
        }
        // ── packed integer ────────────────────────────────────────────────────
        for m in ["paddb", "paddw", "paddd", "paddq", "psubb", "psubw", "psubd", "psubq", "pmullw", "pmulld", "pand", "pandn", "por", "pxor", "pcmpeqb", "pcmpeqd"] {
            emit(&mut o, m, "sse.packed.int", &[Xmm, XmmRm]);
        }
        // ── SSE moves ─────────────────────────────────────────────────────────
        for m in ["movss", "movsd", "movaps", "movups", "movapd", "movupd", "movdqa", "movdqu"] {
            emit(&mut o, m, "sse.mov", &[Xmm, XmmRm]);
            emit(&mut o, m, "sse.mov", &[XmmRm, Xmm]); // store direction
        }
        emit(&mut o, "movd", "sse.mov", &[Xmm, Rm(D)]);
        emit(&mut o, "movd", "sse.mov", &[Rm(D), Xmm]);
        emit(&mut o, "movq", "sse.mov", &[Xmm, Rm(Q)]);
        emit(&mut o, "movq", "sse.mov", &[Rm(Q), Xmm]);
        emit(&mut o, "movq", "sse.mov", &[Xmm, XmmRm]);

        // ── conversions ───────────────────────────────────────────────────────
        emit(&mut o, "cvtsi2sd", "sse.cvt", &[Xmm, Rm(Q)]);
        emit(&mut o, "cvtsi2sd", "sse.cvt", &[Xmm, Rm(D)]);
        emit(&mut o, "cvtsi2ss", "sse.cvt", &[Xmm, Rm(Q)]);
        emit(&mut o, "cvtsi2ss", "sse.cvt", &[Xmm, Rm(D)]);
        emit(&mut o, "cvttsd2si", "sse.cvt", &[Reg(Q), XmmRm]);
        emit(&mut o, "cvttsd2si", "sse.cvt", &[Reg(D), XmmRm]);
        emit(&mut o, "cvtsd2si", "sse.cvt", &[Reg(Q), XmmRm]);
        emit(&mut o, "cvttss2si", "sse.cvt", &[Reg(Q), XmmRm]);
        emit(&mut o, "cvtss2si", "sse.cvt", &[Reg(Q), XmmRm]);
        for m in ["cvtsd2ss", "cvtss2sd", "cvtdq2pd", "cvtdq2ps", "cvtpd2ps", "cvtps2pd"] {
            emit(&mut o, m, "sse.cvt", &[Xmm, XmmRm]);
        }

        // ── shuffles / permutes (imm8) ────────────────────────────────────────
        emit(&mut o, "shufps", "sse.shuffle", &[Xmm, XmmRm, Imm(B)]);
        emit(&mut o, "shufpd", "sse.shuffle", &[Xmm, XmmRm, Imm(B)]);
        emit(&mut o, "pshufd", "sse.shuffle", &[Xmm, XmmRm, Imm(B)]);

        // ── AVX / AVX2 (VEX-encoded) ──────────────────────────────────────────
        // packed FP, 3-operand, both xmm (VEX.128) and ymm (VEX.256)
        for m in [
            "vaddps", "vsubps", "vmulps", "vdivps", "vminps", "vmaxps", "vandps", "vandnps",
            "vorps", "vxorps", "vunpcklps", "vunpckhps", "vaddpd", "vsubpd", "vmulpd", "vdivpd",
            "vminpd", "vmaxpd", "vandpd", "vandnpd", "vorpd", "vxorpd",
        ] {
            emit(&mut o, m, "vex.packed.fp", &[Xmm, Xmm, XmmRm]);
            emit(&mut o, m, "vex.packed.fp", &[Ymm, Ymm, YmmRm]);
        }
        // scalar FP, 3-operand (always xmm)
        for m in [
            "vaddss", "vsubss", "vmulss", "vdivss", "vminss", "vmaxss", "vsqrtss", "vaddsd",
            "vsubsd", "vmulsd", "vdivsd", "vminsd", "vmaxsd", "vsqrtsd",
        ] {
            emit(&mut o, m, "vex.scalar.fp", &[Xmm, Xmm, XmmRm]);
        }
        // packed integer, 3-operand
        for m in [
            "vpaddb", "vpaddw", "vpaddd", "vpaddq", "vpsubb", "vpsubw", "vpsubd", "vpsubq",
            "vpand", "vpandn", "vpor", "vpxor", "vpcmpeqb", "vpcmpeqd", "vpmullw", "vpmulld",
        ] {
            emit(&mut o, m, "vex.packed.int", &[Xmm, Xmm, XmmRm]);
            emit(&mut o, m, "vex.packed.int", &[Ymm, Ymm, YmmRm]);
        }
        // 2-operand packed (sqrt / reciprocal / conversions)
        for m in ["vsqrtps", "vsqrtpd", "vrcpps", "vrsqrtps", "vcvtdq2ps", "vcvtps2dq", "vcvttps2dq"] {
            emit(&mut o, m, "vex.packed.2op", &[Xmm, XmmRm]);
            emit(&mut o, m, "vex.packed.2op", &[Ymm, YmmRm]);
        }
        // moves: load (reg ← reg/mem) and store (mem ← reg)
        for m in ["vmovaps", "vmovups", "vmovapd", "vmovupd", "vmovdqa", "vmovdqu"] {
            emit(&mut o, m, "vex.mov", &[Xmm, XmmRm]);
            emit(&mut o, m, "vex.mov", &[Ymm, YmmRm]);
            emit(&mut o, m, "vex.mov", &[Mem, Xmm]);
            emit(&mut o, m, "vex.mov", &[Mem, Ymm]);
        }
        // shuffles (imm8)
        for m in ["vshufps", "vshufpd"] {
            emit(&mut o, m, "vex.shuffle", &[Xmm, Xmm, XmmRm, Imm(B)]);
            emit(&mut o, m, "vex.shuffle", &[Ymm, Ymm, YmmRm, Imm(B)]);
        }
        emit(&mut o, "vpshufd", "vex.shuffle", &[Xmm, XmmRm, Imm(B)]);
        emit(&mut o, "vpshufd", "vex.shuffle", &[Ymm, YmmRm, Imm(B)]);

        // ── AVX-512 (EVEX, 512-bit zmm; unmasked) ─────────────────────────────
        // Same opcodes as AVX, EVEX-encoded because the operands are zmm. Only
        // forms with a real 512-bit encoding (excludes AVX-only vrcpps/vrsqrtps).
        for m in [
            "vaddps", "vsubps", "vmulps", "vdivps", "vminps", "vmaxps", "vandps", "vandnps",
            "vorps", "vxorps", "vunpcklps", "vunpckhps", "vaddpd", "vsubpd", "vmulpd", "vdivpd",
            "vminpd", "vmaxpd", "vandpd", "vandnpd", "vorpd", "vxorpd",
        ] {
            emit(&mut o, m, "evex.packed.fp", &[Zmm, Zmm, ZmmRm]);
        }
        // Integer adds/subs/muls keep the same mnemonic under EVEX. (The
        // bitwise ops become vpandd/vpandq etc., and vpcmpeq* take a mask
        // destination — both are out of scope for this unmasked increment.)
        for m in [
            "vpaddb", "vpaddw", "vpaddd", "vpaddq", "vpsubb", "vpsubw", "vpsubd", "vpsubq",
            "vpmullw", "vpmulld",
        ] {
            emit(&mut o, m, "evex.packed.int", &[Zmm, Zmm, ZmmRm]);
        }
        for m in ["vsqrtps", "vsqrtpd", "vcvtdq2ps", "vcvtps2dq", "vcvttps2dq"] {
            emit(&mut o, m, "evex.packed.2op", &[Zmm, ZmmRm]);
        }
        for m in ["vmovaps", "vmovups", "vmovapd", "vmovupd"] {
            emit(&mut o, m, "evex.mov", &[Zmm, ZmmRm]);
            emit(&mut o, m, "evex.mov", &[Mem, Zmm]);
        }
        emit(&mut o, "vshufps", "evex.shuffle", &[Zmm, Zmm, ZmmRm, Imm(B)]);
        emit(&mut o, "vshufpd", "evex.shuffle", &[Zmm, Zmm, ZmmRm, Imm(B)]);
        emit(&mut o, "vpshufd", "evex.shuffle", &[Zmm, ZmmRm, Imm(B)]);

        o
    }
}
