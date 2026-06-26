//! Encode one parsed AArch64 instruction into its 32-bit little-endian word(s)
//! plus any [`Fixup`]s the two-pass driver resolves.
//!
//! Every base opcode and field layout here was derived from vectors verified
//! with `llvm-mc -triple=aarch64-apple-darwin --show-encoding` (see
//! docs/design/aarch64-apple-silicon.md §5). Byte-identity against the LLVM-MC
//! oracle is the gate.
//!
//! Scope (first slice): data movement (mov/mvn/movz/movk/movn), integer ALU
//! (add/sub reg+imm, and/orr/eor reg), loads/stores (unsigned-offset), and
//! branches (b/bl/b.cond/cbz/cbnz). Pre/post-index loads, the bitmask-immediate
//! logical forms, shifts/mul/div, and pc-relative addressing land in later
//! phases.

use anyhow::{bail, Context, Result};

use super::parse::{Addr, IndexExt, Mem, Operand, Reg, RegClass, Shift};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FixupKind {
    /// `b`/`bl` — 26-bit field at bits [25:0], value = (target-site) >> 2.
    Branch26,
    /// `b.cond`/`cbz`/`cbnz` — 19-bit field at bits [23:5], value = (target-site) >> 2.
    Branch19,
    /// `adrp` — 21-bit PC-relative *page*; always relocated (the page address is
    /// load-time, so even local symbols carry an `ARM64_RELOC_PAGE21`).
    AdrpPage21,
    /// `add`/`ldr` `…@PAGEOFF` low-12 page offset; always relocated.
    AddPageOff12,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Fixup {
    /// Byte offset of the affected 4-byte word within this instruction's bytes
    /// (always 0 — every AArch64 instruction is one word).
    pub at: usize,
    pub kind: FixupKind,
    pub target: String,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Encoded {
    pub bytes: Vec<u8>,
    pub fixups: Vec<Fixup>,
}

impl Encoded {
    fn word(w: u32) -> Self {
        Encoded { bytes: w.to_le_bytes().to_vec(), fixups: vec![] }
    }
    fn word_fixup(w: u32, kind: FixupKind, target: &str) -> Self {
        Encoded { bytes: w.to_le_bytes().to_vec(), fixups: vec![Fixup { at: 0, kind, target: target.to_string() }] }
    }
}

/// Map a `b.<cond>` suffix to its 4-bit condition code.
pub fn cond_code(suffix: &str) -> Option<u32> {
    Some(match suffix {
        "eq" => 0,
        "ne" => 1,
        "cs" | "hs" => 2,
        "cc" | "lo" => 3,
        "mi" => 4,
        "pl" => 5,
        "vs" => 6,
        "vc" => 7,
        "hi" => 8,
        "ls" => 9,
        "ge" => 10,
        "lt" => 11,
        "gt" => 12,
        "le" => 13,
        "al" => 14,
        "nv" => 15,
        _ => return None,
    })
}

fn rn(r: Reg) -> u32 {
    r.num as u32
}

fn want(r: Reg, class: RegClass, what: &str) -> Result<u32> {
    if r.class != class {
        bail!("expected {what} register, got {:?}", r.class);
    }
    Ok(r.num as u32)
}

/// `sf` bit (bit 31): 1 for X (64-bit), 0 for W (32-bit). Errors on FP/SIMD.
fn sf_of(r: Reg) -> Result<u32> {
    match r.class {
        RegClass::X => Ok(1),
        RegClass::W => Ok(0),
        _ => bail!("expected a general-purpose register here, got {:?}", r.class),
    }
}

pub fn encode(mnemonic: &str, ops: &[Operand]) -> Result<Encoded> {
    use Operand::*;

    // Vector modified-immediate (movi/mvni/bic/orr #imm) — before try_simd so
    // the register forms of bic/orr still fall through to the logical path.
    if let Some(r) = try_movi(mnemonic, ops) {
        return r;
    }

    // Crypto extensions (AES / SHA).
    if let Some(r) = try_crypto(mnemonic, ops) {
        return r;
    }

    // NEON / SIMD family (checked first; matches only vector operands).
    if let Some(r) = try_simd(mnemonic, ops) {
        return r;
    }

    // System, atomics, and the integer-ISA tail.
    if let Some(r) = try_system(mnemonic, ops) {
        return r;
    }
    if let Some(r) = try_atomics(mnemonic, ops) {
        return r;
    }
    if let Some(r) = try_intmisc(mnemonic, ops) {
        return r;
    }

    // Scalar floating-point family (kept modular, like x86's `try_sse`).
    if let Some(r) = try_fp(mnemonic, ops) {
        return r;
    }

    // ── condition branches: `b.<cond> sym` ──────────────────────────────────
    if let Some(suffix) = mnemonic.strip_prefix("b.") {
        let cond = cond_code(suffix).ok_or_else(|| anyhow::anyhow!("unknown condition `{suffix}`"))?;
        let [Sym(t)] = ops else { bail!("b.{suffix} expects a label") };
        // 0x54000000 | imm19<<5 | cond ; imm19 filled by the driver.
        return Ok(Encoded::word_fixup(0x5400_0000 | cond, FixupKind::Branch19, t));
    }

    match (mnemonic, ops) {
        // ── system / no-operand ─────────────────────────────────────────────
        ("nop", []) => Ok(Encoded::word(0xD503_201F)),
        ("ret", []) => Ok(Encoded::word(0xD65F_0000 | (30 << 5))),
        ("ret", [Reg(r)]) => Ok(Encoded::word(0xD65F_0000 | (want(*r, RegClass::X, "x")? << 5))),
        ("brk", [Imm(n)]) => Ok(Encoded::word(0xD420_0000 | (((*n as u32) & 0xFFFF) << 5))),

        // ── moves ───────────────────────────────────────────────────────────
        ("mov", [Reg(d), Reg(s)]) => mov_reg(*d, *s),
        ("mov", [Reg(d), Imm(_) | ImmShift(_, _)]) => mov_imm(*d, &ops[1]),
        // mvn Rd, Rn  →  ORN Rd, ZR, Rn (logical ORR with N=1)
        ("mvn", [Reg(d), Reg(s)]) => logical_reg(0b01, true, *d, zr(*d), *s, Shift::Lsl, 0),
        ("mvn", [Reg(d), RegShift(s, sh, a)]) => logical_reg(0b01, true, *d, zr(*d), *s, *sh, *a),
        ("movz", [Reg(d), imm]) => move_wide(*d, imm, MoveWide::Z),
        ("movn", [Reg(d), imm]) => move_wide(*d, imm, MoveWide::N),
        ("movk", [Reg(d), imm]) => move_wide(*d, imm, MoveWide::K),

        // ── pc-relative address materialization ─────────────────────────────
        ("adrp", [Reg(d), Sym(s)]) => {
            let target = s.strip_suffix("@PAGE").unwrap_or(s);
            Ok(Encoded::word_fixup(0x9000_0000 | rn(*d), FixupKind::AdrpPage21, target))
        }
        // `add Xd, Xn, sym@PAGEOFF` — the low-12 companion to `adrp`.
        ("add", [Reg(d), Reg(n), Sym(s)]) => {
            let target = s.strip_suffix("@PAGEOFF").unwrap_or(s);
            let w = 0x9100_0000 | (rn(*n) << 5) | rn(*d);
            Ok(Encoded::word_fixup(w, FixupKind::AddPageOff12, target))
        }

        // ── add/sub (+ S forms; SP operands use the extended-register form) ──
        ("add", [Reg(d), Reg(n), third]) => add_sub(false, false, *d, *n, third),
        ("adds", [Reg(d), Reg(n), third]) => add_sub(false, true, *d, *n, third),
        ("sub", [Reg(d), Reg(n), third]) => add_sub(true, false, *d, *n, third),
        ("subs", [Reg(d), Reg(n), third]) => add_sub(true, true, *d, *n, third),
        ("cmn", [Reg(n), third]) => add_sub(false, true, zr(*n), *n, third),
        ("cmp", [Reg(n), third]) => add_sub(true, true, zr(*n), *n, third),
        ("neg", [Reg(d), third]) => add_sub(true, false, *d, zr(*d), third),
        ("negs", [Reg(d), third]) => add_sub(true, true, *d, zr(*d), third),

        // ── logical (shifted register + bitmask immediate) ──────────────────
        ("and", [Reg(d), Reg(n), third]) => logical(0b00, *d, *n, third),
        ("orr", [Reg(d), Reg(n), third]) => logical(0b01, *d, *n, third),
        ("eor", [Reg(d), Reg(n), third]) => logical(0b10, *d, *n, third),
        ("ands", [Reg(d), Reg(n), third]) => logical(0b11, *d, *n, third),
        ("tst", [Reg(n), third]) => logical(0b11, zr(*n), *n, third),
        // Logical with inverted operand (N=1): bic/orn/eon/bics (register only).
        ("bic", [Reg(d), Reg(n), third]) => logical_n(0b00, *d, *n, third),
        ("orn", [Reg(d), Reg(n), third]) => logical_n(0b01, *d, *n, third),
        ("eon", [Reg(d), Reg(n), third]) => logical_n(0b10, *d, *n, third),
        ("bics", [Reg(d), Reg(n), third]) => logical_n(0b11, *d, *n, third),

        // ── multiply / divide ───────────────────────────────────────────────
        ("mul", [Reg(d), Reg(n), Reg(m)]) => data3(*d, *n, *m, zr(*d), false),
        ("mneg", [Reg(d), Reg(n), Reg(m)]) => data3(*d, *n, *m, zr(*d), true),
        ("madd", [Reg(d), Reg(n), Reg(m), Reg(a)]) => data3(*d, *n, *m, *a, false),
        ("msub", [Reg(d), Reg(n), Reg(m), Reg(a)]) => data3(*d, *n, *m, *a, true),
        ("smull", [Reg(d), Reg(n), Reg(m)]) => data3_long(*d, *n, *m, true, false),
        ("umull", [Reg(d), Reg(n), Reg(m)]) => data3_long(*d, *n, *m, false, false),
        ("smnegl", [Reg(d), Reg(n), Reg(m)]) => data3_long(*d, *n, *m, true, true),
        ("umnegl", [Reg(d), Reg(n), Reg(m)]) => data3_long(*d, *n, *m, false, true),
        ("sdiv", [Reg(d), Reg(n), Reg(m)]) => div(*d, *n, *m, true),
        ("udiv", [Reg(d), Reg(n), Reg(m)]) => div(*d, *n, *m, false),

        // ── shifts: register variants (lslv/lsrv/asrv/rorv) + immediate ─────
        ("lsl" | "lslv", [Reg(d), Reg(n), Reg(m)]) => shift_var(*d, *n, *m, 0b00),
        ("lsr" | "lsrv", [Reg(d), Reg(n), Reg(m)]) => shift_var(*d, *n, *m, 0b01),
        ("asr" | "asrv", [Reg(d), Reg(n), Reg(m)]) => shift_var(*d, *n, *m, 0b10),
        ("ror" | "rorv", [Reg(d), Reg(n), Reg(m)]) => shift_var(*d, *n, *m, 0b11),
        ("lsl", [Reg(d), Reg(n), Imm(sh)]) => lsl_imm(*d, *n, *sh),
        ("lsr", [Reg(d), Reg(n), Imm(sh)]) => shift_right_imm(*d, *n, *sh, BfmKind::Ubfm),
        ("asr", [Reg(d), Reg(n), Imm(sh)]) => shift_right_imm(*d, *n, *sh, BfmKind::Sbfm),
        ("ror", [Reg(d), Reg(n), Imm(sh)]) => extr(*d, *n, *n, *sh),
        ("extr", [Reg(d), Reg(n), Reg(m), Imm(sh)]) => extr(*d, *n, *m, *sh),

        // ── data-processing (1 source): count / reverse ─────────────────────
        ("rbit", [Reg(d), Reg(n)]) => dp1(*d, *n, 0),
        ("rev16", [Reg(d), Reg(n)]) => dp1(*d, *n, 1),
        ("rev32", [Reg(d), Reg(n)]) => dp1(*d, *n, 2),
        ("rev", [Reg(d), Reg(n)]) => dp1(*d, *n, if d.class == RegClass::X { 3 } else { 2 }),
        ("clz", [Reg(d), Reg(n)]) => dp1(*d, *n, 4),
        ("cls", [Reg(d), Reg(n)]) => dp1(*d, *n, 5),
        // bitfield insert / insert-low (BFM aliases)
        ("bfi", [Reg(d), Reg(n), Imm(lsb), Imm(w)]) => bfiz(*d, *n, *lsb, *w, BfmKind::Bfm),
        ("bfxil", [Reg(d), Reg(n), Imm(lsb), Imm(w)]) => bfx(*d, *n, *lsb, *w, BfmKind::Bfm),

        // ── bitfield extract / sign+zero extend ─────────────────────────────
        ("ubfx", [Reg(d), Reg(n), Imm(lsb), Imm(w)]) => bfx(*d, *n, *lsb, *w, BfmKind::Ubfm),
        ("sbfx", [Reg(d), Reg(n), Imm(lsb), Imm(w)]) => bfx(*d, *n, *lsb, *w, BfmKind::Sbfm),
        ("ubfiz", [Reg(d), Reg(n), Imm(lsb), Imm(w)]) => bfiz(*d, *n, *lsb, *w, BfmKind::Ubfm),
        ("sbfiz", [Reg(d), Reg(n), Imm(lsb), Imm(w)]) => bfiz(*d, *n, *lsb, *w, BfmKind::Sbfm),
        ("sxtw", [Reg(d), Reg(n)]) => bfm(*d, *n, 0, 31, BfmKind::Sbfm),
        ("sxtb", [Reg(d), Reg(n)]) => bfm(*d, *n, 0, 7, BfmKind::Sbfm),
        ("sxth", [Reg(d), Reg(n)]) => bfm(*d, *n, 0, 15, BfmKind::Sbfm),
        ("uxtw", [Reg(d), Reg(n)]) => bfm(*d, *n, 0, 31, BfmKind::Ubfm),
        ("uxtb", [Reg(d), Reg(n)]) => bfm(*d, *n, 0, 7, BfmKind::Ubfm),
        ("uxth", [Reg(d), Reg(n)]) => bfm(*d, *n, 0, 15, BfmKind::Ubfm),

        // ── conditional select family ───────────────────────────────────────
        ("csel", [Reg(d), Reg(n), Reg(m), Sym(c)]) => csel(*d, *n, *m, cc(c)?, CSel::Csel),
        ("csinc", [Reg(d), Reg(n), Reg(m), Sym(c)]) => csel(*d, *n, *m, cc(c)?, CSel::Csinc),
        ("csinv", [Reg(d), Reg(n), Reg(m), Sym(c)]) => csel(*d, *n, *m, cc(c)?, CSel::Csinv),
        ("csneg", [Reg(d), Reg(n), Reg(m), Sym(c)]) => csel(*d, *n, *m, cc(c)?, CSel::Csneg),
        ("cset", [Reg(d), Sym(c)]) => csel(*d, zr(*d), zr(*d), cc(c)? ^ 1, CSel::Csinc),
        ("csetm", [Reg(d), Sym(c)]) => csel(*d, zr(*d), zr(*d), cc(c)? ^ 1, CSel::Csinv),
        ("cinc", [Reg(d), Reg(n), Sym(c)]) => csel(*d, *n, *n, cc(c)? ^ 1, CSel::Csinc),
        ("cinv", [Reg(d), Reg(n), Sym(c)]) => csel(*d, *n, *n, cc(c)? ^ 1, CSel::Csinv),
        ("cneg", [Reg(d), Reg(n), Sym(c)]) => csel(*d, *n, *n, cc(c)? ^ 1, CSel::Csneg),

        // ── loads / stores ──────────────────────────────────────────────────
        ("ldr" | "str" | "ldrb" | "strb" | "ldrh" | "strh" | "ldrsb" | "ldrsh" | "ldrsw",
            [Reg(t), Mem(m)]) => ldst(mnemonic, *t, m),
        ("ldur" | "stur" | "ldurb" | "sturb" | "ldurh" | "sturh" | "ldursb" | "ldursh" | "ldursw",
            [Reg(t), Mem(m)]) => ldst_ur(mnemonic, *t, m),
        ("ldp" | "stp", [Reg(t1), Reg(t2), Mem(m)]) => ldst_pair(mnemonic, *t1, *t2, m),

        // ── branches ────────────────────────────────────────────────────────
        ("b", [Sym(t)]) => Ok(Encoded::word_fixup(0x1400_0000, FixupKind::Branch26, t)),
        ("bl", [Sym(t)]) => Ok(Encoded::word_fixup(0x9400_0000, FixupKind::Branch26, t)),
        ("br", [Reg(r)]) => Ok(Encoded::word(0xD61F_0000 | (want(*r, RegClass::X, "x")? << 5))),
        ("blr", [Reg(r)]) => Ok(Encoded::word(0xD63F_0000 | (want(*r, RegClass::X, "x")? << 5))),
        ("cbz", [Reg(r), Sym(t)]) => compare_branch(*r, t, false),
        ("cbnz", [Reg(r), Sym(t)]) => compare_branch(*r, t, true),

        _ => bail!("unsupported AArch64 instruction `{mnemonic}` with {} operand(s)", ops.len()),
    }
}

/// The all-zero-or-XZR register of the same width as `r` (#31, not SP).
fn zr(r: Reg) -> Reg {
    Reg { class: r.class, num: 31, is_sp: false }
}

/// Condition-code lookup for a bare condition token (`Operand::Sym`).
fn cc(name: &str) -> Result<u32> {
    cond_code(name).ok_or_else(|| anyhow::anyhow!("unknown condition `{name}`"))
}

/// `mov Rd, Rn` — `ORR Rd, XZR, Rn`, except when SP is involved, where it is
/// `ADD Rd, Rn, #0` (XZR and SP both encode #31, so ORR can't name SP).
fn mov_reg(d: Reg, s: Reg) -> Result<Encoded> {
    if d.class != s.class || !matches!(d.class, RegClass::X | RegClass::W) {
        bail!("mov register width mismatch (GP only; use fmov for FP)");
    }
    if d.is_sp || s.is_sp {
        return add_sub_imm(d, s, 0, 0, false, false);
    }
    let base = if d.class == RegClass::X { 0xAA00_03E0 } else { 0x2A00_03E0 };
    Ok(Encoded::word(base | (rn(s) << 16) | rn(d)))
}

enum MoveWide {
    N,
    Z,
    K,
}

fn move_wide(d: Reg, imm: &Operand, kind: MoveWide) -> Result<Encoded> {
    let sf = sf_of(d)?;
    let (val, shift) = match imm {
        Operand::Imm(v) => (*v, 0u32),
        Operand::ImmShift(v, s) => (*v, *s),
        _ => bail!("move-wide expects an immediate"),
    };
    if shift % 16 != 0 || shift > 48 || (sf == 0 && shift > 16) {
        bail!("invalid move-wide shift #{shift}");
    }
    let hw = shift / 16;
    let imm16 = (val as u64) & 0xFFFF;
    if (val as u64) > 0xFFFF && val >= 0 {
        bail!("move-wide immediate {val:#x} exceeds 16 bits (use lsl)");
    }
    let opc: u32 = match kind {
        MoveWide::N => 0b00,
        MoveWide::Z => 0b10,
        MoveWide::K => 0b11,
    };
    let w = (sf << 31) | (opc << 29) | (0b100101 << 23) | (hw << 21) | ((imm16 as u32) << 5) | rn(d);
    Ok(Encoded::word(w))
}

/// `add`/`sub` (and `adds`/`subs`) dispatcher. SP as `Rd`/`Rn` forces the
/// extended-register form (XZR and SP both encode #31, so the shifted-register
/// form can't name SP). `third` is the second source operand.
fn add_sub(sub: bool, s: bool, d: Reg, n: Reg, third: &Operand) -> Result<Encoded> {
    let use_ext = d.is_sp || n.is_sp;
    match third {
        Operand::Imm(i) => add_sub_imm(d, n, *i, 0, sub, s),
        Operand::ImmShift(i, sh) => add_sub_imm(d, n, *i, *sh, sub, s),
        Operand::Reg(m) if use_ext => add_sub_ext(d, n, *m, 0, sub, s),
        Operand::Reg(m) => add_sub_reg(d, n, *m, Shift::Lsl, 0, sub, s),
        Operand::RegShift(m, sh, amt) if use_ext => {
            if *sh != Shift::Lsl {
                bail!("add/sub with SP only allows `lsl` shift");
            }
            add_sub_ext(d, n, *m, *amt, sub, s)
        }
        Operand::RegShift(m, sh, amt) => add_sub_reg(d, n, *m, *sh, *amt, sub, s),
        _ => bail!("invalid add/sub operand"),
    }
}

fn add_sub_imm(d: Reg, n: Reg, imm: i64, shift: u32, sub: bool, s: bool) -> Result<Encoded> {
    let sf = sf_of(d)?;
    if d.class != n.class {
        bail!("add/sub operand width mismatch");
    }
    let sh = match shift {
        0 => 0u32,
        12 => 1u32,
        _ => bail!("add/sub immediate shift must be 0 or 12"),
    };
    if !(0..=4095).contains(&imm) {
        bail!("add/sub immediate {imm} out of range 0..=4095 (negative not folded yet)");
    }
    let w = (sf << 31)
        | ((sub as u32) << 30)
        | ((s as u32) << 29)
        | (0b100010 << 23)
        | (sh << 22)
        | ((imm as u32) << 10)
        | (rn(n) << 5)
        | rn(d);
    Ok(Encoded::word(w))
}

fn shift_bits(sh: Shift) -> u32 {
    match sh {
        Shift::Lsl => 0b00,
        Shift::Lsr => 0b01,
        Shift::Asr => 0b10,
        Shift::Ror => 0b11,
    }
}

fn add_sub_reg(d: Reg, n: Reg, m: Reg, sh: Shift, amt: u32, sub: bool, s: bool) -> Result<Encoded> {
    let sf = sf_of(d)?;
    if d.class != n.class || d.class != m.class {
        bail!("add/sub register width mismatch");
    }
    let max = if sf == 1 { 63 } else { 31 };
    if amt > max {
        bail!("shift amount #{amt} out of range");
    }
    let w = (sf << 31)
        | ((sub as u32) << 30)
        | ((s as u32) << 29)
        | (0b01011 << 24)
        | (shift_bits(sh) << 22)
        | (rn(m) << 16)
        | (amt << 10)
        | (rn(n) << 5)
        | rn(d);
    Ok(Encoded::word(w))
}

/// `add`/`sub` extended-register form (used when SP is `Rd`/`Rn`). The shift
/// amount rides in `imm3` (0..=4) with option = UXTX (64-bit) / UXTW (32-bit) —
/// the canonical form LLVM emits for `add x0, sp, x1{, lsl #n}`.
fn add_sub_ext(d: Reg, n: Reg, m: Reg, imm3: u32, sub: bool, s: bool) -> Result<Encoded> {
    let sf = sf_of(d)?;
    if imm3 > 4 {
        bail!("extended-register shift amount #{imm3} out of range 0..=4");
    }
    let option = if sf == 1 { 0b011 } else { 0b010 }; // UXTX / UXTW
    let w = (sf << 31)
        | ((sub as u32) << 30)
        | ((s as u32) << 29)
        | (0b01011 << 24)
        | (1 << 21)
        | (rn(m) << 16)
        | (option << 13)
        | (imm3 << 10)
        | (rn(n) << 5)
        | rn(d);
    Ok(Encoded::word(w))
}

/// `and`/`orr`/`eor`/`ands` dispatcher — register or bitmask-immediate.
fn logical(opc: u32, d: Reg, n: Reg, third: &Operand) -> Result<Encoded> {
    match third {
        Operand::Reg(m) => logical_reg(opc, false, d, n, *m, Shift::Lsl, 0),
        Operand::RegShift(m, sh, amt) => logical_reg(opc, false, d, n, *m, *sh, *amt),
        Operand::Imm(i) => logical_imm(opc, d, n, *i),
        _ => bail!("invalid logical operand"),
    }
}

fn logical_reg(opc: u32, n_bit: bool, d: Reg, n: Reg, m: Reg, sh: Shift, amt: u32) -> Result<Encoded> {
    let sf = sf_of(d)?;
    if d.class != n.class || d.class != m.class {
        bail!("logical register width mismatch");
    }
    let max = if sf == 1 { 63 } else { 31 };
    if amt > max {
        bail!("shift amount #{amt} out of range");
    }
    let w = (sf << 31)
        | (opc << 29)
        | (0b01010 << 24)
        | (shift_bits(sh) << 22)
        | ((n_bit as u32) << 21)
        | (rn(m) << 16)
        | (amt << 10)
        | (rn(n) << 5)
        | rn(d);
    Ok(Encoded::word(w))
}

/// `bic`/`orn`/`eon`/`bics` — logical with inverted second operand (N=1),
/// register form only.
fn logical_n(opc: u32, d: Reg, n: Reg, third: &Operand) -> Result<Encoded> {
    match third {
        Operand::Reg(m) => logical_reg(opc, true, d, n, *m, Shift::Lsl, 0),
        Operand::RegShift(m, sh, amt) => logical_reg(opc, true, d, n, *m, *sh, *amt),
        _ => bail!("bic/orn/eon take register operands"),
    }
}

/// `and`/`orr`/`eor`/`ands` with a logical (bitmask) immediate.
fn logical_imm(opc: u32, d: Reg, n: Reg, imm: i64) -> Result<Encoded> {
    let sf = sf_of(d)?;
    let reg_size = if sf == 1 { 64 } else { 32 };
    let (nbit, immr, imms) = encode_bitmask(imm as u64, reg_size)
        .ok_or_else(|| anyhow::anyhow!("{imm:#x} is not a valid AArch64 logical (bitmask) immediate"))?;
    let w = (sf << 31)
        | (opc << 29)
        | (0b100100 << 23)
        | (nbit << 22)
        | (immr << 16)
        | (imms << 10)
        | (rn(n) << 5)
        | rn(d);
    Ok(Encoded::word(w))
}

fn is_mask_64(v: u64) -> bool {
    v != 0 && (v.wrapping_add(1) & v) == 0
}

fn is_shifted_mask_64(v: u64) -> bool {
    v != 0 && is_mask_64((v - 1) | v)
}

/// Encode a logical (bitmask) immediate to `(N, immr, imms)`, or `None` if the
/// value is not a legal bitmask. Faithful port of LLVM's
/// `AArch64_AM::processLogicalImmediate`.
fn encode_bitmask(mut imm: u64, reg_size: u32) -> Option<(u32, u32, u32)> {
    if reg_size == 32 {
        if imm >> 32 != 0 {
            return None;
        }
        imm |= imm << 32;
    }
    if imm == 0 || imm == u64::MAX {
        return None;
    }
    // Determine the element size (largest power of two whose halves match).
    let mut size = 64u32;
    loop {
        size >>= 1;
        let mask = (1u64 << size) - 1;
        if (imm & mask) != ((imm >> size) & mask) {
            size <<= 1;
            break;
        }
        if size <= 2 {
            break;
        }
    }
    let mask = u64::MAX >> (64 - size);
    imm &= mask;

    let (i, cto): (u32, u32);
    if is_shifted_mask_64(imm) {
        i = imm.trailing_zeros();
        cto = (imm >> i).trailing_ones();
    } else {
        imm |= !mask;
        if !is_shifted_mask_64(!imm) {
            return None;
        }
        let clo = imm.leading_ones();
        i = 64 - clo;
        cto = clo + imm.trailing_ones() - (64 - size);
    }
    let immr = (size - i) & (size - 1);
    let mut nimms: u32 = !(size - 1) << 1;
    nimms |= cto - 1;
    let n = ((nimms >> 6) & 1) ^ 1;
    Some((n, immr, nimms & 0x3f))
}

/// `mov Rd, #imm` — LLVM's assembler lowering: a single `movz`, else a single
/// `movn`, else `orr Rd, ZR, #bitmask`; otherwise not a one-instruction move.
fn mov_imm(d: Reg, imm_op: &Operand) -> Result<Encoded> {
    let sf = sf_of(d)?;
    let reg_size = if sf == 1 { 64 } else { 32 };
    let raw = match imm_op {
        Operand::Imm(v) => *v as u64,
        Operand::ImmShift(v, s) => (*v as u64) << s,
        _ => bail!("mov immediate expected"),
    };
    let mask = if reg_size == 64 { u64::MAX } else { 0xFFFF_FFFF };
    let val = raw & mask;

    // movz: one 16-bit group set, the rest zero.
    if let Some((hw, g)) = movz_fields(val, reg_size) {
        let w = (sf << 31) | (0b10 << 29) | (0b100101 << 23) | (hw << 21) | (g << 5) | rn(d);
        return Ok(Encoded::word(w));
    }
    // movn: the inverse fits a movz pattern.
    let inv = (!val) & mask;
    if let Some((hw, g)) = movz_fields(inv, reg_size) {
        let w = (sf << 31) | (0b00 << 29) | (0b100101 << 23) | (hw << 21) | (g << 5) | rn(d);
        return Ok(Encoded::word(w));
    }
    // orr Rd, ZR, #bitmask.
    if let Some((nbit, immr, imms)) = encode_bitmask(val, reg_size) {
        let w = (sf << 31)
            | (0b01 << 29)
            | (0b100100 << 23)
            | (nbit << 22)
            | (immr << 16)
            | (imms << 10)
            | (31 << 5)
            | rn(d);
        return Ok(Encoded::word(w));
    }
    bail!("mov {raw:#x}: not encodable as a single movz/movn/orr (multi-instruction mov not supported)")
}

/// If `val` is a single 16-bit group at some `hw` (rest zero), return `(hw, g)`.
fn movz_fields(val: u64, reg_size: u32) -> Option<(u32, u32)> {
    let groups = reg_size / 16;
    for hw in 0..groups {
        let shifted = 0xFFFFu64 << (16 * hw);
        if val & !shifted == 0 {
            return Some((hw, ((val >> (16 * hw)) & 0xFFFF) as u32));
        }
    }
    None
}

/// Three-source (`madd`/`msub`; `mul`/`mneg` alias with `Ra = ZR`).
fn data3(d: Reg, n: Reg, m: Reg, a: Reg, sub: bool) -> Result<Encoded> {
    let sf = sf_of(d)?;
    let base = if sf == 1 { 0x9B00_0000 } else { 0x1B00_0000 };
    let o0 = if sub { 0x8000 } else { 0 };
    Ok(Encoded::word(base | o0 | (rn(m) << 16) | (rn(a) << 10) | (rn(n) << 5) | rn(d)))
}

/// Widening multiply-long (`smull`/`umull`; `Ra = XZR`). `Rd` is X, `Rn`/`Rm` W.
fn data3_long(d: Reg, n: Reg, m: Reg, signed: bool, sub: bool) -> Result<Encoded> {
    let base = if signed { 0x9B20_0000 } else { 0x9BA0_0000 };
    let o0 = if sub { 0x8000 } else { 0 };
    Ok(Encoded::word(base | o0 | (rn(m) << 16) | (31 << 10) | (rn(n) << 5) | rn(d)))
}

fn div(d: Reg, n: Reg, m: Reg, signed: bool) -> Result<Encoded> {
    let sf = sf_of(d)?;
    let base = if sf == 1 { 0x9AC0_0000 } else { 0x1AC0_0000 };
    let op = if signed { 0xC00 } else { 0x800 };
    Ok(Encoded::word(base | op | (rn(m) << 16) | (rn(n) << 5) | rn(d)))
}

/// Variable shift (`lslv`/`lsrv`/`asrv`/`rorv`).
fn shift_var(d: Reg, n: Reg, m: Reg, op2: u32) -> Result<Encoded> {
    let sf = sf_of(d)?;
    let base = if sf == 1 { 0x9AC0_2000 } else { 0x1AC0_2000 };
    Ok(Encoded::word(base | (op2 << 10) | (rn(m) << 16) | (rn(n) << 5) | rn(d)))
}

#[derive(Clone, Copy)]
enum BfmKind {
    Sbfm,
    Ubfm,
    Bfm,
}

/// Core bitfield-move encoder (`SBFM`/`UBFM`/`BFM`). The `N` bit equals `sf`.
fn bfm(d: Reg, n: Reg, immr: u32, imms: u32, kind: BfmKind) -> Result<Encoded> {
    let sf = sf_of(d)?;
    let base = match kind {
        BfmKind::Sbfm => 0x1300_0000,
        BfmKind::Bfm => 0x3300_0000,
        BfmKind::Ubfm => 0x5300_0000,
    };
    let w = (sf << 31) | base | (sf << 22) | (immr << 16) | (imms << 10) | (rn(n) << 5) | rn(d);
    Ok(Encoded::word(w))
}

/// Data-processing (1 source): `clz`/`cls`/`rbit`/`rev`/`rev16`/`rev32`.
fn dp1(d: Reg, n: Reg, opc2: u32) -> Result<Encoded> {
    let sf = sf_of(d)?;
    Ok(Encoded::word((sf << 31) | 0x5AC0_0000 | (opc2 << 10) | (rn(n) << 5) | rn(d)))
}

/// `extr Rd, Rn, Rm, #lsb` (and the `ror Rd, Rn, #lsb` alias with `Rm = Rn`).
fn extr(d: Reg, n: Reg, m: Reg, lsb: i64) -> Result<Encoded> {
    let sf = sf_of(d)?;
    let max = if sf == 1 { 63 } else { 31 };
    if !(0..=max).contains(&lsb) {
        bail!("extr/ror amount #{lsb} out of range");
    }
    let w = (sf << 31) | 0x1380_0000 | (sf << 22) | (rn(m) << 16) | ((lsb as u32) << 10) | (rn(n) << 5) | rn(d);
    Ok(Encoded::word(w))
}

fn reg_bits(r: Reg) -> Result<u32> {
    Ok(if sf_of(r)? == 1 { 64 } else { 32 })
}

/// `lsl Rd, Rn, #sh` → `UBFM Rd, Rn, #(-sh mod bits), #(bits-1-sh)`.
fn lsl_imm(d: Reg, n: Reg, sh: i64) -> Result<Encoded> {
    let bits = reg_bits(d)?;
    let sh = sh as u32;
    if sh >= bits {
        bail!("lsl #{sh} out of range for a {bits}-bit register");
    }
    bfm(d, n, (bits - sh) & (bits - 1), bits - 1 - sh, BfmKind::Ubfm)
}

/// `lsr`/`asr Rd, Rn, #sh` → `UBFM`/`SBFM Rd, Rn, #sh, #(bits-1)`.
fn shift_right_imm(d: Reg, n: Reg, sh: i64, kind: BfmKind) -> Result<Encoded> {
    let bits = reg_bits(d)?;
    let sh = sh as u32;
    if sh >= bits {
        bail!("shift #{sh} out of range for a {bits}-bit register");
    }
    bfm(d, n, sh, bits - 1, kind)
}

/// `ubfx`/`sbfx Rd, Rn, #lsb, #width` → `*BFM Rd, Rn, #lsb, #(lsb+width-1)`.
fn bfx(d: Reg, n: Reg, lsb: i64, width: i64, kind: BfmKind) -> Result<Encoded> {
    if width < 1 {
        bail!("bitfield width must be ≥ 1");
    }
    bfm(d, n, lsb as u32, (lsb + width - 1) as u32, kind)
}

/// `ubfiz`/`sbfiz Rd, Rn, #lsb, #width` → `*BFM Rd, Rn, #(-lsb mod bits), #(width-1)`.
fn bfiz(d: Reg, n: Reg, lsb: i64, width: i64, kind: BfmKind) -> Result<Encoded> {
    let bits = reg_bits(d)?;
    if width < 1 {
        bail!("bitfield width must be ≥ 1");
    }
    bfm(d, n, (bits - lsb as u32) & (bits - 1), (width - 1) as u32, kind)
}

#[derive(Clone, Copy)]
enum CSel {
    Csel,
    Csinc,
    Csinv,
    Csneg,
}

// ── NEON / SIMD ─────────────────────────────────────────────────────────────

use super::parse::Arr;

/// Arrangement → `(Q, size)` for integer vector ops (size at bits[23:22]).
fn arr_qsize(a: Arr) -> (u32, u32) {
    match a {
        Arr::B8 => (0, 0),
        Arr::B16 => (1, 0),
        Arr::H4 => (0, 1),
        Arr::H8 => (1, 1),
        Arr::S2 => (0, 2),
        Arr::S4 => (1, 2),
        Arr::D1 => (0, 3),
        Arr::D2 => (1, 3),
    }
}

/// Arrangement → `(Q, sz)` for FP vector ops (`sz` at bit22; .2s/.4s/.2d only).
fn arr_fp_qsz(a: Arr) -> Option<(u32, u32)> {
    match a {
        Arr::S2 => Some((0, 0)),
        Arr::S4 => Some((1, 0)),
        Arr::D2 => Some((1, 1)),
        _ => None,
    }
}

/// Element size of an arrangement (B=0,H=1,S=2,D=3).
fn arr_esize(a: Arr) -> u32 {
    arr_qsize(a).1
}

fn elem_size(es: super::parse::ElemSize) -> u32 {
    use super::parse::ElemSize::*;
    match es {
        B => 0,
        H => 1,
        S => 2,
        D => 3,
    }
}

/// 3-same integer base (Q=0,size=0,Rm/Rn/Rd=0): U + opcode + fixed bits.
fn simd3_int_base(m: &str) -> Option<u32> {
    Some(match m {
        "add" => 0x0E20_8400,
        "sub" => 0x2E20_8400,
        "mul" => 0x0E20_9C00,
        "mla" => 0x0E20_9400,
        "mls" => 0x2E20_9400,
        "smax" => 0x0E20_6400,
        "umax" => 0x2E20_6400,
        "smin" => 0x0E20_6C00,
        "umin" => 0x2E20_6C00,
        "sshl" => 0x0E20_4400,
        "ushl" => 0x2E20_4400,
        "sqadd" => 0x0E20_0C00,
        "uqadd" => 0x2E20_0C00,
        "sqsub" => 0x0E20_2C00,
        "uqsub" => 0x2E20_2C00,
        "cmgt" => 0x0E20_3400,
        "cmge" => 0x0E20_3C00,
        "cmhi" => 0x2E20_3400,
        "cmhs" => 0x2E20_3C00,
        "cmtst" => 0x0E20_8C00,
        "cmeq" => 0x2E20_8C00,
        "addp" => 0x0E20_BC00,
        _ => return None,
    })
}

/// 3-same logical base (Q=0, Rm/Rn/Rd=0); arrangement is .8b/.16b (Q only).
fn simd3_logical_base(m: &str) -> Option<u32> {
    Some(match m {
        "and" => 0x0E20_1C00,
        "bic" => 0x0E60_1C00,
        "orr" => 0x0EA0_1C00,
        "orn" => 0x0EE0_1C00,
        "eor" => 0x2E20_1C00,
        "bsl" => 0x2E60_1C00,
        "bit" => 0x2EA0_1C00,
        "bif" => 0x2EE0_1C00,
        _ => return None,
    })
}

/// 3-same FP base (Q=0,sz=0,Rm/Rn/Rd=0); arrangement .2s/.4s/.2d.
fn simd3_fp_base(m: &str) -> Option<u32> {
    Some(match m {
        "fadd" => 0x0E20_D400,
        "fsub" => 0x0EA0_D400,
        "fmul" => 0x2E20_DC00,
        "fdiv" => 0x2E20_FC00,
        "fmla" => 0x0E20_CC00,
        "fmls" => 0x0EA0_CC00,
        "fmax" => 0x0E20_F400,
        "fmin" => 0x0EA0_F400,
        "fcmeq" => 0x0E20_E400,
        "fcmge" => 0x2E20_E400,
        "fcmgt" => 0x2EA0_E400,
        "fabd" => 0x2EA0_D400,
        "fmulx" => 0x0E20_DC00,
        "fmaxnm" => 0x0E20_C400,
        "fminnm" => 0x0EA0_C400,
        "faddp" => 0x2E20_D400,
        "fmaxp" => 0x2E20_F400,
        "fminp" => 0x2EA0_F400,
        "fmaxnmp" => 0x2E20_C400,
        "fminnmp" => 0x2EA0_C400,
        _ => return None,
    })
}

/// 2-misc FP compare-against-zero base (Q=0,sz=0,Rn/Rd=0): `Vd, Vn, #0.0`.
fn simd2_fcmp_zero_base(m: &str) -> Option<u32> {
    Some(match m {
        "fcmeq" => 0x0EA0_D800,
        "fcmge" => 0x2EA0_C800,
        "fcmgt" => 0x0EA0_C800,
        "fcmle" => 0x2EA0_D800,
        "fcmlt" => 0x0EA0_E800,
        _ => return None,
    })
}

/// 2-misc integer base (Q=0,size=0,Rn/Rd=0).
fn simd2_int_base(m: &str) -> Option<u32> {
    Some(match m {
        "neg" => 0x2E20_B800,
        "abs" => 0x0E20_B800,
        "not" | "mvn" => 0x2E20_5800,
        "cnt" => 0x0E20_5800,
        "cls" => 0x0E20_4800,
        "clz" => 0x2E20_4800,
        "rev16" => 0x0E20_1800,
        "rev32" => 0x2E20_0800,
        "rev64" => 0x0E20_0800,
        _ => return None,
    })
}

/// 2-misc FP base (Q=0,sz=0,Rn/Rd=0).
fn simd2_fp_base(m: &str) -> Option<u32> {
    Some(match m {
        "fabs" => 0x0EA0_F800,
        "fneg" => 0x2EA0_F800,
        "fsqrt" => 0x2EA1_F800,
        "scvtf" => 0x0E21_D800,
        "ucvtf" => 0x2E21_D800,
        "fcvtzs" => 0x0EA1_B800,
        "fcvtzu" => 0x2EA1_B800,
        "frintn" => 0x0E21_8800,
        "frintp" => 0x0EA1_8800,
        "frintm" => 0x0E21_9800,
        "frintz" => 0x0EA1_9800,
        _ => return None,
    })
}

/// Same-width vector shift-by-immediate: `(base, is_left_shift)`.
/// base has Q=0, immh:immb=0, Rn/Rd=0 (U + opcode + fixed bit10 baked in).
fn simd_shift_imm(m: &str) -> Option<(u32, bool)> {
    Some(match m {
        "shl" => (0x0F00_5400, true),
        "sli" => (0x2F00_5400, true),
        "sshr" => (0x0F00_0400, false),
        "ushr" => (0x2F00_0400, false),
        "ssra" => (0x0F00_1400, false),
        "usra" => (0x2F00_1400, false),
        "srshr" => (0x0F00_2400, false),
        "urshr" => (0x2F00_2400, false),
        "srsra" => (0x0F00_3400, false),
        "ursra" => (0x2F00_3400, false),
        "sri" => (0x2F00_4400, false),
        _ => return None,
    })
}

fn is_struct_ldst(m: &str) -> bool {
    matches!(m, "ld1" | "st1" | "ld2" | "st2" | "ld3" | "st3" | "ld4" | "st4")
}

/// Structured load/store opcode (bits[15:12]) from mnemonic + register count,
/// validating the count. ld1/st1 take 1–4 (each a distinct opcode).
fn struct_opcode(m: &str, count: u8) -> Result<u32> {
    let (want, opcode) = match m {
        "ld1" | "st1" => {
            return Ok(match count {
                1 => 0b0111,
                2 => 0b1010,
                3 => 0b0110,
                4 => 0b0010,
                _ => bail!("ld1/st1 needs 1–4 registers"),
            })
        }
        "ld2" | "st2" => (2, 0b1000),
        "ld3" | "st3" => (3, 0b0100),
        "ld4" | "st4" => (4, 0b0000),
        _ => bail!("not a structured load/store"),
    };
    if count != want {
        bail!("{m} needs {want} registers, got {count}");
    }
    Ok(opcode)
}

/// By-element op base (Q=0,size=0,index=0,regs=0).
fn simd_byelem(m: &str) -> Option<u32> {
    Some(match m {
        "mul" => 0x0F00_8000,
        "mla" => 0x2F00_0000,
        "mls" => 0x2F00_4000,
        "fmul" => 0x0F00_9000,
        "fmla" => 0x0F00_1000,
        "fmls" => 0x0F00_5000,
        "fmulx" => 0x2F00_9000,
        "sqdmulh" => 0x0F00_C000,
        "sqrdmulh" => 0x0F00_D000,
        "smull" => 0x0F00_A000,
        "umull" => 0x2F00_A000,
        "smlal" => 0x0F00_2000,
        "umlal" => 0x2F00_2000,
        "smlsl" => 0x0F00_6000,
        "umlsl" => 0x2F00_6000,
        _ => return None,
    })
}

/// By-element `(size, index_bits, Rm)` from element size + Vm reg + lane index.
/// S-size: index = H:L (Rm is 5-bit). H-size: index = H:L:M (Rm ≤ v15, 4-bit).
fn byelem_index(es: super::parse::ElemSize, m: u8, idx: u8) -> Result<(u32, u32, u32)> {
    use super::parse::ElemSize::*;
    let (idx, m) = (idx as u32, m as u32);
    match es {
        S => Ok((2, ((idx & 1) << 21) | ((idx >> 1) << 11), m)),
        H => {
            if m > 15 {
                bail!("by-element .h index requires Vm in v0–v15");
            }
            Ok((1, (((idx >> 1) & 1) << 21) | ((idx & 1) << 20) | ((idx >> 2) << 11), m))
        }
        D => Ok((3, idx << 11, m)),
        B => bail!("no by-element byte form"),
    }
}

/// Which operand's arrangement supplies (Q,size) for a widen/narrow op.
#[derive(Clone, Copy)]
enum QFrom {
    Vn,
    Vm,
    /// 3-operand narrowing (addhn/subhn): size from the narrow destination.
    Vd,
    /// 2-operand narrowing (xtn/sqxtn): size from the destination.
    Vd2,
}

/// Widen/narrow/long base (Q=0,size=0,regs=0) + the (Q,size) source operand.
fn simd_widen(m: &str) -> Option<(u32, QFrom)> {
    Some(match m {
        "saddl" => (0x0E20_0000, QFrom::Vn),
        "uaddl" => (0x2E20_0000, QFrom::Vn),
        "ssubl" => (0x0E20_2000, QFrom::Vn),
        "usubl" => (0x2E20_2000, QFrom::Vn),
        "saddw" => (0x0E20_1000, QFrom::Vm),
        "uaddw" => (0x2E20_1000, QFrom::Vm),
        "ssubw" => (0x0E20_3000, QFrom::Vm),
        "usubw" => (0x2E20_3000, QFrom::Vm),
        "addhn" => (0x0E20_4000, QFrom::Vd),
        "raddhn" => (0x2E20_4000, QFrom::Vd),
        "subhn" => (0x0E20_6000, QFrom::Vd),
        "rsubhn" => (0x2E20_6000, QFrom::Vd),
        "smull" => (0x0E20_C000, QFrom::Vn),
        "umull" => (0x2E20_C000, QFrom::Vn),
        "smlal" => (0x0E20_8000, QFrom::Vn),
        "umlal" => (0x2E20_8000, QFrom::Vn),
        "smlsl" => (0x0E20_A000, QFrom::Vn),
        "umlsl" => (0x2E20_A000, QFrom::Vn),
        "xtn" => (0x0E21_2800, QFrom::Vd2),
        "sqxtn" => (0x0E21_4800, QFrom::Vd2),
        "uqxtn" => (0x2E21_4800, QFrom::Vd2),
        "sqxtun" => (0x2E21_2800, QFrom::Vd2),
        _ => return None,
    })
}

/// Permute base (Q=0,size=0,Rm/Rn/Rd=0): zip/uzp/trn.
fn simd_permute_base(m: &str) -> Option<u32> {
    Some(match m {
        "uzp1" => 0x0E00_1800,
        "trn1" => 0x0E00_2800,
        "zip1" => 0x0E00_3800,
        "uzp2" => 0x0E00_5800,
        "trn2" => 0x0E00_6800,
        "zip2" => 0x0E00_7800,
        _ => return None,
    })
}

/// Across-lane reduction base (Q=0,size=0,Rn/Rd=0).
fn simd_across_base(m: &str) -> Option<u32> {
    Some(match m {
        "addv" => 0x0E31_B800,
        "smaxv" => 0x0E30_A800,
        "sminv" => 0x0E31_A800,
        "umaxv" => 0x2E30_A800,
        "uminv" => 0x2E31_A800,
        "saddlv" => 0x0E30_3800,
        "uaddlv" => 0x2E30_3800,
        _ => return None,
    })
}

/// If `v64` is a per-byte 0x00/0xFF mask, return the 8-bit `movi .2d` immediate.
fn byte_mask_imm8(v64: u64) -> Option<u8> {
    let mut imm8 = 0u8;
    for i in 0..8 {
        match (v64 >> (8 * i)) & 0xFF {
            0x00 => {}
            0xFF => imm8 |= 1 << i,
            _ => return None,
        }
    }
    Some(imm8)
}

/// Vector modified immediate: movi / mvni / bic(#imm) / orr(#imm), and scalar
/// `movi Dd, #imm64`. Returns `None` for non-immediate forms (register bic/orr
/// fall through to the logical path).
fn try_movi(mnem: &str, ops: &[Operand]) -> Option<Result<Encoded>> {
    use Operand::{Imm, ImmMsl, ImmShift, Reg, VReg};
    if !matches!(mnem, "movi" | "mvni" | "bic" | "orr") {
        return None;
    }
    let word = |q: u32, op: u32, cmode: u32, imm8: u8, d: u32| {
        let abc = ((imm8 >> 5) & 7) as u32;
        let defgh = (imm8 & 0x1F) as u32;
        Encoded::word(0x0F00_0400 | (q << 30) | (op << 29) | (abc << 16) | (cmode << 12) | (defgh << 5) | d)
    };

    // Scalar `movi Dd, #imm64` (byte-mask): op=1, cmode=1110, Q=0.
    if mnem == "movi" {
        if let [Reg(d), Imm(v)] = ops {
            if d.class == RegClass::D {
                return Some((|| {
                    let imm8 = byte_mask_imm8(*v as u64)
                        .context("movi Dd immediate must be a per-byte 0x00/0xFF mask")?;
                    Ok(word(0, 1, 0b1110, imm8, d.num as u32))
                })());
            }
        }
    }

    // Vector form: `Vd.<arr>, #imm{, lsl/msl #n}`.
    let [VReg { num: d, arr }, immop] = ops else { return None };
    // Pull the immediate value + shift kind/amount; bail to None if op[1] isn't
    // an immediate (so register bic/orr go to the logical encoder).
    let (val, lsl, msl): (i64, Option<u32>, Option<u32>) = match immop {
        Imm(v) => (*v, None, None),
        ImmShift(v, s) => (*v, Some(*s), None),
        ImmMsl(v, s) => (*v, None, Some(*s)),
        _ => return None,
    };
    let d = *d as u32;
    let (q, esize) = arr_qsize(*arr);

    Some((|| {
        // .2d movi: 64-bit byte-mask immediate.
        if esize == 3 {
            if mnem != "movi" {
                bail!("only movi supports the .2d immediate form");
            }
            let imm8 = byte_mask_imm8(val as u64)
                .context("movi .2d immediate must be a per-byte 0x00/0xFF mask")?;
            return Ok(word(q, 1, 0b1110, imm8, d));
        }
        let imm8 = (val as u64 & 0xFF) as u8;
        let lsl_x = |amt: u32| -> Result<u32> {
            match amt {
                0 => Ok(0),
                8 => Ok(1),
                16 => Ok(2),
                24 => Ok(3),
                _ => bail!("invalid lsl #{amt}"),
            }
        };
        let (op, cmode) = match (mnem, esize) {
            // .8b/.16b: movi only, no shift.
            ("movi", 0) => {
                if lsl.is_some() || msl.is_some() {
                    bail!("movi .8b/.16b takes no shift");
                }
                (0, 0b1110)
            }
            // .4h/.8h: movi/mvni cmode 100x; bic/orr cmode 10x1.
            ("movi" | "mvni", 1) => {
                let x = lsl_x(lsl.unwrap_or(0))?;
                if x > 1 || msl.is_some() {
                    bail!("16-bit immediate shift must be lsl #0 or #8");
                }
                ((mnem == "mvni") as u32, 0b1000 | (x << 1))
            }
            ("bic" | "orr", 1) => {
                let x = lsl_x(lsl.unwrap_or(0))?;
                if x > 1 || msl.is_some() {
                    bail!("16-bit immediate shift must be lsl #0 or #8");
                }
                ((mnem == "bic") as u32, 0b1001 | (x << 1))
            }
            // .2s/.4s: movi/mvni lsl cmode 0xx0 or msl cmode 110x; bic/orr lsl 0xx1.
            ("movi" | "mvni", 2) => {
                let op = (mnem == "mvni") as u32;
                if let Some(m) = msl {
                    let b = match m {
                        8 => 0,
                        16 => 1,
                        _ => bail!("msl must be #8 or #16"),
                    };
                    (op, 0b1100 | b)
                } else {
                    (op, lsl_x(lsl.unwrap_or(0))? << 1)
                }
            }
            ("bic" | "orr", 2) => {
                if msl.is_some() {
                    bail!("bic/orr immediate has no msl form");
                }
                ((mnem == "bic") as u32, 0b0001 | (lsl_x(lsl.unwrap_or(0))? << 1))
            }
            _ => bail!("unsupported {mnem} immediate arrangement"),
        };
        Ok(word(q, op, cmode, imm8, d))
    })())
}

/// NEON dispatcher — matches only vector operands, so GP/FP-scalar forms of
/// shared mnemonics (`add`, `neg`, `fadd`, …) fall through.
fn try_simd(mnem: &str, ops: &[Operand]) -> Option<Result<Encoded>> {
    use Operand::{Reg, VElem, VReg};

    // ── 3-same: Vd, Vn, Vm (same arrangement) ───────────────────────────────
    // The arrangement-match check is done *inside* each arm so that
    // differing-arrangement ops (widen/narrow) fall through to their own block.
    if let [VReg { num: d, arr: ad }, VReg { num: n, arr: an }, VReg { num: m, arr: am }] = ops {
        let (d, n, m) = (*d as u32, *n as u32, *m as u32);
        let same = ad == an && ad == am;
        let mismatch = || Some(Err(anyhow::anyhow!("vector arrangement mismatch")));
        if let Some(base) = simd3_int_base(mnem) {
            if !same {
                return mismatch();
            }
            let (q, sz) = arr_qsize(*ad);
            return Some(Ok(Encoded::word(base | (q << 30) | (sz << 22) | (m << 16) | (n << 5) | d)));
        }
        if let Some(base) = simd3_logical_base(mnem) {
            if !same {
                return mismatch();
            }
            let q = match ad {
                Arr::B16 => 1,
                Arr::B8 => 0,
                _ => return Some(Err(anyhow::anyhow!("vector logical needs .8b/.16b"))),
            };
            return Some(Ok(Encoded::word(base | (q << 30) | (m << 16) | (n << 5) | d)));
        }
        if let Some(base) = simd3_fp_base(mnem) {
            if !same {
                return mismatch();
            }
            let Some((q, sz)) = arr_fp_qsz(*ad) else {
                return Some(Err(anyhow::anyhow!("fp vector needs .2s/.4s/.2d")));
            };
            return Some(Ok(Encoded::word(base | (q << 30) | (sz << 22) | (m << 16) | (n << 5) | d)));
        }
        if let Some(base) = simd_permute_base(mnem) {
            if !same {
                return mismatch();
            }
            let (q, sz) = arr_qsize(*ad);
            return Some(Ok(Encoded::word(base | (q << 30) | (sz << 22) | (m << 16) | (n << 5) | d)));
        }
    }

    // ── by-element: Vd.<arr>, Vn.<arr>, Vm.<ts>[index] ──────────────────────
    {
        let base_mnem = mnem.strip_suffix('2').unwrap_or(mnem);
        if let Some(base) = simd_byelem(base_mnem) {
            if let [VReg { num: d, arr: _ }, VReg { num: n, arr: an }, VElem { num: m, esize, index }] = ops {
                return Some((|| {
                    let q = arr_qsize(*an).0;
                    let (size, idx_bits, rm) = byelem_index(*esize, *m, *index)?;
                    Ok(Encoded::word(
                        base | (q << 30) | (size << 22) | idx_bits | (rm << 16) | ((*n as u32) << 5) | *d as u32,
                    ))
                })());
            }
        }
    }

    // ── structured load/store: ld1–ld4 / st1–st4 {list}, [Xn]{, post} ───────
    if is_struct_ldst(mnem) {
        // [VList, Mem]  or  [VList, Mem, Reg(rm)] (register post-index).
        let (list, mem, rm_post) = match ops {
            [l @ Operand::VList { .. }, Operand::Mem(m)] => (l, m, None),
            [l @ Operand::VList { .. }, Operand::Mem(m), Reg(rm)] => (l, m, Some(*rm)),
            _ => return Some(Err(anyhow::anyhow!("structured ld/st expects `{{list}}, [Xn]{{, post}}`"))),
        };
        let Operand::VList { first, count, arr } = list else { unreachable!() };
        return Some((|| {
            let opcode = struct_opcode(mnem, *count)?;
            let load = mnem.starts_with("ld");
            let (q, size) = arr_qsize(*arr);
            let mut w = 0x0C00_0000
                | (q << 30)
                | ((load as u32) << 22)
                | (opcode << 12)
                | (size << 10)
                | (rn(mem.base) << 5)
                | *first as u32;
            // Post-index: bit23, with Rm = the index reg, or 31 for the `#imm` form.
            match (&mem.addr, rm_post) {
                (Addr::Base, None) => {}
                (Addr::Base, Some(rm)) => w |= (1 << 23) | (rn(rm) << 16),
                (Addr::PostIndex(_), None) => w |= (1 << 23) | (31 << 16),
                _ => bail!("structured ld/st: only `[Xn]`, `[Xn], #imm`, or `[Xn], Xm`"),
            }
            Ok(Encoded::word(w))
        })());
    }

    // ── tbl / tbx: Vd.<8b|16b>, {Vt.16b, …}, Vm.<8b|16b> ────────────────────
    if mnem == "tbl" || mnem == "tbx" {
        if let [VReg { num: d, arr: ad }, Operand::VList { first, count, arr: _ }, VReg { num: m, arr: am }] = ops {
            return Some((|| {
                if ad != am {
                    bail!("tbl/tbx Vd and Vm arrangement must match");
                }
                let q = match ad {
                    Arr::B16 => 1,
                    Arr::B8 => 0,
                    _ => bail!("tbl/tbx needs .8b/.16b"),
                };
                if *count < 1 || *count > 4 {
                    bail!("tbl/tbx table must be 1–4 registers");
                }
                let len = (*count - 1) as u32;
                let base = if mnem == "tbx" { 0x0E00_1000 } else { 0x0E00_0000 };
                Ok(Encoded::word(
                    base | (q << 30) | ((*m as u32) << 16) | (len << 13) | ((*first as u32) << 5) | *d as u32,
                ))
            })());
        }
    }

    // ── ext: Vd, Vn, Vm, #index (byte rotate; .8b/.16b) ─────────────────────
    if mnem == "ext" {
        if let [VReg { num: d, arr: ad }, VReg { num: n, arr: an }, VReg { num: m, arr: am }, Operand::Imm(idx)] = ops {
            if ad != an || ad != am {
                return Some(Err(anyhow::anyhow!("ext arrangement mismatch")));
            }
            let q = match ad {
                Arr::B16 => 1,
                Arr::B8 => 0,
                _ => return Some(Err(anyhow::anyhow!("ext needs .8b/.16b"))),
            };
            let w = 0x2E00_0000 | (q << 30) | ((*m as u32) << 16) | (((*idx as u32) & 0xF) << 11) | ((*n as u32) << 5) | *d as u32;
            return Some(Ok(Encoded::word(w)));
        }
    }

    // ── copy: ins / umov / smov ─────────────────────────────────────────────
    if mnem == "ins" || mnem == "mov" {
        // ins Vd.<ts>[i], Vn.<ts>[j]  (element ← element)
        if let [VElem { num: d, esize: ed, index: id }, VElem { num: n, esize: en, index: jn }] = ops {
            if ed != en {
                return Some(Err(anyhow::anyhow!("ins element size mismatch")));
            }
            let sz = elem_size(*ed);
            let imm5 = ((*id as u32) << (sz + 1)) | (1 << sz);
            let imm4 = (*jn as u32) << sz;
            let w = 0x6E00_0400 | (imm5 << 16) | (imm4 << 11) | ((*n as u32) << 5) | *d as u32;
            return Some(Ok(Encoded::word(w)));
        }
        // ins Vd.<ts>[i], Wn/Xn  (element ← GP)
        if let [VElem { num: d, esize, index }, Operand::Reg(g)] = ops {
            if !matches!(g.class, RegClass::X | RegClass::W) {
                return Some(Err(anyhow::anyhow!("ins source must be a GP register")));
            }
            let sz = elem_size(*esize);
            let imm5 = ((*index as u32) << (sz + 1)) | (1 << sz);
            let w = 0x4E00_1C00 | (imm5 << 16) | ((g.num as u32) << 5) | *d as u32;
            return Some(Ok(Encoded::word(w)));
        }
    }
    if mnem == "umov" || mnem == "smov" || mnem == "mov" {
        if let [Operand::Reg(g), VElem { num: n, esize, index }] = ops {
            let to_x = match g.class {
                RegClass::X => 1u32,
                RegClass::W => 0,
                _ => return Some(Err(anyhow::anyhow!("umov/smov destination must be a GP register"))),
            };
            let sz = elem_size(*esize);
            let imm5 = ((*index as u32) << (sz + 1)) | (1 << sz);
            let base = if mnem == "smov" { 0x0E00_2C00 } else { 0x0E00_3C00 };
            let w = base | (to_x << 30) | (imm5 << 16) | ((*n as u32) << 5) | g.num as u32;
            return Some(Ok(Encoded::word(w)));
        }
    }

    // ── shift by immediate: Vd, Vn, #shift (same arrangement) ───────────────
    if let Some((base, left)) = simd_shift_imm(mnem) {
        if let [VReg { num: d, arr: ad }, VReg { num: n, arr: an }, Operand::Imm(sh)] = ops {
            if ad != an {
                return Some(Err(anyhow::anyhow!("shift arrangement mismatch")));
            }
            let (q, _) = arr_qsize(*ad);
            let ebits = 8u32 << arr_esize(*ad);
            let sh = *sh as u32;
            let immhb = if left {
                if sh >= ebits {
                    return Some(Err(anyhow::anyhow!("left shift #{sh} out of range")));
                }
                ebits + sh
            } else {
                if sh < 1 || sh > ebits {
                    return Some(Err(anyhow::anyhow!("right shift #{sh} out of range")));
                }
                2 * ebits - sh
            };
            return Some(Ok(Encoded::word(base | (q << 30) | (immhb << 16) | ((*n as u32) << 5) | *d as u32)));
        }
    }

    // ── 2-misc FP compare against zero: Vd, Vn, #0.0 ────────────────────────
    if let [VReg { num: d, arr: ad }, VReg { num: n, arr: an }, Operand::FpImm(z)] = ops {
        if let Some(base) = simd2_fcmp_zero_base(mnem) {
            if ad != an {
                return Some(Err(anyhow::anyhow!("vector arrangement mismatch")));
            }
            if *z != 0.0 {
                return Some(Err(anyhow::anyhow!("vector fcmp immediate must be #0.0")));
            }
            let Some((q, sz)) = arr_fp_qsz(*ad) else {
                return Some(Err(anyhow::anyhow!("fp vector needs .2s/.4s/.2d")));
            };
            return Some(Ok(Encoded::word(base | (q << 30) | (sz << 22) | ((*n as u32) << 5) | *d as u32)));
        }
    }

    // ── 2-misc: Vd, Vn (same arrangement) ───────────────────────────────────
    // Mismatch check inside each arm so differing-arrangement 2-op narrowing
    // (xtn/sqxtn/…) falls through to the widen/narrow block.
    if let [VReg { num: d, arr: ad }, VReg { num: n, arr: an }] = ops {
        let (d, n) = (*d as u32, *n as u32);
        let same = ad == an;
        if let Some(base) = simd2_int_base(mnem) {
            if !same {
                return Some(Err(anyhow::anyhow!("vector arrangement mismatch")));
            }
            let (q, sz) = arr_qsize(*ad);
            return Some(Ok(Encoded::word(base | (q << 30) | (sz << 22) | (n << 5) | d)));
        }
        if let Some(base) = simd2_fp_base(mnem) {
            if !same {
                return Some(Err(anyhow::anyhow!("vector arrangement mismatch")));
            }
            let Some((q, sz)) = arr_fp_qsz(*ad) else {
                return Some(Err(anyhow::anyhow!("fp vector needs .2s/.4s/.2d")));
            };
            return Some(Ok(Encoded::word(base | (q << 30) | (sz << 22) | (n << 5) | d)));
        }
    }

    // ── widen / narrow / long (3-diff + 2-misc narrowing) ───────────────────
    // (Q,size) comes from the *narrow* operand's arrangement; the `2` suffix is
    // conveyed by that arrangement (.16b ⇒ Q=1), so we strip it.
    {
        let base_mnem = mnem.strip_suffix('2').unwrap_or(mnem);
        if let Some((base, from)) = simd_widen(base_mnem) {
            // 2-operand narrowing (xtn/sqxtn/…): Vd, Vn.
            if matches!(from, QFrom::Vd2) {
                if let [VReg { num: d, arr: ad }, VReg { num: n, arr: _an }] = ops {
                    let (q, sz) = arr_qsize(*ad);
                    return Some(Ok(Encoded::word(base | (q << 30) | (sz << 22) | ((*n as u32) << 5) | *d as u32)));
                }
            } else if let [VReg { num: d, arr: ad }, VReg { num: n, arr: an }, VReg { num: m, arr: am }] = ops {
                let src = match from {
                    QFrom::Vn => *an,
                    QFrom::Vm => *am,
                    _ => *ad,
                };
                let (q, sz) = arr_qsize(src);
                let w = base | (q << 30) | (sz << 22) | ((*m as u32) << 16) | ((*n as u32) << 5) | *d as u32;
                return Some(Ok(Encoded::word(w)));
            }
        }
    }

    // ── fcvtn/fcvtxn (narrow) + fcvtl (long) ────────────────────────────────
    {
        let base_mnem = mnem.strip_suffix('2').unwrap_or(mnem);
        let narrow = match base_mnem {
            "fcvtn" => Some(0x0E21_6800u32),
            "fcvtxn" => Some(0x2E21_6800),
            _ => None,
        };
        if let Some(base) = narrow {
            if let [VReg { num: d, arr: ad }, VReg { num: n, arr: an }] = ops {
                // Q from the narrow destination, sz from the wide source.
                let q = arr_qsize(*ad).0;
                let sz = arr_esize(*an).wrapping_sub(2);
                return Some(Ok(Encoded::word(base | (q << 30) | (sz << 22) | ((*n as u32) << 5) | *d as u32)));
            }
        }
        if base_mnem == "fcvtl" {
            if let [VReg { num: d, arr: ad }, VReg { num: n, arr: an }] = ops {
                // Q from the narrow source, sz from the wide destination.
                let q = arr_qsize(*an).0;
                let sz = arr_esize(*ad).wrapping_sub(2);
                return Some(Ok(Encoded::word(0x0E21_7800 | (q << 30) | (sz << 22) | ((*n as u32) << 5) | *d as u32)));
            }
        }
    }

    // ── across-lane reduction: scalar Vd, Vn.<arr> ──────────────────────────
    if let [Reg(d), VReg { num: n, arr: an }] = ops {
        if let Some(base) = simd_across_base(mnem) {
            let (q, sz) = arr_qsize(*an);
            return Some(Ok(Encoded::word(base | (q << 30) | (sz << 22) | ((*n as u32) << 5) | d.num as u32)));
        }
    }

    // ── dup: from a vector element, or from a GP register ────────────────────
    if mnem == "dup" {
        if let [VReg { num: d, arr: ad }, VElem { num: n, esize, index }] = ops {
            let sz = elem_size(*esize);
            let imm5 = ((*index as u32) << (sz + 1)) | (1 << sz);
            let q = arr_qsize(*ad).0;
            return Some(Ok(Encoded::word(
                (q << 30) | 0x0E00_0400 | (imm5 << 16) | ((*n as u32) << 5) | *d as u32,
            )));
        }
        if let [VReg { num: d, arr: ad }, Reg(g)] = ops {
            if !matches!(g.class, RegClass::X | RegClass::W) {
                return Some(Err(anyhow::anyhow!("dup source must be a GP register or vector element")));
            }
            let sz = arr_esize(*ad);
            let imm5 = 1u32 << sz;
            let q = arr_qsize(*ad).0;
            return Some(Ok(Encoded::word(
                (q << 30) | 0x0E00_0C00 | (imm5 << 16) | ((g.num as u32) << 5) | *d as u32,
            )));
        }
    }

    None
}

/// The register number of any register-ish operand (GP, FP-scalar, or vector).
fn opnum(op: &Operand) -> Option<u32> {
    match op {
        Operand::Reg(r) => Some(r.num as u32),
        Operand::VReg { num, .. } | Operand::VElem { num, .. } => Some(*num as u32),
        _ => None,
    }
}

/// Crypto extensions: AES (2-op, `.16b`) and SHA1/SHA256 (2-op + 3-op). These
/// have no Q/size variation, so each is a fixed base + register numbers.
fn try_crypto(mnem: &str, ops: &[Operand]) -> Option<Result<Encoded>> {
    // AES: Vd.16b, Vn.16b.
    let aes = match mnem {
        "aese" => 0x4E28_4800u32,
        "aesd" => 0x4E28_5800,
        "aesmc" => 0x4E28_6800,
        "aesimc" => 0x4E28_7800,
        _ => 0,
    };
    // SHA 2-operand: sha1h Sd,Sn ; sha1su1/sha256su0 Vd.4s,Vn.4s.
    let sha2 = match mnem {
        "sha1h" => 0x5E28_0800u32,
        "sha1su1" => 0x5E28_1800,
        "sha256su0" => 0x5E28_2800,
        _ => 0,
    };
    if aes != 0 || sha2 != 0 {
        let base = if aes != 0 { aes } else { sha2 };
        let [a, b] = ops else {
            return Some(Err(anyhow::anyhow!("{mnem} expects two register operands")));
        };
        let (Some(d), Some(n)) = (opnum(a), opnum(b)) else {
            return Some(Err(anyhow::anyhow!("{mnem} expects registers")));
        };
        return Some(Ok(Encoded::word(base | (n << 5) | d)));
    }
    // SHA 3-operand: Rd, Rn, Rm (mixed q/s/v register classes).
    let sha3 = match mnem {
        "sha1c" => 0x5E00_0000u32,
        "sha1p" => 0x5E00_1000,
        "sha1m" => 0x5E00_2000,
        "sha1su0" => 0x5E00_3000,
        "sha256h" => 0x5E00_4000,
        "sha256h2" => 0x5E00_5000,
        "sha256su1" => 0x5E00_6000,
        _ => 0,
    };
    if sha3 != 0 {
        let [a, b, c] = ops else {
            return Some(Err(anyhow::anyhow!("{mnem} expects three register operands")));
        };
        let (Some(d), Some(n), Some(m)) = (opnum(a), opnum(b), opnum(c)) else {
            return Some(Err(anyhow::anyhow!("{mnem} expects registers")));
        };
        return Some(Ok(Encoded::word(sha3 | (m << 16) | (n << 5) | d)));
    }
    None
}

/// FP type field for a scalar FP register: S=00, D=01, H=11.
fn ftype_of(r: Reg) -> Option<u32> {
    match r.class {
        RegClass::S => Some(0b00),
        RegClass::D => Some(0b01),
        RegClass::H => Some(0b11),
        _ => None,
    }
}

/// Scalar floating-point data-processing (1/2/3-source, fcvt, fmov reg).
/// Returns `None` when `mnemonic`/operands aren't a scalar-FP form handled here.
fn try_fp(mnem: &str, ops: &[Operand]) -> Option<Result<Encoded>> {
    use Operand::Reg;

    // 2-source: base 0x1E200800 | ftype<<22 | opcode<<12 | Rm<<16 | Rn<<5 | Rd.
    let two = |opc: u32| -> Option<Result<Encoded>> {
        let [Reg(d), Reg(n), Reg(m)] = ops else { return None };
        let ft = ftype_of(*d)?;
        if d.class != n.class || d.class != m.class {
            return Some(Err(anyhow::anyhow!("fp operand type mismatch")));
        }
        Some(Ok(Encoded::word(
            0x1E20_0800 | (ft << 22) | (opc << 12) | (rn(*m) << 16) | (rn(*n) << 5) | rn(*d),
        )))
    };
    match mnem {
        "fmul" => return two(0),
        "fdiv" => return two(1),
        "fadd" => return two(2),
        "fsub" => return two(3),
        "fmax" => return two(4),
        "fmin" => return two(5),
        "fmaxnm" => return two(6),
        "fminnm" => return two(7),
        "fnmul" => return two(8),
        _ => {}
    }

    // 1-source: per-mnemonic S-variant base | ftype<<22 | Rn<<5 | Rd.
    let one = |base_s: u32| -> Option<Result<Encoded>> {
        let [Reg(d), Reg(n)] = ops else { return None };
        let ft = ftype_of(*d)?;
        if d.class != n.class {
            return Some(Err(anyhow::anyhow!("fp operand type mismatch")));
        }
        Some(Ok(Encoded::word(base_s | (ft << 22) | (rn(*n) << 5) | rn(*d))))
    };
    match mnem {
        "fabs" => return one(0x1E20_C000),
        "fneg" => return one(0x1E21_4000),
        "fsqrt" => return one(0x1E21_C000),
        "frintn" => return one(0x1E24_4000),
        "frintp" => return one(0x1E24_C000),
        "frintm" => return one(0x1E25_4000),
        "frintz" => return one(0x1E25_C000),
        "frinta" => return one(0x1E26_4000),
        "frintx" => return one(0x1E27_4000),
        "frinti" => return one(0x1E27_C000),
        _ => {}
    }

    // fcvt (between H/S/D): base 0x1E224000 | src_ftype<<22 | dst_opc<<15.
    if mnem == "fcvt" {
        if let [Reg(d), Reg(n)] = ops {
            let src = ftype_of(*n)?;
            let Some(dst) = ftype_of(*d) else {
                return Some(Err(anyhow::anyhow!("fcvt destination not S/D/H")));
            };
            return Some(Ok(Encoded::word(
                0x1E22_4000 | (src << 22) | (dst << 15) | (rn(*n) << 5) | rn(*d),
            )));
        }
    }

    // 3-source: per-mnemonic S base | ftype<<22 | Rm<<16 | Ra<<10 | Rn<<5 | Rd.
    let three = |base_s: u32| -> Option<Result<Encoded>> {
        let [Reg(d), Reg(n), Reg(m), Reg(a)] = ops else { return None };
        let ft = ftype_of(*d)?;
        Some(Ok(Encoded::word(
            base_s | (ft << 22) | (rn(*m) << 16) | (rn(*a) << 10) | (rn(*n) << 5) | rn(*d),
        )))
    };
    match mnem {
        "fmadd" => return three(0x1F00_0000),
        "fmsub" => return three(0x1F00_8000),
        "fnmadd" => return three(0x1F20_0000),
        "fnmsub" => return three(0x1F20_8000),
        _ => {}
    }

    // ── compare: fcmp / fcmpe (reg + #0.0) ──────────────────────────────────
    if mnem == "fcmp" || mnem == "fcmpe" {
        let e = if mnem == "fcmpe" { 0x10 } else { 0 };
        return Some((|| {
            match ops {
                [Reg(n), Reg(m)] => {
                    let ft = ftype_of(*n).context("fcmp needs FP operands")?;
                    Ok(Encoded::word(0x1E20_2000 | (ft << 22) | (rn(*m) << 16) | (rn(*n) << 5) | e))
                }
                [Reg(n), Operand::FpImm(z)] if *z == 0.0 => {
                    let ft = ftype_of(*n).context("fcmp needs FP operands")?;
                    Ok(Encoded::word(0x1E20_2000 | (ft << 22) | (rn(*n) << 5) | e | 0x08))
                }
                _ => bail!("fcmp: expected `Sn, Sm` or `Sn, #0.0`"),
            }
        })());
    }

    // ── conditional compare: fccmp / fccmpe ─────────────────────────────────
    if mnem == "fccmp" || mnem == "fccmpe" {
        let e = if mnem == "fccmpe" { 0x10 } else { 0 };
        if let [Reg(n), Reg(m), Operand::Imm(nzcv), Operand::Sym(c)] = ops {
            return Some((|| {
                let ft = ftype_of(*n).context("fccmp needs FP operands")?;
                let cond = cond_code(c).context("fccmp condition")?;
                Ok(Encoded::word(
                    0x1E20_0400 | (ft << 22) | (rn(*m) << 16) | (cond << 12) | (rn(*n) << 5) | ((*nzcv as u32) & 0xF) | e,
                ))
            })());
        }
        return Some(Err(anyhow::anyhow!("fccmp: expected `Sn, Sm, #nzcv, cond`")));
    }

    // ── conditional select: fcsel ───────────────────────────────────────────
    if mnem == "fcsel" {
        if let [Reg(d), Reg(n), Reg(m), Operand::Sym(c)] = ops {
            return Some((|| {
                let ft = ftype_of(*d).context("fcsel needs FP operands")?;
                let cond = cond_code(c).context("fcsel condition")?;
                Ok(Encoded::word(0x1E20_0C00 | (ft << 22) | (rn(*m) << 16) | (cond << 12) | (rn(*n) << 5) | rn(*d)))
            })());
        }
        return Some(Err(anyhow::anyhow!("fcsel: expected `Sd, Sn, Sm, cond`")));
    }

    // ── convert FP↔int (fcvt*s/u, scvtf/ucvtf) — integer + fixed-point ───────
    if let Some((rmode, opcode)) = fcvt_int_fields(mnem) {
        // Fixed-point FP→int: [Reg(gpr), Reg(fp), #fbits] (bit21=0, scale field).
        if let [Reg(d), Reg(n), Operand::Imm(fbits)] = ops {
            return Some((|| {
                let sf = sf_of(*d).context("fcvt* destination must be Wn/Xn")?;
                let ft = ftype_of(*n).context("fcvt* source must be a FP register")?;
                let scale = fbits_to_scale(*fbits, sf)?;
                Ok(Encoded::word(
                    (sf << 31) | 0x1E00_0000 | (ft << 22) | (rmode << 19) | (opcode << 16) | (scale << 10) | (rn(*n) << 5) | rn(*d),
                ))
            })());
        }
        // Integer FP→int: [Reg(gpr_dst), Reg(fp_src)].
        if let [Reg(d), Reg(n)] = ops {
            return Some((|| {
                let sf = sf_of(*d).context("fcvt* destination must be Wn/Xn")?;
                let ft = ftype_of(*n).context("fcvt* source must be a FP register")?;
                Ok(Encoded::word(
                    (sf << 31) | 0x1E20_0000 | (ft << 22) | (rmode << 19) | (opcode << 16) | (rn(*n) << 5) | rn(*d),
                ))
            })());
        }
    }
    if mnem == "scvtf" || mnem == "ucvtf" {
        let opcode = if mnem == "scvtf" { 0b010 } else { 0b011 };
        // Fixed-point int→FP: [Reg(fp), Reg(gpr), #fbits].
        if let [Reg(d), Reg(n), Operand::Imm(fbits)] = ops {
            return Some((|| {
                let sf = sf_of(*n).context("scvtf/ucvtf source must be Wn/Xn")?;
                let ft = ftype_of(*d).context("scvtf/ucvtf destination must be a FP register")?;
                let scale = fbits_to_scale(*fbits, sf)?;
                Ok(Encoded::word(
                    (sf << 31) | 0x1E00_0000 | (ft << 22) | (opcode << 16) | (scale << 10) | (rn(*n) << 5) | rn(*d),
                ))
            })());
        }
        // Integer int→FP: [Reg(fp_dst), Reg(gpr_src)].
        if let [Reg(d), Reg(n)] = ops {
            return Some((|| {
                let sf = sf_of(*n).context("scvtf/ucvtf source must be Wn/Xn")?;
                let ft = ftype_of(*d).context("scvtf/ucvtf destination must be a FP register")?;
                Ok(Encoded::word(
                    (sf << 31) | 0x1E20_0000 | (ft << 22) | (opcode << 16) | (rn(*n) << 5) | rn(*d),
                ))
            })());
        }
    }

    // ── fmov: reg-reg (same FP type), gpr↔fp, and #imm ──────────────────────
    if mnem == "fmov" {
        return Some((|| {
            match ops {
                // reg-reg, same FP type → 1-source fmov.
                [Reg(d), Reg(n)] if ftype_of(*d).is_some() && d.class == n.class => {
                    let ft = ftype_of(*d).unwrap();
                    Ok(Encoded::word(0x1E20_4000 | (ft << 22) | (rn(*n) << 5) | rn(*d)))
                }
                // GP → FP (opcode 111).
                [Reg(d), Reg(n)] if ftype_of(*d).is_some() && matches!(n.class, RegClass::X | RegClass::W) => {
                    let ft = ftype_of(*d).unwrap();
                    let sf = sf_of(*n)?;
                    Ok(Encoded::word((sf << 31) | 0x1E20_0000 | (ft << 22) | (7 << 16) | (rn(*n) << 5) | rn(*d)))
                }
                // FP → GP (opcode 110).
                [Reg(d), Reg(n)] if matches!(d.class, RegClass::X | RegClass::W) && ftype_of(*n).is_some() => {
                    let ft = ftype_of(*n).unwrap();
                    let sf = sf_of(*d)?;
                    Ok(Encoded::word((sf << 31) | 0x1E20_0000 | (ft << 22) | (6 << 16) | (rn(*n) << 5) | rn(*d)))
                }
                // FP immediate.
                [Reg(d), Operand::FpImm(v)] if ftype_of(*d).is_some() => {
                    let ft = ftype_of(*d).unwrap();
                    let imm8 = fp_imm8(*v).with_context(|| format!("{v} is not an 8-bit FP immediate"))?;
                    Ok(Encoded::word(0x1E20_1000 | (ft << 22) | ((imm8 as u32) << 13) | rn(*d)))
                }
                _ => bail!("unsupported fmov form"),
            }
        })());
    }

    None
}

/// Fixed-point `scale` field (`64 - fbits`) for the FP↔fixed converts.
fn fbits_to_scale(fbits: i64, sf: u32) -> Result<u32> {
    let max = if sf == 1 { 64 } else { 32 };
    if !(1..=max).contains(&fbits) {
        bail!("fixed-point #fbits {fbits} out of range 1..={max}");
    }
    Ok((64 - fbits) as u32)
}

/// `(rmode, opcode)` for an FP→int convert mnemonic, else `None`.
fn fcvt_int_fields(mnem: &str) -> Option<(u32, u32)> {
    Some(match mnem {
        "fcvtns" => (0b00, 0b000),
        "fcvtnu" => (0b00, 0b001),
        "fcvtas" => (0b00, 0b100),
        "fcvtau" => (0b00, 0b101),
        "fcvtps" => (0b01, 0b000),
        "fcvtpu" => (0b01, 0b001),
        "fcvtms" => (0b10, 0b000),
        "fcvtmu" => (0b10, 0b001),
        "fcvtzs" => (0b11, 0b000),
        "fcvtzu" => (0b11, 0b001),
        _ => return None,
    })
}

/// Encode a float as the AArch64 8-bit FP immediate (VFPExpandImm inverse), or
/// `None` if not representable: value = ±2^e·(1+m/16), e∈[-3,4], m∈[0,15].
fn fp_imm8(x: f64) -> Option<u8> {
    if x == 0.0 || !x.is_finite() {
        return None;
    }
    let bits = x.to_bits();
    let sign = ((bits >> 63) & 1) as u8;
    let exp = (((bits >> 52) & 0x7FF) as i64) - 1023;
    let mant = bits & 0x000F_FFFF_FFFF_FFFF;
    if mant & 0x0000_FFFF_FFFF_FFFF != 0 {
        return None; // low 48 mantissa bits must be zero
    }
    if !(-3..=4).contains(&exp) {
        return None;
    }
    let m4 = (mant >> 48) as u8; // top 4 mantissa bits
    let exp_field = ((exp + 7) & 0x7) as u8; // == NOT(bit2(E)):bits1:0(E), E=exp+3
    Some((sign << 7) | (exp_field << 4) | m4)
}

/// Conditional select family (`csel`/`csinc`/`csinv`/`csneg` and their
/// `cset`/`csetm`/`cinc`/… aliases, which pass `Rn = Rm` and the inverted cond).
fn csel(d: Reg, n: Reg, m: Reg, cond: u32, kind: CSel) -> Result<Encoded> {
    let sf = sf_of(d)?;
    let base = match kind {
        CSel::Csel => if sf == 1 { 0x9A80_0000 } else { 0x1A80_0000 },
        CSel::Csinc => if sf == 1 { 0x9A80_0400 } else { 0x1A80_0400 },
        CSel::Csinv => if sf == 1 { 0xDA80_0000 } else { 0x5A80_0000 },
        CSel::Csneg => if sf == 1 { 0xDA80_0400 } else { 0x5A80_0400 },
    };
    Ok(Encoded::word(base | (rn(m) << 16) | (cond << 12) | (rn(n) << 5) | rn(d)))
}

/// `(size, opc, scale, V)` for a load/store mnemonic + register. `V=1` for
/// FP/SIMD scalar registers (`b/h/s/d/q`); the width then comes from the
/// register class, not a mnemonic suffix (`ldr q0` / `ldr d0` / …).
fn ldst_kind(mnem: &str, t: Reg) -> Result<(u32, u32, i64, u32)> {
    let load = mnem.starts_with("ld");
    let ld = if load { 0b01 } else { 0b00 };
    // FP/SIMD scalar (V=1) — plain ldr/str only.
    let fp = match t.class {
        RegClass::B => Some((0b00, ld, 1)),
        RegClass::H => Some((0b01, ld, 2)),
        RegClass::S => Some((0b10, ld, 4)),
        RegClass::D => Some((0b11, ld, 8)),
        RegClass::Q => Some((0b00, if load { 0b11 } else { 0b10 }, 16)),
        _ => None,
    };
    if let Some((size, opc, scale)) = fp {
        if !matches!(mnem, "ldr" | "str") {
            bail!("FP/SIMD load/store uses ldr/str (got `{mnem}`)");
        }
        return Ok((size, opc, scale, 1));
    }
    if t.class == RegClass::V {
        bail!("vector .arrangement load needs ld1/st1 (not implemented)");
    }
    let is_x = t.class == RegClass::X;
    let (size, opc, scale) = match mnem {
        "str" => (if is_x { 0b11 } else { 0b10 }, 0b00, if is_x { 8 } else { 4 }),
        "ldr" => (if is_x { 0b11 } else { 0b10 }, 0b01, if is_x { 8 } else { 4 }),
        "strb" => (0b00, 0b00, 1),
        "ldrb" => (0b00, 0b01, 1),
        "strh" => (0b01, 0b00, 2),
        "ldrh" => (0b01, 0b01, 2),
        "ldrsb" => (0b00, if is_x { 0b10 } else { 0b11 }, 1),
        "ldrsh" => (0b01, if is_x { 0b10 } else { 0b11 }, 2),
        "ldrsw" => (0b10, 0b10, 4), // 64-bit dst only
        _ => bail!("not a load/store mnemonic `{mnem}`"),
    };
    Ok((size, opc, scale, 0))
}

/// Scalar load/store dispatcher over all addressing modes.
fn ldst(mnem: &str, t: Reg, m: &Mem) -> Result<Encoded> {
    let (size, opc, scale, v) = ldst_kind(mnem, t)?;
    match &m.addr {
        Addr::Base => ldst_uoff(size, opc, scale, v, 0, t, m.base),
        Addr::Offset(off) => {
            // Prefer the unsigned scaled form; fall back to LDUR/STUR (unscaled
            // imm9) for negative or unaligned offsets — what LLVM does.
            if *off >= 0 && off % scale == 0 && off / scale <= 0xFFF {
                ldst_uoff(size, opc, scale, v, *off, t, m.base)
            } else {
                ldst_unscaled(size, opc, v, *off, t, m.base, 0b00)
            }
        }
        Addr::PreIndex(off) => ldst_unscaled(size, opc, v, *off, t, m.base, 0b11),
        Addr::PostIndex(off) => ldst_unscaled(size, opc, v, *off, t, m.base, 0b01),
        Addr::RegOffset { index, ext, amount, scaled } => {
            ldst_regoff(size, opc, v, t, m.base, *index, *ext, *amount, *scaled)
        }
    }
}

/// Unsigned scaled offset form (`[Rn, #imm]`, `imm` a non-negative multiple of
/// the access size).
fn ldst_uoff(size: u32, opc: u32, scale: i64, v: u32, off: i64, t: Reg, base: Reg) -> Result<Encoded> {
    if off < 0 || off % scale != 0 {
        bail!("unsigned-offset load/store needs a non-negative multiple of {scale}, got {off}");
    }
    let imm12 = (off / scale) as u32;
    if imm12 > 0xFFF {
        bail!("load/store offset {off} out of unsigned-offset range");
    }
    let w = (size << 30) | (0b111 << 27) | (v << 26) | (0b01 << 24) | (opc << 22) | (imm12 << 10) | (rn(base) << 5) | rn(t);
    Ok(Encoded::word(w))
}

/// Explicit `ldur`/`stur…` — always the unscaled imm9 form (`idx=00`). The
/// mnemonic maps to its scaled sibling's `(size, opc)` by dropping the `u`.
fn ldst_ur(mnem: &str, t: Reg, m: &Mem) -> Result<Encoded> {
    let scaled_name = mnem.replacen("ur", "r", 1);
    let (size, opc, _scale, v) = ldst_kind(&scaled_name, t)?;
    let off = match m.addr {
        Addr::Base => 0,
        Addr::Offset(o) => o,
        _ => bail!("ldur/stur takes a simple `[Rn, #imm]` offset"),
    };
    ldst_unscaled(size, opc, v, off, t, m.base, 0b00)
}

/// Unscaled imm9 forms: LDUR/STUR (`idx=00`), post-index (`01`), pre-index (`11`).
fn ldst_unscaled(size: u32, opc: u32, v: u32, off: i64, t: Reg, base: Reg, idx: u32) -> Result<Encoded> {
    if !(-256..=255).contains(&off) {
        bail!("unscaled load/store offset {off} out of imm9 range -256..=255");
    }
    let imm9 = (off as u32) & 0x1FF;
    let w = (size << 30) | (0b111 << 27) | (v << 26) | (opc << 22) | (imm9 << 12) | (idx << 10) | (rn(base) << 5) | rn(t);
    Ok(Encoded::word(w))
}

/// Register-offset form (`[Rn, Rm{, ext{ #amt}}]`).
fn ldst_regoff(
    size: u32,
    opc: u32,
    v: u32,
    t: Reg,
    base: Reg,
    index: Reg,
    ext: IndexExt,
    _amount: u32,
    scaled: bool,
) -> Result<Encoded> {
    let option = match ext {
        IndexExt::Lsl | IndexExt::Uxtx => 0b011,
        IndexExt::Uxtw => 0b010,
        IndexExt::Sxtw => 0b110,
        IndexExt::Sxtx => 0b111,
    };
    let s = scaled as u32; // S bit: shift by log2(access size) when set
    let w = (size << 30)
        | (0b111 << 27)
        | (v << 26)
        | (opc << 22)
        | (1 << 21)
        | (rn(index) << 16)
        | (option << 13)
        | (s << 12)
        | (0b10 << 10)
        | (rn(base) << 5)
        | rn(t);
    Ok(Encoded::word(w))
}

/// `ldp`/`stp` — load/store pair, signed imm7 (offset / pre / post).
fn ldst_pair(mnem: &str, t1: Reg, t2: Reg, m: &Mem) -> Result<Encoded> {
    let load = mnem == "ldp";
    if t1.class != t2.class {
        bail!("ldp/stp register width mismatch");
    }
    // GP: W→opc=00/scale4, X→opc=10/scale8, V=0.
    // FP: S→opc=00/scale4, D→opc=01/scale8, Q→opc=10/scale16, V=1.
    let (opc, scale, v): (u32, i64, u32) = match t1.class {
        RegClass::W => (0b00, 4, 0),
        RegClass::X => (0b10, 8, 0),
        RegClass::S => (0b00, 4, 1),
        RegClass::D => (0b01, 8, 1),
        RegClass::Q => (0b10, 16, 1),
        _ => bail!("ldp/stp unsupported register class"),
    };
    let (off, idx) = match &m.addr {
        Addr::Base => (0i64, 0b10u32),
        Addr::Offset(o) => (*o, 0b10),
        Addr::PreIndex(o) => (*o, 0b11),
        Addr::PostIndex(o) => (*o, 0b01),
        Addr::RegOffset { .. } => bail!("ldp/stp takes an immediate offset, not a register"),
    };
    if off % scale != 0 {
        bail!("ldp/stp offset {off} must be a multiple of {scale}");
    }
    let imm7 = off / scale;
    if !(-64..=63).contains(&imm7) {
        bail!("ldp/stp offset {off} out of imm7 range");
    }
    let imm7 = (imm7 as u32) & 0x7F;
    let w = (opc << 30)
        | (0b101 << 27)
        | (v << 26)
        | (idx << 23)
        | ((load as u32) << 22)
        | (imm7 << 15)
        | (rn(t2) << 10)
        | (rn(m.base) << 5)
        | rn(t1);
    Ok(Encoded::word(w))
}

fn compare_branch(r: Reg, target: &str, nz: bool) -> Result<Encoded> {
    let sf = sf_of(r)?;
    let op = if nz { 1u32 } else { 0u32 };
    // sf | 011010 | op | imm19<<5 | Rt ; imm19 filled by the driver.
    let w = (sf << 31) | (0b011010 << 25) | (op << 24) | rn(r);
    Ok(Encoded::word_fixup(w, FixupKind::Branch19, target))
}

// ── integer ISA tail: ccmp/ccmn, adc/sbc, mulh, crc32 ───────────────────────

fn try_intmisc(mnem: &str, ops: &[Operand]) -> Option<Result<Encoded>> {
    use Operand::{Imm, Reg, Sym};

    // Conditional compare (register + immediate).
    if mnem == "ccmp" || mnem == "ccmn" {
        let reg_base = |sf: u32| -> u32 {
            let op = if mnem == "ccmp" { 1 } else { 0 }; // ccmp subtracts (op=1)
            (sf << 31) | (op << 30) | (1 << 29) | (0b11010010 << 21)
        };
        return Some((|| {
            match ops {
                [Reg(n), Reg(m), Imm(nzcv), Sym(c)] => {
                    let sf = sf_of(*n)?;
                    let cond = cond_code(c).context("ccmp condition")?;
                    Ok(Encoded::word(reg_base(sf) | (rn(*m) << 16) | (cond << 12) | (rn(*n) << 5) | ((*nzcv as u32) & 0xF)))
                }
                [Reg(n), Imm(imm5), Imm(nzcv), Sym(c)] => {
                    let sf = sf_of(*n)?;
                    let cond = cond_code(c).context("ccmp condition")?;
                    Ok(Encoded::word(
                        reg_base(sf) | (1 << 11) | (((*imm5 as u32) & 0x1F) << 16) | (cond << 12) | (rn(*n) << 5) | ((*nzcv as u32) & 0xF),
                    ))
                }
                _ => bail!("ccmp/ccmn: expected `Rn, Rm|#imm, #nzcv, cond`"),
            }
        })());
    }

    // Add/subtract with carry (and ngc/ngcs aliases).
    let adc_base = |sf: u32, op: u32, s: u32| (sf << 31) | (op << 30) | (s << 29) | (0b11010000 << 21);
    let carry = match mnem {
        "adc" => Some((0u32, 0u32)),
        "adcs" => Some((0, 1)),
        "sbc" => Some((1, 0)),
        "sbcs" => Some((1, 1)),
        _ => None,
    };
    if let Some((op, s)) = carry {
        if let [Reg(d), Reg(n), Reg(m)] = ops {
            return Some((|| {
                let sf = sf_of(*d)?;
                Ok(Encoded::word(adc_base(sf, op, s) | (rn(*m) << 16) | (rn(*n) << 5) | rn(*d)))
            })());
        }
    }
    if mnem == "ngc" || mnem == "ngcs" {
        let s = if mnem == "ngcs" { 1 } else { 0 };
        if let [Reg(d), Reg(m)] = ops {
            return Some((|| {
                let sf = sf_of(*d)?;
                Ok(Encoded::word(adc_base(sf, 1, s) | (rn(*m) << 16) | (31 << 5) | rn(*d)))
            })());
        }
    }

    // High multiply.
    if mnem == "smulh" || mnem == "umulh" {
        let base: u32 = if mnem == "smulh" { 0x9B40_7C00 } else { 0x9BC0_7C00 };
        if let [Reg(d), Reg(n), Reg(m)] = ops {
            return Some(Ok(Encoded::word(base | (rn(*m) << 16) | (rn(*n) << 5) | rn(*d))));
        }
    }

    // CRC32 / CRC32C.
    if let Some(base) = crc_base(mnem) {
        if let [Reg(d), Reg(n), Reg(m)] = ops {
            return Some(Ok(Encoded::word(base | (rn(*m) << 16) | (rn(*n) << 5) | rn(*d))));
        }
    }

    None
}

fn crc_base(m: &str) -> Option<u32> {
    Some(match m {
        "crc32b" => 0x1AC0_4000,
        "crc32h" => 0x1AC0_4400,
        "crc32w" => 0x1AC0_4800,
        "crc32x" => 0x9AC0_4C00,
        "crc32cb" => 0x1AC0_5000,
        "crc32ch" => 0x1AC0_5400,
        "crc32cw" => 0x1AC0_5800,
        "crc32cx" => 0x9AC0_5C00,
        _ => return None,
    })
}

// ── system: barriers, hints, exceptions, mrs/msr ────────────────────────────

fn try_system(mnem: &str, ops: &[Operand]) -> Option<Result<Encoded>> {
    use Operand::{Imm, Reg, Sym};

    // Hints with dedicated mnemonics.
    let hint = match mnem {
        "nop" => Some(0u32),
        "yield" => Some(1),
        "wfe" => Some(2),
        "wfi" => Some(3),
        "sev" => Some(4),
        "sevl" => Some(5),
        _ => None,
    };
    if let Some(h) = hint {
        if ops.is_empty() {
            return Some(Ok(Encoded::word(0xD503_201F | (h << 5))));
        }
    }
    if mnem == "hint" {
        if let [Imm(n)] = ops {
            return Some(Ok(Encoded::word(0xD503_201F | (((*n as u32) & 0x7F) << 5))));
        }
    }

    // Exception generation.
    let exc = match mnem {
        "svc" => Some(0xD400_0001u32),
        "hvc" => Some(0xD400_0002),
        "smc" => Some(0xD400_0003),
        "brk" => Some(0xD420_0000),
        "hlt" => Some(0xD440_0000),
        _ => None,
    };
    if let Some(base) = exc {
        if let [Imm(n)] = ops {
            return Some(Ok(Encoded::word(base | (((*n as u32) & 0xFFFF) << 5))));
        }
    }

    // Barriers: dmb/dsb (with option), isb (default sy).
    if mnem == "dmb" || mnem == "dsb" || mnem == "isb" {
        let base = match mnem {
            "dmb" => 0xD503_30BFu32, // opc=101
            "dsb" => 0xD503_309F,    // opc=100
            _ => 0xD503_30DF,        // isb, opc=110
        };
        let crm = match ops {
            [] => 0xF, // default sy
            [Sym(o)] => match barrier_option(o) {
                Some(v) => v,
                None => return Some(Err(anyhow::anyhow!("unknown barrier option `{o}`"))),
            },
            [Imm(v)] => (*v as u32) & 0xF,
            _ => return Some(Err(anyhow::anyhow!("bad barrier operand"))),
        };
        return Some(Ok(Encoded::word(base | (crm << 8))));
    }

    // PSTATE immediate: `msr <field>, #imm`.
    if mnem == "msr" {
        if let [Sym(field), Imm(imm)] = ops {
            if let Some((op1, op2)) = pstate_field(field) {
                let crm = (*imm as u32) & 0xF;
                return Some(Ok(Encoded::word(0xD500_401F | (op1 << 16) | (crm << 8) | (op2 << 5))));
            }
            // else fall through to the `msr SYSREG, Xt` form below.
        }
        if let [Sym(reg), Reg(t)] = ops {
            return Some((|| {
                let enc = sysreg(reg).with_context(|| format!("unknown system register `{reg}`"))?;
                Ok(Encoded::word(0xD510_0000 | (enc << 5) | rn(*t)))
            })());
        }
    }
    if mnem == "mrs" {
        if let [Reg(t), Sym(reg)] = ops {
            return Some((|| {
                let enc = sysreg(reg).with_context(|| format!("unknown system register `{reg}`"))?;
                Ok(Encoded::word(0xD530_0000 | (enc << 5) | rn(*t)))
            })());
        }
    }

    None
}

/// Barrier option name → 4-bit CRm.
fn barrier_option(o: &str) -> Option<u32> {
    Some(match o.to_ascii_lowercase().as_str() {
        "sy" => 15,
        "st" => 14,
        "ld" => 13,
        "ish" => 11,
        "ishst" => 10,
        "ishld" => 9,
        "nsh" => 7,
        "nshst" => 6,
        "nshld" => 5,
        "osh" => 3,
        "oshst" => 2,
        "oshld" => 1,
        _ => return None,
    })
}

/// PSTATE field for `msr <field>, #imm` → (op1, op2).
fn pstate_field(f: &str) -> Option<(u32, u32)> {
    Some(match f.to_ascii_lowercase().as_str() {
        "spsel" => (0, 5),
        "daifset" => (3, 6),
        "daifclr" => (3, 7),
        _ => return None,
    })
}

/// Common named system registers → 15-bit `op0:op1:CRn:CRm:op2` encoding.
fn sysreg(name: &str) -> Option<u32> {
    // enc = (op0<<14)|(op1<<11)|(CRn<<7)|(CRm<<3)|op2
    let pack = |op0: u32, op1: u32, crn: u32, crm: u32, op2: u32| (op0 << 14) | (op1 << 11) | (crn << 7) | (crm << 3) | op2;
    Some(match name.to_ascii_lowercase().as_str() {
        "nzcv" => pack(3, 3, 4, 2, 0),
        "fpcr" => pack(3, 3, 4, 4, 0),
        "fpsr" => pack(3, 3, 4, 4, 1),
        "tpidr_el0" => pack(3, 3, 13, 0, 2),
        "tpidrro_el0" => pack(3, 3, 13, 0, 3),
        "ctr_el0" => pack(3, 3, 0, 0, 1),
        "dczid_el0" => pack(3, 3, 0, 0, 7),
        "cntvct_el0" => pack(3, 3, 14, 0, 2),
        "cntfrq_el0" => pack(3, 3, 14, 0, 0),
        "midr_el1" => pack(3, 0, 0, 0, 0),
        "mpidr_el1" => pack(3, 0, 0, 0, 5),
        _ => return None,
    })
}

// ── atomics: exclusives + LSE ───────────────────────────────────────────────

fn try_atomics(mnem: &str, ops: &[Operand]) -> Option<Result<Encoded>> {
    use Operand::{Mem, Reg};

    // Load/store exclusive + acquire/release. [Rt, [Rn]] or [Rs, Rt, [Rn]].
    if let Some((base0, has_rs, size_opt)) = exclusive_base(mnem) {
        return Some((|| {
            // `size` (bits[31:30]) from the b/h suffix, else the register width.
            let rt = match ops {
                [Reg(t), Mem(_)] => *t,
                [Reg(_), Reg(t), Mem(_)] => *t,
                _ => bail!("bad exclusive operands"),
            };
            let size = match size_opt {
                Some(s) => s,
                None => match rt.class {
                    RegClass::X => 3,
                    RegClass::W => 2,
                    _ => bail!("exclusive needs a W/X register"),
                },
            };
            let base = base0 | (size << 30);
            match ops {
                [Reg(t), Mem(m)] if !has_rs => {
                    if !matches!(m.addr, Addr::Base) {
                        bail!("exclusive load/store takes `[Rn]` only");
                    }
                    Ok(Encoded::word(base | (rn(m.base) << 5) | rn(*t)))
                }
                [Reg(s), Reg(t), Mem(m)] if has_rs => {
                    if !matches!(m.addr, Addr::Base) {
                        bail!("exclusive store takes `[Rn]` only");
                    }
                    Ok(Encoded::word(base | (rn(*s) << 16) | (rn(m.base) << 5) | rn(*t)))
                }
                _ => bail!("bad exclusive operands"),
            }
        })());
    }

    // LSE: ld<op>{a}{l}{b|h}, st<op>… (alias), swp*, cas*.
    if let Some((size_opt, a, r, opc, kind)) = parse_lse(mnem) {
        return Some((|| {
            let [Reg(s), Reg(t), Mem(m)] = ops else {
                bail!("LSE atomic expects `Rs, Rt, [Rn]`");
            };
            if !matches!(m.addr, Addr::Base) {
                bail!("LSE atomic takes `[Rn]` only");
            }
            // No b/h suffix → size from the data register width (W=2, X=3).
            let size = match size_opt {
                Some(sz) => sz,
                None => match t.class {
                    RegClass::X => 3,
                    RegClass::W => 2,
                    _ => bail!("LSE atomic needs W/X registers"),
                },
            };
            let w = match kind {
                LseKind::LdOp => {
                    (size << 30) | 0x3820_0000 | (a << 23) | (r << 22) | (opc << 12) | (rn(*s) << 16) | (rn(m.base) << 5) | rn(*t)
                }
                LseKind::Swp => {
                    (size << 30) | 0x3820_8000 | (a << 23) | (r << 22) | (rn(*s) << 16) | (rn(m.base) << 5) | rn(*t)
                }
                LseKind::Cas => {
                    // size 00=0x08,01=0x48,10=0x88,11=0xC8 base; L=acquire(bit22), o0=release(bit15).
                    (size << 30) | 0x08A0_7C00 | (a << 22) | (r << 15) | (rn(*s) << 16) | (rn(m.base) << 5) | rn(*t)
                }
            };
            Ok(Encoded::word(w))
        })());
    }

    None
}

/// Exclusive base with `size` bits[31:30] cleared, whether it has an Rs operand,
/// and the fixed size (`Some(0/1)` for b/h, `None` to take it from the register).
fn exclusive_base(m: &str) -> Option<(u32, bool, Option<u32>)> {
    Some(match m {
        "ldxr" => (0x085F_7C00, false, None),
        "ldxrb" => (0x085F_7C00, false, Some(0)),
        "ldxrh" => (0x085F_7C00, false, Some(1)),
        "ldaxr" => (0x085F_FC00, false, None),
        "ldaxrb" => (0x085F_FC00, false, Some(0)),
        "ldaxrh" => (0x085F_FC00, false, Some(1)),
        "ldar" => (0x08DF_FC00, false, None),
        "ldarb" => (0x08DF_FC00, false, Some(0)),
        "ldarh" => (0x08DF_FC00, false, Some(1)),
        "stxr" => (0x0800_7C00, true, None),
        "stxrb" => (0x0800_7C00, true, Some(0)),
        "stxrh" => (0x0800_7C00, true, Some(1)),
        "stlxr" => (0x0800_FC00, true, None),
        "stlxrb" => (0x0800_FC00, true, Some(0)),
        "stlxrh" => (0x0800_FC00, true, Some(1)),
        "stlr" => (0x089F_FC00, false, None),
        "stlrb" => (0x089F_FC00, false, Some(0)),
        "stlrh" => (0x089F_FC00, false, Some(1)),
        _ => return None,
    })
}

#[derive(Clone, Copy)]
enum LseKind {
    LdOp,
    Swp,
    Cas,
}

/// Decompose an LSE mnemonic into `(size_opt, A, R, opc, kind)`. `size_opt` is
/// `Some(0)`/`Some(1)` for a `b`/`h` suffix, else `None` (resolve from the
/// register width at the call site).
fn parse_lse(m: &str) -> Option<(Option<u32>, u32, u32, u32, LseKind)> {
    let (body, size): (&str, Option<u32>) = if let Some(b) = m.strip_suffix('b') {
        (b, Some(0))
    } else if let Some(b) = m.strip_suffix('h') {
        (b, Some(1))
    } else {
        (m, None)
    };

    if let Some(rest) = body.strip_prefix("swp") {
        let (a, r) = lse_order(rest)?;
        return Some((size, a, r, 0, LseKind::Swp));
    }
    if let Some(rest) = body.strip_prefix("cas") {
        let (a, r) = lse_order(rest)?;
        return Some((size, a, r, 0, LseKind::Cas));
    }
    if let Some(rest) = body.strip_prefix("ld") {
        for (name, opc) in [
            ("add", 0u32), ("clr", 1), ("eor", 2), ("set", 3),
            ("smax", 4), ("smin", 5), ("umax", 6), ("umin", 7),
        ] {
            if let Some(order) = rest.strip_prefix(name) {
                let (a, r) = lse_order(order)?;
                return Some((size, a, r, opc, LseKind::LdOp));
            }
        }
    }
    None
}

/// LSE order suffix `""`/`"a"`/`"l"`/`"al"` → (A, R) bits.
fn lse_order(s: &str) -> Option<(u32, u32)> {
    Some(match s {
        "" => (0, 0),
        "a" => (1, 0),
        "l" => (0, 1),
        "al" => (1, 1),
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::parse::parse_line;
    use super::super::parse::Line;

    fn enc(asm: &str) -> Vec<u8> {
        let Line::Insn { mnemonic, ops } = parse_line(asm).unwrap() else { panic!("not an insn: {asm}") };
        encode(&mnemonic, &ops).unwrap_or_else(|e| panic!("encode `{asm}`: {e}")).bytes
    }

    /// Spot-check against the verified golden bytes (no LLVM needed).
    #[test]
    fn known_goldens() {
        assert_eq!(enc("ret"), [0xc0, 0x03, 0x5f, 0xd6]);
        assert_eq!(enc("nop"), [0x1f, 0x20, 0x03, 0xd5]);
        assert_eq!(enc("brk #0"), [0x00, 0x00, 0x20, 0xd4]);
        assert_eq!(enc("mov x0, x1"), [0xe0, 0x03, 0x01, 0xaa]);
        assert_eq!(enc("mov w0, w1"), [0xe0, 0x03, 0x01, 0x2a]);
        assert_eq!(enc("mvn x0, x1"), [0xe0, 0x03, 0x21, 0xaa]);
        assert_eq!(enc("movz x0, #0x1234"), [0x80, 0x46, 0x82, 0xd2]);
        assert_eq!(enc("movk x0, #0xabcd, lsl #32"), [0xa0, 0x79, 0xd5, 0xf2]);
        assert_eq!(enc("movn x0, #0"), [0x00, 0x00, 0x80, 0x92]);
        assert_eq!(enc("movn x0, #0xffff, lsl #16"), [0xe0, 0xff, 0xbf, 0x92]);
        assert_eq!(enc("add x0, x1, x2"), [0x20, 0x00, 0x02, 0x8b]);
        assert_eq!(enc("sub sp, sp, #16"), [0xff, 0x43, 0x00, 0xd1]);
        assert_eq!(enc("ldr x0, [x1, #8]"), [0x20, 0x04, 0x40, 0xf9]);
    }

    /// Offline goldens for the extended families (verified with llvm-mc).
    #[test]
    fn extended_goldens() {
        // aliases
        assert_eq!(enc("cmp x1, x2"), [0x3f, 0x00, 0x02, 0xeb]);
        assert_eq!(enc("cmp x1, #42"), [0x3f, 0xa8, 0x00, 0xf1]);
        assert_eq!(enc("tst x1, x2"), [0x3f, 0x00, 0x02, 0xea]);
        assert_eq!(enc("neg x0, x1"), [0xe0, 0x03, 0x01, 0xcb]);
        // extended-register add/sub for SP
        assert_eq!(enc("add x0, sp, x1"), [0xe0, 0x63, 0x21, 0x8b]);
        assert_eq!(enc("add x0, sp, x1, lsl #3"), [0xe0, 0x6f, 0x21, 0x8b]);
        // mul / div
        assert_eq!(enc("mul x0, x1, x2"), [0x20, 0x7c, 0x02, 0x9b]);
        assert_eq!(enc("madd x0, x1, x2, x3"), [0x20, 0x0c, 0x02, 0x9b]);
        assert_eq!(enc("smull x0, w1, w2"), [0x20, 0x7c, 0x22, 0x9b]);
        assert_eq!(enc("sdiv x0, x1, x2"), [0x20, 0x0c, 0xc2, 0x9a]);
        assert_eq!(enc("udiv x0, x1, x2"), [0x20, 0x08, 0xc2, 0x9a]);
        // shifts — variable + immediate
        assert_eq!(enc("lsl x0, x1, x2"), [0x20, 0x20, 0xc2, 0x9a]);
        assert_eq!(enc("lsl x0, x1, #3"), [0x20, 0xf0, 0x7d, 0xd3]);
        assert_eq!(enc("lsr x0, x1, #3"), [0x20, 0xfc, 0x43, 0xd3]);
        assert_eq!(enc("asr x0, x1, #3"), [0x20, 0xfc, 0x43, 0x93]);
        // bitfield / extends
        assert_eq!(enc("ubfx x0, x1, #4, #8"), [0x20, 0x2c, 0x44, 0xd3]);
        assert_eq!(enc("sxtw x0, w1"), [0x20, 0x7c, 0x40, 0x93]);
        assert_eq!(enc("uxtb w0, w1"), [0x20, 0x1c, 0x00, 0x53]);
        // bitmask-immediate logicals
        assert_eq!(enc("and x0, x1, #0xff"), [0x20, 0x1c, 0x40, 0x92]);
        assert_eq!(enc("orr x0, x1, #0xf"), [0x20, 0x0c, 0x40, 0xb2]);
        // mov #imm lowering: movz / movn / orr-bitmask
        assert_eq!(enc("mov x0, #0x1234"), [0x80, 0x46, 0x82, 0xd2]);
        assert_eq!(enc("mov x0, #-1"), [0x00, 0x00, 0x80, 0x92]);
        assert_eq!(enc("mov x0, #0x5555555555555555"), [0xe0, 0xf3, 0x00, 0xb2]);
        // conditional select
        assert_eq!(enc("csel x0, x1, x2, eq"), [0x20, 0x00, 0x82, 0x9a]);
        assert_eq!(enc("cset x0, eq"), [0xe0, 0x17, 0x9f, 0x9a]);
        assert_eq!(enc("csetm x0, lt"), [0xe0, 0xa3, 0x9f, 0xda]);
    }
}
