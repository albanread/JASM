//! Parse one line of AArch64 (LLVM/GAS) assembly into a [`Line`].
//!
//! AArch64 syntax is regular: `mnemonic op, op, op` with `#imm` immediates,
//! `[base, #off]` memory, and `, lsl #n` / `, sxtw #n` modifiers attached to the
//! preceding operand. There is no Intel-style `ptr` sizing — the register width
//! (x vs w) carries the size. This is the AArch64 sibling of `rasm/parse.rs`.

use anyhow::{bail, Context, Result};

/// Register bank. `Sp`/`Xzr` both encode field 31; which one #31 *means* is
/// positional (SP in `add`-imm/load base, XZR in shifted-register forms), so we
/// keep the bank and a `is_sp` marker the encoder consults when it matters.
///
/// `B`/`H`/`S`/`D`/`Q` are the scalar *views* of a SIMD&FP register (8/16/32/
/// 64/128-bit); `V` is the vector view (paired with an [`Arr`] arrangement in
/// [`Operand::VReg`]). All share one 0..=31 register file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RegClass {
    /// 64-bit general (`x0`–`x30`, `sp`/`xzr` = 31).
    X,
    /// 32-bit general (`w0`–`w30`, `wsp`/`wzr` = 31).
    W,
    /// Scalar FP/SIMD views.
    B,
    H,
    S,
    D,
    Q,
    /// Vector view (`v0`–`v31`); arrangement carried by [`Operand::VReg`].
    V,
}

/// SIMD vector arrangement (`Vn.<arr>`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Arr {
    B8,
    B16,
    H4,
    H8,
    S2,
    S4,
    D1,
    D2,
}

/// Element size for a lane-indexed vector operand (`Vn.<es>[i]`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ElemSize {
    B,
    H,
    S,
    D,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Reg {
    pub class: RegClass,
    /// 0..=31. For `sp`/`wsp` and `xzr`/`wzr` this is 31; `is_sp` disambiguates.
    pub num: u8,
    /// True when the token was `sp`/`wsp` (stack pointer), false for `xzr`/`wzr`
    /// or any numbered register.
    pub is_sp: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Shift {
    Lsl,
    Lsr,
    Asr,
    Ror,
}

/// Index extend/shift for a register-offset memory operand.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IndexExt {
    /// `lsl` (or bare `[xn, xm]`) — encodes option UXTX for a 64-bit index.
    Lsl,
    Uxtw,
    Sxtw,
    Sxtx,
    Uxtx,
}

/// Addressing form of a `[...]` memory operand.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Addr {
    /// `[xn]` — base only (offset 0, unsigned form).
    Base,
    /// `[xn, #imm]` — unsigned scaled / signed unscaled offset (encoder decides).
    Offset(i64),
    /// `[xn, #imm]!` — pre-index writeback.
    PreIndex(i64),
    /// `[xn], #imm` — post-index writeback.
    PostIndex(i64),
    /// `[xn, xm{, lsl #s}]` / `[xn, wm, (s|u)xtw{ #s}]` — register offset.
    /// `scaled` is the S bit (a shift amount was given).
    RegOffset { index: Reg, ext: IndexExt, amount: u32, scaled: bool },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Mem {
    pub base: Reg,
    pub addr: Addr,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Operand {
    Reg(Reg),
    /// `xm, <shift> #amount` (shifted register for ALU/logical).
    RegShift(Reg, Shift, u32),
    /// `#imm`.
    Imm(i64),
    /// `#imm, lsl #amount` (movz/movk/movn hw; add/sub `lsl #12`).
    ImmShift(i64, u32),
    /// `#imm, msl #amount` (movi/mvni masking-shift-left).
    ImmMsl(i64, u32),
    Mem(Mem),
    /// A bare symbol — branch/adr target.
    Sym(String),
    /// Vector register with arrangement: `v3.4s`.
    VReg { num: u8, arr: Arr },
    /// Lane-indexed vector element: `v3.s[1]`.
    VElem { num: u8, esize: ElemSize, index: u8 },
    /// Floating-point immediate: `#1.0`, `#0.5`, `#-2.0`.
    FpImm(f64),
    /// Register list `{v0.16b, v1.16b, …}` or `{v0.16b-v3.16b}` (tbl/structured
    /// load-store). `first` register, `count` consecutive regs, common `arr`.
    VList { first: u8, count: u8, arr: Arr },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Directive {
    Globl(String),
    Text,
    /// `.p2align n` / `.align n` (normalized to a power-of-two exponent).
    P2align(u32),
    Quad(Vec<i64>),
    Byte(u8),
    Zero(usize),
    Ascii(Vec<u8>, bool),
    Other(String),
}

#[derive(Debug, Clone, PartialEq)]
pub enum Line {
    Empty,
    Label(String),
    Directive(Directive),
    Insn { mnemonic: String, ops: Vec<Operand> },
}

/// Strip a `//` line comment. AArch64 (LLVM/GAS) uses `//` for comments; `@` is
/// NOT a comment here — it introduces relocation specifiers (`sym@PAGE`,
/// `sym@PAGEOFF`), so it must survive into the operand.
pub fn strip_comment(s: &str) -> &str {
    match s.find("//") {
        Some(i) => s[..i].trim_end(),
        None => s.trim_end(),
    }
}

/// Peel a leading `label:` (returns the name and the remainder of the line).
pub fn split_leading_label(line: &str) -> (Option<&str>, &str) {
    let t = line.trim_start();
    // A label is `ident:` where ident is the first token and the `:` is not part
    // of an operand (AArch64 has no `:` operands in our subset).
    if let Some(colon) = t.find(':') {
        let name = &t[..colon];
        if !name.is_empty() && is_ident(name) {
            return (Some(name), t[colon + 1..].trim_start());
        }
    }
    (None, t)
}

fn is_ident(s: &str) -> bool {
    let mut chars = s.chars();
    match chars.next() {
        Some(c) if c.is_ascii_alphabetic() || c == '_' || c == '.' || c == '$' => {}
        _ => return false,
    }
    // `@` is allowed *inside* a symbol so relocation specifiers like
    // `sym@PAGE` / `sym@PAGEOFF` parse as a single operand token.
    s.chars().all(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '.' | '$' | '@'))
}

pub fn parse_line(raw: &str) -> Result<Line> {
    let line = raw.trim();
    if line.is_empty() {
        return Ok(Line::Empty);
    }
    if line.starts_with('.') {
        return parse_directive(line);
    }
    // mnemonic + operands
    let (mnem, rest) = match line.split_once(char::is_whitespace) {
        Some((m, r)) => (m, r.trim()),
        None => (line, ""),
    };
    let mnemonic = mnem.to_ascii_lowercase();
    let ops = parse_operands(rest)
        .with_context(|| format!("operands of `{raw}`"))?;
    Ok(Line::Insn { mnemonic, ops })
}

fn parse_directive(line: &str) -> Result<Line> {
    let (name, rest) = match line.split_once(char::is_whitespace) {
        Some((n, r)) => (n, r.trim()),
        None => (line, ""),
    };
    let d = match name {
        ".text" => Directive::Text,
        ".globl" | ".global" => Directive::Globl(rest.to_string()),
        ".p2align" => Directive::P2align(rest.split(',').next().unwrap_or("0").trim().parse().context("p2align")?),
        ".align" => {
            // GAS .align on AArch64/ELF is a power-of-two exponent; on Mach-O it
            // is also the exponent. Treat the operand as the exponent.
            let n: u32 = rest.split(',').next().unwrap_or("0").trim().parse().context(".align")?;
            Directive::P2align(n)
        }
        ".quad" | ".xword" => {
            let vs = rest
                .split(',')
                .map(|t| parse_int(t.trim()))
                .collect::<Result<Vec<_>>>()?;
            Directive::Quad(vs)
        }
        ".byte" => Directive::Byte(parse_int(rest)? as u8),
        ".zero" | ".space" => Directive::Zero(parse_int(rest)? as usize),
        ".asciz" | ".string" => Directive::Ascii(unquote(rest)?, true),
        ".ascii" => Directive::Ascii(unquote(rest)?, false),
        _ => Directive::Other(name.to_string()),
    };
    Ok(Line::Directive(d))
}

/// Split operands on top-level commas (commas inside `[...]` are part of one
/// memory operand) and parse each, folding `lsl/sxtw …` modifiers onto the
/// preceding operand.
fn parse_operands(rest: &str) -> Result<Vec<Operand>> {
    if rest.is_empty() {
        return Ok(vec![]);
    }
    // Tokenize into comma-separated chunks, respecting brackets.
    let mut chunks: Vec<String> = Vec::new();
    let mut depth = 0i32;
    let mut cur = String::new();
    for c in rest.chars() {
        match c {
            '[' | '{' => {
                depth += 1;
                cur.push(c);
            }
            ']' | '}' => {
                depth -= 1;
                cur.push(c);
            }
            ',' if depth == 0 => {
                chunks.push(cur.trim().to_string());
                cur.clear();
            }
            _ => cur.push(c),
        }
    }
    if !cur.trim().is_empty() {
        chunks.push(cur.trim().to_string());
    }

    let mut ops: Vec<Operand> = Vec::new();
    for chunk in chunks {
        // A `lsl/lsr/asr/ror/sxtw/uxtw #n` chunk modifies the previous operand.
        if let Some((shift, amount)) = parse_shift_modifier(&chunk)? {
            match ops.last_mut() {
                Some(Operand::Reg(r)) => {
                    let reg = *r;
                    *ops.last_mut().unwrap() = Operand::RegShift(reg, shift, amount);
                }
                Some(Operand::Imm(v)) => {
                    let v = *v;
                    *ops.last_mut().unwrap() = Operand::ImmShift(v, amount);
                }
                _ => bail!("misplaced shift modifier `{chunk}`"),
            }
            continue;
        }
        // `msl #n` modifier (movi/mvni) folds onto the preceding immediate.
        if let Some(amt) = parse_msl(&chunk)? {
            match ops.last_mut() {
                Some(Operand::Imm(v)) => {
                    let v = *v;
                    *ops.last_mut().unwrap() = Operand::ImmMsl(v, amt);
                    continue;
                }
                _ => bail!("misplaced `msl` modifier"),
            }
        }
        let op = parse_operand(&chunk)?;
        // `[xn], #imm` arrives as two top-level chunks (`[xn]` then `#imm`); a
        // memory operand followed by a bare immediate is *only* ever post-index
        // in AArch64, so fold it.
        if let Operand::Imm(v) = op {
            if let Some(Operand::Mem(m)) = ops.last() {
                if m.addr == Addr::Base {
                    let base = m.base;
                    *ops.last_mut().unwrap() = Operand::Mem(Mem { base, addr: Addr::PostIndex(v) });
                    continue;
                }
            }
        }
        ops.push(op);
    }
    Ok(ops)
}

/// Recognize a standalone `lsl #n` (and friends) chunk.
fn parse_shift_modifier(chunk: &str) -> Result<Option<(Shift, u32)>> {
    let lc = chunk.to_ascii_lowercase();
    let kind = if lc.starts_with("lsl") {
        Shift::Lsl
    } else if lc.starts_with("lsr") {
        Shift::Lsr
    } else if lc.starts_with("asr") {
        Shift::Asr
    } else if lc.starts_with("ror") {
        Shift::Ror
    } else {
        return Ok(None);
    };
    // After the keyword expect `#n` (or bare n).
    let amt = lc[3..].trim();
    let amt = amt.strip_prefix('#').unwrap_or(amt).trim();
    let amount: u32 = amt.parse().with_context(|| format!("shift amount in `{chunk}`"))?;
    Ok(Some((kind, amount)))
}

/// Recognize a standalone `msl #n` chunk (movi/mvni masking shift).
fn parse_msl(chunk: &str) -> Result<Option<u32>> {
    let lc = chunk.trim().to_ascii_lowercase();
    let Some(rest) = lc.strip_prefix("msl") else {
        return Ok(None);
    };
    let amt = rest.trim();
    let amt = amt.strip_prefix('#').unwrap_or(amt).trim();
    Ok(Some(amt.parse().with_context(|| format!("msl amount in `{chunk}`"))?))
}

/// Parse `vN.<arr>` → (num, arr). Returns `None` if not a vector-arrangement reg.
fn vreg_parts(tok: &str) -> Option<(u8, Arr)> {
    match parse_vreg(tok).ok()?? {
        Operand::VReg { num, arr } => Some((num, arr)),
        _ => None,
    }
}

/// Parse a register list `{v0.16b, v1.16b, …}` or `{v0.16b-v3.16b}`.
fn parse_vlist(s: &str) -> Result<Operand> {
    let close = s.rfind('}').context("unterminated `{`")?;
    let inner = s[1..close].trim();
    // Range form `vA.arr - vB.arr`.
    if let Some((a, b)) = inner.split_once('-') {
        let (first, arr) = vreg_parts(a.trim()).context("bad register-list start")?;
        let (last, _) = vreg_parts(b.trim()).context("bad register-list end")?;
        let count = last.wrapping_sub(first).wrapping_add(1);
        return Ok(Operand::VList { first, count, arr });
    }
    let regs: Vec<(u8, Arr)> = inner
        .split(',')
        .map(|t| vreg_parts(t.trim()).context("bad register in list"))
        .collect::<Result<Vec<_>>>()?;
    if regs.is_empty() {
        bail!("empty register list");
    }
    let (first, arr) = regs[0];
    Ok(Operand::VList { first, count: regs.len() as u8, arr })
}

fn parse_operand(s: &str) -> Result<Operand> {
    let s = s.trim();
    if s.starts_with('{') {
        return parse_vlist(s);
    }
    if s.starts_with('[') {
        return parse_mem(s);
    }
    if let Some(imm) = s.strip_prefix('#') {
        let t = imm.trim();
        // Floating-point immediate: a non-hex literal with a `.`/`e`/inf/nan.
        let lower = t.to_ascii_lowercase();
        let is_float = !lower.starts_with("0x")
            && (lower.contains('.') || lower.contains("inf") || lower.contains("nan")
                || (lower.contains('e') && !lower.starts_with("0b")));
        if is_float {
            return Ok(Operand::FpImm(t.parse::<f64>().with_context(|| format!("fp immediate `{t}`"))?));
        }
        return Ok(Operand::Imm(parse_int(t)?));
    }
    // Vector register with arrangement (`v3.4s`) or lane (`v3.s[1]`).
    if let Some(v) = parse_vreg(s)? {
        return Ok(v);
    }
    if let Some(r) = parse_reg(s) {
        return Ok(Operand::Reg(r));
    }
    // Otherwise a symbol (branch target). Reject obvious garbage.
    if is_ident(s) {
        return Ok(Operand::Sym(s.to_string()));
    }
    bail!("unrecognized operand `{s}`")
}

/// Parse a SIMD vector operand `v<n>.<arr>` or `v<n>.<es>[<i>]`. Returns
/// `Ok(None)` if `s` isn't a `v`-register with a `.` suffix.
fn parse_vreg(s: &str) -> Result<Option<Operand>> {
    let Some((head, tail)) = s.split_once('.') else {
        return Ok(None);
    };
    let head = head.trim().to_ascii_lowercase();
    let Some(num_str) = head.strip_prefix('v') else {
        return Ok(None);
    };
    let num: u8 = match num_str.parse() {
        Ok(n) if n <= 31 => n,
        _ => return Ok(None),
    };
    let tail = tail.trim().to_ascii_lowercase();
    if let Some((es, rest)) = tail.split_once('[') {
        let index: u8 = rest.trim_end_matches(']').trim().parse().context("lane index")?;
        let esize = match es {
            "b" => ElemSize::B,
            "h" => ElemSize::H,
            "s" => ElemSize::S,
            "d" => ElemSize::D,
            _ => bail!("bad element size `{es}`"),
        };
        return Ok(Some(Operand::VElem { num, esize, index }));
    }
    let arr = match tail.as_str() {
        "8b" => Arr::B8,
        "16b" => Arr::B16,
        "4h" => Arr::H4,
        "8h" => Arr::H8,
        "2s" => Arr::S2,
        "4s" => Arr::S4,
        "1d" => Arr::D1,
        "2d" => Arr::D2,
        _ => bail!("bad vector arrangement `{tail}`"),
    };
    Ok(Some(Operand::VReg { num, arr }))
}

fn parse_mem(s: &str) -> Result<Operand> {
    let inner_close = s.rfind(']').context("unterminated `[`")?;
    let inner = &s[1..inner_close];
    let trailer = s[inner_close + 1..].trim(); // "" or "!" or ", #imm" (post-index)

    let mut parts = inner.splitn(2, ',');
    let base_tok = parts.next().unwrap().trim();
    let base = parse_reg(base_tok).with_context(|| format!("base register `{base_tok}`"))?;

    let addr = match parts.next() {
        None => {
            // `[xn]` or `[xn], #imm` (post-index lives in the trailer)
            if let Some(imm) = trailer.strip_prefix(',') {
                Addr::PostIndex(parse_hash_int(imm.trim())?)
            } else {
                Addr::Base
            }
        }
        Some(rest) => {
            let rest = rest.trim();
            if rest.starts_with('#') {
                let off = parse_hash_int(rest)?;
                if trailer == "!" {
                    Addr::PreIndex(off)
                } else {
                    Addr::Offset(off)
                }
            } else {
                // Register offset: `xm` / `xm, lsl #s` / `wm, sxtw #s` …
                parse_reg_offset(rest)?
            }
        }
    };
    Ok(Operand::Mem(Mem { base, addr }))
}

/// Parse the index part of a register-offset memory operand (everything after
/// `base,` inside the brackets): `xm`, `xm, lsl #3`, `wm, sxtw`, `wm, sxtw #3`.
fn parse_reg_offset(s: &str) -> Result<Addr> {
    let mut it = s.splitn(2, ',');
    let idx_tok = it.next().unwrap().trim();
    let index = parse_reg(idx_tok).with_context(|| format!("index register `{idx_tok}`"))?;
    match it.next() {
        None => Ok(Addr::RegOffset { index, ext: IndexExt::Lsl, amount: 0, scaled: false }),
        Some(ext_spec) => {
            let ext_spec = ext_spec.trim().to_ascii_lowercase();
            let (kw, rest) = match ext_spec.split_once(char::is_whitespace) {
                Some((k, r)) => (k, r.trim()),
                None => (ext_spec.as_str(), ""),
            };
            let ext = match kw {
                "lsl" => IndexExt::Lsl,
                "uxtw" => IndexExt::Uxtw,
                "sxtw" => IndexExt::Sxtw,
                "sxtx" => IndexExt::Sxtx,
                "uxtx" => IndexExt::Uxtx,
                _ => bail!("unknown index extend `{kw}`"),
            };
            let (amount, scaled) = if rest.is_empty() {
                (0u32, false)
            } else {
                let n = rest.strip_prefix('#').unwrap_or(rest).trim();
                (n.parse().with_context(|| format!("extend amount `{rest}`"))?, true)
            };
            Ok(Addr::RegOffset { index, ext, amount, scaled })
        }
    }
}

fn parse_hash_int(s: &str) -> Result<i64> {
    parse_int(s.strip_prefix('#').unwrap_or(s).trim())
}

/// Parse a register token: `x0`–`x30`, `w0`–`w30`, `sp`/`wsp`, `xzr`/`wzr`,
/// `v0`–`v31` (and `q/d/s/h/b` views, num only). Returns `None` if not a reg.
pub fn parse_reg(tok: &str) -> Option<Reg> {
    let t = tok.trim().to_ascii_lowercase();
    match t.as_str() {
        "sp" => return Some(Reg { class: RegClass::X, num: 31, is_sp: true }),
        "wsp" => return Some(Reg { class: RegClass::W, num: 31, is_sp: true }),
        "xzr" => return Some(Reg { class: RegClass::X, num: 31, is_sp: false }),
        "wzr" => return Some(Reg { class: RegClass::W, num: 31, is_sp: false }),
        _ => {}
    }
    let (class, digits) = match t.split_at(1) {
        ("x", d) => (RegClass::X, d),
        ("w", d) => (RegClass::W, d),
        ("b", d) => (RegClass::B, d),
        ("h", d) => (RegClass::H, d),
        ("s", d) => (RegClass::S, d),
        ("d", d) => (RegClass::D, d),
        ("q", d) => (RegClass::Q, d),
        ("v", d) => (RegClass::V, d),
        _ => return None,
    };
    let num: u8 = digits.parse().ok()?;
    let max = if matches!(class, RegClass::X | RegClass::W) { 30 } else { 31 };
    if num > max {
        return None;
    }
    Some(Reg { class, num, is_sp: false })
}

/// Parse an integer: decimal, `0x..` hex, `0b..` binary, optional leading `-`.
pub fn parse_int(s: &str) -> Result<i64> {
    let s = s.trim();
    let (neg, body) = match s.strip_prefix('-') {
        Some(rest) => (true, rest.trim()),
        None => (false, s),
    };
    // Parse hex/binary as u64 then reinterpret, so a high-bit literal like
    // `0xffffffffffff0000` is accepted (it round-trips through `as u64` in the
    // encoder); decimal stays i64 so leading `-` works.
    let v: i64 = if let Some(h) = body.strip_prefix("0x").or_else(|| body.strip_prefix("0X")) {
        u64::from_str_radix(h, 16).with_context(|| format!("hex int `{s}`"))? as i64
    } else if let Some(b) = body.strip_prefix("0b").or_else(|| body.strip_prefix("0B")) {
        u64::from_str_radix(b, 2).with_context(|| format!("binary int `{s}`"))? as i64
    } else {
        body.parse().with_context(|| format!("int `{s}`"))?
    };
    Ok(if neg { -v } else { v })
}

fn unquote(s: &str) -> Result<Vec<u8>> {
    let s = s.trim();
    let inner = s
        .strip_prefix('"')
        .and_then(|r| r.strip_suffix('"'))
        .context("string literal must be quoted")?;
    let mut out = Vec::new();
    let mut chars = inner.chars();
    while let Some(c) = chars.next() {
        if c == '\\' {
            match chars.next() {
                Some('n') => out.push(b'\n'),
                Some('t') => out.push(b'\t'),
                Some('0') => out.push(0),
                Some('\\') => out.push(b'\\'),
                Some('"') => out.push(b'"'),
                Some(other) => out.push(other as u8),
                None => break,
            }
        } else {
            out.push(c as u8);
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn regs_and_imm() {
        let Line::Insn { mnemonic, ops } = parse_line("add x0, x1, x2").unwrap() else { panic!() };
        assert_eq!(mnemonic, "add");
        assert_eq!(ops[0], Operand::Reg(Reg { class: RegClass::X, num: 0, is_sp: false }));
        assert_eq!(ops[2], Operand::Reg(Reg { class: RegClass::X, num: 2, is_sp: false }));
    }

    #[test]
    fn sp_and_imm_shift() {
        let Line::Insn { ops, .. } = parse_line("sub sp, sp, #16").unwrap() else { panic!() };
        assert_eq!(ops[0], Operand::Reg(Reg { class: RegClass::X, num: 31, is_sp: true }));
        assert_eq!(ops[2], Operand::Imm(16));

        let Line::Insn { ops, .. } = parse_line("movk x0, #0xabcd, lsl #32").unwrap() else { panic!() };
        assert_eq!(ops[1], Operand::ImmShift(0xabcd, 32));
    }

    #[test]
    fn memory_forms() {
        let Line::Insn { ops, .. } = parse_line("ldr x0, [x1, #8]").unwrap() else { panic!() };
        assert_eq!(ops[1], Operand::Mem(Mem { base: Reg { class: RegClass::X, num: 1, is_sp: false }, addr: Addr::Offset(8) }));

        let Line::Insn { ops, .. } = parse_line("str x0, [sp, #-16]!").unwrap() else { panic!() };
        assert_eq!(ops[1], Operand::Mem(Mem { base: Reg { class: RegClass::X, num: 31, is_sp: true }, addr: Addr::PreIndex(-16) }));

        let Line::Insn { ops, .. } = parse_line("ldr x0, [x1], #8").unwrap() else { panic!() };
        assert_eq!(ops[1], Operand::Mem(Mem { base: Reg { class: RegClass::X, num: 1, is_sp: false }, addr: Addr::PostIndex(8) }));
    }

    #[test]
    fn shifted_register() {
        let Line::Insn { ops, .. } = parse_line("add x0, x1, x2, lsl #3").unwrap() else { panic!() };
        assert_eq!(ops[2], Operand::RegShift(Reg { class: RegClass::X, num: 2, is_sp: false }, Shift::Lsl, 3));
    }

    #[test]
    fn fp_and_simd_operands() {
        let Line::Insn { ops, .. } = parse_line("fadd d0, d1, d2").unwrap() else { panic!() };
        assert_eq!(ops[0], Operand::Reg(Reg { class: RegClass::D, num: 0, is_sp: false }));
        assert_eq!(ops[2], Operand::Reg(Reg { class: RegClass::D, num: 2, is_sp: false }));

        let Line::Insn { ops, .. } = parse_line("fmov d0, #1.0").unwrap() else { panic!() };
        assert_eq!(ops[1], Operand::FpImm(1.0));

        let Line::Insn { ops, .. } = parse_line("add v0.4s, v1.4s, v2.4s").unwrap() else { panic!() };
        assert_eq!(ops[0], Operand::VReg { num: 0, arr: Arr::S4 });
        assert_eq!(ops[2], Operand::VReg { num: 2, arr: Arr::S4 });

        let Line::Insn { ops, .. } = parse_line("dup v0.2d, v1.d[1]").unwrap() else { panic!() };
        assert_eq!(ops[0], Operand::VReg { num: 0, arr: Arr::D2 });
        assert_eq!(ops[1], Operand::VElem { num: 1, esize: ElemSize::D, index: 1 });

        let Line::Insn { ops, .. } = parse_line("ldr q3, [x0, #16]").unwrap() else { panic!() };
        assert_eq!(ops[0], Operand::Reg(Reg { class: RegClass::Q, num: 3, is_sp: false }));
    }

    #[test]
    fn label_and_branch() {
        let (label, rest) = split_leading_label("loop: sub x0, x0, #1");
        assert_eq!(label, Some("loop"));
        assert_eq!(rest, "sub x0, x0, #1");
        let Line::Insn { mnemonic, ops } = parse_line("b.eq done").unwrap() else { panic!() };
        assert_eq!(mnemonic, "b.eq");
        assert_eq!(ops[0], Operand::Sym("done".to_string()));
    }
}
