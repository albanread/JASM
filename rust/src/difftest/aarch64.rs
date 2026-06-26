//! `Aarch64Model` — a templated form generator for the AArch64 differential
//! corpus. Mirrors [`x86::X86Model`](super::x86::X86Model): it yields
//! reloc-free single-instruction forms (so the arch-neutral
//! [`compare`](super::compare) needs no per-kind field masking), which the
//! oracle and [`A64Encoder`](crate::a64::A64Encoder) are diffed against and the
//! green ones recorded to `corpus/aarch64.tsv`.
//!
//! Over-generation is fine: forms the oracle rejects (illegal arrangement, etc.)
//! or the encoder can't build are skipped by
//! [`record_corpus`](super::record_corpus). Branch/pc-relative/symbol forms are
//! intentionally excluded (they carry relocations).

use super::{Form, IsaModel};

pub struct Aarch64Model;

/// `(dst, src1, src2)` register-index triples reused across 3-operand forms.
/// Indices stay ≤30 — `31` is `xzr`/`sp`, which must be spelled by name, not
/// as `x31` (LLVM leniently maps `x31`→`xzr`, but it's not a form we generate).
const TRIPLES: &[(u8, u8, u8)] = &[(0, 1, 2), (5, 10, 19), (28, 27, 26), (15, 0, 30)];
const PAIRS: &[(u8, u8)] = &[(0, 1), (5, 19), (28, 3), (15, 30)];

/// Integer-arrangement set (`.8b … .2d`); FP-arrangement set (`.2s/.4s/.2d`).
const ARR_INT: &[&str] = &["8b", "16b", "4h", "8h", "2s", "4s", "2d"];
const ARR_FP: &[&str] = &["2s", "4s", "2d"];

impl IsaModel for Aarch64Model {
    fn triple(&self) -> &str {
        "aarch64-apple-darwin"
    }

    fn forms(&self) -> Vec<Form> {
        let mut f: Vec<Form> = Vec::new();
        let mut add = |asm: String, family: &'static str, mnemonic: &'static str| {
            f.push(Form { asm, family, mnemonic });
        };

        // ── GP ALU: 3-register (X and W) ────────────────────────────────────
        for &m in &[
            "add", "sub", "adds", "subs", "and", "orr", "eor", "ands", "bic", "orn", "eon", "bics",
            "mul", "smulh", "umulh", "sdiv", "udiv", "lslv", "lsrv", "asrv",
            "rorv", "adc", "sbc", "adcs", "sbcs",
        ] {
            for &(d, n, mm) in TRIPLES {
                add(format!("{m} x{d}, x{n}, x{mm}"), "gp.alu3", m);
                add(format!("{m} w{d}, w{n}, w{mm}"), "gp.alu3", m);
            }
        }

        // ── GP ALU: add/sub immediate + lsl #12 ─────────────────────────────
        for &m in &["add", "sub", "adds", "subs"] {
            for &(d, n) in PAIRS {
                for imm in ["#0", "#1", "#42", "#4095"] {
                    add(format!("{m} x{d}, x{n}, {imm}"), "gp.addsub.imm", m);
                    add(format!("{m} w{d}, w{n}, {imm}"), "gp.addsub.imm", m);
                }
                add(format!("{m} x{d}, x{n}, #1, lsl #12"), "gp.addsub.imm", m);
            }
        }

        // ── GP logical immediate (bitmask) ──────────────────────────────────
        for &m in &["and", "orr", "eor", "ands"] {
            for imm in ["#0xff", "#0xf", "#0x1", "#0xfff", "#0x5555555555555555", "#0xffff0000"] {
                add(format!("{m} x0, x1, {imm}"), "gp.logical.imm", m);
            }
            for imm in ["#0xff", "#0xf", "#0x1", "#0xf0f0f0f0"] {
                add(format!("{m} w0, w1, {imm}"), "gp.logical.imm", m);
            }
        }

        // ── mov #imm (movz/movn/orr lowering) + movz/movk/movn ──────────────
        for imm in ["#0", "#1", "#0x1234", "#0xffff", "#-1", "#0x10000", "#0x5555555555555555"] {
            for &d in &[0u8, 5, 28] {
                add(format!("mov x{d}, {imm}"), "gp.mov.imm", "mov");
            }
        }
        for &m in &["movz", "movk", "movn"] {
            for sh in ["", ", lsl #16", ", lsl #32", ", lsl #48"] {
                add(format!("{m} x0, #0x1234{sh}"), "gp.movewide", m);
            }
            add(format!("{m} w1, #0x1234"), "gp.movewide", m);
        }

        // ── shifts/bitfield by immediate ────────────────────────────────────
        for &m in &["lsl", "lsr", "asr"] {
            for sh in ["#1", "#3", "#31"] {
                add(format!("{m} x0, x1, {sh}"), "gp.shift.imm", m);
                add(format!("{m} w0, w1, {sh}"), "gp.shift.imm", m);
            }
        }
        for &m in &["ubfx", "sbfx", "ubfiz", "sbfiz", "bfi", "bfxil"] {
            add(format!("{m} x0, x1, #4, #8"), "gp.bitfield", m);
            add(format!("{m} w0, w1, #2, #6"), "gp.bitfield", m);
        }
        for &m in &["sxtw", "sxtb", "sxth", "uxtw"] {
            add(format!("{m} x0, w1"), "gp.extend", m);
        }
        for &m in &["uxtb", "uxth"] {
            add(format!("{m} w0, w1"), "gp.extend", m);
        }
        for &m in &["clz", "cls", "rbit", "rev", "rev16", "rev32"] {
            add(format!("{m} x0, x1"), "gp.dp1", m);
            add(format!("{m} w0, w1"), "gp.dp1", m);
        }

        // ── conditional select / compare ────────────────────────────────────
        for &m in &["csel", "csinc", "csinv", "csneg"] {
            for c in ["eq", "ne", "lt", "ge", "hi"] {
                add(format!("{m} x0, x1, x2, {c}"), "gp.csel", m);
            }
        }
        for &m in &["cset", "csetm"] {
            for c in ["eq", "ne", "lt", "gt"] {
                add(format!("{m} x0, {c}"), "gp.cset", m);
            }
        }
        for &m in &["cmp", "cmn", "tst"] {
            add(format!("{m} x1, x2"), "gp.cmp", m);
            add(format!("{m} x1, #42"), "gp.cmp", m);
        }
        add("ccmp x0, x1, #0, eq".into(), "gp.ccmp", "ccmp");
        add("ccmp w0, #31, #15, ne".into(), "gp.ccmp", "ccmp");
        for &m in &["crc32b", "crc32h", "crc32w", "crc32cb"] {
            add(format!("{m} w0, w1, w2"), "gp.crc", m);
        }
        add("crc32x w0, w1, x2".into(), "gp.crc", "crc32x");

        // ── loads / stores (GP) ─────────────────────────────────────────────
        for &m in &["ldr", "str"] {
            for off in ["[x1]", "[x1, #8]", "[x1, #-8]!", "[x1], #8", "[x1, x2]", "[x1, x2, lsl #3]"] {
                add(format!("{m} x0, {off}"), "mem.gp", m);
            }
            add(format!("{m} w0, [x1, #4]"), "mem.gp", m);
        }
        for &m in &["ldrb", "strb", "ldrh", "strh", "ldrsb", "ldrsh", "ldrsw"] {
            add(format!("{m} w0, [x1, #4]"), "mem.gp.ext", m);
        }
        for &m in &["ldp", "stp"] {
            for off in ["[x2]", "[x2, #16]", "[sp, #-16]!", "[sp], #16"] {
                add(format!("{m} x0, x1, {off}"), "mem.pair", m);
            }
        }

        // ── atomics ─────────────────────────────────────────────────────────
        for &m in &["ldxr", "ldaxr", "ldar"] {
            add(format!("{m} x0, [x1]"), "atomic.excl", m);
            add(format!("{m} w0, [x1]"), "atomic.excl", m);
        }
        for &m in &["stxr", "stlxr"] {
            add(format!("{m} w0, x1, [x2]"), "atomic.excl", m);
        }
        for &m in &[
            "ldadd", "ldclr", "ldeor", "ldset", "ldsmax", "ldumin", "ldadda", "ldaddl", "ldaddal",
            "swp", "swpal", "cas", "casal",
        ] {
            add(format!("{m} x0, x1, [x2]"), "atomic.lse", m);
            add(format!("{m} w0, w1, [x2]"), "atomic.lse", m);
        }

        // ── system ──────────────────────────────────────────────────────────
        for &m in &["nop", "yield", "wfe", "wfi", "sev", "sevl"] {
            add(m.to_string(), "sys.hint", m);
        }
        for o in ["sy", "ish", "ishst", "nsh", "osh", "ld", "st"] {
            add(format!("dmb {o}"), "sys.barrier", "dmb");
            add(format!("dsb {o}"), "sys.barrier", "dsb");
        }
        add("isb".into(), "sys.barrier", "isb");
        add("svc #0".into(), "sys.exc", "svc");
        add("brk #0".into(), "sys.exc", "brk");
        for r in ["nzcv", "fpcr", "fpsr", "tpidr_el0", "cntvct_el0", "ctr_el0"] {
            add(format!("mrs x0, {r}"), "sys.reg", "mrs");
        }
        add("msr nzcv, x0".into(), "sys.reg", "msr");
        add("msr daifset, #2".into(), "sys.pstate", "msr");

        // ── scalar FP ───────────────────────────────────────────────────────
        for &m in &["fadd", "fsub", "fmul", "fdiv", "fmax", "fmin", "fnmul", "fmaxnm", "fminnm"] {
            add(format!("{m} s0, s1, s2"), "fp.arith3", m);
            add(format!("{m} d0, d1, d2"), "fp.arith3", m);
        }
        for &m in &["fabs", "fneg", "fsqrt", "fmov", "frinta", "frintn", "frintm", "frintz", "frintp"] {
            add(format!("{m} s0, s1"), "fp.arith1", m);
            add(format!("{m} d0, d1"), "fp.arith1", m);
        }
        for &m in &["fmadd", "fmsub", "fnmadd", "fnmsub"] {
            add(format!("{m} s0, s1, s2, s3"), "fp.arith4", m);
            add(format!("{m} d0, d1, d2, d3"), "fp.arith4", m);
        }
        add("fcvt d0, s1".into(), "fp.cvt", "fcvt");
        add("fcvt s0, d1".into(), "fp.cvt", "fcvt");
        for &m in &["fcvtzs", "fcvtzu", "fcvtns", "fcvtms", "fcvtps", "fcvtas"] {
            add(format!("{m} w0, s1"), "fp.cvt.int", m);
            add(format!("{m} x0, d1"), "fp.cvt.int", m);
        }
        for &m in &["scvtf", "ucvtf"] {
            add(format!("{m} s0, w1"), "fp.cvt.int", m);
            add(format!("{m} d0, x1"), "fp.cvt.int", m);
        }
        // Fixed-point FP↔int convert (#fbits).
        for &m in &["fcvtzs", "fcvtzu"] {
            add(format!("{m} w0, s1, #4"), "fp.cvt.fixed", m);
            add(format!("{m} x0, d1, #16"), "fp.cvt.fixed", m);
        }
        for &m in &["scvtf", "ucvtf"] {
            add(format!("{m} s0, w1, #4"), "fp.cvt.fixed", m);
            add(format!("{m} d0, x1, #16"), "fp.cvt.fixed", m);
        }
        add("fcmp s0, s1".into(), "fp.cmp", "fcmp");
        add("fcmp d0, #0.0".into(), "fp.cmp", "fcmp");
        add("fcsel s0, s1, s2, eq".into(), "fp.csel", "fcsel");
        add("fmov w0, s1".into(), "fp.mov", "fmov");
        add("fmov s0, w1".into(), "fp.mov", "fmov");
        add("fmov x0, d1".into(), "fp.mov", "fmov");
        add("fmov d0, x1".into(), "fp.mov", "fmov");
        for imm in ["#1.0", "#2.0", "#0.5", "#-1.0", "#1.9375"] {
            add(format!("fmov d0, {imm}"), "fp.mov.imm", "fmov");
            add(format!("fmov s0, {imm}"), "fp.mov.imm", "fmov");
        }

        // ── NEON 3-same integer ─────────────────────────────────────────────
        for &m in &[
            "add", "sub", "mul", "mla", "mls", "smax", "smin", "umax", "umin", "sshl", "ushl",
            "sqadd", "uqadd", "sqsub", "uqsub", "cmeq", "cmgt", "cmge", "cmhi", "cmhs", "cmtst",
            "addp",
        ] {
            for &a in ARR_INT {
                add(format!("{m} v0.{a}, v1.{a}, v2.{a}"), "neon.3same.int", m);
            }
        }
        for &m in &["and", "orr", "eor", "bic", "orn", "bsl"] {
            for a in ["8b", "16b"] {
                add(format!("{m} v0.{a}, v1.{a}, v2.{a}"), "neon.3same.logic", m);
            }
        }
        for &m in &[
            "fadd", "fsub", "fmul", "fdiv", "fmla", "fmls", "fmax", "fmin", "fabd", "fmulx",
            "fmaxnm", "fminnm", "faddp", "fmaxp", "fminp", "fcmeq", "fcmge", "fcmgt",
        ] {
            for &a in ARR_FP {
                add(format!("{m} v0.{a}, v1.{a}, v2.{a}"), "neon.3same.fp", m);
            }
        }

        // ── NEON 2-misc ─────────────────────────────────────────────────────
        for &m in &["neg", "abs", "cls", "clz", "rev64"] {
            for &a in &["8b", "16b", "4h", "8h", "2s", "4s"] {
                add(format!("{m} v0.{a}, v1.{a}"), "neon.2misc.int", m);
            }
        }
        for a in ["8b", "16b"] {
            add(format!("not v0.{a}, v1.{a}"), "neon.2misc.int", "not");
            add(format!("cnt v0.{a}, v1.{a}"), "neon.2misc.int", "cnt");
            add(format!("rev16 v0.{a}, v1.{a}"), "neon.2misc.int", "rev16");
        }
        for &m in &["fneg", "fabs", "fsqrt", "scvtf", "ucvtf", "fcvtzs", "fcvtzu", "frintn", "frintp", "frintm", "frintz"] {
            for &a in ARR_FP {
                add(format!("{m} v0.{a}, v1.{a}"), "neon.2misc.fp", m);
            }
        }
        for &m in &["fcmeq", "fcmge", "fcmgt", "fcmle", "fcmlt"] {
            for &a in ARR_FP {
                add(format!("{m} v0.{a}, v1.{a}, #0.0"), "neon.2misc.fcmp0", m);
            }
        }

        // ── NEON across-lane / dup / copy / permute ─────────────────────────
        add("addv s0, v1.4s".into(), "neon.across", "addv");
        add("addv b0, v1.8b".into(), "neon.across", "addv");
        add("addv h0, v1.8h".into(), "neon.across", "addv");
        for &m in &["smaxv", "sminv", "umaxv", "uminv"] {
            add(format!("{m} s0, v1.4s"), "neon.across", m);
        }
        add("dup v0.4s, v1.s[1]".into(), "neon.dup", "dup");
        add("dup v0.16b, w1".into(), "neon.dup", "dup");
        add("dup v0.2d, x1".into(), "neon.dup", "dup");
        add("ins v0.s[1], v1.s[2]".into(), "neon.copy", "ins");
        add("ins v0.d[1], x2".into(), "neon.copy", "ins");
        add("umov w0, v1.s[2]".into(), "neon.copy", "umov");
        add("smov x0, v1.h[2]".into(), "neon.copy", "smov");
        for &m in &["zip1", "zip2", "uzp1", "uzp2", "trn1", "trn2"] {
            for a in ["16b", "4s", "2d"] {
                add(format!("{m} v0.{a}, v1.{a}, v2.{a}"), "neon.permute", m);
            }
        }
        add("ext v0.16b, v1.16b, v2.16b, #4".into(), "neon.permute", "ext");
        add("tbl v0.16b, {v1.16b}, v2.16b".into(), "neon.tbl", "tbl");
        add("tbl v0.16b, {v1.16b, v2.16b}, v3.16b".into(), "neon.tbl", "tbl");
        add("tbx v0.16b, {v1.16b, v2.16b, v3.16b}, v4.16b".into(), "neon.tbl", "tbx");

        // ── NEON shift-by-immediate ─────────────────────────────────────────
        for &m in &["shl", "sli"] {
            for (a, sh) in [("16b", "#3"), ("8h", "#5"), ("4s", "#10"), ("2d", "#40")] {
                add(format!("{m} v0.{a}, v1.{a}, {sh}"), "neon.shift.left", m);
            }
        }
        for &m in &["sshr", "ushr", "ssra", "usra", "srshr", "urshr", "sri"] {
            for (a, sh) in [("16b", "#3"), ("8h", "#5"), ("4s", "#10"), ("2d", "#40")] {
                add(format!("{m} v0.{a}, v1.{a}, {sh}"), "neon.shift.right", m);
            }
        }

        // ── NEON widen / narrow / long ──────────────────────────────────────
        for &m in &["saddl", "uaddl", "ssubl", "usubl", "smull", "umull", "smlal", "umlal", "smlsl", "umlsl"] {
            add(format!("{m} v0.8h, v1.8b, v2.8b"), "neon.long", m);
            add(format!("{m} v0.4s, v1.4h, v2.4h"), "neon.long", m);
            add(format!("{m}2 v0.4s, v1.8h, v2.8h"), "neon.long", m);
        }
        for &m in &["saddw", "uaddw", "ssubw", "usubw"] {
            add(format!("{m} v0.8h, v1.8h, v2.8b"), "neon.widen", m);
        }
        for &m in &["addhn", "raddhn", "subhn", "rsubhn"] {
            add(format!("{m} v0.8b, v1.8h, v2.8h"), "neon.narrow", m);
            add(format!("{m}2 v0.16b, v1.8h, v2.8h"), "neon.narrow", m);
        }
        for &m in &["xtn", "sqxtn", "uqxtn", "sqxtun"] {
            add(format!("{m} v0.8b, v1.8h"), "neon.narrow2", m);
            add(format!("{m}2 v0.16b, v1.8h"), "neon.narrow2", m);
        }

        // ── NEON by-element ─────────────────────────────────────────────────
        for &m in &["mul", "mla", "mls", "sqdmulh", "sqrdmulh"] {
            add(format!("{m} v0.4s, v1.4s, v2.s[1]"), "neon.byelem", m);
            add(format!("{m} v0.8h, v1.8h, v2.h[3]"), "neon.byelem", m);
        }
        for &m in &["fmul", "fmla", "fmls", "fmulx"] {
            add(format!("{m} v0.4s, v1.4s, v2.s[2]"), "neon.byelem.fp", m);
            add(format!("{m} v0.2d, v1.2d, v2.d[1]"), "neon.byelem.fp", m);
        }
        for &m in &["smull", "umull", "smlal", "umlal"] {
            add(format!("{m} v0.4s, v1.4h, v2.h[3]"), "neon.byelem.long", m);
        }

        // ── NEON modified immediate ─────────────────────────────────────────
        add("movi v0.16b, #0xab".into(), "neon.movi", "movi");
        add("movi v0.8b, #0xab".into(), "neon.movi", "movi");
        for sh in ["", ", lsl #8", ", lsl #16", ", lsl #24", ", msl #8", ", msl #16"] {
            add(format!("movi v0.4s, #0xab{sh}"), "neon.movi", "movi");
            add(format!("mvni v0.4s, #0xab{sh}"), "neon.movi", "mvni");
        }
        for sh in ["", ", lsl #8"] {
            add(format!("movi v0.8h, #0xab{sh}"), "neon.movi", "movi");
            add(format!("bic v0.4s, #0xab{sh}"), "neon.movi", "bic");
            add(format!("orr v0.4s, #0xab{sh}"), "neon.movi", "orr");
        }
        add("movi v0.2d, #0xff00ff00ff00ff00".into(), "neon.movi", "movi");
        add("movi d0, #0xffffffffffffffff".into(), "neon.movi", "movi");

        // ── NEON FP narrow / long + structured load/store ───────────────────
        add("fcvtn v0.4h, v1.4s".into(), "neon.fcvt", "fcvtn");
        add("fcvtn2 v0.8h, v1.4s".into(), "neon.fcvt", "fcvtn");
        add("fcvtl v0.4s, v1.4h".into(), "neon.fcvt", "fcvtl");
        add("fcvtl2 v0.4s, v1.8h".into(), "neon.fcvt", "fcvtl");
        for &m in &["ld1", "st1"] {
            for a in ["16b", "8h", "4s", "2d"] {
                add(format!("{m} {{v0.{a}}}, [x0]"), "neon.ldst1", m);
                add(format!("{m} {{v0.{a}, v1.{a}}}, [x0]"), "neon.ldst1", m);
            }
        }
        add("ld2 {v0.4s, v1.4s}, [x0]".into(), "neon.ldstN", "ld2");
        add("ld3 {v0.8h, v1.8h, v2.8h}, [x0]".into(), "neon.ldstN", "ld3");
        add("ld4 {v0.4s, v1.4s, v2.4s, v3.4s}, [x0]".into(), "neon.ldstN", "ld4");
        add("ld1 {v0.16b}, [x0], #16".into(), "neon.ldst.post", "ld1");
        add("ld1 {v0.16b}, [x0], x2".into(), "neon.ldst.post", "ld1");

        // ── crypto: AES + SHA1/SHA256 ───────────────────────────────────────
        for &m in &["aese", "aesd", "aesmc", "aesimc"] {
            add(format!("{m} v0.16b, v1.16b"), "crypto.aes", m);
        }
        add("sha1h s0, s1".into(), "crypto.sha", "sha1h");
        add("sha1su1 v0.4s, v1.4s".into(), "crypto.sha", "sha1su1");
        add("sha256su0 v0.4s, v1.4s".into(), "crypto.sha", "sha256su0");
        for &m in &["sha1c", "sha1p", "sha1m"] {
            add(format!("{m} q0, s1, v2.4s"), "crypto.sha", m);
        }
        add("sha1su0 v0.4s, v1.4s, v2.4s".into(), "crypto.sha", "sha1su0");
        for &m in &["sha256h", "sha256h2"] {
            add(format!("{m} q0, q1, v2.4s"), "crypto.sha", m);
        }
        add("sha256su1 v0.4s, v1.4s, v2.4s".into(), "crypto.sha", "sha256su1");

        f
    }
}
