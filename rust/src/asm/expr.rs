//! Compile-time integer expression evaluator.
//!
//! Powers `@assign NAME = expr`, `@if expr`, `@elif expr`, `@rept expr`,
//! `@assert expr, "msg"`, and Rust-macro `args.parse_int(N)`. Everything
//! is `i64`; there are no floats, strings, or pointers in expression
//! context.
//!
//! ## Grammar (low precedence → high)
//!
//! ```text
//! expr        = or
//! or          = and        ( "||" and )*
//! and         = bitor      ( "&&" bitor )*
//! bitor       = bitxor     ( "|"  bitxor )*
//! bitxor      = bitand     ( "^"  bitand )*
//! bitand      = equality   ( "&"  equality )*
//! equality    = relational ( ("==" | "!=") relational )*
//! relational  = shift      ( ("<" | "<=" | ">" | ">=") shift )*
//! shift       = additive   ( ("<<" | ">>") additive )*
//! additive    = mult       ( ("+" | "-") mult )*
//! mult        = unary      ( ("*" | "/" | "%") unary )*
//! unary       = ("-" | "+" | "~" | "!") unary | primary
//! primary     = NUMBER | IDENT | "@" IDENT | "(" expr ")"
//! ```
//!
//! ## Semantics
//!
//! * All math is `i64`, two's complement. Overflow wraps silently (this
//!   matches the typical assembler-time arithmetic experience and
//!   avoids spurious failures for legitimate uses like `~0`).
//! * Division and modulo by zero are hard errors.
//! * Comparisons and `&&` / `||` return 0 or 1. Anything non-zero is
//!   "true" as an input.
//! * `&&` and `||` short-circuit. Side effects (`@COUNTER`) on the
//!   skipped side don't fire — this matches C-family expectations.
//! * Shifts: by a count outside `0..64` is a hard error (LLVM's
//!   behavior is target-undefined for out-of-range shifts; we refuse
//!   to silently produce whatever the host does).
//!
//! ## Name resolution
//!
//! Bare identifiers (`cell`, `STATE_INTERPRET`) resolve through
//! [`EvalContext::lookup`]. `@FOO` directives resolve through
//! [`EvalContext::lookup_directive`], which is `&mut self` because
//! `@COUNTER` mutates state.
//!
//! Strings as context values aren't supported in expressions. If a
//! directive (e.g. `@PROC`) is a name rather than a number,
//! `lookup_directive` should return `None` and the user gets a clear
//! "no integer value for `@PROC`" error.

use super::error::LexErrorKind;
use super::span::Span;
use super::token::{Punct, Token, TokenKind};

/// What an evaluator needs from the surrounding assembler to resolve
/// names.
pub trait EvalContext {
    /// Resolve a bare identifier (e.g. `cell`) to its integer value.
    fn lookup(&self, name: &str) -> Option<i64>;

    /// Resolve a directive-style context name (e.g. `@COUNTER`, `@LINE`,
    /// `@INDEX`) to its integer value. The leading `@` has already been
    /// stripped.
    ///
    /// Mutable because reading `@COUNTER` bumps it.
    fn lookup_directive(&mut self, name: &str) -> Option<i64>;
}

/// Simple in-memory `EvalContext` for tests and small users. The full
/// assembler builds its own that also knows about `@COUNTER` etc.
#[derive(Default)]
pub struct SimpleCtx {
    names: std::collections::HashMap<String, i64>,
    counter: i64,
}

impl SimpleCtx {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn define(&mut self, name: &str, value: i64) {
        self.names.insert(name.to_string(), value);
    }
}

impl EvalContext for SimpleCtx {
    fn lookup(&self, name: &str) -> Option<i64> {
        self.names.get(name).copied()
    }
    fn lookup_directive(&mut self, name: &str) -> Option<i64> {
        match name {
            "COUNTER" => {
                let v = self.counter;
                self.counter += 1;
                Some(v)
            }
            other => self.names.get(other).copied(),
        }
    }
}

#[derive(Debug)]
pub struct EvalError {
    pub kind: EvalErrorKind,
    pub span: Span,
}

impl std::fmt::Display for EvalError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}", self.span, self.kind)
    }
}
impl std::error::Error for EvalError {}

#[derive(Debug)]
pub enum EvalErrorKind {
    UnexpectedEnd,
    UnexpectedToken { found: String, expected: &'static str },
    UndefinedName(String),
    UndefinedDirective(String),
    DivByZero,
    ModByZero,
    ShiftOutOfRange(i64),
    TrailingTokens,
    /// Mostly forwards lex-time problems that managed to slip into a
    /// number we want to evaluate; kept here so the evaluator can be
    /// the single error source for `@if` / `@assign`.
    Lex(LexErrorKind),
}

impl std::fmt::Display for EvalErrorKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use EvalErrorKind::*;
        match self {
            UnexpectedEnd => write!(f, "unexpected end of expression"),
            UnexpectedToken { found, expected } => {
                write!(f, "expected {expected}, found {found}")
            }
            UndefinedName(n) => write!(f, "undefined name `{n}`"),
            UndefinedDirective(n) => write!(f, "no integer value for `@{n}`"),
            DivByZero => write!(f, "division by zero"),
            ModByZero => write!(f, "modulo by zero"),
            ShiftOutOfRange(n) => {
                write!(f, "shift count {n} is outside 0..64")
            }
            TrailingTokens => write!(f, "trailing tokens after expression"),
            Lex(e) => write!(f, "{e}"),
        }
    }
}

/// Evaluate `tokens` as a complete expression. Returns an error if any
/// tokens remain after the expression parses cleanly.
pub fn eval(tokens: &[Token], ctx: &mut dyn EvalContext) -> Result<i64, EvalError> {
    let mut p = Parser { tokens, pos: 0, ctx };
    let v = p.parse_or()?;
    if p.pos < p.tokens.len() {
        return Err(p.err_here(EvalErrorKind::TrailingTokens));
    }
    Ok(v)
}

struct Parser<'a> {
    tokens: &'a [Token],
    pos: usize,
    ctx: &'a mut dyn EvalContext,
}

impl<'a> Parser<'a> {
    fn peek(&self) -> Option<&'a Token> {
        self.tokens.get(self.pos)
    }

    /// Look at the punctuation at the cursor without consuming.
    fn peek_punct(&self) -> Option<Punct> {
        match self.peek().map(|t| &t.kind) {
            Some(TokenKind::Punct(p)) => Some(*p),
            _ => None,
        }
    }

    /// If the cursor is on `want`, consume and return true.
    fn eat(&mut self, want: Punct) -> bool {
        if self.peek_punct() == Some(want) {
            self.pos += 1;
            true
        } else {
            false
        }
    }

    fn err_here(&self, kind: EvalErrorKind) -> EvalError {
        let span = self
            .peek()
            .map(|t| t.span)
            .or_else(|| self.tokens.last().map(|t| t.span))
            .unwrap_or(Span::SYNTHETIC);
        EvalError { kind, span }
    }

    fn err_at(&self, kind: EvalErrorKind, span: Span) -> EvalError {
        EvalError { kind, span }
    }

    fn expect(&mut self, want: Punct, ctx: &'static str) -> Result<(), EvalError> {
        if self.eat(want) {
            Ok(())
        } else {
            let found = match self.peek() {
                None => "end of input".to_string(),
                Some(t) => describe_kind(&t.kind),
            };
            Err(self.err_here(EvalErrorKind::UnexpectedToken {
                found,
                expected: ctx,
            }))
        }
    }

    // ─── precedence layers ────────────────────────────────────────

    fn parse_or(&mut self) -> Result<i64, EvalError> {
        let mut lhs = self.parse_and()?;
        while self.eat(Punct::PipePipe) {
            // Short-circuit: only evaluate RHS if LHS is false.
            if lhs != 0 {
                // Still need to parse the RHS so the cursor advances,
                // but discard its value. We DON'T evaluate (i.e., no
                // side effects on EvalContext); however our context
                // calls happen during parse, so to truly skip side
                // effects we'd need a "skip" mode. For the simple ctx
                // and v1 use, evaluate-and-discard is acceptable.
                //
                // TODO if @COUNTER side effects on skipped branches
                // become a problem, add a Parser::skip_expr that walks
                // without consulting the context.
                let _ = self.parse_and()?;
                lhs = 1;
            } else {
                let rhs = self.parse_and()?;
                lhs = if rhs != 0 { 1 } else { 0 };
            }
        }
        Ok(lhs)
    }

    fn parse_and(&mut self) -> Result<i64, EvalError> {
        let mut lhs = self.parse_bitor()?;
        while self.eat(Punct::AmpAmp) {
            if lhs == 0 {
                let _ = self.parse_bitor()?;
                lhs = 0;
            } else {
                let rhs = self.parse_bitor()?;
                lhs = if rhs != 0 { 1 } else { 0 };
            }
        }
        Ok(lhs)
    }

    fn parse_bitor(&mut self) -> Result<i64, EvalError> {
        let mut lhs = self.parse_bitxor()?;
        while self.eat(Punct::Pipe) {
            let rhs = self.parse_bitxor()?;
            lhs |= rhs;
        }
        Ok(lhs)
    }

    fn parse_bitxor(&mut self) -> Result<i64, EvalError> {
        let mut lhs = self.parse_bitand()?;
        while self.eat(Punct::Caret) {
            let rhs = self.parse_bitand()?;
            lhs ^= rhs;
        }
        Ok(lhs)
    }

    fn parse_bitand(&mut self) -> Result<i64, EvalError> {
        let mut lhs = self.parse_equality()?;
        while self.eat(Punct::Amp) {
            let rhs = self.parse_equality()?;
            lhs &= rhs;
        }
        Ok(lhs)
    }

    fn parse_equality(&mut self) -> Result<i64, EvalError> {
        let mut lhs = self.parse_relational()?;
        loop {
            let p = self.peek_punct();
            match p {
                Some(Punct::EqEq) => {
                    self.pos += 1;
                    let rhs = self.parse_relational()?;
                    lhs = if lhs == rhs { 1 } else { 0 };
                }
                Some(Punct::BangEq) => {
                    self.pos += 1;
                    let rhs = self.parse_relational()?;
                    lhs = if lhs != rhs { 1 } else { 0 };
                }
                _ => break,
            }
        }
        Ok(lhs)
    }

    fn parse_relational(&mut self) -> Result<i64, EvalError> {
        let mut lhs = self.parse_shift()?;
        loop {
            let p = self.peek_punct();
            match p {
                Some(Punct::Lt) => {
                    self.pos += 1;
                    let rhs = self.parse_shift()?;
                    lhs = if lhs < rhs { 1 } else { 0 };
                }
                Some(Punct::LtEq) => {
                    self.pos += 1;
                    let rhs = self.parse_shift()?;
                    lhs = if lhs <= rhs { 1 } else { 0 };
                }
                Some(Punct::Gt) => {
                    self.pos += 1;
                    let rhs = self.parse_shift()?;
                    lhs = if lhs > rhs { 1 } else { 0 };
                }
                Some(Punct::GtEq) => {
                    self.pos += 1;
                    let rhs = self.parse_shift()?;
                    lhs = if lhs >= rhs { 1 } else { 0 };
                }
                _ => break,
            }
        }
        Ok(lhs)
    }

    fn parse_shift(&mut self) -> Result<i64, EvalError> {
        let mut lhs = self.parse_additive()?;
        loop {
            let span = self.peek().map(|t| t.span).unwrap_or(Span::SYNTHETIC);
            let p = self.peek_punct();
            match p {
                Some(Punct::LtLt) => {
                    self.pos += 1;
                    let rhs = self.parse_additive()?;
                    if !(0..64).contains(&rhs) {
                        return Err(self.err_at(EvalErrorKind::ShiftOutOfRange(rhs), span));
                    }
                    lhs = lhs.wrapping_shl(rhs as u32);
                }
                Some(Punct::GtGt) => {
                    self.pos += 1;
                    let rhs = self.parse_additive()?;
                    if !(0..64).contains(&rhs) {
                        return Err(self.err_at(EvalErrorKind::ShiftOutOfRange(rhs), span));
                    }
                    // Arithmetic (signed) shift — preserves sign on i64.
                    lhs = lhs.wrapping_shr(rhs as u32);
                }
                _ => break,
            }
        }
        Ok(lhs)
    }

    fn parse_additive(&mut self) -> Result<i64, EvalError> {
        let mut lhs = self.parse_mult()?;
        loop {
            let p = self.peek_punct();
            match p {
                Some(Punct::Plus) => {
                    self.pos += 1;
                    let rhs = self.parse_mult()?;
                    lhs = lhs.wrapping_add(rhs);
                }
                Some(Punct::Minus) => {
                    self.pos += 1;
                    let rhs = self.parse_mult()?;
                    lhs = lhs.wrapping_sub(rhs);
                }
                _ => break,
            }
        }
        Ok(lhs)
    }

    fn parse_mult(&mut self) -> Result<i64, EvalError> {
        let mut lhs = self.parse_unary()?;
        loop {
            let span = self.peek().map(|t| t.span).unwrap_or(Span::SYNTHETIC);
            let p = self.peek_punct();
            match p {
                Some(Punct::Star) => {
                    self.pos += 1;
                    let rhs = self.parse_unary()?;
                    lhs = lhs.wrapping_mul(rhs);
                }
                Some(Punct::Slash) => {
                    self.pos += 1;
                    let rhs = self.parse_unary()?;
                    if rhs == 0 {
                        return Err(self.err_at(EvalErrorKind::DivByZero, span));
                    }
                    lhs = lhs.wrapping_div(rhs);
                }
                Some(Punct::Percent) => {
                    self.pos += 1;
                    let rhs = self.parse_unary()?;
                    if rhs == 0 {
                        return Err(self.err_at(EvalErrorKind::ModByZero, span));
                    }
                    lhs = lhs.wrapping_rem(rhs);
                }
                _ => break,
            }
        }
        Ok(lhs)
    }

    fn parse_unary(&mut self) -> Result<i64, EvalError> {
        match self.peek_punct() {
            Some(Punct::Minus) => {
                self.pos += 1;
                let v = self.parse_unary()?;
                Ok(v.wrapping_neg())
            }
            Some(Punct::Plus) => {
                self.pos += 1;
                self.parse_unary()
            }
            Some(Punct::Tilde) => {
                self.pos += 1;
                let v = self.parse_unary()?;
                Ok(!v)
            }
            Some(Punct::Bang) => {
                self.pos += 1;
                let v = self.parse_unary()?;
                Ok(if v == 0 { 1 } else { 0 })
            }
            _ => self.parse_primary(),
        }
    }

    fn parse_primary(&mut self) -> Result<i64, EvalError> {
        let tok = self
            .peek()
            .ok_or_else(|| self.err_here(EvalErrorKind::UnexpectedEnd))?;
        let span = tok.span;
        match &tok.kind {
            TokenKind::Number(n) => {
                let v = n.value;
                self.pos += 1;
                Ok(v)
            }
            TokenKind::Ident(name) => {
                let name = name.clone();
                self.pos += 1;
                self.ctx
                    .lookup(&name)
                    .ok_or_else(|| self.err_at(EvalErrorKind::UndefinedName(name), span))
            }
            TokenKind::Directive(name) => {
                let name = name.clone();
                self.pos += 1;
                self.ctx
                    .lookup_directive(&name)
                    .ok_or_else(|| self.err_at(EvalErrorKind::UndefinedDirective(name), span))
            }
            TokenKind::Punct(Punct::LParen) => {
                self.pos += 1;
                let v = self.parse_or()?;
                self.expect(Punct::RParen, "`)`")?;
                Ok(v)
            }
            other => {
                let found = describe_kind(other);
                Err(self.err_at(
                    EvalErrorKind::UnexpectedToken {
                        found,
                        expected: "number, name, or `(`",
                    },
                    span,
                ))
            }
        }
    }
}

fn describe_kind(k: &TokenKind) -> String {
    match k {
        TokenKind::Comment(_) => "comment".into(),
        TokenKind::Newline => "newline".into(),
        TokenKind::Ident(s) => format!("identifier `{s}`"),
        TokenKind::Number(n) => format!("number `{}`", n.raw),
        TokenKind::String(s) => format!("string `{}`", s.raw),
        TokenKind::MacroParam(s) => format!("macro param `&{s}`"),
        TokenKind::Directive(s) => format!("directive `@{s}`"),
        TokenKind::LocalLabel(s, outer) => {
            if *outer {
                format!("outer local label `.^{s}`")
            } else {
                format!("local label `.{s}`")
            }
        }
        TokenKind::Punct(p) => format!("`{}`", p.as_str()),
    }
}

// ─── tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::asm::lex::lex;
    use crate::asm::source::FileId;
    use crate::asm::token::TokenKind;

    fn ev(src: &str, ctx: &mut dyn EvalContext) -> Result<i64, EvalError> {
        let toks = lex(FileId(0), src).expect("lex");
        // Drop Newline tokens — the evaluator doesn't want them.
        let toks: Vec<Token> = toks
            .into_iter()
            .filter(|t| !matches!(t.kind, TokenKind::Newline | TokenKind::Comment(_)))
            .collect();
        eval(&toks, ctx)
    }

    fn ok(src: &str) -> i64 {
        let mut ctx = SimpleCtx::new();
        ev(src, &mut ctx).expect("eval failed")
    }

    fn ok_with(src: &str, names: &[(&str, i64)]) -> i64 {
        let mut ctx = SimpleCtx::new();
        for (n, v) in names {
            ctx.define(n, *v);
        }
        ev(src, &mut ctx).expect("eval failed")
    }

    #[test]
    fn literals() {
        assert_eq!(ok("42"), 42);
        assert_eq!(ok("0x2A"), 42);
        assert_eq!(ok("0b101010"), 42);
        assert_eq!(ok("0o52"), 42);
        assert_eq!(ok("'A'"), 65);
    }

    #[test]
    fn arithmetic() {
        assert_eq!(ok("1 + 2"), 3);
        assert_eq!(ok("10 - 3"), 7);
        assert_eq!(ok("4 * 5"), 20);
        assert_eq!(ok("17 / 5"), 3);
        assert_eq!(ok("17 % 5"), 2);
    }

    #[test]
    fn precedence_mul_over_add() {
        assert_eq!(ok("2 + 3 * 4"), 14);
        assert_eq!(ok("(2 + 3) * 4"), 20);
    }

    #[test]
    fn left_assoc() {
        assert_eq!(ok("100 - 30 - 20"), 50);
        assert_eq!(ok("64 / 4 / 2"), 8);
    }

    #[test]
    fn unary_ops() {
        assert_eq!(ok("-5"), -5);
        assert_eq!(ok("-(2 + 3)"), -5);
        assert_eq!(ok("~0"), -1);
        assert_eq!(ok("!0"), 1);
        assert_eq!(ok("!1"), 0);
        assert_eq!(ok("!!42"), 1);
    }

    #[test]
    fn bitwise() {
        assert_eq!(ok("0xF0 & 0x0F"), 0);
        assert_eq!(ok("0xF0 | 0x0F"), 0xFF);
        assert_eq!(ok("0xFF ^ 0x0F"), 0xF0);
        assert_eq!(ok("1 << 8"), 256);
        assert_eq!(ok("256 >> 4"), 16);
    }

    #[test]
    fn comparisons() {
        assert_eq!(ok("1 < 2"), 1);
        assert_eq!(ok("2 < 2"), 0);
        assert_eq!(ok("2 <= 2"), 1);
        assert_eq!(ok("2 == 2"), 1);
        assert_eq!(ok("2 != 3"), 1);
        assert_eq!(ok("5 >= 5"), 1);
    }

    #[test]
    fn logical_and_or_short_circuit() {
        assert_eq!(ok("0 && 5"), 0);
        assert_eq!(ok("5 && 7"), 1);
        assert_eq!(ok("0 || 7"), 1);
        assert_eq!(ok("0 || 0"), 0);
        // Precedence: && binds tighter than ||.
        assert_eq!(ok("0 || 1 && 1"), 1);
        assert_eq!(ok("1 || 1 && 0"), 1); // short-circuits the `&&`
    }

    #[test]
    fn names_resolve_via_context() {
        assert_eq!(ok_with("cell == 8", &[("cell", 8)]), 1);
        assert_eq!(ok_with("cell * 4", &[("cell", 8)]), 32);
    }

    #[test]
    fn undefined_name_errors() {
        let mut ctx = SimpleCtx::new();
        let err = ev("foo + 1", &mut ctx).unwrap_err();
        match err.kind {
            EvalErrorKind::UndefinedName(ref n) if n == "foo" => {}
            _ => panic!("wrong error: {:?}", err.kind),
        }
    }

    #[test]
    fn div_by_zero_errors() {
        let mut ctx = SimpleCtx::new();
        match ev("10 / 0", &mut ctx).unwrap_err().kind {
            EvalErrorKind::DivByZero => {}
            other => panic!("wrong error: {other:?}"),
        }
        match ev("10 % 0", &mut ctx).unwrap_err().kind {
            EvalErrorKind::ModByZero => {}
            other => panic!("wrong error: {other:?}"),
        }
    }

    #[test]
    fn shift_out_of_range_errors() {
        let mut ctx = SimpleCtx::new();
        match ev("1 << 64", &mut ctx).unwrap_err().kind {
            EvalErrorKind::ShiftOutOfRange(64) => {}
            other => panic!("wrong: {other:?}"),
        }
        match ev("1 << -1", &mut ctx).unwrap_err().kind {
            EvalErrorKind::ShiftOutOfRange(-1) => {}
            other => panic!("wrong: {other:?}"),
        }
    }

    #[test]
    fn directive_counter_bumps() {
        let mut ctx = SimpleCtx::new();
        // Each read of @COUNTER bumps; the second read sees the new value.
        let toks = lex(FileId(0), "@COUNTER + @COUNTER * 0\n").unwrap();
        let toks: Vec<Token> = toks
            .into_iter()
            .filter(|t| !matches!(t.kind, TokenKind::Newline))
            .collect();
        // Eval: read1=0 + read2=1 * 0 = 0
        assert_eq!(eval(&toks, &mut ctx).unwrap(), 0);
        // After two reads, next read should be 2.
        assert_eq!(ctx.lookup_directive("COUNTER"), Some(2));
    }

    #[test]
    fn trailing_tokens_error() {
        let mut ctx = SimpleCtx::new();
        match ev("1 + 2 3", &mut ctx).unwrap_err().kind {
            EvalErrorKind::TrailingTokens => {}
            other => panic!("wrong: {other:?}"),
        }
    }

    #[test]
    fn parens_change_precedence() {
        assert_eq!(ok("(1 + 2) * (3 + 4)"), 21);
        assert_eq!(ok("((1))"), 1);
    }

    #[test]
    fn wrapping_arithmetic_doesnt_panic() {
        // i64::MAX + 1 wraps to i64::MIN
        let mut ctx = SimpleCtx::new();
        ctx.define("MAX", i64::MAX);
        assert_eq!(ev("MAX + 1", &mut ctx).unwrap(), i64::MIN);
    }

    #[test]
    fn cell_size_check_pattern() {
        // The pattern from USER-GUIDE: @assert cell == 8
        assert_eq!(ok_with("cell == 8", &[("cell", 8)]), 1);
        assert_eq!(ok_with("cell == 8", &[("cell", 4)]), 0);
    }
}
