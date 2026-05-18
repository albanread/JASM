//! Assembler errors.
//!
//! `AsmError` is the top-level error for the whole assembler pipeline
//! (lex / parse / expand / emit). Each phase has its own sub-enum with
//! a `Span` so diagnostics can resolve through a `SourceMap`.
//!
//! Display impls render `path:line:col: kind` when the `SourceMap` is
//! available. For now the lexer-only build prints `file#N:line:col`
//! since the renderer doesn't yet know the map.

use super::span::Span;

#[derive(Debug)]
pub enum AsmError {
    Lex(LexError),
    Expand(super::expand::ExpandError),
    Emit(super::emit::EmitError),
    // Parse(ParseError),     // future
}

impl std::fmt::Display for AsmError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AsmError::Lex(e) => write!(f, "{e}"),
            AsmError::Expand(e) => write!(f, "{e}"),
            AsmError::Emit(e) => write!(f, "{e}"),
        }
    }
}

impl From<super::emit::EmitError> for AsmError {
    fn from(e: super::emit::EmitError) -> Self {
        AsmError::Emit(e)
    }
}

impl std::error::Error for AsmError {}

impl From<LexError> for AsmError {
    fn from(e: LexError) -> Self {
        AsmError::Lex(e)
    }
}

#[derive(Debug)]
pub struct LexError {
    pub kind: LexErrorKind,
    pub span: Span,
}

impl std::fmt::Display for LexError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: lex error: {}", self.span, self.kind)
    }
}

impl std::error::Error for LexError {}

#[derive(Debug)]
pub enum LexErrorKind {
    /// `"...` with no closing `"` before end of file or newline.
    UnterminatedString,
    /// `'…` with no closing `'` before end of line.
    UnterminatedCharLit,
    /// `'ab'` (more than one char), `''` (zero chars), or unhandled escape.
    BadCharLit(String),
    /// `0xZZ`, `0b12`, `0o9` — digit outside the base.
    BadDigit { base: &'static str, raw: String },
    /// Numeric literal that overflows `i64`.
    NumberOverflow(String),
    /// `\?` inside a string or char literal where `?` isn't an
    /// implemented escape.
    BadEscape(char),
    /// `&` not followed by an identifier and not followed by `&` to
    /// form `&&`.
    StrayAmpersand,
    /// `@` not followed by an identifier — reserved sigil, but `@`
    /// alone has no meaning.
    StrayAt,
    /// A character the lexer doesn't recognize at all. Most input is
    /// passed through to MC, but the lexer needs each char to fit
    /// *some* token class.
    StrayChar(char),
    /// `&a&b` — paste operator unspecified between two parameter
    /// substitutions. Use `&a##&b`.
    AmbiguousParamGlue,
}

impl std::fmt::Display for LexErrorKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use LexErrorKind::*;
        match self {
            UnterminatedString => write!(f, "unterminated string literal"),
            UnterminatedCharLit => write!(f, "unterminated character literal"),
            BadCharLit(s) => write!(f, "bad character literal `{s}`"),
            BadDigit { base, raw } => {
                write!(f, "invalid digit in {base} literal `{raw}`")
            }
            NumberOverflow(s) => write!(f, "numeric literal `{s}` overflows i64"),
            BadEscape(c) => write!(f, "bad escape sequence `\\{c}`"),
            StrayAmpersand => write!(
                f,
                "stray `&` — must be followed by an identifier (for &name) or another `&` (for &&)"
            ),
            StrayAt => write!(
                f,
                "stray `@` — must be followed by an identifier (e.g. `@scope`, `@PROC`)"
            ),
            StrayChar(c) => write!(f, "unrecognized character `{}` (0x{:02X})", c, *c as u32),
            AmbiguousParamGlue => write!(
                f,
                "ambiguous `&NAME&NAME` — use `&a##&b` for paste or insert whitespace"
            ),
        }
    }
}
