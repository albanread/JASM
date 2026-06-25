//! Token types.
//!
//! The lexer produces a flat `Vec<Token>` per file. `Newline` tokens are
//! preserved (the language is line-oriented) but interior whitespace is
//! not — instead, each token records whether whitespace separated it
//! from its predecessor (`space_before`). That's enough to emit
//! readable output without carrying a `Whitespace` variant.
//!
//! Comments are preserved as `Comment` tokens because they're useful for
//! source-map-aware diagnostics and (later) for pass-through to MC's
//! output for round-trippable disassembly.

use super::span::Span;

#[derive(Debug, Clone)]
pub struct Token {
    pub kind: TokenKind,
    pub span: Span,
    /// True if at least one space/tab separated this token from the
    /// previous token *on the same line*. Newlines set the flag back to
    /// false for the first token of the next line — start-of-line is
    /// "no space before" by convention.
    pub space_before: bool,
}

impl Token {
    pub fn new(kind: TokenKind, span: Span, space_before: bool) -> Self {
        Self { kind, span, space_before }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TokenKind {
    /// `; …` to end of line. Stored without the leading `;`. The lexer
    /// strips a single leading space if present (so `; foo` and `;foo`
    /// both store `foo`).
    Comment(String),

    /// `\n` (or `\r\n`). Structural — directives and statements are
    /// line-bounded.
    Newline,

    /// `[A-Za-z_][A-Za-z0-9_]*`. Mnemonics, register names, label names
    /// (when followed by `:`), macro names, defined constants — all
    /// flow through this single variant. The parser decides which.
    Ident(String),

    /// Numeric literal — value plus the original text so we can emit it
    /// back to MC verbatim if we never need to do arithmetic on it.
    Number(NumberLit),

    /// String literal in double quotes. Stores both the unescaped value
    /// and the original raw text including quotes.
    String(StringLit),

    /// `&name` — macro parameter substitution slot. Stored without the
    /// leading `&`.
    MacroParam(String),

    /// `@name` — wfasm directive or built-in context name. Stored
    /// without the leading `@`. The parser distinguishes directive
    /// names (`@scope`, `@if`, `@macro`, `@include`, etc.) from
    /// context names (`@PROC`, `@COUNTER`, `@LINE`, etc.) — they share
    /// this token type.
    Directive(String),

    /// `.name` — local label reference. Stored without the leading
    /// dot. The parser checks whether it's a definition (followed by
    /// `:`) or a reference, and applies scope/macro mangling.
    ///
    /// The `bool` (`outer`) is true for `.^name` — a label reference
    /// that should resolve in the innermost enclosing `@scope` frame,
    /// skipping over any macro-invocation frames on top of it. This
    /// lets a macro body jump to labels defined in the calling proc.
    LocalLabel(String, bool),

    /// Punctuation token. Multi-char punctuation is lexed greedily.
    Punct(Punct),
}

/// Numeric literal. Examples: `42`, `0x2A`, `0b101010`, `0o52`, `'A'`,
/// `'\n'`. The lexer parses the value; `raw` keeps the original text
/// for verbatim re-emission and diagnostics.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NumberLit {
    pub value: i64,
    pub raw: String,
    pub base: NumberBase,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum NumberBase {
    Dec,
    Hex,
    Bin,
    Oct,
    Char,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StringLit {
    /// Unescaped contents.
    pub value: String,
    /// Original text including the surrounding `"` quotes and any
    /// escape sequences.
    pub raw: String,
}

/// Punctuation tokens. Multi-character forms are lexed as a single
/// token (e.g. `==`, `<<`, `&&`, `...`).
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum Punct {
    Colon,        // :
    Comma,        // ,
    LParen,       // (
    RParen,       // )
    LBrace,       // {
    RBrace,       // }
    LBracket,     // [
    RBracket,     // ]

    Plus,         // +
    Minus,        // -
    Star,         // *
    Slash,        // /
    Percent,      // %
    Tilde,        // ~
    Bang,         // !

    Amp,          // &  (bitwise / address-of in MC operands)
    Pipe,         // |
    Caret,        // ^

    AmpAmp,       // &&  (logical AND in @if; literal `&` escape in macro bodies)
    PipePipe,     // ||

    Eq,           // =
    EqEq,         // ==
    BangEq,       // !=
    Lt,           // <
    LtEq,         // <=
    Gt,           // >
    GtEq,         // >=
    LtLt,         // <<
    GtGt,         // >>

    /// `##` — token paste operator inside macro bodies.
    HashHash,

    /// `...` — variadic-arg marker in `@macro name(args...)`.
    Ellipsis,

    Question,     // ?
    Dot,          // bare `.` (rare; mostly `.label` is its own token)
    At,           // bare `@` (error; reserved for future)
    Hash,         // bare `#` (rare; not currently meaningful, parser may emit)
}

impl Punct {
    /// Render back to source form for emission to MC.
    pub fn as_str(self) -> &'static str {
        use Punct::*;
        match self {
            Colon => ":",
            Comma => ",",
            LParen => "(",
            RParen => ")",
            LBrace => "{",
            RBrace => "}",
            LBracket => "[",
            RBracket => "]",
            Plus => "+",
            Minus => "-",
            Star => "*",
            Slash => "/",
            Percent => "%",
            Tilde => "~",
            Bang => "!",
            Amp => "&",
            Pipe => "|",
            Caret => "^",
            AmpAmp => "&&",
            PipePipe => "||",
            Eq => "=",
            EqEq => "==",
            BangEq => "!=",
            Lt => "<",
            LtEq => "<=",
            Gt => ">",
            GtEq => ">=",
            LtLt => "<<",
            GtGt => ">>",
            HashHash => "##",
            Ellipsis => "...",
            Question => "?",
            Dot => ".",
            At => "@",
            Hash => "#",
        }
    }
}
