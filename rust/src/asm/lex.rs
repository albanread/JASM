//! The lexer.
//!
//! One file at a time: text in, `Vec<Token>` out. The lexer is
//! single-pass with one byte of lookahead, byte-oriented (ASCII
//! identifiers; non-ASCII bytes inside strings and comments pass
//! through but aren't interpreted).
//!
//! Whitespace doesn't appear as a token. Instead, every emitted token
//! carries `space_before: bool` indicating whether at least one
//! space/tab separated it from the previous token *on the same line*.
//! That's enough to reconstruct `mov rax, rcx` vs `mov rax,rcx` on
//! output, and it lets `1b` (GAS numeric back-reference) round-trip:
//! the `b` token has `space_before = false` after the `1` and the
//! emitter joins them with no gap.

use super::error::{LexError, LexErrorKind};
use super::source::FileId;
use super::span::Span;
use super::token::{NumberBase, NumberLit, Punct, StringLit, Token, TokenKind};

/// Lex one file's text into a flat token stream.
pub fn lex(file: FileId, text: &str) -> Result<Vec<Token>, LexError> {
    let mut lx = Lexer::new(file, text.as_bytes());
    let mut out = Vec::new();
    while !lx.eof() {
        if let Some(tok) = lx.next_token()? {
            out.push(tok);
        }
    }
    Ok(out)
}

struct Lexer<'a> {
    text: &'a [u8],
    /// Current byte index into `text`.
    pos: usize,
    /// 1-based line number at `pos`.
    line: u32,
    /// 1-based column number at `pos`.
    col: u32,
    /// File this lexer is reading.
    file: FileId,
    /// Set by `skip_whitespace`; consumed (set to false) by
    /// `make_token`. Initialized to true so the first token of the file
    /// reads "no space before" — `make_token` flips it.
    space_pending: bool,
    /// True after a Newline token until the next non-whitespace char.
    /// Used to suppress `space_before` on the first token of each line.
    at_line_start: bool,
}

impl<'a> Lexer<'a> {
    fn new(file: FileId, text: &'a [u8]) -> Self {
        Self {
            text,
            pos: 0,
            line: 1,
            col: 1,
            file,
            space_pending: false,
            at_line_start: true,
        }
    }

    fn eof(&self) -> bool {
        self.pos >= self.text.len()
    }

    fn peek(&self) -> u8 {
        self.text[self.pos]
    }

    fn peek_at(&self, offset: usize) -> Option<u8> {
        self.text.get(self.pos + offset).copied()
    }

    fn advance(&mut self) -> u8 {
        let c = self.peek();
        self.pos += 1;
        if c == b'\n' {
            self.line += 1;
            self.col = 1;
        } else {
            self.col += 1;
        }
        c
    }

    fn span_at(&self, start_line: u32, start_col: u32, len: u32) -> Span {
        Span::new(self.file, start_line, start_col, len)
    }

    /// Skip horizontal whitespace (spaces and tabs). Track that we saw
    /// some, so the next token gets `space_before = true`.
    fn skip_h_ws(&mut self) {
        while !self.eof() {
            match self.peek() {
                b' ' | b'\t' => {
                    self.advance();
                    self.space_pending = true;
                }
                _ => break,
            }
        }
    }

    /// Make a token from kind + start position. Computes the span and
    /// resolves `space_before`.
    fn make(&mut self, kind: TokenKind, start_line: u32, start_col: u32) -> Token {
        let len = self.col.saturating_sub(start_col).max(1);
        let span = self.span_at(start_line, start_col, len);
        let space_before = self.space_pending && !self.at_line_start;
        self.space_pending = false;
        self.at_line_start = false;
        Token { kind, span, space_before }
    }

    /// Build a span for a token whose end is the current position.
    /// Reserved for multi-line tokens (multi-line strings, raw blocks)
    /// added in later phases.
    #[allow(dead_code)]
    fn span_from(&self, start_line: u32, start_col: u32) -> Span {
        let len = if self.line == start_line {
            self.col.saturating_sub(start_col).max(1)
        } else {
            // Multi-line tokens are unusual but possible (multi-line
            // strings if we ever support them). For now we just stop
            // pretending `len` means anything beyond start.
            1
        };
        self.span_at(start_line, start_col, len)
    }

    // ---- main dispatch ----------------------------------------------

    fn next_token(&mut self) -> Result<Option<Token>, LexError> {
        self.skip_h_ws();
        if self.eof() {
            return Ok(None);
        }

        let start_line = self.line;
        let start_col = self.col;
        let c = self.peek();

        // Newline (with optional preceding CR).
        if c == b'\r' || c == b'\n' {
            if c == b'\r' {
                self.advance();
                if !self.eof() && self.peek() == b'\n' {
                    self.advance();
                }
            } else {
                self.advance();
            }
            let span = self.span_at(start_line, start_col, 1);
            self.space_pending = false;
            self.at_line_start = true;
            return Ok(Some(Token {
                kind: TokenKind::Newline,
                span,
                space_before: false,
            }));
        }

        // Comment to end of line.
        if c == b';' {
            return Ok(Some(self.lex_comment(start_line, start_col)));
        }

        // String / char.
        if c == b'"' {
            return Ok(Some(self.lex_string(start_line, start_col)?));
        }
        if c == b'\'' {
            return Ok(Some(self.lex_char(start_line, start_col)?));
        }

        // Sigil-prefixed tokens.
        if c == b'&' {
            return Ok(Some(self.lex_amp_or_param(start_line, start_col)?));
        }
        if c == b'@' {
            return Ok(Some(self.lex_at(start_line, start_col)?));
        }
        if c == b'.' {
            return Ok(Some(self.lex_dot_or_local(start_line, start_col)?));
        }

        // Numbers and identifiers.
        if c.is_ascii_digit() {
            return Ok(Some(self.lex_number(start_line, start_col)?));
        }
        if c == b'_' || c.is_ascii_alphabetic() {
            return Ok(Some(self.lex_ident(start_line, start_col)));
        }

        // Punctuation.
        Ok(Some(self.lex_punct(start_line, start_col)?))
    }

    // ---- individual token kinds -------------------------------------

    fn lex_comment(&mut self, start_line: u32, start_col: u32) -> Token {
        // Consume the leading `;` plus everything to (but not including)
        // the next newline.
        self.advance(); // ;
        let body_start = self.pos;
        while !self.eof() && self.peek() != b'\n' && self.peek() != b'\r' {
            self.advance();
        }
        let raw = std::str::from_utf8(&self.text[body_start..self.pos])
            .unwrap_or("")
            .to_string();
        // Trim a single leading space for friendlier storage: `;foo`
        // and `; foo` both give "foo".
        let stored = raw.strip_prefix(' ').unwrap_or(&raw).to_string();
        self.make(TokenKind::Comment(stored), start_line, start_col)
    }

    fn lex_string(&mut self, start_line: u32, start_col: u32) -> Result<Token, LexError> {
        let start_pos = self.pos;
        self.advance(); // opening "
        let mut value = String::new();
        loop {
            if self.eof() || self.peek() == b'\n' || self.peek() == b'\r' {
                return Err(LexError {
                    kind: LexErrorKind::UnterminatedString,
                    span: self.span_at(start_line, start_col, 1),
                });
            }
            let c = self.peek();
            if c == b'"' {
                self.advance();
                break;
            }
            if c == b'\\' {
                self.advance();
                if self.eof() {
                    return Err(LexError {
                        kind: LexErrorKind::UnterminatedString,
                        span: self.span_at(start_line, start_col, 1),
                    });
                }
                let esc = self.advance();
                value.push(self.decode_escape(esc, start_line, start_col)?);
                continue;
            }
            // Non-escape, non-quote: include the byte as-is. (Non-ASCII
            // bytes flow through; we don't decode UTF-8.)
            value.push(c as char);
            self.advance();
        }
        let raw = std::str::from_utf8(&self.text[start_pos..self.pos])
            .unwrap_or("")
            .to_string();
        Ok(self.make(
            TokenKind::String(StringLit { value, raw }),
            start_line,
            start_col,
        ))
    }

    fn lex_char(&mut self, start_line: u32, start_col: u32) -> Result<Token, LexError> {
        let start_pos = self.pos;
        self.advance(); // opening '
        if self.eof() || self.peek() == b'\n' {
            return Err(LexError {
                kind: LexErrorKind::UnterminatedCharLit,
                span: self.span_at(start_line, start_col, 1),
            });
        }
        let raw_byte_start = self.pos;
        let value: i64 = if self.peek() == b'\\' {
            self.advance();
            if self.eof() {
                return Err(LexError {
                    kind: LexErrorKind::UnterminatedCharLit,
                    span: self.span_at(start_line, start_col, 1),
                });
            }
            let esc = self.advance();
            self.decode_escape(esc, start_line, start_col)? as i64
        } else {
            let c = self.advance();
            c as i64
        };
        if self.eof() || self.peek() != b'\'' {
            // Read until the closing quote or EOL for a useful error.
            while !self.eof() && self.peek() != b'\'' && self.peek() != b'\n' {
                self.advance();
            }
            let raw = std::str::from_utf8(&self.text[raw_byte_start..self.pos])
                .unwrap_or("")
                .to_string();
            return Err(LexError {
                kind: LexErrorKind::BadCharLit(raw),
                span: self.span_at(start_line, start_col, 1),
            });
        }
        self.advance(); // closing '
        let raw = std::str::from_utf8(&self.text[start_pos..self.pos])
            .unwrap_or("")
            .to_string();
        Ok(self.make(
            TokenKind::Number(NumberLit {
                value,
                raw,
                base: NumberBase::Char,
            }),
            start_line,
            start_col,
        ))
    }

    fn decode_escape(&self, esc: u8, line: u32, col: u32) -> Result<char, LexError> {
        Ok(match esc {
            b'n' => '\n',
            b'r' => '\r',
            b't' => '\t',
            b'0' => '\0',
            b'\\' => '\\',
            b'\'' => '\'',
            b'"' => '"',
            // `\xNN` would need two more bytes; supported but only here
            // not in the simple case. Implement on demand.
            other => {
                return Err(LexError {
                    kind: LexErrorKind::BadEscape(other as char),
                    span: self.span_at(line, col, 2),
                });
            }
        })
    }

    fn lex_amp_or_param(
        &mut self,
        start_line: u32,
        start_col: u32,
    ) -> Result<Token, LexError> {
        self.advance(); // &
        match self.peek_at(0) {
            // `&&` — single punct.
            Some(b'&') => {
                self.advance();
                Ok(self.make(
                    TokenKind::Punct(Punct::AmpAmp),
                    start_line,
                    start_col,
                ))
            }
            // `&name` — macro parameter substitution.
            Some(c) if c == b'_' || c.is_ascii_alphabetic() => {
                let name = self.read_ident_chars();
                Ok(self.make(
                    TokenKind::MacroParam(name),
                    start_line,
                    start_col,
                ))
            }
            // Stand-alone `&` — bitwise-AND. Useful inside @assign /
            // @if expressions and forwarded to MC for operand
            // expressions. We accept it as Punct::Amp.
            _ => Ok(self.make(
                TokenKind::Punct(Punct::Amp),
                start_line,
                start_col,
            )),
        }
    }

    fn lex_at(&mut self, start_line: u32, start_col: u32) -> Result<Token, LexError> {
        self.advance(); // @
        match self.peek_at(0) {
            Some(c) if c == b'_' || c.is_ascii_alphabetic() => {
                let name = self.read_ident_chars();
                Ok(self.make(
                    TokenKind::Directive(name),
                    start_line,
                    start_col,
                ))
            }
            _ => Err(LexError {
                kind: LexErrorKind::StrayAt,
                span: self.span_at(start_line, start_col, 1),
            }),
        }
    }

    fn lex_dot_or_local(
        &mut self,
        start_line: u32,
        start_col: u32,
    ) -> Result<Token, LexError> {
        // `...` ellipsis — three dots in a row.
        if self.peek_at(1) == Some(b'.') && self.peek_at(2) == Some(b'.') {
            self.advance();
            self.advance();
            self.advance();
            return Ok(self.make(
                TokenKind::Punct(Punct::Ellipsis),
                start_line,
                start_col,
            ));
        }
        self.advance(); // .
        match self.peek_at(0) {
            // `.name` — local label.
            Some(c) if c == b'_' || c.is_ascii_alphanumeric() => {
                let name = self.read_ident_chars();
                Ok(self.make(
                    TokenKind::LocalLabel(name),
                    start_line,
                    start_col,
                ))
            }
            // Bare `.` — pass through. MC sometimes uses `.` as the
            // current address; we don't model that, but we don't reject it.
            _ => Ok(self.make(
                TokenKind::Punct(Punct::Dot),
                start_line,
                start_col,
            )),
        }
    }

    fn lex_number(&mut self, start_line: u32, start_col: u32) -> Result<Token, LexError> {
        let start_pos = self.pos;
        let first = self.peek();
        let (base, raw, value) = if first == b'0' && matches!(
            self.peek_at(1),
            Some(b'x') | Some(b'X') | Some(b'b') | Some(b'B') | Some(b'o') | Some(b'O')
        ) {
            self.advance(); // 0
            let kind = self.advance(); // x / b / o
            match kind.to_ascii_lowercase() {
                b'x' => self.lex_radix_number(start_pos, NumberBase::Hex, 16, "hex")?,
                b'b' => self.lex_radix_number(start_pos, NumberBase::Bin, 2, "binary")?,
                b'o' => self.lex_radix_number(start_pos, NumberBase::Oct, 8, "octal")?,
                _ => unreachable!(),
            }
        } else {
            self.lex_radix_number(start_pos, NumberBase::Dec, 10, "decimal")?
        };
        Ok(self.make(
            TokenKind::Number(NumberLit { value, raw, base }),
            start_line,
            start_col,
        ))
    }

    /// Consume digits for the given radix and return (base, raw, value).
    /// Allows `_` as a separator (`0xDEAD_BEEF`).
    fn lex_radix_number(
        &mut self,
        start_pos: usize,
        base_enum: NumberBase,
        radix: u32,
        base_name: &'static str,
    ) -> Result<(NumberBase, String, i64), LexError> {
        let digits_start = self.pos;
        while !self.eof() {
            let c = self.peek();
            if c == b'_' {
                self.advance();
                continue;
            }
            let d = (c as char).to_digit(36); // accept more chars; reject later
            match d {
                Some(d) if d < radix => {
                    self.advance();
                }
                _ => break,
            }
        }
        let raw = std::str::from_utf8(&self.text[start_pos..self.pos])
            .unwrap_or("")
            .to_string();
        let digits_str = std::str::from_utf8(&self.text[digits_start..self.pos])
            .unwrap_or("")
            .replace('_', "");
        if digits_str.is_empty() {
            return Err(LexError {
                kind: LexErrorKind::BadDigit {
                    base: base_name,
                    raw: raw.clone(),
                },
                span: self.span_at(self.line, self.col, 1),
            });
        }
        let value = match i64::from_str_radix(&digits_str, radix) {
            Ok(v) => v,
            Err(e) if e.kind() == &std::num::IntErrorKind::PosOverflow => {
                return Err(LexError {
                    kind: LexErrorKind::NumberOverflow(raw),
                    span: self.span_at(self.line, self.col, 1),
                });
            }
            Err(_) => {
                return Err(LexError {
                    kind: LexErrorKind::BadDigit {
                        base: base_name,
                        raw,
                    },
                    span: self.span_at(self.line, self.col, 1),
                });
            }
        };
        Ok((base_enum, raw, value))
    }

    fn lex_ident(&mut self, start_line: u32, start_col: u32) -> Token {
        let name = self.read_ident_chars();
        self.make(TokenKind::Ident(name), start_line, start_col)
    }

    fn read_ident_chars(&mut self) -> String {
        let start = self.pos;
        while !self.eof() {
            let c = self.peek();
            if c == b'_' || c.is_ascii_alphanumeric() {
                self.advance();
            } else {
                break;
            }
        }
        std::str::from_utf8(&self.text[start..self.pos])
            .unwrap_or("")
            .to_string()
    }

    fn lex_punct(&mut self, start_line: u32, start_col: u32) -> Result<Token, LexError> {
        let c = self.advance();
        let punct = match c {
            b':' => Punct::Colon,
            b',' => Punct::Comma,
            b'(' => Punct::LParen,
            b')' => Punct::RParen,
            b'{' => Punct::LBrace,
            b'}' => Punct::RBrace,
            b'[' => Punct::LBracket,
            b']' => Punct::RBracket,
            b'+' => Punct::Plus,
            b'-' => Punct::Minus,
            b'*' => Punct::Star,
            b'/' => Punct::Slash,
            b'%' => Punct::Percent,
            b'~' => Punct::Tilde,
            b'?' => Punct::Question,
            b'^' => Punct::Caret,
            b'#' => {
                if self.peek_at(0) == Some(b'#') {
                    self.advance();
                    Punct::HashHash
                } else {
                    Punct::Hash
                }
            }
            b'|' => {
                if self.peek_at(0) == Some(b'|') {
                    self.advance();
                    Punct::PipePipe
                } else {
                    Punct::Pipe
                }
            }
            b'!' => {
                if self.peek_at(0) == Some(b'=') {
                    self.advance();
                    Punct::BangEq
                } else {
                    Punct::Bang
                }
            }
            b'=' => {
                if self.peek_at(0) == Some(b'=') {
                    self.advance();
                    Punct::EqEq
                } else {
                    Punct::Eq
                }
            }
            b'<' => match self.peek_at(0) {
                Some(b'=') => {
                    self.advance();
                    Punct::LtEq
                }
                Some(b'<') => {
                    self.advance();
                    Punct::LtLt
                }
                _ => Punct::Lt,
            },
            b'>' => match self.peek_at(0) {
                Some(b'=') => {
                    self.advance();
                    Punct::GtEq
                }
                Some(b'>') => {
                    self.advance();
                    Punct::GtGt
                }
                _ => Punct::Gt,
            },
            other => {
                return Err(LexError {
                    kind: LexErrorKind::StrayChar(other as char),
                    span: self.span_at(start_line, start_col, 1),
                });
            }
        };
        Ok(self.make(TokenKind::Punct(punct), start_line, start_col))
    }
}

// ─── tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn lex_text(text: &str) -> Vec<TokenKind> {
        let toks = lex(FileId(0), text).expect("lex error");
        toks.into_iter().map(|t| t.kind).collect()
    }

    fn lex_full(text: &str) -> Vec<Token> {
        lex(FileId(0), text).expect("lex error")
    }

    #[test]
    fn empty_file_no_tokens() {
        assert!(lex_text("").is_empty());
    }

    #[test]
    fn simple_instruction() {
        let toks = lex_text("mov rax, 42\n");
        assert_eq!(
            toks,
            vec![
                TokenKind::Ident("mov".into()),
                TokenKind::Ident("rax".into()),
                TokenKind::Punct(Punct::Comma),
                TokenKind::Number(NumberLit {
                    value: 42,
                    raw: "42".into(),
                    base: NumberBase::Dec,
                }),
                TokenKind::Newline,
            ]
        );
    }

    #[test]
    fn hex_bin_oct() {
        let toks = lex_text("0x2A 0b101010 0o52 42");
        let values: Vec<i64> = toks
            .iter()
            .filter_map(|k| match k {
                TokenKind::Number(n) => Some(n.value),
                _ => None,
            })
            .collect();
        assert_eq!(values, vec![42, 42, 42, 42]);
    }

    #[test]
    fn hex_keeps_raw_form() {
        let toks = lex_text("0xDEAD_BEEF");
        match &toks[0] {
            TokenKind::Number(n) => {
                assert_eq!(n.value, 0xDEAD_BEEF);
                assert_eq!(n.raw, "0xDEAD_BEEF");
                assert_eq!(n.base, NumberBase::Hex);
            }
            _ => panic!("expected number, got {:?}", toks[0]),
        }
    }

    #[test]
    fn char_literals() {
        let toks = lex_text("'A' '\\n' '\\\\' '\\0'");
        let values: Vec<i64> = toks
            .iter()
            .filter_map(|k| match k {
                TokenKind::Number(n) => Some(n.value),
                _ => None,
            })
            .collect();
        assert_eq!(values, vec![65, 10, 92, 0]);
    }

    #[test]
    fn strings() {
        let toks = lex_text(r#""hello\nworld""#);
        match &toks[0] {
            TokenKind::String(s) => {
                assert_eq!(s.value, "hello\nworld");
                assert_eq!(s.raw, r#""hello\nworld""#);
            }
            _ => panic!("expected string"),
        }
    }

    #[test]
    fn unterminated_string_errors() {
        let err = lex(FileId(0), "\"hello\n").unwrap_err();
        match err.kind {
            LexErrorKind::UnterminatedString => {}
            _ => panic!("wrong error"),
        }
    }

    #[test]
    fn macro_param() {
        let toks = lex_text("mov &val, rax");
        assert!(matches!(toks[1], TokenKind::MacroParam(ref n) if n == "val"));
    }

    #[test]
    fn directive() {
        let toks = lex_text("@scope foo");
        assert!(matches!(toks[0], TokenKind::Directive(ref n) if n == "scope"));
        assert!(matches!(toks[1], TokenKind::Ident(ref n) if n == "foo"));
    }

    #[test]
    fn local_label() {
        let toks = lex_text(".done:");
        assert!(matches!(toks[0], TokenKind::LocalLabel(ref n) if n == "done"));
        assert!(matches!(toks[1], TokenKind::Punct(Punct::Colon)));
    }

    #[test]
    fn comment_to_eol() {
        let toks = lex_text("mov rax, 42 ; a comment\nret");
        let comments: Vec<&str> = toks
            .iter()
            .filter_map(|k| match k {
                TokenKind::Comment(s) => Some(s.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(comments, vec!["a comment"]);
    }

    #[test]
    fn multi_char_punct() {
        let toks = lex_text("== != <= >= && || << >> ##");
        let p: Vec<Punct> = toks
            .iter()
            .filter_map(|k| match k {
                TokenKind::Punct(p) => Some(*p),
                _ => None,
            })
            .collect();
        assert_eq!(
            p,
            vec![
                Punct::EqEq,
                Punct::BangEq,
                Punct::LtEq,
                Punct::GtEq,
                Punct::AmpAmp,
                Punct::PipePipe,
                Punct::LtLt,
                Punct::GtGt,
                Punct::HashHash,
            ]
        );
    }

    #[test]
    fn ellipsis_three_dots() {
        let toks = lex_text("@macro foo(args...)\n");
        let p: Vec<Punct> = toks
            .iter()
            .filter_map(|k| match k {
                TokenKind::Punct(p) => Some(*p),
                _ => None,
            })
            .collect();
        assert!(p.contains(&Punct::Ellipsis));
    }

    #[test]
    fn space_before_tracked() {
        let toks = lex_full("mov rax,rcx");
        // mov | rax | , | rcx | (eof, no newline)
        assert_eq!(toks.len(), 4);
        assert!(!toks[0].space_before); // start of line
        assert!(toks[1].space_before); // after `mov`
        assert!(!toks[2].space_before); // `,` is glued to `rax`
        assert!(!toks[3].space_before); // `rcx` glued to `,`
    }

    #[test]
    fn space_before_after_newline_resets() {
        let toks = lex_full("a\nb");
        assert_eq!(toks.len(), 3); // a, Newline, b
        assert!(!toks[0].space_before);
        assert!(!toks[2].space_before); // first token of new line
    }

    #[test]
    fn gas_numeric_back_ref_glued() {
        // `1b` should lex as Number(1) followed by Ident(b), with the
        // `b` carrying space_before = false so the emitter can re-glue
        // them into a single GAS-style operand.
        let toks = lex_full("jmp 1b");
        assert_eq!(toks.len(), 3);
        match &toks[1].kind {
            TokenKind::Number(n) => assert_eq!(n.value, 1),
            _ => panic!(),
        }
        match &toks[2].kind {
            TokenKind::Ident(s) => assert_eq!(s, "b"),
            _ => panic!(),
        }
        assert!(!toks[2].space_before);
    }

    #[test]
    fn stray_at_errors() {
        let err = lex(FileId(0), "@\n").unwrap_err();
        assert!(matches!(err.kind, LexErrorKind::StrayAt));
    }

    #[test]
    fn underscore_separator_in_number() {
        let toks = lex_text("1_000_000");
        match &toks[0] {
            TokenKind::Number(n) => assert_eq!(n.value, 1_000_000),
            _ => panic!(),
        }
    }

    #[test]
    fn full_forth_proc_snippet() {
        let src = r#"
@scope plus
.globl plus
plus:
    add  rax, [rbp]
    add  rbp, 8
    ret
@endscope
"#;
        // Just ensure it lexes without error and produces a sensible
        // number of tokens. Detail validation is the parser's job.
        let toks = lex_full(src);
        assert!(toks.iter().any(|t| matches!(&t.kind,
            TokenKind::Directive(n) if n == "scope")));
        assert!(toks.iter().any(|t| matches!(&t.kind,
            TokenKind::Directive(n) if n == "endscope")));
        assert!(toks.iter().any(|t| matches!(&t.kind,
            TokenKind::Ident(n) if n == "plus")));
    }
}
