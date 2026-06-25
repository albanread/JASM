//! Token stream → MC-flavor assembly text.
//!
//! The expander produces a token sequence with all wfasm directives and
//! macros resolved. The emitter walks that sequence and writes a
//! string that LLVM's integrated assembler can consume.
//!
//! ## Spacing
//!
//! Token-level whitespace was discarded at lex time and replaced with a
//! per-token `space_before` bool. The emitter respects that bool: a
//! single space when set, nothing when not. Source like `mov rax,rcx`
//! round-trips with no space after the comma; `mov rax, rcx` keeps the
//! space. GAS-style `1b` stays glued because the `b` token has
//! `space_before = false` after the `1`.
//!
//! ## What stays, what goes
//!
//! | Token kind     | Emitted as                                              |
//! |----------------|---------------------------------------------------------|
//! | `Newline`      | `'\n'`                                                  |
//! | `Comment`      | dropped (no MC meaning)                                 |
//! | `Ident(s)`     | `s`                                                     |
//! | `Number`       | `raw` (preserves `0x2A` vs `42`)                        |
//! | `String`       | `raw` (preserves quote form and escapes)                |
//! | `LocalLabel(s)`| `.s` — for MC pass-through directives like `.globl`     |
//! | `Punct(p)`     | `p.as_str()`                                            |
//! | `MacroParam`   | error: should have been substituted by the expander     |
//! | `Directive`    | error: should have been resolved by the expander        |
//!
//! Unresolved `MacroParam` or `Directive` tokens in the input are bugs
//! upstream — the expander is supposed to remove them. We surface them
//! as `EmitError` rather than silently writing `&foo` or `@COUNTER`
//! into MC input.

use super::span::Span;
use super::token::{Token, TokenKind};

#[derive(Debug)]
pub struct EmitError {
    pub kind: EmitErrorKind,
    pub span: Span,
}

impl std::fmt::Display for EmitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: emit error: {}", self.span, self.kind)
    }
}
impl std::error::Error for EmitError {}

#[derive(Debug)]
pub enum EmitErrorKind {
    /// `&name` reached the emitter — the expander must have missed it.
    UnresolvedMacroParam(String),
    /// `@name` reached the emitter — same.
    UnresolvedDirective(String),
}

impl std::fmt::Display for EmitErrorKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EmitErrorKind::UnresolvedMacroParam(n) => {
                write!(f, "macro parameter `&{n}` left unresolved by the expander")
            }
            EmitErrorKind::UnresolvedDirective(n) => {
                write!(f, "directive `@{n}` left unresolved by the expander")
            }
        }
    }
}

/// Emit `tokens` as MC-ready assembly text.
pub fn emit(tokens: &[Token]) -> Result<String, EmitError> {
    let mut out = String::with_capacity(tokens.len() * 4);
    let mut at_line_start = true;
    for tok in tokens {
        match &tok.kind {
            TokenKind::Newline => {
                out.push('\n');
                at_line_start = true;
                continue;
            }
            TokenKind::Comment(_) => {
                // Comments are stripped — they carry no MC semantics.
                // (Later we could pass them through as `#` comments,
                //  which MC accepts, for round-trippable disassembly.)
                continue;
            }
            TokenKind::MacroParam(n) => {
                return Err(EmitError {
                    kind: EmitErrorKind::UnresolvedMacroParam(n.clone()),
                    span: tok.span,
                });
            }
            TokenKind::Directive(n) => {
                return Err(EmitError {
                    kind: EmitErrorKind::UnresolvedDirective(n.clone()),
                    span: tok.span,
                });
            }
            _ => {}
        }
        // For non-newline / non-comment tokens, optionally emit a
        // separating space, then the token's text.
        if !at_line_start && tok.space_before {
            out.push(' ');
        }
        write_token(&mut out, &tok.kind);
        at_line_start = false;
    }
    Ok(out)
}

fn write_token(out: &mut String, kind: &TokenKind) {
    match kind {
        TokenKind::Ident(s) => out.push_str(s),
        TokenKind::Number(n) => out.push_str(&n.raw),
        TokenKind::String(s) => out.push_str(&s.raw),
        TokenKind::LocalLabel(s, outer) => {
            out.push('.');
            if *outer {
                out.push('^');
            }
            out.push_str(s);
        }
        TokenKind::Punct(p) => out.push_str(p.as_str()),
        // Handled separately above:
        TokenKind::Newline | TokenKind::Comment(_) => {}
        TokenKind::MacroParam(_) | TokenKind::Directive(_) => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::asm::lex::lex;
    use crate::asm::source::FileId;

    fn round_trip(src: &str) -> String {
        let toks = lex(FileId(0), src).expect("lex");
        emit(&toks).expect("emit")
    }

    #[test]
    fn instruction_round_trips() {
        let s = round_trip("mov rax, 42\nret\n");
        assert_eq!(s, "mov rax, 42\nret\n");
    }

    #[test]
    fn no_space_after_comma_preserved() {
        let s = round_trip("mov rax,rcx\n");
        assert_eq!(s, "mov rax,rcx\n");
    }

    #[test]
    fn space_after_comma_preserved() {
        let s = round_trip("mov rax, rcx\n");
        assert_eq!(s, "mov rax, rcx\n");
    }

    #[test]
    fn gas_numeric_label_glued() {
        // `jmp 1b` — the `b` after `1` is glued (no space).
        let s = round_trip("jmp 1b\n");
        assert_eq!(s, "jmp 1b\n");
    }

    #[test]
    fn mc_directive_passes_through() {
        let s = round_trip(".globl forth_main\n");
        assert_eq!(s, ".globl forth_main\n");
    }

    #[test]
    fn label_definition() {
        // Leading whitespace on a line collapses — the first token of
        // each line has `space_before = false`. MC doesn't care about
        // indentation, so this is the right call.
        let s = round_trip("forth_main:\n    ret\n");
        assert_eq!(s, "forth_main:\nret\n");
    }

    #[test]
    fn comments_dropped() {
        let s = round_trip("mov rax, 42 ; the answer\nret\n");
        assert_eq!(s, "mov rax, 42\nret\n");
    }

    #[test]
    fn hex_literal_preserves_form() {
        let s = round_trip("mov rax, 0xDEAD_BEEF\n");
        assert_eq!(s, "mov rax, 0xDEAD_BEEF\n");
    }

    #[test]
    fn macro_param_unresolved_errors() {
        let mut asm = crate::asm::Assembler::new();
        // Force a MacroParam token into the output (this shouldn't
        // happen via Assembler::expand — we construct it directly).
        let _ = asm;
        let tok = Token {
            kind: TokenKind::MacroParam("oops".into()),
            span: Span::SYNTHETIC,
            space_before: false,
        };
        let err = emit(&[tok]).unwrap_err();
        assert!(matches!(
            err.kind,
            EmitErrorKind::UnresolvedMacroParam(ref n) if n == "oops"
        ));
    }
}
