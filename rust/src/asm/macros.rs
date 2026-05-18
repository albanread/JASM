//! Macro and scope types.
//!
//! Lives in its own module to keep [`crate::asm::expand`] focused on
//! the expander's control flow. The data shapes here are simple and
//! self-contained; the expander reaches in and reads/clones them.
//!
//! Why a separate file: macros and scope frames are conceptually one
//! domain ("how do user-written macros and `@scope` interact with
//! local-label mangling") that is easy to discuss without the surrounding
//! expander state.

use super::span::Span;
use super::token::Token;

/// A user-defined text macro.
#[derive(Debug, Clone)]
pub struct MacroDef {
    pub name: String,
    pub params: Vec<MacroParam>,
    /// Tokens between `@macro NAME(...)` and `@endmacro`, with the
    /// `Newline` separating the header from the body trimmed. Body
    /// `Newline`s are preserved for accurate emission.
    pub body: Vec<Token>,
    /// Source span of the `@macro` directive — used in diagnostics.
    pub span: Span,
}

#[derive(Debug, Clone)]
pub struct MacroParam {
    pub name: String,
    /// True if this is the variadic terminal (`name...`). Only the
    /// final param may be variadic; the parser enforces this.
    pub variadic: bool,
}

/// One frame on the expander's scope stack. Pushed by `@scope NAME`
/// (with `kind = Scope` and `id = None`) and by macro invocations
/// (with `kind = MacroInvocation` and a unique `id`).
#[derive(Debug, Clone)]
pub struct ScopeFrame {
    pub kind: ScopeKind,
    pub name: String,
    pub id: Option<u64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScopeKind {
    /// Opened by `@scope NAME`. Closed by `@endscope`.
    Scope,
    /// Pushed for the duration of a single macro body expansion. Each
    /// invocation gets a unique `id` for per-invocation label hygiene.
    MacroInvocation,
}

impl ScopeFrame {
    /// Prefix used when mangling local labels inside this frame.
    /// `@scope foo` → `foo`; macro `bar` invocation id 7 → `bar$$7`.
    pub fn mangle_prefix(&self) -> String {
        match self.kind {
            ScopeKind::Scope => self.name.clone(),
            ScopeKind::MacroInvocation => {
                format!("{}$${}", self.name, self.id.unwrap_or(0))
            }
        }
    }
}

/// MC directive names that look like `.foo` but are *not* wfasm local
/// labels — they belong to LLVM's integrated assembler and must pass
/// through unchanged even when a `@scope` is open.
///
/// This list isn't exhaustive (LLVM MC understands many more, plus
/// per-target ones), but it covers what real-world asm sources actually
/// emit. New entries are cheap to add. The fallback for an unrecognized
/// `.foo` outside any scope is also "pass through," so a missing entry
/// only matters inside a `@scope`/`@macro` body.
pub const MC_DIRECTIVE_NAMES: &[&str] = &[
    // Syntax mode
    "intel_syntax",
    "att_syntax",
    // Sections
    "text",
    "data",
    "rodata",
    "bss",
    "section",
    "subsection",
    "popsection",
    "previous",
    // Symbol attributes
    "globl",
    "global",
    "weak",
    "local",
    "hidden",
    "internal",
    "protected",
    "type",
    "size",
    "comm",
    "lcomm",
    // Data emission
    "byte",
    "short",
    "word",
    "long",
    "quad",
    "octa",
    "single",
    "double",
    "asciz",
    "ascii",
    "string",
    "zero",
    "skip",
    "fill",
    "space",
    // Alignment / org
    "balign",
    "p2align",
    "align",
    "org",
    // Symbol aliases
    "set",
    "equ",
    "equiv",
    // Code-mode switching
    "code64",
    "code32",
    "code16",
    "arch",
    "arch_extension",
    "ident",
    // Source position
    "file",
    "line",
    "loc",
];

/// True if `name` is one of LLVM MC's recognized directive names
/// (`globl`, `text`, `quad`, etc.) or one of the patterned families
/// (`cfi_*`, `eh_*`). Used by the expander to decide whether a
/// `LocalLabel(name)` token represents wfasm's local-label sigil or an
/// MC directive that should pass through unchanged.
pub fn is_mc_directive(name: &str) -> bool {
    if MC_DIRECTIVE_NAMES.contains(&name) {
        return true;
    }
    // Pattern families. Each `cfi_*` is a distinct directive
    // (`cfi_startproc`, `cfi_def_cfa_offset`, ...). Same for the
    // `eh_*` and `seh_*` families.
    name.starts_with("cfi_") || name.starts_with("eh_") || name.starts_with("seh_")
}

// ── Built-in Rust macros for Forth STC use ─────────────────────────

/// `stk(in, out)` — emit the stack-effect adjustment for a Forth
/// primitive whose signature is `(in -> out)` on the data stack.
///
/// Equivalent to WF32's auto-`ste-adjust`: the difference between input
/// and output cells turns into a single `add` or `sub` of `rbp`. With
/// no net change, emits nothing.
///
/// Cell size is read from `@assign cell = N` (defaults to 8 if not set).
///
/// Register it with `asm.register_macro("stk", wfasm::asm::macros::stk)`.
///
/// Example use in source:
///
/// ```text
/// proc(plus)        ; ( a b -- a+b )
///     stk(2, 1)     ; emits: add rbp, 8
///     add rax, [rbp]
///     next()
/// endp()
/// ```
pub fn stk(ctx: &mut super::expand::RustMacroCtx<'_>) -> Result<(), String> {
    if ctx.count() != 2 {
        return Err(format!("stk: expected 2 args (in, out), got {}", ctx.count()));
    }
    let in_count = ctx.parse_int(0)?;
    let out_count = ctx.parse_int(1)?;
    if in_count < 0 || out_count < 0 {
        return Err(format!(
            "stk: counts must be non-negative (got in={in_count}, out={out_count})"
        ));
    }
    let cell = ctx.lookup_int("cell").unwrap_or(8);
    let delta = (out_count - in_count) * cell;
    use std::cmp::Ordering;
    match delta.cmp(&0) {
        Ordering::Greater => ctx.emit_line(&format!("sub rbp, {delta}\n"))?,
        Ordering::Less => ctx.emit_line(&format!("add rbp, {}\n", -delta))?,
        Ordering::Equal => {}
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mc_directive_recognition() {
        assert!(is_mc_directive("globl"));
        assert!(is_mc_directive("intel_syntax"));
        assert!(is_mc_directive("quad"));
        assert!(is_mc_directive("cfi_startproc"));
        assert!(is_mc_directive("cfi_def_cfa"));
        assert!(is_mc_directive("eh_frame"));
        assert!(!is_mc_directive("done"));
        assert!(!is_mc_directive("loop_top"));
        assert!(!is_mc_directive("my_label"));
    }

    #[test]
    fn mangle_prefix_scope() {
        let f = ScopeFrame {
            kind: ScopeKind::Scope,
            name: "plus".into(),
            id: None,
        };
        assert_eq!(f.mangle_prefix(), "plus");
    }

    #[test]
    fn mangle_prefix_macro_invocation() {
        let f = ScopeFrame {
            kind: ScopeKind::MacroInvocation,
            name: "win64_call".into(),
            id: Some(7),
        };
        assert_eq!(f.mangle_prefix(), "win64_call$$7");
    }
}
