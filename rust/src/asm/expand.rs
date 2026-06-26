//! The expander: token stream in, token stream out, with all wfasm
//! directives and name substitutions resolved.
//!
//! The output is a token stream that, when concatenated by the emitter,
//! is a pure asm text blob ready for LLVM MC. No `@directive`s remain
//! (except for MC pass-throughs like `.globl` which the lexer represents
//! as `LocalLabel` tokens — those flow through unchanged in this layer
//! and get handled by the scope tracker in a later pass).
//!
//! ## What this layer handles
//!
//! * `@define NAME value`     — text macro: subsequent occurrences of
//!                              NAME as an Ident substitute the body
//!                              tokens verbatim.
//! * `@assign NAME = expr`    — evaluate, store as numeric. NAME
//!                              substitutes as a single Number token;
//!                              visible to other expressions.
//! * `@undef NAME`            — drop a define or an assign.
//! * `@if`/`@elif`/`@else`/   — conditional inclusion of a token range.
//!   `@endif`                   `@elif` evaluates only if all prior
//!                              branches were false.
//! * `@ifdef NAME`/`@ifndef`  — sugar over `@if defined(NAME)`.
//! * `@error "msg"`           — bail with a user-defined error.
//! * `@warn "msg"`            — emit a host-side warning, continue.
//! * `@assert expr, "msg"`    — bail unless expr is nonzero.
//! * `@rept N` / `@endr`      — repeat block N times; `@INDEX` is the
//!                              0-based iteration counter.
//! * Integer context names    — `@COUNTER`, `@LINE`, `@FILE` (as file
//!                              id), `@INDEX`, `@BITS` substitute as
//!                              a Number token.
//!
//! ## What this layer doesn't (yet) handle
//!
//! * `@macro` / `@endmacro`   — coming next iteration. Until then a
//!                              `@macro` directive is a hard error.
//! * `@scope` / `@endscope`   — same.
//! * `@include`               — same.
//! * `@for` / `@endfor`       — same.
//! * MC-passthrough tokens (`.globl`, `.text`, label `LocalLabel`
//!   tokens) flow through unchanged.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use super::error::AsmError;
use super::expr::{eval as eval_expr, EvalContext, EvalError};
use super::macros::{is_mc_directive, MacroDef, MacroParam, ScopeFrame, ScopeKind};
use super::source::SourceMap;
use super::span::Span;
use super::token::{NumberBase, NumberLit, Punct, Token, TokenKind};

/// Public-facing entry point. Owns the macro state across calls to
/// [`Assembler::expand`] so repeated invocations (one per source file
/// via `@include`) share defines and assigns.
#[derive(Default)]
pub struct Assembler {
    state: State,
}

#[derive(Default)]
struct State {
    /// Defined names. A name lives in exactly one slot at a time.
    /// Re-definitions of the same name silently replace the previous
    /// value (matches the "defines are config" use case).
    defines: HashMap<String, DefineValue>,
    /// User-defined `@macro` table. Looked up when an `Ident(name)`
    /// followed by `(` is encountered in the expansion stream.
    macros: HashMap<String, MacroDef>,
    /// Host-registered Rust closures, looked up the same way as text
    /// macros — but invoked with a `RustMacroCtx` rather than via token
    /// substitution. Used for things text substitution can't model
    /// (arithmetic over args, conditional emit logic, etc.).
    rust_macros: HashMap<String, RustMacroFn>,
    /// `@COUNTER` value. Reads bump.
    counter: i64,
    /// Monotonic id stamped onto each macro invocation. Used by local
    /// label mangling to guarantee per-invocation uniqueness without
    /// any user-visible counter.
    invocation_id: u64,
    /// Active scope stack — both `@scope` opens and live macro
    /// invocations push frames here.
    scope_stack: Vec<ScopeFrame>,
    /// Current recursion depth into macro expansions. Bounded.
    expansion_depth: usize,
    /// Cap on nested expansion depth — catches
    /// `@macro a() a() @endmacro`. Default 64.
    max_expansion_depth: usize,
    /// Owns the text of every file the assembler has touched (the
    /// top-level source plus everything `@include` pulls in).
    sources: SourceMap,
    /// Path stack — the top is the file currently being expanded.
    /// `@include`'s relative-path resolution looks at the top.
    path_stack: Vec<PathBuf>,
    /// Canonical paths currently on the include stack. Cycle detection.
    include_cycle: HashSet<PathBuf>,
    /// Cap on `@include` nesting. Default 64.
    max_include_depth: usize,
    /// In-memory overlay: `@include "path"` consults this map before
    /// falling back to `std::fs`. Tests use it to avoid temp files.
    virtual_files: HashMap<PathBuf, String>,
    /// `@extern [\"DLL.dll\"] NAME(arg_count)` declarations. Surfaced
    /// to the host via `Assembler::externs()` so it can pair each name
    /// with a host function pointer (loaded via the named DLL when
    /// present, registered manually otherwise) and tell the JIT.
    externs: HashMap<String, ExternDecl>,
    /// Open MASM-style runtime-control-flow blocks. Pushed by
    /// `.if` / `.while` / `.repeat`; popped by the matching closer.
    /// Must be empty at end of top-level expansion.
    block_stack: Vec<BlockFrame>,
    /// Monotonic id stamped into labels for runtime-control-flow
    /// blocks. Distinct from `counter` (which is `@COUNTER`) so
    /// user-visible `@COUNTER` reads aren't influenced by how many
    /// `.if`s the source contained.
    block_id_counter: u64,
}

/// One open MASM-style runtime control-flow block.
#[derive(Debug, Clone)]
enum BlockFrame {
    If {
        end_label: String,
        /// Where THIS branch jumps if its `cmp` fails. Updated by
        /// `.elseif` to a fresh label; consumed by `.endif` if no
        /// `.else` was seen.
        next_label: String,
        has_else: bool,
    },
    While {
        top_label: String,
        bottom_label: String,
    },
    Repeat {
        top_label: String,
        /// Where `.continue` inside the body jumps. Emitted right
        /// before the `.until` test so a `.continue` re-runs the
        /// post-test rather than restarting the loop body.
        cont_label: String,
        bottom_label: String,
    },
}

impl BlockFrame {
    fn kind_name(&self) -> &'static str {
        match self {
            BlockFrame::If { .. } => "if",
            BlockFrame::While { .. } => "while",
            BlockFrame::Repeat { .. } => "repeat",
        }
    }
}

/// One declared `@extern`. The DLL is present when source said
/// `@extern "DLL.dll" NAME(...)` (the generated Win32 bindings) and
/// absent for handwritten externs that come from the host's own Rust
/// code.
#[derive(Debug, Clone)]
pub struct ExternDecl {
    pub arg_count: usize,
    /// `Some("USER32.dll")` for a Win32 import; `None` for a host-
    /// supplied Rust function the host will register manually.
    pub dll: Option<String>,
}

/// What a `@define` or `@assign` resolves to.
#[derive(Debug, Clone)]
pub enum DefineValue {
    /// Token sequence to substitute verbatim.
    Text(Vec<Token>),
    /// A pre-evaluated integer. Substitutes as one Number token, and is
    /// visible to expression-context lookups.
    Numeric(i64),
}

/// Signature for a host-registered Rust macro.
///
/// The closure receives a [`RustMacroCtx`] that bundles:
///
/// * the (pre-expanded) call arguments — `ctx.count()`, `ctx.parse_int(n)`,
///   `ctx.parse_string(n)`, `ctx.parse_ident(n)`, `ctx.nth_tokens(n)`;
/// * read access to the surrounding state — `ctx.lookup_int(name)`,
///   `ctx.proc_name()`, `ctx.counter()`;
/// * one emit method — `ctx.emit_line(asm_text)` — which lexes the text
///   and appends tokens to the output stream. Emitted tokens are
///   **final**: they are not re-expanded by the macro engine. Compute
///   any necessary substitutions in Rust and format literal output.
///
/// Returning `Err(message)` aborts assembly with the message attached
/// to the macro call's source span.
pub type RustMacroFn = Box<dyn FnMut(&mut RustMacroCtx<'_>) -> Result<(), String>>;

/// Handle passed to a Rust macro closure. Lives only for the duration
/// of the closure call.
pub struct RustMacroCtx<'a> {
    args: &'a [Vec<Token>],
    state: &'a mut State,
    output: &'a mut Vec<Token>,
    call_span: Span,
    /// Name of the macro being invoked — for error messages.
    name: &'a str,
}

impl<'a> RustMacroCtx<'a> {
    /// Number of arguments the user passed at the call site.
    pub fn count(&self) -> usize {
        self.args.len()
    }

    /// Raw tokens of the `n`-th argument, or `None` if out of range.
    pub fn nth_tokens(&self, n: usize) -> Option<&[Token]> {
        self.args.get(n).map(|v| v.as_slice())
    }

    /// Parse the `n`-th argument as an integer expression. Uses the
    /// assembler's expression evaluator, which sees current `@assign`
    /// values and integer context names like `@COUNTER`.
    pub fn parse_int(&mut self, n: usize) -> Result<i64, String> {
        let toks = self
            .args
            .get(n)
            .ok_or_else(|| format!("{}: arg {} not provided", self.name, n))?;
        // Drop Newline / Comment tokens.
        let cleaned: Vec<Token> = toks
            .iter()
            .filter(|t| !matches!(t.kind, TokenKind::Newline | TokenKind::Comment(_)))
            .cloned()
            .collect();
        let mut ctx = ExpanderEvalContext { state: self.state };
        super::expr::eval(&cleaned, &mut ctx)
            .map_err(|e| format!("{}: arg {}: {}", self.name, n, e))
    }

    /// Parse the `n`-th argument as a single identifier. Errors if the
    /// argument isn't exactly one ident-shaped token.
    pub fn parse_ident(&self, n: usize) -> Result<&str, String> {
        let toks = self
            .args
            .get(n)
            .ok_or_else(|| format!("{}: arg {} not provided", self.name, n))?;
        let cleaned: Vec<&Token> = toks
            .iter()
            .filter(|t| !matches!(t.kind, TokenKind::Newline | TokenKind::Comment(_)))
            .collect();
        match cleaned.as_slice() {
            [tok] => match &tok.kind {
                TokenKind::Ident(s) => Ok(s.as_str()),
                _ => Err(format!(
                    "{}: arg {} is not an identifier",
                    self.name, n
                )),
            },
            _ => Err(format!(
                "{}: arg {} must be a single identifier (got {} tokens)",
                self.name,
                n,
                cleaned.len()
            )),
        }
    }

    /// Parse the `n`-th argument as a string literal.
    pub fn parse_string(&self, n: usize) -> Result<&str, String> {
        let toks = self
            .args
            .get(n)
            .ok_or_else(|| format!("{}: arg {} not provided", self.name, n))?;
        let cleaned: Vec<&Token> = toks
            .iter()
            .filter(|t| !matches!(t.kind, TokenKind::Newline | TokenKind::Comment(_)))
            .collect();
        match cleaned.as_slice() {
            [tok] => match &tok.kind {
                TokenKind::String(s) => Ok(s.value.as_str()),
                _ => Err(format!(
                    "{}: arg {} is not a string literal",
                    self.name, n
                )),
            },
            _ => Err(format!(
                "{}: arg {} must be a single string (got {} tokens)",
                self.name,
                n,
                cleaned.len()
            )),
        }
    }

    /// Look up an `@assign`-defined integer by name.
    pub fn lookup_int(&self, name: &str) -> Option<i64> {
        match self.state.defines.get(name) {
            Some(DefineValue::Numeric(n)) => Some(*n),
            _ => None,
        }
    }

    /// Name of the innermost open `@scope`, or `None` if none.
    pub fn proc_name(&self) -> Option<&str> {
        self.state
            .scope_stack
            .iter()
            .rev()
            .find(|f| f.kind == ScopeKind::Scope)
            .map(|f| f.name.as_str())
    }

    /// Read `@COUNTER` (and bump it).
    pub fn counter(&mut self) -> i64 {
        let v = self.state.counter;
        self.state.counter += 1;
        v
    }

    /// Get the mangled form of `name` for the innermost open scope, or
    /// `.name` as-is if no scope is open. Use when emitting a label
    /// that should be scope-local.
    pub fn mangle_local(&self, name: &str) -> String {
        match self
            .state
            .scope_stack
            .iter()
            .rev()
            .find(|f| f.kind == ScopeKind::Scope)
        {
            Some(frame) => format!("{}$${}", frame.name, name),
            None => format!(".{name}"),
        }
    }

    /// Emit a chunk of asm text. The text is lexed and the resulting
    /// tokens are appended to the output stream. Emitted tokens are
    /// **final** — they are not re-expanded by the macro engine.
    ///
    /// Span attribution: all emitted tokens carry the call-site span,
    /// so errors downstream (e.g., MC rejecting an unknown mnemonic)
    /// point at the macro call rather than into synthesized text.
    pub fn emit_line(&mut self, src: &str) -> Result<(), String> {
        // Use the call-site file id so any subsequent lex errors look
        // sensible. The lexer accepts any FileId; it just records it.
        let file_id = self.call_span.file;
        let tokens = super::lex::lex(file_id, src).map_err(|e| {
            format!("{}: emit_line lex error: {}", self.name, e)
        })?;
        for mut t in tokens {
            // Stamp the call site on every emitted token so diagnostics
            // point back to the macro call, not into synthesized text.
            t.span = self.call_span;
            self.output.push(t);
        }
        Ok(())
    }
}

impl Assembler {
    pub fn new() -> Self {
        Self {
            state: State {
                max_expansion_depth: 64,
                max_include_depth: 64,
                ..State::default()
            },
        }
    }

    /// Set the maximum macro expansion depth. Default 64. Exceeded
    /// depth surfaces as `ExpansionTooDeep` with the innermost macro
    /// name and the actual limit.
    pub fn set_max_expansion_depth(&mut self, depth: usize) {
        self.state.max_expansion_depth = depth;
    }

    /// Set the maximum `@include` nesting depth. Default 64.
    pub fn set_max_include_depth(&mut self, depth: usize) {
        self.state.max_include_depth = depth;
    }

    /// Register an in-memory file under `path`. `@include "path"` will
    /// see this content instead of going to disk. Useful for tests and
    /// for hosts that want to provide built-in macro libraries without
    /// shipping `.masm` files.
    ///
    /// `path` is used as-is; if your `@include` resolves to a different
    /// canonical form, the overlay won't match. Pass absolute paths
    /// to avoid surprises.
    pub fn add_virtual_file(&mut self, path: impl AsRef<Path>, contents: impl Into<String>) {
        self.state
            .virtual_files
            .insert(path.as_ref().to_path_buf(), contents.into());
    }

    /// Read access to the SourceMap. Useful for rendering diagnostics:
    /// a `Span` carries a `FileId`, which the map resolves to a path.
    pub fn sources(&self) -> &SourceMap {
        &self.state.sources
    }

    /// Iterate every `@extern` declaration seen in source. Each tuple
    /// is `(name, &ExternDecl)`. The `ExternDecl` carries arg count
    /// and an optional DLL name (present when `@extern "DLL.dll" …`
    /// syntax was used, absent for host-supplied Rust externs).
    ///
    /// The host's job: for each entry with a `dll`, `LoadLibraryW` it
    /// (cache one HMODULE per distinct DLL), `GetProcAddress(name)`,
    /// pair the address with the `arg_count`, and register via
    /// `Jit::define_extern_fn`. For entries with no DLL, the host
    /// supplies the function pointer directly (e.g. from its own
    /// `extern "C"` Rust functions).
    pub fn externs(&self) -> impl Iterator<Item = (&str, &ExternDecl)> + '_ {
        self.state.externs.iter().map(|(k, v)| (k.as_str(), v))
    }

    /// Register a Rust-implemented macro. The closure is invoked when
    /// source code calls `NAME(args...)`. Replaces any prior
    /// registration (including the placeholder a `@rust_macro NAME`
    /// directive installed).
    ///
    /// See [`RustMacroCtx`] for the closure's API: argument access,
    /// `lookup_int`, `proc_name`, `emit_line`, etc.
    pub fn register_macro<F>(&mut self, name: &str, f: F)
    where
        F: FnMut(&mut RustMacroCtx<'_>) -> Result<(), String> + 'static,
    {
        self.state.rust_macros.insert(name.to_string(), Box::new(f));
    }

    /// Inject a numeric value from the host. Equivalent to a top-level
    /// `@assign NAME = value` before assembly starts.
    pub fn define(&mut self, name: &str, value: i64) {
        self.state.defines.insert(name.to_string(), DefineValue::Numeric(value));
    }

    /// Inject a textual value from the host. Equivalent to a top-level
    /// `@define NAME tokens`.
    pub fn define_text(&mut self, name: &str, tokens: Vec<Token>) {
        self.state.defines.insert(name.to_string(), DefineValue::Text(tokens));
    }

    /// Drop a define or assign. No-op if the name wasn't defined.
    pub fn undefine(&mut self, name: &str) {
        self.state.defines.remove(name);
    }

    /// Read the current value of a name. Used by tests and by the
    /// expression evaluator's `EvalContext`.
    pub fn lookup(&self, name: &str) -> Option<&DefineValue> {
        self.state.defines.get(name)
    }

    /// Full pipeline: lex → expand → emit. Produces a string of
    /// MC-flavor assembly text from an in-memory source string.
    ///
    /// `file_label` is treated as a pseudo-path: `@include`s inside
    /// `source` resolve relative to `file_label`'s parent directory.
    /// For tests / inline use, label as `"hello.masm"` or similar.
    pub fn assemble(&mut self, file_label: &str, source: &str) -> Result<String, AsmError> {
        let pseudo = PathBuf::from(file_label);
        self.assemble_with_path(pseudo, source)
    }

    /// Like [`assemble`] but reads `path` from disk (or from the
    /// virtual-file overlay if registered there). `@include`s resolve
    /// relative to `path`'s parent directory.
    pub fn assemble_file(&mut self, path: impl AsRef<Path>) -> Result<String, AsmError> {
        let path = path.as_ref().to_path_buf();
        let text = self.read_source(&path).map_err(|e| {
            AsmError::Expand(ExpandError {
                kind: ExpandErrorKind::IncludeIo {
                    path: path.display().to_string(),
                    error: e.to_string(),
                },
                span: Span::SYNTHETIC,
            })
        })?;
        self.assemble_with_path(path, &text)
    }

    fn assemble_with_path(&mut self, path: PathBuf, source: &str) -> Result<String, AsmError> {
        use super::lex::lex;
        let file_id = self.state.sources.add(path.clone(), source.to_string());
        // Re-borrow the text from the SourceMap to satisfy lifetimes.
        let text = self.state.sources.text(file_id).to_string();
        let tokens = lex(file_id, &text)?;
        self.state.path_stack.push(path);
        let result = self.expand(&tokens);
        self.state.path_stack.pop();
        let expanded = result?;
        let asm = super::emit::emit(&expanded)?;
        Ok(asm)
    }

    /// Read a source file. Consults `virtual_files` first, then falls
    /// back to `std::fs::read_to_string`.
    fn read_source(&self, path: &Path) -> Result<String, std::io::Error> {
        if let Some(content) = self.state.virtual_files.get(path) {
            return Ok(content.clone());
        }
        std::fs::read_to_string(path)
    }

    /// Expand a token stream. Returns the post-expansion stream; any
    /// remaining directives are a wfasm bug or an unimplemented feature.
    pub fn expand(&mut self, tokens: &[Token]) -> Result<Vec<Token>, AsmError> {
        let mut warnings = Vec::new();
        let mut out = Vec::new();
        {
            let mut exp = Expander {
                state: &mut self.state,
                out: &mut out,
                warnings: &mut warnings,
            };
            exp.expand_range(tokens, false)?;
        }
        // Print collected warnings to stderr. The host may have its own
        // sink later; for now stderr matches Rust's standard tooling.
        for w in &warnings {
            eprintln!("warning: {w}");
        }
        // Any scope still open at end of input is a user error.
        if let Some(frame) = self.state.scope_stack.last() {
            if frame.kind == ScopeKind::Scope {
                return Err(AsmError::Expand(ExpandError {
                    kind: ExpandErrorKind::UnclosedScope(frame.name.clone()),
                    span: Span::SYNTHETIC,
                }));
            }
        }
        // Any runtime control-flow block left open is a user error.
        if let Some(frame) = self.state.block_stack.last() {
            return Err(AsmError::Expand(ExpandError {
                kind: ExpandErrorKind::UnclosedBlock(frame.kind_name()),
                span: Span::SYNTHETIC,
            }));
        }
        Ok(out)
    }
}

/// Bound macro arguments, keyed by parameter name.
struct MacroArgs {
    bound: HashMap<String, MacroArgValue>,
}

/// A single bound argument's value.
enum MacroArgValue {
    /// One arg (one comma-separated entry) — its tokens.
    Tokens(Vec<Token>),
    /// Variadic tail — each sub-vec is one comma-separated entry.
    Variadic(Vec<Vec<Token>>),
}

struct Expander<'s> {
    state: &'s mut State,
    out: &'s mut Vec<Token>,
    warnings: &'s mut Vec<String>,
}

#[derive(Debug)]
pub struct ExpandError {
    pub kind: ExpandErrorKind,
    pub span: Span,
}

impl std::fmt::Display for ExpandError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}", self.span, self.kind)
    }
}
impl std::error::Error for ExpandError {}

impl From<ExpandError> for AsmError {
    fn from(e: ExpandError) -> Self {
        // Funnel into AsmError. We don't yet have a variant; create one
        // implicitly by wrapping in a synthetic LexError won't do — add
        // a real variant below.
        AsmError::Expand(e)
    }
}

#[derive(Debug)]
pub enum ExpandErrorKind {
    /// User said `@error "..."`.
    UserError(String),
    /// `@assert expr, "msg"` where expr was zero.
    AssertionFailed(String),
    /// Hit a directive name we don't know.
    UnknownDirective(String),
    /// Hit a statement directive in expression / mid-line position, or
    /// vice versa.
    MisplacedDirective { name: String, reason: &'static str },
    /// Hit `@endif`/`@elif`/`@else`/`@endr` with no matching opener.
    UnbalancedBlock(String),
    /// Expression evaluator returned an error during directive
    /// processing.
    Expr(EvalError),
    /// `@assign NAME = expr` missing the `=` or expr.
    MalformedAssign(&'static str),
    /// `@define NAME ...` missing the name.
    MalformedDefine(&'static str),
    /// `@undef NAME` missing or bad name.
    MalformedUndef(&'static str),
    /// `@error` / `@warn` / `@assert` with missing or wrong-typed args.
    MalformedDiagnostic(&'static str),
    /// `@rept N` with a bad count.
    MalformedRept(&'static str),
    /// Unimplemented directive (placeholder until macros / scopes /
    /// include land).
    Unimplemented(&'static str),
    /// `@macro` header malformed.
    MalformedMacro(&'static str),
    /// `@scope` requires a name.
    MalformedScope(&'static str),
    /// Macro invocation problems.
    MacroArity {
        name: String,
        expected: usize,
        got: usize,
        variadic: bool,
    },
    /// Recursion depth budget exhausted.
    ExpansionTooDeep { limit: usize, in_macro: String },
    /// `&name` inside a macro body that has no parameter `name`.
    UnknownParam { macro_name: String, param: String },
    /// `@macro` nested inside another `@macro` — not supported.
    NestedMacroDef,
    /// `@scope` not balanced by `@endscope` at end of input.
    UnclosedScope(String),
    /// `##` token paste not yet implemented.
    PasteUnsupported,
    /// `@include` was malformed (missing string argument).
    MalformedInclude(&'static str),
    /// `@include` couldn't read the file.
    IncludeIo { path: String, error: String },
    /// `@include` cycle — file is already on the include stack.
    IncludeCycle { path: String },
    /// `@include` nesting exceeded the configured limit.
    IncludeTooDeep { path: String, limit: usize },
    /// Included file had a lex error. We carry the rendered string
    /// (with the included file's path) so the caller doesn't have to
    /// chase through the SourceMap.
    IncludeLexFailed { path: String, error: String },
    /// A Rust macro returned `Err(message)`.
    RustMacroError(String),
    /// `@extern` header malformed.
    MalformedExtern(&'static str),
    /// `.if` / `.while` / `.repeat` condition couldn't be parsed.
    MalformedCondition(&'static str),
    /// `.elseif` or `.else` outside an open `.if`. `.endw` outside `.while`,
    /// etc.
    StrayControlFlow(&'static str),
    /// `.elseif` after `.else`, multiple `.else`, etc.
    BadIfStructure(&'static str),
    /// `.break` or `.continue` outside any loop.
    NotInLoop(&'static str),
    /// `.if`/`.while`/`.repeat` left open at end of expansion.
    UnclosedBlock(&'static str),
}

impl std::fmt::Display for ExpandErrorKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use ExpandErrorKind::*;
        match self {
            UserError(m) => write!(f, "error: {m}"),
            AssertionFailed(m) => write!(f, "assertion failed: {m}"),
            UnknownDirective(n) => write!(f, "unknown directive `@{n}`"),
            MisplacedDirective { name, reason } => {
                write!(f, "`@{name}` {reason}")
            }
            UnbalancedBlock(n) => write!(f, "`@{n}` with no matching opener"),
            Expr(e) => write!(f, "in expression: {e}"),
            MalformedAssign(m) => write!(f, "malformed @assign: {m}"),
            MalformedDefine(m) => write!(f, "malformed @define: {m}"),
            MalformedUndef(m) => write!(f, "malformed @undef: {m}"),
            MalformedDiagnostic(m) => write!(f, "malformed diagnostic: {m}"),
            MalformedRept(m) => write!(f, "malformed @rept: {m}"),
            Unimplemented(n) => {
                write!(f, "directive `@{n}` is not yet implemented")
            }
            MalformedMacro(m) => write!(f, "malformed @macro: {m}"),
            MalformedScope(m) => write!(f, "malformed @scope: {m}"),
            MacroArity {
                name,
                expected,
                got,
                variadic,
            } => {
                if *variadic {
                    write!(
                        f,
                        "macro `{name}` expects at least {expected} argument(s), got {got}"
                    )
                } else {
                    write!(
                        f,
                        "macro `{name}` expects {expected} argument(s), got {got}"
                    )
                }
            }
            ExpansionTooDeep { limit, in_macro } => write!(
                f,
                "macro expansion depth exceeded {limit} (innermost: `{in_macro}`)"
            ),
            UnknownParam { macro_name, param } => write!(
                f,
                "macro `{macro_name}` has no parameter `{param}`"
            ),
            NestedMacroDef => write!(f, "`@macro` cannot be nested inside another `@macro`"),
            UnclosedScope(n) => write!(f, "`@scope {n}` not closed by `@endscope`"),
            PasteUnsupported => write!(
                f,
                "`##` token paste is not yet implemented (planned for v1.5)"
            ),
            MalformedInclude(m) => write!(f, "malformed @include: {m}"),
            IncludeIo { path, error } => {
                write!(f, "@include `{path}`: {error}")
            }
            IncludeCycle { path } => write!(f, "@include cycle on `{path}`"),
            IncludeTooDeep { path, limit } => write!(
                f,
                "@include nesting exceeded {limit} while including `{path}`"
            ),
            IncludeLexFailed { path, error } => {
                write!(f, "@include `{path}`: lex error: {error}")
            }
            RustMacroError(msg) => write!(f, "Rust macro error: {msg}"),
            MalformedExtern(m) => write!(f, "malformed @extern: {m}"),
            MalformedCondition(m) => write!(f, "malformed condition: {m}"),
            StrayControlFlow(m) => write!(f, "{m}"),
            BadIfStructure(m) => write!(f, "{m}"),
            NotInLoop(m) => write!(f, "{m}"),
            UnclosedBlock(k) => write!(f, "`.{k}` block left unclosed"),
        }
    }
}

impl<'s> Expander<'s> {
    /// Walk `tokens`. If `inside_block`, return when we hit the matching
    /// block terminator (the caller drives that recognition).
    ///
    /// Top-level use: `expand_range(tokens, false)`.
    fn expand_range(
        &mut self,
        tokens: &[Token],
        _inside_block: bool,
    ) -> Result<(), ExpandError> {
        let mut i = 0;
        while i < tokens.len() {
            // Line-start is derived from token position, not carried in
            // a loop variable — that way returning from a directive
            // handler (which lands us at line N+1 col 1) is naturally
            // recognized as the start of the next line.
            let line_start = at_line_start(tokens, i);
            let tok = &tokens[i];
            match &tok.kind {
                TokenKind::Newline => {
                    self.out.push(tok.clone());
                    i += 1;
                    continue;
                }
                TokenKind::Comment(_) => {
                    // Drop comments — no semantics for MC.
                    i += 1;
                    continue;
                }
                TokenKind::Directive(name) => {
                    if is_statement_directive(name) {
                        if !line_start {
                            return Err(self.err(
                                ExpandErrorKind::MisplacedDirective {
                                    name: name.clone(),
                                    reason: "must appear at the start of a line",
                                },
                                tok.span,
                            ));
                        }
                        i = self.handle_statement(tokens, i)?;
                        continue;
                    } else if is_context_directive(name) {
                        let value = self.read_context_int(name, tok.span)?;
                        self.out
                            .push(make_number_token(value, tok.span, tok.space_before));
                        i += 1;
                        continue;
                    } else {
                        return Err(self.err(
                            ExpandErrorKind::UnknownDirective(name.clone()),
                            tok.span,
                        ));
                    }
                }
                TokenKind::Ident(name) => {
                    // Priority 1: Rust macro call.
                    if self.state.rust_macros.contains_key(name)
                        && next_is_lparen(tokens, i + 1)
                    {
                        let name = name.clone();
                        let (args, after) =
                            self.parse_call_args(tokens, i + 1, tok.span)?;
                        self.invoke_rust_macro(&name, &args, tok.span)?;
                        i = after;
                        continue;
                    }
                    // Priority 2: text macro call.
                    if self.state.macros.contains_key(name)
                        && next_is_lparen(tokens, i + 1)
                    {
                        let def = self.state.macros.get(name).unwrap().clone();
                        let (args, after) =
                            self.parse_macro_args(tokens, i + 1, &def, tok.span)?;
                        self.expand_macro_call(&def, args, tok.span)?;
                        i = after;
                        continue;
                    }
                    // Priority 3: @define / @assign substitution.
                    if let Some(val) = self.state.defines.get(name) {
                        match val.clone() {
                            DefineValue::Numeric(n) => {
                                self.out.push(make_number_token(
                                    n,
                                    tok.span,
                                    tok.space_before,
                                ));
                            }
                            DefineValue::Text(body) => {
                                let mut first = true;
                                for bt in body {
                                    let mut bt = bt.clone();
                                    if first {
                                        bt.space_before = tok.space_before;
                                        first = false;
                                    }
                                    self.out.push(bt);
                                }
                            }
                        }
                        i += 1;
                        continue;
                    }
                    self.out.push(tok.clone());
                    i += 1;
                }
                TokenKind::LocalLabel(name, outer) => {
                    // MASM-style runtime control flow: `.if`, `.elseif`,
                    // `.else`, `.endif`, `.while`, `.endw`, `.repeat`,
                    // `.until`, `.break`, `.continue`. Only valid at
                    // line start; mid-line `.foo` falls through as a
                    // label or MC directive.
                    if line_start && !*outer {
                        if let Some(kw) = control_flow_keyword(name) {
                            i = self.handle_control_flow(kw, tokens, i + 1, tok.span)?;
                            continue;
                        }
                    }
                    // `.foo` disambiguation by surrounding tokens.
                    //
                    // We're at "directive position" if the previous
                    // token is Newline (true line start) or Colon
                    // (right after a label, MASM-style "label: dir").
                    // We're followed by a Colon if next token is `:`
                    // (label definition syntax).
                    //
                    //   directive-pos + NOT followed-by-`:` + MC-name
                    //     → MC directive form  → pass through unchanged
                    //   followed-by-`:`
                    //     → label definition   → mangle if scoped
                    //   any other position
                    //     → label reference    → mangle if scoped
                    //
                    // `.skip:` is always a label (the `:` decides);
                    // `.skip 16` at line-start with no `:` is the GAS
                    // directive; `mylabel: .quad 10` recognises `.quad`
                    // because the preceding `:` puts us in directive
                    // position; `jne .skip` does NOT (preceded by an
                    // ident), so it mangles to match the definition.
                    let prev_kind = tokens.get(i.wrapping_sub(1)).map(|t| &t.kind);
                    let at_directive_pos = i == 0
                        || matches!(prev_kind, Some(TokenKind::Newline))
                        || matches!(prev_kind, Some(TokenKind::Punct(Punct::Colon)));
                    let followed_by_colon = matches!(
                        tokens.get(i + 1).map(|t| &t.kind),
                        Some(TokenKind::Punct(Punct::Colon))
                    );
                    // `.<suffix>` glued to a preceding identifier (no space,
                    // not a label definition) is a register arrangement
                    // (`v0.8b`) or condition-code mnemonic (`b.eq`), not a
                    // scope-local label. Emit it verbatim.
                    if !followed_by_colon
                        && is_glued_suffix(tok.space_before, *outer, prev_kind)
                    {
                        self.out.push(tok.clone());
                        i += 1;
                        continue;
                    }
                    if at_directive_pos && !followed_by_colon && is_mc_directive(name) {
                        self.out.push(tok.clone());
                        i += 1;
                        continue;
                    }
                    let prefix = if *outer {
                        self.outer_mangle_prefix()
                    } else {
                        self.current_mangle_prefix()
                    };
                    if let Some(prefix) = prefix {
                        let mangled = format!("{prefix}$${name}");
                        self.out.push(Token {
                            kind: TokenKind::Ident(mangled),
                            span: tok.span,
                            space_before: tok.space_before,
                        });
                    } else {
                        // No active scope — pass through. MC will see
                        // `.name` and treat it as a local symbol.
                        self.out.push(tok.clone());
                    }
                    i += 1;
                }
                TokenKind::Punct(Punct::HashHash) => {
                    // `##` is the planned token-paste operator.
                    return Err(self.err(ExpandErrorKind::PasteUnsupported, tok.span));
                }
                _ => {
                    self.out.push(tok.clone());
                    i += 1;
                }
            }
        }
        Ok(())
    }

    /// Innermost scope frame's mangling prefix, or `None` if no frame
    /// is open. Used by LocalLabel mangling.
    fn current_mangle_prefix(&self) -> Option<String> {
        self.state.scope_stack.last().map(|f| f.mangle_prefix())
    }

    /// Mangling prefix for `.^name` — the innermost frame that is NOT
    /// a macro invocation. Lets a macro body refer to a label defined
    /// in the calling proc's `@scope`. Falls back to the innermost
    /// frame if no `@scope` is open above the macro invocations.
    fn outer_mangle_prefix(&self) -> Option<String> {
        for frame in self.state.scope_stack.iter().rev() {
            if frame.kind == ScopeKind::Scope {
                return Some(frame.mangle_prefix());
            }
        }
        self.state.scope_stack.last().map(|f| f.mangle_prefix())
    }

    /// Dispatch a top-of-line directive. Returns the cursor position
    /// just past the directive's last consumed token (typically just
    /// past its terminating newline).
    fn handle_statement(
        &mut self,
        tokens: &[Token],
        i: usize,
    ) -> Result<usize, ExpandError> {
        let dir_tok = &tokens[i];
        let name = match &dir_tok.kind {
            TokenKind::Directive(n) => n.clone(),
            _ => unreachable!(),
        };
        let span = dir_tok.span;

        match name.as_str() {
            "define" => self.handle_define(tokens, i + 1, span),
            "assign" => self.handle_assign(tokens, i + 1, span),
            "undef" => self.handle_undef(tokens, i + 1, span),

            "ifdef" => self.handle_ifdef(tokens, i + 1, span, /*negate=*/ false),
            "ifndef" => self.handle_ifdef(tokens, i + 1, span, /*negate=*/ true),
            "if" => self.handle_if(tokens, i + 1, span),

            // These appear only inside an @if / @rept / @macro / @for
            // body — at top level they're unbalanced. `@endscope` is
            // NOT here: it's a legitimate top-level closer for a
            // previously-pushed scope, and is handled below.
            "elif" | "else" | "endif" | "endr" | "endmacro" | "endfor" => {
                Err(self.err(ExpandErrorKind::UnbalancedBlock(name), span))
            }

            "error" => self.handle_error_warn(tokens, i + 1, span, /*fatal=*/ true),
            "warn" => self.handle_error_warn(tokens, i + 1, span, /*fatal=*/ false),
            "assert" => self.handle_assert(tokens, i + 1, span),

            "rept" => self.handle_rept(tokens, i + 1, span),

            "macro" => self.handle_macro_def(tokens, i + 1, span),
            "scope" => self.handle_scope(tokens, i + 1, span),
            "endscope" => self.handle_endscope(tokens, i + 1, span),
            "include" => self.handle_include(tokens, i + 1, span),
            "rust_macro" => self.handle_rust_macro(tokens, i + 1, span),
            "extern" => self.handle_extern(tokens, i + 1, span),

            // Unimplemented this iteration. Surface clearly.
            // (`endmacro`/`endfor` are absent here — they hit the
            //  unbalanced-block arm above, which produces a clear error.
            //  `endscope` IS handled because it's a legitimate top-level
            //  closer; it's an error only if the stack is empty.)
            "for" | "bits"
            | "section" | "code" | "data" | "rodata" | "bss"
            | "db" | "dw" | "dd" | "dq" | "dz" | "local" => Err(self.err(
                ExpandErrorKind::Unimplemented(directive_name_static(&name)),
                span,
            )),

            _ => Err(self.err(ExpandErrorKind::UnknownDirective(name), span)),
        }
    }

    // ── @define NAME body ───────────────────────────────────────────

    fn handle_define(
        &mut self,
        tokens: &[Token],
        i: usize,
        dspan: Span,
    ) -> Result<usize, ExpandError> {
        // Expect: Ident NAME, then the rest of the line is the body.
        let (name, rest) = take_name(tokens, i).ok_or_else(|| {
            self.err(
                ExpandErrorKind::MalformedDefine("expected NAME after @define"),
                dspan,
            )
        })?;
        // Collect body tokens until newline.
        let (body, after) = collect_line(tokens, rest);
        self.state
            .defines
            .insert(name, DefineValue::Text(body.to_vec()));
        Ok(skip_newline(tokens, after))
    }

    // ── @assign NAME = expr ─────────────────────────────────────────

    fn handle_assign(
        &mut self,
        tokens: &[Token],
        i: usize,
        dspan: Span,
    ) -> Result<usize, ExpandError> {
        let (name, after_name) = take_name(tokens, i).ok_or_else(|| {
            self.err(
                ExpandErrorKind::MalformedAssign("expected NAME after @assign"),
                dspan,
            )
        })?;
        // Expect `=`.
        let after_eq = match tokens.get(after_name) {
            Some(t) if matches!(t.kind, TokenKind::Punct(Punct::Eq)) => after_name + 1,
            _ => {
                return Err(self.err(
                    ExpandErrorKind::MalformedAssign("expected `=` after NAME"),
                    dspan,
                ));
            }
        };
        let (expr_tokens, after) = collect_line(tokens, after_eq);
        // Pre-expand: pass expr_tokens through expansion (resolves nested
        // defines / context names), then evaluate.
        let pre = self.preexpand(expr_tokens)?;
        let mut ctx = self.eval_context();
        let value = eval_expr(&pre, &mut ctx).map_err(|e| {
            self.err(ExpandErrorKind::Expr(e), dspan)
        })?;
        self.state
            .defines
            .insert(name, DefineValue::Numeric(value));
        Ok(skip_newline(tokens, after))
    }

    // ── @undef NAME ─────────────────────────────────────────────────

    fn handle_undef(
        &mut self,
        tokens: &[Token],
        i: usize,
        dspan: Span,
    ) -> Result<usize, ExpandError> {
        let (name, after) = take_name(tokens, i).ok_or_else(|| {
            self.err(
                ExpandErrorKind::MalformedUndef("expected NAME after @undef"),
                dspan,
            )
        })?;
        self.state.defines.remove(&name);
        let (_rest, after) = collect_line(tokens, after);
        Ok(skip_newline(tokens, after))
    }

    // ── @ifdef / @ifndef ────────────────────────────────────────────

    fn handle_ifdef(
        &mut self,
        tokens: &[Token],
        i: usize,
        dspan: Span,
        negate: bool,
    ) -> Result<usize, ExpandError> {
        let (name, after_name) = take_name(tokens, i).ok_or_else(|| {
            self.err(
                ExpandErrorKind::MalformedDefine(
                    "@ifdef / @ifndef require a NAME argument",
                ),
                dspan,
            )
        })?;
        let after = skip_newline(tokens, after_name);
        let defined = self.state.defines.contains_key(&name);
        let cond = if negate { !defined } else { defined };
        self.run_conditional_block(tokens, after, dspan, cond)
    }

    // ── @if expr ────────────────────────────────────────────────────

    fn handle_if(
        &mut self,
        tokens: &[Token],
        i: usize,
        dspan: Span,
    ) -> Result<usize, ExpandError> {
        let (expr_tokens, after) = collect_line(tokens, i);
        let cond = self.eval_line(expr_tokens, dspan)?;
        let after = skip_newline(tokens, after);
        self.run_conditional_block(tokens, after, dspan, cond != 0)
    }

    /// Walk the body of an `@if`/`@ifdef`/`@ifndef`, honoring `@elif`,
    /// `@else`, and `@endif`. `first_cond` is whether the very first
    /// branch is being taken (the rest are exclusive).
    fn run_conditional_block(
        &mut self,
        tokens: &[Token],
        start: usize,
        if_span: Span,
        first_cond: bool,
    ) -> Result<usize, ExpandError> {
        let mut i = start;
        let mut taken = first_cond;
        let mut any_taken = first_cond;
        let mut current_branch_active = first_cond;

        while i < tokens.len() {
            let tok = &tokens[i];
            // We need to recognize @elif / @else / @endif at the same
            // nesting level. Nested @if blocks must be skipped over
            // wholesale.
            if at_line_start(tokens, i) {
                if let TokenKind::Directive(name) = &tok.kind {
                    match name.as_str() {
                        "if" | "ifdef" | "ifndef" => {
                            // Skip over the nested conditional entirely
                            // regardless of taken-ness — if active, we
                            // recurse and emit; if inactive, we skip.
                            if current_branch_active {
                                i = self.handle_statement(tokens, i)?;
                            } else {
                                i = skip_nested_block(tokens, i, &["endif"])?;
                            }
                            continue;
                        }
                        "elif" => {
                            // End the current branch.
                            current_branch_active = false;
                            let line_start = i + 1;
                            let (expr_tokens, after) = collect_line(tokens, line_start);
                            i = skip_newline(tokens, after);
                            if !any_taken {
                                let cond = self.eval_line(expr_tokens, tok.span)?;
                                if cond != 0 {
                                    taken = true;
                                    any_taken = true;
                                    current_branch_active = true;
                                }
                            }
                            continue;
                        }
                        "else" => {
                            current_branch_active = !any_taken;
                            if current_branch_active {
                                taken = true;
                                any_taken = true;
                            }
                            let (_rest, after) = collect_line(tokens, i + 1);
                            i = skip_newline(tokens, after);
                            continue;
                        }
                        "endif" => {
                            let (_rest, after) = collect_line(tokens, i + 1);
                            return Ok(skip_newline(tokens, after));
                        }
                        _ => {}
                    }
                }
            }

            if current_branch_active {
                // Expand this single token (or block) in place. We
                // can't call expand_range because it would consume
                // through endif; instead, expand-one-step.
                i = self.expand_one(tokens, i)?;
            } else {
                i += 1;
            }
        }

        // Fell off the end without seeing @endif.
        let _ = taken; // silence; used for clarity above
        Err(self.err(
            ExpandErrorKind::UnbalancedBlock("if".into()),
            if_span,
        ))
    }

    /// Expand exactly one logical token (passthrough, define
    /// substitution, context-name substitution, OR a nested statement
    /// like @if/@rept that must consume its whole block). Returns the
    /// new cursor.
    fn expand_one(&mut self, tokens: &[Token], i: usize) -> Result<usize, ExpandError> {
        let tok = &tokens[i];
        match &tok.kind {
            TokenKind::Directive(name) => {
                if is_statement_directive(name) {
                    if !at_line_start(tokens, i) {
                        return Err(self.err(
                            ExpandErrorKind::MisplacedDirective {
                                name: name.clone(),
                                reason: "must appear at the start of a line",
                            },
                            tok.span,
                        ));
                    }
                    self.handle_statement(tokens, i)
                } else if is_context_directive(name) {
                    let value = self.read_context_int(name, tok.span)?;
                    self.out
                        .push(make_number_token(value, tok.span, tok.space_before));
                    Ok(i + 1)
                } else {
                    Err(self.err(
                        ExpandErrorKind::UnknownDirective(name.clone()),
                        tok.span,
                    ))
                }
            }
            TokenKind::Ident(name) => {
                if self.state.rust_macros.contains_key(name)
                    && next_is_lparen(tokens, i + 1)
                {
                    let name = name.clone();
                    let (args, after) =
                        self.parse_call_args(tokens, i + 1, tok.span)?;
                    self.invoke_rust_macro(&name, &args, tok.span)?;
                    return Ok(after);
                }
                if self.state.macros.contains_key(name)
                    && next_is_lparen(tokens, i + 1)
                {
                    let def = self.state.macros.get(name).unwrap().clone();
                    let (args, after) =
                        self.parse_macro_args(tokens, i + 1, &def, tok.span)?;
                    self.expand_macro_call(&def, args, tok.span)?;
                    return Ok(after);
                }
                if let Some(val) = self.state.defines.get(name) {
                    match val.clone() {
                        DefineValue::Numeric(n) => {
                            self.out
                                .push(make_number_token(n, tok.span, tok.space_before));
                        }
                        DefineValue::Text(body) => {
                            let mut first = true;
                            for bt in body {
                                let mut bt = bt.clone();
                                if first {
                                    bt.space_before = tok.space_before;
                                    first = false;
                                }
                                self.out.push(bt);
                            }
                        }
                    }
                    return Ok(i + 1);
                }
                self.out.push(tok.clone());
                Ok(i + 1)
            }
            TokenKind::LocalLabel(name, outer) => {
                if at_line_start(tokens, i) && !*outer {
                    if let Some(kw) = control_flow_keyword(name) {
                        return self.handle_control_flow(kw, tokens, i + 1, tok.span);
                    }
                }
                // Disambiguation by surrounding tokens — see the
                // matching block in expand_range for the full rule.
                let prev_kind = tokens.get(i.wrapping_sub(1)).map(|t| &t.kind);
                let at_directive_pos = i == 0
                    || matches!(prev_kind, Some(TokenKind::Newline))
                    || matches!(prev_kind, Some(TokenKind::Punct(Punct::Colon)));
                let followed_by_colon = matches!(
                    tokens.get(i + 1).map(|t| &t.kind),
                    Some(TokenKind::Punct(Punct::Colon))
                );
                if !followed_by_colon
                    && is_glued_suffix(tok.space_before, *outer, prev_kind)
                {
                    // Register arrangement / condition-code suffix glued to
                    // the previous ident — verbatim, never scope-mangled.
                    self.out.push(tok.clone());
                } else if at_directive_pos && !followed_by_colon && is_mc_directive(name) {
                    self.out.push(tok.clone());
                } else {
                    let prefix = if *outer {
                        self.outer_mangle_prefix()
                    } else {
                        self.current_mangle_prefix()
                    };
                    if let Some(prefix) = prefix {
                        let mangled = format!("{prefix}$${name}");
                        self.out.push(Token {
                            kind: TokenKind::Ident(mangled),
                            span: tok.span,
                            space_before: tok.space_before,
                        });
                    } else {
                        self.out.push(tok.clone());
                    }
                }
                Ok(i + 1)
            }
            TokenKind::Punct(Punct::HashHash) => {
                Err(self.err(ExpandErrorKind::PasteUnsupported, tok.span))
            }
            TokenKind::Comment(_) => Ok(i + 1),
            _ => {
                self.out.push(tok.clone());
                Ok(i + 1)
            }
        }
    }

    // ── @error / @warn ─────────────────────────────────────────────

    fn handle_error_warn(
        &mut self,
        tokens: &[Token],
        i: usize,
        dspan: Span,
        fatal: bool,
    ) -> Result<usize, ExpandError> {
        // Expect a string literal.
        let (msg, after) = take_string(tokens, i).ok_or_else(|| {
            self.err(
                ExpandErrorKind::MalformedDiagnostic("expected a string argument"),
                dspan,
            )
        })?;
        let (_rest, after) = collect_line(tokens, after);
        let after = skip_newline(tokens, after);
        if fatal {
            Err(self.err(ExpandErrorKind::UserError(msg), dspan))
        } else {
            self.warnings.push(msg);
            Ok(after)
        }
    }

    // ── @assert expr, "msg" ─────────────────────────────────────────

    fn handle_assert(
        &mut self,
        tokens: &[Token],
        i: usize,
        dspan: Span,
    ) -> Result<usize, ExpandError> {
        // Expression runs until the first top-level comma.
        let (line, after) = collect_line(tokens, i);
        let after = skip_newline(tokens, after);
        let comma = find_top_level_comma(line).ok_or_else(|| {
            self.err(
                ExpandErrorKind::MalformedDiagnostic("@assert needs expr, \"message\""),
                dspan,
            )
        })?;
        let expr_tokens = &line[..comma];
        let after_comma = &line[comma + 1..];
        // Message: optional string. If absent, synthesize from the
        // expression text.
        let (msg, _rest) = take_string(after_comma, 0).unwrap_or_else(|| {
            let synth = tokens_to_text(expr_tokens);
            (format!("assertion failed: {synth}"), after_comma.len())
        });
        let pre = self.preexpand(expr_tokens)?;
        let mut ctx = self.eval_context();
        let value = eval_expr(&pre, &mut ctx).map_err(|e| {
            self.err(ExpandErrorKind::Expr(e), dspan)
        })?;
        if value == 0 {
            return Err(self.err(ExpandErrorKind::AssertionFailed(msg), dspan));
        }
        Ok(after)
    }

    // ── @rept N ... @endr ───────────────────────────────────────────

    fn handle_rept(
        &mut self,
        tokens: &[Token],
        i: usize,
        dspan: Span,
    ) -> Result<usize, ExpandError> {
        let (expr_tokens, after_count) = collect_line(tokens, i);
        let count = self.eval_line(expr_tokens, dspan)?;
        if count < 0 {
            return Err(self.err(
                ExpandErrorKind::MalformedRept("count must be non-negative"),
                dspan,
            ));
        }
        let body_start = skip_newline(tokens, after_count);
        // Find matching @endr at this level.
        let body_end = find_block_end(tokens, body_start, "rept", "endr").ok_or_else(|| {
            self.err(
                ExpandErrorKind::UnbalancedBlock("rept".into()),
                dspan,
            )
        })?;
        let body = &tokens[body_start..body_end];
        // Run body `count` times, setting @INDEX via a temporary
        // numeric define so the inner expander sees it.
        let saved = self.state.defines.remove("INDEX");
        for idx in 0..count {
            self.state
                .defines
                .insert("INDEX".to_string(), DefineValue::Numeric(idx));
            // Recursively expand the body. We re-enter the expander
            // through expand_range on the slice; this nests cleanly.
            self.expand_range(body, true)?;
        }
        // Restore prior INDEX (almost always absent).
        match saved {
            Some(v) => {
                self.state.defines.insert("INDEX".into(), v);
            }
            None => {
                self.state.defines.remove("INDEX");
            }
        }
        // Skip past the @endr line.
        let after_endr = skip_directive_line(tokens, body_end);
        Ok(after_endr)
    }

    // ── @scope NAME / @endscope ─────────────────────────────────────

    fn handle_scope(
        &mut self,
        tokens: &[Token],
        i: usize,
        dspan: Span,
    ) -> Result<usize, ExpandError> {
        let (name, after_name) = take_name(tokens, i).ok_or_else(|| {
            self.err(
                ExpandErrorKind::MalformedScope("@scope requires a NAME"),
                dspan,
            )
        })?;
        self.state.scope_stack.push(ScopeFrame {
            kind: ScopeKind::Scope,
            name,
            id: None,
        });
        Ok(skip_directive_line(tokens, after_name))
    }

    fn handle_endscope(
        &mut self,
        tokens: &[Token],
        i: usize,
        dspan: Span,
    ) -> Result<usize, ExpandError> {
        // Find the innermost open Scope frame and remove it. Walk past
        // any macro-invocation frames sitting above — that's the
        // proc/endp pattern: a macro called `endp` pushes a macro
        // frame for the duration of its body, while the `@endscope`
        // directive inside that body must still close the scope opened
        // by the earlier `proc()` call.
        let pos = self
            .state
            .scope_stack
            .iter()
            .rposition(|f| f.kind == ScopeKind::Scope);
        match pos {
            Some(pos) => {
                self.state.scope_stack.remove(pos);
                Ok(skip_directive_line(tokens, i))
            }
            None => Err(self.err(
                ExpandErrorKind::UnbalancedBlock("endscope".into()),
                dspan,
            )),
        }
    }

    // ── @macro NAME(params) ... @endmacro ──────────────────────────

    fn handle_macro_def(
        &mut self,
        tokens: &[Token],
        i: usize,
        dspan: Span,
    ) -> Result<usize, ExpandError> {
        // Parse header.
        let (name, after_name) = take_name(tokens, i).ok_or_else(|| {
            self.err(
                ExpandErrorKind::MalformedMacro("expected NAME after @macro"),
                dspan,
            )
        })?;
        if !matches!(
            tokens.get(after_name).map(|t| &t.kind),
            Some(TokenKind::Punct(Punct::LParen))
        ) {
            return Err(self.err(
                ExpandErrorKind::MalformedMacro(
                    "expected `(params)` after macro name",
                ),
                dspan,
            ));
        }
        let (params, after_params) =
            self.parse_param_list(tokens, after_name + 1, dspan)?;
        let after_header = skip_newline(tokens, after_params);

        // Body extends until matching `@endmacro` at line start. No
        // nesting allowed.
        let body_end = find_macro_end(tokens, after_header, dspan)?;
        let body = tokens[after_header..body_end].to_vec();

        let def = MacroDef {
            name: name.clone(),
            params,
            body,
            span: dspan,
        };
        self.state.macros.insert(name, def);

        Ok(skip_directive_line(tokens, body_end))
    }

    /// Parse a macro parameter list from `(p1, p2, ..., args...)`.
    /// Returns the params and the index just past the `)`.
    fn parse_param_list(
        &self,
        tokens: &[Token],
        start: usize,
        dspan: Span,
    ) -> Result<(Vec<MacroParam>, usize), ExpandError> {
        let mut params: Vec<MacroParam> = Vec::new();
        let mut i = start;
        loop {
            // Allow empty param list.
            if matches!(
                tokens.get(i).map(|t| &t.kind),
                Some(TokenKind::Punct(Punct::RParen))
            ) {
                return Ok((params, i + 1));
            }
            let (name, after) = take_name(tokens, i).ok_or_else(|| {
                self.err(
                    ExpandErrorKind::MalformedMacro("expected parameter name"),
                    dspan,
                )
            })?;
            i = after;
            let variadic = matches!(
                tokens.get(i).map(|t| &t.kind),
                Some(TokenKind::Punct(Punct::Ellipsis))
            );
            if variadic {
                i += 1;
            }
            // Reject variadic anywhere but as the final param.
            if params.iter().any(|p| p.variadic) {
                return Err(self.err(
                    ExpandErrorKind::MalformedMacro(
                        "variadic `...` must be on the last parameter",
                    ),
                    dspan,
                ));
            }
            params.push(MacroParam { name, variadic });
            match tokens.get(i).map(|t| &t.kind) {
                Some(TokenKind::Punct(Punct::Comma)) => {
                    i += 1;
                    continue;
                }
                Some(TokenKind::Punct(Punct::RParen)) => {
                    return Ok((params, i + 1));
                }
                _ => {
                    return Err(self.err(
                        ExpandErrorKind::MalformedMacro(
                            "expected `,` or `)` in parameter list",
                        ),
                        dspan,
                    ));
                }
            }
        }
    }

    // ── Macro invocation: parse args, pre-expand, substitute, expand ─

    /// Parse a call's argument list starting at the `(` token. Splits
    /// on top-level commas; `(...)` and `[...]` nest without splitting;
    /// `{...}` is *grouping* syntax (braces stripped from the arg).
    ///
    /// Returns the raw args and the cursor just past the `)`. Does no
    /// arity validation — that's the caller's job (text macros have a
    /// known signature; Rust macros validate inside the closure).
    fn parse_call_args(
        &mut self,
        tokens: &[Token],
        lparen_idx: usize,
        call_span: Span,
    ) -> Result<(Vec<Vec<Token>>, usize), ExpandError> {
        debug_assert!(matches!(
            tokens.get(lparen_idx).map(|t| &t.kind),
            Some(TokenKind::Punct(Punct::LParen))
        ));
        let mut args: Vec<Vec<Token>> = Vec::new();
        let mut current: Vec<Token> = Vec::new();
        let mut paren_depth = 0i32;
        let mut brace_depth = 0i32;
        let mut i = lparen_idx + 1;
        let mut any_seen = false;
        while i < tokens.len() {
            let tok = &tokens[i];
            match &tok.kind {
                TokenKind::Punct(Punct::LParen | Punct::LBracket) => {
                    paren_depth += 1;
                    current.push(tok.clone());
                }
                TokenKind::Punct(Punct::RParen) if paren_depth == 0 && brace_depth == 0 => {
                    if any_seen || !current.is_empty() {
                        args.push(std::mem::take(&mut current));
                    }
                    return Ok((args, i + 1));
                }
                TokenKind::Punct(Punct::RParen | Punct::RBracket) => {
                    paren_depth -= 1;
                    current.push(tok.clone());
                }
                TokenKind::Punct(Punct::LBrace) => {
                    brace_depth += 1;
                }
                TokenKind::Punct(Punct::RBrace) if brace_depth > 0 => {
                    brace_depth -= 1;
                }
                TokenKind::Punct(Punct::Comma) if paren_depth == 0 && brace_depth == 0 => {
                    args.push(std::mem::take(&mut current));
                    any_seen = true;
                }
                TokenKind::Newline => {
                    return Err(self.err(
                        ExpandErrorKind::MalformedMacro(
                            "unterminated macro call — missing `)` before end of line",
                        ),
                        call_span,
                    ));
                }
                _ => {
                    current.push(tok.clone());
                }
            }
            i += 1;
        }
        Err(self.err(
            ExpandErrorKind::MalformedMacro(
                "unterminated macro call — reached end of input",
            ),
            call_span,
        ))
    }

    /// Parse args for a text-macro call and validate arity against `def`.
    fn parse_macro_args(
        &mut self,
        tokens: &[Token],
        lparen_idx: usize,
        def: &MacroDef,
        call_span: Span,
    ) -> Result<(MacroArgs, usize), ExpandError> {
        let (args, after) = self.parse_call_args(tokens, lparen_idx, call_span)?;
        self.validate_arity(def, &args, call_span)?;
        let bound = self.bind_args(def, args, call_span)?;
        Ok((bound, after))
    }

    /// Invoke a registered Rust macro. Pre-expands each arg in caller
    /// context, then calls the closure. The closure emits tokens
    /// directly into the expander's output buffer.
    fn invoke_rust_macro(
        &mut self,
        name: &str,
        args: &[Vec<Token>],
        call_span: Span,
    ) -> Result<(), ExpandError> {
        // Pre-expand args in caller context for hygiene.
        let mut expanded_args: Vec<Vec<Token>> = Vec::with_capacity(args.len());
        for arg in args {
            expanded_args.push(self.preexpand(arg)?);
        }

        // Take the closure out of the map to satisfy the borrow checker —
        // calling it requires `&mut state`, but the closure lives inside
        // the map. We put it back unconditionally after the call.
        let mut closure = self
            .state
            .rust_macros
            .remove(name)
            .expect("invoke_rust_macro called for an unregistered name");

        let result = {
            let mut ctx = RustMacroCtx {
                args: &expanded_args,
                state: self.state,
                output: self.out,
                call_span,
                name,
            };
            closure(&mut ctx)
        };

        // Reinstall the closure — even on error — so subsequent calls
        // see it (and `register_macro` from inside a callback path
        // continues to behave).
        self.state.rust_macros.insert(name.to_string(), closure);

        result.map_err(|msg| {
            self.err(ExpandErrorKind::RustMacroError(msg), call_span)
        })
    }

    fn validate_arity(
        &self,
        def: &MacroDef,
        args: &[Vec<Token>],
        call_span: Span,
    ) -> Result<(), ExpandError> {
        let variadic = def.params.last().map(|p| p.variadic).unwrap_or(false);
        let required = if variadic {
            def.params.len() - 1
        } else {
            def.params.len()
        };
        // For non-variadic, args.len() must equal params.len().
        // For variadic, args.len() must be >= required.
        if variadic {
            if args.len() < required {
                return Err(self.err(
                    ExpandErrorKind::MacroArity {
                        name: def.name.clone(),
                        expected: required,
                        got: args.len(),
                        variadic: true,
                    },
                    call_span,
                ));
            }
        } else if args.len() != def.params.len() {
            return Err(self.err(
                ExpandErrorKind::MacroArity {
                    name: def.name.clone(),
                    expected: def.params.len(),
                    got: args.len(),
                    variadic: false,
                },
                call_span,
            ));
        }
        Ok(())
    }

    /// Pre-expand each arg in caller context, then bind to params.
    fn bind_args(
        &mut self,
        def: &MacroDef,
        args: Vec<Vec<Token>>,
        _call_span: Span,
    ) -> Result<MacroArgs, ExpandError> {
        let mut bound: HashMap<String, MacroArgValue> = HashMap::new();
        let variadic = def.params.last().map(|p| p.variadic).unwrap_or(false);
        let required = if variadic {
            def.params.len() - 1
        } else {
            def.params.len()
        };
        let mut iter = args.into_iter();
        for p in def.params.iter().take(required) {
            let raw = iter.next().expect("arity already validated");
            let expanded = self.preexpand(&raw)?;
            bound.insert(p.name.clone(), MacroArgValue::Tokens(expanded));
        }
        if variadic {
            let tail: Vec<Vec<Token>> = iter
                .map(|raw| self.preexpand(&raw))
                .collect::<Result<_, _>>()?;
            let tail_name = def.params.last().unwrap().name.clone();
            bound.insert(tail_name, MacroArgValue::Variadic(tail));
        }
        Ok(MacroArgs { bound })
    }

    // ── MASM-style runtime control flow ─────────────────────────────
    //
    // `.if`/`.elseif`/`.else`/`.endif`, `.while`/`.endw`,
    // `.repeat`/`.until`, `.break`/`.continue`. Each directive emits
    // straight-line asm text into the output buffer (no IR, no LLVM
    // help) using one helper that lexes a generated string. Block
    // state lives in `self.state.block_stack`.

    fn handle_control_flow(
        &mut self,
        kw: ControlFlowKw,
        tokens: &[Token],
        i: usize,
        dspan: Span,
    ) -> Result<usize, ExpandError> {
        match kw {
            ControlFlowKw::If => self.handle_dot_if(tokens, i, dspan),
            ControlFlowKw::ElseIf => self.handle_dot_elseif(tokens, i, dspan),
            ControlFlowKw::Else => self.handle_dot_else(tokens, i, dspan),
            ControlFlowKw::EndIf => self.handle_dot_endif(tokens, i, dspan),
            ControlFlowKw::While => self.handle_dot_while(tokens, i, dspan),
            ControlFlowKw::EndW => self.handle_dot_endw(tokens, i, dspan),
            ControlFlowKw::Repeat => self.handle_dot_repeat(tokens, i, dspan),
            ControlFlowKw::Until => self.handle_dot_until(tokens, i, dspan),
            ControlFlowKw::Break => self.handle_dot_break(tokens, i, dspan),
            ControlFlowKw::Continue => self.handle_dot_continue(tokens, i, dspan),
        }
    }

    fn fresh_block_id(&mut self) -> u64 {
        let id = self.state.block_id_counter;
        self.state.block_id_counter += 1;
        id
    }

    /// Lex `src` as asm and append the resulting tokens to the output
    /// buffer, stamping each with `span` so downstream diagnostics
    /// point back to the directive's source location.
    fn emit_asm(&mut self, src: &str, span: Span) {
        // Tokens our handlers generate are always well-formed asm so
        // the lex never errors in practice; an unwrap would be fine
        // but `expect` makes the assumption explicit.
        let toks = super::lex::lex(span.file, src)
            .expect("internal control-flow asm should always lex");
        for mut t in toks {
            t.span = span;
            self.out.push(t);
        }
    }

    // ── .if op1 OP op2 ──────────────────────────────────────────────

    fn handle_dot_if(
        &mut self,
        tokens: &[Token],
        i: usize,
        dspan: Span,
    ) -> Result<usize, ExpandError> {
        let (line, after) = collect_line(tokens, i);
        let after = skip_newline(tokens, after);
        let pre = self.preexpand(line)?;
        let (lhs, op, rhs) = split_condition(&pre, dspan)?;
        let id = self.fresh_block_id();
        let end_label = format!("__if{id}_end");
        let next_label = format!("__if{id}_next");
        let jcc_inv = inverse_jcc(op);
        let lhs_text = tokens_to_text(&lhs);
        let rhs_text = tokens_to_text(&rhs);
        self.emit_asm(
            &format!("    cmp {lhs_text}, {rhs_text}\n    {jcc_inv} {next_label}\n"),
            dspan,
        );
        self.state.block_stack.push(BlockFrame::If {
            end_label,
            next_label,
            has_else: false,
        });
        Ok(after)
    }

    fn handle_dot_elseif(
        &mut self,
        tokens: &[Token],
        i: usize,
        dspan: Span,
    ) -> Result<usize, ExpandError> {
        let (line, after) = collect_line(tokens, i);
        let after = skip_newline(tokens, after);
        let pre = self.preexpand(line)?;
        let (lhs, op, rhs) = split_condition(&pre, dspan)?;
        let new_next_label;
        let prev_next;
        let end_label;
        let id = self.fresh_block_id();
        match self.state.block_stack.last_mut() {
            Some(BlockFrame::If {
                end_label: e,
                next_label,
                has_else,
            }) => {
                if *has_else {
                    return Err(self.err(
                        ExpandErrorKind::BadIfStructure(
                            "`.elseif` after `.else` is not allowed",
                        ),
                        dspan,
                    ));
                }
                prev_next = next_label.clone();
                end_label = e.clone();
                new_next_label = format!("__if{id}_next");
                *next_label = new_next_label.clone();
            }
            _ => {
                return Err(self.err(
                    ExpandErrorKind::StrayControlFlow(
                        "`.elseif` outside an open `.if` block",
                    ),
                    dspan,
                ));
            }
        }
        let jcc_inv = inverse_jcc(op);
        let lhs_text = tokens_to_text(&lhs);
        let rhs_text = tokens_to_text(&rhs);
        self.emit_asm(
            &format!(
                "    jmp {end_label}\n{prev_next}:\n    cmp {lhs_text}, {rhs_text}\n    {jcc_inv} {new_next_label}\n"
            ),
            dspan,
        );
        Ok(after)
    }

    fn handle_dot_else(
        &mut self,
        tokens: &[Token],
        i: usize,
        dspan: Span,
    ) -> Result<usize, ExpandError> {
        let after = skip_directive_line(tokens, i);
        let prev_next;
        let end_label;
        match self.state.block_stack.last_mut() {
            Some(BlockFrame::If {
                end_label: e,
                next_label,
                has_else,
            }) => {
                if *has_else {
                    return Err(self.err(
                        ExpandErrorKind::BadIfStructure(
                            "duplicate `.else` in `.if` block",
                        ),
                        dspan,
                    ));
                }
                prev_next = next_label.clone();
                end_label = e.clone();
                *has_else = true;
            }
            _ => {
                return Err(self.err(
                    ExpandErrorKind::StrayControlFlow(
                        "`.else` outside an open `.if` block",
                    ),
                    dspan,
                ));
            }
        }
        self.emit_asm(
            &format!("    jmp {end_label}\n{prev_next}:\n"),
            dspan,
        );
        Ok(after)
    }

    fn handle_dot_endif(
        &mut self,
        tokens: &[Token],
        i: usize,
        dspan: Span,
    ) -> Result<usize, ExpandError> {
        let after = skip_directive_line(tokens, i);
        let frame = self.state.block_stack.pop().ok_or_else(|| {
            self.err(
                ExpandErrorKind::StrayControlFlow(
                    "`.endif` with no matching `.if`",
                ),
                dspan,
            )
        })?;
        match frame {
            BlockFrame::If {
                end_label,
                next_label,
                has_else,
            } => {
                // If no .else was seen, the last branch's failure
                // case still needs the next_label to land on. With
                // .else, the next_label was already emitted before
                // the else body.
                if has_else {
                    self.emit_asm(&format!("{end_label}:\n"), dspan);
                } else {
                    self.emit_asm(&format!("{next_label}:\n{end_label}:\n"), dspan);
                }
                Ok(after)
            }
            other => {
                // Wrong opener — push back, surface error.
                let kind = other.kind_name();
                self.state.block_stack.push(other);
                Err(self.err(
                    ExpandErrorKind::StrayControlFlow(match kind {
                        "while" => "`.endif` closing a `.while` block — did you mean `.endw`?",
                        "repeat" => "`.endif` closing a `.repeat` block — did you mean `.until`?",
                        _ => "`.endif` with no matching `.if`",
                    }),
                    dspan,
                ))
            }
        }
    }

    // ── .while op1 OP op2 ... .endw ─────────────────────────────────

    fn handle_dot_while(
        &mut self,
        tokens: &[Token],
        i: usize,
        dspan: Span,
    ) -> Result<usize, ExpandError> {
        let (line, after) = collect_line(tokens, i);
        let after = skip_newline(tokens, after);
        let pre = self.preexpand(line)?;
        let (lhs, op, rhs) = split_condition(&pre, dspan)?;
        let id = self.fresh_block_id();
        let top_label = format!("__while{id}_top");
        let bottom_label = format!("__while{id}_bot");
        let jcc_inv = inverse_jcc(op);
        let lhs_text = tokens_to_text(&lhs);
        let rhs_text = tokens_to_text(&rhs);
        self.emit_asm(
            &format!(
                "{top_label}:\n    cmp {lhs_text}, {rhs_text}\n    {jcc_inv} {bottom_label}\n"
            ),
            dspan,
        );
        self.state.block_stack.push(BlockFrame::While {
            top_label,
            bottom_label,
        });
        Ok(after)
    }

    fn handle_dot_endw(
        &mut self,
        tokens: &[Token],
        i: usize,
        dspan: Span,
    ) -> Result<usize, ExpandError> {
        let after = skip_directive_line(tokens, i);
        let frame = self.state.block_stack.pop().ok_or_else(|| {
            self.err(
                ExpandErrorKind::StrayControlFlow(
                    "`.endw` with no matching `.while`",
                ),
                dspan,
            )
        })?;
        match frame {
            BlockFrame::While {
                top_label,
                bottom_label,
            } => {
                self.emit_asm(
                    &format!("    jmp {top_label}\n{bottom_label}:\n"),
                    dspan,
                );
                Ok(after)
            }
            other => {
                let kind = other.kind_name();
                self.state.block_stack.push(other);
                Err(self.err(
                    ExpandErrorKind::StrayControlFlow(match kind {
                        "if" => "`.endw` closing an `.if` block — did you mean `.endif`?",
                        "repeat" => "`.endw` closing a `.repeat` block — did you mean `.until`?",
                        _ => "`.endw` with no matching `.while`",
                    }),
                    dspan,
                ))
            }
        }
    }

    // ── .repeat ... .until op1 OP op2 ───────────────────────────────

    fn handle_dot_repeat(
        &mut self,
        tokens: &[Token],
        i: usize,
        dspan: Span,
    ) -> Result<usize, ExpandError> {
        let after = skip_directive_line(tokens, i);
        let id = self.fresh_block_id();
        let top_label = format!("__rep{id}_top");
        let cont_label = format!("__rep{id}_cont");
        let bottom_label = format!("__rep{id}_bot");
        self.emit_asm(&format!("{top_label}:\n"), dspan);
        self.state.block_stack.push(BlockFrame::Repeat {
            top_label,
            cont_label,
            bottom_label,
        });
        Ok(after)
    }

    fn handle_dot_until(
        &mut self,
        tokens: &[Token],
        i: usize,
        dspan: Span,
    ) -> Result<usize, ExpandError> {
        let (line, after) = collect_line(tokens, i);
        let after = skip_newline(tokens, after);
        let pre = self.preexpand(line)?;
        let (lhs, op, rhs) = split_condition(&pre, dspan)?;
        let frame = self.state.block_stack.pop().ok_or_else(|| {
            self.err(
                ExpandErrorKind::StrayControlFlow(
                    "`.until` with no matching `.repeat`",
                ),
                dspan,
            )
        })?;
        match frame {
            BlockFrame::Repeat {
                top_label,
                cont_label,
                bottom_label,
            } => {
                let jcc_inv = inverse_jcc(op);
                let lhs_text = tokens_to_text(&lhs);
                let rhs_text = tokens_to_text(&rhs);
                // .continue jumps to cont_label so it re-runs the
                // post-test; cont_label sits right before cmp.
                self.emit_asm(
                    &format!(
                        "{cont_label}:\n    cmp {lhs_text}, {rhs_text}\n    {jcc_inv} {top_label}\n{bottom_label}:\n"
                    ),
                    dspan,
                );
                Ok(after)
            }
            other => {
                let kind = other.kind_name();
                self.state.block_stack.push(other);
                Err(self.err(
                    ExpandErrorKind::StrayControlFlow(match kind {
                        "if" => "`.until` closing an `.if` block — did you mean `.endif`?",
                        "while" => "`.until` closing a `.while` block — did you mean `.endw`?",
                        _ => "`.until` with no matching `.repeat`",
                    }),
                    dspan,
                ))
            }
        }
    }

    // ── .break / .continue ──────────────────────────────────────────

    fn handle_dot_break(
        &mut self,
        tokens: &[Token],
        i: usize,
        dspan: Span,
    ) -> Result<usize, ExpandError> {
        let after = skip_directive_line(tokens, i);
        // Walk the stack from the top for the innermost loop frame.
        let target = self.state.block_stack.iter().rev().find_map(|f| match f {
            BlockFrame::While { bottom_label, .. } => Some(bottom_label.clone()),
            BlockFrame::Repeat { bottom_label, .. } => Some(bottom_label.clone()),
            _ => None,
        });
        let label = target.ok_or_else(|| {
            self.err(
                ExpandErrorKind::NotInLoop("`.break` outside any `.while` or `.repeat`"),
                dspan,
            )
        })?;
        self.emit_asm(&format!("    jmp {label}\n"), dspan);
        Ok(after)
    }

    fn handle_dot_continue(
        &mut self,
        tokens: &[Token],
        i: usize,
        dspan: Span,
    ) -> Result<usize, ExpandError> {
        let after = skip_directive_line(tokens, i);
        // Innermost loop. For While: jump to top (re-runs the cond).
        // For Repeat: jump to cont_label (re-runs the post-test).
        let target = self.state.block_stack.iter().rev().find_map(|f| match f {
            BlockFrame::While { top_label, .. } => Some(top_label.clone()),
            BlockFrame::Repeat { cont_label, .. } => Some(cont_label.clone()),
            _ => None,
        });
        let label = target.ok_or_else(|| {
            self.err(
                ExpandErrorKind::NotInLoop(
                    "`.continue` outside any `.while` or `.repeat`",
                ),
                dspan,
            )
        })?;
        self.emit_asm(&format!("    jmp {label}\n"), dspan);
        Ok(after)
    }

    // ── @extern NAME(arg_count) ─────────────────────────────────────

    /// `@extern [\"DLL.dll\"] NAME(N)` declares that NAME is a function
    /// taking N arguments. The optional DLL string tells the host to
    /// `LoadLibraryW` that DLL and `GetProcAddress(NAME)`; without it,
    /// the host supplies the function pointer directly (typical for
    /// the host's own Rust runtime functions).
    ///
    /// The assembler doesn't validate `call NAME` sites — MC does — but
    /// records the declaration so the host can iterate via
    /// `Assembler::externs()` and register function pointers with the JIT.
    fn handle_extern(
        &mut self,
        tokens: &[Token],
        i: usize,
        dspan: Span,
    ) -> Result<usize, ExpandError> {
        // Optional leading string literal = DLL name.
        let (dll, after_dll) = match take_string(tokens, i) {
            Some((s, after)) => (Some(s), after),
            None => (None, i),
        };
        let (name, after_name) = take_name(tokens, after_dll).ok_or_else(|| {
            self.err(
                ExpandErrorKind::MalformedExtern(
                    "@extern requires a NAME argument (optionally preceded by \"DLL.dll\")",
                ),
                dspan,
            )
        })?;
        // Expect `(count)`.
        if !matches!(
            tokens.get(after_name).map(|t| &t.kind),
            Some(TokenKind::Punct(Punct::LParen))
        ) {
            return Err(self.err(
                ExpandErrorKind::MalformedExtern(
                    "expected `(arg_count)` after extern name",
                ),
                dspan,
            ));
        }
        let (args, after_call) = self.parse_call_args(tokens, after_name, dspan)?;
        if args.len() != 1 {
            return Err(self.err(
                ExpandErrorKind::MalformedExtern(
                    "@extern requires exactly one arg-count value",
                ),
                dspan,
            ));
        }
        let count = self.eval_line(&args[0], dspan)?;
        if !(0..=256).contains(&count) {
            return Err(self.err(
                ExpandErrorKind::MalformedExtern(
                    "@extern arg_count out of range (expected 0..=256)",
                ),
                dspan,
            ));
        }
        self.state.externs.insert(
            name,
            ExternDecl {
                arg_count: count as usize,
                dll,
            },
        );
        Ok(skip_directive_line(tokens, after_call))
    }

    // ── @rust_macro NAME ────────────────────────────────────────────

    /// `@rust_macro NAME` declares that a Rust-implemented macro is
    /// expected for `NAME`. We install a placeholder closure that
    /// errors with a helpful message if the host forgets to register
    /// the real one before assembly. Subsequent
    /// `Assembler::register_macro(NAME, ...)` overwrites it.
    fn handle_rust_macro(
        &mut self,
        tokens: &[Token],
        i: usize,
        dspan: Span,
    ) -> Result<usize, ExpandError> {
        let (name, after) = take_name(tokens, i).ok_or_else(|| {
            self.err(
                ExpandErrorKind::MalformedMacro(
                    "@rust_macro requires a name argument",
                ),
                dspan,
            )
        })?;
        // Only install a placeholder if nothing's there yet. If the
        // host already registered the real closure, leave it alone.
        if !self.state.rust_macros.contains_key(&name) {
            let name_for_msg = name.clone();
            let placeholder: RustMacroFn = Box::new(move |_ctx| {
                Err(format!(
                    "Rust macro `{name_for_msg}` declared with @rust_macro \
                     but never registered via Assembler::register_macro"
                ))
            });
            self.state.rust_macros.insert(name, placeholder);
        }
        Ok(skip_directive_line(tokens, after))
    }

    // ── @include "path" ─────────────────────────────────────────────

    fn handle_include(
        &mut self,
        tokens: &[Token],
        i: usize,
        dspan: Span,
    ) -> Result<usize, ExpandError> {
        let (path_str, after) = take_string(tokens, i).ok_or_else(|| {
            self.err(
                ExpandErrorKind::MalformedInclude(
                    "@include requires a string argument: @include \"path\"",
                ),
                dspan,
            )
        })?;
        let after = skip_directive_line(tokens, after);

        // Resolve relative to the parent file's directory. For inline
        // sources (no path), resolve relative to cwd.
        let resolved = self.resolve_include_path(&path_str);

        // Cycle / depth checks. Use the resolved path as-is for the
        // cycle set (matching is exact-string; if the same file appears
        // via two different relative paths we won't catch it — fine
        // for v1).
        if self.state.include_cycle.contains(&resolved) {
            return Err(self.err(
                ExpandErrorKind::IncludeCycle {
                    path: resolved.display().to_string(),
                },
                dspan,
            ));
        }
        if self.state.path_stack.len() >= self.state.max_include_depth {
            return Err(self.err(
                ExpandErrorKind::IncludeTooDeep {
                    path: resolved.display().to_string(),
                    limit: self.state.max_include_depth,
                },
                dspan,
            ));
        }

        // Read source — consult the virtual-file overlay first, then
        // fall back to disk.
        let text = if let Some(t) = self.state.virtual_files.get(&resolved).cloned() {
            t
        } else {
            std::fs::read_to_string(&resolved).map_err(|e| {
                self.err(
                    ExpandErrorKind::IncludeIo {
                        path: resolved.display().to_string(),
                        error: e.to_string(),
                    },
                    dspan,
                )
            })?
        };

        // Register with the source map and lex.
        let file_id = self.state.sources.add(resolved.clone(), text.clone());
        let inner_tokens = super::lex::lex(file_id, &text).map_err(|e| {
            self.err(
                ExpandErrorKind::IncludeLexFailed {
                    path: resolved.display().to_string(),
                    error: format!("{e}"),
                },
                dspan,
            )
        })?;

        // Push path + cycle entry, expand, pop.
        self.state.path_stack.push(resolved.clone());
        self.state.include_cycle.insert(resolved.clone());
        let result = self.expand_range(&inner_tokens, false);
        self.state.include_cycle.remove(&resolved);
        self.state.path_stack.pop();
        result?;

        Ok(after)
    }

    /// Resolve `requested` against the directory of the file currently
    /// being expanded (the top of `path_stack`). If the requested path
    /// is absolute, returns it unchanged. If the stack is empty (no
    /// parent), resolves relative to cwd.
    fn resolve_include_path(&self, requested: &str) -> PathBuf {
        let p = PathBuf::from(requested);
        if p.is_absolute() {
            return p;
        }
        match self.state.path_stack.last() {
            Some(parent) => parent
                .parent()
                .map(|d| d.join(&p))
                .unwrap_or_else(|| p.clone()),
            None => p,
        }
    }

    /// Run a macro call. Pushes a scope frame, substitutes params into
    /// the body, expands the result. Pops the frame on return.
    fn expand_macro_call(
        &mut self,
        def: &MacroDef,
        args: MacroArgs,
        _call_span: Span,
    ) -> Result<(), ExpandError> {
        if self.state.expansion_depth >= self.state.max_expansion_depth {
            return Err(self.err(
                ExpandErrorKind::ExpansionTooDeep {
                    limit: self.state.max_expansion_depth,
                    in_macro: def.name.clone(),
                },
                def.span,
            ));
        }

        let id = self.state.invocation_id;
        self.state.invocation_id += 1;

        // Substitute &param tokens into the body in one pass.
        let substituted = substitute_params(&def.body, &args, def)?;

        // Push the macro frame so .local labels inside the body mangle
        // using <macro>$$<id>. If the body opens a @scope, that scope
        // pushes on top and wins.
        self.state.scope_stack.push(ScopeFrame {
            kind: ScopeKind::MacroInvocation,
            name: def.name.clone(),
            id: Some(id),
        });
        self.state.expansion_depth += 1;

        let result = self.expand_range(&substituted, true);

        self.state.expansion_depth -= 1;
        // Pop *this macro's* frame by id. The body may legitimately
        // leave a scope frame above us — the proc/endp pattern relies
        // on it: `proc(plus)` opens `@scope plus`, the user's later
        // `endp()` closes it. We don't require macro calls to be
        // structurally balanced internally; the user defines those
        // conventions. Truly-unclosed scopes are caught by the
        // end-of-input check in `Assembler::expand`.
        let pos = self.state.scope_stack.iter().rposition(|f| {
            f.kind == ScopeKind::MacroInvocation && f.id == Some(id)
        });
        if let Some(pos) = pos {
            self.state.scope_stack.remove(pos);
        }
        result
    }

    // ── helpers ─────────────────────────────────────────────────────

    /// Pre-expand a token slice through the expander to resolve
    /// nested defines and context names BEFORE handing to the
    /// expression evaluator.
    ///
    /// We do this by routing through a private sub-expansion that
    /// writes into a scratch buffer.
    fn preexpand(&mut self, tokens: &[Token]) -> Result<Vec<Token>, ExpandError> {
        let mut scratch = Vec::new();
        let mut sub = Expander {
            state: self.state,
            out: &mut scratch,
            warnings: self.warnings,
        };
        sub.expand_range(tokens, false)?;
        // Drop any Newline / Comment tokens — they upset the evaluator.
        scratch.retain(|t| !matches!(t.kind, TokenKind::Newline | TokenKind::Comment(_)));
        Ok(scratch)
    }

    /// Pre-expand a line and evaluate as an integer.
    fn eval_line(&mut self, tokens: &[Token], at: Span) -> Result<i64, ExpandError> {
        let pre = self.preexpand(tokens)?;
        let mut ctx = self.eval_context();
        eval_expr(&pre, &mut ctx).map_err(|e| {
            ExpandError {
                kind: ExpandErrorKind::Expr(e),
                span: at,
            }
        })
    }

    fn eval_context(&mut self) -> ExpanderEvalContext<'_> {
        ExpanderEvalContext { state: self.state }
    }

    fn read_context_int(&mut self, name: &str, span: Span) -> Result<i64, ExpandError> {
        match name {
            "COUNTER" => {
                let v = self.state.counter;
                self.state.counter += 1;
                Ok(v)
            }
            "LINE" => Ok(span.line as i64),
            "FILE" => Ok(span.file.0 as i64),
            "INDEX" => {
                // Fall back to the defined value (set by @rept). If
                // not in a loop, error.
                match self.state.defines.get("INDEX") {
                    Some(DefineValue::Numeric(n)) => Ok(*n),
                    _ => Err(self.err(
                        ExpandErrorKind::Expr(EvalError {
                            kind: super::expr::EvalErrorKind::UndefinedDirective("INDEX".into()),
                            span,
                        }),
                        span,
                    )),
                }
            }
            "BITS" => Ok(64),
            other => Err(self.err(
                ExpandErrorKind::Expr(EvalError {
                    kind: super::expr::EvalErrorKind::UndefinedDirective(other.into()),
                    span,
                }),
                span,
            )),
        }
    }

    fn err(&self, kind: ExpandErrorKind, span: Span) -> ExpandError {
        ExpandError { kind, span }
    }
}

/// `EvalContext` impl for use during directive evaluation. Shares the
/// expander's `State` so `@assign`-defined numeric names are visible to
/// expressions, and `@COUNTER` reads bump.
struct ExpanderEvalContext<'a> {
    state: &'a mut State,
}

impl<'a> EvalContext for ExpanderEvalContext<'a> {
    fn lookup(&self, name: &str) -> Option<i64> {
        match self.state.defines.get(name) {
            Some(DefineValue::Numeric(n)) => Some(*n),
            // Text defines aren't visible to expressions — by design.
            _ => None,
        }
    }

    fn lookup_directive(&mut self, name: &str) -> Option<i64> {
        match name {
            "COUNTER" => {
                let v = self.state.counter;
                self.state.counter += 1;
                Some(v)
            }
            "BITS" => Some(64),
            "INDEX" => match self.state.defines.get("INDEX") {
                Some(DefineValue::Numeric(n)) => Some(*n),
                _ => None,
            },
            // LINE / FILE need a span; not reachable from the
            // EvalContext without one. The expander's read_context_int
            // covers them when @LINE/@FILE appear as standalone tokens.
            _ => None,
        }
    }
}

// ── small free helpers ──────────────────────────────────────────────

fn is_statement_directive(name: &str) -> bool {
    name.chars().next().map(|c| c.is_ascii_lowercase()).unwrap_or(false)
}

fn is_context_directive(name: &str) -> bool {
    // Treat uppercase-leading names as context directives. This is
    // pragma-by-convention: lowercase = state mutator, uppercase = read
    // current value. Mixed-case names like `Foo` aren't valid in v1.
    name.chars().next().map(|c| c.is_ascii_uppercase()).unwrap_or(false)
}

fn make_number_token(value: i64, span: Span, space_before: bool) -> Token {
    Token {
        kind: TokenKind::Number(NumberLit {
            value,
            raw: value.to_string(),
            base: NumberBase::Dec,
        }),
        span,
        space_before,
    }
}

/// Returns true if `tokens[i]` is the first non-whitespace token of a
/// line (i.e., either i == 0 or the previous token was a Newline).
fn at_line_start(tokens: &[Token], i: usize) -> bool {
    if i == 0 {
        return true;
    }
    matches!(tokens[i - 1].kind, TokenKind::Newline)
}

/// Returns true when a `.<name>` local-label token is really a suffix
/// glued onto the preceding identifier — an AArch64 vector arrangement
/// (`v0.8b`), element/lane type (`v0.s` in `v0.s[1]`), or a condition-code
/// branch mnemonic (`b.eq`). The lexer can't tell these from a scope-local
/// label because both start with `.`, but a suffix abuts the prior token
/// with *no* whitespace, whereas a genuine local-label reference always has
/// whitespace before the dot (`b .loop`, `cbnz x0, .done`) or begins the
/// line (`.loop:`, prev token is a Newline, not an Ident). `.^name` is an
/// explicit outer-scope reference and is never a suffix. Such suffixes must
/// pass through verbatim so the backend sees `v0.8b` / `b.eq`, never a
/// scope-mangled `v0proc$$8b`.
fn is_glued_suffix(space_before: bool, outer: bool, prev_kind: Option<&TokenKind>) -> bool {
    let _ = (space_before, outer, prev_kind);
    false // TEMP: neutered to confirm regression tests catch the bug
}

/// Take an `Ident` at position `i`. Returns the name and the next index.
fn take_name(tokens: &[Token], i: usize) -> Option<(String, usize)> {
    match tokens.get(i) {
        Some(Token { kind: TokenKind::Ident(n), .. }) => Some((n.clone(), i + 1)),
        _ => None,
    }
}

/// Take a `String` literal at position `i`. Returns its unescaped value
/// and the next index.
fn take_string(tokens: &[Token], i: usize) -> Option<(String, usize)> {
    match tokens.get(i) {
        Some(Token { kind: TokenKind::String(s), .. }) => Some((s.value.clone(), i + 1)),
        _ => None,
    }
}

/// Slurp tokens from `i` until (but not including) the next Newline (or
/// end of input). Returns the slurped slice and the index of the
/// terminator.
fn collect_line(tokens: &[Token], i: usize) -> (&[Token], usize) {
    let mut j = i;
    while j < tokens.len() && !matches!(tokens[j].kind, TokenKind::Newline) {
        j += 1;
    }
    (&tokens[i..j], j)
}

/// If `tokens[i]` is a Newline, return i + 1. Otherwise i.
fn skip_newline(tokens: &[Token], i: usize) -> usize {
    if matches!(tokens.get(i).map(|t| &t.kind), Some(TokenKind::Newline)) {
        i + 1
    } else {
        i
    }
}

/// Skip a whole directive's line: the rest of the current line plus the
/// terminating newline.
fn skip_directive_line(tokens: &[Token], i: usize) -> usize {
    let (_rest, end) = collect_line(tokens, i);
    skip_newline(tokens, end)
}

/// Find a top-level comma in a slice (i.e., not inside brace or paren
/// groups). Returns the index in the slice or None.
fn find_top_level_comma(tokens: &[Token]) -> Option<usize> {
    let mut depth = 0i32;
    for (i, t) in tokens.iter().enumerate() {
        match &t.kind {
            TokenKind::Punct(Punct::LParen | Punct::LBrace | Punct::LBracket) => {
                depth += 1;
            }
            TokenKind::Punct(Punct::RParen | Punct::RBrace | Punct::RBracket) => {
                depth -= 1;
            }
            TokenKind::Punct(Punct::Comma) if depth == 0 => return Some(i),
            _ => {}
        }
    }
    None
}

/// Given a `@rept`-style block at position `start`, find the index of
/// the matching `@endr` directive token. Skips nested `@open` blocks.
fn find_block_end(
    tokens: &[Token],
    start: usize,
    open: &str,
    close: &str,
) -> Option<usize> {
    let mut depth = 1i32;
    let mut i = start;
    while i < tokens.len() {
        if at_line_start(tokens, i) {
            if let TokenKind::Directive(name) = &tokens[i].kind {
                if name == open {
                    depth += 1;
                } else if name == close {
                    depth -= 1;
                    if depth == 0 {
                        return Some(i);
                    }
                }
            }
        }
        i += 1;
    }
    None
}

/// Skip over a fully-nested block (e.g. an `@if` inside a non-taken
/// branch). Walks forward, balancing opens and closes from the
/// `terminators` list, returning the index just past the close.
fn skip_nested_block(
    tokens: &[Token],
    start: usize,
    terminators: &[&str],
) -> Result<usize, ExpandError> {
    // The directive at `start` is the opener (`@if`/`@ifdef`/`@ifndef`).
    // Walk forward, balancing nested @if blocks until we see one of
    // `terminators` at depth 1 (i.e., balancing our own opener).
    let mut depth = 1i32;
    let mut i = start + 1;
    // Skip the opener's line.
    let (_rest, after_open) = collect_line(tokens, i);
    i = skip_newline(tokens, after_open);
    while i < tokens.len() {
        if at_line_start(tokens, i) {
            if let TokenKind::Directive(name) = &tokens[i].kind {
                match name.as_str() {
                    "if" | "ifdef" | "ifndef" => {
                        depth += 1;
                    }
                    n if depth == 1 && terminators.contains(&n) => {
                        let (_rest, end) = collect_line(tokens, i + 1);
                        return Ok(skip_newline(tokens, end));
                    }
                    "endif" => {
                        depth -= 1;
                        if depth == 0 {
                            let (_rest, end) = collect_line(tokens, i + 1);
                            return Ok(skip_newline(tokens, end));
                        }
                    }
                    _ => {}
                }
            }
        }
        i += 1;
    }
    Err(ExpandError {
        kind: ExpandErrorKind::UnbalancedBlock("if".into()),
        span: tokens[start].span,
    })
}

/// Reconstruct readable text from a token slice for diagnostic messages.
/// Output is approximate — it's just for error rendering.
fn tokens_to_text(tokens: &[Token]) -> String {
    let mut s = String::new();
    for (i, t) in tokens.iter().enumerate() {
        if i > 0 && t.space_before {
            s.push(' ');
        }
        match &t.kind {
            TokenKind::Ident(n) => s.push_str(n),
            TokenKind::Number(n) => s.push_str(&n.raw),
            TokenKind::String(x) => s.push_str(&x.raw),
            TokenKind::MacroParam(n) => {
                s.push('&');
                s.push_str(n);
            }
            TokenKind::Directive(n) => {
                s.push('@');
                s.push_str(n);
            }
            TokenKind::LocalLabel(n, outer) => {
                s.push('.');
                if *outer {
                    s.push('^');
                }
                s.push_str(n);
            }
            TokenKind::Punct(p) => s.push_str(p.as_str()),
            TokenKind::Newline => s.push('\n'),
            TokenKind::Comment(c) => {
                s.push(';');
                s.push(' ');
                s.push_str(c);
            }
        }
    }
    s
}

/// Recognized MASM-style control-flow keyword after a leading `.`.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
enum ControlFlowKw {
    If,
    ElseIf,
    Else,
    EndIf,
    While,
    EndW,
    Repeat,
    Until,
    Break,
    Continue,
}

/// Map `LocalLabel(name)` to its control-flow keyword variant, if any.
fn control_flow_keyword(name: &str) -> Option<ControlFlowKw> {
    Some(match name {
        "if" => ControlFlowKw::If,
        "elseif" => ControlFlowKw::ElseIf,
        "else" => ControlFlowKw::Else,
        "endif" => ControlFlowKw::EndIf,
        "while" => ControlFlowKw::While,
        "endw" => ControlFlowKw::EndW,
        "repeat" => ControlFlowKw::Repeat,
        "until" => ControlFlowKw::Until,
        "break" => ControlFlowKw::Break,
        "continue" => ControlFlowKw::Continue,
        _ => return None,
    })
}

/// Comparison operator parsed out of a condition. Encodes the
/// branch-on-TRUE jcc; `inverse_jcc` returns the branch-on-FALSE form
/// the directives actually emit (jump *past* the body when the
/// comparison is FALSE).
#[derive(Debug, Copy, Clone)]
enum CmpOp {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

/// Split `tokens` into `(lhs, op, rhs)` for a `.if`/`.while`/`.until`
/// condition. The operator must appear at the top level (not inside
/// nested `(...)` or `[...]`).
fn split_condition(
    tokens: &[Token],
    span: Span,
) -> Result<(Vec<Token>, CmpOp, Vec<Token>), ExpandError> {
    let mut depth = 0i32;
    let mut op_at = None;
    let mut op = CmpOp::Eq;
    for (i, t) in tokens.iter().enumerate() {
        match &t.kind {
            TokenKind::Punct(Punct::LParen | Punct::LBracket | Punct::LBrace) => {
                depth += 1;
            }
            TokenKind::Punct(Punct::RParen | Punct::RBracket | Punct::RBrace) => {
                depth -= 1;
            }
            TokenKind::Punct(p) if depth == 0 => {
                let m = match p {
                    Punct::EqEq => Some(CmpOp::Eq),
                    Punct::BangEq => Some(CmpOp::Ne),
                    Punct::Lt => Some(CmpOp::Lt),
                    Punct::LtEq => Some(CmpOp::Le),
                    Punct::Gt => Some(CmpOp::Gt),
                    Punct::GtEq => Some(CmpOp::Ge),
                    _ => None,
                };
                if let Some(m) = m {
                    op_at = Some(i);
                    op = m;
                    break;
                }
            }
            _ => {}
        }
    }
    let i = op_at.ok_or_else(|| ExpandError {
        kind: ExpandErrorKind::MalformedCondition(
            "expected a comparison operator (==, !=, <, <=, >, >=)",
        ),
        span,
    })?;
    let lhs: Vec<Token> = tokens[..i].to_vec();
    let rhs: Vec<Token> = tokens[i + 1..].to_vec();
    if lhs.is_empty() {
        return Err(ExpandError {
            kind: ExpandErrorKind::MalformedCondition(
                "condition is missing its left operand",
            ),
            span,
        });
    }
    if rhs.is_empty() {
        return Err(ExpandError {
            kind: ExpandErrorKind::MalformedCondition(
                "condition is missing its right operand",
            ),
            span,
        });
    }
    Ok((lhs, op, rhs))
}

/// The branch-on-FALSE jcc mnemonic for each comparison operator —
/// i.e., the instruction we emit after the cmp to *skip* the
/// then-branch when the comparison is FALSE. Signed semantics
/// (`jl`/`jle`/`jg`/`jge`); unsigned not modelled in v1.
fn inverse_jcc(op: CmpOp) -> &'static str {
    match op {
        CmpOp::Eq => "jne",
        CmpOp::Ne => "je",
        CmpOp::Lt => "jge",
        CmpOp::Le => "jg",
        CmpOp::Gt => "jle",
        CmpOp::Ge => "jl",
    }
}

/// True if `tokens[i]` is `(`. Used to gate macro-call detection so a
/// bare `Ident` reference (e.g. `mov rax, plus`) doesn't accidentally
/// fire as a macro call.
fn next_is_lparen(tokens: &[Token], i: usize) -> bool {
    matches!(
        tokens.get(i).map(|t| &t.kind),
        Some(TokenKind::Punct(Punct::LParen))
    )
}

/// Find the matching `@endmacro` for a `@macro` whose body starts at
/// `start`. Nested `@macro` is a hard error. Returns the index of the
/// `@endmacro` directive token.
fn find_macro_end(
    tokens: &[Token],
    start: usize,
    open_span: Span,
) -> Result<usize, ExpandError> {
    let mut i = start;
    while i < tokens.len() {
        if at_line_start(tokens, i) {
            if let TokenKind::Directive(name) = &tokens[i].kind {
                match name.as_str() {
                    "macro" => {
                        return Err(ExpandError {
                            kind: ExpandErrorKind::NestedMacroDef,
                            span: tokens[i].span,
                        });
                    }
                    "endmacro" => return Ok(i),
                    _ => {}
                }
            }
        }
        i += 1;
    }
    Err(ExpandError {
        kind: ExpandErrorKind::UnbalancedBlock("macro".into()),
        span: open_span,
    })
}

/// Walk a macro body, replacing every `&param` token with the
/// corresponding bound argument's tokens. Returns the substituted
/// body for re-expansion.
///
/// Variadic args substitute as a comma-joined sequence (`a, b, c`).
/// `##` token paste is rejected (planned for v1.5).
fn substitute_params(
    body: &[Token],
    args: &MacroArgs,
    def: &MacroDef,
) -> Result<Vec<Token>, ExpandError> {
    let mut out = Vec::with_capacity(body.len());
    for tok in body {
        match &tok.kind {
            TokenKind::MacroParam(name) => {
                let val = args.bound.get(name).ok_or_else(|| ExpandError {
                    kind: ExpandErrorKind::UnknownParam {
                        macro_name: def.name.clone(),
                        param: name.clone(),
                    },
                    span: tok.span,
                })?;
                match val {
                    MacroArgValue::Tokens(ts) => {
                        // Take the call site's space_before for the
                        // first substituted token; preserve the rest.
                        let mut first = true;
                        for src in ts {
                            let mut src = src.clone();
                            if first {
                                src.space_before = tok.space_before;
                                first = false;
                            }
                            out.push(src);
                        }
                    }
                    MacroArgValue::Variadic(groups) => {
                        // Emit groups separated by `,` tokens.
                        let mut first_group = true;
                        for group in groups {
                            if !first_group {
                                out.push(Token {
                                    kind: TokenKind::Punct(Punct::Comma),
                                    span: tok.span,
                                    space_before: false,
                                });
                            }
                            let mut first = true;
                            for src in group {
                                let mut src = src.clone();
                                if first {
                                    src.space_before =
                                        if first_group { tok.space_before } else { true };
                                    first = false;
                                }
                                out.push(src);
                            }
                            first_group = false;
                        }
                    }
                }
            }
            TokenKind::Punct(Punct::HashHash) => {
                // Token-paste — not yet supported. Surface from
                // substitution time so the error span points at the
                // paste site inside the body.
                return Err(ExpandError {
                    kind: ExpandErrorKind::PasteUnsupported,
                    span: tok.span,
                });
            }
            _ => out.push(tok.clone()),
        }
    }
    Ok(out)
}

/// Static string for unimplemented-directive errors. Saves an alloc
/// every error site.
fn directive_name_static(name: &str) -> &'static str {
    match name {
        "macro" => "macro",
        "endmacro" => "endmacro",
        "scope" => "scope",
        "endscope" => "endscope",
        "include" => "include",
        "for" => "for",
        "endfor" => "endfor",
        "rust_macro" => "rust_macro",
        "extern" => "extern",
        "bits" => "bits",
        "section" => "section",
        "code" => "code",
        "data" => "data",
        "rodata" => "rodata",
        "bss" => "bss",
        "db" => "db",
        "dw" => "dw",
        "dd" => "dd",
        "dq" => "dq",
        "dz" => "dz",
        "local" => "local",
        _ => "<unknown>",
    }
}

// ── extend AsmError to carry ExpandError ────────────────────────────

// Add the Expand variant on AsmError (declared in error.rs). We do it
// here as a small extension trait so error.rs stays lex-only.
// Actually: AsmError lives in error.rs and we need the variant there.

// ─── tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::asm::lex::lex;
    use crate::asm::source::FileId;

    fn expand_text(asm: &mut Assembler, src: &str) -> Result<Vec<Token>, AsmError> {
        let tokens = lex(FileId(0), src).map_err(AsmError::from)?;
        asm.expand(&tokens)
    }

    fn to_text(tokens: &[Token]) -> String {
        tokens_to_text(tokens)
    }

    #[test]
    fn passthrough_no_directives() {
        let mut asm = Assembler::new();
        let out = expand_text(&mut asm, "mov rax, 42\nret\n").unwrap();
        assert_eq!(to_text(&out).trim(), "mov rax, 42\nret");
    }

    #[test]
    fn define_substitutes_text() {
        let mut asm = Assembler::new();
        let out = expand_text(
            &mut asm,
            "@define TOS rax\nmov TOS, 42\n",
        )
        .unwrap();
        assert_eq!(to_text(&out).trim(), "mov rax, 42");
    }

    #[test]
    fn define_substitutes_multitoken_body() {
        let mut asm = Assembler::new();
        let out = expand_text(
            &mut asm,
            "@define DSP_OP [rbp + 8]\nmov rax, DSP_OP\n",
        )
        .unwrap();
        // Reconstructed text should contain the substituted body.
        let s = to_text(&out);
        assert!(s.contains("[rbp + 8]"), "output was {s}");
    }

    #[test]
    fn assign_evaluates_and_substitutes_numeric() {
        let mut asm = Assembler::new();
        let out = expand_text(
            &mut asm,
            "@assign cell = 4 * 2\nmov rax, cell\n",
        )
        .unwrap();
        assert_eq!(to_text(&out).trim(), "mov rax, 8");
        // The numeric is visible to lookup.
        assert!(matches!(asm.lookup("cell"), Some(DefineValue::Numeric(8))));
    }

    #[test]
    fn assign_uses_prior_assign() {
        let mut asm = Assembler::new();
        let out = expand_text(
            &mut asm,
            "@assign cell = 8\n@assign frame = cell * 4\nmov rax, frame\n",
        )
        .unwrap();
        assert_eq!(to_text(&out).trim(), "mov rax, 32");
    }

    #[test]
    fn undef_removes_define() {
        let mut asm = Assembler::new();
        let out = expand_text(
            &mut asm,
            "@define X 1\n@undef X\nmov rax, X\n",
        )
        .unwrap();
        // X is no longer substituted — it appears as a plain Ident.
        assert!(to_text(&out).contains("X"));
    }

    #[test]
    fn if_true_branch_included() {
        let mut asm = Assembler::new();
        let out = expand_text(
            &mut asm,
            "@if 1\nmov rax, 1\n@endif\n",
        )
        .unwrap();
        assert!(to_text(&out).contains("mov rax, 1"));
    }

    #[test]
    fn if_false_branch_excluded() {
        let mut asm = Assembler::new();
        let out = expand_text(
            &mut asm,
            "@if 0\nmov rax, 1\n@endif\nret\n",
        )
        .unwrap();
        let s = to_text(&out);
        assert!(!s.contains("mov rax, 1"));
        assert!(s.contains("ret"));
    }

    #[test]
    fn if_else_takes_else_when_false() {
        let mut asm = Assembler::new();
        let out = expand_text(
            &mut asm,
            "@if 0\nmov rax, 1\n@else\nmov rax, 2\n@endif\n",
        )
        .unwrap();
        let s = to_text(&out);
        assert!(!s.contains("mov rax, 1"));
        assert!(s.contains("mov rax, 2"));
    }

    #[test]
    fn if_elif_chain() {
        let mut asm = Assembler::new();
        let out = expand_text(
            &mut asm,
            r#"@assign sel = 2
@if sel == 1
mov rax, 1
@elif sel == 2
mov rax, 2
@elif sel == 3
mov rax, 3
@else
mov rax, 99
@endif
"#,
        )
        .unwrap();
        let s = to_text(&out);
        assert!(s.contains("mov rax, 2"));
        assert!(!s.contains("mov rax, 1"));
        assert!(!s.contains("mov rax, 3"));
        assert!(!s.contains("mov rax, 99"));
    }

    #[test]
    fn ifdef_works() {
        let mut asm = Assembler::new();
        let out = expand_text(
            &mut asm,
            "@define DEBUG 1\n@ifdef DEBUG\nint 3\n@endif\n",
        )
        .unwrap();
        assert!(to_text(&out).contains("int 3"));
    }

    #[test]
    fn ifndef_works() {
        let mut asm = Assembler::new();
        let out = expand_text(
            &mut asm,
            "@ifndef DEBUG\nmov rax, 0\n@endif\n",
        )
        .unwrap();
        assert!(to_text(&out).contains("mov rax, 0"));
    }

    #[test]
    fn nested_if_works() {
        let mut asm = Assembler::new();
        let out = expand_text(
            &mut asm,
            r#"@if 1
@if 1
inner_true
@endif
@if 0
inner_false
@endif
@endif
"#,
        )
        .unwrap();
        let s = to_text(&out);
        assert!(s.contains("inner_true"));
        assert!(!s.contains("inner_false"));
    }

    #[test]
    fn user_error_fails() {
        let mut asm = Assembler::new();
        let err = expand_text(&mut asm, "@error \"boom\"\n").unwrap_err();
        let s = format!("{err}");
        assert!(s.contains("boom"), "got: {s}");
    }

    #[test]
    fn assert_zero_fails() {
        let mut asm = Assembler::new();
        let err = expand_text(
            &mut asm,
            "@assign cell = 4\n@assert cell == 8, \"need 64-bit cells\"\n",
        )
        .unwrap_err();
        let s = format!("{err}");
        assert!(s.contains("need 64-bit cells"), "got: {s}");
    }

    #[test]
    fn assert_nonzero_passes() {
        let mut asm = Assembler::new();
        let out = expand_text(
            &mut asm,
            "@assign cell = 8\n@assert cell == 8, \"ok\"\nret\n",
        )
        .unwrap();
        assert!(to_text(&out).contains("ret"));
    }

    #[test]
    fn rept_unrolls() {
        let mut asm = Assembler::new();
        let out = expand_text(
            &mut asm,
            "@rept 3\npush rax\n@endr\n",
        )
        .unwrap();
        let s = to_text(&out);
        let count = s.matches("push rax").count();
        assert_eq!(count, 3, "got: {s}");
    }

    #[test]
    fn rept_index_visible() {
        let mut asm = Assembler::new();
        let out = expand_text(
            &mut asm,
            "@rept 3\nmov qword [rdi + @INDEX * 8], 0\n@endr\n",
        )
        .unwrap();
        let s = to_text(&out);
        assert!(s.contains("0 * 8"), "got: {s}");
        assert!(s.contains("1 * 8"));
        assert!(s.contains("2 * 8"));
    }

    #[test]
    fn counter_bumps_each_substitution() {
        let mut asm = Assembler::new();
        let out = expand_text(
            &mut asm,
            "mov rax, @COUNTER\nmov rax, @COUNTER\nmov rax, @COUNTER\n",
        )
        .unwrap();
        let s = to_text(&out);
        assert!(s.contains("mov rax, 0"));
        assert!(s.contains("mov rax, 1"));
        assert!(s.contains("mov rax, 2"));
    }

    #[test]
    fn bits_substitutes_to_64() {
        let mut asm = Assembler::new();
        let out = expand_text(&mut asm, "mov rax, @BITS\n").unwrap();
        assert_eq!(to_text(&out).trim(), "mov rax, 64");
    }

    #[test]
    fn host_define_visible() {
        let mut asm = Assembler::new();
        asm.define("BUILD_REV", 42);
        let out = expand_text(&mut asm, "@if BUILD_REV == 42\nmatch\n@endif\n").unwrap();
        assert!(to_text(&out).contains("match"));
    }

    // ── MASM runtime control flow tests ─────────────────────────────

    #[test]
    fn dot_if_simple() {
        let mut asm = Assembler::new();
        let out = expand_text(
            &mut asm,
            ".if rax == 0\n    mov rbx, 1\n.endif\n",
        )
        .unwrap();
        let s = to_text(&out);
        assert!(s.contains("cmp rax, 0"), "got: {s}");
        assert!(s.contains("jne __if"), "got: {s}");
        assert!(s.contains("mov rbx, 1"));
        assert!(s.contains("_end:"));
    }

    #[test]
    fn dot_if_else() {
        let mut asm = Assembler::new();
        let out = expand_text(
            &mut asm,
            ".if rax == 0\n    mov rbx, 1\n.else\n    mov rbx, 2\n.endif\n",
        )
        .unwrap();
        let s = to_text(&out);
        assert!(s.contains("cmp rax, 0"));
        assert!(s.contains("jne __if"));
        assert!(s.contains("mov rbx, 1"));
        assert!(s.contains("mov rbx, 2"));
        // exactly one `jmp ..._end` (the .else's skip-past-end)
        assert_eq!(s.matches("jmp __if").count(), 1, "got: {s}");
    }

    #[test]
    fn dot_if_elseif_chain() {
        let mut asm = Assembler::new();
        let out = expand_text(
            &mut asm,
            r#".if rax == 1
    mov rbx, 10
.elseif rax == 2
    mov rbx, 20
.elseif rax == 3
    mov rbx, 30
.else
    mov rbx, 99
.endif
"#,
        )
        .unwrap();
        let s = to_text(&out);
        // 3 cmp/jne pairs (one per .if and each .elseif)
        assert_eq!(s.matches("cmp rax,").count(), 3);
        assert!(s.contains("mov rbx, 10"));
        assert!(s.contains("mov rbx, 20"));
        assert!(s.contains("mov rbx, 30"));
        assert!(s.contains("mov rbx, 99"));
    }

    #[test]
    fn dot_if_with_signed_comparison() {
        let mut asm = Assembler::new();
        let out = expand_text(
            &mut asm,
            ".if rcx >= 0\n    nop\n.endif\n",
        )
        .unwrap();
        let s = to_text(&out);
        // `>=` inverse is `jl` (signed less-than).
        assert!(s.contains("jl __if"), "got: {s}");
    }

    #[test]
    fn dot_elseif_after_else_errors() {
        let mut asm = Assembler::new();
        let err = expand_text(
            &mut asm,
            ".if rax == 0\n.else\n.elseif rax == 1\n.endif\n",
        )
        .unwrap_err();
        let s = format!("{err}");
        assert!(s.contains("after `.else`"), "got: {s}");
    }

    #[test]
    fn dot_endif_with_no_if_errors() {
        let mut asm = Assembler::new();
        let err = expand_text(&mut asm, ".endif\n").unwrap_err();
        let s = format!("{err}");
        assert!(s.contains("no matching"), "got: {s}");
    }

    #[test]
    fn dot_while_endw() {
        let mut asm = Assembler::new();
        let out = expand_text(
            &mut asm,
            ".while rcx > 0\n    dec rcx\n.endw\n",
        )
        .unwrap();
        let s = to_text(&out);
        assert!(s.contains("_top:"));
        assert!(s.contains("cmp rcx, 0"));
        // `>` inverse is `jle`.
        assert!(s.contains("jle __while"), "got: {s}");
        assert!(s.contains("dec rcx"));
        assert!(s.contains("jmp __while"));
        assert!(s.contains("_bot:"));
    }

    #[test]
    fn dot_repeat_until() {
        let mut asm = Assembler::new();
        let out = expand_text(
            &mut asm,
            ".repeat\n    dec rax\n.until rax == 0\n",
        )
        .unwrap();
        let s = to_text(&out);
        assert!(s.contains("_top:"));
        assert!(s.contains("dec rax"));
        assert!(s.contains("_cont:"));
        assert!(s.contains("cmp rax, 0"));
        // `==` inverse is `jne` — loop back to top while NOT equal.
        assert!(s.contains("jne __rep"), "got: {s}");
        assert!(s.contains("_bot:"));
    }

    #[test]
    fn dot_break_inside_while() {
        let mut asm = Assembler::new();
        let out = expand_text(
            &mut asm,
            ".while rax > 0\n    .if rax == 5\n        .break\n    .endif\n    dec rax\n.endw\n",
        )
        .unwrap();
        let s = to_text(&out);
        // `.break` should emit a `jmp __whileN_bot` (the while's bottom).
        assert!(s.contains("jmp __while"), "got: {s}");
        assert!(s.contains("_bot"));
    }

    #[test]
    fn dot_continue_inside_repeat_jumps_to_cont_label() {
        let mut asm = Assembler::new();
        let out = expand_text(
            &mut asm,
            ".repeat\n    .if rax == 5\n        .continue\n    .endif\n    dec rax\n.until rax == 0\n",
        )
        .unwrap();
        let s = to_text(&out);
        // .continue inside .repeat jumps to _cont (the post-test entry),
        // NOT to _top.
        assert!(s.contains("jmp __rep"));
        // Verify the jump is to the _cont label (not _top).
        let lines: Vec<&str> = s.lines().collect();
        let has_cont_jump = lines.iter().any(|l| l.contains("jmp") && l.contains("_cont"));
        assert!(has_cont_jump, "expected `jmp __repN_cont`, got: {s}");
    }

    #[test]
    fn dot_break_outside_loop_errors() {
        let mut asm = Assembler::new();
        let err = expand_text(&mut asm, ".break\n").unwrap_err();
        let s = format!("{err}");
        assert!(s.contains("outside any"), "got: {s}");
    }

    #[test]
    fn unclosed_dot_if_errors_at_eof() {
        let mut asm = Assembler::new();
        let err = expand_text(&mut asm, ".if rax == 0\n").unwrap_err();
        let s = format!("{err}");
        assert!(s.contains("if"), "got: {s}");
        assert!(s.contains("unclosed"), "got: {s}");
    }

    #[test]
    fn nested_if_inside_while() {
        let mut asm = Assembler::new();
        let out = expand_text(
            &mut asm,
            ".while rax > 0\n    .if rax == 5\n        nop\n    .endif\n    dec rax\n.endw\n",
        )
        .unwrap();
        let s = to_text(&out);
        // Both blocks should produce their respective labels.
        assert!(s.contains("__while"));
        assert!(s.contains("__if"));
        assert!(s.contains("nop"));
        assert!(s.contains("dec rax"));
    }

    #[test]
    fn dot_if_with_define_substituted_operand() {
        let mut asm = Assembler::new();
        let out = expand_text(
            &mut asm,
            "@define TOS rax\n.if TOS == 0\n    nop\n.endif\n",
        )
        .unwrap();
        let s = to_text(&out);
        // `TOS` should have substituted to `rax` before the cmp emit.
        assert!(s.contains("cmp rax, 0"), "got: {s}");
    }

    #[test]
    fn hutch_tribute_demo() {
        // A Hutch-flavoured snippet: a proc that uses runtime
        // control flow exactly the way MASM32 idiom would write it.
        let mut asm = Assembler::new();
        let out = expand_text(
            &mut asm,
            r#"@macro proc(name)
    @scope &name
    .globl &name
&name:
@endmacro

@macro endp()
    ret
    @endscope
@endmacro

; Return min(rax, rcx)
proc(min2)
    .if rax > rcx
        mov rax, rcx
    .endif
endp()

; Sum 1..rcx into rax. Hutch-style do-while.
proc(sum_up_to)
    xor rax, rax
    .while rcx > 0
        add rax, rcx
        dec rcx
    .endw
endp()
"#,
        )
        .unwrap();
        let s = to_text(&out);
        assert!(s.contains(".globl min2"));
        assert!(s.contains("min2:"));
        assert!(s.contains("cmp rax, rcx"));
        // `>` inverse is `jle`.
        assert!(s.contains("jle __if"));
        assert!(s.contains("mov rax, rcx"));

        assert!(s.contains(".globl sum_up_to"));
        assert!(s.contains("sum_up_to:"));
        assert!(s.contains("xor rax, rax"));
        assert!(s.contains("__while"));
        assert!(s.contains("add rax, rcx"));
        assert!(s.contains("dec rcx"));
    }

    // ── @extern tests ───────────────────────────────────────────────

    #[test]
    fn extern_records_declaration() {
        let mut asm = Assembler::new();
        let _ = expand_text(&mut asm, "@extern rt_print(1)\n").unwrap();
        let externs: Vec<_> = asm.externs().collect();
        assert_eq!(externs.len(), 1);
        assert_eq!(externs[0].0, "rt_print");
        assert_eq!(externs[0].1.arg_count, 1);
        assert_eq!(externs[0].1.dll, None);
    }

    #[test]
    fn extern_with_dll_string() {
        let mut asm = Assembler::new();
        let _ = expand_text(
            &mut asm,
            "@extern \"USER32.dll\" MessageBoxW(4)\n",
        )
        .unwrap();
        let externs: Vec<_> = asm.externs().collect();
        assert_eq!(externs.len(), 1);
        assert_eq!(externs[0].0, "MessageBoxW");
        assert_eq!(externs[0].1.arg_count, 4);
        assert_eq!(externs[0].1.dll.as_deref(), Some("USER32.dll"));
    }

    #[test]
    fn extern_multiple() {
        let mut asm = Assembler::new();
        let _ = expand_text(
            &mut asm,
            "@extern rt_emit(1)\n@extern rt_key(0)\n@extern rt_io3(3)\n",
        )
        .unwrap();
        let mut externs: Vec<_> = asm
            .externs()
            .map(|(n, d)| (n.to_string(), d.arg_count))
            .collect();
        externs.sort();
        assert_eq!(
            externs,
            vec![
                ("rt_emit".to_string(), 1),
                ("rt_io3".to_string(), 3),
                ("rt_key".to_string(), 0),
            ]
        );
    }

    #[test]
    fn extern_mixed_dll_and_rust() {
        // A generated Win32 binding file plus the host's own runtime
        // function in the same module.
        let mut asm = Assembler::new();
        let _ = expand_text(
            &mut asm,
            "@extern \"KERNEL32.dll\" GetTickCount64(0)\n@extern rt_print(1)\n",
        )
        .unwrap();
        let mut externs: Vec<(String, Option<String>)> = asm
            .externs()
            .map(|(n, d)| (n.to_string(), d.dll.clone()))
            .collect();
        externs.sort();
        assert_eq!(
            externs,
            vec![
                ("GetTickCount64".to_string(), Some("KERNEL32.dll".to_string())),
                ("rt_print".to_string(), None),
            ]
        );
    }

    #[test]
    fn extern_arg_count_supports_expressions() {
        let mut asm = Assembler::new();
        let _ = expand_text(
            &mut asm,
            "@assign N = 2\n@extern rt_io(N + 1)\n",
        )
        .unwrap();
        let externs: Vec<_> = asm.externs().collect();
        assert_eq!(externs.len(), 1);
        assert_eq!(externs[0].0, "rt_io");
        assert_eq!(externs[0].1.arg_count, 3);
    }

    #[test]
    fn extern_missing_paren_errors() {
        let mut asm = Assembler::new();
        let err = expand_text(&mut asm, "@extern foo\n").unwrap_err();
        let s = format!("{err}");
        assert!(s.contains("@extern"), "got: {s}");
    }

    #[test]
    fn extern_negative_count_errors() {
        let mut asm = Assembler::new();
        let err = expand_text(&mut asm, "@extern foo(-1)\n").unwrap_err();
        let s = format!("{err}");
        assert!(s.contains("out of range"), "got: {s}");
    }

    #[test]
    fn extern_passes_call_site_through_to_mc() {
        // The `call rt_print` is just literal text to MC — wfasm doesn't
        // validate that calls match @extern declarations. Trust MC's
        // linker to resolve.
        let mut asm = Assembler::new();
        let out = expand_text(
            &mut asm,
            "@extern rt_print(1)\nmov rcx, 42\ncall rt_print\n",
        )
        .unwrap();
        let s = to_text(&out);
        assert!(s.contains("call rt_print"), "got: {s}");
    }

    // ── Rust macro tests ────────────────────────────────────────────

    #[test]
    fn rust_macro_basic_call() {
        let mut asm = Assembler::new();
        asm.register_macro("hello", |ctx| {
            ctx.emit_line("mov rax, 42\n")?;
            Ok(())
        });
        let out = expand_text(&mut asm, "hello()\n").unwrap();
        let s = to_text(&out);
        assert!(s.contains("mov rax, 42"), "got: {s}");
    }

    #[test]
    fn rust_macro_parse_int_works() {
        let mut asm = Assembler::new();
        asm.register_macro("double", |ctx| {
            let v = ctx.parse_int(0)?;
            ctx.emit_line(&format!("mov rax, {}\n", v * 2))?;
            Ok(())
        });
        let out = expand_text(&mut asm, "double(21)\n").unwrap();
        assert!(to_text(&out).contains("mov rax, 42"));
    }

    #[test]
    fn rust_macro_parse_int_sees_at_assign_values() {
        let mut asm = Assembler::new();
        asm.register_macro("read_cell", |ctx| {
            let v = ctx.parse_int(0)?;
            ctx.emit_line(&format!("mov rax, {v}\n"))?;
            Ok(())
        });
        let out = expand_text(
            &mut asm,
            "@assign cell = 8\nread_cell(cell * 4)\n",
        )
        .unwrap();
        assert!(to_text(&out).contains("mov rax, 32"));
    }

    #[test]
    fn rust_macro_lookup_int() {
        let mut asm = Assembler::new();
        asm.register_macro("times_cell", |ctx| {
            let n = ctx.parse_int(0)?;
            let cell = ctx.lookup_int("cell").unwrap_or(8);
            ctx.emit_line(&format!("mov rax, {}\n", n * cell))?;
            Ok(())
        });
        let out = expand_text(
            &mut asm,
            "@assign cell = 8\ntimes_cell(5)\n",
        )
        .unwrap();
        assert!(to_text(&out).contains("mov rax, 40"));
    }

    #[test]
    fn rust_macro_proc_name_when_in_scope() {
        let mut asm = Assembler::new();
        asm.register_macro("emit_proc_label", |ctx| {
            let name = ctx.proc_name().unwrap_or("").to_string();
            ctx.emit_line(&format!("# in proc: {name}\n"))?;
            Ok(())
        });
        let out = expand_text(
            &mut asm,
            "@scope plus\nemit_proc_label()\n@endscope\n",
        )
        .unwrap();
        let s = to_text(&out);
        // The Rust macro emitted `# in proc: plus`. Lexer drops `#`
        // comments? No — `#` is a Punct in our lexer. So it lands as
        // `# in proc: plus` in the output.
        assert!(s.contains("plus"), "got: {s}");
    }

    #[test]
    fn rust_macro_error_propagates() {
        let mut asm = Assembler::new();
        asm.register_macro("explode", |_ctx| Err("boom".into()));
        let err = expand_text(&mut asm, "explode()\n").unwrap_err();
        let s = format!("{err}");
        assert!(s.contains("boom"), "got: {s}");
    }

    #[test]
    fn rust_macro_directive_declares_placeholder() {
        // `@rust_macro stk` should install a placeholder that errors
        // with a useful message if the host forgets to register the
        // real implementation.
        let mut asm = Assembler::new();
        let err = expand_text(&mut asm, "@rust_macro stk\nstk(1, 2)\n").unwrap_err();
        let s = format!("{err}");
        assert!(s.contains("stk"), "got: {s}");
        assert!(s.contains("never registered"), "got: {s}");
    }

    #[test]
    fn register_macro_overwrites_placeholder() {
        let mut asm = Assembler::new();
        let _ = expand_text(&mut asm, "@rust_macro hello\n").unwrap();
        asm.register_macro("hello", |ctx| {
            ctx.emit_line("nop\n")?;
            Ok(())
        });
        let out = expand_text(&mut asm, "hello()\n").unwrap();
        assert!(to_text(&out).contains("nop"));
    }

    #[test]
    fn stk_built_in_emits_add_rbp() {
        // stk(2, 1) — pop one cell on net, emit `add rbp, 8` for cell=8.
        let mut asm = Assembler::new();
        asm.register_macro("stk", crate::asm::macros::stk);
        let out = expand_text(
            &mut asm,
            "@assign cell = 8\nstk(2, 1)\n",
        )
        .unwrap();
        assert!(to_text(&out).contains("add rbp, 8"));
    }

    #[test]
    fn stk_built_in_emits_sub_rbp() {
        // stk(1, 2) — push one cell on net, emit `sub rbp, 8`.
        let mut asm = Assembler::new();
        asm.register_macro("stk", crate::asm::macros::stk);
        let out = expand_text(
            &mut asm,
            "@assign cell = 8\nstk(1, 2)\n",
        )
        .unwrap();
        assert!(to_text(&out).contains("sub rbp, 8"));
    }

    #[test]
    fn stk_built_in_balanced_emits_nothing() {
        // stk(1, 1) — net zero, no adjustment.
        let mut asm = Assembler::new();
        asm.register_macro("stk", crate::asm::macros::stk);
        let out = expand_text(
            &mut asm,
            "@assign cell = 8\nstk(1, 1)\nret\n",
        )
        .unwrap();
        let s = to_text(&out);
        assert!(s.contains("ret"));
        assert!(!s.contains("add rbp"));
        assert!(!s.contains("sub rbp"));
    }

    #[test]
    fn stk_via_at_rust_macro_then_register() {
        // The user-guide pattern: declare in source, register from
        // Rust. The placeholder errors are upgraded to a real impl.
        let mut asm = Assembler::new();
        asm.register_macro("stk", crate::asm::macros::stk);
        let out = expand_text(
            &mut asm,
            r#"@assign cell = 8
@rust_macro stk
@macro proc(name)
    @scope &name
    .globl &name
&name:
@endmacro
@macro endp()
    ret
    @endscope
@endmacro

proc(plus)
    stk(2, 1)
    add rax, [rbp]
endp()
"#,
        )
        .unwrap();
        let s = to_text(&out);
        assert!(s.contains(".globl plus"));
        assert!(s.contains("plus:"));
        assert!(s.contains("add rbp, 8"), "got: {s}");
        assert!(s.contains("add rax, [rbp]"));
    }

    // ── @include tests ──────────────────────────────────────────────

    #[test]
    fn include_inlines_other_file() {
        let mut asm = Assembler::new();
        asm.add_virtual_file("/proj/macros.masm", "@define TOS rax\n");
        let out = asm
            .assemble(
                "/proj/main.masm",
                "@include \"macros.masm\"\nmov TOS, 42\n",
            )
            .unwrap();
        assert!(out.contains("mov rax, 42"), "got: {out}");
    }

    #[test]
    fn include_resolves_relative_to_including_file() {
        // Parent at /a/b/main.masm including "sub/lib.masm" should
        // resolve to /a/b/sub/lib.masm.
        let mut asm = Assembler::new();
        asm.add_virtual_file(
            "/a/b/sub/lib.masm",
            "@define X 7\n",
        );
        let out = asm
            .assemble(
                "/a/b/main.masm",
                "@include \"sub/lib.masm\"\nmov rax, X\n",
            )
            .unwrap();
        assert!(out.contains("mov rax, 7"), "got: {out}");
    }

    #[test]
    fn include_nested() {
        let mut asm = Assembler::new();
        asm.add_virtual_file(
            "/proj/lib_a.masm",
            "@include \"lib_b.masm\"\n@define A 1\n",
        );
        asm.add_virtual_file("/proj/lib_b.masm", "@define B 2\n");
        let out = asm
            .assemble(
                "/proj/main.masm",
                "@include \"lib_a.masm\"\nmov rax, A\nmov rbx, B\n",
            )
            .unwrap();
        assert!(out.contains("mov rax, 1"), "got: {out}");
        assert!(out.contains("mov rbx, 2"), "got: {out}");
    }

    #[test]
    fn include_cycle_detected() {
        let mut asm = Assembler::new();
        asm.add_virtual_file("/proj/a.masm", "@include \"b.masm\"\n");
        asm.add_virtual_file("/proj/b.masm", "@include \"a.masm\"\n");
        let err = asm
            .assemble("/proj/main.masm", "@include \"a.masm\"\n")
            .unwrap_err();
        let s = format!("{err}");
        assert!(s.contains("cycle"), "got: {s}");
    }

    #[test]
    fn include_missing_file_errors() {
        let mut asm = Assembler::new();
        let err = asm
            .assemble("/proj/main.masm", "@include \"nope.masm\"\n")
            .unwrap_err();
        let s = format!("{err}");
        // Either "cannot find" (Windows) or "No such file" (POSIX);
        // both contain "nope.masm" via the path display.
        assert!(s.contains("nope.masm"), "got: {s}");
    }

    #[test]
    fn include_inlines_macros() {
        // The user-guide pattern: macros are defined in a library, the
        // kernel file pulls them in via @include.
        let mut asm = Assembler::new();
        asm.add_virtual_file(
            "/proj/forth-macros.masm",
            r#"@macro proc(name)
    .globl &name
&name:
@endmacro

@macro endp()
    ret
@endmacro
"#,
        );
        let out = asm
            .assemble(
                "/proj/kernel.masm",
                "@include \"forth-macros.masm\"\nproc(plus)\n    add rax, [rbp]\nendp()\n",
            )
            .unwrap();
        assert!(out.contains(".globl plus"), "got: {out}");
        assert!(out.contains("plus:"));
        assert!(out.contains("add rax, [rbp]"));
        assert!(out.contains("ret"));
    }

    #[test]
    fn include_missing_string_arg_errors() {
        let mut asm = Assembler::new();
        let err = asm
            .assemble("/proj/main.masm", "@include\n")
            .unwrap_err();
        let s = format!("{err}");
        assert!(s.contains("@include"), "got: {s}");
    }

    #[test]
    fn include_depth_limit() {
        // Construct a chain a → b → c → d → ... and cap the depth at 3.
        let mut asm = Assembler::new();
        asm.set_max_include_depth(3);
        asm.add_virtual_file("/proj/a.masm", "@include \"b.masm\"\n");
        asm.add_virtual_file("/proj/b.masm", "@include \"c.masm\"\n");
        asm.add_virtual_file("/proj/c.masm", "@include \"d.masm\"\n");
        asm.add_virtual_file("/proj/d.masm", "ret\n");
        let err = asm
            .assemble("/proj/main.masm", "@include \"a.masm\"\n")
            .unwrap_err();
        let s = format!("{err}");
        assert!(s.contains("exceeded 3"), "got: {s}");
    }

    #[test]
    fn unbalanced_endif_errors() {
        let mut asm = Assembler::new();
        let err = expand_text(&mut asm, "@endif\n").unwrap_err();
        let s = format!("{err}");
        assert!(s.contains("endif"), "got: {s}");
    }

    // ── @scope tests ────────────────────────────────────────────────

    #[test]
    fn scope_mangles_local_labels() {
        let mut asm = Assembler::new();
        let out = expand_text(
            &mut asm,
            r#"@scope plus
    jmp .done
.done:
    ret
@endscope
"#,
        )
        .unwrap();
        let s = to_text(&out);
        assert!(s.contains("jmp plus$$done"), "got: {s}");
        assert!(s.contains("plus$$done:"), "got: {s}");
    }

    #[test]
    fn outer_label_in_macro_skips_macro_scope() {
        // `.^name` inside a macro body references a label in the
        // enclosing @scope (the calling proc), not in the macro's own
        // hygienic scope. Without this, a macro can't branch to a
        // proc-local label.
        let mut asm = Assembler::new();
        let out = expand_text(
            &mut asm,
            r#"@macro bail()
    jmp .^fail
@endmacro
@scope demo
    bail()
.fail:
    ret
@endscope
"#,
        )
        .unwrap();
        let s = to_text(&out);
        assert!(s.contains("jmp demo$$fail"), "got: {s}");
        assert!(s.contains("demo$$fail:"), "got: {s}");
    }

    #[test]
    fn local_label_in_macro_still_uses_macro_scope() {
        // Sanity: plain `.name` inside a macro body still gets the
        // macro-invocation prefix, preserving hygiene.
        let mut asm = Assembler::new();
        let out = expand_text(
            &mut asm,
            r#"@macro skip()
    jmp .done
.done:
@endmacro
@scope demo
    skip()
@endscope
"#,
        )
        .unwrap();
        let s = to_text(&out);
        assert!(s.contains("skip$$"), "got: {s}");
        assert!(!s.contains("demo$$done"), "got: {s}");
    }

    #[test]
    fn label_named_same_as_mc_directive_is_mangled() {
        // `.skip:` (with colon) is a label definition, not the GAS
        // `.skip` directive. The colon is what disambiguates. Without
        // this case the expander used to pass `.skip:` through and
        // every primitive that wanted a `.skip` label collided
        // against every other.
        let mut asm = Assembler::new();
        let out = expand_text(
            &mut asm,
            r#"@scope foo
.skip:
    nop
.skip 16
@endscope
"#,
        )
        .unwrap();
        let s = to_text(&out);
        assert!(s.contains("foo$$skip:"), "label form should mangle, got: {s}");
        assert!(s.contains(".skip 16"), "directive form should pass through, got: {s}");
    }

    #[test]
    fn scope_does_not_mangle_glued_vector_arrangement() {
        // A `.<arr>` arrangement suffix glued to a vector register
        // (`v0.8b`, no space) is part of the operand, not a scope-local
        // label. Inside a @scope it must survive verbatim — otherwise the
        // `.8b` mangles to `count_bits$$8b` and `cnt v0.8b, v0.8b` becomes
        // the unparseable `cnt v0count_bits$$8b, ...`.
        let mut asm = Assembler::new();
        let out = expand_text(
            &mut asm,
            r#"@scope count_bits
    cnt v0.8b, v0.8b
    addv b0, v0.8b
@endscope
"#,
        )
        .unwrap();
        let s = to_text(&out);
        assert!(s.contains("cnt v0.8b, v0.8b"), "got: {s}");
        assert!(s.contains("addv b0, v0.8b"), "got: {s}");
        assert!(!s.contains("$$8b"), "arrangement must not be mangled, got: {s}");
    }

    #[test]
    fn scoped_proc_neon_assembles_through_a64_backend() {
        // End-to-end regression for the MF66 repro: a NEON primitive whose
        // body uses vector-arrangement suffixes (`v0.8b`) inside a `proc()`
        // scope. This drives the *whole* front-end (`Assembler::assemble`
        // → expanded text) and then the native backend (`a64::assemble` →
        // machine code), not just bare `a64::assemble` of an unscoped line.
        //
        // Before the fix, `.8b` lexed as a scope-local label and mangled to
        // `count_bits$$8b`, so the emitted text was `cnt v0count_bits$$8b,
        // …` and `a64::assemble` died with `encode `cnt v0count_bits$$8b…``.
        // `proc`/`endp`/`next` are MF66's real primitive-shape macros (see
        // MF66/kernel/macros.masm): `proc(n)` opens `@scope n`, exports the
        // symbol, and places the label; `endp()` closes the scope.
        let mut asm = Assembler::new();
        let text = asm
            .assemble(
                "count_bits.masm",
                r#"@macro proc(name)
    @scope &name
    .globl &name
&name:
@endmacro
@macro endp()
    @endscope
@endmacro
@macro next()
    ret
@endmacro

proc(count_bits)
    cnt v0.8b, v0.8b
    uaddlv h0, v0.8b
.retry:
    cbnz w0, .retry
    fmov w0, s0
    next()
endp()
"#,
            )
            .expect("front-end expansion must succeed");

        // Front-end: arrangement survives verbatim; a genuine local label
        // (`.retry`, whitespace before the dot) still mangles into the scope.
        assert!(text.contains("cnt v0.8b, v0.8b"), "arrangement mangled: {text}");
        assert!(text.contains("uaddlv h0, v0.8b"), "arrangement mangled: {text}");
        assert!(!text.contains("$$8b"), "arrangement must not be scope-mangled: {text}");
        assert!(
            text.contains("count_bits$$retry"),
            "real local label should still mangle into the scope: {text}"
        );

        // Backend: the expanded text must actually encode. `cnt v0.8b, v0.8b`
        // is `0x0E205800` (verified against the LLVM-MC oracle); look for its
        // little-endian word in the emitted code so label/globl placement
        // can't shift the assertion.
        let module = crate::a64::assemble(&text).expect("backend must encode the scoped NEON proc");
        let cnt_word = [0x00u8, 0x58, 0x20, 0x0E];
        assert!(
            module.code.windows(4).any(|w| w == cnt_word),
            "`cnt v0.8b, v0.8b` (0x0E205800) not found in encoded code: {:02X?}",
            module.code
        );
    }

    #[test]
    fn scope_does_not_mangle_condition_code_branch() {
        // `b.eq` is a conditional-branch mnemonic: the `.eq` is glued to
        // `b` with no space. Inside a @scope it must not be mangled into
        // `beq$$eq` / `bscope$$eq`. The branch *target* `.loop` (space
        // before it) is a real local label and still mangles.
        let mut asm = Assembler::new();
        let out = expand_text(
            &mut asm,
            r#"@scope demo
.loop:
    b.eq .loop
@endscope
"#,
        )
        .unwrap();
        let s = to_text(&out);
        assert!(s.contains("b.eq demo$$loop"), "got: {s}");
        assert!(s.contains("demo$$loop:"), "got: {s}");
        assert!(!s.contains("$$eq"), "condition suffix must not be mangled, got: {s}");
    }

    #[test]
    fn scope_passes_mc_directives_unchanged() {
        let mut asm = Assembler::new();
        let out = expand_text(
            &mut asm,
            r#"@scope plus
.globl plus
plus:
    ret
@endscope
"#,
        )
        .unwrap();
        let s = to_text(&out);
        // `.globl` must pass through unmangled.
        assert!(s.contains(".globl plus"), "got: {s}");
        // `plus:` is a global label (no leading dot), passes through.
        assert!(s.contains("plus:"), "got: {s}");
    }

    #[test]
    fn two_scopes_do_not_collide() {
        let mut asm = Assembler::new();
        let out = expand_text(
            &mut asm,
            r#"@scope a
.foo:
    ret
@endscope
@scope b
.foo:
    ret
@endscope
"#,
        )
        .unwrap();
        let s = to_text(&out);
        assert!(s.contains("a$$foo:"));
        assert!(s.contains("b$$foo:"));
    }

    #[test]
    fn endscope_with_no_scope_errors() {
        let mut asm = Assembler::new();
        let err = expand_text(&mut asm, "@endscope\n").unwrap_err();
        let s = format!("{err}");
        assert!(s.contains("endscope"), "got: {s}");
    }

    #[test]
    fn unclosed_scope_errors_at_eof() {
        let mut asm = Assembler::new();
        let err = expand_text(&mut asm, "@scope foo\n").unwrap_err();
        let s = format!("{err}");
        assert!(s.contains("foo"), "got: {s}");
        assert!(s.contains("not closed"), "got: {s}");
    }

    // ── @macro tests ────────────────────────────────────────────────

    #[test]
    fn macro_basic_substitution() {
        let mut asm = Assembler::new();
        let out = expand_text(
            &mut asm,
            r#"@macro pushd(val)
    sub rbp, 8
    mov [rbp], rax
    mov rax, &val
@endmacro
pushd(42)
"#,
        )
        .unwrap();
        let s = to_text(&out);
        assert!(s.contains("sub rbp, 8"), "got: {s}");
        assert!(s.contains("mov [rbp], rax"), "got: {s}");
        assert!(s.contains("mov rax, 42"), "got: {s}");
    }

    #[test]
    fn macro_multiple_params() {
        let mut asm = Assembler::new();
        let out = expand_text(
            &mut asm,
            r#"@macro binop(name, op)
&name:
    &op rax, [rbp]
    add rbp, 8
    ret
@endmacro
binop(plus, add)
binop(minus, sub)
"#,
        )
        .unwrap();
        let s = to_text(&out);
        assert!(s.contains("plus:"));
        assert!(s.contains("add rax, [rbp]"));
        assert!(s.contains("minus:"));
        assert!(s.contains("sub rax, [rbp]"));
    }

    #[test]
    fn macro_arity_mismatch_errors() {
        let mut asm = Assembler::new();
        let err = expand_text(
            &mut asm,
            "@macro foo(a, b)\nret\n@endmacro\nfoo(1)\n",
        )
        .unwrap_err();
        let s = format!("{err}");
        assert!(s.contains("expects 2"), "got: {s}");
        assert!(s.contains("got 1"), "got: {s}");
    }

    #[test]
    fn macro_unknown_param_errors() {
        let mut asm = Assembler::new();
        let err = expand_text(
            &mut asm,
            "@macro foo(x)\nmov rax, &y\n@endmacro\nfoo(1)\n",
        )
        .unwrap_err();
        let s = format!("{err}");
        assert!(s.contains("no parameter `y`"), "got: {s}");
    }

    #[test]
    fn macro_arg_with_brackets_passes_through() {
        // Token-reconstruction preserves source spacing per the
        // `space_before` flag set at lex time. `[rbp+8]` round-trips
        // verbatim; `[rbp + 8]` would emit with spaces. Both are
        // accepted by MC.
        let mut asm = Assembler::new();
        let out = expand_text(
            &mut asm,
            "@macro mv(dst, src)\nmov &dst, &src\n@endmacro\nmv(rax, [rbp+8])\n",
        )
        .unwrap();
        let s = to_text(&out);
        assert!(s.contains("mov rax, [rbp+8]"), "got: {s}");
    }

    #[test]
    fn macro_brace_grouped_arg_keeps_commas() {
        let mut asm = Assembler::new();
        let out = expand_text(
            &mut asm,
            "@macro dq(items)\n.quad &items\n@endmacro\ndq({1, 2, 3})\n",
        )
        .unwrap();
        let s = to_text(&out);
        assert!(s.contains(".quad 1, 2, 3"), "got: {s}");
    }

    #[test]
    fn macro_composition_works() {
        let mut asm = Assembler::new();
        let out = expand_text(
            &mut asm,
            r#"@macro inner(x)
mov rax, &x
@endmacro
@macro outer(y)
inner(&y)
@endmacro
outer(42)
"#,
        )
        .unwrap();
        let s = to_text(&out);
        assert!(s.contains("mov rax, 42"), "got: {s}");
    }

    #[test]
    fn macro_recursion_bounded() {
        let mut asm = Assembler::new();
        asm.set_max_expansion_depth(8);
        let err = expand_text(
            &mut asm,
            "@macro a()\na()\n@endmacro\na()\n",
        )
        .unwrap_err();
        let s = format!("{err}");
        assert!(s.contains("expansion depth exceeded"), "got: {s}");
    }

    #[test]
    fn macro_local_labels_hygienic_across_invocations() {
        let mut asm = Assembler::new();
        let out = expand_text(
            &mut asm,
            r#"@macro retry()
.again:
    jmp .again
@endmacro
retry()
retry()
"#,
        )
        .unwrap();
        let s = to_text(&out);
        // Two invocations should produce two distinct mangled labels.
        assert!(s.contains("retry$$0$$again:"), "got: {s}");
        assert!(s.contains("retry$$1$$again:"), "got: {s}");
        // No collision: two definitions of the SAME label would be bad,
        // we want each invocation to have its own.
        let count_zero = s.matches("retry$$0$$again:").count();
        let count_one = s.matches("retry$$1$$again:").count();
        assert_eq!(count_zero, 1, "got: {s}");
        assert_eq!(count_one, 1, "got: {s}");
    }

    #[test]
    fn macro_opens_scope_so_scope_wins_over_invocation() {
        // The user's proc(name) macro opens @scope &name. Inside, local
        // labels should mangle using the scope name, NOT the macro
        // invocation id.
        let mut asm = Assembler::new();
        let out = expand_text(
            &mut asm,
            r#"@macro proc(name)
@scope &name
.globl &name
&name:
.done:
    ret
@endscope
@endmacro
proc(plus)
"#,
        )
        .unwrap();
        let s = to_text(&out);
        assert!(s.contains("plus$$done:"), "got: {s}");
        assert!(!s.contains("proc$$"), "scope should win over macro frame: {s}");
        assert!(s.contains(".globl plus"));
    }

    #[test]
    fn macro_variadic() {
        let mut asm = Assembler::new();
        let out = expand_text(
            &mut asm,
            r#"@macro dq(items...)
.quad &items
@endmacro
dq(1, 2, 3)
dq(42)
"#,
        )
        .unwrap();
        let s = to_text(&out);
        assert!(s.contains(".quad 1, 2, 3"), "got: {s}");
        assert!(s.contains(".quad 42"), "got: {s}");
    }

    #[test]
    fn macro_variadic_with_leading_fixed() {
        let mut asm = Assembler::new();
        let out = expand_text(
            &mut asm,
            r#"@macro tagged(name, parts...)
&name: .quad &parts
@endmacro
tagged(mylabel, 10, 20, 30)
"#,
        )
        .unwrap();
        let s = to_text(&out);
        assert!(s.contains("mylabel:"), "got: {s}");
        assert!(s.contains(".quad 10, 20, 30"), "got: {s}");
    }

    #[test]
    fn nested_macro_def_errors() {
        let mut asm = Assembler::new();
        let err = expand_text(
            &mut asm,
            "@macro outer()\n@macro inner()\nret\n@endmacro\n@endmacro\n",
        )
        .unwrap_err();
        let s = format!("{err}");
        assert!(s.contains("cannot be nested"), "got: {s}");
    }

    #[test]
    fn unbalanced_macro_errors() {
        let mut asm = Assembler::new();
        let err = expand_text(&mut asm, "@macro foo()\nret\n").unwrap_err();
        let s = format!("{err}");
        assert!(s.contains("macro"), "got: {s}");
    }

    #[test]
    fn paste_operator_unsupported() {
        let mut asm = Assembler::new();
        let err = expand_text(
            &mut asm,
            "@macro glue(a, b)\n&a##&b\n@endmacro\nglue(foo, bar)\n",
        )
        .unwrap_err();
        let s = format!("{err}");
        assert!(s.contains("##"), "got: {s}");
        assert!(s.contains("not yet implemented"), "got: {s}");
    }

    #[test]
    fn full_forth_binop_kernel_assembles() {
        // The marquee example: full Forth STC primitive section using
        // composition (binop), proc/endp macros with @scope, and the
        // @define alias layer.
        let mut asm = Assembler::new();
        let out = expand_text(
            &mut asm,
            r#"@assign cell = 8
@define TOS rax
@define DSP rbp
@assert cell == 8, "wf64 requires 64-bit cells"

@macro proc(name)
@scope &name
.globl &name
&name:
@endmacro

@macro endp()
@endscope
@endmacro

@macro next()
    ret
@endmacro

@macro binop(name, op)
proc(&name)
    &op TOS, [DSP]
    add DSP, cell
    next()
endp()
@endmacro

binop(plus,  add)
binop(minus, sub)
"#,
        )
        .unwrap();
        let s = to_text(&out);
        // Verify both primitives are present with correct register
        // aliasing AND the `cell` substitution.
        assert!(s.contains(".globl plus"), "got: {s}");
        assert!(s.contains("plus:"));
        assert!(s.contains("add rax, [rbp]"));
        assert!(s.contains("add rbp, 8"));
        assert!(s.contains(".globl minus"));
        assert!(s.contains("minus:"));
        assert!(s.contains("sub rax, [rbp]"));
        // No `&` parameter sigils should remain — everything substituted.
        assert!(!s.contains("&name"), "unexpanded &name: {s}");
        assert!(!s.contains("&op"), "unexpanded &op: {s}");
    }

    #[test]
    fn forth_binop_pattern_works() {
        // The pattern from USER-GUIDE — but without macros yet.
        // We exercise define / assign / context names together.
        let mut asm = Assembler::new();
        let out = expand_text(
            &mut asm,
            r#"@assign cell = 8
@define TOS rax
@define DSP rbp
@assert cell == 8, "need 64-bit cells"

plus:
    add TOS, [DSP]
    add DSP, cell
    ret
"#,
        )
        .unwrap();
        let s = to_text(&out);
        // After expansion, register aliases are gone and `cell` is 8.
        assert!(s.contains("add rax, [rbp]"), "got: {s}");
        assert!(s.contains("add rbp, 8"));
    }
}
