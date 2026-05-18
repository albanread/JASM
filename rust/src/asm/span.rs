//! Source spans: every token and every diagnostic carries one.
//!
//! Positions are 1-based for `line` and `col` (matching the convention
//! editors and most tools use). `len` is a count of bytes — wfasm is
//! byte-oriented; identifiers are ASCII; non-ASCII bytes inside strings
//! and comments are passed through but not interpreted.

use super::source::FileId;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Span {
    pub file: FileId,
    pub line: u32,
    pub col: u32,
    pub len: u32,
}

impl Span {
    pub const fn new(file: FileId, line: u32, col: u32, len: u32) -> Self {
        Self { file, line, col, len }
    }

    /// Sentinel span for synthesized tokens that don't trace to source.
    /// Use sparingly — most synthesized tokens should carry the span of
    /// their generator (e.g., a macro expansion stamps tokens with the
    /// call site span).
    pub const SYNTHETIC: Span = Span {
        file: FileId::SYNTHETIC,
        line: 0,
        col: 0,
        len: 0,
    };

    /// Build the smallest span containing both inputs. Both spans must
    /// be in the same file (we don't model cross-file ranges).
    pub fn merge(self, other: Span) -> Span {
        debug_assert_eq!(
            self.file, other.file,
            "Span::merge across files is meaningless"
        );
        let self_end = self.col + self.len;
        let other_end = other.col + other.len;
        let start_line = self.line.min(other.line);
        let start_col = if self.line < other.line {
            self.col
        } else if other.line < self.line {
            other.col
        } else {
            self.col.min(other.col)
        };
        let end_line = self.line.max(other.line);
        let end_col = if self.line > other.line {
            self_end
        } else if other.line > self.line {
            other_end
        } else {
            self_end.max(other_end)
        };
        // For multi-line merges we don't pretend `len` is meaningful;
        // callers that need the full bytes count should track it
        // separately. We set len to the run within the *end* line.
        let len = if start_line == end_line {
            end_col.saturating_sub(start_col)
        } else {
            end_col
        };
        Span {
            file: self.file,
            line: start_line,
            col: start_col,
            len,
        }
    }
}

impl std::fmt::Display for Span {
    /// Render as `path:line:col`. The `SourceMap` isn't reachable from
    /// `Span`, so this prints the file id; higher layers that have the
    /// map should render their own form.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.file.is_synthetic() {
            write!(f, "<synthetic>:{}:{}", self.line, self.col)
        } else {
            write!(f, "file#{}:{}:{}", self.file.0, self.line, self.col)
        }
    }
}
