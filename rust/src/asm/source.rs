//! Source bookkeeping.
//!
//! Every file the assembler touches — the top-level source plus
//! everything `@include` pulls in — is registered with the `SourceMap`
//! and identified by a stable `FileId`. Spans on tokens carry that id;
//! diagnostics resolve through the map when they need a path or a
//! source line for an error message.

use std::path::PathBuf;

/// A compact, stable id for a file in the [`SourceMap`].
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct FileId(pub u32);

impl FileId {
    /// Sentinel for "nowhere" / synthesized tokens that don't trace to a
    /// real source location. Use sparingly.
    pub const SYNTHETIC: FileId = FileId(u32::MAX);

    pub fn is_synthetic(self) -> bool {
        self == Self::SYNTHETIC
    }
}

/// One file's text plus its on-disk path. The text is owned because the
/// lexer borrows from it and we want a single home.
#[derive(Debug)]
pub struct SourceFile {
    pub path: PathBuf,
    pub text: String,
}

#[derive(Debug, Default)]
pub struct SourceMap {
    files: Vec<SourceFile>,
}

impl SourceMap {
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a file's contents. The returned `FileId` is stable for
    /// the lifetime of the map.
    pub fn add(&mut self, path: PathBuf, text: String) -> FileId {
        let id = FileId(self.files.len() as u32);
        self.files.push(SourceFile { path, text });
        id
    }

    /// Convenience: register a synthetic in-memory source under a
    /// pseudo-path. Used by tests and by host-injected snippets.
    pub fn add_anon(&mut self, label: &str, text: String) -> FileId {
        self.add(PathBuf::from(format!("<{label}>")), text)
    }

    pub fn get(&self, id: FileId) -> &SourceFile {
        &self.files[id.0 as usize]
    }

    pub fn path(&self, id: FileId) -> &std::path::Path {
        &self.get(id).path
    }

    pub fn text(&self, id: FileId) -> &str {
        &self.get(id).text
    }

    /// Number of registered files.
    pub fn len(&self) -> usize {
        self.files.len()
    }

    pub fn is_empty(&self) -> bool {
        self.files.is_empty()
    }

    /// Pull a specific line (1-based) from a file. Returns `None` if the
    /// line is out of range. Useful for error rendering.
    pub fn line(&self, id: FileId, line: u32) -> Option<&str> {
        let text = self.text(id);
        text.lines().nth(line.saturating_sub(1) as usize)
    }
}
