//! `wfasm` — the WF Forth assembler + JIT.
//!
//! Layering (bottom up):
//!
//! * [`llvm`]   — raw `extern "C"` bindings to LLVM-C.dll. Unsafe, thin.
//! * [`jit`]    — safe Rust wrapper around ORC LLJIT. Holds the context,
//!                module, and dylib references. Owns symbol lookup.
//! * (parser)   — MASM-style lexer/parser. Not yet implemented. Will produce
//!                an IR that lowers to a string of LLVM-flavor Intel asm.
//! * (macros)   — Rust closures that expand into IR. Registered with the
//!                parser. Not yet implemented.
//!
//! The first milestone (`bin/hello_jit.rs`) sidesteps the parser entirely:
//! it feeds a fixed assembly string straight into the JIT to prove the
//! LLVM pipe is wired up. The MASM parser then plugs in above this layer.

pub mod asm;

// LLVM-MC assembler + MCJIT loader. Behind the `llvm` feature (default ON) —
// the Rasm migration replaces these with a native encoder+loader behind the
// `backend` traits. With `llvm` off, the crate compiles to the front-end +
// trait skeleton + SEH, with no LLVM-C dependency.
#[cfg(feature = "llvm")]
pub mod llvm;
#[cfg(feature = "llvm")]
pub mod jit;

/// Backend seam (Encoder / Loader traits) for the Rasm migration — replacing
/// LLVM-MC + MCJIT with a native Rust assembler + loader. Always compiled; the
/// LLVM impls live alongside `Jit`.
pub mod backend;

#[cfg(windows)]
pub mod win32;

#[cfg(windows)]
pub mod seh;

/// Native loader (`NativeJit`) — the Rasm replacement for MCJIT's place +
/// relocate + protect job. No LLVM; always compiled (Windows).
#[cfg(windows)]
pub mod native;

pub use asm::Assembler;
pub use backend::{EncodedModule, Encoder, Loader, Reloc, RelocKind};
#[cfg(feature = "llvm")]
pub use jit::{CodeArena, Jit, JitError};
