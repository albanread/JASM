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

/// Near (rel32-reachable) RWX code arena for runtime words. LLVM-independent.
pub mod arena;

// LLVM-MC assembler + MCJIT loader. Behind the `llvm` feature (default ON) —
// the Rasm migration replaces these with a native encoder+loader behind the
// `backend` traits. With `llvm` off, the crate compiles to the front-end +
// trait skeleton + SEH, with no LLVM-C dependency.
#[cfg(feature = "llvm")]
pub mod llvm;
#[cfg(feature = "llvm")]
pub mod jit;

/// `LlvmMcEncoder` — the LLVM-MC differential oracle (an `Encoder` impl that
/// emits a relocatable object and parses it back). Build/test-time only; the
/// byte-for-byte ground truth rasm is gated against. See
/// `docs/design/rasm-difftest.md`.
#[cfg(feature = "llvm")]
pub mod oracle;

/// Backend seam (Encoder / Loader traits) for the Rasm migration — replacing
/// LLVM-MC + MCJIT with a native Rust assembler + loader. Always compiled; the
/// LLVM impls live alongside `Jit`.
pub mod backend;

/// Rasm — the native x86-64 encoder (text → machine code) replacing LLVM-MC.
/// Pure Rust, no LLVM. See WF65 docs/design/rasm-replace-llvm.md.
pub mod rasm;

/// Differential driver: diff rasm against an oracle `Encoder` (byte + reloc),
/// with reloc-field masking. Arch/oracle-neutral; gated tests use the LLVM
/// oracle. See `docs/design/rasm-difftest.md`.
pub mod difftest;

#[cfg(windows)]
pub mod win32;

#[cfg(windows)]
pub mod seh;

/// Native loader (`NativeJit`) — the Rasm replacement for MCJIT's place +
/// relocate + protect job. No LLVM; always compiled (Windows).
#[cfg(windows)]
pub mod native;

pub use arena::CodeArena;
pub use asm::Assembler;
pub use backend::{EncodedModule, Encoder, Loader, Reloc, RelocKind};
#[cfg(feature = "llvm")]
pub use jit::{Jit, JitError};
#[cfg(feature = "llvm")]
pub use oracle::LlvmMcEncoder;
