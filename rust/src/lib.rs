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
pub mod llvm;
pub mod jit;

#[cfg(windows)]
pub mod win32;

#[cfg(windows)]
pub mod seh;

pub use asm::Assembler;
pub use jit::{Jit, JitError};
