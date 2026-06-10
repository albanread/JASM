//! Rasm — the native x86-64 encoder that replaces LLVM-MC.
//!
//! Input: the assembled (post-macro-expansion) MC-flavour Intel-syntax text the
//! `asm/` front-end already produces. Output: an [`EncodedModule`](crate::backend::EncodedModule)
//! the native [`NativeJit`](crate::native::NativeJit) loads. Tables/logic are
//! derived from LLVM-MC for byte-identity (see WF65 docs/design/rasm-replace-llvm.md).
//!
//! Layering: [`parse`] (text → [`Line`]) → [`encode`] (one instruction → bytes)
//! → this module's two-pass driver (assign offsets, resolve internal labels +
//! branch relaxation, emit relocs) → `EncodedModule`.

pub mod parse;

pub use parse::{Directive, Line, Mem, MemSize, Operand, Reg, RegClass};
