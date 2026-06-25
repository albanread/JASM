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

pub mod assemble;
pub mod encode;
pub mod parse;

pub use assemble::assemble;
pub use encode::{encode, Encoded, Fixup, FixupKind};
pub use parse::{Directive, Line, Mem, MemSize, Operand, Reg, RegClass};

/// The native from-scratch x86-64 [`Encoder`](crate::backend::Encoder) — the
/// owned replacement for LLVM-MC.
#[derive(Debug, Default, Clone, Copy)]
pub struct RasmEncoder;

impl crate::backend::Encoder for RasmEncoder {
    fn encode(&self, asm_text: &str) -> anyhow::Result<crate::backend::EncodedModule> {
        assemble(asm_text)
    }
}
