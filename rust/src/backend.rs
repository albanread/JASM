//! Backend seam for the Rasm migration (replacing LLVM-MC + MCJIT).
//!
//! Two traits split the two jobs LLVM does today, so a native Rust
//! implementation can be plugged in behind each without touching the
//! `asm/` macro front-end:
//!
//! * [`Encoder`] — assembled text → machine code + symbols + relocations.
//!   Today LLVM-MC does this *inside* MCJIT; the native `RasmEncoder`
//!   (Sprint 2) and the `IcedEncoder` bootstrap (Sprint 1) implement this.
//! * [`Loader`] — declare symbols / map host externs / place code in
//!   executable memory / resolve addresses. [`crate::jit::Jit`] (MCJIT)
//!   implements it today as `LlvmJit`; the native `NativeJit` (Sprint 1)
//!   implements it over `VirtualAlloc` + relocations + W^X.
//!
//! Sprint 0 only establishes the skeleton: the traits compile always, the
//! LLVM impls live behind the `llvm` feature. Nothing re-points onto the
//! trait objects until Sprint 1.

use std::collections::BTreeMap;
use std::ffi::c_void;

use anyhow::Result;

/// A relocation the loader must resolve once final addresses are known.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Reloc {
    /// Byte offset of the field to patch within the encoded code blob.
    pub at: usize,
    /// Field width in bytes (4 for rel32 / RIP-rel disp32, 8 for abs64).
    pub size: u8,
    pub kind: RelocKind,
    /// Target symbol name (internal label or host extern).
    pub target: String,
    /// Constant added to the resolved target before encoding.
    pub addend: i64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RelocKind {
    // ── x86-64 (rel32 family) ────────────────────────────────────────────
    /// `call`/`jmp`/`jcc rel32`: field = target - (field_addr + 4).
    BranchRel32,
    /// `lea reg,[rip+disp32]` and friends: field = target - (field_addr + 4).
    RipRel32,
    /// 64-bit absolute address embedded in a data cell.
    /// Shared with AArch64 (`.quad sym` → `ARM64_RELOC_UNSIGNED`).
    Abs64,

    // ── AArch64 (fields packed *within* the 32-bit instruction word) ──────
    // Unlike x86 rel32 (a contiguous little-endian field), these patch a
    // bit-slice of the 4-byte word, so the loader/encoder must insert the
    // immediate by masking — not by writing N little-endian bytes. See
    // docs/design/aarch64-apple-silicon.md.
    /// `b`/`bl` 26-bit PC-relative branch (±128 MB), `<<2`.
    /// Mach-O `ARM64_RELOC_BRANCH26`.
    Branch26,
    /// `adrp` 21-bit PC-relative *page* address (±4 GB), bits split
    /// `immlo[30:29]`/`immhi[23:5]`. Mach-O `ARM64_RELOC_PAGE21`.
    AdrpPage21,
    /// Low-12 page offset for the `add`/`ldr` that follows an `adrp`
    /// (`imm[21:10]`, scaled by access size for `ldr`). Mach-O
    /// `ARM64_RELOC_PAGEOFF12`.
    AddPageOff12,
}

/// The product of [`Encoder::encode`]: a position-independent code blob plus
/// the symbol table and relocation list the [`Loader`] needs to place it.
#[derive(Debug, Clone, Default)]
pub struct EncodedModule {
    /// The encoded bytes, with reloc fields left as placeholders (0).
    pub code: Vec<u8>,
    /// `name -> byte offset` for every `.globl`/labelled symbol defined here.
    pub symbols: BTreeMap<String, usize>,
    /// Relocations to apply at load time.
    pub relocs: Vec<Reloc>,
    /// Names referenced but not defined here (host externs to bind).
    pub externs: Vec<String>,
}

/// Encode assembled (post-macro-expansion) assembly into machine code.
///
/// Sprint 0 fixes the *contract*; the first implementation lands in Sprint 1.
/// Input is the assembled text the `asm/` front-end already produces today —
/// the same text LLVM-MC is fed — which keeps the encoder a drop-in for the
/// existing pipeline and lets it parse exactly what LLVM parses (aiding
/// byte-identity). A later refactor may pass a structured instruction stream
/// instead.
pub trait Encoder {
    fn encode(&self, asm_text: &str) -> Result<EncodedModule>;
}

/// Place encoded code in executable memory, bind host externs, resolve
/// symbol addresses. Mirrors the public surface of [`crate::jit::Jit`] so
/// the LLVM and native backends are interchangeable.
///
/// Object-safe (the generic [`Loader::lookup_fn`] is `where Self: Sized`), so
/// callers may hold `&mut dyn Loader` once the call path is re-pointed.
pub trait Loader {
    /// Append a chunk of assembly (LLVM path) or, for a native loader, hand it
    /// the assembled text it will encode+place on finalize. Build-time only.
    fn add_asm(&mut self, asm_text: &str) -> Result<()>;

    /// Advertise `name` (with `arg_count` i64 params) so it can be looked up.
    fn declare_fn(&mut self, name: &str, arg_count: usize) -> Result<()>;

    /// Declare `name` AND bind it to a host-process address (an `extern "C"`
    /// function with `arg_count` i64 args).
    fn define_extern_fn(&mut self, name: &str, arg_count: usize, addr: *mut c_void) -> Result<()>;

    /// Finalize if needed and return the runtime address of `name`.
    fn lookup_addr(&mut self, name: &str) -> Result<u64>;

    /// Look up `name` and transmute to a function-pointer type `F`.
    ///
    /// # Safety
    /// The caller asserts the JITed symbol matches `F`'s ABI.
    unsafe fn lookup_fn<F: Copy>(&mut self, name: &str) -> Result<F>
    where
        Self: Sized,
    {
        debug_assert_eq!(
            std::mem::size_of::<F>(),
            std::mem::size_of::<*const ()>(),
            "F must be a function pointer type"
        );
        let addr = self.lookup_addr(name)?;
        Ok(unsafe { std::mem::transmute_copy::<u64, F>(&addr) })
    }
}
