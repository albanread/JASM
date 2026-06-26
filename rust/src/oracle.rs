//! `LlvmMcEncoder` — the LLVM-MC differential oracle.
//!
//! Implements the same [`Encoder`](crate::backend::Encoder) trait as
//! [`RasmEncoder`](crate::rasm::RasmEncoder), but produces its bytes by handing
//! the assembled Intel-syntax text to LLVM-MC as module-level inline asm and
//! emitting a **relocatable object** (`TargetMachine::EmitToMemoryBuffer`). The
//! object is parsed with the [`object`] crate into an
//! [`EncodedModule`](crate::backend::EncodedModule) — `.text` bytes (reloc
//! fields left as zero placeholders), `.globl` symbols, and a relocation table —
//! the exact shape rasm emits, so the two can be diffed byte-for-byte.
//!
//! This is the byte-for-byte ground truth rasm is built against (`encode.rs`:
//! *"chosen to match LLVM-MC … the golden differential gates that"*). It is a
//! build/test-time tool only, behind the `llvm` feature; nothing here is on the
//! shipping native path.
//!
//! Multi-arch: the object route is target-neutral. `LlvmMcEncoder::with_triple`
//! retargets LLVM-MC; the `object` crate parses COFF/ELF/Mach-O uniformly. The
//! only per-arch seam is [`map_reloc`] (object reloc type → [`RelocKind`]).
//! See `docs/design/rasm-difftest.md`.
#![cfg(feature = "llvm")]

use std::collections::BTreeMap;
use std::ffi::{c_char, c_void, CStr, CString};

use anyhow::{anyhow, bail, Context, Result};
use object::{Object, ObjectSection, ObjectSymbol, RelocationTarget, SymbolKind};

use crate::backend::{EncodedModule, Encoder, Reloc, RelocKind};
use crate::llvm::*;

/// LLVM-MC-backed [`Encoder`] used as the differential oracle for rasm.
#[derive(Debug, Clone, Default)]
pub struct LlvmMcEncoder {
    /// Target triple. `None` = the host default (what the shipping JIT uses).
    triple: Option<String>,
}

impl LlvmMcEncoder {
    /// Oracle for the host target (matches the MCJIT path's triple).
    pub fn new() -> Self {
        Self { triple: None }
    }

    /// Oracle for an explicit triple, e.g. `"aarch64-apple-darwin"`. The
    /// multi-arch hook — the rest of the pipeline is target-neutral.
    pub fn with_triple(triple: impl Into<String>) -> Self {
        Self { triple: Some(triple.into()) }
    }

    /// Oracle pinned to the project's canonical x86-64 target (Windows/COFF) —
    /// the triple the committed corpus was recorded against. Host-independent,
    /// so the x86 differential against [`RasmEncoder`](crate::rasm::RasmEncoder)
    /// runs identically whether the build host is x86-64 or Apple Silicon. LLVM
    /// cross-assembles, so this is correct on an AArch64 Mac too.
    pub fn x86_64() -> Self {
        Self::with_triple("x86_64-pc-windows-msvc")
    }

    /// Oracle pinned to AArch64 macOS (Mach-O) — the target the Apple Silicon
    /// native encoder is gated against. See docs/design/aarch64-apple-silicon.md.
    pub fn aarch64_macos() -> Self {
        Self::with_triple("aarch64-apple-darwin")
    }

    /// Resolve the triple to an owned C string (host default if unset).
    fn resolve_triple(&self) -> Result<CString> {
        if let Some(t) = &self.triple {
            return CString::new(t.as_str()).context("triple has interior NUL");
        }
        unsafe {
            let p = LLVMGetDefaultTargetTriple();
            if p.is_null() {
                bail!("LLVMGetDefaultTargetTriple returned null");
            }
            let s = CStr::from_ptr(p).to_owned();
            LLVMDisposeMessage(p);
            Ok(s)
        }
    }

    /// Assemble `asm_text` and return the emitted relocatable object's bytes.
    fn emit_object(&self, asm_text: &str) -> Result<Vec<u8>> {
        init_targets_and_mcjit(); // registers X86 + AArch64 + AsmParser/Printer (idempotent)
        let triple = self.resolve_triple()?;
        let is_x86 = triple_is_x86(&triple);

        // Collect LLVM error-severity diagnostics (e.g. inline-asm parse errors)
        // instead of letting them print+abort. The box must outlive the context;
        // declared first so it drops last (Rust drops locals in reverse order).
        let mut diag_errors: Box<Vec<String>> = Box::new(Vec::new());

        unsafe {
            // Resolve target for the triple.
            let mut target: LLVMTargetRef = std::ptr::null_mut();
            let mut err: *mut c_char = std::ptr::null_mut();
            if LLVMGetTargetFromTriple(triple.as_ptr(), &mut target, &mut err) != 0 {
                bail!("LLVMGetTargetFromTriple({triple:?}): {}", take_msg(err));
            }

            let empty = CString::new("").unwrap();
            // x86: enable AVX-512 so the assembler accepts zmm / EVEX forms (the
            // explicit asm chooses the encoding; features only gate availability).
            // AArch64: leave features empty — base ARMv8-A covers the kernel
            // surface; the `+avx512*` flags are X86-only and would be rejected.
            let features = if is_x86 {
                CString::new("+avx512f,+avx512vl,+avx512dq,+avx512bw").unwrap()
            } else {
                // Apple Silicon baseline extensions so the oracle accepts the
                // full AArch64 surface (LSE atomics, CRC32, half-precision, …).
                // Features only gate *acceptance*; the explicit asm picks the
                // encoding, so this never changes emitted bytes.
                CString::new("+lse,+crc,+fullfp16,+rcpc,+rdm,+dotprod,+aes,+sha2").unwrap()
            };
            let tm = LLVMCreateTargetMachine(
                target,
                triple.as_ptr(),
                empty.as_ptr(),
                features.as_ptr(),
                LLVMCodeGenOptLevel::Default,
                LLVMRelocMode::Static,
                LLVMCodeModel::Small,
            );
            if tm.is_null() {
                bail!("LLVMCreateTargetMachine({triple:?}) returned null");
            }
            let _tm = TargetMachineGuard(tm);

            let ctx = LLVMContextCreate();
            assert!(!ctx.is_null(), "LLVMContextCreate returned null");
            let _ctx = ContextGuard(ctx);
            let errors_ptr = (&mut *diag_errors as *mut Vec<String>) as *mut c_void;
            LLVMContextSetDiagnosticHandler(ctx, Some(diag_handler), errors_ptr);

            let modname = CString::new("rasm_oracle").unwrap();
            let module = LLVMModuleCreateWithNameInContext(modname.as_ptr(), ctx);
            assert!(!module.is_null(), "LLVMModuleCreateWithNameInContext returned null");
            let _module = ModuleGuard(module);
            LLVMSetTarget(module, triple.as_ptr());

            // x86: LLVM-MC inline asm defaults to AT&T; rasm always parses Intel,
            // so prepend `.intel_syntax noprefix` unless the text already has it.
            // AArch64 has a single (non-prefixed) GAS syntax — pass it through.
            let body = if is_x86 {
                ensure_intel(asm_text)
            } else {
                ensure_trailing_newline(asm_text)
            };
            LLVMAppendModuleInlineAsm(module, body.as_ptr() as *const c_char, body.len());

            // Emit the relocatable object into a memory buffer. Param order is
            // (TargetMachine, Module, FileType, &out_err, &out_buf); does not
            // consume the module.
            let mut buf: LLVMMemoryBufferRef = std::ptr::null_mut();
            let mut err2: *mut c_char = std::ptr::null_mut();
            let rc = LLVMTargetMachineEmitToMemoryBuffer(
                tm,
                module,
                LLVMCodeGenFileType::ObjectFile,
                &mut err2,
                &mut buf,
            );

            // Inline-asm parse errors surface through the diagnostic handler;
            // surface them as a Result rather than trusting `rc` alone.
            if !diag_errors.is_empty() {
                let joined = diag_errors.join("; ");
                if !buf.is_null() {
                    LLVMDisposeMemoryBuffer(buf);
                }
                bail!("LLVM-MC rejected asm: {joined}");
            }
            if rc != 0 || buf.is_null() {
                bail!("LLVMTargetMachineEmitToMemoryBuffer failed: {}", take_msg(err2));
            }

            let start = LLVMGetBufferStart(buf) as *const u8;
            let len = LLVMGetBufferSize(buf);
            let object_bytes = std::slice::from_raw_parts(start, len).to_vec();
            LLVMDisposeMemoryBuffer(buf);
            Ok(object_bytes)
        }
    }
}

impl Encoder for LlvmMcEncoder {
    fn encode(&self, asm_text: &str) -> Result<EncodedModule> {
        let obj = self.emit_object(asm_text)?;
        parse_object(&obj)
    }
}

// ── object → EncodedModule ───────────────────────────────────────────────────

/// Parse an emitted relocatable object into the same shape rasm produces.
fn parse_object(bytes: &[u8]) -> Result<EncodedModule> {
    let file = object::File::parse(bytes).context("parse emitted object")?;
    // COFF/ELF call it `.text`; Mach-O calls it `__text`. Fall back to the
    // first executable text section so the route stays object-format-neutral.
    let text = file
        .section_by_name(".text")
        .or_else(|| file.section_by_name("__text"))
        .or_else(|| file.sections().find(|s| s.kind() == object::SectionKind::Text))
        .context("emitted object has no text section")?;
    let text_index = text.index();
    let text_base = text.address();
    let code = text.data().context("read .text data")?.to_vec();

    // Symbol names are kept verbatim across all object formats: the assembler
    // emits assembly-level labels as-is on every target (the leading-`_` seen on
    // Mach-O for *C* symbols is a compiler convention, not an assembler one — a
    // bare `.globl w` stays `w`), so rasm and the oracle agree without mangling.

    // `.globl` symbols defined in .text → name -> offset. Skip section symbols
    // and local labels (rasm only exports globls).
    let mut symbols = BTreeMap::new();
    for sym in file.symbols() {
        if sym.kind() == SymbolKind::Section || !sym.is_definition() || !sym.is_global() {
            continue;
        }
        if sym.section_index() != Some(text_index) {
            continue;
        }
        let name = sym.name().context("symbol name not UTF-8")?;
        if name.is_empty() {
            continue;
        }
        let off = sym.address().saturating_sub(text_base) as usize;
        symbols.insert(name.to_string(), off);
    }

    // Relocations against .text → Reloc list + extern names (undefined targets).
    let mut relocs = Vec::new();
    let mut externs = Vec::new();
    for (off, rel) in text.relocations() {
        let (name, undefined) = match rel.target() {
            RelocationTarget::Symbol(idx) => {
                let s = file.symbol_by_index(idx).context("reloc target symbol")?;
                (s.name().context("reloc symbol name")?.to_string(), s.is_undefined())
            }
            other => bail!("unsupported relocation target {other:?}"),
        };
        let kind = map_reloc(&rel)
            .with_context(|| format!("reloc at {off:#x} targeting {name}"))?;
        relocs.push(Reloc {
            at: off as usize,
            size: (rel.size() / 8).max(1),
            kind,
            target: name.clone(),
            addend: rel.addend(),
        });
        if undefined {
            externs.push(name);
        }
    }
    externs.sort();
    externs.dedup();

    Ok(EncodedModule { code, symbols, relocs, externs })
}

/// Map an object-file relocation to rasm's [`RelocKind`]. Per-arch seam.
///
/// Caveat (x86-64): both branch `rel32` (`call`/`jmp`/`jcc`) and RIP-relative
/// `disp32` (`lea [rip+sym]`) emit the *same* machine relocation
/// (`IMAGE_REL_AMD64_REL32` / `R_X86_64_PC32`), so they're indistinguishable
/// from the object alone — both map to [`RelocKind::BranchRel32`]. The diff
/// driver must therefore treat `BranchRel32` and `RipRel32` as one class.
///
/// AArch64 (Mach-O): each field shape carries a distinct reloc type, so they
/// map 1:1. `object` would otherwise fold `BRANCH26` into the generic 32-bit
/// `Relative` bucket — we classify by the raw Mach-O type first to avoid that.
fn map_reloc(rel: &object::Relocation) -> Result<RelocKind> {
    use object::{RelocationFlags, RelocationKind as K};

    // Mach-O ARM64 first (types from <mach-o/arm64/reloc.h>).
    if let RelocationFlags::MachO { r_type, .. } = rel.flags() {
        return match r_type {
            0 => Ok(RelocKind::Abs64),         // ARM64_RELOC_UNSIGNED (.quad sym)
            2 => Ok(RelocKind::Branch26),      // ARM64_RELOC_BRANCH26 (b/bl)
            3 => Ok(RelocKind::AdrpPage21),    // ARM64_RELOC_PAGE21 (adrp)
            4 => Ok(RelocKind::AddPageOff12),  // ARM64_RELOC_PAGEOFF12 (add/ldr lo12)
            // GOT-load + subtractor/addend variants are not emitted by the
            // kernel's @PAGE/@PAGEOFF forms; surface them rather than silently
            // treating an indirect GOT load as a direct page reference.
            other => Err(anyhow!("unmapped Mach-O ARM64 reloc type {other}")),
        };
    }

    match (rel.kind(), rel.size()) {
        (K::Absolute, 64) => return Ok(RelocKind::Abs64),
        (K::Relative, 32) | (K::PltRelative, 32) => return Ok(RelocKind::BranchRel32),
        _ => {}
    }
    // Fall back to the raw container reloc type (object reports some COFF kinds
    // as `Unknown`). COFF AMD64 types from `winnt.h`.
    match rel.flags() {
        RelocationFlags::Coff { typ } => match typ {
            0x0001 => Ok(RelocKind::Abs64),       // IMAGE_REL_AMD64_ADDR64
            0x0004..=0x0009 => Ok(RelocKind::BranchRel32), // REL32[_1.._5]
            other => Err(anyhow!("unmapped COFF reloc type {other:#06x}")),
        },
        other => Err(anyhow!("unmapped reloc {:?} size {}", other, rel.size())),
    }
}

// ── helpers ──────────────────────────────────────────────────────────────────

/// Does this triple target an x86 family CPU (needs Intel-syntax + AVX flags)?
fn triple_is_x86(triple: &CStr) -> bool {
    let t = triple.to_string_lossy();
    t.starts_with("x86_64") || t.starts_with("i686") || t.starts_with("i386") || t.starts_with("i586")
}

/// Prepend `.intel_syntax noprefix` unless the text already selects it, and
/// guarantee a trailing newline so the final line parses. (x86 only.)
fn ensure_intel(asm: &str) -> Vec<u8> {
    let mut s = String::new();
    if !asm.contains(".intel_syntax") {
        s.push_str(".intel_syntax noprefix\n");
    }
    s.push_str(asm);
    if !s.ends_with('\n') {
        s.push('\n');
    }
    s.into_bytes()
}

/// Pass the text through unchanged except for a guaranteed trailing newline
/// (AArch64 / non-x86 — single GAS syntax, no Intel directive).
fn ensure_trailing_newline(asm: &str) -> Vec<u8> {
    let mut s = asm.to_string();
    if !s.ends_with('\n') {
        s.push('\n');
    }
    s.into_bytes()
}

/// Take ownership of an LLVM-allocated message string and free it.
unsafe fn take_msg(p: *mut c_char) -> String {
    if p.is_null() {
        return String::new();
    }
    let s = CStr::from_ptr(p).to_string_lossy().into_owned();
    LLVMDisposeMessage(p);
    s
}

/// Diagnostic handler: collect error-severity messages into the `Vec<String>`
/// passed as the opaque context. Mirrors `jit.rs`.
unsafe extern "C" fn diag_handler(diag: LLVMDiagnosticInfoRef, ctx: *mut c_void) {
    if ctx.is_null() {
        return;
    }
    let errors = &mut *(ctx as *mut Vec<String>);
    let sev = LLVMGetDiagInfoSeverity(diag);
    let desc = LLVMGetDiagInfoDescription(diag);
    if !desc.is_null() {
        let msg = CStr::from_ptr(desc).to_string_lossy().into_owned();
        LLVMDisposeMessage(desc);
        if sev == LLVMDiagnosticSeverity::LLVMDSError {
            errors.push(msg);
        }
    }
}

// RAII guards so early `bail!`s don't leak LLVM objects. Declaration order in
// `emit_object` ensures the module is disposed before its context.
struct TargetMachineGuard(LLVMTargetMachineRef);
impl Drop for TargetMachineGuard {
    fn drop(&mut self) {
        unsafe { LLVMDisposeTargetMachine(self.0) }
    }
}
struct ContextGuard(LLVMContextRef);
impl Drop for ContextGuard {
    fn drop(&mut self) {
        unsafe { LLVMContextDispose(self.0) }
    }
}
struct ModuleGuard(LLVMModuleRef);
impl Drop for ModuleGuard {
    fn drop(&mut self) {
        unsafe { LLVMDisposeModule(self.0) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rasm::RasmEncoder;

    #[test]
    fn matches_rasm_on_reg_imm_alu() {
        // No relocations → a direct byte-for-byte oracle comparison. Pinned to
        // the x86-64 triple so it runs identically on an AArch64 build host.
        let asm = "mov rax, 42\nsub rbp, 8\nadd rax, [rbp]\nmovzx ecx, byte ptr [rbp - 8]\nret\n";
        let o = LlvmMcEncoder::x86_64().encode(asm).expect("oracle");
        let r = RasmEncoder.encode(asm).expect("rasm");
        assert_eq!(o.code, r.code, "oracle {:02x?} != rasm {:02x?}", o.code, r.code);
    }

    #[test]
    fn extracts_globl_symbol_and_extern_reloc() {
        let asm = ".globl w\nw:\ncall rt_emit\nret\n";
        let o = LlvmMcEncoder::x86_64().encode(asm).expect("oracle");
        assert_eq!(o.symbols.get("w"), Some(&0), "w must be exported at offset 0");
        assert_eq!(o.externs, vec!["rt_emit".to_string()]);
        assert_eq!(o.relocs.len(), 1, "one reloc, got {:?}", o.relocs);
        assert_eq!(o.relocs[0].kind, RelocKind::BranchRel32);
        assert_eq!(o.relocs[0].target, "rt_emit");
        assert_eq!(o.code[0], 0xE8, "call rel32 opcode");
    }

    #[test]
    fn rejects_garbage_via_diagnostics_not_abort() {
        let err = LlvmMcEncoder::x86_64().encode("this_is_not_an_instruction foo\n");
        assert!(err.is_err(), "bad asm must return Err, not abort");
    }

    // ── AArch64 / macOS Mach-O oracle bring-up (the Apple Silicon ground truth) ──

    /// The oracle assembles AArch64 and returns byte-identical machine code.
    /// Expected bytes verified independently with
    /// `llvm-mc -triple=aarch64-apple-darwin --show-encoding`.
    #[test]
    fn aarch64_oracle_emits_known_bytes() {
        let asm = "ret\nnop\nmov x0, #42\nadd x0, x1, x2\nsub sp, sp, #16\n";
        let o = LlvmMcEncoder::aarch64_macos().encode(asm).expect("aarch64 oracle");
        #[rustfmt::skip]
        let expected: Vec<u8> = vec![
            0xc0, 0x03, 0x5f, 0xd6, // ret
            0x1f, 0x20, 0x03, 0xd5, // nop
            0x40, 0x05, 0x80, 0xd2, // mov  x0, #42
            0x20, 0x00, 0x02, 0x8b, // add  x0, x1, x2
            0xff, 0x43, 0x00, 0xd1, // sub  sp, sp, #16
        ];
        assert_eq!(o.code, expected, "aarch64 oracle {:02x?}", o.code);
        assert!(o.relocs.is_empty() && o.externs.is_empty());
    }

    /// Mach-O path: `__text` section, verbatim global symbol, undefined extern,
    /// and the `ARM64_RELOC_BRANCH26` → [`RelocKind::Branch26`] mapping.
    #[test]
    fn aarch64_oracle_extracts_symbol_and_branch26_reloc() {
        let asm = ".globl w\nw:\nbl rt_emit\nret\n";
        let o = LlvmMcEncoder::aarch64_macos().encode(asm).expect("aarch64 oracle");
        assert_eq!(o.symbols.get("w"), Some(&0), "w exported at offset 0");
        assert_eq!(o.externs, vec!["rt_emit".to_string()]);
        assert_eq!(o.relocs.len(), 1, "one reloc, got {:?}", o.relocs);
        assert_eq!(o.relocs[0].kind, RelocKind::Branch26);
        assert_eq!(o.relocs[0].target, "rt_emit");
        // bl is a 4-byte word; the internal `ret` follows at offset 4.
        assert_eq!(o.code.len(), 8);
    }
}
