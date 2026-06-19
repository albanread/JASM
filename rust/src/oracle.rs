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

    /// Oracle for an explicit triple, e.g. `"aarch64-unknown-linux-gnu"`. The
    /// multi-arch hook — the rest of the pipeline is target-neutral.
    pub fn with_triple(triple: impl Into<String>) -> Self {
        Self { triple: Some(triple.into()) }
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
        init_x86_mcjit(); // registers target + AsmParser/AsmPrinter (idempotent)
        let triple = self.resolve_triple()?;

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
            let tm = LLVMCreateTargetMachine(
                target,
                triple.as_ptr(),
                empty.as_ptr(),
                empty.as_ptr(),
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

            // LLVM-MC inline asm defaults to AT&T; rasm always parses Intel.
            // Prepend the directive unless the assembled text already carries it.
            let body = ensure_intel(asm_text);
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
    let text = file
        .section_by_name(".text")
        .context("emitted object has no .text section")?;
    let text_index = text.index();
    let text_base = text.address();
    let code = text.data().context("read .text data")?.to_vec();

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
fn map_reloc(rel: &object::Relocation) -> Result<RelocKind> {
    use object::{RelocationFlags, RelocationKind as K};
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

/// Prepend `.intel_syntax noprefix` unless the text already selects it, and
/// guarantee a trailing newline so the final line parses.
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
        // No relocations → a direct byte-for-byte oracle comparison.
        let asm = "mov rax, 42\nsub rbp, 8\nadd rax, [rbp]\nmovzx ecx, byte ptr [rbp - 8]\nret\n";
        let o = LlvmMcEncoder::new().encode(asm).expect("oracle");
        let r = RasmEncoder.encode(asm).expect("rasm");
        assert_eq!(o.code, r.code, "oracle {:02x?} != rasm {:02x?}", o.code, r.code);
    }

    #[test]
    fn extracts_globl_symbol_and_extern_reloc() {
        let asm = ".globl w\nw:\ncall rt_emit\nret\n";
        let o = LlvmMcEncoder::new().encode(asm).expect("oracle");
        assert_eq!(o.symbols.get("w"), Some(&0), "w must be exported at offset 0");
        assert_eq!(o.externs, vec!["rt_emit".to_string()]);
        assert_eq!(o.relocs.len(), 1, "one reloc, got {:?}", o.relocs);
        assert_eq!(o.relocs[0].kind, RelocKind::BranchRel32);
        assert_eq!(o.relocs[0].target, "rt_emit");
        assert_eq!(o.code[0], 0xE8, "call rel32 opcode");
    }

    #[test]
    fn rejects_garbage_via_diagnostics_not_abort() {
        let err = LlvmMcEncoder::new().encode("this_is_not_an_instruction foo\n");
        assert!(err.is_err(), "bad asm must return Err, not abort");
    }
}
