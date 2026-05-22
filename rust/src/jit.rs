//! Safe wrapper around LLVM **MCJIT** for our assembler use case.
//!
//! The deal:
//!
//! 1. We hold a single `LLVMModule` and grow it by appending module-level
//!    inline asm strings (`LLVMAppendModuleInlineAsm`) and `declare`ing each
//!    asm-defined symbol as an extern IR function. Symbol resolution
//!    matches the IR `declare` to the asm-emitted symbol at link time —
//!    this only works with MCJIT, NOT with ORC LLJIT.
//!
//! 2. On first `lookup_fn`, MCJIT compiles + finalizes the whole module.
//!    Subsequent lookups return cached addresses for free.
//!
//! 3. Rust runtime functions are plumbed in with `define_extern_fn` (calls
//!    `LLVMAddGlobalMapping` under the hood). The JIT'd asm can `call`
//!    them by name as long as the asm `.extern`'s them.
//!
//! Trade-off: MCJIT is whole-module — once it finalizes you cannot append.
//! For a Forth where the kernel is one bag of primitives plus runtime
//! helpers, that's fine. If we later want per-definition lazy compilation
//! (for a REPL `:` colon definition), we'll add modules via
//! `LLVMAddModule`, but that's not what this layer does today.

use std::ffi::{c_void, CStr, CString};
use std::os::raw::c_uint;
use std::ptr;

use crate::llvm::*;

#[derive(Debug)]
pub enum JitError {
    Llvm(String),
    NotFound(String),
    Nul(String),
    AlreadyFinalized,
}

impl std::fmt::Display for JitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            JitError::Llvm(s) => write!(f, "LLVM error: {s}"),
            JitError::NotFound(s) => write!(f, "symbol `{s}` not found"),
            JitError::Nul(s) => write!(f, "string {s:?} contained an interior NUL"),
            JitError::AlreadyFinalized => write!(
                f,
                "JIT module already finalized; further add_asm/define_extern_fn not allowed"
            ),
        }
    }
}
impl std::error::Error for JitError {}

/// Owns the MCJIT engine, the underlying LLVMContext, and (until handed
/// over) the module we accumulate into.
///
/// Lifecycle:
///
/// * `new()` → empty module ready to receive asm + declarations.
/// * `add_asm()` / `declare_fn()` / `define_extern_fn()` → build the module.
///   Allowed any number of times before the first lookup.
/// * `lookup_fn()` → first call hands the module to MCJIT, which compiles
///   and finalizes it. From that point on, the module pointer is owned by
///   MCJIT — we drop our `module` slot to `None` to make further mutation
///   attempts hit `AlreadyFinalized` instead of corrupting the engine.
pub struct Jit {
    ctx: LLVMContextRef,
    /// `Some(module)` while we're building. `None` after the engine has
    /// consumed it (i.e. after the first lookup_fn).
    module: Option<LLVMModuleRef>,
    /// `Some(ee)` once we've handed the module to MCJIT. The engine owns
    /// the module from that point.
    engine: Option<LLVMExecutionEngineRef>,
    /// Pending host-process mappings (IR value, host address). We collect
    /// these during build because `LLVMAddGlobalMapping` requires an
    /// existing engine, but we don't materialize the engine until first
    /// lookup. On finalize we apply all queued mappings.
    pending_mappings: Vec<(LLVMValueRef, *mut c_void)>,
    /// Heap-allocated bag of error messages captured by our LLVM
    /// diagnostic handler. Boxed so its address is stable across moves —
    /// we hand the raw pointer to `LLVMContextSetDiagnosticHandler` once
    /// at construction. Drained by `take_errors()` before each public
    /// operation reports success.
    diag_errors: Box<Vec<String>>,
}

/// LLVM diagnostic handler: captures error-severity diagnostics into a
/// `Vec<String>` whose address LLVM stashed for us via the
/// `DiagnosticContext` argument to `LLVMContextSetDiagnosticHandler`.
///
/// We only capture severity == Error; warnings/remarks/notes are
/// dropped on the floor for now (we'd otherwise need to plumb them
/// up through the Result return type, which has no place for them).
unsafe extern "C" fn diag_handler(diag: LLVMDiagnosticInfoRef, ctx: *mut c_void) {
    if ctx.is_null() {
        return;
    }
    unsafe {
        let severity = LLVMGetDiagInfoSeverity(diag);
        if severity != LLVMDiagnosticSeverity::LLVMDSError {
            return;
        }
        let raw = LLVMGetDiagInfoDescription(diag);
        if raw.is_null() {
            return;
        }
        let msg = CStr::from_ptr(raw).to_string_lossy().into_owned();
        LLVMDisposeMessage(raw);
        let errors = &mut *(ctx as *mut Vec<String>);
        errors.push(msg);
    }
}

impl Jit {
    /// Build an empty JIT ready to accept assembly.
    pub fn new(module_name: &str) -> Result<Self, JitError> {
        init_x86_mcjit();
        unsafe {
            let ctx = LLVMContextCreate();
            assert!(!ctx.is_null(), "LLVMContextCreate returned null");

            let cname = c_string(module_name)?;
            let module = LLVMModuleCreateWithNameInContext(cname.as_ptr(), ctx);
            assert!(!module.is_null(), "LLVMModuleCreateWithNameInContext returned null");

            // Set the host triple so MC picks the right ABI/syntax.
            let triple = LLVMGetDefaultTargetTriple();
            if !triple.is_null() {
                LLVMSetTarget(module, triple);
                LLVMDisposeMessage(triple);
            }

            let mut diag_errors: Box<Vec<String>> = Box::new(Vec::new());
            // Install the diagnostic handler with our errors box as the
            // opaque context pointer. The box outlives the LLVMContext
            // (Drop runs in struct-field order: ctx is disposed before
            // the box is freed), so LLVM never sees a dangling pointer.
            let errors_ptr = (&mut *diag_errors as *mut Vec<String>) as *mut c_void;
            LLVMContextSetDiagnosticHandler(ctx, Some(diag_handler), errors_ptr);

            Ok(Jit {
                ctx,
                module: Some(module),
                engine: None,
                pending_mappings: Vec::new(),
                diag_errors,
            })
        }
    }

    /// Drain and return all error-severity diagnostics LLVM has reported
    /// since the last call. Callers should invoke this immediately after
    /// any LLVM operation that might produce errors and short-circuit
    /// with `JitError::Llvm` when the result is non-empty.
    fn take_errors(&mut self) -> Vec<String> {
        std::mem::take(&mut *self.diag_errors)
    }

    /// Append a chunk of assembly to the module.
    ///
    /// Must include `.globl <name>` for every symbol you intend to look up
    /// later. The asm body's `<name>:` is what actually emits the bytes;
    /// `declare_fn(name)` then advertises the symbol to IR so it survives
    /// MCJIT's link step and `LLVMGetFunctionAddress` can find it.
    ///
    /// You can call this many times before the first `lookup_fn`. Each
    /// call concatenates onto the module's asm blob.
    pub fn add_asm(&mut self, asm_text: &str) -> Result<(), JitError> {
        let module = self.require_module()?;
        unsafe {
            LLVMAppendModuleInlineAsm(
                module,
                asm_text.as_ptr() as *const _,
                asm_text.len(),
            );
        }
        Ok(())
    }

    /// Declare `name` as an extern `i64 (i64, i64, ...) → i64` function in
    /// IR, with `arg_count` parameters. The IR has no body; the
    /// implementation must come from somewhere — for our case, either
    /// from inline asm appended via [`add_asm`] (with a matching `.globl`)
    /// or from a host function plumbed in via [`define_extern_fn`].
    ///
    /// Returns the IR `LLVMValueRef` so callers can stash it for later
    /// global-mapping calls.
    pub fn declare_fn(&mut self, name: &str, arg_count: usize) -> Result<LLVMValueRef, JitError> {
        let module = self.require_module()?;
        unsafe {
            let i64_ty = LLVMInt64TypeInContext(self.ctx);
            let mut params: Vec<LLVMTypeRef> = vec![i64_ty; arg_count];
            let fn_ty = LLVMFunctionType(
                i64_ty,
                if params.is_empty() {
                    ptr::null_mut()
                } else {
                    params.as_mut_ptr()
                },
                arg_count as c_uint,
                0,
            );
            let cname = c_string(name)?;
            let fv = LLVMAddFunction(module, cname.as_ptr(), fn_ty);
            Ok(fv)
        }
    }

    /// Declare `name` as an extern function AND map it to a host process
    /// address. The JITed code can call `name` and it lands in the Rust
    /// function at `addr`.
    ///
    /// `addr` is the address of a Rust `extern "C"` function with
    /// `arg_count` `i64` arguments returning `i64`. On x86-64 Windows this
    /// is the Win64 ABI; on Linux it's SysV — both match `extern "C"`.
    ///
    /// We queue the mapping until the engine exists; it's applied on first
    /// `lookup_fn` before any code runs.
    pub fn define_extern_fn(
        &mut self,
        name: &str,
        arg_count: usize,
        addr: *mut c_void,
    ) -> Result<(), JitError> {
        let fv = self.declare_fn(name, arg_count)?;
        self.pending_mappings.push((fv, addr));
        Ok(())
    }

    /// Dump the module IR to stderr (for debugging).
    pub fn dump_ir(&self) {
        let Some(module) = self.module else {
            eprintln!("=== module already consumed by MCJIT — IR dump unavailable ===");
            return;
        };
        unsafe {
            let raw = LLVMPrintModuleToString(module);
            let s = CStr::from_ptr(raw).to_string_lossy().into_owned();
            LLVMDisposeMessage(raw);
            eprintln!("=== JIT module IR ===\n{s}===");
        }
    }

    /// Force compilation, look up `name`, return its address. First call
    /// finalizes the module — after this, no more `add_asm` / `declare_fn`
    /// allowed.
    pub fn lookup_addr(&mut self, name: &str) -> Result<u64, JitError> {
        let engine = self.finalize()?;
        // finalize() triggers MCJIT codegen, which is where MC's inline-asm
        // parser runs.  Errors from that path are captured into
        // diag_errors via our installed handler — surface them now
        // instead of returning a misleading NotFound.
        let errors = self.take_errors();
        if !errors.is_empty() {
            return Err(JitError::Llvm(errors.join("\n")));
        }
        unsafe {
            let cname = c_string(name)?;
            let addr = LLVMGetFunctionAddress(engine, cname.as_ptr());
            // Codegen of a single symbol can also fire diagnostics.
            let errors = self.take_errors();
            if !errors.is_empty() {
                return Err(JitError::Llvm(errors.join("\n")));
            }
            if addr == 0 {
                return Err(JitError::NotFound(name.to_string()));
            }
            Ok(addr)
        }
    }

    /// Look up a symbol and transmute to a function pointer.
    ///
    /// # Safety
    /// The caller asserts the JITed symbol matches the type `F`. On x86-64
    /// Windows that means an `extern "C" fn(...)` whose signature matches
    /// the asm's actual register usage.
    pub unsafe fn lookup_fn<F: Copy>(&mut self, name: &str) -> Result<F, JitError> {
        debug_assert_eq!(
            std::mem::size_of::<F>(),
            std::mem::size_of::<*const ()>(),
            "F must be a function pointer type"
        );
        let addr = self.lookup_addr(name)?;
        Ok(std::mem::transmute_copy::<u64, F>(&addr))
    }

    // ---- internals -----------------------------------------------------

    fn require_module(&self) -> Result<LLVMModuleRef, JitError> {
        self.module.ok_or(JitError::AlreadyFinalized)
    }

    /// Hand the module to MCJIT (consuming it), apply queued global
    /// mappings, return the engine. Idempotent.
    fn finalize(&mut self) -> Result<LLVMExecutionEngineRef, JitError> {
        if let Some(engine) = self.engine {
            return Ok(engine);
        }
        let module = self.module.take().ok_or(JitError::AlreadyFinalized)?;

        unsafe {
            // Default options + the OptLevel/EnableFastISel knobs LLVM picks.
            let mut opts: LLVMMCJITCompilerOptions = std::mem::zeroed();
            LLVMInitializeMCJITCompilerOptions(
                &mut opts,
                std::mem::size_of::<LLVMMCJITCompilerOptions>(),
            );
            // OptLevel 0 = no optimization; fine for our hand-written asm.
            // (LLVM optimizes IR, not the bodies of inline asm, so this
            // mostly affects helper functions we add later.)
            opts.OptLevel = 0;

            let mut engine: LLVMExecutionEngineRef = ptr::null_mut();
            let mut err_msg: *mut std::os::raw::c_char = ptr::null_mut();
            let rc = LLVMCreateMCJITCompilerForModule(
                &mut engine,
                module,
                &mut opts,
                std::mem::size_of::<LLVMMCJITCompilerOptions>(),
                &mut err_msg,
            );
            if rc != 0 || engine.is_null() {
                let msg = if err_msg.is_null() {
                    "LLVMCreateMCJITCompilerForModule failed with no message".to_string()
                } else {
                    let s = CStr::from_ptr(err_msg).to_string_lossy().into_owned();
                    LLVMDisposeMessage(err_msg);
                    s
                };
                // The module is NOT consumed on failure per the C API.
                // Put it back so cleanup is sane.
                self.module = Some(module);
                return Err(JitError::Llvm(msg));
            }

            // Apply queued mappings BEFORE the first GetFunctionAddress
            // triggers code emission.
            for (fv, addr) in self.pending_mappings.drain(..) {
                LLVMAddGlobalMapping(engine, fv, addr);
            }

            // Force MCJIT to materialize: compile, run MC over module-
            // level inline asm, link with RTDyld. Without this,
            // `LLVMGetFunctionAddress` returns 0 for IR symbols whose
            // bodies live entirely in `module asm` — LLVM doesn't see a
            // body in IR, so it doesn't bother triggering codegen on
            // demand.
            //
            // `LLVMRunStaticConstructors` is the standard hook for
            // this. It has no static ctors to run in our case, so the
            // useful side effect is the finalize itself.
            LLVMRunStaticConstructors(engine);

            self.engine = Some(engine);
            Ok(engine)
        }
    }
}

impl Drop for Jit {
    fn drop(&mut self) {
        unsafe {
            // If we got as far as an engine, dispose it — that also frees
            // the module it consumed.
            if let Some(engine) = self.engine.take() {
                LLVMDisposeExecutionEngine(engine);
                // The engine owns the module from here on; don't double-free.
            } else if let Some(module) = self.module.take() {
                // Never handed to MCJIT — we own it, must dispose.
                LLVMDisposeModule(module);
            }
            if !self.ctx.is_null() {
                LLVMContextDispose(self.ctx);
                self.ctx = ptr::null_mut();
            }
        }
    }
}

fn c_string(s: &str) -> Result<CString, JitError> {
    CString::new(s).map_err(|_| JitError::Nul(s.to_string()))
}
