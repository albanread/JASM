//! Hand-written FFI bindings to LLVM-C.dll.
//!
//! Scope: enough to drive **MCJIT** with module-level inline assembly plus
//! IR `declare`s. That pattern is the one battle-tested in NewBCPL and
//! NewFB — the asm body provides the bytes, the IR `declare` makes the
//! symbol resolvable, and MCJIT/RTDyld matches them at link time.
//!
//! We deliberately do not use ORC LLJIT here: ORC's lazy materialization
//! inventories which symbols a module *provides* from IR alone, before
//! MC runs on the inline asm — so asm-defined symbols are invisible to
//! ORC and lookups fail. MCJIT compiles eagerly and reaps both IR and
//! asm symbols out of the final object in one pass, which is exactly
//! what this project needs.
//!
//! C signatures are transcribed from the public LLVM 22 headers
//! (Core.h, Target.h, ExecutionEngine.h, Error.h) and confirmed against
//! the export table of the shipped `LLVM-C.dll`.
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(dead_code)]

use std::os::raw::{c_char, c_int, c_uint};

// ---- Opaque handle types --------------------------------------------------

pub enum LLVMOpaqueContext {}
pub enum LLVMOpaqueModule {}
pub enum LLVMOpaqueType {}
pub enum LLVMOpaqueValue {}
pub enum LLVMOpaqueExecutionEngine {}
pub enum LLVMOpaqueMCJITMemoryManager {}

pub type LLVMContextRef = *mut LLVMOpaqueContext;
pub type LLVMModuleRef = *mut LLVMOpaqueModule;
pub type LLVMTypeRef = *mut LLVMOpaqueType;
pub type LLVMValueRef = *mut LLVMOpaqueValue;
pub type LLVMExecutionEngineRef = *mut LLVMOpaqueExecutionEngine;
pub type LLVMMCJITMemoryManagerRef = *mut LLVMOpaqueMCJITMemoryManager;

pub type LLVMBool = c_int;

/// Mirror of `LLVMMCJITCompilerOptions` from `llvm-c/ExecutionEngine.h`.
///
/// Field order, types, and packing must match LLVM exactly. Always pass the
/// `size_of` so LLVM can detect a version mismatch and zero-fill any
/// trailing fields you don't know about.
#[repr(C)]
pub struct LLVMMCJITCompilerOptions {
    pub OptLevel: c_uint,
    pub CodeModel: c_int, // LLVMCodeModel enum
    pub NoFramePointerElim: LLVMBool,
    pub EnableFastISel: LLVMBool,
    pub MCJMM: LLVMMCJITMemoryManagerRef,
}

// ---- Core ----------------------------------------------------------------

#[link(name = "LLVM-C")]
extern "C" {
    pub fn LLVMContextCreate() -> LLVMContextRef;
    pub fn LLVMContextDispose(C: LLVMContextRef);

    pub fn LLVMModuleCreateWithNameInContext(
        ModuleID: *const c_char,
        C: LLVMContextRef,
    ) -> LLVMModuleRef;
    pub fn LLVMDisposeModule(M: LLVMModuleRef);
    pub fn LLVMSetTarget(M: LLVMModuleRef, Triple: *const c_char);
    pub fn LLVMGetTarget(M: LLVMModuleRef) -> *const c_char;
    pub fn LLVMGetDefaultTargetTriple() -> *mut c_char;
    pub fn LLVMAppendModuleInlineAsm(M: LLVMModuleRef, Asm: *const c_char, Len: usize);
    pub fn LLVMPrintModuleToString(M: LLVMModuleRef) -> *mut c_char;
    pub fn LLVMDisposeMessage(Msg: *mut c_char);

    pub fn LLVMAddFunction(
        M: LLVMModuleRef,
        Name: *const c_char,
        FunctionTy: LLVMTypeRef,
    ) -> LLVMValueRef;
    pub fn LLVMFunctionType(
        ReturnType: LLVMTypeRef,
        ParamTypes: *mut LLVMTypeRef,
        ParamCount: c_uint,
        IsVarArg: LLVMBool,
    ) -> LLVMTypeRef;
    pub fn LLVMVoidTypeInContext(C: LLVMContextRef) -> LLVMTypeRef;
    pub fn LLVMInt32TypeInContext(C: LLVMContextRef) -> LLVMTypeRef;
    pub fn LLVMInt64TypeInContext(C: LLVMContextRef) -> LLVMTypeRef;
    pub fn LLVMPointerTypeInContext(C: LLVMContextRef, AddressSpace: c_uint) -> LLVMTypeRef;
}

// ---- Target init (X86 only) ---------------------------------------------

#[link(name = "LLVM-C")]
extern "C" {
    pub fn LLVMInitializeX86TargetInfo();
    pub fn LLVMInitializeX86Target();
    pub fn LLVMInitializeX86TargetMC();
    pub fn LLVMInitializeX86AsmParser();
    pub fn LLVMInitializeX86AsmPrinter();
}

// ---- Execution engine (MCJIT) -------------------------------------------

#[link(name = "LLVM-C")]
extern "C" {
    /// One-shot symbol that pulls MCJIT into the link. Calling
    /// `CreateMCJIT*` before this returns "Interpreter has not been linked
    /// in." Safe to call any number of times.
    pub fn LLVMLinkInMCJIT();

    /// Zero-fill `Options` and set library defaults. Pass the actual size
    /// of the struct as known by the caller; LLVM uses this for forward-
    /// compatibility when fields get added.
    pub fn LLVMInitializeMCJITCompilerOptions(
        Options: *mut LLVMMCJITCompilerOptions,
        SizeOfOptions: usize,
    );

    /// Build an MCJIT execution engine that compiles `Module`. **Consumes**
    /// `Module` — do not dispose or reuse it after a successful call. On
    /// failure, the module is still owned by the caller and `OutError`
    /// holds a message that must be freed with `LLVMDisposeMessage`.
    pub fn LLVMCreateMCJITCompilerForModule(
        OutJIT: *mut LLVMExecutionEngineRef,
        M: LLVMModuleRef,
        Options: *mut LLVMMCJITCompilerOptions,
        SizeOfOptions: usize,
        OutError: *mut *mut c_char,
    ) -> LLVMBool;

    pub fn LLVMDisposeExecutionEngine(EE: LLVMExecutionEngineRef);

    /// Force codegen + RTDyld linking on every module added to `EE`.
    /// This is the call that makes asm-emitted symbols visible — for
    /// declare-only IR functions (whose bodies live in module-level
    /// inline asm), `LLVMGetFunctionAddress` won't trigger codegen
    /// on its own because LLVM doesn't see a body in IR.
    /// `LLVMRunStaticConstructors` finalizes the module unconditionally.
    pub fn LLVMRunStaticConstructors(EE: LLVMExecutionEngineRef);
    pub fn LLVMRunStaticDestructors(EE: LLVMExecutionEngineRef);

    /// Trigger codegen for `Name` (if not yet emitted), return its address.
    /// First call on any function in a module forces the whole module to
    /// compile + finalize.
    pub fn LLVMGetFunctionAddress(EE: LLVMExecutionEngineRef, Name: *const c_char) -> u64;

    /// Get the address of a global value (function or global variable)
    /// by name. Returns 0 if not found.
    pub fn LLVMGetGlobalValueAddress(EE: LLVMExecutionEngineRef, Name: *const c_char) -> u64;

    /// Tell MCJIT that the named global (`Global` is the IR `LLVMValueRef`
    /// from `LLVMAddFunction` / `LLVMAddGlobal`) corresponds to host
    /// process memory at `Addr`. Used to plumb Rust runtime functions in.
    pub fn LLVMAddGlobalMapping(
        EE: LLVMExecutionEngineRef,
        Global: LLVMValueRef,
        Addr: *mut std::ffi::c_void,
    );
}

// ---- Convenience ---------------------------------------------------------

/// Initialize the X86 backend AND link in MCJIT. Idempotent.
pub fn init_x86_mcjit() {
    use std::sync::Once;
    static ONCE: Once = Once::new();
    ONCE.call_once(|| unsafe {
        LLVMInitializeX86TargetInfo();
        LLVMInitializeX86Target();
        LLVMInitializeX86TargetMC();
        LLVMInitializeX86AsmParser();
        LLVMInitializeX86AsmPrinter();
        LLVMLinkInMCJIT();
    });
}
