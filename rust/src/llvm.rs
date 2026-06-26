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

use std::os::raw::{c_char, c_int, c_uint, c_void};

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

// ---- MCJIT memory manager callbacks --------------------------------------
//
// A `SimpleMCJITMemoryManager` lets us choose where MCJIT places emitted
// sections. We use it to allocate code/data from a near arena (within
// ±1.75 GB of the kernel) so runtime-JITed words are rel32-reachable —
// no far-segment jump trampoline needed.  Signatures from
// `llvm-c/ExecutionEngine.h`.
pub type LLVMMemoryManagerAllocateCodeSectionCallback = unsafe extern "C" fn(
    Opaque: *mut c_void,
    Size: usize,
    Alignment: c_uint,
    SectionID: c_uint,
    SectionName: *const c_char,
) -> *mut u8;

pub type LLVMMemoryManagerAllocateDataSectionCallback = unsafe extern "C" fn(
    Opaque: *mut c_void,
    Size: usize,
    Alignment: c_uint,
    SectionID: c_uint,
    SectionName: *const c_char,
    IsReadOnly: LLVMBool,
) -> *mut u8;

pub type LLVMMemoryManagerFinalizeMemoryCallback =
    unsafe extern "C" fn(Opaque: *mut c_void, ErrMsg: *mut *mut c_char) -> LLVMBool;

pub type LLVMMemoryManagerDestroyCallback = unsafe extern "C" fn(Opaque: *mut c_void);

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

// ---- Diagnostics (error capture) ----------------------------------------
//
// LLVM-MC reports inline-asm parse errors (and similar non-fatal
// diagnostics) through the LLVMContext's diagnostic handler. When no
// handler is installed, LLVM prints to stderr and — for severity
// `LLVMDSError` — typically calls `report_fatal_error`, which aborts
// the process. Installing our own handler lets the caller convert
// those errors into a returnable `Result` and recover.

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LLVMDiagnosticSeverity {
    LLVMDSError   = 0,
    LLVMDSWarning = 1,
    LLVMDSRemark  = 2,
    LLVMDSNote    = 3,
}

pub enum LLVMOpaqueDiagnosticInfo {}
pub type LLVMDiagnosticInfoRef = *mut LLVMOpaqueDiagnosticInfo;

pub type LLVMDiagnosticHandler =
    Option<unsafe extern "C" fn(diag: LLVMDiagnosticInfoRef, ctx: *mut std::ffi::c_void)>;

pub type LLVMFatalErrorHandler =
    Option<unsafe extern "C" fn(reason: *const c_char)>;

#[link(name = "LLVM-C")]
extern "C" {
    /// Install a per-context handler called whenever LLVM produces a
    /// diagnostic (errors, warnings, remarks, notes). The `ctx` pointer
    /// is opaque to LLVM and gets passed back to the handler verbatim.
    pub fn LLVMContextSetDiagnosticHandler(
        C: LLVMContextRef,
        Handler: LLVMDiagnosticHandler,
        DiagnosticContext: *mut std::ffi::c_void,
    );

    /// Returns the diagnostic's message as a newly-allocated string;
    /// must be freed by the caller with `LLVMDisposeMessage`.
    pub fn LLVMGetDiagInfoDescription(DI: LLVMDiagnosticInfoRef) -> *mut c_char;

    /// Returns the severity (error / warning / remark / note).
    pub fn LLVMGetDiagInfoSeverity(DI: LLVMDiagnosticInfoRef) -> LLVMDiagnosticSeverity;

    /// Install a process-wide handler for fatal errors. By default LLVM
    /// calls `abort()` after printing to stderr; with a handler installed
    /// it calls our handler instead. The handler is NOT expected to
    /// return — LLVM considers the program unrecoverable past this point
    /// — so use with care.
    pub fn LLVMInstallFatalErrorHandler(Handler: LLVMFatalErrorHandler);

    /// Restore the default fatal-error behaviour (print + abort).
    pub fn LLVMResetFatalErrorHandler();
}

// ---- Target init (X86 + AArch64) ----------------------------------------
//
// The `LLVMInitializeAllTargets*` umbrella functions are inline macros in the
// C headers, not exported symbols, so each backend must be registered by name.
// Standard LLVM distributions (Homebrew on macOS, the official Windows package,
// distro `libLLVM`) build *all* targets, so both X86 and AArch64 init symbols
// are present in the single `LLVM-C` shared object on every platform we target.

#[link(name = "LLVM-C")]
extern "C" {
    pub fn LLVMInitializeX86TargetInfo();
    pub fn LLVMInitializeX86Target();
    pub fn LLVMInitializeX86TargetMC();
    pub fn LLVMInitializeX86AsmParser();
    pub fn LLVMInitializeX86AsmPrinter();

    pub fn LLVMInitializeAArch64TargetInfo();
    pub fn LLVMInitializeAArch64Target();
    pub fn LLVMInitializeAArch64TargetMC();
    pub fn LLVMInitializeAArch64AsmParser();
    pub fn LLVMInitializeAArch64AsmPrinter();
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

    /// Create a memory manager whose section-allocation decisions are
    /// delegated to the supplied callbacks. We use this to place runtime
    /// code in a near arena. Ownership passes to the engine built from the
    /// options that reference it; the engine calls `Destroy` on dispose.
    pub fn LLVMCreateSimpleMCJITMemoryManager(
        Opaque: *mut c_void,
        AllocateCodeSection: LLVMMemoryManagerAllocateCodeSectionCallback,
        AllocateDataSection: LLVMMemoryManagerAllocateDataSectionCallback,
        FinalizeMemory: LLVMMemoryManagerFinalizeMemoryCallback,
        Destroy: LLVMMemoryManagerDestroyCallback,
    ) -> LLVMMCJITMemoryManagerRef;

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

// ---- TargetMachine + object emission (the rasm differential oracle) ------
//
// The `LlvmMcEncoder` oracle (see `src/oracle.rs`) needs LLVM-MC to emit a
// *relocatable object* — `.text` bytes with zeroed reloc placeholders plus a
// relocation table — so it produces the same shape as rasm's `EncodedModule`
// and the two can be diffed byte-for-byte. That object comes from a
// `TargetMachine.EmitToMemoryBuffer(ObjectFile)`. Signatures transcribed from
// `llvm-c/TargetMachine.h`, `llvm-c/Target.h`, and `llvm-c/Core.h` (LLVM 22).

pub enum LLVMTarget {}
pub type LLVMTargetRef = *mut LLVMTarget;

pub enum LLVMOpaqueTargetMachine {}
pub type LLVMTargetMachineRef = *mut LLVMOpaqueTargetMachine;

pub enum LLVMOpaqueMemoryBuffer {}
pub type LLVMMemoryBufferRef = *mut LLVMOpaqueMemoryBuffer;

#[repr(C)]
#[derive(Clone, Copy)]
pub enum LLVMCodeGenOptLevel {
    None = 0,
    Less,
    Default,
    Aggressive,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub enum LLVMRelocMode {
    Default = 0,
    Static,
    PIC,
    DynamicNoPic,
    ROPI,
    RWPI,
    ROPI_RWPI,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub enum LLVMCodeModel {
    Default = 0,
    JITDefault,
    Tiny,
    Small,
    Kernel,
    Medium,
    Large,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub enum LLVMCodeGenFileType {
    AssemblyFile = 0,
    ObjectFile,
}

#[link(name = "LLVM-C")]
extern "C" {
    /// Look up the registered `LLVMTarget` for a triple. Returns nonzero on
    /// failure with a message in `ErrorMessage` (free via `LLVMDisposeMessage`).
    pub fn LLVMGetTargetFromTriple(
        Triple: *const c_char,
        T: *mut LLVMTargetRef,
        ErrorMessage: *mut *mut c_char,
    ) -> LLVMBool;

    /// Construct a `TargetMachine`. `CPU`/`Features` may be empty C strings —
    /// they don't affect assembling already-chosen instructions (inline asm),
    /// only IR codegen. Dispose with `LLVMDisposeTargetMachine`.
    pub fn LLVMCreateTargetMachine(
        T: LLVMTargetRef,
        Triple: *const c_char,
        CPU: *const c_char,
        Features: *const c_char,
        Level: LLVMCodeGenOptLevel,
        Reloc: LLVMRelocMode,
        CodeModel: LLVMCodeModel,
    ) -> LLVMTargetMachineRef;

    pub fn LLVMDisposeTargetMachine(T: LLVMTargetMachineRef);

    /// Run codegen for `M` and write the result (`ObjectFile` for us) into a
    /// freshly allocated `MemoryBuffer`. Does **not** consume `M`. Returns
    /// nonzero on failure with a message in `ErrorMessage`.
    pub fn LLVMTargetMachineEmitToMemoryBuffer(
        T: LLVMTargetMachineRef,
        M: LLVMModuleRef,
        codegen: LLVMCodeGenFileType,
        ErrorMessage: *mut *mut c_char,
        OutMemBuf: *mut LLVMMemoryBufferRef,
    ) -> LLVMBool;

    pub fn LLVMGetBufferStart(MemBuf: LLVMMemoryBufferRef) -> *const c_char;
    pub fn LLVMGetBufferSize(MemBuf: LLVMMemoryBufferRef) -> usize;
    pub fn LLVMDisposeMemoryBuffer(MemBuf: LLVMMemoryBufferRef);
}

// ---- Convenience ---------------------------------------------------------

/// Initialize the X86 + AArch64 backends AND link in MCJIT. Idempotent.
///
/// Both backends are registered so the oracle ([`crate::oracle`]) can assemble
/// for either triple regardless of host, and so MCJIT codegen ([`crate::jit`])
/// targets the host (x86-64 on Windows/Linux, AArch64 on Apple Silicon).
pub fn init_targets_and_mcjit() {
    use std::sync::Once;
    static ONCE: Once = Once::new();
    ONCE.call_once(|| unsafe {
        LLVMInitializeX86TargetInfo();
        LLVMInitializeX86Target();
        LLVMInitializeX86TargetMC();
        LLVMInitializeX86AsmParser();
        LLVMInitializeX86AsmPrinter();

        LLVMInitializeAArch64TargetInfo();
        LLVMInitializeAArch64Target();
        LLVMInitializeAArch64TargetMC();
        LLVMInitializeAArch64AsmParser();
        LLVMInitializeAArch64AsmPrinter();

        LLVMLinkInMCJIT();
    });
}

/// Back-compat alias — callers predating multi-arch init. Registers both
/// backends (see [`init_targets_and_mcjit`]).
pub fn init_x86_mcjit() {
    init_targets_and_mcjit();
}
