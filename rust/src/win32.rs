//! Runtime helper that pairs `Assembler::externs()` with real Win32
//! function addresses via `LoadLibraryW` / `GetProcAddress`, and
//! registers each mapping with the JIT.
//!
//! Usage:
//!
//! ```ignore
//! let mut asm = Assembler::new();
//! let asm_text = asm.assemble("kernel.masm", source)?;
//!
//! let mut jit = Jit::new("kernel")?;
//! wfasm::win32::bind_externs(&asm, &mut jit, |name| {
//!     // Host's own Rust runtime functions, keyed by `@extern NAME(N)`
//!     // entries with no DLL string.
//!     match name {
//!         "rt_emit" => Some(rt_emit as *mut c_void),
//!         _        => None,
//!     }
//! })?;
//! jit.add_asm(&asm_text)?;
//! ```
//!
//! What `bind_externs` does:
//!
//! 1. For each `@extern "DLL.dll" NAME(N)` entry: `LoadLibraryW` the
//!    DLL once (cached across externs that share a DLL), call
//!    `GetProcAddress(NAME)`, register the resulting pointer with the
//!    JIT via `Jit::define_extern_fn`.
//! 2. For each `@extern NAME(N)` entry (no DLL): call the user-
//!    supplied resolver. If it returns `Some(ptr)`, register; if
//!    `None`, error out clearly.
//!
//! Loaded DLLs are kept loaded for the process lifetime. No
//! `FreeLibrary` — JIT'd code may hold pointers indefinitely, and the
//! cost of leaking 12 HMODULEs is trivial vs. the risk of unloading
//! while a JIT'd function pointer is still live.
//!
//! Platform: Windows only. Compile-flagged out elsewhere.

#![cfg(windows)]
#![allow(non_snake_case, non_camel_case_types)]

use std::collections::HashMap;
use std::ffi::{c_void, CString};

use crate::asm::Assembler;
use crate::backend::Loader;

// ─── Win32 FFI (minimal — three functions) ──────────────────────────

#[repr(C)]
struct HMODULE(*mut c_void);
unsafe impl Send for HMODULE {}
unsafe impl Sync for HMODULE {}

#[link(name = "kernel32")]
extern "system" {
    fn LoadLibraryW(lpLibFileName: *const u16) -> *mut c_void;
    fn GetProcAddress(hModule: *mut c_void, lpProcName: *const i8) -> *mut c_void;
    fn GetLastError() -> u32;
}

// ─── public API ──────────────────────────────────────────────────────

#[derive(Debug)]
pub enum BindError {
    LoadLibrary { dll: String, win_error: u32 },
    GetProcAddress { dll: String, name: String, win_error: u32 },
    UnresolvedHostExtern { name: String },
    Loader(String),
    Cstring(String),
}

impl std::fmt::Display for BindError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BindError::LoadLibrary { dll, win_error } => {
                write!(f, "LoadLibraryW(\"{dll}\") failed (GetLastError = {win_error})")
            }
            BindError::GetProcAddress { dll, name, win_error } => write!(
                f,
                "GetProcAddress(\"{name}\") in \"{dll}\" failed (GetLastError = {win_error})"
            ),
            BindError::UnresolvedHostExtern { name } => write!(
                f,
                "@extern `{name}` has no DLL string and the resolver returned None"
            ),
            BindError::Loader(e) => write!(f, "loader bind failed: {e}"),
            BindError::Cstring(s) => write!(f, "name `{s}` contains an interior NUL"),
        }
    }
}
impl std::error::Error for BindError {}

/// Outcome of a `bind_externs` call.
#[derive(Debug, Default)]
pub struct BindReport {
    /// Number of externs successfully registered with the JIT.
    pub bound: usize,
    /// `(name, dll, win_error)` for DLL-backed externs whose
    /// `GetProcAddress` returned null (and so were skipped).
    /// Some DLLs export different sets per Windows version; the
    /// generator emits a name → all calls of unresolved names will
    /// surface later when MCJIT tries to link them.
    pub missing_proc: Vec<(String, String, u32)>,
}

/// Bind every `@extern` declaration in `asm` to a real function pointer
/// and register it with `jit`. See module docs for the contract.
///
/// `host_resolver` is consulted for externs with no DLL string (the
/// host's own Rust runtime functions). It's a closure
/// `&str -> Option<*mut c_void>`.
///
/// **Tolerance model:**
///
/// * `LoadLibraryW` failure is fatal — if the host has an
///   `@extern "USER32.dll" …` and USER32 can't load, nothing else
///   from that DLL will work either. Bail out.
/// * `GetProcAddress` returning null is NOT fatal — the generated
///   bindings include functions some Windows versions don't export.
///   Such names get logged in `BindReport.missing_proc` and skipped.
///   If JITed code actually `call`s a missing function, RTDyld
///   surfaces the unresolved symbol when the module materializes.
/// * `host_resolver` returning `None` for a host-side extern IS
///   fatal — the host should know exactly which Rust functions it
///   provides.
pub fn bind_externs<F>(
    asm: &Assembler,
    loader: &mut dyn Loader,
    mut host_resolver: F,
) -> Result<BindReport, BindError>
where
    F: FnMut(&str) -> Option<*mut c_void>,
{
    // Snapshot the externs first — registering with the JIT borrows
    // `jit` mutably, and we don't want that to fight with iterating
    // through the assembler's state.
    let externs: Vec<(String, usize, Option<String>)> = asm
        .externs()
        .map(|(n, d)| (n.to_string(), d.arg_count, d.dll.clone()))
        .collect();

    // Cache loaded DLLs across the registration loop.
    let mut loaded: HashMap<String, *mut c_void> = HashMap::new();
    let mut report = BindReport::default();

    for (name, arg_count, dll_opt) in externs {
        let addr_opt = match dll_opt.as_deref() {
            Some(dll) => resolve_via_dll_opt(dll, &name, &mut loaded, &mut report)?,
            None => Some(host_resolver(&name).ok_or_else(|| {
                BindError::UnresolvedHostExtern { name: name.clone() }
            })?),
        };
        if let Some(addr) = addr_opt {
            loader
                .define_extern_fn(&name, arg_count, addr)
                .map_err(|e| BindError::Loader(e.to_string()))?;
            report.bound += 1;
        }
    }
    Ok(report)
}

/// Look up `name` in `dll`, loading the DLL on first use. Returns
/// `Ok(Some(addr))` on success, `Ok(None)` when the DLL loaded but the
/// proc isn't exported (logged in `report.missing_proc`), or
/// `Err(LoadLibrary)` on DLL-load failure.
fn resolve_via_dll_opt(
    dll: &str,
    name: &str,
    loaded: &mut HashMap<String, *mut c_void>,
    report: &mut BindReport,
) -> Result<Option<*mut c_void>, BindError> {
    let hmod = if let Some(h) = loaded.get(dll) {
        *h
    } else {
        let wide: Vec<u16> = dll.encode_utf16().chain(std::iter::once(0)).collect();
        let h = unsafe { LoadLibraryW(wide.as_ptr()) };
        if h.is_null() {
            return Err(BindError::LoadLibrary {
                dll: dll.to_string(),
                win_error: unsafe { GetLastError() },
            });
        }
        loaded.insert(dll.to_string(), h);
        h
    };
    let cname = CString::new(name).map_err(|_| BindError::Cstring(name.to_string()))?;
    let addr = unsafe { GetProcAddress(hmod, cname.as_ptr()) };
    if addr.is_null() {
        report.missing_proc.push((
            name.to_string(),
            dll.to_string(),
            unsafe { GetLastError() },
        ));
        return Ok(None);
    }
    Ok(Some(addr))
}
