//! Native loader (`NativeJit`) — the Rasm replacement for MCJIT's place +
//! relocate + protect job. No LLVM.
//!
//! Sprint 1 brings this up driven directly by [`EncodedModule`]s (from the
//! golden/encoder), proving the genuinely-new loader risk — near-host
//! executable allocation, the 3-kind relocator, W^X protection, and executing
//! relocated code that calls back into the Rust host — independent of any
//! encoder. The `Loader` trait impl (driven by `add_asm` text) lands once an
//! encoder exists.
//!
//! Memory model: one code region `VirtualAlloc2`'d within ±~1.75 GB of an
//! anchor address (so `call rel32` reaches host `rt_*` externs), mapped
//! **RW** while we copy + relocate, then flipped to **RX** (never RWX) on
//! [`NativeJit::finalize`].

#![cfg(windows)]

use std::collections::HashMap;
use std::ffi::c_void;
use std::ptr;

use anyhow::{bail, Context, Result};

use crate::backend::{EncodedModule, Reloc, RelocKind};

// ── Win32 FFI (VirtualAlloc2 windowed alloc + VirtualProtect) ───────────────

#[repr(C)]
struct MemAddressRequirements {
    lowest_starting_address: *mut c_void,
    highest_ending_address: *mut c_void,
    alignment: usize,
}
const MEM_EXTENDED_PARAMETER_ADDRESS_REQUIREMENTS: u64 = 1;

#[repr(C)]
struct MemExtendedParameter {
    type_and_reserved: u64,
    pointer: *mut c_void,
}

const MEM_RESERVE: u32 = 0x0000_2000;
const MEM_COMMIT: u32 = 0x0000_1000;
const MEM_RELEASE: u32 = 0x0000_8000;
const PAGE_READWRITE: u32 = 0x04;
const PAGE_EXECUTE_READ: u32 = 0x20;

type VirtualAlloc2Fn = unsafe extern "system" fn(
    *mut c_void,
    *mut c_void,
    usize,
    u32,
    u32,
    *mut MemExtendedParameter,
    u32,
) -> *mut c_void;

#[link(name = "kernel32")]
extern "system" {
    fn LoadLibraryA(name: *const i8) -> *mut c_void;
    fn GetProcAddress(module: *mut c_void, name: *const i8) -> *mut c_void;
    fn GetLastError() -> u32;
    fn VirtualFree(addr: *mut c_void, size: usize, free_type: u32) -> i32;
    fn VirtualProtect(addr: *mut c_void, size: usize, new: u32, old: *mut u32) -> i32;
}

fn virtual_alloc2() -> Result<VirtualAlloc2Fn> {
    unsafe {
        let lib = LoadLibraryA(b"kernelbase.dll\0".as_ptr() as *const i8);
        let lib = if lib.is_null() {
            LoadLibraryA(b"kernel32.dll\0".as_ptr() as *const i8)
        } else {
            lib
        };
        if lib.is_null() {
            bail!("LoadLibraryA(kernelbase/kernel32) failed");
        }
        let p = GetProcAddress(lib, b"VirtualAlloc2\0".as_ptr() as *const i8);
        if p.is_null() {
            bail!("GetProcAddress(VirtualAlloc2) failed — needs Windows 10+");
        }
        Ok(std::mem::transmute::<*mut c_void, VirtualAlloc2Fn>(p))
    }
}

/// Allocate `size` bytes of RW memory anywhere the OS picks (a roomy spot, not
/// crowded against the host image). Far externs are reached via stubs, so the
/// code region has no rel32 constraint to the host.
fn alloc_anywhere(size: usize) -> Result<*mut u8> {
    let va2 = virtual_alloc2().context("locate VirtualAlloc2")?;
    let base = unsafe {
        va2(ptr::null_mut(), ptr::null_mut(), size, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE, ptr::null_mut(), 0)
    };
    if base.is_null() {
        bail!("VirtualAlloc2 (anywhere) returned null (GetLastError = {})", unsafe { GetLastError() });
    }
    Ok(base as *mut u8)
}

/// Allocate `size` bytes of RW memory within ±window of `anchor` so emitted
/// code reaches host externs by `call rel32`.
#[allow(dead_code)]
fn alloc_near(anchor: u64, size: usize) -> Result<*mut u8> {
    let va2 = virtual_alloc2().context("locate VirtualAlloc2")?;
    const GRANULARITY: u64 = 0x10000;
    const WINDOW: u64 = 0x7000_0000; // ~1.75 GB, comfortably inside rel32 (±2 GB)
    let low = (anchor.saturating_sub(WINDOW) + GRANULARITY - 1) & !(GRANULARITY - 1);
    let low = low.max(GRANULARITY);
    let high = (anchor.saturating_add(WINDOW) & !(GRANULARITY - 1)).saturating_sub(1);

    let mut req = MemAddressRequirements {
        lowest_starting_address: low as *mut c_void,
        highest_ending_address: high as *mut c_void,
        alignment: 0,
    };
    let mut param = MemExtendedParameter {
        type_and_reserved: MEM_EXTENDED_PARAMETER_ADDRESS_REQUIREMENTS,
        pointer: &mut req as *mut _ as *mut c_void,
    };
    let base = unsafe {
        va2(
            ptr::null_mut(),
            ptr::null_mut(),
            size,
            MEM_RESERVE | MEM_COMMIT,
            PAGE_READWRITE,
            &mut param,
            1,
        )
    };
    if base.is_null() {
        bail!(
            "VirtualAlloc2 near {anchor:#018x} returned null (GetLastError = {})",
            unsafe { GetLastError() }
        );
    }
    Ok(base as *mut u8)
}

// ── NativeJit ───────────────────────────────────────────────────────────────

/// A pending placed module: its base offset in the region and the source.
struct Placed {
    base: u64,
    relocs: Vec<Reloc>,
}

pub struct NativeJit {
    region: *mut u8,
    cap: usize,
    used: usize,
    /// Defined symbol name → absolute runtime address.
    symbols: HashMap<String, u64>,
    /// Host extern name → absolute address.
    externs: HashMap<String, u64>,
    placed: Vec<Placed>,
    finalized: bool,
    /// Accumulated assembly text (Loader builder path); assembled by
    /// `RasmEncoder` on first lookup. Empty in the low-level module path.
    pending_text: String,
}

impl NativeJit {
    /// Reserve `cap` bytes of RW code space within rel32 of `anchor`.
    pub fn new_near(anchor: u64, cap: usize) -> Result<Self> {
        let region = alloc_near(anchor, cap)?;
        Ok(NativeJit {
            region,
            cap,
            used: 0,
            symbols: HashMap::new(),
            externs: HashMap::new(),
            placed: Vec::new(),
            finalized: false,
            pending_text: String::new(),
        })
    }

    /// A builder-mode loader: accumulate assembly text + externs, then assemble
    /// + place on first `lookup_addr`. No region is reserved until then (the
    /// anchor is derived from the bound externs, so emitted code is rel32-near
    /// the host `rt_*` functions). This is the [`Loader`](crate::backend::Loader)
    /// path the kernel boots through.
    pub fn new() -> Self {
        NativeJit {
            region: ptr::null_mut(),
            cap: 0,
            used: 0,
            symbols: HashMap::new(),
            externs: HashMap::new(),
            placed: Vec::new(),
            finalized: false,
            pending_text: String::new(),
        }
    }

    /// Builder path: assemble the accumulated text with `RasmEncoder`, reserve
    /// a code region near the externs, place + relocate + RX-protect. Idempotent.
    fn build_if_needed(&mut self) -> Result<()> {
        if self.finalized {
            return Ok(());
        }
        if self.region.is_null() {
            let module = crate::rasm::assemble(&self.pending_text)
                .context("RasmEncoder: assemble kernel text")?;
            // Reserve space for the code + far-call stubs (12 bytes/extern) + slack.
            // Placed anywhere roomy; ALL externs (host rt_* and DLL imports) are
            // reached via stubs, so there is no rel32 constraint to the host.
            let cap = (module.code.len() + module.externs.len() * 16 + 4096 + 0xFFF) & !0xFFF;
            self.region = alloc_anywhere(cap)?;
            self.cap = cap;
            self.load_module(&module)?;
        }
        self.finalize()
    }

    /// Bind a host extern (a Rust `extern "C"` function) by name.
    pub fn define_extern(&mut self, name: &str, addr: u64) {
        self.externs.insert(name.to_string(), addr);
    }

    /// Copy `m`'s code into the region, record its symbols at their final
    /// addresses, and queue its relocations. Returns the module base address.
    pub fn load_module(&mut self, m: &EncodedModule) -> Result<u64> {
        if self.finalized {
            bail!("NativeJit already finalized");
        }
        // 16-byte align each module for tidy disassembly / future unwind.
        let start = (self.used + 15) & !15;
        let end = start + m.code.len();
        if end > self.cap {
            bail!("code region exhausted: need {end} bytes, cap {}", self.cap);
        }
        let base = self.region as u64 + start as u64;
        unsafe {
            ptr::copy_nonoverlapping(m.code.as_ptr(), self.region.add(start), m.code.len());
        }
        for (name, off) in &m.symbols {
            self.symbols.insert(name.clone(), base + *off as u64);
        }
        self.placed.push(Placed { base, relocs: m.relocs.clone() });
        self.used = end;
        Ok(base)
    }

    /// Resolve `name` to a defined symbol or a bound extern.
    fn resolve(&self, name: &str) -> Option<u64> {
        self.symbols.get(name).copied().or_else(|| self.externs.get(name).copied())
    }

    /// Apply all relocations, then flip the region to RX. Idempotent.
    pub fn finalize(&mut self) -> Result<()> {
        if self.finalized {
            return Ok(());
        }
        // Collect first to avoid borrowing self while patching.
        let mut patches: Vec<(u64, Reloc)> = Vec::new();
        for p in &self.placed {
            for r in &p.relocs {
                patches.push((p.base, r.clone()));
            }
        }
        // Apply relocations. A far branch target (e.g. a kernel32 DLL import
        // >2GB from our code) can't be reached by `call rel32`, so route it
        // through a 12-byte `movabs rax,target ; jmp rax` stub appended after
        // the code (one per distinct target) — the same trick RTDyld uses.
        let mut stub_off = self.used;
        let mut stubs: HashMap<u64, u64> = HashMap::new();
        for (base, r) in patches {
            let target = self
                .resolve(&r.target)
                .with_context(|| format!("unresolved reloc target `{}`", r.target))?;
            let field = base + r.at as u64;
            match r.kind {
                RelocKind::BranchRel32 | RelocKind::RipRel32 => {
                    let mut rel = (target as i64 + r.addend) - (field as i64 + 4);
                    if i32::try_from(rel).is_err() {
                        if r.kind == RelocKind::RipRel32 {
                            bail!("RIP-rel disp32 out of range for `{}` (no stub possible)", r.target);
                        }
                        let stub = match stubs.get(&target) {
                            Some(&s) => s,
                            None => {
                                if stub_off + 12 > self.cap {
                                    bail!("far-call stub region exhausted");
                                }
                                let s = self.region as u64 + stub_off as u64;
                                unsafe {
                                    let p = self.region.add(stub_off);
                                    *p = 0x48; // REX.W
                                    *p.add(1) = 0xB8; // movabs rax, imm64
                                    ptr::copy_nonoverlapping(target.to_le_bytes().as_ptr(), p.add(2), 8);
                                    *p.add(10) = 0xFF; // jmp rax
                                    *p.add(11) = 0xE0;
                                }
                                stub_off += 12;
                                stubs.insert(target, s);
                                s
                            }
                        };
                        rel = stub as i64 - (field as i64 + 4);
                    }
                    let rel32 = i32::try_from(rel)
                        .map_err(|_| anyhow::anyhow!("rel32 still out of range for `{}` via stub", r.target))?;
                    unsafe { ptr::copy_nonoverlapping(rel32.to_le_bytes().as_ptr(), field as *mut u8, 4); }
                }
                RelocKind::Abs64 => {
                    let val = (target as i64 + r.addend) as u64;
                    unsafe { ptr::copy_nonoverlapping(val.to_le_bytes().as_ptr(), field as *mut u8, 8); }
                }
            }
        }
        self.used = stub_off;

        let mut old = 0u32;
        let ok = unsafe {
            VirtualProtect(self.region as *mut c_void, self.cap, PAGE_EXECUTE_READ, &mut old)
        };
        if ok == 0 {
            bail!("VirtualProtect RX failed (GetLastError = {})", unsafe { GetLastError() });
        }
        self.finalized = true;
        Ok(())
    }

    /// Runtime address of a defined symbol (after [`finalize`]).
    pub fn lookup(&self, name: &str) -> Option<u64> {
        self.symbols.get(name).copied()
    }

    /// Convenience: a CString-free symbol existence check for tests.
    pub fn has_symbol(&self, name: &str) -> bool {
        self.symbols.contains_key(name)
    }
}

impl Default for NativeJit {
    fn default() -> Self {
        Self::new()
    }
}

/// The native [`Loader`](crate::backend::Loader) — assembles accumulated text
/// with `RasmEncoder` and places it, replacing MCJIT. `declare_fn` is a no-op
/// (symbols come from the encoder's symbol table).
impl crate::backend::Loader for NativeJit {
    fn add_asm(&mut self, asm_text: &str) -> Result<()> {
        self.pending_text.push_str(asm_text);
        self.pending_text.push('\n');
        Ok(())
    }

    fn declare_fn(&mut self, _name: &str, _arg_count: usize) -> Result<()> {
        Ok(())
    }

    fn define_extern_fn(&mut self, name: &str, _arg_count: usize, addr: *mut c_void) -> Result<()> {
        self.externs.insert(name.to_string(), addr as u64);
        Ok(())
    }

    fn lookup_addr(&mut self, name: &str) -> Result<u64> {
        self.build_if_needed()?;
        self.symbols
            .get(name)
            .copied()
            .ok_or_else(|| anyhow::anyhow!("native loader: symbol `{name}` not found"))
    }
}

impl Drop for NativeJit {
    fn drop(&mut self) {
        if !self.region.is_null() {
            unsafe {
                VirtualFree(self.region as *mut c_void, 0, MEM_RELEASE);
            }
            self.region = ptr::null_mut();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::{EncodedModule, Reloc, RelocKind};
    use std::collections::BTreeMap;

    // Host extern the JIT'd code calls back into.
    extern "C" fn host_inc(x: u64) -> u64 {
        x + 1
    }

    /// End-to-end loader proof: place a 2-symbol module with an INTERNAL
    /// branch reloc (call helper) and an EXTERN branch reloc (call host_inc),
    /// relocate, RX-protect, and execute — verifying the result computed by
    /// relocated code that called back into Rust.
    #[test]
    fn load_relocate_execute_with_host_callback() {
        // helper(x) = x + x   (lea rax,[rcx+rcx] ; ret) — leaf, no reloc.
        // entry(x)  = helper(host_inc(x)) = (x+1)*2
        //   sub rsp,40 ; call host_inc ; mov rcx,rax ; call helper ; add rsp,40 ; ret
        let mut code: Vec<u8> = Vec::new();
        // helper @ 0
        code.extend_from_slice(&[0x48, 0x8D, 0x04, 0x09]); // lea rax,[rcx+rcx]
        code.push(0xC3); // ret
        let helper_off = 0usize;
        // entry @ 5
        let entry_off = code.len();
        code.extend_from_slice(&[0x48, 0x83, 0xEC, 0x28]); // sub rsp,40
        let call_host_at = code.len() + 1; // rel32 follows the E8
        code.extend_from_slice(&[0xE8, 0, 0, 0, 0]); // call host_inc
        code.extend_from_slice(&[0x48, 0x89, 0xC1]); // mov rcx,rax
        let call_helper_at = code.len() + 1;
        code.extend_from_slice(&[0xE8, 0, 0, 0, 0]); // call helper
        code.extend_from_slice(&[0x48, 0x83, 0xC4, 0x28]); // add rsp,40
        code.push(0xC3); // ret

        let mut symbols = BTreeMap::new();
        symbols.insert("helper".to_string(), helper_off);
        symbols.insert("entry".to_string(), entry_off);

        let module = EncodedModule {
            code,
            symbols,
            relocs: vec![
                Reloc { at: call_host_at, size: 4, kind: RelocKind::BranchRel32, target: "host_inc".into(), addend: 0 },
                Reloc { at: call_helper_at, size: 4, kind: RelocKind::BranchRel32, target: "helper".into(), addend: 0 },
            ],
            externs: vec!["host_inc".to_string()],
        };

        let anchor = host_inc as usize as u64;
        let mut jit = NativeJit::new_near(anchor, 0x1000).expect("alloc near host");
        jit.define_extern("host_inc", host_inc as usize as u64);
        jit.load_module(&module).expect("load module");
        jit.finalize().expect("finalize");

        let entry_addr = jit.lookup("entry").expect("entry symbol");
        let entry: extern "C" fn(u64) -> u64 = unsafe { std::mem::transmute(entry_addr) };
        assert_eq!(entry(10), 22, "(10+1)*2 via host callback + internal call");
        assert_eq!(entry(0), 2, "(0+1)*2");
    }
}
