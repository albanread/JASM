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

/// Allocate `size` bytes of RW memory within ±window of `anchor` so emitted
/// code reaches host externs by `call rel32`.
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
        })
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
        for (base, r) in patches {
            let target = self
                .resolve(&r.target)
                .with_context(|| format!("unresolved reloc target `{}`", r.target))?;
            self.apply_reloc(base, &r, target)?;
        }

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

    fn apply_reloc(&self, base: u64, r: &Reloc, target: u64) -> Result<()> {
        let field_addr = base + r.at as u64;
        match r.kind {
            RelocKind::BranchRel32 | RelocKind::RipRel32 => {
                debug_assert_eq!(r.size, 4);
                let rel = (target as i64 + r.addend) - (field_addr as i64 + 4);
                let rel32 = i32::try_from(rel).map_err(|_| {
                    anyhow::anyhow!(
                        "rel32 out of range for `{}`: {rel} (target {target:#x}, site {field_addr:#x})",
                        r.target
                    )
                })?;
                unsafe {
                    ptr::copy_nonoverlapping(
                        rel32.to_le_bytes().as_ptr(),
                        field_addr as *mut u8,
                        4,
                    );
                }
            }
            RelocKind::Abs64 => {
                debug_assert_eq!(r.size, 8);
                let val = (target as i64 + r.addend) as u64;
                unsafe {
                    ptr::copy_nonoverlapping(val.to_le_bytes().as_ptr(), field_addr as *mut u8, 8);
                }
            }
        }
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
