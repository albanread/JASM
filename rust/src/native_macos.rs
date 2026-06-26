//! macOS / Apple Silicon native loader (`MacJit`) — the AArch64 sibling of the
//! Windows [`NativeJit`](crate::native). It places an
//! [`EncodedModule`](crate::backend::EncodedModule) from the native
//! [`A64Encoder`](crate::a64::A64Encoder) into executable memory, resolves
//! relocations, and hands back a callable function pointer. No LLVM.
//!
//! Memory model (see docs/design/aarch64-apple-silicon.md §3.6):
//!
//! * One region is `mmap`'d `MAP_JIT` (RWX once; the page protection never
//!   changes after that).
//! * Per-cycle write/exec is the **per-thread** `pthread_jit_write_protect_np`
//!   toggle — NOT `mprotect`. We flip to writable at allocation, build + relocate
//!   the whole module, then flip to executable in [`MacJit::finalize`].
//! * `sys_icache_invalidate` runs after the writes are in place and before
//!   execution (ARM has split I/D caches — mandatory).
//! * Far branch targets (host externs resolved via `dlsym`, routinely > ±128 MB
//!   from the region) are reached through a per-target absolute veneer
//!   (`movz/movk x16…; br x16`), the analogue of the x86 `movabs rax; jmp rax`.
//!
//! The executing thread must be the one that flipped to exec mode (the toggle is
//! per-thread); building + finalizing + calling on one thread satisfies that.

#![cfg(target_os = "macos")]

use std::collections::HashMap;
use std::ffi::c_void;
use std::ptr;

use anyhow::{bail, Context, Result};

use crate::backend::{EncodedModule, Reloc, RelocKind};

// ── libSystem FFI (linked by default on macOS) ──────────────────────────────

extern "C" {
    fn mmap(addr: *mut c_void, len: usize, prot: i32, flags: i32, fd: i32, offset: i64) -> *mut c_void;
    fn munmap(addr: *mut c_void, len: usize) -> i32;
    /// `<libkern/OSCacheControl.h>` — data-cache clean + instruction-cache
    /// invalidate over `[start, start+len)`. Public since macOS 10.5.
    fn sys_icache_invalidate(start: *mut c_void, len: usize);
    /// `<pthread.h>` — per-thread W^X toggle for `MAP_JIT` pages. `1` =
    /// write-protected (executable), `0` = writable (non-executable).
    fn pthread_jit_write_protect_np(enabled: i32);
}

const PROT_READ: i32 = 0x1;
const PROT_WRITE: i32 = 0x2;
const PROT_EXEC: i32 = 0x4;
const MAP_PRIVATE: i32 = 0x0002;
const MAP_ANON: i32 = 0x1000;
const MAP_JIT: i32 = 0x0800;
const MAP_FAILED: *mut c_void = usize::MAX as *mut c_void;
const PAGE: usize = 0x4000; // 16 KiB on Apple Silicon

fn round_up(n: usize, to: usize) -> usize {
    (n + to - 1) & !(to - 1)
}

/// `movz/movk x16, …; br x16` — load `addr` into x16 and branch. 5 words.
fn abs_veneer(addr: u64) -> [u32; 5] {
    let g = |i: u32| ((addr >> (16 * i)) & 0xFFFF) as u32;
    [
        0xD280_0000 | (g(0) << 5) | 16,             // movz x16, #g0
        0xF280_0000 | (1 << 21) | (g(1) << 5) | 16, // movk x16, #g1, lsl #16
        0xF280_0000 | (2 << 21) | (g(2) << 5) | 16, // movk x16, #g2, lsl #32
        0xF280_0000 | (3 << 21) | (g(3) << 5) | 16, // movk x16, #g3, lsl #48
        0xD61F_0000 | (16 << 5),                    // br x16
    ]
}
const VENEER_LEN: usize = 20;

struct Placed {
    base: u64,
    relocs: Vec<Reloc>,
}

pub struct MacJit {
    region: *mut u8,
    cap: usize,
    used: usize,
    /// Defined symbol → absolute runtime address.
    symbols: HashMap<String, u64>,
    /// Host extern name → absolute address.
    externs: HashMap<String, u64>,
    placed: Vec<Placed>,
    finalized: bool,
    writable: bool,
    /// Accumulated text for the [`Loader`](crate::backend::Loader) builder path.
    pending_text: String,
}

impl MacJit {
    /// Reserve `cap` bytes (rounded to a page) of `MAP_JIT` RWX code space, left
    /// in the *writable* state for building.
    pub fn with_capacity(cap: usize) -> Result<Self> {
        let cap = round_up(cap.max(PAGE), PAGE);
        let region = unsafe {
            mmap(
                ptr::null_mut(),
                cap,
                PROT_READ | PROT_WRITE | PROT_EXEC,
                MAP_PRIVATE | MAP_ANON | MAP_JIT,
                -1,
                0,
            )
        };
        if region == MAP_FAILED || region.is_null() {
            bail!("mmap(MAP_JIT) failed — JIT memory unavailable");
        }
        // Enter write mode for this thread so we can populate the region.
        unsafe { pthread_jit_write_protect_np(0) };
        Ok(MacJit {
            region: region as *mut u8,
            cap,
            used: 0,
            symbols: HashMap::new(),
            externs: HashMap::new(),
            placed: Vec::new(),
            finalized: false,
            writable: true,
            pending_text: String::new(),
        })
    }

    /// A builder-mode loader: accumulate text + externs, assemble + place on the
    /// first `lookup_addr`.
    pub fn new() -> Self {
        MacJit {
            region: ptr::null_mut(),
            cap: 0,
            used: 0,
            symbols: HashMap::new(),
            externs: HashMap::new(),
            placed: Vec::new(),
            finalized: false,
            writable: false,
            pending_text: String::new(),
        }
    }

    /// Bind a host extern (a Rust `extern "C"` function) by name.
    pub fn define_extern(&mut self, name: &str, addr: u64) {
        self.externs.insert(name.to_string(), addr);
    }

    /// Copy `m`'s code into the region (16-byte aligned), record its symbols at
    /// their final addresses, and queue its relocations. Returns the base.
    pub fn load_module(&mut self, m: &EncodedModule) -> Result<u64> {
        if self.finalized {
            bail!("MacJit already finalized");
        }
        debug_assert!(self.writable, "region must be writable to load");
        let start = round_up(self.used, 16);
        let end = start + m.code.len();
        if end > self.cap {
            bail!("code region exhausted: need {end} bytes, cap {}", self.cap);
        }
        let base = self.region as u64 + start as u64;
        unsafe { ptr::copy_nonoverlapping(m.code.as_ptr(), self.region.add(start), m.code.len()) };
        for (name, off) in &m.symbols {
            self.symbols.insert(name.clone(), base + *off as u64);
        }
        self.placed.push(Placed { base, relocs: m.relocs.clone() });
        self.used = end;
        Ok(base)
    }

    fn resolve(&self, name: &str) -> Option<u64> {
        self.symbols.get(name).copied().or_else(|| self.externs.get(name).copied())
    }

    /// Apply all relocations (building far-call veneers as needed), flip the
    /// region to executable, and invalidate the icache. Idempotent.
    pub fn finalize(&mut self) -> Result<()> {
        if self.finalized {
            return Ok(());
        }
        // Resolve every reloc target up front (avoids borrowing self mid-patch).
        let mut work: Vec<(u64, RelocKind, u64, i64)> = Vec::new();
        for p in &self.placed {
            for r in &p.relocs {
                let target = self
                    .resolve(&r.target)
                    .with_context(|| format!("unresolved reloc target `{}`", r.target))?;
                work.push((p.base + r.at as u64, r.kind, target, r.addend));
            }
        }

        let region = self.region;
        let cap = self.cap;
        let mut veneer_off = self.used;
        let mut veneers: HashMap<u64, u64> = HashMap::new();

        for (field, kind, target, addend) in work {
            let target = (target as i64 + addend) as u64;
            match kind {
                RelocKind::Branch26 => {
                    let mut disp = target as i64 - field as i64;
                    if !(-(1 << 27)..(1 << 27)).contains(&disp) {
                        // Out of ±128 MB → route through an absolute veneer.
                        let v = match veneers.get(&target) {
                            Some(&v) => v,
                            None => {
                                if veneer_off + VENEER_LEN > cap {
                                    bail!("veneer region exhausted");
                                }
                                let v_addr = region as u64 + veneer_off as u64;
                                let words = abs_veneer(target);
                                for (i, w) in words.iter().enumerate() {
                                    unsafe { write_u32(region.add(veneer_off + i * 4), *w) };
                                }
                                veneer_off += VENEER_LEN;
                                veneers.insert(target, v_addr);
                                v_addr
                            }
                        };
                        disp = v as i64 - field as i64;
                    }
                    or_word(field, ((disp >> 2) as u32) & 0x03FF_FFFF);
                }
                RelocKind::AdrpPage21 => {
                    let delta = ((target & !0xFFF) as i64 - (field & !0xFFF) as i64) >> 12;
                    if !(-(1 << 20)..(1 << 20)).contains(&delta) {
                        bail!("adrp page offset out of ±4 GB range");
                    }
                    let d = (delta as u32) & 0x1F_FFFF;
                    let immlo = d & 0x3;
                    let immhi = (d >> 2) & 0x7_FFFF;
                    or_word(field, (immlo << 29) | (immhi << 5));
                }
                RelocKind::AddPageOff12 => {
                    or_word(field, ((target & 0xFFF) as u32) << 10);
                }
                RelocKind::Abs64 => unsafe {
                    ptr::copy_nonoverlapping(target.to_le_bytes().as_ptr(), field as *mut u8, 8);
                },
                RelocKind::BranchRel32 | RelocKind::RipRel32 => {
                    bail!("x86 relocation kind {kind:?} in an AArch64 module");
                }
            }
        }
        self.used = veneer_off;

        // Flip this thread to exec mode, then invalidate the icache.
        unsafe {
            pthread_jit_write_protect_np(1);
            sys_icache_invalidate(self.region as *mut c_void, self.cap);
        }
        self.writable = false;
        self.finalized = true;
        Ok(())
    }

    /// Builder path: assemble accumulated text with the native AArch64 encoder,
    /// reserve a region, place + relocate + protect. Idempotent.
    fn build_if_needed(&mut self) -> Result<()> {
        if self.finalized {
            return Ok(());
        }
        if self.region.is_null() {
            let module = crate::a64::assemble(&self.pending_text)
                .context("A64Encoder: assemble kernel text")?;
            // code + one veneer per extern + slack.
            let cap = module.code.len() + module.externs.len() * VENEER_LEN + PAGE;
            let externs = std::mem::take(&mut self.externs);
            *self = MacJit::with_capacity(cap)?;
            self.externs = externs;
            self.load_module(&module)?;
        }
        self.finalize()
    }

    /// Runtime address of a defined symbol (after [`finalize`]).
    pub fn lookup(&self, name: &str) -> Option<u64> {
        self.symbols.get(name).copied()
    }

    pub fn has_symbol(&self, name: &str) -> bool {
        self.symbols.contains_key(name)
    }
}

impl Default for MacJit {
    fn default() -> Self {
        Self::new()
    }
}

/// OR `bits` into the 32-bit instruction word at `field` (read-modify-write).
fn or_word(field: u64, bits: u32) {
    unsafe {
        let p = field as *mut u8;
        let mut buf = [0u8; 4];
        ptr::copy_nonoverlapping(p, buf.as_mut_ptr(), 4);
        let w = u32::from_le_bytes(buf) | bits;
        ptr::copy_nonoverlapping(w.to_le_bytes().as_ptr(), p, 4);
    }
}

unsafe fn write_u32(p: *mut u8, w: u32) {
    ptr::copy_nonoverlapping(w.to_le_bytes().as_ptr(), p, 4);
}

impl crate::backend::Loader for MacJit {
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
            .ok_or_else(|| anyhow::anyhow!("macos loader: symbol `{name}` not found"))
    }
}

impl Drop for MacJit {
    fn drop(&mut self) {
        if !self.region.is_null() {
            // Return to write mode so a subsequent builder on this thread starts
            // clean, then release the mapping.
            unsafe {
                if !self.writable {
                    pthread_jit_write_protect_np(0);
                }
                munmap(self.region as *mut c_void, self.cap);
            }
            self.region = ptr::null_mut();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::a64::A64Encoder;
    use crate::backend::Encoder;

    fn place(asm: &str) -> MacJit {
        let m = A64Encoder.encode(asm).expect("encode");
        let mut jit = MacJit::with_capacity(m.code.len() + PAGE).expect("mmap");
        jit.load_module(&m).expect("load");
        jit.finalize().expect("finalize");
        jit
    }

    /// End-to-end: encode `(x+1)*2`, JIT it, and run it. Proves the whole
    /// pipe — A64Encoder → MAP_JIT → W^X toggle → icache → execute.
    #[test]
    fn leaf_executes() {
        let jit = place(".globl entry\nentry:\nadd x0, x0, #1\nadd x0, x0, x0\nret\n");
        let f: extern "C" fn(u64) -> u64 = unsafe { std::mem::transmute(jit.lookup("entry").unwrap()) };
        assert_eq!(f(10), 22);
        assert_eq!(f(0), 2);
        assert_eq!(f(100), 202);
    }

    /// Internal `bl` (in-range, patched, no veneer): entry calls a leaf helper.
    #[test]
    fn internal_call_executes() {
        let src = "\
.globl entry
entry:
stp x29, x30, [sp, #-16]!
bl dbl
ldp x29, x30, [sp], #16
ret
dbl:
add x0, x0, x0
ret
";
        let jit = place(src);
        let f: extern "C" fn(u64) -> u64 = unsafe { std::mem::transmute(jit.lookup("entry").unwrap()) };
        assert_eq!(f(21), 42);
    }

    extern "C" fn host_inc(x: u64) -> u64 {
        x + 1
    }

    /// Host callback: JIT'd code `bl`s a Rust `extern "C"` function. The target
    /// is in the test binary, typically far from the JIT region, so this
    /// exercises the absolute veneer path (and is correct either way).
    #[test]
    fn host_callback_executes() {
        let src = "\
.globl entry
entry:
stp x29, x30, [sp, #-16]!
bl host_inc
add x0, x0, x0
ldp x29, x30, [sp], #16
ret
";
        let m = A64Encoder.encode(src).expect("encode");
        let mut jit = MacJit::with_capacity(m.code.len() + PAGE).expect("mmap");
        jit.define_extern("host_inc", host_inc as *const () as u64);
        jit.load_module(&m).expect("load");
        jit.finalize().expect("finalize");
        let f: extern "C" fn(u64) -> u64 = unsafe { std::mem::transmute(jit.lookup("entry").unwrap()) };
        assert_eq!(f(10), 22, "(10+1)*2 via host callback");
        assert_eq!(f(0), 2);
    }

    #[test]
    fn abs_veneer_encoding() {
        // movz/movk x16, 0x1122_3344_5566_7788 ; br x16
        let v = abs_veneer(0x1122_3344_5566_7788);
        assert_eq!(v[0], 0xD2800000 | (0x7788 << 5) | 16);
        assert_eq!(v[4], 0xD61F0000 | (16 << 5));
    }
}
