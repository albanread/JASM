//! Executable memory for the AArch64 loaders, per platform.
//!
//! * macOS: one `MAP_JIT` mapping, RWX once; writes and execution are switched
//!   per thread with `pthread_jit_write_protect_np`; `sys_icache_invalidate`
//!   cleans the caches.
//! * Linux AArch64: a plain RWX anonymous mapping (Linux has no per-thread
//!   toggle to make), and the cache maintenance `__clear_cache` does, inline:
//!   clean the data cache to the point of unification, invalidate the
//!   instruction cache over the range, barriers.
//!
//! * Linux x86-64: plain anonymous mappings, read-write while code is placed
//!   and read-execute after ([`protect_rx`]), or read-write-execute for hosts
//!   that patch code as they go; [`map_near`] puts a mapping within `call
//!   rel32` reach of an address. x86 needs no cache maintenance.
//!
//! Used by [`crate::native_macos::MacJit`], by [`crate::native::NativeJit`] on
//! Linux, and by hosts with their own code arenas (MF66's `CodeArena`, WF66's
//! regions).

#![cfg(any(target_os = "macos", target_os = "linux"))]

use std::ffi::c_void;
use std::ptr;

extern "C" {
    fn mmap(addr: *mut c_void, len: usize, prot: i32, flags: i32, fd: i32, offset: i64) -> *mut c_void;
    fn munmap(addr: *mut c_void, len: usize) -> i32;
    fn mprotect(addr: *mut c_void, len: usize, prot: i32) -> i32;
}

#[cfg(target_os = "macos")]
extern "C" {
    fn sys_icache_invalidate(start: *mut c_void, len: usize);
    fn pthread_jit_write_protect_np(enabled: i32);
}

const PROT_READ: i32 = 0x1;
const PROT_WRITE: i32 = 0x2;
const PROT_EXEC: i32 = 0x4;
const MAP_PRIVATE: i32 = 0x0002;
#[cfg(target_os = "macos")]
const MAP_FLAGS: i32 = MAP_PRIVATE | 0x1000 /* MAP_ANON */ | 0x0800 /* MAP_JIT */;
#[cfg(target_os = "linux")]
const MAP_FLAGS: i32 = MAP_PRIVATE | 0x20 /* MAP_ANONYMOUS */;
const MAP_FAILED: *mut c_void = usize::MAX as *mut c_void;

/// Map `len` bytes of read-write-execute memory, or null.
pub fn map_rwx(len: usize) -> *mut u8 {
    let p = unsafe { mmap(ptr::null_mut(), len, PROT_READ | PROT_WRITE | PROT_EXEC, MAP_FLAGS, -1, 0) };
    if p == MAP_FAILED {
        ptr::null_mut()
    } else {
        p as *mut u8
    }
}

/// Map `len` bytes of read-write memory, or null.
pub fn map_rw(len: usize) -> *mut u8 {
    let p = unsafe { mmap(ptr::null_mut(), len, PROT_READ | PROT_WRITE, MAP_FLAGS, -1, 0) };
    if p == MAP_FAILED {
        ptr::null_mut()
    } else {
        p as *mut u8
    }
}

/// Map `len` bytes (read-write, and execute if `exec`) within about 1.75 GB
/// of `anchor`, so code there and at `anchor` reach each other by rel32; null
/// if no hole was found. Tries addresses outward from `anchor` in 16 MB steps,
/// each with MAP_FIXED_NOREPLACE (an address in use fails rather than moving
/// or replacing anything).
#[cfg(target_os = "linux")]
pub fn map_near(anchor: u64, len: usize, exec: bool) -> *mut u8 {
    const STEP: u64 = 0x100_0000;
    const WINDOW: u64 = 0x7000_0000;
    const MAP_FIXED_NOREPLACE: i32 = 0x10_0000;
    let prot = PROT_READ | PROT_WRITE | if exec { PROT_EXEC } else { 0 };
    let base = anchor & !(STEP - 1);
    let mut d = STEP;
    while d < WINDOW {
        for want in [base.wrapping_add(d), base.wrapping_sub(d + len as u64)] {
            let want = want & !(STEP - 1);
            if want < 0x1_0000_0000 || want.abs_diff(anchor) + len as u64 > WINDOW {
                continue; // never the low 4 GB: the BOX's RISC OS map is there
            }
            let p = unsafe {
                mmap(want as *mut c_void, len, prot, MAP_FLAGS | MAP_FIXED_NOREPLACE, -1, 0)
            };
            if p == MAP_FAILED {
                continue;
            }
            if p as u64 == want {
                return p as *mut u8;
            }
            unsafe { munmap(p, len) }; // an old kernel took it as a hint
        }
        d += STEP;
    }
    ptr::null_mut()
}

/// Make `[p, p+len)` read-execute: false if that failed.
pub fn protect_rx(p: *mut u8, len: usize) -> bool {
    unsafe { mprotect(p as *mut c_void, len, PROT_READ | PROT_EXEC) == 0 }
}

/// Give back a mapping made by [`map_rwx`].
pub fn unmap(p: *mut u8, len: usize) {
    unsafe { munmap(p as *mut c_void, len) };
}

/// This thread may write JIT memory (`false`) or execute it (`true`).
/// A no-op on Linux.
pub fn write_protect(exec: bool) {
    #[cfg(target_os = "macos")]
    unsafe {
        pthread_jit_write_protect_np(exec as i32)
    };
    #[cfg(target_os = "linux")]
    let _ = exec;
}

/// Make freshly written code in `[p, p+len)` visible to instruction fetch.
pub fn icache_invalidate(p: *mut u8, len: usize) {
    #[cfg(target_os = "macos")]
    unsafe {
        sys_icache_invalidate(p as *mut c_void, len)
    };
    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
    let _ = (p, len); // x86 keeps instruction fetch coherent itself
    #[cfg(all(target_os = "linux", target_arch = "aarch64"))]
    unsafe {
        let ctr: u64;
        core::arch::asm!("mrs {}, ctr_el0", out(reg) ctr);
        let dline = 4usize << ((ctr >> 16) & 0xF);
        let iline = 4usize << (ctr & 0xF);
        let start = p as usize;
        let end = start + len;
        let mut a = start & !(dline - 1);
        while a < end {
            core::arch::asm!("dc cvau, {}", in(reg) a);
            a += dline;
        }
        core::arch::asm!("dsb ish");
        let mut a = start & !(iline - 1);
        while a < end {
            core::arch::asm!("ic ivau, {}", in(reg) a);
            a += iline;
        }
        core::arch::asm!("dsb ish", "isb");
    }
}
