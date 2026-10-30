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
//! Used by [`crate::native_macos::MacJit`] and by hosts with their own code
//! arenas (MF66's `CodeArena`).

#![cfg(any(target_os = "macos", all(target_os = "linux", target_arch = "aarch64")))]

use std::ffi::c_void;
use std::ptr;

extern "C" {
    fn mmap(addr: *mut c_void, len: usize, prot: i32, flags: i32, fd: i32, offset: i64) -> *mut c_void;
    fn munmap(addr: *mut c_void, len: usize) -> i32;
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
    #[cfg(target_os = "linux")]
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
