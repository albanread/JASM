//! A near (rel32-reachable) code/data arena the host allocates within ±1.75 GB
//! of the kernel, so emitted runtime words (`CODE:`/`LET`) are reachable by a
//! plain `call rel32` from the kernel/dictionary. LLVM-independent: the MCJIT
//! memory-manager callbacks that drive it (`arena_alloc_*`) live in `jit.rs`
//! behind the `llvm` feature, while the native `CODE:` path calls [`alloc`]
//! directly. The arena is mapped **RWX** and the host owns the backing memory:
//! it must outlive every loader that allocates from it.

use std::ptr;

#[repr(C)]
pub struct CodeArena {
    pub(crate) base: *mut u8,
    pub(crate) size: usize,
    pub(crate) offset: usize,
    /// Bytes reserved immediately before each CODE section so the host can
    /// stash per-function metadata at `[section_base - code_header ..
    /// section_base)` — e.g. a dictionary xt back-offset cell, the same way
    /// boot-time primitives carry one. 0 = no reservation.
    pub(crate) code_header: usize,
}

impl CodeArena {
    /// `base`/`size` must describe an RWX region kept alive by the caller.
    pub fn new(base: *mut u8, size: usize) -> Self {
        CodeArena { base, size, offset: 0, code_header: 0 }
    }

    /// Like [`new`](CodeArena::new), but reserve `code_header` bytes before
    /// every code section (see the field docs).
    pub fn with_code_header(base: *mut u8, size: usize, code_header: usize) -> Self {
        CodeArena { base, size, offset: 0, code_header }
    }

    /// Bytes handed out so far.
    pub fn used(&self) -> usize {
        self.offset
    }

    /// Bump-allocate `size` bytes (`align`-aligned) with NO header reservation —
    /// the caller supplies its own leading xt-metadata cell (e.g. a `.quad 0`
    /// already in the assembled bytes). Returns null if the arena is exhausted.
    /// Used by the native `CODE:` path to place a RasmEncoder-assembled word.
    pub fn alloc(&mut self, size: usize, align: usize) -> *mut u8 {
        self.bump(size, align, 0)
    }

    pub(crate) fn bump(&mut self, size: usize, align: usize, reserve: usize) -> *mut u8 {
        let align = align.max(1);
        // Reserve first, then align: the returned pointer is `align`-aligned
        // and `[ptr - reserve .. ptr)` is free space past the prior section.
        let start = (self.offset + reserve + align - 1) & !(align - 1);
        match start.checked_add(size) {
            Some(end) if end <= self.size => {
                self.offset = end;
                unsafe { self.base.add(start) }
            }
            // Exhausted (or overflow): return null so the caller fails cleanly
            // rather than scribbling out of bounds.
            _ => ptr::null_mut(),
        }
    }
}
