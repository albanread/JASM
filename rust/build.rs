//! Build script: locate LLVM-C and wire link/runtime paths.
//!
//! The LLVM-C link is gated on the `llvm` cargo feature (default OFF). With the
//! feature off (Rasm native backend) we link nothing and require no LLVM install.
//! See WF65 docs/design/rasm-replace-llvm.md.
//!
//! Per-OS discovery:
//!
//! * **Windows** — the LLVM binary distribution at `C:\Program Files\LLVM\`
//!   ships `LLVM-C.dll` in `bin\` and `LLVM-C.lib` in `lib\`. It does NOT ship
//!   llvm-config or the per-component static archives, so we link the single
//!   C-API import lib and copy the DLL next to the binaries for runtime.
//! * **macOS** — Homebrew LLVM (`brew install llvm`) ships
//!   `lib/libLLVM-C.dylib`, a thin shim that re-exports `@rpath/libLLVM.dylib`.
//!   We link `-lLLVM-C` against `lib/` and add an `-rpath` so the reexported
//!   `libLLVM.dylib` resolves at runtime. No copy needed.
//! * **Linux** — same shape as macOS (`libLLVM-C.so`), discovered via
//!   `llvm-config --prefix` or `LLVM_DIR`.
//!
//! Override the LLVM root with `LLVM_DIR` (the folder containing `lib/` and
//! `bin/`) on any platform. See docs/design/aarch64-apple-silicon.md.

use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=LLVM_DIR");
    if env::var_os("CARGO_FEATURE_LLVM").is_none() {
        return;
    }

    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    match target_os.as_str() {
        "windows" => link_windows(),
        "macos" => link_unix_dylib(&["libLLVM-C.dylib"]),
        _ => link_unix_dylib(&["libLLVM-C.so", "libLLVM-C.so.22", "libLLVM-C.so.1"]),
    }
}

/// Resolve the LLVM install root: `LLVM_DIR` wins; otherwise ask `llvm-config`
/// (`--prefix`); otherwise fall back to the Homebrew default on macOS.
fn llvm_root() -> Option<PathBuf> {
    if let Ok(d) = env::var("LLVM_DIR") {
        return Some(PathBuf::from(d));
    }
    // `llvm-config --prefix` — works for Homebrew and most distro packages.
    for cfg in ["llvm-config", "/opt/homebrew/opt/llvm/bin/llvm-config"] {
        if let Ok(out) = Command::new(cfg).arg("--prefix").output() {
            if out.status.success() {
                let p = String::from_utf8_lossy(&out.stdout).trim().to_string();
                if !p.is_empty() {
                    return Some(PathBuf::from(p));
                }
            }
        }
    }
    let brew = PathBuf::from("/opt/homebrew/opt/llvm");
    if brew.exists() {
        return Some(brew);
    }
    None
}

/// macOS/Linux: link the LLVM-C shared library and add an rpath so its
/// reexported `libLLVM` resolves at runtime.
fn link_unix_dylib(candidates: &[&str]) {
    let root = llvm_root().unwrap_or_else(|| {
        panic!(
            "LLVM not found. Set LLVM_DIR to the install root (the folder with \
             lib/ and bin/), or install LLVM (`brew install llvm`)."
        )
    });
    let lib_dir = root.join("lib");
    let found = candidates.iter().any(|c| lib_dir.join(c).exists());
    if !found {
        panic!(
            "no LLVM-C shared library ({:?}) under {}\n\
             Set LLVM_DIR to the LLVM install root.",
            candidates,
            lib_dir.display()
        );
    }
    println!("cargo:rustc-link-search=native={}", lib_dir.display());
    println!("cargo:rustc-link-lib=dylib=LLVM-C");
    // libLLVM-C re-exports @rpath/libLLVM.{dylib,so}; make that rpath available.
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", lib_dir.display());
}

/// Windows: link `LLVM-C.lib` and copy `LLVM-C.dll` next to the binaries.
fn link_windows() {
    let llvm_dir = env::var("LLVM_DIR").unwrap_or_else(|_| r"C:\Program Files\LLVM".to_string());
    let llvm_dir = PathBuf::from(llvm_dir);

    let lib_dir = llvm_dir.join("lib");
    let bin_dir = llvm_dir.join("bin");
    let dll = bin_dir.join("LLVM-C.dll");
    let import_lib = lib_dir.join("LLVM-C.lib");

    if !import_lib.exists() {
        panic!(
            "LLVM-C.lib not found at {}\n\
             Set LLVM_DIR to the root of your LLVM install (the folder with bin/ and lib/).",
            import_lib.display()
        );
    }
    if !dll.exists() {
        panic!(
            "LLVM-C.dll not found at {}\n\
             Set LLVM_DIR to the root of your LLVM install (the folder with bin/ and lib/).",
            dll.display()
        );
    }

    println!("cargo:rustc-link-search=native={}", lib_dir.display());
    println!("cargo:rustc-link-lib=dylib=LLVM-C");

    // Copy LLVM-C.dll next to the built binaries so they run without
    // C:\Program Files\LLVM\bin on PATH.
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let target_dir = out_dir
        .ancestors()
        .nth(3) // out/ -> <crate>-<hash>/ -> build/ -> <profile>/
        .expect("OUT_DIR has unexpected layout");

    let dest = target_dir.join("LLVM-C.dll");
    if !dest.exists() || file_differs(&dll, &dest) {
        fs::copy(&dll, &dest).unwrap_or_else(|e| {
            panic!("copy {} -> {} failed: {}", dll.display(), dest.display(), e)
        });
    }
    for sub in &["deps", "examples"] {
        let d = target_dir.join(sub);
        if d.is_dir() {
            let _ = fs::copy(&dll, d.join("LLVM-C.dll"));
        }
    }
}

fn file_differs(a: &Path, b: &Path) -> bool {
    match (fs::metadata(a), fs::metadata(b)) {
        (Ok(ma), Ok(mb)) => ma.len() != mb.len(),
        _ => true,
    }
}
