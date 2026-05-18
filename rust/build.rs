//! Build script: tell rustc where to find LLVM-C.lib (link-time) and copy
//! LLVM-C.dll next to the built binaries (run-time).
//!
//! The LLVM Windows binary distribution at C:\Program Files\LLVM\ ships
//! LLVM-C.dll in `bin\` and LLVM-C.lib in `lib\`. It does NOT ship
//! llvm-config or the full per-component static archives, so we link the
//! single C-API import lib and rely on the DLL at runtime.
//!
//! Override the LLVM root with LLVM_DIR if you've installed it elsewhere.

use std::env;
use std::fs;
use std::path::{Path, PathBuf};

fn main() {
    let llvm_dir = env::var("LLVM_DIR")
        .unwrap_or_else(|_| r"C:\Program Files\LLVM".to_string());
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
    // C:\Program Files\LLVM\bin on PATH. Cargo runs build.rs before
    // building, so OUT_DIR points into target/<profile>/build/<crate>-<hash>/out.
    // The actual binaries land in target/<profile>/. Walk up to find it.
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

    // Also drop it next to examples/, deps/ for tests etc.
    for sub in &["deps", "examples"] {
        let d = target_dir.join(sub);
        if d.is_dir() {
            let _ = fs::copy(&dll, d.join("LLVM-C.dll"));
        }
    }

    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=LLVM_DIR");
}

fn file_differs(a: &Path, b: &Path) -> bool {
    match (fs::metadata(a), fs::metadata(b)) {
        (Ok(ma), Ok(mb)) => ma.len() != mb.len(),
        _ => true,
    }
}
