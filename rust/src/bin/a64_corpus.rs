//! a64-corpus — regenerate the no-LLVM AArch64 regression corpus from the
//! `Aarch64Model` generator, using LLVM-MC (`aarch64-apple-darwin`) as oracle.
//!
//!   cargo run --bin a64-corpus --features llvm
//!
//! Writes `corpus/aarch64.tsv` (oracle goldens for every form the native
//! `A64Encoder` currently matches). The `difftest` corpus-replay test then gates
//! the encoder against this file with **no LLVM dependency**. Re-run after
//! adding encoder coverage. See docs/design/aarch64-apple-silicon.md.

use std::path::Path;

use wfasm::a64::A64Encoder;
use wfasm::difftest::{aarch64::Aarch64Model, diff_model, record_corpus};
use wfasm::oracle::LlvmMcEncoder;

fn main() -> std::io::Result<()> {
    // Diagnostic: surface mismatches (encoder bugs) + the gap worklist.
    let report = diff_model(&A64Encoder, &LlvmMcEncoder::aarch64_macos(), &Aarch64Model);
    eprintln!("{}", report.summary());
    for (form, v) in report.mismatches() {
        eprintln!("MISMATCH [{}] {}", form.family, v);
    }
    eprintln!("{}", report.worklist());

    let build = record_corpus(&A64Encoder, &LlvmMcEncoder::aarch64_macos(), &Aarch64Model);
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("corpus").join("aarch64.tsv");
    std::fs::create_dir_all(path.parent().unwrap())?;
    std::fs::write(&path, &build.text)?;
    eprintln!(
        "recorded {} forms -> {}\n  ({} gaps, {} oracle-rejects, {} mismatches skipped)",
        build.recorded,
        path.display(),
        build.gaps,
        build.oracle_rejects,
        build.mismatches,
    );
    Ok(())
}
