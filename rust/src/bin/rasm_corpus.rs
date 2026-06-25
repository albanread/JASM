//! rasm-corpus — regenerate the no-LLVM regression corpus from the `X86Model`
//! generator, using LLVM-MC as the golden oracle.
//!
//!   cargo run --bin rasm-corpus --features llvm
//!
//! Writes `corpus/x86_64.tsv` (oracle goldens for every form rasm currently
//! matches). The `difftest::corpus_replay_matches_golden` test then gates rasm
//! against this file with **no LLVM dependency**. Re-run after closing gaps.

use std::path::Path;

use wfasm::difftest::{record_corpus, x86::X86Model};
use wfasm::oracle::LlvmMcEncoder;
use wfasm::rasm::RasmEncoder;

fn main() -> std::io::Result<()> {
    let build = record_corpus(&RasmEncoder, &LlvmMcEncoder::new(), &X86Model);
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("corpus").join("x86_64.tsv");
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
