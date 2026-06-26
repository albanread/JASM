//! rasm-gaps — run the `X86Model` form generator through the LLVM-MC differential
//! and print the coverage worklist: which instructions/forms a WF66 code word
//! can't yet assemble (`gaps`), plus any form rasm encodes WRONG (`MISMATCH`).
//!
//!   cargo run --bin rasm-gaps --features llvm
//!
//! `MISMATCH` is a bug (rasm accepts but mis-encodes vs LLVM) and must be zero.
//! `gaps` are the prioritized to-implement list. `oracle-reject` are generator
//! sizing artifacts (forms LLVM itself won't assemble), reported for triage.

use wfasm::difftest::{diff_model, x86::X86Model};
use wfasm::oracle::LlvmMcEncoder;
use wfasm::rasm::RasmEncoder;

fn main() {
    let model = X86Model;
    let rasm = RasmEncoder;
    let oracle = LlvmMcEncoder::x86_64();

    let report = diff_model(&rasm, &oracle, &model);
    println!("{}", report.summary());
    println!("{}", report.worklist());
}
