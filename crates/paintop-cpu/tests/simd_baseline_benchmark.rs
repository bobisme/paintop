//! Baseline SIMD benchmark artifact (`bn-m0k`).
//!
//! Runs the full optimized-kernel benchmark sweep and writes the selection
//! artifact to `target/verification/cpu.optimized/benchmarks/baseline.json`. The
//! artifact identifies which pointwise kernels were *selected* for a
//! `cpu.optimized` backend (cleared the speedup gate) and records every kernel's
//! precision tag and measured speedup — the evidence `bn-m0k` requires and `bn-2ja`
//! extends with the differential tolerance.
//!
//! Timing is host- and load-dependent, so this test does **not** assert an absolute
//! nanosecond bound. It asserts the structural invariants the bone's acceptance
//! calls for: every kernel carries a precision tag and an explicit accept/reject
//! decision, the artifact serialises and is written, and the selection gate is the
//! mechanism that rejects a no-measurable-win kernel.

use std::path::PathBuf;

use paintop_cpu::optimized::bench::{KernelKind, benchmark_all};

/// A representative pointwise working set (1024x1024 RGBA ≈ 1M pixels) so the
/// timings reflect a real image-sized buffer, not a cache-resident toy.
const PIXELS: usize = 1024 * 1024;

fn artifact_path() -> PathBuf {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("crates dir")
        .parent()
        .expect("repo root")
        .join("target/verification/cpu.optimized/benchmarks");
    std::fs::create_dir_all(&root).expect("create benchmark dir");
    root.join("baseline.json")
}

// A full 1M-pixel sweep is too slow for the default (debug) `just test` gate, and
// the speedup ratio is only meaningful under optimization. Run it explicitly with
// `cargo test --release -- --ignored`; the fast structural invariants are covered
// in-suite by the `optimized::bench` unit tests on a small working set.
#[test]
#[ignore = "slow benchmark sweep; run with --release --ignored to refresh the perf artifact"]
fn emits_baseline_benchmark_artifact() {
    let artifact = benchmark_all(PIXELS);

    // Every kernel is measured and recorded with its precision tag and an explicit
    // accept/reject decision (selected union rejected == the whole kernel set).
    assert_eq!(artifact.selections.len(), KernelKind::ALL.len());
    assert_eq!(
        artifact.selected().len() + artifact.rejected().len(),
        KernelKind::ALL.len(),
        "every kernel has an explicit selection decision"
    );

    // The selection gate is the rejection mechanism: a kernel is selected iff its
    // measured speedup cleared the bar, so the artifact never silently ships a
    // no-measurable-win kernel.
    for sel in &artifact.selections {
        assert_eq!(
            sel.selected,
            sel.measurement.speedup >= artifact.min_speedup,
            "{:?} selection must follow the speedup gate",
            sel.kernel
        );
    }

    // Write the artifact so the perf evidence exists on disk for the milestone.
    let json = artifact.to_json().expect("serialise artifact");
    let path = artifact_path();
    std::fs::write(&path, format!("{json}\n")).expect("write artifact");
    assert!(path.exists(), "artifact written to {}", path.display());

    // Surface the decision in the test log for a quick human read.
    eprintln!(
        "SIMD baseline: {} selected, {} rejected (gate >= {:.2}x)",
        artifact.selected().len(),
        artifact.rejected().len(),
        artifact.min_speedup
    );
    for sel in &artifact.selections {
        eprintln!(
            "  {:<24} {:>6.2}x  {}  [{:?}]",
            sel.kernel.op_id(),
            sel.measurement.speedup,
            if sel.selected { "SELECT" } else { "reject" },
            sel.precision,
        );
    }
}
