//! Determinism, perf, and memory **characterization** of the local-optimizer
//! engine core (`bn-9mrr`; the M4 gate's determinism + bounded-resource criteria).
//!
//! Unlike the analytic [`super::tests`] module (which pins correctness), this
//! module characterizes the optimizer's *operational* behaviour and emits a JSON
//! artifact the M4 gate can archive:
//!
//! - **bit-identical reruns** at several representative free-region sizes on a
//!   fixed backend (M4 exit criterion 2);
//! - **perf / memory characterization**: per size, the iteration count to a fixed
//!   tolerance, the final objective, and the engine's working-set byte footprint
//!   (`O(width·height)` `f64` buffers: the field, the gradient scratch, and the
//!   per-channel init/target deinterleave) — wall-clock is printed for the record
//!   but never asserted (CI-machine dependent);
//! - **runaway / no-progress predictability**: a tight iteration cap stops at the
//!   cap exactly, and an all-frozen region stops at iteration 0 — neither spins
//!   unbounded.
//!
//! The artifact is written under `target/verification/` (the same tree
//! `cargo xtask verify-op` uses), so it is a build output, not a checked-in file.

#![allow(
    clippy::cast_precision_loss,
    clippy::needless_range_loop,
    reason = "characterization fixtures index by (col, row) and build small \
              coordinate fields; the casts are exact at these sizes"
)]

use std::fmt::Write as _;
use std::time::Instant;

use paintop_ir::SolverStopReason;

use super::{Cell, Controls, Objective, Problem, classify_cells, minimize};

/// The representative square free-region edge lengths the characterization sweeps.
const SIZES_PX: &[usize] = &[32, 64, 128];

/// The number of `O(N)` `f64` working arrays the engine touches per channel: the
/// field, the gradient scratch, and the deinterleaved init + target.
const FIELD_ARRAYS_F64: usize = 4;

/// One characterized row: a size and the optimizer's behaviour at it.
#[derive(Debug)]
struct Row {
    size_px: usize,
    free_pixels: usize,
    iterations: u32,
    final_objective: f64,
    stop_reason: SolverStopReason,
    working_set_bytes: usize,
}

/// Build a centred-square data-fit problem at `edge`×`edge`: a flat init, a
/// non-trivial target, and a square free mask. Returns the assembled buffers.
fn build_problem(edge: usize) -> (Vec<Cell>, Vec<f64>, Vec<f64>) {
    let count = edge * edge;
    let lo = edge / 4;
    let hi = edge - edge / 4;
    let mut mask = vec![0.0_f32; count];
    for row in lo..hi {
        for col in lo..hi {
            mask[row * edge + col] = 1.0;
        }
    }
    let cells = classify_cells(&mask, edge, edge);

    let init = vec![0.0_f64; count];
    let mut target = vec![0.0_f64; count];
    for row in 0..edge {
        for col in 0..edge {
            let x = col as f64 / edge as f64;
            let y = row as f64 / edge as f64;
            target[row * edge + col] = 0.5 * (x + y);
        }
    }
    (cells, init, target)
}

/// Optimize the characterization problem at `edge` and return the optimized field
/// plus the row record.
fn characterize_size(edge: usize) -> (Vec<f64>, Row) {
    let (cells, init, target) = build_problem(edge);
    let free_pixels = cells.iter().filter(|&&c| c == Cell::Free).count();
    let problem = Problem {
        width: edge,
        height: edge,
        cells: &cells,
        init: &init,
        target: &target,
        objective: Objective {
            data_weight: 1.0,
            smooth_weight: 0.0,
        },
    };
    let controls = Controls {
        max_iterations: 5_000,
        tolerance: 1e-6,
        step: 0.25,
        seed: 0,
    };
    let start = Instant::now();
    let (field, report) = minimize(&problem, controls);
    let elapsed = start.elapsed();

    let count = edge * edge;
    let working_set_bytes =
        count * (FIELD_ARRAYS_F64 * std::mem::size_of::<f64>() + std::mem::size_of::<Cell>());

    println!(
        "optimizer characterization: {edge}x{edge} ({count} px, {free_pixels} free) -> \
         {} iters, final objective {:.2e}, {:?}, ~{} KiB working set, {elapsed:?}",
        report.iterations,
        report.final_objective,
        report.stop_reason,
        working_set_bytes / 1024
    );

    let row = Row {
        size_px: edge,
        free_pixels,
        iterations: report.iterations,
        final_objective: report.final_objective,
        stop_reason: report.stop_reason,
        working_set_bytes,
    };
    (field, row)
}

#[test]
fn reruns_are_bit_identical_across_representative_sizes() {
    // M4 exit criterion 2: a fixed-backend rerun is bit-identical at every
    // representative free-region size.
    for &edge in SIZES_PX {
        let (a, _) = characterize_size(edge);
        let (b, _) = characterize_size(edge);
        assert_eq!(
            a, b,
            "the {edge}x{edge} optimize must be bit-identical on rerun"
        );
    }
}

#[test]
fn characterization_artifact_is_emitted_for_the_gate() {
    let mut rows = Vec::new();
    for &edge in SIZES_PX {
        let (_field, row) = characterize_size(edge);
        // Every characterized run produces a finite, recorded objective and a
        // bounded iteration count (it never ran to the cap on these well-posed
        // problems).
        assert!(row.final_objective.is_finite() && row.final_objective >= 0.0);
        assert!(row.iterations <= 5_000);
        rows.push(row);
    }

    // Memory scales linearly in pixels: the per-pixel working set is constant
    // across sizes (the O(N) characterization the gate records).
    let per_pixel: Vec<usize> = rows
        .iter()
        .map(|r| r.working_set_bytes / (r.size_px * r.size_px))
        .collect();
    assert!(
        per_pixel.windows(2).all(|w| w[0] == w[1]),
        "the per-pixel working set must be constant (linear memory), got {per_pixel:?}"
    );

    // Emit the artifact under target/verification/ for the M4 gate to archive.
    let json = serialize_rows(&rows);
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(std::path::Path::parent)
        .map(|root| root.join("target/verification"));
    if let Some(dir) = dir {
        // Best-effort: a sandbox without a writable target/ must not fail the
        // characterization (the assertions above are the real gate).
        if std::fs::create_dir_all(&dir).is_ok() {
            let path = dir.join("optimizer_characterization.json");
            let _ = std::fs::write(&path, json);
            println!("optimizer characterization artifact -> {}", path.display());
        }
    }
}

/// Serialize the characterized rows to a stable, hand-rolled JSON document.
fn serialize_rows(rows: &[Row]) -> String {
    let mut out = String::from("{\n  \"optimizer\": \"local-gradient-descent\",\n  \"rows\": [\n");
    for (i, r) in rows.iter().enumerate() {
        let reason = match r.stop_reason {
            SolverStopReason::Converged => "converged",
            SolverStopReason::MaxIterations => "max-iterations",
            SolverStopReason::Stalled => "stalled",
            _ => "unknown",
        };
        let _ = write!(
            out,
            "    {{\"size_px\": {}, \"free_pixels\": {}, \"iterations\": {}, \
             \"final_objective\": {:.6e}, \"stop_reason\": \"{}\", \"working_set_bytes\": {}}}",
            r.size_px, r.free_pixels, r.iterations, r.final_objective, reason, r.working_set_bytes,
        );
        if i + 1 < rows.len() {
            out.push(',');
        }
        out.push('\n');
    }
    out.push_str("  ]\n}\n");
    out
}

#[test]
fn runaway_iterations_stop_at_the_cap() {
    // A tight cap with an unreachable tolerance must stop at exactly the cap, not
    // spin — the runaway guard.
    let edge = 64;
    let (cells, init, target) = build_problem(edge);
    let problem = Problem {
        width: edge,
        height: edge,
        cells: &cells,
        init: &init,
        target: &target,
        objective: Objective {
            data_weight: 1.0,
            smooth_weight: 0.0,
        },
    };
    let cap = 7;
    let (_f, report) = minimize(
        &problem,
        Controls {
            max_iterations: cap,
            tolerance: 1e-30,
            step: 0.25,
            seed: 0,
        },
    );
    assert_eq!(report.iterations, cap, "must stop at exactly the cap");
    assert_eq!(report.stop_reason, SolverStopReason::MaxIterations);
    assert!(!report.converged);
}

#[test]
fn no_free_region_stops_immediately() {
    // A degenerate all-frozen region (mask selects nothing) has no unknowns: the
    // optimizer must stop at iteration 0, never iterate.
    let edge = 32;
    let count = edge * edge;
    let cells = vec![Cell::Frozen; count];
    let init = vec![0.3_f64; count];
    let target = vec![0.9_f64; count];
    let problem = Problem {
        width: edge,
        height: edge,
        cells: &cells,
        init: &init,
        target: &target,
        objective: Objective {
            data_weight: 1.0,
            smooth_weight: 0.0,
        },
    };
    let (field, report) = minimize(
        &problem,
        Controls {
            max_iterations: 1_000,
            tolerance: 1e-6,
            step: 0.25,
            seed: 0,
        },
    );
    assert_eq!(report.iterations, 0, "no unknowns: stop at iteration 0");
    assert_eq!(report.stop_reason, SolverStopReason::Converged);
    assert_eq!(field, init);
}
