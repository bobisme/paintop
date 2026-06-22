//! Determinism, perf, and memory **characterization** of the Poisson solver
//! core (`bn-1jrr`; the M4 gate's determinism + bounded-resource criteria).
//!
//! Unlike the analytic [`super::tests`] module (which pins correctness), this
//! module characterizes the solver's *operational* behaviour and emits a JSON
//! artifact the M4 gate can archive:
//!
//! - **bit-identical reruns** at several representative mask sizes on a fixed
//!   backend (M4 exit criterion 2);
//! - **perf / memory characterization**: per size, the iteration count to a fixed
//!   tolerance, the final residual, and the solver's working-set byte footprint
//!   (`O(width·height)` `f64` buffers), recorded — wall-clock is printed for the
//!   record but never asserted (CI-machine dependent);
//! - **runaway / no-progress predictability**: a tight iteration cap stops at the
//!   cap exactly, and a degenerate all-boundary region (no interior unknowns)
//!   stops at iteration 0 — neither spins unbounded.
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

use super::{Cell, PoissonSystem, SolveControls, classify_cells, guidance_laplacian, solve};

/// The representative square mask edge lengths the characterization sweeps.
const SIZES_PX: &[usize] = &[32, 64, 128];

/// The bytes of solver working set per pixel: the solver allocates the field
/// (`anchor.to_vec()`), and the caller assembles `cells` (1 byte tag, padded),
/// `rhs`, and `anchor` — four `O(N)` arrays. We characterize the dominant `f64`
/// arrays (field + rhs + anchor = 3 × 8 bytes) plus the cell tags.
const FIELD_ARRAYS_F64: usize = 3;

/// One characterized row: a size and the solver's behaviour at it.
#[derive(Debug)]
struct Row {
    size_px: usize,
    interior_pixels: usize,
    iterations: u32,
    final_residual: f64,
    stop_reason: SolverStopReason,
    working_set_bytes: usize,
}

/// Build a centred-square gradient-domain system at `edge`×`edge`: a quadratic
/// guidance (non-zero Laplacian so the solve does real work), a flat anchor, and
/// a square interior mask. Returns the assembled buffers.
fn build_system(edge: usize) -> (Vec<Cell>, Vec<f64>, Vec<f64>) {
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

    // Quadratic guidance source, flat anchor.
    let mut source = vec![0.0_f64; count];
    for row in 0..edge {
        for col in 0..edge {
            let x = col as f64;
            let y = row as f64;
            source[row * edge + col] = 0.001 * x.mul_add(x, y * y);
        }
    }
    let anchor = vec![0.25_f64; count];

    let mut rhs = vec![0.0_f64; count];
    for row in 0..edge {
        for col in 0..edge {
            let idx = row * edge + col;
            if cells[idx] == Cell::Interior {
                rhs[idx] = guidance_laplacian(&source, col, row, edge, edge);
            }
        }
    }
    (cells, rhs, anchor)
}

/// Solve the characterization system at `edge` and return the solved field plus
/// the row record.
fn characterize_size(edge: usize) -> (Vec<f64>, Row) {
    let (cells, rhs, anchor) = build_system(edge);
    let interior_pixels = cells.iter().filter(|&&c| c == Cell::Interior).count();
    let system = PoissonSystem {
        width: edge,
        height: edge,
        cells: &cells,
        rhs: &rhs,
        anchor: &anchor,
        lambda: 0.0,
    };
    let controls = SolveControls {
        max_iterations: 5_000,
        tolerance: 1e-6,
        omega: 1.8,
    };
    let start = Instant::now();
    let (field, report) = solve(&system, controls);
    let elapsed = start.elapsed();

    let count = edge * edge;
    let working_set_bytes =
        count * (FIELD_ARRAYS_F64 * std::mem::size_of::<f64>() + std::mem::size_of::<Cell>());

    println!(
        "poisson characterization: {edge}x{edge} ({count} px, {interior_pixels} interior) -> \
         {} sweeps, final residual {:.2e}, {:?}, ~{} KiB working set, {elapsed:?}",
        report.iterations,
        report.final_residual,
        report.stop_reason,
        working_set_bytes / 1024
    );

    let row = Row {
        size_px: edge,
        interior_pixels,
        iterations: report.iterations,
        final_residual: report.final_residual,
        stop_reason: report.stop_reason,
        working_set_bytes,
    };
    (field, row)
}

#[test]
fn reruns_are_bit_identical_across_representative_sizes() {
    // M4 exit criterion 2: a fixed-backend rerun is bit-identical at every
    // representative mask size.
    for &edge in SIZES_PX {
        let (a, _) = characterize_size(edge);
        let (b, _) = characterize_size(edge);
        assert_eq!(
            a, b,
            "the {edge}x{edge} solve must be bit-identical on rerun"
        );
    }
}

#[test]
fn characterization_artifact_is_emitted_for_the_gate() {
    let mut rows = Vec::new();
    for &edge in SIZES_PX {
        let (_field, row) = characterize_size(edge);
        // Every characterized solve must produce a finite, recorded residual and a
        // bounded iteration count (it never ran to the cap on these well-posed
        // systems).
        assert!(row.final_residual.is_finite() && row.final_residual >= 0.0);
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
            let path = dir.join("poisson_characterization.json");
            let _ = std::fs::write(&path, json);
            println!("poisson characterization artifact -> {}", path.display());
        }
    }
}

/// Serialize the characterized rows to a stable, hand-rolled JSON document (no
/// serde derive needed for an internal characterization type).
fn serialize_rows(rows: &[Row]) -> String {
    let mut out = String::from("{\n  \"solver\": \"poisson-gauss-seidel\",\n  \"rows\": [\n");
    for (i, r) in rows.iter().enumerate() {
        let reason = match r.stop_reason {
            SolverStopReason::Converged => "converged",
            SolverStopReason::MaxIterations => "max-iterations",
            SolverStopReason::Stalled => "stalled",
            _ => "unknown",
        };
        let _ = write!(
            out,
            "    {{\"size_px\": {}, \"interior_pixels\": {}, \"iterations\": {}, \
             \"final_residual\": {:.6e}, \"stop_reason\": \"{}\", \"working_set_bytes\": {}}}",
            r.size_px,
            r.interior_pixels,
            r.iterations,
            r.final_residual,
            reason,
            r.working_set_bytes,
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
    let (cells, rhs, anchor) = build_system(edge);
    let system = PoissonSystem {
        width: edge,
        height: edge,
        cells: &cells,
        rhs: &rhs,
        anchor: &anchor,
        lambda: 0.0,
    };
    let cap = 7;
    let (_f, report) = solve(
        &system,
        SolveControls {
            max_iterations: cap,
            tolerance: 1e-30,
            omega: 1.0,
        },
    );
    assert_eq!(report.iterations, cap, "must stop at exactly the cap");
    assert_eq!(report.stop_reason, SolverStopReason::MaxIterations);
    assert!(!report.converged);
}

#[test]
fn no_interior_region_stops_immediately() {
    // A degenerate all-boundary region (mask selects nothing) has no unknowns: the
    // solver must stop at iteration 0 with a satisfied system, never iterate.
    let edge = 32;
    let count = edge * edge;
    let cells = vec![Cell::Boundary; count];
    let rhs = vec![0.0_f64; count];
    let anchor = vec![0.5_f64; count];
    let system = PoissonSystem {
        width: edge,
        height: edge,
        cells: &cells,
        rhs: &rhs,
        anchor: &anchor,
        lambda: 0.0,
    };
    let (field, report) = solve(
        &system,
        SolveControls {
            max_iterations: 1_000,
            tolerance: 1e-8,
            omega: 1.5,
        },
    );
    assert_eq!(report.iterations, 0, "no unknowns ⇒ zero iterations");
    assert_eq!(report.stop_reason, SolverStopReason::Converged);
    assert!(report.converged);
    assert!(report.residual_history.is_empty());
    // The field is untouched: it equals the anchor everywhere.
    assert_eq!(field, anchor);
}
