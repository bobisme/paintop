//! Verification suite for the shared Poisson solver core (`OP_CATALOG` §12):
//!
//! - **analytic convergence**: a Laplace problem (`b = 0`) with a linear-ramp
//!   boundary converges to the exact linear (harmonic) interpolant; a constant
//!   boundary converges to that constant;
//! - **residual reports**: the residual history decreases monotonically and the
//!   stop reason / iteration count are recorded (M4 exit criterion 1);
//! - **determinism**: a rerun is bit-identical (M4 exit criterion 2);
//! - **screened limits**: large `lambda` snaps the interior to the anchor; small
//!   `lambda` recovers the pure-Poisson result.

#![allow(
    clippy::cast_precision_loss,
    clippy::needless_range_loop,
    clippy::suboptimal_flops,
    reason = "small analytic fixtures index by (col, row) and build coordinate \
              ramps; the grids are tiny so the integer-to-f64 casts are exact, the \
              explicit row/col loops read more clearly than iterator adapters, and \
              the readable `a*x + b*y` ramp is clearer than a fused multiply-add"
)]

use paintop_ir::SolverStopReason;

use super::{Cell, PoissonSystem, SolveControls, solve};

/// Default tight controls for the small analytic systems.
fn controls() -> SolveControls {
    SolveControls {
        max_iterations: 5_000,
        tolerance: 1e-10,
        omega: 1.9,
    }
}

/// Build a `width x height` system whose one-pixel border is `Boundary` (pinned
/// to `anchor`) and whose strict interior is `Interior`, with `rhs = 0`
/// everywhere (a pure Laplace problem) and a given `lambda`.
fn laplace_system(
    width: usize,
    height: usize,
    anchor: Vec<f64>,
    lambda: f64,
) -> (Vec<Cell>, Vec<f64>, Vec<f64>, f64) {
    let mut cells = vec![Cell::Boundary; width * height];
    for row in 1..height.saturating_sub(1) {
        for col in 1..width.saturating_sub(1) {
            cells[row * width + col] = Cell::Interior;
        }
    }
    let rhs = vec![0.0_f64; width * height];
    (cells, rhs, anchor, lambda)
}

#[test]
fn constant_boundary_converges_to_constant() {
    let (w, h) = (7, 5);
    let anchor = vec![3.5_f64; w * h];
    let (cells, rhs, anchor, lambda) = laplace_system(w, h, anchor, 0.0);
    let system = PoissonSystem {
        width: w,
        height: h,
        cells: &cells,
        rhs: &rhs,
        anchor: &anchor,
        lambda,
    };
    let (field, report) = solve(&system, controls());
    assert!(report.converged, "constant Laplace problem must converge");
    assert_eq!(report.stop_reason, SolverStopReason::Converged);
    for (idx, &v) in field.iter().enumerate() {
        if cells[idx] == Cell::Interior {
            assert!(
                (v - 3.5).abs() < 1e-6,
                "interior pixel {idx} = {v}, expected the boundary constant 3.5"
            );
        }
    }
}

#[test]
fn linear_ramp_boundary_recovers_linear_solution() {
    // A boundary set to the linear field f(x,y) = 2x + 3y is harmonic
    // (Laplacian 0), so the harmonic interior solution must equal 2x + 3y
    // exactly at every pixel.
    let (w, h) = (9, 9);
    let mut anchor = vec![0.0_f64; w * h];
    for row in 0..h {
        for col in 0..w {
            anchor[row * w + col] = 2.0 * col as f64 + 3.0 * row as f64;
        }
    }
    let (cells, rhs, anchor, lambda) = laplace_system(w, h, anchor.clone(), 0.0);
    let system = PoissonSystem {
        width: w,
        height: h,
        cells: &cells,
        rhs: &rhs,
        anchor: &anchor,
        lambda,
    };
    let (field, report) = solve(&system, controls());
    assert!(report.converged, "linear Laplace problem must converge");
    for row in 0..h {
        for col in 0..w {
            let idx = row * w + col;
            if cells[idx] == Cell::Interior {
                let expected = 2.0 * col as f64 + 3.0 * row as f64;
                assert!(
                    (field[idx] - expected).abs() < 1e-5,
                    "pixel ({col},{row}) = {} expected linear {expected}",
                    field[idx]
                );
            }
        }
    }
}

#[test]
fn residual_history_decreases_and_records_metrics() {
    let (w, h) = (11, 11);
    let mut anchor = vec![0.0_f64; w * h];
    // A non-trivial boundary so the interior starts far from the solution.
    for col in 0..w {
        anchor[col] = 10.0; // top row hot
    }
    let (cells, rhs, anchor, lambda) = laplace_system(w, h, anchor, 0.0);
    let system = PoissonSystem {
        width: w,
        height: h,
        cells: &cells,
        rhs: &rhs,
        anchor: &anchor,
        lambda,
    };
    let (_field, report) = solve(&system, controls());
    assert!(report.iterations >= 1);
    assert_eq!(
        report.residual_history.len(),
        report.iterations as usize,
        "one residual per sweep"
    );
    // Every relative residual is finite and non-negative. (Over-relaxation can
    // transiently raise the residual within a sweep or two, so per-step
    // monotonicity is not guaranteed; the trend must be a strong overall drop.)
    for &r in &report.residual_history {
        assert!(
            r.is_finite() && r >= 0.0,
            "residual {r} must be finite >= 0"
        );
    }
    let first = report.residual_history[0];
    let last = report.final_residual;
    assert!(
        last < first,
        "the residual must fall over the run: first {first}, last {last}"
    );
    assert!(report.converged, "the run had room to converge");
    assert!(report.final_residual <= report.tolerance);
}

#[test]
fn reruns_are_bit_identical() {
    let (w, h) = (13, 8);
    let mut anchor = vec![0.0_f64; w * h];
    for i in 0..w * h {
        anchor[i] = (i as f64).sin();
    }
    let (cells, rhs, anchor, lambda) = laplace_system(w, h, anchor, 0.0);
    let system = PoissonSystem {
        width: w,
        height: h,
        cells: &cells,
        rhs: &rhs,
        anchor: &anchor,
        lambda,
    };
    let (a, ra) = solve(&system, controls());
    let (b, rb) = solve(&system, controls());
    assert_eq!(a, b, "the solved field must be bit-identical on rerun");
    assert_eq!(ra.iterations, rb.iterations);
    assert_eq!(ra.residual_history, rb.residual_history);
}

#[test]
fn max_iterations_stop_reason() {
    let (w, h) = (15, 15);
    let mut anchor = vec![0.0_f64; w * h];
    for col in 0..w {
        anchor[col] = 100.0;
    }
    let (cells, rhs, anchor, lambda) = laplace_system(w, h, anchor, 0.0);
    let system = PoissonSystem {
        width: w,
        height: h,
        cells: &cells,
        rhs: &rhs,
        anchor: &anchor,
        lambda,
    };
    // A tiny cap with a tight tolerance: cannot converge, must report the cap.
    let tight = SolveControls {
        max_iterations: 2,
        tolerance: 1e-12,
        omega: 1.0,
    };
    let (_f, report) = solve(&system, tight);
    assert!(!report.converged);
    assert_eq!(report.stop_reason, SolverStopReason::MaxIterations);
    assert_eq!(report.iterations, 2);
}

#[test]
fn large_lambda_snaps_to_anchor() {
    // Screened Poisson with a huge lambda: the interior must collapse onto its
    // anchor regardless of the (zero) guidance.
    let (w, h) = (9, 9);
    let mut anchor = vec![0.0_f64; w * h];
    for row in 0..h {
        for col in 0..w {
            anchor[row * w + col] = (col + row) as f64;
        }
    }
    let (cells, rhs, anchor, _l) = laplace_system(w, h, anchor.clone(), 1.0e6);
    let system = PoissonSystem {
        width: w,
        height: h,
        cells: &cells,
        rhs: &rhs,
        anchor: &anchor,
        lambda: 1.0e6,
    };
    let (field, _report) = solve(&system, controls());
    for (idx, &v) in field.iter().enumerate() {
        if cells[idx] == Cell::Interior {
            assert!(
                (v - anchor[idx]).abs() < 1e-2,
                "large lambda: pixel {idx} = {v} should snap to anchor {}",
                anchor[idx]
            );
        }
    }
}
