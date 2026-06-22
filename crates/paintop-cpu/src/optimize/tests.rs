//! Engine-core verification for the local optimizer ([`super`]):
//!
//! - **known minimum**: with `smooth_weight = 0` the descent drives a free region
//!   to the target (the analytic minimizer), and the objective trajectory decays
//!   monotonically;
//! - **stop rules**: a tiny iteration cap stops at the cap; an already-minimized
//!   problem stops at iteration 0; a no-progress problem stalls;
//! - **determinism**: a rerun with the same controls is bit-identical;
//! - **locality**: frozen (outside-mask) pixels never change.

#![allow(
    clippy::cast_precision_loss,
    clippy::needless_range_loop,
    clippy::float_cmp,
    reason = "small analytic fixtures index by (col, row) and build coordinate \
              fields; the casts are exact at these sizes, and the float equalities \
              are deterministic bit-identity / exact-constant checks"
)]

use paintop_ir::SolverStopReason;

use super::{Cell, Controls, MASK_THRESHOLD, Objective, Problem, classify_cells, minimize};

/// A `width`×`height` grid with a centred square free region `[lo, hi)`.
fn centred_square_mask(width: usize, height: usize, lo: usize, hi: usize) -> Vec<f32> {
    let mut mask = vec![0.0_f32; width * height];
    for row in lo..hi {
        for col in lo..hi {
            mask[row * width + col] = 1.0;
        }
    }
    mask
}

/// Build a data-only problem (`smooth_weight = 0`): the known minimum inside the
/// mask is exactly the target.
fn data_problem<'a>(
    width: usize,
    height: usize,
    cells: &'a [Cell],
    init: &'a [f64],
    target: &'a [f64],
) -> Problem<'a> {
    Problem {
        width,
        height,
        cells,
        init,
        target,
        objective: Objective {
            data_weight: 1.0,
            smooth_weight: 0.0,
        },
    }
}

fn controls(max_iterations: u32, tolerance: f64, step: f64) -> Controls {
    Controls {
        max_iterations,
        tolerance,
        step,
        seed: 0,
    }
}

#[test]
fn converges_to_the_known_minimum() {
    let (w, h) = (8, 8);
    let mask = centred_square_mask(w, h, 2, 6);
    let cells = classify_cells(&mask, w, h);
    let init = vec![0.0_f64; w * h];
    // A constant target inside the mask; the minimizer is u = target there.
    let target = vec![0.7_f64; w * h];

    let problem = data_problem(w, h, &cells, &init, &target);
    let (field, report) = minimize(&problem, controls(2000, 1e-9, 0.25));

    assert!(report.converged, "should reach the tolerance: {report:?}");
    assert_eq!(report.stop_reason, SolverStopReason::Converged);
    // Free pixels reach the target within a tight bound.
    for row in 2..6 {
        for col in 2..6 {
            let v = field[row * w + col];
            assert!((v - 0.7).abs() < 1e-3, "free pixel {col},{row} = {v}");
        }
    }
    // The objective trajectory is monotonically non-increasing.
    for pair in report.objective_history.windows(2) {
        assert!(pair[1] <= pair[0] + 1e-12, "non-monotone: {pair:?}");
    }
    // The final relative objective is at or below the tolerance.
    assert!(report.final_objective <= report.tolerance);
}

#[test]
fn frozen_pixels_never_change() {
    let (w, h) = (6, 6);
    let mask = centred_square_mask(w, h, 2, 4);
    let cells = classify_cells(&mask, w, h);
    let init: Vec<f64> = (0..w * h).map(|i| i as f64).collect();
    let target = vec![100.0_f64; w * h];

    let problem = data_problem(w, h, &cells, &init, &target);
    let (field, _) = minimize(&problem, controls(500, 1e-6, 0.25));

    for idx in 0..w * h {
        if cells[idx] == Cell::Frozen {
            assert!(
                (field[idx] - init[idx]).abs() < f64::EPSILON,
                "frozen pixel {idx} moved: {} -> {}",
                init[idx],
                field[idx]
            );
        }
    }
}

#[test]
fn deterministic_reruns_are_bit_identical() {
    let (w, h) = (10, 10);
    let mask = centred_square_mask(w, h, 1, 9);
    let cells = classify_cells(&mask, w, h);
    let init: Vec<f64> = (0..w * h).map(|i| (i as f64).sin()).collect();
    let target: Vec<f64> = (0..w * h).map(|i| (i as f64 * 0.3).cos()).collect();

    let problem = data_problem(w, h, &cells, &init, &target);
    let (field_a, report_a) = minimize(&problem, controls(123, 1e-8, 0.2));
    let (field_b, report_b) = minimize(&problem, controls(123, 1e-8, 0.2));

    assert_eq!(field_a, field_b, "fields must be bit-identical");
    assert_eq!(report_a.iterations, report_b.iterations);
    assert_eq!(report_a.objective_history, report_b.objective_history);
    assert_eq!(report_a.final_objective, report_b.final_objective);
}

#[test]
fn iteration_cap_stops_predictably() {
    let (w, h) = (12, 12);
    let mask = centred_square_mask(w, h, 1, 11);
    let cells = classify_cells(&mask, w, h);
    let init = vec![0.0_f64; w * h];
    let target = vec![1.0_f64; w * h];

    let problem = data_problem(w, h, &cells, &init, &target);
    // A tiny cap with a tight tolerance: it cannot converge, so it stops at the cap.
    let (_, report) = minimize(&problem, controls(5, 1e-12, 0.1));

    assert_eq!(report.iterations, 5, "must stop exactly at the cap");
    assert_eq!(report.stop_reason, SolverStopReason::MaxIterations);
    assert!(!report.converged);
}

#[test]
fn already_minimized_stops_at_iteration_zero() {
    let (w, h) = (5, 5);
    // No free pixels: an all-frozen problem is solved at iteration 0.
    let mask = vec![0.0_f32; w * h];
    let cells = classify_cells(&mask, w, h);
    let init = vec![0.3_f64; w * h];
    let target = vec![0.9_f64; w * h];

    let problem = data_problem(w, h, &cells, &init, &target);
    let (field, report) = minimize(&problem, controls(500, 1e-6, 0.25));

    assert_eq!(report.iterations, 0);
    assert_eq!(report.stop_reason, SolverStopReason::Converged);
    assert!(report.objective_history.is_empty());
    // Nothing moved.
    assert_eq!(field, init);
}

#[test]
fn no_progress_stalls_rather_than_runs_away() {
    let (w, h) = (8, 8);
    let mask = centred_square_mask(w, h, 1, 7);
    let cells = classify_cells(&mask, w, h);
    let init = vec![0.0_f64; w * h];
    let target = vec![1.0_f64; w * h];

    // A step of 1.0 on the data objective E = Σ(u−t)² makes the update
    // u ← u − 1.0·2·(u−t) = u − 2(u−t) = 2t − u, which oscillates without
    // decreasing the objective — the no-progress guard must trip it.
    let problem = data_problem(w, h, &cells, &init, &target);
    let (_, report) = minimize(&problem, controls(10_000, 1e-9, 1.0));

    assert_eq!(report.stop_reason, SolverStopReason::Stalled);
    assert!(!report.converged);
    // It stopped well before the cap (within a small multiple of the stall window).
    assert!(
        report.iterations < 100,
        "stalled run must stop early, ran {}",
        report.iterations
    );
}

#[test]
fn classify_respects_the_threshold() {
    let mask = vec![0.0, 0.5, 0.50001, 1.0];
    let cells = classify_cells(&mask, 4, 1);
    assert_eq!(cells[0], Cell::Frozen);
    assert_eq!(cells[1], Cell::Frozen, "exactly the threshold is frozen");
    assert_eq!(cells[2], Cell::Free, "just above the threshold is free");
    assert_eq!(cells[3], Cell::Free);
    assert_eq!(MASK_THRESHOLD, 0.5);
}

#[test]
fn smoothness_term_reduces_roughness() {
    // With a noisy target and a positive smoothness weight, the optimized field is
    // smoother (smaller total-variation) than the raw target inside the mask.
    let (w, h) = (10, 10);
    let mask = centred_square_mask(w, h, 1, 9);
    let cells = classify_cells(&mask, w, h);
    let init = vec![0.5_f64; w * h];
    let target: Vec<f64> = (0..w * h)
        .map(|i| if i % 2 == 0 { 0.0 } else { 1.0 })
        .collect();

    let problem = Problem {
        width: w,
        height: h,
        cells: &cells,
        init: &init,
        target: &target,
        objective: Objective {
            data_weight: 1.0,
            smooth_weight: 2.0,
        },
    };
    let (field, report) = minimize(&problem, controls(2000, 1e-8, 0.05));
    assert!(report.iterations > 0);

    let tv = |buf: &[f64]| -> f64 {
        let mut sum = 0.0;
        for row in 1..h - 1 {
            for col in 1..w - 1 {
                let c = buf[row * w + col];
                sum += (buf[row * w + col + 1] - c).abs() + (buf[(row + 1) * w + col] - c).abs();
            }
        }
        sum
    };
    assert!(
        tv(&field) < tv(&target),
        "smoothed field TV {} should be below target TV {}",
        tv(&field),
        tv(&target)
    );
}
