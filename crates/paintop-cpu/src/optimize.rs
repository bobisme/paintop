//! The shared **contract-driven local-optimizer core** for the `optimize`
//! domain (`plan.md` §1428 final deliverable; `ALIEN_OPS` §7 — the contract-driven
//! micro-optimizer).
//!
//! This module is the deterministic iterative optimizer that `optimize.local@1`
//! builds on. It is not itself an op; it exposes a single entry point,
//! [`minimize`], that the op kernel calls once it has assembled the candidate's
//! initial field, the data target, and the coverage mask.
//!
//! # The objective
//!
//! The optimizer drives a per-pixel candidate field `u` (one channel; the op runs
//! it per channel) toward minimizing a **declared, mask-restricted objective**
//! built from two analytic terms:
//!
//! ```text
//! E(u) = w_data   · Σ_{i ∈ mask}  (u_i − target_i)²
//!      + w_smooth · Σ_{i ∈ mask}  ‖∇u_i‖²
//! ```
//!
//! where `∇u` is the forward-difference gradient on the pixel grid. The objective
//! is declared by two non-negative weights — there is **no arbitrary code
//! execution**; the optimizer only ever evaluates this fixed analytic family, so a
//! malformed objective is a schema error the op rejects before the engine runs.
//!
//! With `w_smooth = 0` the unique minimizer inside the mask is `u = target` (the
//! known-minimum fixture the convergence tests pin); a positive `w_smooth` trades
//! data fidelity for spatial smoothness.
//!
//! # The engine
//!
//! [`minimize`] runs **deterministic gradient descent** with a fixed step size
//! (the schedule). Each iteration computes the analytic gradient of `E` at every
//! masked pixel in fixed row-major order, takes a step `u ← u − step · ∇E`, and
//! records the new objective value. Pixels outside the mask are frozen at their
//! initial value (they are the optimizer's boundary), so the edit stays local.
//!
//! There is no RNG, no parallel reduction, and no iteration-order dependence
//! beyond the declared row-major order, so a rerun with the same seed and schedule
//! is **bit-identical** on a fixed backend (M4 exit criterion 2; asserted by the
//! op test suite). The `seed` does not steer the deterministic descent — it is
//! carried in the report as the schedule identity so a reproducible-tier run is
//! auditable.
//!
//! # Stop rules (no runaway)
//!
//! The engine stops at the first of three terminal conditions, recorded as the
//! [`SolverStopReason`]:
//!
//! - **[`Converged`](SolverStopReason::Converged)**: the relative objective
//!   `E_k / E_0` fell to or below the requested `tolerance`;
//! - **[`Stalled`](SolverStopReason::Stalled)**: the best-so-far objective did not
//!   improve by a meaningful fraction for [`STALL_WINDOW`] consecutive iterations
//!   (the no-progress guard);
//! - **[`MaxIterations`](SolverStopReason::MaxIterations)**: the iteration cap was
//!   reached first.
//!
//! A non-finite objective (a diverging step) trips the stall/cap guards rather
//! than spinning unbounded — the op additionally caps `max_iterations` and rejects
//! a step size that is not a finite positive number, so the engine always
//! terminates.
//!
//! # Convergence metrics
//!
//! [`minimize`] returns a [`MinimizeReport`] carrying the per-iteration relative
//! objective `objective_history`, the `iterations` actually run, the `converged`
//! flag, the [`SolverStopReason`], the targeted `tolerance`, and the
//! `final_objective`. These map directly onto the iterative fields of
//! [`SolverData`](paintop_ir::SolverData) the op attaches to its report.

use paintop_ir::SolverStopReason;

/// The no-progress window: consecutive non-improving iterations before stopping.
///
/// After this many iterations without a meaningful improvement on the best-so-far
/// objective the optimizer declares no progress and stops. A window (rather than a
/// single non-decreasing step) tolerates the rare flat step while still tripping
/// on a genuine stall.
pub const STALL_WINDOW: u32 = 16;

/// The fractional improvement on the best-so-far objective that counts as real
/// progress; a smaller relative gain is treated as no progress for the stall
/// window.
pub const STALL_IMPROVEMENT: f64 = 1e-12;

/// A pixel's role in the optimization.
///
/// A `Free` pixel is an unknown the optimizer updates each iteration; a `Frozen`
/// pixel keeps its initial value forever (it is outside the mask, the optimizer's
/// fixed boundary). A pixel is decided once, before the first iteration, from the
/// coverage mask.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Cell {
    /// A free unknown the optimizer relaxes each iteration.
    Free,
    /// A frozen pixel pinned to its initial value (outside the mask).
    Frozen,
}

/// The mask coverage above which a pixel is a `Free` unknown the optimizer
/// updates. At or below it the pixel is `Frozen` at its initial value.
pub const MASK_THRESHOLD: f32 = 0.5;

/// The declared objective: the two non-negative term weights.
///
/// This is the *whole* objective language — there is no code to execute, only the
/// fixed analytic data + smoothness terms weighted here. Both weights are finite
/// and `>= 0`; at least one is positive (an all-zero objective is rejected by the
/// op as degenerate).
#[derive(Debug, Clone, Copy)]
pub struct Objective {
    /// The weight on the masked data term `Σ (u − target)²`.
    pub data_weight: f64,
    /// The weight on the masked smoothness term `Σ ‖∇u‖²`.
    pub smooth_weight: f64,
}

/// The assembled single-channel problem the [`minimize`] entry point relaxes.
///
/// All buffers are row-major, length `width * height`. The op assembles them once
/// per channel; the optimizer reads `cells`, `init`, and `target` and returns a
/// fresh optimized field.
pub struct Problem<'a> {
    /// The grid width.
    pub width: usize,
    /// The grid height.
    pub height: usize,
    /// Per-pixel role (row-major, length `width*height`).
    pub cells: &'a [Cell],
    /// The initial candidate value at each pixel (row-major); also the frozen
    /// boundary value outside the mask and the descent's starting guess.
    pub init: &'a [f64],
    /// The data-term target at each pixel (row-major).
    pub target: &'a [f64],
    /// The declared objective weights.
    pub objective: Objective,
}

/// The resolved iteration controls for a [`minimize`] call.
#[derive(Debug, Clone, Copy)]
pub struct Controls {
    /// The maximum number of descent iterations before stopping at the cap.
    pub max_iterations: u32,
    /// The relative-objective tolerance to declare convergence (`E_k/E_0`).
    pub tolerance: f64,
    /// The fixed gradient-descent step size (the schedule). Finite and `> 0`.
    pub step: f64,
    /// The schedule seed, carried into the report as the run's identity. It does
    /// not steer the deterministic descent.
    pub seed: u64,
}

/// The optimizer's convergence record, mirroring the iterative
/// [`SolverData`](paintop_ir::SolverData) fields.
#[derive(Debug, Clone)]
pub struct MinimizeReport {
    /// The per-iteration relative objective `E_k / E_0` (length `iterations`). A
    /// monotone-decaying series evidences convergence.
    pub objective_history: Vec<f64>,
    /// The number of iterations actually run.
    pub iterations: u32,
    /// Whether the relative objective reached `tolerance`.
    pub converged: bool,
    /// Why the optimizer halted.
    pub stop_reason: SolverStopReason,
    /// The targeted relative-objective tolerance.
    pub tolerance: f64,
    /// The final relative objective reached (`0` if the initial objective was
    /// already zero or there were no free pixels).
    pub final_objective: f64,
    /// The absolute initial objective `E_0` (before the first step).
    pub initial_objective: f64,
}

/// Minimize `problem`'s declared objective by deterministic gradient descent,
/// returning the optimized row-major field and the convergence [`MinimizeReport`].
///
/// `Frozen` pixels keep their `init` value; `Free` pixels start at `init` and are
/// driven toward the minimizer. The objective is evaluated and stepped in fixed
/// row-major order with no RNG, so a rerun with the same `controls` is
/// bit-identical on a fixed backend.
#[must_use]
pub fn minimize(problem: &Problem<'_>, controls: Controls) -> (Vec<f64>, MinimizeReport) {
    let mut field = problem.init.to_vec();

    let initial_objective = objective_value(problem, &field);
    let tolerance = controls.tolerance;

    // An already-minimized problem (no free pixels, or a zero objective) is solved
    // at iteration 0: nothing to relax.
    if initial_objective == 0.0 {
        return (
            field,
            MinimizeReport {
                objective_history: Vec::new(),
                iterations: 0,
                converged: true,
                stop_reason: SolverStopReason::Converged,
                tolerance,
                final_objective: 0.0,
                initial_objective: 0.0,
            },
        );
    }

    let mut objective_history = Vec::new();
    let mut converged = false;
    let mut stop_reason = SolverStopReason::MaxIterations;
    let mut last_relative = 1.0_f64;
    let mut iterations = 0_u32;
    // No-progress tracking: the best (lowest) relative objective seen and how many
    // iterations have passed since it last improved.
    let mut best_relative = f64::INFINITY;
    let mut iters_since_improvement = 0_u32;

    for iter in 0..controls.max_iterations {
        descent_step(problem, &mut field, controls.step);
        iterations = iter + 1;

        let absolute = objective_value(problem, &field);
        let relative = absolute / initial_objective;
        objective_history.push(relative);
        last_relative = relative;

        if relative <= tolerance {
            converged = true;
            stop_reason = SolverStopReason::Converged;
            break;
        }
        // No-progress guard: a non-finite or non-improving objective for a whole
        // window means the descent can make no further headway and stops, so a
        // diverging step never spins unbounded.
        let improved = relative.is_finite() && relative < best_relative * (1.0 - STALL_IMPROVEMENT);
        if improved {
            best_relative = relative;
            iters_since_improvement = 0;
        } else {
            iters_since_improvement += 1;
            if iters_since_improvement >= STALL_WINDOW {
                stop_reason = SolverStopReason::Stalled;
                break;
            }
        }
    }

    let report = MinimizeReport {
        objective_history,
        iterations,
        converged,
        stop_reason,
        tolerance,
        final_objective: last_relative,
        initial_objective,
    };
    (field, report)
}

/// Take one gradient-descent step `u ← u − step · ∇E` over the free pixels, in
/// fixed row-major order, reading the gradient from the pre-step field so the step
/// is a deterministic simultaneous update.
fn descent_step(problem: &Problem<'_>, field: &mut [f64], step: f64) {
    let width = problem.width;
    let height = problem.height;
    // Simultaneous (Jacobi-style) update: compute every gradient from the current
    // field, then apply, so the result is independent of within-sweep visitation.
    let mut grad = vec![0.0_f64; field.len()];
    for row in 0..height {
        for col in 0..width {
            let idx = row * width + col;
            if problem.cells[idx] != Cell::Free {
                continue;
            }
            grad[idx] = objective_gradient(problem, field, col, row);
        }
    }
    for (value, g) in field.iter_mut().zip(&grad) {
        *value = step.mul_add(-*g, *value);
    }
}

/// The partial derivative `∂E/∂u_i` at the free pixel `(col, row)`.
///
/// `∂E/∂u_i = 2·w_data·(u_i − target_i) + w_smooth·(smoothness gradient)`. The
/// forward-difference smoothness energy `Σ ‖∇u‖²` has the well-known discrete
/// gradient `2·(neighbours_contribution)`, which for the interior reduces to the
/// negative 5-point Laplacian scaled by `2`; at the grid edge the absent forward
/// differences simply drop out (a clamped, replicate-free boundary).
fn objective_gradient(problem: &Problem<'_>, field: &[f64], col: usize, row: usize) -> f64 {
    let width = problem.width;
    let idx = row * width + col;
    let data = 2.0 * problem.objective.data_weight * (field[idx] - problem.target[idx]);

    let smooth = if problem.objective.smooth_weight == 0.0 {
        0.0
    } else {
        2.0 * problem.objective.smooth_weight * smoothness_gradient(problem, field, col, row)
    };
    data + smooth
}

/// The discrete gradient of the forward-difference smoothness energy at
/// `(col, row)`: `4·u − (sum of in-grid 4-neighbours)`, with out-of-grid
/// neighbours dropped (no replicate), summed in a fixed order.
fn smoothness_gradient(problem: &Problem<'_>, field: &[f64], col: usize, row: usize) -> f64 {
    let width = problem.width;
    let height = problem.height;
    let centre = field[row * width + col];
    let mut neighbours = 0.0_f64;
    let mut count = 0.0_f64;
    if col > 0 {
        neighbours += field[row * width + col - 1];
        count += 1.0;
    }
    if col + 1 < width {
        neighbours += field[row * width + col + 1];
        count += 1.0;
    }
    if row > 0 {
        neighbours += field[(row - 1) * width + col];
        count += 1.0;
    }
    if row + 1 < height {
        neighbours += field[(row + 1) * width + col];
        count += 1.0;
    }
    count.mul_add(centre, -neighbours)
}

/// The objective value `E(field)`, summed over the free pixels in fixed row-major
/// order with a stable accumulation.
fn objective_value(problem: &Problem<'_>, field: &[f64]) -> f64 {
    let width = problem.width;
    let height = problem.height;
    let mut data_sum = 0.0_f64;
    let mut smooth_sum = 0.0_f64;
    for row in 0..height {
        for col in 0..width {
            let idx = row * width + col;
            if problem.cells[idx] != Cell::Free {
                continue;
            }
            let diff = field[idx] - problem.target[idx];
            data_sum = diff.mul_add(diff, data_sum);
            if problem.objective.smooth_weight != 0.0 {
                smooth_sum += gradient_norm_sq(problem, field, col, row);
            }
        }
    }
    problem
        .objective
        .smooth_weight
        .mul_add(smooth_sum, problem.objective.data_weight * data_sum)
}

/// The squared forward-difference gradient magnitude `‖∇u‖²` at `(col, row)`:
/// `(u(x+1,y) − u)² + (u(x,y+1) − u)²`, with an out-of-grid forward difference
/// contributing `0`.
fn gradient_norm_sq(problem: &Problem<'_>, field: &[f64], col: usize, row: usize) -> f64 {
    let width = problem.width;
    let height = problem.height;
    let centre = field[row * width + col];
    let dx = if col + 1 < width {
        field[row * width + col + 1] - centre
    } else {
        0.0
    };
    let dy = if row + 1 < height {
        field[(row + 1) * width + col] - centre
    } else {
        0.0
    };
    dx.mul_add(dx, dy * dy)
}

/// Classify every grid pixel as `Free` or `Frozen` from a coverage `mask`.
///
/// A pixel is `Free` (an unknown the optimizer relaxes) iff its mask coverage
/// exceeds [`MASK_THRESHOLD`]; otherwise it is `Frozen` at its initial value. The
/// frozen pixels are the optimizer's fixed boundary, keeping the edit local.
#[must_use]
pub fn classify_cells(mask: &[f32], width: usize, height: usize) -> Vec<Cell> {
    let mut cells = vec![Cell::Frozen; width * height];
    for (cell, &coverage) in cells.iter_mut().zip(mask) {
        if coverage > MASK_THRESHOLD {
            *cell = Cell::Free;
        }
    }
    cells
}

#[cfg(test)]
mod characterization;
#[cfg(test)]
mod tests;
