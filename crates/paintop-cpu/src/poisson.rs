//! The shared **Poisson / screened-Poisson solver core** for the `repair`
//! domain (`OP_CATALOG` §12; `plan.md` §1444 — M4 gradient-domain editing).
//!
//! This module is the deterministic iterative solver that
//! `repair.poisson_blend@1` and `repair.screened_poisson@1` build on. It is not
//! itself an op; it exposes a single entry point, [`solve`], that the op kernels
//! call after they have assembled the divergence-of-guidance field, the boundary
//! (anchor) values, and the interior mask.
//!
//! # The discrete problem
//!
//! Gradient-domain editing reconstructs an unknown field `u` whose Laplacian
//! matches a guidance divergence `b` inside a region, subject to fixed boundary
//! values on the region's edge. The classical Poisson problem is
//!
//! ```text
//! Δu = b            inside the masked region
//! u  = anchor       on the boundary (Dirichlet)
//! ```
//!
//! Discretized on the pixel grid with the 5-point Laplacian
//! `Δu(x,y) = u(x-1,y) + u(x+1,y) + u(x,y-1) + u(x,y+1) - 4·u(x,y)`, the update
//! that drives the residual to zero at an interior pixel is the Gauss-Seidel
//! sweep
//!
//! ```text
//! u(x,y) <- ( neighbour_sum - b(x,y) ) / 4
//! ```
//!
//! where each neighbour contributes its *current* value (an interior neighbour's
//! freshly-updated value within the same sweep, a boundary neighbour's fixed
//! `anchor` value). This is the workhorse the two ops share.
//!
//! # Screened Poisson (the `lambda` term)
//!
//! The **screened** variant adds a data-fidelity term pulling the solution toward
//! an `anchor` field with weight `lambda >= 0`:
//!
//! ```text
//! (Δ - lambda·I) u = b - lambda·anchor
//! ```
//!
//! so the interior update becomes
//!
//! ```text
//! u(x,y) <- ( neighbour_sum - b(x,y) + lambda·anchor(x,y) ) / (4 + lambda)
//! ```
//!
//! With `lambda = 0` this is exactly the pure Poisson update (the seamless-clone
//! limit); as `lambda -> infinity` the update approaches `anchor(x,y)` (the
//! solution snaps to the anchor). The two limits are the explicit semantics
//! `repair.screened_poisson@1` exposes.
//!
//! # Determinism (M4 exit criterion 2)
//!
//! Every sweep visits interior pixels in a fixed **row-major** order and reads
//! neighbours from a single shared buffer, so a rerun on a fixed backend is
//! bit-identical (asserted by the op test suites). There is no RNG, no
//! parallel reduction, and no iteration-order dependence beyond the declared
//! row-major Gauss-Seidel order.
//!
//! # Convergence metrics (M4 exit criterion 1)
//!
//! [`solve`] returns a [`SolveReport`] carrying the per-sweep relative-residual
//! `residual_history`, the `iterations` actually run, the `converged` flag, the
//! [`SolverStopReason`], the targeted `tolerance`, and the `final_residual`.
//! These map directly onto the iterative fields of
//! [`SolverData`](paintop_ir::SolverData) the ops attach to their report.

use paintop_ir::SolverStopReason;

/// The 5-point Laplacian's centre weight (`4` for the unit-grid stencil).
const CENTRE_WEIGHT: f64 = 4.0;

/// The number of consecutive sweeps without a new best (lower) residual after
/// which the solver declares a stall and stops. Over-relaxation can raise the
/// residual transiently, so the guard tracks the best-so-far over a window
/// rather than reacting to a single non-decreasing step — a genuine stall (e.g.
/// a degenerate all-boundary region) trips it, SOR oscillation does not.
const STALL_WINDOW: u32 = 16;

/// The fractional improvement on the best-so-far residual that counts as real
/// progress; a smaller relative gain is treated as no progress for the stall
/// window.
const STALL_IMPROVEMENT: f64 = 1e-12;

/// The boundary condition for a solver pixel.
///
/// An `Interior` pixel is a free unknown the solver updates each sweep; a
/// `Boundary` pixel holds a fixed Dirichlet value (the anchor) and is never
/// updated. A pixel is decided once, before the first sweep, from the mask.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Cell {
    /// A free interior unknown the solver relaxes each sweep.
    Interior,
    /// A fixed boundary pixel pinned to its anchor value.
    Boundary,
}

/// The assembled linear system the [`solve`] entry point relaxes.
///
/// All buffers are row-major, length `width * height`. The caller assembles them
/// once per channel; the solver reads `cells`, `rhs`, and `anchor` and writes the
/// reconstructed field into a fresh buffer it returns.
pub struct PoissonSystem<'a> {
    /// The grid width.
    pub width: usize,
    /// The grid height.
    pub height: usize,
    /// Per-pixel boundary classification (row-major, length `width*height`).
    pub cells: &'a [Cell],
    /// The right-hand-side guidance divergence `b` at each pixel (row-major).
    pub rhs: &'a [f64],
    /// The anchor / boundary value at each pixel (row-major). For an `Interior`
    /// pixel this is also the screened-term target and the initial guess; for a
    /// `Boundary` pixel it is the fixed Dirichlet value.
    pub anchor: &'a [f64],
    /// The screened-Poisson data weight `lambda >= 0`. `0` is pure Poisson; a
    /// large value pulls the interior toward `anchor`.
    pub lambda: f64,
}

/// The solver's convergence record, mirroring the iterative
/// [`SolverData`](paintop_ir::SolverData) fields.
#[derive(Debug, Clone)]
pub struct SolveReport {
    /// The per-sweep relative residual `||r_k||_2 / ||r_0||_2` (length
    /// `iterations`). A monotone-decaying series evidences convergence.
    pub residual_history: Vec<f64>,
    /// The number of sweeps actually run.
    pub iterations: u32,
    /// Whether the relative residual reached `tolerance`.
    pub converged: bool,
    /// Why the solver halted.
    pub stop_reason: SolverStopReason,
    /// The targeted relative-residual tolerance.
    pub tolerance: f64,
    /// The final relative residual reached (`0` if there were no interior pixels
    /// or the initial residual was already zero).
    pub final_residual: f64,
}

/// The resolved iteration controls for a [`solve`] call.
#[derive(Debug, Clone, Copy)]
pub struct SolveControls {
    /// The maximum number of Gauss-Seidel sweeps.
    pub max_iterations: u32,
    /// The relative-residual tolerance to declare convergence.
    pub tolerance: f64,
    /// The successive-over-relaxation factor in `(0, 2)`; `1.0` is plain
    /// Gauss-Seidel. A value in `(1, 2)` accelerates convergence on the smooth
    /// Poisson problem while staying deterministic.
    pub omega: f64,
}

/// Relax `system` to convergence (or the iteration cap), returning the
/// reconstructed row-major field and the convergence [`SolveReport`].
///
/// The solver runs damped/over-relaxed Gauss-Seidel sweeps in fixed row-major
/// order. `Boundary` pixels keep their `anchor` value; `Interior` pixels start at
/// their `anchor` value (a deterministic initial guess) and are relaxed toward
/// the screened-Poisson solution. The relative residual is the L2 norm of the
/// per-interior-pixel equation residual, normalized by the initial residual.
///
/// The solve is a deterministic function of `system` and `controls` (fixed
/// ordering, no RNG), so a rerun is bit-identical on a fixed backend.
#[must_use]
pub fn solve(system: &PoissonSystem<'_>, controls: SolveControls) -> (Vec<f64>, SolveReport) {
    // Initial guess: the anchor everywhere (boundary pixels keep it forever).
    let mut field = system.anchor.to_vec();

    let initial_residual = residual_l2(system, &field);
    let tolerance = controls.tolerance;

    // A region with no interior unknowns (or an already-satisfied system) is
    // solved at iteration 0: nothing to relax.
    if initial_residual == 0.0 {
        return (
            field,
            SolveReport {
                residual_history: Vec::new(),
                iterations: 0,
                converged: true,
                stop_reason: SolverStopReason::Converged,
                tolerance,
                final_residual: 0.0,
            },
        );
    }

    let denom = CENTRE_WEIGHT + system.lambda;
    let mut residual_history = Vec::new();
    let mut converged = false;
    let mut stop_reason = SolverStopReason::MaxIterations;
    let mut last_relative = 1.0_f64;
    let mut iterations = 0_u32;
    // Stall tracking: the best (lowest) relative residual seen and how many
    // sweeps have passed since it last improved.
    let mut best_relative = f64::INFINITY;
    let mut sweeps_since_improvement = 0_u32;

    for sweep in 0..controls.max_iterations {
        sweep_once(system, &mut field, denom, controls.omega);
        iterations = sweep + 1;

        let relative = residual_l2(system, &field) / initial_residual;
        residual_history.push(relative);
        last_relative = relative;

        if relative <= tolerance {
            converged = true;
            stop_reason = SolverStopReason::Converged;
            break;
        }
        // Stall guard: track the best-so-far residual; if it has not improved by
        // a meaningful fraction for STALL_WINDOW consecutive sweeps the solver
        // can make no further headway and stops. Over-relaxation oscillation does
        // not trip this because a later sweep still lowers the best-so-far.
        if relative < best_relative * (1.0 - STALL_IMPROVEMENT) {
            best_relative = relative;
            sweeps_since_improvement = 0;
        } else {
            sweeps_since_improvement += 1;
            if sweeps_since_improvement >= STALL_WINDOW {
                stop_reason = SolverStopReason::Stalled;
                break;
            }
        }
    }

    let report = SolveReport {
        residual_history,
        iterations,
        converged,
        stop_reason,
        tolerance,
        final_residual: last_relative,
    };
    (field, report)
}

/// Run one row-major Gauss-Seidel / SOR sweep, updating every `Interior` pixel
/// in place from its current neighbours.
fn sweep_once(system: &PoissonSystem<'_>, field: &mut [f64], denom: f64, omega: f64) {
    let width = system.width;
    let height = system.height;
    for row in 0..height {
        for col in 0..width {
            let idx = row * width + col;
            if system.cells[idx] != Cell::Interior {
                continue;
            }
            let neighbour_sum = neighbour_sum(field, col, row, width, height);
            // Screened Gauss-Seidel target:
            //   u <- (neighbour_sum - b + lambda*anchor) / (4 + lambda)
            let target = system
                .lambda
                .mul_add(system.anchor[idx], neighbour_sum - system.rhs[idx])
                / denom;
            // Over-relaxation: u <- u + omega*(target - u). omega == 1 is plain
            // Gauss-Seidel.
            field[idx] = omega.mul_add(target - field[idx], field[idx]);
        }
    }
}

/// The sum of the four 5-point-stencil neighbours of `(col, row)`, treating an
/// out-of-grid neighbour as absent (its value is folded into the boundary by the
/// caller's anchor pixels, so the grid edge contributes nothing extra here).
///
/// In this formulation the solver region is padded by at least one ring of
/// `Boundary` pixels, so an `Interior` pixel always has four in-grid neighbours;
/// the edge clamp is a defensive fallback that reuses the centre value, keeping a
/// degenerate 1-pixel grid well-defined rather than panicking.
fn neighbour_sum(field: &[f64], col: usize, row: usize, width: usize, height: usize) -> f64 {
    let centre = field[row * width + col];
    let left = if col == 0 {
        centre
    } else {
        field[row * width + col - 1]
    };
    let right = if col + 1 == width {
        centre
    } else {
        field[row * width + col + 1]
    };
    let up = if row == 0 {
        centre
    } else {
        field[(row - 1) * width + col]
    };
    let down = if row + 1 == height {
        centre
    } else {
        field[(row + 1) * width + col]
    };
    left + right + up + down
}

/// The L2 norm of the per-interior-pixel equation residual
/// `r = (Δ - lambda·I) u - (b - lambda·anchor)`, summed in fixed row-major order.
///
/// Boundary pixels are satisfied by construction (they hold their anchor value),
/// so they contribute no residual.
fn residual_l2(system: &PoissonSystem<'_>, field: &[f64]) -> f64 {
    let width = system.width;
    let height = system.height;
    let mut sum_sq = 0.0_f64;
    for row in 0..height {
        for col in 0..width {
            let idx = row * width + col;
            if system.cells[idx] != Cell::Interior {
                continue;
            }
            let neighbour_sum = neighbour_sum(field, col, row, width, height);
            // Laplacian = neighbour_sum - 4*u; screened operator subtracts
            // lambda*u; rhs is b - lambda*anchor.
            let lhs = (system.lambda + CENTRE_WEIGHT).mul_add(-field[idx], neighbour_sum);
            let rhs = system.lambda.mul_add(-system.anchor[idx], system.rhs[idx]);
            let r = lhs - rhs;
            sum_sq = r.mul_add(r, sum_sq);
        }
    }
    sum_sq.sqrt()
}

/// The mask coverage above which a pixel is treated as inside the edited region
/// (an `Interior` unknown). At or below it the pixel is a fixed `Boundary`.
pub const MASK_THRESHOLD: f32 = 0.5;

/// Classify every grid pixel as `Interior` or `Boundary` from a coverage `mask`.
///
/// A pixel is `Interior` (a free unknown the solver relaxes) iff its mask
/// coverage exceeds [`MASK_THRESHOLD`] **and** it is not on the image border, so
/// every interior pixel always has four in-grid neighbours and the masked region
/// is wrapped by a ring of fixed boundary pixels (the seam continuity comes from
/// those pinned target values). A border pixel, or a pixel the mask does not
/// select, is `Boundary`.
#[must_use]
pub fn classify_cells(mask: &[f32], width: usize, height: usize) -> Vec<Cell> {
    let mut cells = vec![Cell::Boundary; width * height];
    if width < 3 || height < 3 {
        // Too small to hold an interior ring; every pixel is boundary.
        return cells;
    }
    for row in 1..height - 1 {
        for col in 1..width - 1 {
            let idx = row * width + col;
            if mask[idx] > MASK_THRESHOLD {
                cells[idx] = Cell::Interior;
            }
        }
    }
    cells
}

/// The 5-point discrete Laplacian of `field` at `(col, row)` under a clamped
/// (replicate) border.
///
/// Used as the guidance divergence `b = Δ(source)` for a gradient-domain blend.
/// The clamp only affects border pixels, which are always `Boundary` and so never
/// read this value — the interior values are exact.
#[must_use]
pub fn guidance_laplacian(
    field: &[f64],
    col: usize,
    row: usize,
    width: usize,
    height: usize,
) -> f64 {
    let centre = field[row * width + col];
    CENTRE_WEIGHT.mul_add(-centre, neighbour_sum(field, col, row, width, height))
}

#[cfg(test)]
mod characterization;
#[cfg(test)]
mod tests;
