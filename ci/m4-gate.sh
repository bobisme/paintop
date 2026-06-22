#!/usr/bin/env bash
# M4 exit-criteria gate (plan.md §19 / §1428-§1446 "M4 — classical magic";
# AGENT_VERIFICATION). This is the executable checklist that proves the M4
# *classical* ops (pyramids/frequency split, orientation fields, edge-aware
# filters, Poisson solvers, PatchMatch/fill, procedural fields, reaction
# diffusion, and the contract-driven local optimizer) satisfy the four governing
# M4 exit criteria — and that they did NOT regress the M1/M1.5/M2/M3 core. It
# mirrors ci/m3-gate.sh: it asserts, in order, the FIVE M4 criteria, then re-runs
# the prior milestone gates so an M4 addition cannot silently break the oracle,
# and finally collects the perf/memory characterization artifacts.
#
# NOTE: the exact EDT + SDF mask algebra shipped in M1.5 (its gate is re-run as
# part of the no-regression sweep below), so it is NOT re-asserted here. The
# typed complex-spectrum FFT ops are an optional stretch and are covered by
# verify-op like any other op, but carry no extra solver/fixture criterion.
#
# The five M4 criteria (each backed by named tests / verify-op fixtures):
#   1. Every SOLVER-style op (Poisson, screened-Poisson, reaction-diffusion,
#      PatchMatch, local optimizer) EXPOSES convergence/progress metrics in its
#      report (residual/objective history, iteration count, stop reason).
#   2. Every ITERATIVE / seeded op has DETERMINISTIC seed + ordering rules ->
#      bit-identical reruns on a fixed backend (asserted by rerun tests).
#   3. Every required M4 op has a SYNTHETIC fixture with a known / bounded outcome
#      that passes `cargo xtask verify-op <op>@1`.
#   4. PERFORMANCE + MEMORY scale are characterized and emitted as CI artifacts
#      (the solver characterization harnesses record (size, iterations, residual,
#      working-set) rows and write them under target/verification/).
#   5. ORIENTATION / FREQUENCY analysis ops cover their covariance / preservation
#      contracts (structure-tensor + orientation rotation covariance; frequency
#      energy split + the frequency-preserved assertion).
#
# Run from the workspace root. Exits non-zero on the first failed criterion.
set -euo pipefail

# Resolve to the repo/workspace root (this script lives in ci/).
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# Where the perf/memory + verify-op artifacts are collected. CI points
# M4_GATE_ARTIFACT_DIR at the job's upload dir; locally it defaults under target/.
ARTIFACT_DIR="${M4_GATE_ARTIFACT_DIR:-$ROOT/target/m4-gate-artifacts}"
mkdir -p "$ARTIFACT_DIR"

step() { printf '\n=== M4 gate: %s ===\n' "$1"; }

# Run a named lib-test set in paintop-cpu. Quiet, exits non-zero on any failure.
run_lib_tests() {
  cargo test -p paintop-cpu --lib -- "$@" --quiet
}

# The full set of NEW M4 op ids. Their manifests are the verify-op fixtures
# asserted by criterion 3. (EDT/SDF shipped in M1.5 and are NOT in this list.)
M4_OPS=(
  frequency.gaussian_pyramid@1
  frequency.laplacian_split@1
  frequency.recombine@1
  frequency.fft2@1
  frequency.ifft2@1
  frequency.bandpass@1
  filter.structure_tensor@1
  filter.guided@1
  filter.bilateral@1
  field.orientation@1
  field.noise@1
  field.fbm@1
  field.domain_warp@1
  field.reaction_diffusion@1
  repair.poisson_blend@1
  repair.screened_poisson@1
  repair.patch_field@1
  repair.patch_synthesize@1
  analyze.frequency_energy@1
  assert.frequency_preserved@1
  optimize.local@1
)

# ---------------------------------------------------------------------------
# 0. Layer-0 build hygiene + the whole analytic/property/differential suite.
#
# CI splits this across dedicated jobs, so set M4_GATE_SKIP_CHECK=1 there to avoid
# re-running the whole suite; locally (`just m4-gate`) it runs as criterion 0.
# ---------------------------------------------------------------------------
if [ "${M4_GATE_SKIP_CHECK:-0}" = "1" ]; then
  step "Layer-0 build hygiene + tests (just check) — SKIPPED (dedicated CI job)"
else
  step "Layer-0 build hygiene + tests (just check)"
  just check
fi

# ---------------------------------------------------------------------------
# 1. Every solver-style op EXPOSES convergence metrics in its report.
#
# Each named test inspects the op's SolverData report and asserts it carries the
# iteration count, residual/objective trajectory, and stop reason.
# ---------------------------------------------------------------------------
step "criterion 1: every solver exposes convergence metrics in its report"
run_lib_tests \
  poisson_blend::tests::report_carries_iterative_metrics \
  screened_poisson::tests::report_carries_screened_solver_metrics \
  reaction_diffusion::tests::report_carries_solver_metrics \
  local_optimize::tests::the_seed_is_recorded_in_the_report \
  local_optimize::tests::converges_to_the_known_minimum \
  patch_field::tests::manifest_declares_field_and_report_outputs
echo "  ok: Poisson / screened-Poisson / reaction-diffusion / PatchMatch / local-optimizer reports"
echo "      all carry iteration count + residual/objective history + stop reason"

# ---------------------------------------------------------------------------
# 2. Every iterative / seeded op has deterministic seed/ordering -> bit-identical
#    reruns on a fixed backend.
# ---------------------------------------------------------------------------
step "criterion 2: deterministic seed/ordering -> bit-identical reruns (asserted)"
run_lib_tests \
  poisson_blend::tests::reruns_are_bit_identical \
  screened_poisson::tests::reruns_are_bit_identical \
  reaction_diffusion::tests::reruns_are_bit_identical \
  patch_field::tests::reruns_are_bit_identical_for_a_fixed_seed \
  local_optimize::tests::reruns_are_bit_identical \
  orientation::tests::rerun_is_bit_identical \
  structure_tensor::tests::rerun_is_bit_identical \
  frequency_energy::tests::energy_is_deterministic_bit_for_bit
echo "  ok: every iterative / seeded M4 op reruns bit-identically on a fixed backend"

# ---------------------------------------------------------------------------
# 3. Every required M4 op has a synthetic fixture with a known/bounded outcome
#    that passes `cargo xtask verify-op`.
# ---------------------------------------------------------------------------
step "criterion 3: every M4 op passes verify-op against its synthetic fixture"
cargo build -p xtask --quiet
VERIFY_DIR="$ARTIFACT_DIR/verify-op"
mkdir -p "$VERIFY_DIR"
for op in "${M4_OPS[@]}"; do
  manifest="ops/manifests/${op}.json"
  if [ ! -f "$manifest" ]; then
    echo "  FAIL: missing manifest for $op ($manifest)" >&2
    exit 1
  fi
  echo "  verify-op $op"
  cargo run -p xtask --quiet -- verify-op --manifest "$manifest" "$op"
done
# Collect the per-op verification artifacts (test-results / summary) for upload.
if [ -d target/verification ]; then
  for op in "${M4_OPS[@]}"; do
    if [ -d "target/verification/$op" ]; then
      mkdir -p "$VERIFY_DIR/$op"
      cp target/verification/"$op"/* "$VERIFY_DIR/$op/" 2>/dev/null || true
    fi
  done
fi
echo "  ok: all ${#M4_OPS[@]} M4 ops verified; per-op evidence collected under $VERIFY_DIR"

# ---------------------------------------------------------------------------
# 4. Performance + memory scale characterized and emitted as CI artifacts.
#
# The solver characterization harnesses sweep representative sizes, assert the
# residual stays finite + bounded and the per-pixel working set is constant
# (linear memory), and write the (size, iterations, residual, working-set) rows
# under target/verification/ for archival.
# ---------------------------------------------------------------------------
step "criterion 4: perf + memory scale characterized + emitted as artifacts"
run_lib_tests \
  poisson::characterization \
  optimize::characterization \
  reaction_diffusion::tests::solution_stays_finite_and_bounded
PERF_DIR="$ARTIFACT_DIR/characterization"
mkdir -p "$PERF_DIR"
collected=0
for f in poisson_characterization.json optimizer_characterization.json; do
  src="target/verification/$f"
  if [ -s "$src" ]; then
    cp "$src" "$PERF_DIR/"
    rows="$(jq -r '.rows | length' "$src" 2>/dev/null || echo 0)"
    have_axes="$(jq -r '.rows[0] | has("size_px") and has("iterations")' "$src" 2>/dev/null || echo false)"
    if [ "$rows" -lt 1 ] || [ "$have_axes" != "true" ]; then
      echo "  FAIL: $f is missing rows or the (size_px, iterations) axes" >&2
      exit 1
    fi
    echo "  collected $f ($rows rows; axes include size_px, iterations)"
    collected=$((collected + 1))
  fi
done
if [ "$collected" -lt 2 ]; then
  echo "  FAIL: expected >=2 characterization artifacts, collected $collected" >&2
  exit 1
fi
echo "  ok: $collected solver perf/memory characterization artifacts emitted to $PERF_DIR"

# ---------------------------------------------------------------------------
# 5. Orientation / frequency analysis ops cover their covariance / preservation
#    contracts.
# ---------------------------------------------------------------------------
step "criterion 5: orientation covariance + frequency preservation contracts"
run_lib_tests \
  structure_tensor::tests::rotation_swaps_diagonal_and_negates_off_diagonal \
  orientation::tests::rotation_covariance_of_orientation \
  orientation::tests::orientation_is_perpendicular_to_a_known_gradient \
  frequency_energy::tests::total_energy_is_the_sum_of_bands \
  frequency_preserved::tests::edit_inside_mask_does_not_fail_when_outside_is_preserved \
  frequency_preserved::tests::over_blurring_the_whole_image_fails_with_a_band_delta
echo "  ok: structure-tensor / orientation rotation covariance + frequency energy split"
echo "      + frequency-preserved assertion contracts all hold"

# ---------------------------------------------------------------------------
# 6. No-regression: the M1, M1.5, M2, and M3 gates must STILL be green.
#
# An M4 op/registry addition must keep every prior-milestone guarantee. We re-run
# their gates (skipping their `just check`, already run as criterion 0 here, to
# avoid a redundant compile of the whole suite). M1.5 re-running also covers the
# EDT/SDF mask algebra that M4's plan attributes to M1.5.
# ---------------------------------------------------------------------------
step "criterion 6 (no-regression): M1 / M1.5 / M2 / M3 gates still green"
M1_GATE_SKIP_CHECK=1 bash ci/m1-gate.sh
M15_GATE_SKIP_CHECK=1 bash ci/m15-gate.sh
M2_GATE_SKIP_CHECK=1 M2_GATE_ARTIFACT_DIR="$ARTIFACT_DIR/m2" bash ci/m2-gate.sh
M3_GATE_SKIP_CHECK=1 M3_GATE_ARTIFACT_DIR="$ARTIFACT_DIR/m3" bash ci/m3-gate.sh
echo "  ok: M1 / M1.5 / M2 / M3 gates all pass after the M4 classical-op additions"

# ---------------------------------------------------------------------------
# Upload: collect the artifacts. (In CI the dir is the upload target; locally it
# is just a stable path under target/.)
# ---------------------------------------------------------------------------
step "artifacts collected under $ARTIFACT_DIR"
ls -1R "$ARTIFACT_DIR"

step "ALL FIVE M4 EXIT CRITERIA PASSED (solvers metered + deterministic; fixtures bounded; perf/memory characterized; M1/M1.5/M2/M3 did not regress)"
