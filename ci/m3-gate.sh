#!/usr/bin/env bash
# M3 exit-criteria gate (plan.md §19 "M3 — optimized CPU and wgpu backends"; §12
# backend strategy; AGENT_VERIFICATION). This is the executable checklist that
# proves the M3 *faster backends* are correct, leak-free, measured, and degrade
# cleanly — and that they did NOT regress the M1/M1.5/M2 reference core. It
# mirrors ci/m2-gate.sh: it asserts, in order, the FOUR M3 exit criteria, then
# re-runs the prior milestone gates so a backend change cannot silently break the
# oracle, and finally collects the perf + backend artifacts.
#
# The four M3 criteria (each backed by a named differential/trace/fallback test):
#   1. every GPU op passes its TOLERANCE CONTRACT against the cpu.reference oracle
#      — the cross-backend differential harness compares each optimized/GPU backend
#      to the oracle within the op's declared tier (exact bit-for-bit, bounded
#      within the L∞/L2 envelope). The cpu.optimized differential ALWAYS runs; the
#      wgpu differentials run when a GPU adapter is present and self-skip cleanly
#      when not (so this gate passes GPU-less too).
#   2. NO UNPLANNED READBACK in a fully GPU-compatible chain — the readback trace
#      shows zero host readbacks except at the declared export, and a host-side
#      materialize introduces exactly one expected readback. Adapter-gated.
#   3. PERF BASELINES checked into CI artifacts — the perf-baseline harness emits
#      the (op, backend, size, throughput) artifact; the checked-in reference and
#      the freshly-emitted artifact are collected for upload. Machine-tolerant: no
#      absolute wall-clock is asserted.
#   4. GPU ABSENCE -> CLEAN FALLBACK / EXPLICIT UNSUPPORTED ERROR — ALWAYS tested,
#      even on a GPU-less runner: the forced-no-adapter probe yields a clean typed
#      Unavailable (never a panic), it escalates to a typed E_GPU_UNAVAILABLE /
#      E_BACKEND_UNSUPPORTED error, and a *required* GPU backend that cannot run an
#      op is an explicit dispatch error — never a silent wrong answer.
#
# Run from the workspace root. Exits non-zero on the first failed criterion.
set -euo pipefail

# Resolve to the repo/workspace root (this script lives in ci/).
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# Where the perf + backend artifacts are collected. CI points
# M3_GATE_ARTIFACT_DIR at the job's upload dir; locally it defaults under target/.
ARTIFACT_DIR="${M3_GATE_ARTIFACT_DIR:-$ROOT/target/m3-gate-artifacts}"
mkdir -p "$ARTIFACT_DIR"

# A machine/runner identity for the perf baseline so a comparison only ever runs
# against the same machine class. CI sets PAINTOP_PERF_MACHINE to the runner label.
PERF_MACHINE="${PAINTOP_PERF_MACHINE:-ci}"

step() { printf '\n=== M3 gate: %s ===\n' "$1"; }

# Run a wgpu/cpu differential or trace suite. The GPU suites self-skip cleanly with
# no adapter, so this is safe on a GPU-less runner.
run_test() {
  local crate="$1" target="$2"
  cargo test -p "$crate" --test "$target" --quiet
}

# ---------------------------------------------------------------------------
# Adapter probe. The gate reports whether the GPU criteria actually RAN against a
# real device or were adapter-skipped, so a green GPU-less run is never mistaken
# for a verified-on-hardware run. `xtask perf-baseline --no-gpu` is never used
# here; the probe is purely informational for the banner.
# ---------------------------------------------------------------------------
step "GPU adapter probe"
if cargo run -q -p xtask -- gpu-probe 2>/dev/null; then
  GPU_PRESENT=1
  echo "  adapter present: GPU differential / readback criteria will run on hardware"
else
  GPU_PRESENT=0
  echo "  NO adapter: GPU differential / readback criteria self-skip cleanly;"
  echo "  the criterion-4 fallback path is STILL fully tested below."
fi

# ---------------------------------------------------------------------------
# 0. Layer-0 build hygiene + the whole analytic/property/differential suite.
#
# CI splits this across dedicated jobs, so set M3_GATE_SKIP_CHECK=1 there to avoid
# re-running the whole suite; locally (`just m3-gate`) it runs as criterion 0.
# ---------------------------------------------------------------------------
if [ "${M3_GATE_SKIP_CHECK:-0}" = "1" ]; then
  step "Layer-0 build hygiene + tests (just check) — SKIPPED (dedicated CI job)"
else
  step "Layer-0 build hygiene + tests (just check)"
  just check
fi

# ---------------------------------------------------------------------------
# 1. Every GPU op passes its tolerance contract vs the cpu.reference oracle.
# ---------------------------------------------------------------------------
step "criterion 1: optimized/GPU backends match the cpu.reference oracle (tolerance contract)"
# cpu.optimized differential — ALWAYS runs (no adapter needed).
run_test paintop-cpu simd_differential
run_test paintop-cpu gaussian_separable_differential
# wgpu differentials — run on hardware, self-skip cleanly with no adapter.
run_test paintop-wgpu fusion_differential_gpu
run_test paintop-wgpu separable_differential_gpu
run_test paintop-wgpu splat_differential_gpu
if [ "$GPU_PRESENT" = "1" ]; then
  echo "  ok: cpu.optimized AND wgpu backends matched the oracle within tolerance (on hardware)"
else
  echo "  ok: cpu.optimized matched the oracle; wgpu differentials adapter-skipped cleanly"
fi

# ---------------------------------------------------------------------------
# 2. No unplanned readback in a fully GPU-compatible chain.
# ---------------------------------------------------------------------------
step "criterion 2: no unplanned readback in a GPU-compatible chain (adapter-gated)"
run_test paintop-wgpu readback_trace_gpu
if [ "$GPU_PRESENT" = "1" ]; then
  echo "  ok: the fully-GPU chain readback-trace showed zero readbacks except the declared export"
else
  echo "  ok: readback-trace suite adapter-skipped cleanly (no GPU on this runner)"
fi

# ---------------------------------------------------------------------------
# 3. Performance baselines checked into CI artifacts.
#
# Emit the (op, backend, size, throughput) artifact and collect it for upload
# alongside the checked-in reference. Machine-tolerant: no absolute wall-clock is
# asserted; a regression check (relative slack) only runs when a same-machine-class
# reference exists. Built --release so the throughput is representative.
# ---------------------------------------------------------------------------
step "criterion 3: perf baselines emitted + collected as CI artifacts"
PERF_OUT="$ARTIFACT_DIR/perf-baseline.json"
PERF_ARGS=(perf-baseline --out "$PERF_OUT" --machine "$PERF_MACHINE")
# Compare against a checked-in reference for THIS machine class when one exists, so
# a regression beyond the (generous, configurable) threshold is flagged.
REF="$ROOT/ci/perf/baseline.${PERF_MACHINE}.json"
if [ -f "$REF" ]; then
  PERF_ARGS+=(--baseline "$REF" --threshold "${PERF_THRESHOLD:-0.25}")
  echo "  comparing against checked-in reference $REF"
else
  echo "  no checked-in reference for machine '$PERF_MACHINE'; recording a fresh baseline"
fi
cargo run -q --release -p xtask -- "${PERF_ARGS[@]}"
if [ ! -s "$PERF_OUT" ]; then
  echo "  FAIL: perf baseline artifact was not written" >&2
  exit 1
fi
# Structural self-check: the artifact carries (op, backend, size_px, throughput)
# rows. (jq is already a gate dependency — see verify-ops.)
rows="$(jq -r '.rows | length' "$PERF_OUT")"
have_axes="$(jq -r '.rows[0] | has("op") and has("backend") and has("size_px") and has("throughput_mpps")' "$PERF_OUT" 2>/dev/null || echo false)"
if [ "$rows" -lt 1 ] || [ "$have_axes" != "true" ]; then
  echo "  FAIL: perf baseline artifact is missing rows or the (op,backend,size,throughput) axes" >&2
  exit 1
fi
# Collect the checked-in reference baselines too, so the upload bundle is complete.
cp "$ROOT"/ci/perf/baseline.*.json "$ARTIFACT_DIR/" 2>/dev/null || true
echo "  ok: $rows-row perf baseline written to $PERF_OUT (axes: op, backend, size, throughput)"

# ---------------------------------------------------------------------------
# 4. GPU absence -> clean fallback / explicit unsupported error.
#
# ALWAYS tested, even on a GPU-less runner (and even WITH a GPU here — the forced
# no-adapter path simulates absence). These are unit tests that run GPU-less.
# ---------------------------------------------------------------------------
step "criterion 4: GPU absence yields a clean fallback / explicit unsupported error (ALWAYS tested)"
# The forced-no-adapter probe is a clean typed Unavailable (never a panic), and it
# escalates to a typed unsupported error.
cargo test -p paintop-wgpu --lib -- gpu::probe::tests::forced_no_adapter_is_a_clean_unavailable_not_a_panic \
  gpu::probe::tests::unavailable_escalates_to_typed_unsupported_error --quiet
# A REQUIRED backend that cannot run an op is an explicit dispatch error, and the
# default policy falls back to the reference oracle — never a silent wrong answer.
cargo test -p paintop-core --lib -- \
  executor::dispatch::tests::required_backend_absent_is_explicit_error \
  executor::dispatch::tests::falls_back_to_reference_when_optimized_absent --quiet
echo "  ok: forced no-adapter -> clean typed Unavailable; required-GPU-absent -> explicit error; default -> oracle fallback"

# ---------------------------------------------------------------------------
# 5. No-regression: the M1, M1.5, and M2 gates must STILL be green.
#
# A backend/parallelism change must keep every prior-milestone guarantee. We re-run
# their gates (skipping their `just check`, already run as criterion 0 here, to
# avoid a redundant compile of the whole suite).
# ---------------------------------------------------------------------------
step "criterion 5 (no-regression): M1, M1.5, and M2 gates still green"
M1_GATE_SKIP_CHECK=1 bash ci/m1-gate.sh
M15_GATE_SKIP_CHECK=1 bash ci/m15-gate.sh
M2_GATE_SKIP_CHECK=1 M2_GATE_ARTIFACT_DIR="$ARTIFACT_DIR/m2" bash ci/m2-gate.sh
echo "  ok: M1 / M1.5 / M2 gates all pass after the M3 backend additions"

# ---------------------------------------------------------------------------
# Upload: collect the artifacts. (In CI the dir is the upload target; locally it
# is just a stable path under target/.)
# ---------------------------------------------------------------------------
step "artifacts collected under $ARTIFACT_DIR"
ls -1R "$ARTIFACT_DIR"

if [ "$GPU_PRESENT" = "1" ]; then
  step "ALL FOUR M3 EXIT CRITERIA PASSED (GPU criteria verified ON HARDWARE; M1/M1.5/M2 did not regress)"
else
  step "ALL FOUR M3 EXIT CRITERIA PASSED (GPU criteria adapter-skipped; fallback path verified; M1/M1.5/M2 did not regress)"
fi
