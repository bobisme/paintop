#!/usr/bin/env bash
# M2 exit-criteria gate (plan.md §19 "M2 — graph compiler, tiles, ROI, cache";
# §11 demand/tile model; AGENT_VERIFICATION §12). This is the executable
# checklist that proves the demand-driven tiled compiler is complete and has not
# regressed the M1 exact 2D CPU core. It asserts, in order, the FOUR M2 exit
# criteria, then re-runs the M1 op suite so a tiling change cannot silently break
# an op, and uploads the tile-count / performance artifacts.
#
# The four M2 criteria (each backed by a named differential/efficiency test):
#   1. tiled == whole-image, BIT-IDENTICAL for exact ops — the pointwise chain and
#      the neighbourhood (convolution) halo path both reproduce the whole-image
#      executor sample-for-sample across tile sizes incl. ragged edges
#      (crates/paintop-cpu tiled_pointwise + tiled_convolution; deterministic
#      tiled reductions in tiled_reductions).
#   2. ROI execution is differentially EQUIVALENT to full execution — perturbing an
#      input outside its backward-demanded region never changes a pixel inside the
#      demanded region (crates/paintop-core roi_differential).
#   3. a small masked edit on a 4K image touches ONLY the predicted, halo-expanded
#      tile set — the executor's executed-tile count is <= an independently computed
#      conservative prediction, and the localized run is bit-identical to a full run
#      inside the region (crates/paintop-cpu masked_edit_4k). Emits tiles.json.
#   4. cache replay performs ZERO unnecessary execution — a second run over an
#      unchanged plan recomputes no producer node and every output equals the
#      uncached output (crates/paintop-core cache_replay).
#
# Run from the workspace root. Exits non-zero on the first failed criterion.
set -euo pipefail

# Resolve to the repo/workspace root (this script lives in ci/).
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# Where the tile-count / performance artifacts are collected. CI points
# M2_GATE_ARTIFACT_DIR at the job's upload dir; locally it defaults under target/.
ARTIFACT_DIR="${M2_GATE_ARTIFACT_DIR:-$ROOT/target/m2-gate-artifacts}"
mkdir -p "$ARTIFACT_DIR"

# A single cargo test invocation: a crate, a test target, and the suite name (for
# the banner). All criteria run with --quiet so the artifact-bearing test's stdout
# is the only noise.
run_suite() {
  local crate="$1" target="$2"
  cargo test -p "$crate" --test "$target" --quiet
}

step() { printf '\n=== M2 gate: %s ===\n' "$1"; }

# ---------------------------------------------------------------------------
# 0. Layer-0 build hygiene + the whole analytic/property/differential suite.
#
# CI splits this across dedicated jobs, so set M2_GATE_SKIP_CHECK=1 there to avoid
# re-running the whole suite; locally (`just m2-gate`) it runs as criterion 0.
# ---------------------------------------------------------------------------
if [ "${M2_GATE_SKIP_CHECK:-0}" = "1" ]; then
  step "Layer-0 build hygiene + tests (just check) — SKIPPED (dedicated CI job)"
else
  step "Layer-0 build hygiene + tests (just check)"
  just check
fi

# ---------------------------------------------------------------------------
# 1. tiled == whole-image, bit-identical for exact ops.
# ---------------------------------------------------------------------------
step "criterion 1: tiled == whole-image (bit-identical for exact ops)"
run_suite paintop-cpu tiled_pointwise
run_suite paintop-cpu tiled_convolution
run_suite paintop-cpu tiled_reductions
echo "  ok: pointwise, neighbourhood-halo, and reduction tiling all match whole-image"

# ---------------------------------------------------------------------------
# 2. ROI execution differentially equivalent to full execution.
# ---------------------------------------------------------------------------
step "criterion 2: ROI execution == full execution (differential)"
run_suite paintop-core roi_differential
echo "  ok: perturbing outside the demanded region never changes the demanded output"

# ---------------------------------------------------------------------------
# 3. masked 4K edit touches only the predicted halo-expanded tile set.
#
# Point the artifact-bearing test at the upload dir so its tiles.json lands where
# CI can collect it. --nocapture surfaces the executed/predicted counts in the log.
# ---------------------------------------------------------------------------
step "criterion 3: masked 4K edit touches only the predicted (halo-expanded) tiles"
PAINTOP_TILE_ARTIFACT="$ARTIFACT_DIR/masked_edit_4k.tiles.json" \
  cargo test -p paintop-cpu --test masked_edit_4k --quiet
if [ ! -s "$ARTIFACT_DIR/masked_edit_4k.tiles.json" ]; then
  echo "  FAIL: masked-edit tile-count artifact was not written" >&2
  exit 1
fi
# The artifact's own self-checks must agree the executor stayed within prediction,
# both in aggregate and per node.
within="$(jq -r '.executed_within_prediction' "$ARTIFACT_DIR/masked_edit_4k.tiles.json")"
per_node="$(jq -r '.per_node_within_prediction' "$ARTIFACT_DIR/masked_edit_4k.tiles.json")"
executed="$(jq -r '.tiles.executed' "$ARTIFACT_DIR/masked_edit_4k.tiles.json")"
predicted="$(jq -r '.aggregate_prediction' "$ARTIFACT_DIR/masked_edit_4k.tiles.json")"
grid="$(jq -r '.grid_tiles' "$ARTIFACT_DIR/masked_edit_4k.tiles.json")"
if [ "$within" != "true" ] || [ "$per_node" != "true" ]; then
  echo "  FAIL: executed-tile count exceeded the conservative prediction" >&2
  exit 1
fi
echo "  ok: executed=$executed within prediction=$predicted of $grid total 4K tiles"

# ---------------------------------------------------------------------------
# 4. cache replay performs zero unnecessary execution.
# ---------------------------------------------------------------------------
step "criterion 4: cache replay performs zero unnecessary execution"
run_suite paintop-core cache_replay
echo "  ok: a replay over an unchanged plan recomputes no node; hits equal uncached output"

# ---------------------------------------------------------------------------
# 5. M1 must NOT regress: re-run the M1 op conformance + verify-op gate.
#
# A tiling/ROI/cache change must keep every M1 op exact. We re-run the M1 gate's
# op-list + verify-op + conformance criteria (skipping its `just check`, already
# run as criterion 0 here, to avoid a double compile).
# ---------------------------------------------------------------------------
step "criterion 5 (no-regression): the full M1 op suite still passes"
M1_GATE_SKIP_CHECK=1 bash ci/m1-gate.sh
echo "  ok: all M1 ops still pass verify-op + conformance after the M2 changes"

# ---------------------------------------------------------------------------
# Upload: collect the artifacts. (In CI the dir is the upload target; locally it
# is just a stable path under target/.)
# ---------------------------------------------------------------------------
step "artifacts collected under $ARTIFACT_DIR"
ls -1 "$ARTIFACT_DIR"

step "ALL FOUR M2 EXIT CRITERIA PASSED (and M1 did not regress)"
