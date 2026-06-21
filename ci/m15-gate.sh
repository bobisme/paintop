#!/usr/bin/env bash
# M1.5 exit-criteria gate (plan.md §25 original SDF variant; M0_DECISIONS D1
# "SDF lands second"; OP_CATALOG §4; ALIEN_OPS §2). This is the executable
# checklist that proves the exact-EDT + SDF mask calculus is complete and that
# the deferred SDF north-star scenario runs end-to-end.
#
# It asserts, in order:
#   1. Layer-0 build hygiene + the whole analytic/property/differential test
#      suite (`just check`: fmt, the lint wall, tests, docs) — this also runs the
#      EDT brute-force differential, the offset-composition and boolean SDF law
#      property suites, the mask-topology tests, and all four conformance
#      integration tests (blemish + banner + SDF north-star + leaking variants).
#   2. `op list` exposes the full M1.5 SDF + topology op set, each backed by a
#      discoverable manifest.
#   3. Every NEW M1.5 op manifest passes `cargo xtask verify-op`.
#   4. The SDF north-star scenario (mask.ellipse → mask.to_sdf → sdf.offset →
#      sdf.to_mask feather chain feeding the touch-up loop) runs to exit 0 with a
#      passing `assert.no_change_outside_mask` localization assertion (zero pixels
#      changed outside the SDF-authorized region) and a byte-identical reproducible
#      rerun hash.
#
# Run from the workspace root. Exits non-zero on the first failed criterion.
set -euo pipefail

# Resolve to the repo/workspace root (this script lives in ci/).
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# How to invoke the CLI and xtask. CI/local default to building through cargo;
# override PAINTOP_BIN / XTASK_BIN with prebuilt binaries to avoid nested cargo.
run_paintop() {
  if [ -n "${PAINTOP_BIN:-}" ]; then "$PAINTOP_BIN" "$@"; else cargo run --quiet -p paintop-cli -- "$@"; fi
}
run_xtask() {
  if [ -n "${XTASK_BIN:-}" ]; then "$XTASK_BIN" "$@"; else cargo run --quiet -p xtask -- "$@"; fi
}

step() { printf '\n=== M1.5 gate: %s ===\n' "$1"; }

# The M1.5 SDF + topology op set: the new operations landed by this milestone.
# Every one of these must have a manifest and pass verify-op for M1.5 to be
# complete. (mask.grow/shrink/feather are macro ops; sdf.* are the new domain;
# mask.to_sdf and the topology ops register in the mask domain.)
M15_OPS=(
  mask.to_sdf
  sdf.to_mask sdf.offset
  sdf.union sdf.intersect sdf.subtract
  mask.grow mask.shrink mask.feather
  mask.connected_components mask.fill_holes mask.remove_components
)

# ---------------------------------------------------------------------------
# 1. Layer-0 build hygiene + analytic/property/differential/conformance suite.
#
# CI splits this across dedicated jobs, so set M15_GATE_SKIP_CHECK=1 there to
# avoid re-running the whole suite; locally (`just m15-gate`) it runs as the
# first criterion.
# ---------------------------------------------------------------------------
if [ "${M15_GATE_SKIP_CHECK:-0}" = "1" ]; then
  step "Layer-0 build hygiene + tests (just check) — SKIPPED (run in a dedicated CI job)"
else
  step "Layer-0 build hygiene + tests (just check)"
  just check
fi

# ---------------------------------------------------------------------------
# 2. op list exposes the full M1.5 SDF + topology op set, each with a manifest.
# ---------------------------------------------------------------------------
step "op list exposes the full M1.5 SDF + topology op set (OP_CATALOG §4)"
listed="$(run_paintop op list --format json | jq -r '.operations[].id' | sed 's/@[0-9]*$//' | sort -u)"
missing=0
for op in "${M15_OPS[@]}"; do
  if ! grep -qx "$op" <<<"$listed"; then
    echo "  FAIL: M1.5 op '$op' is not exposed by 'op list'" >&2
    missing=1
  fi
  if [ ! -f "ops/manifests/${op}@1.json" ]; then
    echo "  FAIL: M1.5 op '$op' has no manifest under ops/manifests/" >&2
    missing=1
  fi
done
if [ "$missing" -ne 0 ]; then exit 1; fi
echo "  ok: all ${#M15_OPS[@]} M1.5 ops are listed and have a manifest"

# ---------------------------------------------------------------------------
# 3. Every NEW M1.5 op manifest passes verify-op.
# ---------------------------------------------------------------------------
step "verify-op for every M1.5 op manifest"
if [ -z "${XTASK_BIN:-}" ]; then cargo build -p xtask --quiet; fi
for op in "${M15_OPS[@]}"; do
  manifest="ops/manifests/${op}@1.json"
  echo "  verify-op ${op}@1"
  run_xtask verify-op --manifest "$manifest" "${op}@1"
done
echo "  ok: every M1.5 op passed verify-op"

# ---------------------------------------------------------------------------
# 4. The SDF north-star scenario: mask.ellipse → mask.to_sdf → sdf.offset →
#    sdf.to_mask feather chain feeding the touch-up loop. Green loop, passing
#    localization assertion, reproducible byte-identical rerun hash.
# ---------------------------------------------------------------------------
mkdir -p conformance/out

step "SDF north-star scenario runs green (ellipse → to_sdf → offset → to_mask feather chain)"
set +e
out_a="$(run_paintop run conformance/plans/blemish-sdf.json --bundle target/m15-gate-sdf-a)"
code_a=$?
set -e
if [ "$code_a" -ne 0 ]; then echo "  FAIL: SDF north-star run exited $code_a" >&2; exit 1; fi
status_a="$(jq -r '.status' <<<"$out_a")"
if [ "$status_a" != "success" ]; then echo "  FAIL: SDF north-star status '$status_a' (expected success)" >&2; exit 1; fi
hash_a="$(jq -r '.output_content_hash' <<<"$out_a")"
echo "  ok: SDF north-star ran green; output hash $hash_a"

step "SDF north-star localization assertion passed (no change outside the SDF-authorized region)"
loc_status="$(jq -r '.assertions[] | select(.id=="localized") | .status' target/m15-gate-sdf-a/assertions.json)"
loc_outside="$(jq -r '.assertions[] | select(.id=="localized") | .metrics.changed_pixels_outside' target/m15-gate-sdf-a/assertions.json)"
if [ "$loc_status" != "passed" ] || [ "$loc_outside" != "0" ]; then
  echo "  FAIL: localized assertion status='$loc_status' changed_outside='$loc_outside'" >&2
  exit 1
fi
echo "  ok: assert.no_change_outside_mask passed with 0 pixels changed outside"

step "SDF north-star rerun is hash-reproducible"
out_b="$(run_paintop run conformance/plans/blemish-sdf.json --bundle target/m15-gate-sdf-b)"
hash_b="$(jq -r '.output_content_hash' <<<"$out_b")"
if [ "$hash_a" != "$hash_b" ]; then
  echo "  FAIL: output content hash drifted across reruns: $hash_a != $hash_b" >&2
  exit 1
fi
if ! cmp -s target/m15-gate-sdf-a/outputs/final.png target/m15-gate-sdf-b/outputs/final.png; then
  echo "  FAIL: rerun produced a non-byte-identical output image" >&2
  exit 1
fi
echo "  ok: rerun produced an identical hash and byte-identical output PNG"

step "ALL M1.5 EXIT CRITERIA PASSED"
