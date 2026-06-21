#!/usr/bin/env bash
# M1 exit-criteria gate (plan.md §19 "M1 — exact 2D CPU core"; OP_CATALOG §18 P0
# conformance set; AGENT_VERIFICATION §6). This is the executable checklist that
# proves the exact 2D CPU core is complete: every P0 operation has a manifest and
# passes verify-op, the analytic/property tests pass, and TWO distinct
# agent-authored fixture edits run end-to-end with a passing
# no-change-outside-mask localization assertion and a reproducible rerun hash.
#
# It asserts, in order:
#   1. Layer-0 build hygiene + the whole analytic/property test suite (`just
#      check`: fmt, the lint wall, tests, docs) — this also runs both
#      conformance integration tests (blemish + banner) and the leaking variants.
#   2. `op list` exposes the full P0 conformance set (OP_CATALOG §18), every entry
#      backed by a discoverable manifest.
#   3. Every P0 op manifest passes `cargo xtask verify-op`.
#   4. The unauthorized-pixel assertion is demonstrated: the NEW banner scenario
#      (gradient + blur + polygon composite, distinct from the blemish keystone)
#      runs to exit 0 with a passing `assert.no_change_outside_mask` and a
#      reproducible rerun hash, and its deliberately-leaking variant fails with
#      the stable assertion exit class (6).
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

step() { printf '\n=== M1 gate: %s ===\n' "$1"; }

# The OP_CATALOG §18 P0 conformance set: the first stable operation set. Every one
# of these must have a manifest and pass verify-op for M1 to be complete.
P0_OPS=(
  io.decode_image io.encode_image image.create image.inspect
  color.convert color.adjust alpha.premultiply alpha.unpremultiply
  mask.empty mask.full mask.rect mask.ellipse mask.polygon
  mask.invert mask.union mask.intersect mask.subtract mask.bounds
  image.crop image.pad image.resize
  paint.fill paint.linear_gradient paint.radial_gradient paint.gaussian_splats
  composite.over composite.masked_replace composite.blend
  filter.convolve filter.gaussian_blur
  analyze.statistics analyze.histogram analyze.diff analyze.changed_bounds
  assert.finite assert.range assert.no_change_outside_mask assert.changed_bounds
  assert.alpha_valid debug.materialize
)

# ---------------------------------------------------------------------------
# 1. Layer-0 build hygiene + analytic/property/conformance test suite.
#
# CI splits this across dedicated jobs (build-hygiene + conformance), so set
# M1_GATE_SKIP_CHECK=1 there to avoid re-running the whole suite; locally
# (`just m1-gate`) it runs as the first criterion.
# ---------------------------------------------------------------------------
if [ "${M1_GATE_SKIP_CHECK:-0}" = "1" ]; then
  step "Layer-0 build hygiene + tests (just check) — SKIPPED (run in a dedicated CI job)"
else
  step "Layer-0 build hygiene + tests (just check)"
  just check
fi

# ---------------------------------------------------------------------------
# 2. op list exposes the full P0 conformance set, each backed by a manifest.
# ---------------------------------------------------------------------------
step "op list exposes the full P0 conformance set (OP_CATALOG §18)"
listed="$(run_paintop op list --format json | jq -r '.operations[].id' | sed 's/@[0-9]*$//' | sort -u)"
missing=0
for op in "${P0_OPS[@]}"; do
  if ! grep -qx "$op" <<<"$listed"; then
    echo "  FAIL: P0 op '$op' is not exposed by 'op list'" >&2
    missing=1
  fi
  if [ ! -f "ops/manifests/${op}@1.json" ]; then
    echo "  FAIL: P0 op '$op' has no manifest under ops/manifests/" >&2
    missing=1
  fi
done
if [ "$missing" -ne 0 ]; then exit 1; fi
echo "  ok: all ${#P0_OPS[@]} P0 ops are listed and have a manifest"

# ---------------------------------------------------------------------------
# 3. Every P0 op manifest passes verify-op.
# ---------------------------------------------------------------------------
step "verify-op for every P0 op manifest"
if [ -z "${XTASK_BIN:-}" ]; then cargo build -p xtask --quiet; fi
for op in "${P0_OPS[@]}"; do
  manifest="ops/manifests/${op}@1.json"
  echo "  verify-op ${op}@1"
  run_xtask verify-op --manifest "$manifest" "${op}@1"
done
echo "  ok: every P0 op passed verify-op"

# ---------------------------------------------------------------------------
# 4. The NEW agent-authored fixture edit: green loop, reproducible rerun hash,
#    passing localization assertion, and a leaking variant that fails with 6.
# ---------------------------------------------------------------------------
mkdir -p conformance/out

step "banner scenario runs green (gradient + blur + polygon composite)"
set +e
out_a="$(run_paintop run conformance/plans/banner.json --bundle target/m1-gate-banner-a)"
code_a=$?
set -e
if [ "$code_a" -ne 0 ]; then echo "  FAIL: banner run exited $code_a" >&2; exit 1; fi
status_a="$(jq -r '.status' <<<"$out_a")"
if [ "$status_a" != "success" ]; then echo "  FAIL: banner status '$status_a' (expected success)" >&2; exit 1; fi
hash_a="$(jq -r '.output_content_hash' <<<"$out_a")"
echo "  ok: banner ran green; output hash $hash_a"

step "banner localization assertion passed (no change outside the polygon)"
loc_status="$(jq -r '.assertions[] | select(.id=="localized") | .status' target/m1-gate-banner-a/assertions.json)"
loc_outside="$(jq -r '.assertions[] | select(.id=="localized") | .metrics.changed_pixels_outside' target/m1-gate-banner-a/assertions.json)"
if [ "$loc_status" != "passed" ] || [ "$loc_outside" != "0" ]; then
  echo "  FAIL: localized assertion status='$loc_status' changed_outside='$loc_outside'" >&2
  exit 1
fi
echo "  ok: assert.no_change_outside_mask passed with 0 pixels changed outside"

step "banner rerun is hash-reproducible"
out_b="$(run_paintop run conformance/plans/banner.json --bundle target/m1-gate-banner-b)"
hash_b="$(jq -r '.output_content_hash' <<<"$out_b")"
if [ "$hash_a" != "$hash_b" ]; then
  echo "  FAIL: output content hash drifted across reruns: $hash_a != $hash_b" >&2
  exit 1
fi
if ! cmp -s target/m1-gate-banner-a/outputs/final.png target/m1-gate-banner-b/outputs/final.png; then
  echo "  FAIL: rerun produced a non-byte-identical output image" >&2
  exit 1
fi
echo "  ok: rerun produced an identical hash and byte-identical output PNG"

step "banner leaking variant fails with the stable assertion exit class (6)"
set +e
run_paintop run conformance/plans/banner-leak.json --bundle target/m1-gate-banner-leak >/dev/null
code_leak=$?
set -e
if [ "$code_leak" -ne 6 ]; then
  echo "  FAIL: leaking banner exited $code_leak (expected 6)" >&2
  exit 1
fi
leak_status="$(jq -r '.assertions[] | select(.id=="localized") | .status' target/m1-gate-banner-leak/assertions.json)"
if [ "$leak_status" != "failed" ]; then
  echo "  FAIL: leaking variant localized assertion status='$leak_status' (expected failed)" >&2
  exit 1
fi
if [ ! -f target/m1-gate-banner-leak/replays/localized.json ]; then
  echo "  FAIL: leaking variant did not emit a minimal replay" >&2
  exit 1
fi
echo "  ok: leaking variant failed with exit 6, recorded the leak, and emitted a replay"

step "ALL M1 EXIT CRITERIA PASSED"
