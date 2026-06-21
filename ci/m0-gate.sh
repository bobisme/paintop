#!/usr/bin/env bash
# M0 exit-criteria gate (plan.md §19 "M0 — repository and contracts" + the
# first-half-M1 "all operations have manifests" criterion). This is the
# executable checklist that proves the M0 foundation is viable; the CI `m0-gate`
# job and a fresh-clone walkthrough both run it.
#
# It asserts, in order:
#   1. Layer-0 build hygiene (`just check`: fmt, the lint wall, tests, docs).
#   2. The §19 CLI validate criteria, with the exact exit codes plan.md demands:
#        paintop validate fixtures/plans/empty-valid.json     -> 0
#        paintop validate fixtures/plans/unknown-field.json   -> fails (2)
#   3. `op list` exposes all 14 MVP ops (M1: every operation has a manifest).
#   4. Every MVP op manifest passes `cargo xtask verify-op`.
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

step() { printf '\n=== M0 gate: %s ===\n' "$1"; }

# ---------------------------------------------------------------------------
# 1. Layer-0 build hygiene.
# ---------------------------------------------------------------------------
step "Layer-0 build hygiene (just check)"
just check

# ---------------------------------------------------------------------------
# 2. §19 CLI validate criteria, with the stable §15.4 exit classes.
# ---------------------------------------------------------------------------
step "paintop validate fixtures/plans/empty-valid.json -> 0"
run_paintop validate fixtures/plans/empty-valid.json
echo "  ok: empty-valid plan validated (exit 0)"

step "paintop validate fixtures/plans/unknown-field.json -> must fail predictably"
set +e
run_paintop validate fixtures/plans/unknown-field.json
code=$?
set -e
if [ "$code" -ne 2 ]; then
  echo "  FAIL: expected exit 2 (schema class) on an unknown field, got $code" >&2
  exit 1
fi
echo "  ok: unknown-field plan rejected with the stable schema exit class (2)"

# ---------------------------------------------------------------------------
# 3. op list exposes all 14 MVP ops.
# ---------------------------------------------------------------------------
step "op list exposes all 14 MVP operations"
op_count=$(run_paintop op list --format json | jq '.operations | length')
if [ "$op_count" -ne 14 ]; then
  echo "  FAIL: op list shows $op_count operations, expected 14" >&2
  exit 1
fi
echo "  ok: op list shows all 14 MVP operations"

# ---------------------------------------------------------------------------
# 4. Every MVP op manifest passes verify-op (M1: all operations have manifests).
# ---------------------------------------------------------------------------
step "verify-op for every MVP op manifest"
# Build xtask once so per-op runs are fast and a compile error fails clearly.
if [ -z "${XTASK_BIN:-}" ]; then cargo build -p xtask --quiet; fi
for manifest in ops/manifests/*.json; do
  op=$(jq -r '.id' "$manifest")
  echo "  verify-op $op"
  run_xtask verify-op --manifest "$manifest" "$op"
done
echo "  ok: every MVP op passed verify-op"

step "ALL M0 EXIT CRITERIA PASSED"
