# paintop repository tasks.
# `just check` is the bone exit gate (AGENT_VERIFICATION §2.1): it must pass
# CLEAN — zero warnings, zero failures — before any bone is committed/merged.

# List available recipes.
default:
    @just --list

# Full quality gate: format check, the workspace lint wall, tests, and docs.
# Mirrors AGENT_VERIFICATION §2.1 Layer-0 build hygiene.
check: fmt-check clippy test doc

# Verify formatting without modifying files.
fmt-check:
    cargo fmt --all -- --check

# Apply rustfmt to the whole workspace.
fmt:
    cargo fmt --all

# Workspace lint wall: pedantic + nursery + unwrap_used at deny, warnings fatal.
clippy:
    cargo clippy --workspace --all-targets --all-features -- -D warnings

# Run the whole test suite.
test:
    cargo test --workspace

# Build the docs for every crate (no third-party deps), denying broken links.
doc:
    cargo doc --workspace --no-deps

# Build the whole workspace.
build:
    cargo build --workspace

# M0 exit-criteria gate (plan.md §19): runs `just check`, the §19 CLI validate
# criteria with their exact exit codes, `op list` (all 14 MVP ops), and
# `verify-op` for every MVP op manifest. This is the executable M0 checklist and
# the fresh-clone walkthrough entry point.
m0-gate:
    bash ci/m0-gate.sh

# M1 exit-criteria gate (plan.md §19 M1; OP_CATALOG §18 P0 conformance set):
# runs `just check`, asserts `op list` exposes the full P0 set (each with a
# manifest), runs `verify-op` for every P0 op, and runs the NEW agent-authored
# banner scenario (gradient + blur + polygon composite) green with a passing
# no-change-outside-mask assertion + reproducible rerun hash + a leaking variant
# that fails with exit 6. This is the executable M1 checklist.
m1-gate:
    bash ci/m1-gate.sh

# M1.5 exit-criteria gate (plan.md §25 original SDF variant; M0_DECISIONS D1;
# OP_CATALOG §4): runs `just check` (which includes the EDT brute-force
# differential, offset-composition + boolean SDF law property suites, the
# mask-topology tests, and the SDF north-star conformance integration test),
# asserts `op list` exposes the full M1.5 SDF + topology op set (each with a
# manifest), runs `verify-op` for every NEW M1.5 op, and runs the deferred SDF
# north-star scenario (mask.ellipse -> mask.to_sdf -> sdf.offset -> sdf.to_mask
# feather chain feeding the touch-up loop) green with a passing
# no-change-outside-mask assertion + reproducible byte-identical rerun hash.
# This is the executable M1.5 checklist.
m15-gate:
    bash ci/m15-gate.sh
# M2 exit-criteria gate (plan.md §19 M2; §11 demand/tile model; AGENT_VERIFICATION
# §12): runs `just check`, then asserts the FOUR M2 criteria — (1) tiled ==
# whole-image bit-identical for exact ops, (2) ROI execution differentially
# equivalent to full execution, (3) a small masked 4K edit touches only the
# predicted halo-expanded tile set (executed-tile count <= conservative
# prediction), (4) cache replay performs zero unnecessary execution — re-runs the
# M1 op suite to prove no regression, and collects the tile-count / performance
# artifacts. This is the executable M2 checklist.
m2-gate:
    bash ci/m2-gate.sh

# M3 exit-criteria gate (plan.md §19 M3 "optimized CPU + wgpu backends"; §12
# backend strategy): runs `just check`, then asserts the FOUR M3 criteria —
# (1) every GPU op passes its tolerance contract vs the cpu.reference oracle
# (cpu.optimized always; wgpu differentials on hardware, adapter-skipped cleanly
# otherwise), (2) no unplanned readback in a fully GPU-compatible chain
# (adapter-gated), (3) perf baselines emitted + collected as CI artifacts
# (machine-tolerant, no absolute wall-clock), (4) GPU absence yields a clean
# fallback / explicit unsupported error (ALWAYS tested, even GPU-less) — then
# re-runs the M1/M1.5/M2 gates to prove no regression and collects the perf +
# backend artifacts. This is the executable M3 checklist.
m3-gate:
    bash ci/m3-gate.sh

# Run `cargo xtask verify-op` for every MVP op manifest under ops/manifests/.
verify-ops:
    #!/usr/bin/env bash
    set -euo pipefail
    cargo build -p xtask --quiet
    for manifest in ops/manifests/*.json; do
        op=$(jq -r '.id' "$manifest")
        cargo run -p xtask --quiet -- verify-op --manifest "$manifest" "$op"
    done

# Install the CLI locally (post-merge step per AGENTS.md).
install:
    cargo install --path crates/paintop-cli --locked

# Performance baselines (plan.md §19 M3 exit criterion 3; bn-7k0).
#
# Sweeps the optimized-CPU pointwise kernels (cpu.reference vs cpu.optimized) and
# the wgpu.separable Gaussian (adapter-gated; skipped cleanly with no GPU), and
# writes the (op, backend, size, throughput) artifact to target/verification/perf/.
# Built --release so the throughput is representative. Pass PERF_BASELINE to
# compare against a checked-in reference at PERF_THRESHOLD relative slack (a
# regression beyond the threshold exits non-zero); machine-tolerant — no absolute
# wall-clock is asserted. See ci/perf/README.md.
PERF_MACHINE := env_var_or_default("PAINTOP_PERF_MACHINE", "local")
PERF_THRESHOLD := env_var_or_default("PERF_THRESHOLD", "0.25")
PERF_OUT := env_var_or_default("PERF_OUT", "target/verification/perf/baseline.json")
PERF_BASELINE := env_var_or_default("PERF_BASELINE", "")

perf-baseline:
    #!/usr/bin/env bash
    set -euo pipefail
    args=(perf-baseline --out "{{PERF_OUT}}" --machine "{{PERF_MACHINE}}" --threshold "{{PERF_THRESHOLD}}")
    if [ -n "{{PERF_BASELINE}}" ]; then
        args+=(--baseline "{{PERF_BASELINE}}")
    fi
    cargo run --release -p xtask -- "${args[@]}"

# ----------------------------------------------------------------------------
# Fuzzing (plan.md §19 M0; AGENT_VERIFICATION §2.1/§2.2).
#
# The fuzz harness lives in the detached `fuzz/` crate (its own workspace) so
# the lint wall and `just check` never compile these libFuzzer targets. These
# recipes are a bounded smoke / nightly gate, NOT part of `just check`.
#
# Requirements: a nightly toolchain (`rustup toolchain install nightly`) and
# cargo-fuzz (`cargo install cargo-fuzz`). The `FUZZ_NIGHTLY` variable selects
# the nightly channel; override it to pin a known-good date in CI.
# ----------------------------------------------------------------------------

# Nightly channel used to build the libFuzzer targets. Override in CI to pin.
FUZZ_NIGHTLY := env_var_or_default("FUZZ_NIGHTLY", "nightly")

# Seconds each target runs in the bounded smoke job. Keep small for CI.
FUZZ_SECONDS := env_var_or_default("FUZZ_SECONDS", "30")

# Build both fuzz targets without running them (cheap CI smoke / compile gate).
fuzz-build:
    cargo +{{FUZZ_NIGHTLY}} fuzz build --fuzz-dir fuzz

# Bounded smoke run: fuzz each target for FUZZ_SECONDS, seeded from the checked-in
# corpus. Crash artifacts (if any) land under `fuzz/artifacts/<target>/`; see
# `fuzz/README.md` for reproduction. Exits non-zero on a crash, failing CI.
fuzz-smoke: fuzz-build
    cargo +{{FUZZ_NIGHTLY}} fuzz run --fuzz-dir fuzz plan_parse -- \
        -max_total_time={{FUZZ_SECONDS}} -rss_limit_mb=2048
    cargo +{{FUZZ_NIGHTLY}} fuzz run --fuzz-dir fuzz png_decode -- \
        -max_total_time={{FUZZ_SECONDS}} -rss_limit_mb=2048

# List the discovered fuzz targets.
fuzz-list:
    cargo +{{FUZZ_NIGHTLY}} fuzz list --fuzz-dir fuzz
