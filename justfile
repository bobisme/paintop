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
