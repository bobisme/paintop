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

# Install the CLI locally (post-merge step per AGENTS.md).
install:
    cargo install --path crates/paintop-cli --locked
