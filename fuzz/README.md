# paintop fuzz harness

Fuzz scaffolding for the two attacker-facing boundaries called out in
`plan.md` §19 (M0 exit) and `AGENT_VERIFICATION` §2.1/§2.2:

| Target        | Boundary under test                          | Entry point                          |
| ------------- | -------------------------------------------- | ------------------------------------ |
| `plan_parse`  | strict plan parse / canonicalize / hash      | `paintop_ir::parse_plan` (→ limits → scan → serde → normalize → BLAKE3) |
| `png_decode`  | PNG decode/encode under hardened limits      | `paintop_cpu::io::decode_png` / `encode_png` |

The harness is a **detached crate** (`fuzz/Cargo.toml` declares an empty
`[workspace]`), so the repository lint wall and `just check` never compile
these libFuzzer targets. Fuzzing is a bounded smoke / nightly gate, not part
of every local check.

## Requirements

```bash
rustup toolchain install nightly      # libFuzzer needs nightly
cargo install cargo-fuzz              # provides `cargo fuzz`
```

## Running

All recipes live in the repo `justfile` and target this `fuzz/` directory via
`--fuzz-dir fuzz`:

```bash
just fuzz-list      # show targets
just fuzz-build     # compile both targets (cheap CI compile gate)
just fuzz-smoke     # bounded run: each target for FUZZ_SECONDS (default 30s)
```

Tune the run without editing the justfile:

```bash
FUZZ_SECONDS=120 FUZZ_NIGHTLY=nightly-2025-11-21 just fuzz-smoke
```

To run a single target ad hoc for longer:

```bash
cargo +nightly fuzz run --fuzz-dir fuzz plan_parse -- -max_total_time=300
cargo +nightly fuzz run --fuzz-dir fuzz png_decode -- -max_total_time=300
```

## Seed corpus

Seeds are checked in under `fuzz/corpus/<target>/` and are the campaign's
starting inputs:

- `corpus/plan_parse/` — every malformed-plan fixture used by the IR/CLI tests
  (duplicate keys, `NaN`/`Infinity`/overflow numbers, unknown fields,
  depth/node-count/inline-payload limit cases) plus the valid plans, so the
  fuzzer starts from inputs that already reach deep into the pipeline.
- `corpus/png_decode/` — deterministic tiny valid PNGs (RGBA/RGB/gray) plus
  malformed, truncated, oversized (`bomb-header-100k.png`), zero-dimension, and
  unsupported-format (`unsupported-16bit-rgb.png`) inputs that exercise the
  decode limit and malformed-input branches.

## Crash artifacts and reproduction

cargo-fuzz writes any crashing input to `fuzz/artifacts/<target>/` (a
`crash-<sha1>` file). The job exits non-zero on a crash, failing CI. To
reproduce and triage locally:

```bash
# Re-run the exact crashing input under the target (prints the backtrace):
cargo +nightly fuzz run --fuzz-dir fuzz <target> fuzz/artifacts/<target>/crash-<sha1>

# Or minimize it to the smallest reproducer first:
cargo +nightly fuzz tmin --fuzz-dir fuzz <target> fuzz/artifacts/<target>/crash-<sha1>
```

Check the minimized reproducer into the corpus (and add a regression test
against the relevant `parse_plan` / `decode_png` entry point) when fixing the
bug, so the case is covered by `just check` going forward.

## What counts as a bug

These targets assert **liveness**, not a particular error code: for *any*
input the entry point must return `Ok`/`Err` without panicking, aborting,
overflowing, or exhausting memory. A classified rejection
(`E_DECODE_MALFORMED`, `E_MAX_DEPTH`, `E_DUPLICATE_KEY`, …) is the expected,
correct outcome for hostile input — not a finding.
