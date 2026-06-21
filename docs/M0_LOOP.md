# Running the M0 loop: author → run → diagnose → reproduce

This is the agent-facing guide for the M0 milestone (`notes/paintop-plan/plan.md`
§19). It shows a fresh agent how to author a plan, run it through the real CLI,
read the evidence to diagnose a failure, and prove a reproducible rerun — using
only the CLI's machine surface and the checked-in evidence bundle.

Everything here runs from the **workspace root** (plan paths are
workspace-relative). The CLI binary is `paintop` (crate `paintop-cli`); run it
with `cargo run -p paintop-cli -- <args>` until you `just install` it.

---

## 0. Fresh-clone walkthrough (the smoke you can copy/paste)

A fresh clone is healthy if these four commands pass, in order:

```bash
just check                                              # Layer-0 hygiene: fmt + lint wall + tests + docs
just m0-gate                                            # the full M0 exit-criteria gate (see §5)
cargo run -p paintop-cli -- run conformance/plans/blemish.json \
  --bundle target/conformance-bundle                    # the keystone loop, end to end
cargo run -p paintop-cli -- run conformance/plans/blemish-leak.json \
  --bundle target/conformance-leak; echo "exit: $?"     # the negative variant -> exit 6
```

Expected: `just check` and `just m0-gate` exit `0`; the blemish run prints a
one-line JSON outcome with `"status":"success"` and a `plan_semantic_hash`; the
leak run exits `6` (the stable assertion class, `plan.md` §15.4) and still writes
a complete bundle. `conformance/README.md` documents the scenario in full.

---

## 1. Discover the op set (introspection only)

You never need to read the source to author a plan. The registry is queryable:

```bash
cargo run -p paintop-cli -- op list --format json          # all 14 MVP ops
cargo run -p paintop-cli -- op schema paint.gaussian_splats@1
```

`op list` returns every operation's id, one-line summary, and determinism tier.
`op schema <id>` returns the op's full manifest (ports, params, units, defaults,
`cpu.reference` impl id) **plus** the manifest JSON schema, so you can both read
the op and validate a manifest against it. The 14 MVP ops are the D2 set
(`notes/paintop-plan/M0_DECISIONS.md`): `io.decode_image`, `io.encode_image`,
`image.inspect`, `color.convert`, `alpha.premultiply`, `alpha.unpremultiply`,
`mask.ellipse`, `paint.gaussian_splats`, `color.adjust`,
`composite.masked_replace`, `analyze.diff`, `assert.no_change_outside_mask`,
`assert.finite`, `debug.materialize`.

---

## 2. Author a plan

A plan is a typed JSON operation graph. The smallest valid plan:

```json
{ "paintop": "1.0", "inputs": {}, "nodes": [], "exports": {} }
```

The keystone plan `conformance/plans/blemish.json` is the worked example: decode
→ inspect → sRGB→linear → premultiply → ellipse mask → splats on an edit layer →
adjust → `composite.masked_replace(edited, base, authorized_mask)` → unpremultiply
→ linear→sRGB → diff → `assert.no_change_outside_mask` → `assert.finite` → encode.
`composite.masked_replace` is the single authorization boundary;
`assert.no_change_outside_mask` is the check that nothing changed outside it.
`notes/paintop-plan/docs/IR_SPEC.md` §20 is the canonical (non-SDF) graph.

The parser is strict (`serde` `deny_unknown_fields`): a typo'd field is a hard
schema error, not a silent default — see §4.

---

## 3. Run a plan and emit evidence

```bash
mkdir -p conformance/out
cargo run -p paintop-cli -- run conformance/plans/blemish.json \
  --bundle target/conformance-bundle
```

`--bundle <dir>` writes the evidence bundle (`AGENT_VERIFICATION` §5):

| Path                       | What it holds                                            |
| -------------------------- | -------------------------------------------------------- |
| `manifest.json`            | semantic hash, status, exit code, outputs, failures      |
| `normalized-plan.json`     | the exact §17-normalized graph that ran                  |
| `graph.dot`                | the dependency DAG (DOT)                                  |
| `trace.jsonl`              | per-node structured trace                                |
| `assertions.json`          | every assertion's verdict, thresholds, metrics           |
| `outputs/final.png`        | the encoded result                                       |
| `masks/`, `intermediates/` | materialized mask + edit-layer previews                  |
| `contact-sheet.png`        | before / after / amplified-diff                          |
| `diffs/final-diff.png`     | the standalone amplified diff                            |

Other introspection commands (no execution):

```bash
cargo run -p paintop-cli -- validate plan.json     # parse + resolve + typecheck; exit 0 if valid
cargo run -p paintop-cli -- explain plan.json       # normalized plan + semantic hash
cargo run -p paintop-cli -- graph plan.json --out graph.svg
cargo run -p paintop-cli -- diff before.png after.png --mask allowed.png
```

---

## 4. Diagnose a failure

The CLI speaks a stable machine contract: **stdout is pure JSON, logs go to
stderr, and the exit code is one of the §15.4 stable classes.** Diagnose without
guessing:

- **Exit code** names the class: `2` schema, `6` assertion, `9` asset/IO. (Full
  table: `plan.md` §15.4.) A schema failure prints `{"error":{"class":...,
  "code":...,"message":...}}` with the offending field and line/column.
- **`assertions.json`** records each assertion's verdict plus its metrics. For a
  leaking edit, `assert.no_change_outside_mask` shows `"failed"` with a
  `changed_pixels_outside` count, and the run writes a minimal reproducer at
  `replays/<assertion>.json`.
- **`contact-sheet.png` / `diffs/final-diff.png`** show *where* pixels changed —
  the amplified diff makes a boundary leak visible at a glance.

Worked negative example (the leaking variant, which authorizes a wide ellipse but
asserts against a narrow one):

```bash
cargo run -p paintop-cli -- run conformance/plans/blemish-leak.json \
  --bundle target/conformance-leak
echo "exit: $?"                                  # 6
jq '.' target/conformance-leak/assertions.json   # localized: failed + changed_pixels_outside
```

---

## 5. The M0 exit-criteria gate

`just m0-gate` (script: `ci/m0-gate.sh`) is the executable M0 checklist
(`plan.md` §19 plus the first-half-M1 "all operations have manifests"
criterion). It runs:

1. `just check` — fmt, the clippy lint wall (pedantic + nursery + `unwrap_used`
   at deny), `cargo test --workspace`, `cargo doc --workspace --no-deps`.
2. `paintop validate fixtures/plans/empty-valid.json` → exit `0`.
3. `paintop validate fixtures/plans/unknown-field.json` → must fail with exit `2`.
4. `op list` exposes all 14 MVP ops.
5. `cargo xtask verify-op` for every manifest under `ops/manifests/`.

In CI the same criteria run as the `M0 exit criteria` job in
`.github/workflows/ci.yml`; the `Keystone conformance`, `Changed-op
verification`, and `Fuzz smoke` jobs cover the rest of §19.

### Fuzz smoke

The two attacker-facing boundaries (strict plan parse; PNG decode/encode) have
bounded libFuzzer smoke targets. They are off `just check` (nightly + libFuzzer)
and time-boxed:

```bash
just fuzz-build            # compile both targets (cheap)
just fuzz-smoke            # run each for FUZZ_SECONDS (default 30s), seeded from corpus
```

A crash exits non-zero and drops an artifact under `fuzz/artifacts/<target>/`;
`fuzz/README.md` documents reproduction (`cargo +nightly fuzz run --fuzz-dir fuzz
<target> <artifact>`).

---

## 6. Reproducibility

Determinism is a contract, not a hope. Running a plan twice produces a
**byte-identical** `outputs/final.png` and an **identical** `plan_semantic_hash`
and `output_content_hash`. This is asserted by
`second_run_is_byte_identical_and_hash_stable` in
`crates/paintop-cpu/tests/blemish_conformance.rs`, which `cargo test --workspace`
(hence `just check`) runs. Hashes are over canonical bytes (BLAKE3), never raw
`serde_json` output, so insignificant formatting can never change a hash.

---

## 7. Completion report (AGENT_VERIFICATION §6.1)

When you finish an op or a verified change, emit both a machine-readable and a
human-readable summary. Template:

```json
{
  "task": "<op-id or change> <impl/class>",
  "status": "complete",
  "changed_contracts": [],
  "tests_added": [
    "impulse_response",
    "constant_preserving",
    "translation_metamorphic",
    "boundary_modes"
  ],
  "verification_commands": [
    "cargo xtask verify-op --manifest ops/manifests/<op>.json <op-id>"
  ],
  "evidence": "target/verification/<op-id>/index.json",
  "benchmarks": { "1024x1024_ms": 0.0 },
  "known_limits": ["reference path is scalar"]
}
```

Rules (§6.2): do not weaken a tolerance to hide a failure without a derived error
bound; do not update goldens without inspecting and recording the diff; do not
disable a property test because the implementation is inconvenient; do not use an
optimized implementation as its own oracle; do not conflate "no panic" with
correctness; do not merge a new op without evidence-bundle support.

---

## 8. M0 / first-half-M1 exit-criteria checklist (with evidence paths)

| # | Criterion (`plan.md` §19) | Evidence |
| - | ------------------------- | -------- |
| 1 | `cargo fmt --check` clean | `just fmt-check` / CI `Layer-0` job |
| 2 | `cargo clippy … -D warnings` (pedantic + nursery + `unwrap_used`) | `just clippy` / CI `Layer-0` job; `clippy.toml` |
| 3 | `cargo test --workspace` green | `just test` / CI `Layer-0` job |
| 4 | `cargo doc --workspace --no-deps` clean | `just doc` / CI `Layer-0` job (`RUSTDOCFLAGS=-D warnings`) |
| 5 | `paintop validate fixtures/plans/empty-valid.json` → 0 | `ci/m0-gate.sh` step 2; CI `M0 exit criteria` job |
| 6 | `paintop validate fixtures/plans/unknown-field.json` fails predictably (exit 2) | `ci/m0-gate.sh` step 3; `crates/paintop-cli/tests/cli.rs::validate_unknown_field_exits_two_with_stable_code` |
| 7 | Every operation has a manifest (M1) | `ops/manifests/*.json` (14); `op list` shows 14 (`tests/cli.rs::op_list_emits_all_mvp_ops`) |
| 8 | Every op passes `verify-op` | `just verify-ops`; CI `M0 exit criteria` job; reports under `target/verification/<op>/` |
| 9 | Unauthorized-pixel assertion demonstrated | `conformance/plans/blemish-leak.json` → exit 6; `crates/paintop-cpu/tests/blemish_conformance.rs` |
| 10 | One agent can complete a fixture edit via CLI + evidence only | this doc + `conformance/README.md`; the green keystone run |
| 11 | Reproducible: byte-identical rerun + stable semantic hash | `second_run_is_byte_identical_and_hash_stable` in `blemish_conformance.rs` |
| 12 | Bounded fuzz smoke for the parse / PNG boundaries | `just fuzz-smoke`; CI `Fuzz smoke` job; `fuzz/README.md` |
