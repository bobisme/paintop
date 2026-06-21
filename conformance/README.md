# Conformance: "an agent edits a blemish" (the M0 keystone)

This directory holds the first end-to-end conformance scenario — the deliberately
boring, strict, **non-SDF** MVP loop from `notes/paintop-plan/plan.md` §25 and
`notes/paintop-plan/docs/IR_SPEC.md` §20 (the rewritten, non-SDF form per
`M0_DECISIONS.md` D1/D2). If the runtime can run, diagnose, and reproduce this
workflow, the M0 foundation is viable (`plan.md` §19 exit criteria).

## Layout

```
conformance/
├── README.md                  ← you are here
├── fixtures/
│   └── blemish-input.png      deterministic procedural RGBA8 input (256×192)
├── plans/
│   ├── blemish.json           the M0 keystone plan: the 14-op non-SDF loop
│   ├── blemish-leak.json      its negative variant: MUST fail with exit 6
│   ├── banner.json            the M1 scenario: gradient + blur + polygon composite
│   └── banner-leak.json       its negative variant: MUST fail with exit 6
└── out/                       io.encode_image sink (gitignored, regenerated)
```

## The M1 scenario: "an agent paints a masked gradient banner"

`plans/banner.json` is the **second, distinct** agent-authored fixture edit
demanded by the M1 exit criteria (`plan.md` §19 M1; `OP_CATALOG` §18;
`AGENT_VERIFICATION` §6). It is deliberately different from the blemish keystone
— it exercises the **new M1 P0 operations** end-to-end:

```
io.decode_image            decode the fixture PNG → image
image.inspect              record extent / ranges / content hash (evidence)
color.convert  (srgb→linear-srgb)
alpha.premultiply          → the linear, premultiplied BASE
paint.linear_gradient      synthesize a 3-stop linear-light color gradient        [NEW]
alpha.premultiply          premultiply the gradient edit layer
filter.gaussian_blur       soften the gradient (reference Gaussian, sigma 3.5)     [NEW]
mask.polygon               authorized banner polygon (nonzero winding)            [NEW]
composite.masked_replace   blend the blurred gradient over the base THROUGH it
alpha.unpremultiply
color.convert  (linear-srgb→srgb)
analyze.diff               before/after diff summary (evidence)
assert.no_change_outside_mask   the authorization-boundary check
assert.finite              every composited sample is finite
io.encode_image            write the final PNG
```

Like the keystone, `composite.masked_replace` is the **single authorization
boundary** and `assert.no_change_outside_mask` proves nothing leaked outside the
authorized polygon. Running it twice is byte-identical and hash-stable.
`plans/banner-leak.json` authorizes the wide banner polygon but checks against a
narrow inner polygon, so the assertion legitimately fails with exit class 6 and
emits a minimal replay. All five behaviours (green loop, passing localization,
reproducible rerun, leaking variant, full bundle) are asserted by
`crates/paintop-cpu/tests/banner_conformance.rs` and the `just m1-gate` target.

The M1 exit gate is the executable script `ci/m1-gate.sh` (`just m1-gate`): it
runs `just check`, asserts `op list` exposes the full `OP_CATALOG` §18 P0 set
(each with a manifest), runs `verify-op` for every P0 op, and drives this banner
scenario through its green run, reproducible rerun, and leaking variant.

## What the plan does

`plans/blemish.json` composes every MVP operation (`M0_DECISIONS` D2), wired
through the real op ports:

```
io.decode_image            decode the fixture PNG → image
image.inspect              record extent / ranges / content hash (evidence)
color.convert  (srgb→linear-srgb)
alpha.premultiply          → the linear, premultiplied BASE
mask.ellipse               analytic smoothstep-feathered authorized ellipse
paint.gaussian_splats      paint a bounded splat batch on the EDIT LAYER
color.adjust               linear-light grade of the edit layer
composite.masked_replace   blend edit over base THROUGH the authorized mask
alpha.unpremultiply
color.convert  (linear-srgb→srgb)
analyze.diff               before/after diff summary (evidence)
assert.no_change_outside_mask   the authorization-boundary check
assert.finite              every composited sample is finite
io.encode_image            write the final PNG
```

`composite.masked_replace` is the **single authorization boundary**, and
`assert.no_change_outside_mask` is the check that nothing changed outside it.

## Running it

From the **workspace root** (paths in the plan are workspace-relative):

```bash
mkdir -p conformance/out
cargo run -p paintop-cli -- run conformance/plans/blemish.json \
  --bundle target/conformance-bundle
```

Expected: exit code `0` and a one-line JSON outcome on stdout with
`"status":"success"`, a `plan_semantic_hash`, and an `output_content_hash`. The
evidence bundle under `target/conformance-bundle/` contains:

```
manifest.json          semantic hash, status, exit code, outputs, failures
normalized-plan.json   the exact §17-normalized graph that ran
input-... / graph.dot  the dependency graph (DOT)
trace.jsonl            per-node structured trace
assertions.json        every assertion's verdict + thresholds + metrics
outputs/final.png      the encoded result
masks/ intermediates/  the materialized mask and edit-layer previews
contact-sheet.png      before / after / amplified-diff
diffs/final-diff.png   the standalone amplified diff
```

### Reproducibility

Running the plan a second time produces a **byte-identical** `outputs/final.png`
and an **identical** `plan_semantic_hash` and `output_content_hash`. This is
asserted by `second_run_is_byte_identical_and_hash_stable` in
`crates/paintop-cpu/tests/blemish_conformance.rs`.

### The leaking variant (negative test)

`plans/blemish-leak.json` is identical except the composite authorizes the *wide*
ellipse while `assert.no_change_outside_mask` checks against a *narrow* ellipse,
so the edit legitimately changes pixels the assertion mask forbids:

```bash
cargo run -p paintop-cli -- run conformance/plans/blemish-leak.json \
  --bundle target/conformance-leak
echo "exit: $?"      # 6  (the stable assertion exit class, plan.md §15.4)
```

The run still writes a complete bundle, including `assertions.json` (with the
`localized` assertion `failed` and its `changed_pixels_outside` metric) and a
minimal reproducer at `replays/localized.json`.

## The fixture is deterministic

`fixtures/blemish-input.png` is generated from a fixed integer formula (a smooth
two-axis gradient with an off-center darker lobe) — no external assets, no RNG.
The test `blemish_conformance.rs` regenerates it when missing and otherwise
asserts the checked-in bytes match the formula, so the input can never silently
drift.

## In CI

The `Keystone conformance` job in `.github/workflows/ci.yml` runs all three
checks (green loop, reproducible rerun, leaking variant) and uploads the evidence
bundle. `just check` also runs them as part of `cargo test --workspace`.
