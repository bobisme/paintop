# paintop

`paintop` is a command-line runtime for deterministic, verifiable visual edits.

The project is aimed at coding agents and automation systems that need to edit images, and later material textures, without relying on unchecked pixel scripts or subjective visual inspection. A user or agent writes a typed JSON operation graph; `paintop` validates it, normalizes it, executes it, and emits outputs plus machine-checkable evidence.

## What It Is For

- Localized image touch-ups inside explicit masks, with a machine-checkable no-change-outside-mask guarantee.
- Typed image, mask, field, SDF, label-map, pyramid, and patch-field semantics with explicit color, alpha, and boundary policies.
- A CPU reference (the semantic oracle) plus optimized CPU and GPU backends validated against it within declared tolerances.
- Evidence bundles with normalized plans, traces, metrics, diffs, artifacts, and assertion results.
- Agent-facing verification workflows where failures are inspectable and replayable.
- Future surface-aware material editing for GLB assets, with rendering and geometric queries delegated to `msh`.

`paintop` is not a GUI editor, Photoshop clone, prompt-to-image system, or general-purpose image scripting sandbox.

## Current Status

Milestones **M0 through M4 are implemented** — **77 operations**, each passing the
`cargo xtask verify-op` multi-oracle suite, plus reference-grade CPU, optimized
CPU, and GPU backends. What landed, by milestone:

- **M0 — runtime spine.** Strict plan parser + op contracts, canonicalization +
  BLAKE3 semantic/content hashing, the evidence-bundle writer, the `verify-op`
  runner, and the end-to-end keystone touch-up loop (byte-identical reruns;
  no-change-outside-mask assertion).
- **M1 — exact 2D CPU core (P0 ops).** Blank-canvas creation, resize/crop/pad/
  flip/rotate, masks (rect/ellipse/polygon, boolean algebra, bounds), fills and
  gradients, composite (over/blend), convolution and Gaussian blur, statistics/
  histogram, and the range/alpha/changed-bounds assertions.
- **M1.5 — EDT + SDF mask calculus.** Exact Euclidean distance transform,
  `mask.to_sdf`, SDF offset/boolean algebra, grow/shrink/feather macros, and mask
  topology (connected components / fill holes / remove components) with a
  `LabelMap` resource.
- **M2 — graph compiler.** Backward ROI/demand propagation with dead-node
  elimination, demand-driven tiled execution (tiled == whole-image, bit-identical
  for exact ops), deterministic reductions, a content-addressed BLAKE3 result
  cache with zero-recompute replay, safe graph simplification, and DOT/SVG graph
  visualization.
- **M3 — optimized CPU + GPU backends.** A backend-dispatch layer
  (`cpu.reference` / `cpu.optimized` / `wgpu`), SIMD pointwise kernels, Rayon tile
  parallelism (bit-identical across threads), a separable Gaussian, and a
  `wgpu` (GPU) crate with pointwise fusion, batched splats, and separable filters —
  all checked against the CPU oracle by a cross-backend differential harness, with
  performance baselines in CI.
- **M4 — classical "magic".** Image pyramids and frequency split (incl. FFT /
  bandpass), structure tensor + orientation fields, guided and bilateral filters,
  Poisson / screened-Poisson solvers, PatchMatch + patch synthesis, procedural
  noise / fbm / domain warp / reaction-diffusion, and a contract-driven local
  optimizer — every solver reporting convergence metrics, every iterative op
  deterministic.

Each milestone has an executable exit gate (`just m0-gate` … `just m4-gate`); all
run in CI (`.github/workflows/ci.yml`), and the M3 GPU criteria are verified on
hardware.

- **Run the loop:** `docs/M0_LOOP.md` — author → run → diagnose → reproduce, plus
  the fresh-clone walkthrough and exit-criteria checklists.
- **The keystone scenarios:** `conformance/README.md`.
- **Quality gate:** `just check` (formatting, the lint wall, tests, docs) is the
  per-change gate; `just verify-ops` runs the full op suite.

Later milestones — model adapters and perception candidates (M5), `msh`
material editing and multi-view evidence (M6/M7), and compiler/alien-ops research
(M8) — are planned. The canonical plan and operation backlog live under:

- `notes/paintop-plan/plan.md`
- `notes/paintop-plan/M0_DECISIONS.md`
- `notes/paintop-plan/docs/IR_SPEC.md`
- `notes/paintop-plan/docs/AGENT_VERIFICATION.md`
- `notes/paintop-plan/docs/OP_CATALOG.md`

## CLI

```bash
paintop validate plan.json
paintop run plan.json --bundle target/evidence/run-001
paintop explain plan.json --format json
paintop graph plan.json --out graph.svg
paintop diff before.png after.png
paintop op list
paintop op schema paint.gaussian_splats@1
```

Run via `cargo run -p paintop-cli -- <args>` or the installed `paintop` binary
(after `just install`). `paintop run` writes a complete evidence bundle —
normalized plan, trace, metrics, diffs, output artifacts, and assertion results.
See `docs/M0_LOOP.md`.

## Development Workflow

This repo uses the Edict workflow with `bones` for task tracking and `maw` workspaces for isolated changes. See `AGENTS.md` for the project-specific agent workflow.

Useful orientation commands:

```bash
bn status
bn triage
bn next
maw ws list
```

The quality gate is `just check` (formatting, the lint wall, tests, and docs),
with `cargo xtask verify-op` providing per-operation multi-oracle verification.
