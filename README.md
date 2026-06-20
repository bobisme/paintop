# paintop

`paintop` is a planned command-line runtime for deterministic, verifiable visual edits.

The project is aimed at coding agents and automation systems that need to edit images, and later material textures, without relying on unchecked pixel scripts or subjective visual inspection. A user or agent writes a typed JSON operation graph; `paintop` validates it, normalizes it, executes it, and emits outputs plus machine-checkable evidence.

## What It Is For

- Localized image touch-ups inside explicit masks.
- Typed image, mask, field, color, alpha, and boundary semantics.
- Reproducible CPU reference behavior before optimized backends.
- Evidence bundles with normalized plans, traces, metrics, diffs, artifacts, and assertion results.
- Agent-facing verification workflows where failures are inspectable and replayable.
- Future surface-aware material editing for GLB assets, with rendering and geometric queries delegated to `msh`.

`paintop` is not a GUI editor, Photoshop clone, prompt-to-image system, or general-purpose image scripting sandbox.

## Current Status

This repository is currently in early planning and backlog setup. The implementation workspace is not scaffolded yet. The canonical plan and operation backlog live under:

- `notes/paintop-plan/plan.md`
- `notes/paintop-plan/M0_DECISIONS.md`
- `notes/paintop-plan/docs/IR_SPEC.md`
- `notes/paintop-plan/docs/AGENT_VERIFICATION.md`
- `notes/paintop-plan/docs/OP_CATALOG.md`

The first build milestone is M0 plus the non-SDF MVP touch-up loop: strict parser and contracts, operation manifests, canonicalization and hashing, evidence skeletons, fixture generation, CLI commands, and a small verified operation set for a reproducible localized edit.

## Intended CLI Shape

The planned CLI includes commands such as:

```bash
paintop validate plan.json
paintop run plan.json --bundle target/evidence/run-001
paintop explain plan.json --format json
paintop graph plan.json --out graph.svg
paintop diff before.png after.png --mask allowed.png
paintop op list
paintop op schema paint.gaussian_splats@1
```

Until the Rust workspace lands, these commands are design targets rather than runnable tools.

## Development Workflow

This repo uses the Edict workflow with `bones` for task tracking and `maw` workspaces for isolated changes. See `AGENTS.md` for the project-specific agent workflow.

Useful orientation commands:

```bash
bn status
bn triage
bn next
maw ws list
```

When implementation begins, the expected quality gate is `just check`, backed by formatting, linting, tests, and operation-specific verification.
