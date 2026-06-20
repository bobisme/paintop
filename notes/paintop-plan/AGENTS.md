# AGENTS.md

This repository is designed to be implemented substantially by coding agents. These rules are part of the product architecture.

---

## 1. Prime directive

Do not optimize for producing a plausible image quickly. Optimize for producing an operation whose semantics, failure modes, and implementation can be independently verified.

A change is not complete until another agent can reproduce and diagnose it from commands and artifacts.

---

## 2. Read order

Before making architectural changes, read:

1. `plan.md`
2. `docs/IR_SPEC.md`
3. `docs/AGENT_VERIFICATION.md`
4. the affected operation manifest
5. the nearest existing reference implementation and conformance tests

Read `docs/ALIEN_OPS.md` for experimental work and `docs/RESEARCH.md` for adoption context.

---

## 3. Repository invariants

- Canonical edit programs are typed JSON graphs.
- Lua, if present, only generates serializable graph data.
- Graph resources are immutable logically.
- File writes are explicit exports.
- Image color, alpha, scalar format, and coordinate semantics are explicit.
- Masks, fields, IDs, and material channels are not generic color images.
- Foundational operations have a CPU reference semantic path.
- Optimized CPU/GPU paths are differential candidates, never their own oracle.
- Model operations produce perception resources or candidate sets by default.
- `paintop-image` does not depend on 3D/material crates.
- `msh` owns rendering/geometric queries; `paintop` owns edit semantics and mutation.
- Unknown JSON fields fail unless under a namespaced `extensions` object.
- Fixed seed is required after normalization for every stochastic operation.
- Stdout is machine-readable in machine mode; logs go to stderr.

Do not violate an invariant casually. Propose a decision-record change with evidence.

---

## 4. Standard verification commands

Run before claiming completion:

```bash
cargo fmt --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace
```

For an operation:

```bash
cargo xtask verify-op <operation-id>
```

For end-to-end changes:

```bash
paintop selftest --backend cpu-reference
paintop conformance run
```

When touching optimized/GPU code:

```bash
paintop selftest --backend all
cargo xtask differential <operation-id>
```

When touching schemas:

```bash
cargo xtask schema check
cargo xtask normalize-goldens check
```

When touching `msh` integration or material code:

```bash
cargo xtask gltf-conformance
cargo xtask render-goldens
cargo xtask material-integrity
```

If a command does not exist yet and the task depends on it, implementing the command may be part of the task.

---

## 5. Workflow for a new operation

### Step 1: write the contract first

Create or update the operation manifest:

- stable ID and semantic major;
- input/output types;
- parameters/defaults/ranges/units;
- color/alpha requirements;
- boundary behavior;
- output shape;
- ROI mapping and halo;
- determinism tier;
- seed behavior;
- postconditions;
- applicable properties;
- implementation list;
- resource limits;
- debug artifacts.

### Step 2: create failing tests

At minimum:

- minimal valid plan;
- invalid parameter/type plan;
- analytic fixture;
- identity/empty-mask property;
- one meaningful metamorphic relation;
- ROI or boundary test if applicable.

### Step 3: implement the reference path

Prefer readable scalar code. Checked arithmetic and explicit boundary behavior are mandatory. Do not hide conversions or clamping.

### Step 4: add evidence

The operation must appear in:

- structured trace;
- normalized graph;
- operation introspection;
- failure materialization;
- verification report.

### Step 5: optimize only after the oracle passes

Add differential tests before or with optimized code. Record implementation choice and tolerance.

### Step 6: benchmark

Measure representative extents, ROI densities, and parameter regimes. Report memory as well as time.

### Step 7: document limitations

State unsupported formats, approximation, pathological cases, and determinism constraints.

---

## 6. Required test independence

Avoid tests that restate the implementation.

Bad:

```text
implementation generates Gaussian kernel
unit test calls same kernel generator and compares it to itself
```

Better:

- impulse response;
- kernel sum and symmetry;
- known small coefficients from a separately derived formula;
- semigroup relation;
- dense vs separable implementation;
- external numeric calculation for a tiny case.

For each operation identify at least one oracle based on a different derivation or implementation path.

---

## 7. Tolerance rules

Do not increase tolerances just to pass CI.

A tolerance change requires:

1. failing evidence and error maps;
2. explanation of numeric source;
3. derived or conservative bound where possible;
4. measurements across adversarial fixtures;
5. confirmation that semantic bugs such as gamma, coordinates, or boundaries are absent;
6. checked-in tolerance-profile update.

ID maps, schema output, and declared exact operations do not use perceptual tolerances.

---

## 8. Golden update rules

A golden update must include:

- before/after/diff artifacts;
- reason for change;
- semantic version decision;
- confirmation that property/differential tests still pass;
- no unrelated golden churn.

Never auto-accept all snapshots.

---

## 9. Debugging protocol

When a test fails:

1. reproduce with a fixed seed;
2. pin backend and implementation;
3. emit an evidence bundle;
4. generate a minimal replay plan;
5. inspect resource descriptors and conversions;
6. compare full vs ROI/tile;
7. compare reference vs candidate;
8. inspect worst-pixel neighborhood and boundary mode;
9. check color/alpha/coordinate assumptions;
10. only then change implementation or tolerance.

Likely visual bug signatures are cataloged in `docs/AGENT_VERIFICATION.md`.

---

## 10. Performance rules

- The scheduler owns parallelism. Do not spawn independent unbounded pools in operations.
- Do not add GPU dispatches without considering residency and readback.
- Do not materialize full intermediates when an ROI/tile contract exists.
- Do not add a cache key that omits semantic version, resource semantics, seed, or model hash.
- Do not claim speedup from a single timing.
- Keep reference paths simple even if optimized paths share some utilities.
- Any approximate algorithm requires explicit opt-in or a semantic contract that permits it.
- Trace selected algorithm: direct, separable, FFT, low-rank, CPU, GPU, etc.

---

## 11. Neural/model rules

- Core conformance must run without network or model downloads.
- Every model has a manifest, immutable hash, source, license, preprocessing version, postprocessing version, limits, and test vectors.
- Validate model outputs as untrusted data.
- Models do not receive arbitrary paths or shell access.
- A model candidate cannot mutate source outside the explicit mask.
- Candidate generation and candidate selection are separate nodes.
- Confidence is carried forward rather than discarded.
- A model/provider timeout or crash is a structured failure.
- Do not couple core IR types to PyTorch, ONNX, or a specific provider.

---

## 12. `msh` and material rules

- Do not evolve `MeshWithColors` into the scene-preserving material asset model.
- Preserve scene/node/primitive/material/texture identity.
- Treat the glTF specification as the source of truth for channel/color semantics.
- Do not pack roughness/metallic/occlusion/normal data as sRGB.
- Do not silently unwrap or rewrite UVs.
- Surface projection must declare visibility, overlap, mirror, seam, wrap, and texture-transform policies.
- Exported asset changes must pass an allowlist integrity assertion.
- Do not combine `msh` library extraction with a `wgpu` major upgrade.
- Diagnostic ID/depth/UV buffers need exact representations; PNG previews are not the oracle.

---

## 13. Security rules

- Use checked arithmetic for dimensions, offsets, strides, buffer sizes, and dispatch sizes.
- Apply decode limits before decompression/allocation.
- Plan paths remain under explicit roots.
- Writes are atomic and do not overwrite without policy.
- Graph/model execution has deadlines and memory limits.
- No arbitrary user shader, plugin, Lua C module, dynamic library, or process command in the trusted plan runtime.
- Fuzz parsers, decoders, GLB handlers, and graph validation.
- Do not weaken sandboxing for developer convenience without an explicit privileged mode.

---

## 14. Change sizing

Prefer changes that one agent can fully verify:

- one operation contract + reference implementation;
- one optimized backend for an existing operation;
- one execution/compiler feature with focused conformance fixtures;
- one `msh` diagnostic pass;
- one model adapter with fake and real manifest tests.

Avoid PRs that simultaneously:

- change IR semantics;
- add multiple operations;
- refactor the scheduler;
- update `wgpu`;
- change goldens;
- add a model.

Split semantic, mechanical, optimization, and dependency changes.

---

## 15. Completion template

Include this in the task/PR report:

````markdown
## Contract
- Operation/subsystem:
- Semantic version affected:
- Inputs/outputs:
- Determinism tier:
- Unsupported cases:

## Verification
- Analytic oracle:
- Properties:
- Metamorphic tests:
- Differential comparisons:
- ROI/tile/cache tests:
- Fuzz/adversarial tests:

## Commands
```bash
# exact commands
```

## Evidence
- Bundle/report path:
- Worst numeric error:
- Relevant contact sheet/diff:

## Performance
- Benchmark matrix:
- Peak memory:
- Regression status:

## Remaining uncertainty
- Explicit list; never omit this section.
````

---

## 16. Forbidden shortcuts

- “It compiles, so it works.”
- “The image looks right.”
- “The model is state of the art.”
- “GPU floating point is just different.”
- “The golden changed because the implementation changed.”
- “This conversion is obvious, so it can be implicit.”
- “We can add the contract later.”
- “The user asked for speed, so tests can wait.”
- “The agent will know what this generic tensor means.”
- “We only edit one material today, so flattening is fine.”

Each sentence indicates missing engineering evidence.

---

## 17. Escalation criteria

Stop and raise an architectural issue rather than patch around it when:

- the operation cannot state its color/alpha semantics;
- an optimized backend cannot be compared to an independent oracle;
- a model output cannot be constrained to a typed resource;
- a plan needs arbitrary code execution to express a common case;
- ROI behavior cannot be defined;
- a material edit cannot identify allowed asset mutations;
- determinism claims depend on undocumented hardware behavior;
- performance requires changing semantics;
- a test failure cannot be reduced or explained with evidence.

The correct output may be a research spike or contract revision, not more code.
