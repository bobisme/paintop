# Agent verification and autonomous development protocol

**Purpose:** make `paintop` implementable, testable, and debuggable by coding agents without relying on a human eyeballing every output.

The core rule is:

> Visual plausibility is evidence, not proof. Every operation needs independent, machine-checkable contracts.

---

## 1. Verification model

A trustworthy visual operation is surrounded by several partially independent oracles:

```text
                 ┌──────────────────┐
                 │ analytic oracle  │
                 └────────┬─────────┘
                          │
┌─────────────┐   ┌───────▼────────┐   ┌────────────────┐
│ properties  ├──►│ implementation ├◄──┤ differential   │
└─────────────┘   └───────┬────────┘   └────────────────┘
                          │
                 ┌────────▼─────────┐
                 │ metamorphic tests│
                 └────────┬─────────┘
                          │
          ┌───────────────▼────────────────┐
          │ runtime assertions + evidence │
          └────────────────────────────────┘
```

No single oracle is sufficient:

- a golden can enshrine a bug;
- a property can be too weak;
- a differential test can compare two implementations sharing the same mistake;
- a perceptual metric can approve a semantically wrong edit;
- an analytic test can cover only small synthetic cases.

Confidence comes from overlap.

---

## 2. Required verification layers

### 2.1 Layer 0: build hygiene

Every change must pass:

```bash
cargo fmt --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace
cargo doc --workspace --no-deps
```

Additional checks should be introduced gradually:

```bash
cargo deny check
cargo machete
cargo audit
cargo llvm-cov --workspace --fail-under-lines 80
```

Coverage percentage is not a quality target by itself. It catches unexecuted branches; it does not prove image semantics.

### 2.2 Layer 1: schema and contract tests

For every operation:

- valid minimal plan parses;
- every required field omission fails;
- unknown fields fail;
- invalid enum values fail;
- numeric limits fail at boundaries;
- incompatible resource types fail;
- color/alpha mismatch fails;
- unsupported backend policy fails before execution;
- deterministic seed normalization is stable;
- manifest and Rust implementation agree about ports and defaults.

Generate negative tests from the operation manifest where possible.

### 2.3 Layer 2: analytic fixtures

Synthetic images with known structure let agents derive expected behavior instead of trusting a rendering.

Required fixture families:

| Fixture | Detects |
|---|---|
| Constant scalar/color | Bias, non-unit kernels, boundary errors. |
| Single impulse | Kernel shape, centering, support, splat normalization. |
| Horizontal/vertical ramp | Derivative, interpolation, coordinate, gamma errors. |
| Checkerboard | Aliasing, phase, resampling, tile seams. |
| Sine gratings | Frequency response, blur/sharpen behavior. |
| Binary rectangle/circle | Rasterization, SDF, morphology, antialiasing. |
| Alpha edge with hidden RGB | Premultiplication and fringe bugs. |
| Tiny images: 1×1, 1×N, 2×2 | Boundary and dimension assumptions. |
| NaN/Inf injected field | Finite-value validation. |
| Label map with large IDs | Integer attachment/encoding loss. |
| UV seam toy mesh | Surface projection and seam consistency. |

Analytic fixtures should be generated from code with fixed formulas and versioned parameters, not manually painted images.

### 2.4 Layer 3: property-based tests

Use `proptest` or a comparable generator. Inputs should be small enough for reference implementations and shrinking.

General properties:

- output extent matches shape inference;
- no NaN/Inf unless operation explicitly permits it;
- declared ranges hold;
- empty mask is identity;
- full mask equals unmasked operation;
- zero-opacity edit is identity;
- operation is deterministic for fixed seed/tier;
- tiled and whole-image execution agree;
- ROI execution agrees inside demanded region;
- cache hit output equals uncached output;
- serializing and reparsing normalized plan preserves semantic hash.

Operation-specific properties are listed later.

### 2.5 Layer 4: metamorphic tests

Metamorphic testing checks relationships under transformed inputs when exact outputs are unknown.

Examples:

#### Translation equivariance

For an operation `F` that should be translation equivariant away from boundaries:

```math
F(T_\Delta x) = T_\Delta F(x)
```

Use padded inputs and compare the valid interior.

Applies to:

- convolution;
- splats when centers are translated;
- morphology;
- pointwise color operations;
- local patch descriptors.

#### Rotation/reflection covariance

For isotropic blur or Euclidean distance transform:

```math
F(Rx) = R F(x)
```

Use 90-degree rotations for exact raster correspondence.

#### Scale covariance

Scale image coordinates and all physical radii/sigmas together. Compare after resampling under a declared tolerance.

#### Decomposition equivalence

- split a splat batch into two batches and composite in the same defined order;
- separable kernel vs dense outer-product kernel;
- process image as tiles vs whole;
- process connected masked components independently vs together when the operation is component-local;
- combine two consecutive exposure changes: `EV(a)` then `EV(b)` equals `EV(a+b)` in unclamped linear light.

#### Round-trip relations

- premultiply then unpremultiply for alpha above epsilon;
- encode then decode transfer function;
- pack then unpack metallic/roughness;
- normal decode/encode within quantization tolerance;
- affine transform then inverse on band-limited fixtures within resampling tolerance.

#### Mask algebra

For hard masks:

- commutativity: `A ∪ B = B ∪ A`;
- associativity;
- idempotence: `A ∪ A = A`;
- complement laws;
- De Morgan laws;
- `A - A = ∅`.

For soft masks, only assert laws defined by the selected fuzzy algebra.

### 2.6 Layer 5: differential testing

Compare independent implementations.

Required pairs when available:

| Reference | Candidate |
|---|---|
| scalar CPU | SIMD CPU |
| scalar CPU | tiled CPU |
| scalar CPU | GPU |
| dense convolution | separable convolution |
| dense convolution | low-rank approximation with declared error |
| direct convolution | FFT overlap-save |
| full-frame | ROI/tile execution |
| exact EDT | GPU/approximate distance transform |
| CPU ray/triangle hit | G-buffer surface hit |
| internal PNG path | known decoder/encoder test vectors |

Differential comparison must use operation-specific metrics. “Images look close” is not a comparator.

For scalar data:

```math
|a-b| \le \epsilon_{abs} + \epsilon_{rel} \max(|a|,|b|)
```

Also report:

- max absolute error;
- mean absolute error;
- RMS error;
- percentile errors;
- count above threshold;
- location and tile of worst error.

For vectors:

- component error;
- norm error;
- angular error.

For masks:

- max coverage error;
- area difference;
- boundary Hausdorff-like distance where meaningful.

For IDs:

- exact equality only.

### 2.7 Layer 6: goldens

Use goldens for end-to-end behavior and external formats, not as the only operation oracle.

Classes:

- exact bytes for canonical JSON and lossless deterministic encodings where guaranteed;
- exact pixel arrays for integer/reference operations;
- numeric arrays plus tolerance metadata for floating output;
- render goldens plus diagnostic buffers for `msh` integration;
- structural JSON reports for GLB before/after.

A golden update requires:

1. a diff artifact;
2. an explanation tied to a semantic or implementation change;
3. unchanged property/differential tests;
4. explicit reviewer or supervising-agent acceptance.

Never use “accept all changed snapshots” as an automated fix.

### 2.8 Layer 7: perceptual evidence

Use PSNR, SSIM, LPIPS, DISTS, or related metrics only as secondary signals.

Perceptual metrics are useful for:

- identifying gross restoration regressions;
- ranking candidates when hard constraints already pass;
- detecting unexpected visual changes in render goldens;
- characterizing approximation quality.

They cannot establish:

- that only the authorized region changed;
- that a material map obeys PBR channel semantics;
- that object identity is preserved;
- that a texture edit is physically correct;
- that a small but semantically critical logo was not altered.

### 2.9 Layer 8: fuzzing and adversarial input

Fuzz targets:

- JSON parser and duplicate-key handling;
- schema/normalizer;
- resource graph/cycle detection;
- dimension and stride calculations;
- image decoders;
- convolution kernels and boundary modes;
- masks with pathological floats;
- splat batches with extreme covariance;
- GLB parser and exporter;
- model tensor shape adapters;
- cache serialization.

Seed corpus:

- zero-sized declarations;
- maximum legal dimensions;
- truncated files;
- huge metadata chunks;
- degenerate polygons;
- self-intersecting paths;
- singular transforms;
- zero/negative sigma;
- all-transparent images with nonzero RGB;
- overlapping/mirrored UVs;
- degenerate UV triangles;
- malformed material texture references.

### 2.10 Layer 9: performance and resource verification

An operation can be numerically correct and operationally unusable.

Measure:

- median/p95/p99 wall time;
- CPU time;
- peak resident memory;
- bytes allocated;
- tile count requested/executed/skipped;
- cache hits/misses;
- GPU dispatch count;
- GPU upload/readback bytes;
- pipeline compilation count;
- model load/inference time;
- evidence artifact size.

Performance tests should detect algorithmic cliffs:

- kernel radius transitions;
- sparse-to-dense mask thresholds;
- CPU-to-GPU crossover;
- FFT crossover;
- image sizes near row-alignment boundaries;
- pathological connected-component counts.

---

## 3. Operation property catalog

### 3.1 Pointwise color operations

Properties:

- empty mask identity;
- full mask equals unmasked output;
- output finite;
- alpha unchanged unless declared;
- exposure in linear light satisfies:

  ```math
  exposure(x, e) = x \cdot 2^e
  ```

- exposure composition before clamp: `E_a(E_b(x)) = E_{a+b}(x)`;
- zero adjustment identity;
- monotonicity for exposure/gamma over valid nonnegative domains;
- color matrix composition matches matrix multiplication;
- no changes outside mask at exact reference precision.

Adversarial fixtures:

- negative scene-linear values if supported;
- HDR values above 1;
- alpha zero with arbitrary RGB;
- exact channel extremes;
- subnormal values.

### 3.2 Alpha compositing

For premultiplied `over`:

```math
C_o = C_s + C_d(1-\alpha_s)
```

```math
\alpha_o = \alpha_s + \alpha_d(1-\alpha_s)
```

Properties:

- transparent source identity;
- opaque source replacement;
- output alpha in `[0,1]` for valid inputs;
- output premultiplied constraint `|C_i| <= alpha` for bounded nonnegative color;
- associativity within floating tolerance for premultiplied `over`;
- no colored fringe on alpha-edge fixtures after export.

### 3.3 Convolution

Properties:

- impulse response equals kernel under matching boundary interior;
- constant image scales by kernel sum;
- unit-sum kernel preserves constant values;
- zero-sum kernel yields zero on constant interior;
- linearity before clamp;
- exact outer-product kernel agrees with separable path;
- sparse and dense forms agree;
- direct and FFT forms agree within declared tolerance;
- tile seams absent;
- transpose/rotate relations for transformed kernels.

### 3.4 Gaussian blur

Properties:

- positive unit-sum kernel;
- constant preserving;
- isotropic under 90-degree rotations;
- semigroup approximately holds for discretized kernels:

  ```math
  G_{σ1} * G_{σ2} ≈ G_{sqrt(σ1²+σ2²)}
  ```

- variance of blurred impulse matches requested variance within discretization bound;
- sigma→0 approaches identity under defined cutoff policy.

### 3.5 Splats

For Gaussian splat with covariance `Σ`:

```math
w(p) = \exp(-\tfrac{1}{2}(p-\mu)^T Σ^{-1}(p-\mu))
```

Properties:

- center symmetry;
- covariance axes align with specified rotation;
- translation covariance;
- zero opacity identity;
- clipping mask respected;
- bounded support truncation error reported;
- batch order behavior matches blend mode semantics;
- split-batch equivalence for commutative accumulation modes;
- CPU/GPU accumulated coverage agrees within tolerance.

### 3.6 Rasterized shapes

Properties:

- bounds agree with analytic geometry plus antialias support;
- area converges to analytic area with increasing resolution;
- 90-degree rotation/reflection covariance;
- polygon winding/fill rule explicit;
- degenerate edges handled without NaN;
- tile boundary does not alter coverage;
- half-open coordinate convention tested on exact pixel-aligned rectangles.

### 3.7 Signed distance and morphology

Properties:

- sign is correct inside/outside;
- zero contour corresponds to declared threshold boundary;
- distance gradient magnitude is approximately one away from medial axis;
- offset composition: `offset(offset(s,d1),d2)=offset(s,d1+d2)`;
- grow/shrink by zero identity;
- dilation/erosion duality under complement for hard masks;
- Euclidean rotation covariance on symmetric fixtures;
- exact EDT agrees with brute force on small images.

### 3.8 Resampling and warps

Properties:

- identity transform identity;
- integer translation with nearest filter exact;
- inverse transform round trip on band-limited fixtures;
- constant preserving for normalized kernels;
- output does not depend on tiling;
- singular transform fails before execution;
- support/ROI mapping contains all contributing source pixels;
- transparent boundary behavior respects premultiplication.

### 3.9 Pyramids and frequency decomposition

Properties:

- split then recombine reconstructs input within declared tolerance;
- total dimensions/phase conventions stable;
- constant image has zero residual bands;
- impulse response documented;
- tile execution agrees with full image;
- high/low energy shifts as expected for sine fixtures.

### 3.10 PatchMatch and texture synthesis

Properties are weaker because the algorithm is stochastic/iterative:

- fixed seed and backend reproduce declared tier;
- nearest-neighbor energy is non-increasing across accepted update iterations;
- all correspondences lie in allowed source region;
- hole pixels are filled;
- outside-hole pixels unchanged;
- patch score recomputation agrees with stored score;
- brute-force nearest patch agrees on tiny images;
- splitting source region into impossible/valid subsets behaves predictably;
- termination limit honored.

### 3.11 Poisson and PDE solvers

Properties:

- boundary values satisfy constraints within residual tolerance;
- reported residual decreases or meets stopping criterion;
- constant guidance plus constant boundary returns constant solution;
- linear system matrix construction matches finite-difference stencil;
- direct solver and iterative solver agree on small domains;
- connected components solved independently produce same result;
- no change outside target domain;
- failure to converge is explicit and does not silently export partial output unless policy allows candidates.

### 3.12 Model operations

Properties focus on wrapper correctness, not proving the model:

- exact model hash checked;
- preprocessing test vector matches expected tensor;
- postprocessing test vector matches expected resource;
- output shapes and ranges validated;
- confidence field extent matches prediction;
- mask outside requested crop is zero or explicitly mapped;
- deterministic seed/provider behavior recorded;
- timeout/cancellation/OOM handled;
- malformed model output rejected;
- model cannot write outside adapter sandbox;
- source image remains immutable;
- candidate operation never implicitly selects an output.

### 3.13 Material map operations

Properties:

- base-color sRGB decode/encode explicit;
- roughness packed to glTF green channel;
- metallic packed to blue channel;
- occlusion uses red channel;
- normal maps decoded to signed tangent vectors and renormalized;
- untouched channels remain bit-identical when packing permits it;
- scalar maps remain linear;
- material factors compose with texture values according to glTF semantics;
- image, texture, and material identity outside allowlist remains unchanged.

### 3.14 Surface projection

Properties:

- CPU ray hit and G-buffer hit agree on triangle/material/UV for interior samples;
- barycentric coordinates sum to one and lie within tolerance;
- reconstructed world position matches interpolated triangle position;
- screen→surface→screen round trip agrees within pixel tolerance;
- projected UV footprint conserves normalized coverage under simple planar mappings;
- frontmost visibility rejects occluded surfaces;
- material filter is exact;
- mirrored/overlapping UV policies trigger expected report/rejection;
- seam-paired texels receive consistent coverage when policy requests it;
- tile boundaries do not create UV holes.

---

## 4. Fixture repository

### 4.1 Directory shape

```text
fixtures/
├── analytic/
│   ├── constants/
│   ├── impulses/
│   ├── ramps/
│   ├── sinusoids/
│   ├── alpha/
│   └── geometry/
├── photographic/
│   ├── public-domain/
│   └── manifests.json
├── adversarial/
│   ├── malformed-images/
│   ├── extreme-dimensions/
│   └── pathological-masks/
├── materials/
│   ├── pbr-channel-toys/
│   └── uv-atlases/
└── gltf/
    ├── single-plane/
    ├── cube-seams/
    ├── overlapping-uv/
    ├── mirrored-uv/
    ├── multi-material/
    ├── texture-transform/
    └── skinned/
```

### 4.2 Fixture manifest

Every non-generated fixture needs:

```json
{
  "id": "photo.wall-crack.001",
  "path": "photographic/public-domain/wall-crack-001.png",
  "sha256": "...",
  "license": "CC0-1.0",
  "source": "...",
  "semantic_tags": ["texture", "crack", "repair"],
  "expected_properties": ["opaque", "srgb"]
}
```

No mystery images copied from the internet.

### 4.3 Generated fixture formulas

The fixture generator should expose versioned formulas:

```bash
cargo xtask fixture generate impulse \
  --width 65 --height 65 --x 32 --y 32 \
  --format f32 --out fixtures/generated/impulse-65.json
```

A JSON or binary array representation should accompany preview PNGs so tests do not compare against quantized screenshots.

---

## 5. Evidence bundle specification

### 5.1 Bundle manifest

```json
{
  "paintop_runtime": "0.1.0+gitsha",
  "plan_semantic_hash": "blake3:...",
  "normalized_plan": "normalized-plan.json",
  "started_at": "2026-06-20T18:42:10Z",
  "platform": {
    "os": "linux",
    "arch": "x86_64",
    "cpu": "...",
    "gpu": "...",
    "driver": "..."
  },
  "determinism": "bounded",
  "status": "assertion-failed",
  "exit_code": 6,
  "outputs": [],
  "failures": ["localized"]
}
```

Time and host data are provenance, not semantic identity.

### 5.2 Trace events

Use JSON Lines for streamability. Event types:

- plan parsed;
- node normalized;
- graph optimized;
- resource demanded;
- cache lookup;
- tile scheduled;
- implementation selected;
- dispatch started/completed;
- assertion measured;
- output written;
- cancellation/failure.

Each event has stable keys and a schema version.

### 5.3 Assertion report

```json
{
  "id": "localized",
  "op": "assert.no_change_outside_mask@1",
  "status": "failed",
  "metrics": {
    "max_abs_delta_outside": 0.00312,
    "changed_pixels_outside": 17,
    "worst_pixel": [721, 418]
  },
  "thresholds": {
    "max_abs_delta": 0.000001,
    "changed_pixels": 0
  },
  "artifacts": {
    "outside_diff": "diffs/localized-outside.png",
    "minimal_replay": "replays/localized.json"
  }
}
```

### 5.4 Minimal replay generation

On failure, automatically emit a reduced plan containing:

- required external inputs or cropped reproducer;
- transitive producer nodes only;
- failing assertion;
- fixed backend/implementation;
- exact seed;
- failing ROI plus sufficient halo;
- relevant resource metadata.

This is analogous to a compiler testcase reducer and is essential for agent debugging.

---

## 6. Autonomous agent workflow

An implementation agent should follow this loop:

```text
1. read operation manifest and task contract
2. run existing tests and capture baseline
3. add/confirm failing analytic/property test
4. implement smallest semantic reference path
5. run verify-op
6. add differential or alternate oracle
7. add ROI/tile tests
8. benchmark representative matrix
9. inspect evidence artifacts on at least one end-to-end plan
10. update manifest/docs and emit completion report
```

### 6.1 Required completion report

An agent should produce machine-readable and human-readable summaries:

```json
{
  "task": "filter.gaussian_blur@1 cpu reference",
  "status": "complete",
  "changed_contracts": [],
  "tests_added": [
    "impulse_response",
    "constant_preserving",
    "translation_metamorphic",
    "boundary_modes"
  ],
  "verification_commands": [
    "cargo xtask verify-op filter.gaussian_blur@1"
  ],
  "evidence": "target/verification/filter.gaussian_blur/index.json",
  "benchmarks": {
    "1024x1024_sigma_4_ms": 18.4
  },
  "known_limits": ["reference path is scalar"]
}
```

### 6.2 What agents may not do

- weaken a tolerance to make a failure disappear without deriving a justified error bound;
- update goldens without inspecting and recording the diff;
- disable a property test because the implementation is inconvenient;
- silently insert color/alpha conversions;
- use the optimized implementation as its own oracle;
- make network calls during core tests;
- add model weights without manifest, hash, and license review;
- claim performance improvement from one unrepeatable timing;
- conflate “no panic” with correctness;
- merge a new op without evidence bundle support.

---

## 7. Tolerance policy

### 7.1 Tolerances belong to contracts

A tolerance is not a global magic constant. It depends on:

- operation;
- numeric format;
- image dynamic range;
- implementation class;
- number of accumulated operations;
- boundary region;
- approximation setting.

Store tolerance profiles in checked-in data:

```json
{
  "op": "filter.gaussian_blur@1",
  "comparison": "cpu-reference-vs-wgpu-separable",
  "format": "f32",
  "domain": "[0,1]",
  "max_abs": 0.00005,
  "max_rel": 0.0002,
  "rms": 0.00001,
  "exclude": {"none": true}
}
```

### 7.2 Derive, then measure

Where possible, derive a conservative bound from:

- coefficient quantization;
- accumulation length;
- floating-point rounding model;
- interpolation support;
- iterative residual.

Then measure across adversarial/random fixtures. If empirical errors exceed the bound, investigate. Do not simply fit tolerance to observed output.

### 7.3 Error maps

Every failed differential test should save:

- absolute error image;
- signed error image;
- relative error image where stable;
- histogram;
- worst-pixel neighborhood;
- tile/pipeline metadata.

Spatial patterns often reveal the bug immediately: seams, boundary mode, gamma, transposition, or row padding.

---

## 8. CI matrix

### 8.1 Per-pull-request

Fast, deterministic:

- format/lint;
- unit/schema tests;
- small property tests;
- scalar reference conformance;
- selected optimized CPU differential tests;
- no-network model-wrapper tests with fake models;
- documentation examples;
- changed-op verification.

### 8.2 Main branch/nightly

Broader:

- all property tests with higher case count;
- fuzz smoke;
- full conformance suite;
- cross-platform CPU differential tests;
- GPU tests on available vendors;
- render/material goldens;
- model adapter tests for approved models;
- performance regression suite;
- cache corruption/replay tests;
- artifact upload.

### 8.3 Scheduled deep runs

- long fuzzing;
- sanitizer/Miri-compatible subsets;
- cross-vendor GPU comparisons;
- huge-image bounded-memory tests;
- randomized graph generation;
- GLB corpus round trips;
- model/provider matrix;
- benchmark trend analysis.

### 8.4 Hardware labels

Record and group results by:

- CPU ISA;
- core count;
- GPU vendor/model;
- driver;
- backend;
- memory limits.

Do not compare absolute performance across dissimilar runners as a regression signal. Compare each runner to its own historical baseline or normalized kernels.

---

## 9. Generated random graph testing

A typed graph generator can explore interactions humans will miss.

### 9.1 Generator constraints

- small extents, typically 1–64 pixels;
- bounded node count;
- resources selected by type compatibility;
- parameters generated within legal ranges, emphasizing boundaries;
- seeds fixed in repro output;
- exports and assertions always reachable;
- optional invalid-graph mode for validator fuzzing.

### 9.2 Metamorphic graph transforms

Given a valid graph, derive equivalent or predictably related graphs:

- insert identity nodes;
- duplicate common subgraph then compare;
- tile executor on/off;
- cache on/off;
- materialize extra intermediates;
- choose alternate backend;
- replace exact separable kernel with dense form;
- reorder independent nodes;
- rename IDs and verify semantic outputs remain equal;
- serialize/deserialize normalized plan.

### 9.3 Failure reduction

When a generated graph fails:

1. remove unrelated exports/assertions;
2. remove dead nodes;
3. minimize node count;
4. shrink image extent;
5. shrink parameters and masks;
6. pin implementation and seed;
7. emit minimal replay and evidence.

This should be automated using a delta-debugging reducer guided by the failure predicate.

---

## 10. Model verification protocol

### 10.1 Model manifest

```json
{
  "id": "sam2-tiny",
  "version": "...",
  "format": "onnx",
  "sha256": "...",
  "source": "...",
  "license": "...",
  "inputs": {},
  "outputs": {},
  "preprocess": "sam2-preprocess@1",
  "postprocess": "sam2-mask-postprocess@1",
  "providers": ["cpu", "cuda"],
  "determinism": "bounded",
  "limits": {
    "max_input_pixels": 16777216,
    "max_memory_bytes": 4294967296,
    "timeout_ms": 10000
  },
  "test_vectors": []
}
```

### 10.2 Tests

- checksum mismatch fails;
- unsupported provider fails or falls back according to policy;
- preprocessing tensor exact test;
- output tensor parser exact test;
- known fixture prompt produces mask with broad expected geometry, not an overfitted pixel golden;
- crop/resize coordinate mapping round trip;
- confidence range/shape;
- repeated fixed-seed run characterization;
- timeout and cancellation;
- OOM simulation or configured memory rejection;
- process adapter crash isolation;
- malformed output rejection;
- model license/source displayed by introspection.

### 10.3 Trust boundary

Model output must be treated as untrusted structured data:

- validate all dimensions and counts;
- reject nonfinite values;
- clamp only when postprocessing semantics explicitly say so;
- never use model-provided paths or commands;
- never let model output bypass authorized edit masks;
- combine model confidence with deterministic selection constraints.

---

## 11. `msh` and material verification

### 11.1 Scene integrity report

Before and after a GLB edit, compute a canonical structural report:

- scene/node counts and hierarchy;
- transforms;
- meshes/primitives;
- accessors and topology hashes;
- material assignments;
- texture/image/sampler identities;
- animation/skin hashes;
- extensions and extras;
- buffer lengths and relevant content hashes.

An allowlist assertion describes permitted differences.

### 11.2 Render-pass verification

For each diagnostic pass:

- exact ID buffers where possible;
- depth monotonicity and known-plane fixtures;
- UV interpolation on a plane/triangle;
- barycentric sum and vertex-corner values;
- normal transforms under rotation and nonuniform scale;
- material ID boundaries;
- camera serialization round trip;
- headless vs viewer render agreement for fixed state.

### 11.3 Projection fixtures

Create tiny GLBs:

1. single front-facing textured plane;
2. tilted plane with known homography;
3. cube with six UV islands;
4. two overlapping planes at different depth;
5. mirrored UV halves;
6. overlapping UV islands;
7. texture transform extension;
8. repeated/wrapped UVs;
9. multi-material primitive set;
10. skinned plane at a fixed animation frame.

For the plane, derive screen→UV analytically and compare every sampled hit.

### 11.4 Multi-view edit assertions

- authorized surface set is stable across views;
- edited texels correspond to visible target surfaces;
- no other material samples changed;
- seam-paired texels agree within tolerance;
- coverage holes below threshold;
- rendered difference outside projected surface silhouette below threshold;
- target effect visible in at least one requested view;
- hidden/backface policy honored.

---

## 12. Performance verification strategy

### 12.1 Benchmark dimensions

For each operation benchmark across:

- sizes: tiny, 256², 1K², 4K², 8K² where applicable;
- ROI density: 0%, 1%, 10%, 50%, 100%;
- mask pattern: compact, scattered, checker, many components;
- formats: `u8`, `f32`, later `f16`;
- radius/support;
- CPU thread count;
- backend;
- warm/cold cache;
- GPU pipeline cold/warm.

### 12.2 Performance contracts

Avoid brittle exact milliseconds in core tests. Use:

- asymptotic guards;
- maximum memory bounds;
- maximum executed tile counts;
- maximum dispatch/readback counts;
- broad regression ratios on stable runners;
- benchmark artifacts and trend alerts.

Examples:

- a 1% compact ROI must not execute more than the conservative halo-expanded tile set;
- separable Gaussian should scale approximately with radius linearly, not quadratically;
- cache replay should execute zero pure producer nodes;
- GPU-resident chain should perform no intermediate CPU readback;
- evidence disabled should not materialize unrequested intermediates.

### 12.3 Autotuner verification

The autotuner may choose implementations, but:

- every candidate must already conform;
- selected schedule is recorded;
- disabling autotune yields a stable fallback;
- profile corruption triggers rebuild, not undefined behavior;
- profile is keyed by hardware/runtime version;
- performance exploration is time-bounded.

---

## 13. Bug taxonomy and likely signatures

| Symptom | Likely cause |
|---|---|
| Bright/dark halos around transparency | Nonlinear or unpremultiplied filtering. |
| One-pixel shift | Pixel-center/edge convention mismatch. |
| Tile grid visible | Missing halo, inconsistent boundary, write overlap. |
| GPU differs only at row ends | Copy-row alignment/stride bug. |
| Colors differ strongly but structure matches | sRGB/linear mismatch or channel order. |
| UV paint has holes on tilted surfaces | Nearest scatter instead of anisotropic footprint. |
| Paint appears on back side | UV overlap/mirroring or visibility policy. |
| Roughness edit affects metalness | glTF channel packing error. |
| Diff outside mask on transparent pixels | Hidden RGB or encoding comparison mismatch. |
| Iterative solver looks plausible but varies | Nondeterministic reduction/update order. |
| Cache-only failure | Incomplete semantic cache key or mutable input. |
| Whole image works, ROI fails | Incorrect backward ROI or halo mapping. |
| CPU/GPU only disagree near boundaries | Boundary mode or support truncation. |

This table should be extended as real failures occur.

---

## 14. `cargo xtask verify-op`

The operation verification command should automate the definition of done.

```bash
cargo xtask verify-op paint.gaussian_splats@1
```

Expected stages:

1. locate and validate manifest;
2. run schema examples;
3. run unit/analytic tests tagged with op ID;
4. run property tests with standard seed set;
5. run metamorphic tests;
6. run differential matrix for available implementations;
7. run ROI/tile/cache variants;
8. run smoke benchmarks;
9. generate a verification report and contact sheet;
10. fail if any required category is missing.

Report:

```text
target/verification/paint.gaussian_splats@1/
├── index.json
├── summary.md
├── test-results.json
├── differential/
├── properties/
├── benchmarks/
└── contact-sheet.png
```

The manifest declares which verification categories are applicable. “Not applicable” requires a reason.

---

## 15. Definition of autonomous confidence

An agent may call a task complete only when it can answer, with paths to evidence:

1. What exact semantics were implemented?
2. Which invalid inputs are rejected?
3. What is the independent oracle?
4. Which algebraic properties hold?
5. Which metamorphic relationships were tested?
6. Do tile, ROI, cache, and full execution agree?
7. Do optimized backends agree with the reference?
8. What is the numeric tolerance and why?
9. What happens on cancellation/resource exhaustion?
10. What performance envelope was measured?
11. Which debug artifacts are emitted on failure?
12. What remains uncertain?

If an agent cannot answer those, the operation is not verifiable enough to merge.

---

## 16. Primary testing references

- Property-based testing in Rust (`proptest`): <https://docs.rs/proptest/latest/proptest/>
- Metamorphic testing for computer vision systems: <https://arxiv.org/abs/1912.12162>
- Cross-vendor differential testing of GPU numerical behavior: <https://arxiv.org/abs/2410.09172>
- LPIPS: <https://arxiv.org/abs/1801.03924>
- DISTS/A-DISTS family: <https://arxiv.org/abs/2110.08521>
- Shift-tolerant perceptual similarity: <https://arxiv.org/abs/2207.13686>
- Tolerant testing of image properties: <https://arxiv.org/abs/1503.01363>
