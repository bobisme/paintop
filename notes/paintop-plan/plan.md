# paintop: build plan

**Status:** proposed implementation blueprint  
**Date:** 2026-06-20  
**Primary audience:** coding agents and the humans supervising them  
**Language:** Rust  
**Canonical program format:** typed JSON operation graph  
**Optional authoring frontend:** Lua, later, as a graph generator only  
**Initial scope:** deterministic 2D image editing and verification  
**Expansion scope:** surface-aware material editing for glTF/GLB assets, using `msh` for rendering and geometric inspection

---

## 1. Mission

Build a headless visual transformation runtime that lets coding agents make **localized, composable, inspectable, reversible, and verifiable** edits to images and, later, texture-backed materials on 3D meshes.

The project is not a GUI editor and not a Photoshop clone. It is an **actuator and compiler for visual edits**:

```text
agent-authored plan
        │
        ▼
strict typed graph
        │
        ▼
validation + normalization + optimization
        │
        ▼
CPU/GPU/model execution
        │
        ▼
outputs + evidence bundle + machine-checkable assertions
```

A successful run does not merely produce an image. It produces evidence that tells an agent:

- what executed;
- what changed;
- where it changed;
- what invariants held;
- how closely alternate implementations agreed;
- which outputs are deterministic;
- which results are candidates rather than trusted edits;
- why the runtime accepted or rejected the plan.

The central engineering bet is:

> A small, typed visual algebra with strong contracts and exceptional observability will be more useful to agents than a broad, weakly specified image API.

---

## 2. Product definition

### 2.1 One sentence

`paintop` is a deterministic, graph-based image and material editing runtime designed to be safely driven and independently verified by coding agents.

### 2.2 Primary use cases

1. Touch up an image inside a constrained mask.
2. Create and manipulate masks, distance fields, orientation fields, pyramids, and patch correspondences.
3. Paint controlled batches of Gaussian or anisotropic splats.
4. Repair texture defects with clone, PatchMatch, Poisson, PDE, and neural candidate operations.
5. Run perception models to produce masks or fields that feed deterministic edits.
6. Emit before/after/diff/debug artifacts for autonomous critique.
7. Project screen-space selections onto mesh UV maps and edit PBR material channels.
8. Optimize a small edit program against measurable goals without giving an agent unrestricted pixel loops.

### 2.3 Non-goals

- Interactive GUI.
- Full PSD compatibility.
- Full Blender or Photoshop parity.
- General-purpose arbitrary code execution inside plans.
- Prompt-to-image generation as the core abstraction.
- Silent destructive neural edits.
- Bit-identical floating-point output across every GPU vendor.
- Reimplementing a complete color-management system before the core contracts work.
- Mesh topology editing in `paintop`; that remains `msh` territory.

---

## 3. Success criteria

### 3.1 First credible release

A coding agent can:

1. inspect an input image;
2. produce a valid JSON plan;
3. create and refine a mask;
4. apply color, convolution, splat, and compositing operations;
5. assert that pixels outside the authorized mask did not change;
6. run the plan against a scalar CPU reference backend and an optimized backend;
7. inspect an evidence bundle;
8. revise the plan based on numerical and visual feedback;
9. reproduce the accepted output from the normalized plan and content hashes.

### 3.2 First credible material-editing release

A coding agent can:

1. inspect a GLB while preserving its full scene/material structure;
2. request beauty and diagnostic render passes from `msh`;
3. define a screen-space region from a stable camera;
4. project that region to visible triangles and UV texels;
5. edit base color, roughness, metallic, emissive, or alpha channels;
6. export a structurally valid GLB;
7. render before/after views from multiple cameras;
8. validate no unauthorized material, texture, or texel changed;
9. inspect UV seam and multi-view consistency reports.

### 3.3 Quantitative quality bars

The initial bars should be explicit and modest:

- 100% of public operations have a machine-readable manifest and schema.
- 100% of deterministic operations have analytic or property-based tests.
- Every optimized implementation has a reference or differential oracle.
- Every run can emit a normalized plan, trace, metrics, and assertion results.
- No operation may mutate an input resource in place from the graph’s perspective.
- CPU reference outputs are reproducible for fixed version, inputs, parameters, and seed.
- GPU outputs declare tolerances and determinism tier.
- Invalid finite ranges, NaNs, malformed images, and resource-limit violations fail explicitly.
- A standard 4K masked pointwise edit should avoid touching tiles outside the propagated region of interest.
- A standard run should not require a GPU unless the selected operation explicitly does.

---

## 4. Hard design decisions

These decisions should be treated as defaults until implementation evidence disproves them.

### 4.1 JSON is the canonical IR

JSON is not chosen for beauty. It is chosen because it can be:

- generated by agents;
- parsed without executing code;
- validated before resource allocation;
- canonicalized and hashed;
- diffed and audited;
- replayed exactly;
- restricted by policy;
- translated from other frontends.

A Lua DSL may be added later, but Lua must return a data graph. Lua must never become the only representation of an edit.

### 4.2 Graph semantics, sequential syntax

Humans and agents may author a mostly sequential `ops` array. The runtime resolves named references and normalizes it into an immutable directed acyclic graph.

This yields readable plans without giving up:

- common-subexpression elimination;
- lazy execution;
- parallel scheduling;
- dead-node elimination;
- content-addressed caching;
- selective debug materialization;
- graph rewrites.

### 4.3 CPU reference semantics first

The first implementation of each foundational operation should be a clear CPU reference implementation. It may be slow. It exists to define behavior and test optimized CPU/GPU variants.

GPU-first development would maximize demo speed and minimize confidence. That is the wrong trade for an agent-built system.

### 4.4 Neural operations produce candidates or perception resources

Models are useful for segmentation, matting, depth, restoration, inpainting, intrinsic decomposition, and material inference. They are not trusted deterministic mutations.

By default, model operations produce one of:

- a `Mask`;
- a `Field` plus confidence;
- a `CandidateSet<Image>`;
- a `CandidateSet<MaterialChannels>`;
- a `Report`.

A separate explicit node selects and composites a candidate after assertions.

### 4.5 Explicit image semantics

Every image-like resource carries explicit metadata:

- extent;
- channel layout;
- scalar type;
- color encoding;
- alpha representation;
- coordinate convention;
- valid numeric range;
- semantic role.

The runtime must reject ambiguous operations instead of guessing.

### 4.6 No hidden side effects

Operations are pure from the graph’s perspective. File writes, debug artifact writes, and GLB exports are explicit sink nodes or top-level exports.

### 4.7 `msh` renders and locates; `paintop` mutates

The boundary is:

```text
msh
  asset inspection
  scene-preserving geometry/material access
  camera and render configuration
  diagnostic G-buffers
  ray/screen/surface/UV queries

paintop
  edit IR
  image/mask/field computation
  texture and material mutation
  assertions and evidence
  agent-facing edit workflow
```

Initially, `paintop` may invoke `msh` as a process. Shared Rust crates should be extracted only after the interfaces stabilize.

### 4.8 Do not extend `msh::MeshWithColors` into a scene model

The current `msh` loader flattens one selected mesh, concatenates primitives, and retains an optional first base-color texture. That is appropriate for its current viewer but insufficient for material editing. Material work requires preserving:

- scenes and nodes;
- node transforms;
- mesh/primitive identity;
- material identity;
- all texture slots;
- texture coordinate sets;
- samplers and wrap modes;
- image provenance;
- tangent data;
- relevant glTF extensions;
- byte-level export relationships.

Create a new scene-preserving asset model rather than turning `MeshWithColors` into a god object.

### 4.9 Determinism is tiered

The runtime should report a determinism class, not promise fantasy:

| Tier | Meaning |
|---|---|
| `exact` | Bit-exact for the declared scalar format and backend contract. |
| `reproducible` | Same build, backend, device class, seed, and inputs reproduce within a declared bound. |
| `bounded` | Alternate implementations agree within operation-specific absolute/relative/perceptual bounds. |
| `stochastic` | Seeded but model/provider behavior may vary; results are candidates with evidence. |

---

## 5. System architecture

```text
┌─────────────────────────────────────────────────────────────────────┐
│ Agent / CI / human                                                  │
│  JSON plan, policy, resource limits, requested evidence             │
└──────────────────────────────┬──────────────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│ paintop-cli                                                         │
│ validate | run | explain | graph | diff | selftest | bench          │
└──────────────────────────────┬──────────────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│ paintop-ir                                                          │
│ schema, resource types, op manifests, canonicalization, hashing     │
└──────────────────────────────┬──────────────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│ paintop-core                                                        │
│ type checking, policy, DAG, ROI/halo analysis, optimizer, scheduler │
│ cache, memory planner, tracing, assertions                          │
└───────────────┬───────────────────────────────┬─────────────────────┘
                ▼                               ▼
┌────────────────────────────┐   ┌────────────────────────────────────┐
│ deterministic backends     │   │ adapters                           │
│ paintop-cpu                │   │ paintop-model: ONNX/process models │
│ paintop-wgpu               │   │ paintop-msh-bridge                 │
└───────────────┬────────────┘   └─────────────────┬──────────────────┘
                ▼                                  ▼
┌────────────────────────────┐   ┌────────────────────────────────────┐
│ image/mask/field resources │   │ GLB + render passes + candidates   │
└───────────────┬────────────┘   └─────────────────┬──────────────────┘
                └────────────────────┬─────────────┘
                                     ▼
                         evidence bundle + exports
```

---

## 6. Proposed workspace

```text
paintop/
├── Cargo.toml
├── rust-toolchain.toml
├── plan.md
├── AGENTS.md
├── crates/
│   ├── paintop-ir/          # serde types, canonical graph, schema export
│   ├── paintop-core/        # compiler, scheduler, cache, policies, tracing
│   ├── paintop-image/       # image/mask/field abstractions and contracts
│   ├── paintop-cpu/         # scalar oracle + SIMD/Rayon implementations
│   ├── paintop-wgpu/        # GPU kernels and fused pointwise pipelines
│   ├── paintop-model/       # model manifests and isolated inference adapters
│   ├── paintop-material/    # material channels, UV edits, GLB mutation
│   ├── paintop-msh-bridge/  # initial process/RPC bridge to msh
│   ├── paintop-testkit/     # fixtures, metrics, metamorphic/differential harness
│   ├── paintop-cli/         # stable machine-facing CLI
│   └── xtask/               # repository automation
├── schemas/
│   ├── plan-v1.schema.json
│   ├── op-manifest-v1.schema.json
│   └── model-manifest-v1.schema.json
├── ops/
│   └── manifests/           # checked-in generated/curated op manifests
├── fixtures/
│   ├── analytic/
│   ├── photographic/
│   ├── adversarial/
│   ├── materials/
│   └── gltf/
├── conformance/
│   ├── plans/
│   ├── expected/
│   └── tolerances/
├── models/
│   └── manifests/           # no untracked mystery weights
├── benches/
├── fuzz/
└── docs/
    ├── IR_SPEC.md
    ├── AGENT_VERIFICATION.md
    ├── RESEARCH.md
    └── ALIEN_OPS.md
```

### 6.1 Dependency direction

```text
paintop-ir
    ▲
paintop-image
    ▲
paintop-core ◄──── paintop-testkit
    ▲     ▲
    │     ├──── paintop-cpu
    │     ├──── paintop-wgpu
    │     └──── paintop-model
    ▲
paintop-material ◄──── paintop-msh-bridge
    ▲
paintop-cli
```

Rules:

- `paintop-image` must not know that 3D exists.
- `paintop-cpu` and `paintop-wgpu` implement contracts owned by core/image crates.
- `paintop-material` translates surface/material intent into image/field operations.
- Core crates must not depend on a model runtime.
- CLI JSON types must reuse the canonical IR rather than invent parallel structs.

---

## 7. Resource algebra

The graph becomes more powerful when intermediate representations are first-class.

### 7.1 Initial resource kinds

| Resource | Purpose |
|---|---|
| `Image` | Typed color or scalar raster. |
| `Mask` | Coverage field constrained to `[0,1]`. |
| `SdfMask` | Signed distance to a boundary in physical pixel units. |
| `Field1` | Scalar field: depth, roughness, confidence, distance. |
| `Field2` | Vector field: flow, orientation, displacement. |
| `Field3` | Vector field: normals or 3-vector features. |
| `LabelMap` | Integer object/material/component IDs. |
| `Transform2D` | Affine/projective mapping with declared coordinate spaces. |
| `Histogram` | Distribution summary with bins and domain. |
| `Palette` | Discrete colors and weights. |
| `PatchField` | Nearest-neighbor patch correspondence and score. |
| `Pyramid` | Gaussian/Laplacian/steerable multiscale representation. |
| `CandidateSet<T>` | Ranked alternatives plus confidence and provenance. |
| `Report` | Machine-readable metrics, diagnostics, or validation result. |

Later:

| Resource | Purpose |
|---|---|
| `FeatureMap` | Opaque model embedding with strict producer/consumer compatibility. |
| `SurfaceMap` | Per-pixel triangle, barycentric, UV, depth, visibility data. |
| `MaterialSet` | Named PBR channels and metadata. |
| `MeshField` | Scalar/vector values on vertices, edges, or faces. |
| `SparseAttribution` | Per-tile or per-pixel provenance of edit influence. |

### 7.2 Why masks and fields are separate from images

Treating every raster as “an image” invites silent mistakes. A mask has closure properties, allowed ranges, morphology, and coverage semantics. A normal field has vector normalization and tangent-space semantics. Roughness is linear scalar data, not sRGB color.

Typed resources allow the compiler to reject:

- applying hue rotation to depth;
- sRGB decoding a normal map;
- alpha-compositing an integer label map;
- using a hard ID map as a feathered mask without explicit conversion;
- packing glTF roughness into the wrong channel.

---

## 8. Image and coordinate semantics

These choices must be settled before implementing dozens of operations.

### 8.1 Coordinate convention

Recommended convention:

- integer `(x, y)` identifies a pixel cell;
- pixel center is `(x + 0.5, y + 0.5)`;
- rectangles are half-open: `[x0, x1) × [y0, y1)`;
- image origin is upper-left;
- positive `x` is right, positive `y` is down;
- normalized image coordinates refer to edges: `[0,1] × [0,1]`;
- transforms explicitly declare source and destination spaces;
- resampling specifies center and boundary behavior.

This aligns naturally with common raster conventions and glTF texture-coordinate interpretation, while avoiding off-by-half ambiguity.

### 8.2 Color

Recommended internal policy:

- color computations occur in linear light;
- 8-bit sRGB is an import/export encoding, not an arithmetic space;
- alpha is linear coverage;
- compositing uses premultiplied alpha internally;
- unassociated alpha is converted explicitly at boundaries;
- scalar material maps stay linear;
- normal maps are decoded to signed vectors before vector operations;
- conversions are explicit graph nodes and visible in traces.

Initial supported encodings:

- `srgb`;
- `linear-srgb`;
- `display-p3` only when a reliable conversion backend is available;
- `raw` for data textures.

Do not implement half a color-management system. Support a narrow explicit set correctly, and reject unsupported ICC/profile behavior.

### 8.3 Numeric formats

Start with:

- `u8` for import/export and exact integer fixtures;
- `f32` for internal color, masks, fields, and reference semantics;
- optional `f16` GPU storage after equivalence tests exist;
- `u32` for label maps and IDs.

A resource carries valid-range policy:

```text
bounded: clamp or reject outside [min,max]
unbounded: finite values required
normalized-vector: finite and norm constrained
```

Clamping must be an explicit policy or node; silent clamp hides bugs.

### 8.4 Boundary conditions

Every neighborhood operation declares one of:

- `constant(value)`;
- `clamp`;
- `mirror`;
- `wrap`;
- `transparent`;
- `valid-only`.

Boundary mode is part of the operation hash and conformance tests.

---

## 9. Operation contract

Every operation has a machine-readable manifest. The manifest is not documentation garnish; it is input to validation, scheduling, testing, and agent introspection.

```yaml
name: filter.gaussian_blur
version: 1
inputs:
  image: Image
  mask: Mask?
outputs:
  image: Image
parameters:
  sigma: { type: f32, exclusive_min: 0 }
  boundary: { enum: [clamp, mirror, wrap, constant] }
semantics:
  purity: pure
  color_requirement: linear
  alpha_requirement: premultiplied
  roi: expand_by_halo
  halo: ceil(3 * sigma)
  determinism: bounded
implementations:
  - cpu.reference
  - cpu.separable
  - wgpu.separable
properties:
  - constant_preserving
  - nonnegative_kernel
  - unit_sum_kernel
  - translation_equivariant_away_from_boundaries
limits:
  sigma_max_default: 512
```

Required manifest concepts:

- stable operation ID and semantic version;
- input/output resource types;
- parameter schema and defaults;
- color/alpha expectations;
- shape inference;
- ROI propagation;
- halo computation;
- purity and cacheability;
- determinism tier;
- seed requirements;
- available implementations;
- cost-model inputs;
- mathematical properties;
- known limitations;
- debug visualizations;
- postconditions;
- reference fixture IDs.

---

## 10. Compilation and execution pipeline

A run is a compilation pipeline, not a loop over JSON objects.

### 10.1 Phases

1. **Parse**
   - Reject duplicate keys and invalid numeric forms.
   - Apply strict size/depth limits before allocating large structures.

2. **Schema validation**
   - Reject unknown fields unless nested under a namespaced `extensions` object.
   - Validate operation and plan versions.

3. **Reference resolution**
   - Resolve resources and node outputs.
   - Detect cycles and missing references.

4. **Type, shape, color, and alpha checking**
   - Infer output extents and semantics.
   - Insert no implicit conversions except identity-preserving canonicalization.

5. **Policy validation**
   - Enforce maximum pixels, nodes, splats, model calls, runtime, memory, and output paths.
   - Require seeds where appropriate.

6. **Normalization**
   - Expand syntactic sugar.
   - Canonicalize parameters.
   - Normalize operation versions.
   - Produce a stable normalized plan.

7. **Backward demand and ROI analysis**
   - Start from exports, assertions, and requested debug resources.
   - Eliminate dead nodes.
   - Propagate required regions backward through transforms and halos.

8. **Graph simplification**
   - Constant folding.
   - Identity elimination.
   - Common-subexpression elimination.
   - Safe pointwise fusion.
   - Mask algebra simplification.
   - Conversion cancellation.

9. **Algorithm selection**
   - Direct/separable/FFT convolution.
   - Sparse/dense mask handling.
   - Scalar/SIMD/GPU backend.
   - Full image/tiled/ROI execution.

10. **Memory and residency planning**
    - Liveness analysis.
    - Tile buffer reuse.
    - CPU/GPU residency grouping.
    - Readback minimization.

11. **Execution**
    - Bounded parallelism.
    - Cancellation and deadline checks.
    - Structured trace events.

12. **Assertions and differential checks**
    - Run required validators.
    - Optionally execute alternate implementation on sampled/full regions.

13. **Evidence assembly**
    - Atomic outputs.
    - Hashes, metrics, contact sheets, masks, graph, and trace.

### 10.2 Immutable lazy resources

Resources are immutable logical values. Physical buffers may be reused when liveness proves safety, but that is invisible to graph semantics.

The executor should borrow from the strengths of libvips and Halide:

- demand-driven regions;
- tile/strip execution;
- immutable operation graphs;
- separation of algorithm from schedule;
- operation caching;
- bounded working sets.

### 10.3 Content-addressed caching

Cache keys should include:

```text
hash(
  op_id + op_semantic_version +
  canonical_parameters +
  ordered_input_content_hashes +
  resource_semantics +
  seed +
  model_hash + model_runtime_contract +
  backend_semantics_version
)
```

Use BLAKE3 or another fast cryptographic hash. Cache entries should include validation metadata and must not be reused across incompatible semantic versions.

Do not hash a path as the input identity; hash content and relevant metadata.

---

## 11. Tile and ROI execution

### 11.1 Why tiling is foundational

A 4K image in several `f32` channels plus intermediates can consume hundreds of megabytes. Material workflows may involve multiple 4K or 8K maps. Most touch-ups affect a small region.

Tile execution enables:

- bounded memory;
- sparse masks;
- cache locality;
- parallelism;
- partial recomputation;
- efficient debug sampling;
- incremental agent iterations.

### 11.2 Suggested tile model

- logical base tile: configurable, initially 128×128 or 256×256;
- each operation computes an input halo per output tile;
- masks carry tile occupancy summaries: empty, full, mixed;
- empty masked tiles become identity without evaluating the expensive branch;
- full masked tiles skip blend-mask reads when legal;
- transforms request conservative source footprints;
- reductions use deterministic merge trees on the reference backend.

### 11.3 ROI propagation

Each operation implements a backward mapping:

```text
required_input_region = roi(op, required_output_region, parameters)
```

Examples:

- pointwise op: same region;
- Gaussian blur: expand by kernel support;
- affine warp: inverse-transform output bounds, then expand by reconstruction support;
- Poisson solve: generally the connected masked domain plus boundary ring;
- FFT operation: may force a larger block or whole image depending on algorithm;
- segmentation model: whole resized input unless model supports crops;
- composite: source and target requirements depend on mask occupancy.

### 11.4 Sparse mask index

Maintain a compact occupancy structure over mask tiles:

```text
MaskTileSummary {
    min_coverage,
    max_coverage,
    nonzero_count,
    bounds_of_nonzero,
}
```

This supports fast:

- no-op elimination;
- assertion bounds;
- scheduling;
- changed-region prediction;
- GPU dispatch compaction.

---

## 12. Backend strategy

### 12.1 CPU reference

Requirements:

- straightforward loops;
- stable accumulation order where practical;
- explicit boundary behavior;
- no architecture-dependent approximate math unless declared;
- small, readable functions;
- exact or tight analytic tests.

This backend is the semantic oracle.

### 12.2 Optimized CPU

Use:

- Rayon for tile-level parallelism;
- explicit SIMD where benchmarks prove value;
- cache-aware separable passes;
- row/column transposition where beneficial;
- vectorized pointwise kernels;
- precomputed kernels and lookup tables with versioned precision;
- thread caps from policy.

Avoid nested uncontrolled parallelism. The scheduler owns parallelism; individual ops should request execution strategies rather than spawning arbitrary thread pools.

### 12.3 `wgpu`

Use GPU execution for:

- large pointwise chains;
- large splat batches;
- separable filters;
- morphology/distance approximations when exactness is not required;
- pyramids and reductions;
- texture-map workflows already resident on GPU;
- differentiable or iterative local optimization kernels.

Design rules:

- group GPU-resident subgraphs to avoid readbacks;
- compile/cache pipelines by normalized fused expression and format;
- use storage textures/buffers intentionally;
- validate dispatch dimensions and overflow;
- expose adapter/device identity in evidence;
- keep the CPU oracle available for sampled differential checks.

`msh` currently uses `wgpu` 27. Do not combine renderer extraction with a `wgpu` major upgrade. First share interfaces at the existing version; perform an isolated upgrade with render-golden and cross-backend tests later.

### 12.4 Model adapter

Start with an isolated adapter, preferably ONNX Runtime where model export quality permits it. Allow an out-of-process protocol for models that require Python/PyTorch.

The core runtime should see:

```text
ModelManifest + typed inputs + typed outputs + evidence
```

not framework-specific objects.

A model manifest includes:

- immutable model hash;
- source and license;
- accepted tensor shapes/types;
- preprocessing and postprocessing versions;
- output semantics;
- execution providers tested;
- seed behavior;
- determinism tier;
- memory/runtime limits;
- reference vectors;
- known failure modes.

---

## 13. Algorithm selection and performance

### 13.1 Convolution chooser

Convolution is not one implementation. The compiler should select among:

1. **Unrolled direct convolution** for tiny dense kernels.
2. **Sparse direct convolution** when most coefficients are zero.
3. **Separable convolution** when the kernel is exactly rank 1.
4. **Low-rank factorization** using SVD when approximation is allowed:

   \[
   K \approx \sum_{i=1}^{r} \sigma_i u_i v_i^T
   \]

   Choose the smallest `r` satisfying an operation-specified Frobenius or output-error budget.

5. **Integral-image methods** for box-like filters.
6. **Recursive/IIR approximations** for very large Gaussian radii when the requested contract permits approximation.
7. **FFT overlap-save/overlap-add** for large kernels and large dense regions.

The chosen algorithm and estimated error must appear in the trace.

### 13.2 Pointwise fusion

Fuse chains such as:

```text
linearize → matrix color transform → exposure → saturation → mask blend → encode
```

into one CPU loop or generated WGSL kernel when:

- the operations are pure and pointwise;
- precision and clamping order are preserved;
- debug materialization does not require intermediates;
- the rewrite has conformance tests.

Do not algebraically reorder operations just because they look similar. Color transforms, clamping, premultiplication, and nonlinear transfer functions are order-sensitive.

### 13.3 Autotuning

Static heuristics will be wrong across CPUs and GPUs. Add a device-profile autotuner after correctness:

```text
cost = f(op, extent, ROI density, halo, format, channels, backend, device)
```

At install/selftest time, benchmark a bounded matrix of representative kernels. Cache the chosen schedule keyed by hardware and runtime version.

Autotuning must never change semantics, only select among conforming implementations.

### 13.4 Equality saturation, later

An e-graph optimizer can explore equivalent graph forms without committing to one rewrite order. Candidate equivalences:

- merge consecutive affine transforms;
- combine compatible color matrices;
- push/collapse conversions;
- simplify mask Boolean/SDF algebra;
- choose dense vs separable vs low-rank convolution representations;
- fuse pointwise expressions;
- eliminate redundant resampling.

This is post-v1. Every rewrite needs a proof sketch, generated property tests, and cost extraction that respects precision and debug barriers.

---

## 14. Initial operation coverage

The detailed catalog belongs in op manifests. The implementation sequence matters more than total count.

### 14.1 Phase A: substrate

- `io.decode_image`
- `io.encode_image`
- `image.create`
- `image.convert_format`
- `color.decode_transfer`
- `color.encode_transfer`
- `alpha.premultiply`
- `alpha.unpremultiply`
- `debug.export`
- `report.inspect`

### 14.2 Phase B: masks and geometry

- rectangle, ellipse, polygon, path rasterization;
- alpha/luminance/color-range masks;
- invert/union/intersect/subtract/xor;
- threshold/smoothstep;
- exact Euclidean distance transform;
- `Mask ↔ SdfMask`;
- grow/shrink/feather via signed distance;
- connected components, fill holes, remove small components;
- affine and projective transforms;
- crop/pad/resize/warp;
- nearest, bilinear, bicubic, and Lanczos resampling with fixed definitions.

### 14.3 Phase C: edits

- fill color;
- linear/radial/conic gradients;
- Gaussian and anisotropic splats;
- pointwise color adjustments;
- curves/levels;
- blend/composite modes;
- Gaussian/box/median/bilateral/guided filters;
- sharpen/unsharp;
- morphology;
- copy/clone/paste;
- pyramids and frequency split/recombine;
- diff and heatmap.

### 14.4 Phase D: advanced classical operations

- Poisson/screened-Poisson blend;
- structure tensor and orientation field;
- anisotropic diffusion;
- PatchMatch nearest-neighbor field;
- patch fill and texture synthesis;
- boundary-conditioned heal;
- reaction-diffusion and procedural fields;
- domain warp;
- local optimization over splats and compact edit programs.

### 14.5 Phase E: perception/model operations

- promptable segmentation;
- matte refinement;
- depth estimation;
- normal estimation;
- restoration candidate generation;
- masked inpainting candidates;
- intrinsic decomposition candidates;
- material-channel inference candidates.

### 14.6 Phase F: material operations

- glTF texture slot extraction and replacement;
- linear/raw channel handling;
- metallic/roughness pack/unpack;
- normal-map decode/renormalize/encode;
- material parameter mutation;
- screen mask to surface map;
- surface map to UV coverage;
- seam-aware dilation and filtering;
- geodesic surface selection;
- multi-view projection and consistency reports.

---

## 15. Agent verification is part of the runtime

The project is being built by agents. Therefore, each layer needs executable self-knowledge.

### 15.1 Evidence bundle

A normal run with `--bundle DIR` emits:

```text
DIR/
├── manifest.json              # version, platform, hashes, policy, backend
├── input-manifest.json        # content hashes and decoded semantics
├── normalized-plan.json       # exact graph executed
├── graph.dot
├── trace.jsonl                # structured per-node events
├── metrics.json
├── assertions.json
├── outputs/
├── intermediates/             # requested or failure-relevant only
├── masks/
├── diffs/
├── contact-sheet.png
└── logs/
```

### 15.2 Verification layers

Every nontrivial operation should be checked through several independent lenses:

1. **Analytic fixtures**
   - impulse, step, ramp, sine, checker, constant field, exact geometry.

2. **Property tests**
   - range, identity, monotonicity, normalization, conservation, idempotence, inverse relationships.

3. **Metamorphic tests**
   - translate input and selection, then translate output;
   - tile/reassemble equivalence;
   - split a batch of splats and compose;
   - scale coordinates and sigma consistently;
   - reorder operations only when algebra says they commute.

4. **Differential tests**
   - scalar CPU vs optimized CPU;
   - CPU vs GPU;
   - direct vs separable vs FFT;
   - dense vs tiled;
   - whole image vs ROI;
   - internal implementation vs a trusted external reference for sampled cases.

5. **Golden tests**
   - exact for integer/reference subsets;
   - bounded numeric and perceptual thresholds for floating/GPU output.

6. **Fuzzing**
   - decoders;
   - JSON/schema parser;
   - graph validation;
   - dimensions/strides/overflow;
   - operation parameters;
   - malformed GLB structures.

7. **Performance regression tests**
   - runtime distribution;
   - peak resident memory;
   - GPU allocation/readback counts;
   - cache hit rate;
   - unnecessary tile evaluation.

### 15.3 Runtime assertions

Plans may include assertions such as:

- no change outside mask;
- changed bounds contained in region;
- minimum change inside mask;
- maximum absolute/relative delta;
- finite pixels only;
- alpha in range;
- image mean/histogram target;
- edge continuity improves;
- high-frequency energy preserved;
- number of connected components unchanged;
- normal vectors remain unit length;
- roughness and metallic remain in `[0,1]`;
- UV seam disagreement below tolerance;
- GLB structure/material assignment unchanged except declared targets.

An assertion failure is a failed run, not a warning hidden in logs.

### 15.4 Agent-facing CLI contract

Stdout should be pure machine-readable JSON for machine modes. Logs go to stderr.

```bash
paintop validate plan.json
paintop explain plan.json --format json
paintop run plan.json --bundle run/
paintop graph plan.json --out graph.svg
paintop diff before.png after.png --bundle diff/
paintop op list --format json
paintop op schema filter.gaussian_blur@1
paintop selftest --backend cpu-reference
paintop selftest --backend all
paintop conformance run
paintop bench smoke
paintop model verify models/manifests/sam2.json
cargo xtask verify-op filter.gaussian_blur
```

Stable exit classes:

```text
0 success
2 parse/schema error
3 type/semantic error
4 policy/resource-limit rejection
5 execution failure
6 assertion failure
7 differential/conformance failure
8 model adapter failure
9 export/asset integrity failure
```

### 15.5 Task contract for coding agents

Every implementation task should state:

- operation or subsystem contract;
- accepted inputs and outputs;
- exact or bounded semantics;
- fixtures to add;
- properties to test;
- alternate oracle;
- failure behavior;
- performance budget;
- commands that prove completion;
- evidence artifact path.

No task should be “implement blur.” It should be “implement `filter.gaussian_blur@1` reference semantics and prove properties X–Y under boundary modes A–D.”

See `docs/AGENT_VERIFICATION.md` and `AGENTS.md`.

---

## 16. `msh` integration plan

### 16.1 Current useful capabilities

`msh` already provides:

- Rust/`wgpu` rendering;
- an interactive viewer;
- headless PNG rendering;
- camera and projection controls;
- multi-angle/sprite-sheet rendering;
- GLB inspection;
- JSON-RPC control and screenshots;
- UV coordinates and a base-color texture in its flattened render path.

This makes it a strong preview and geometric-query substrate.

### 16.2 Required new scene-preserving layer

Add or extract an asset crate with data structures roughly like:

```rust
pub struct Asset {
    pub scenes: Vec<Scene>,
    pub nodes: Vec<Node>,
    pub meshes: Vec<Mesh>,
    pub materials: Vec<Material>,
    pub textures: Vec<Texture>,
    pub images: Vec<ImageAsset>,
    pub samplers: Vec<Sampler>,
    pub source: AssetSource,
}

pub struct Primitive {
    pub id: PrimitiveId,
    pub indices: BufferView,
    pub attributes: VertexAttributes,
    pub material: Option<MaterialId>,
    pub mode: PrimitiveMode,
}
```

Identity must survive round trips. Avoid assigning meaning solely by array position when a stable internal ID can be derived.

### 16.3 Diagnostic render passes

`msh-render` should eventually expose a multi-render-target or configurable-pass API for:

- beauty;
- unlit base color;
- metallic;
- roughness;
- emissive;
- alpha/coverage;
- world/view/tangent normals;
- linear depth;
- object ID;
- mesh ID;
- primitive ID;
- triangle ID;
- material ID;
- UV0/UV1;
- barycentric coordinates;
- front/back facing;
- optional motion vectors.

Integer IDs should use integer attachments where supported rather than encoding large IDs into lossy sRGB PNGs. Export an exact binary/NPY-like sidecar when PNG cannot preserve the data.

### 16.4 Screen-to-surface query

Given a camera and screen sample, return:

```json
{
  "hit": true,
  "node": 4,
  "mesh": 2,
  "primitive": 1,
  "triangle": 18422,
  "material": 5,
  "barycentric": [0.2, 0.6, 0.2],
  "uv0": [0.734, 0.421],
  "world_position": [0.4, 1.1, -0.2],
  "world_normal": [0.1, 0.9, 0.3],
  "linear_depth": 2.71,
  "front_facing": true
}
```

For masks, return a `SurfaceMap` raster rather than millions of JSON samples.

### 16.5 Projection painting

Naively scattering screen pixels into nearest UV texels will alias and leave holes. Use an anisotropic footprint derived from the local Jacobian:

\[
J = \frac{\partial (u,v)}{\partial (x,y)}
\]

A screen-space Gaussian with covariance `Σ_screen` maps to:

\[
Σ_{uv} = J Σ_{screen} J^T + εI
\]

Rasterize an elliptical weighted average footprint in UV space, weighted by:

- screen mask coverage;
- visibility;
- foreshortening;
- sample confidence;
- material/primitive filter;
- optional depth discontinuity rejection.

This is both a useful practical algorithm and a strong candidate for a novel `paintop` surface-splat primitive.

### 16.6 UV pathologies must be policy-visible

The system must report and require policy for:

- missing UVs;
- overlapping UVs;
- mirrored UVs;
- degenerate UV triangles;
- UVs outside `[0,1]` with wrap modes;
- multiple UV sets;
- texture transforms;
- shared textures across materials;
- compressed/KTX2 images;
- insufficient gutters;
- animated/skinned geometry.

Never silently “fix” these during an edit.

---

## 17. Security, safety, and supply chain

### 17.1 Plan sandbox

JSON plans contain no arbitrary filesystem traversal or process execution. Inputs and outputs are resolved under explicit roots or passed as pre-opened handles.

Enforce:

- maximum decoded pixels;
- maximum dimensions;
- maximum graph nodes;
- maximum parameter nesting/string sizes;
- maximum splats/patch iterations/model calls;
- CPU wall/compute budget;
- GPU memory budget;
- output byte budget;
- cancellation and deadlines;
- atomic writes and no overwrite unless allowed.

### 17.2 Decoder hardening

Image and GLB decoders are attack surfaces. Add:

- library decode limits;
- integer overflow checks;
- corpus fuzzing;
- truncated/malformed fixture sets;
- decompression-bomb tests;
- isolated model adapters where practical.

### 17.3 Model provenance

Do not accept “download this checkpoint” as a reproducible dependency. Every model needs:

- source URL;
- license summary;
- exact checksum;
- model format/opset;
- preprocessing code/version;
- test vectors;
- expected resource envelope;
- known provider differences.

No model weights should be required for core conformance.

---

## 18. Observability and debugging

### 18.1 Structured trace

Per-node trace events should include:

```json
{
  "node": "blur_1",
  "op": "filter.gaussian_blur@1",
  "implementation": "wgpu.separable@1",
  "input_regions": {"image": [128, 64, 768, 512]},
  "output_region": [160, 96, 704, 448],
  "halo": 32,
  "tiles": {"requested": 12, "executed": 8, "identity": 4},
  "elapsed_ms": 1.82,
  "alloc_bytes": 4194304,
  "cache": "miss",
  "numeric_contract": {"max_abs_error": 0.00002},
  "warnings": []
}
```

### 18.2 Failure-driven materialization

Do not write every intermediate by default. On assertion or differential failure, materialize:

- failing input tile plus halo;
- reference output;
- optimized output;
- absolute/relative diff;
- masks and fields involved;
- operation parameters;
- minimal replay plan.

This creates a self-contained bug specimen an agent can attack.

### 18.3 Provenance

Every exported resource records:

- producer node;
- transitive input hashes;
- operation semantic versions;
- implementation choices;
- model hashes;
- assertions passed;
- timestamp excluded from content identity.

Later, add sparse delta provenance: identify which operation most influenced changed pixels or tiles.

---

## 19. Milestones

Milestones are gated by executable exit criteria, not dates.

### M0 — repository and contracts

Deliver:

- Rust workspace;
- formatting/lint/test/fuzz scaffolding;
- strict plan parser;
- operation manifest model;
- schema generation;
- resource metadata model;
- `paintop validate`, `op list`, and `op schema`;
- analytic fixture generator;
- evidence-bundle skeleton;
- `AGENTS.md`.

Exit criteria:

```bash
cargo fmt --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace
paintop validate fixtures/plans/empty-valid.json
paintop validate fixtures/plans/unknown-field.json   # must fail predictably
```

### M1 — exact 2D CPU core

Deliver:

- decode/encode PNG;
- typed `Image`, `Mask`, `Field1`;
- explicit linearization and premultiplication;
- primitive masks;
- exact mask algebra;
- pointwise adjustments;
- alpha-over composite;
- Gaussian splats;
- direct convolution;
- diff and assertions;
- scalar CPU oracle;
- evidence contact sheet.

Exit criteria:

- all operations have manifests;
- analytic/property tests pass;
- unauthorized-pixel assertion is demonstrated;
- one agent can complete a fixture edit using only CLI introspection and evidence.

### M2 — graph compiler, tiles, and cache

Deliver:

- normalized DAG;
- dead-node elimination;
- backward ROI/halo propagation;
- tile scheduler;
- sparse mask summaries;
- deterministic reductions;
- content-addressed cache;
- execution trace and graph visualization.

Exit criteria:

- whole-image and tiled outputs are identical for exact operations;
- ROI execution is differentially equivalent to full execution;
- a small masked edit on 4K touches only predicted tiles;
- cache replay shows no unnecessary execution.

### M3 — optimized CPU and `wgpu`

Deliver:

- Rayon tile parallelism;
- SIMD pointwise path;
- separable Gaussian;
- GPU pointwise fusion;
- GPU splat batch;
- GPU separable filters;
- cross-backend differential harness;
- backend/device evidence.

Exit criteria:

- every GPU op passes tolerance contracts against CPU reference;
- no unplanned readback in a fully GPU-compatible chain;
- performance baselines are checked into CI artifacts;
- GPU absence yields a clean fallback or explicit unsupported error.

### M4 — classical “magic”

Deliver:

- exact Euclidean distance transform and SDF mask algebra;
- pyramids/frequency split;
- structure tensor/orientation field;
- guided or bilateral filtering;
- Poisson/screened-Poisson solver;
- PatchMatch and patch fill;
- procedural noise/domain warp/reaction-diffusion;
- contract-driven local optimizer.

Exit criteria:

- each solver exposes convergence metrics;
- each iterative operation has deterministic seed/ordering rules;
- each operation has synthetic fixtures with known or bounded outcomes;
- performance and memory scale are characterized.

### M5 — model adapter and perception

Deliver:

- model manifest schema;
- ONNX Runtime adapter;
- optional process adapter;
- one promptable segmentation model;
- one matte refinement or depth model;
- confidence fields;
- candidate-set semantics;
- model verification command.

Exit criteria:

- core test suite runs with no model downloads;
- model adapter validates exact weight hash and preprocessing;
- outputs are constrained to declared types/ranges;
- candidate outputs cannot overwrite source without explicit selection/composite;
- failure/timeout/OOM paths are tested.

### M6 — extract `msh` render/asset boundaries

Deliver in `msh`:

- scene-preserving asset model;
- render library API independent of CLI/window;
- stable camera serialization;
- beauty and initial diagnostic passes;
- exact ID/depth/UV readback;
- screen-to-surface query;
- existing CLI behavior preserved.

Exit criteria:

- old `msh` commands and goldens still pass;
- no `wgpu` major upgrade in the extraction PR;
- full material/primitive identity survives load-render-query;
- representative GLBs retain scene/material structure.

### M7 — material editing

Deliver:

- GLB material/texture extraction and replacement;
- PBR channel semantics;
- screen-mask projection to UV coverage;
- anisotropic UV splatting;
- seam/gutter diagnostics;
- base-color/roughness/metallic/emissive edits;
- multi-view evidence bundle;
- asset-integrity assertions.

Exit criteria:

- a screen-local edit changes only the intended material/UV region;
- before/after render evidence covers at least four views;
- shared/mirrored/overlapping UV behavior is explicit and tested;
- exported GLB passes structural validation and reloads identically outside declared mutations.

### M8 — compiler research and alien operations

Deliver selectively:

- low-rank convolution factoring;
- backend autotuning;
- equality-saturation graph optimization;
- topology-aware seam-graph diffusion;
- multi-view inverse material optimization;
- delta provenance;
- optional Lua graph builder.

Exit criteria:

- each research feature beats a defined baseline;
- each optimization can be disabled;
- normalized semantics remain reproducible;
- no feature lands solely because the demo looks cool.

---

## 20. Suggested first 24 pull requests

Keep early PRs small enough for agents to reason about and verify.

1. Workspace, CI, `xtask`, lint policy.
2. Resource metadata and coordinate/color enums.
3. Plan parser with duplicate-key and size rejection.
4. Operation registry and manifest schema.
5. Canonical JSON normalization and hash.
6. Analytic fixture generator.
7. Evidence bundle and structured trace skeleton.
8. PNG decode/encode with limits.
9. Linear-sRGB and premultiplied-alpha conversions.
10. Rect/ellipse/polygon mask rasterization.
11. Mask Boolean algebra.
12. Mask inspection, bounds, occupancy summaries.
13. Pointwise expression engine on scalar CPU.
14. Alpha-over composite and masked replace.
15. Gaussian splats CPU reference.
16. Direct convolution CPU reference.
17. Diff resources and core assertions.
18. Graph resolution/type checking.
19. ROI and halo interfaces with tests.
20. Tile executor for pointwise operations.
21. Tiled convolution and whole-image differential tests.
22. Content-addressed cache.
23. Contact-sheet/debug visualization.
24. End-to-end “agent edits blemish” conformance scenario.

After PR 24, stop and evaluate whether the operation contracts and evidence are actually pleasant for an agent. Do not sprint into GPU code if the graph language is still awkward.

---

## 21. Assumptions challenged

### “Lua will make the system expressive.”

True, but premature. It also introduces execution, sandbox, control-flow, and reproducibility complexity. JSON IR plus reusable subgraphs and parameter binding may cover most needs. Add Lua only after repeated plans demonstrate painful static duplication.

### “GPU means high performance.”

Not for small masked edits. Dispatch, upload, synchronization, and readback can dominate. The scheduler needs a cost model and a good CPU path.

### “A perceptual metric can decide whether an edit is good.”

No single metric captures edit intent. LPIPS/DISTS-like measures are secondary evidence. Hard invariants and task-specific assertions are primary.

### “Segmentation solves targeting.”

It helps, but segmentation masks are often too coarse for touch-up. Matte refinement, edge-aware morphology, confidence, and agent-visible overlays remain necessary.

### “Image coordinates map cleanly to mesh textures.”

They do not. Visibility, foreshortening, seams, overlap, mirroring, wrap modes, and texture transforms complicate projection. Surface-aware footprints and policies are mandatory.

### “A model can infer physically correct PBR maps from one image.”

At best it produces plausible candidates under priors. Material inference is ill-posed. Treat inferred channels as uncertain hypotheses and validate through multi-view rendering.

### “Bit-exact GPU output is the right goal.”

Only for restricted integer operations. For floating kernels, define stable reference semantics and bounded cross-backend error.

### “More operations make the tool more useful.”

Only if operations compose and are diagnosable. A smaller algebra with typed fields and excellent contracts is more powerful than hundreds of opaque filters.

---

## 22. Risks and mitigations

| Risk | Mitigation |
|---|---|
| IR churn | Version operations independently; normalize old syntax; keep plans small during M0–M2. |
| Floating-point backend drift | CPU oracle, per-op tolerances, sampled differential checks, tiered determinism. |
| GPU complexity consumes project | Gate GPU until M1/M2 semantics and evidence are complete. |
| Neural dependency sprawl | Isolated adapter, immutable manifests, no model required for core. |
| Agent writes tests that mirror implementation bugs | Require analytic, metamorphic, and alternate-oracle tests. |
| Visual goldens are flaky | Exact scalar subsets; tolerance maps; perceptual metrics only as secondary checks. |
| Cache returns semantically stale output | Include op/model/backend semantic versions and resource metadata in keys. |
| Hidden color/alpha bugs | Explicit typed conversions; test transparent colored pixels and linear-light fixtures. |
| GLB export corrupts unrelated data | Scene-preserving asset model, byte/structure diffs, allowlisted mutation assertions. |
| UV projection creates holes/seams | Anisotropic footprints, coverage accumulation, seam graph, gutters, multi-view checks. |
| Agent cannot understand failure | Minimal replay plan, failing tiles, reference/optimized diffs, structured exit classes. |
| “Alien” features become research sink | Each feature needs baseline, measurable win, disable switch, and bounded milestone. |

---

## 23. Open questions to resolve experimentally

1. Is 128×128 or 256×256 the better default tile on representative CPUs/GPUs?
2. Is `f32` everywhere initially acceptable, or does memory pressure justify `f16` storage earlier?
3. How much schema verbosity will agents tolerate before reusable subgraph templates are needed?
4. Should masks use dense `f32`, `f16`, or hybrid sparse/dense storage?
5. Which operation subset deserves exact fixed-point semantics?
6. Is ONNX Runtime sufficient for the first segmentation/depth models, or is a process adapter required first?
7. Should `msh` and `paintop` eventually share a workspace or remain repos connected by stable crates/protocols?
8. What exact GLB write strategy preserves unknown extensions and binary layout most safely?
9. Can UV projection use a G-buffer alone, or is exact CPU ray/triangle reconstruction needed for validation?
10. Which perceptual metrics correlate with our actual touch-up tasks rather than generic image similarity?
11. Can an e-graph optimizer justify its complexity before there are dozens of algebraic operations?
12. Does sparse delta provenance need per-pixel precision, or is tile/node attribution enough?

These are benchmark questions, not architecture debates. Build minimal probes and collect evidence.

---

## 24. Definition of done for an operation

An operation is not done when it renders a plausible image. It is done when:

- its semantic version and manifest exist;
- inputs, outputs, coordinate, color, alpha, and boundary behavior are explicit;
- invalid parameters fail clearly;
- the CPU reference implementation exists or a justified alternate oracle is documented;
- analytic fixtures exist;
- property tests exist;
- at least one metamorphic relation is tested where applicable;
- optimized variants are differentially tested;
- ROI and halo behavior are tested;
- empty/full/mixed mask behavior is tested;
- finite/range postconditions are tested;
- trace and evidence output identify the implementation used;
- benchmarks cover representative sizes and ROI densities;
- documentation contains a minimal valid plan;
- `cargo xtask verify-op OP_ID` passes.

---

## 25. Immediate next action

Implement M0 and the first half of M1 without GPU, Lua, models, or GLB mutation.

The first end-to-end conformance scenario should be deliberately boring and strict
(non-SDF MVP variant; see [`M0_DECISIONS.md`](M0_DECISIONS.md)):

```text
load a photographic fixture
→ inspect it (extent, ranges, content hash)
→ create a soft-edged ellipse mask (analytic feather by a physical pixel radius)
→ paint a bounded batch of Gaussian splats onto an edit layer
→ apply a linear-light color adjustment to the edit layer
→ composite the edit over the source through the authorized mask
→ assert no change outside the authorized mask
→ emit output, mask, diff, contact sheet, normalized plan, and trace
→ rerun and verify identical CPU-reference content hash
```

The signed-distance feather/grow path (`mask.to_sdf` → `sdf.offset` → `sdf.to_mask`)
is deferred out of this first slice and lands immediately after the loop is green
("M1.5"), per [`M0_DECISIONS.md`](M0_DECISIONS.md) D1.

If an agent can implement, run, diagnose, and revise that workflow autonomously, the project has a viable foundation. Everything else—including the alien mathematics—is leverage on top of that contract.

---

## 26. Primary references

Research synthesis and adoption notes live in [`docs/RESEARCH.md`](docs/RESEARCH.md). Core sources include:

- libvips demand-driven execution: <https://www.libvips.org/API/current/how-it-works.html>
- Halide: <https://halide-lang.org/>
- OpenCV G-API graph model: <https://docs.opencv.org/4.x/d0/d1e/gapi.html>
- `egg` equality saturation: <https://arxiv.org/abs/2004.03082>
- `wgpu`: <https://docs.rs/wgpu/latest/wgpu/>
- ONNX Runtime Rust bindings: <https://docs.rs/ort/latest/ort/>
- glTF 2.0 specification: <https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html>
- Segment Anything 2: <https://arxiv.org/abs/2408.00714>
- ZIM matting: <https://arxiv.org/abs/2411.00626>
- Depth Anything V2: <https://arxiv.org/abs/2406.09414>
- LaMa inpainting: <https://arxiv.org/abs/2109.07161>
- Material Anything: <https://arxiv.org/abs/2411.15138>
- Geodesics in Heat: <https://arxiv.org/abs/1204.6216>
- `msh`: <https://github.com/bobisme/msh>
