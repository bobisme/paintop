# Research notes and assumption audit

**Snapshot date:** 2026-06-20  
**Scope:** execution engines, verification, image processing, perception models, restoration, PBR/material inference, and surface-aware editing  
**Caution:** many recent methods below are preprints. Paper claims are research signals, not automatic engineering dependencies. Model licenses, weight availability, exportability, memory use, and provider behavior must be verified before adoption.

---

## 1. Executive synthesis

The research supports five major conclusions.

### 1.1 The execution engine should look more like libvips/Halide than Photoshop scripting

The useful lineage is:

- immutable operation graph;
- demand-driven regions;
- tile/strip execution;
- algorithm separated from schedule;
- multiple backends;
- operation fusion and caching.

libvips demonstrates the practical value of lazy, demand-driven regions and low-memory pipelines. Halide demonstrates that the image algorithm and its execution schedule should be separable. OpenCV G-API and GEGL validate graph-based operation systems, though neither gives exactly the agent-facing contracts required here.

### 1.2 The “alien artifact” is an algebra of intermediate representations

Recent models are most useful when converted into typed intermediate resources:

```text
segmentation model → Mask
matting model      → Mask + confidence
monocular depth    → Field1 + confidence
normal estimator   → Field3 + confidence
restoration model  → CandidateSet<Image>
material estimator → CandidateSet<MaterialChannels>
```

This makes them composable with deterministic operations and assertions. Exposing a generic “AI edit” op would discard most of the architectural advantage.

### 1.3 Verification must be multi-oracle

Recent work on metamorphic testing and GPU differential testing reinforces the need for:

- analytic fixtures;
- algebraic properties;
- metamorphic relationships;
- scalar/optimized/GPU differential tests;
- hard region/range/integrity assertions;
- perceptual metrics only as secondary signals.

### 1.4 Material inference is useful but ill-posed

Recent methods can infer plausible PBR channels from photographs or mesh renders, but inverse rendering from limited views is underdetermined. Material estimation should therefore produce candidates and uncertainty, then be validated by rendering across views and by edit-locality constraints.

### 1.5 Surface-aware editing requires geometry-aware sampling, not UV pixel hacks

For projected paint, a screen pixel maps to an anisotropic footprint in UV space. Visibility, seams, overlap, mirroring, texture transforms, and surface geodesics are first-class concerns. The right tools include:

- diagnostic G-buffers;
- barycentric interpolation;
- screen-to-UV Jacobians;
- elliptical weighted footprints;
- seam graphs;
- heat-method geodesics;
- optional differentiable rendering for local optimization.

---

## 2. Execution engines and compilers

### 2.1 libvips

**Source:** <https://www.libvips.org/API/current/how-it-works.html>

Relevant ideas:

- operations form a demand-driven graph;
- only required image regions are computed;
- images are effectively immutable at the operation level;
- pipelines can process partial images and avoid materializing every intermediate;
- operation caching and SIMD reduce repeated work and memory.

**Adopt now:**

- lazy immutable graph;
- backward region demand;
- bounded tiles;
- operation cache;
- avoid full intermediate images.

**Do not copy blindly:**

- `paintop` requires typed masks/fields/material channels rather than a mostly image-centric API;
- agent evidence may intentionally materialize selected intermediates;
- GPU residency requires a schedule beyond a CPU streaming pipeline.

### 2.2 Halide

**Sources:**

- <https://halide-lang.org/>
- GPU autoscheduling: <https://arxiv.org/abs/2012.07145>
- formal semantics work: <https://arxiv.org/abs/2210.15740>
- guided optimization: <https://arxiv.org/abs/2107.12567>

Halide’s decisive idea is separation of:

```text
algorithm: what value each output pixel means
schedule:  where, when, tiled how, vectorized how, on which device
```

**Adopt now:** operation semantics separate from implementation/schedule.

**Adopt later:** device-specific autotuning and learned/analytic cost models.

**Avoid initially:** creating a new general-purpose image DSL/compiler. `paintop` can use fixed operation contracts and a simpler scheduler before considering code generation beyond fused pointwise kernels.

### 2.3 OpenCV G-API

**Source:** <https://docs.opencv.org/4.x/d0/d1e/gapi.html>

G-API validates a graph model with backend execution. Its value here is architectural precedent, especially for operation declarations and alternate backends.

**Gap for `paintop`:** it is not designed around immutable evidence bundles, strict image semantics, agent-safe policies, candidate sets, or material projection.

### 2.4 GEGL

**Source:** <https://www.gegl.org/operations/>

GEGL demonstrates a broad graph of image operations and is useful as an operation-coverage reference. It is also a warning: a huge catalog can become inconsistent if coordinate, color, alpha, and verification contracts are not uniform.

### 2.5 Equality saturation with `egg`

**Source:** <https://arxiv.org/abs/2004.03082>

Equality saturation uses an e-graph to represent many equivalent expressions simultaneously, then extracts a low-cost form. This is unusually well matched to a mature `paintop` graph because there may be many legal equivalents:

- fused pointwise expressions;
- combined transforms;
- conversion cancellation;
- dense/separable/low-rank kernel representations;
- mask algebra rewrites.

**Adoption:** research milestone only. Handwritten canonical simplifications are enough initially. E-graphs become justified when rewrite ordering and backend-dependent extraction are real bottlenecks.

### 2.6 Rust runtime choices

#### `wgpu`

Official Rust docs: <https://docs.rs/wgpu/latest/wgpu/>

`wgpu` gives a safe cross-platform graphics/compute abstraction. It fits both `paintop` GPU kernels and `msh` rendering. The main risks are API churn, shader/pipeline management, backend numerical differences, and readback cost.

`msh` currently pins `wgpu` 27 in its `Cargo.toml`. Current published docs in this research snapshot are newer. Renderer extraction and dependency upgrades should be separate changes.

#### ONNX Runtime / `ort`

Official Rust docs: <https://docs.rs/ort/latest/ort/>

The `ort` crate provides ONNX Runtime bindings and multiple execution providers. It is a reasonable first model adapter because it avoids embedding Python for models that export cleanly.

Caveats:

- ONNX export quality varies by model;
- preprocessing/postprocessing often contain significant semantics;
- provider output and determinism may differ;
- some research models rely on unsupported custom operators.

Use an out-of-process adapter as an escape hatch, not as core architecture.

#### `image`

Official docs: <https://docs.rs/image/latest/image/>

Useful for file decoding/encoding and basic storage. It is not the execution engine. Enforce decoding limits and convert into explicit `paintop` resource types.

#### `proptest`

Official docs: <https://docs.rs/proptest/latest/proptest/>

Useful for shrinking small generated image/graph failures. Pair with analytic and differential oracles.

---

## 3. Classical operations with disproportionate leverage

These methods are often more controllable, deterministic, and composable than recent generative models.

### 3.1 Exact Euclidean distance transform

Pedro Felzenszwalb and Daniel Huttenlocher’s sampled-function distance transform gives a linear-time route for separable squared Euclidean distance on grids.

**Reference:** <https://cs.brown.edu/people/pfelzens/papers/dt-final.pdf>

Applications:

- signed-distance masks;
- exact grow/shrink;
- feathering in physical pixel units;
- nearest-region maps;
- medial-axis features;
- distance-weighted blending;
- compact-region assertions.

This should land early. It turns ad hoc morphology into a coherent mask algebra.

### 3.2 Poisson image editing and screened Poisson systems

Poisson editing reconstructs an image whose gradients follow guidance while boundary conditions anchor the solution.

Original reference:

- Patrick Pérez, Michel Gangnet, Andrew Blake, “Poisson Image Editing,” SIGGRAPH 2003.

Useful operations:

- seamless clone;
- gradient-domain blend;
- illumination-preserving patch insertion;
- boundary-conditioned heal;
- low-frequency correction;
- seam diffusion.

Implementation notes:

- solve per connected mask component;
- expose residual and iteration count;
- provide a small-domain direct oracle;
- use multigrid/preconditioned conjugate gradient later;
- distinguish ordinary, screened, and mixed-gradient formulations.

### 3.3 PatchMatch

**Primary project/paper:** <https://gfx.cs.princeton.edu/pubs/Barnes_2009_PAR/>

PatchMatch efficiently estimates an approximate nearest-neighbor field between image patches through propagation and random search.

Useful outputs:

- `PatchField` rather than immediate destructive fill;
- patch score/confidence;
- source-validity mask;
- multi-scale field.

Use cases:

- deterministic-ish texture fill under fixed seed/order;
- clone-source recommendation;
- texture similarity masks;
- high-frequency residual transplant;
- seam repair.

PatchMatch remains valuable because the agent can inspect the field and source patches. A diffusion inpaint model cannot provide the same transparent correspondence semantics.

### 3.4 Guided filtering

**Primary paper:** <https://kaiminghe.com/publications/eccv10guidedfilter.pdf>

Guided filtering performs edge-aware smoothing using a local linear model. It can refine masks and transfer structure efficiently.

Applications:

- soft-mask edge refinement;
- confidence smoothing without crossing strong edges;
- base/detail decomposition;
- guided scalar-material edits.

### 3.5 Structure tensor and anisotropic diffusion

The local structure tensor derives dominant orientation and coherence from gradients. It is cheap, deterministic, and broadly useful:

- orient splats along cloth/hair/grain;
- directional blur;
- scratch continuation;
- anisotropic patch metrics;
- edge-flow guidance;
- confidence-aware texture synthesis.

This should become a first-class `Field2`/`Field3` producer rather than an internal detail of one filter.

### 3.6 Pyramids and steerable/frequency representations

Laplacian or steerable pyramids separate low-frequency illumination/color from high-frequency texture. They enable:

- frequency separation touch-up;
- texture-preserving recolor;
- multi-scale PatchMatch;
- high-frequency residual transfer;
- image-quality diagnostics;
- multi-resolution PDE initialization.

A reconstruction identity is testable, making pyramids suitable for agent-built code.

### 3.7 Reaction-diffusion and procedural fields

Gray–Scott-like reaction-diffusion, fractal noise, cellular noise, and domain warp are deterministic texture-field generators. They are not “AI,” but can create excellent material masks:

- corrosion;
- patina;
- mottling;
- stains;
- organic roughness;
- paint breakup.

Their value comes from composability and parameterization, not photorealistic standalone output.

---

## 4. Segmentation, matting, depth, and perception

### 4.1 Segment Anything 2

**Paper:** <https://arxiv.org/abs/2408.00714>  
**Project:** <https://github.com/facebookresearch/sam2>

SAM 2 is a promptable segmentation foundation model for images and video. For `paintop`, the relevant interface is point/box/mask prompting producing one or more masks and confidence scores.

**Adoption recommendation:** model adapter after the deterministic core works. Use it as a mask proposal generator, not as a hard dependency.

Questions to benchmark:

- smallest model with acceptable latency;
- ONNX/export path viability;
- coordinate mapping exactness;
- prompt batching;
- memory on CPU and common GPUs;
- license and redistribution conditions at integration time.

### 4.2 EfficientSAM and EdgeSAM

- EfficientSAM: <https://arxiv.org/abs/2312.00863>
- EdgeSAM: <https://arxiv.org/abs/2312.06660>

These investigate lighter promptable segmentation. They may be more suitable for an embedded tool than the largest foundation model, depending on quality and exportability.

**Research task:** benchmark prompt-to-mask latency, boundary quality, model size, provider support, and downstream matte-refinement quality on `paintop` fixtures rather than adopting based on generic benchmarks.

### 4.3 ZIM zero-shot matting

**Paper:** <https://arxiv.org/abs/2411.00626>  
**Project:** <https://github.com/naver-ai/ZIM>

ZIM targets fine-grained matte generation from prompts, addressing the gap between segmentation and alpha-quality masks.

**Architectural implication:** segmentation and matte refinement should be distinct operations. A hard object mask is not sufficient for hair, fur, translucent boundaries, anti-aliased decals, or soft material transitions.

Recommended output:

```text
Mask matte
Field1 confidence
Report boundary statistics
```

### 4.4 Depth Anything V2

**Paper:** <https://arxiv.org/abs/2406.09414>  
**Project:** <https://github.com/DepthAnything/Depth-Anything-V2>

Depth Anything V2 offers monocular depth models at several scales and emphasizes finer, robust depth estimates.

Useful `paintop` compositions:

- depth-slice masks;
- depth-aware blur;
- background/foreground separation;
- approximate relighting;
- occlusion-aware candidate ranking;
- projected shadow guidance.

Caveat: monocular depth scale and geometry are uncertain. The resource must declare relative vs metric depth and carry confidence or validity metadata.

### 4.5 Normal and edge estimation

Normal estimators can complement depth, but inferred normals are also uncertain. Prefer consistency checks:

- depth-gradient vs normal agreement;
- integrability residual;
- confidence gating;
- multi-scale smoothing.

Classical edges and structure tensors should remain available even if model estimators exist.

---

## 5. Restoration and inpainting

### 5.1 LaMa

**Paper:** <https://arxiv.org/abs/2109.07161>  
**Project:** <https://github.com/advimman/lama>

LaMa uses fast Fourier convolutions to obtain image-wide receptive fields and is strong motivation for a constrained masked inpainting candidate op.

Recommended semantics:

```text
Image + Mask + Seed + ModelManifest
    → CandidateSet<Image> + confidence/report
```

The wrapper should hard-copy source pixels outside the authorized mask after inference unless the operation explicitly studies model leakage as a report.

### 5.2 MAT

**Paper:** <https://arxiv.org/abs/2203.15270>

MAT targets large-hole inpainting with a mask-aware transformer. It is another candidate backend, not a core semantic operation.

### 5.3 NAFNet

**Paper:** <https://arxiv.org/abs/2204.04676>

NAFNet shows that strong image restoration can emerge from deliberately simple nonlinear-block designs. The lesson is not necessarily to embed this exact network, but to avoid assuming attention-heavy architectures are automatically superior for denoising/deblurring.

### 5.4 All-in-one restoration

Recent all-in-one restoration methods attempt to handle multiple degradations with a single model or degradation controls. Example:

- AllRestorer: <https://arxiv.org/abs/2411.10708>
- HOGformer: <https://arxiv.org/abs/2504.09377>
- FoundIR-v2: <https://arxiv.org/abs/2512.09282>

**Adoption stance:** watch. Composite restoration is attractive for agents, but it increases ambiguity: what degradation did the model infer, and what details did it hallucinate? A better `paintop` interface may expose explicit candidate controls and compare against classical baselines.

### 5.5 Recommended hierarchy

For texture repair, try in this order:

1. deterministic clone/transform;
2. structure-guided patch transfer;
3. PatchMatch;
4. Poisson/screened-Poisson boundary blend;
5. classical restoration;
6. neural candidate generation;
7. candidate ranking and explicit selection.

This hierarchy maximizes inspectability and minimizes hallucination.

---

## 6. Intrinsic decomposition and material inference

### 6.1 Why this matters

Changing “material color” in a rendered photograph is difficult because observed RGB mixes:

- reflectance/albedo;
- illumination;
- geometry/shading;
- view-dependent effects;
- camera response.

Intrinsic decomposition attempts to separate some of these factors. For `paintop`, even imperfect decomposition can provide candidate layers that allow texture-preserving recolor.

### 6.2 Recent intrinsic decomposition work

Research signals include:

- Colorful Diffuse Intrinsic Image Decomposition: <https://arxiv.org/abs/2409.13690>
- SAIL albedo estimation: <https://arxiv.org/abs/2505.19751>
- FlowIID: <https://arxiv.org/abs/2601.12329>

These are recent research and must be evaluated on actual touch-up/material fixtures. Do not treat paper benchmark wins as production validity.

Recommended output:

```text
CandidateSet<{
  albedo: Image,
  shading: Field1 or Image,
  specular?: Image,
  confidence: Field1
}>
```

A recomposition assertion should measure how closely the decomposition reconstructs the input.

### 6.3 Material Anything

**Paper:** <https://arxiv.org/abs/2411.15138>

Material Anything proposes diffusion-based PBR material generation for 3D objects, including confidence masks and UV-space refinement.

Relevant ideas:

- confidence should be an explicit resource;
- textured and textureless regions may need different handling;
- UV-space refinement is valuable after view-space inference;
- rendering loss can enforce material consistency.

Not an initial dependency. It informs later candidate material inference.

### 6.4 StableMaterials and MatFuse

- StableMaterials: <https://arxiv.org/abs/2406.09293>
- MatFuse: <https://arxiv.org/abs/2308.11408>

These represent generative material modeling directions. They may eventually back operations that generate tileable or PBR candidates, but they are outside the initial localized-touch-up core.

### 6.5 Recent material/3D signals

Research to monitor:

- PBR-SR: <https://arxiv.org/abs/2506.02846>
- PBR3DGen: <https://arxiv.org/abs/2503.11368>
- MeshGen: <https://arxiv.org/abs/2505.04656>
- Meta AssetGen: <https://arxiv.org/abs/2407.02445>
- DragTex: <https://arxiv.org/abs/2403.02217>
- TexGen: <https://arxiv.org/abs/2408.01291>
- PacTure: <https://arxiv.org/abs/2505.22394>
- MatE: <https://arxiv.org/abs/2512.18312>
- HumanMaterial: <https://arxiv.org/abs/2507.18385>

The likely useful pattern is not integrating every paper. It is defining stable operations such as:

```text
infer.material_channels
infer.albedo_shading
generate.texture_candidate
refine.uv_material
optimize.material_against_views
```

Backends can change while the operation contracts remain stable.

---

## 7. 3D rendering, projection, and surface mathematics

### 7.1 glTF 2.0 material semantics

**Official specification:** <https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html>

Critical details for `paintop`:

- base color and emissive textures are color data requiring sRGB decoding;
- metallic-roughness is a data texture: roughness in green, metallic in blue;
- occlusion uses the red channel;
- normal maps encode tangent-space vectors and are linear data;
- texture coordinate origin and sampling conventions must be followed;
- `KHR_texture_transform` can change UV mapping;
- `KHR_texture_basisu` may introduce KTX2/BasisU handling.

This is why material channels need semantic types rather than generic RGBA image operations.

### 7.2 Differentiable rendering

Relevant work:

- Soft Rasterizer: <https://arxiv.org/abs/1901.05567>
- nvdiffrast: <https://nvlabs.github.io/nvdiffrast/>
- Differentiable Surface Splatting: <https://arxiv.org/abs/1906.04173>

Differentiable rendering can support local optimization of:

- texture deltas;
- splat positions/covariances;
- material factors;
- roughness/base-color fields;
- view-consistent edits.

However, embedding a full differentiable renderer early is excessive. `paintop` can begin with finite differences, analytic gradients for its own kernels, or an external research adapter.

### 7.3 Geodesic selection with the heat method

**Paper:** <https://arxiv.org/abs/1204.6216>

The heat method computes geodesic distance on meshes through two elliptic solves and can reuse prefactorizations. It is ideal for a surface brush or selection that should follow the mesh rather than UV distortion.

Potential operation:

```text
surface.geodesic_distance(seed vertices/faces) → MeshField
surface.select_distance(field, radius)         → SurfaceSelection
```

A later scalable variant may draw on parallel heat-method work:

- <https://arxiv.org/abs/1812.06060>

### 7.4 Surface splatting

Differentiable and classical surface splatting research motivates a generalized surface footprint rather than point sampling. For `paintop`, the practical version is:

- G-buffer or ray hit gives triangle and barycentrics;
- local screen→UV Jacobian maps brush covariance;
- an elliptical footprint accumulates weighted coverage in UV space;
- visibility and seam policies control contribution;
- normalization avoids density changes under foreshortening.

This is a high-value area for original implementation work.

### 7.5 Seam graph

A UV atlas disconnects texels that are neighbors on the mesh. Construct a graph whose edges connect:

- ordinary adjacent texels inside an island;
- paired texels across chart seams based on surface-edge correspondence;
- optionally duplicated overlap/mirror instances under policy.

Graph diffusion or screened Poisson operations over this topology can produce seam-aware blur, heal, and mask propagation. This is not standard image convolution and may be one of the project’s distinctive algorithms.

---

## 8. Image quality metrics and verification research

### 8.1 LPIPS

**Paper:** <https://arxiv.org/abs/1801.03924>

LPIPS uses deep features to estimate perceptual similarity. It is useful for candidate ranking and regression characterization, but not authorization or semantic integrity.

### 8.2 A-DISTS / DISTS

**Paper:** <https://arxiv.org/abs/2110.08521>

DISTS-like methods attempt to capture structure and texture similarity. This may be useful in texture-repair evaluation where pure pixel error penalizes legitimate texture variation.

### 8.3 Shift tolerance

**Paper:** <https://arxiv.org/abs/2207.13686>

Perceptual metrics can be overly sensitive to tiny shifts. Shift-tolerant evaluation is relevant to resampling/render comparisons, but should not excuse coordinate bugs in exact operations.

### 8.4 Robust perceptual metrics

- R-LPIPS: <https://arxiv.org/abs/2307.15157>
- LASI: <https://arxiv.org/abs/2310.05986>

These are candidates for evaluation experiments, not immediate runtime dependencies.

### 8.5 Metamorphic and differential testing

- Metamorphic testing for object detection: <https://arxiv.org/abs/1912.12162>
- Cross-vendor GPU numerical differential testing: <https://arxiv.org/abs/2410.09172>
- Tolerant image-property testing: <https://arxiv.org/abs/1503.01363>

The direct lesson is that transformations and alternate executions can expose failures even when exact expected outputs are unavailable.

---

## 9. Assumption audit

### Assumption: “The coding agent wants a powerful language.”

**Challenge:** agents benefit more from constrained affordances and high-quality errors than arbitrary expressiveness. JSON plus typed graph templates is likely superior at first.

**Experiment:** compare error/revision rates for equivalent tasks authored in strict JSON and a Lua graph builder after both exist.

### Assumption: “Every edit should be a single op.”

**Challenge:** too-large ops become opaque and unverifiable. Prefer composable producers and consumers:

```text
compute patch field → synthesize candidate → blend boundary → assert
```

rather than `content_aware_fill_everything`.

### Assumption: “Neural ops are the alien part.”

**Challenge:** the more novel system may be the combination of typed fields, SDF algebra, topology-aware diffusion, compiler optimization, and autonomous evidence. Models can be interchangeable backends.

### Assumption: “A normal image library is enough for the CPU backend.”

**Challenge:** most image crates do not provide strict linear-light/premultiplied semantics, ROI scheduling, operation manifests, or material map types. Use libraries for codecs and selected kernels, not as the semantic core.

### Assumption: “All graph outputs can be images.”

**Challenge:** masks, fields, IDs, patches, candidates, reports, and surface maps need distinct types. Making them generic images would sacrifice validation and optimization.

### Assumption: “GPU implementation should mirror CPU loops.”

**Challenge:** the right GPU unit is often a fused subgraph with persistent residency, not one dispatch per op. The scheduler should optimize across operations.

### Assumption: “The best backend can be selected by image size.”

**Challenge:** ROI density, halo, format, current residency, model load state, and readback requirements matter. Use a multivariate cost model.

### Assumption: “Projection is just rasterization in reverse.”

**Challenge:** reverse mapping is many-to-many around seams, overlaps, occlusion, and filtering footprints. Treat it as surface sampling with explicit policies and confidence.

### Assumption: “Recent PBR inference makes material editing solved.”

**Challenge:** single-view inverse rendering remains ill-posed. Recent results are useful priors, not truth. Multi-view rendering and uncertainty are mandatory.

### Assumption: “Perceptual metrics can authorize autonomous edits.”

**Challenge:** they cannot protect semantic regions or data channels. Authorization must be geometric/mask/structural.

---

## 10. Adoption matrix

### Adopt immediately

- strict typed JSON graph;
- immutable resources;
- explicit color/alpha/coordinates;
- scalar CPU reference;
- analytic/property/metamorphic/differential tests;
- evidence bundles;
- demand-driven tiled ROI execution design;
- exact distance transform and SDF mask algebra;
- `msh` process bridge for initial rendering;
- glTF specification as material truth.

### Adopt after the CPU core

- optimized CPU backend;
- `wgpu` fused pointwise/splat/filter paths;
- Poisson and PatchMatch;
- structure tensor/orientation fields;
- model manifest and ONNX adapter;
- one segmentation model and one depth/matting model;
- scene-preserving `msh` asset/render extraction;
- screen-to-surface/UV maps.

### Research prototypes

- low-rank automatic convolution factoring;
- backend autotuning;
- e-graph optimization;
- seam-graph diffusion;
- Jacobian-driven anisotropic surface splats;
- compact differentiable edit optimizer;
- intrinsic/material inference candidate ops;
- multi-view inverse material fitting;
- sparse delta provenance.

### Avoid until evidence demands it

- GUI;
- arbitrary shader/plugin execution in plans;
- full Photoshop blend/filter parity;
- embedded Python as a core requirement;
- automatic UV unwrapping in the first material release;
- full differentiable renderer;
- PSD authoring;
- universal color-profile support;
- unbounded Lua execution;
- prompt-to-image as a primitive mutation.

---

## 11. Recommended research spikes

Each spike should end in a benchmark/report and be disposable.

### Spike A: ROI/tile prototype

Implement pointwise + Gaussian blur on a 4K image with compact and scattered masks. Compare full frame, tile CPU, and GPU. Measure crossover and memory.

### Spike B: SDF morphology

Implement brute-force and linear-time EDT on small/large fixtures. Validate exactness and demonstrate grow/shrink/feather composition.

### Spike C: convolution chooser

Compare direct, separable, low-rank SVD, and FFT for representative kernels and ROI densities. Produce decision surfaces, not one headline number.

### Spike D: model adapter

Export or adapt one lightweight segmentation model. Measure model load, preprocessing, inference, coordinate mapping, and boundary quality. Verify hash and offline execution.

### Spike E: `msh` G-buffer

Add one exact diagnostic pass—triangle/material ID plus UV—and validate it on a plane/cube fixture. Do not refactor every renderer component first.

### Spike F: anisotropic UV splat

Paint a screen-space circle onto tilted planes and curved surfaces using nearest scatter vs Jacobian EWA. Measure holes, coverage conservation, and view consistency.

### Spike G: seam graph

Build seam adjacency for a cube atlas. Compare ordinary UV blur with seam-aware graph diffusion. Define exact seam disagreement metrics.

### Spike H: local edit optimizer

Optimize 8–64 Gaussian splats to match a synthetic target under locality/complexity penalties. Compare finite-difference L-BFGS, analytic gradients, and derivative-free search.

---

## 12. Source index

### Systems

- libvips: <https://www.libvips.org/API/current/how-it-works.html>
- Halide: <https://halide-lang.org/>
- OpenCV G-API: <https://docs.opencv.org/4.x/d0/d1e/gapi.html>
- GEGL operations: <https://www.gegl.org/operations/>
- `egg`: <https://arxiv.org/abs/2004.03082>
- `wgpu`: <https://docs.rs/wgpu/latest/wgpu/>
- `ort`: <https://docs.rs/ort/latest/ort/>
- `msh`: <https://github.com/bobisme/msh>

### Classical image and geometry processing

- Distance transforms: <https://cs.brown.edu/people/pfelzens/papers/dt-final.pdf>
- PatchMatch: <https://gfx.cs.princeton.edu/pubs/Barnes_2009_PAR/>
- Guided filter: <https://kaiminghe.com/publications/eccv10guidedfilter.pdf>
- Heat method: <https://arxiv.org/abs/1204.6216>
- Parallel heat method: <https://arxiv.org/abs/1812.06060>
- Differentiable surface splatting: <https://arxiv.org/abs/1906.04173>

### Perception and restoration

- SAM 2: <https://arxiv.org/abs/2408.00714>
- EfficientSAM: <https://arxiv.org/abs/2312.00863>
- EdgeSAM: <https://arxiv.org/abs/2312.06660>
- ZIM: <https://arxiv.org/abs/2411.00626>
- Depth Anything V2: <https://arxiv.org/abs/2406.09414>
- LaMa: <https://arxiv.org/abs/2109.07161>
- MAT: <https://arxiv.org/abs/2203.15270>
- NAFNet: <https://arxiv.org/abs/2204.04676>

### Materials and 3D

- glTF 2.0: <https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html>
- Material Anything: <https://arxiv.org/abs/2411.15138>
- StableMaterials: <https://arxiv.org/abs/2406.09293>
- MatFuse: <https://arxiv.org/abs/2308.11408>
- PBR-SR: <https://arxiv.org/abs/2506.02846>
- DragTex: <https://arxiv.org/abs/2403.02217>
- TexGen: <https://arxiv.org/abs/2408.01291>
- Soft Rasterizer: <https://arxiv.org/abs/1901.05567>
- nvdiffrast: <https://nvlabs.github.io/nvdiffrast/>

### Testing and metrics

- `proptest`: <https://docs.rs/proptest/latest/proptest/>
- LPIPS: <https://arxiv.org/abs/1801.03924>
- A-DISTS: <https://arxiv.org/abs/2110.08521>
- Shift-tolerant perceptual metric: <https://arxiv.org/abs/2207.13686>
- Metamorphic CV testing: <https://arxiv.org/abs/1912.12162>
- GPU numerical differential testing: <https://arxiv.org/abs/2410.09172>
