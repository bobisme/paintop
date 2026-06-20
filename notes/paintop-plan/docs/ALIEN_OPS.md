# Alien operations: advanced and experimental operators

**Status:** research design notebook  
**Purpose:** define high-leverage composable operations that go beyond ordinary paint-program filters while remaining typed, inspectable, and testable  
**Honesty note:** several designs below combine established mathematics in project-specific ways. They are proposed algorithms, not claims of first publication or proven superiority. Each must beat a baseline under explicit tests before entering the stable operation set.

---

## 1. What makes an operation “alien” without making it opaque

An alien operation should still have a compact type signature:

```text
Image → Image
Image → Mask
Image → Field
Image + Mask → Image
Image + Field → Image
Image + Mask + Field → Image
SurfaceMap + Mask → UV Coverage
Resources → CandidateSet + Report
```

It should also expose:

- deterministic or seeded semantics;
- intermediate fields;
- confidence/uncertainty;
- residuals or convergence reports;
- changed-region bounds;
- a reference or reduced oracle;
- postconditions;
- an explicit failure mode.

The target is not “magic.” The target is **weirdly powerful machinery an agent can reason about**.

---

## 2. Signed-distance mask calculus

### 2.1 Motivation

Traditional paint APIs expose grow, shrink, blur, feather, union, and intersection as unrelated operations. A signed distance field unifies them geometrically.

Let `φ(p)` be signed Euclidean distance to the mask boundary, negative inside and positive outside.

Then:

- grow by `d`: `φ'(p) = φ(p) - d`;
- shrink by `d`: `φ'(p) = φ(p) + d`;
- hard mask: `m(p) = [φ(p) ≤ 0]`;
- feather:

  ```math
  m(p) = 1 - smoothstep(-w, w, φ(p))
  ```

- union: `min(φ_A, φ_B)`;
- intersection: `max(φ_A, φ_B)`;
- subtraction: `max(φ_A, -φ_B)`;
- complement: `-φ`.

Smooth union can use a controlled smooth-min, for example:

```math
smin_k(a,b) = -\frac{1}{k}\log(e^{-ka}+e^{-kb})
```

or a compact polynomial form with an explicit smoothing radius.

### 2.2 Operations

```text
mask.to_sdf
sdf.to_mask
sdf.offset
sdf.union
sdf.intersect
sdf.subtract
sdf.smooth_union
sdf.smooth_intersect
sdf.curvature
sdf.normal
sdf.medialness
```

### 2.3 Why it matters

- morphology becomes resolution-aware;
- feathering is tied to physical pixels;
- agent plans can reason in distances;
- smooth Boolean shape construction becomes possible;
- change authorization can be expanded by a precise margin;
- brush falloff can be generated from shape distance.

### 2.4 Implementation

1. Threshold or otherwise define the source boundary.
2. Compute exact squared Euclidean distance to inside and outside sets using the separable linear-time distance transform.
3. Take square roots and combine signs.
4. For repeated edits to the same mask, cache the SDF.

For soft-mask input, require a contour threshold or an explicit generalized-distance policy. Do not pretend a soft coverage map has one unique SDF.

### 2.5 Verification

- brute-force distance oracle on small images;
- circle/rectangle analytic distances away from rasterized boundary ambiguity;
- offset composition;
- Boolean algebra on hard masks;
- gradient norm near one away from the medial axis;
- rotation covariance on symmetric fixtures;
- tile/full equivalence if a tiled exact implementation is attempted.

### 2.6 Stable status

This is not speculative. It should be a core advanced mask representation.

---

## 3. Self-tuning anisotropic splats (STAS)

### 3.1 Motivation

A fixed circular brush ignores local structure. A useful agent brush should be able to follow hair, cloth, wood grain, shadows, or projected surface distortion.

Represent each splat as a Gaussian with covariance `Σ`:

```math
w(p) = \exp\left[-\frac{1}{2}(p-\mu)^T Σ^{-1}(p-\mu)\right]
```

Instead of requiring the agent to choose every covariance, derive it from an orientation/coherence field.

### 3.2 2D structure-adaptive covariance

From image gradients, compute a smoothed structure tensor:

```math
J = G_ρ *
\begin{bmatrix}
I_x^2 & I_x I_y \\
I_x I_y & I_y^2
\end{bmatrix}
```

Let eigenvectors be `e1`, `e2`, eigenvalues `λ1 ≥ λ2`. Define coherence:

```math
c = \frac{λ_1-λ_2}{λ_1+λ_2+ε}
```

Construct covariance:

```math
Σ = R
\begin{bmatrix}
σ_\parallel^2 & 0 \\
0 & σ_\perp^2
\end{bmatrix}
R^T
```

where `R=[e2,e1]` if the long axis should follow isophotes rather than gradients, and:

```math
σ_\parallel = σ_0(1 + a c)
σ_\perp = σ_0/(1 + b c)
```

### 3.3 Surface-adaptive covariance

For projection painting, map a screen covariance through the local Jacobian:

```math
J_{uv} = \frac{\partial(u,v)}{\partial(x,y)}
```

```math
Σ_{uv} = J_{uv} Σ_{screen} J_{uv}^T + εI
```

This produces an elliptical UV footprint that accounts for foreshortening and texture distortion.

### 3.4 Operation shape

```json
{
  "op": "paint.adaptive_splats@1",
  "in": {
    "base": "...",
    "orientation": "...",
    "coherence": "...",
    "clip": "..."
  },
  "params": {
    "centers": "...",
    "base_sigma_px": 8.0,
    "anisotropy_max": 6.0,
    "align": "isophote",
    "normalization": "peak",
    "blend": "normal"
  }
}
```

For surface mode, consume `SurfaceMap` and output one or more texture-space coverage/paint layers.

### 3.5 Accumulation modes

Define explicitly:

- ordered alpha-over;
- additive energy;
- normalized weighted average;
- max coverage;
- optical-density accumulation;
- multiply/screen in linear light.

For normalized accumulation:

```math
C(p) = \frac{\sum_i w_i(p) a_i C_i}{\sum_i w_i(p) a_i + ε}
```

Coverage is tracked separately to avoid density-dependent color shifts.

### 3.6 GPU strategy

- bin splats into tiles using conservative covariance bounds;
- prefix-sum tile lists;
- dispatch one workgroup per occupied tile;
- use a stable local accumulation strategy;
- for huge splats, switch to screen-aligned quad rasterization;
- avoid global atomics on color where possible;
- record truncation radius and omitted tail-energy bound.

### 3.7 Verification

- analytic center/axis samples;
- covariance reconstruction from weighted moments;
- 90-degree rotation covariance;
- screen plane with analytic screen→UV homography;
- coverage conservation under plane tilt;
- CPU/GPU comparison;
- seam and tile boundary tests;
- upper bound on truncated Gaussian mass.

### 3.8 Research question

Does structure adaptation actually help agents use fewer splats for common repairs? Measure fit quality vs splat count on synthetic and real targets.

---

## 4. Seam-graph diffusion

### 4.1 Motivation

A UV atlas cuts a continuous surface into disconnected 2D islands. Ordinary blur, morphology, Poisson fill, and patch propagation stop at island boundaries, creating seams.

Build a graph over texel samples that reconnects surface-adjacent texels across UV seams.

### 4.2 Graph construction

Nodes may represent valid texels or coarser texel cells. Edges include:

1. regular 4/8-neighbor UV adjacency inside a chart;
2. seam adjacency between samples corresponding to the two sides of the same mesh edge;
3. optional overlap-instance edges under an explicit mirror/overlap policy;
4. multi-resolution edges for coarse solves.

Edge weight can depend on:

- physical surface distance;
- normal discontinuity;
- material boundary;
- UV texel density;
- seam orientation;
- confidence/visibility.

Example:

```math
w_{ij} = \exp(-d_s^2/2σ_s^2)
         \exp(-(1-n_i\cdot n_j)/σ_n)
         q_{ij}
```

### 4.3 Graph Laplacian

```math
L = D - W
```

Screened diffusion/repair can solve:

```math
(λI + L)x = λb + g
```

where `b` anchors known texture values and `g` encodes guidance.

### 4.4 Operations

```text
surface.build_seam_graph
surface.diffuse_field
surface.seam_aware_blur
surface.seam_aware_feather
surface.seam_aware_poisson
surface.seam_disagreement
```

### 4.5 Implementation strategy

- construct seam correspondences from mesh topology and UV edge parameterization;
- begin with sparse CPU matrices and conjugate gradient;
- precondition with Jacobi or incomplete factorization;
- cache topology-dependent matrices per asset/texture resolution;
- later add multigrid or GPU sparse solvers;
- preserve material boundaries unless explicitly crossed.

### 4.6 Verification

- cube atlas with known constant field should remain constant across seams;
- impulse diffuses symmetrically over surface distance, not atlas distance;
- ordinary adjacency plus seam graph agrees with a single-chart reference plane;
- graph is symmetric/nonnegative when configured for symmetric diffusion;
- Laplacian nullspace behavior understood;
- solver residual reported;
- disconnected materials remain disconnected.

### 4.7 Potential novelty

The ingredients are established graph/PDE methods. The project-specific contribution is exposing topology-aware diffusion as a composable agent operation over ordinary PBR texture maps.

---

## 5. Boundary-conditioned harmonic heal

### 5.1 Motivation

Clone and neural inpaint operations often fail for different reasons:

- clone preserves texture but mismatches low-frequency color/lighting;
- Poisson blend matches gradients but can smear texture;
- neural inpaint may hallucinate structure.

Separate the repair into low-frequency structure and high-frequency residual.

### 5.2 Decomposition

Construct a lowpass `L` and residual `H`:

```math
I = L + H
```

Inside a hole `Ω`:

1. solve a screened harmonic/Poisson problem for `L_Ω` constrained by the boundary;
2. synthesize or transplant `H_Ω` from valid source patches;
3. modulate residual amplitude and orientation to match the target boundary;
4. recombine and perform a narrow boundary blend.

### 5.3 Low-frequency solve

A simple screened objective:

```math
E(L) = \int_Ω ||\nabla L - v||^2 dp
     + λ\int_Ω ||L - L_0||^2 dp
```

- `v` may be zero for harmonic fill, mixed source/target gradients, or an extrapolated boundary field;
- `L0` may be a coarse PatchMatch or model candidate.

### 5.4 High-frequency residual transfer

Choose source patches by a metric combining:

- normalized high-frequency patch distance;
- structure-tensor orientation difference;
- low-frequency color compatibility;
- distance to forbidden source regions;
- optional semantic labels.

Rotate/warp residual patches only under an explicit transform model.

### 5.5 Operation shape

```text
heal.decompose
heal.solve_low_frequency
heal.compute_residual_patch_field
heal.synthesize_residual
heal.recombine
```

A convenience macro may build the subgraph, but the canonical IR retains the pieces.

### 5.6 Verification

Synthetic fixture:

- smooth gradient background plus known periodic texture;
- remove a region;
- check low-frequency reconstruction error;
- check residual spectrum/orientation;
- check boundary gradient discontinuity;
- assert outside unchanged.

Compare against:

- ordinary clone;
- Poisson clone;
- plain PatchMatch;
- neural candidate.

### 5.7 Research metric

Optimize for a vector of metrics, not one score:

- boundary gradient error;
- low-frequency RMSE;
- local power-spectrum distance;
- patch repetition artifact score;
- outside-mask delta.

---

## 6. Spectral residual transplant

### 6.1 Motivation

Agents frequently want to change tone/color without destroying fine detail, or transplant texture without transferring illumination.

Use a multiscale decomposition:

```math
I = L_N + \sum_{k=0}^{N-1} R_k
```

where `L_N` is coarse lowpass and `R_k` are band-limited residuals.

### 6.2 Operations

```text
frequency.split_pyramid
frequency.align_residual
frequency.transfer_residual
frequency.match_band_energy
frequency.recombine
```

### 6.3 Local orientation alignment

A residual patch from source may be rotated into the target’s local orientation. Let source and target structure-tensor frames be `R_s`, `R_t`. Approximate local transform:

```math
A = R_t S R_s^T
```

where `S` optionally scales along axes based on local feature frequencies.

Use conservative transforms; high-frequency resampling can create artifacts.

### 6.4 Energy matching

For each band and local window:

```math
R'_k = \frac{σ_{target,k}}{σ_{source,k}+ε}(R_k-μ_k)
```

then add a controlled mean if appropriate. Clamp the gain to prevent noise amplification.

### 6.5 Material use

- preserve weave while changing fabric color;
- transfer scratch detail into roughness;
- remove lighting from a base-color candidate;
- create correlated but not identical variation across PBR channels.

### 6.6 Verification

- split/recombine identity;
- sine-band fixtures;
- known rotated texture;
- local spectrum comparison;
- no low-frequency leakage beyond tolerance;
- no change outside mask;
- gain cap behavior.

---

## 7. Contract-driven micro-optimizer

### 7.1 Motivation

An agent can describe intent but may be poor at selecting exact splat positions, opacities, curve points, or material deltas. Optimize a **small parameterized edit program**, not millions of pixels.

### 7.2 Program

Example parameters:

```text
32 splat centers
32 covariance pairs
32 opacities
4 color coefficients
1 mask expansion
```

Let renderer/executor produce `Y(θ)` from parameters `θ`.

Objective:

```math
E(θ) =
w_t D_{target}(Y(θ), T)
+ w_o D_{outside}(Y(θ), X, M)
+ w_b D_{boundary}(Y(θ), X, M)
+ w_c C(θ)
+ w_r R(θ)
```

Where:

- `D_target`: task-specific target error;
- `D_outside`: massive penalty for unauthorized change;
- `D_boundary`: gradient/color continuity;
- `C`: complexity, e.g. number/opacity of splats;
- `R`: regularization and parameter bounds.

### 7.3 Optimizers

- analytic gradients for splat/color kernels;
- automatic differentiation if a compact differentiable backend exists;
- finite-difference L-BFGS for small parameter counts;
- coordinate descent;
- CMA-ES or other derivative-free strategy for discontinuous objectives;
- mixed strategy: coarse derivative-free initialization, gradient refinement.

### 7.4 Output semantics

The op returns:

```text
CandidateSet<PlanFragment>
OptimizationReport
SensitivityMap
```

It does not silently apply the winner.

### 7.5 Operation example

```json
{
  "op": "optimize.edit_program@1",
  "in": {
    "base": "...",
    "target": "...",
    "allowed": "..."
  },
  "params": {
    "program": "node:initial_splats/fragment",
    "variables": ["splats[*].center", "splats[*].opacity"],
    "objective": [
      {"metric": "masked-l2", "weight": 1.0},
      {"metric": "outside-max-delta", "weight": 1000.0},
      {"metric": "boundary-gradient", "weight": 2.0},
      {"metric": "l1-opacity", "weight": 0.01}
    ],
    "optimizer": {"kind": "lbfgs", "steps": 80}
  }
}
```

### 7.6 Verification

- synthetic target generated by known parameter vector;
- objective decreases;
- recovered output matches target within bound;
- constraints never violated or are rejected;
- fixed initialization deterministic;
- finite-difference gradient check against analytic gradient;
- complexity penalty reduces redundant splats;
- failure/iteration limit returns report and candidate, not success.

### 7.7 Why this may be more valuable than a large model

It converts a vague spatial-control problem into a small constrained numerical problem while preserving exact locality and provenance.

---

## 8. Multi-view constraint projection

### 8.1 Motivation

A material texture edit should look coherent from multiple views. A single screen-space projection can undersample hidden regions and create view-dependent artifacts.

Given views `v=1..V`, a UV texture delta `ΔT`, and renderer `R_v`, solve:

```math
\min_{ΔT}
\sum_v w_v D_v(R_v(T+ΔT), Y_v)
+ λ_s ||\nabla_{surface} ΔT||^2
+ λ_m ||(1-M)ΔT||^2
+ λ_c C_{seam}(ΔT)
```

Where:

- `Y_v` may be a target render or desired residual;
- `M` is authorized surface/UV mask;
- surface smoothness and seam consistency regularize unseen areas.

### 8.2 Practical non-differentiable version

Before a differentiable renderer exists:

1. project each view’s desired screen residual into UV with Jacobian footprints;
2. accumulate weighted normal equations per texel;
3. normalize by accumulated confidence;
4. solve a seam-aware smooth correction;
5. render and iterate.

### 8.3 Confidence weighting

View sample weight can include:

```math
w = m_{screen}
    c_{model}
    max(n\cdot v, 0)^γ
    q_{resolution}
    q_{depth-edge}
```

Avoid overweighting grazing-angle views.

### 8.4 Outputs

- UV delta candidate;
- coverage/confidence map;
- per-view residuals;
- disagreement map;
- unseen-region mask;
- convergence report.

### 8.5 Verification

- planar texture with known homographies;
- two-view synthetic target generated from a known UV edit;
- recovered UV delta compared to truth;
- view-order invariance under deterministic reductions;
- occlusion rejection;
- unseen area remains unchanged unless regularization policy allows fill;
- seam disagreement metric.

---

## 9. Uncertainty-carrying fields

### 9.1 Motivation

Model-derived masks, depth, normals, and material channels are not equally reliable everywhere. A bare field encourages overconfidence.

Represent uncertain output as:

```text
value field
confidence field in [0,1]
optional covariance/entropy
validity mask
provenance report
```

### 9.2 Operations

```text
uncertainty.gate
uncertainty.combine_independent
uncertainty.combine_conservative
uncertainty.propagate_pointwise
uncertainty.propagate_resample
uncertainty.threshold_to_validity
uncertainty.visualize
```

### 9.3 Conservative propagation

For a weighted average:

```math
\hat{x} = \frac{\sum_i w_i c_i x_i}{\sum_i w_i c_i + ε}
```

Confidence should not simply average upward. A conservative heuristic might combine support and agreement:

```math
c_{out} = support \cdot agreement \cdot max_i(c_i)
```

The exact rule is operation-specific and must be documented. Avoid pretending heuristic confidence is calibrated probability.

### 9.4 Confidence-aware editing

Examples:

- only use inferred depth where confidence exceeds threshold;
- feather selection more heavily near uncertain boundaries;
- request another view where UV projection confidence is low;
- rank model candidates by confidence-weighted residual;
- keep uncertain material channels as candidates.

### 9.5 Verification

- confidence stays in range;
- zero-confidence inputs cannot dominate;
- identical confident inputs preserve confidence under declared rule;
- disagreement lowers confidence;
- confidence transforms with coordinates exactly;
- synthetic noise/calibration experiments characterize, not overclaim, meaning.

---

## 10. Delta provenance and influence maps

### 10.1 Motivation

When an assertion detects 17 changed pixels outside a mask, the agent needs to know which node caused them.

### 10.2 Coarse provenance

Track per output tile:

- producer node;
- contributing input tile IDs;
- mask occupancy;
- operations whose output differs from identity;
- cache lineage.

This may be enough for most debugging.

### 10.3 Sparse pixel provenance

For changed pixels, retain a compact attribution set:

```text
pixel/tile → [(node_id, estimated_contribution)]
```

Exact provenance is expensive for nonlinear graphs. Use modes:

- structural: transitive producer set;
- delta: compare node input/output and attribute first change;
- sensitivity: finite-difference or analytic derivative contribution;
- sampled: detailed attribution only near failures.

### 10.4 Edit influence probe

Given parameter `θ_j`, estimate:

```math
S_j(p) = \frac{\partial Y(p)}{\partial θ_j}
```

or finite difference:

```math
S_j(p) \approx \frac{Y(θ+εδ_j)-Y(θ)}{ε}
```

Outputs:

- spatial influence mask;
- sign/magnitude heatmap;
- affected bounds;
- coupling matrix between parameters and metrics.

This tells an agent which control to adjust.

### 10.5 Verification

- identity operations have empty delta provenance;
- provenance contains all true producer nodes on small exact graphs;
- analytic and finite-difference sensitivities agree;
- sampled mode reproduces a known assertion failure;
- provenance tracking disabled does not change outputs.

---

## 11. Automatic low-rank convolution factoring

### 11.1 Motivation

Agents or future ops may supply arbitrary kernels. Many are separable or approximately low rank, making dense `O(k²)` convolution wasteful.

### 11.2 Factorization

Compute SVD:

```math
K = UΣV^T
```

Approximate with rank `r`:

```math
K_r = \sum_{i=1}^r σ_i u_i v_i^T
```

Choose smallest `r` satisfying:

```math
\frac{||K-K_r||_F}{||K||_F} \le ε_K
```

But kernel error is not always output error. For bounded input `||x||`, derive or conservatively estimate output bound, and optionally run a sampled differential probe.

### 11.3 Planner choices

```text
rank 1 → two 1D passes
small r → 2r 1D passes, perhaps fused accumulation
large r → dense direct or FFT
sparse K → sparse direct
box-like → integral image
```

### 11.4 Operation semantics

The semantic op is still exact convolution if approximation is not enabled. Low-rank approximation requires an explicit error budget:

```json
{
  "approximation": {
    "allowed": true,
    "kernel_relative_frobenius": 0.0001,
    "output_max_abs_probe": 0.0005
  }
}
```

### 11.5 Verification

- known rank-1/2 kernels;
- random matrices with controlled singular spectrum;
- adversarial high-frequency images;
- measured output error vs bound;
- direct/low-rank performance crossover;
- stable sign/order conventions for factors.

---

## 12. Adaptive backend autotuner

### 12.1 Motivation

Backend crossover depends on more than image dimensions:

```text
extent
ROI density and topology
halo/radius
format/channels
CPU ISA/cores
GPU/device/driver
current residency
pipeline warmth
readback requirements
```

### 12.2 Feature vector

```text
x = [
  pixels_requested,
  occupied_tiles,
  perimeter_estimate,
  halo,
  channels,
  bytes_per_channel,
  op_parameters,
  input_residency,
  output_sink,
  device_profile
]
```

### 12.3 Model

Start with transparent piecewise/linear regression or decision tables. Avoid a neural scheduler until evidence says it helps.

Predicted cost:

```math
T = T_{setup} + T_{transfer} + T_{compute} + T_{sync}
```

### 12.4 Online bounded exploration

- benchmark a small candidate subset during explicit `paintop tune`;
- never explore during a strict deterministic production run unless policy allows;
- cache profile keyed by hardware/runtime/driver;
- decay or invalidate stale profiles;
- always retain a safe heuristic fallback.

### 12.5 Verification

- all selected implementations already conform;
- tuning cannot alter output semantics;
- corrupted profile rejected;
- selected choice no worse than fallback beyond tolerance on benchmark suite;
- exploration budget honored;
- profile details appear in evidence.

---

## 13. Equality-saturation graph optimizer

### 13.1 Motivation

As the operation algebra grows, local rewrite order may trap the optimizer. Equality saturation can represent alternatives and choose by cost.

### 13.2 Candidate rewrite rules

Only semantic-preserving rules with side conditions:

```text
convert(A→B) ∘ convert(B→A) → identity
premultiply ∘ unpremultiply → identity where alpha > ε
exposure(a) ∘ exposure(b) → exposure(a+b) before clamp
matrix(A) ∘ matrix(B) → matrix(A·B)
affine(A) ∘ affine(B) → affine(A·B)
mask ∩ full → mask
mask ∪ empty → mask
convolve(outer(u,v)) → separable(u,v)
```

Approximate rewrites live in a separate class with explicit error budgets.

### 13.3 Barriers

Do not rewrite across:

- requested debug materialization;
- stochastic candidate generation;
- assertions that observe an intermediate;
- clamping/nonlinear encodings unless algebra proves safety;
- model calls;
- side-effect sinks.

### 13.4 Cost extraction

Cost includes:

```text
predicted runtime
peak memory
transfers/readbacks
pipeline compilation
approximation error
requested determinism
cache reuse probability
```

### 13.5 Verification

- property-generated equivalent expression pairs;
- compare optimized and unoptimized normalized semantics;
- differential output tests;
- rewrite-specific side-condition tests;
- optimizer can be disabled;
- extraction is deterministic for a fixed device profile;
- e-graph resource limits prevent explosion.

---

## 14. Surface geodesic brush

### 14.1 Motivation

A circular region in UV or screen space does not represent a constant-radius region on a curved mesh. Use geodesic distance.

### 14.2 Heat method

Given source set `S`, solve a short-time heat diffusion, normalize its gradient, then solve a Poisson problem to recover distance. See “Geodesics in Heat.”

Operation chain:

```text
surface.seed_from_screen
surface.heat_geodesic_distance
surface.distance_to_coverage
surface.rasterize_material_uv
```

### 14.3 Brush profile

Given geodesic distance `d_g` and radius `r`:

```math
m = 1 - smoothstep(r-w, r+w, d_g)
```

This produces a surface-space feather independent of UV distortion.

### 14.4 Caching

The mesh Laplacian/factorization depends on topology and metric, not source seed. Cache it per mesh state. Animated/skinned geometry requires policy:

- rest-pose metric;
- current-pose recomputation;
- approximate deformation update.

### 14.5 Verification

- plane agrees with Euclidean distance;
- cylinder/sphere analytic or high-resolution reference;
- invariant to UV atlas changes;
- seed/view mapping agrees with exact hit;
- cached vs uncached solve;
- distance nonnegative and source approximately zero;
- triangle refinement convergence characterization.

---

## 15. Candidate consensus and disagreement maps

### 15.1 Motivation

Different model/classical backends may each fail differently. Instead of selecting one blindly, compare them.

Inputs:

```text
CandidateSet<T> from model A
CandidateSet<T> from model B
classical candidate C
```

Outputs:

- consensus candidate;
- disagreement field;
- confidence field;
- cluster/ranking report.

### 15.2 Consensus for masks

Combine calibrated or heuristic probabilities:

- intersection for conservative authorization;
- union for recall;
- weighted average plus edge-aware refinement;
- disagreement band sent to agent for another prompt.

### 15.3 Consensus for images/material maps

Do not average arbitrary candidates blindly. Compare:

- boundary agreement;
- low/high-frequency components;
- render residuals;
- outside-mask leakage;
- physical channel ranges;
- multi-view consistency.

A consensus op may select per-region sources and then seam-blend, but must expose the selection map.

### 15.4 Verification

- identical candidates return identity consensus and zero disagreement;
- outlier candidate lowers confidence;
- conservative mask mode never includes pixels absent from all inputs;
- selection map reconstructs output exactly;
- no candidate can affect outside authorized mask.

---

## 16. Topology-aware material wear field

### 16.1 Motivation

Procedural wear should depend on geometry, not only 2D image edges.

Compute mesh fields:

- mean/Gaussian curvature;
- ambient occlusion proxy;
- thickness;
- upward/downward orientation;
- geodesic distance to boundaries/features;
- contact/accessibility heuristics.

Combine into a wear potential:

```math
W = σ(
a κ_+ +
b (1-AO) +
c B +
d N_{up} +
e \eta
)
```

where `η` is correlated procedural noise and `σ` is a bounded transfer.

### 16.2 Operations

```text
surface.curvature
surface.ambient_occlusion
surface.thickness
surface.feature_distance
material.wear_potential
material.apply_wear
```

The wear field becomes a mask consumed by normal image/material operations.

### 16.3 Channel coupling

A wear event may produce correlated deltas:

- base color lightening/darkening;
- roughness change;
- metallic exposure;
- normal attenuation;
- edge chipping mask.

Represent this as a `MaterialDelta` resource so channel relationships are explicit.

### 16.4 Verification

- known primitives: sphere, cube, torus;
- curvature sign/magnitude sanity;
- rotation behavior of orientation terms;
- deterministic noise seed;
- channel ranges and correlation;
- no material boundary crossing unless allowed.

---

## 17. Differentiable material micro-fitting

### 17.1 Motivation

Given before/target views, fit a small local material edit rather than regenerate all maps.

Parameters might include:

```text
base-color affine transform
roughness offset/scale
normal strength
small UV splat basis coefficients
```

Objective:

```math
E(θ)=
\sum_v ||M_v(R_v(T;θ)-Y_v)||_ρ
+ λ_{outside}||\bar{M}_{uv}ΔT||^2
+ λ_{tv}TV(ΔT)
+ λ_{seam}C_{seam}(ΔT)
```

`ρ` may be robust Charbonnier/Huber loss.

### 17.2 Two implementation paths

1. External differentiable-rendering research adapter.
2. Project-specific differentiable raster/material subset using analytic derivatives or finite differences.

Start with finite differences over a very small parameter vector and cached render passes. The point is testing the operation contract, not winning inverse-rendering benchmarks.

### 17.3 Outputs

- candidate plan fragment;
- fitted parameter covariance/sensitivity;
- per-view residuals;
- unauthorized-change report;
- convergence report.

### 17.4 Verification

- synthetic target generated by known parameters;
- recover parameters/output under controlled lighting;
- multi-view improves identifiability over one view;
- parameter bounds enforced;
- unchanged geometry and unrelated material channels;
- optimizer failure remains a candidate/report.

---

## 18. Graph-level edit basis discovery

### 18.1 Motivation

Agents may repeatedly make similar edits with many correlated parameters. Discover a low-dimensional basis from prior accepted edit deltas or generated probes.

Given parameter-to-output Jacobian `J`, compute truncated SVD:

```math
J \approx U_r Σ_r V_r^T
```

`V_r` gives parameter directions with distinct visual effects. Expose these as macro controls:

```text
shadow_length
shadow_softness
warmth
texture_strength
edge_wear
```

Names can remain agent-assigned; the runtime provides numeric basis vectors and influence previews.

### 18.2 Operation

```text
analyze.parameter_jacobian
analyze.discover_edit_basis
apply.edit_basis_coordinate
```

### 18.3 Verification

- reconstructed probe deltas;
- orthogonality/numerical stability;
- held-out edit approximation;
- basis version tied to exact graph structure;
- no claim of semantic meaning without labels.

This is a longer-term agent ergonomics experiment.

---

## 19. Operations that look alien but should probably be rejected

### Arbitrary user WGSL/CUDA kernels

Too much attack surface and no stable semantics. Permit only registered, versioned kernels with manifests.

### “Enhance image”

No measurable contract. Split into explicit restoration candidates and metrics.

### “Make material realistic”

Underspecified and untestable. Expose material inference candidates, render objectives, and selected channel edits.

### End-to-end prompt edit as a destructive node

It hides targeting, provenance, model leakage, and uncertainty. At most, make it a candidate-plan generator outside the trusted core.

### Per-pixel Lua callbacks

Catastrophic performance and unbounded execution. Lua may generate graph data, not run hot loops.

### Universal learned feature map

Opaque embeddings are only composable when producer/model/version and consumer compatibility are exact. Do not create a generic “AI features” type with implied interoperability.

---

## 20. Prioritization

### High value, low research risk

1. Signed-distance mask calculus.
2. Structure tensor/orientation field.
3. Frequency split/recombine.
4. Poisson/screened-Poisson blend.
5. Patch field as a first-class resource.
6. Contract-driven micro-optimizer over splats.

### High value for material editing

1. Jacobian-driven UV splats.
2. Exact surface-map G-buffer.
3. Geodesic brush.
4. Seam graph and seam disagreement metric.
5. Multi-view coverage/confidence aggregation.

### High research risk but distinctive

1. Seam-graph Poisson repair.
2. Multi-view inverse material micro-fitting.
3. Delta provenance and parameter sensitivity.
4. Equality-saturation scheduling.
5. Learned/empirical backend autotuning.
6. Edit-basis discovery.

---

## 21. Experimental acceptance template

No alien op enters stable status without this record:

```markdown
# Experiment: <name>

## Hypothesis
What measurable problem does it solve better?

## Baselines
At least one simple baseline and one strong baseline.

## Contract
Typed inputs/outputs, determinism, limits, failure behavior.

## Fixtures
Synthetic truth plus representative real cases.

## Metrics
Locality, numeric error, boundary error, texture metric, runtime, memory.

## Results
Raw artifact paths and summary statistics.

## Failure modes
Where does it lose or become unstable?

## Decision
Reject / keep experimental / promote to versioned operation.
```

A cool image is not an acceptance criterion.

---

## 22. Relevant foundations

- Distance transforms: <https://cs.brown.edu/people/pfelzens/papers/dt-final.pdf>
- PatchMatch: <https://gfx.cs.princeton.edu/pubs/Barnes_2009_PAR/>
- Guided filter: <https://kaiminghe.com/publications/eccv10guidedfilter.pdf>
- Heat-method geodesics: <https://arxiv.org/abs/1204.6216>
- Differentiable surface splatting: <https://arxiv.org/abs/1906.04173>
- Soft Rasterizer: <https://arxiv.org/abs/1901.05567>
- `egg` equality saturation: <https://arxiv.org/abs/2004.03082>
- Kornia differentiable vision operators: <https://arxiv.org/abs/1910.02190>
