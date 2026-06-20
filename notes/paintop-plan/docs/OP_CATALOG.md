# Operation catalog and implementation priority

**Status:** proposed backlog; names and semantic boundaries should be validated during M0/M1  
**Legend:**

- **P0** — required for the first autonomous 2D loop.
- **P1** — strong classical editing and compiler foundation.
- **P2** — material/advanced perception expansion.
- **R** — research/experimental; promote only after benchmarked evidence.

Determinism labels use the tiers from `plan.md`.

---

## 1. Resource and I/O operations

| Operation | Signature | Priority | Determinism | Key contract/tests |
|---|---|---:|---|---|
| `io.decode_image@1` | file → `Image` | P0 | exact/bounded by codec | Decode limits, metadata, channel order, malformed files. |
| `io.encode_image@1` | `Image` → file export | P0 | exact for fixed lossless encoder contract | Round trip, alpha/color metadata, atomic write. |
| `image.create@1` | descriptor + fill → `Image` | P0 | exact | Extent/range/format. |
| `image.inspect@1` | resource → `Report` | P0 | exact | Extent, ranges, finite stats, hashes. |
| `image.convert_scalar@1` | `Image<T>` → `Image<U>` | P0 | exact/bounded | Quantization/rounding policy. |
| `image.extract_channel@1` | `Image` → `Field1`/`Mask` | P0 | exact | Channel semantics. |
| `image.assemble_channels@1` | fields → `Image` | P1 | exact | Shape/range/channel mapping. |
| `resource.hash@1` | resource → `Report` | P0 | exact | Canonical logical content hash. |
| `resource.copy@1` | resource → same type | P0 | exact | Mostly a debugging/materialization barrier. |
| `debug.materialize@1` | resource → same + artifact | P0 | exact | Must not change semantics. |

---

## 2. Color and alpha operations

| Operation | Signature | Priority | Determinism | Notes |
|---|---|---:|---|---|
| `color.convert@1` | color `Image` → color `Image` | P0 | bounded | Explicit source/destination encoding. |
| `color.matrix@1` | `Image` + 3×3/4×4 → `Image` | P0 | bounded | Linear domain only. |
| `color.adjust@1` | `Image` + optional `Mask` → `Image` | P0 | bounded | Exposure, contrast, saturation, hue, temperature as explicit sub-parameters. |
| `color.levels@1` | `Image` + ranges/gamma → `Image` | P1 | bounded | Reject degenerate ranges. |
| `color.curve@1` | `Image` + monotonic/general curve → `Image` | P1 | bounded | Interpolation fixed and tested. |
| `color.replace_range@1` | `Image` + color metric/mask → `Image` | P1 | bounded | Metric/space explicit. |
| `color.match_statistics@1` | source + reference + masks → `Image` | P2 | bounded | Mean/covariance match; guard singular covariance. |
| `alpha.premultiply@1` | straight `Image` → premultiplied `Image` | P0 | bounded/exact subset | Hidden RGB and alpha zero. |
| `alpha.unpremultiply@1` | premultiplied → straight | P0 | bounded | Epsilon/policy explicit. |
| `alpha.from_mask@1` | `Image` + `Mask` → `Image` | P0 | bounded | Replace/multiply semantics explicit. |
| `alpha.extract@1` | `Image` → `Mask` | P0 | exact | Coverage semantics. |
| `alpha.decontaminate_edges@1` | image + matte → image | P2 | bounded | Candidate operation with evidence. |

Avoid a monolithic “Photoshop adjustment” operation. Smaller typed nodes permit fusion without hiding semantics.

---

## 3. Primitive masks and paths

| Operation | Signature | Priority | Determinism | Key tests |
|---|---|---:|---|---|
| `mask.empty@1` | extent → `Mask` | P0 | exact | All zero. |
| `mask.full@1` | extent → `Mask` | P0 | exact | All one. |
| `mask.rect@1` | geometry → `Mask` | P0 | exact/bounded AA | Half-open pixel convention. |
| `mask.rounded_rect@1` | geometry → `Mask` | P1 | bounded | Radius constraints. |
| `mask.ellipse@1` | geometry → `Mask` | P0 | bounded | Analytic coverage/rotation. |
| `mask.polygon@1` | vertices/fill rule → `Mask` | P0 | bounded | Winding/even-odd, degenerate edges. |
| `mask.path@1` | path/stroke/fill → `Mask` | P1 | bounded | Curve flattening tolerance. |
| `mask.from_alpha@1` | `Image` → `Mask` | P0 | exact | Alpha representation-independent. |
| `mask.from_luminance@1` | `Image` → `Mask` | P1 | bounded | Color space explicit. |
| `mask.color_range@1` | `Image` + metric → `Mask` | P1 | bounded | Lab/linear RGB metric fixed. |
| `mask.threshold@1` | `Field1/Mask` → `Mask` | P0 | exact/bounded | Comparison semantics at equality. |
| `mask.smoothstep@1` | `Field1` → `Mask` | P0 | bounded | Edge ordering/range. |

---

## 4. Mask algebra, topology, and SDF

| Operation | Signature | Priority | Determinism | Key tests |
|---|---|---:|---|---|
| `mask.invert@1` | `Mask` → `Mask` | P0 | exact/bounded | Double inverse. |
| `mask.union@1` | masks → `Mask` | P0 | exact/bounded | Soft algebra variant explicit. |
| `mask.intersect@1` | masks → `Mask` | P0 | exact/bounded | Commutativity. |
| `mask.subtract@1` | masks → `Mask` | P0 | exact/bounded | `A-A=0`. |
| `mask.xor@1` | masks → `Mask` | P1 | exact/bounded | Hard/soft definition. |
| `mask.bounds@1` | `Mask` → `Report` | P0 | exact | Empty mask behavior. |
| `mask.connected_components@1` | hard mask → `LabelMap` + report | P1 | exact | Label stability policy. |
| `mask.fill_holes@1` | hard mask → hard mask | P1 | exact | Border-connected definition. |
| `mask.remove_components@1` | mask + size policy → mask | P1 | exact | Area/connectivity. |
| `mask.to_sdf@1` | mask + contour policy → `SdfMask` | P1 | exact reference | Brute-force oracle. |
| `sdf.to_mask@1` | SDF + profile → `Mask` | P1 | bounded | Physical width. |
| `sdf.offset@1` | SDF + distance → SDF | P1 | bounded | Composition. |
| `sdf.union@1` | SDFs → SDF | P1 | bounded | `min`. |
| `sdf.intersect@1` | SDFs → SDF | P1 | bounded | `max`. |
| `sdf.subtract@1` | SDFs → SDF | P1 | bounded | Difference law. |
| `sdf.smooth_union@1` | SDFs + radius → SDF | P2 | bounded | Limit to hard union. |
| `sdf.normal@1` | SDF → `Field2` | P2 | bounded | Gradient orientation/norm. |
| `sdf.curvature@1` | SDF → `Field1` | R | bounded | Circle analytic behavior. |

Compatibility convenience macros may expose `mask.grow`, `mask.shrink`, and `mask.feather`, but normalization should reduce them to SDF or explicitly defined morphology nodes.

> **Priority note (see [`../M0_DECISIONS.md`](../M0_DECISIONS.md) D1):** exact-EDT + the SDF mask calculus (`mask.to_sdf`, `sdf.to_mask`, `sdf.offset`, and the boolean SDF ops) are the **M1.5 priority** — the first slice to land immediately after the MVP touch-up loop is green — not M4. They are deferred out of the first vertical slice only because the MVP feathers analytically on `mask.ellipse@1`; they are foundational and should land second.

---

## 5. Geometry and resampling

| Operation | Signature | Priority | Determinism | Notes |
|---|---|---:|---|---|
| `transform.affine@1` | transform parameters → `Transform2D` | P0 | exact params | Matrix convention explicit. |
| `transform.projective@1` | matrix → `Transform2D` | P1 | exact params | Reject singular. |
| `image.crop@1` | image + rect → image | P0 | exact | Half-open bounds. |
| `image.pad@1` | image + margins/boundary → image | P0 | exact/bounded | Negative padding rejected or normalized to crop. |
| `image.resize@1` | image + extent/filter → image | P0 | bounded | Pixel centers and kernel fixed. |
| `image.warp@1` | image + inverse transform → image | P1 | bounded | ROI footprint. |
| `image.flip@1` | image + axis → image | P0 | exact | Double flip identity. |
| `image.rotate90@1` | image + quarter turns → image | P0 | exact | Useful metamorphic fixture. |
| `image.rotate@1` | image + angle/filter → image | P1 | bounded | Extent policy. |
| `field.warp@1` | field + transform → field | P1 | bounded | Vector-space transform policy. |
| `mask.warp@1` | mask + transform → mask | P1 | bounded | Coverage resampling, not Boolean nearest by default. |

Initial resamplers:

- nearest;
- bilinear;
- bicubic with fixed cubic parameter;
- Lanczos with fixed lobe count.

Every kernel declares support and halo.

---

## 6. Paint and fill

| Operation | Signature | Priority | Determinism | Notes |
|---|---|---:|---|---|
| `paint.fill@1` | base + mask + color/value → image | P0 | bounded | Typed scalar/color fill. |
| `paint.linear_gradient@1` | geometry/stops → image/field | P0 | bounded | Stop interpolation/color space. |
| `paint.radial_gradient@1` | geometry/stops → image/field | P0 | bounded | Degenerate radii. |
| `paint.conic_gradient@1` | geometry/stops → image/field | P1 | bounded | Angle wrap. |
| `paint.gaussian_splats@1` | base + splat batch + clip → image | P0 | bounded | Batch limits, covariance, truncation. |
| `paint.adaptive_splats@1` | base + orientation/surface map → image/UV layer | R | bounded | See `ALIEN_OPS.md`. |
| `paint.erase@1` | image + mask/brush → image | P0 | bounded | Alpha operation, not RGB painting. |
| `paint.stroke_splats@1` | polyline + brush sampling → splat batch/image | P1 | bounded | Sampling spacing/jitter seed. |
| `paint.clone_stamps@1` | source + mapping + stamps → image | P1 | bounded | Source transform/overlap. |
| `paint.dodge_burn@1` | image + mask → image | P2 | bounded | Define linear/tone-space behavior; may remain macro. |

A full physically simulated bristle brush is not an initial goal.

---

## 7. Compositing and blending

| Operation | Signature | Priority | Determinism | Notes |
|---|---|---:|---|---|
| `composite.over@1` | src + dst → image | P0 | bounded | Premultiplied linear semantics. |
| `composite.masked_replace@1` | edited + base + mask → image | P0 | bounded | Core authorization primitive. |
| `composite.blend@1` | src + dst + mode/opacity/mask → image | P0/P1 | bounded | Mode color space explicit. |
| `composite.stack@1` | ordered layers → image | P1 | bounded | Normalizes to pairwise operations or optimized fold. |
| `composite.edge_blend@1` | source/target/mask → image | P1 | bounded | Prefer explicit method. |
| `effect.drop_shadow@1` | mask + params → layer | P2 | bounded | Macro over offset/blur/fill. |
| `effect.glow@1` | mask + params → layer | P2 | bounded | Macro. |

Initial blend modes:

- normal/over;
- add;
- subtract;
- multiply;
- screen;
- darken/lighten;
- difference;
- overlay/soft-light after exact semantics are pinned.

Do not inherit ambiguous application-specific behavior without a documented formula and color space.

---

## 8. Convolution, filtering, and morphology

| Operation | Signature | Priority | Determinism | Key path |
|---|---|---:|---|---|
| `filter.convolve@1` | image/field + kernel → same | P0 | bounded/exact subset | Scalar direct oracle. |
| `filter.gaussian_blur@1` | image/field + sigma → same | P0 | bounded | Separable optimized path. |
| `filter.box_blur@1` | image/field + radius → same | P1 | bounded | Integral or sliding window. |
| `filter.median@1` | image/field + window → same | P1 | bounded/exact | Tiny brute oracle. |
| `filter.bilateral@1` | image + spatial/range sigmas → image | P1 | bounded | Expensive reference, possible approximation. |
| `filter.guided@1` | input + guide → output | P1 | bounded | Local linear model. |
| `filter.unsharp@1` | image + sigma/amount → image | P1 | bounded | Macro or fused operation. |
| `filter.sobel@1` | field/image → `Field2` | P1 | bounded | Kernel convention. |
| `filter.laplacian@1` | field/image → field/image | P1 | bounded | Discrete stencil. |
| `filter.structure_tensor@1` | image/field → tensor/orientation/coherence | P1 | bounded | Gradient and smoothing scales. |
| `filter.anisotropic_diffusion@1` | image/field → same + report | P2 | bounded | Iteration/stability. |
| `morphology.dilate@1` | hard mask/field → same | P1 | exact/bounded | Structuring element explicit. |
| `morphology.erode@1` | hard mask/field → same | P1 | exact/bounded | Duality. |
| `morphology.open@1` | mask → mask | P1 | exact/bounded | Macro. |
| `morphology.close@1` | mask → mask | P1 | exact/bounded | Macro. |

The compiler may choose direct, sparse, separable, low-rank, integral, recursive, or FFT schedules according to the operation contract.

---

## 9. Frequency and multiscale operations

| Operation | Signature | Priority | Determinism | Notes |
|---|---|---:|---|---|
| `frequency.gaussian_pyramid@1` | image/field → `Pyramid` | P1 | bounded | Phase/downsample convention. |
| `frequency.laplacian_split@1` | image → `Pyramid` | P1 | bounded | Reconstruction test. |
| `frequency.recombine@1` | `Pyramid` → image | P1 | bounded | Inverse contract. |
| `frequency.fft2@1` | field/image → complex spectrum | P2 | bounded | Internal typed complex resource. |
| `frequency.ifft2@1` | spectrum → field/image | P2 | bounded | Round trip. |
| `frequency.bandpass@1` | image/field + response → same | P2 | bounded | Boundary/window policy. |
| `frequency.match_band_energy@1` | source + reference → image | R | bounded | Gain limits. |
| `frequency.transfer_residual@1` | source/target pyramids + mapping → image | R | bounded | See alien ops. |

FFT should not be exposed early merely because it is impressive. The first consumer should justify complex resource semantics and padding/window rules.

---

## 10. Texture repair and PDE operations

| Operation | Signature | Priority | Determinism | Notes |
|---|---|---:|---|---|
| `repair.copy_region@1` | source + transform + mask → candidate layer | P1 | bounded | Exact source mapping. |
| `repair.clone@1` | image + source map + target mask → image | P1 | bounded | Outside identity. |
| `repair.patch_field@1` | source/target/masks → `PatchField` | P1 | reproducible | PatchMatch or exact tiny backend. |
| `repair.patch_synthesize@1` | image + field + hole → candidate | P1 | reproducible | Field validity. |
| `repair.poisson_blend@1` | source/target/mask → candidate + report | P1 | bounded | Residual/convergence. |
| `repair.screened_poisson@1` | guidance/anchor/mask → candidate + report | P1 | bounded | Lambda semantics. |
| `repair.boundary_harmonic@1` | image + hole → low-frequency candidate | R | bounded | See alien ops. |
| `repair.residual_transplant@1` | source/target/field → candidate | R | bounded | Frequency/orientation tests. |
| `repair.seam_aware@1` | texture + seam graph + mask → candidate | R | bounded | Surface topology. |

Convenience `repair.heal@1` may normalize into a subgraph after the components stabilize. Avoid one opaque implementation early.

---

## 11. Procedural generators and fields

| Operation | Signature | Priority | Determinism | Notes |
|---|---|---:|---|---|
| `field.constant@1` | extent + value → field | P0 | exact | Fixture/helper. |
| `field.coordinate@1` | extent → `Field2` | P1 | exact | Pixel/normalized spaces. |
| `field.noise@1` | extent + type + seed → field | P1 | reproducible | Hash-based deterministic noise. |
| `field.fbm@1` | extent + octaves + seed → field | P1 | reproducible | Frequency normalization. |
| `field.cellular@1` | extent + seed → fields | P2 | reproducible | Distance/cell ID outputs. |
| `field.reaction_diffusion@1` | initial fields + params → fields/report | P2 | reproducible/bounded | Stability and steps. |
| `field.domain_warp@1` | image/field + displacement → same | P1 | bounded | Uses resampling contract. |
| `field.distance_to_points@1` | point set → field | P1 | exact/bounded | Brute oracle. |
| `field.orientation@1` | structure tensor → `Field2` + coherence | P1 | bounded | Eigenvector sign convention. |
| `field.color_ramp@1` | scalar field + stops → image | P1 | bounded | Interpolation. |

---

## 12. Analysis, reports, and assertions

| Operation | Signature | Priority | Determinism | Notes |
|---|---|---:|---|---|
| `analyze.statistics@1` | resource + optional mask → report | P0 | exact/bounded reduction | Stable reduction tree. |
| `analyze.histogram@1` | image/field + bins → histogram | P0 | exact | Domain/bins explicit. |
| `analyze.sample@1` | resource + coordinates → report | P0 | exact/bounded | Sample filter explicit. |
| `analyze.diff@1` | before + after → fields/report | P0 | exact/bounded | Comparison space. |
| `analyze.changed_bounds@1` | diff + threshold → report | P0 | exact | Empty behavior. |
| `analyze.edge_map@1` | image/field → field | P1 | bounded | Built from explicit filters. |
| `analyze.connected_components@1` | mask/labels → labels/report | P1 | exact | Connectivity. |
| `analyze.frequency_energy@1` | image/field + bands → report | P1 | bounded | Useful texture preservation metric. |
| `analyze.perceptual_metric@1` | images + metric/model → report | P2 | bounded/stochastic | Secondary evidence only. |
| `assert.finite@1` | resource → report/fail | P0 | exact | NaN/Inf locations. |
| `assert.range@1` | field/image → report/fail | P0 | exact | Count/worst value. |
| `assert.no_change_outside_mask@1` | before/after/mask → report/fail | P0 | exact/bounded | Core safety assertion. |
| `assert.changed_bounds@1` | before/after/bounds → report/fail | P0 | exact | Threshold explicit. |
| `assert.max_delta@1` | before/after → report/fail | P0 | exact/bounded | Spaces/ranges. |
| `assert.min_change_inside_mask@1` | before/after/mask → report/fail | P1 | exact/bounded | Avoid successful no-op. |
| `assert.alpha_valid@1` | image → report/fail | P0 | exact/bounded | Range/premul. |
| `assert.normalized_vectors@1` | field → report/fail | P2 | bounded | Angular/norm tolerance. |
| `assert.edge_continuity@1` | image + boundary → report/fail | P1 | bounded | Gradient jump metric. |
| `assert.frequency_preserved@1` | images + mask/bands → report/fail | P2 | bounded | Texture touch-up. |
| `assert.asset_mutation_allowlist@1` | assets → report/fail | P2 | exact structural | Material safety. |
| `assert.uv_seam_consistency@1` | texture + seam graph → report/fail | P2 | bounded | Surface-aware. |

Assertions should be ordinary typed operations internally so they participate in graph demand and evidence.

---

## 13. Model/perception operations

| Operation | Signature | Priority | Determinism | Default output |
|---|---|---:|---|---|
| `model.segment_prompted@1` | image + prompts + manifest → candidates | P2 | bounded/stochastic | `CandidateSet<Mask>` + confidence. |
| `model.refine_matte@1` | image + coarse mask/prompts → mask/confidence | P2 | bounded/stochastic | Soft matte. |
| `model.estimate_depth@1` | image → field/confidence | P2 | bounded/stochastic | Relative/metric depth explicit. |
| `model.estimate_normals@1` | image → field/confidence | P2 | bounded/stochastic | Camera-space normals. |
| `model.restore_candidates@1` | image + mask/degradation controls → candidates | P2 | stochastic | Candidate images. |
| `model.inpaint_candidates@1` | image + mask + seed → candidates | P2 | stochastic | Candidate images. |
| `model.intrinsic_candidates@1` | image → decomposition candidates | R | stochastic | Albedo/shading/specular/confidence. |
| `model.material_candidates@1` | image/views/asset → material candidates | R | stochastic | PBR channels + uncertainty. |
| `candidate.rank@1` | candidate set + objectives → ranked set | P2 | bounded | All metric values visible. |
| `candidate.select@1` | ranked set + explicit choice → resource | P2 | exact | Selection never implicit. |
| `candidate.consensus@1` | candidate sets → candidate/report | R | bounded | Disagreement map. |

---

## 14. glTF material operations

| Operation | Signature | Priority | Determinism | Notes |
|---|---|---:|---|---|
| `material.inspect@1` | asset → report/material set | P2 | exact | Full slot/factor/sampler metadata. |
| `material.extract_texture@1` | asset + material/slot → typed image | P2 | exact/bounded codec | Preserve semantic role. |
| `material.replace_texture@1` | asset + typed image → asset | P2 | exact structural | Allowlist mutation. |
| `material.set_factor@1` | asset + factor → asset | P2 | exact | PBR factor ranges. |
| `material.unpack_metallic_roughness@1` | packed image → fields | P2 | exact | G/B channels. |
| `material.pack_metallic_roughness@1` | roughness + metallic + original? → image | P2 | exact/bounded | Preserve unused channels. |
| `material.extract_occlusion@1` | packed image → field | P2 | exact | Red channel. |
| `material.decode_normal@1` | normal texture → `Field3` | P2 | bounded | Tangent vector semantics. |
| `material.encode_normal@1` | `Field3` → normal texture | P2 | bounded | Renormalize/quantize. |
| `material.perturb_normal@1` | normal field + perturbation → normal field | P2 | bounded | Tangent-space composition. |
| `material.height_to_normal@1` | height field → normal field | P2 | bounded | Scale/texel units. |
| `material.delta@1` | channel deltas → `MaterialDelta` | R | bounded | Correlated channel edits. |
| `material.apply_delta@1` | material set + delta/mask → material set | R | bounded | Range/locality. |

---

## 15. Surface and `msh` bridge operations

| Operation | Signature | Priority | Determinism | Notes |
|---|---|---:|---|---|
| `surface.render_passes@1` | asset + camera + pass set → resources | P2 | bounded/exact IDs | Backed by `msh-render`. |
| `surface.query_screen@1` | asset + camera + points → report | P2 | bounded/exact IDs | Triangle/barycentric/UV/depth. |
| `surface.project_screen_mask@1` | asset + camera + mask → `SurfaceMap` | P2 | bounded | Visibility/filter policy. |
| `surface.rasterize_uv_coverage@1` | surface map → mask/confidence | P2 | bounded | Jacobian EWA footprint. |
| `surface.build_seam_graph@1` | asset + texture target → graph | R | exact topology/bounded samples | Cacheable. |
| `surface.seam_disagreement@1` | texture/field + graph → report/field | P2/R | bounded | Diagnostic first. |
| `surface.geodesic_distance@1` | mesh + seeds → mesh field | R | bounded | Heat method. |
| `surface.distance_to_uv_mask@1` | mesh field + radius + target → mask | R | bounded | UV rasterization. |
| `surface.curvature@1` | mesh → mesh field | R | bounded | Discretization explicit. |
| `surface.ambient_occlusion@1` | mesh/render policy → field | R | bounded/stochastic if sampled | Seed and rays. |
| `surface.multi_view_accumulate@1` | surface projections → UV candidate/confidence | R | bounded | View weights. |
| `surface.optimize_material@1` | asset + target views + variables → candidates/report | R | stochastic/bounded | Micro-fitting. |

---

## 16. Compiler/meta operations

These are not visual filters, but should be visible to tooling.

| Operation/tooling concept | Priority | Purpose |
|---|---:|---|
| `graph.bind_parameters` | P1 | Instantiate reusable static subgraphs without arbitrary code. |
| `graph.select` | P1 | Explicit data/resource selection under a typed condition known at compile time. |
| `graph.barrier` | P1 | Preserve an intermediate for evidence; blocks some fusion. |
| `graph.cache_hint` | P1 | Non-semantic schedule hint. |
| `graph.compare_backends` | P1 | Evidence-only alternate execution. |
| `graph.optimize_program` | R | Contract-driven micro-optimizer returning plan candidates. |
| `graph.influence_probe` | R | Parameter sensitivity maps. |

Runtime data-dependent general control flow should not enter v1. Iterative numerical operations remain bounded nodes with internal convergence contracts.

---

## 17. Macros versus primitive operations

A macro is authoring sugar normalized into ordinary nodes. Prefer macros for:

- unsharp mask = Gaussian blur + residual scale + add;
- drop shadow = mask transform + blur + fill + composite;
- grow/shrink/feather = SDF nodes;
- heal = decomposition + patch field + synthesis + Poisson blend;
- material wear = geometry fields + procedural field + channel deltas;
- before/after contact sheet = layout/composite/debug exports.

A macro should not hide:

- model selection;
- candidate choice;
- authorization mask;
- approximation budget;
- destructive export;
- failure/convergence.

The normalized plan always exposes the expanded graph.

---

## 18. P0 conformance set

The first stable operation set should remain small:

```text
io.decode_image
io.encode_image
image.create
image.inspect
color.convert
color.adjust
alpha.premultiply
alpha.unpremultiply
mask.empty
mask.full
mask.rect
mask.ellipse
mask.polygon
mask.invert
mask.union
mask.intersect
mask.subtract
mask.bounds
image.crop
image.pad
image.resize
paint.fill
paint.linear_gradient
paint.radial_gradient
paint.gaussian_splats
composite.over
composite.masked_replace
composite.blend (restricted modes)
filter.convolve
filter.gaussian_blur
analyze.statistics
analyze.histogram
analyze.diff
analyze.changed_bounds
assert.finite
assert.range
assert.no_change_outside_mask
assert.changed_bounds
assert.alpha_valid
debug.materialize
```

This is enough to prove the compiler, evidence, and agent loop. Do not add P1/P2 operations merely to make the catalog look impressive.

---

## 19. Operation review questions

Before accepting a proposed operation, ask:

1. Is this one coherent mathematical transformation or an opaque workflow?
2. Does it produce a reusable intermediate representation?
3. Can its ROI and halo be defined?
4. Are color, alpha, coordinates, boundary, and units explicit?
5. Is it pure and cacheable?
6. What is the reference oracle?
7. Which properties and metamorphic relations hold?
8. Does it need candidate semantics?
9. Can it be expressed as a macro over existing nodes?
10. Is its output useful to more than one downstream operation?
11. Can a coding agent diagnose failure from evidence?
12. Does it justify its implementation and maintenance cost?

If most answers are weak, reject or keep it outside the stable IR.
