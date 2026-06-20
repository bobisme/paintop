# paintop IR specification sketch

**Status:** design target for `plan-v1`; intentionally strict and subject to implementation feedback  
**Purpose:** define a canonical, typed, hashable, side-effect-free graph that coding agents can generate and the runtime can validate before execution

---

## 1. Design goals

The IR must be:

- safe to parse without executing code;
- explicit about image, mask, field, color, alpha, and coordinate semantics;
- easy for agents to discover through schemas and examples;
- stable enough for conformance fixtures;
- expressive enough for DAGs, reusable resources, candidates, assertions, and material edits;
- optimizable without changing visible semantics;
- replayable and content-addressable;
- strict enough that typos fail rather than silently change output.

It is not intended to be a pleasant handwritten language. Human convenience belongs in templates, YAML conversion, or a later Lua graph builder.

---

## 2. Top-level document

```json
{
  "paintop": "1.0",
  "name": "darken-and-repair",
  "description": "Localized repair with bounded changes",
  "policy": {},
  "inputs": {},
  "nodes": [],
  "assertions": [],
  "exports": {},
  "evidence": {},
  "extensions": {}
}
```

### 2.1 Required fields

| Field | Meaning |
|---|---|
| `paintop` | Plan language major/minor version. |
| `inputs` | External resource declarations. |
| `nodes` | Operation nodes in any topological-compatible order. |
| `exports` | Explicit resource sinks. |

### 2.2 Optional fields

| Field | Meaning |
|---|---|
| `name` | Stable human-readable plan name. |
| `description` | Non-semantic text. Excluded from content identity unless policy says otherwise. |
| `policy` | Resource, execution, model, path, and determinism constraints. |
| `assertions` | Postconditions or cross-resource predicates. |
| `evidence` | Requested debug artifacts and trace detail. |
| `extensions` | Namespaced non-core metadata. |

Unknown top-level or node fields are errors. Experimental data must live under a reverse-domain or repository-qualified namespace inside `extensions`.

---

## 3. Identifiers and references

### 3.1 Node identifiers

Node IDs are ASCII strings matching:

```regex
^[A-Za-z][A-Za-z0-9_.-]{0,127}$
```

They are unique within a plan.

Good:

```text
mask.jacket
blur.low_frequency
candidate.inpaint.0
```

Bad:

```text
../output
mask/jacket
🔥
```

Unicode labels may be stored in non-semantic metadata, not identifiers.

### 3.2 Resource references

Canonical reference syntax:

```text
input:<input-id>
node:<node-id>/<output-port>
```

Examples:

```json
"image": "input:source"
"mask": "node:mask.feathered/mask"
```

Authoring sugar may accept `source` or `mask.feathered.mask`, but normalization emits canonical references.

### 3.3 Output ports

Each operation manifest defines named output ports. A node may not invent ports.

```json
{
  "id": "split",
  "op": "frequency.laplacian_split@1",
  "in": {"image": "input:source"},
  "params": {"levels": 4}
}
```

May produce:

```text
node:split/lowpass
node:split/residuals
node:split/metadata
```

The runtime should expose the output schema through `paintop op schema`.

---

## 4. Input declarations

External inputs are typed and policy-bound.

```json
{
  "inputs": {
    "source": {
      "kind": "image.file",
      "path": "input.png",
      "decode": {
        "desired_format": "rgba8",
        "color": "from-file-or-srgb",
        "alpha": "straight"
      },
      "limits": {
        "max_width": 8192,
        "max_height": 8192,
        "max_pixels": 67108864
      }
    },
    "allowed": {
      "kind": "mask.file",
      "path": "allowed.png",
      "decode": {"channel": "luminance"}
    }
  }
}
```

Initial input kinds:

- `image.file`;
- `mask.file`;
- `json.file` for structured non-image parameters under a schema;
- `binary.file` only for explicitly registered consumers;
- later `gltf.file`, `surface-map.file`, `model.asset`.

Paths are resolved under the invocation’s declared input root. Plans cannot escape roots.

A caller embedding `paintop` may supply handles or byte streams while preserving the same logical declarations.

---

## 5. Node form

```json
{
  "id": "blurred",
  "op": "filter.gaussian_blur@1",
  "in": {
    "image": "node:linear/image",
    "mask": "node:selection/mask"
  },
  "params": {
    "sigma": 8.0,
    "boundary": {"mode": "mirror"}
  },
  "hints": {
    "preferred_backend": "auto",
    "materialize": false
  },
  "extensions": {}
}
```

Required:

- `id`;
- `op`;
- `in` if the operation has inputs;
- `params` if required parameters have no defaults.

`hints` may influence schedule but not semantics. Unsupported hints are warnings or policy-controlled errors; they are excluded from semantic content hashes unless they change a declared implementation contract.

---

## 6. Operation IDs and versioning

Format:

```text
<namespace>.<name>@<semantic-major>
```

Examples:

```text
mask.rect@1
mask.to_sdf@1
paint.gaussian_splats@1
filter.convolve@1
assert.no_change_outside_mask@1
material.pack_metallic_roughness@1
```

The major version defines semantics. Backward-compatible parameter additions may occur without changing the major only if normalization fills a fixed default and old normalized plans retain their meaning.

Implementations are separately versioned:

```text
cpu.reference@1
cpu.simd-separable@3
wgpu.separable@2
```

An implementation version does not change operation semantics; it changes execution provenance and cache compatibility.

---

## 7. Resource descriptors

The compiler infers resource descriptors, but inputs and explicit conversions expose them.

### 7.1 Image descriptor

```json
{
  "kind": "Image",
  "extent": {"width": 2048, "height": 2048},
  "layout": "rgba",
  "scalar": "f32",
  "color": {
    "encoding": "linear-srgb",
    "range": "scene-referred"
  },
  "alpha": "premultiplied",
  "coordinates": "pixel-center-upper-left",
  "semantic": "color"
}
```

### 7.2 Mask descriptor

```json
{
  "kind": "Mask",
  "extent": {"width": 2048, "height": 2048},
  "scalar": "f32",
  "range": [0.0, 1.0],
  "meaning": "coverage"
}
```

### 7.3 Field descriptor

```json
{
  "kind": "Field3",
  "extent": {"width": 2048, "height": 2048},
  "scalar": "f32",
  "semantic": "normal",
  "space": "tangent",
  "normalization": "unit",
  "encoding": "signed-vector"
}
```

### 7.4 SDF descriptor

```json
{
  "kind": "SdfMask",
  "extent": {"width": 2048, "height": 2048},
  "scalar": "f32",
  "units": "pixels",
  "sign": "negative-inside"
}
```

The sign convention must never be implicit.

---

## 8. Color and alpha conversion nodes

No operation silently converts transfer functions or alpha representation.

```json
{
  "id": "linear",
  "op": "color.convert@1",
  "in": {"image": "input:source"},
  "params": {
    "to": "linear-srgb",
    "rendering_intent": "relative-colorimetric"
  }
},
{
  "id": "premul",
  "op": "alpha.premultiply@1",
  "in": {"image": "node:linear/image"}
}
```

The compiler may fuse these physically, but normalized semantics retain distinct logical nodes unless a proven canonical rewrite replaces them.

Rules:

- material scalar maps use `raw-linear`, not color encodings;
- normal maps require explicit decode/encode nodes;
- premultiplication on nonlinear encoded RGB is rejected;
- operations that require linear light reject sRGB inputs;
- transparent RGB values are preserved or normalized according to explicit node semantics, never accidentally discarded.

---

## 9. Mask semantics

### 9.1 Coverage

A `Mask` value is coverage, not Boolean truth. Operations consuming masks use:

```math
out = m \cdot edited + (1-m) \cdot original
```

in the declared compositing space, usually premultiplied linear light.

### 9.2 Hard-mask conversion

```json
{
  "id": "hard",
  "op": "mask.threshold@1",
  "in": {"mask": "node:soft/mask"},
  "params": {
    "threshold": 0.5,
    "comparison": "greater-or-equal"
  }
}
```

### 9.3 SDF morphology

Preferred morphology path:

```text
Mask → threshold or contour policy → SdfMask
SdfMask offset by distance
SdfMask → coverage using explicit reconstruction profile
```

This provides resolution-aware grow, shrink, feather, smooth union, and smooth intersection.

Example:

```json
{
  "id": "sdf",
  "op": "mask.to_sdf@1",
  "in": {"mask": "node:ellipse/mask"},
  "params": {"threshold": 0.5, "sign": "negative-inside"}
},
{
  "id": "grown",
  "op": "sdf.offset@1",
  "in": {"sdf": "node:sdf/sdf"},
  "params": {"distance_px": -6.0}
},
{
  "id": "feathered",
  "op": "sdf.to_mask@1",
  "in": {"sdf": "node:grown/sdf"},
  "params": {
    "edge": {"profile": "smoothstep", "half_width_px": 4.0}
  }
}
```

---

## 10. Parameters and units

Parameters with dimensional meaning include units in their name or typed form.

Preferred:

```json
{"sigma_px": 4.0}
{"angle_rad": 0.7853981633974483}
{"exposure_ev": -0.5}
```

Acceptable typed unit:

```json
{"angle": {"value": 45.0, "unit": "deg"}}
```

Normalization converts to canonical units.

Avoid bare ambiguous values:

```json
{"radius": 4}
{"angle": 45}
```

Numeric requirements:

- JSON `NaN` and infinity are never accepted;
- integer parameters reject fractional values;
- dimensions and byte calculations use checked arithmetic;
- negative zero is normalized where semantics do not distinguish it;
- canonical floating formatting is defined by the canonical serializer.

---

## 11. Seeds and stochasticity

Every stochastic operation must have a seed after normalization.

```json
{
  "id": "texture",
  "op": "synthesis.patchmatch@1",
  "in": {"image": "input:source", "hole": "node:hole/mask"},
  "params": {
    "seed": 1296630431,
    "iterations": 6,
    "patch_radius_px": 4
  }
}
```

A top-level policy may provide a default seed derivation:

```text
seed(node) = keyed_hash(plan_seed, normalized_node_id)
```

The normalized plan must include the resolved numeric seed.

Model providers may still be nondeterministic. Their determinism tier and provider identity are recorded.

---

## 12. Candidate sets

Opaque or stochastic operations should not directly become the final image.

```json
{
  "id": "inpaint",
  "op": "model.inpaint_candidates@1",
  "in": {
    "image": "node:premul/image",
    "mask": "node:hole/mask"
  },
  "params": {
    "model": "model:lama-onnx-sha256:...",
    "count": 4,
    "seed": 42
  }
},
{
  "id": "rank",
  "op": "candidate.rank@1",
  "in": {
    "candidates": "node:inpaint/candidates",
    "reference": "node:premul/image",
    "mask": "node:hole/mask"
  },
  "params": {
    "objectives": [
      {"metric": "boundary-gradient-error", "weight": 1.0},
      {"metric": "outside-mask-delta", "weight": 100.0},
      {"metric": "patch-coherence", "weight": 0.4}
    ]
  }
},
{
  "id": "selected",
  "op": "candidate.select@1",
  "in": {"ranked": "node:rank/candidates"},
  "params": {"index": 0}
}
```

Candidate metadata should include:

- producer/model hash;
- seed;
- confidence;
- objective metrics;
- uncertainty map if available;
- changed bounds;
- preview path in the evidence bundle.

---

## 13. Assertions

Assertions may appear as top-level declarations or normal graph nodes. Top-level syntax is concise and normalizes to nodes.

```json
{
  "assertions": [
    {
      "id": "localized",
      "op": "assert.no_change_outside_mask@1",
      "in": {
        "before": "input:source",
        "after": "node:encoded/image",
        "allowed": "node:authorized/mask"
      },
      "params": {
        "max_abs_delta": 0,
        "comparison_space": "decoded-linear"
      }
    },
    {
      "id": "finite",
      "op": "assert.finite@1",
      "in": {"resource": "node:final/image"}
    }
  ]
}
```

Assertions produce a `Report` and determine run success.

Assertion severity:

- `error`: fail run;
- `warning`: retain output but mark evidence;
- `metric`: never fail; record measurement.

Severity is explicit and may be constrained by policy.

---

## 14. Exports

Exports are the only ordinary filesystem side effects.

```json
{
  "exports": {
    "final": {
      "resource": "node:encoded/image",
      "kind": "image",
      "path": "out.png",
      "encoding": {
        "format": "png",
        "compression": 6
      },
      "overwrite": false
    },
    "mask": {
      "resource": "node:authorized/mask",
      "kind": "debug-mask",
      "path": "allowed-mask.png"
    },
    "report": {
      "resource": "node:stats/report",
      "kind": "json",
      "path": "stats.json"
    }
  }
}
```

Rules:

- paths are relative to an explicit output root;
- writes are temporary-then-atomic-rename where supported;
- an existing file is not overwritten unless both plan and invocation policy allow it;
- output content hashes are recorded;
- encoding settings are semantic for the file export but do not alter upstream resource identity.

---

## 15. Policy

Example:

```json
{
  "policy": {
    "resources": {
      "max_nodes": 256,
      "max_pixels_per_resource": 67108864,
      "max_live_bytes": 2147483648,
      "max_splats": 100000,
      "max_iterations": 2000
    },
    "execution": {
      "deadline_ms": 30000,
      "threads": 8,
      "allowed_backends": ["cpu-reference", "cpu-optimized", "wgpu"],
      "required_determinism": "bounded",
      "network": "deny"
    },
    "models": {
      "allowed": ["sam2-tiny:*", "zim:*"],
      "max_calls": 4,
      "require_hash": true,
      "allow_process_adapter": false
    },
    "paths": {
      "input_root": ".",
      "output_root": "./run",
      "allow_overwrite": false
    },
    "debug": {
      "max_artifact_bytes": 536870912
    }
  }
}
```

Invocation policy may tighten but not loosen a plan’s constraints unless an explicit privileged mode is used.

---

## 16. Evidence request

```json
{
  "evidence": {
    "trace": "detailed",
    "graph": ["dot", "svg"],
    "contact_sheet": true,
    "materialize": [
      "node:authorized/mask",
      "node:painted/image",
      "node:final/image"
    ],
    "diffs": [
      {
        "before": "input:source",
        "after": "node:final/image",
        "space": "linear",
        "heatmap": true
      }
    ],
    "differential": {
      "backend": "cpu-reference",
      "sample": {"mode": "changed-tiles"}
    }
  }
}
```

Evidence requests may force materialization and therefore change schedule and cost, but not output semantics.

---

## 17. Canonicalization

Canonicalization is required for hashing, caching, replay, and meaningful diffs.

Rules:

1. Resolve all defaults.
2. Resolve all seeds.
3. Expand aliases and sequential sugar.
4. Emit canonical operation IDs with major version.
5. Sort object keys lexicographically in canonical serialization.
6. Preserve array order where order is semantic.
7. Normalize unit-bearing values to canonical units.
8. Normalize resource references.
9. Normalize boundary, color, alpha, and coordinate descriptors.
10. Remove non-semantic comments/descriptions from the semantic hash.
11. Represent floats with a single round-trippable format.
12. Reject duplicate keys before canonicalization.
13. Include extensions in the semantic hash only when their namespace declares semantic impact.

The runtime emits both:

- `normalized-plan.json`: readable, stable formatting;
- `semantic-hash`: hash of canonical semantic bytes.

A compiler version must not be included in the semantic hash, but operation semantic versions must be. Execution provenance separately records compiler/runtime versions.

---

## 18. Shape, ROI, and halo functions

Operation manifests name executable contract functions:

```rust
trait OpContract {
    fn infer_outputs(&self, inputs: &Descriptors, params: &Value)
        -> Result<OutputDescriptors>;

    fn required_inputs(
        &self,
        requested_outputs: &OutputRegions,
        inputs: &Descriptors,
        params: &Value,
    ) -> Result<InputRegions>;

    fn validate_postconditions(
        &self,
        outputs: &Resources,
        params: &Value,
    ) -> Result<Vec<AssertionResult>>;
}
```

These functions must be deterministic and cheap. They are part of the operation semantic implementation and tested independently.

Examples:

| Operation | Required input region |
|---|---|
| pointwise | same output region |
| blur radius `r` | output expanded by `r` |
| affine warp | inverse-transformed output footprint + reconstruction halo |
| composite | target region; source only where mask may be nonzero |
| SDF conversion | potentially whole connected domain; exact implementation may force full image |
| Poisson | masked connected component plus boundary |
| histogram | whole demanded input unless tiled reduction is possible |

---

## 19. Error model

Errors have stable machine codes and structured context.

```json
{
  "ok": false,
  "error": {
    "class": "semantic",
    "code": "E_COLOR_ENCODING_MISMATCH",
    "message": "filter.gaussian_blur@1 requires a linear image",
    "node": "blurred",
    "path": "/nodes/7/in/image",
    "actual": "srgb",
    "expected": "linear-*",
    "suggestions": [
      {
        "op": "color.convert@1",
        "params": {"to": "linear-srgb"}
      }
    ]
  }
}
```

Error classes:

- `parse`;
- `schema`;
- `reference`;
- `type`;
- `semantic`;
- `policy`;
- `execution`;
- `assertion`;
- `conformance`;
- `model`;
- `asset`;
- `export`.

Suggestions are non-authoritative data, not auto-applied edits.

---

## 20. Example: complete v1 touch-up plan

This is the **non-SDF MVP variant** of the first conformance scenario (see
[`../M0_DECISIONS.md`](../M0_DECISIONS.md) D1/D2): the authorized mask is a
soft-edged ellipse (analytic feather), and `composite.masked_replace@1` is the
single authorization boundary that `assert.no_change_outside_mask@1` checks.

```json
{
  "paintop": "1.0",
  "name": "localized-warm-shadow-repair",
  "policy": {
    "resources": {
      "max_nodes": 64,
      "max_pixels_per_resource": 33554432,
      "max_splats": 256
    },
    "execution": {
      "deadline_ms": 10000,
      "allowed_backends": ["cpu-reference", "cpu-optimized", "wgpu"]
    }
  },
  "inputs": {
    "source": {
      "kind": "image.file",
      "path": "input.png",
      "decode": {
        "desired_format": "rgba8",
        "color": "srgb",
        "alpha": "straight"
      }
    }
  },
  "nodes": [
    {
      "id": "linear",
      "op": "color.convert@1",
      "in": {"image": "input:source"},
      "params": {"to": "linear-srgb"}
    },
    {
      "id": "base",
      "op": "alpha.premultiply@1",
      "in": {"image": "node:linear/image"}
    },
    {
      "id": "allowed",
      "op": "mask.ellipse@1",
      "params": {
        "extent_from": "input:source",
        "center_px": [512.0, 406.0],
        "radii_px": [92.0, 36.0],
        "angle_rad": -0.16,
        "antialias": "analytic",
        "edge": {
          "profile": "smoothstep",
          "half_width_px": 8.0
        }
      }
    },
    {
      "id": "splats",
      "op": "paint.gaussian_splats@1",
      "in": {
        "base": "node:base/image"
      },
      "params": {
        "space": "linear-srgb",
        "splats": [
          {
            "center_px": [486.0, 401.0],
            "sigma_px": [42.0, 9.0],
            "angle_rad": -0.18,
            "color": [0.16, 0.08, 0.035, 1.0],
            "opacity": 0.12,
            "blend": "multiply"
          },
          {
            "center_px": [538.0, 414.0],
            "sigma_px": [50.0, 12.0],
            "angle_rad": -0.10,
            "color": [0.20, 0.10, 0.04, 1.0],
            "opacity": 0.08,
            "blend": "multiply"
          }
        ]
      }
    },
    {
      "id": "graded",
      "op": "color.adjust@1",
      "in": {
        "image": "node:splats/image"
      },
      "params": {
        "exposure_ev": -0.04,
        "saturation": -0.03,
        "temperature": 0.02
      }
    },
    {
      "id": "composited",
      "op": "composite.masked_replace@1",
      "in": {
        "edited": "node:graded/image",
        "base": "node:base/image",
        "mask": "node:allowed/mask"
      }
    },
    {
      "id": "straight",
      "op": "alpha.unpremultiply@1",
      "in": {"image": "node:composited/image"}
    },
    {
      "id": "encoded",
      "op": "color.convert@1",
      "in": {"image": "node:straight/image"},
      "params": {"to": "srgb"}
    }
  ],
  "assertions": [
    {
      "id": "localized",
      "op": "assert.no_change_outside_mask@1",
      "in": {
        "before": "input:source",
        "after": "node:encoded/image",
        "allowed": "node:allowed/mask"
      },
      "params": {
        "comparison_space": "decoded-linear",
        "outside_threshold": 0.000001,
        "coverage_epsilon": 0.0001
      }
    },
    {
      "id": "finite",
      "op": "assert.finite@1",
      "in": {"resource": "node:composited/image"}
    }
  ],
  "exports": {
    "final": {
      "resource": "node:encoded/image",
      "kind": "image",
      "path": "out.png",
      "encoding": {"format": "png"}
    }
  },
  "evidence": {
    "trace": "detailed",
    "graph": ["dot", "svg"],
    "contact_sheet": true,
    "materialize": [
      "node:allowed/mask",
      "node:splats/image",
      "node:encoded/image"
    ],
    "diffs": [
      {
        "before": "input:source",
        "after": "node:encoded/image",
        "heatmap": true
      }
    ]
  }
}
```

---

## 21. Example: material projection plan shape

This is a later extension, shown to ensure the 2D IR grows cleanly.

```json
{
  "paintop": "1.0",
  "inputs": {
    "asset": {"kind": "gltf.file", "path": "character.glb"},
    "screen_mask": {"kind": "mask.file", "path": "jacket-screen-mask.png"},
    "camera": {"kind": "json.file", "path": "front-camera.json"}
  },
  "nodes": [
    {
      "id": "surface",
      "op": "surface.project_screen_mask@1",
      "in": {
        "asset": "input:asset",
        "mask": "input:screen_mask",
        "camera": "input:camera"
      },
      "params": {
        "material_filter": ["Jacket"],
        "visibility": "frontmost",
        "backfaces": "reject"
      }
    },
    {
      "id": "uv_mask",
      "op": "surface.rasterize_uv_coverage@1",
      "in": {"surface": "node:surface/surface_map"},
      "params": {
        "channel": "baseColor",
        "footprint": "jacobian-ewa",
        "overlap_policy": "report-and-reject"
      }
    },
    {
      "id": "edited",
      "op": "color.adjust@1",
      "in": {
        "image": "node:surface/base_color_texture",
        "mask": "node:uv_mask/mask"
      },
      "params": {
        "exposure_ev": -0.3,
        "saturation": -0.1
      }
    },
    {
      "id": "asset_out",
      "op": "material.replace_texture@1",
      "in": {
        "asset": "input:asset",
        "texture": "node:edited/image"
      },
      "params": {
        "material": "Jacket",
        "slot": "baseColor"
      }
    }
  ],
  "assertions": [
    {
      "id": "asset_scope",
      "op": "assert.asset_mutation_allowlist@1",
      "in": {
        "before": "input:asset",
        "after": "node:asset_out/asset"
      },
      "params": {
        "allowed": ["materials[Jacket].baseColorTexture.image"]
      }
    }
  ],
  "exports": {
    "asset": {
      "resource": "node:asset_out/asset",
      "kind": "gltf",
      "path": "character-edited.glb"
    }
  }
}
```

---

## 22. Lua frontend contract, later

Lua is permitted only as a graph-construction frontend:

```lua
return paintop.plan {
  inputs = {
    source = paintop.image("input.png")
  },
  nodes = {
    paintop.mask.ellipse {
      id = "region",
      center_px = {512, 406},
      radii_px = {92, 36}
    }
  }
}
```

The Lua process returns serializable plan data, which is then validated exactly like JSON. Restrictions:

- no direct image buffer access;
- no hidden execution during graph construction;
- deterministic standard library subset;
- bounded instructions and memory;
- no network;
- filesystem access limited to approved module roots;
- normalized JSON emitted and retained in evidence.

A Lua plan that cannot emit canonical JSON is invalid.

---

## 23. Compatibility policy

- Plan major version changes only for structural language breaks.
- Operation semantic versions change independently.
- The runtime may support reading several older plan versions by normalizing them to the current internal graph.
- Normalization must be deterministic and covered by golden fixtures.
- Deprecated operations remain readable for a documented window but may be unavailable under strict policy.
- A plan may pin exact operation semantic versions and model hashes for archival replay.
- Implementation choice remains a scheduler decision unless policy pins it for diagnostics.

---

## 24. Questions to settle during M0/M1

1. Use a single `in` object or explicit top-level named input fields per operation?
2. Should output references require `/port` when there is exactly one output?
3. Should masks and images share an underlying generic tensor type internally while remaining distinct in IR?
4. How should reusable subgraphs/templates bind parameters without introducing general code execution?
5. Is JSON Schema alone sufficient for agent discovery, or should the CLI emit a compact operation grammar optimized for LLM context?
6. How should large literal data such as splat batches be referenced—inline JSON, binary sidecar, or content-addressed blob?
7. Should evidence requests live in the plan’s semantic hash? They change execution cost, not exported resource semantics.
8. Which metadata survives canonicalization but remains excluded from semantic identity?

Resolve these by writing real plans and measuring agent error rates, not aesthetic preference.
