# M0 decisions addendum

**Status:** resolved decomposition-prep decisions — binding defaults for M0 and the first vertical slice
**Date:** 2026-06-20
**Method:** grill-me interview against `plan.md`, `IR_SPEC.md`, `OP_CATALOG.md`, `AGENT_VERIFICATION.md`
**Purpose:** close the blocking decisions and one internal contradiction so M0 + the MVP slice can be broken into bones. Supersedes the deferred questions it names; does not override the rest of the plan.

Each decision is binding until implementation evidence disproves it (same convention as `plan.md §4`).

---

## D1 — SDF deferred; first slice feathers analytically

The general signed-distance mask chain (`mask.to_sdf` → `sdf.offset` → `sdf.to_mask`, plus the exact-EDT primitive) is **out of the first vertical slice.**

- The first end-to-end scenario feathers via an **analytic soft-edge parameter on `mask.ellipse@1`** (`edge: { profile: "smoothstep", half_width_px: N }`), evaluated from the ellipse's implicit quadratic. No distance-field infrastructure required to close the loop once.
- Exact-EDT + the full SDF mask calculus (`ALIEN_OPS §2`) is the **immediate next slice after the loop is green** — call it "M1.5", ahead of the rest of M4. The research (`RESEARCH §3.1`, `ALIEN_OPS §2.6`) is right that it is foundational and low-risk; it simply must not block the first green run.

**Contradiction this closes:** `plan.md §25` and the `IR_SPEC §20` example currently require the SDF chain to demonstrate the "viable foundation," yet `OP_CATALOG §4/§18` marks SDF P1 and absent from P0, and the milestone list parks it under M4.

**Required doc edits (do as part of M0):**
- Rewrite `plan.md §25` steps 3–4 to "create a soft-edged ellipse mask (analytic feather by a physical pixel radius)" — drop "convert mask to signed distance / feather."
- Rewrite the `IR_SPEC §20` example to remove the `sdf` / `allowed` (`mask.to_sdf` → `sdf.to_mask`) nodes and feather directly on `mask.ellipse`.
- Add a one-line note in `OP_CATALOG §4` that exact-EDT + SDF is the M1.5 priority (not M4) once the loop closes.

---

## D2 — MVP op set (14 ops) that closes the autonomous touch-up loop

The first slice ships exactly these operations. Each gets the full per-op treatment from `plan.md §24` + `AGENT_VERIFICATION §15`.

| Op | Role in the loop |
|---|---|
| `io.decode_image@1` | load the fixture |
| `image.inspect@1` | success criterion `§3.1.1`; agent authors the plan from extent/range/hash stats (exercises the `Report` type early) |
| `color.convert@1` | sRGB ↔ linear (both directions, one op) |
| `alpha.premultiply@1` | enter linear premultiplied working space |
| `alpha.unpremultiply@1` | leave it before encode |
| `mask.ellipse@1` | the selection **and** the analytic feather (per D1) |
| `paint.gaussian_splats@1` | the edit, painted onto an **edit layer** (not clipped in-place) |
| `color.adjust@1` | linear-light grade on the edit layer |
| `composite.masked_replace@1` | **the single authorization boundary** (per D2 sub-decision) |
| `analyze.diff@1` | diff/heatmap evidence artifact (`§25`) |
| `assert.no_change_outside_mask@1` | the core safety assertion |
| `assert.finite@1` | cheap finiteness guard (in the `§20` example) |
| `io.encode_image@1` | write the output PNG |
| `debug.materialize@1` | mask/intermediate evidence |

**Sub-decision — explicit compositing.** Authorization is made explicit: splats paint onto an edit layer, `color.adjust` modifies that layer, then `composite.masked_replace(edited, base, authorized_mask)` enforces locality in **one** auditable graph edge. The authorized mask appears exactly once — the cleanest possible target for `assert.no_change_outside_mask`. (The `IR_SPEC §20` graph instead spread masking across per-op `clip`/`mask` params; the rewrite in D1 should also adopt the explicit `masked_replace` boundary.)

**Sub-decision — `image.inspect` included** (not deferred): the agent needs it to author the first plan, and it gives the `Report` resource type early test coverage.

**Deferred from full P0 to "finish-P0" (after the loop is green), not dropped:** `mask.empty/full/rect/polygon` + boolean algebra (`invert/union/intersect/subtract`) + `mask.bounds`; `image.create/crop/pad/resize`; `paint.fill` + linear/radial gradients; `composite.over` + `composite.blend`; `filter.convolve` + `filter.gaussian_blur`; `color.matrix/levels`; `analyze.statistics/histogram/changed_bounds`; `assert.range/changed_bounds/alpha_valid`; `resource.hash/copy`. None are needed to close the loop once; all remain P0 for the full conformance set.

---

## D3 — IR open questions (resolves `IR_SPEC §24` Q1, Q2, Q6)

All three resolve strict-and-minimal; sugar is deferred, not built.

- **Q1 — single `in` object.** Node inputs stay under one `in: { … }` object (as every example already shows). Validation rule: *every key under `in` must be a declared input port.* Keeps inputs cleanly separated from `params`/`hints`/`extensions`; no name collisions.
- **Q2 — always require `/port`; no shorthand.** Every reference is `node:id/port`, even single-output nodes. No bare-`node:id` sugar and no normalization-sugar layer in v1 — future-proof against ops gaining a second output, and additive to introduce later without breaking canonical plans.
- **Q6 — inline JSON splats, additive seam.** Splat batches are inline in `params` (bounded by `policy.resources.max_splats`). The op is designed so a future large-batch path (a `splats` input-port backed by a sidecar/content-addressed blob) is **additive** — its content hash would feed the semantic hash identically. **No CAS/sidecar infrastructure is built now.**

The remaining `§24` questions (subgraph templates, single-tensor backing, e-graph, evidence-in-hash, metadata survival) stay open and are resolved later by writing real plans, per the doc.

---

## D4 — PR#1 bootstrap pins

Grounded against the local `../msh` checkout (wgpu 27, edition 2024, gltf 1.4.1, nalgebra 0.33, image 0.25, serde 1.0 + serde_json 1.0.145, clap 4.5; **no** `rust-toolchain.toml`, **no** MSRV, **no** error crate).

- **Toolchain / edition / MSRV.** Pin `rust-toolchain.toml` to a current **stable** channel with `rustfmt` + `clippy` components; **edition 2024** (match `msh`); **no MSRV policy** (internal tool, no downstream consumers — an MSRV gate is pure CI cost). `msh` pinning nothing is a gap, not a model to copy.
- **Hashing.** **BLAKE3** for all internal semantic-plan / content / cache hashing (`plan.md §10.3`); **SHA-256** only at external interop boundaries (fixture manifests `§4.2`, model-weight checksums, glTF integrity, `SHA256SUMS`). Serialized hashes carry an algorithm prefix (`blake3:…`, `sha256:…`).
- **Errors.** `thiserror` typed enums per crate; a **central error-taxonomy module in `paintop-ir`** owns the stable `class` + `code` strings and the exit codes from `IR_SPEC §19` / `plan.md §15.4`. `anyhow` permitted **only** in the `cli`/`xtask` binaries — never in library crates. The structured error is part of the agent-facing contract.
- **serde / canonicalization.** `serde` + `serde_json` with **`deny_unknown_fields` on every plan/manifest struct**; a **dedicated canonicalization + hashing module** does duplicate-key rejection at parse time and emits canonical bytes (lexicographically sorted keys, single round-trippable float format) per `IR_SPEC §17` rules 5/11/12. **Never hash raw `serde_json` output.**

- **Workspace lints (binding, root `Cargo.toml`).** The workspace declares, and every crate inherits via `[lints] workspace = true`:

  ```toml
  [workspace.lints.rust]
  unsafe_code = "forbid"

  [workspace.lints.clippy]
  pedantic = { level = "deny", priority = -1 }
  nursery  = { level = "deny", priority = -1 }
  unwrap_used = "deny"
  ```

  Implications baked into the bootstrap bone: (1) `unsafe_code = "forbid"` means no `unsafe` anywhere — any future SIMD/GPU need that requires it must justify a scoped, separately-reviewed exception, not a blanket relax; (2) `unwrap_used = "deny"` reinforces the `thiserror` error model — library code returns typed errors, never `.unwrap()`/`.expect()`; (3) add a `clippy.toml` with `allow-unwrap-in-tests = true` and `allow-expect-in-tests = true` so test code may still `unwrap`/`expect`; (4) `pedantic` + `nursery` at `deny` is intentional — targeted `#[allow(clippy::lint, reason = "…")]` with a written justification is the only escape hatch, and `clippy --all-targets -- -D warnings` must pass clean (`AGENT_VERIFICATION §2.1`).

**Forward constraint (not active in M0/M1):** M0/M1 ship **zero GPU**. When `paintop-wgpu` lands at M3, pin **`wgpu = "27"`** to match `msh`, and per `plan.md §12.3` treat any wgpu upgrade as an isolated, separately-tested change — never bundled with renderer extraction (`M6`).

**Shared-with-msh dep choices (for consistency):** `clap` 4.5 derive for the CLI, `image` 0.25 for codecs (behind enforced decode limits, `plan.md §17.2`), `nalgebra` 0.33 for small-matrix linear algebra (color matrices, transforms).

---

## What this unblocks

M0 (`plan.md §19`) + the D2 MVP slice can now be decomposed into bones (~15–20). Suggested grouping, dependency-ordered:

1. **Bootstrap** — workspace, `rust-toolchain.toml`, `justfile` (`just check`/`just install`), CI hygiene gates (`AGENT_VERIFICATION §2.1`), `xtask` skeleton. (`plan.md §20` PR 1)
2. **IR foundation** — resource metadata + coordinate/color enums; strict plan parser (deny-unknown-fields, duplicate-key + size rejection); op registry + manifest schema; canonical normalization + BLAKE3 hashing; error taxonomy. (PRs 2–5)
3. **Evidence + fixtures** — analytic fixture generator; evidence-bundle + structured-trace skeleton. (PRs 6–7)
4. **MVP ops** — the 14 ops of D2, each per `plan.md §24`, with `cargo xtask verify-op` support.
5. **Conformance** — the end-to-end "agent edits blemish" scenario (D1-rewritten, non-SDF), reproducible-hash rerun.

Leave M2+ at milestone granularity until the loop is proven (per `plan.md §20`: "After PR 24, stop and evaluate").

> **Workflow note:** the bones above are code work and follow the maw/edict flow (bone → workspace → `just check` → merge). This addendum is a planning-notes doc and was written directly to the `notes/` bundle.
