# paintop planning bundle

A research-backed implementation blueprint for an agent-verifiable visual transformation runtime in Rust, with a typed JSON graph and a future surface-aware material-editing layer integrated with [`msh`](https://github.com/bobisme/msh).

## Documents

- [`plan.md`](plan.md) — architecture, boundaries, execution model, operation roadmap, `msh` integration, milestones, and first pull requests.
- [`AGENTS.md`](AGENTS.md) — operational rules for coding agents working on the repository.
- [`docs/IR_SPEC.md`](docs/IR_SPEC.md) — canonical JSON graph, typed resources, policies, assertions, candidate sets, and examples.
- [`docs/AGENT_VERIFICATION.md`](docs/AGENT_VERIFICATION.md) — analytic/property/metamorphic/differential testing, evidence bundles, CI, fuzzing, and autonomous completion rules.
- [`docs/RESEARCH.md`](docs/RESEARCH.md) — primary-source survey and assumption audit across execution engines, CV, restoration, materials, rendering, and metrics.
- [`docs/ALIEN_OPS.md`](docs/ALIEN_OPS.md) — advanced and proposed algorithms: SDF mask calculus, adaptive splats, seam-graph diffusion, micro-optimization, multi-view projection, provenance, and more.
- [`docs/OP_CATALOG.md`](docs/OP_CATALOG.md) — prioritized typed operation backlog from P0 substrate through material and research operators.

## Integrity

- [`MANIFEST.json`](MANIFEST.json) records file sizes, line counts, and SHA-256 hashes.
- [`SHA256SUMS`](SHA256SUMS) provides standard checksum output.

## Recommended starting point

Implement milestones M0 and M1 from `plan.md`: strict contracts, CPU reference semantics, masks, linear-light premultiplied compositing, Gaussian splats, convolution, assertions, and evidence bundles. Delay GPU, Lua, models, and GLB mutation until the agent verification loop works end to end.
