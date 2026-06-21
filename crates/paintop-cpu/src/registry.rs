//! The MVP operation registry: assembles every op's manifest and
//! `cpu.reference` implementation so the executor and the CLI dispatch the real
//! op set rather than stubs.
//!
//! Registration is **partitioned by op domain** (`io`, `color`, `alpha`,
//! `image`, `mask`, `paint`, `composite`, `filter`, `analyze`, `assert`,
//! `debug`). Each domain owns one `register` fn under [`crate::domains`] that
//! adds its ops' manifests AND impls; this module just iterates those domain
//! functions in a fixed order (`crate::domains::REGISTERS`). Adding an op to
//! an existing domain edits only that domain's module — never this file — so
//! parallel op work no longer serializes on a single seam.
//!
//! Two registries are produced from the same domain functions, so a manifest
//! can never drift from its implementation:
//!
//! * an [`OperationRegistry`] of manifests, which `resolve_plan` /
//!   `check_graph` type-check against, and
//! * an [`ImplRegistry`] of compute kernels, which the executor dispatches.
//!
//! Both registries are [`BTreeMap`](std::collections::BTreeMap)-backed, so they
//! iterate in canonical op-id order independent of registration order; the
//! determinism test below pins that id sequence so it can never silently move.

use paintop_core::executor::ImplRegistry;
use paintop_ir::{Error, OperationRegistry};

/// The manifest [`OperationRegistry`] for the whole MVP op set.
///
/// This is the registry a plan resolves and type-checks against; it is the
/// authority on each op's declared ports, params, and `cpu.reference`
/// implementation id. Built by running every domain's `register` fn in fixed
/// order.
///
/// # Errors
/// Propagates a [`schema`](paintop_ir::ErrorClass::Schema) error if a manifest is
/// invalid, or a duplicate-registration error if two manifests share an id
/// (neither occurs for the fixed MVP set).
pub fn operation_registry() -> Result<OperationRegistry, Error> {
    let (registry, _impls) = build()?;
    Ok(registry)
}

/// The executable [`ImplRegistry`] for the whole MVP op set.
///
/// Each entry is the op's `cpu.reference` compute kernel — the deterministic
/// oracle the M0 executor dispatches. Keyed by the same op ids as
/// [`operation_registry`], so a resolved node always finds its kernel.
///
/// # Errors
/// Propagates a [`schema`](paintop_ir::ErrorClass::Schema) error if an op id is
/// invalid or an [`execution`](paintop_ir::ErrorClass::Execution) error if an id
/// is registered twice (neither occurs for the fixed MVP set).
pub fn implementation_registry() -> Result<ImplRegistry, Error> {
    let (_registry, impls) = build()?;
    Ok(impls)
}

/// Build both registries together by running every domain's `register` fn in
/// the fixed [`crate::domains::REGISTERS`] order.
///
/// Building them in lockstep is what guarantees the manifest set and impl set
/// stay identical: each domain adds an op to *both* or neither.
///
/// # Errors
/// Propagates the first domain's registration error (invalid manifest or
/// duplicate id).
fn build() -> Result<(OperationRegistry, ImplRegistry), Error> {
    let mut registry = OperationRegistry::new();
    let mut impls = ImplRegistry::new();
    for register in crate::domains::REGISTERS {
        register(&mut registry, &mut impls)?;
    }
    Ok((registry, impls))
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use super::{build, implementation_registry, operation_registry};

    /// Collect the on-disk `ops/manifests/*.json` op ids (file stem == op id).
    fn on_disk_manifest_ids() -> BTreeSet<String> {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .expect("crates dir")
            .parent()
            .expect("repo root")
            .join("ops/manifests");
        let mut ids = BTreeSet::new();
        for entry in std::fs::read_dir(&dir).expect("read ops/manifests") {
            let path = entry.expect("dir entry").path();
            if path.extension().and_then(|e| e.to_str()) != Some("json") {
                continue;
            }
            let stem = path
                .file_stem()
                .and_then(|s| s.to_str())
                .expect("utf8 stem")
                .to_owned();
            ids.insert(stem);
        }
        ids
    }

    #[test]
    fn emit_manifests_when_requested() {
        // Dev helper: with PAINTOP_EMIT_MANIFESTS set, (re)write every op's
        // checked-in ops/manifests/<id>.json from the Rust builder (the source of
        // truth), then return. Off by default so a normal test run is read-only.
        if std::env::var_os("PAINTOP_EMIT_MANIFESTS").is_none() {
            return;
        }
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .expect("crates dir")
            .parent()
            .expect("repo root")
            .join("ops/manifests");
        let only = std::env::var("PAINTOP_EMIT_MANIFESTS").unwrap_or_default();
        let ops = operation_registry().expect("op registry");
        for manifest in ops.iter() {
            let id = manifest.id.to_string();
            if only != "all" && !only.split(',').any(|t| t == id) {
                continue;
            }
            let path = root.join(format!("{id}.json"));
            let json = serde_json::to_string_pretty(&manifest).expect("serialize");
            std::fs::write(&path, format!("{json}\n")).expect("write manifest");
        }
    }

    /// Completeness: the on-disk manifest set, the registered manifest set, and
    /// the registered impl set must be exactly equal.
    ///
    /// This replaces the old `len() == N` magic-number assertion. It fails if a
    /// domain's `register` fn was never wired into
    /// [`crate::domains::REGISTERS`] (its ops vanish from both registries while
    /// their `*.json` stays on disk), if an op is dropped from a manifest but
    /// not its impl (or vice-versa), or if a `*.json` is added with no code — no
    /// number to bump, ever.
    #[test]
    fn registered_set_matches_on_disk_manifests() {
        let ops = operation_registry().expect("op registry");
        let impls = implementation_registry().expect("impls");

        let manifest_ids: BTreeSet<String> = ops.iter().map(|m| m.id.to_string()).collect();
        let on_disk = on_disk_manifest_ids();

        // Every registered manifest has a registered impl, and vice-versa.
        for manifest in ops.iter() {
            assert!(
                impls.contains(&manifest.id),
                "manifest {} has no registered implementation",
                manifest.id
            );
        }
        assert_eq!(
            ops.len(),
            impls.len(),
            "manifest count {} != impl count {} (an impl has no manifest or vice-versa)",
            ops.len(),
            impls.len()
        );

        // The registered manifest set == the on-disk manifest set: catches a
        // domain that was never wired in (json on disk, nothing registered) and
        // a manifest registered with no checked-in json.
        assert_eq!(
            manifest_ids, on_disk,
            "registered manifest ids differ from ops/manifests/*.json on disk"
        );
    }

    /// Determinism: the registry's iteration order (canonical op-id order) must
    /// stay byte-identical to the order captured from `main` before the
    /// by-domain refactor. If any op id moves, appears, or disappears, this
    /// fails — the runtime must see the exact same op sequence.
    #[test]
    fn registry_order_is_byte_identical_to_baseline() {
        const BASELINE: &[&str] = &[
            "alpha.premultiply@1",
            "alpha.unpremultiply@1",
            "analyze.changed_bounds@1",
            "analyze.diff@1",
            "analyze.histogram@1",
            "analyze.statistics@1",
            "assert.alpha_valid@1",
            "assert.changed_bounds@1",
            "assert.finite@1",
            "assert.no_change_outside_mask@1",
            "assert.range@1",
            "color.adjust@1",
            "color.convert@1",
            "composite.blend@1",
            "composite.masked_replace@1",
            "composite.over@1",
            "debug.materialize@1",
            "filter.convolve@1",
            "filter.gaussian_blur@1",
            "image.assemble_channels@1",
            "image.create@1",
            "image.crop@1",
            "image.extract_channel@1",
            "image.flip@1",
            "image.inspect@1",
            "image.pad@1",
            "image.resize@1",
            "image.rotate90@1",
            "io.decode_image@1",
            "io.encode_image@1",
            "mask.bounds@1",
            "mask.connected_components@1",
            "mask.ellipse@1",
            "mask.empty@1",
            "mask.feather@1",
            "mask.fill_holes@1",
            "mask.full@1",
            "mask.grow@1",
            "mask.intersect@1",
            "mask.invert@1",
            "mask.polygon@1",
            "mask.rect@1",
            "mask.remove_components@1",
            "mask.shrink@1",
            "mask.subtract@1",
            "mask.to_sdf@1",
            "mask.union@1",
            "paint.fill@1",
            "paint.gaussian_splats@1",
            "paint.linear_gradient@1",
            "paint.radial_gradient@1",
            "sdf.intersect@1",
            "sdf.offset@1",
            "sdf.subtract@1",
            "sdf.to_mask@1",
            "sdf.union@1",
        ];
        let ops = operation_registry().expect("op registry");
        let ids: Vec<String> = ops.iter().map(|m| m.id.to_string()).collect();
        assert_eq!(
            ids.as_slice(),
            BASELINE
                .iter()
                .map(|s| (*s).to_owned())
                .collect::<Vec<_>>()
                .as_slice(),
            "registry op-id order moved relative to the captured baseline"
        );
    }

    /// Both registries are built from the same domain functions in lockstep, so
    /// they expose the same op-id set.
    #[test]
    fn manifest_and_impl_sets_agree() {
        let (registry, impls) = build().expect("build registries");
        assert_eq!(registry.len(), impls.len());
        for manifest in registry.iter() {
            assert!(impls.contains(&manifest.id), "{} missing impl", manifest.id);
        }
    }
}
