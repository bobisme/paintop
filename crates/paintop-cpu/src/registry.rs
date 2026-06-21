//! The MVP operation registry: every M0 op's manifest and `cpu.reference`
//! implementation, assembled in one place so the executor and the CLI dispatch
//! the real op set rather than stubs.
//!
//! M0 ships fourteen operations (`M0_DECISIONS` D2). Each lives in its own module
//! (`crate::io`, `crate::color`, …) and exposes a manifest builder plus a
//! zero-sized [`OpImplementation`] kernel. This module is the single seam that
//! collects them into the two registries the runtime needs:
//!
//! * an [`OperationRegistry`] of manifests, which `resolve_plan` /
//!   `check_graph` type-check against, and
//! * an [`ImplRegistry`] of compute kernels, which the executor dispatches.
//!
//! Both are built from the same source list so a manifest can never drift from
//! its implementation: adding an op here wires it into both at once.

use paintop_core::executor::{ImplRegistry, OpImplementation};
use paintop_ir::{Error, OperationManifest, OperationRegistry};

use crate::{
    adjust::Adjust,
    alpha::{Premultiply, Unpremultiply},
    assert::{Finite, NoChangeOutsideMask},
    color::Convert,
    composite::MaskedReplace,
    diff::Diff,
    ellipse::EllipseMask,
    inspect::Inspect,
    io::{DecodeImage, EncodeImage},
    materialize::Materialize,
    splat::GaussianSplats,
};

/// Build the manifest list for every MVP operation, in a stable declaration
/// order.
///
/// # Errors
/// Propagates the first op's [`schema`](paintop_ir::ErrorClass::Schema) error if
/// a hard-coded manifest is somehow invalid (it is not).
fn manifests() -> Result<Vec<OperationManifest>, Error> {
    Ok(vec![
        DecodeImage::manifest()?,
        EncodeImage::manifest()?,
        Inspect::manifest()?,
        Convert::manifest()?,
        Premultiply::manifest()?,
        Unpremultiply::manifest()?,
        EllipseMask::manifest()?,
        GaussianSplats::manifest()?,
        Adjust::manifest()?,
        MaskedReplace::manifest()?,
        Diff::manifest()?,
        NoChangeOutsideMask::manifest()?,
        Finite::manifest()?,
        Materialize::manifest()?,
    ])
}

/// The manifest [`OperationRegistry`] for the whole MVP op set.
///
/// This is the registry a plan resolves and type-checks against; it is the
/// authority on each op's declared ports, params, and `cpu.reference`
/// implementation id.
///
/// # Errors
/// Propagates a [`schema`](paintop_ir::ErrorClass::Schema) error if a manifest is
/// invalid, or a duplicate-registration error if two manifests share an id
/// (neither occurs for the fixed MVP set).
pub fn operation_registry() -> Result<OperationRegistry, Error> {
    OperationRegistry::from_manifests(manifests()?)
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
    let mut registry = ImplRegistry::new();
    let entries: Vec<(&str, Box<dyn OpImplementation>)> = vec![
        (crate::io::DECODE_OP_ID, Box::new(DecodeImage::new())),
        (crate::io::ENCODE_OP_ID, Box::new(EncodeImage::new())),
        (crate::inspect::INSPECT_OP_ID, Box::new(Inspect::new())),
        (crate::color::CONVERT_OP_ID, Box::new(Convert::new())),
        (
            crate::alpha::PREMULTIPLY_OP_ID,
            Box::new(Premultiply::new()),
        ),
        (
            crate::alpha::UNPREMULTIPLY_OP_ID,
            Box::new(Unpremultiply::new()),
        ),
        (crate::ellipse::ELLIPSE_OP_ID, Box::new(EllipseMask::new())),
        (crate::splat::SPLAT_OP_ID, Box::new(GaussianSplats::new())),
        (crate::adjust::ADJUST_OP_ID, Box::new(Adjust::new())),
        (
            crate::composite::MASKED_REPLACE_OP_ID,
            Box::new(MaskedReplace::new()),
        ),
        (crate::diff::DIFF_OP_ID, Box::new(Diff::new())),
        (
            crate::assert::NO_CHANGE_OUTSIDE_MASK_OP_ID,
            Box::new(NoChangeOutsideMask::new()),
        ),
        (crate::assert::FINITE_OP_ID, Box::new(Finite::new())),
        (
            crate::materialize::MATERIALIZE_OP_ID,
            Box::new(Materialize::new()),
        ),
    ];
    for (id, implementation) in entries {
        registry.register(id.parse()?, implementation)?;
    }
    Ok(registry)
}

#[cfg(test)]
mod tests {
    use super::{implementation_registry, manifests, operation_registry};

    #[test]
    fn registers_every_mvp_op() {
        // The fourteen MVP operations (`M0_DECISIONS` D2).
        assert_eq!(manifests().expect("manifests").len(), 14);
        assert_eq!(operation_registry().expect("op registry").len(), 14);
        assert_eq!(implementation_registry().expect("impls").len(), 14);
    }

    #[test]
    fn every_manifest_has_a_matching_implementation() {
        let ops = operation_registry().expect("op registry");
        let impls = implementation_registry().expect("impls");
        for manifest in ops.iter() {
            assert!(
                impls.contains(&manifest.id),
                "no implementation registered for {}",
                manifest.id
            );
        }
    }
}
