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
    blend::Blend,
    bounds_assert::{AssertAlphaValid, AssertChangedBounds, AssertRange, ChangedBounds},
    canvas::CreateImage,
    channel::{AssembleChannels, ExtractChannel},
    color::Convert,
    composite::MaskedReplace,
    composite_over::Over,
    convolve::Convolve,
    crop::Crop,
    diff::Diff,
    ellipse::EllipseMask,
    fill::Fill,
    flip::Flip,
    gaussian_blur::GaussianBlur,
    gradient::{LinearGradient, RadialGradient},
    inspect::Inspect,
    io::{DecodeImage, EncodeImage},
    mask::{EmptyMask, FullMask, RectMask},
    mask_algebra::{BinaryMaskOp, InvertMask},
    mask_bounds::MaskBounds,
    mask_polygon::PolygonMask,
    materialize::Materialize,
    pad::Pad,
    resize::Resize,
    rotate::Rotate90,
    splat::GaussianSplats,
    statistics::{Histogram, Statistics},
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
        ExtractChannel::manifest()?,
        AssembleChannels::manifest()?,
        CreateImage::manifest()?,
        EmptyMask::manifest()?,
        FullMask::manifest()?,
        RectMask::manifest()?,
        Fill::manifest()?,
        LinearGradient::manifest()?,
        RadialGradient::manifest()?,
        Over::manifest()?,
        Blend::manifest()?,
        InvertMask::manifest()?,
        BinaryMaskOp::union_manifest()?,
        BinaryMaskOp::intersect_manifest()?,
        BinaryMaskOp::subtract_manifest()?,
        MaskBounds::manifest()?,
        PolygonMask::manifest()?,
        Crop::manifest()?,
        Pad::manifest()?,
        Flip::manifest()?,
        Rotate90::manifest()?,
        Resize::manifest()?,
        Convolve::manifest()?,
        GaussianBlur::manifest()?,
        Statistics::manifest()?,
        Histogram::manifest()?,
        ChangedBounds::manifest()?,
        AssertRange::manifest()?,
        AssertAlphaValid::manifest()?,
        AssertChangedBounds::manifest()?,
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
#[allow(
    clippy::too_many_lines,
    reason = "a flat one-entry-per-op registration table; splitting it would obscure the 1:1 \
              op-to-kernel mapping"
)]
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
        (
            crate::channel::EXTRACT_OP_ID,
            Box::new(ExtractChannel::new()),
        ),
        (
            crate::channel::ASSEMBLE_OP_ID,
            Box::new(AssembleChannels::new()),
        ),
        (crate::canvas::CREATE_OP_ID, Box::new(CreateImage::new())),
        (crate::mask::EMPTY_OP_ID, Box::new(EmptyMask::new())),
        (crate::mask::FULL_OP_ID, Box::new(FullMask::new())),
        (crate::mask::RECT_OP_ID, Box::new(RectMask::new())),
        (crate::fill::FILL_OP_ID, Box::new(Fill::new())),
        (
            crate::gradient::LINEAR_GRADIENT_OP_ID,
            Box::new(LinearGradient::new()),
        ),
        (
            crate::gradient::RADIAL_GRADIENT_OP_ID,
            Box::new(RadialGradient::new()),
        ),
        (crate::composite_over::OVER_OP_ID, Box::new(Over::new())),
        (crate::blend::BLEND_OP_ID, Box::new(Blend::new())),
        (
            crate::mask_algebra::INVERT_OP_ID,
            Box::new(InvertMask::new()),
        ),
        (
            crate::mask_algebra::UNION_OP_ID,
            Box::new(BinaryMaskOp::union()),
        ),
        (
            crate::mask_algebra::INTERSECT_OP_ID,
            Box::new(BinaryMaskOp::intersect()),
        ),
        (
            crate::mask_algebra::SUBTRACT_OP_ID,
            Box::new(BinaryMaskOp::subtract()),
        ),
        (
            crate::mask_bounds::BOUNDS_OP_ID,
            Box::new(MaskBounds::new()),
        ),
        (
            crate::mask_polygon::POLYGON_OP_ID,
            Box::new(PolygonMask::new()),
        ),
        (crate::crop::CROP_OP_ID, Box::new(Crop::new())),
        (crate::pad::PAD_OP_ID, Box::new(Pad::new())),
        (crate::flip::FLIP_OP_ID, Box::new(Flip::new())),
        (crate::rotate::ROTATE90_OP_ID, Box::new(Rotate90::new())),
        (crate::resize::RESIZE_OP_ID, Box::new(Resize::new())),
        (crate::convolve::CONVOLVE_OP_ID, Box::new(Convolve::new())),
        (
            crate::gaussian_blur::GAUSSIAN_BLUR_OP_ID,
            Box::new(GaussianBlur::new()),
        ),
        (
            crate::statistics::STATISTICS_OP_ID,
            Box::new(Statistics::new()),
        ),
        (
            crate::statistics::HISTOGRAM_OP_ID,
            Box::new(Histogram::new()),
        ),
        (
            crate::bounds_assert::CHANGED_BOUNDS_OP_ID,
            Box::new(ChangedBounds::new()),
        ),
        (
            crate::bounds_assert::RANGE_OP_ID,
            Box::new(AssertRange::new()),
        ),
        (
            crate::bounds_assert::ALPHA_VALID_OP_ID,
            Box::new(AssertAlphaValid::new()),
        ),
        (
            crate::bounds_assert::ASSERT_CHANGED_BOUNDS_OP_ID,
            Box::new(AssertChangedBounds::new()),
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
        for manifest in manifests().expect("manifests") {
            let id = manifest.id.to_string();
            if only != "all" && !only.split(',').any(|t| t == id) {
                continue;
            }
            let path = root.join(format!("{id}.json"));
            let json = serde_json::to_string_pretty(&manifest).expect("serialize");
            std::fs::write(&path, format!("{json}\n")).expect("write manifest");
        }
    }

    #[test]
    fn registers_every_mvp_op() {
        // The fourteen M0 ops (`M0_DECISIONS` D2) plus the M1 P0 additions wired in
        // this workspace.
        const REGISTERED_OPS: usize = 44;
        assert_eq!(manifests().expect("manifests").len(), REGISTERED_OPS);
        assert_eq!(
            operation_registry().expect("op registry").len(),
            REGISTERED_OPS
        );
        assert_eq!(
            implementation_registry().expect("impls").len(),
            REGISTERED_OPS
        );
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
