//! Registration for the `mask` op domain.
//!
//! Owns the `mask.*` operations, which span several op modules (mask,
//! `mask_algebra`, `mask_bounds`, `mask_polygon`, `ellipse`). Adding a `mask`
//! op edits only this file plus the op's own module.

use paintop_core::executor::ImplRegistry;
use paintop_ir::{Error, OperationRegistry};

use crate::ellipse::{self, EllipseMask};
use crate::mask::{self, EmptyMask, FullMask, RectMask};
use crate::mask_algebra::{self, BinaryMaskOp, InvertMask};
use crate::mask_bounds::{self, MaskBounds};
use crate::mask_macros::{self, MaskMacro};
use crate::mask_polygon::{self, PolygonMask};
use crate::mask_to_sdf::MaskToSdf;
use crate::mask_topology::{self, ConnectedComponents, FillHoles, RemoveComponents};

/// Register every `mask.*` manifest and implementation, in fixed declaration
/// order.
///
/// # Errors
/// Propagates a schema error from an invalid manifest or a duplicate-id error.
pub(crate) fn register(reg: &mut OperationRegistry, impls: &mut ImplRegistry) -> Result<(), Error> {
    reg.register(MaskBounds::manifest()?)?;
    impls.register(
        mask_bounds::BOUNDS_OP_ID.parse()?,
        Box::new(MaskBounds::new()),
    )?;

    reg.register(EllipseMask::manifest()?)?;
    impls.register(
        ellipse::ELLIPSE_OP_ID.parse()?,
        Box::new(EllipseMask::new()),
    )?;

    reg.register(EmptyMask::manifest()?)?;
    impls.register(mask::EMPTY_OP_ID.parse()?, Box::new(EmptyMask::new()))?;

    reg.register(FullMask::manifest()?)?;
    impls.register(mask::FULL_OP_ID.parse()?, Box::new(FullMask::new()))?;

    reg.register(BinaryMaskOp::intersect_manifest()?)?;
    impls.register(
        mask_algebra::INTERSECT_OP_ID.parse()?,
        Box::new(BinaryMaskOp::intersect()),
    )?;

    reg.register(InvertMask::manifest()?)?;
    impls.register(
        mask_algebra::INVERT_OP_ID.parse()?,
        Box::new(InvertMask::new()),
    )?;

    reg.register(PolygonMask::manifest()?)?;
    impls.register(
        mask_polygon::POLYGON_OP_ID.parse()?,
        Box::new(PolygonMask::new()),
    )?;

    reg.register(RectMask::manifest()?)?;
    impls.register(mask::RECT_OP_ID.parse()?, Box::new(RectMask::new()))?;

    reg.register(BinaryMaskOp::subtract_manifest()?)?;
    impls.register(
        mask_algebra::SUBTRACT_OP_ID.parse()?,
        Box::new(BinaryMaskOp::subtract()),
    )?;

    reg.register(MaskToSdf::manifest()?)?;
    impls.register(
        crate::mask_to_sdf::OP_ID.parse()?,
        Box::new(MaskToSdf::new()),
    )?;

    reg.register(ConnectedComponents::manifest()?)?;
    impls.register(
        mask_topology::CONNECTED_COMPONENTS_OP_ID.parse()?,
        Box::new(ConnectedComponents::new()),
    )?;

    reg.register(FillHoles::manifest()?)?;
    impls.register(
        mask_topology::FILL_HOLES_OP_ID.parse()?,
        Box::new(FillHoles::new()),
    )?;

    reg.register(RemoveComponents::manifest()?)?;
    impls.register(
        mask_topology::REMOVE_COMPONENTS_OP_ID.parse()?,
        Box::new(RemoveComponents::new()),
    )?;

    reg.register(MaskMacro::feather_manifest()?)?;
    impls.register(
        mask_macros::FEATHER_OP_ID.parse()?,
        Box::new(MaskMacro::feather()),
    )?;

    reg.register(MaskMacro::grow_manifest()?)?;
    impls.register(
        mask_macros::GROW_OP_ID.parse()?,
        Box::new(MaskMacro::grow()),
    )?;

    reg.register(MaskMacro::shrink_manifest()?)?;
    impls.register(
        mask_macros::SHRINK_OP_ID.parse()?,
        Box::new(MaskMacro::shrink()),
    )?;

    reg.register(BinaryMaskOp::union_manifest()?)?;
    impls.register(
        mask_algebra::UNION_OP_ID.parse()?,
        Box::new(BinaryMaskOp::union()),
    )?;

    Ok(())
}
