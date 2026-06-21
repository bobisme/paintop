//! Registration for the `sdf` op domain.
//!
//! Owns the `sdf.*` signed-distance-field operations — reconstruction
//! (`sdf.to_mask`), offset/grow/shrink (`sdf.offset`), and the boolean algebra
//! (`sdf.union`/`sdf.intersect`/`sdf.subtract`) — all in [`crate::sdf_ops`].
//! Adding an `sdf` op edits only this file plus the op's own module.

use paintop_core::executor::ImplRegistry;
use paintop_ir::{Error, OperationRegistry};

use crate::sdf_algebra::{self, SdfBooleanOp};
use crate::sdf_ops::{self, SdfOffset, SdfToMask};

/// Register every `sdf.*` manifest and implementation, in canonical id order.
///
/// # Errors
/// Propagates a schema error from an invalid manifest or a duplicate-id error.
pub(crate) fn register(reg: &mut OperationRegistry, impls: &mut ImplRegistry) -> Result<(), Error> {
    reg.register(SdfOffset::manifest()?)?;
    impls.register(sdf_ops::OFFSET_OP_ID.parse()?, Box::new(SdfOffset::new()))?;

    reg.register(SdfToMask::manifest()?)?;
    impls.register(sdf_ops::TO_MASK_OP_ID.parse()?, Box::new(SdfToMask::new()))?;

    reg.register(SdfBooleanOp::union_manifest()?)?;
    impls.register(
        sdf_algebra::UNION_OP_ID.parse()?,
        Box::new(SdfBooleanOp::union()),
    )?;

    reg.register(SdfBooleanOp::intersect_manifest()?)?;
    impls.register(
        sdf_algebra::INTERSECT_OP_ID.parse()?,
        Box::new(SdfBooleanOp::intersect()),
    )?;

    reg.register(SdfBooleanOp::subtract_manifest()?)?;
    impls.register(
        sdf_algebra::SUBTRACT_OP_ID.parse()?,
        Box::new(SdfBooleanOp::subtract()),
    )?;

    Ok(())
}
