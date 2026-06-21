//! Registration for the `paint` op domain.
//!
//! Owns the `paint.*` operations (fill, gradients, gaussian splats). Adding a
//! `paint` op edits only this file plus the op's own module.

use paintop_core::executor::ImplRegistry;
use paintop_ir::{Error, OperationRegistry};

use crate::fill::{self, Fill};
use crate::gradient::{self, LinearGradient, RadialGradient};
use crate::splat::{self, GaussianSplats};

/// Register every `paint.*` manifest and implementation, in fixed declaration
/// order.
///
/// # Errors
/// Propagates a schema error from an invalid manifest or a duplicate-id error.
pub(crate) fn register(reg: &mut OperationRegistry, impls: &mut ImplRegistry) -> Result<(), Error> {
    reg.register(Fill::manifest()?)?;
    impls.register(fill::FILL_OP_ID.parse()?, Box::new(Fill::new()))?;

    reg.register(GaussianSplats::manifest()?)?;
    impls.register(splat::SPLAT_OP_ID.parse()?, Box::new(GaussianSplats::new()))?;

    reg.register(LinearGradient::manifest()?)?;
    impls.register(
        gradient::LINEAR_GRADIENT_OP_ID.parse()?,
        Box::new(LinearGradient::new()),
    )?;

    reg.register(RadialGradient::manifest()?)?;
    impls.register(
        gradient::RADIAL_GRADIENT_OP_ID.parse()?,
        Box::new(RadialGradient::new()),
    )?;

    Ok(())
}
