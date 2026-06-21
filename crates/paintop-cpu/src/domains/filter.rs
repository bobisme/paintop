//! Registration for the `filter` op domain.
//!
//! Owns the `filter.*` operations (convolve, gaussian blur). Adding a `filter`
//! op edits only this file plus the op's own module.

use paintop_core::executor::ImplRegistry;
use paintop_ir::{Error, OperationRegistry};

use crate::convolve::{self, Convolve};
use crate::gaussian_blur::{self, GaussianBlur};

/// Register every `filter.*` manifest and implementation, in fixed declaration
/// order.
///
/// # Errors
/// Propagates a schema error from an invalid manifest or a duplicate-id error.
pub(crate) fn register(reg: &mut OperationRegistry, impls: &mut ImplRegistry) -> Result<(), Error> {
    reg.register(Convolve::manifest()?)?;
    impls.register(convolve::CONVOLVE_OP_ID.parse()?, Box::new(Convolve::new()))?;

    reg.register(GaussianBlur::manifest()?)?;
    impls.register(
        gaussian_blur::GAUSSIAN_BLUR_OP_ID.parse()?,
        Box::new(GaussianBlur::new()),
    )?;

    Ok(())
}
