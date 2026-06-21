//! Registration for the `alpha` op domain.
//!
//! Owns the `alpha.*` operations. Adding an `alpha` op edits only this file
//! plus the op's own module.

use paintop_core::executor::ImplRegistry;
use paintop_ir::{Error, OperationRegistry};

use crate::alpha::{self, Premultiply, Unpremultiply};

/// Register every `alpha.*` manifest and implementation, in fixed declaration
/// order.
///
/// # Errors
/// Propagates a schema error from an invalid manifest or a duplicate-id error.
pub(crate) fn register(reg: &mut OperationRegistry, impls: &mut ImplRegistry) -> Result<(), Error> {
    reg.register(Premultiply::manifest()?)?;
    impls.register(
        alpha::PREMULTIPLY_OP_ID.parse()?,
        Box::new(Premultiply::new()),
    )?;

    reg.register(Unpremultiply::manifest()?)?;
    impls.register(
        alpha::UNPREMULTIPLY_OP_ID.parse()?,
        Box::new(Unpremultiply::new()),
    )?;

    Ok(())
}
