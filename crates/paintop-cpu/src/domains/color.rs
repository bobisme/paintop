//! Registration for the `color` op domain.
//!
//! Owns the `color.*` operations (`color.adjust`, `color.convert`). Adding a
//! `color` op edits only this file plus the op's own module.

use paintop_core::executor::ImplRegistry;
use paintop_ir::{Error, OperationRegistry};

use crate::adjust::{self, Adjust};
use crate::color::{self, Convert};

/// Register every `color.*` manifest and implementation, in fixed declaration
/// order.
///
/// # Errors
/// Propagates a schema error from an invalid manifest or a duplicate-id error.
pub(crate) fn register(reg: &mut OperationRegistry, impls: &mut ImplRegistry) -> Result<(), Error> {
    reg.register(Adjust::manifest()?)?;
    impls.register(adjust::ADJUST_OP_ID.parse()?, Box::new(Adjust::new()))?;

    reg.register(Convert::manifest()?)?;
    impls.register(color::CONVERT_OP_ID.parse()?, Box::new(Convert::new()))?;

    Ok(())
}
