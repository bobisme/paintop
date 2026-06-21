//! Registration for the `debug` op domain.
//!
//! Owns the `debug.*` operations (materialize). Adding a `debug` op edits only
//! this file plus the op's own module.

use paintop_core::executor::ImplRegistry;
use paintop_ir::{Error, OperationRegistry};

use crate::materialize::{self, Materialize};

/// Register every `debug.*` manifest and implementation, in fixed declaration
/// order.
///
/// # Errors
/// Propagates a schema error from an invalid manifest or a duplicate-id error.
pub(crate) fn register(reg: &mut OperationRegistry, impls: &mut ImplRegistry) -> Result<(), Error> {
    reg.register(Materialize::manifest()?)?;
    impls.register(
        materialize::MATERIALIZE_OP_ID.parse()?,
        Box::new(Materialize::new()),
    )?;

    Ok(())
}
