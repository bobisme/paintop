//! Registration for the `composite` op domain.
//!
//! Owns the `composite.*` operations (blend, masked replace, over). Adding a
//! `composite` op edits only this file plus the op's own module.

use paintop_core::executor::ImplRegistry;
use paintop_ir::{Error, OperationRegistry};

use crate::blend::{self, Blend};
use crate::composite::{self, MaskedReplace};
use crate::composite_over::{self, Over};

/// Register every `composite.*` manifest and implementation, in fixed
/// declaration order.
///
/// # Errors
/// Propagates a schema error from an invalid manifest or a duplicate-id error.
pub(crate) fn register(reg: &mut OperationRegistry, impls: &mut ImplRegistry) -> Result<(), Error> {
    reg.register(Blend::manifest()?)?;
    impls.register(blend::BLEND_OP_ID.parse()?, Box::new(Blend::new()))?;

    reg.register(MaskedReplace::manifest()?)?;
    impls.register(
        composite::MASKED_REPLACE_OP_ID.parse()?,
        Box::new(MaskedReplace::new()),
    )?;

    reg.register(Over::manifest()?)?;
    impls.register(composite_over::OVER_OP_ID.parse()?, Box::new(Over::new()))?;

    Ok(())
}
