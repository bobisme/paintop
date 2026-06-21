//! Registration for the `assert` op domain.
//!
//! Owns the `assert.*` operations, which span the `assert` and `bounds_assert`
//! op modules. Adding an `assert` op edits only this file plus the op's own
//! module.

use paintop_core::executor::ImplRegistry;
use paintop_ir::{Error, OperationRegistry};

use crate::assert::{self, Finite, NoChangeOutsideMask};
use crate::bounds_assert::{self, AssertAlphaValid, AssertChangedBounds, AssertRange};

/// Register every `assert.*` manifest and implementation, in fixed declaration
/// order.
///
/// # Errors
/// Propagates a schema error from an invalid manifest or a duplicate-id error.
pub(crate) fn register(reg: &mut OperationRegistry, impls: &mut ImplRegistry) -> Result<(), Error> {
    reg.register(AssertAlphaValid::manifest()?)?;
    impls.register(
        bounds_assert::ALPHA_VALID_OP_ID.parse()?,
        Box::new(AssertAlphaValid::new()),
    )?;

    reg.register(AssertChangedBounds::manifest()?)?;
    impls.register(
        bounds_assert::ASSERT_CHANGED_BOUNDS_OP_ID.parse()?,
        Box::new(AssertChangedBounds::new()),
    )?;

    reg.register(Finite::manifest()?)?;
    impls.register(assert::FINITE_OP_ID.parse()?, Box::new(Finite::new()))?;

    reg.register(NoChangeOutsideMask::manifest()?)?;
    impls.register(
        assert::NO_CHANGE_OUTSIDE_MASK_OP_ID.parse()?,
        Box::new(NoChangeOutsideMask::new()),
    )?;

    reg.register(AssertRange::manifest()?)?;
    impls.register(
        bounds_assert::RANGE_OP_ID.parse()?,
        Box::new(AssertRange::new()),
    )?;

    Ok(())
}
