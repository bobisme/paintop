//! Registration for the `analyze` op domain.
//!
//! Owns the `analyze.*` operations, which span several op modules (diff,
//! statistics, bounds). Adding an `analyze` op edits only this file plus the
//! op's own module.

use paintop_core::executor::ImplRegistry;
use paintop_ir::{Error, OperationRegistry};

use crate::bounds_assert::{self, ChangedBounds};
use crate::diff::{self, Diff};
use crate::statistics::{self, Histogram, Statistics};

/// Register every `analyze.*` manifest and implementation, in fixed
/// declaration order.
///
/// # Errors
/// Propagates a schema error from an invalid manifest or a duplicate-id error.
pub(crate) fn register(reg: &mut OperationRegistry, impls: &mut ImplRegistry) -> Result<(), Error> {
    reg.register(ChangedBounds::manifest()?)?;
    impls.register(
        bounds_assert::CHANGED_BOUNDS_OP_ID.parse()?,
        Box::new(ChangedBounds::new()),
    )?;

    reg.register(Diff::manifest()?)?;
    impls.register(diff::DIFF_OP_ID.parse()?, Box::new(Diff::new()))?;

    reg.register(Histogram::manifest()?)?;
    impls.register(
        statistics::HISTOGRAM_OP_ID.parse()?,
        Box::new(Histogram::new()),
    )?;

    reg.register(Statistics::manifest()?)?;
    impls.register(
        statistics::STATISTICS_OP_ID.parse()?,
        Box::new(Statistics::new()),
    )?;

    Ok(())
}
