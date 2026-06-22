//! Registration for the `optimize` op domain (`plan.md` §1428 final deliverable;
//! `ALIEN_OPS` §7 — the contract-driven micro-optimizer).
//!
//! Owns the contract-driven local optimizer `optimize.local@1`, which drives a
//! candidate image toward minimizing a declared mask-restricted objective. Adding
//! an `optimize` op edits only this file plus the op's own module.

use paintop_core::executor::ImplRegistry;
use paintop_ir::{Error, OperationRegistry};

use crate::local_optimize::{self, LocalOptimize};

/// Register every `optimize.*` manifest and implementation, in fixed declaration
/// order.
///
/// # Errors
/// Propagates a schema error from an invalid manifest or a duplicate-id error.
pub(crate) fn register(reg: &mut OperationRegistry, impls: &mut ImplRegistry) -> Result<(), Error> {
    reg.register(LocalOptimize::manifest()?)?;
    impls.register(
        local_optimize::LOCAL_OPTIMIZE_OP_ID.parse()?,
        Box::new(LocalOptimize::new()),
    )?;

    Ok(())
}
