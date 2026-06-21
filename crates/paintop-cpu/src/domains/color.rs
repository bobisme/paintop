//! Registration for the `color` op domain.
//!
//! Owns the `color.*` operations (`color.adjust`, `color.convert`). Adding a
//! `color` op edits only this file plus the op's own module.

use paintop_core::executor::ImplRegistry;
use paintop_ir::{Error, ImplId, OperationRegistry};

use crate::adjust::{self, Adjust, AdjustOptimized};
use crate::color::{self, Convert, ConvertOptimized};

/// The `cpu.optimized@1` backend id every optimized impl is registered under.
fn optimized_impl_id() -> Result<ImplId, Error> {
    ImplId::new("cpu", "optimized", 1)
}

/// Register every `color.*` manifest and implementation, in fixed declaration
/// order.
///
/// Each op carries its `cpu.reference` oracle plus a `cpu.optimized` autovectorized
/// backend (M3 cluster 2), validated against the oracle by the cross-backend
/// differential harness within tier tolerance.
///
/// # Errors
/// Propagates a schema error from an invalid manifest or a duplicate-id error.
pub(crate) fn register(reg: &mut OperationRegistry, impls: &mut ImplRegistry) -> Result<(), Error> {
    let optimized = optimized_impl_id()?;

    reg.register(Adjust::manifest()?)?;
    impls.register(adjust::ADJUST_OP_ID.parse()?, Box::new(Adjust::new()))?;
    impls.register_backend(
        adjust::ADJUST_OP_ID.parse()?,
        &optimized,
        Box::new(AdjustOptimized::new()),
    )?;

    reg.register(Convert::manifest()?)?;
    impls.register(color::CONVERT_OP_ID.parse()?, Box::new(Convert::new()))?;
    impls.register_backend(
        color::CONVERT_OP_ID.parse()?,
        &optimized,
        Box::new(ConvertOptimized::new()),
    )?;

    Ok(())
}
