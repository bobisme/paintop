//! Registration for the `alpha` op domain.
//!
//! Owns the `alpha.*` operations. Adding an `alpha` op edits only this file
//! plus the op's own module.

use paintop_core::executor::ImplRegistry;
use paintop_ir::{Error, ImplId, OperationRegistry};

use crate::alpha::{
    self, Premultiply, PremultiplyOptimized, Unpremultiply, UnpremultiplyOptimized,
};

/// The `cpu.optimized@1` backend id every optimized impl is registered under.
fn optimized_impl_id() -> Result<ImplId, Error> {
    ImplId::new("cpu", "optimized", 1)
}

/// Register every `alpha.*` manifest and implementation, in fixed declaration
/// order.
///
/// Each op carries its `cpu.reference` oracle plus a `cpu.optimized` autovectorized
/// backend (M3 cluster 2); the scheduler selects between them by policy, and the
/// cross-backend differential harness validates the optimized result against the
/// oracle within tier tolerance.
///
/// # Errors
/// Propagates a schema error from an invalid manifest or a duplicate-id error.
pub(crate) fn register(reg: &mut OperationRegistry, impls: &mut ImplRegistry) -> Result<(), Error> {
    let optimized = optimized_impl_id()?;

    reg.register(Premultiply::manifest()?)?;
    impls.register(
        alpha::PREMULTIPLY_OP_ID.parse()?,
        Box::new(Premultiply::new()),
    )?;
    impls.register_backend(
        alpha::PREMULTIPLY_OP_ID.parse()?,
        &optimized,
        Box::new(PremultiplyOptimized::new()),
    )?;

    reg.register(Unpremultiply::manifest()?)?;
    impls.register(
        alpha::UNPREMULTIPLY_OP_ID.parse()?,
        Box::new(Unpremultiply::new()),
    )?;
    impls.register_backend(
        alpha::UNPREMULTIPLY_OP_ID.parse()?,
        &optimized,
        Box::new(UnpremultiplyOptimized::new()),
    )?;

    Ok(())
}
