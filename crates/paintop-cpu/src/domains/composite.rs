//! Registration for the `composite` op domain.
//!
//! Owns the `composite.*` operations (blend, masked replace, over). Adding a
//! `composite` op edits only this file plus the op's own module.

use paintop_core::executor::ImplRegistry;
use paintop_ir::{Error, ImplId, OperationRegistry};

use crate::blend::{self, Blend, BlendOptimized};
use crate::composite::{self, MaskedReplace};
use crate::composite_over::{self, Over, OverOptimized};

/// The `cpu.optimized@1` backend id every optimized impl is registered under.
fn optimized_impl_id() -> Result<ImplId, Error> {
    ImplId::new("cpu", "optimized", 1)
}

/// Register every `composite.*` manifest and implementation, in fixed
/// declaration order.
///
/// `composite.over` and `composite.blend` carry a `cpu.optimized` autovectorized
/// backend alongside their `cpu.reference` oracle (M3 cluster 2), validated against
/// the oracle by the cross-backend differential harness within tier tolerance.
/// `composite.masked_replace` stays reference-only (not a cluster-2 kernel).
///
/// # Errors
/// Propagates a schema error from an invalid manifest or a duplicate-id error.
pub(crate) fn register(reg: &mut OperationRegistry, impls: &mut ImplRegistry) -> Result<(), Error> {
    let optimized = optimized_impl_id()?;

    reg.register(Blend::manifest()?)?;
    impls.register(blend::BLEND_OP_ID.parse()?, Box::new(Blend::new()))?;
    impls.register_backend(
        blend::BLEND_OP_ID.parse()?,
        &optimized,
        Box::new(BlendOptimized::new()),
    )?;

    reg.register(MaskedReplace::manifest()?)?;
    impls.register(
        composite::MASKED_REPLACE_OP_ID.parse()?,
        Box::new(MaskedReplace::new()),
    )?;

    reg.register(Over::manifest()?)?;
    impls.register(composite_over::OVER_OP_ID.parse()?, Box::new(Over::new()))?;
    impls.register_backend(
        composite_over::OVER_OP_ID.parse()?,
        &optimized,
        Box::new(OverOptimized::new()),
    )?;

    Ok(())
}
