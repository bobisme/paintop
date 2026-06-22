//! Registration for the `repair` op domain (`OP_CATALOG` §12).
//!
//! Owns the gradient-domain Poisson editing ops — `repair.poisson_blend`
//! (seamless cloning) and `repair.screened_poisson` (screened reconstruction) —
//! built on the shared [`crate::poisson`] solver core, plus the `PatchMatch`
//! correspondence op `repair.patch_field` and the patch-fill op
//! `repair.patch_synthesize`. Adding a `repair` op edits only this file plus the
//! op's own module.

use paintop_core::executor::ImplRegistry;
use paintop_ir::{Error, OperationRegistry};

use crate::patch_field::{self, PatchField};
use crate::patch_synthesize::{self, PatchSynthesize};
use crate::poisson_blend::{self, PoissonBlend};
use crate::screened_poisson::{self, ScreenedPoisson};

/// Register every `repair.*` manifest and implementation, in fixed declaration
/// order.
///
/// # Errors
/// Propagates a schema error from an invalid manifest or a duplicate-id error.
pub(crate) fn register(reg: &mut OperationRegistry, impls: &mut ImplRegistry) -> Result<(), Error> {
    reg.register(PoissonBlend::manifest()?)?;
    impls.register(
        poisson_blend::POISSON_BLEND_OP_ID.parse()?,
        Box::new(PoissonBlend::new()),
    )?;

    reg.register(ScreenedPoisson::manifest()?)?;
    impls.register(
        screened_poisson::SCREENED_POISSON_OP_ID.parse()?,
        Box::new(ScreenedPoisson::new()),
    )?;

    reg.register(PatchField::manifest()?)?;
    impls.register(
        patch_field::PATCH_FIELD_OP_ID.parse()?,
        Box::new(PatchField::new()),
    )?;

    reg.register(PatchSynthesize::manifest()?)?;
    impls.register(
        patch_synthesize::PATCH_SYNTHESIZE_OP_ID.parse()?,
        Box::new(PatchSynthesize::new()),
    )?;

    Ok(())
}
