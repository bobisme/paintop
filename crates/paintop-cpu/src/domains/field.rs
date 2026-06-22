//! Registration for the `field` op domain (`OP_CATALOG` §10.4).
//!
//! Owns the vector-field analysis ops (`field.orientation`). Adding a `field`
//! op edits only this file plus the op's own module.

use paintop_core::executor::ImplRegistry;
use paintop_ir::{Error, OperationRegistry};

use crate::domain_warp::{self, DomainWarp};
use crate::noise::{self, Fbm, Noise};
use crate::orientation::{self, Orientation};
use crate::reaction_diffusion::{self, ReactionDiffusion};

/// Register every `field.*` manifest and implementation, in fixed declaration
/// order.
///
/// # Errors
/// Propagates a schema error from an invalid manifest or a duplicate-id error.
pub(crate) fn register(reg: &mut OperationRegistry, impls: &mut ImplRegistry) -> Result<(), Error> {
    reg.register(Orientation::manifest()?)?;
    impls.register(
        orientation::ORIENTATION_OP_ID.parse()?,
        Box::new(Orientation::new()),
    )?;

    reg.register(Noise::manifest()?)?;
    impls.register(noise::NOISE_OP_ID.parse()?, Box::new(Noise::new()))?;

    reg.register(Fbm::manifest()?)?;
    impls.register(noise::FBM_OP_ID.parse()?, Box::new(Fbm::new()))?;

    reg.register(DomainWarp::manifest()?)?;
    impls.register(
        domain_warp::DOMAIN_WARP_OP_ID.parse()?,
        Box::new(DomainWarp::new()),
    )?;

    reg.register(ReactionDiffusion::manifest()?)?;
    impls.register(
        reaction_diffusion::REACTION_DIFFUSION_OP_ID.parse()?,
        Box::new(ReactionDiffusion::new()),
    )?;

    Ok(())
}
