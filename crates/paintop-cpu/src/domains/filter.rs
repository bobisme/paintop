//! Registration for the `filter` op domain.
//!
//! Owns the `filter.*` operations (convolve, gaussian blur). Adding a `filter`
//! op edits only this file plus the op's own module.

use paintop_core::executor::ImplRegistry;
use paintop_ir::{Error, ImplId, OperationRegistry};

use crate::bilateral::{self, Bilateral};
use crate::convolve::{self, Convolve};
use crate::gaussian_blur::{self, GaussianBlur, GaussianBlurOptimized};
use crate::guided::{self, Guided};
use crate::structure_tensor::{self, StructureTensor};

/// The `cpu.optimized@1` backend id the separable Gaussian is registered under.
fn optimized_impl_id() -> Result<ImplId, Error> {
    ImplId::new("cpu", "optimized", 1)
}

/// Register every `filter.*` manifest and implementation, in fixed declaration
/// order.
///
/// `filter.gaussian_blur@1` carries its `cpu.reference` direct-convolution oracle
/// plus a `cpu.optimized` separable backend (M3 cluster 3); the scheduler selects
/// between them by policy, and the cross-backend differential harness validates the
/// separable result against the oracle within the op's bounded tolerance.
///
/// # Errors
/// Propagates a schema error from an invalid manifest or a duplicate-id error.
pub(crate) fn register(reg: &mut OperationRegistry, impls: &mut ImplRegistry) -> Result<(), Error> {
    reg.register(Convolve::manifest()?)?;
    impls.register(convolve::CONVOLVE_OP_ID.parse()?, Box::new(Convolve::new()))?;

    reg.register(GaussianBlur::manifest()?)?;
    impls.register(
        gaussian_blur::GAUSSIAN_BLUR_OP_ID.parse()?,
        Box::new(GaussianBlur::new()),
    )?;
    impls.register_backend(
        gaussian_blur::GAUSSIAN_BLUR_OP_ID.parse()?,
        &optimized_impl_id()?,
        Box::new(GaussianBlurOptimized::new()),
    )?;

    reg.register(StructureTensor::manifest()?)?;
    impls.register(
        structure_tensor::STRUCTURE_TENSOR_OP_ID.parse()?,
        Box::new(StructureTensor::new()),
    )?;

    reg.register(Guided::manifest()?)?;
    impls.register(guided::GUIDED_OP_ID.parse()?, Box::new(Guided::new()))?;

    reg.register(Bilateral::manifest()?)?;
    impls.register(
        bilateral::BILATERAL_OP_ID.parse()?,
        Box::new(Bilateral::new()),
    )?;

    Ok(())
}
