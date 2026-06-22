//! Registration for the `frequency` op domain (`OP_CATALOG` §13).
//!
//! Owns the multi-resolution pyramid ops (`frequency.gaussian_pyramid`,
//! `frequency.laplacian_split`, `frequency.recombine`) and the Fourier ops
//! (`frequency.fft2`, `frequency.ifft2`, `frequency.bandpass`). Adding a
//! `frequency` op edits only this file plus the op's own module.

use paintop_core::executor::ImplRegistry;
use paintop_ir::{Error, OperationRegistry};

use crate::bandpass::{self, Bandpass};
use crate::fft::{self, Fft2, Ifft2};
use crate::gaussian_pyramid::{self, GaussianPyramid};
use crate::laplacian::{self, LaplacianSplit, Recombine};

/// Register every `frequency.*` manifest and implementation, in fixed
/// declaration order.
///
/// # Errors
/// Propagates a schema error from an invalid manifest or a duplicate-id error.
pub(crate) fn register(reg: &mut OperationRegistry, impls: &mut ImplRegistry) -> Result<(), Error> {
    reg.register(GaussianPyramid::manifest()?)?;
    impls.register(
        gaussian_pyramid::GAUSSIAN_PYRAMID_OP_ID.parse()?,
        Box::new(GaussianPyramid::new()),
    )?;

    reg.register(LaplacianSplit::manifest()?)?;
    impls.register(
        laplacian::LAPLACIAN_SPLIT_OP_ID.parse()?,
        Box::new(LaplacianSplit::new()),
    )?;

    reg.register(Recombine::manifest()?)?;
    impls.register(
        laplacian::RECOMBINE_OP_ID.parse()?,
        Box::new(Recombine::new()),
    )?;

    reg.register(Fft2::manifest()?)?;
    impls.register(fft::FFT2_OP_ID.parse()?, Box::new(Fft2::new()))?;

    reg.register(Ifft2::manifest()?)?;
    impls.register(fft::IFFT2_OP_ID.parse()?, Box::new(Ifft2::new()))?;

    reg.register(Bandpass::manifest()?)?;
    impls.register(bandpass::BANDPASS_OP_ID.parse()?, Box::new(Bandpass::new()))?;

    Ok(())
}
