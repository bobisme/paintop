//! Registration for the `image` op domain.
//!
//! Owns the `image.*` operations, which span several op modules (channel,
//! canvas, crop, pad, flip, rotate, resize, inspect). Adding an `image` op
//! edits only this file plus the op's own module.

use paintop_core::executor::ImplRegistry;
use paintop_ir::{Error, OperationRegistry};

use crate::canvas::{self, CreateImage};
use crate::channel::{self, AssembleChannels, ExtractChannel};
use crate::crop::{self, Crop};
use crate::flip::{self, Flip};
use crate::inspect::{self, Inspect};
use crate::pad::{self, Pad};
use crate::resize::{self, Resize};
use crate::rotate::{self, Rotate90};

/// Register every `image.*` manifest and implementation, in fixed declaration
/// order.
///
/// # Errors
/// Propagates a schema error from an invalid manifest or a duplicate-id error.
pub(crate) fn register(reg: &mut OperationRegistry, impls: &mut ImplRegistry) -> Result<(), Error> {
    reg.register(AssembleChannels::manifest()?)?;
    impls.register(
        channel::ASSEMBLE_OP_ID.parse()?,
        Box::new(AssembleChannels::new()),
    )?;

    reg.register(CreateImage::manifest()?)?;
    impls.register(canvas::CREATE_OP_ID.parse()?, Box::new(CreateImage::new()))?;

    reg.register(Crop::manifest()?)?;
    impls.register(crop::CROP_OP_ID.parse()?, Box::new(Crop::new()))?;

    reg.register(ExtractChannel::manifest()?)?;
    impls.register(
        channel::EXTRACT_OP_ID.parse()?,
        Box::new(ExtractChannel::new()),
    )?;

    reg.register(Flip::manifest()?)?;
    impls.register(flip::FLIP_OP_ID.parse()?, Box::new(Flip::new()))?;

    reg.register(Inspect::manifest()?)?;
    impls.register(inspect::INSPECT_OP_ID.parse()?, Box::new(Inspect::new()))?;

    reg.register(Pad::manifest()?)?;
    impls.register(pad::PAD_OP_ID.parse()?, Box::new(Pad::new()))?;

    reg.register(Resize::manifest()?)?;
    impls.register(resize::RESIZE_OP_ID.parse()?, Box::new(Resize::new()))?;

    reg.register(Rotate90::manifest()?)?;
    impls.register(rotate::ROTATE90_OP_ID.parse()?, Box::new(Rotate90::new()))?;

    Ok(())
}
