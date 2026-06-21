//! Registration for the `io` op domain.
//!
//! Owns the `io.*` operations. Adding an `io` op edits only this file plus the
//! op's own module — never the central `registry.rs`.

use paintop_core::executor::ImplRegistry;
use paintop_ir::{Error, OperationRegistry};

use crate::io::{self, DecodeImage, EncodeImage};

/// Register every `io.*` manifest and implementation, in fixed declaration
/// order.
///
/// # Errors
/// Propagates a schema error from an invalid manifest or a duplicate-id error
/// (neither occurs for the fixed op set).
pub(crate) fn register(reg: &mut OperationRegistry, impls: &mut ImplRegistry) -> Result<(), Error> {
    reg.register(DecodeImage::manifest()?)?;
    impls.register(io::DECODE_OP_ID.parse()?, Box::new(DecodeImage::new()))?;

    reg.register(EncodeImage::manifest()?)?;
    impls.register(io::ENCODE_OP_ID.parse()?, Box::new(EncodeImage::new()))?;

    Ok(())
}
