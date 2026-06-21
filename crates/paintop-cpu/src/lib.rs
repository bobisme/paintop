//! `paintop-cpu`: the scalar reference ("oracle") and SIMD/Rayon op backends.
//!
//! Per `plan.md` §6.1 this crate implements contracts **owned** by the
//! core/image/ir crates rather than defining its own. Each MVP operation lives in
//! its own module here: the `cpu.reference` [`OpContract`](paintop_ir::OpContract)
//! (descriptor/ROI/postcondition) and the executable
//! [`OpImplementation`](paintop_core::executor::OpImplementation) compute kernel,
//! plus the op's [`OperationManifest`](paintop_ir::OperationManifest) builder.

pub use paintop_core as core;

pub mod adjust;
pub mod alpha;
pub mod assert;
pub mod color;
pub mod composite;
pub mod diff;
pub mod ellipse;
pub mod inspect;
pub mod io;
pub mod materialize;
pub mod pipeline;
pub mod registry;
pub mod splat;
