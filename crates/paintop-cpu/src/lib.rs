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
pub mod blend;
pub mod bounds_assert;
pub mod canvas;
pub mod channel;
pub mod color;
pub mod composite;
pub mod composite_over;
pub mod convolve;
pub mod crop;
pub mod diff;
pub mod domains;
pub mod edt;
pub mod ellipse;
pub mod fill;
pub mod flip;
pub mod gaussian_blur;
pub mod gradient;
pub mod inspect;
pub mod io;
pub mod mask;
pub mod mask_algebra;
pub mod mask_bounds;
pub mod mask_macros;
pub mod mask_polygon;
pub mod mask_to_sdf;
pub mod mask_topology;
pub mod materialize;
pub mod pad;
pub mod pipeline;
pub mod registry;
pub mod resize;
pub mod rotate;
pub mod sdf_algebra;
pub mod sdf_ops;
pub mod splat;
pub mod statistics;
