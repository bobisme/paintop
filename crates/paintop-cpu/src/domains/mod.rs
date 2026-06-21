//! Per-domain op registration.
//!
//! Each `op DOMAIN` (io, color, alpha, image, mask, sdf, paint, composite,
//! filter, analyze, assert, debug) owns one `register` function that adds *its* ops'
//! manifests and `cpu.reference` implementations to the two runtime registries.
//!
//! The central [`crate::registry`] iterates `REGISTERS` in this explicit,
//! fixed order — it changes only when a *whole new domain* is added, so an
//! op-adding bone edits only its domain module here and never serializes on a
//! shared file.
//!
//! Ordering note: the runtime registries are [`BTreeMap`](std::collections::BTreeMap)-backed
//! and so always iterate in canonical op-id order regardless of *registration*
//! order; this fixed domain order keeps registration reviewable and stable. A
//! determinism test in [`crate::registry`] pins the resulting id sequence.

use paintop_core::executor::ImplRegistry;
use paintop_ir::{Error, OperationRegistry};

pub mod alpha;
pub mod analyze;
pub mod assert;
pub mod color;
pub mod composite;
pub mod debug;
pub mod filter;
pub mod image;
pub mod io;
pub mod mask;
pub mod paint;
pub mod sdf;

/// A domain's registration function: adds every op in that domain to both the
/// manifest registry and the implementation registry.
type DomainRegister = fn(&mut OperationRegistry, &mut ImplRegistry) -> Result<(), Error>;

/// Every domain's `register` fn, in the fixed domain order the runtime applies.
///
/// Adding a new *domain* appends one entry here; adding an op to an *existing*
/// domain touches only that domain's module.
pub(crate) const REGISTERS: &[DomainRegister] = &[
    io::register,
    color::register,
    alpha::register,
    image::register,
    mask::register,
    sdf::register,
    paint::register,
    composite::register,
    filter::register,
    analyze::register,
    assert::register,
    debug::register,
];
