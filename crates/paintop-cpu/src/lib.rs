//! `paintop-cpu`: the scalar reference ("oracle") and SIMD/Rayon op backends.
//!
//! Per `plan.md` §6.1 this crate implements contracts **owned** by the
//! core/image/ir crates rather than defining its own. Filled in by later bones.

pub use paintop_core as core;
