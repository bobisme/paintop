//! `paintop-image`: image / mask / field abstractions and contracts.
//!
//! Per `plan.md` §6.1 this crate must **not** know that 3D exists. It depends
//! only on [`paintop_ir`]. Concrete types are filled in by later bones.

pub use paintop_ir as ir;
