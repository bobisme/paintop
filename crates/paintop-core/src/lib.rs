//! `paintop-core`: the compiler, scheduler, cache, policy engine, and tracing.
//!
//! Per `plan.md` §6.1 core sits above [`paintop_image`]/[`paintop_ir`] and must
//! not depend on a model runtime. Concrete types are filled in by later bones.

pub mod cache;
pub mod evidence;
pub mod executor;
pub mod graphviz;
pub mod tile;

pub use paintop_image as image;
pub use paintop_ir as ir;
