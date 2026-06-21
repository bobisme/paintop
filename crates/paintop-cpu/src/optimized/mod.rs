//! The `cpu.optimized` backend: safe, autovectorization-friendly pointwise kernels
//! that reproduce the `cpu.reference` oracle within each op's declared tolerance
//! (`plan.md` §12.2; M3 cluster 2).
//!
//! M3 adds *faster backends*, never new ops. This module is the optimized CPU
//! backend for the pointwise color / alpha / compositing ops: the kernels
//! themselves ([`mod@kernels`]) plus the measurement-and-selection substrate
//! ([`mod@bench`]) that proves each kernel is worth a second backend before one is
//! wired (`plan.md` §12.2: "explicit SIMD where benchmarks prove value").
//!
//! Every kernel is **safe Rust** — the crate forbids `unsafe`, so there are no raw
//! SIMD intrinsics. The kernels are written in the shape the compiler
//! autovectorizes (tight fixed-stride loops, branch-free inner arithmetic), and
//! they reproduce the reference arithmetic operation-for-operation so the exact-
//! tier ops stay bit-identical and the bounded-tier ops stay within envelope. The
//! cross-backend differential harness (`bn-2ja`) is the authority that this holds.
//!
//! The optimized op wrappers that adapt these kernels to the
//! [`OpImplementation`](paintop_core::executor::OpImplementation) trait, and the
//! `register_backend` wiring, live next to each op (`bn-yhf`, `bn-1pu`).

pub mod bench;
pub mod kernels;
