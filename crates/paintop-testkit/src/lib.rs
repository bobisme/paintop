//! `paintop-testkit`: analytic fixtures, metrics, and the
//! metamorphic/differential test harness.
//!
//! Per `plan.md` §6.1 the testkit consumes the core/image/ir crates.
//!
//! # Analytic fixtures
//!
//! [`fixtures`] generates synthetic images with *known structure* from fixed,
//! versioned formulas (`AGENT_VERIFICATION` §2.3, §4.3). Each fixture is an
//! **exact numeric array** — never a quantized screenshot — so an op test can
//! derive the expected behavior analytically instead of trusting a render. A
//! preview PNG is produced only as a human aid and is never the source of
//! truth.
//!
//! Generation is **deterministic**: identical parameters yield byte-identical
//! arrays on every run and machine, and the [`fixtures::Manifest`] records the
//! formula name, its parameters, a version, and the `sha256` of the canonical
//! bytes (`AGENT_VERIFICATION` §4.2 shape).

pub use paintop_core as core;

pub mod fixtures;
pub mod metamorphic;
