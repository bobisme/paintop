//! Evidence-bundle manifest schema + atomic writer (`plan.md` §15.1,
//! `AGENT_VERIFICATION` §5).
//!
//! An *evidence bundle* is a directory that tells an agent what a run did and
//! whether to trust it. This module owns the bundle's **manifest schema**
//! ([`BundleManifest`]), its **on-disk layout** ([`BundleLayout`]), and the
//! **atomic writer** ([`BundleWriter`]) that lays out the directory and publishes
//! the canonical artifacts (`manifest.json`, `normalized-plan.json`) with
//! temp-then-rename semantics so a crash mid-write never leaves a partial
//! canonical artifact.
//!
//! The trace / metrics / assertion / contact-sheet stages (later bones) attach to
//! the same writer via [`BundleWriter::write_artifact`] /
//! [`BundleWriter::write_bytes`]; this bone deliberately provides only the
//! manifest + canonical-artifact substrate they build on.
//!
//! ```no_run
//! use paintop_core::evidence::{BundleManifest, BundleWriter, RunStatus};
//! use paintop_ir::{DeterminismTier, parse_plan};
//!
//! # fn demo() -> Result<(), Box<dyn std::error::Error>> {
//! let plan = parse_plan(r#"{"paintop":"1.0","inputs":{},"nodes":[],"exports":{}}"#)?;
//! let writer = BundleWriter::create("/tmp/run-bundle")?;
//! let hash = writer.write_normalized_plan(&plan)?;
//! let manifest = BundleManifest::new(
//!     "0.0.0+devsha",
//!     hash,
//!     DeterminismTier::Exact,
//!     RunStatus::Success,
//! );
//! writer.write_manifest(&manifest)?;
//! # Ok(())
//! # }
//! ```

pub mod assertions;
pub mod atomic;
pub mod contact;
pub mod error;
pub mod layout;
pub mod manifest;
pub mod materialize;
pub mod png;
pub mod replay;
pub mod trace;
pub mod writer;

pub use assertions::{AssertionArtifacts, AssertionEntry, AssertionReport, MetricBag};
pub use atomic::write_atomic;
pub use contact::{ContactSheet, Panel};
pub use error::{BundleError, BundleResult, E_BUNDLE_IO, E_BUNDLE_SERIALIZE};
pub use layout::{BundleLayout, SUBDIRS};
pub use manifest::{BundleManifest, OutputEntry, Platform, RunStatus};
pub use materialize::{FailureInputs, materialize_failure};
pub use png::encode_rgba;
pub use replay::{MinimalReplay, ReplaySpec};
pub use trace::{
    AssertionStatus, CacheOutcome, DispatchStatus, TileCounts, TraceEvent, TraceWriter,
};
pub use writer::BundleWriter;
