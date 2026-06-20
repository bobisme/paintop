//! The well-known on-disk layout of an evidence bundle (`plan.md` §15.1).
//!
//! A bundle is a directory with a fixed set of file and subdirectory names. This
//! module is the single source of truth for those names so the writer, the
//! manifest's relative paths, and any later reader agree. Subdirectories are
//! created eagerly so a downstream stage can drop an artifact into, e.g.,
//! `diffs/` without first checking whether the directory exists.
//!
//! Optional artifacts (`trace.jsonl`, `metrics.json`, `contact-sheet.png`, …)
//! are *named* here but only materialized by the stages that own them; a missing
//! optional artifact is simply absent, never a malformed placeholder
//! (`plan.md` §15.1 acceptance).

use std::path::{Path, PathBuf};

/// The canonical top-level artifact file names of a bundle (`plan.md` §15.1).
pub mod files {
    /// The bundle index / manifest.
    pub const MANIFEST: &str = "manifest.json";
    /// The exact normalized graph that was executed.
    pub const NORMALIZED_PLAN: &str = "normalized-plan.json";
    /// Decoded-input content hashes and semantics.
    pub const INPUT_MANIFEST: &str = "input-manifest.json";
    /// The Graphviz rendering of the executed graph.
    pub const GRAPH_DOT: &str = "graph.dot";
    /// The structured per-node trace (JSON Lines).
    pub const TRACE: &str = "trace.jsonl";
    /// Recorded run metrics.
    pub const METRICS: &str = "metrics.json";
    /// The assertion report.
    pub const ASSERTIONS: &str = "assertions.json";
    /// The before/after/diff contact sheet.
    pub const CONTACT_SHEET: &str = "contact-sheet.png";
}

/// The canonical subdirectory names of a bundle (`plan.md` §15.1).
pub mod dirs {
    /// Produced outputs.
    pub const OUTPUTS: &str = "outputs";
    /// Requested or failure-relevant intermediates.
    pub const INTERMEDIATES: &str = "intermediates";
    /// Masks involved in the run.
    pub const MASKS: &str = "masks";
    /// Absolute / relative diffs.
    pub const DIFFS: &str = "diffs";
    /// Minimal-replay reproducers.
    pub const REPLAYS: &str = "replays";
    /// Free-form logs.
    pub const LOGS: &str = "logs";
}

/// The fixed set of subdirectories created when a bundle is laid out, in a
/// stable order.
pub const SUBDIRS: [&str; 6] = [
    dirs::OUTPUTS,
    dirs::INTERMEDIATES,
    dirs::MASKS,
    dirs::DIFFS,
    dirs::REPLAYS,
    dirs::LOGS,
];

/// A typed handle to a bundle's root directory that resolves the well-known
/// artifact and subdirectory paths beneath it.
///
/// The layout only *computes* paths; it does not touch the filesystem (the
/// [`BundleWriter`](crate::evidence::BundleWriter) creates the directories). Use
/// it to keep manifest-relative names and on-disk paths in lockstep.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BundleLayout {
    root: PathBuf,
}

impl BundleLayout {
    /// Create a layout rooted at `root`. No filesystem access occurs.
    #[must_use]
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    /// The bundle's root directory.
    #[must_use]
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Resolve a bundle-relative path against the root.
    #[must_use]
    pub fn join(&self, relative: impl AsRef<Path>) -> PathBuf {
        self.root.join(relative)
    }

    /// The absolute path of `manifest.json`.
    #[must_use]
    pub fn manifest(&self) -> PathBuf {
        self.join(files::MANIFEST)
    }

    /// The absolute path of `normalized-plan.json`.
    #[must_use]
    pub fn normalized_plan(&self) -> PathBuf {
        self.join(files::NORMALIZED_PLAN)
    }

    /// The absolute paths of every well-known subdirectory, in [`SUBDIRS`] order.
    #[must_use]
    pub fn subdirs(&self) -> Vec<PathBuf> {
        SUBDIRS.iter().map(|d| self.join(d)).collect()
    }
}
