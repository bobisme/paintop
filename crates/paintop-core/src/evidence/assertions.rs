//! The evidence-bundle assertion report (`assertions.json`,
//! `AGENT_VERIFICATION` §5.3).
//!
//! A run's `assertions.json` records what each postcondition assertion measured,
//! what threshold it was held to, and — on failure — where the failure-driven
//! artifacts (`plan.md` §18.2) landed. The report is the agent-facing answer to
//! "did the edit do *only* what it claimed", so it is deliberately
//! machine-precise: stable assertion `id`, the versioned assertion `op`, a
//! [`status`](AssertionStatus), and free-form numeric `metrics` / `thresholds`
//! bags whose keys are owned by each assertion op (e.g.
//! `max_abs_delta_outside`, `changed_pixels_outside`, `worst_pixel`).
//!
//! ## Failure artifacts travel with the report
//!
//! When an assertion fails, the run materializes a self-contained bug specimen
//! (`plan.md` §18.2): an outside-mask diff image, a minimal replay plan, and so
//! on. The [`AssertionArtifacts`] block carries the *bundle-relative* paths to
//! those artifacts so a reader follows the report straight to the evidence
//! without guessing file names. Absent artifacts are simply omitted — a passing
//! assertion produces a clean object with no `artifacts` key, never a malformed
//! placeholder (`plan.md` §15.1 acceptance).
//!
//! This module owns only the **schema and its builders**. The executor /
//! assertion stage (later bones) measures the numbers and fills the bags; the
//! [`BundleWriter`](crate::evidence::BundleWriter) writes the canonical
//! `assertions.json`. The wire form is the contract, so every struct is
//! `deny_unknown_fields`.
//!
//! ```
//! use paintop_core::evidence::assertions::{AssertionEntry, AssertionReport};
//! use paintop_core::evidence::AssertionStatus;
//!
//! let entry = AssertionEntry::new("localized", "assert.no_change_outside_mask@1", AssertionStatus::Passed);
//! let report = AssertionReport::from(vec![entry]);
//! assert!(report.all_passed());
//! assert!(report.failures().next().is_none());
//! ```

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::evidence::trace::AssertionStatus;

/// A free-form numeric measurement bag keyed by an assertion-op-owned key.
///
/// Modeled as a [`BTreeMap`] of `String` → JSON value so the keys are
/// order-stable (canonicalization is deterministic) while the exact metric set
/// stays open to each assertion op (`max_abs_delta_outside`, `worst_pixel`, …).
pub type MetricBag = BTreeMap<String, serde_json::Value>;

/// The bundle-relative paths of a failing assertion's failure artifacts
/// (`AGENT_VERIFICATION` §5.3 `artifacts`; materialized per `plan.md` §18.2).
///
/// Every field is optional and omitted when absent, so a partially materialized
/// failure (e.g. a diff written but no replay reduced yet) still serializes
/// cleanly. Paths are bundle-relative (`"diffs/localized-outside.png"`,
/// `"replays/localized.json"`) so a reader resolves them against the bundle root.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AssertionArtifacts {
    /// The before/after/diff image isolating the violation, if written.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub outside_diff: Option<String>,
    /// The minimal replay plan reproducing the failure, if emitted.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub minimal_replay: Option<String>,
    /// The contact sheet contextualizing the failure, if composited.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub contact_sheet: Option<String>,
}

impl AssertionArtifacts {
    /// Whether no artifact path is set (the block would serialize empty).
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.outside_diff.is_none() && self.minimal_replay.is_none() && self.contact_sheet.is_none()
    }
}

/// One assertion's measured outcome (`AGENT_VERIFICATION` §5.3).
///
/// Construct it from the assertion's stable `id`, its versioned `op`, and a
/// [`status`](AssertionStatus); attach `metrics`, `thresholds`, and failure
/// `artifacts` with the builder methods. The `metrics`/`thresholds` bags and the
/// `artifacts` block are omitted from the wire form when empty so a passing
/// assertion is a terse object.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AssertionEntry {
    /// The assertion's stable id (e.g. `"localized"`).
    pub id: String,
    /// The versioned assertion op (e.g. `"assert.no_change_outside_mask@1"`).
    pub op: String,
    /// Whether the assertion held.
    pub status: AssertionStatus,
    /// What the assertion measured, keyed by assertion-op-owned metric names.
    #[serde(default, skip_serializing_if = "MetricBag::is_empty")]
    pub metrics: MetricBag,
    /// The thresholds the assertion was held to.
    #[serde(default, skip_serializing_if = "MetricBag::is_empty")]
    pub thresholds: MetricBag,
    /// The bundle-relative failure artifacts (only meaningful on failure).
    #[serde(default, skip_serializing_if = "AssertionArtifacts::is_empty")]
    pub artifacts: AssertionArtifacts,
}

impl AssertionEntry {
    /// Build an entry from the assertion's identity and outcome, with empty
    /// metric/threshold bags and no artifacts.
    #[must_use]
    pub fn new(id: impl Into<String>, op: impl Into<String>, status: AssertionStatus) -> Self {
        Self {
            id: id.into(),
            op: op.into(),
            status,
            metrics: MetricBag::new(),
            thresholds: MetricBag::new(),
            artifacts: AssertionArtifacts::default(),
        }
    }

    /// Replace the measured metrics bag.
    #[must_use]
    pub fn with_metrics(mut self, metrics: MetricBag) -> Self {
        self.metrics = metrics;
        self
    }

    /// Replace the thresholds bag.
    #[must_use]
    pub fn with_thresholds(mut self, thresholds: MetricBag) -> Self {
        self.thresholds = thresholds;
        self
    }

    /// Replace the failure artifacts block.
    #[must_use]
    pub fn with_artifacts(mut self, artifacts: AssertionArtifacts) -> Self {
        self.artifacts = artifacts;
        self
    }

    /// Whether this assertion failed.
    #[must_use]
    pub const fn is_failed(&self) -> bool {
        matches!(self.status, AssertionStatus::Failed)
    }
}

/// The whole `assertions.json` report: every measured assertion in plan order
/// (`AGENT_VERIFICATION` §5.3).
///
/// The order mirrors the plan's `assertions` list so the report is stable across
/// runs. [`failures`](AssertionReport::failures) and
/// [`failure_ids`](AssertionReport::failure_ids) let the manifest writer stamp
/// the failing ids without re-deriving them.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AssertionReport {
    /// The measured assertions, in plan order.
    pub assertions: Vec<AssertionEntry>,
}

impl AssertionReport {
    /// The conventional bundle-relative path of the assertion report artifact.
    pub const PATH: &'static str = "assertions.json";

    /// Whether every measured assertion passed (an empty report passes
    /// vacuously).
    #[must_use]
    pub fn all_passed(&self) -> bool {
        self.assertions
            .iter()
            .all(|a| !matches!(a.status, AssertionStatus::Failed))
    }

    /// The failed assertions, in plan order.
    pub fn failures(&self) -> impl Iterator<Item = &AssertionEntry> {
        self.assertions.iter().filter(|a| a.is_failed())
    }

    /// The stable ids of the failed assertions, in plan order — the value the
    /// manifest's `failures` field carries (`AGENT_VERIFICATION` §5.1).
    #[must_use]
    pub fn failure_ids(&self) -> Vec<String> {
        self.failures().map(|a| a.id.clone()).collect()
    }
}

impl From<Vec<AssertionEntry>> for AssertionReport {
    fn from(assertions: Vec<AssertionEntry>) -> Self {
        Self { assertions }
    }
}

#[cfg(test)]
mod tests {
    use super::{AssertionArtifacts, AssertionEntry, AssertionReport, MetricBag};
    use crate::evidence::trace::AssertionStatus;
    use serde_json::json;

    fn localized_failure() -> AssertionEntry {
        let mut metrics = MetricBag::new();
        metrics.insert("max_abs_delta_outside".to_owned(), json!(0.003_12));
        metrics.insert("changed_pixels_outside".to_owned(), json!(17));
        metrics.insert("worst_pixel".to_owned(), json!([721, 418]));
        let mut thresholds = MetricBag::new();
        thresholds.insert("max_abs_delta".to_owned(), json!(0.000_001));
        thresholds.insert("changed_pixels".to_owned(), json!(0));
        AssertionEntry::new(
            "localized",
            "assert.no_change_outside_mask@1",
            AssertionStatus::Failed,
        )
        .with_metrics(metrics)
        .with_thresholds(thresholds)
        .with_artifacts(AssertionArtifacts {
            outside_diff: Some("diffs/localized-outside.png".to_owned()),
            minimal_replay: Some("replays/localized.json".to_owned()),
            contact_sheet: None,
        })
    }

    #[test]
    fn failure_entry_matches_the_spec_example_shape() {
        // Mirrors `AGENT_VERIFICATION` §5.3.
        let v = serde_json::to_value(localized_failure()).expect("serialize");
        let obj = v.as_object().expect("object");
        assert_eq!(obj.get("id"), Some(&json!("localized")));
        assert_eq!(obj.get("status"), Some(&json!("failed")));
        assert_eq!(
            obj["metrics"]["worst_pixel"],
            json!([721, 418]),
            "metric keys are preserved verbatim"
        );
        assert_eq!(
            obj["artifacts"]["minimal_replay"],
            json!("replays/localized.json")
        );
        // An absent artifact field is omitted, never null.
        assert!(
            !obj["artifacts"]
                .as_object()
                .unwrap()
                .contains_key("contact_sheet")
        );
    }

    #[test]
    fn passing_entry_is_terse() {
        let entry = AssertionEntry::new(
            "localized",
            "assert.no_change_outside_mask@1",
            AssertionStatus::Passed,
        );
        let v = serde_json::to_value(&entry).expect("serialize");
        let obj = v.as_object().expect("object");
        // No empty bags / artifacts block leak onto the wire.
        assert!(!obj.contains_key("metrics"));
        assert!(!obj.contains_key("thresholds"));
        assert!(!obj.contains_key("artifacts"));
    }

    #[test]
    fn report_round_trips_and_reports_failures() {
        let report = AssertionReport::from(vec![
            AssertionEntry::new("a", "assert.x@1", AssertionStatus::Passed),
            localized_failure(),
            AssertionEntry::new("c", "assert.z@1", AssertionStatus::Skipped),
        ]);
        assert!(!report.all_passed());
        assert_eq!(report.failure_ids(), vec!["localized".to_owned()]);
        assert_eq!(report.failures().count(), 1);

        let v = serde_json::to_value(&report).expect("serialize");
        let back: AssertionReport = serde_json::from_value(v).expect("round trip");
        assert_eq!(back, report);
    }

    #[test]
    fn empty_report_passes_vacuously() {
        let report = AssertionReport::default();
        assert!(report.all_passed());
        assert!(report.failure_ids().is_empty());
    }

    #[test]
    fn unknown_entry_field_is_rejected() {
        let v = json!({
            "id": "localized",
            "op": "assert.no_change_outside_mask@1",
            "status": "failed",
            "surprise": true
        });
        assert!(serde_json::from_value::<AssertionEntry>(v).is_err());
    }
}
