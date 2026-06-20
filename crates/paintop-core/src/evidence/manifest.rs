//! The evidence-bundle `manifest.json` schema (`AGENT_VERIFICATION` §5.1).
//!
//! The manifest is the bundle's index: it records the runtime + git provenance,
//! the plan **semantic hash**, the host platform, the determinism tier, the run
//! status + stable exit code, the produced outputs, and the failure ids. It is
//! deliberately split into two halves of meaning:
//!
//! * **Semantic identity** — `plan_semantic_hash`. Two runs of the same plan
//!   share it regardless of when or where they ran.
//! * **Provenance** — `started_at`, `platform`, and the runtime/git build string.
//!   These describe *this* execution and are explicitly **excluded from semantic
//!   identity** (`AGENT_VERIFICATION` §5.1: "Time and host data are provenance,
//!   not semantic identity").
//!
//! The manifest never *contains* a wall-clock-derived identity field; the
//! timestamp is a free-form provenance string the caller supplies (or omits).

use serde::{Deserialize, Serialize};

use paintop_ir::DeterminismTier;

/// The terminal status of a run, as recorded in the manifest (`AGENT_VERIFICATION`
/// §5.1; statuses mirror the stable exit classes of `plan.md` §15.4).
///
/// The wire form is kebab-case (`"assertion-failed"`) to match the §5.1 example.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
#[non_exhaustive]
pub enum RunStatus {
    /// The plan executed and every assertion passed.
    Success,
    /// The plan failed to parse / validate against the schema.
    ParseError,
    /// The plan was type- or semantically invalid.
    SemanticError,
    /// A resource or execution policy limit rejected the plan.
    PolicyRejected,
    /// A backend failed while executing an otherwise-valid plan.
    ExecutionFailed,
    /// A runtime assertion failed.
    AssertionFailed,
    /// A differential / conformance comparison failed.
    ConformanceFailed,
    /// A model adapter failed.
    ModelFailed,
    /// An asset or export integrity contract failed.
    ExportFailed,
}

impl RunStatus {
    /// The stable process exit code for this status (`plan.md` §15.4).
    ///
    /// `0` for [`Success`](Self::Success); otherwise the code of the matching
    /// [`ErrorClass`](paintop_ir::ErrorClass).
    #[must_use]
    pub const fn exit_code(self) -> i32 {
        use paintop_ir::ErrorClass;
        match self {
            Self::Success => 0,
            Self::ParseError => ErrorClass::Parse.exit_code(),
            Self::SemanticError => ErrorClass::Semantic.exit_code(),
            Self::PolicyRejected => ErrorClass::Policy.exit_code(),
            Self::ExecutionFailed => ErrorClass::Execution.exit_code(),
            Self::AssertionFailed => ErrorClass::Assertion.exit_code(),
            Self::ConformanceFailed => ErrorClass::Conformance.exit_code(),
            Self::ModelFailed => ErrorClass::Model.exit_code(),
            Self::ExportFailed => ErrorClass::Export.exit_code(),
        }
    }

    /// Whether this status denotes a fully successful run.
    #[must_use]
    pub const fn is_success(self) -> bool {
        matches!(self, Self::Success)
    }
}

/// Host / device provenance (`AGENT_VERIFICATION` §5.1 `platform`).
///
/// All fields are provenance only and are excluded from semantic identity. The
/// optional device fields (`cpu`, `gpu`, `driver`) are absent — not null — when
/// unknown, so a CPU-only M0 run produces a clean object without placeholder
/// strings.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Platform {
    /// The operating system, e.g. `"linux"`.
    pub os: String,
    /// The CPU architecture, e.g. `"x86_64"`.
    pub arch: String,
    /// A human-readable CPU description, if known.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cpu: Option<String>,
    /// A human-readable GPU description, if known.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gpu: Option<String>,
    /// The graphics/compute driver version, if known.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub driver: Option<String>,
}

impl Platform {
    /// The platform for the **current host**, filling `os`/`arch` from the build
    /// target. Device fields are left empty (M0 is CPU-only and records no GPU).
    #[must_use]
    pub fn current() -> Self {
        Self {
            os: std::env::consts::OS.to_owned(),
            arch: std::env::consts::ARCH.to_owned(),
            cpu: None,
            gpu: None,
            driver: None,
        }
    }
}

/// One produced output recorded in the manifest (`AGENT_VERIFICATION` §5.1
/// `outputs`).
///
/// `path` is the bundle-relative location (e.g. `"outputs/result.png"`); the
/// optional `content_hash` is the algorithm-prefixed content id of the bytes.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OutputEntry {
    /// The plan export/output name this artifact realizes.
    pub name: String,
    /// The bundle-relative path to the written artifact.
    pub path: String,
    /// The algorithm-prefixed content hash of the artifact bytes, if computed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content_hash: Option<String>,
}

/// The evidence-bundle manifest (`AGENT_VERIFICATION` §5.1).
///
/// Construct it from a [`semantic hash`](paintop_ir::SemanticHash) and a
/// [`RunStatus`]; the `exit_code` is derived from the status so the two can
/// never disagree. Provenance (`started_at`, `platform`, runtime build string)
/// is attached separately and excluded from semantic identity.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BundleManifest {
    /// The runtime build string, conventionally `"<version>+<gitsha>"`
    /// (provenance, not semantic identity).
    pub paintop_runtime: String,
    /// The plan **semantic hash** (`blake3:…`) — the bundle's semantic identity.
    pub plan_semantic_hash: String,
    /// The bundle-relative path to the normalized plan that was executed.
    pub normalized_plan: String,
    /// A free-form RFC-3339 provenance timestamp for when the run started, if
    /// recorded. Excluded from semantic identity.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub started_at: Option<String>,
    /// Host / device provenance.
    pub platform: Platform,
    /// The determinism tier the run executed under.
    pub determinism: DeterminismTier,
    /// The terminal run status.
    pub status: RunStatus,
    /// The stable process exit code (derived from `status`).
    pub exit_code: i32,
    /// The produced outputs (possibly empty).
    #[serde(default)]
    pub outputs: Vec<OutputEntry>,
    /// The ids of failed assertions / checks (possibly empty).
    #[serde(default)]
    pub failures: Vec<String>,
}

impl BundleManifest {
    /// The conventional bundle-relative path of the normalized plan artifact.
    pub const NORMALIZED_PLAN_PATH: &'static str = "normalized-plan.json";

    /// Build a manifest from the run's semantic identity and terminal status.
    ///
    /// `exit_code` is taken from [`RunStatus::exit_code`] so it can never drift
    /// from `status`. `platform` defaults to [`Platform::current`]; provenance
    /// (`started_at`, device fields) and `outputs`/`failures` are attached with
    /// the builder methods.
    #[must_use]
    pub fn new(
        runtime: impl Into<String>,
        plan_semantic_hash: impl Into<String>,
        determinism: DeterminismTier,
        status: RunStatus,
    ) -> Self {
        Self {
            paintop_runtime: runtime.into(),
            plan_semantic_hash: plan_semantic_hash.into(),
            normalized_plan: Self::NORMALIZED_PLAN_PATH.to_owned(),
            started_at: None,
            platform: Platform::current(),
            determinism,
            status,
            exit_code: status.exit_code(),
            outputs: Vec::new(),
            failures: Vec::new(),
        }
    }

    /// Attach a free-form provenance start timestamp.
    #[must_use]
    pub fn with_started_at(mut self, started_at: impl Into<String>) -> Self {
        self.started_at = Some(started_at.into());
        self
    }

    /// Replace the host/device platform provenance.
    #[must_use]
    pub fn with_platform(mut self, platform: Platform) -> Self {
        self.platform = platform;
        self
    }

    /// Replace the recorded outputs.
    #[must_use]
    pub fn with_outputs(mut self, outputs: Vec<OutputEntry>) -> Self {
        self.outputs = outputs;
        self
    }

    /// Replace the recorded failure ids.
    #[must_use]
    pub fn with_failures(mut self, failures: Vec<String>) -> Self {
        self.failures = failures;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::{BundleManifest, OutputEntry, Platform, RunStatus};
    use paintop_ir::DeterminismTier;
    use serde_json::json;

    #[test]
    fn status_exit_codes_match_the_stable_classes() {
        // `plan.md` §15.4: success is 0; every failure status shares its error
        // class's code.
        assert_eq!(RunStatus::Success.exit_code(), 0);
        assert_eq!(RunStatus::ParseError.exit_code(), 2);
        assert_eq!(RunStatus::SemanticError.exit_code(), 3);
        assert_eq!(RunStatus::PolicyRejected.exit_code(), 4);
        assert_eq!(RunStatus::ExecutionFailed.exit_code(), 5);
        assert_eq!(RunStatus::AssertionFailed.exit_code(), 6);
        assert_eq!(RunStatus::ConformanceFailed.exit_code(), 7);
        assert_eq!(RunStatus::ModelFailed.exit_code(), 8);
        assert_eq!(RunStatus::ExportFailed.exit_code(), 9);
    }

    #[test]
    fn new_derives_exit_code_from_status() {
        let m = BundleManifest::new(
            "rt",
            "blake3:00",
            DeterminismTier::Bounded,
            RunStatus::AssertionFailed,
        );
        assert_eq!(m.exit_code, 6);
        assert_eq!(m.normalized_plan, BundleManifest::NORMALIZED_PLAN_PATH);
        assert!(m.started_at.is_none());
        assert!(m.outputs.is_empty());
    }

    #[test]
    fn status_wire_form_is_kebab_case() {
        // Matches the `AGENT_VERIFICATION` §5.1 example string.
        let v = serde_json::to_value(RunStatus::AssertionFailed).unwrap();
        assert_eq!(v, json!("assertion-failed"));
    }

    #[test]
    fn platform_omits_unknown_device_fields() {
        let p = Platform {
            os: "linux".to_owned(),
            arch: "x86_64".to_owned(),
            cpu: None,
            gpu: None,
            driver: None,
        };
        let v = serde_json::to_value(&p).unwrap();
        let obj = v.as_object().unwrap();
        assert!(obj.contains_key("os"));
        assert!(obj.contains_key("arch"));
        assert!(!obj.contains_key("cpu"));
        assert!(!obj.contains_key("gpu"));
        assert!(!obj.contains_key("driver"));
    }

    #[test]
    fn current_platform_fills_os_and_arch() {
        let p = Platform::current();
        assert!(!p.os.is_empty());
        assert!(!p.arch.is_empty());
    }

    #[test]
    fn manifest_round_trips_through_serde() {
        let manifest = BundleManifest::new(
            "0.0.0+sha",
            "blake3:abc",
            DeterminismTier::Exact,
            RunStatus::Success,
        )
        .with_started_at("2026-06-20T18:42:10Z")
        .with_outputs(vec![OutputEntry {
            name: "result".to_owned(),
            path: "outputs/result.png".to_owned(),
            content_hash: Some("blake3:dead".to_owned()),
        }])
        .with_failures(vec!["a".to_owned()]);
        let v = serde_json::to_value(&manifest).unwrap();
        let back: BundleManifest = serde_json::from_value(v).unwrap();
        assert_eq!(back, manifest);
    }

    #[test]
    fn unknown_manifest_field_is_rejected() {
        let v = json!({
            "paintop_runtime": "rt",
            "plan_semantic_hash": "blake3:00",
            "normalized_plan": "normalized-plan.json",
            "platform": {"os": "linux", "arch": "x86_64"},
            "determinism": "exact",
            "status": "success",
            "exit_code": 0,
            "surprise": true
        });
        assert!(serde_json::from_value::<BundleManifest>(v).is_err());
    }
}
