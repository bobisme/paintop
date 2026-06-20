//! The `xtask verify-op <op-id>` report runner (`AGENT_VERIFICATION` §14).
//!
//! `verify-op` automates an operation's *definition of done*: it locates and
//! validates the op manifest, derives which verification categories apply
//! (`paintop_ir::verify`), evaluates the stages it can in M0, and writes a
//! report tree under `target/verification/<op-id>/`:
//!
//! ```text
//! target/verification/<op-id>/
//! ├── index.json          machine-readable per-category result + overall pass
//! ├── summary.md          human-readable digest
//! ├── test-results.json   the per-stage outcomes
//! ├── differential/       (per §14 layout; populated by later stage bones)
//! ├── properties/
//! └── benchmarks/
//! ```
//!
//! # What runs in M0
//!
//! The full §14 stage list (run unit/property/metamorphic/differential tests,
//! benchmark, render a contact sheet) needs the executor and the per-op test
//! suites that land in later segments. This bone wires the *runner*: the report
//! layout, the index/summary/results emitters, and the category gate. The
//! stages that have a real M0 oracle are evaluated for real:
//!
//! - **build-hygiene** and **schema-contract** map to "the manifest loads,
//!   parses under `deny_unknown_fields`, and passes
//!   [`OperationManifest::validate`]". A manifest that fails to load fails the
//!   whole run before any report is written.
//! - the remaining categories are reported from the manifest's *declarations*
//!   (`paintop_ir::verify::VerificationDeclarations`): an applicable category
//!   that is neither covered nor not-applicable-with-a-reason is a **missing**
//!   category and fails the run; a covered category is reported `pass`; a
//!   justified not-applicable is reported `skipped` with its reason.
//!
//! The command exits non-zero (a non-panicking [`anyhow::Error`]) whenever the
//! run does not pass, so CI and the agent loop get a clear signal.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow, bail};
use paintop_ir::manifest::{OpId, OperationManifest};
use paintop_ir::verify::{CategoryStatus, VerificationCategory};
use serde::Serialize;

/// The default report root, relative to the workspace, matching the §14 layout
/// (`target/verification/<op-id>/`).
const DEFAULT_REPORT_ROOT: &str = "target/verification";

/// The §14 report subdirectories created eagerly so a later stage can drop an
/// artifact in without first checking the directory exists.
const REPORT_SUBDIRS: [&str; 3] = ["differential", "properties", "benchmarks"];

/// The outcome of one verification category as reported in the index/results.
///
/// Distinct from a bare bool so an *incomplete* op (missing an applicable
/// category) is reported as such rather than collapsing to "fail", which lets
/// the summary tell "this layer was tested and failed" from "this layer was
/// never declared".
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
enum CategoryOutcome {
    /// The category applies and is declared covered.
    Pass,
    /// The category does not apply (or is declared not-applicable) and carries a
    /// justification.
    Skipped,
    /// The category applies but is not declared — the op is incomplete.
    Missing,
}

impl CategoryOutcome {
    /// Whether this outcome blocks an overall pass. Only [`Missing`](Self::Missing)
    /// does: a justified skip and a covered category both pass the gate.
    const fn is_failure(self) -> bool {
        matches!(self, Self::Missing)
    }

    /// The stable token used in the human-readable summary.
    const fn token(self) -> &'static str {
        match self {
            Self::Pass => "pass",
            Self::Skipped => "skipped",
            Self::Missing => "MISSING",
        }
    }
}

/// One category's line in the report.
#[derive(Debug, Clone, Serialize)]
struct CategoryReport {
    /// The category's stable kebab token (e.g. `analytic-fixtures`).
    category: &'static str,
    /// The §2 layer number (0–9).
    layer: u8,
    /// Whether the category applies to this op (derived from the manifest).
    applicable: bool,
    /// The evaluated outcome.
    outcome: CategoryOutcome,
    /// The justification for a not-applicable / skipped category, if any.
    #[serde(skip_serializing_if = "Option::is_none")]
    reason: Option<String>,
}

/// The machine-readable `index.json` payload (`AGENT_VERIFICATION` §14, §6.1).
#[derive(Debug, Clone, Serialize)]
struct ReportIndex {
    /// The verified op id, canonical form.
    op: String,
    /// Overall pass/fail: false if any applicable category is missing.
    passed: bool,
    /// The number of applicable categories that are still undeclared.
    missing_categories: Vec<String>,
    /// The per-category breakdown, in spec (layer 0..=9) order.
    categories: Vec<CategoryReport>,
}

/// The `test-results.json` payload: the per-stage outcomes the §14 stage list
/// produces. In M0 the stages are derived from the category model; later bones
/// attach real test counts here.
#[derive(Debug, Clone, Serialize)]
struct TestResults {
    /// The verified op id.
    op: String,
    /// One entry per category/stage.
    stages: Vec<StageResult>,
}

/// One stage's result in `test-results.json`.
#[derive(Debug, Clone, Serialize)]
struct StageResult {
    /// The category/stage token.
    stage: &'static str,
    /// The stage outcome token (`pass` / `skipped` / `missing`).
    outcome: CategoryOutcome,
}

/// Options controlling a `verify-op` run, kept separate from CLI parsing so the
/// runner is unit-testable without a process.
#[derive(Debug, Clone)]
pub struct VerifyOpOptions {
    /// The op id to verify.
    pub op: String,
    /// Explicit path to the op manifest. M0 has no on-disk registry, so the
    /// manifest location is provided rather than discovered.
    pub manifest: PathBuf,
    /// The report root; defaults to `target/verification`.
    pub out_dir: Option<PathBuf>,
}

/// Run `verify-op`: locate and validate the manifest, evaluate the verification
/// categories, write the report tree, and fail if any applicable category is
/// missing.
///
/// # Errors
/// Returns an error if the manifest cannot be read/parsed/validated, if the
/// requested op id does not match the manifest's id, if the report tree cannot
/// be written, or — the gate — if any applicable verification category is
/// undeclared (an *incomplete* op). The message names the missing categories.
pub fn run(options: &VerifyOpOptions) -> Result<()> {
    let requested: OpId = options
        .op
        .parse()
        .with_context(|| format!("`{}` is not a valid op id", options.op))?;

    let manifest = load_manifest(&options.manifest)?;

    // The requested op id must match the manifest we were pointed at, so a
    // mislabeled report can never be written.
    if manifest.id != requested {
        bail!(
            "manifest {} declares op {}, but verify-op was asked to verify {}",
            options.manifest.display(),
            manifest.id,
            requested,
        );
    }

    let index = evaluate(&manifest);
    let results = stage_results(&manifest, &index);

    let root = options
        .out_dir
        .clone()
        .unwrap_or_else(|| PathBuf::from(DEFAULT_REPORT_ROOT))
        .join(manifest.id.to_string());
    write_report(&root, &index, &results)?;

    if index.passed {
        println!(
            "verify-op {}: PASS ({} categories) -> {}",
            index.op,
            index.categories.len(),
            root.display(),
        );
        Ok(())
    } else {
        // Clear, nonzero failure naming the gap (§14 step 10), with the written
        // report path so the agent can inspect it.
        Err(anyhow!(
            "verify-op {}: INCOMPLETE — missing required verification categories: {}. report: {}",
            index.op,
            index.missing_categories.join(", "),
            root.display(),
        ))
    }
}

/// Read, parse (enforcing `deny_unknown_fields`), and structurally validate the
/// manifest at `path`. This is §14 step 1 ("locate and validate manifest") and
/// covers the build-hygiene / schema-contract layers' M0 oracle.
fn load_manifest(path: &Path) -> Result<OperationManifest> {
    let bytes = std::fs::read(path)
        .with_context(|| format!("failed to read manifest {}", path.display()))?;
    let manifest: OperationManifest = serde_json::from_slice(&bytes)
        .with_context(|| format!("manifest {} failed schema parse", path.display()))?;
    manifest
        .validate()
        .with_context(|| format!("manifest {} is internally inconsistent", path.display()))?;
    Ok(manifest)
}

/// Evaluate every verification category against the manifest's declarations and
/// derived applicability, producing the `index.json` payload.
fn evaluate(manifest: &OperationManifest) -> ReportIndex {
    let declarations = &manifest.test.verification;
    let mut categories = Vec::with_capacity(VerificationCategory::ALL.len());
    let mut missing = Vec::new();

    for category in VerificationCategory::ALL {
        let applicable = category.is_applicable(manifest);
        let (outcome, reason) = match declarations.get(category) {
            Some(CategoryStatus::Covered) => (CategoryOutcome::Pass, None),
            Some(CategoryStatus::NotApplicable { reason }) => {
                (CategoryOutcome::Skipped, Some(reason.clone()))
            }
            None => {
                if applicable {
                    (CategoryOutcome::Missing, None)
                } else {
                    // Not applicable and undeclared: a benign gap, reported as a
                    // skip with a derived reason rather than a failure.
                    (
                        CategoryOutcome::Skipped,
                        Some("not applicable to this op".to_owned()),
                    )
                }
            }
            // `CategoryStatus` is `#[non_exhaustive]`: a future status we do not
            // recognise cannot be assumed to satisfy an applicable category, so
            // it is treated as missing (covered ones above are explicit).
            Some(_) => {
                if applicable {
                    (CategoryOutcome::Missing, None)
                } else {
                    (CategoryOutcome::Skipped, None)
                }
            }
        };
        if outcome.is_failure() {
            missing.push(category.as_str().to_owned());
        }
        categories.push(CategoryReport {
            category: category.as_str(),
            layer: category.layer(),
            applicable,
            outcome,
            reason,
        });
    }

    ReportIndex {
        op: manifest.id.to_string(),
        passed: missing.is_empty(),
        missing_categories: missing,
        categories,
    }
}

/// Project the category breakdown into the per-stage `test-results.json` view.
fn stage_results(manifest: &OperationManifest, index: &ReportIndex) -> TestResults {
    let stages = index
        .categories
        .iter()
        .map(|c| StageResult {
            stage: c.category,
            outcome: c.outcome,
        })
        .collect();
    TestResults {
        op: manifest.id.to_string(),
        stages,
    }
}

/// Create the report tree at `root` and write `index.json`, `summary.md`, and
/// `test-results.json`, plus the §14 stage subdirectories.
fn write_report(root: &Path, index: &ReportIndex, results: &TestResults) -> Result<()> {
    std::fs::create_dir_all(root)
        .with_context(|| format!("failed to create report dir {}", root.display()))?;
    for sub in REPORT_SUBDIRS {
        let dir = root.join(sub);
        std::fs::create_dir_all(&dir)
            .with_context(|| format!("failed to create report subdir {}", dir.display()))?;
    }

    let index_json = serde_json::to_string_pretty(index)
        .map_err(|e| anyhow!("failed to serialize index.json: {e}"))?;
    write_file(&root.join("index.json"), index_json.as_bytes())?;

    let results_json = serde_json::to_string_pretty(results)
        .map_err(|e| anyhow!("failed to serialize test-results.json: {e}"))?;
    write_file(&root.join("test-results.json"), results_json.as_bytes())?;

    write_file(&root.join("summary.md"), render_summary(index).as_bytes())?;
    Ok(())
}

/// Write `bytes` to `path`, wrapping the IO error with the path for context.
fn write_file(path: &Path, bytes: &[u8]) -> Result<()> {
    std::fs::write(path, bytes).with_context(|| format!("failed to write {}", path.display()))
}

/// Render the human-readable `summary.md` digest.
fn render_summary(index: &ReportIndex) -> String {
    use std::fmt::Write as _;

    let mut out = String::new();
    let verdict = if index.passed { "PASS" } else { "INCOMPLETE" };
    // Writing to a `String` is infallible, so the `write!` results are ignored.
    let _ = writeln!(out, "# verify-op {} — {verdict}\n", index.op);

    if index.passed {
        out.push_str("All applicable verification categories are covered or justified.\n\n");
    } else {
        let _ = writeln!(
            out,
            "Missing required verification categories: {}.\n",
            index.missing_categories.join(", ")
        );
    }

    out.push_str("| layer | category | applicable | outcome | reason |\n");
    out.push_str("|------:|----------|:----------:|---------|--------|\n");
    for c in &index.categories {
        let applicable = if c.applicable { "yes" } else { "no" };
        let reason = c.reason.as_deref().unwrap_or("");
        let _ = writeln!(
            out,
            "| {} | {} | {} | {} | {} |",
            c.layer,
            c.category,
            applicable,
            c.outcome.token(),
            reason,
        );
    }
    out
}

#[cfg(test)]
mod tests {
    use super::{CategoryOutcome, VerifyOpOptions, evaluate, run};
    use paintop_ir::manifest::{
        DeterminismTier, OperationManifest, OutputSpec, ResourceKind, RoiCategory, RoiPolicy,
        TestMetadata,
    };
    use paintop_ir::verify::{CategoryStatus, VerificationCategory, VerificationDeclarations};

    /// A minimal single-reference, exact op. Differential and perceptual do not
    /// apply; the other eight categories are applicable.
    fn base_manifest() -> OperationManifest {
        OperationManifest {
            id: "filter.invert@1".parse().unwrap(),
            impl_version: 1,
            summary: String::new(),
            determinism: DeterminismTier::Exact,
            roi: RoiPolicy {
                category: RoiCategory::Pointwise,
                halo_px: None,
            },
            inputs: vec![],
            outputs: vec![OutputSpec {
                name: "image".to_owned(),
                kind: ResourceKind::Image,
                doc: String::new(),
            }],
            params: vec![],
            implementations: vec!["cpu.reference@1".parse().unwrap()],
            test: TestMetadata::default(),
        }
    }

    /// Declarations covering every applicable category for `manifest`.
    fn cover_all(manifest: &OperationManifest) -> VerificationDeclarations {
        let mut decls = VerificationDeclarations::new();
        for category in VerificationCategory::applicable_to(manifest) {
            decls = decls.with(category, CategoryStatus::Covered);
        }
        decls
    }

    fn write_manifest(manifest: &OperationManifest) -> std::path::PathBuf {
        use std::sync::atomic::{AtomicU32, Ordering};
        static COUNTER: AtomicU32 = AtomicU32::new(0);
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir();
        let path = dir.join(format!(
            "paintop_verifyop_manifest_{}_{n}.json",
            std::process::id(),
        ));
        let json = serde_json::to_string(manifest).unwrap();
        std::fs::write(&path, json).unwrap();
        path
    }

    #[test]
    fn complete_op_passes_and_writes_a_report() {
        let mut m = base_manifest();
        m.test.verification = cover_all(&m);
        let manifest_path = write_manifest(&m);
        let out_dir = std::env::temp_dir().join(format!("paintop_vo_pass_{}", std::process::id()));

        let opts = VerifyOpOptions {
            op: "filter.invert@1".to_owned(),
            manifest: manifest_path.clone(),
            out_dir: Some(out_dir.clone()),
        };
        run(&opts).expect("a fully-covered op should pass verify-op");

        // The §14 report tree exists with the three top-level artifacts.
        let root = out_dir.join("filter.invert@1");
        let index = std::fs::read_to_string(root.join("index.json")).unwrap();
        assert!(index.contains("\"passed\": true"), "{index}");
        assert!(root.join("summary.md").exists());
        assert!(root.join("test-results.json").exists());
        for sub in ["differential", "properties", "benchmarks"] {
            assert!(root.join(sub).is_dir(), "missing subdir {sub}");
        }
        let summary = std::fs::read_to_string(root.join("summary.md")).unwrap();
        assert!(summary.contains("PASS"), "{summary}");

        let _ = std::fs::remove_file(manifest_path);
        let _ = std::fs::remove_dir_all(out_dir);
    }

    #[test]
    fn incomplete_op_reports_missing_categories_and_fails() {
        let mut m = base_manifest();
        let mut decls = cover_all(&m);
        // Drop an applicable category: the op is now incomplete.
        decls
            .by_category
            .remove(&VerificationCategory::PropertyTests);
        m.test.verification = decls;
        let manifest_path = write_manifest(&m);
        let out_dir = std::env::temp_dir().join(format!("paintop_vo_inc_{}", std::process::id()));

        let opts = VerifyOpOptions {
            op: "filter.invert@1".to_owned(),
            manifest: manifest_path.clone(),
            out_dir: Some(out_dir.clone()),
        };
        let err = run(&opts).expect_err("an incomplete op must fail verify-op");
        let msg = err.to_string();
        assert!(msg.contains("INCOMPLETE"), "{msg}");
        assert!(msg.contains("property-tests"), "{msg}");

        // The report is still written (so the agent can inspect the gap) and
        // records the missing category.
        let root = out_dir.join("filter.invert@1");
        let index = std::fs::read_to_string(root.join("index.json")).unwrap();
        assert!(index.contains("\"passed\": false"), "{index}");
        assert!(index.contains("property-tests"), "{index}");

        let _ = std::fs::remove_file(manifest_path);
        let _ = std::fs::remove_dir_all(out_dir);
    }

    #[test]
    fn justified_not_applicable_is_skipped_not_missing() {
        let mut m = base_manifest();
        let mut decls = cover_all(&m);
        decls = decls.with(
            VerificationCategory::AnalyticFixtures,
            CategoryStatus::not_applicable("pass-through op has no closed-form fixture"),
        );
        m.test.verification = decls;

        let index = evaluate(&m);
        assert!(index.passed);
        let analytic = index
            .categories
            .iter()
            .find(|c| c.category == "analytic-fixtures")
            .unwrap();
        assert_eq!(analytic.outcome, CategoryOutcome::Skipped);
        assert_eq!(
            analytic.reason.as_deref(),
            Some("pass-through op has no closed-form fixture")
        );
    }

    #[test]
    fn inapplicable_undeclared_category_is_a_benign_skip() {
        // Differential does not apply to a single-reference op and is undeclared;
        // it must be a skip, not a missing category.
        let mut m = base_manifest();
        m.test.verification = cover_all(&m);
        let index = evaluate(&m);
        assert!(index.passed);
        let differential = index
            .categories
            .iter()
            .find(|c| c.category == "differential")
            .unwrap();
        assert!(!differential.applicable);
        assert_eq!(differential.outcome, CategoryOutcome::Skipped);
    }

    #[test]
    fn mismatched_op_id_is_rejected() {
        let mut m = base_manifest();
        m.test.verification = cover_all(&m);
        let manifest_path = write_manifest(&m);
        let opts = VerifyOpOptions {
            op: "filter.brighten@1".to_owned(),
            manifest: manifest_path.clone(),
            out_dir: Some(std::env::temp_dir()),
        };
        let err = run(&opts).expect_err("a mismatched op id must be rejected");
        assert!(err.to_string().contains("but verify-op was asked"), "{err}");
        let _ = std::fs::remove_file(manifest_path);
    }

    #[test]
    fn missing_manifest_file_fails_cleanly() {
        let opts = VerifyOpOptions {
            op: "filter.invert@1".to_owned(),
            manifest: std::path::PathBuf::from("/nonexistent/paintop/manifest.json"),
            out_dir: Some(std::env::temp_dir()),
        };
        let err = run(&opts).expect_err("a missing manifest must fail");
        assert!(err.to_string().contains("failed to read manifest"), "{err}");
    }
}
