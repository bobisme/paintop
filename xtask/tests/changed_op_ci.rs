//! Integration coverage for the changed-op CI wiring (`AGENT_VERIFICATION`
//! §8.1, §14): the `ci/verify-changed-ops.sh` driver that the `changed-op` CI
//! job runs.
//!
//! These tests drive the *real* shell script with `XTASK_BIN` pointed at the
//! test binary's compiled `xtask`, so the exact CI code path is exercised: a
//! complete manifest passes (and leaves a `target/verification/<op-id>/` report
//! for upload), a deliberately-incomplete manifest fails with the
//! missing-category message, and an empty change set is a clean no-op.
//!
//! The tests are skipped (not failed) when `bash` or `jq` are unavailable, so
//! the suite stays green on hosts without them; CI runners provide both.

use std::path::{Path, PathBuf};
use std::process::Command;

use paintop_ir::manifest::{
    DeterminismTier, OperationManifest, OutputSpec, ResourceKind, RoiCategory, RoiPolicy,
    TestMetadata,
};
use paintop_ir::verify::{CategoryStatus, VerificationCategory, VerificationDeclarations};

/// Absolute path to the checked-in `ci/verify-changed-ops.sh` driver.
fn script_path() -> PathBuf {
    // CARGO_MANIFEST_DIR is `<repo>/xtask`; the script lives at `<repo>/ci/...`.
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("xtask has a parent dir")
        .join("ci")
        .join("verify-changed-ops.sh")
}

/// Whether `tool` is runnable, so a host missing `bash`/`jq` skips rather than
/// fails.
fn have(tool: &str) -> bool {
    Command::new(tool)
        .arg("--version")
        .output()
        .is_ok_and(|o| o.status.success())
}

/// A minimal complete `color.convert@1`-shaped manifest: a single-reference,
/// bounded op with every applicable verification category declared covered.
fn complete_manifest() -> OperationManifest {
    let mut m = OperationManifest {
        id: "color.convert@1".parse().expect("valid op id"),
        impl_version: 1,
        summary: String::new(),
        determinism: DeterminismTier::Bounded,
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
        implementations: vec!["cpu.reference@1".parse().expect("valid impl id")],
        test: TestMetadata::default(),
    };
    let mut decls = VerificationDeclarations::new();
    for category in VerificationCategory::applicable_to(&m) {
        decls = decls.with(category, CategoryStatus::Covered);
    }
    m.test.verification = decls;
    m
}

/// Write `manifest` to `dir/<op-id>.json` (the checked-in `ops/manifests/`
/// naming convention) and return the path.
fn write_manifest(dir: &Path, manifest: &OperationManifest) -> PathBuf {
    let path = dir.join(format!("{}.json", manifest.id));
    let json = serde_json::to_string_pretty(manifest).expect("manifest serializes");
    std::fs::write(&path, json).expect("write manifest");
    path
}

/// Run the driver with `CHANGED_MANIFESTS=manifests`, the test's xtask binary,
/// and an isolated `target/verification` working dir. Returns success + combined
/// output.
fn run_driver(work: &Path, manifests: &str) -> (bool, String) {
    let output = Command::new("bash")
        .arg(script_path())
        .current_dir(work)
        .env("CHANGED_MANIFESTS", manifests)
        .env("XTASK_BIN", env!("CARGO_BIN_EXE_xtask"))
        .output()
        .expect("spawn bash driver");
    let mut combined = String::from_utf8_lossy(&output.stdout).into_owned();
    combined.push_str(&String::from_utf8_lossy(&output.stderr));
    (output.status.success(), combined)
}

#[test]
fn complete_manifest_passes_and_writes_report() {
    if !have("bash") || !have("jq") {
        eprintln!("skipping: bash/jq unavailable");
        return;
    }
    let work = tempdir("paintop_ci_pass");
    let manifest = write_manifest(&work, &complete_manifest());

    let (ok, out) = run_driver(&work, manifest.to_str().expect("utf-8 path"));
    assert!(ok, "complete manifest should pass; output:\n{out}");

    // The report tree is left under target/verification/<op-id>/ for upload.
    let report = work
        .join("target")
        .join("verification")
        .join("color.convert@1");
    assert!(
        report.join("index.json").is_file(),
        "expected index.json under {}; output:\n{out}",
        report.display()
    );

    cleanup(&work);
}

#[test]
fn incomplete_manifest_fails_with_missing_category() {
    if !have("bash") || !have("jq") {
        eprintln!("skipping: bash/jq unavailable");
        return;
    }
    let work = tempdir("paintop_ci_incomplete");
    let mut m = complete_manifest();
    // Drop an applicable category: the op is now incomplete.
    m.test
        .verification
        .by_category
        .remove(&VerificationCategory::PropertyTests);
    let manifest = write_manifest(&work, &m);

    let (ok, out) = run_driver(&work, manifest.to_str().expect("utf-8 path"));
    assert!(
        !ok,
        "incomplete manifest must fail the driver; output:\n{out}"
    );
    assert!(
        out.contains("property-tests"),
        "failure should name the missing category; output:\n{out}"
    );

    cleanup(&work);
}

#[test]
fn empty_change_set_is_a_clean_no_op() {
    if !have("bash") {
        eprintln!("skipping: bash unavailable");
        return;
    }
    let work = tempdir("paintop_ci_empty");
    let (ok, out) = run_driver(&work, "");
    assert!(ok, "empty change set should succeed; output:\n{out}");
    cleanup(&work);
}

/// Create a fresh per-test working directory under the system temp dir.
fn tempdir(tag: &str) -> PathBuf {
    use std::sync::atomic::{AtomicU32, Ordering};
    static COUNTER: AtomicU32 = AtomicU32::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!("{tag}_{}_{n}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("create temp work dir");
    dir
}

fn cleanup(dir: &Path) {
    let _ = std::fs::remove_dir_all(dir);
}
