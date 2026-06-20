//! Integration coverage for the failure path of an evidence bundle
//! (`AGENT_VERIFICATION` §5.3/§5.4, `plan.md` §15.1/§18.2).
//!
//! This bone's exit gate: a *failing* plan, beyond the manifest, additionally
//! produces `assertions.json`, a minimal replay file under `replays/`, and a
//! contact sheet — and where inputs are unavailable those artifacts are simply
//! absent, never malformed. These tests drive the failure-driven materialization
//! hook end to end through [`BundleWriter`].

use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use paintop_core::evidence::{
    AssertionArtifacts, AssertionEntry, AssertionReport, AssertionStatus, BundleManifest,
    BundleWriter, ContactSheet, FailureInputs, MinimalReplay, Panel, ReplaySpec, RunStatus,
    layout::files, materialize_failure,
};
use paintop_ir::{DeterminismTier, parse_plan};

/// A unique scratch directory under the OS temp dir, removed on drop.
struct Scratch {
    path: PathBuf,
}

impl Scratch {
    fn new(tag: &str) -> Self {
        static SEQ: AtomicU64 = AtomicU64::new(0);
        let seq = SEQ.fetch_add(1, Ordering::Relaxed);
        let mut path = std::env::temp_dir();
        path.push(format!(
            "paintop-failure-test-{}-{tag}-{seq}",
            std::process::id(),
        ));
        Self { path }
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.path);
    }
}

/// A small plan with one external input and a producer node a failing assertion
/// can target.
const FAILING_PLAN: &str = r#"{
    "paintop": "1.0",
    "name": "repair",
    "inputs": {"src": {"kind": "image.file", "path": "src.png"}},
    "nodes": [
        {"id": "blur", "op": "filter.gaussian_blur@1", "in": {"image": "input:src"}}
    ],
    "exports": {"result": {"from": "node:blur/image"}}
}"#;

fn solid(w: u32, h: u32, c: [u8; 4]) -> Panel {
    let rgba = c
        .iter()
        .copied()
        .cycle()
        .take((w as usize) * (h as usize) * 4)
        .collect();
    Panel::new(w, h, rgba).expect("panel")
}

#[test]
fn failing_run_emits_manifest_assertions_replay_and_contact_sheet() {
    let scratch = Scratch::new("full");
    let plan = parse_plan(FAILING_PLAN).expect("plan parses");
    let writer = BundleWriter::create(&scratch.path).expect("skeleton");

    // The run failed an assertion: materialize the bug specimen.
    let spec = ReplaySpec::new("localized", "blur")
        .with_implementation("cpu.separable@1")
        .with_seed(7);
    let inputs = FailureInputs::default()
        .with_panels(
            solid(4, 4, [0, 0, 0, 255]),
            solid(4, 4, [255, 255, 255, 255]),
        )
        .with_replay(spec);
    let artifacts = materialize_failure(&writer, &plan, &inputs).expect("materialize");

    // The assertion report names the failure and points at the artifacts.
    let entry = AssertionEntry::new(
        "localized",
        "assert.no_change_outside_mask@1",
        AssertionStatus::Failed,
    )
    .with_artifacts(artifacts);
    let report = AssertionReport::from(vec![entry]);
    writer
        .write_assertions(&report)
        .expect("assertions written");

    // The manifest carries the assertion-failed status, exit code, and failure id.
    let manifest = BundleManifest::new(
        "0.0.0+testsha",
        "blake3:dead",
        DeterminismTier::Bounded,
        RunStatus::AssertionFailed,
    )
    .with_failures(report.failure_ids());
    writer.write_manifest(&manifest).expect("manifest written");

    // 1. assertions.json exists and re-parses, carrying the failing id.
    let assertions_path = scratch.path.join(files::ASSERTIONS);
    assert!(assertions_path.is_file(), "assertions.json missing");
    let parsed_report: AssertionReport =
        serde_json::from_slice(&fs::read(&assertions_path).expect("read")).expect("reparse");
    assert_eq!(parsed_report.failure_ids(), vec!["localized".to_owned()]);

    // 2. a minimal replay file exists under replays/ and re-parses to a plan.
    let replay_rel = MinimalReplay::path_for("localized");
    let replay_path = scratch.path.join(&replay_rel);
    assert!(replay_path.is_file(), "replay file missing");
    let replay: MinimalReplay =
        serde_json::from_slice(&fs::read(&replay_path).expect("read")).expect("replay reparse");
    assert_eq!(replay.spec.assertion, "localized");
    assert_eq!(
        replay.spec.implementation.as_deref(),
        Some("cpu.separable@1")
    );
    assert_eq!(replay.plan.nodes.len(), 1);
    assert_eq!(replay.plan.nodes[0].id, "blur");

    // 3. a contact sheet exists and is a real PNG.
    let sheet_path = scratch.path.join(files::CONTACT_SHEET);
    assert!(sheet_path.is_file(), "contact-sheet.png missing");
    let png = fs::read(&sheet_path).expect("read sheet");
    assert!(png.starts_with(&[0x89, b'P', b'N', b'G']));

    // The outside diff was also materialized and is referenced from the report.
    assert_eq!(
        parsed_report.assertions[0]
            .artifacts
            .minimal_replay
            .as_deref(),
        Some(replay_rel.as_str())
    );
    assert!(
        scratch
            .path
            .join(
                parsed_report.assertions[0]
                    .artifacts
                    .outside_diff
                    .as_deref()
                    .expect("diff path")
            )
            .is_file()
    );

    // The manifest re-parses with the failure status and exit code.
    let manifest_back: BundleManifest =
        serde_json::from_slice(&fs::read(scratch.path.join(files::MANIFEST)).expect("read"))
            .expect("manifest reparse");
    assert_eq!(manifest_back.status, RunStatus::AssertionFailed);
    assert_eq!(manifest_back.exit_code, 6);
    assert_eq!(manifest_back.failures, vec!["localized".to_owned()]);
}

#[test]
fn missing_inputs_leave_optional_artifacts_absent_not_malformed() {
    let scratch = Scratch::new("absent");
    let plan = parse_plan(FAILING_PLAN).expect("plan parses");
    let writer = BundleWriter::create(&scratch.path).expect("skeleton");

    // No panels: only the replay can be materialized.
    let inputs = FailureInputs::default().with_replay(ReplaySpec::new("localized", "blur"));
    let artifacts = materialize_failure(&writer, &plan, &inputs).expect("materialize");
    assert_eq!(artifacts.outside_diff, None);
    assert_eq!(artifacts.contact_sheet, None);
    assert!(artifacts.minimal_replay.is_some());

    // The contact sheet and diff files are simply absent.
    assert!(!scratch.path.join(files::CONTACT_SHEET).exists());
    assert!(!scratch.path.join("diffs/localized-outside.png").exists());
    // The replay that *was* possible is present.
    assert!(
        scratch
            .path
            .join(MinimalReplay::path_for("localized"))
            .is_file()
    );
}

#[test]
fn assertion_artifacts_block_omits_unset_paths_on_disk() {
    // A passing report writes a terse assertions.json with no artifacts noise.
    let scratch = Scratch::new("passing");
    let writer = BundleWriter::create(&scratch.path).expect("skeleton");
    let report = AssertionReport::from(vec![AssertionEntry::new(
        "localized",
        "assert.no_change_outside_mask@1",
        AssertionStatus::Passed,
    )]);
    writer.write_assertions(&report).expect("written");

    let value: serde_json::Value =
        serde_json::from_slice(&fs::read(scratch.path.join(files::ASSERTIONS)).expect("read"))
            .expect("json");
    let entry = &value["assertions"][0];
    assert_eq!(entry["status"], "passed");
    assert!(entry.get("artifacts").is_none(), "no empty artifacts block");
    assert!(entry.get("metrics").is_none(), "no empty metrics bag");
}

#[test]
fn contact_sheet_round_trips_through_writer() {
    let scratch = Scratch::new("sheet");
    let writer = BundleWriter::create(&scratch.path).expect("skeleton");
    let sheet = ContactSheet::compose(
        &solid(2, 2, [10, 20, 30, 255]),
        &solid(2, 2, [40, 50, 60, 255]),
    );
    let rel = writer.write_contact_sheet(&sheet).expect("write sheet");
    assert_eq!(rel, "contact-sheet.png");

    // The diff artifact `AssertionArtifacts` can carry exactly that path.
    let artifacts = AssertionArtifacts {
        contact_sheet: Some(rel.clone()),
        ..AssertionArtifacts::default()
    };
    assert_eq!(
        artifacts.contact_sheet.as_deref(),
        Some("contact-sheet.png")
    );
    assert!(scratch.path.join(&rel).is_file());
}
