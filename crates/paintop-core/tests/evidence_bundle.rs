//! Integration coverage for the evidence-bundle manifest schema + atomic writer
//! (`plan.md` §15.1, `AGENT_VERIFICATION` §5).
//!
//! The bone's exit gate: a trivial successful run writes a *valid, complete*
//! bundle with no partial canonical artifacts. These tests drive
//! [`BundleWriter`] end to end and assert the directory structure, manifest
//! fields (provenance excluded from semantic identity), canonical artifact
//! bytes, and atomic-write behavior.

use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use paintop_core::evidence::{
    BundleManifest, BundleWriter, RunStatus,
    layout::{SUBDIRS, files},
};
use paintop_ir::{DeterminismTier, parse_plan, semantic_hash, to_canonical_bytes};

/// A unique scratch directory under the OS temp dir, removed on drop so the test
/// leaves nothing behind. Avoids pulling in an extra crate just for tests.
struct Scratch {
    path: PathBuf,
}

impl Scratch {
    fn new(tag: &str) -> Self {
        static SEQ: AtomicU64 = AtomicU64::new(0);
        let seq = SEQ.fetch_add(1, Ordering::Relaxed);
        let mut path = std::env::temp_dir();
        path.push(format!(
            "paintop-evidence-test-{}-{}-{tag}-{seq}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(0, |d| d.as_nanos()),
        ));
        Self { path }
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.path);
    }
}

/// The minimal valid plan: the four required keys and nothing else.
const fn trivial_plan_json() -> &'static str {
    r#"{"paintop":"1.0","inputs":{},"nodes":[],"exports":{}}"#
}

#[test]
fn trivial_run_writes_a_valid_complete_bundle() {
    let scratch = Scratch::new("complete");
    let plan = parse_plan(trivial_plan_json()).expect("trivial plan parses");

    let writer = BundleWriter::create(&scratch.path).expect("bundle skeleton created");

    // Every well-known subdirectory exists after layout.
    for sub in SUBDIRS {
        let dir = scratch.path.join(sub);
        assert!(dir.is_dir(), "subdirectory `{sub}` was not created");
    }

    let hash = writer
        .write_normalized_plan(&plan)
        .expect("normalized plan written");
    let manifest = BundleManifest::new(
        "0.0.0+testsha",
        hash.clone(),
        DeterminismTier::Exact,
        RunStatus::Success,
    )
    .with_started_at("2026-06-20T18:42:10Z");
    writer.write_manifest(&manifest).expect("manifest written");

    // The two canonical artifacts exist.
    let manifest_path = scratch.path.join(files::MANIFEST);
    let plan_path = scratch.path.join(files::NORMALIZED_PLAN);
    assert!(manifest_path.is_file(), "manifest.json missing");
    assert!(plan_path.is_file(), "normalized-plan.json missing");

    // Optional artifacts that this bone does not write are simply *absent*, not
    // malformed placeholders.
    for optional in [files::TRACE, files::METRICS, files::ASSERTIONS] {
        assert!(
            !scratch.path.join(optional).exists(),
            "optional artifact `{optional}` should be absent until its stage writes it"
        );
    }

    // The manifest re-parses and carries the expected fields.
    let manifest_bytes = fs::read(&manifest_path).expect("read manifest");
    let parsed: BundleManifest =
        serde_json::from_slice(&manifest_bytes).expect("manifest re-parses");
    assert_eq!(parsed, manifest);
    assert_eq!(parsed.status, RunStatus::Success);
    assert_eq!(parsed.exit_code, 0);
    assert_eq!(parsed.plan_semantic_hash, hash);
    assert!(parsed.plan_semantic_hash.starts_with("blake3:"));
    assert_eq!(parsed.normalized_plan, files::NORMALIZED_PLAN);
    assert!(parsed.outputs.is_empty());
    assert!(parsed.failures.is_empty());
}

#[test]
fn normalized_plan_artifact_reparses_to_same_semantic_hash() {
    let scratch = Scratch::new("reparse");
    let plan = parse_plan(trivial_plan_json()).expect("plan parses");
    let writer = BundleWriter::create(&scratch.path).expect("skeleton");

    let written_hash = writer.write_normalized_plan(&plan).expect("plan written");

    // Read the on-disk artifact back, re-parse it, and confirm its semantic hash
    // matches — the canonical artifact is a faithful, stable record of the graph.
    let bytes = fs::read(scratch.path.join(files::NORMALIZED_PLAN)).expect("read plan");
    let reparsed = parse_plan(std::str::from_utf8(&bytes).expect("utf8")).expect("reparse");
    let reparsed_hash = semantic_hash(&reparsed).expect("hash").to_string();
    assert_eq!(written_hash, reparsed_hash);
    assert_eq!(
        written_hash,
        semantic_hash(&plan).expect("hash").to_string()
    );
}

#[test]
fn manifest_bytes_are_canonical_and_stable() {
    let scratch = Scratch::new("canonical");
    let writer = BundleWriter::create(&scratch.path).expect("skeleton");
    let manifest = BundleManifest::new(
        "0.0.0+testsha",
        "blake3:00",
        DeterminismTier::Bounded,
        RunStatus::Success,
    );

    writer.write_manifest(&manifest).expect("write");
    let on_disk = fs::read(scratch.path.join(files::MANIFEST)).expect("read");

    // The on-disk bytes are exactly the canonical emission of the manifest value:
    // sorted keys, single float format, no insignificant whitespace.
    let value = serde_json::to_value(&manifest).expect("to value");
    let expected = to_canonical_bytes(&value).expect("canonical");
    assert_eq!(on_disk, expected);

    // Re-writing the same manifest yields byte-identical output (atomic overwrite
    // of an already-present file).
    writer.write_manifest(&manifest).expect("rewrite");
    let again = fs::read(scratch.path.join(files::MANIFEST)).expect("read again");
    assert_eq!(again, on_disk);
}

#[test]
fn failed_empty_run_still_writes_a_valid_manifest_with_exit_code_and_failures() {
    let scratch = Scratch::new("failed");
    let writer = BundleWriter::create(&scratch.path).expect("skeleton");

    let manifest = BundleManifest::new(
        "0.0.0+testsha",
        "blake3:dead",
        DeterminismTier::Bounded,
        RunStatus::AssertionFailed,
    )
    .with_failures(vec!["localized".to_owned()]);
    writer.write_manifest(&manifest).expect("manifest written");

    let parsed: BundleManifest =
        serde_json::from_slice(&fs::read(scratch.path.join(files::MANIFEST)).expect("read"))
            .expect("reparse");
    assert_eq!(parsed.status, RunStatus::AssertionFailed);
    assert_eq!(parsed.exit_code, 6, "assertion failure exit code");
    assert_eq!(parsed.failures, vec!["localized".to_owned()]);
}

#[test]
fn no_temp_files_remain_after_writes() {
    let scratch = Scratch::new("notemp");
    let plan = parse_plan(trivial_plan_json()).expect("plan");
    let writer = BundleWriter::create(&scratch.path).expect("skeleton");
    let hash = writer.write_normalized_plan(&plan).expect("plan written");
    let manifest = BundleManifest::new("rt", hash, DeterminismTier::Exact, RunStatus::Success);
    writer.write_manifest(&manifest).expect("manifest");

    // A clean bundle root: no `.tmp.` sidecar from the temp-then-rename dance is
    // left behind once the writes complete.
    for entry in fs::read_dir(&scratch.path).expect("read root") {
        let name = entry.expect("entry").file_name();
        let name = name.to_string_lossy();
        assert!(
            !name.contains(".tmp."),
            "leftover temp file in bundle root: {name}"
        );
    }
}
