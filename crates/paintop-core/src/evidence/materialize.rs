//! Failure-driven materialization (`plan.md` §18.2).
//!
//! Paintop does **not** write every intermediate by default — that would bury
//! the signal and bloat every bundle. Instead, *on a failure* (an assertion or
//! differential violation) it materializes a self-contained bug specimen: the
//! diff that isolates the change, the minimal replay that reproduces it, and a
//! contact sheet that contextualizes it (`plan.md` §18.2). This module is the
//! **hook** that, given the failure inputs, writes those artifacts through the
//! [`BundleWriter`] and reports back where they landed so the assertion entry can
//! point at them.
//!
//! The hook is intentionally a thin orchestrator over the artifact builders:
//! the executor / assertion stage (later bones) decides *that* a failure
//! occurred and supplies the panels + reduced-plan inputs; this module decides
//! *what files* a failure produces and writes them atomically. Every artifact is
//! optional — if an input is missing the corresponding file is simply absent, and
//! the returned [`AssertionArtifacts`] omits it (`plan.md` §15.1 acceptance).

use paintop_ir::Plan;

use crate::evidence::assertions::AssertionArtifacts;
use crate::evidence::contact::{ContactSheet, Panel};
use crate::evidence::error::BundleResult;
use crate::evidence::replay::{MinimalReplay, ReplaySpec};
use crate::evidence::writer::BundleWriter;

/// The inputs a failed assertion offers up for materialization (`plan.md`
/// §18.2).
///
/// Everything is optional: a failure detected before the rasters are available
/// can still emit a replay, and vice versa. The hook materializes exactly the
/// artifacts whose inputs are present.
#[derive(Debug, Clone, Default)]
pub struct FailureInputs {
    /// The before (input) raster of the failing node, if captured.
    pub before: Option<Panel>,
    /// The after (output) raster of the failing node, if captured.
    pub after: Option<Panel>,
    /// The pinned replay context for reducing the plan, if known.
    pub replay: Option<ReplaySpec>,
}

impl FailureInputs {
    /// Attach the before/after panels.
    #[must_use]
    pub fn with_panels(mut self, before: Panel, after: Panel) -> Self {
        self.before = Some(before);
        self.after = Some(after);
        self
    }

    /// Attach the replay context.
    #[must_use]
    pub fn with_replay(mut self, replay: ReplaySpec) -> Self {
        self.replay = Some(replay);
        self
    }
}

/// Materialize one assertion's failure artifacts against `plan` (`plan.md`
/// §18.2).
///
/// Writes each artifact through `writer` and returns the bundle-relative paths
/// to stamp into the assertion's [`artifacts`](AssertionArtifacts) block.
///
/// Writes, when the corresponding input is present:
/// * `contact-sheet.png` — the before/after/diff sheet (needs both panels);
/// * `diffs/<id>-outside.png` — the standalone diff panel (needs both panels);
/// * `replays/<id>.json` — the minimal replay reducing `plan` to the target cone.
///
/// A missing input yields an absent file and an unset field in the returned
/// block, never a malformed placeholder.
///
/// # Errors
/// Returns the first [`BundleError`](crate::evidence::error::BundleError) from
/// any artifact write (serialize or I/O).
pub fn materialize_failure(
    writer: &BundleWriter,
    plan: &Plan,
    inputs: &FailureInputs,
) -> BundleResult<AssertionArtifacts> {
    let mut artifacts = AssertionArtifacts::default();

    if let (Some(before), Some(after)) = (&inputs.before, &inputs.after) {
        let sheet = ContactSheet::compose(before, after);
        artifacts.contact_sheet = Some(writer.write_contact_sheet(&sheet)?);

        // The standalone "outside diff" is the diff panel on its own, keyed by
        // the assertion id so multiple failures do not collide.
        if let Some(spec) = &inputs.replay
            && let Some(png) = Panel::diff_panel(before, after).encode_png()
        {
            let path = format!("diffs/{}-outside.png", spec.assertion);
            writer.write_bytes(&path, &png)?;
            artifacts.outside_diff = Some(path);
        }
    }

    if let Some(spec) = &inputs.replay {
        let replay = MinimalReplay::reduce(plan, spec.clone());
        artifacts.minimal_replay = Some(writer.write_replay(&replay)?);
    }

    Ok(artifacts)
}

#[cfg(test)]
mod tests {
    use super::{FailureInputs, materialize_failure};
    use crate::evidence::contact::Panel;
    use crate::evidence::replay::ReplaySpec;
    use crate::evidence::writer::BundleWriter;
    use paintop_ir::parse_plan;
    use std::sync::atomic::{AtomicU64, Ordering};

    fn scratch_dir(tag: &str) -> std::path::PathBuf {
        static SEQ: AtomicU64 = AtomicU64::new(0);
        let seq = SEQ.fetch_add(1, Ordering::Relaxed);
        let mut dir = std::env::temp_dir();
        dir.push(format!(
            "paintop-materialize-{}-{tag}-{seq}",
            std::process::id()
        ));
        dir
    }

    fn solid(w: u32, h: u32, c: [u8; 4]) -> Panel {
        let rgba = c
            .iter()
            .copied()
            .cycle()
            .take((w as usize) * (h as usize) * 4)
            .collect();
        Panel::new(w, h, rgba).expect("panel")
    }

    const PLAN: &str = r#"{
        "paintop": "1.0",
        "inputs": {"src": {"kind": "image.file", "path": "src.png"}},
        "nodes": [{"id": "blur", "op": "filter.gaussian_blur@1", "in": {"image": "input:src"}}],
        "exports": {"result": {"from": "node:blur/image"}}
    }"#;

    #[test]
    fn full_failure_materializes_every_artifact() {
        let dir = scratch_dir("full");
        let writer = BundleWriter::create(&dir).expect("writer");
        let plan = parse_plan(PLAN).expect("parse");
        let inputs = FailureInputs::default()
            .with_panels(
                solid(2, 2, [0, 0, 0, 255]),
                solid(2, 2, [255, 255, 255, 255]),
            )
            .with_replay(ReplaySpec::new("localized", "blur").with_seed(7));

        let artifacts = materialize_failure(&writer, &plan, &inputs).expect("materialize");

        let contact = artifacts.contact_sheet.expect("contact sheet path");
        let diff = artifacts.outside_diff.expect("diff path");
        let replay = artifacts.minimal_replay.expect("replay path");
        assert_eq!(replay, "replays/localized.json");
        assert_eq!(diff, "diffs/localized-outside.png");

        // Every promised file is on disk.
        assert!(writer.layout().join(&contact).exists());
        assert!(writer.layout().join(&diff).exists());
        assert!(writer.layout().join(&replay).exists());

        // The replay file re-parses as a plan-shaped document.
        let bytes = std::fs::read(writer.layout().join(&replay)).expect("read replay");
        let value: serde_json::Value = serde_json::from_slice(&bytes).expect("json");
        assert!(value["plan"]["nodes"].is_array());
        assert_eq!(value["spec"]["assertion"], "localized");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn missing_panels_skip_image_artifacts_but_still_replay() {
        let dir = scratch_dir("partial");
        let writer = BundleWriter::create(&dir).expect("writer");
        let plan = parse_plan(PLAN).expect("parse");
        let inputs = FailureInputs::default().with_replay(ReplaySpec::new("localized", "blur"));

        let artifacts = materialize_failure(&writer, &plan, &inputs).expect("materialize");
        assert!(artifacts.contact_sheet.is_none());
        assert!(artifacts.outside_diff.is_none());
        assert!(artifacts.minimal_replay.is_some());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn no_inputs_materializes_nothing() {
        let dir = scratch_dir("empty");
        let writer = BundleWriter::create(&dir).expect("writer");
        let plan = parse_plan(PLAN).expect("parse");
        let artifacts =
            materialize_failure(&writer, &plan, &FailureInputs::default()).expect("materialize");
        assert!(artifacts.is_empty());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
