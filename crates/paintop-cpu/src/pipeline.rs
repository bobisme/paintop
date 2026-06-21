//! The end-to-end run orchestrator: parse a plan, execute the MVP op graph
//! whole-image, evaluate assertions, and write a complete evidence bundle.
//!
//! This is the seam the CLI's `paintop run` and the conformance test both drive.
//! It composes the pieces other bones built — the IR parser/normalizer/resolver
//! ([`paintop_ir`]), the whole-image executor ([`paintop_core::executor`]), the
//! real MVP op registries ([`crate::registry`]), and the evidence-bundle writer
//! ([`paintop_core::evidence`]) — into one deterministic call.
//!
//! # What a run produces
//!
//! Given a plan and a bundle directory, [`run_plan`] writes (`plan.md` §15.1):
//!
//! * `manifest.json` — semantic hash, status, exit code, outputs, failure ids;
//! * `normalized-plan.json` — the exact §17-normalized graph that ran;
//! * `input-manifest.json` — the decoded inputs' extents and content hashes;
//! * `graph.dot` — the dependency graph;
//! * `trace.jsonl` — the per-node structured trace;
//! * `assertions.json` — every assertion's verdict, thresholds, and metrics;
//! * `outputs/` — the encoded export images;
//! * `masks/` and `intermediates/` — requested materializations;
//! * `diffs/` and `contact-sheet.png` — the before/after evidence;
//! * `replays/<id>.json` — a minimal reproducer for each failed assertion.
//!
//! # Determinism
//!
//! Every stage is a deterministic function of the plan and its input files: the
//! executor runs sequentially in the resolved topological order, the PNG encoders
//! pin compression/filter, and the artifact writers canonicalize their JSON. Two
//! runs of the same plan therefore produce a byte-identical output image and an
//! identical plan semantic hash — the reproducibility the keystone asserts.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use paintop_core::evidence::layout::{dirs, files};
use paintop_core::evidence::{
    AssertionEntry, AssertionReport, AssertionStatus, BundleManifest, BundleWriter, ContactSheet,
    MetricBag, MinimalReplay, OutputEntry, Panel, Platform, ReplaySpec, RunStatus, TraceEvent,
    TraceWriter,
};
use paintop_core::executor::{Execution, ResourceValue, execute};
use paintop_ir::{
    Error, ErrorClass, HashDomain, Plan, Reference, Report, ResolvedGraph, hash_canonical_bytes,
    parse_plan, resolve_plan, semantic_hash,
};

use crate::io::encode_png;
use crate::registry::{implementation_registry, operation_registry};

/// The runtime build string stamped into the manifest's provenance.
const RUNTIME: &str = concat!("paintop-cpu/", env!("CARGO_PKG_VERSION"));

/// The terminal outcome of a [`run_plan`] call.
///
/// Carries the bundle root, the plan semantic hash, the overall status (and its
/// stable exit code), the content hash of the primary output, and the ids of any
/// failed assertions, so a caller can report the run and assert reproducibility
/// without re-reading the bundle.
#[derive(Debug, Clone)]
pub struct RunOutcome {
    /// The bundle directory the artifacts were written to.
    pub bundle: PathBuf,
    /// The plan's `blake3:` semantic hash (`AGENT_VERIFICATION` §5.1).
    pub plan_semantic_hash: String,
    /// The terminal run status.
    pub status: RunStatus,
    /// The stable process exit code derived from [`status`](Self::status).
    pub exit_code: i32,
    /// The content hash of the primary encoded output, if one was produced.
    pub output_content_hash: Option<String>,
    /// The stable ids of failed assertions, in plan order.
    pub failures: Vec<String>,
}

impl RunOutcome {
    /// Whether the run fully succeeded (status `success`, exit code `0`).
    #[must_use]
    pub const fn ok(&self) -> bool {
        self.status.is_success()
    }
}

/// Run `plan_path` end-to-end and write a complete evidence bundle to `bundle`.
///
/// Parses and normalizes the plan, resolves it against the MVP op registry,
/// executes the demanded subgraph whole-image, evaluates every assertion node,
/// and writes the bundle. The returned [`RunOutcome`] reports the status, exit
/// code, and reproducibility hashes.
///
/// # Errors
/// Returns the plan's native IR [`Error`] for a parse / resolve / execution
/// failure (the caller maps it to the matching exit class), or a bundle error
/// surfaced as an [`export`](ErrorClass::Export) error if an artifact cannot be
/// written. A *failed assertion* is **not** an error: it is reported in the
/// outcome with status [`RunStatus::AssertionFailed`] and exit code `6`, and the
/// bundle is written in full.
pub fn run_plan(plan_path: &Path, bundle: &Path) -> Result<RunOutcome, Error> {
    let text = std::fs::read_to_string(plan_path).map_err(|e| {
        Error::new(
            ErrorClass::Asset,
            "E_PLAN_READ_FAILED",
            format!("failed to read plan {}: {e}", plan_path.display()),
        )
    })?;
    let plan = parse_plan(&text)?;
    run_parsed_plan(&plan, bundle)
}

/// Run an already-parsed `plan`, writing its bundle to `bundle`. Shared by
/// [`run_plan`] and tests that build the plan in memory.
///
/// # Errors
/// As [`run_plan`].
pub fn run_parsed_plan(plan: &Plan, bundle: &Path) -> Result<RunOutcome, Error> {
    let registry = operation_registry()?;
    let implementations = implementation_registry()?;
    let graph = resolve_plan(plan, &registry)?;
    let hash = semantic_hash(plan)?.to_string();

    // No external `input:` resources flow into the MVP loop — every source is an
    // `io.decode_image` node reading a file param — so the executor's external
    // input map is empty.
    let inputs: BTreeMap<String, ResourceValue> = BTreeMap::new();
    let execution = execute(plan, &graph, &registry, &implementations, &inputs)?;

    let report = collect_assertions(plan, &execution);
    let status = if report.all_passed() {
        RunStatus::Success
    } else {
        RunStatus::AssertionFailed
    };

    let writer = BundleWriter::create(bundle).map_err(bundle_err)?;
    writer.write_normalized_plan(plan).map_err(bundle_err)?;
    write_input_manifest(&writer, plan, &execution)?;
    write_trace(&writer, &hash, plan, &execution)?;
    write_graph_dot(&writer, &graph)?;
    let outputs = write_outputs(&writer, plan, &execution)?;
    write_materializations(&writer, plan, &execution)?;
    let (before, after) = export_panels(plan, &execution);
    write_contact_and_diffs(&writer, before.as_ref(), after.as_ref())?;
    writer.write_assertions(&report).map_err(bundle_err)?;
    write_replays(&writer, plan, &report)?;

    let manifest = BundleManifest::new(
        RUNTIME,
        hash.clone(),
        graph_determinism(&registry, plan),
        status,
    )
    .with_platform(Platform::current())
    .with_outputs(outputs.clone())
    .with_failures(report.failure_ids());
    writer.write_manifest(&manifest).map_err(bundle_err)?;

    Ok(RunOutcome {
        bundle: bundle.to_path_buf(),
        plan_semantic_hash: hash,
        status,
        exit_code: status.exit_code(),
        output_content_hash: outputs.first().and_then(|o| o.content_hash.clone()),
        failures: report.failure_ids(),
    })
}

/// Collect every assertion node's verdict into an [`AssertionReport`].
///
/// An assertion is an ordinary node whose output is a [`Report`] carrying an
/// [`AssertionOutcome`](paintop_ir::AssertionOutcome). We walk the plan's nodes in
/// declaration order, pull the produced report for each assertion op, and project
/// its verdict, thresholds, and metrics into the bundle's assertion schema.
fn collect_assertions(plan: &Plan, execution: &Execution) -> AssertionReport {
    let mut entries = Vec::new();
    for node in &plan.nodes {
        let op = node.op.clone();
        if !op.starts_with("assert.") {
            continue;
        }
        let Some(report) = execution
            .output(&node.id, "report")
            .and_then(ResourceValue::as_report)
        else {
            // A demanded-but-dead assertion (eliminated by demand) has no report;
            // record it as skipped so the bundle is honest about coverage.
            entries.push(AssertionEntry::new(
                node.id.clone(),
                op,
                AssertionStatus::Skipped,
            ));
            continue;
        };
        entries.push(assertion_entry(&node.id, &op, report));
    }
    AssertionReport::from(entries)
}

/// Project one assertion [`Report`] into a bundle [`AssertionEntry`].
fn assertion_entry(id: &str, op: &str, report: &Report) -> AssertionEntry {
    let Some(verdict) = report.assertion.as_ref() else {
        return AssertionEntry::new(id, op, AssertionStatus::Skipped);
    };
    let status = if verdict.passed {
        AssertionStatus::Passed
    } else {
        AssertionStatus::Failed
    };
    let mut metrics = MetricBag::new();
    if let Some(d) = verdict.max_abs_delta_outside {
        metrics.insert("max_abs_delta_outside".to_owned(), serde_json::json!(d));
    }
    if let Some(c) = verdict.changed_pixels_outside {
        metrics.insert("changed_pixels_outside".to_owned(), serde_json::json!(c));
    }
    if let Some(c) = verdict.nonfinite_count {
        metrics.insert("nonfinite_count".to_owned(), serde_json::json!(c));
    }
    if let Some(p) = verdict.worst_pixel {
        metrics.insert("worst_pixel".to_owned(), serde_json::json!(p));
    }
    if let Some(v) = verdict.violations {
        metrics.insert("violations".to_owned(), serde_json::json!(v));
    }
    if let Some(w) = verdict.worst_value {
        metrics.insert("worst_value".to_owned(), serde_json::json!(w));
    }
    if let Some(b) = verdict.changed_bounds {
        metrics.insert(
            "changed_bounds".to_owned(),
            serde_json::json!([b.x0, b.y0, b.x1, b.y1]),
        );
    }
    if let Some(b) = verdict.expected_bounds {
        metrics.insert(
            "expected_bounds".to_owned(),
            serde_json::json!([b.x0, b.y0, b.x1, b.y1]),
        );
    }
    AssertionEntry::new(id, op, status).with_metrics(metrics)
}

/// Whether the run as a whole is exact or bounded: bounded if any executed op is
/// bounded, else exact (`AGENT_VERIFICATION` §5.1 `determinism`).
fn graph_determinism(
    registry: &paintop_ir::OperationRegistry,
    plan: &Plan,
) -> paintop_ir::DeterminismTier {
    use paintop_ir::DeterminismTier;
    for node in &plan.nodes {
        let Ok(op_id) = node.op.parse() else {
            continue;
        };
        if let Ok(manifest) = registry.get(&op_id)
            && matches!(manifest.determinism, DeterminismTier::Bounded)
        {
            return DeterminismTier::Bounded;
        }
    }
    DeterminismTier::Exact
}

/// Write the `trace.jsonl`: a leading `plan_parsed` event then every executor
/// dispatch event, one canonical JSON object per line.
fn write_trace(
    writer: &BundleWriter,
    hash: &str,
    plan: &Plan,
    execution: &Execution,
) -> Result<(), Error> {
    let node_count = u64::try_from(plan.nodes.len()).unwrap_or(u64::MAX);
    let mut trace = TraceWriter::create(writer.layout()).map_err(bundle_err)?;
    trace
        .write_event(&TraceEvent::plan_parsed(hash, node_count))
        .map_err(bundle_err)?;
    for event in execution.trace() {
        trace.write_event(event).map_err(bundle_err)?;
    }
    Ok(())
}

/// Render and write the resolved graph as Graphviz DOT under `graph.dot`.
fn write_graph_dot(writer: &BundleWriter, graph: &ResolvedGraph) -> Result<(), Error> {
    writer
        .write_bytes(files::GRAPH_DOT, render_dot(graph).as_bytes())
        .map_err(bundle_err)
}

/// Render a resolved graph to a deterministic Graphviz DOT document.
fn render_dot(graph: &ResolvedGraph) -> String {
    use std::fmt::Write as _;
    let mut out = String::from("digraph paintop {\n");
    for (id, node) in graph.nodes() {
        let label = format!("{id}\n{}", node.op);
        let _ = writeln!(out, "  {id:?} [label={label:?}];");
    }
    for (id, node) in graph.nodes() {
        for reference in node.inputs.values() {
            if let paintop_ir::Reference::Node { node: src, .. } = reference {
                let _ = writeln!(out, "  {src:?} -> {id:?};");
            }
        }
    }
    out.push_str("}\n");
    out
}

/// Encode and write every export image under `outputs/`, recording each as a
/// manifest [`OutputEntry`] with its content hash.
fn write_outputs(
    writer: &BundleWriter,
    plan: &Plan,
    execution: &Execution,
) -> Result<Vec<OutputEntry>, Error> {
    let mut entries = Vec::new();
    for (name, value) in execution.exports() {
        // Only image exports are encodable in M0; a report export is recorded by
        // the assertion stage, not here.
        if value.as_report().is_some() {
            continue;
        }
        let png = encode_png(value)?;
        let relative = format!("{}/{name}.png", dirs::OUTPUTS);
        writer.write_bytes(&relative, &png).map_err(bundle_err)?;
        let hash = hash_canonical_bytes(HashDomain::Content, &png).to_string();
        entries.push(OutputEntry {
            name: name.clone(),
            path: relative,
            content_hash: Some(hash),
        });
    }
    let _ = plan;
    Ok(entries)
}

/// Write `input-manifest.json`: the decoded sources' extents and content hashes
/// (`AGENT_VERIFICATION` §5.1). Every `io.decode_image` node is a decoded source;
/// its produced image is summarized so a reader can confirm which bytes fed the
/// run.
fn write_input_manifest(
    writer: &BundleWriter,
    plan: &Plan,
    execution: &Execution,
) -> Result<(), Error> {
    let mut sources = Vec::new();
    for node in &plan.nodes {
        if node.op != crate::io::DECODE_OP_ID {
            continue;
        }
        let Some(value) = execution.output(&node.id, "image") else {
            continue;
        };
        let bytes: Vec<u8> = value
            .samples()
            .iter()
            .flat_map(|s| s.to_le_bytes())
            .collect();
        let content_hash = hash_canonical_bytes(HashDomain::Content, &bytes).to_string();
        let path = node
            .params
            .get("path")
            .and_then(serde_json::Value::as_str)
            .unwrap_or_default();
        sources.push(serde_json::json!({
            "node": node.id,
            "path": path,
            "extent": [value.extent().width, value.extent().height],
            "channels": value.channels(),
            "content_hash": content_hash,
        }));
    }
    writer
        .write_artifact(
            files::INPUT_MANIFEST,
            &serde_json::json!({ "sources": sources }),
        )
        .map_err(bundle_err)
}

/// Parse the `node:<id>/<port>` reference strings under a plan's
/// `evidence.<key>` array (e.g. `materialize`), discarding malformed entries.
fn evidence_references(plan: &Plan, key: &str) -> Vec<Reference> {
    plan.evidence
        .get(key)
        .and_then(serde_json::Value::as_array)
        .map(|entries| {
            entries
                .iter()
                .filter_map(serde_json::Value::as_str)
                .filter_map(|s| Reference::parse(s).ok())
                .collect()
        })
        .unwrap_or_default()
}

/// Write each `evidence.materialize` request: masks under `masks/`, other
/// resources under `intermediates/`, as RGBA8 PNG previews.
fn write_materializations(
    writer: &BundleWriter,
    plan: &Plan,
    execution: &Execution,
) -> Result<(), Error> {
    for reference in evidence_references(plan, "materialize") {
        let Reference::Node { node, port } = &reference else {
            continue;
        };
        let Some(value) = execution.output(node, port) else {
            continue;
        };
        let Some(rgba) = value_to_rgba8(value) else {
            continue;
        };
        let Some(png) =
            paintop_core::evidence::encode_rgba(value.extent().width, value.extent().height, &rgba)
        else {
            continue;
        };
        let is_mask = matches!(value.descriptor(), paintop_ir::ResourceDescriptor::Mask(_));
        let dir = if is_mask {
            dirs::MASKS
        } else {
            dirs::INTERMEDIATES
        };
        let relative = format!("{dir}/{node}-{port}.png");
        writer.write_bytes(&relative, &png).map_err(bundle_err)?;
    }
    Ok(())
}

/// Build the before/after panels for the contact sheet and diffs from the plan's
/// declared evidence diff (the §16 `diffs` request), if any.
fn export_panels(plan: &Plan, execution: &Execution) -> (Option<Panel>, Option<Panel>) {
    let Some(diff) = plan
        .evidence
        .get("diffs")
        .and_then(serde_json::Value::as_array)
        .and_then(|d| d.first())
        .and_then(serde_json::Value::as_object)
    else {
        return (None, None);
    };
    let before = diff
        .get("before")
        .and_then(serde_json::Value::as_str)
        .and_then(|s| reference_panel(s, execution));
    let after = diff
        .get("after")
        .and_then(serde_json::Value::as_str)
        .and_then(|s| reference_panel(s, execution));
    (before, after)
}

/// Resolve a reference string to a renderable [`Panel`], if it names a produced
/// image node output.
fn reference_panel(reference: &str, execution: &Execution) -> Option<Panel> {
    let Reference::Node { node, port } = Reference::parse(reference).ok()? else {
        return None;
    };
    let value = execution.output(&node, &port)?;
    let rgba = value_to_rgba8(value)?;
    Panel::new(value.extent().width, value.extent().height, rgba)
}

/// Write the before/after `contact-sheet.png` and the standalone `diffs/` diff,
/// when both panels are available.
fn write_contact_and_diffs(
    writer: &BundleWriter,
    before: Option<&Panel>,
    after: Option<&Panel>,
) -> Result<(), Error> {
    let (Some(before), Some(after)) = (before, after) else {
        return Ok(());
    };
    let sheet = ContactSheet::compose(before, after);
    writer.write_contact_sheet(&sheet).map_err(bundle_err)?;
    if let Some(diff_png) = Panel::diff_panel(before, after).encode_png() {
        writer
            .write_bytes(format!("{}/final-diff.png", dirs::DIFFS), &diff_png)
            .map_err(bundle_err)?;
    }
    Ok(())
}

/// Write a minimal replay reproducer for each failed assertion under `replays/`.
fn write_replays(
    writer: &BundleWriter,
    plan: &Plan,
    report: &AssertionReport,
) -> Result<(), Error> {
    for failure in report.failures() {
        let spec = ReplaySpec::new(failure.id.clone(), failure.op.clone());
        let replay = MinimalReplay::reduce(plan, spec);
        writer.write_replay(&replay).map_err(bundle_err)?;
    }
    Ok(())
}

/// Quantize a resource value to an 8-bit RGBA preview buffer (row-major).
///
/// Images map their channels onto RGBA; a single-channel mask is broadcast to
/// gray with opaque alpha. A report (no raster) yields `None`.
fn value_to_rgba8(value: &ResourceValue) -> Option<Vec<u8>> {
    if value.as_report().is_some() {
        return None;
    }
    let extent = value.extent();
    let px = (extent.width as usize).checked_mul(extent.height as usize)?;
    let channels = value.channels() as usize;
    if channels == 0 {
        return None;
    }
    let samples = value.samples();
    let mut rgba = vec![0u8; px.checked_mul(4)?];
    for pixel in 0..px {
        let base = pixel * channels;
        let channel = |offset: usize| -> u8 {
            let sample = samples.get(base + offset).copied().unwrap_or(0.0);
            quantize_unit(sample)
        };
        let (red, green, blue, alpha) = match channels {
            1 => {
                let luma = channel(0);
                (luma, luma, luma, 255)
            }
            2 => (channel(0), channel(0), channel(0), channel(1)),
            3 => (channel(0), channel(1), channel(2), 255),
            _ => (channel(0), channel(1), channel(2), channel(3)),
        };
        let out = pixel * 4;
        rgba[out] = red;
        rgba[out + 1] = green;
        rgba[out + 2] = blue;
        rgba[out + 3] = alpha;
    }
    Some(rgba)
}

/// Clamp a normalized sample to `[0, 1]` and round to 8-bit. Non-finite is `0`.
fn quantize_unit(sample: f32) -> u8 {
    if !sample.is_finite() {
        return 0;
    }
    let rounded = (sample.clamp(0.0, 1.0) * 255.0).round();
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "rounded lies in [0, 255], so the cast is exact"
    )]
    let byte = rounded as u8;
    byte
}

/// Map a bundle-writer error into the IR error taxonomy as an export failure.
///
/// Takes the error by value so it can be passed directly as a `map_err` function
/// argument (`Result::map_err` hands its closure the error by value).
#[allow(
    clippy::needless_pass_by_value,
    reason = "used as a `map_err` fn argument, which is invoked with the error by value"
)]
fn bundle_err(err: paintop_core::evidence::BundleError) -> Error {
    Error::new(
        ErrorClass::Export,
        "E_BUNDLE_WRITE_FAILED",
        format!("evidence bundle write failed: {err}"),
    )
}
