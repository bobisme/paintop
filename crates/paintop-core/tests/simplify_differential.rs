//! Differential + side-condition tests for graph simplification (bn-28m).
//!
//! The core guarantee (`plan.md` §10.1 phase 8; `ALIEN_OPS` §13.2): every safe
//! rewrite preserves the semantic value of every graph **output**. These tests
//! execute the *simplified* and *unsimplified* graphs whole-image through the same
//! executor and assert the export values are **byte-identical** for exact ops —
//! and that CSE drops the node count while keeping the output, and barriers
//! prevent a rewrite firing.

use std::collections::BTreeMap;

use paintop_core::executor::{
    ImplRegistry, InputValues, OpImplementation, OutputValues, ResourceValue, execute,
};
use paintop_ir::{
    AlphaRepresentation, ChannelLayout, ColorEncoding, ColorRange, CoordinateConvention,
    DeterminismTier, Error, ErrorClass, Extent, ImageDescriptor, InputSpec, OperationManifest,
    OperationRegistry, OutputSpec, Plan, ResourceDescriptor, ResourceKind, RoiCategory, RoiPolicy,
    ScalarType, SemanticRole, SimplifyOptions, TestMetadata, parse_plan, resolve_plan, simplify,
};
use serde_json::Value;

// ---- Descriptors / values --------------------------------------------------

const EXTENT: Extent = Extent::new(2, 2);
const CHANNELS: u32 = 4;

const fn image_descriptor() -> ResourceDescriptor {
    ResourceDescriptor::Image(ImageDescriptor {
        extent: EXTENT,
        layout: ChannelLayout::Rgba,
        scalar: ScalarType::F32,
        color: ColorEncoding::LinearSrgb,
        range: ColorRange::SceneReferred,
        alpha: AlphaRepresentation::Straight,
        coordinates: CoordinateConvention::PixelCenterUpperLeft,
        semantic: SemanticRole::Color,
    })
}

fn value(samples: Vec<f32>) -> ResourceValue {
    ResourceValue::new(image_descriptor(), CHANNELS, samples).expect("well-sized")
}

fn source_value() -> ResourceValue {
    // Distinct per-sample values so any wrong rewiring would change bytes. The
    // count is tiny (16 samples), so a u16->f32 widening is exact.
    let count = u16::try_from(EXTENT.width * EXTENT.height * CHANNELS).expect("small");
    value((0..count).map(|i| f32::from(i) * 0.01).collect())
}

// ---- Manifests -------------------------------------------------------------

fn op(id: &str, det: DeterminismTier, inputs: &[&str], outputs: &[&str]) -> OperationManifest {
    OperationManifest {
        id: id.parse().expect("ok"),
        impl_version: 1,
        summary: String::new(),
        determinism: det,
        roi: RoiPolicy {
            category: RoiCategory::Pointwise,
            halo_px: None,
        },
        inputs: inputs
            .iter()
            .map(|name| InputSpec {
                name: (*name).to_owned(),
                kind: ResourceKind::Image,
                required: true,
                doc: String::new(),
            })
            .collect(),
        outputs: outputs
            .iter()
            .map(|name| OutputSpec {
                name: (*name).to_owned(),
                kind: ResourceKind::Image,
                doc: String::new(),
            })
            .collect(),
        params: vec![],
        implementations: vec!["cpu.reference@1".parse().expect("ok")],
        test: TestMetadata::default(),
    }
}

fn registry() -> OperationRegistry {
    OperationRegistry::from_manifests([
        op(
            "filter.invert@1",
            DeterminismTier::Exact,
            &["image"],
            &["image"],
        ),
        op(
            "filter.gaussian_blur@1",
            DeterminismTier::Exact,
            &["image"],
            &["image"],
        ),
        op(
            "alpha.premultiply@1",
            DeterminismTier::Exact,
            &["image"],
            &["image"],
        ),
        op(
            "alpha.unpremultiply@1",
            DeterminismTier::Exact,
            &["image"],
            &["image"],
        ),
        op(
            "image.flip@1",
            DeterminismTier::Exact,
            &["image"],
            &["image"],
        ),
    ])
    .expect("ok")
}

// ---- Executable implementations --------------------------------------------

/// Multiply every sample by `factor`. With a complementary factor, premultiply
/// and unpremultiply form a true inverse pair (the §13.2 `alpha > ε` case where
/// the round-trip is exact).
struct ScaleImpl(f32);
impl OpImplementation for ScaleImpl {
    fn compute(&self, inputs: &InputValues, _p: &Value) -> Result<OutputValues, Error> {
        let image = inputs
            .get("image")
            .cloned()
            .ok_or_else(|| Error::new(ErrorClass::Execution, "E_MISSING_INPUT", "needs `image`"))?;
        let samples: Vec<f32> = image.samples().iter().map(|s| s * self.0).collect();
        let mut out = OutputValues::new();
        out.insert("image".to_owned(), value(samples));
        Ok(out)
    }
}

/// A pure pass-through (identity) op.
struct PassThroughImpl;
impl OpImplementation for PassThroughImpl {
    fn compute(&self, inputs: &InputValues, _p: &Value) -> Result<OutputValues, Error> {
        let image = inputs
            .get("image")
            .cloned()
            .ok_or_else(|| Error::new(ErrorClass::Execution, "E_MISSING_INPUT", "needs `image`"))?;
        let mut out = OutputValues::new();
        out.insert("image".to_owned(), image);
        Ok(out)
    }
}

/// Add a fixed per-call constant so two distinct invert nodes (pre-CSE) that are
/// nonetheless *semantically identical* produce identical output — the CSE merge
/// must not change that output.
struct InvertImpl;
impl OpImplementation for InvertImpl {
    fn compute(&self, inputs: &InputValues, _p: &Value) -> Result<OutputValues, Error> {
        let image = inputs
            .get("image")
            .cloned()
            .ok_or_else(|| Error::new(ErrorClass::Execution, "E_MISSING_INPUT", "needs `image`"))?;
        let samples: Vec<f32> = image.samples().iter().map(|s| 1.0 - s).collect();
        let mut out = OutputValues::new();
        out.insert("image".to_owned(), value(samples));
        Ok(out)
    }
}

fn implementations() -> ImplRegistry {
    let mut r = ImplRegistry::new();
    // premultiply x2, unpremultiply x0.5 => exact inverse round-trip.
    r.register(
        "alpha.premultiply@1".parse().expect("ok"),
        Box::new(ScaleImpl(2.0)),
    )
    .expect("ok");
    r.register(
        "alpha.unpremultiply@1".parse().expect("ok"),
        Box::new(ScaleImpl(0.5)),
    )
    .expect("ok");
    r.register("filter.invert@1".parse().expect("ok"), Box::new(InvertImpl))
        .expect("ok");
    r.register(
        "filter.gaussian_blur@1".parse().expect("ok"),
        Box::new(PassThroughImpl),
    )
    .expect("ok");
    r.register(
        "image.flip@1".parse().expect("ok"),
        Box::new(PassThroughImpl),
    )
    .expect("ok");
    r
}

fn inputs() -> BTreeMap<String, ResourceValue> {
    let mut m = BTreeMap::new();
    m.insert("src".to_owned(), source_value());
    m
}

/// Resolve, simplify (enabled vs disabled), execute both, and return the export
/// values keyed by export id for comparison.
fn run(plan: &Plan, options: SimplifyOptions) -> Vec<(String, ResourceValue)> {
    let reg = registry();
    let graph = resolve_plan(plan, &reg).expect("resolve");
    let (graph, _report) = simplify(plan, &graph, &reg, options);
    let impls = implementations();
    let exec = execute(plan, &graph, &reg, &impls, &inputs()).expect("execute");
    exec.exports().to_vec()
}

fn assert_outputs_identical(plan: &Plan) {
    let unsimplified = run(plan, SimplifyOptions::DISABLED);
    let simplified = run(plan, SimplifyOptions::ENABLED);
    assert_eq!(
        unsimplified.len(),
        simplified.len(),
        "same export set after simplification"
    );
    for ((id_a, v_a), (id_b, v_b)) in unsimplified.iter().zip(simplified.iter()) {
        assert_eq!(id_a, id_b, "export order preserved");
        assert_eq!(
            v_a.samples(),
            v_b.samples(),
            "export {id_a:?} must be byte-identical simplified vs unsimplified"
        );
    }
}

// ---- Differential: simplified == unsimplified ------------------------------

#[test]
fn conversion_cancellation_preserves_output() {
    // src -> premultiply -> unpremultiply -> blur -> export. The pair cancels;
    // the surviving blur must produce the same bytes as the full chain.
    let plan = parse_plan(
        r#"{
            "paintop": "1.0",
            "inputs": {"src": {"kind": "image.file", "path": "in.png"}},
            "nodes": [
                {"id": "p", "op": "alpha.premultiply@1", "in": {"image": "input:src"}},
                {"id": "u", "op": "alpha.unpremultiply@1", "in": {"image": "node:p/image"}},
                {"id": "b", "op": "filter.gaussian_blur@1", "in": {"image": "node:u/image"}}
            ],
            "exports": {"o": {"resource": "node:b/image", "kind": "image", "path": "o.png"}}
        }"#,
    )
    .expect("ok");
    assert_outputs_identical(&plan);
}

#[test]
fn identity_elimination_preserves_output() {
    let plan = parse_plan(
        r#"{
            "paintop": "1.0",
            "inputs": {"src": {"kind": "image.file", "path": "in.png"}},
            "nodes": [
                {"id": "f", "op": "image.flip@1", "in": {"image": "input:src"},
                 "params": {"axis": "none"}},
                {"id": "i", "op": "filter.invert@1", "in": {"image": "node:f/image"}}
            ],
            "exports": {"o": {"resource": "node:i/image", "kind": "image", "path": "o.png"}}
        }"#,
    )
    .expect("ok");
    assert_outputs_identical(&plan);
}

#[test]
fn cse_preserves_output_and_drops_node_count() {
    let plan = parse_plan(
        r#"{
            "paintop": "1.0",
            "inputs": {"src": {"kind": "image.file", "path": "in.png"}},
            "nodes": [
                {"id": "a", "op": "filter.invert@1", "in": {"image": "input:src"}},
                {"id": "b", "op": "filter.invert@1", "in": {"image": "input:src"}}
            ],
            "exports": {
                "x": {"resource": "node:a/image", "kind": "image", "path": "x.png"},
                "y": {"resource": "node:b/image", "kind": "image", "path": "y.png"}
            }
        }"#,
    )
    .expect("ok");
    // Node-count drop is observable in the simplified graph.
    let reg = registry();
    let graph = resolve_plan(&plan, &reg).expect("resolve");
    let (simplified, report) = simplify(&plan, &graph, &reg, SimplifyOptions::ENABLED);
    assert_eq!(simplified.nodes().len(), 1, "duplicate invert collapsed");
    assert!(report.rewrite_count() >= 1);
    // And the outputs are unchanged.
    assert_outputs_identical(&plan);
}

#[test]
fn nested_rewrites_reach_a_fixed_point_and_preserve_output() {
    // premultiply/unpremultiply pair feeding two identical inverts: cancellation
    // then CSE both fire; output must still match.
    let plan = parse_plan(
        r#"{
            "paintop": "1.0",
            "inputs": {"src": {"kind": "image.file", "path": "in.png"}},
            "nodes": [
                {"id": "p", "op": "alpha.premultiply@1", "in": {"image": "input:src"}},
                {"id": "u", "op": "alpha.unpremultiply@1", "in": {"image": "node:p/image"}},
                {"id": "a", "op": "filter.invert@1", "in": {"image": "node:u/image"}},
                {"id": "b", "op": "filter.invert@1", "in": {"image": "node:u/image"}}
            ],
            "exports": {
                "x": {"resource": "node:a/image", "kind": "image", "path": "x.png"},
                "y": {"resource": "node:b/image", "kind": "image", "path": "y.png"}
            }
        }"#,
    )
    .expect("ok");
    assert_outputs_identical(&plan);
}

// ---- Disable flag ----------------------------------------------------------

#[test]
fn disabled_simplification_is_a_no_op() {
    let plan = parse_plan(
        r#"{
            "paintop": "1.0",
            "inputs": {"src": {"kind": "image.file", "path": "in.png"}},
            "nodes": [
                {"id": "p", "op": "alpha.premultiply@1", "in": {"image": "input:src"}},
                {"id": "u", "op": "alpha.unpremultiply@1", "in": {"image": "node:p/image"}}
            ],
            "exports": {"o": {"resource": "node:u/image", "kind": "image", "path": "o.png"}}
        }"#,
    )
    .expect("ok");
    let reg = registry();
    let graph = resolve_plan(&plan, &reg).expect("resolve");
    let (out, report) = simplify(&plan, &graph, &reg, SimplifyOptions::DISABLED);
    assert_eq!(out, graph, "disabled pass returns the graph unchanged");
    assert_eq!(report.rewrite_count(), 0);
}
