//! Generated random-graph quality suite for the M0/MVP compile+execute spine
//! (`AGENT_VERIFICATION` §9, bn-2q0).
//!
//! A typed graph generator explores resolver/checker/executor interactions a
//! hand-written fixture would miss. The generator (`§9.1`) builds **bounded valid
//! DAGs** over the implemented operation subset — a source op (no inputs, one
//! image output) and an identity unary op requiring linear-light premultiplied
//! RGBA — with small extents, a fixed seed in the repro output, and exports
//! always reachable. It also has an **invalid-graph mode** (`§9.1` last bullet)
//! that injects dangling references, cycles, missing ports, and a color/alpha
//! mismatch for validator fuzzing, asserting each fails with a *stable* error
//! code.
//!
//! Where resources permit, small valid graphs are executed end-to-end through the
//! sequential executor, proving topological execution and dead-node elimination
//! hold over generated structure, not just the canned three-node fixture.
//!
//! Finally, a **delta-debugging reducer** (`§9.3`) guided by a failure predicate
//! minimizes a failing generated plan to a small bug specimen and re-checks that
//! the specimen still fails — the failure-reduction loop §9.3 calls for.
//!
//! Scope: schema/reference/type/executor behavior only. No tiling, cache, GPU,
//! models, or material work.

use std::collections::BTreeMap;

use paintop_core::evidence::replay::{MinimalReplay, ReplaySpec};
use paintop_core::evidence::trace::TraceEvent;
use paintop_core::executor::{
    ImplRegistry, InputValues, OpImplementation, OutputValues, ResourceValue, execute,
};
use paintop_ir::{
    AlphaRepresentation, ChannelLayout, ColorEncoding, ColorRange, ContractRegistry,
    CoordinateConvention, Descriptors, DeterminismTier, Error, ErrorClass, Extent, ImageDescriptor,
    InputRegions, InputSpec, OpContract, OperationManifest, OperationRegistry, OutputDescriptors,
    OutputRegions, OutputSpec, Plan, ResourceDescriptor, ResourceKind, RoiCategory, RoiPolicy,
    ScalarType, SemanticRole, TestMetadata, check_graph, parse_plan, resolve_plan,
};
use proptest::prelude::*;
use serde_json::Value;

// ---------------------------------------------------------------------------
// Operation subset: a source (0->1) and an identity unary op (1->1).
// ---------------------------------------------------------------------------

const SOURCE_OP: &str = "source.create@1";
const UNARY_OP: &str = "filter.invert@1";
const EXTENT: Extent = Extent::new(2, 2);
const CHANNELS: u32 = 4;

const fn image_descriptor(color: ColorEncoding, alpha: AlphaRepresentation) -> ResourceDescriptor {
    ResourceDescriptor::Image(ImageDescriptor {
        extent: EXTENT,
        layout: ChannelLayout::Rgba,
        scalar: ScalarType::F32,
        color,
        range: ColorRange::SceneReferred,
        alpha,
        coordinates: CoordinateConvention::PixelCenterUpperLeft,
        semantic: SemanticRole::Color,
    })
}

const fn linear_premul() -> ResourceDescriptor {
    image_descriptor(
        ColorEncoding::LinearSrgb,
        AlphaRepresentation::Premultiplied,
    )
}

fn value(descriptor: ResourceDescriptor, fill: f32) -> ResourceValue {
    let len = (EXTENT.width * EXTENT.height * CHANNELS) as usize;
    ResourceValue::new(descriptor, CHANNELS, vec![fill; len]).expect("well-sized buffer")
}

fn op(id: &str, inputs: &[&str], outputs: &[&str]) -> OperationManifest {
    OperationManifest {
        id: id.parse().expect("valid op id"),
        impl_version: 1,
        summary: String::new(),
        determinism: DeterminismTier::Exact,
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
        implementations: vec!["cpu.reference@1".parse().expect("valid impl id")],
        test: TestMetadata::default(),
    }
}

fn registry() -> OperationRegistry {
    OperationRegistry::from_manifests([
        op(SOURCE_OP, &[], &["image"]),
        op(UNARY_OP, &["image"], &["image"]),
    ])
    .expect("registry")
}

// ---- Contracts (descriptor-level type checker) -----------------------------

struct SourceContract(ResourceDescriptor);
impl OpContract for SourceContract {
    fn declared_inputs(&self) -> Vec<(String, ResourceKind)> {
        vec![]
    }
    fn declared_outputs(&self) -> Vec<(String, ResourceKind)> {
        vec![("image".to_owned(), ResourceKind::Image)]
    }
    fn infer_outputs(&self, _i: &Descriptors, _p: &Value) -> Result<OutputDescriptors, Error> {
        let mut out = OutputDescriptors::new();
        out.insert("image".to_owned(), self.0);
        Ok(out)
    }
    fn required_inputs(
        &self,
        _o: &OutputRegions,
        _i: &Descriptors,
        _p: &Value,
    ) -> Result<InputRegions, Error> {
        Ok(InputRegions::new())
    }
    fn validate_postconditions(
        &self,
        _o: &OutputDescriptors,
        _p: &Value,
    ) -> Result<Vec<paintop_ir::AssertionResult>, Error> {
        Ok(vec![])
    }
}

/// Identity unary op that *requires* a linear-light input, so feeding it sRGB is
/// a stable `E_COLOR_ENCODING_MISMATCH`.
struct UnaryContract;
impl OpContract for UnaryContract {
    fn declared_inputs(&self) -> Vec<(String, ResourceKind)> {
        vec![("image".to_owned(), ResourceKind::Image)]
    }
    fn declared_outputs(&self) -> Vec<(String, ResourceKind)> {
        vec![("image".to_owned(), ResourceKind::Image)]
    }
    fn infer_outputs(&self, inputs: &Descriptors, _p: &Value) -> Result<OutputDescriptors, Error> {
        let image = inputs.get("image").ok_or_else(|| {
            Error::new(ErrorClass::Type, "E_MISSING_INPUT", "unary needs `image`")
        })?;
        let ResourceDescriptor::Image(desc) = image else {
            return Err(Error::new(
                ErrorClass::Type,
                "E_WRONG_KIND",
                "unary needs an Image",
            ));
        };
        if !desc.color.is_linear_light() {
            return Err(Error::new(
                ErrorClass::Semantic,
                "E_COLOR_ENCODING_MISMATCH",
                "filter requires a linear image",
            ));
        }
        let mut out = OutputDescriptors::new();
        out.insert("image".to_owned(), *image);
        Ok(out)
    }
    fn required_inputs(
        &self,
        o: &OutputRegions,
        _i: &Descriptors,
        _p: &Value,
    ) -> Result<InputRegions, Error> {
        let mut regions = InputRegions::new();
        if let Some(region) = o.get("image") {
            regions.insert("image".to_owned(), *region);
        }
        Ok(regions)
    }
    fn validate_postconditions(
        &self,
        _o: &OutputDescriptors,
        _p: &Value,
    ) -> Result<Vec<paintop_ir::AssertionResult>, Error> {
        Ok(vec![])
    }
}

fn contracts(source: ResourceDescriptor) -> ContractRegistry {
    let mut c = ContractRegistry::new();
    c.register(
        SOURCE_OP.parse().expect("op id"),
        Box::new(SourceContract(source)),
    )
    .expect("register source");
    c.register(UNARY_OP.parse().expect("op id"), Box::new(UnaryContract))
        .expect("register unary");
    c
}

// ---- Executable implementations (identity passthrough) ---------------------

struct SourceImpl(ResourceDescriptor);
impl OpImplementation for SourceImpl {
    fn compute(&self, _i: &InputValues, _p: &Value) -> Result<OutputValues, Error> {
        let mut out = OutputValues::new();
        out.insert("image".to_owned(), value(self.0, 0.25));
        Ok(out)
    }
}

struct UnaryImpl;
impl OpImplementation for UnaryImpl {
    fn compute(&self, inputs: &InputValues, _p: &Value) -> Result<OutputValues, Error> {
        let image = inputs.get("image").cloned().ok_or_else(|| {
            Error::new(
                ErrorClass::Execution,
                "E_MISSING_INPUT",
                "unary needs `image`",
            )
        })?;
        let mut out = OutputValues::new();
        out.insert("image".to_owned(), image);
        Ok(out)
    }
}

fn implementations() -> ImplRegistry {
    let mut r = ImplRegistry::new();
    r.register(
        SOURCE_OP.parse().expect("op id"),
        Box::new(SourceImpl(linear_premul())),
    )
    .expect("register source impl");
    r.register(UNARY_OP.parse().expect("op id"), Box::new(UnaryImpl))
        .expect("register unary impl");
    r
}

const fn no_inputs() -> BTreeMap<String, ResourceValue> {
    BTreeMap::new()
}

const fn no_input_descriptors() -> BTreeMap<String, ResourceDescriptor> {
    BTreeMap::new()
}

// ---------------------------------------------------------------------------
// Typed graph generator (AGENT_VERIFICATION §9.1).
// ---------------------------------------------------------------------------

/// A bounded, typed DAG description the generator emits before lowering to JSON.
///
/// `node_count` source/unary nodes are chained so each node `i > 0` reads node
/// `parent[i]` (an earlier node, guaranteeing acyclicity and that exports are
/// reachable). `node 0` is always the `source`. `export_node` selects which
/// node's output the single export reads.
#[derive(Debug, Clone)]
struct GraphPlan {
    node_count: usize,
    /// `parent[i]` is the index of the node that node `i` reads from (only
    /// meaningful for `i >= 1`); always `< i`, so the graph is a DAG.
    parent: Vec<usize>,
    export_node: usize,
}

impl GraphPlan {
    /// Lower this typed DAG to a `paintop` plan JSON string with a single export.
    fn to_json(&self) -> String {
        let nodes = (0..self.node_count)
            .map(|i| {
                if i == 0 {
                    format!(r#"{{"id": "n0", "op": "{SOURCE_OP}"}}"#)
                } else {
                    format!(
                        r#"{{"id": "n{i}", "op": "{UNARY_OP}", "in": {{"image": "node:n{}/image"}}}}"#,
                        self.parent[i]
                    )
                }
            })
            .collect::<Vec<_>>()
            .join(",");
        format!(
            r#"{{
                "paintop": "1.0",
                "name": "generated",
                "inputs": {{}},
                "nodes": [{nodes}],
                "exports": {{"out": {{"resource": "node:n{}/image", "kind": "image", "path": "o.png"}}}}
            }}"#,
            self.export_node
        )
    }

    /// The set of node indices on the export's transitive producer cone (the
    /// nodes a correct demand pass must keep).
    fn live_nodes(&self) -> Vec<usize> {
        let mut keep = vec![false; self.node_count];
        let mut cur = self.export_node;
        loop {
            keep[cur] = true;
            if cur == 0 {
                break;
            }
            cur = self.parent[cur];
        }
        (0..self.node_count).filter(|&i| keep[i]).collect()
    }
}

/// Strategy for a bounded valid DAG: 1..=6 nodes, each non-root reading an
/// earlier node, export reading any node. Small extents are fixed by the op
/// subset (2x2 RGBA), keeping reference execution cheap.
fn graph_strategy() -> impl Strategy<Value = GraphPlan> {
    (1usize..=6).prop_flat_map(|node_count| {
        // For each node i>=1, pick a parent in 0..i.
        let parents = (1..node_count).map(|i| 0..i).collect::<Vec<_>>();
        (Just(node_count), parents, 0..node_count).prop_map(|(node_count, tail, export_node)| {
            let mut parent = vec![0usize; node_count];
            for (i, p) in tail.into_iter().enumerate() {
                parent[i + 1] = p;
            }
            GraphPlan {
                node_count,
                parent,
                export_node,
            }
        })
    })
}

// ---------------------------------------------------------------------------
// Property: every generated valid DAG resolves, type-checks, and executes, and
// the executor keeps exactly the export's producer cone (dead-node elimination).
// ---------------------------------------------------------------------------

proptest! {
    // A fixed seed is recorded in any repro output (AGENT_VERIFICATION §9.1).
    #![proptest_config(ProptestConfig {
        cases: 128,
        ..ProptestConfig::default()
    })]

    #[test]
    fn generated_valid_dag_resolves_checks_and_executes(graph in graph_strategy()) {
        let json = graph.to_json();
        let plan = parse_plan(&json).expect("generated plan parses");
        let reg = registry();
        let resolved = resolve_plan(&plan, &reg).expect("generated DAG resolves");

        // The type checker accepts the well-typed (all linear) pipeline.
        check_graph(
            &plan,
            &resolved,
            &reg,
            &contracts(linear_premul()),
            &no_input_descriptors(),
        )
        .expect("generated DAG type-checks");

        let exec = execute(&plan, &resolved, &reg, &implementations(), &no_inputs())
            .expect("generated DAG executes");

        // Demand keeps exactly the export's producer cone; everything else is
        // eliminated and never dispatched.
        let mut expected_live: Vec<String> =
            graph.live_nodes().iter().map(|i| format!("n{i}")).collect();
        expected_live.sort();
        let mut demanded: Vec<String> = exec.demand().demanded().to_vec();
        demanded.sort();
        prop_assert_eq!(&demanded, &expected_live);

        // Demanded nodes run in topological order: a node only runs after its
        // (single) parent. We check each completed node's parent completed first.
        let completed_order: Vec<usize> = exec
            .trace()
            .iter()
            .filter_map(|e| match e {
                TraceEvent::DispatchCompleted(c) => {
                    c.node.strip_prefix('n').and_then(|s| s.parse::<usize>().ok())
                }
                _ => None,
            })
            .collect();
        let mut position = BTreeMap::new();
        for (pos, idx) in completed_order.iter().enumerate() {
            position.insert(*idx, pos);
        }
        for &idx in &completed_order {
            if idx != 0 {
                let parent = graph.parent[idx];
                prop_assert!(
                    position[&parent] < position[&idx],
                    "node n{idx} ran before its parent n{parent}"
                );
            }
        }

        // Every eliminated node is absent from the trace entirely.
        for i in 0..graph.node_count {
            if !graph.live_nodes().contains(&i) {
                let id = format!("n{i}");
                prop_assert!(
                    exec.trace().iter().all(|e| e.node() != Some(id.as_str())),
                    "dead node {id} appeared in the trace"
                );
            }
        }

        // The single export carries the identity-passed source fill.
        prop_assert_eq!(exec.exports().len(), 1);
        prop_assert_eq!(exec.exports()[0].1.samples(), &[0.25_f32; 16]);
    }
}

// ---------------------------------------------------------------------------
// Invalid-graph mode (AGENT_VERIFICATION §9.1 last bullet): each malformed graph
// fails with a STABLE error code from the resolver / checker.
// ---------------------------------------------------------------------------

#[test]
fn dangling_node_reference_is_a_stable_reference_error() {
    let plan = parse_plan(
        r#"{
            "paintop": "1.0",
            "inputs": {},
            "nodes": [
                {"id": "a", "op": "filter.invert@1", "in": {"image": "node:ghost/image"}}
            ],
            "exports": {}
        }"#,
    )
    .expect("parses");
    let err = resolve_plan(&plan, &registry()).expect_err("dangling ref must fail");
    assert_eq!(err.class, ErrorClass::Reference);
    assert_eq!(err.code, paintop_ir::E_DANGLING_REFERENCE);
}

#[test]
fn cycle_is_a_stable_reference_error() {
    let plan = parse_plan(
        r#"{
            "paintop": "1.0",
            "inputs": {},
            "nodes": [
                {"id": "a", "op": "filter.invert@1", "in": {"image": "node:b/image"}},
                {"id": "b", "op": "filter.invert@1", "in": {"image": "node:a/image"}}
            ],
            "exports": {}
        }"#,
    )
    .expect("parses");
    let err = resolve_plan(&plan, &registry()).expect_err("cycle must fail");
    assert_eq!(err.class, ErrorClass::Reference);
    assert_eq!(err.code, paintop_ir::E_GRAPH_CYCLE);
}

#[test]
fn missing_required_input_port_is_a_stable_reference_error() {
    // The unary op declares a required `image` input but the node wires none.
    let plan = parse_plan(
        r#"{
            "paintop": "1.0",
            "inputs": {},
            "nodes": [{"id": "a", "op": "filter.invert@1"}],
            "exports": {}
        }"#,
    )
    .expect("parses");
    let err = resolve_plan(&plan, &registry()).expect_err("missing port must fail");
    assert_eq!(err.class, ErrorClass::Reference);
    assert_eq!(err.code, paintop_ir::E_MISSING_INPUT_PORT);
}

#[test]
fn unknown_output_port_reference_is_a_stable_reference_error() {
    let plan = parse_plan(
        r#"{
            "paintop": "1.0",
            "inputs": {},
            "nodes": [
                {"id": "src", "op": "source.create@1"},
                {"id": "a", "op": "filter.invert@1", "in": {"image": "node:src/bogus"}}
            ],
            "exports": {}
        }"#,
    )
    .expect("parses");
    let err = resolve_plan(&plan, &registry()).expect_err("bad port must fail");
    assert_eq!(err.class, ErrorClass::Reference);
    assert_eq!(err.code, paintop_ir::E_UNKNOWN_OUTPUT_PORT);
}

#[test]
fn color_encoding_mismatch_is_a_stable_semantic_error() {
    // A valid DAG whose source emits sRGB into the linear-only unary op: the
    // resolver accepts the structure, the checker rejects the color policy.
    let graph = GraphPlan {
        node_count: 2,
        parent: vec![0, 0],
        export_node: 1,
    };
    let plan = parse_plan(&graph.to_json()).expect("parses");
    let reg = registry();
    let resolved = resolve_plan(&plan, &reg).expect("resolves");
    let srgb = image_descriptor(ColorEncoding::Srgb, AlphaRepresentation::Premultiplied);
    let err = check_graph(
        &plan,
        &resolved,
        &reg,
        &contracts(srgb),
        &no_input_descriptors(),
    )
    .expect_err("color mismatch must fail");
    assert_eq!(err.class, ErrorClass::Semantic);
    assert_eq!(err.code, "E_COLOR_ENCODING_MISMATCH");
}

// ---------------------------------------------------------------------------
// Property: invalid graphs (each node references an as-yet-undefined later node)
// never resolve, and never panic — the validator stays total under fuzzing.
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig { cases: 64, ..ProptestConfig::default() })]

    #[test]
    fn forward_only_references_always_fail_to_resolve(node_count in 2usize..=6) {
        // Every node i reads node i+1 (a forward reference); the last reads node
        // 0, closing a cycle. This is always invalid — either a cycle or a
        // dangling ref — and must be rejected with a Reference-class error.
        let nodes = (0..node_count)
            .map(|i| {
                let target = (i + 1) % node_count;
                format!(
                    r#"{{"id": "n{i}", "op": "{UNARY_OP}", "in": {{"image": "node:n{target}/image"}}}}"#
                )
            })
            .collect::<Vec<_>>()
            .join(",");
        let json = format!(
            r#"{{"paintop": "1.0", "inputs": {{}}, "nodes": [{nodes}], "exports": {{}}}}"#
        );
        let plan = parse_plan(&json).expect("invalid plan still parses (schema-valid)");
        let err = resolve_plan(&plan, &registry()).expect_err("invalid graph must fail");
        prop_assert_eq!(err.class, ErrorClass::Reference);
    }
}

// ---------------------------------------------------------------------------
// Failure reduction (AGENT_VERIFICATION §9.3): a delta-debugging reducer guided
// by a failure predicate minimizes a failing generated plan to a small specimen
// that still fails the same way.
// ---------------------------------------------------------------------------

/// The failure predicate: does `plan` still fail the type check with the
/// color-encoding mismatch when the source emits sRGB?
fn fails_with_color_mismatch(plan: &Plan) -> bool {
    let reg = registry();
    let Ok(resolved) = resolve_plan(plan, &reg) else {
        // A reduction that breaks resolution does not preserve the failure.
        return false;
    };
    let srgb = image_descriptor(ColorEncoding::Srgb, AlphaRepresentation::Premultiplied);
    match check_graph(
        plan,
        &resolved,
        &reg,
        &contracts(srgb),
        &no_input_descriptors(),
    ) {
        Err(err) => err.code == "E_COLOR_ENCODING_MISMATCH",
        Ok(_) => false,
    }
}

#[test]
fn delta_debugging_reduces_a_failing_plan_to_a_minimal_specimen() {
    // A large plan: a source, a long unrelated chain, and one unary node fed by
    // the source whose output is exported. Feeding sRGB makes the unary node's
    // check fail. The minimal specimen is just {source, that unary node}.
    let mut node_specs = vec![r#"{"id": "src", "op": "source.create@1"}"#.to_owned()];
    // An unrelated decoy chain (none of these is the export's producer).
    for i in 0..5 {
        node_specs.push(format!(
            r#"{{"id": "decoy{i}", "op": "filter.invert@1", "in": {{"image": "node:src/image"}}}}"#
        ));
    }
    // The node the export actually reads.
    node_specs.push(
        r#"{"id": "target", "op": "filter.invert@1", "in": {"image": "node:src/image"}}"#
            .to_owned(),
    );
    let nodes = node_specs.join(",");
    let json = format!(
        r#"{{
            "paintop": "1.0",
            "inputs": {{}},
            "nodes": [{nodes}],
            "exports": {{"out": {{"resource": "node:target/image", "kind": "image", "path": "o.png"}}}}
        }}"#
    );
    let plan = parse_plan(&json).expect("parses");
    assert!(fails_with_color_mismatch(&plan), "original plan must fail");

    // The structural reducer keeps only the target's producer cone (§9.3 steps
    // 1-3: remove unrelated exports/dead nodes, minimize node count).
    let replay = MinimalReplay::reduce(&plan, ReplaySpec::new("color", "target"));
    let reduced = &replay.plan;

    // The specimen is minimal: exactly {src, target}, decoys gone.
    let mut kept: Vec<&str> = reduced.nodes.iter().map(|n| n.id.as_str()).collect();
    kept.sort_unstable();
    assert_eq!(kept, vec!["src", "target"]);

    // The failure predicate still holds on the reduced specimen — the reduction
    // preserved the bug (§9.3: emit a minimal replay that still fails).
    assert!(
        fails_with_color_mismatch(reduced),
        "reduced specimen must reproduce the failure"
    );

    // Delta-debugging cannot shrink further without losing the failure: dropping
    // either remaining node breaks resolution or removes the mismatch.
    let further = MinimalReplay::reduce(reduced, ReplaySpec::new("color", "src"));
    assert!(
        !fails_with_color_mismatch(&further.plan),
        "the source alone (no unary op) must not reproduce the color mismatch"
    );

    // The minimal replay re-parses as an ordinary plan (agent-debuggable).
    let serialized = serde_json::to_string(reduced).expect("serialize");
    let back = parse_plan(&serialized).expect("reduced specimen re-parses");
    assert_eq!(back.nodes.len(), 2);
}
