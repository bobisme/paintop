//! Integration tests for the sequential whole-image executor (bn-2sb).
//!
//! These exercise the compile+execute spine end to end with stub/identity ops:
//! a ≥3-node synthetic plan resolves, type-checks, and executes in topological
//! order; a node demanded by no export is eliminated and never dispatched
//! (proved via the trace); a cycle and a color/alpha mismatch are rejected by the
//! earlier resolve/check phases this executor sits on top of.

use std::collections::BTreeMap;

use paintop_core::evidence::trace::TraceEvent;
use paintop_core::executor::{
    BackendId, BackendPolicy, E_BACKEND_UNSUPPORTED, E_OUTPUT_NOT_PRODUCED, ExecError,
    ImplRegistry, InputValues, OpImplementation, OutputValues, ResourceValue, execute,
    execute_with_policy,
};
use paintop_ir::{
    AlphaRepresentation, ChannelLayout, ColorEncoding, ColorRange, ContractRegistry,
    CoordinateConvention, Descriptors, DeterminismTier, Error, ErrorClass, Extent, ImageDescriptor,
    InputSpec, OpContract, OperationManifest, OperationRegistry, OutputDescriptors, OutputSpec,
    Plan, ResourceDescriptor, ResourceKind, RoiCategory, RoiPolicy, ScalarType, SemanticRole,
    TestMetadata, check_graph, parse_plan, resolve_plan,
};
use serde_json::Value;

// ---- Descriptors -----------------------------------------------------------

const fn image_descriptor(
    extent: Extent,
    color: ColorEncoding,
    alpha: AlphaRepresentation,
) -> ResourceDescriptor {
    ResourceDescriptor::Image(ImageDescriptor {
        extent,
        layout: ChannelLayout::Rgba,
        scalar: ScalarType::F32,
        color,
        range: ColorRange::SceneReferred,
        alpha,
        coordinates: CoordinateConvention::PixelCenterUpperLeft,
        semantic: SemanticRole::Color,
    })
}

const fn linear_premul(extent: Extent) -> ResourceDescriptor {
    image_descriptor(
        extent,
        ColorEncoding::LinearSrgb,
        AlphaRepresentation::Premultiplied,
    )
}

const EXTENT: Extent = Extent::new(2, 2);
const CHANNELS: u32 = 4;

fn value(descriptor: ResourceDescriptor, fill: f32) -> ResourceValue {
    let len = (EXTENT.width * EXTENT.height * CHANNELS) as usize;
    ResourceValue::new(descriptor, CHANNELS, vec![fill; len]).expect("well-sized buffer")
}

// ---- Manifests -------------------------------------------------------------

fn op(id: &str, inputs: &[&str], outputs: &[&str]) -> OperationManifest {
    OperationManifest {
        id: id.parse().expect("ok"),
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
        implementations: vec!["cpu.reference@1".parse().expect("ok")],
        test: TestMetadata::default(),
    }
}

fn registry() -> OperationRegistry {
    OperationRegistry::from_manifests([
        op("source.create@1", &[], &["image"]),
        op("filter.invert@1", &["image"], &["image"]),
    ])
    .expect("ok")
}

// ---- Descriptor-level contracts (for the type checker) ---------------------

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
        _o: &paintop_ir::OutputRegions,
        _i: &Descriptors,
        _p: &Value,
    ) -> Result<paintop_ir::InputRegions, Error> {
        Ok(paintop_ir::InputRegions::new())
    }
    fn validate_postconditions(
        &self,
        _o: &OutputDescriptors,
        _p: &Value,
    ) -> Result<Vec<paintop_ir::AssertionResult>, Error> {
        Ok(vec![])
    }
}

/// An identity "invert" contract that *requires* linear-light premultiplied input
/// — the color/alpha policy that makes the mismatch test bite.
struct InvertContract;
impl OpContract for InvertContract {
    fn declared_inputs(&self) -> Vec<(String, ResourceKind)> {
        vec![("image".to_owned(), ResourceKind::Image)]
    }
    fn declared_outputs(&self) -> Vec<(String, ResourceKind)> {
        vec![("image".to_owned(), ResourceKind::Image)]
    }
    fn infer_outputs(&self, inputs: &Descriptors, _p: &Value) -> Result<OutputDescriptors, Error> {
        let image = inputs.get("image").ok_or_else(|| {
            Error::new(ErrorClass::Type, "E_MISSING_INPUT", "invert needs `image`")
        })?;
        let ResourceDescriptor::Image(desc) = image else {
            return Err(Error::new(
                ErrorClass::Type,
                "E_WRONG_KIND",
                "invert needs an Image",
            ));
        };
        if !desc.color.is_linear_light() {
            return Err(Error::new(
                ErrorClass::Semantic,
                "E_COLOR_ENCODING_MISMATCH",
                "filter.invert requires a linear image",
            ));
        }
        let mut out = OutputDescriptors::new();
        out.insert("image".to_owned(), *image);
        Ok(out)
    }
    fn required_inputs(
        &self,
        o: &paintop_ir::OutputRegions,
        _i: &Descriptors,
        _p: &Value,
    ) -> Result<paintop_ir::InputRegions, Error> {
        let mut regions = paintop_ir::InputRegions::new();
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
        "source.create@1".parse().expect("ok"),
        Box::new(SourceContract(source)),
    )
    .expect("ok");
    c.register(
        "filter.invert@1".parse().expect("ok"),
        Box::new(InvertContract),
    )
    .expect("ok");
    c
}

// ---- Executable implementations (for the executor) -------------------------

/// A source op: ignores inputs, emits a fixed-fill 2x2 RGBA value.
struct SourceImpl(ResourceDescriptor);
impl OpImplementation for SourceImpl {
    fn compute(&self, _i: &InputValues, _p: &Value) -> Result<OutputValues, Error> {
        let mut out = OutputValues::new();
        out.insert("image".to_owned(), value(self.0, 0.25));
        Ok(out)
    }
}

/// An identity "invert" op: passes its `image` input through unchanged so the
/// test can assert exact output equality whole-image.
struct InvertImpl;
impl OpImplementation for InvertImpl {
    fn compute(&self, inputs: &InputValues, _p: &Value) -> Result<OutputValues, Error> {
        let image = inputs.get("image").cloned().ok_or_else(|| {
            Error::new(
                ErrorClass::Execution,
                "E_MISSING_INPUT",
                "invert needs `image`",
            )
        })?;
        let mut out = OutputValues::new();
        out.insert("image".to_owned(), image);
        Ok(out)
    }
}

/// An op whose implementation forgets to produce its declared `image` output.
struct UnderproducingImpl;
impl OpImplementation for UnderproducingImpl {
    fn compute(&self, _i: &InputValues, _p: &Value) -> Result<OutputValues, Error> {
        Ok(OutputValues::new())
    }
}

fn implementations() -> ImplRegistry {
    let mut r = ImplRegistry::new();
    r.register(
        "source.create@1".parse().expect("ok"),
        Box::new(SourceImpl(linear_premul(EXTENT))),
    )
    .expect("ok");
    r.register("filter.invert@1".parse().expect("ok"), Box::new(InvertImpl))
        .expect("ok");
    r
}

const fn no_inputs() -> BTreeMap<String, ResourceValue> {
    BTreeMap::new()
}

const fn no_input_descriptors() -> BTreeMap<String, ResourceDescriptor> {
    BTreeMap::new()
}

// ---- The ≥3-node synthetic plan: src -> used -> (export); dead is unused ----

fn three_node_plan() -> Plan {
    parse_plan(
        r#"{
            "paintop": "1.0",
            "inputs": {},
            "nodes": [
                {"id": "src", "op": "source.create@1"},
                {"id": "used", "op": "filter.invert@1", "in": {"image": "node:src/image"}},
                {"id": "dead", "op": "filter.invert@1", "in": {"image": "node:src/image"}}
            ],
            "exports": {"out": {"resource": "node:used/image", "kind": "image", "path": "o.png"}}
        }"#,
    )
    .expect("ok")
}

// ---- Exit gate: end-to-end execution with trace assertions -----------------

#[test]
fn three_node_plan_executes_in_topological_order_with_trace() {
    let plan = three_node_plan();
    let reg = registry();
    let graph = resolve_plan(&plan, &reg).expect("ok");
    // The type checker (prior bone) accepts the well-typed pipeline.
    check_graph(
        &plan,
        &graph,
        &reg,
        &contracts(linear_premul(EXTENT)),
        &no_input_descriptors(),
    )
    .expect("ok");

    let exec = execute(&plan, &graph, &reg, &implementations(), &no_inputs()).expect("ok");

    // Only the demanded nodes ran, in topological order; `dead` was eliminated.
    assert_eq!(exec.demand().demanded(), &["src", "used"]);
    assert_eq!(exec.demand().eliminated(), &["dead"]);

    // The export carries the produced value (identity-passed from the source).
    assert_eq!(exec.exports().len(), 1);
    assert_eq!(exec.exports()[0].0, "out");
    assert_eq!(exec.exports()[0].1.samples(), &[0.25_f32; 16]);
    assert_eq!(
        exec.output("used", "image").expect("ok").samples(),
        &[0.25_f32; 16]
    );
}

#[test]
fn each_executed_node_emits_a_dispatch_trace_event() {
    let plan = three_node_plan();
    let reg = registry();
    let graph = resolve_plan(&plan, &reg).expect("ok");
    let exec = execute(&plan, &graph, &reg, &implementations(), &no_inputs()).expect("ok");

    // Every demanded node emits a dispatch_completed event naming its op and the
    // selected cpu.reference implementation.
    let completed: Vec<&str> = exec
        .trace()
        .iter()
        .filter_map(|e| match e {
            TraceEvent::DispatchCompleted(c) => Some(c.node.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(completed, vec!["src", "used"]);

    for e in exec.trace() {
        if let TraceEvent::DispatchCompleted(c) = e {
            assert_eq!(c.implementation, "cpu.reference@1");
            assert!(c.elapsed_ms.is_some(), "elapsed time must be recorded");
        }
    }

    // The eliminated node `dead` appears in NO trace event — the proof it never ran.
    assert!(
        exec.trace().iter().all(|e| e.node() != Some("dead")),
        "dead node must never appear in the trace"
    );

    // Identity keys (selected impl) are present for each executed node.
    assert!(
        exec.trace()
            .iter()
            .filter_map(TraceEvent::implementation)
            .any(|i| i == "cpu.reference@1")
    );
}

#[test]
fn cache_lookup_is_bypassed_in_m0() {
    let plan = three_node_plan();
    let reg = registry();
    let graph = resolve_plan(&plan, &reg).expect("ok");
    let exec = execute(&plan, &graph, &reg, &implementations(), &no_inputs()).expect("ok");

    let bypasses = exec
        .trace()
        .iter()
        .filter(|e| {
            matches!(
                e,
                TraceEvent::CacheLookup(c)
                    if c.outcome == paintop_core::evidence::trace::CacheOutcome::Bypass
            )
        })
        .count();
    assert_eq!(bypasses, 2, "one bypassed cache lookup per executed node");
}

// ---- The spine rejects cycles and color mismatches upstream -----------------

#[test]
fn a_cycle_is_rejected_before_execution() {
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
    .expect("ok");
    let err = resolve_plan(&plan, &registry()).expect_err("expected error");
    assert_eq!(err.class, ErrorClass::Reference);
    assert_eq!(err.code, paintop_ir::E_GRAPH_CYCLE);
}

#[test]
fn an_srgb_color_mismatch_is_rejected_by_the_checker() {
    let plan = three_node_plan();
    let reg = registry();
    let graph = resolve_plan(&plan, &reg).expect("ok");
    // A source that emits sRGB into the linear-only invert.
    let srgb = image_descriptor(
        EXTENT,
        ColorEncoding::Srgb,
        AlphaRepresentation::Premultiplied,
    );
    let err = check_graph(
        &plan,
        &graph,
        &reg,
        &contracts(srgb),
        &no_input_descriptors(),
    )
    .expect_err("expected error");
    assert_eq!(err.class, ErrorClass::Semantic);
    assert_eq!(err.code, "E_COLOR_ENCODING_MISMATCH");
}

// ---- Executor-level failures -----------------------------------------------

#[test]
fn an_underproducing_implementation_fails_with_a_stable_code() {
    let plan = parse_plan(
        r#"{
            "paintop": "1.0",
            "inputs": {},
            "nodes": [{"id": "src", "op": "source.create@1"}],
            "exports": {"out": {"resource": "node:src/image", "kind": "image", "path": "o.png"}}
        }"#,
    )
    .expect("ok");
    let reg = registry();
    let graph = resolve_plan(&plan, &reg).expect("ok");
    let mut impls = ImplRegistry::new();
    impls
        .register(
            "source.create@1".parse().expect("ok"),
            Box::new(UnderproducingImpl),
        )
        .expect("ok");

    let err = execute(&plan, &graph, &reg, &impls, &no_inputs()).expect_err("expected error");
    assert!(matches!(err, ExecError::OutputNotProduced { .. }));
    assert_eq!(err.code(), E_OUTPUT_NOT_PRODUCED);
    assert_eq!(err.node(), "src");
    // Lifts into the central taxonomy as an execution-class failure.
    assert_eq!(err.into_paintop().class, ErrorClass::Execution);
}

#[test]
fn a_missing_implementation_fails_for_a_demanded_node() {
    let plan = parse_plan(
        r#"{
            "paintop": "1.0",
            "inputs": {},
            "nodes": [{"id": "src", "op": "source.create@1"}],
            "exports": {"out": {"resource": "node:src/image", "kind": "image", "path": "o.png"}}
        }"#,
    )
    .expect("ok");
    let reg = registry();
    let graph = resolve_plan(&plan, &reg).expect("ok");
    // Empty impl registry: the demanded node has no kernel.
    let err = execute(&plan, &graph, &reg, &ImplRegistry::new(), &no_inputs())
        .expect_err("expected error");
    assert!(matches!(err, ExecError::ImplementationNotFound { .. }));
    assert_eq!(err.node(), "src");
}

// ---- M3: backend selection + dispatch (bn-2b3) ------------------------------

/// An "invert" op exposing two backends: the `cpu.reference` oracle (identity
/// pass-through) and a `cpu.optimized` kernel that computes the *same logical
/// result* (the reference is the oracle, so the optimized kernel must agree).
fn multi_impl_registry() -> OperationRegistry {
    let mut invert = op("filter.invert@1", &["image"], &["image"]);
    invert.implementations = vec![
        "cpu.reference@1".parse().expect("ok"),
        "cpu.optimized@1".parse().expect("ok"),
    ];
    OperationRegistry::from_manifests([op("source.create@1", &[], &["image"]), invert]).expect("ok")
}

/// The reference and the optimized invert kernels, both registered for the op.
fn multi_impl_implementations() -> ImplRegistry {
    let mut r = ImplRegistry::new();
    r.register(
        "source.create@1".parse().expect("ok"),
        Box::new(SourceImpl(linear_premul(EXTENT))),
    )
    .expect("ok");
    r.register("filter.invert@1".parse().expect("ok"), Box::new(InvertImpl))
        .expect("ok");
    r.register_backend(
        "filter.invert@1".parse().expect("ok"),
        &"cpu.optimized@1".parse().expect("ok"),
        Box::new(InvertImpl),
    )
    .expect("ok");
    r
}

#[test]
fn default_policy_dispatches_the_reference_oracle() {
    let plan = three_node_plan();
    let reg = multi_impl_registry();
    let graph = resolve_plan(&plan, &reg).expect("ok");
    let exec = execute(
        &plan,
        &graph,
        &reg,
        &multi_impl_implementations(),
        &no_inputs(),
    )
    .expect("ok");

    // Even though an optimized backend exists, the default policy stays on the
    // oracle so the plan is byte-identical to the pre-M3 path.
    for e in exec.trace() {
        if let TraceEvent::DispatchCompleted(c) = e {
            assert_eq!(c.implementation, "cpu.reference@1");
        }
    }
}

#[test]
fn preferring_policy_dispatches_optimized_and_matches_reference() {
    let plan = three_node_plan();
    let reg = multi_impl_registry();
    let graph = resolve_plan(&plan, &reg).expect("ok");

    // Reference run (the oracle).
    let reference = execute(
        &plan,
        &graph,
        &reg,
        &multi_impl_implementations(),
        &no_inputs(),
    )
    .expect("ok");

    // Optimized run under a preferring policy.
    let policy = BackendPolicy::prefer([BackendId::new("cpu", "optimized")]);
    let optimized = execute_with_policy(
        &plan,
        &graph,
        &reg,
        &multi_impl_implementations(),
        &no_inputs(),
        &policy,
    )
    .expect("ok");

    // The `used` node (filter.invert) dispatched on the optimized backend; the
    // source (only a reference impl) fell back to the oracle — both recorded.
    let used_impl = optimized
        .trace()
        .iter()
        .find_map(|e| match e {
            TraceEvent::ImplementationSelected(s) if s.node == "used" => {
                Some(s.implementation.as_str())
            }
            _ => None,
        })
        .expect("used node selection recorded");
    assert_eq!(used_impl, "cpu.optimized@1");

    let src_impl = optimized
        .trace()
        .iter()
        .find_map(|e| match e {
            TraceEvent::ImplementationSelected(s) if s.node == "src" => {
                Some(s.implementation.as_str())
            }
            _ => None,
        })
        .expect("src node selection recorded");
    assert_eq!(src_impl, "cpu.reference@1", "src falls back to the oracle");

    // The optimized result is the same logical value as the oracle.
    assert_eq!(
        optimized.output("used", "image").expect("ok").samples(),
        reference.output("used", "image").expect("ok").samples(),
    );
}

#[test]
fn required_unavailable_backend_is_an_explicit_dispatch_error() {
    let plan = three_node_plan();
    let reg = multi_impl_registry();
    let graph = resolve_plan(&plan, &reg).expect("ok");

    // Require wgpu, which no op here exposes: an explicit error, never a silent
    // wrong answer or a quiet fallback.
    let policy = BackendPolicy::require(BackendId::new("wgpu", "separable"));
    let err = execute_with_policy(
        &plan,
        &graph,
        &reg,
        &multi_impl_implementations(),
        &no_inputs(),
        &policy,
    )
    .expect_err("required wgpu must fail");
    // The selection error is carried as the dispatch failure's source, so the
    // unsupported-backend code is preserved end to end.
    match err {
        ExecError::Dispatch { source, .. } => {
            assert_eq!(source.code, E_BACKEND_UNSUPPORTED);
            assert_eq!(source.class, ErrorClass::Policy);
        }
        other => panic!("expected a dispatch failure, got {other:?}"),
    }
}
