//! Backward region-of-interest (ROI) demand analysis (`plan.md` §10–§11,
//! `IR_SPEC` §18).
//!
//! [`compute_demand`](super::demand::compute_demand) answers the *node*-level
//! question — which nodes are reachable from a demand root — but not the
//! *region*-level one: of the pixels a node produces, which does anything
//! downstream actually need? This module answers that by walking the resolved
//! graph **backward** in reverse topological order, accumulating the [`Region`]
//! each node output is demanded over and pushing each operation's
//! [`required_inputs`](paintop_ir::OpContract::required_inputs) onto its
//! producers.
//!
//! The roots are the same as node-level demand — exports, assertions, and
//! requested debug resources (`plan.md` §10.1 phase 7) — but each root now
//! contributes a *region*: an export writes the whole referenced resource, so it
//! demands that output port's full extent; an assertion or debug request likewise
//! demands the full extent of every node output it names (those reductions and
//! dumps read the whole resource).
//!
//! The result is a [`RoiAnalysis`]: the demanded [`Region`] of every live node
//! output, and the set of nodes with a non-empty demand. A node whose every
//! output is demanded over the empty region is **dead** — region-level dead-node
//! elimination, strictly stronger than the node-level pass (a node can be
//! reachable yet contribute no demanded pixel).
//!
//! The analysis is conservative throughout: a [`Region`] is the bounding box of
//! the rects unioned into it, so a demanded region is always a superset of the
//! true contributor set. ROI-restricted execution over these regions therefore
//! reproduces full-image execution *bit-for-bit inside the demanded region* —
//! the property the differential suite pins.

use std::collections::{BTreeMap, BTreeSet};

use paintop_ir::{
    CheckedGraph, ContractRegistry, Descriptors, OutputRegionDemand, Plan, Reference, Region,
    ResolvedGraph, propagate_demand,
};

use super::error::{ExecError, ExecResult};

/// The demand [`Region`] of one node output port, keyed `(node_id, port)`.
type OutputDemandMap = BTreeMap<(String, String), Region>;

/// The result of backward ROI analysis over a checked graph.
///
/// Carries the demanded [`Region`] of every node output that any root needs, and
/// the set of nodes that contribute at least one demanded pixel. A node absent
/// from [`demanded_nodes`](RoiAnalysis::demanded_nodes) is dead: nothing
/// downstream reads any pixel it produces.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RoiAnalysis {
    output_demand: OutputDemandMap,
    input_demand: BTreeMap<String, Region>,
    demanded_nodes: BTreeSet<String>,
}

impl RoiAnalysis {
    /// The [`Region`] demanded of node `node`'s output port `port`, or
    /// [`Region::EMPTY`] if nothing downstream reads it.
    #[must_use]
    pub fn output_region(&self, node: &str, port: &str) -> Region {
        self.output_demand
            .get(&(node.to_owned(), port.to_owned()))
            .copied()
            .unwrap_or(Region::EMPTY)
    }

    /// The [`Region`] demanded of the external `input:<input>` resource — the
    /// union of the regions every consuming node demands of it. [`Region::EMPTY`]
    /// when no live node reads the input.
    #[must_use]
    pub fn input_region(&self, input: &str) -> Region {
        self.input_demand
            .get(input)
            .copied()
            .unwrap_or(Region::EMPTY)
    }

    /// Whether `node` contributes at least one demanded pixel (is *live* at the
    /// region level). A node not demanded here is eliminated.
    #[must_use]
    pub fn is_demanded(&self, node: &str) -> bool {
        self.demanded_nodes.contains(node)
    }

    /// The region-level demanded node set, in deterministic id order.
    pub fn demanded_nodes(&self) -> impl Iterator<Item = &str> {
        self.demanded_nodes.iter().map(String::as_str)
    }

    /// The number of node outputs carrying a non-empty demand.
    #[must_use]
    pub fn demanded_output_count(&self) -> usize {
        self.output_demand
            .values()
            .filter(|r| !r.is_empty())
            .count()
    }
}

/// Compute the backward ROI demand of `graph` from its export/assertion/debug
/// roots (`plan.md` §10–§11).
///
/// `checked` supplies the concrete input descriptors every node's
/// [`required_inputs`](paintop_ir::OpContract::required_inputs) needs (a
/// full-domain op reads its input extent from them); `contracts` supplies the
/// executable contracts; `plan` supplies node params and the assertion/debug
/// roots. Each export and each node-reference inside an assertion or debug
/// request demands the **full extent** of the referenced output.
///
/// # Errors
/// - [`ExecError::ImplementationNotFound`] if a live node has no registered
///   contract (the contract registry is incomplete for this graph).
/// - any contract error raised by an operation's
///   [`required_inputs`](paintop_ir::OpContract::required_inputs).
pub fn analyze_roi(
    plan: &Plan,
    graph: &ResolvedGraph,
    checked: &CheckedGraph,
    contracts: &ContractRegistry,
) -> ExecResult<RoiAnalysis> {
    // Seed roots: each export / assertion / debug reference demands the full
    // extent of the referenced node output.
    let mut seeds: OutputDemandMap = OutputDemandMap::new();
    for export in graph.exports() {
        seed_full_extent(&export.resource, checked, &mut seeds);
    }
    let mut roots: Vec<Reference> = Vec::new();
    for assertion in &plan.assertions {
        collect_node_refs(assertion, graph, &mut roots);
    }
    collect_node_refs(
        &serde_json::Value::Object(plan.evidence.clone()),
        graph,
        &mut roots,
    );
    for reference in &roots {
        seed_full_extent(reference, checked, &mut seeds);
    }

    analyze_roi_from_seeds(plan, graph, checked, contracts, &seeds)
}

/// Backward ROI analysis from explicit per-output seed [`Region`]s, keyed
/// `(node_id, port)` (`plan.md` §10–§11).
///
/// The same backward walk [`analyze_roi`] runs, but with the demand roots given
/// directly rather than derived from exports/assertions/debug. This lets a caller
/// (notably the ROI-restricted-vs-full differential suite) demand an arbitrary
/// *sub-region* of an output and obtain the input regions that feed it.
///
/// # Errors
/// Same as [`analyze_roi`].
pub fn analyze_roi_from_seeds(
    plan: &Plan,
    graph: &ResolvedGraph,
    checked: &CheckedGraph,
    contracts: &ContractRegistry,
    seeds: &BTreeMap<(String, String), Region>,
) -> ExecResult<RoiAnalysis> {
    let params_by_node: BTreeMap<&str, serde_json::Value> = plan
        .nodes
        .iter()
        .map(|n| (n.id.as_str(), serde_json::Value::Object(n.params.clone())))
        .collect();

    let mut output_demand: OutputDemandMap = OutputDemandMap::new();
    let mut input_demand: BTreeMap<String, Region> = BTreeMap::new();
    for (key, region) in seeds {
        if !region.is_empty() {
            let entry = output_demand.entry(key.clone()).or_insert(Region::EMPTY);
            *entry = entry.union(*region);
        }
    }

    // Walk backward in reverse topological order: a node is processed only
    //    after every consumer that could demand it, so its accumulated output
    //    demand is complete when we propagate it onto its inputs.
    for node_id in graph.topological_order().iter().rev() {
        let Some(node) = graph.node(node_id) else {
            continue;
        };

        // Gather this node's accumulated per-output-port demand.
        let mut out_demand: OutputRegionDemand = OutputRegionDemand::new();
        if let Some(ports) = checked.node_outputs(node_id) {
            for port in ports.keys() {
                let region = output_demand
                    .get(&(node_id.clone(), port.clone()))
                    .copied()
                    .unwrap_or(Region::EMPTY);
                if !region.is_empty() {
                    out_demand.insert(port.clone(), region);
                }
            }
        }
        // No demanded output -> nothing to propagate (the node is dead so far).
        if out_demand.is_empty() {
            continue;
        }

        // Propagate backward through the contract to per-input-port regions.
        let contract =
            contracts
                .get(&node.op)
                .ok_or_else(|| ExecError::ImplementationNotFound {
                    node: node_id.clone(),
                    op: node.op.to_string(),
                })?;
        let input_descriptors = input_descriptors(node, checked);
        let params = params_by_node
            .get(node_id.as_str())
            .cloned()
            .unwrap_or(serde_json::Value::Null);
        let needed = propagate_demand(contract, &out_demand, &input_descriptors, &params).map_err(
            |source| ExecError::Dispatch {
                node: node_id.clone(),
                op: node.op.to_string(),
                source: Box::new(source),
            },
        )?;

        // Push each input-port demand onto the producing node output, or onto the
        // external input it reads.
        for (port, region) in needed {
            if region.is_empty() {
                continue;
            }
            match node.inputs.get(&port) {
                Some(Reference::Node {
                    node: upstream,
                    port: upstream_port,
                }) => {
                    let key = (upstream.clone(), upstream_port.clone());
                    let entry = output_demand.entry(key).or_insert(Region::EMPTY);
                    *entry = entry.union(region);
                }
                Some(Reference::Input { input }) => {
                    let entry = input_demand.entry(input.clone()).or_insert(Region::EMPTY);
                    *entry = entry.union(region);
                }
                None => {}
            }
        }
    }

    // 3. A node is region-level live iff one of its outputs has a non-empty demand.
    let mut demanded_nodes: BTreeSet<String> = BTreeSet::new();
    for ((node, _port), region) in &output_demand {
        if !region.is_empty() {
            demanded_nodes.insert(node.clone());
        }
    }

    Ok(RoiAnalysis {
        output_demand,
        input_demand,
        demanded_nodes,
    })
}

/// Seed `out` with the full-extent demand of the node output `reference` names.
///
/// An `input:` reference names no node output (external inputs are demanded
/// implicitly through their consumers), and a `node:` reference whose descriptor
/// the checked graph does not carry contributes nothing.
fn seed_full_extent(reference: &Reference, checked: &CheckedGraph, out: &mut OutputDemandMap) {
    if let Reference::Node { node, port } = reference
        && let Some(descriptor) = checked.output(node, port)
    {
        let region = Region::from_extent(descriptor.extent());
        let entry = out
            .entry((node.clone(), port.clone()))
            .or_insert(Region::EMPTY);
        *entry = entry.union(region);
    }
}

/// Assemble the input descriptors of `node` from the checked graph, keyed by
/// input port — the `Descriptors` an op's `required_inputs` reads.
fn input_descriptors(node: &paintop_ir::ResolvedNode, checked: &CheckedGraph) -> Descriptors {
    let mut descriptors = Descriptors::new();
    for (port, reference) in &node.inputs {
        match reference {
            Reference::Node {
                node: upstream,
                port: upstream_port,
            } => {
                if let Some(descriptor) = checked.output(upstream, upstream_port) {
                    descriptors.insert(port.clone(), *descriptor);
                }
            }
            // `input:` edges read their concrete descriptor from the checked
            // graph's retained external-input map. A neighbourhood op's
            // `required_inputs` needs the input extent (to clamp the kernel halo),
            // so supplying it here makes the ROI of an op reading an external input
            // correct rather than relying on the conservative whole-plane fallback.
            Reference::Input { input } => {
                if let Some(descriptor) = checked.input(input) {
                    descriptors.insert(port.clone(), *descriptor);
                }
            }
        }
    }
    descriptors
}

/// Recursively collect every `node:<id>/<port>` reference in `value` whose node
/// exists in `graph`.
fn collect_node_refs(value: &serde_json::Value, graph: &ResolvedGraph, out: &mut Vec<Reference>) {
    match value {
        serde_json::Value::String(s) => {
            if let Ok(reference @ Reference::Node { .. }) = Reference::parse(s)
                && let Reference::Node { node, .. } = &reference
                && graph.node(node).is_some()
            {
                out.push(reference);
            }
        }
        serde_json::Value::Array(items) => {
            for item in items {
                collect_node_refs(item, graph, out);
            }
        }
        serde_json::Value::Object(map) => {
            for item in map.values() {
                collect_node_refs(item, graph, out);
            }
        }
        _ => {}
    }
}

#[allow(
    clippy::missing_errors_doc,
    reason = "test-only helper, errors are asserted directly"
)]
#[cfg(test)]
mod tests {
    use super::analyze_roi;
    use paintop_ir::{
        AlphaRepresentation, ChannelLayout, ColorEncoding, ColorRange, ContractRegistry,
        CoordinateConvention, Descriptors, Extent, ImageDescriptor, InputRegions, OpContract,
        OutputDescriptors, OutputRegions, Plan, Rect, Region, ResourceDescriptor, ResourceKind,
        ScalarType, SemanticRole, parse_plan, resolve_plan,
    };
    use paintop_ir::{
        AssertionResult, DeterminismTier, InputSpec, OperationManifest, OperationRegistry,
        OutputSpec, RoiCategory, RoiPolicy, TestMetadata, check_graph,
    };
    use std::collections::BTreeMap;

    const EXTENT: Extent = Extent::new(64, 48);

    fn image() -> ResourceDescriptor {
        ResourceDescriptor::Image(ImageDescriptor {
            extent: EXTENT,
            layout: ChannelLayout::Rgba,
            scalar: ScalarType::F32,
            color: ColorEncoding::LinearSrgb,
            range: ColorRange::SceneReferred,
            alpha: AlphaRepresentation::Premultiplied,
            coordinates: CoordinateConvention::PixelCenterUpperLeft,
            semantic: SemanticRole::Color,
        })
    }

    // A source op (no inputs, one `image` output).
    struct Source;
    impl OpContract for Source {
        fn declared_inputs(&self) -> Vec<(String, ResourceKind)> {
            vec![]
        }
        fn declared_outputs(&self) -> Vec<(String, ResourceKind)> {
            vec![("image".to_owned(), ResourceKind::Image)]
        }
        fn infer_outputs(
            &self,
            _i: &Descriptors,
            _p: &serde_json::Value,
        ) -> paintop_ir::Result<OutputDescriptors> {
            let mut o = OutputDescriptors::new();
            o.insert("image".to_owned(), image());
            Ok(o)
        }
        fn required_inputs(
            &self,
            _o: &OutputRegions,
            _i: &Descriptors,
            _p: &serde_json::Value,
        ) -> paintop_ir::Result<InputRegions> {
            Ok(InputRegions::new())
        }
        fn validate_postconditions(
            &self,
            _o: &OutputDescriptors,
            _p: &serde_json::Value,
        ) -> paintop_ir::Result<Vec<AssertionResult>> {
            Ok(vec![])
        }
    }

    // A pointwise op (input region == output region).
    struct Pointwise;
    impl OpContract for Pointwise {
        fn declared_inputs(&self) -> Vec<(String, ResourceKind)> {
            vec![("image".to_owned(), ResourceKind::Image)]
        }
        fn declared_outputs(&self) -> Vec<(String, ResourceKind)> {
            vec![("image".to_owned(), ResourceKind::Image)]
        }
        fn infer_outputs(
            &self,
            i: &Descriptors,
            _p: &serde_json::Value,
        ) -> paintop_ir::Result<OutputDescriptors> {
            let mut o = OutputDescriptors::new();
            o.insert("image".to_owned(), i["image"]);
            Ok(o)
        }
        fn required_inputs(
            &self,
            o: &OutputRegions,
            _i: &Descriptors,
            _p: &serde_json::Value,
        ) -> paintop_ir::Result<InputRegions> {
            let mut r = InputRegions::new();
            if let Some(region) = o.get("image") {
                r.insert("image".to_owned(), *region);
            }
            Ok(r)
        }
        fn validate_postconditions(
            &self,
            _o: &OutputDescriptors,
            _p: &serde_json::Value,
        ) -> paintop_ir::Result<Vec<AssertionResult>> {
            Ok(vec![])
        }
    }

    // A halo op (input region == output region dilated by a fixed radius).
    struct Halo(u32);
    impl OpContract for Halo {
        fn declared_inputs(&self) -> Vec<(String, ResourceKind)> {
            vec![("image".to_owned(), ResourceKind::Image)]
        }
        fn declared_outputs(&self) -> Vec<(String, ResourceKind)> {
            vec![("image".to_owned(), ResourceKind::Image)]
        }
        fn infer_outputs(
            &self,
            i: &Descriptors,
            _p: &serde_json::Value,
        ) -> paintop_ir::Result<OutputDescriptors> {
            let mut o = OutputDescriptors::new();
            o.insert("image".to_owned(), i["image"]);
            Ok(o)
        }
        fn required_inputs(
            &self,
            o: &OutputRegions,
            i: &Descriptors,
            _p: &serde_json::Value,
        ) -> paintop_ir::Result<InputRegions> {
            let mut r = InputRegions::new();
            if let Some(region) = o.get("image") {
                let extent = i["image"].extent();
                let grown = Region::from_rect(*region)
                    .dilate(self.0)
                    .clamp_to_extent(extent);
                r.insert("image".to_owned(), grown.bounding_rect());
            }
            Ok(r)
        }
        fn validate_postconditions(
            &self,
            _o: &OutputDescriptors,
            _p: &serde_json::Value,
        ) -> paintop_ir::Result<Vec<AssertionResult>> {
            Ok(vec![])
        }
    }

    fn manifest(id: &str, inputs: &[&str], outputs: &[&str]) -> OperationManifest {
        OperationManifest {
            id: id.parse().unwrap(),
            impl_version: 1,
            summary: String::new(),
            determinism: DeterminismTier::Exact,
            roi: RoiPolicy {
                category: RoiCategory::Pointwise,
                halo_px: None,
            },
            inputs: inputs
                .iter()
                .map(|n| InputSpec {
                    name: (*n).to_owned(),
                    kind: ResourceKind::Image,
                    required: true,
                    doc: String::new(),
                })
                .collect(),
            outputs: outputs
                .iter()
                .map(|n| OutputSpec {
                    name: (*n).to_owned(),
                    kind: ResourceKind::Image,
                    doc: String::new(),
                })
                .collect(),
            params: vec![],
            implementations: vec!["cpu.reference@1".parse().unwrap()],
            test: TestMetadata::default(),
        }
    }

    fn registry() -> OperationRegistry {
        OperationRegistry::from_manifests([
            manifest("source.create@1", &[], &["image"]),
            manifest("filter.point@1", &["image"], &["image"]),
            manifest("filter.halo@1", &["image"], &["image"]),
        ])
        .unwrap()
    }

    fn contracts(halo: u32) -> ContractRegistry {
        let mut c = ContractRegistry::new();
        c.register("source.create@1".parse().unwrap(), Box::new(Source))
            .unwrap();
        c.register("filter.point@1".parse().unwrap(), Box::new(Pointwise))
            .unwrap();
        c.register("filter.halo@1".parse().unwrap(), Box::new(Halo(halo)))
            .unwrap();
        c
    }

    fn analyze(plan: &Plan, halo: u32) -> super::RoiAnalysis {
        let reg = registry();
        let graph = resolve_plan(plan, &reg).unwrap();
        let checked = check_graph(plan, &graph, &reg, &contracts(halo), &BTreeMap::new()).unwrap();
        analyze_roi(plan, &graph, &checked, &contracts(halo)).unwrap()
    }

    #[test]
    fn pointwise_chain_propagates_the_export_extent() {
        let plan = parse_plan(
            r#"{
                "paintop": "1.0", "inputs": {},
                "nodes": [
                    {"id": "src", "op": "source.create@1"},
                    {"id": "a", "op": "filter.point@1", "in": {"image": "node:src/image"}},
                    {"id": "b", "op": "filter.point@1", "in": {"image": "node:a/image"}}
                ],
                "exports": {"out": {"resource": "node:b/image", "kind": "image", "path": "o.png"}}
            }"#,
        )
        .unwrap();
        let roi = analyze(&plan, 0);
        // The export demands b's full extent; pointwise propagation keeps it full
        // back through a and src.
        let full = Rect::from_extent(EXTENT);
        assert_eq!(roi.output_region("b", "image").bounding_rect(), full);
        assert_eq!(roi.output_region("a", "image").bounding_rect(), full);
        assert_eq!(roi.output_region("src", "image").bounding_rect(), full);
        assert!(roi.is_demanded("src") && roi.is_demanded("a") && roi.is_demanded("b"));
    }

    #[test]
    fn halo_op_grows_the_demand_upstream() {
        let plan = parse_plan(
            r#"{
                "paintop": "1.0", "inputs": {},
                "nodes": [
                    {"id": "src", "op": "source.create@1"},
                    {"id": "blur", "op": "filter.halo@1", "in": {"image": "node:src/image"}}
                ],
                "exports": {"out": {"resource": "node:blur/image", "kind": "image", "path": "o.png"}}
            }"#,
        )
        .unwrap();
        // The export demands blur's full extent, so the halo clamps back to the
        // full source extent — the upstream demand is still the whole image.
        let roi = analyze(&plan, 4);
        assert_eq!(
            roi.output_region("src", "image").bounding_rect(),
            Rect::from_extent(EXTENT)
        );
    }

    #[test]
    fn a_node_outside_the_demand_set_is_eliminated() {
        let plan = parse_plan(
            r#"{
                "paintop": "1.0", "inputs": {},
                "nodes": [
                    {"id": "src", "op": "source.create@1"},
                    {"id": "used", "op": "filter.point@1", "in": {"image": "node:src/image"}},
                    {"id": "dead", "op": "filter.point@1", "in": {"image": "node:src/image"}}
                ],
                "exports": {"out": {"resource": "node:used/image", "kind": "image", "path": "o.png"}}
            }"#,
        )
        .unwrap();
        let roi = analyze(&plan, 0);
        assert!(roi.is_demanded("used"));
        assert!(roi.is_demanded("src"));
        // `dead` feeds no export/assertion/debug: region-level eliminated.
        assert!(!roi.is_demanded("dead"));
        assert!(roi.output_region("dead", "image").is_empty());
    }

    #[test]
    fn an_assertion_reference_demands_the_full_extent() {
        let plan = parse_plan(
            r#"{
                "paintop": "1.0", "inputs": {},
                "nodes": [
                    {"id": "src", "op": "source.create@1"},
                    {"id": "kept", "op": "filter.point@1", "in": {"image": "node:src/image"}}
                ],
                "assertions": [{"kind": "assert.x@1", "subject": "node:kept/image"}],
                "exports": {}
            }"#,
        )
        .unwrap();
        let roi = analyze(&plan, 0);
        assert!(roi.is_demanded("kept"));
        assert_eq!(
            roi.output_region("kept", "image").bounding_rect(),
            Rect::from_extent(EXTENT)
        );
    }
}
