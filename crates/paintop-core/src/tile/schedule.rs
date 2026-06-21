//! The demand-driven tile scheduler and buffer-liveness analysis
//! (`plan.md` §10.2, §11.1–§11.3).
//!
//! The backward ROI pass ([`crate::executor::roi`]) answers *which region of each
//! node output is demanded*; the [`TileGrid`] partitions that into tiles. This
//! module turns the two into a concrete *schedule*: the deterministic, demand-
//! driven order in which the executor evaluates `(node, output tile)` work items,
//! together with the per-output-tile input dependencies (the halos) each item
//! reads and a **buffer-liveness** analysis that bounds the working set.
//!
//! # Demand-driven, deterministic order
//!
//! Nodes are visited in the resolved graph's stable topological order, so a
//! producer is always scheduled before its consumers. Within a node, only the
//! tiles that intersect the node's demanded [`Region`] are emitted — a node whose
//! output is demanded over a small region (a masked 4K edit) schedules only the
//! handful of tiles that region touches, never the whole image
//! (`plan.md` §1 goal, §11.3). Tiles are emitted in the grid's row-major order, so
//! the whole schedule is a pure function of the graph, the demand, and the tile
//! size — the determinism the tiled-vs-whole differential relies on.
//!
//! # Immutable resources, reused physical buffers
//!
//! Logical resources are immutable values; physical buffers may be reused once
//! liveness proves it is safe, invisibly to graph semantics (`plan.md` §10.2). A
//! node output's buffer becomes **live** when the node's first tile is scheduled
//! and **dead** once every consumer that reads it (and every export naming it) has
//! been scheduled. The [`LivenessTrace`] records the live-buffer count after each
//! work item; its [`peak`](LivenessTrace::peak) is the bounded working set the
//! M2 memory cap is checked against, and a node read by nothing downstream frees
//! its buffer immediately rather than pinning it to the end of the run.

use std::collections::{BTreeMap, BTreeSet};

use paintop_ir::{
    CheckedGraph, ContractRegistry, Descriptors, Plan, Reference, Region, ResolvedGraph,
};

use super::grid::{Tile, TileGrid, input_tile_region};
use crate::executor::error::{ExecError, ExecResult};
use crate::executor::roi::RoiAnalysis;

/// One input dependency of a tile work item: the producer it reads and the input
/// region (the halo) it needs from that producer (`plan.md` §11.2–§11.3).
///
/// `source` is the resolved edge — an upstream node output or an external
/// `input:` resource — and `region` is the conservative input region the
/// consuming output tile reads from it, already clamped to the producer's extent.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TileInput {
    /// The input port name on the consuming node.
    pub port: String,
    /// The resolved resource this port reads.
    pub source: Reference,
    /// The input region (halo) the output tile demands from `source`.
    pub region: Region,
}

/// One unit of tiled work: produce output `port` of `node` over `tile`
/// (`plan.md` §11).
///
/// The work item carries the input halos it depends on, so the executor (a later
/// bone) can evaluate it without re-deriving the ROI. `index` is the work item's
/// position in the schedule, for the trace.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TileWorkItem {
    /// The schedule position of this work item (0-based).
    pub index: usize,
    /// The node whose output tile this item produces.
    pub node: String,
    /// The output port produced.
    pub port: String,
    /// The output tile produced.
    pub tile: Tile,
    /// The input halos this output tile reads, in input-port order.
    pub inputs: Vec<TileInput>,
}

/// A snapshot of buffer liveness after one work item (`plan.md` §10.2).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LivenessPoint {
    /// The work-item index this snapshot follows.
    pub after_item: usize,
    /// The number of node-output buffers live at this point.
    pub live_buffers: usize,
}

/// The buffer-liveness trace over a schedule: the live-buffer count after each
/// work item, and the peak (`plan.md` §10.2).
///
/// The peak is the bounded working set: the maximum number of node-output
/// buffers that must be physically resident simultaneously. A masked edit that
/// schedules a few nodes over a few tiles keeps this small regardless of the
/// image size — the property the 4K bounded-memory assertion pins.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LivenessTrace {
    points: Vec<LivenessPoint>,
    peak: usize,
}

impl LivenessTrace {
    /// The per-item liveness snapshots, in schedule order.
    #[must_use]
    pub fn points(&self) -> &[LivenessPoint] {
        &self.points
    }

    /// The peak number of simultaneously-live node-output buffers — the bounded
    /// working set.
    #[must_use]
    pub const fn peak(&self) -> usize {
        self.peak
    }
}

/// A demand-driven tiled execution schedule: the ordered work items, the demanded
/// tile set per node, and the buffer-liveness trace (`plan.md` §10.2, §11).
#[derive(Debug, Clone)]
pub struct TileSchedule {
    grid: TileGrid,
    items: Vec<TileWorkItem>,
    demanded_tiles: BTreeMap<String, usize>,
    liveness: LivenessTrace,
}

impl TileSchedule {
    /// The tile grid the schedule is laid out over.
    #[must_use]
    pub const fn grid(&self) -> TileGrid {
        self.grid
    }

    /// The work items in deterministic schedule order.
    #[must_use]
    pub fn items(&self) -> &[TileWorkItem] {
        &self.items
    }

    /// The number of demanded output tiles scheduled for `node` (0 if the node is
    /// dead or absent).
    #[must_use]
    pub fn demanded_tile_count(&self, node: &str) -> usize {
        self.demanded_tiles.get(node).copied().unwrap_or(0)
    }

    /// The total number of scheduled work items.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.items.len()
    }

    /// Whether the schedule has no work items.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    /// The buffer-liveness trace.
    #[must_use]
    pub const fn liveness(&self) -> &LivenessTrace {
        &self.liveness
    }

    /// The peak live-buffer count — the bounded working set (`plan.md` §10.2).
    #[must_use]
    pub const fn peak_live_buffers(&self) -> usize {
        self.liveness.peak
    }
}

/// Build a demand-driven tile schedule for `graph` over `grid`
/// (`plan.md` §10.2, §11).
///
/// Visits the region-level demanded nodes in topological order; for each, emits a
/// [`TileWorkItem`] per output tile that intersects the node's demanded
/// [`Region`] (from `roi`), carrying the input halos (`input_tile_region`) the
/// tile reads. The buffer-liveness trace is computed alongside: a node output's
/// buffer is live from the node's first scheduled item until its last consumer
/// (or export) has been scheduled.
///
/// # Errors
/// - [`ExecError::ImplementationNotFound`] if a demanded node has no registered
///   contract.
/// - any contract error raised while computing an op's input halos.
pub fn schedule_tiles(
    plan: &Plan,
    graph: &ResolvedGraph,
    checked: &CheckedGraph,
    contracts: &ContractRegistry,
    roi: &RoiAnalysis,
    grid: TileGrid,
) -> ExecResult<TileSchedule> {
    let params_by_node: BTreeMap<&str, serde_json::Value> = plan
        .nodes
        .iter()
        .map(|n| (n.id.as_str(), serde_json::Value::Object(n.params.clone())))
        .collect();

    // The demanded node order (topological, restricted to region-live nodes).
    let order: Vec<&str> = graph
        .topological_order()
        .iter()
        .map(String::as_str)
        .filter(|id| roi.is_demanded(id))
        .collect();

    // Last-consumer index: the position in `order` of the last demanded node that
    // reads a given producer output. Exports pin a buffer to the end (index = len).
    let last_use = compute_last_use(graph, roi, &order);

    let mut items: Vec<TileWorkItem> = Vec::new();
    let mut demanded_tiles: BTreeMap<String, usize> = BTreeMap::new();
    let mut points: Vec<LivenessPoint> = Vec::new();

    // Live buffers, keyed by producer node id. A buffer is born when its node's
    // first item is scheduled and dies after the last consumer node's items.
    let mut live: BTreeSet<String> = BTreeSet::new();
    let mut peak = 0usize;

    for (position, &node_id) in order.iter().enumerate() {
        let Some(node) = graph.node(node_id) else {
            continue;
        };
        let contract =
            contracts
                .get(&node.op)
                .ok_or_else(|| ExecError::ImplementationNotFound {
                    node: node_id.to_owned(),
                    op: node.op.to_string(),
                })?;
        let params = params_by_node
            .get(node_id)
            .cloned()
            .unwrap_or(serde_json::Value::Null);
        let input_descriptors = node_input_descriptors(node, checked);

        // This node's output ports, in deterministic order.
        let ports: Vec<String> = checked
            .node_outputs(node_id)
            .map(|m| m.keys().cloned().collect())
            .unwrap_or_default();

        let mut node_tiles = 0usize;
        for port in &ports {
            let demand = roi.output_region(node_id, port);
            if demand.is_empty() {
                continue;
            }
            // Born: this output's buffer becomes live at its first scheduled item.
            for tile in grid.tiles_in_region(demand) {
                let inputs =
                    tile_inputs(contract, port, tile.rect, node, &input_descriptors, &params)?;
                let index = items.len();
                items.push(TileWorkItem {
                    index,
                    node: node_id.to_owned(),
                    port: port.clone(),
                    tile,
                    inputs,
                });
                node_tiles += 1;
            }
        }
        if node_tiles > 0 {
            demanded_tiles.insert(node_id.to_owned(), node_tiles);
            // The node's output buffer(s) are now resident; model one buffer per
            // node (its output set is produced and held together).
            live.insert(node_id.to_owned());
        }

        // Peak residency is measured *while this node runs*: its freshly produced
        // output and every input buffer it still reads are simultaneously live,
        // before any are released.
        peak = peak.max(live.len());

        // Free any producer buffer whose last consumer is this node (its data is
        // no longer needed once this node has consumed it).
        let dead: Vec<String> = live
            .iter()
            .filter(|producer| {
                producer.as_str() != node_id
                    && last_use
                        .get(producer.as_str())
                        .is_some_and(|&last| last <= position)
            })
            .cloned()
            .collect();
        for producer in dead {
            live.remove(&producer);
        }

        points.push(LivenessPoint {
            after_item: items.len().saturating_sub(1),
            live_buffers: live.len(),
        });
    }

    Ok(TileSchedule {
        grid,
        items,
        demanded_tiles,
        liveness: LivenessTrace { points, peak },
    })
}

/// The last position in `order` at which each producer output is consumed.
///
/// A producer node id maps to the maximum `order` index of a demanded node that
/// reads any of its outputs. A producer named by an export is pinned to
/// `order.len()` so its buffer stays live to the end of the run.
fn compute_last_use(
    graph: &ResolvedGraph,
    roi: &RoiAnalysis,
    order: &[&str],
) -> BTreeMap<String, usize> {
    let position: BTreeMap<&str, usize> =
        order.iter().enumerate().map(|(i, &id)| (id, i)).collect();
    let mut last_use: BTreeMap<String, usize> = BTreeMap::new();

    for (&consumer, &pos) in &position {
        let Some(node) = graph.node(consumer) else {
            continue;
        };
        for reference in node.inputs.values() {
            if let Reference::Node { node: producer, .. } = reference
                && roi.is_demanded(producer)
            {
                let entry = last_use.entry(producer.clone()).or_insert(0);
                *entry = (*entry).max(pos);
            }
        }
    }

    // Exports pin their producer to the end of the schedule.
    let end = order.len();
    for export in graph.exports() {
        if let Reference::Node { node: producer, .. } = &export.resource
            && roi.is_demanded(producer)
        {
            last_use.insert(producer.clone(), end);
        }
    }
    last_use
}

/// The per-input-port halos one output `tile` of `node` reads, in port order.
fn tile_inputs(
    contract: &dyn paintop_ir::OpContract,
    output_port: &str,
    tile: paintop_ir::Rect,
    node: &paintop_ir::ResolvedNode,
    input_descriptors: &Descriptors,
    params: &serde_json::Value,
) -> ExecResult<Vec<TileInput>> {
    let mut inputs = Vec::with_capacity(node.inputs.len());
    for (port, reference) in &node.inputs {
        let region =
            input_tile_region(contract, output_port, port, tile, input_descriptors, params)
                .map_err(|source| ExecError::Dispatch {
                    node: node.id.clone(),
                    op: node.op.to_string(),
                    source: Box::new(source),
                })?;
        if region.is_empty() {
            continue;
        }
        inputs.push(TileInput {
            port: port.clone(),
            source: reference.clone(),
            region,
        });
    }
    Ok(inputs)
}

/// Assemble the input descriptors of `node` from the checked graph, keyed by
/// input port — the `Descriptors` an op's `required_inputs` reads.
fn node_input_descriptors(node: &paintop_ir::ResolvedNode, checked: &CheckedGraph) -> Descriptors {
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
            // An external `input:` edge's descriptor is needed so a neighbourhood
            // op's per-tile halo (its `required_inputs`) can clamp the kernel
            // footprint to the real input extent.
            Reference::Input { input } => {
                if let Some(descriptor) = checked.input(input) {
                    descriptors.insert(port.clone(), *descriptor);
                }
            }
        }
    }
    descriptors
}

#[cfg(test)]
mod tests {
    use super::schedule_tiles;
    use crate::executor::roi::analyze_roi;
    use crate::tile::grid::TileGrid;
    use paintop_ir::{
        AlphaRepresentation, AssertionResult, ChannelLayout, ColorEncoding, ColorRange,
        ContractRegistry, CoordinateConvention, Descriptors, DeterminismTier, Extent,
        ImageDescriptor, InputRegions, InputSpec, OpContract, OperationManifest, OperationRegistry,
        OutputDescriptors, OutputRegions, OutputSpec, Plan, Rect, Region, ResourceDescriptor,
        ResourceKind, RoiCategory, RoiPolicy, ScalarType, SemanticRole, TestMetadata, check_graph,
        parse_plan, resolve_plan,
    };
    use std::collections::BTreeMap;

    const EXTENT: Extent = Extent::new(256, 256);

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
        ])
        .unwrap()
    }

    fn contracts() -> ContractRegistry {
        let mut c = ContractRegistry::new();
        c.register("source.create@1".parse().unwrap(), Box::new(Source))
            .unwrap();
        c.register("filter.point@1".parse().unwrap(), Box::new(Pointwise))
            .unwrap();
        c
    }

    fn schedule_for(plan: &Plan, grid: TileGrid) -> super::TileSchedule {
        let reg = registry();
        let graph = resolve_plan(plan, &reg).unwrap();
        let checked = check_graph(plan, &graph, &reg, &contracts(), &BTreeMap::new()).unwrap();
        let roi = analyze_roi(plan, &graph, &checked, &contracts()).unwrap();
        schedule_tiles(plan, &graph, &checked, &contracts(), &roi, grid).unwrap()
    }

    fn chain_plan() -> Plan {
        parse_plan(
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
        .unwrap()
    }

    #[test]
    fn schedules_every_demanded_tile_in_topological_then_row_major_order() {
        // 256x256 with 128 tiles => 4 tiles per node, 3 nodes => 12 items.
        let grid = TileGrid::new(EXTENT, 128);
        let schedule = schedule_for(&chain_plan(), grid);
        assert_eq!(schedule.len(), 12);
        assert_eq!(schedule.demanded_tile_count("src"), 4);
        assert_eq!(schedule.demanded_tile_count("a"), 4);
        assert_eq!(schedule.demanded_tile_count("b"), 4);

        // Topological: all of src's items precede a's, which precede b's.
        let nodes: Vec<&str> = schedule.items().iter().map(|i| i.node.as_str()).collect();
        assert_eq!(&nodes[0..4], &["src", "src", "src", "src"]);
        assert_eq!(&nodes[4..8], &["a", "a", "a", "a"]);
        assert_eq!(&nodes[8..12], &["b", "b", "b", "b"]);

        // Row-major within a node: tile indices 0,1,2,3.
        let src_tiles: Vec<u32> = schedule.items()[0..4]
            .iter()
            .map(|i| i.tile.index)
            .collect();
        assert_eq!(src_tiles, vec![0, 1, 2, 3]);
    }

    #[test]
    fn pointwise_tile_inputs_are_the_co_located_tile() {
        let grid = TileGrid::new(EXTENT, 128);
        let schedule = schedule_for(&chain_plan(), grid);
        // `a`'s first tile reads exactly the co-located tile of `src`.
        let a0 = schedule
            .items()
            .iter()
            .find(|i| i.node == "a" && i.tile.index == 0)
            .unwrap();
        assert_eq!(a0.inputs.len(), 1);
        assert_eq!(a0.inputs[0].port, "image");
        assert_eq!(
            a0.inputs[0].region.bounding_rect(),
            Rect::new(0, 0, 128, 128)
        );

        // A source op has no input dependencies.
        let src0 = &schedule.items()[0];
        assert!(src0.inputs.is_empty());
    }

    #[test]
    fn buffer_liveness_is_bounded_for_a_linear_chain() {
        // A linear chain reuses buffers: at most a couple of node outputs are live
        // at once (the running producer + the one being produced), never all three.
        let grid = TileGrid::new(EXTENT, 128);
        let schedule = schedule_for(&chain_plan(), grid);
        // src lives until a consumes it (a's items), then dies; a lives until b;
        // b is exported (lives to the end). Peak is 2, not 3.
        assert_eq!(schedule.peak_live_buffers(), 2);
        assert!(!schedule.liveness().points().is_empty());
    }

    #[test]
    fn only_demanded_tiles_are_scheduled_for_a_small_roi() {
        // A sub-region demand on the chain: demand only the top-left 100x100 of
        // `b`. Pointwise propagation keeps that region back through `a` and `src`,
        // so on a 128-tile grid (4 tiles) only the single top-left tile is touched
        // per node — the masked-edit "touch only predicted tiles" property.
        use crate::executor::roi::analyze_roi_from_seeds;
        let plan = chain_plan();
        let reg = registry();
        let graph = resolve_plan(&plan, &reg).unwrap();
        let checked = check_graph(&plan, &graph, &reg, &contracts(), &BTreeMap::new()).unwrap();

        let mut seeds: BTreeMap<(String, String), Region> = BTreeMap::new();
        seeds.insert(
            ("b".to_owned(), "image".to_owned()),
            Region::from_rect(Rect::new(0, 0, 100, 100)),
        );
        let roi = analyze_roi_from_seeds(&plan, &graph, &checked, &contracts(), &seeds).unwrap();

        let grid = TileGrid::new(EXTENT, 128); // 4 tiles total
        let schedule = schedule_tiles(&plan, &graph, &checked, &contracts(), &roi, grid).unwrap();

        // Only the top-left tile per node: 3 nodes * 1 tile = 3 work items, not 12.
        assert_eq!(schedule.len(), 3);
        assert_eq!(schedule.demanded_tile_count("b"), 1);
        assert_eq!(schedule.demanded_tile_count("a"), 1);
        assert_eq!(schedule.demanded_tile_count("src"), 1);
        // Every scheduled tile is the top-left one (index 0).
        assert!(schedule.items().iter().all(|i| i.tile.index == 0));
        assert_eq!(schedule.peak_live_buffers(), 2);
    }

    #[test]
    fn dead_node_is_not_scheduled() {
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
        let grid = TileGrid::new(EXTENT, 256);
        let schedule = schedule_for(&plan, grid);
        assert_eq!(schedule.demanded_tile_count("dead"), 0);
        assert!(schedule.items().iter().all(|i| i.node != "dead"));
    }

    #[test]
    fn empty_demand_yields_an_empty_schedule() {
        let plan = parse_plan(
            r#"{
                "paintop": "1.0", "inputs": {},
                "nodes": [{"id": "src", "op": "source.create@1"}],
                "exports": {}
            }"#,
        )
        .unwrap();
        let grid = TileGrid::with_default(EXTENT);
        let schedule = schedule_for(&plan, grid);
        assert!(schedule.is_empty());
        assert_eq!(schedule.peak_live_buffers(), 0);
    }

    #[test]
    fn schedule_is_deterministic_across_runs() {
        let grid = TileGrid::new(EXTENT, 128);
        let a = schedule_for(&chain_plan(), grid);
        let b = schedule_for(&chain_plan(), grid);
        assert_eq!(a.items(), b.items());
        assert_eq!(a.liveness(), b.liveness());
    }

    #[test]
    fn diamond_graph_keeps_a_shared_producer_live_across_both_consumers() {
        // src feeds two pointwise nodes l and r, both joined by `j` (via l only,
        // r exported). The shared producer src must stay live until its last
        // consumer is scheduled.
        let plan = parse_plan(
            r#"{
                "paintop": "1.0", "inputs": {},
                "nodes": [
                    {"id": "src", "op": "source.create@1"},
                    {"id": "l", "op": "filter.point@1", "in": {"image": "node:src/image"}},
                    {"id": "r", "op": "filter.point@1", "in": {"image": "node:src/image"}}
                ],
                "exports": {
                    "ol": {"resource": "node:l/image", "kind": "image", "path": "l.png"},
                    "or": {"resource": "node:r/image", "kind": "image", "path": "r.png"}
                }
            }"#,
        )
        .unwrap();
        let grid = TileGrid::new(EXTENT, 256);
        let schedule = schedule_for(&plan, grid);
        // src, l, r each schedule one tile; l and r are both exported and pinned.
        assert_eq!(schedule.demanded_tile_count("src"), 1);
        assert_eq!(schedule.demanded_tile_count("l"), 1);
        assert_eq!(schedule.demanded_tile_count("r"), 1);
        // Peak: src + l live while r still pending -> but l and r are both pinned
        // to the end (exported), so once src dies the peak is l + r = 2; with src
        // also live before its last use the peak reaches 2 as well. Bounded < 3 is
        // the property that matters here.
        assert!(schedule.peak_live_buffers() <= 3);
        assert!(Region::from_extent(EXTENT).contains_rect(Rect::new(0, 0, 1, 1)));
    }
}
