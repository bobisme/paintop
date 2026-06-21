//! The pointwise tiled execution path (`plan.md` §11, exit gate: tiled ==
//! whole-image, bit-identical for exact ops).
//!
//! [`schedule_tiles`](super::schedule::schedule_tiles()) lays out *which*
//! `(node, output tile)` work items run and in what order; this module *runs*
//! them. For a **pointwise** op, an output pixel `(x, y)` is a pure function of
//! the co-located input pixel(s) and the op's params, independent of position and
//! of every other pixel. That is exactly the property that makes tiling
//! bit-exact: cropping each input to the output tile, running the op's ordinary
//! whole-image kernel on that tile-sized window, and scattering the result back
//! into the full output buffer produces — sample for sample — the same bytes the
//! whole-image executor produces, because
//!
//! ```text
//! crop(compute(input), tile) == compute(crop(input, tile))
//! ```
//!
//! holds for a position-independent pointwise op. A tile carries no neighbour
//! data, so there are no seam artifacts, and the row-major scatter writes each
//! output pixel exactly once.
//!
//! Nodes the scheduler marks as non-pointwise (sources with no input, or ops
//! whose ROI category is not [`Pointwise`](paintop_ir::RoiCategory::Pointwise))
//! fall back to a single whole-image dispatch, so the tiled executor still
//! produces every node's output correctly; only the pointwise interior is tiled.
//! The result is the same node-output value map the whole-image executor yields,
//! which the differential suite compares byte-for-byte.

use std::collections::BTreeMap;

use paintop_ir::{
    OperationRegistry, Plan, Rect, Reference, Region, ResolvedGraph, ResourceDescriptor,
    RoiCategory,
};

use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

use super::grid::TileGrid;
use super::parallel::ThreadCap;
use super::schedule::{TileSchedule, TileWorkItem, schedule_tiles};
use crate::executor::error::{ExecError, ExecResult};
use crate::executor::op_impl::{ImplRegistry, InputValues, OpImplementation, OutputValues};
use crate::executor::roi::RoiAnalysis;
use crate::executor::value::ResourceValue;

/// The per-node, per-tile counters a tiled run accumulates, for the trace and the
/// `tiles { requested, executed, identity }` metric (`plan.md` §15, the bundle
/// `tiles` block).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct TileStats {
    /// The number of output tiles the schedule requested.
    pub requested: usize,
    /// The number of output tiles actually computed (a dispatch ran).
    pub executed: usize,
    /// The number of output tiles served as identity / pass-through without a
    /// dispatch (an empty-mask tile, a whole-image fallback counted once).
    pub identity: usize,
}

/// The product of a tiled execution: every produced node-output value, the
/// resolved export values, and the tile statistics (`plan.md` §11).
#[derive(Debug)]
pub struct TiledExecution {
    node_outputs: BTreeMap<String, OutputValues>,
    exports: Vec<(String, ResourceValue)>,
    stats: TileStats,
}

impl TiledExecution {
    /// The value produced on node `node`'s output port `port`, if it ran.
    #[must_use]
    pub fn output(&self, node: &str, port: &str) -> Option<&ResourceValue> {
        self.node_outputs.get(node).and_then(|p| p.get(port))
    }

    /// The resolved export values, in export order.
    #[must_use]
    pub fn exports(&self) -> &[(String, ResourceValue)] {
        &self.exports
    }

    /// The tile statistics accumulated over the run.
    #[must_use]
    pub const fn stats(&self) -> TileStats {
        self.stats
    }
}

/// Execute the demanded subgraph of `graph` tile-by-tile, tiling pointwise ops
/// and dispatching the rest whole-image (`plan.md` §11).
///
/// `roi` supplies the per-node demanded regions; a [`TileGrid`] of `tile_size`
/// partitions them. The returned [`TiledExecution`] carries the same node-output
/// values a whole-image run produces — bit-identical for exact pointwise ops —
/// plus the tile statistics.
///
/// # Errors
/// - [`ExecError::ImplementationNotFound`] if a demanded node has no manifest or
///   implementation.
/// - [`ExecError::InputNotAvailable`] if a node's wired input has no value.
/// - [`ExecError::Dispatch`] if an op kernel raises.
/// - [`ExecError::OutputNotProduced`] if a kernel omits a declared output.
#[allow(
    clippy::too_many_arguments,
    reason = "the tiled executor threads the manifest/contract/impl registries, ROI, inputs, and tile size"
)]
pub fn execute_tiled(
    plan: &Plan,
    graph: &ResolvedGraph,
    checked: &paintop_ir::CheckedGraph,
    manifests: &OperationRegistry,
    contracts: &paintop_ir::ContractRegistry,
    implementations: &ImplRegistry,
    roi: &RoiAnalysis,
    inputs: &BTreeMap<String, ResourceValue>,
    tile_size: u32,
) -> ExecResult<TiledExecution> {
    execute_tiled_with(
        plan,
        graph,
        checked,
        manifests,
        contracts,
        implementations,
        roi,
        inputs,
        tile_size,
        ThreadCap::fixed(1),
    )
}

/// Execute the demanded subgraph of `graph` tile-by-tile under an explicit
/// [`ThreadCap`], fanning a node's independent output tiles across the
/// scheduler-owned Rayon pool (`plan.md` §12.2).
///
/// Identical in *result* to [`execute_tiled`] for every `cap`: the cap only governs
/// *how many threads* evaluate the per-tile compute, never *what* each tile
/// computes. A node's tiles are independent (disjoint output regions over read-only
/// inputs) and every scatter is replayed on the calling thread in fixed schedule
/// order, so the produced buffers are **bit-identical across thread counts** and
/// equal to the single-threaded [`execute_tiled`] result. `cap` of
/// [`ThreadCap::fixed(1)`](ThreadCap::fixed) is exactly the sequential M2 path.
///
/// # Errors
/// The same failures as [`execute_tiled`].
#[allow(
    clippy::too_many_arguments,
    reason = "the tiled executor threads the manifest/contract/impl registries, ROI, inputs, tile size, and the thread cap"
)]
pub fn execute_tiled_with(
    plan: &Plan,
    graph: &ResolvedGraph,
    checked: &paintop_ir::CheckedGraph,
    manifests: &OperationRegistry,
    contracts: &paintop_ir::ContractRegistry,
    implementations: &ImplRegistry,
    roi: &RoiAnalysis,
    inputs: &BTreeMap<String, ResourceValue>,
    tile_size: u32,
    cap: ThreadCap,
) -> ExecResult<TiledExecution> {
    let extent = graph_extent(graph, checked, inputs);
    let grid = TileGrid::new(extent, tile_size);
    let schedule = schedule_tiles(plan, graph, checked, contracts, roi, grid)?;

    let params_by_node: BTreeMap<&str, serde_json::Value> = plan
        .nodes
        .iter()
        .map(|n| (n.id.as_str(), serde_json::Value::Object(n.params.clone())))
        .collect();

    let mut node_outputs: BTreeMap<String, OutputValues> = BTreeMap::new();
    let mut stats = TileStats::default();

    // The schedule is in topological-then-tile order; group work items by node and
    // run each node once (pointwise: per tile; otherwise: whole image).
    for node_id in demanded_order(graph, roi) {
        let Some(node) = graph.node(&node_id) else {
            continue;
        };
        let params = params_by_node
            .get(node_id.as_str())
            .cloned()
            .unwrap_or(serde_json::Value::Null);
        let implementation =
            implementations
                .get(&node.op)
                .ok_or_else(|| ExecError::ImplementationNotFound {
                    node: node_id.clone(),
                    op: node.op.to_string(),
                })?;
        let category = roi_category(manifests, node);

        let tile_count = schedule.demanded_tile_count(&node_id);
        if tileable(category, node, contracts, checked, &params) && tile_count > 0 {
            let outputs = run_pointwise_tiled(
                &node_id,
                node,
                implementation,
                &params,
                &schedule,
                checked,
                inputs,
                &node_outputs,
                &mut stats,
                cap,
            )?;
            node_outputs.insert(node_id.clone(), outputs);
        } else if neighborhood_tileable(category, node) && tile_count > 0 {
            let outputs = run_neighborhood_tiled(
                &node_id,
                node,
                implementation,
                &params,
                &schedule,
                checked,
                inputs,
                &node_outputs,
                &mut stats,
                cap,
            )?;
            node_outputs.insert(node_id.clone(), outputs);
        } else {
            // Whole-image fallback (sources, non-pointwise ops): one dispatch.
            let input_values = assemble_inputs(node, inputs, &node_outputs)?;
            let produced = implementation
                .compute(&input_values, &params)
                .map_err(|source| ExecError::Dispatch {
                    node: node_id.clone(),
                    op: node.op.to_string(),
                    source: Box::new(source),
                })?;
            let outputs = collect_outputs(&node_id, node, manifests, &produced)?;
            node_outputs.insert(node_id.clone(), outputs);
        }
    }

    let exports = resolve_exports(graph, inputs, &node_outputs)?;
    Ok(TiledExecution {
        node_outputs,
        exports,
        stats,
    })
}

/// The owned product of computing one output tile of a node: the absolute output
/// tile rect and the kernel's per-port output values for that tile.
///
/// The window origin is carried so a neighbourhood tile can map its window-local
/// output back into the absolute output frame during the sequential scatter; a
/// pointwise tile leaves it at the tile origin (the identity offset).
struct ProducedTile {
    /// The absolute output tile rect this product fills.
    tile: Rect,
    /// The halo-window origin in the output frame (the top-left of the input crop),
    /// or the tile origin for a pointwise tile.
    origin: (i64, i64),
    /// The kernel's produced outputs for this tile (window output for a
    /// neighbourhood op, co-located output for a pointwise op).
    produced: OutputValues,
}

/// Compute every demanded output tile of `node_id` — optionally in parallel over a
/// scheduler-owned, capped Rayon pool — returning the produced tiles **in schedule
/// order**.
///
/// `prepare` builds the per-tile input crops for one work item (the co-located tile
/// for a pointwise op, the haloed window for a neighbourhood op) and yields the
/// kernel inputs plus the window origin. The kernel is a pure, [`Send`] + [`Sync`]
/// function of those inputs, so evaluating the tiles concurrently is sound and the
/// per-tile outputs are independent of the worker that produced them. Results are
/// collected back into the work items' fixed schedule order, so the subsequent
/// scatter is deterministic regardless of `cap`.
fn compute_node_tiles<F>(
    node_id: &str,
    node: &paintop_ir::ResolvedNode,
    implementation: &dyn OpImplementation,
    params: &serde_json::Value,
    schedule: &TileSchedule,
    cap: ThreadCap,
    prepare: F,
) -> ExecResult<Vec<ProducedTile>>
where
    F: Fn(&TileWorkItem) -> ExecResult<(InputValues, (i64, i64))> + Sync,
{
    let items: Vec<&TileWorkItem> = schedule
        .items()
        .iter()
        .filter(|i| i.node == node_id)
        .collect();

    // The per-tile compute: prepare the input crops, dispatch the kernel, keep the
    // owned product. A pure function of read-only shared state, so it is safe to run
    // concurrently; the result order is restored by `par_iter`'s index-preserving
    // `collect`.
    let compute = |item: &&TileWorkItem| -> ExecResult<ProducedTile> {
        let (tile_inputs, origin) = prepare(item)?;
        let produced = implementation
            .compute(&tile_inputs, params)
            .map_err(|source| ExecError::Dispatch {
                node: node_id.to_owned(),
                op: node.op.to_string(),
                source: Box::new(source),
            })?;
        Ok(ProducedTile {
            tile: item.tile.rect,
            origin,
            produced,
        })
    };

    if cap.is_sequential() {
        // The M2 single-threaded path: no pool, evaluate in schedule order.
        items.iter().map(compute).collect()
    } else {
        // Fan the independent tiles across a pool capped at the policy width. The
        // scheduler owns this pool — ops never spawn their own — so a deep graph
        // cannot oversubscribe the machine with nested parallelism. `par_iter`
        // preserves index order in `collect`, so the produced tiles come back in
        // schedule order regardless of which worker finished first.
        run_on_capped_pool(cap, || items.par_iter().map(compute).collect())
    }
}

/// Run `work` on a Rayon pool capped at `cap` worker threads (`plan.md` §12.2).
///
/// A scoped, locally-built [`rayon::ThreadPool`] bounds the parallel width to the
/// policy cap; if the pool cannot be built the work runs on the calling thread (a
/// safe sequential fallback, never a failure). The pool is dropped when this
/// returns, so it never leaks workers across nodes.
fn run_on_capped_pool<R, W>(cap: ThreadCap, work: W) -> R
where
    W: FnOnce() -> R + Send,
    R: Send,
{
    match rayon::ThreadPoolBuilder::new()
        .num_threads(cap.resolve())
        .build()
    {
        Ok(pool) => pool.install(work),
        // A pool-build failure (e.g. a resource limit) degrades to sequential on
        // the caller's thread; the result is identical, just not parallel.
        Err(_) => work(),
    }
}

/// Run a pointwise node tile-by-tile: for each demanded output tile, crop the
/// inputs to the tile, dispatch the kernel on the tile-sized window (optionally in
/// parallel across the scheduler pool), and scatter the result into the node's
/// full-extent output buffers.
#[allow(
    clippy::too_many_arguments,
    reason = "the tiled dispatch threads the borrowed registries and the scatter target"
)]
fn run_pointwise_tiled(
    node_id: &str,
    node: &paintop_ir::ResolvedNode,
    implementation: &dyn crate::executor::op_impl::OpImplementation,
    params: &serde_json::Value,
    schedule: &TileSchedule,
    checked: &paintop_ir::CheckedGraph,
    inputs: &BTreeMap<String, ResourceValue>,
    node_outputs: &BTreeMap<String, OutputValues>,
    stats: &mut TileStats,
    cap: ThreadCap,
) -> ExecResult<OutputValues> {
    // Allocate the node's full-extent output buffers (zero-filled), to scatter
    // each computed tile into. The output descriptors come from the checked graph.
    let mut outputs: OutputValues = OutputValues::new();
    if let Some(ports) = checked.node_outputs(node_id) {
        for (port, descriptor) in ports {
            outputs.insert(port.clone(), zero_value(*descriptor));
        }
    }

    // Compute every tile (parallel per the cap), then scatter in schedule order.
    let tiles = compute_node_tiles(
        node_id,
        node,
        implementation,
        params,
        schedule,
        cap,
        |item| {
            // Crop every input to the tile region (pointwise: the co-located tile).
            let mut tile_inputs: InputValues = InputValues::new();
            for input in &item.inputs {
                let value =
                    resolve_value(&input.source, inputs, node_outputs).ok_or_else(|| {
                        ExecError::InputNotAvailable {
                            node: node_id.to_owned(),
                            port: input.port.clone(),
                            detail: format!("input `{}` had no value", input.port),
                        }
                    })?;
                tile_inputs.insert(input.port.clone(), crop(value, item.tile.rect));
            }
            Ok((tile_inputs, (item.tile.rect.x0, item.tile.rect.y0)))
        },
    )?;

    for produced_tile in &tiles {
        stats.requested += 1;
        stats.executed += 1;
        let tile = produced_tile.tile;
        // Scatter each produced tile into the matching full-extent output buffer.
        for (port, target) in &mut outputs {
            let Some(tile_value) = produced_tile.produced.get(port) else {
                return Err(ExecError::OutputNotProduced {
                    node: node_id.to_owned(),
                    op: node.op.to_string(),
                    port: port.clone(),
                });
            };
            scatter(target, tile_value, tile);
        }
    }
    Ok(outputs)
}

/// Whether a node is safe to tile **pointwise**: its declared ROI category is
/// [`Pointwise`](RoiCategory::Pointwise), it has at least one wired input (a source
/// has none and must run whole-image), **and** it is *position-independent* — it
/// reads its input at the co-located output region (`output(x, y) = f(input(x, y))`).
///
/// The last condition is the bn-3ai guard. A `Pointwise` ROI category alone is not
/// sufficient to tile: `paint.linear_gradient@1` / `paint.radial_gradient@1` declare
/// `Pointwise` but are **position-dependent generators** — they read *no* input
/// samples (only the `extent_from` size) and compute each output pixel from its
/// *absolute* coordinate. Tiling such an op would crop the input to the tile and run
/// the kernel with a shifted tile origin, silently producing a wrong (per-tile
/// restarted) gradient. We detect this by asking the op's contract what input region
/// a non-empty output region demands: a true pointwise transform demands a non-empty,
/// co-located region; a generator demands the empty region from every input. Only the
/// former is tiled; a generator falls back to a single whole-image dispatch (still
/// correct, just not tiled).
fn tileable(
    category: Option<RoiCategory>,
    node: &paintop_ir::ResolvedNode,
    contracts: &paintop_ir::ContractRegistry,
    checked: &paintop_ir::CheckedGraph,
    params: &serde_json::Value,
) -> bool {
    matches!(category, Some(RoiCategory::Pointwise))
        && !node.inputs.is_empty()
        && reads_colocated_input(node, contracts, checked, params)
}

/// Whether a pointwise node genuinely **reads its input samples** at the co-located
/// region — the position-independence test that distinguishes a true pointwise
/// transform (`output(R) = f(input(R))`, tileable) from a position-dependent
/// generator (a gradient that reads only the input *extent*, not tileable).
///
/// We probe the op's [`OpContract::required_inputs`](paintop_ir::OpContract::required_inputs)
/// with a small, non-trivial output region offset away from the origin. A true
/// pointwise op demands a **non-empty** input region from at least one wired input;
/// a generator demands the empty region from every input (it consumes no samples).
/// If the contract is unavailable or errors, we conservatively answer `false` so the
/// node runs whole-image — never tiled on an unverified assumption.
fn reads_colocated_input(
    node: &paintop_ir::ResolvedNode,
    contracts: &paintop_ir::ContractRegistry,
    checked: &paintop_ir::CheckedGraph,
    params: &serde_json::Value,
) -> bool {
    let Some(contract) = contracts.get(&node.op) else {
        return false;
    };
    // The node's input descriptors (so the contract can clamp to real extents) and a
    // representative output port to probe.
    let input_descriptors = node_input_descriptors(node, checked);
    let Some(ports) = checked.node_outputs(&node.id) else {
        return false;
    };
    let Some((output_port, _)) = ports.iter().next() else {
        return false;
    };
    // A small probe rect offset from the origin: a position-dependent generator would
    // still demand the empty region here, while a true pointwise op demands this rect
    // (or a superset) back from its input.
    let probe = Rect::new(1, 1, 2, 2);
    let mut requested = paintop_ir::OutputRegions::new();
    requested.insert(output_port.clone(), probe);
    let Ok(needed) = contract.required_inputs(&requested, &input_descriptors, params) else {
        return false;
    };
    // Tileable iff some wired input is read over a non-empty region.
    node.inputs.keys().any(|port| {
        needed
            .get(port)
            .is_some_and(|region| !Region::from_rect(*region).is_empty())
    })
}

/// The node's input-port descriptors, resolved from the checked graph, so a contract
/// probe can clamp halos to real input extents (mirrors the scheduler's helper).
fn node_input_descriptors(
    node: &paintop_ir::ResolvedNode,
    checked: &paintop_ir::CheckedGraph,
) -> paintop_ir::Descriptors {
    let mut descriptors = paintop_ir::Descriptors::new();
    for (port, reference) in &node.inputs {
        match reference {
            Reference::Node { node, port: up } => {
                if let Some(descriptor) = checked.output(node, up) {
                    descriptors.insert(port.clone(), *descriptor);
                }
            }
            Reference::Input { input } => {
                if let Some(descriptor) = checked.input(input) {
                    descriptors.insert(port.clone(), *descriptor);
                }
            }
        }
    }
    descriptors
}

/// Whether a node is safe to tile as a single-input **neighbourhood** op: its
/// declared ROI category is [`LocalHalo`](RoiCategory::LocalHalo) or
/// [`Geometric`](RoiCategory::Geometric) (a blur, a convolution, a warp — the
/// output reads a haloed/transformed input footprint), and it has exactly one
/// input port.
///
/// The single-input restriction keeps the output-frame origin unambiguous: the
/// op runs on the halo window cropped from that one spatial input, and the window-
/// output local frame is offset by the window's origin. A multi-input
/// neighbourhood op has no single window origin to align against, so it falls back
/// to a whole-image dispatch (still correct, just not tiled).
fn neighborhood_tileable(category: Option<RoiCategory>, node: &paintop_ir::ResolvedNode) -> bool {
    matches!(
        category,
        Some(RoiCategory::LocalHalo | RoiCategory::Geometric)
    ) && node.inputs.len() == 1
}

/// Run a single-input neighbourhood node tile-by-tile with correct per-tile input
/// halos, matching the whole-image reference (`plan.md` §11.3,
/// `AGENT_VERIFICATION` §3.3).
///
/// For each demanded output tile the schedule carries the haloed input region the
/// op's backward ROI contract demands (the kernel-dilated footprint, clamped to
/// the input extent). The executor
///
/// 1. crops the input to that **halo window** (not the co-located tile),
/// 2. runs the op's ordinary whole-image kernel on the window, and
/// 3. extracts the sub-rect of the window output that corresponds to the output
///    tile and scatters it into the node's full-extent output buffer.
///
/// This is the standard "compute-on-halo, keep the interior" construction. For an
/// interior tile the halo window lies strictly inside the input, so the op's
/// boundary handling at the *window* edge never touches the tile's pixels — they
/// are computed only from real, in-bounds samples, exactly as the whole-image run
/// computes them. For an edge tile the halo window is clamped to the input extent,
/// so the window edge *is* the image edge and the op applies the real boundary
/// mode there. Either way the tile's output samples equal the whole-image run's,
/// so there is no visible tile grid (`AGENT_VERIFICATION` §13 "tile grid visible").
///
/// The window-output local frame is offset by the window origin `(W.x0, W.y0)`;
/// an output pixel at absolute coord `(x, y)` lives at window-local
/// `(x - W.x0, y - W.y0)` under both the extent-preserving edge modes and the
/// extent-shrinking `valid` mode (the op's output indexing is offset by the same
/// window origin in both cases), so a single mapping serves every boundary mode.
#[allow(
    clippy::too_many_arguments,
    reason = "the tiled dispatch threads the borrowed registries and the scatter target"
)]
fn run_neighborhood_tiled(
    node_id: &str,
    node: &paintop_ir::ResolvedNode,
    implementation: &dyn crate::executor::op_impl::OpImplementation,
    params: &serde_json::Value,
    schedule: &TileSchedule,
    checked: &paintop_ir::CheckedGraph,
    inputs: &BTreeMap<String, ResourceValue>,
    node_outputs: &BTreeMap<String, OutputValues>,
    stats: &mut TileStats,
    cap: ThreadCap,
) -> ExecResult<OutputValues> {
    let mut outputs: OutputValues = OutputValues::new();
    if let Some(ports) = checked.node_outputs(node_id) {
        for (port, descriptor) in ports {
            outputs.insert(port.clone(), zero_value(*descriptor));
        }
    }

    // Compute every window output (parallel per the cap), then extract + scatter the
    // interior in schedule order.
    let tiles = compute_node_tiles(
        node_id,
        node,
        implementation,
        params,
        schedule,
        cap,
        |item| {
            let tile = item.tile.rect;
            // Crop the single spatial input to its halo window (the conservative
            // kernel-dilated footprint the schedule computed), not the co-located
            // tile.
            let mut tile_inputs: InputValues = InputValues::new();
            // The window origin in the output coordinate frame: the input halo's
            // top-left. A single-input neighbourhood op has exactly one such window.
            let mut window_origin: Option<(i64, i64)> = None;
            for input in &item.inputs {
                let value =
                    resolve_value(&input.source, inputs, node_outputs).ok_or_else(|| {
                        ExecError::InputNotAvailable {
                            node: node_id.to_owned(),
                            port: input.port.clone(),
                            detail: format!("input `{}` had no value", input.port),
                        }
                    })?;
                let window = input.region.bounding_rect();
                window_origin.get_or_insert((window.x0, window.y0));
                tile_inputs.insert(input.port.clone(), crop(value, window));
            }
            // The window origin (top-left of the halo crop) anchors the window output
            // back into absolute output coordinates. With no input region the op reads
            // nothing for this tile; treat the tile origin as the anchor.
            let origin = window_origin.unwrap_or((tile.x0, tile.y0));
            Ok((tile_inputs, origin))
        },
    )?;

    for produced_tile in &tiles {
        stats.requested += 1;
        stats.executed += 1;
        let tile = produced_tile.tile;
        let (origin_left, origin_top) = produced_tile.origin;
        // Extract the output tile from the window output and scatter it into the
        // full-extent buffer. The window output's local coordinate `(lx, ly)` maps
        // to absolute output `(origin_left + lx, origin_top + ly)`, so the output
        // tile lives at window-local rect `tile - (origin_left, origin_top)`.
        for (port, target) in &mut outputs {
            let Some(window_value) = produced_tile.produced.get(port) else {
                return Err(ExecError::OutputNotProduced {
                    node: node_id.to_owned(),
                    op: node.op.to_string(),
                    port: port.clone(),
                });
            };
            let local = tile.translate(-origin_left, -origin_top);
            let tile_value = crop(window_value, local);
            scatter(target, &tile_value, tile);
        }
    }
    Ok(outputs)
}

/// The demanded node ids in topological order.
fn demanded_order(graph: &ResolvedGraph, roi: &RoiAnalysis) -> Vec<String> {
    graph
        .topological_order()
        .iter()
        .filter(|id| roi.is_demanded(id))
        .cloned()
        .collect()
}

/// The ROI category declared by `node`'s manifest, if the op resolves.
fn roi_category(
    manifests: &OperationRegistry,
    node: &paintop_ir::ResolvedNode,
) -> Option<RoiCategory> {
    manifests.get(&node.op).ok().map(|m| m.roi.category)
}

/// A zero-filled [`ResourceValue`] for a raster descriptor, sized to its extent
/// and channel count.
///
/// The buffer length is derived from the descriptor, so [`ResourceValue::new`]
/// always succeeds; the `unwrap_or_else` fallback (a zero-extent buffer, itself
/// always valid) only exists to keep the function total without a panic.
fn zero_value(descriptor: ResourceDescriptor) -> ResourceValue {
    let extent = descriptor.extent();
    let channels = channel_count(&descriptor);
    let len = (extent.width as usize)
        .saturating_mul(extent.height as usize)
        .saturating_mul(channels as usize);
    ResourceValue::new(descriptor, channels, vec![0.0; len]).unwrap_or_else(|_| {
        let empty = descriptor.with_extent(paintop_ir::Extent::new(0, 0));
        ResourceValue::new(empty, channels, Vec::new())
            .unwrap_or_else(|_| ResourceValue::report(report_placeholder(&empty)))
    })
}

/// A degenerate empty report, the last-resort total fallback for [`zero_value`]
/// (never reached for a correctly-sized raster descriptor).
const fn report_placeholder(descriptor: &ResourceDescriptor) -> paintop_ir::Report {
    paintop_ir::Report {
        extent: descriptor.extent(),
        channels: 0,
        channel_stats: Vec::new(),
        all_finite: true,
        content_hash: String::new(),
        diff: None,
        assertion: None,
        histogram: None,
        components: None,
    }
}

/// The interleaved sample-per-pixel count a raster descriptor carries.
///
/// The catch-all keeps the function total against a future non-exhaustive
/// [`ResourceDescriptor`] variant: an unknown raster kind defaults to a single
/// channel, a kind-agnostic conservative choice.
const fn channel_count(descriptor: &ResourceDescriptor) -> u32 {
    match descriptor {
        ResourceDescriptor::Image(d) => d.layout.channel_count(),
        ResourceDescriptor::Field2(_) => 2,
        ResourceDescriptor::Field3(_) => 3,
        ResourceDescriptor::Report(_) => 0,
        // Mask, SdfMask, Field1, and any future raster kind default to one channel.
        _ => 1,
    }
}

/// Crop `value` to the half-open `tile` rect, returning a tile-extent
/// [`ResourceValue`] whose descriptor is `value`'s at the tile's extent.
///
/// The tile is clamped to the source extent, and pixels are copied row-major. The
/// crop preserves absolute pixel values (no resampling), so for a pointwise op the
/// cropped output equals the corresponding window of a whole-image output.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "tile coords are clamped to [0, extent) and tile dims fit u32 (sub-extent)"
)]
fn crop(value: &ResourceValue, tile: Rect) -> ResourceValue {
    let extent = value.extent();
    let clipped = tile.clamp_to_extent(extent);
    let channels = value.channels();
    let src = value.samples();
    let src_w = extent.width as usize;
    let tile_w = clipped.width() as usize;
    let tile_h = clipped.height() as usize;
    let ch = channels as usize;

    let mut out = vec![0.0_f32; tile_w * tile_h * ch];
    for row in 0..tile_h {
        let src_y = clipped.y0 as usize + row;
        let src_base = (src_y * src_w + clipped.x0 as usize) * ch;
        let dst_base = row * tile_w * ch;
        let span = tile_w * ch;
        if let (Some(s), Some(d)) = (
            src.get(src_base..src_base + span),
            out.get_mut(dst_base..dst_base + span),
        ) {
            d.copy_from_slice(s);
        }
    }

    let tile_extent = paintop_ir::Extent::new(tile_w as u32, tile_h as u32);
    let descriptor = value.descriptor().with_extent(tile_extent);
    ResourceValue::new(descriptor, channels, out).unwrap_or_else(|_| zero_value(descriptor))
}

/// Scatter `tile_value` into `target` at the half-open `tile` rect, copying
/// row-major. The tile and target share channel count; pixels outside the tile
/// are left untouched.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "tile coords are clamped to [0, extent) and channel count fits u32"
)]
fn scatter(target: &mut ResourceValue, tile_value: &ResourceValue, tile: Rect) {
    let extent = target.extent();
    let clipped = tile.clamp_to_extent(extent);
    let channels = target.channels() as usize;
    let dst_w = extent.width as usize;
    let tile_w = clipped.width() as usize;
    let tile_h = clipped.height() as usize;
    let src = tile_value.samples().to_vec();

    let descriptor = *target.descriptor();
    let mut dst = target.samples().to_vec();
    for row in 0..tile_h {
        let dst_y = clipped.y0 as usize + row;
        let dst_base = (dst_y * dst_w + clipped.x0 as usize) * channels;
        let src_base = row * tile_w * channels;
        let span = tile_w * channels;
        if let (Some(s), Some(d)) = (
            src.get(src_base..src_base + span),
            dst.get_mut(dst_base..dst_base + span),
        ) {
            d.copy_from_slice(s);
        }
    }
    *target = ResourceValue::new(descriptor, channels as u32, dst)
        .unwrap_or_else(|_| zero_value(descriptor));
}

/// Assemble a node's whole-image input values from external inputs and produced
/// node outputs.
fn assemble_inputs(
    node: &paintop_ir::ResolvedNode,
    inputs: &BTreeMap<String, ResourceValue>,
    node_outputs: &BTreeMap<String, OutputValues>,
) -> ExecResult<InputValues> {
    let mut values: InputValues = InputValues::new();
    for (port, reference) in &node.inputs {
        let value = resolve_value(reference, inputs, node_outputs).ok_or_else(|| {
            ExecError::InputNotAvailable {
                node: node.id.clone(),
                port: port.clone(),
                detail: format!("input `{port}` had no value"),
            }
        })?;
        values.insert(port.clone(), value.clone());
    }
    Ok(values)
}

/// Verify a whole-image dispatch produced every declared output port.
fn collect_outputs(
    node_id: &str,
    node: &paintop_ir::ResolvedNode,
    manifests: &OperationRegistry,
    produced: &OutputValues,
) -> ExecResult<OutputValues> {
    let Ok(manifest) = manifests.get(&node.op) else {
        return Err(ExecError::ImplementationNotFound {
            node: node_id.to_owned(),
            op: node.op.to_string(),
        });
    };
    let mut outputs: OutputValues = OutputValues::new();
    for spec in &manifest.outputs {
        let value = produced
            .get(&spec.name)
            .ok_or_else(|| ExecError::OutputNotProduced {
                node: node_id.to_owned(),
                op: node.op.to_string(),
                port: spec.name.clone(),
            })?;
        outputs.insert(spec.name.clone(), value.clone());
    }
    Ok(outputs)
}

/// Resolve every export's value from the produced outputs / external inputs.
fn resolve_exports(
    graph: &ResolvedGraph,
    inputs: &BTreeMap<String, ResourceValue>,
    node_outputs: &BTreeMap<String, OutputValues>,
) -> ExecResult<Vec<(String, ResourceValue)>> {
    let mut exports = Vec::with_capacity(graph.exports().len());
    for export in graph.exports() {
        let value = resolve_value(&export.resource, inputs, node_outputs).ok_or_else(|| {
            let node = match &export.resource {
                Reference::Node { node, .. } => node.clone(),
                Reference::Input { input } => input.clone(),
            };
            ExecError::InputNotAvailable {
                node,
                port: export.id.clone(),
                detail: format!("export `{}` had no value", export.id),
            }
        })?;
        exports.push((export.id.clone(), value.clone()));
    }
    Ok(exports)
}

/// Resolve a [`Reference`] to the concrete value it carries.
fn resolve_value<'a>(
    reference: &Reference,
    inputs: &'a BTreeMap<String, ResourceValue>,
    node_outputs: &'a BTreeMap<String, OutputValues>,
) -> Option<&'a ResourceValue> {
    match reference {
        Reference::Input { input } => inputs.get(input),
        Reference::Node { node, port } => node_outputs.get(node).and_then(|p| p.get(port)),
    }
}

/// The working extent of the graph: the extent of the largest demanded node
/// output, falling back to the first external input. Used to size the tile grid.
fn graph_extent(
    graph: &ResolvedGraph,
    checked: &paintop_ir::CheckedGraph,
    inputs: &BTreeMap<String, ResourceValue>,
) -> paintop_ir::Extent {
    let mut best = paintop_ir::Extent::new(0, 0);
    for (_, descriptor) in checked.exports() {
        let e = descriptor.extent();
        if u64::from(e.width) * u64::from(e.height) > u64::from(best.width) * u64::from(best.height)
        {
            best = e;
        }
    }
    if (best.width == 0 || best.height == 0)
        && let Some(value) = inputs.values().next()
    {
        best = value.extent();
    }
    let _ = graph;
    best
}

/// The conservative region of an export's resource, for callers that drive the
/// grid from a sub-region demand (`plan.md` §11.3). Currently the full extent.
#[must_use]
pub const fn export_region(descriptor: &ResourceDescriptor) -> Region {
    Region::from_extent(descriptor.extent())
}

#[cfg(test)]
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    clippy::float_cmp,
    clippy::unwrap_used,
    reason = "test fixtures build exact small ramp buffers from integer indices"
)]
mod tests {
    use super::{crop, scatter};
    use crate::executor::roi::analyze_roi;
    use crate::executor::value::ResourceValue;
    use crate::executor::{InputValues, OpImplementation, OutputValues, execute};
    use paintop_ir::{
        AlphaRepresentation, ChannelLayout, ColorEncoding, ColorRange, ContractRegistry,
        CoordinateConvention, Descriptors, DeterminismTier, Error, Extent, ImageDescriptor,
        InputRegions, InputSpec, OpContract, OperationManifest, OperationRegistry,
        OutputDescriptors, OutputRegions, OutputSpec, Plan, Rect, ResourceDescriptor, ResourceKind,
        RoiCategory, RoiPolicy, ScalarType, SemanticRole, TestMetadata, check_graph, parse_plan,
        resolve_plan,
    };
    use std::collections::BTreeMap;

    const EXTENT: Extent = Extent::new(64, 48);

    fn image_descriptor(extent: Extent) -> ResourceDescriptor {
        ResourceDescriptor::Image(ImageDescriptor {
            extent,
            layout: ChannelLayout::Rgba,
            scalar: ScalarType::F32,
            color: ColorEncoding::LinearSrgb,
            range: ColorRange::SceneReferred,
            alpha: AlphaRepresentation::Premultiplied,
            coordinates: CoordinateConvention::PixelCenterUpperLeft,
            semantic: SemanticRole::Color,
        })
    }

    // A source op: produces a deterministic ramp keyed by absolute pixel index.
    struct Ramp;
    impl OpContract for Ramp {
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
            o.insert("image".to_owned(), image_descriptor(EXTENT));
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
        ) -> paintop_ir::Result<Vec<paintop_ir::AssertionResult>> {
            Ok(vec![])
        }
    }
    impl OpImplementation for Ramp {
        fn compute(
            &self,
            _inputs: &InputValues,
            _params: &serde_json::Value,
        ) -> Result<OutputValues, Error> {
            let len = (EXTENT.width * EXTENT.height * 4) as usize;
            let samples: Vec<f32> = (0..len).map(|i| i as f32).collect();
            let mut out = OutputValues::new();
            out.insert(
                "image".to_owned(),
                ResourceValue::new(image_descriptor(EXTENT), 4, samples).unwrap(),
            );
            Ok(out)
        }
    }

    // A pointwise op: scale every sample by 2 and add 1 (position-independent).
    struct PointwiseScale;
    impl OpContract for PointwiseScale {
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
        ) -> paintop_ir::Result<Vec<paintop_ir::AssertionResult>> {
            Ok(vec![])
        }
    }
    impl OpImplementation for PointwiseScale {
        fn compute(
            &self,
            inputs: &InputValues,
            _params: &serde_json::Value,
        ) -> Result<OutputValues, Error> {
            let value = &inputs["image"];
            let samples: Vec<f32> = value
                .samples()
                .iter()
                .map(|s| s.mul_add(2.0, 1.0))
                .collect();
            let mut out = OutputValues::new();
            out.insert(
                "image".to_owned(),
                ResourceValue::new(*value.descriptor(), value.channels(), samples).unwrap(),
            );
            Ok(out)
        }
    }

    // A single-input neighbourhood op: a clamped 3x3 box-sum (halo of 1 on every
    // side). It is a genuine neighbourhood op — each output reads its 8 neighbours
    // — so tiling it must crop the haloed window, compute, and keep the interior.
    struct HaloBox;
    impl OpContract for HaloBox {
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
                let grown = paintop_ir::Region::from_rect(*region)
                    .dilate(1)
                    .clamp_to_extent(extent);
                r.insert("image".to_owned(), grown.bounding_rect());
            }
            Ok(r)
        }
        fn validate_postconditions(
            &self,
            _o: &OutputDescriptors,
            _p: &serde_json::Value,
        ) -> paintop_ir::Result<Vec<paintop_ir::AssertionResult>> {
            Ok(vec![])
        }
    }
    impl OpImplementation for HaloBox {
        fn compute(
            &self,
            inputs: &InputValues,
            _params: &serde_json::Value,
        ) -> Result<OutputValues, Error> {
            let value = &inputs["image"];
            let extent = value.extent();
            let w = i64::from(extent.width);
            let h = i64::from(extent.height);
            let ch = value.channels() as usize;
            let src = value.samples();
            let mut out = vec![0.0_f32; src.len()];
            for y in 0..h {
                for x in 0..w {
                    for c in 0..ch {
                        let mut acc = 0.0_f32;
                        // Clamped 3x3 box sum (the edge-replication boundary).
                        for dy in -1..=1_i64 {
                            for dx in -1..=1_i64 {
                                let sx = (x + dx).clamp(0, w - 1);
                                let sy = (y + dy).clamp(0, h - 1);
                                let idx = ((sy * w + sx) as usize) * ch + c;
                                acc += src[idx];
                            }
                        }
                        let didx = ((y * w + x) as usize) * ch + c;
                        out[didx] = acc;
                    }
                }
            }
            let mut o = OutputValues::new();
            o.insert(
                "image".to_owned(),
                ResourceValue::new(*value.descriptor(), value.channels(), out).unwrap(),
            );
            Ok(o)
        }
    }

    // A position-dependent **generator** modelling paint.linear_gradient@1: it
    // declares `Pointwise` ROI and is wired to an `extent_from` input, but reads
    // *no* input samples (its `required_inputs` returns the empty region) and
    // computes each output pixel from its **absolute** coordinate. The bn-3ai guard
    // must keep this op off the tiled path; tiling it would restart the ramp at each
    // tile origin and diverge.
    struct Gradientish;
    impl OpContract for Gradientish {
        fn declared_inputs(&self) -> Vec<(String, ResourceKind)> {
            vec![("extent_from".to_owned(), ResourceKind::Image)]
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
            o.insert("image".to_owned(), i["extent_from"]);
            Ok(o)
        }
        fn required_inputs(
            &self,
            _o: &OutputRegions,
            _i: &Descriptors,
            _p: &serde_json::Value,
        ) -> paintop_ir::Result<InputRegions> {
            // Reads only the input *extent*, no samples: an empty demanded region.
            let mut r = InputRegions::new();
            r.insert("extent_from".to_owned(), Rect::new(0, 0, 0, 0));
            Ok(r)
        }
        fn validate_postconditions(
            &self,
            _o: &OutputDescriptors,
            _p: &serde_json::Value,
        ) -> paintop_ir::Result<Vec<paintop_ir::AssertionResult>> {
            Ok(vec![])
        }
    }
    impl OpImplementation for Gradientish {
        fn compute(
            &self,
            inputs: &InputValues,
            _params: &serde_json::Value,
        ) -> Result<OutputValues, Error> {
            let value = &inputs["extent_from"];
            let extent = value.extent();
            let w = extent.width as usize;
            let h = extent.height as usize;
            let ch = value.channels() as usize;
            // A diagonal ramp keyed by absolute (x, y): position-dependent.
            let mut out = vec![0.0_f32; w * h * ch];
            for y in 0..h {
                for x in 0..w {
                    let base = (y * w + x) * ch;
                    for (c, slot) in out[base..base + ch].iter_mut().enumerate() {
                        #[allow(
                            clippy::cast_precision_loss,
                            reason = "small test extents; absolute-coordinate ramp"
                        )]
                        {
                            *slot = (x + y + c) as f32;
                        }
                    }
                }
            }
            let mut o = OutputValues::new();
            o.insert(
                "image".to_owned(),
                ResourceValue::new(*value.descriptor(), value.channels(), out).unwrap(),
            );
            Ok(o)
        }
    }

    fn manifest(
        id: &str,
        inputs: &[&str],
        outputs: &[&str],
        cat: RoiCategory,
    ) -> OperationManifest {
        OperationManifest {
            id: id.parse().unwrap(),
            impl_version: 1,
            summary: String::new(),
            determinism: DeterminismTier::Exact,
            roi: RoiPolicy {
                category: cat,
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
            manifest("source.ramp@1", &[], &["image"], RoiCategory::Pointwise),
            manifest(
                "filter.scale@1",
                &["image"],
                &["image"],
                RoiCategory::Pointwise,
            ),
            manifest(
                "filter.box@1",
                &["image"],
                &["image"],
                RoiCategory::Geometric,
            ),
            // A position-dependent pointwise generator (Pointwise ROI, one wired
            // input it does not read): the bn-3ai guard must run it whole-image.
            manifest(
                "paint.gradientish@1",
                &["extent_from"],
                &["image"],
                RoiCategory::Pointwise,
            ),
        ])
        .unwrap()
    }

    fn contracts() -> ContractRegistry {
        let mut c = ContractRegistry::new();
        c.register("source.ramp@1".parse().unwrap(), Box::new(Ramp))
            .unwrap();
        c.register("filter.scale@1".parse().unwrap(), Box::new(PointwiseScale))
            .unwrap();
        c.register("filter.box@1".parse().unwrap(), Box::new(HaloBox))
            .unwrap();
        c.register(
            "paint.gradientish@1".parse().unwrap(),
            Box::new(Gradientish),
        )
        .unwrap();
        c
    }

    fn implementations() -> crate::executor::op_impl::ImplRegistry {
        let mut r = crate::executor::op_impl::ImplRegistry::new();
        r.register("source.ramp@1".parse().unwrap(), Box::new(Ramp))
            .unwrap();
        r.register("filter.scale@1".parse().unwrap(), Box::new(PointwiseScale))
            .unwrap();
        r.register("filter.box@1".parse().unwrap(), Box::new(HaloBox))
            .unwrap();
        r.register(
            "paint.gradientish@1".parse().unwrap(),
            Box::new(Gradientish),
        )
        .unwrap();
        r
    }

    fn chain_plan() -> Plan {
        parse_plan(
            r#"{
                "paintop": "1.0", "inputs": {},
                "nodes": [
                    {"id": "src", "op": "source.ramp@1"},
                    {"id": "a", "op": "filter.scale@1", "in": {"image": "node:src/image"}},
                    {"id": "b", "op": "filter.scale@1", "in": {"image": "node:a/image"}}
                ],
                "exports": {"out": {"resource": "node:b/image", "kind": "image", "path": "o.png"}}
            }"#,
        )
        .unwrap()
    }

    #[test]
    fn crop_and_scatter_round_trip() {
        let extent = Extent::new(8, 8);
        let samples: Vec<f32> = (0..(8 * 8 * 4)).map(|i| i as f32).collect();
        let value = ResourceValue::new(image_descriptor(extent), 4, samples.clone()).unwrap();

        let tile = Rect::new(2, 2, 6, 6);
        let cropped = crop(&value, tile);
        assert_eq!(cropped.extent(), Extent::new(4, 4));

        // Scatter the crop back into a zero buffer and confirm the tile matches.
        let mut target =
            ResourceValue::new(image_descriptor(extent), 4, vec![0.0; 8 * 8 * 4]).unwrap();
        scatter(&mut target, &cropped, tile);
        for y in 2..6 {
            for x in 2..6 {
                for c in 0..4 {
                    let idx = ((y * 8 + x) * 4 + c) as usize;
                    assert_eq!(target.samples()[idx], samples[idx], "({x},{y},{c})");
                }
            }
        }
    }

    fn whole_image(plan: &Plan) -> BTreeMap<String, Vec<f32>> {
        let reg = registry();
        let graph = resolve_plan(plan, &reg).unwrap();
        let inputs: BTreeMap<String, ResourceValue> = BTreeMap::new();
        let execution = execute(plan, &graph, &reg, &implementations(), &inputs).unwrap();
        let mut out = BTreeMap::new();
        for node in &plan.nodes {
            if let Some(value) = execution.output(&node.id, "image") {
                out.insert(node.id.clone(), value.samples().to_vec());
            }
        }
        out
    }

    fn tiled(plan: &Plan, tile_size: u32) -> super::TiledExecution {
        tiled_with(plan, tile_size, super::ThreadCap::fixed(1))
    }

    fn tiled_with(plan: &Plan, tile_size: u32, cap: super::ThreadCap) -> super::TiledExecution {
        let reg = registry();
        let graph = resolve_plan(plan, &reg).unwrap();
        let checked = check_graph(plan, &graph, &reg, &contracts(), &BTreeMap::new()).unwrap();
        let roi = analyze_roi(plan, &graph, &checked, &contracts()).unwrap();
        let inputs: BTreeMap<String, ResourceValue> = BTreeMap::new();
        super::execute_tiled_with(
            plan,
            &graph,
            &checked,
            &reg,
            &contracts(),
            &implementations(),
            &roi,
            &inputs,
            tile_size,
            cap,
        )
        .unwrap()
    }

    /// The parallel-determinism property (bn-3ny): a tiled pointwise chain produces
    /// **bit-identical** output across thread counts {1, 2, 8}, and that output
    /// equals the M2 single-threaded tiled result. The thread cap is scheduler-owned
    /// — the kernels never spawn their own pool — and the M2 disjoint-tile / fixed
    /// scatter order make the result a pure function of the schedule, not the worker
    /// that produced each tile.
    #[test]
    fn parallel_tiled_is_bit_identical_across_thread_counts() {
        let plan = chain_plan();
        // The single-threaded M2 baseline.
        let baseline = tiled_with(&plan, 16, super::ThreadCap::fixed(1));
        for threads in [1_usize, 2, 8] {
            for tile_size in [8, 13, 16, 32] {
                let baseline_ts = tiled_with(&plan, tile_size, super::ThreadCap::fixed(1));
                let parallel = tiled_with(&plan, tile_size, super::ThreadCap::fixed(threads));
                for node in ["src", "a", "b"] {
                    let want = baseline_ts.output(node, "image").unwrap().samples();
                    let got = parallel.output(node, "image").unwrap().samples();
                    assert_eq!(
                        got, want,
                        "node {node} differs at {threads} threads, tile_size {tile_size}"
                    );
                }
            }
        }
        // The export also matches the single-threaded baseline byte-for-byte.
        let parallel = tiled_with(&plan, 16, super::ThreadCap::fixed(8));
        assert_eq!(
            parallel.exports()[0].1.samples(),
            baseline.exports()[0].1.samples()
        );
    }

    /// The parallel path equals the whole-image reference, not just itself across
    /// thread counts — proving the pool changes only *which thread* evaluates a
    /// pure tile, never the result.
    #[test]
    fn parallel_tiled_equals_whole_image() {
        let plan = chain_plan();
        let whole = whole_image(&plan);
        for threads in [2_usize, 8] {
            let parallel = tiled_with(&plan, 13, super::ThreadCap::fixed(threads));
            for (node, expected) in &whole {
                let got = parallel.output(node, "image").unwrap().samples();
                assert_eq!(got, expected.as_slice(), "node {node} at {threads} threads");
            }
        }
    }

    /// The neighbourhood (haloed) tiled path is also bit-identical across thread
    /// counts and equal to the whole-image reference — the halo crop + interior
    /// extract is a pure per-tile function, so parallel evaluation cannot introduce
    /// a seam.
    #[test]
    fn parallel_neighborhood_tiled_is_bit_identical() {
        let plan = halo_plan();
        let whole = whole_image(&plan);
        for threads in [1_usize, 2, 8] {
            for tile_size in [8, 13, 20] {
                let parallel = tiled_with(&plan, tile_size, super::ThreadCap::fixed(threads));
                let got = parallel.output("box", "image").unwrap().samples();
                assert_eq!(
                    got,
                    whole["box"].as_slice(),
                    "neighbourhood node differs at {threads} threads, tile_size {tile_size}"
                );
            }
        }
    }

    /// The tile stats are independent of the thread count: the same requested /
    /// executed / identity counts whether run on 1 or 8 threads.
    #[test]
    fn parallel_tile_stats_match_sequential() {
        let plan = chain_plan();
        let seq = tiled_with(&plan, 32, super::ThreadCap::fixed(1));
        let par = tiled_with(&plan, 32, super::ThreadCap::fixed(8));
        assert_eq!(seq.stats(), par.stats());
    }

    /// A throughput smoke test: a small tile size over the 64x48 extent yields many
    /// tiles per node (≈24 at `tile_size` 8), so the 8-thread pool genuinely fans the
    /// work out. The run completes (no panic / deadlock / pool leak) and is
    /// bit-identical to the single-threaded baseline — the cheapest possible proof
    /// that the scheduler-owned pool drives real parallel tile work correctly.
    #[test]
    fn parallel_throughput_smoke_many_tiles() {
        let plan = chain_plan();
        let seq = tiled_with(&plan, 8, super::ThreadCap::fixed(1));
        // Run several times so a data race or a pool-reuse bug would surface as a
        // non-deterministic divergence across iterations.
        for _ in 0..8 {
            let par = tiled_with(&plan, 8, super::ThreadCap::fixed(8));
            assert_eq!(
                par.output("b", "image").unwrap().samples(),
                seq.output("b", "image").unwrap().samples()
            );
        }
        // The auto cap (host parallelism) also runs and matches.
        let auto = tiled_with(&plan, 8, super::ThreadCap::auto());
        assert_eq!(
            auto.output("b", "image").unwrap().samples(),
            seq.output("b", "image").unwrap().samples()
        );
    }

    #[test]
    fn tiled_equals_whole_image_bit_identical() {
        let plan = chain_plan();
        let whole = whole_image(&plan);
        for tile_size in [16, 32, 64, 256] {
            let tiled = tiled(&plan, tile_size);
            for (node, expected) in &whole {
                let got = tiled.output(node, "image").unwrap().samples();
                assert_eq!(
                    got,
                    expected.as_slice(),
                    "node {node} differs at tile_size {tile_size}"
                );
            }
        }
    }

    #[test]
    fn no_tile_seam_at_boundaries() {
        // A tile size that does not divide the extent (64x48 / 16 => 4x3 tiles
        // exactly; use 20 to force ragged edge tiles 20,20,20,4 wide).
        let plan = chain_plan();
        let whole = whole_image(&plan);
        let tiled = tiled(&plan, 20);
        let got = tiled.output("b", "image").unwrap().samples();
        assert_eq!(got, whole["b"].as_slice());
    }

    #[test]
    fn tile_stats_count_requested_and_executed() {
        let plan = chain_plan();
        // 64x48 with 32-tile => 2x2 = 4 tiles per pointwise node (a, b); src is a
        // whole-image source fallback that does not contribute tile stats.
        let tiled = tiled(&plan, 32);
        let stats = tiled.stats();
        // a and b: 4 tiles each, all executed (no masking) => 8 requested, 8 run.
        assert_eq!(stats.requested, 8);
        assert_eq!(stats.executed, 8);
        assert_eq!(stats.identity, 0);
    }

    #[test]
    fn export_matches_whole_image() {
        let plan = chain_plan();
        let whole = whole_image(&plan);
        let tiled = tiled(&plan, 16);
        let export = &tiled.exports()[0];
        assert_eq!(export.0, "out");
        assert_eq!(export.1.samples(), whole["b"].as_slice());
    }

    /// An external `input:ext` image of the working extent, the generator's
    /// `extent_from` source (mirroring a real gradient fed by `input:src`).
    fn ext_input() -> BTreeMap<String, ResourceValue> {
        let len = (EXTENT.width * EXTENT.height * 4) as usize;
        let value = ResourceValue::new(image_descriptor(EXTENT), 4, vec![1.0; len]).unwrap();
        let mut m = BTreeMap::new();
        m.insert("ext".to_owned(), value);
        m
    }

    /// Whole-image / tiled runners that supply the external `input:ext`, for the
    /// generator-guard plans.
    fn whole_image_ext(plan: &Plan) -> BTreeMap<String, Vec<f32>> {
        let reg = registry();
        let graph = resolve_plan(plan, &reg).unwrap();
        let inputs = ext_input();
        let execution = execute(plan, &graph, &reg, &implementations(), &inputs).unwrap();
        let mut out = BTreeMap::new();
        for node in &plan.nodes {
            if let Some(value) = execution.output(&node.id, "image") {
                out.insert(node.id.clone(), value.samples().to_vec());
            }
        }
        out
    }

    fn tiled_ext(plan: &Plan, tile_size: u32) -> super::TiledExecution {
        let reg = registry();
        let graph = resolve_plan(plan, &reg).unwrap();
        let mut input_descriptors: BTreeMap<String, ResourceDescriptor> = BTreeMap::new();
        input_descriptors.insert("ext".to_owned(), image_descriptor(EXTENT));
        let checked = check_graph(plan, &graph, &reg, &contracts(), &input_descriptors).unwrap();
        let roi = analyze_roi(plan, &graph, &checked, &contracts()).unwrap();
        let inputs = ext_input();
        super::execute_tiled(
            plan,
            &graph,
            &checked,
            &reg,
            &contracts(),
            &implementations(),
            &roi,
            &inputs,
            tile_size,
        )
        .unwrap()
    }

    /// A plan whose only working node is a position-dependent **generator**
    /// (`paint.gradientish@1`): a `Pointwise` ROI op with a wired `extent_from`
    /// input it does not read. Without the bn-3ai guard it would be tiled and each
    /// tile would restart its absolute-coordinate ramp at the tile origin, diverging
    /// from the whole-image result.
    fn gradient_plan() -> Plan {
        parse_plan(
            r#"{
                "paintop": "1.0",
                "inputs": {"ext": {"kind": "image.file", "path": "e.png"}},
                "nodes": [
                    {"id": "g", "op": "paint.gradientish@1", "in": {"extent_from": "input:ext"}}
                ],
                "exports": {"out": {"resource": "node:g/image", "kind": "image", "path": "o.png"}}
            }"#,
        )
        .unwrap()
    }

    /// The bn-3ai guard: tiling a position-dependent generator equals the
    /// whole-image result bit-for-bit (it falls back to a single whole-image
    /// dispatch rather than being tiled), across tile sizes that divide the extent
    /// and ragged ones.
    #[test]
    fn position_dependent_generator_is_not_tiled() {
        let plan = gradient_plan();
        let whole = whole_image_ext(&plan);
        for tile_size in [8, 13, 16, 20, 32, 64] {
            let tiled = tiled_ext(&plan, tile_size);
            let got = tiled.output("g", "image").unwrap().samples();
            assert_eq!(
                got,
                whole["g"].as_slice(),
                "tiled generator diverges from whole-image at tile_size {tile_size}"
            );
            // The generator runs whole-image: no per-tile work is recorded, so the
            // stats stay zero.
            assert_eq!(
                tiled.stats(),
                super::TileStats::default(),
                "a position-dependent generator must not be tiled (tile_size {tile_size})"
            );
        }
    }

    /// The guard does not over-restrict: a real position-independent pointwise op
    /// (`filter.scale@1`) downstream of the generator is still tiled.
    #[test]
    fn pointwise_op_after_a_generator_still_tiles() {
        let plan = parse_plan(
            r#"{
                "paintop": "1.0",
                "inputs": {"ext": {"kind": "image.file", "path": "e.png"}},
                "nodes": [
                    {"id": "g", "op": "paint.gradientish@1", "in": {"extent_from": "input:ext"}},
                    {"id": "s", "op": "filter.scale@1", "in": {"image": "node:g/image"}}
                ],
                "exports": {"out": {"resource": "node:s/image", "kind": "image", "path": "o.png"}}
            }"#,
        )
        .unwrap();
        let whole = whole_image_ext(&plan);
        // 64x48 / 32 => 2x2 = 4 tiles for the lone pointwise node `s`; `g` runs
        // whole-image.
        let tiled = tiled_ext(&plan, 32);
        assert_eq!(
            tiled.output("s", "image").unwrap().samples(),
            whole["s"].as_slice()
        );
        assert_eq!(
            tiled.output("g", "image").unwrap().samples(),
            whole["g"].as_slice()
        );
        let stats = tiled.stats();
        assert_eq!(stats.executed, 4, "only the pointwise consumer should tile");
        assert_eq!(stats.requested, 4);
    }

    /// A plan mixing a pointwise op with a genuine neighbourhood op
    /// (`filter.box@1`, a clamped 3x3 box-sum). The neighbourhood node reads a
    /// haloed input window per tile, so a halo off-by-one or a boundary applied at
    /// the tile edge would diverge from the whole-image reference.
    fn halo_plan() -> Plan {
        parse_plan(
            r#"{
                "paintop": "1.0", "inputs": {},
                "nodes": [
                    {"id": "src", "op": "source.ramp@1"},
                    {"id": "a", "op": "filter.scale@1", "in": {"image": "node:src/image"}},
                    {"id": "box", "op": "filter.box@1", "in": {"image": "node:a/image"}}
                ],
                "exports": {"out": {"resource": "node:box/image", "kind": "image", "path": "o.png"}}
            }"#,
        )
        .unwrap()
    }

    #[test]
    fn neighborhood_tiled_equals_whole_image_bit_identical() {
        let plan = halo_plan();
        let whole = whole_image(&plan);
        // Tile sizes that divide the 64x48 extent and ones that leave ragged edge
        // tiles, so the haloed crop is exercised at interior and boundary tiles.
        for tile_size in [8, 13, 16, 20, 32, 64] {
            let tiled = tiled(&plan, tile_size);
            let got = tiled.output("box", "image").unwrap().samples();
            assert_eq!(
                got,
                whole["box"].as_slice(),
                "neighbourhood node differs from whole-image at tile_size {tile_size}"
            );
        }
    }

    #[test]
    fn neighborhood_tiled_export_matches_whole_image() {
        let plan = halo_plan();
        let whole = whole_image(&plan);
        let tiled = tiled(&plan, 13);
        assert_eq!(tiled.exports()[0].1.samples(), whole["box"].as_slice());
    }
}
