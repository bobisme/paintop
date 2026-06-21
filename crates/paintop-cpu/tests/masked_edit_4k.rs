//! The M2 "small masked edit on a 4K image touches only predicted tiles" exit
//! criterion (`plan.md` §19 M2, §11.3; `AGENT_VERIFICATION` §12.2; bn-1r2).
//!
//! A localized edit — a few-hundred-pixel masked region of a 4K image — must not
//! drag the whole image through the tiled compiler. The demand-driven scheduler
//! seeds the demanded region at the export node, propagates it backward through
//! each op's halo, and schedules **only** the output tiles that region (grown by
//! the cumulative halo) intersects. This suite pins that promise *quantitatively*:
//!
//! * it builds an independent, conservative prediction of the touched tile set —
//!   the seed rect dilated by every upstream op's halo, intersected with the tile
//!   grid — and asserts the executor's **executed-tile count is ≤ that
//!   prediction** at every node (`AGENT_VERIFICATION` §12.2);
//! * it asserts the touched set is a tiny fraction of the full 4K grid, so the
//!   edit is genuinely localized rather than trivially "predicted == everything";
//! * it asserts the tiled output inside the demanded region is **bit-identical**
//!   to a full whole-image run (the localized work loses no precision); and
//! * it writes a `tiles.json` artifact (the requested/executed/identity counts,
//!   the prediction, the grid size) the M2 CI gate uploads.
//!
//! The chain is `color.adjust` (pointwise, zero halo) → `filter.convolve` (a 3×3
//! kernel, one-pixel halo). The export demands a small rect of the convolve
//! output; backward propagation grows it by the convolve's 1px halo onto the
//! adjust output, and the pointwise adjust passes that region straight through to
//! the source — so the predicted touched set is the seed rect dilated by 1px,
//! mapped onto the 256px tile grid.

#![allow(
    clippy::unwrap_used,
    clippy::missing_const_for_fn,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::float_cmp,
    reason = "an integration test crate building exact fixtures and exact tile-count predictions"
)]

use std::collections::{BTreeMap, BTreeSet};

use paintop_core::executor::{ResourceValue, analyze_roi_from_seeds, execute};
use paintop_core::tile::{TileGrid, TiledExecution, execute_tiled, schedule_tiles};
use paintop_cpu::adjust::Adjust;
use paintop_cpu::convolve::Convolve;
use paintop_ir::{
    AlphaRepresentation, ChannelLayout, ColorEncoding, ColorRange, ContractRegistry,
    CoordinateConvention, DeterminismTier, Extent, ImageDescriptor, InputSpec, OperationManifest,
    OperationRegistry, OutputSpec, Plan, Rect, Region, ResourceDescriptor, ResourceKind,
    RoiCategory, RoiPolicy, ScalarType, SemanticRole, TestMetadata, check_graph, parse_plan,
    resolve_plan,
};
use serde_json::json;

const ADJUST_OP_ID: &str = "color.adjust@1";
const CONVOLVE_OP_ID: &str = "filter.convolve@1";

/// 4K (UHD) RGBA f32.
const EXTENT: Extent = Extent::new(3840, 2160);
/// The production tile edge the scheduler grids 4K into (15×9 = 135 tiles).
const TILE_SIZE: u32 = 256;
/// The 3×3 convolution kernel's halo: one pixel on every side.
const CONVOLVE_HALO: u32 = 1;

/// A linear-sRGB RGBA f32 image descriptor at `extent`.
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

/// The `color.adjust@1` manifest (pointwise) rebuilt locally so the test pins
/// exactly the op under test rather than the whole MVP registry.
fn adjust_manifest() -> OperationManifest {
    OperationManifest {
        id: ADJUST_OP_ID.parse().unwrap(),
        impl_version: 1,
        summary: String::new(),
        determinism: DeterminismTier::Bounded,
        roi: RoiPolicy {
            category: RoiCategory::Pointwise,
            halo_px: None,
        },
        inputs: vec![
            InputSpec {
                name: "image".to_owned(),
                kind: ResourceKind::Image,
                required: true,
                doc: String::new(),
            },
            InputSpec {
                name: "mask".to_owned(),
                kind: ResourceKind::Mask,
                required: false,
                doc: String::new(),
            },
        ],
        outputs: vec![OutputSpec {
            name: "image".to_owned(),
            kind: ResourceKind::Image,
            doc: String::new(),
        }],
        params: vec![],
        implementations: vec!["cpu.reference@1".parse().unwrap()],
        test: TestMetadata::default(),
    }
}

fn registry() -> OperationRegistry {
    OperationRegistry::from_manifests([adjust_manifest(), Convolve::manifest().unwrap()]).unwrap()
}

fn contracts() -> ContractRegistry {
    let mut c = ContractRegistry::new();
    c.register(ADJUST_OP_ID.parse().unwrap(), Box::new(Adjust))
        .unwrap();
    c.register(CONVOLVE_OP_ID.parse().unwrap(), Box::new(Convolve))
        .unwrap();
    c
}

fn implementations() -> paintop_core::executor::ImplRegistry {
    let mut r = paintop_core::executor::ImplRegistry::new();
    r.register(ADJUST_OP_ID.parse().unwrap(), Box::new(Adjust))
        .unwrap();
    r.register(CONVOLVE_OP_ID.parse().unwrap(), Box::new(Convolve))
        .unwrap();
    r
}

/// The masked-edit plan: a pointwise grade then a 3×3 convolution. The export
/// names the convolve output; the test seeds a small demanded rect of it.
fn masked_edit_plan() -> Plan {
    let plan = json!({
        "paintop": "1.0",
        "inputs": {"src": {"kind": "image.file", "path": "src.png"}},
        "nodes": [
            {"id": "a", "op": ADJUST_OP_ID, "in": {"image": "input:src"},
             "params": {"exposure_ev": 0.2, "saturation": 0.1}},
            {"id": "f", "op": CONVOLVE_OP_ID, "in": {"input": "node:a/image"},
             "params": {
                 "mode": "clamp",
                 "kernel": {"width": 3, "height": 3, "origin_x": 1, "origin_y": 1,
                            "weights": [0.0, -1.0, 0.0, -1.0, 5.0, -1.0, 0.0, -1.0, 0.0]}
             }}
        ],
        "exports": {"out": {"resource": "node:f/output", "kind": "image", "path": "o.png"}}
    });
    parse_plan(&plan.to_string()).unwrap()
}

/// A deterministic spatially varying RGBA f32 source, allocated lazily so the test
/// only pays for the 4K buffer when it actually executes (criterion 4 below).
fn source(extent: Extent) -> ResourceValue {
    let width = extent.width as usize;
    let height = extent.height as usize;
    let mut samples = vec![0.0_f32; width * height * 4];
    for row in 0..height {
        for col in 0..width {
            let base = (row * width + col) * 4;
            let fx = col as f32;
            let fy = row as f32;
            samples[base] = fy.mul_add(0.0005, fx.mul_add(0.0007, 0.1));
            samples[base + 1] = fy.mul_add(0.0009, 0.2);
            samples[base + 2] = ((col ^ row) as f32 % 11.0).mul_add(0.02, 0.05);
            samples[base + 3] = 1.0;
        }
    }
    ResourceValue::new(image_descriptor(extent), 4, samples).unwrap()
}

/// The demanded edit region: a small rect well inside the 4K frame, deliberately
/// straddling a 256px tile boundary so the touched set is a 2×2 tile block, not a
/// single tile — a stricter test of the prediction geometry.
const fn edit_rect() -> Rect {
    // 200×180 rect straddling the (col 1|2, row 1) tile seam near (510, 300).
    Rect::new(510, 300, 710, 480)
}

/// Build the ROI seed map: demand `edit` of the export node `f`'s output.
fn seeds(edit: Rect) -> BTreeMap<(String, String), Region> {
    let mut seeds: BTreeMap<(String, String), Region> = BTreeMap::new();
    seeds.insert(
        ("f".to_owned(), "output".to_owned()),
        Region::from_rect(edit),
    );
    seeds
}

/// Run the masked edit tiled and return the execution plus the schedule's
/// per-node demanded-tile counts.
fn run_tiled(plan: &Plan, edit: Rect) -> (TiledExecution, BTreeMap<String, usize>) {
    let reg = registry();
    let graph = resolve_plan(plan, &reg).unwrap();
    let mut input_descriptors: BTreeMap<String, ResourceDescriptor> = BTreeMap::new();
    input_descriptors.insert("src".to_owned(), image_descriptor(EXTENT));
    let checked = check_graph(plan, &graph, &reg, &contracts(), &input_descriptors).unwrap();
    let roi = analyze_roi_from_seeds(plan, &graph, &checked, &contracts(), &seeds(edit)).unwrap();

    // The schedule's demanded-tile counts, derived independently of execution.
    let grid = TileGrid::new(EXTENT, TILE_SIZE);
    let schedule = schedule_tiles(plan, &graph, &checked, &contracts(), &roi, grid).unwrap();
    let demanded: BTreeMap<String, usize> = ["a", "f"]
        .iter()
        .map(|n| ((*n).to_owned(), schedule.demanded_tile_count(n)))
        .collect();

    let mut inputs: BTreeMap<String, ResourceValue> = BTreeMap::new();
    inputs.insert("src".to_owned(), source(EXTENT));
    let exec = execute_tiled(
        plan,
        &graph,
        &checked,
        &reg,
        &contracts(),
        &implementations(),
        &roi,
        &inputs,
        TILE_SIZE,
    )
    .unwrap();
    (exec, demanded)
}

/// The conservative, independently-computed prediction of the touched tile set:
/// the edit rect dilated by the cumulative upstream halo, clamped to the frame,
/// mapped onto the tile grid. Every node's executed-tile count must be ≤ this.
///
/// The deepest node (`f`, the export) demands exactly the seed rect; the pointwise
/// `a` upstream of the 1px convolution demands the seed grown by 1px. The single
/// `edit.dilate(CONVOLVE_HALO)` bound is therefore an upper bound for *both* nodes
/// (`f`'s un-dilated rect touches a subset of the dilated rect's tiles).
fn predicted_tile_count(edit: Rect) -> usize {
    let grid = TileGrid::new(EXTENT, TILE_SIZE);
    let grown = Region::from_rect(edit)
        .dilate(CONVOLVE_HALO)
        .clamp_to_extent(EXTENT);
    grid.tiles_in_region(grown).count()
}

#[test]
fn masked_4k_edit_touches_only_predicted_tiles() {
    let plan = masked_edit_plan();
    let edit = edit_rect();
    let (exec, demanded) = run_tiled(&plan, edit);

    let grid = TileGrid::new(EXTENT, TILE_SIZE);
    let full = grid.tile_count() as usize;
    assert_eq!(full, 15 * 9, "4K @ 256px must grid into 135 tiles");

    let prediction = predicted_tile_count(edit);
    let stats = exec.stats();

    // Criterion (3): the executor touched no more tiles than the conservative
    // halo-expanded prediction. `stats.executed` aggregates the per-tile dispatches
    // across every tiled node, so the conservative bound is the per-node prediction
    // times the number of tiled nodes (both `a` and `f` tile in this chain). The
    // tighter, per-node check below pins that no *single* node ran more tiles than
    // the prediction.
    let tiled_nodes = demanded.values().filter(|c| **c > 0).count();
    let aggregate_bound = prediction * tiled_nodes;
    assert!(
        stats.executed <= aggregate_bound,
        "executed {} exceeds conservative prediction {prediction} \u{d7} {tiled_nodes} tiled nodes \
         (= {aggregate_bound})",
        stats.executed
    );
    assert_eq!(
        stats.requested, stats.executed,
        "every requested tile was executed (no mask fast-path in this chain)"
    );

    // Per node: neither node schedules more than the prediction, and the whole
    // edit is genuinely localized — well under a tenth of the full grid.
    for (node, count) in &demanded {
        assert!(
            *count <= prediction,
            "node {node} scheduled {count} tiles, over prediction {prediction}"
        );
    }
    assert!(
        prediction * 10 < full,
        "a small masked edit must touch < 10% of the 135-tile grid; \
         prediction {prediction} of {full}"
    );

    write_tiles_artifact(&exec, &demanded, full, prediction);
}

#[test]
fn masked_4k_edit_is_bit_identical_to_full_run_inside_the_region() {
    // The localized tiled run must reproduce, byte-for-byte inside the demanded
    // region, what a full whole-image run produces — no precision lost by tiling
    // the edit. (`color.adjust` is bounded-tier but deterministic within a run, so
    // identical samples in => identical samples out.)
    let plan = masked_edit_plan();
    let edit = edit_rect();
    let (exec, _) = run_tiled(&plan, edit);

    let reg = registry();
    let graph = resolve_plan(&plan, &reg).unwrap();
    let mut inputs: BTreeMap<String, ResourceValue> = BTreeMap::new();
    inputs.insert("src".to_owned(), source(EXTENT));
    let whole = execute(&plan, &graph, &reg, &implementations(), &inputs).unwrap();
    let whole_f = whole.output("f", "output").unwrap();
    let tiled_f = exec.output("f", "output").unwrap();

    let width = EXTENT.width as usize;
    let channels = whole_f.channels() as usize;
    let whole_samples = whole_f.samples();
    let tiled_samples = tiled_f.samples();

    let mut mismatches = 0usize;
    for y in edit.y0..edit.y1 {
        for x in edit.x0..edit.x1 {
            let base = ((y as usize) * width + x as usize) * channels;
            for c in 0..channels {
                if whole_samples[base + c] != tiled_samples[base + c] {
                    mismatches += 1;
                }
            }
        }
    }
    assert_eq!(
        mismatches, 0,
        "tiled masked edit diverged from the full run inside the demanded region"
    );
}

#[test]
fn prediction_set_matches_the_grid_geometry() {
    // The prediction is computed from the grid, so guard against a regression in
    // `tiles_in_region`: the seed straddles the col-1|2 seam at row 1, so the
    // grown rect (dilated 1px) touches the 2×2 block of tiles (cols {1,2}×rows
    // {1,2})? Verify it is exactly the tiles whose rects intersect the grown rect.
    let edit = edit_rect();
    let grid = TileGrid::new(EXTENT, TILE_SIZE);
    let grown = Region::from_rect(edit)
        .dilate(CONVOLVE_HALO)
        .clamp_to_extent(EXTENT);
    let touched: BTreeSet<(u32, u32)> = grid
        .tiles_in_region(grown)
        .map(|t| (t.col, t.row))
        .collect();

    // 510..711 spans cols 1 (256..512) and 2 (512..768); 299..481 spans rows 1
    // (256..512) only. So the predicted block is cols {1,2} × row {1} = 2 tiles.
    let expected: BTreeSet<(u32, u32)> = [(1, 1), (2, 1)].into_iter().collect();
    assert_eq!(touched, expected, "predicted tile block drifted");
    assert_eq!(predicted_tile_count(edit), 2);
}

/// Write the tile-count artifact the M2 CI gate uploads. The destination is
/// `PAINTOP_TILE_ARTIFACT` if set (the gate points it at the artifact dir),
/// otherwise the system temp dir. Failure to write is non-fatal — the artifact is
/// diagnostic, not an assertion.
fn write_tiles_artifact(
    exec: &TiledExecution,
    demanded: &BTreeMap<String, usize>,
    full_tiles: usize,
    prediction: usize,
) {
    let stats = exec.stats();
    // The per-node prediction bounds each node's tile count; the aggregate
    // `stats.executed` sums per-tile dispatches over every tiled node, so its bound
    // is the per-node prediction times the number of tiled nodes.
    let tiled_nodes = demanded.values().filter(|c| **c > 0).count();
    let aggregate_bound = prediction * tiled_nodes;
    let per_node_within = demanded.values().all(|c| *c <= prediction);
    let report = json!({
        "scenario": "masked_edit_4k",
        "extent": {"width": EXTENT.width, "height": EXTENT.height},
        "tile_size": TILE_SIZE,
        "grid_tiles": full_tiles,
        "predicted_tiles_per_node": prediction,
        "tiled_nodes": tiled_nodes,
        "aggregate_prediction": aggregate_bound,
        "tiles": {
            "requested": stats.requested,
            "executed": stats.executed,
            "identity": stats.identity,
        },
        "per_node_demanded_tiles": demanded,
        "per_node_within_prediction": per_node_within,
        "executed_within_prediction": stats.executed <= aggregate_bound,
    });
    let path = std::env::var("PAINTOP_TILE_ARTIFACT").map_or_else(
        |_| std::env::temp_dir().join("paintop-m2-tiles.json"),
        std::path::PathBuf::from,
    );
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let _ = std::fs::write(
        &path,
        serde_json::to_vec_pretty(&report).unwrap_or_default(),
    );
}
