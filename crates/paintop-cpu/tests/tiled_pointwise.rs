//! Tiled-vs-whole differential for the pointwise M1 chain, and the 4K
//! bounded-memory assertion (`plan.md` §11; M2 exit gate, bn-1vx).
//!
//! The M2 promise is that tiled execution equals whole-image execution
//! **bit-for-bit** for exact pointwise ops, with no seam artifacts, while keeping
//! the resident working set bounded regardless of image size. This suite pins
//! both on a real M1 pointwise chain (`color.adjust@1` chained over a spatially
//! varying input):
//!
//! * `tiled_chain_is_bit_identical_to_whole_image` — every node output and the
//!   export match the whole-image executor byte-for-byte, across several tile
//!   sizes including ones that do not divide the extent (ragged edge tiles), so a
//!   seam or off-by-one in the crop/scatter geometry would be caught;
//! * `bounded_working_set_for_a_4k_pointwise_plan` — the demand-driven schedule's
//!   peak live-buffer count is a small constant independent of chain length, so
//!   the derived peak resident bytes for a 4K input stay under a fixed cap and
//!   well below the whole-image residency of holding every node output at once.
//!
//! The metrics the M2 gate consumes — `TileStats { requested, executed,
//! identity }` and the schedule's peak working set — are asserted directly here.

#![allow(
    clippy::unwrap_used,
    clippy::missing_const_for_fn,
    clippy::suboptimal_flops,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "an integration test crate building exact fixtures from integer indices"
)]

use std::collections::BTreeMap;

use paintop_core::executor::{ResourceValue, analyze_roi, execute};
use paintop_core::tile::{TileGrid, execute_tiled, schedule_tiles};
use paintop_cpu::adjust::Adjust;
use paintop_ir::{
    AlphaRepresentation, ChannelLayout, ColorEncoding, ColorRange, ContractRegistry,
    CoordinateConvention, DeterminismTier, Extent, ImageDescriptor, InputSpec, OperationManifest,
    OperationRegistry, OutputSpec, Plan, ResourceDescriptor, ResourceKind, RoiCategory, RoiPolicy,
    ScalarType, SemanticRole, TestMetadata, check_graph, parse_plan, resolve_plan,
};

const ADJUST_OP_ID: &str = "color.adjust@1";

/// A linear-sRGB RGBA `f32` image descriptor at `extent`.
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

/// The `color.adjust@1` manifest the registry resolves the plan against. We
/// rebuild it locally (rather than pull the whole MVP registry) so the test
/// pins exactly the op under test.
fn adjust_manifest() -> OperationManifest {
    OperationManifest {
        id: ADJUST_OP_ID.parse().unwrap(),
        impl_version: 1,
        summary: String::new(),
        // `color.adjust` is bounded-determinism cross-platform (exposure uses
        // `exp2`), but within a single run it is a deterministic per-pixel
        // function, so tiled and whole-image execution call it on identical
        // samples and agree bit-for-bit.
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
    OperationRegistry::from_manifests([adjust_manifest()]).unwrap()
}

fn contracts() -> ContractRegistry {
    let mut c = ContractRegistry::new();
    c.register(ADJUST_OP_ID.parse().unwrap(), Box::new(Adjust))
        .unwrap();
    c
}

fn implementations() -> paintop_core::executor::ImplRegistry {
    let mut r = paintop_core::executor::ImplRegistry::new();
    r.register(ADJUST_OP_ID.parse().unwrap(), Box::new(Adjust))
        .unwrap();
    r
}

/// A two-node `color.adjust` chain reading an external `input:src` image. Each
/// adjust applies a different exposure so the chain composes two grades.
fn chain_plan() -> Plan {
    parse_plan(
        r#"{
            "paintop": "1.0",
            "inputs": {"src": {"kind": "image.file", "path": "src.png"}},
            "nodes": [
                {"id": "a", "op": "color.adjust@1", "in": {"image": "input:src"},
                 "params": {"saturation": 0.3, "temperature": 0.1}},
                {"id": "b", "op": "color.adjust@1", "in": {"image": "node:a/image"},
                 "params": {"exposure_ev": -0.25, "saturation": -0.2}}
            ],
            "exports": {"out": {"resource": "node:b/image", "kind": "image", "path": "o.png"}}
        }"#,
    )
    .unwrap()
}

/// A spatially varying RGBA `f32` ramp: each channel of each pixel is a distinct
/// function of `(x, y)`, scaled into a moderate linear-light range. Variation in
/// every dimension is what lets a tile-seam bug surface.
fn ramp(extent: Extent) -> ResourceValue {
    let w = extent.width as usize;
    let h = extent.height as usize;
    let mut samples = vec![0.0_f32; w * h * 4];
    for y in 0..h {
        for x in 0..w {
            let base = (y * w + x) * 4;
            #[allow(clippy::cast_precision_loss, reason = "small test extents")]
            {
                samples[base] = (x as f32) * 0.01 + 0.1;
                samples[base + 1] = (y as f32) * 0.013 + 0.2;
                samples[base + 2] = ((x + y) as f32) * 0.007 + 0.05;
                samples[base + 3] = 1.0;
            }
        }
    }
    ResourceValue::new(image_descriptor(extent), 4, samples).unwrap()
}

/// The whole-image node outputs of the chain, keyed by node id.
fn whole_image(plan: &Plan, extent: Extent) -> BTreeMap<String, Vec<f32>> {
    let reg = registry();
    let graph = resolve_plan(plan, &reg).unwrap();
    let mut inputs: BTreeMap<String, ResourceValue> = BTreeMap::new();
    inputs.insert("src".to_owned(), ramp(extent));
    let execution = execute(plan, &graph, &reg, &implementations(), &inputs).unwrap();
    let mut out = BTreeMap::new();
    for node in &plan.nodes {
        if let Some(value) = execution.output(&node.id, "image") {
            out.insert(node.id.clone(), value.samples().to_vec());
        }
    }
    out
}

/// The tiled execution of the chain at `tile_size`.
fn tiled(plan: &Plan, extent: Extent, tile_size: u32) -> paintop_core::tile::TiledExecution {
    let reg = registry();
    let graph = resolve_plan(plan, &reg).unwrap();
    let mut input_descriptors: BTreeMap<String, ResourceDescriptor> = BTreeMap::new();
    input_descriptors.insert("src".to_owned(), image_descriptor(extent));
    let checked = check_graph(plan, &graph, &reg, &contracts(), &input_descriptors).unwrap();
    let roi = analyze_roi(plan, &graph, &checked, &contracts()).unwrap();
    let mut inputs: BTreeMap<String, ResourceValue> = BTreeMap::new();
    inputs.insert("src".to_owned(), ramp(extent));
    execute_tiled(
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

#[test]
fn tiled_chain_is_bit_identical_to_whole_image() {
    let extent = Extent::new(200, 150);
    let plan = chain_plan();
    let whole = whole_image(&plan, extent);

    // Tile sizes that divide the extent and ones that do not (ragged edges):
    // 200x150 / 64 => 4x3 with edge tiles 64,64,64,8 wide and 64,64,22 tall.
    for tile_size in [32, 50, 64, 128, 256] {
        let exec = tiled(&plan, extent, tile_size);
        for (node, expected) in &whole {
            let got = exec.output(node, "image").unwrap().samples();
            assert_eq!(
                got,
                expected.as_slice(),
                "node {node} differs from whole-image at tile_size {tile_size}"
            );
        }
        // The export equals the whole-image final node, byte for byte.
        let export = &exec.exports()[0];
        assert_eq!(export.1.samples(), whole["b"].as_slice());
    }
}

#[test]
fn no_seam_artifacts_on_ragged_tiles() {
    // A prime-ish tile size guarantees ragged right/bottom edge tiles across the
    // whole image; any seam handling error would show as a mismatch row/column.
    let extent = Extent::new(127, 83);
    let plan = chain_plan();
    let whole = whole_image(&plan, extent);
    let exec = tiled(&plan, extent, 37);
    assert_eq!(exec.output("b", "image").unwrap().samples(), whole["b"]);
}

#[test]
fn tile_stats_account_for_every_requested_tile() {
    let extent = Extent::new(128, 128);
    let plan = chain_plan();
    let exec = tiled(&plan, extent, 64); // 2x2 = 4 tiles per adjust node
    let stats = exec.stats();
    // Two adjust nodes, 4 tiles each, all executed (no masking).
    assert_eq!(stats.requested, 8);
    assert_eq!(stats.executed, 8);
    assert_eq!(stats.identity, 0);
}

#[test]
fn bounded_working_set_for_a_4k_pointwise_plan() {
    // A four-stage adjust chain on a 4K RGBA f32 image. Scheduling and liveness
    // are computed analytically (no 100+ MB allocation), so the test is cheap.
    let extent = Extent::new(3840, 2160);
    let plan = parse_plan(
        r#"{
            "paintop": "1.0",
            "inputs": {"src": {"kind": "image.file", "path": "src.png"}},
            "nodes": [
                {"id": "a", "op": "color.adjust@1", "in": {"image": "input:src"}, "params": {"exposure_ev": 0.1}},
                {"id": "b", "op": "color.adjust@1", "in": {"image": "node:a/image"}, "params": {"exposure_ev": 0.1}},
                {"id": "c", "op": "color.adjust@1", "in": {"image": "node:b/image"}, "params": {"exposure_ev": 0.1}},
                {"id": "d", "op": "color.adjust@1", "in": {"image": "node:c/image"}, "params": {"exposure_ev": 0.1}}
            ],
            "exports": {"out": {"resource": "node:d/image", "kind": "image", "path": "o.png"}}
        }"#,
    )
    .unwrap();

    let reg = registry();
    let graph = resolve_plan(&plan, &reg).unwrap();
    let mut input_descriptors: BTreeMap<String, ResourceDescriptor> = BTreeMap::new();
    input_descriptors.insert("src".to_owned(), image_descriptor(extent));
    let checked = check_graph(&plan, &graph, &reg, &contracts(), &input_descriptors).unwrap();
    let roi = analyze_roi(&plan, &graph, &checked, &contracts()).unwrap();
    let grid = TileGrid::with_default(extent);
    let schedule = schedule_tiles(&plan, &graph, &checked, &contracts(), &roi, grid).unwrap();

    // A linear chain holds at most two node-output buffers live at once (the
    // producer being consumed and the consumer being produced), independent of the
    // chain length — the bounded working set the M2 gate requires.
    let peak = schedule.peak_live_buffers();
    assert_eq!(peak, 2, "linear chain working set must be a small constant");

    // The 4K full-buffer residency: peak_live * (W*H*4ch*4bytes). With peak == 2
    // this is ~254 MB; the whole-image executor that keeps all four node outputs
    // resident would need ~507 MB. Assert the tiled working set is under a fixed
    // 300 MiB cap and strictly below the all-nodes-resident figure.
    let cap_bytes: u64 = 300 * 1024 * 1024;
    let full_buffer_bytes = u64::from(extent.width) * u64::from(extent.height) * 4 * 4;
    let tiled_resident = peak as u64 * full_buffer_bytes;
    let all_nodes_resident = plan.nodes.len() as u64 * full_buffer_bytes;
    assert!(
        tiled_resident <= cap_bytes,
        "tiled working set {tiled_resident} exceeds cap {cap_bytes}"
    );
    assert!(
        tiled_resident < all_nodes_resident,
        "tiled working set must be below the all-nodes-resident figure"
    );

    // Every node schedules the full 4K tile set (15x9 = 135 tiles at 256px), and
    // the schedule is demand-driven, not whole-image dispatch.
    assert_eq!(grid.tile_count(), 15 * 9);
    assert_eq!(schedule.demanded_tile_count("a"), 135);
    assert_eq!(schedule.demanded_tile_count("d"), 135);
}
