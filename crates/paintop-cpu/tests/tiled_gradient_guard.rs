//! Tile-executor guard for **position-dependent pointwise** ops (bn-3ai).
//!
//! `paint.linear_gradient@1` / `paint.radial_gradient@1` declare
//! [`RoiCategory::Pointwise`](paintop_ir::RoiCategory::Pointwise) but are
//! *position-dependent generators*: every output pixel is a function of its
//! **absolute** coordinate, and the op reads *no* input samples (only the
//! `extent_from` size). The naive `tileable()` predicate — "tile any Pointwise op
//! with ≥1 input" — would crop the (unread) input to each tile and run the kernel
//! with a shifted tile origin, silently restarting the gradient per tile and
//! producing a wrong image with no panic.
//!
//! The bn-3ai fix adds a position-independence guard: a pointwise op is tiled only
//! when its contract demands a **non-empty, co-located** input region (a true
//! `output(R) = f(input(R))` transform). A generator demands the empty region from
//! every input, so it falls back to a single whole-image dispatch.
//!
//! This suite pins the guard end-to-end on the real gradient op:
//!
//! * `tiled_gradient_equals_whole_image` — the gradient node's tiled output is
//!   **bit-identical** to the whole-image executor across tile sizes that divide
//!   the extent and ragged ones (a per-tile-restart bug would diverge);
//! * `gradient_is_not_tiled` — the gradient runs whole-image (it contributes no
//!   per-tile work to the tile stats), confirming the guard took the fallback path
//!   rather than tiling it correctly by luck;
//! * `pointwise_consumer_downstream_of_a_gradient_still_tiles` — a genuine
//!   position-independent pointwise op (`color.adjust@1`) fed by the gradient is
//!   still tiled, so the guard does not over-restrict real pointwise work.

#![allow(
    clippy::unwrap_used,
    clippy::missing_const_for_fn,
    reason = "an integration test crate exercising the real gradient op end-to-end"
)]

use std::collections::BTreeMap;

use paintop_core::executor::{ImplRegistry, ResourceValue, analyze_roi, execute};
use paintop_core::tile::{TiledExecution, execute_tiled};
use paintop_cpu::adjust::Adjust;
use paintop_cpu::gradient::LinearGradient;
use paintop_ir::{
    AlphaRepresentation, ChannelLayout, ColorEncoding, ColorRange, ContractRegistry,
    CoordinateConvention, DeterminismTier, Extent, ImageDescriptor, InputSpec, OperationManifest,
    OperationRegistry, OutputSpec, Plan, ResourceDescriptor, ResourceKind, RoiCategory, RoiPolicy,
    ScalarType, SemanticRole, TestMetadata, check_graph, parse_plan, resolve_plan,
};

const GRADIENT_OP_ID: &str = "paint.linear_gradient@1";
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

/// The real `paint.linear_gradient@1` manifest (the source of truth builder).
fn gradient_manifest() -> OperationManifest {
    LinearGradient::manifest().unwrap()
}

/// A minimal `color.adjust@1` manifest, so a genuine pointwise op can sit
/// downstream of the gradient in the consumer test.
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

fn registry(with_adjust: bool) -> OperationRegistry {
    if with_adjust {
        OperationRegistry::from_manifests([gradient_manifest(), adjust_manifest()]).unwrap()
    } else {
        OperationRegistry::from_manifests([gradient_manifest()]).unwrap()
    }
}

fn contracts(with_adjust: bool) -> ContractRegistry {
    let mut c = ContractRegistry::new();
    c.register(GRADIENT_OP_ID.parse().unwrap(), Box::new(LinearGradient))
        .unwrap();
    if with_adjust {
        c.register(ADJUST_OP_ID.parse().unwrap(), Box::new(Adjust))
            .unwrap();
    }
    c
}

fn implementations(with_adjust: bool) -> ImplRegistry {
    let mut r = ImplRegistry::new();
    r.register(GRADIENT_OP_ID.parse().unwrap(), Box::new(LinearGradient))
        .unwrap();
    if with_adjust {
        r.register(ADJUST_OP_ID.parse().unwrap(), Box::new(Adjust))
            .unwrap();
    }
    r
}

/// A flat opaque RGBA `f32` source whose only role is to give the gradient its
/// extent (the gradient ignores the input samples entirely).
fn extent_source(extent: Extent) -> ResourceValue {
    let len = (extent.width as usize) * (extent.height as usize) * 4;
    ResourceValue::new(image_descriptor(extent), 4, vec![1.0; len]).unwrap()
}

/// A single diagonal linear gradient reading `input:src` for its extent. The
/// diagonal axis makes the gradient vary in **both** x and y, so a per-tile
/// coordinate restart would diverge on every interior tile boundary.
fn gradient_plan() -> Plan {
    parse_plan(
        r#"{
            "paintop": "1.0",
            "inputs": {"src": {"kind": "image.file", "path": "src.png"}},
            "nodes": [
                {"id": "g", "op": "paint.linear_gradient@1", "in": {"extent_from": "input:src"},
                 "params": {
                    "start_px": [0.0, 0.0], "end_px": [200.0, 150.0],
                    "color": "linear-srgb", "alpha": "premultiplied",
                    "stops": [
                        {"position": 0.0, "color": [0.0, 0.0, 0.0, 1.0]},
                        {"position": 1.0, "color": [1.0, 0.8, 0.2, 1.0]}
                    ]
                 }}
            ],
            "exports": {"out": {"resource": "node:g/image", "kind": "image", "path": "o.png"}}
        }"#,
    )
    .unwrap()
}

/// A gradient feeding a genuine pointwise `color.adjust@1`, so the consumer must
/// still be tiled even though its producer is not.
fn gradient_then_adjust_plan() -> Plan {
    parse_plan(
        r#"{
            "paintop": "1.0",
            "inputs": {"src": {"kind": "image.file", "path": "src.png"}},
            "nodes": [
                {"id": "g", "op": "paint.linear_gradient@1", "in": {"extent_from": "input:src"},
                 "params": {
                    "start_px": [0.0, 0.0], "end_px": [200.0, 150.0],
                    "color": "linear-srgb", "alpha": "premultiplied",
                    "stops": [
                        {"position": 0.0, "color": [0.0, 0.0, 0.0, 1.0]},
                        {"position": 1.0, "color": [1.0, 0.8, 0.2, 1.0]}
                    ]
                 }},
                {"id": "a", "op": "color.adjust@1", "in": {"image": "node:g/image"},
                 "params": {"exposure_ev": -0.2, "saturation": 0.1}}
            ],
            "exports": {"out": {"resource": "node:a/image", "kind": "image", "path": "o.png"}}
        }"#,
    )
    .unwrap()
}

/// The whole-image node outputs of `plan`, keyed by node id.
fn whole_image(plan: &Plan, extent: Extent, with_adjust: bool) -> BTreeMap<String, Vec<f32>> {
    let reg = registry(with_adjust);
    let graph = resolve_plan(plan, &reg).unwrap();
    let mut inputs: BTreeMap<String, ResourceValue> = BTreeMap::new();
    inputs.insert("src".to_owned(), extent_source(extent));
    let execution = execute(plan, &graph, &reg, &implementations(with_adjust), &inputs).unwrap();
    let mut out = BTreeMap::new();
    for node in &plan.nodes {
        if let Some(value) = execution.output(&node.id, "image") {
            out.insert(node.id.clone(), value.samples().to_vec());
        }
    }
    out
}

/// The tiled execution of `plan` at `tile_size`.
fn tiled(plan: &Plan, extent: Extent, tile_size: u32, with_adjust: bool) -> TiledExecution {
    let reg = registry(with_adjust);
    let graph = resolve_plan(plan, &reg).unwrap();
    let mut input_descriptors: BTreeMap<String, ResourceDescriptor> = BTreeMap::new();
    input_descriptors.insert("src".to_owned(), image_descriptor(extent));
    let checked = check_graph(
        plan,
        &graph,
        &reg,
        &contracts(with_adjust),
        &input_descriptors,
    )
    .unwrap();
    let roi = analyze_roi(plan, &graph, &checked, &contracts(with_adjust)).unwrap();
    let mut inputs: BTreeMap<String, ResourceValue> = BTreeMap::new();
    inputs.insert("src".to_owned(), extent_source(extent));
    execute_tiled(
        plan,
        &graph,
        &checked,
        &reg,
        &contracts(with_adjust),
        &implementations(with_adjust),
        &roi,
        &inputs,
        tile_size,
    )
    .unwrap()
}

/// The guard's core promise: tiling a gradient equals the whole-image gradient,
/// **bit-for-bit**, across tile sizes that divide the 200×150 extent and ones that
/// leave ragged edge tiles. Without the position-independence guard the gradient
/// would be tiled and each tile would restart at its own origin, diverging on
/// every interior tile.
#[test]
fn tiled_gradient_equals_whole_image() {
    let extent = Extent::new(200, 150);
    let plan = gradient_plan();
    let whole = whole_image(&plan, extent, false);
    for tile_size in [32, 50, 64, 75, 128, 256] {
        let exec = tiled(&plan, extent, tile_size, false);
        let got = exec.output("g", "image").unwrap().samples();
        assert_eq!(
            got,
            whole["g"].as_slice(),
            "tiled gradient diverges from whole-image at tile_size {tile_size}"
        );
        // The export matches too.
        assert_eq!(exec.exports()[0].1.samples(), whole["g"].as_slice());
    }
}

/// The gradient is **not tiled**: it runs as a single whole-image dispatch (the
/// guard's fallback). A whole-image fallback contributes no per-tile work, so the
/// tile stats stay at zero requested/executed/identity for a lone gradient plan —
/// proving the bit-identical result above came from the fallback, not from
/// accidentally-correct tiling.
#[test]
fn gradient_is_not_tiled() {
    let extent = Extent::new(200, 150);
    let plan = gradient_plan();
    // A small tile size would yield many tiles *if* the gradient were tiled.
    let exec = tiled(&plan, extent, 32, false);
    let stats = exec.stats();
    assert_eq!(
        (stats.requested, stats.executed, stats.identity),
        (0, 0, 0),
        "a position-dependent gradient must run whole-image, contributing no tiles"
    );
}

/// The guard does not over-restrict: a genuine pointwise consumer downstream of the
/// gradient is still tiled (its tiles show up in the stats) and matches whole-image.
#[test]
fn pointwise_consumer_downstream_of_a_gradient_still_tiles() {
    let extent = Extent::new(128, 96);
    let plan = gradient_then_adjust_plan();
    let whole = whole_image(&plan, extent, true);
    let exec = tiled(&plan, extent, 64, true);
    // Both nodes match whole-image bit-for-bit.
    for node in ["g", "a"] {
        assert_eq!(
            exec.output(node, "image").unwrap().samples(),
            whole[node].as_slice(),
            "node {node} diverges from whole-image"
        );
    }
    // The adjust node IS tiled: 128x96 / 64 => 2x2 = 4 tiles. The gradient is not,
    // so the total executed tile count is exactly the adjust node's four.
    let stats = exec.stats();
    assert_eq!(stats.executed, 4, "only the pointwise consumer should tile");
    assert_eq!(stats.requested, 4);
}
