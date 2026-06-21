//! Tiled-vs-whole differential for the neighbourhood filters `filter.convolve@1`
//! and `filter.gaussian_blur@1` across every boundary mode (`plan.md` §11.3,
//! `AGENT_VERIFICATION` §3.3, §13; bn-3r8).
//!
//! A neighbourhood op cannot be tiled like a pointwise op: an output tile reads a
//! kernel-dilated *halo* of its input. The tile executor crops each input to that
//! halo window, runs the op's whole-image kernel on the window, and keeps only the
//! interior that corresponds to the output tile. This suite pins that the
//! construction is **bit-identical** to the whole-image reference for the exact
//! integer-kernel convolution, and within a tight tolerance for the
//! `Bounded`-tier Gaussian blur, at *every* boundary mode and across tile sizes
//! that produce ragged edge tiles — so a halo off-by-one or a boundary applied at
//! the wrong (tile rather than image) edge would surface as a visible tile grid
//! (`AGENT_VERIFICATION` §13 "tile grid visible").
//!
//! On a mismatch the suite writes a PPM **error map** (per-pixel absolute
//! difference, magnified) next to the test's temp dir and names it in the panic,
//! so the failing seam is inspectable (`bn-3r8` exit gate).

#![allow(
    clippy::unwrap_used,
    clippy::missing_const_for_fn,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::float_cmp,
    clippy::needless_pass_by_value,
    reason = "an integration test crate building exact fixtures from integer indices; \
              the exact convolution path demands bit-identical (==) comparison"
)]

use std::collections::BTreeMap;

use paintop_core::executor::{ResourceValue, analyze_roi, execute};
use paintop_core::tile::{TiledExecution, execute_tiled};
use paintop_cpu::convolve::Convolve;
use paintop_cpu::gaussian_blur::GaussianBlur;
use paintop_ir::{
    AlphaRepresentation, ChannelLayout, ColorEncoding, ColorRange, ContractRegistry,
    CoordinateConvention, Extent, ImageDescriptor, OperationManifest, OperationRegistry, Plan,
    ResourceDescriptor, ScalarType, SemanticRole, check_graph, parse_plan, resolve_plan,
};
use serde_json::json;

const CONVOLVE_OP_ID: &str = "filter.convolve@1";
const BLUR_OP_ID: &str = "filter.gaussian_blur@1";

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

/// The registry of the two neighbourhood ops, built from their real manifests so
/// the test pins the declared ROI category (`Geometric`) and contract the
/// scheduler/executor actually consult.
fn registry() -> OperationRegistry {
    let manifests: Vec<OperationManifest> = vec![
        Convolve::manifest().unwrap(),
        GaussianBlur::manifest().unwrap(),
    ];
    OperationRegistry::from_manifests(manifests).unwrap()
}

fn contracts() -> ContractRegistry {
    let mut c = ContractRegistry::new();
    c.register(CONVOLVE_OP_ID.parse().unwrap(), Box::new(Convolve))
        .unwrap();
    c.register(BLUR_OP_ID.parse().unwrap(), Box::new(GaussianBlur))
        .unwrap();
    c
}

fn implementations() -> paintop_core::executor::ImplRegistry {
    let mut r = paintop_core::executor::ImplRegistry::new();
    r.register(CONVOLVE_OP_ID.parse().unwrap(), Box::new(Convolve))
        .unwrap();
    r.register(BLUR_OP_ID.parse().unwrap(), Box::new(GaussianBlur))
        .unwrap();
    r
}

/// A spatially varying RGBA `f32` source with structure in every dimension, so a
/// tile-seam or halo bug surfaces as a mismatch row/column rather than hiding in a
/// flat field.
fn source(extent: Extent) -> ResourceValue {
    let width = extent.width as usize;
    let height = extent.height as usize;
    let mut samples = vec![0.0_f32; width * height * 4];
    for row in 0..height {
        for col in 0..width {
            let base = (row * width + col) * 4;
            let fx = col as f32;
            let fy = row as f32;
            samples[base] = fy.mul_add(0.005, fx.mul_add(0.017, 0.1));
            samples[base + 1] = fy.mul_add(0.013, 0.2);
            samples[base + 2] = ((col ^ row) as f32 % 11.0).mul_add(0.03, 0.05);
            samples[base + 3] = 1.0;
        }
    }
    ResourceValue::new(image_descriptor(extent), 4, samples).unwrap()
}

/// A single-op plan reading an external `input:src` image.
fn single_op_plan(op: &str, params: serde_json::Value) -> Plan {
    let plan = json!({
        "paintop": "1.0",
        "inputs": {"src": {"kind": "image.file", "path": "src.png"}},
        "nodes": [
            {"id": "f", "op": op, "in": {"input": "input:src"}, "params": params}
        ],
        "exports": {"out": {"resource": "node:f/output", "kind": "image", "path": "o.png"}}
    });
    parse_plan(&plan.to_string()).unwrap()
}

/// The whole-image output of node `f`.
fn whole_image(plan: &Plan, extent: Extent) -> Vec<f32> {
    let reg = registry();
    let graph = resolve_plan(plan, &reg).unwrap();
    let mut inputs: BTreeMap<String, ResourceValue> = BTreeMap::new();
    inputs.insert("src".to_owned(), source(extent));
    let execution = execute(plan, &graph, &reg, &implementations(), &inputs).unwrap();
    execution.output("f", "output").unwrap().samples().to_vec()
}

/// The tiled execution of the plan at `tile_size`.
fn tiled(plan: &Plan, extent: Extent, tile_size: u32) -> TiledExecution {
    let reg = registry();
    let graph = resolve_plan(plan, &reg).unwrap();
    let mut input_descriptors: BTreeMap<String, ResourceDescriptor> = BTreeMap::new();
    input_descriptors.insert("src".to_owned(), image_descriptor(extent));
    let checked = check_graph(plan, &graph, &reg, &contracts(), &input_descriptors).unwrap();
    let roi = analyze_roi(plan, &graph, &checked, &contracts()).unwrap();
    let mut inputs: BTreeMap<String, ResourceValue> = BTreeMap::new();
    inputs.insert("src".to_owned(), source(extent));
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

/// Save a PPM error map (per-pixel max-abs-difference over channels, magnified and
/// clamped to white) for a tiled-vs-whole mismatch, returning the path written.
///
/// Self-contained (PPM needs no encoder dependency); a human can open it to see
/// exactly where the seam is.
fn save_error_map(tag: &str, extent: Extent, whole: &[f32], got: &[f32]) -> std::path::PathBuf {
    let width = extent.width as usize;
    let height = extent.height as usize;
    let mut body = format!("P6\n{width} {height}\n255\n").into_bytes();
    for pixel in 0..(width * height) {
        let mut diff = 0.0_f32;
        for channel in 0..4 {
            let idx = pixel * 4 + channel;
            let expected = whole.get(idx).copied().unwrap_or(0.0);
            let actual = got.get(idx).copied().unwrap_or(0.0);
            diff = diff.max((expected - actual).abs());
        }
        // Magnify so even a small error is visible; clamp to [0, 255].
        let intensity = (diff * 4096.0).min(255.0) as u8;
        body.extend_from_slice(&[intensity, 0, 0]);
    }
    let path = std::env::temp_dir().join(format!("paintop-tile-error-{tag}.ppm"));
    let _ = std::fs::write(&path, body);
    path
}

/// Assert `got` equals `whole` exactly; on mismatch, write an error map and panic
/// naming it and the first differing pixel.
fn assert_exact(tag: &str, extent: Extent, whole: &[f32], got: &[f32]) {
    if got == whole {
        return;
    }
    let first = whole.iter().zip(got).position(|(a, b)| a != b).unwrap_or(0);
    let path = save_error_map(tag, extent, whole, got);
    panic!(
        "tiled != whole for {tag}: first diff at sample {first} \
         (whole={}, got={}); error map at {}",
        whole[first],
        got[first],
        path.display()
    );
}

/// Assert `got` is within `tol` of `whole` per sample; on a violation, write an
/// error map and panic naming it.
fn assert_within(tag: &str, extent: Extent, whole: &[f32], got: &[f32], tol: f32) {
    let worst = whole
        .iter()
        .zip(got)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    if worst <= tol {
        return;
    }
    let path = save_error_map(tag, extent, whole, got);
    panic!(
        "tiled differs from whole for {tag} beyond tol {tol}: worst |diff| {worst}; \
         error map at {}",
        path.display()
    );
}

/// The six convolution boundary modes (`valid` shrinks the output extent).
const MODES: [&str; 6] = [
    "constant",
    "transparent",
    "clamp",
    "mirror",
    "wrap",
    "valid",
];

/// A 3x3 asymmetric sharpen-ish kernel: integer weights so the convolution sums
/// are exact in `f64` and the tiled and whole-image runs must agree bit-for-bit.
fn integer_kernel_3x3() -> serde_json::Value {
    json!({
        "width": 3, "height": 3, "origin_x": 1, "origin_y": 1,
        "weights": [0.0, -1.0, 0.0, -1.0, 5.0, -1.0, 0.0, -1.0, 0.0]
    })
}

#[test]
fn tiled_convolution_is_bit_identical_across_boundary_modes() {
    // An extent that does not divide the tile sizes, so every interior+edge tile
    // exercises the halo crop with ragged right/bottom tiles.
    let extent = Extent::new(70, 53);
    for mode in MODES {
        let plan = single_op_plan(
            CONVOLVE_OP_ID,
            json!({"kernel": integer_kernel_3x3(), "mode": mode}),
        );
        let whole = whole_image(&plan, extent);
        // The output extent under `valid` shrinks by (kernel-1); report it so the
        // error map and the exact comparison use the right geometry.
        let out_extent = if mode == "valid" {
            Extent::new(extent.width - 2, extent.height - 2)
        } else {
            extent
        };
        for tile_size in [8, 16, 17, 32, 64] {
            let exec = tiled(&plan, extent, tile_size);
            let got = exec.output("f", "output").unwrap().samples();
            assert_exact(
                &format!("convolve-{mode}-{tile_size}"),
                out_extent,
                &whole,
                got,
            );
            // The export must equal the whole-image final output too.
            assert_exact(
                &format!("convolve-export-{mode}-{tile_size}"),
                out_extent,
                &whole,
                exec.exports()[0].1.samples(),
            );
        }
    }
}

#[test]
fn integer_kernel_larger_than_a_tile_still_tiles_exactly() {
    // A 5x5 kernel with a halo of 2 on a 16px tile: the halo spans into the
    // neighbouring tiles, so a wrong halo would corrupt the tile interior.
    let extent = Extent::new(48, 40);
    let kernel = json!({
        "width": 5, "height": 5, "origin_x": 2, "origin_y": 2,
        "weights": (0..25).map(|i| f64::from(i % 3 - 1)).collect::<Vec<_>>()
    });
    for mode in MODES {
        let plan = single_op_plan(CONVOLVE_OP_ID, json!({"kernel": kernel, "mode": mode}));
        let whole = whole_image(&plan, extent);
        let out_extent = if mode == "valid" {
            Extent::new(extent.width - 4, extent.height - 4)
        } else {
            extent
        };
        for tile_size in [8, 16, 24] {
            let exec = tiled(&plan, extent, tile_size);
            let got = exec.output("f", "output").unwrap().samples();
            assert_exact(
                &format!("convolve5-{mode}-{tile_size}"),
                out_extent,
                &whole,
                got,
            );
        }
    }
}

#[test]
fn tiled_gaussian_blur_matches_whole_image_within_tolerance() {
    // The blur is `Bounded` tier; the reference kernel is the same fixed-order
    // accumulation in both runs, so they actually agree bit-for-bit, but we assert
    // a tight tolerance to honour the declared tier.
    let extent = Extent::new(64, 48);
    // sigma 1.5 => radius ceil(4.5)=5, a 11x11 kernel; clamp/mirror/wrap/constant.
    for mode in ["clamp", "mirror", "wrap", "constant", "transparent"] {
        let plan = single_op_plan(BLUR_OP_ID, json!({"sigma": 1.5, "mode": mode}));
        let whole = whole_image(&plan, extent);
        for tile_size in [8, 16, 32] {
            let exec = tiled(&plan, extent, tile_size);
            let got = exec.output("f", "output").unwrap().samples();
            assert_within(
                &format!("blur-{mode}-{tile_size}"),
                extent,
                &whole,
                got,
                1.0e-6,
            );
        }
    }
}

#[test]
fn small_image_with_large_kernel_clamps_halo_to_extent() {
    // The kernel halo exceeds the whole image on every side, so every tile's halo
    // window clamps to the full extent and the op runs effectively whole-image per
    // tile; the result must still equal the reference.
    let extent = Extent::new(12, 9);
    let plan = single_op_plan(BLUR_OP_ID, json!({"sigma": 3.0, "mode": "clamp"}));
    let whole = whole_image(&plan, extent);
    for tile_size in [4, 8] {
        let exec = tiled(&plan, extent, tile_size);
        let got = exec.output("f", "output").unwrap().samples();
        assert_within(
            &format!("blur-small-{tile_size}"),
            extent,
            &whole,
            got,
            1.0e-6,
        );
    }
}
