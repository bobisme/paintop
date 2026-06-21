//! Verification suite for `image.rotate90@1` (`OP_CATALOG` §5,
//! `AGENT_VERIFICATION` §2.5):
//!
//! - **schema/contract**: the manifest validates, agrees with its contract, gates
//!   clean, and the checked-in JSON matches the builder;
//! - **analytic fixtures**: each turn remaps pixels to the exact rotated position
//!   and transposes the extent for odd turns;
//! - **property**: `turns` is taken modulo 4 (0 and 4 are the identity);
//! - **metamorphic**: four turns is the identity, two turns equals a `both` flip,
//!   and CW∘CCW is the identity (via the shared `metamorphic` harness);
//! - **rejection**: a missing / non-integer `turns` is rejected.

use paintop_core::executor::{InputValues, OpImplementation, ResourceValue};
use paintop_ir::{
    AlphaRepresentation, ChannelLayout, ColorEncoding, ColorRange, CoordinateConvention,
    Descriptors, ErrorClass, Extent, ImageDescriptor, OpContract, ResourceDescriptor, ScalarType,
    SemanticRole,
};
use paintop_testkit::metamorphic::{assert_periodic_identity, samples_bit_identical};

use super::{E_ROTATE_TURNS, ROTATE90_OP_ID, Rotate90};
use crate::flip::Flip;

/// Build an RGBA ramp whose samples encode `(x, y, channel)` positionally.
fn ramp_image(width: u32, height: u32) -> ResourceValue {
    let mut samples = Vec::new();
    for y in 0..height {
        for x in 0..width {
            for c in 0..4u32 {
                #[allow(clippy::cast_precision_loss, reason = "small test extents")]
                samples.push(((y * width + x) * 4 + c) as f32);
            }
        }
    }
    image_value(width, height, ChannelLayout::Rgba, samples)
}

/// Build an image [`ResourceValue`] with the given layout/samples.
fn image_value(width: u32, height: u32, layout: ChannelLayout, samples: Vec<f32>) -> ResourceValue {
    let descriptor = ResourceDescriptor::Image(ImageDescriptor {
        extent: Extent::new(width, height),
        layout,
        scalar: ScalarType::F32,
        color: ColorEncoding::Srgb,
        range: ColorRange::DisplayReferred,
        alpha: AlphaRepresentation::Straight,
        coordinates: CoordinateConvention::PixelCenterUpperLeft,
        semantic: SemanticRole::Color,
    });
    ResourceValue::new(descriptor, layout.channel_count(), samples)
        .expect("sample buffer matches descriptor")
}

/// Run the rotate kernel for `turns` and recover the output image.
fn rotate(value: &ResourceValue, turns: i64) -> ResourceValue {
    let mut inputs = InputValues::new();
    inputs.insert("image".to_owned(), value.clone());
    let params = serde_json::json!({ "turns": turns });
    let mut out = Rotate90::new()
        .compute(&inputs, &params)
        .expect("rotate computes");
    out.remove("image").expect("image port produced")
}

/// Run a flip for `axis`.
fn flip(value: &ResourceValue, axis: &str) -> ResourceValue {
    let mut inputs = InputValues::new();
    inputs.insert("image".to_owned(), value.clone());
    let params = serde_json::json!({ "axis": axis });
    let mut out = Flip::new()
        .compute(&inputs, &params)
        .expect("flip computes");
    out.remove("image").expect("image")
}

/// The base sample index of pixel `(x, y)` in a width-`w` RGBA ramp.
#[allow(clippy::cast_precision_loss, reason = "tiny indices")]
fn ramp_base(x: u32, y: u32, w: u32) -> f32 {
    ((y * w + x) * 4) as f32
}

#[test]
fn manifest_validates_and_agrees_with_contract() {
    let manifest = Rotate90::manifest().expect("rotate manifest");
    manifest.validate().expect("rotate manifest valid");
    paintop_ir::check_contract_consistency(&manifest, &Rotate90::new())
        .expect("rotate manifest agrees with contract");
    paintop_ir::verify_categories(&manifest, &manifest.test.verification)
        .expect("rotate verification declarations gate clean");
    assert_eq!(manifest.id.to_string(), ROTATE90_OP_ID);
}

#[test]
fn one_turn_cw_transposes_and_maps_pixels() {
    // 3x2 input. 90 CW => 2x3 output, out(x, y) = in(y, H-1-x) with H = 2.
    let img = ramp_image(3, 2);
    let out = rotate(&img, 1);
    assert_eq!(out.extent(), Extent::new(2, 3));
    let ow = 2u32;
    for y in 0..3 {
        for x in 0..2 {
            let got = out.samples()[((y * ow + x) * 4) as usize];
            // source (sx, sy) = (y, H-1-x) = (y, 1-x)
            assert!(
                (got - ramp_base(y, 1 - x, 3)).abs() < f32::EPSILON,
                "1 turn mismatch at ({x},{y}): {got}"
            );
        }
    }
}

#[test]
fn two_turns_equals_both_flip() {
    let img = ramp_image(4, 3);
    let rotated = rotate(&img, 2);
    let flipped = flip(&img, "both");
    assert_eq!(rotated.extent(), Extent::new(4, 3));
    assert!(samples_bit_identical(&rotated, &flipped));
}

#[test]
fn four_turns_is_identity() {
    // Periodic identity (§2.5), via the shared metamorphic harness.
    let img = ramp_image(5, 3);
    assert_periodic_identity(&img, 4, |v| rotate(v, 1));
}

#[test]
fn turns_taken_modulo_four() {
    let img = ramp_image(4, 3);
    // 0, 4, 8, -4 are all the identity.
    for k in [0, 4, 8, -4] {
        let out = rotate(&img, k);
        assert!(
            samples_bit_identical(&out, &img),
            "turns {k} should be the identity"
        );
    }
    // -1 (one CCW) equals 3 (three CW).
    let ccw = rotate(&img, -1);
    let cw3 = rotate(&img, 3);
    assert!(samples_bit_identical(&ccw, &cw3));
}

#[test]
fn cw_then_ccw_is_identity() {
    let img = ramp_image(5, 4);
    let round = rotate(&rotate(&img, 1), -1);
    assert!(samples_bit_identical(&round, &img));
}

#[test]
fn missing_turns_is_rejected() {
    let img = ramp_image(2, 2);
    let mut inputs = InputValues::new();
    inputs.insert("image".to_owned(), img);
    let err = Rotate90::new()
        .compute(&inputs, &serde_json::json!({}))
        .expect_err("missing turns must be rejected");
    assert_eq!(err.class, ErrorClass::Schema);
    assert_eq!(err.code, E_ROTATE_TURNS);
}

#[test]
fn non_integer_turns_is_rejected() {
    let img = ramp_image(2, 2);
    let mut inputs = InputValues::new();
    inputs.insert("image".to_owned(), img);
    let err = Rotate90::new()
        .compute(&inputs, &serde_json::json!({ "turns": "quarter" }))
        .expect_err("non-integer turns must be rejected");
    assert_eq!(err.class, ErrorClass::Schema);
    assert_eq!(err.code, E_ROTATE_TURNS);
}

#[test]
fn infer_outputs_transposes_extent_for_odd_turns() {
    let descriptor = ResourceDescriptor::Image(ImageDescriptor {
        extent: Extent::new(10, 8),
        layout: ChannelLayout::Rgb,
        scalar: ScalarType::F32,
        color: ColorEncoding::Srgb,
        range: ColorRange::DisplayReferred,
        alpha: AlphaRepresentation::Straight,
        coordinates: CoordinateConvention::PixelCenterUpperLeft,
        semantic: SemanticRole::Color,
    });
    let mut inputs = Descriptors::new();
    inputs.insert("image".to_owned(), descriptor);

    let out = Rotate90::new()
        .infer_outputs(&inputs, &serde_json::json!({ "turns": 1 }))
        .expect("infer");
    let ResourceDescriptor::Image(d) = out["image"] else {
        panic!("expected image");
    };
    assert_eq!(d.extent, Extent::new(8, 10));

    let out2 = Rotate90::new()
        .infer_outputs(&inputs, &serde_json::json!({ "turns": 2 }))
        .expect("infer");
    let ResourceDescriptor::Image(d2) = out2["image"] else {
        panic!("expected image");
    };
    assert_eq!(d2.extent, Extent::new(10, 8));
}

#[test]
fn checked_in_manifest_matches_builder() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("crates dir")
        .parent()
        .expect("repo root")
        .join("ops/manifests");
    let manifest = Rotate90::manifest().expect("rotate manifest");
    let path = root.join(format!("{}.json", manifest.id));
    let on_disk =
        std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    let expected = serde_json::to_string_pretty(&manifest).expect("serialize manifest");
    assert_eq!(
        on_disk.trim_end(),
        expected.trim_end(),
        "{} is stale; regenerate from the Rust builder",
        path.display()
    );
}
