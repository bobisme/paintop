//! Verification suite for `image.pad@1` (`OP_CATALOG` §5, `IR_SPEC` §8.4):
//!
//! - **schema/contract**: the manifest validates, agrees with its contract, gates
//!   clean, and the checked-in JSON matches the builder;
//! - **analytic fixtures**: each boundary mode (constant/clamp/mirror/wrap)
//!   reproduces a known border on a tiny image;
//! - **property**: a zero-margin pad is the exact identity; the interior of any
//!   pad is the verbatim input; crop ∘ pad round-trips on the interior;
//! - **metamorphic**: an all-negative-margin pad equals the equivalent
//!   `image.crop`;
//! - **rejection**: a margin that over-removes an axis and a mismatched `value`
//!   length are rejected.

use paintop_core::executor::{InputValues, OpImplementation, ResourceValue};
use paintop_ir::{
    AlphaRepresentation, ChannelLayout, ColorEncoding, ColorRange, CoordinateConvention,
    Descriptors, ErrorClass, Extent, ImageDescriptor, OpContract, OutputRegions, Rect,
    ResourceDescriptor, ScalarType, SemanticRole,
};

use super::{E_PAD_EXTENT, E_PAD_VALUE, PAD_OP_ID, Pad};
use crate::crop::Crop;

/// Build a single-channel (gray) image from a row-major sample list.
fn gray(width: u32, height: u32, samples: Vec<f32>) -> ResourceValue {
    image_value(width, height, ChannelLayout::Gray, samples)
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

/// Run the pad kernel and recover the output image.
fn pad(value: &ResourceValue, params: &serde_json::Value) -> ResourceValue {
    let mut inputs = InputValues::new();
    inputs.insert("image".to_owned(), value.clone());
    let mut out = Pad::new().compute(&inputs, params).expect("pad computes");
    out.remove("image").expect("image port produced")
}

#[test]
fn manifest_validates_and_agrees_with_contract() {
    let manifest = Pad::manifest().expect("pad manifest");
    manifest.validate().expect("pad manifest valid");
    paintop_ir::check_contract_consistency(&manifest, &Pad::new())
        .expect("pad manifest agrees with contract");
    paintop_ir::verify_categories(&manifest, &manifest.test.verification)
        .expect("pad verification declarations gate clean");
    assert_eq!(manifest.id.to_string(), PAD_OP_ID);
}

#[test]
fn constant_mode_fills_border_with_value() {
    // A 1x1 image [5]; pad 1 each side with constant 9 => 3x3 with 5 in the center.
    let img = gray(1, 1, vec![5.0]);
    let out = pad(
        &img,
        &serde_json::json!({ "left": 1, "right": 1, "top": 1, "bottom": 1, "mode": "constant", "value": [9.0] }),
    );
    assert_eq!(out.extent(), Extent::new(3, 3));
    let want = vec![
        9.0, 9.0, 9.0, //
        9.0, 5.0, 9.0, //
        9.0, 9.0, 9.0,
    ];
    assert_eq!(out.samples(), want.as_slice());
}

#[test]
fn constant_mode_defaults_to_zero() {
    let img = gray(1, 1, vec![5.0]);
    let out = pad(&img, &serde_json::json!({ "left": 1 }));
    assert_eq!(out.extent(), Extent::new(2, 1));
    assert_eq!(out.samples(), [0.0, 5.0].as_slice());
}

#[test]
fn clamp_mode_replicates_edge() {
    // Row [1, 2, 3]; pad left=2, right=2 clamp => [1,1, 1,2,3, 3,3].
    let img = gray(3, 1, vec![1.0, 2.0, 3.0]);
    let out = pad(
        &img,
        &serde_json::json!({ "left": 2, "right": 2, "mode": "clamp" }),
    );
    assert_eq!(out.extent(), Extent::new(7, 1));
    assert_eq!(
        out.samples(),
        [1.0, 1.0, 1.0, 2.0, 3.0, 3.0, 3.0].as_slice()
    );
}

#[test]
fn mirror_mode_reflects_without_repeating_edge() {
    // Row [1, 2, 3, 4]; half-sample mirror. Left pad 3, right pad 3.
    // Left neighbours of index 0: 2(-1), 3(-2), 4(-3) => [4,3,2 | 1,2,3,4 | 3,2,1]
    let img = gray(4, 1, vec![1.0, 2.0, 3.0, 4.0]);
    let out = pad(
        &img,
        &serde_json::json!({ "left": 3, "right": 3, "mode": "mirror" }),
    );
    assert_eq!(out.extent(), Extent::new(10, 1));
    assert_eq!(
        out.samples(),
        [4.0, 3.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 2.0, 1.0].as_slice()
    );
}

#[test]
fn wrap_mode_tiles_periodically() {
    // Row [1, 2, 3]; pad left=2 right=2 wrap => [2,3, 1,2,3, 1,2].
    let img = gray(3, 1, vec![1.0, 2.0, 3.0]);
    let out = pad(
        &img,
        &serde_json::json!({ "left": 2, "right": 2, "mode": "wrap" }),
    );
    assert_eq!(out.extent(), Extent::new(7, 1));
    assert_eq!(
        out.samples(),
        [2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0].as_slice()
    );
}

#[test]
fn zero_margin_pad_is_exact_identity() {
    let img = gray(3, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let out = pad(&img, &serde_json::json!({}));
    assert_eq!(out.extent(), Extent::new(3, 2));
    assert_eq!(out.samples(), img.samples());
}

#[test]
fn interior_is_verbatim_for_every_mode() {
    let img = gray(3, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    for mode in ["constant", "clamp", "mirror", "wrap"] {
        let out = pad(
            &img,
            &serde_json::json!({ "left": 2, "right": 1, "top": 1, "bottom": 2, "mode": mode }),
        );
        assert_eq!(out.extent(), Extent::new(6, 5));
        // The interior block [2..5) x [1..3) must equal the verbatim input.
        let ow = 6usize;
        for y in 0..2usize {
            for x in 0..3usize {
                let got = out.samples()[(y + 1) * ow + (x + 2)];
                let want = img.samples()[y * 3 + x];
                assert!(
                    (got - want).abs() < f32::EPSILON,
                    "interior mismatch at ({x},{y}) mode {mode}: {got} != {want}"
                );
            }
        }
    }
}

#[test]
fn crop_of_pad_round_trips_on_interior() {
    // pad then crop back the interior reproduces the input exactly, for clamp.
    let img = gray(4, 3, (0..12u8).map(f32::from).collect());
    let padded = pad(
        &img,
        &serde_json::json!({ "left": 2, "right": 3, "top": 1, "bottom": 2, "mode": "clamp" }),
    );
    // Interior is at origin (left, top) = (2, 1), size 4x3.
    let mut inputs = InputValues::new();
    inputs.insert("image".to_owned(), padded);
    let params = serde_json::json!({ "rect": { "x0": 2, "y0": 1, "x1": 6, "y1": 4 } });
    let mut out = Crop::new()
        .compute(&inputs, &params)
        .expect("crop computes");
    let recovered = out.remove("image").expect("image");
    assert_eq!(recovered.extent(), Extent::new(4, 3));
    assert_eq!(recovered.samples(), img.samples());
}

#[test]
fn all_negative_margins_equal_crop() {
    // pad with negative margins removes rows/cols, equalling image.crop of the
    // shrunken interior.
    let img = gray(5, 4, (0..20u8).map(f32::from).collect());
    let padded = pad(
        &img,
        &serde_json::json!({ "left": -1, "right": -1, "top": -1, "bottom": 0 }),
    );
    // Removing 1 each from L/R and 1 from top => crop [1,4) x [1,4) (3x3).
    let mut inputs = InputValues::new();
    inputs.insert("image".to_owned(), img);
    let params = serde_json::json!({ "rect": { "x0": 1, "y0": 1, "x1": 4, "y1": 4 } });
    let mut out = Crop::new()
        .compute(&inputs, &params)
        .expect("crop computes");
    let cropped = out.remove("image").expect("image");
    assert_eq!(padded.extent(), cropped.extent());
    assert_eq!(padded.samples(), cropped.samples());
}

#[test]
fn over_removing_axis_is_rejected() {
    let img = gray(3, 3, vec![0.0; 9]);
    let mut inputs = InputValues::new();
    inputs.insert("image".to_owned(), img);
    let params = serde_json::json!({ "left": -2, "right": -2 });
    let err = Pad::new()
        .compute(&inputs, &params)
        .expect_err("over-removal must be rejected");
    assert_eq!(err.class, ErrorClass::Policy);
    assert_eq!(err.code, E_PAD_EXTENT);
}

#[test]
fn mismatched_value_length_is_rejected() {
    let img = image_value(1, 1, ChannelLayout::Rgb, vec![0.1, 0.2, 0.3]);
    let mut inputs = InputValues::new();
    inputs.insert("image".to_owned(), img);
    let params = serde_json::json!({ "left": 1, "value": [0.5, 0.5] });
    let err = Pad::new()
        .compute(&inputs, &params)
        .expect_err("wrong value length must be rejected");
    assert_eq!(err.class, ErrorClass::Schema);
    assert_eq!(err.code, E_PAD_VALUE);
}

#[test]
fn infer_outputs_grows_extent_and_preserves_descriptor() {
    let descriptor = ResourceDescriptor::Image(ImageDescriptor {
        extent: Extent::new(10, 8),
        layout: ChannelLayout::Rgba,
        scalar: ScalarType::F32,
        color: ColorEncoding::Srgb,
        range: ColorRange::DisplayReferred,
        alpha: AlphaRepresentation::Straight,
        coordinates: CoordinateConvention::PixelCenterUpperLeft,
        semantic: SemanticRole::Color,
    });
    let mut inputs = Descriptors::new();
    inputs.insert("image".to_owned(), descriptor);
    let params = serde_json::json!({ "left": 1, "right": 2, "top": 3, "bottom": 4 });

    let out = Pad::new().infer_outputs(&inputs, &params).expect("infer");
    let ResourceDescriptor::Image(d) = out["image"] else {
        panic!("expected image");
    };
    assert_eq!(d.extent, Extent::new(13, 15));
    assert_eq!(d.layout, ChannelLayout::Rgba);

    // Constant-mode interior ROI maps the requested window back by -lead, clamped.
    let mut requested = OutputRegions::new();
    requested.insert("image".to_owned(), Rect::new(1, 3, 3, 5));
    let needed = Pad::new()
        .required_inputs(&requested, &inputs, &params)
        .expect("required_inputs");
    // (1-1, 3-3) .. (3-1, 5-3) = (0,0)..(2,2), clamped to the 10x8 input.
    assert_eq!(needed["image"], Rect::new(0, 0, 2, 2));
}

#[test]
fn checked_in_manifest_matches_builder() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("crates dir")
        .parent()
        .expect("repo root")
        .join("ops/manifests");
    let manifest = Pad::manifest().expect("pad manifest");
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
