//! Verification suite for `image.flip@1` (`OP_CATALOG` §5,
//! `AGENT_VERIFICATION` §2.5):
//!
//! - **schema/contract**: the manifest validates, agrees with its contract, gates
//!   clean, and the checked-in JSON matches the builder;
//! - **analytic fixtures**: each axis remaps pixels to the exact mirror position;
//! - **property**: extent is preserved; the remap is a verbatim bijection;
//! - **metamorphic**: every flip is an involution (double-flip identity, via the
//!   shared `metamorphic` harness); horizontal∘vertical == both;
//! - **rejection**: a missing / unknown axis is rejected.

use paintop_core::executor::{InputValues, OpImplementation, ResourceValue};
use paintop_ir::{
    AlphaRepresentation, ChannelLayout, ColorEncoding, ColorRange, CoordinateConvention,
    Descriptors, ErrorClass, Extent, ImageDescriptor, OpContract, OutputRegions, Rect,
    ResourceDescriptor, ScalarType, SemanticRole,
};
use paintop_testkit::metamorphic::{assert_involution, samples_bit_identical};

use super::{E_FLIP_AXIS, FLIP_OP_ID, Flip};

/// Build an RGBA ramp whose samples encode `(x, y, channel)` positionally:
/// `sample(x, y, c) = (y * width + x) * 4 + c`.
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

/// Run the flip kernel for `axis` and recover the output image.
fn flip(value: &ResourceValue, axis: &str) -> ResourceValue {
    let mut inputs = InputValues::new();
    inputs.insert("image".to_owned(), value.clone());
    let params = serde_json::json!({ "axis": axis });
    let mut out = Flip::new()
        .compute(&inputs, &params)
        .expect("flip computes");
    out.remove("image").expect("image port produced")
}

/// The base sample index of pixel `(x, y)` in a width-`w` RGBA ramp.
#[allow(clippy::cast_precision_loss, reason = "tiny indices")]
fn ramp_base(x: u32, y: u32, w: u32) -> f32 {
    ((y * w + x) * 4) as f32
}

#[test]
fn manifest_validates_and_agrees_with_contract() {
    let manifest = Flip::manifest().expect("flip manifest");
    manifest.validate().expect("flip manifest valid");
    paintop_ir::check_contract_consistency(&manifest, &Flip::new())
        .expect("flip manifest agrees with contract");
    paintop_ir::verify_categories(&manifest, &manifest.test.verification)
        .expect("flip verification declarations gate clean");
    assert_eq!(manifest.id.to_string(), FLIP_OP_ID);
}

#[test]
fn horizontal_flip_reverses_columns() {
    // 3x2 ramp; out(x, y) = in(2-x, y).
    let img = ramp_image(3, 2);
    let out = flip(&img, "horizontal");
    assert_eq!(out.extent(), Extent::new(3, 2));
    for y in 0..2 {
        for x in 0..3 {
            let got = out.samples()[((y * 3 + x) * 4) as usize];
            assert!((got - ramp_base(2 - x, y, 3)).abs() < f32::EPSILON);
        }
    }
}

#[test]
fn vertical_flip_reverses_rows() {
    let img = ramp_image(3, 2);
    let out = flip(&img, "vertical");
    assert_eq!(out.extent(), Extent::new(3, 2));
    for y in 0..2 {
        for x in 0..3 {
            let got = out.samples()[((y * 3 + x) * 4) as usize];
            assert!((got - ramp_base(x, 1 - y, 3)).abs() < f32::EPSILON);
        }
    }
}

#[test]
fn both_flip_reverses_both_axes() {
    let img = ramp_image(3, 2);
    let out = flip(&img, "both");
    for y in 0..2 {
        for x in 0..3 {
            let got = out.samples()[((y * 3 + x) * 4) as usize];
            assert!((got - ramp_base(2 - x, 1 - y, 3)).abs() < f32::EPSILON);
        }
    }
}

#[test]
fn every_flip_is_an_involution() {
    // Double-flip identity (§2.5), via the shared metamorphic harness.
    let img = ramp_image(4, 3);
    for axis in ["horizontal", "vertical", "both"] {
        assert_involution(&img, |v| flip(v, axis));
    }
}

#[test]
fn horizontal_then_vertical_equals_both() {
    let img = ramp_image(4, 3);
    let hv = flip(&flip(&img, "horizontal"), "vertical");
    let both = flip(&img, "both");
    assert!(samples_bit_identical(&hv, &both));
}

#[test]
fn missing_axis_is_rejected() {
    let img = ramp_image(2, 2);
    let mut inputs = InputValues::new();
    inputs.insert("image".to_owned(), img);
    let err = Flip::new()
        .compute(&inputs, &serde_json::json!({}))
        .expect_err("missing axis must be rejected");
    assert_eq!(err.class, ErrorClass::Schema);
    assert_eq!(err.code, E_FLIP_AXIS);
}

#[test]
fn unknown_axis_is_rejected() {
    let img = ramp_image(2, 2);
    let mut inputs = InputValues::new();
    inputs.insert("image".to_owned(), img);
    let err = Flip::new()
        .compute(&inputs, &serde_json::json!({ "axis": "diagonal" }))
        .expect_err("unknown axis must be rejected");
    assert_eq!(err.class, ErrorClass::Schema);
    assert_eq!(err.code, E_FLIP_AXIS);
}

#[test]
fn infer_outputs_preserves_extent_and_maps_roi() {
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
    let params = serde_json::json!({ "axis": "horizontal" });

    let out = Flip::new().infer_outputs(&inputs, &params).expect("infer");
    let ResourceDescriptor::Image(d) = out["image"] else {
        panic!("expected image");
    };
    assert_eq!(d.extent, Extent::new(10, 8));

    // Horizontal flip: an output region [x0, x1) maps to [W-x1, W-x0).
    let mut requested = OutputRegions::new();
    requested.insert("image".to_owned(), Rect::new(1, 2, 4, 5));
    let needed = Flip::new()
        .required_inputs(&requested, &inputs, &params)
        .expect("required_inputs");
    assert_eq!(needed["image"], Rect::new(6, 2, 9, 5));
}

#[test]
fn checked_in_manifest_matches_builder() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("crates dir")
        .parent()
        .expect("repo root")
        .join("ops/manifests");
    let manifest = Flip::manifest().expect("flip manifest");
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
