//! Verification suite for `paint.fill@1` (`OP_CATALOG` §6) — a masked, typed
//! constant fill:
//!
//! - **schema/contract**: the manifest validates, agrees with its contract, and
//!   its verification declarations gate clean; the checked-in manifest stays in
//!   lockstep with the Rust builder;
//! - **analytic fixtures (mask extremes)**: an empty mask (`m = 0`) is the
//!   identity bit-exactly; a full mask (`m = 1`) reproduces the typed `value`
//!   bit-exactly across every channel;
//! - **property (the safety-critical one)**: wherever the mask is `0` the output
//!   is bit-identical to the input image, regardless of the fill;
//! - **property (soft mask)**: an intermediate `m` lerps each channel between the
//!   base and the fill, stays between them, and is finite;
//! - **type correctness**: a per-channel value is written per channel (color and
//!   alpha distinctly);
//! - **rejection**: a value of the wrong length, a non-finite value, an
//!   out-of-range display-referred / alpha value, and a mask/extent mismatch are
//!   each rejected with the right class.

use paintop_core::executor::{InputValues, OpImplementation, ResourceValue};
use paintop_ir::{
    AlphaRepresentation, AssertionStatus, ChannelLayout, ColorEncoding, ColorRange,
    CoordinateConvention, Descriptors, ErrorClass, Extent, ImageDescriptor, MaskDescriptor,
    MaskMeaning, OpContract, OutputRegions, Rect, ResourceDescriptor, ScalarType, SemanticRole,
    ValidRange, check_contract_consistency, verify_categories,
};

use super::{E_FILL_INPUT, E_FILL_SHAPE, E_FILL_VALUE, FILL_OP_ID, Fill};

/// A display-referred sRGB RGBA image descriptor of side `n`.
fn rgba_descriptor(n: u32) -> ImageDescriptor {
    ImageDescriptor {
        extent: Extent::new(n, n),
        layout: ChannelLayout::Rgba,
        scalar: ScalarType::F32,
        color: ColorEncoding::Srgb,
        range: ColorRange::DisplayReferred,
        alpha: AlphaRepresentation::Straight,
        coordinates: CoordinateConvention::PixelCenterUpperLeft,
        semantic: SemanticRole::Color,
    }
}

/// An RGBA image value of side `n` whose every sample is `fill`.
fn rgba_image(n: u32, fill: f32) -> ResourceValue {
    let descriptor = ResourceDescriptor::Image(rgba_descriptor(n));
    let channels = ChannelLayout::Rgba.channel_count();
    let len = (n as usize) * (n as usize) * channels as usize;
    ResourceValue::new(descriptor, channels, vec![fill; len]).expect("rgba image")
}

/// An RGBA image value of side `n` from an explicit row-major buffer.
fn rgba_image_from(n: u32, samples: Vec<f32>) -> ResourceValue {
    let descriptor = ResourceDescriptor::Image(rgba_descriptor(n));
    ResourceValue::new(descriptor, ChannelLayout::Rgba.channel_count(), samples)
        .expect("rgba image")
}

/// A coverage mask value of side `n` from an explicit row-major buffer.
fn mask_image(n: u32, coverage: Vec<f32>) -> ResourceValue {
    let descriptor = ResourceDescriptor::Mask(MaskDescriptor {
        extent: Extent::new(n, n),
        scalar: ScalarType::F32,
        range: ValidRange::Bounded { min: 0.0, max: 1.0 },
        meaning: MaskMeaning::Coverage,
        coordinates: CoordinateConvention::PixelCenterUpperLeft,
    });
    ResourceValue::new(descriptor, 1, coverage).expect("coverage mask")
}

/// A constant coverage mask of side `n`.
fn const_mask(n: u32, coverage: f32) -> ResourceValue {
    mask_image(n, vec![coverage; (n as usize) * (n as usize)])
}

/// Run `paint.fill`, returning the produced image (panics on a compute error).
fn fill(image: ResourceValue, mask: ResourceValue, value: &serde_json::Value) -> ResourceValue {
    let mut inputs = InputValues::new();
    inputs.insert("image".to_owned(), image);
    inputs.insert("mask".to_owned(), mask);
    let params = serde_json::json!({ "value": value });
    let mut out = Fill::new()
        .compute(&inputs, &params)
        .expect("fill computes");
    out.remove("image").expect("image port produced")
}

/// Run `paint.fill`, returning the error (panics on success).
fn fill_err(
    image: ResourceValue,
    mask: ResourceValue,
    value: &serde_json::Value,
) -> paintop_ir::Error {
    let mut inputs = InputValues::new();
    inputs.insert("image".to_owned(), image);
    inputs.insert("mask".to_owned(), mask);
    let params = serde_json::json!({ "value": value });
    Fill::new()
        .compute(&inputs, &params)
        .expect_err("fill must reject")
}

/// The descriptor inputs for `infer_outputs` / `required_inputs`.
fn descriptors(n: u32) -> Descriptors {
    let mut inputs = Descriptors::new();
    inputs.insert(
        "image".to_owned(),
        ResourceDescriptor::Image(rgba_descriptor(n)),
    );
    inputs.insert(
        "mask".to_owned(),
        ResourceDescriptor::Mask(MaskDescriptor {
            extent: Extent::new(n, n),
            scalar: ScalarType::F32,
            range: ValidRange::Bounded { min: 0.0, max: 1.0 },
            meaning: MaskMeaning::Coverage,
            coordinates: CoordinateConvention::PixelCenterUpperLeft,
        }),
    );
    inputs
}

#[test]
fn manifest_validates_and_agrees_with_contract() {
    let manifest = Fill::manifest().expect("fill manifest");
    manifest.validate().expect("manifest valid");
    check_contract_consistency(&manifest, &Fill::new()).expect("manifest agrees with contract");
    verify_categories(&manifest, &manifest.test.verification)
        .expect("verification declarations gate clean");
    assert_eq!(manifest.id.to_string(), FILL_OP_ID);
}

#[test]
fn produces_image_descriptor_matching_input() {
    let out = fill(
        rgba_image(8, 0.25),
        const_mask(8, 0.5),
        &serde_json::json!([1.0, 0.0, 0.0, 1.0]),
    );
    let ResourceDescriptor::Image(d) = out.descriptor() else {
        panic!("expected image output");
    };
    assert_eq!(d.extent, Extent::new(8, 8));
    assert_eq!(d.layout, ChannelLayout::Rgba);
    assert_eq!(out.channels(), 4);
    assert_eq!(out.samples().len(), 8 * 8 * 4);
}

/// Analytic fixture: an empty mask (`m = 0`) is the identity, bit-exactly.
#[test]
fn empty_mask_is_identity_bit_exact() {
    let image = rgba_image(8, 0.375);
    let before: Vec<f32> = image.samples().to_vec();
    let out = fill(
        image,
        const_mask(8, 0.0),
        // A deliberately very different fill to prove only the mask gates it.
        &serde_json::json!([0.9, 0.9, 0.9, 0.9]),
    );
    assert_eq!(out.samples(), before.as_slice());
}

/// Analytic fixture: a full mask (`m = 1`) reproduces the typed value exactly in
/// every channel of every pixel (type correctness, per-channel).
#[test]
fn full_mask_is_constant_value_bit_exact() {
    let value = [0.125_f32, 0.5, 0.75, 1.0];
    let out = fill(
        rgba_image(4, 0.0),
        const_mask(4, 1.0),
        &serde_json::json!(value),
    );
    for pixel in out.samples().chunks_exact(4) {
        for (c, &got) in pixel.iter().enumerate() {
            assert_eq!(got.to_bits(), value[c].to_bits(), "channel {c}");
        }
    }
}

/// The safety-critical property: wherever the mask is `0`, the output is
/// bit-identical to the input image, even with a per-pixel mask and an arbitrary
/// fill.
#[test]
fn outside_mask_is_bit_identical_to_input() {
    let n = 6;
    let pixels = (n * n) as usize;
    let base_samples: Vec<f32> = (0..pixels * 4)
        .map(|i| f32::from(u8::try_from(i % 251).unwrap()) / 251.0)
        .collect();
    // Checkerboard mask: even pixels filled (m=1), odd pixels untouched (m=0).
    let mask_samples: Vec<f32> = (0..pixels)
        .map(|i| if i % 2 == 0 { 1.0 } else { 0.0 })
        .collect();

    let out = fill(
        rgba_image_from(n, base_samples.clone()),
        mask_image(n, mask_samples.clone()),
        &serde_json::json!([0.3, 0.6, 0.9, 0.4]),
    );
    let result = out.samples();
    for (pixel, &m) in mask_samples.iter().enumerate() {
        if m.to_bits() == 0.0_f32.to_bits() {
            for c in 0..4 {
                let idx = pixel * 4 + c;
                assert_eq!(
                    result[idx].to_bits(),
                    base_samples[idx].to_bits(),
                    "pixel {pixel} channel {c} changed outside the mask"
                );
            }
        }
    }
}

/// Soft-mask lerp property: at intermediate coverage each output channel lies
/// between the base and the fill, equals the analytic lerp, and is finite.
#[test]
fn soft_mask_lerps_between_base_and_value() {
    let n = 4;
    let base_val = 0.2_f32;
    let fill_val = 0.8_f32;
    let m = 0.25_f32;
    let out = fill(
        rgba_image(n, base_val),
        const_mask(n, m),
        &serde_json::json!([fill_val, fill_val, fill_val, fill_val]),
    );
    let expected = m.mul_add(fill_val - base_val, base_val);
    for &s in out.samples() {
        assert!(s.is_finite(), "sample must be finite, got {s}");
        assert!(
            base_val <= s && s <= fill_val,
            "sample {s} not between base {base_val} and fill {fill_val}"
        );
        assert!((s - expected).abs() < 1e-6, "sample {s} != lerp {expected}");
    }
}

/// `infer_outputs` returns the input descriptor and `required_inputs` is pointwise
/// on both ports.
#[test]
fn contract_infers_input_and_pointwise_roi() {
    let inputs = descriptors(10);
    let params = serde_json::json!({ "value": [0.0, 0.0, 0.0, 1.0] });
    let outputs = Fill::new().infer_outputs(&inputs, &params).expect("infer");
    let ResourceDescriptor::Image(d) = outputs.get("image").expect("image out") else {
        panic!("expected image");
    };
    assert_eq!(d.extent, Extent::new(10, 10));

    let region = Rect::new(2, 3, 5, 6);
    let mut requested = OutputRegions::new();
    requested.insert("image".to_owned(), region);
    let regions = Fill::new()
        .required_inputs(&requested, &inputs, &params)
        .expect("required inputs");
    for port in ["image", "mask"] {
        assert_eq!(regions.get(port), Some(&region), "port {port} ROI");
    }
}

/// The postcondition passes for a valid in-range value.
#[test]
fn postcondition_checks_value_in_range() {
    let inputs = descriptors(4);
    let params = serde_json::json!({ "value": [0.5, 0.5, 0.5, 1.0] });
    let outputs = Fill::new().infer_outputs(&inputs, &params).expect("infer");
    let results = Fill::new()
        .validate_postconditions(&outputs, &params)
        .expect("postconditions");
    assert!(
        results.iter().all(|r| r.status == AssertionStatus::Pass),
        "all postconditions pass: {results:?}"
    );
}

/// A value with the wrong number of components is rejected.
#[test]
fn wrong_value_length_is_rejected() {
    let err = fill_err(
        rgba_image(4, 0.0),
        const_mask(4, 1.0),
        &serde_json::json!([0.5, 0.5]),
    );
    assert_eq!(err.class, ErrorClass::Schema);
    assert_eq!(err.code, E_FILL_VALUE);
}

/// A non-finite value component is rejected (the output must never be NaN/Inf).
#[test]
fn non_finite_value_is_rejected() {
    let err = fill_err(
        rgba_image(4, 0.0),
        const_mask(4, 1.0),
        // serde_json cannot hold NaN, so a string is the natural "not a number".
        &serde_json::json!([0.5, 0.5, "nan", 1.0]),
    );
    assert_eq!(err.class, ErrorClass::Schema);
    assert_eq!(err.code, E_FILL_VALUE);
}

/// An out-of-range color component for a display-referred image is rejected
/// (clamping is never implicit).
#[test]
fn out_of_range_display_color_is_rejected() {
    let err = fill_err(
        rgba_image(4, 0.0),
        const_mask(4, 1.0),
        &serde_json::json!([1.5, 0.0, 0.0, 1.0]),
    );
    assert_eq!(err.class, ErrorClass::Policy);
    assert_eq!(err.code, E_FILL_VALUE);
}

/// An out-of-range alpha component is rejected regardless of the color range.
#[test]
fn out_of_range_alpha_is_rejected() {
    // Build a scene-referred image so the color channels are unbounded, isolating
    // the alpha range check.
    let mut d = rgba_descriptor(4);
    d.range = ColorRange::SceneReferred;
    let image = ResourceValue::new(ResourceDescriptor::Image(d), 4, vec![0.0; 4 * 4 * 4])
        .expect("scene image");
    let err = fill_err(
        image,
        const_mask(4, 1.0),
        &serde_json::json!([2.0, 2.0, 2.0, 1.5]),
    );
    assert_eq!(err.class, ErrorClass::Policy);
    assert_eq!(err.code, E_FILL_VALUE);
}

/// A scene-referred color channel accepts an out-of-[0,1] (but finite) fill.
#[test]
fn scene_referred_allows_extended_range() {
    let mut d = rgba_descriptor(2);
    d.range = ColorRange::SceneReferred;
    let image = ResourceValue::new(ResourceDescriptor::Image(d), 4, vec![0.0; 2 * 2 * 4])
        .expect("scene image");
    let out = fill(
        image,
        const_mask(2, 1.0),
        &serde_json::json!([4.0, 2.0, 8.0, 1.0]),
    );
    let expected = [4.0_f32, 2.0, 8.0, 1.0];
    for pixel in out.samples().chunks_exact(4) {
        for (c, &got) in pixel.iter().enumerate() {
            assert_eq!(got.to_bits(), expected[c].to_bits());
        }
    }
}

/// A mask whose extent differs from the image is rejected.
#[test]
fn mask_extent_mismatch_is_rejected() {
    let err = fill_err(
        rgba_image(4, 0.0),
        const_mask(2, 1.0),
        &serde_json::json!([0.5, 0.5, 0.5, 1.0]),
    );
    assert_eq!(err.class, ErrorClass::Semantic);
    assert_eq!(err.code, E_FILL_SHAPE);
}

/// A missing `mask` port is a reference error.
#[test]
fn missing_mask_is_rejected() {
    let mut inputs = InputValues::new();
    inputs.insert("image".to_owned(), rgba_image(4, 0.0));
    let params = serde_json::json!({ "value": [0.5, 0.5, 0.5, 1.0] });
    let err = Fill::new()
        .compute(&inputs, &params)
        .expect_err("missing mask must be rejected");
    assert_eq!(err.class, ErrorClass::Reference);
    assert_eq!(err.code, E_FILL_INPUT);
}

/// The checked-in `ops/manifests/<id>.json` must stay byte-identical to the Rust
/// manifest builder, the source of truth.
#[test]
fn checked_in_manifest_matches_builder() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("crates dir")
        .parent()
        .expect("repo root")
        .join("ops/manifests");
    let manifest = Fill::manifest().expect("fill manifest");
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
