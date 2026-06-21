//! Verification suite for `color.adjust@1` (`OP_CATALOG` §2,
//! `AGENT_VERIFICATION` §3.1):
//!
//! - **schema/contract**: the manifest validates, agrees with its contract, its
//!   verification declarations gate clean, and the checked-in JSON matches the
//!   builder;
//! - **analytic fixtures**: `exposure(x, e) = x · 2^e` reproduces a known value
//!   table; a fully-desaturated pixel collapses to its luminance;
//! - **property**: empty-mask identity, full-mask equals unmasked, zero
//!   adjustment is the exact identity, exposure is monotonic over the nonnegative
//!   domain, every output is finite (incl. HDR / alpha-zero / extreme fixtures),
//!   and alpha is passed through;
//! - **metamorphic**: exposure composition before clamp
//!   (`E_a ∘ E_b = E_{a+b}`) and masked locality (no change where coverage is 0);
//! - **rejection**: a `srgb` (non-linear) input and a mismatched mask extent are
//!   rejected.

use paintop_core::executor::{InputValues, OpImplementation, ResourceValue};
use paintop_ir::{
    AlphaRepresentation, ChannelLayout, ColorEncoding, ColorRange, CoordinateConvention,
    Descriptors, ErrorClass, Extent, ImageDescriptor, MaskDescriptor, MaskMeaning, OpContract,
    OutputRegions, Rect, ResourceDescriptor, ScalarType, SemanticRole, ValidRange,
    check_contract_consistency, verify_categories,
};

use super::{ADJUST_OP_ID, Adjust, E_ADJUST_MASK, E_ADJUST_NONLINEAR};

/// The numeric tolerance for `exp2`-based comparisons (bounded tier).
const TOL: f32 = 1e-5;

/// Build a linear-light color image [`ResourceValue`].
fn image_value(width: u32, height: u32, layout: ChannelLayout, samples: Vec<f32>) -> ResourceValue {
    image_value_encoded(width, height, layout, ColorEncoding::LinearSrgb, samples)
}

/// Build a color image [`ResourceValue`] with an explicit encoding.
fn image_value_encoded(
    width: u32,
    height: u32,
    layout: ChannelLayout,
    color: ColorEncoding,
    samples: Vec<f32>,
) -> ResourceValue {
    let descriptor = ResourceDescriptor::Image(ImageDescriptor {
        extent: Extent::new(width, height),
        layout,
        scalar: ScalarType::F32,
        color,
        range: ColorRange::SceneReferred,
        alpha: AlphaRepresentation::Straight,
        coordinates: CoordinateConvention::PixelCenterUpperLeft,
        semantic: SemanticRole::Color,
    });
    ResourceValue::new(descriptor, layout.channel_count(), samples)
        .expect("sample buffer matches descriptor")
}

/// Build a coverage mask [`ResourceValue`].
fn mask_value(width: u32, height: u32, samples: Vec<f32>) -> ResourceValue {
    let descriptor = ResourceDescriptor::Mask(MaskDescriptor {
        extent: Extent::new(width, height),
        scalar: ScalarType::F32,
        range: ValidRange::Bounded { min: 0.0, max: 1.0 },
        meaning: MaskMeaning::Coverage,
        coordinates: CoordinateConvention::PixelCenterUpperLeft,
    });
    ResourceValue::new(descriptor, 1, samples).expect("mask buffer matches descriptor")
}

/// Run the compute kernel with the given params and recover the output image.
fn adjust(value: &ResourceValue, params: &serde_json::Value) -> ResourceValue {
    let mut inputs = InputValues::new();
    inputs.insert("image".to_owned(), value.clone());
    let mut out = Adjust::new()
        .compute(&inputs, params)
        .expect("adjust computes");
    out.remove("image").expect("image port produced")
}

/// Run the compute kernel with an image and a mask.
fn adjust_masked(
    value: &ResourceValue,
    mask: &ResourceValue,
    params: &serde_json::Value,
) -> ResourceValue {
    let mut inputs = InputValues::new();
    inputs.insert("image".to_owned(), value.clone());
    inputs.insert("mask".to_owned(), mask.clone());
    let mut out = Adjust::new()
        .compute(&inputs, params)
        .expect("masked adjust computes");
    out.remove("image").expect("image port produced")
}

#[test]
fn manifest_validates_and_agrees_with_contract() {
    let manifest = Adjust::manifest().expect("adjust manifest");
    manifest.validate().expect("adjust manifest valid");
    check_contract_consistency(&manifest, &Adjust::new())
        .expect("adjust manifest agrees with contract");
    verify_categories(&manifest, &manifest.test.verification)
        .expect("adjust verification declarations gate clean");
    assert_eq!(manifest.id.to_string(), ADJUST_OP_ID);
}

#[test]
fn exposure_matches_known_value_table() {
    // exposure(x, e) = x * 2^e on every color channel.
    let table = [
        (0.5_f32, 0.0_f32, 0.5_f32),
        (0.5, 1.0, 1.0),
        (0.5, -1.0, 0.25),
        (0.2, 2.0, 0.8),
        (1.0, 3.0, 8.0),
    ];
    for (x, ev, want) in table {
        let value = image_value(1, 1, ChannelLayout::Rgb, vec![x; 3]);
        let out = adjust(&value, &serde_json::json!({ "exposure_ev": ev }));
        for &s in out.samples() {
            assert!(
                (s - want).abs() < TOL,
                "exposure({x}, {ev}) = {s}, want {want}"
            );
        }
    }
}

#[test]
fn zero_adjustment_is_exact_identity() {
    let samples: Vec<f32> = (0..12u8).map(|i| f32::from(i) / 11.0).collect();
    let value = image_value(2, 2, ChannelLayout::Rgb, samples.clone());
    // All defaults => verbatim passthrough (bit-exact).
    let out = adjust(&value, &serde_json::json!({}));
    assert_eq!(out.samples(), samples.as_slice());
    // Explicit zeros are likewise the exact identity.
    let out = adjust(
        &value,
        &serde_json::json!({ "exposure_ev": 0.0, "saturation": 0.0, "temperature": 0.0 }),
    );
    assert_eq!(out.samples(), samples.as_slice());
}

#[test]
fn exposure_composes_additively_before_clamp() {
    // E_a(E_b(x)) = E_{a+b}(x) in unclamped linear light (HDR values preserved).
    let samples = vec![0.1, 0.5, 2.5, 0.3, 4.0, 0.05];
    let value = image_value(2, 1, ChannelLayout::Rgb, samples);
    let a = 0.7_f32;
    let b = -1.3_f32;

    let first = adjust(&value, &serde_json::json!({ "exposure_ev": b }));
    let composed = adjust(&first, &serde_json::json!({ "exposure_ev": a }));
    let direct = adjust(&value, &serde_json::json!({ "exposure_ev": a + b }));

    for (got, want) in composed.samples().iter().zip(direct.samples().iter()) {
        let rel = (got - want).abs() / want.abs().max(1.0);
        assert!(rel < TOL, "composition {got} vs direct {want}");
    }
}

#[test]
fn exposure_is_monotonic_over_nonnegative_domain() {
    // For a fixed nonnegative input, output is monotone non-decreasing in EV.
    let x = 0.3_f32;
    let mut prev = f32::NEG_INFINITY;
    for i in -50..=50i32 {
        #[allow(clippy::cast_precision_loss, reason = "small loop bound")]
        let ev = i as f32 / 10.0;
        let value = image_value(1, 1, ChannelLayout::Gray, vec![x]);
        let out = adjust(&value, &serde_json::json!({ "exposure_ev": ev }));
        let s = out.samples()[0];
        assert!(s >= prev, "exposure not monotone at ev={ev}: {s} < {prev}");
        prev = s;
    }
}

#[test]
fn full_desaturation_collapses_to_luminance() {
    // saturation = -1 maps every channel to the pixel's Rec.709 linear luminance.
    let (r, g, b) = (0.8_f32, 0.3, 0.1);
    let luma = 0.212_6f32.mul_add(r, 0.715_2f32.mul_add(g, 0.072_2 * b));
    let value = image_value(1, 1, ChannelLayout::Rgb, vec![r, g, b]);
    let out = adjust(&value, &serde_json::json!({ "saturation": -1.0 }));
    for &s in out.samples() {
        assert!(
            (s - luma).abs() < TOL,
            "desaturated channel {s} != luma {luma}"
        );
    }
}

#[test]
fn saturation_preserves_luminance() {
    // A saturation change holds the pixel luminance invariant (the blend pivots
    // around luminance), for any saturation factor.
    let (r, g, b) = (0.6_f32, 0.25, 0.4);
    let luma0 = 0.212_6f32.mul_add(r, 0.715_2f32.mul_add(g, 0.072_2 * b));
    for sat in [-0.5_f32, 0.4, 1.5] {
        let value = image_value(1, 1, ChannelLayout::Rgb, vec![r, g, b]);
        let out = adjust(&value, &serde_json::json!({ "saturation": sat }));
        let s = out.samples();
        let luma1 = 0.212_6f32.mul_add(s[0], 0.715_2f32.mul_add(s[1], 0.072_2 * s[2]));
        assert!(
            (luma1 - luma0).abs() < TOL,
            "luminance moved under sat={sat}"
        );
    }
}

#[test]
fn temperature_warms_red_and_cools_blue() {
    let value = image_value(1, 1, ChannelLayout::Rgb, vec![0.4, 0.4, 0.4]);
    let out = adjust(&value, &serde_json::json!({ "temperature": 0.25 }));
    let s = out.samples();
    assert!(
        (s[0] - 0.5).abs() < TOL,
        "red should warm to 0.5, got {}",
        s[0]
    );
    assert!((s[1] - 0.4).abs() < TOL, "green unchanged, got {}", s[1]);
    assert!(
        (s[2] - 0.3).abs() < TOL,
        "blue should cool to 0.3, got {}",
        s[2]
    );
}

#[test]
fn alpha_channel_is_passed_through_unchanged() {
    // RGBA with a strong exposure: color scales, alpha is untouched bit-for-bit.
    let value = image_value(1, 1, ChannelLayout::Rgba, vec![0.5, 0.5, 0.5, 0.5]);
    let out = adjust(&value, &serde_json::json!({ "exposure_ev": 1.0 }));
    let s = out.samples();
    for &c in &s[0..3] {
        assert!((c - 1.0).abs() < TOL, "color got {c}");
    }
    assert_eq!(s[3].to_bits(), 0.5_f32.to_bits(), "alpha must be untouched");
}

#[test]
fn empty_mask_is_identity() {
    let samples: Vec<f32> = (0..12u8).map(|i| f32::from(i) / 11.0).collect();
    let value = image_value(2, 2, ChannelLayout::Rgb, samples.clone());
    let mask = mask_value(2, 2, vec![0.0; 4]);
    let params = serde_json::json!({ "exposure_ev": 2.0, "saturation": 0.5, "temperature": 0.3 });
    let out = adjust_masked(&value, &mask, &params);
    // Empty coverage everywhere => the input is reproduced exactly.
    assert_eq!(out.samples(), samples.as_slice());
}

#[test]
fn full_mask_equals_unmasked() {
    let samples: Vec<f32> = (0..12u8).map(|i| f32::from(i) / 11.0).collect();
    let value = image_value(2, 2, ChannelLayout::Rgb, samples);
    let mask = mask_value(2, 2, vec![1.0; 4]);
    let params = serde_json::json!({ "exposure_ev": 1.3, "saturation": -0.4, "temperature": 0.2 });
    let masked = adjust_masked(&value, &mask, &params);
    let unmasked = adjust(&value, &params);
    assert_eq!(masked.samples(), unmasked.samples());
}

#[test]
fn no_change_outside_mask() {
    // A per-pixel mask: pixel 0 fully on, pixels 1..3 off. Only pixel 0 changes;
    // the others are bit-exact identity (masked locality).
    let samples = vec![
        0.2, 0.3, 0.4, // pixel 0
        0.5, 0.6, 0.7, // pixel 1
        0.1, 0.2, 0.3, // pixel 2
        0.9, 0.8, 0.7, // pixel 3
    ];
    let value = image_value(2, 2, ChannelLayout::Rgb, samples.clone());
    let mask = mask_value(2, 2, vec![1.0, 0.0, 0.0, 0.0]);
    let params = serde_json::json!({ "exposure_ev": 1.0 });
    let out = adjust_masked(&value, &mask, &params);
    let s = out.samples();
    // Pixel 0 doubled by the +1EV exposure.
    for ch in 0..3 {
        assert!(
            samples[ch].mul_add(-2.0, s[ch]).abs() < TOL,
            "pixel 0 ch {ch} not adjusted"
        );
    }
    // Pixels 1..3 untouched, bit-for-bit.
    for i in 3..samples.len() {
        assert_eq!(
            s[i].to_bits(),
            samples[i].to_bits(),
            "sample {i} changed outside mask"
        );
    }
}

#[test]
fn partial_mask_blends_linearly() {
    // Coverage 0.5 with +1EV: out = in + 0.5*(2*in - in) = 1.5*in.
    let value = image_value(1, 1, ChannelLayout::Rgb, vec![0.4, 0.4, 0.4]);
    let mask = mask_value(1, 1, vec![0.5]);
    let out = adjust_masked(&value, &mask, &serde_json::json!({ "exposure_ev": 1.0 }));
    for &s in out.samples() {
        assert!(
            (s - 0.6).abs() < TOL,
            "half-coverage blend got {s}, want 0.6"
        );
    }
}

#[test]
fn outputs_are_finite_on_adversarial_fixtures() {
    // HDR values above 1, alpha zero with arbitrary RGB, channel extremes, and a
    // subnormal — every output sample must be finite.
    let samples = vec![
        8.0,
        16.0,
        32.0,
        0.0, // HDR color, alpha zero
        f32::MIN_POSITIVE,
        1.0,
        0.0,
        0.0, // subnormal-adjacent + extremes, alpha zero
        1e30,
        1e-30,
        5.0,
        1.0, // very large / very small
    ];
    let value = image_value(3, 1, ChannelLayout::Rgba, samples);
    let params = serde_json::json!({ "exposure_ev": 3.0, "saturation": 1.5, "temperature": -0.5 });
    let out = adjust(&value, &params);
    for &s in out.samples() {
        assert!(s.is_finite(), "produced a non-finite sample {s}");
    }
}

#[test]
fn srgb_input_is_rejected_as_nonlinear() {
    let value = image_value_encoded(1, 1, ChannelLayout::Rgb, ColorEncoding::Srgb, vec![0.5; 3]);
    let mut inputs = InputValues::new();
    inputs.insert("image".to_owned(), value);
    let err = Adjust::new()
        .compute(&inputs, &serde_json::json!({ "exposure_ev": 1.0 }))
        .expect_err("srgb input must be rejected");
    assert_eq!(err.class, ErrorClass::Semantic);
    assert_eq!(err.code, E_ADJUST_NONLINEAR);
}

#[test]
fn mismatched_mask_extent_is_rejected() {
    let value = image_value(2, 2, ChannelLayout::Rgb, vec![0.5; 12]);
    let mask = mask_value(1, 1, vec![1.0]);
    let mut inputs = InputValues::new();
    inputs.insert("image".to_owned(), value);
    inputs.insert("mask".to_owned(), mask);
    let err = Adjust::new()
        .compute(&inputs, &serde_json::json!({ "exposure_ev": 1.0 }))
        .expect_err("mismatched mask extent must be rejected");
    assert_eq!(err.class, ErrorClass::Semantic);
    assert_eq!(err.code, E_ADJUST_MASK);
}

#[test]
fn non_finite_param_is_a_schema_error() {
    let value = image_value(1, 1, ChannelLayout::Gray, vec![0.5]);
    let mut inputs = InputValues::new();
    inputs.insert("image".to_owned(), value);
    // JSON has no NaN/Infinity literal, but a string is a clean non-number case.
    let err = Adjust::new()
        .compute(&inputs, &serde_json::json!({ "exposure_ev": "lots" }))
        .expect_err("non-number exposure must error");
    assert_eq!(err.class, ErrorClass::Schema);
}

#[test]
fn infer_outputs_preserves_descriptor_and_pointwise_roi() {
    let descriptor = ResourceDescriptor::Image(ImageDescriptor {
        extent: Extent::new(8, 8),
        layout: ChannelLayout::Rgba,
        scalar: ScalarType::F32,
        color: ColorEncoding::LinearSrgb,
        range: ColorRange::SceneReferred,
        alpha: AlphaRepresentation::Straight,
        coordinates: CoordinateConvention::PixelCenterUpperLeft,
        semantic: SemanticRole::Color,
    });
    let mut inputs = Descriptors::new();
    inputs.insert("image".to_owned(), descriptor);
    let params = serde_json::json!({ "exposure_ev": 0.5 });

    let out = Adjust::new()
        .infer_outputs(&inputs, &params)
        .expect("infer");
    let ResourceDescriptor::Image(d) = out["image"] else {
        panic!("expected image");
    };
    assert_eq!(d.extent, Extent::new(8, 8));
    assert_eq!(d.color, ColorEncoding::LinearSrgb);

    // Pointwise ROI: the demanded image region equals the requested output region.
    let mut requested = OutputRegions::new();
    requested.insert("image".to_owned(), Rect::new(2, 3, 4, 5));
    let needed = Adjust::new()
        .required_inputs(&requested, &inputs, &params)
        .expect("required_inputs");
    assert_eq!(needed["image"], Rect::new(2, 3, 4, 5));
    // No mask wired => no mask region demanded.
    assert!(!needed.contains_key("mask"));

    // Postcondition: an image output is produced.
    let results = Adjust::new()
        .validate_postconditions(&out, &params)
        .expect("postconditions");
    assert!(
        results
            .iter()
            .all(|r| r.status == paintop_ir::AssertionStatus::Pass)
    );
}

/// The checked-in `ops/manifests/<id>.json` file (read by `cargo xtask
/// verify-op`) must stay byte-identical to the Rust manifest builder, the source
/// of truth. Regenerate with `serde_json::to_string_pretty` if this fails.
#[test]
fn checked_in_manifest_matches_builder() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("crates dir")
        .parent()
        .expect("repo root")
        .join("ops/manifests");
    let manifest = Adjust::manifest().expect("adjust manifest");
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
