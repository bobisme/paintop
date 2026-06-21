//! Verification suite for `color.convert@1` (`OP_CATALOG` §2, `plan.md` §8.2,
//! `AGENT_VERIFICATION` §2.5):
//!
//! - **schema/contract**: the manifest validates, agrees with its contract, and
//!   its verification declarations gate clean;
//! - **analytic fixtures**: a known sRGB↔linear value table is reproduced within
//!   tolerance, the knot point maps as specified, and `0`/`1` are fixed points;
//! - **metamorphic**: `encode ∘ decode` and `decode ∘ encode` round-trip within
//!   tolerance; `from == to` is the exact identity; alpha is passed through;
//! - **property**: both transfer directions are monotonic non-decreasing;
//! - **rejection**: `display-p3`/`icc` and `raw-linear`↔color mixes are rejected
//!   with `semantic` errors, and a `from` that disagrees with the input encoding
//!   is rejected.

use paintop_core::executor::{InputValues, OpImplementation, ResourceValue};
use paintop_ir::{
    AlphaRepresentation, ChannelLayout, ColorEncoding, ColorRange, CoordinateConvention,
    Descriptors, ErrorClass, Extent, ImageDescriptor, OpContract, OutputRegions, Rect,
    ResourceDescriptor, ScalarType, SemanticRole, check_contract_consistency, verify_categories,
};

use super::{CONVERT_OP_ID, Convert, srgb_decode, srgb_encode};

/// The numeric tolerance for `powf`-based transfer comparisons (bounded tier).
const TOL: f32 = 1e-5;

/// Build a color image [`ResourceValue`] with the given encoding and samples.
fn image_value(
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
        range: ColorRange::DisplayReferred,
        alpha: AlphaRepresentation::Straight,
        coordinates: CoordinateConvention::PixelCenterUpperLeft,
        semantic: SemanticRole::Color,
    });
    ResourceValue::new(descriptor, layout.channel_count(), samples)
        .expect("sample buffer matches descriptor")
}

/// Run the compute kernel with the given `from`/`to` params and recover the
/// produced image value.
fn convert(value: &ResourceValue, from: &str, to: &str) -> ResourceValue {
    let mut inputs = InputValues::new();
    inputs.insert("image".to_owned(), value.clone());
    let params = serde_json::json!({ "from": from, "to": to });
    let mut out = Convert::new()
        .compute(&inputs, &params)
        .expect("convert computes");
    out.remove("image").expect("image port produced")
}

#[test]
fn manifest_validates_and_agrees_with_contract() {
    let manifest = Convert::manifest().expect("convert manifest");
    manifest.validate().expect("convert manifest valid");
    check_contract_consistency(&manifest, &Convert::new())
        .expect("convert manifest agrees with contract");
    verify_categories(&manifest, &manifest.test.verification)
        .expect("convert verification declarations gate clean");
    assert_eq!(manifest.id.to_string(), CONVERT_OP_ID);
}

#[test]
fn srgb_to_linear_matches_known_value_table() {
    // IEC 61966-2-1 reference values (srgb -> linear).
    let table = [
        (0.0_f32, 0.0_f32),
        (0.040_45, 0.003_130_8),
        (0.5, 0.214_041_14),
        (0.214_041_14, 0.037_649_48),
        (1.0, 1.0),
    ];
    for (srgb, linear) in table {
        let got = srgb_decode(srgb);
        assert!(
            (got - linear).abs() < TOL,
            "srgb_decode({srgb}) = {got}, expected {linear}"
        );
    }
}

#[test]
fn linear_to_srgb_matches_known_value_table() {
    // The inverse direction (linear -> srgb).
    let table = [
        (0.0_f32, 0.0_f32),
        (0.003_130_8, 0.040_45),
        (0.214_041_14, 0.5),
        (1.0, 1.0),
    ];
    for (linear, srgb) in table {
        let got = srgb_encode(linear);
        assert!(
            (got - srgb).abs() < TOL,
            "srgb_encode({linear}) = {got}, expected {srgb}"
        );
    }
}

#[test]
fn decode_then_encode_round_trips() {
    // encode ∘ decode is the identity within tolerance across the [0,1] range.
    for i in 0..=100u32 {
        #[allow(clippy::cast_precision_loss, reason = "small loop bound")]
        let c = i as f32 / 100.0;
        let round = srgb_encode(srgb_decode(c));
        assert!((round - c).abs() < TOL, "encode(decode({c})) = {round}");
        let round2 = srgb_decode(srgb_encode(c));
        assert!((round2 - c).abs() < TOL, "decode(encode({c})) = {round2}");
    }
}

#[test]
fn both_directions_are_monotonic_non_decreasing() {
    let mut prev_dec = f32::NEG_INFINITY;
    let mut prev_enc = f32::NEG_INFINITY;
    for i in 0..=1000u32 {
        #[allow(clippy::cast_precision_loss, reason = "small loop bound")]
        let c = i as f32 / 1000.0;
        let dec = srgb_decode(c);
        let enc = srgb_encode(c);
        assert!(dec >= prev_dec, "decode not monotonic at {c}");
        assert!(enc >= prev_enc, "encode not monotonic at {c}");
        prev_dec = dec;
        prev_enc = enc;
    }
}

#[test]
fn compute_converts_color_channels_and_records_target_encoding() {
    // A 2x1 RGB srgb image of 0.5 grey -> linear ~0.214 on every channel.
    let value = image_value(2, 1, ChannelLayout::Rgb, ColorEncoding::Srgb, vec![0.5; 6]);
    let out = convert(&value, "srgb", "linear-srgb");

    let ResourceDescriptor::Image(d) = out.descriptor() else {
        panic!("expected image output");
    };
    assert_eq!(d.color, ColorEncoding::LinearSrgb);
    for &s in out.samples() {
        assert!((s - 0.214_041_14).abs() < TOL, "got {s}");
    }
}

#[test]
fn alpha_channel_is_passed_through_unchanged() {
    // RGBA: color channels 0.5, alpha 0.5. Alpha must stay 0.5 (no transfer),
    // color channels become ~0.214.
    let value = image_value(
        1,
        1,
        ChannelLayout::Rgba,
        ColorEncoding::Srgb,
        vec![0.5, 0.5, 0.5, 0.5],
    );
    let out = convert(&value, "srgb", "linear-srgb");
    let s = out.samples();
    for &c in &s[0..3] {
        assert!((c - 0.214_041_14).abs() < TOL, "color got {c}");
    }
    // Alpha is passed through bit-for-bit (no transfer function applied).
    assert_eq!(s[3].to_bits(), 0.5_f32.to_bits(), "alpha must be untouched");
}

#[test]
fn identity_conversion_is_exact_passthrough() {
    let samples = vec![0.1, 0.4, 0.9];
    let value = image_value(
        3,
        1,
        ChannelLayout::Gray,
        ColorEncoding::Srgb,
        samples.clone(),
    );
    let out = convert(&value, "srgb", "srgb");
    // from == to: bit-exact passthrough.
    assert_eq!(out.samples(), samples.as_slice());
    let ResourceDescriptor::Image(d) = out.descriptor() else {
        panic!("expected image");
    };
    assert_eq!(d.color, ColorEncoding::Srgb);
}

#[test]
fn raw_linear_identity_is_allowed() {
    let samples = vec![0.2, 0.7];
    let value = image_value(
        2,
        1,
        ChannelLayout::Gray,
        ColorEncoding::RawLinear,
        samples.clone(),
    );
    let out = convert(&value, "raw-linear", "raw-linear");
    assert_eq!(out.samples(), samples.as_slice());
}

#[test]
fn full_round_trip_through_the_op_recovers_the_input() {
    let samples: Vec<f32> = (0..12u8).map(|i| f32::from(i) / 11.0).collect();
    let srgb = image_value(
        4,
        3,
        ChannelLayout::Gray,
        ColorEncoding::Srgb,
        samples.clone(),
    );
    let linear = convert(&srgb, "srgb", "linear-srgb");
    let back = convert(&linear, "linear-srgb", "srgb");
    for (got, want) in back.samples().iter().zip(samples.iter()) {
        assert!((got - want).abs() < TOL, "round-trip {got} vs {want}");
    }
}

#[test]
fn display_p3_request_is_rejected_semantically() {
    let value = image_value(1, 1, ChannelLayout::Gray, ColorEncoding::Srgb, vec![0.5]);
    let mut inputs = InputValues::new();
    inputs.insert("image".to_owned(), value);
    let params = serde_json::json!({ "from": "srgb", "to": "display-p3" });
    let err = Convert::new()
        .compute(&inputs, &params)
        .expect_err("display-p3 must be rejected");
    assert_eq!(err.class, ErrorClass::Semantic);
}

#[test]
fn icc_request_is_rejected_semantically() {
    let value = image_value(1, 1, ChannelLayout::Gray, ColorEncoding::Srgb, vec![0.5]);
    let mut inputs = InputValues::new();
    inputs.insert("image".to_owned(), value);
    let params = serde_json::json!({ "from": "icc", "to": "linear-srgb" });
    let err = Convert::new()
        .compute(&inputs, &params)
        .expect_err("icc must be rejected");
    assert_eq!(err.class, ErrorClass::Semantic);
}

#[test]
fn raw_linear_to_color_is_rejected_semantically() {
    let value = image_value(
        1,
        1,
        ChannelLayout::Gray,
        ColorEncoding::RawLinear,
        vec![0.5],
    );
    let mut inputs = InputValues::new();
    inputs.insert("image".to_owned(), value);
    let params = serde_json::json!({ "from": "raw-linear", "to": "linear-srgb" });
    let err = Convert::new()
        .compute(&inputs, &params)
        .expect_err("raw-linear -> color must be rejected");
    assert_eq!(err.class, ErrorClass::Semantic);
    assert_eq!(err.code, super::E_CONVERT_UNSUPPORTED);
}

#[test]
fn from_disagreeing_with_input_encoding_is_rejected() {
    // Input is srgb, but the plan claims it is already linear: rejected, so the
    // op can never silently double-encode.
    let value = image_value(1, 1, ChannelLayout::Gray, ColorEncoding::Srgb, vec![0.5]);
    let mut inputs = InputValues::new();
    inputs.insert("image".to_owned(), value);
    let params = serde_json::json!({ "from": "linear-srgb", "to": "srgb" });
    let err = Convert::new()
        .compute(&inputs, &params)
        .expect_err("mislabeled `from` must be rejected");
    assert_eq!(err.class, ErrorClass::Semantic);
}

#[test]
fn missing_param_is_a_schema_error() {
    let value = image_value(1, 1, ChannelLayout::Gray, ColorEncoding::Srgb, vec![0.5]);
    let mut inputs = InputValues::new();
    inputs.insert("image".to_owned(), value);
    let params = serde_json::json!({ "from": "srgb" });
    let err = Convert::new()
        .compute(&inputs, &params)
        .expect_err("missing `to` must error");
    assert_eq!(err.class, ErrorClass::Schema);
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
    let manifest = Convert::manifest().expect("convert manifest");
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

#[test]
fn infer_outputs_records_target_encoding_and_pointwise_roi() {
    let descriptor = ResourceDescriptor::Image(ImageDescriptor {
        extent: Extent::new(8, 8),
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
    let params = serde_json::json!({ "from": "srgb", "to": "linear-srgb" });

    let out = Convert::new()
        .infer_outputs(&inputs, &params)
        .expect("infer");
    let ResourceDescriptor::Image(d) = out["image"] else {
        panic!("expected image");
    };
    assert_eq!(d.color, ColorEncoding::LinearSrgb);
    assert_eq!(d.extent, Extent::new(8, 8));

    // Pointwise ROI: the demanded input region equals the requested output region.
    let mut requested = OutputRegions::new();
    requested.insert("image".to_owned(), Rect::new(2, 3, 4, 5));
    let needed = Convert::new()
        .required_inputs(&requested, &inputs, &params)
        .expect("required_inputs");
    assert_eq!(needed["image"], Rect::new(2, 3, 4, 5));

    // Postcondition: the output is encoded to the target.
    let results = Convert::new()
        .validate_postconditions(&out, &params)
        .expect("postconditions");
    assert!(
        results
            .iter()
            .all(|r| r.status == paintop_ir::AssertionStatus::Pass)
    );
}
