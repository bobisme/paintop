//! Verification suite for `alpha.premultiply@1` / `alpha.unpremultiply@1`
//! (`OP_CATALOG` §2, `plan.md` §8.2, `AGENT_VERIFICATION` §3.2):
//!
//! - **schema/contract**: both manifests validate, agree with their contracts,
//!   and their verification declarations gate clean;
//! - **analytic fixtures**: premultiply collapses hidden RGB under zero coverage
//!   to black (no colored fringe); a known value table is reproduced exactly;
//! - **metamorphic**: premultiply ∘ unpremultiply round-trips for every pixel
//!   with `α > ε`;
//! - **property**: premultiplied output obeys `|C'| <= α` for color in `[0,1]`;
//!   alpha is passed through untouched;
//! - **rejection**: a premultiply on `srgb`-encoded input, on an image with no
//!   alpha channel, and on an already-(un)premultiplied image are all rejected
//!   with `semantic` errors.

use paintop_core::executor::{InputValues, OpImplementation, ResourceValue};
use paintop_ir::{
    AlphaRepresentation, ChannelLayout, ColorEncoding, ColorRange, CoordinateConvention,
    Descriptors, ErrorClass, Extent, ImageDescriptor, OpContract, OutputRegions, Rect,
    ResourceDescriptor, ScalarType, SemanticRole, check_contract_consistency, verify_categories,
};

use super::{
    Direction, E_ALPHA_NO_ALPHA, E_ALPHA_REPRESENTATION, PREMULTIPLY_OP_ID, Premultiply,
    UNPREMULTIPLY_EPSILON, UNPREMULTIPLY_OP_ID, Unpremultiply,
};

/// Tolerance for the unpremultiply division round-trip (bounded tier).
const TOL: f32 = 1e-6;

/// Build an RGBA color image value with the given encoding and alpha
/// representation.
fn rgba(
    width: u32,
    height: u32,
    color: ColorEncoding,
    alpha: AlphaRepresentation,
    samples: Vec<f32>,
) -> ResourceValue {
    let descriptor = ResourceDescriptor::Image(ImageDescriptor {
        extent: Extent::new(width, height),
        layout: ChannelLayout::Rgba,
        scalar: ScalarType::F32,
        color,
        range: ColorRange::DisplayReferred,
        alpha,
        coordinates: CoordinateConvention::PixelCenterUpperLeft,
        semantic: SemanticRole::Color,
    });
    ResourceValue::new(descriptor, 4, samples).expect("sample buffer matches descriptor")
}

/// Run premultiply and recover the produced image value.
fn premultiply(value: &ResourceValue) -> ResourceValue {
    let mut inputs = InputValues::new();
    inputs.insert("image".to_owned(), value.clone());
    let mut out = Premultiply::new()
        .compute(&inputs, &serde_json::Value::Null)
        .expect("premultiply computes");
    out.remove("image").expect("image port produced")
}

/// Run unpremultiply and recover the produced image value.
fn unpremultiply(value: &ResourceValue) -> ResourceValue {
    let mut inputs = InputValues::new();
    inputs.insert("image".to_owned(), value.clone());
    let mut out = Unpremultiply::new()
        .compute(&inputs, &serde_json::Value::Null)
        .expect("unpremultiply computes");
    out.remove("image").expect("image port produced")
}

#[test]
fn manifests_validate_and_agree_with_contracts() {
    let pm = Premultiply::manifest().expect("premultiply manifest");
    pm.validate().expect("premultiply manifest valid");
    check_contract_consistency(&pm, &Premultiply::new())
        .expect("premultiply manifest agrees with contract");
    verify_categories(&pm, &pm.test.verification).expect("premultiply declarations gate clean");
    assert_eq!(pm.id.to_string(), PREMULTIPLY_OP_ID);

    let um = Unpremultiply::manifest().expect("unpremultiply manifest");
    um.validate().expect("unpremultiply manifest valid");
    check_contract_consistency(&um, &Unpremultiply::new())
        .expect("unpremultiply manifest agrees with contract");
    verify_categories(&um, &um.test.verification).expect("unpremultiply declarations gate clean");
    assert_eq!(um.id.to_string(), UNPREMULTIPLY_OP_ID);
}

#[test]
fn directions_declare_the_op_ids() {
    assert_eq!(Direction::Premultiply.op_id(), PREMULTIPLY_OP_ID);
    assert_eq!(Direction::Unpremultiply.op_id(), UNPREMULTIPLY_OP_ID);
}

#[test]
fn premultiply_scales_color_and_passes_alpha_through() {
    // One opaque pixel (alpha 1) and one half-coverage pixel.
    let value = rgba(
        2,
        1,
        ColorEncoding::LinearSrgb,
        AlphaRepresentation::Straight,
        vec![0.2, 0.4, 0.6, 1.0, 0.2, 0.4, 0.6, 0.5],
    );
    let out = premultiply(&value);
    let s = out.samples();
    // Opaque pixel: color unchanged (× 1).
    assert!((s[0] - 0.2).abs() < TOL);
    assert!((s[1] - 0.4).abs() < TOL);
    assert!((s[2] - 0.6).abs() < TOL);
    assert_eq!(s[3].to_bits(), 1.0_f32.to_bits(), "alpha untouched");
    // Half pixel: color × 0.5, alpha unchanged.
    assert!((s[4] - 0.1).abs() < TOL);
    assert!((s[5] - 0.2).abs() < TOL);
    assert!((s[6] - 0.3).abs() < TOL);
    assert_eq!(s[7].to_bits(), 0.5_f32.to_bits(), "alpha untouched");

    let ResourceDescriptor::Image(d) = out.descriptor() else {
        panic!("expected image");
    };
    assert_eq!(d.alpha, AlphaRepresentation::Premultiplied);
}

#[test]
fn premultiply_collapses_hidden_rgb_under_zero_coverage() {
    // The §3.2 hidden-RGB fringe fixture: left half opaque visible color, right
    // half transparent with non-zero hidden RGB. After premultiply the hidden
    // color MUST be exactly black so it cannot bleed a colored fringe.
    let value = rgba(
        2,
        1,
        ColorEncoding::LinearSrgb,
        AlphaRepresentation::Straight,
        vec![0.2, 0.4, 0.6, 1.0, 0.9, 0.1, 0.1, 0.0],
    );
    let out = premultiply(&value);
    let s = out.samples();
    // Transparent pixel's premultiplied color is exactly zero (no fringe).
    let zero = 0.0_f32.to_bits();
    assert_eq!(s[4].to_bits(), zero, "hidden R must collapse to 0");
    assert_eq!(s[5].to_bits(), zero, "hidden G must collapse to 0");
    assert_eq!(s[6].to_bits(), zero, "hidden B must collapse to 0");
    assert_eq!(s[7].to_bits(), zero, "alpha stays 0");
}

#[test]
fn premultiplied_output_obeys_color_le_alpha() {
    // Property §3.2: |C'| <= α for color in [0,1].
    let mut samples = Vec::new();
    for i in 0..=10u32 {
        for j in 0..=10u32 {
            #[allow(clippy::cast_precision_loss, reason = "small loop bound")]
            let c = i as f32 / 10.0;
            #[allow(clippy::cast_precision_loss, reason = "small loop bound")]
            let a = j as f32 / 10.0;
            samples.extend_from_slice(&[c, c, c, a]);
        }
    }
    #[allow(clippy::cast_possible_truncation, reason = "121 pixels fits u32")]
    let count = (samples.len() / 4) as u32;
    let value = rgba(
        count,
        1,
        ColorEncoding::LinearSrgb,
        AlphaRepresentation::Straight,
        samples,
    );
    let out = premultiply(&value);
    for pixel in out.samples().chunks(4) {
        let alpha = pixel[3];
        for &color in &pixel[..3] {
            assert!(
                color.abs() <= alpha + TOL,
                "|C'|={} exceeds alpha={alpha}",
                color.abs()
            );
        }
    }
}

#[test]
fn round_trip_recovers_straight_color_where_alpha_above_epsilon() {
    // Metamorphic §2.5 round-trip: unpremultiply ∘ premultiply == identity for
    // α > ε.
    let mut samples = Vec::new();
    for i in 1..=20u32 {
        #[allow(clippy::cast_precision_loss, reason = "small loop bound")]
        let a = i as f32 / 20.0;
        samples.extend_from_slice(&[0.2, 0.5, 0.9, a]);
    }
    #[allow(clippy::cast_possible_truncation, reason = "20 pixels fits u32")]
    let count = (samples.len() / 4) as u32;
    let straight = rgba(
        count,
        1,
        ColorEncoding::LinearSrgb,
        AlphaRepresentation::Straight,
        samples.clone(),
    );
    let pm = premultiply(&straight);
    let back = unpremultiply(&pm);
    for (got, want) in back.samples().iter().zip(samples.iter()) {
        assert!((got - want).abs() < TOL, "round-trip {got} vs {want}");
    }
    let ResourceDescriptor::Image(d) = back.descriptor() else {
        panic!("expected image");
    };
    assert_eq!(d.alpha, AlphaRepresentation::Straight);
}

#[test]
fn unpremultiply_leaves_color_zero_below_epsilon() {
    // A premultiplied transparent pixel stores C'=0; with α below ε the original
    // straight color is unrecoverable, so the policy keeps color at zero rather
    // than dividing by ~0.
    let value = rgba(
        1,
        1,
        ColorEncoding::LinearSrgb,
        AlphaRepresentation::Premultiplied,
        vec![0.0, 0.0, 0.0, UNPREMULTIPLY_EPSILON / 2.0],
    );
    let out = unpremultiply(&value);
    let s = out.samples();
    let zero = 0.0_f32.to_bits();
    assert_eq!(s[0].to_bits(), zero);
    assert_eq!(s[1].to_bits(), zero);
    assert_eq!(s[2].to_bits(), zero);
}

#[test]
fn premultiply_on_srgb_is_rejected_semantically() {
    // §8 rule: premultiplication is only defined in linear light.
    let value = rgba(
        1,
        1,
        ColorEncoding::Srgb,
        AlphaRepresentation::Straight,
        vec![0.2, 0.4, 0.6, 1.0],
    );
    let mut inputs = InputValues::new();
    inputs.insert("image".to_owned(), value);
    let err = Premultiply::new()
        .compute(&inputs, &serde_json::Value::Null)
        .expect_err("srgb premultiply must be rejected");
    assert_eq!(err.class, ErrorClass::Semantic);
    assert_eq!(err.code, E_ALPHA_REPRESENTATION);
}

#[test]
fn premultiply_without_alpha_is_rejected() {
    let descriptor = ResourceDescriptor::Image(ImageDescriptor {
        extent: Extent::new(1, 1),
        layout: ChannelLayout::Rgb,
        scalar: ScalarType::F32,
        color: ColorEncoding::LinearSrgb,
        range: ColorRange::DisplayReferred,
        alpha: AlphaRepresentation::Straight,
        coordinates: CoordinateConvention::PixelCenterUpperLeft,
        semantic: SemanticRole::Color,
    });
    let value = ResourceValue::new(descriptor, 3, vec![0.2, 0.4, 0.6]).expect("value");
    let mut inputs = InputValues::new();
    inputs.insert("image".to_owned(), value);
    let err = Premultiply::new()
        .compute(&inputs, &serde_json::Value::Null)
        .expect_err("no-alpha premultiply must be rejected");
    assert_eq!(err.class, ErrorClass::Semantic);
    assert_eq!(err.code, E_ALPHA_NO_ALPHA);
}

#[test]
fn premultiply_on_already_premultiplied_is_rejected() {
    let value = rgba(
        1,
        1,
        ColorEncoding::LinearSrgb,
        AlphaRepresentation::Premultiplied,
        vec![0.1, 0.2, 0.3, 0.5],
    );
    let mut inputs = InputValues::new();
    inputs.insert("image".to_owned(), value);
    let err = Premultiply::new()
        .compute(&inputs, &serde_json::Value::Null)
        .expect_err("double premultiply must be rejected");
    assert_eq!(err.class, ErrorClass::Semantic);
    assert_eq!(err.code, E_ALPHA_REPRESENTATION);
}

#[test]
fn unpremultiply_on_straight_is_rejected() {
    let value = rgba(
        1,
        1,
        ColorEncoding::LinearSrgb,
        AlphaRepresentation::Straight,
        vec![0.1, 0.2, 0.3, 0.5],
    );
    let mut inputs = InputValues::new();
    inputs.insert("image".to_owned(), value);
    let err = Unpremultiply::new()
        .compute(&inputs, &serde_json::Value::Null)
        .expect_err("unpremultiply on straight must be rejected");
    assert_eq!(err.class, ErrorClass::Semantic);
    assert_eq!(err.code, E_ALPHA_REPRESENTATION);
}

#[test]
fn infer_outputs_and_pointwise_roi() {
    let descriptor = ResourceDescriptor::Image(ImageDescriptor {
        extent: Extent::new(8, 8),
        layout: ChannelLayout::Rgba,
        scalar: ScalarType::F32,
        color: ColorEncoding::LinearSrgb,
        range: ColorRange::DisplayReferred,
        alpha: AlphaRepresentation::Straight,
        coordinates: CoordinateConvention::PixelCenterUpperLeft,
        semantic: SemanticRole::Color,
    });
    let mut inputs = Descriptors::new();
    inputs.insert("image".to_owned(), descriptor);
    let params = serde_json::Value::Null;

    let out = Premultiply::new()
        .infer_outputs(&inputs, &params)
        .expect("infer");
    let ResourceDescriptor::Image(d) = out["image"] else {
        panic!("expected image");
    };
    assert_eq!(d.alpha, AlphaRepresentation::Premultiplied);
    assert_eq!(d.extent, Extent::new(8, 8));

    let mut requested = OutputRegions::new();
    requested.insert("image".to_owned(), Rect::new(2, 3, 4, 5));
    let needed = Premultiply::new()
        .required_inputs(&requested, &inputs, &params)
        .expect("required_inputs");
    assert_eq!(needed["image"], Rect::new(2, 3, 4, 5));

    let results = Premultiply::new()
        .validate_postconditions(&out, &params)
        .expect("postconditions");
    assert!(
        results
            .iter()
            .all(|r| r.status == paintop_ir::AssertionStatus::Pass)
    );
}

/// The checked-in `ops/manifests/<id>.json` files (read by `cargo xtask
/// verify-op`) must stay byte-identical to the Rust manifest builders, the source
/// of truth. Regenerate with `serde_json::to_string_pretty` if this fails.
#[test]
fn checked_in_manifests_match_builders() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("crates dir")
        .parent()
        .expect("repo root")
        .join("ops/manifests");
    for manifest in [
        Premultiply::manifest().expect("premultiply manifest"),
        Unpremultiply::manifest().expect("unpremultiply manifest"),
    ] {
        let path = root.join(format!("{}.json", manifest.id));
        let on_disk = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
        let expected = serde_json::to_string_pretty(&manifest).expect("serialize manifest");
        assert_eq!(
            on_disk.trim_end(),
            expected.trim_end(),
            "{} is stale; regenerate from the Rust builder",
            path.display()
        );
    }
}
