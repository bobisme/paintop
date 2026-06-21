//! Verification suite for `composite.masked_replace@1` (`OP_CATALOG` §7,
//! `IR_SPEC` §9.1, `M0_DECISIONS` D2) — the single MVP authorization boundary:
//!
//! - **schema/contract**: the manifest validates, agrees with its contract, and
//!   its verification declarations gate clean; the checked-in manifest stays in
//!   lockstep with the Rust builder;
//! - **analytic fixtures (mask extremes)**: `m = 0` reproduces `base` bit-exactly,
//!   `m = 1` reproduces `edited` bit-exactly;
//! - **property (the safety-critical one)**: wherever the mask is `0` the output
//!   is bit-identical to `base`, regardless of how different `edited` is;
//! - **property (soft mask)**: an intermediate `m` lerps each channel between
//!   `base` and `edited` and stays between them; the output is finite;
//! - **premultiplied correctness**: a half-coverage blend of two premultiplied
//!   colors is their premultiplied average;
//! - **rejection**: a nonlinear / straight-alpha / no-alpha image, or a
//!   shape/extent mismatch between the ports, is rejected with the right class.

use paintop_core::executor::{InputValues, OpImplementation, ResourceValue};
use paintop_ir::{
    AlphaRepresentation, AssertionStatus, ChannelLayout, ColorEncoding, ColorRange,
    CoordinateConvention, Descriptors, ErrorClass, Extent, ImageDescriptor, MaskDescriptor,
    MaskMeaning, OpContract, OutputRegions, Rect, ResourceDescriptor, ScalarType, SemanticRole,
    ValidRange, check_contract_consistency, verify_categories,
};

use super::{E_MASKED_REPLACE_INPUT, E_MASKED_REPLACE_SHAPE, MASKED_REPLACE_OP_ID, MaskedReplace};

/// A premultiplied linear RGBA image descriptor of side `n`.
fn rgba_descriptor(n: u32) -> ImageDescriptor {
    ImageDescriptor {
        extent: Extent::new(n, n),
        layout: ChannelLayout::Rgba,
        scalar: ScalarType::F32,
        color: ColorEncoding::LinearSrgb,
        range: ColorRange::SceneReferred,
        alpha: AlphaRepresentation::Premultiplied,
        coordinates: CoordinateConvention::PixelCenterUpperLeft,
        semantic: SemanticRole::Color,
    }
}

/// A premultiplied linear RGBA image value of side `n` whose every sample is `fill`.
fn rgba_image(n: u32, fill: f32) -> ResourceValue {
    let descriptor = ResourceDescriptor::Image(rgba_descriptor(n));
    let channels = ChannelLayout::Rgba.channel_count();
    let len = (n as usize) * (n as usize) * channels as usize;
    ResourceValue::new(descriptor, channels, vec![fill; len]).expect("rgba image")
}

/// A premultiplied linear RGBA image value of side `n` from an explicit row-major
/// interleaved buffer.
fn rgba_image_from(n: u32, samples: Vec<f32>) -> ResourceValue {
    let descriptor = ResourceDescriptor::Image(rgba_descriptor(n));
    ResourceValue::new(descriptor, ChannelLayout::Rgba.channel_count(), samples)
        .expect("rgba image")
}

/// A coverage mask value of side `n` from an explicit row-major buffer (one
/// coverage sample per pixel).
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

/// Composite `edited` over `base` through `mask`, returning the produced image.
fn composite(edited: ResourceValue, base: ResourceValue, mask: ResourceValue) -> ResourceValue {
    let mut inputs = InputValues::new();
    inputs.insert("edited".to_owned(), edited);
    inputs.insert("base".to_owned(), base);
    inputs.insert("mask".to_owned(), mask);
    let mut out = MaskedReplace::new()
        .compute(&inputs, &serde_json::Value::Null)
        .expect("masked_replace computes");
    out.remove("image").expect("image port produced")
}

/// The descriptor inputs for `infer_outputs` / `required_inputs`.
fn descriptors(n: u32) -> Descriptors {
    let mut inputs = Descriptors::new();
    inputs.insert(
        "edited".to_owned(),
        ResourceDescriptor::Image(rgba_descriptor(n)),
    );
    inputs.insert(
        "base".to_owned(),
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
    let manifest = MaskedReplace::manifest().expect("masked_replace manifest");
    manifest.validate().expect("manifest valid");
    check_contract_consistency(&manifest, &MaskedReplace::new())
        .expect("manifest agrees with contract");
    verify_categories(&manifest, &manifest.test.verification)
        .expect("verification declarations gate clean");
    assert_eq!(manifest.id.to_string(), MASKED_REPLACE_OP_ID);
}

#[test]
fn produces_image_descriptor_matching_base() {
    let out = composite(rgba_image(8, 1.0), rgba_image(8, 0.0), const_mask(8, 0.5));
    let ResourceDescriptor::Image(d) = out.descriptor() else {
        panic!("expected image output");
    };
    assert_eq!(d.extent, Extent::new(8, 8));
    assert_eq!(d.alpha, AlphaRepresentation::Premultiplied);
    assert_eq!(d.color, ColorEncoding::LinearSrgb);
    assert_eq!(out.channels(), 4);
    assert_eq!(out.samples().len(), 8 * 8 * 4);
}

/// Analytic fixture: `m = 0` reproduces `base` bit-exactly (the
/// no-change-outside-mask invariant at the extreme).
#[test]
fn mask_zero_is_base_bit_exact() {
    let base = rgba_image(8, 0.375);
    let before: Vec<f32> = base.samples().to_vec();
    // The edited image is deliberately *very* different to prove only the mask
    // gates the result.
    let out = composite(rgba_image(8, 0.9), base, const_mask(8, 0.0));
    assert_eq!(out.samples(), before.as_slice());
}

/// Analytic fixture: `m = 1` reproduces `edited` bit-exactly.
#[test]
fn mask_one_is_edited_bit_exact() {
    let edited = rgba_image(8, 0.625);
    let before: Vec<f32> = edited.samples().to_vec();
    let out = composite(edited, rgba_image(8, 0.1), const_mask(8, 1.0));
    assert_eq!(out.samples(), before.as_slice());
}

/// The safety-critical property: wherever the mask is `0`, the output is
/// bit-identical to `base` — even for an arbitrary, wildly different `edited` and a
/// per-pixel mask (some pixels masked in, some out). This is exactly what
/// `assert.no_change_outside_mask@1` verifies against.
#[test]
fn outside_mask_is_bit_identical_to_base() {
    let n = 6;
    let pixels = (n * n) as usize;
    // A non-trivial base and a different edited, both premultiplied-plausible.
    // Values are derived from a lossless `u8 -> f32` cast so the buffer is exact.
    let base_samples: Vec<f32> = (0..pixels * 4)
        .map(|i| f32::from(u8::try_from(i % 251).unwrap()) / 251.0)
        .collect();
    let edited_samples: Vec<f32> = (0..pixels * 4)
        .map(|i| 1.0 - f32::from(u8::try_from(i % 239).unwrap()) / 239.0)
        .collect();
    // Checkerboard mask: even pixels masked in (m=1), odd pixels out (m=0).
    let mask_samples: Vec<f32> = (0..pixels)
        .map(|i| if i % 2 == 0 { 1.0 } else { 0.0 })
        .collect();

    let base = rgba_image_from(n, base_samples.clone());
    let edited = rgba_image_from(n, edited_samples);
    let out = composite(edited, base, mask_image(n, mask_samples.clone()));
    let result = out.samples();

    for (pixel, &m) in mask_samples.iter().enumerate() {
        if m.to_bits() == 0.0_f32.to_bits() {
            for c in 0..4 {
                let idx = pixel * 4 + c;
                // Bit-identical: outside the mask the base passes through verbatim.
                assert_eq!(
                    result[idx].to_bits(),
                    base_samples[idx].to_bits(),
                    "pixel {pixel} channel {c} changed outside the mask"
                );
            }
        }
    }
}

/// Soft-mask blend property: at intermediate coverage every output channel lies
/// between `base` and `edited`, equals the analytic lerp, and stays finite.
#[test]
fn soft_mask_lerps_between_base_and_edited() {
    let n = 4;
    let pixels = (n * n) as usize;
    let base_val = 0.2_f32;
    let edited_val = 0.8_f32;
    let m = 0.25_f32;
    let base = rgba_image(n, base_val);
    let edited = rgba_image(n, edited_val);
    let out = composite(edited, base, const_mask(n, m));

    let expected = m.mul_add(edited_val - base_val, base_val);
    for &s in out.samples() {
        assert!(s.is_finite(), "sample must be finite, got {s}");
        assert!(
            base_val <= s && s <= edited_val,
            "sample {s} not between base {base_val} and edited {edited_val}"
        );
        assert!((s - expected).abs() < 1e-6, "sample {s} != lerp {expected}");
    }
    assert_eq!(out.samples().len(), pixels * 4);
}

/// Premultiplied correctness: a half-coverage blend of two premultiplied colors
/// is their premultiplied average, channel-for-channel (including alpha).
#[test]
fn half_coverage_is_premultiplied_average() {
    // Premultiplied: a 50%-opaque red over a 100%-opaque blue.
    let edited = rgba_image_from(1, vec![0.5, 0.0, 0.0, 0.5]);
    let base = rgba_image_from(1, vec![0.0, 0.0, 1.0, 1.0]);
    let out = composite(edited, base, const_mask(1, 0.5));
    let got = out.samples();
    let expected = [0.25, 0.0, 0.5, 0.75];
    for (g, e) in got.iter().zip(expected.iter()) {
        assert!((g - e).abs() < 1e-6, "got {g}, expected {e}");
    }
}

/// `infer_outputs` returns the base descriptor and `required_inputs` is pointwise
/// on all three ports.
#[test]
fn contract_infers_base_and_pointwise_roi() {
    let inputs = descriptors(10);
    let outputs = MaskedReplace::new()
        .infer_outputs(&inputs, &serde_json::Value::Null)
        .expect("infer");
    let ResourceDescriptor::Image(d) = outputs.get("image").expect("image out") else {
        panic!("expected image");
    };
    assert_eq!(d.extent, Extent::new(10, 10));

    let region = Rect::new(2, 3, 5, 6);
    let mut requested = OutputRegions::new();
    requested.insert("image".to_owned(), region);
    let regions = MaskedReplace::new()
        .required_inputs(&requested, &inputs, &serde_json::Value::Null)
        .expect("required inputs");
    for port in ["edited", "base", "mask"] {
        assert_eq!(regions.get(port), Some(&region), "port {port} ROI");
    }
}

/// The postcondition asserts the composited result stays premultiplied linear.
#[test]
fn postcondition_checks_premultiplied_linear() {
    let inputs = descriptors(4);
    let outputs = MaskedReplace::new()
        .infer_outputs(&inputs, &serde_json::Value::Null)
        .expect("infer");
    let results = MaskedReplace::new()
        .validate_postconditions(&outputs, &serde_json::Value::Null)
        .expect("postconditions");
    assert!(
        results.iter().all(|r| r.status == AssertionStatus::Pass),
        "all postconditions pass: {results:?}"
    );
}

/// A nonlinear (srgb) input is rejected: the blend is only correct in linear light.
#[test]
fn srgb_input_is_rejected() {
    let mut srgb = rgba_descriptor(4);
    srgb.color = ColorEncoding::Srgb;
    let edited = ResourceValue::new(ResourceDescriptor::Image(srgb), 4, vec![0.5; 4 * 4 * 4])
        .expect("srgb image");
    let mut inputs = InputValues::new();
    inputs.insert("edited".to_owned(), edited);
    inputs.insert("base".to_owned(), rgba_image(4, 0.0));
    inputs.insert("mask".to_owned(), const_mask(4, 0.5));
    let err = MaskedReplace::new()
        .compute(&inputs, &serde_json::Value::Null)
        .expect_err("srgb must be rejected");
    assert_eq!(err.class, ErrorClass::Semantic);
    assert_eq!(err.code, E_MASKED_REPLACE_SHAPE);
}

/// A straight-alpha input is rejected: compositing happens in premultiplied space.
#[test]
fn straight_alpha_input_is_rejected() {
    let mut straight = rgba_descriptor(4);
    straight.alpha = AlphaRepresentation::Straight;
    let base = ResourceValue::new(ResourceDescriptor::Image(straight), 4, vec![0.5; 4 * 4 * 4])
        .expect("straight image");
    let mut inputs = InputValues::new();
    inputs.insert("edited".to_owned(), rgba_image(4, 0.0));
    inputs.insert("base".to_owned(), base);
    inputs.insert("mask".to_owned(), const_mask(4, 0.5));
    let err = MaskedReplace::new()
        .compute(&inputs, &serde_json::Value::Null)
        .expect_err("straight alpha must be rejected");
    assert_eq!(err.class, ErrorClass::Semantic);
    assert_eq!(err.code, E_MASKED_REPLACE_SHAPE);
}

/// A mask whose extent differs from the images is rejected.
#[test]
fn mask_extent_mismatch_is_rejected() {
    let mut inputs = InputValues::new();
    inputs.insert("edited".to_owned(), rgba_image(4, 0.0));
    inputs.insert("base".to_owned(), rgba_image(4, 0.5));
    inputs.insert("mask".to_owned(), const_mask(2, 0.5));
    let err = MaskedReplace::new()
        .compute(&inputs, &serde_json::Value::Null)
        .expect_err("mask extent mismatch must be rejected");
    assert_eq!(err.class, ErrorClass::Semantic);
    assert_eq!(err.code, E_MASKED_REPLACE_SHAPE);
}

/// `edited` and `base` with differing extents are rejected.
#[test]
fn edited_base_extent_mismatch_is_rejected() {
    let mut inputs = InputValues::new();
    inputs.insert("edited".to_owned(), rgba_image(8, 0.0));
    inputs.insert("base".to_owned(), rgba_image(4, 0.5));
    inputs.insert("mask".to_owned(), const_mask(4, 0.5));
    let err = MaskedReplace::new()
        .compute(&inputs, &serde_json::Value::Null)
        .expect_err("extent mismatch must be rejected");
    assert_eq!(err.class, ErrorClass::Semantic);
    assert_eq!(err.code, E_MASKED_REPLACE_SHAPE);
}

/// A missing required port is a reference error.
#[test]
fn missing_port_is_rejected() {
    let mut inputs = InputValues::new();
    inputs.insert("edited".to_owned(), rgba_image(4, 0.0));
    inputs.insert("base".to_owned(), rgba_image(4, 0.5));
    // No `mask`.
    let err = MaskedReplace::new()
        .compute(&inputs, &serde_json::Value::Null)
        .expect_err("missing mask must be rejected");
    assert_eq!(err.class, ErrorClass::Reference);
    assert_eq!(err.code, E_MASKED_REPLACE_INPUT);
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
    let manifest = MaskedReplace::manifest().expect("masked_replace manifest");
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
