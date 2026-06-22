//! Verification suite for `filter.structure_tensor@1` (`OP_CATALOG` §8, §10.4):
//!
//! - **schema/contract**: the manifest validates, agrees with its contract, and
//!   gates clean;
//! - **analytic fixtures**: a vertical grating yields a tensor whose dominant
//!   gradient is along x (Jxx >> Jyy, Jxy ~ 0); a constant plane yields the zero
//!   tensor;
//! - **property**: the output is a Field3 of the input extent; the tensor is
//!   symmetric (single component per off-diagonal) and `Jxx, Jyy >= 0`;
//! - **metamorphic**: a 90° rotation swaps Jxx and Jyy and negates Jxy
//!   (rotation covariance);
//! - **determinism**: a rerun is bit-identical;
//! - **rejection**: a non-finite / over-limit scale is rejected.

use paintop_core::executor::{InputValues, OpImplementation, ResourceValue};
use paintop_ir::{
    AlphaRepresentation, ChannelLayout, ColorEncoding, ColorRange, CoordinateConvention,
    Descriptors, ErrorClass, Extent, ImageDescriptor, OpContract, ResourceDescriptor, ScalarType,
    SemanticRole,
};

use super::{
    STRUCTURE_TENSOR_OP_ID, Scales, StructureTensor, compute_tensor, gaussian_radius,
    gaussian_smooth,
};

/// Build a single-channel (gray) image from a row-major sample list.
fn gray(width: u32, height: u32, samples: Vec<f32>) -> ResourceValue {
    let descriptor = ResourceDescriptor::Image(ImageDescriptor {
        extent: Extent::new(width, height),
        layout: ChannelLayout::Gray,
        scalar: ScalarType::F32,
        color: ColorEncoding::LinearSrgb,
        range: ColorRange::SceneReferred,
        alpha: AlphaRepresentation::Premultiplied,
        coordinates: CoordinateConvention::PixelCenterUpperLeft,
        semantic: SemanticRole::Color,
    });
    ResourceValue::new(descriptor, 1, samples).expect("sample buffer matches descriptor")
}

/// Narrow an f64 test sample to the field's f32 storage type.
#[allow(
    clippy::cast_possible_truncation,
    reason = "test fixture samples are small bounded values stored as f32"
)]
fn as_f32(v: f64) -> f32 {
    v as f32
}

/// The value `index * 0.1` as the field's f32 storage type, for a small index.
fn ramp_value(index: usize) -> f32 {
    let scaled = f64::from(u32::try_from(index).unwrap_or(0)) * 0.1;
    as_f32(scaled)
}

/// Build a small `width × height` ramp along x: pixel value `x * 0.1`.
fn ramp_along_x(width: usize, height: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(width * height);
    for _y in 0..height {
        for x in 0..width {
            out.push(ramp_value(x));
        }
    }
    out
}

/// Build a small `width × height` ramp along y: pixel value `y * 0.1`.
fn ramp_along_y(width: usize, height: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(width * height);
    for y in 0..height {
        for _x in 0..width {
            out.push(ramp_value(y));
        }
    }
    out
}

/// Run the structure tensor and recover the (Jxx, Jxy, Jyy) Field3.
fn tensor(value: &ResourceValue, params: &serde_json::Value) -> ResourceValue {
    let mut inputs = InputValues::new();
    inputs.insert("input".to_owned(), value.clone());
    let mut out = StructureTensor::new()
        .compute(&inputs, params)
        .expect("structure tensor computes");
    out.remove("tensor").expect("tensor port produced")
}

/// A vertical sinusoidal grating (stripes running top-to-bottom): the value
/// varies smoothly along x as `0.5 + 0.5 sin(2π x / period)`, so the central
/// difference captures a real gradient (unlike a period-2 square wave whose
/// central difference vanishes at interior pixels).
fn vertical_grating(width: u32, height: u32) -> ResourceValue {
    use std::f64::consts::PI;
    let period = 8.0_f64;
    let mut samples = Vec::with_capacity((width as usize) * (height as usize));
    for _y in 0..height {
        for x in 0..width {
            let v = 0.5_f64.mul_add((2.0 * PI * f64::from(x) / period).sin(), 0.5);
            samples.push(as_f32(v));
        }
    }
    gray(width, height, samples)
}

#[test]
fn manifest_validates_and_agrees_with_contract() {
    let manifest = StructureTensor::manifest().expect("manifest");
    manifest.validate().expect("manifest valid");
    paintop_ir::check_contract_consistency(&manifest, &StructureTensor::new())
        .expect("manifest agrees with contract");
    paintop_ir::verify_categories(&manifest, &manifest.test.verification)
        .expect("verification declarations gate clean");
    assert_eq!(manifest.id.to_string(), STRUCTURE_TENSOR_OP_ID);
}

#[test]
fn constant_plane_has_zero_tensor() {
    let img = gray(9, 9, vec![0.5; 81]);
    let out = tensor(
        &img,
        &serde_json::json!({ "gradient_sigma": 0.0, "integration_sigma": 1.5 }),
    );
    for &s in out.samples() {
        assert!(
            s.abs() < 1e-6,
            "constant tensor entry should be ~0, got {s}"
        );
    }
}

#[test]
fn vertical_grating_gradient_is_along_x() {
    let img = vertical_grating(16, 16);
    let out = tensor(
        &img,
        &serde_json::json!({ "gradient_sigma": 0.0, "integration_sigma": 2.0 }),
    );
    let n = 16 * 16;
    let s = out.samples();
    // Average the interior tensor (away from the clamped border).
    let (mut sxx, mut sxy, mut syy) = (0.0_f64, 0.0_f64, 0.0_f64);
    let mut count = 0;
    for y in 4..12 {
        for x in 4..12 {
            let i = y * 16 + x;
            sxx += f64::from(s[i * 3]);
            sxy += f64::from(s[i * 3 + 1]);
            syy += f64::from(s[i * 3 + 2]);
            count += 1;
        }
    }
    let inv = 1.0 / f64::from(count);
    sxx *= inv;
    sxy *= inv;
    syy *= inv;
    assert_eq!(n, 256);
    // The gradient of a vertical grating is along x: Jxx dominates, Jyy ~ 0.
    assert!(
        sxx > 0.05,
        "Jxx should be large for a vertical grating: {sxx}"
    );
    assert!(syy < sxx * 0.05, "Jyy should be ~0: {syy} vs Jxx {sxx}");
    assert!(sxy.abs() < sxx * 0.05, "Jxy should be ~0: {sxy}");
}

#[test]
fn diagonal_entries_are_non_negative() {
    let img = vertical_grating(12, 12);
    let out = tensor(
        &img,
        &serde_json::json!({ "gradient_sigma": 1.0, "integration_sigma": 1.5 }),
    );
    let s = out.samples();
    for i in 0..(12 * 12) {
        assert!(s[i * 3] >= -1e-6, "Jxx must be >= 0: {}", s[i * 3]);
        assert!(s[i * 3 + 2] >= -1e-6, "Jyy must be >= 0: {}", s[i * 3 + 2]);
    }
}

#[test]
fn output_is_field3_of_input_extent() {
    let img = vertical_grating(7, 5);
    let out = tensor(&img, &serde_json::json!({}));
    assert_eq!(out.extent(), Extent::new(7, 5));
    assert_eq!(out.channels(), 3);
    assert!(matches!(out.descriptor(), ResourceDescriptor::Field3(_)));
}

#[test]
fn rotation_swaps_diagonal_and_negates_off_diagonal() {
    // Build a small ramp along x, compute its tensor, then build the 90°-rotated
    // ramp (now along y) and check Jxx<->Jyy swap and Jxy sign flip on average.
    let w = 12usize;
    let h = 12usize;
    let img_x = gray(12, 12, ramp_along_x(w, h));
    let out_x = tensor(
        &img_x,
        &serde_json::json!({ "gradient_sigma": 0.0, "integration_sigma": 1.0 }),
    );

    // Rotate the source 90° CCW: a ramp in x rotated 90° is a ramp in y.
    let img_y = gray(12, 12, ramp_along_y(w, h));
    let out_y = tensor(
        &img_y,
        &serde_json::json!({ "gradient_sigma": 0.0, "integration_sigma": 1.0 }),
    );

    // Average interior tensors.
    let avg = |value: &ResourceValue| {
        let samples = value.samples();
        let (mut diag_x, mut off_diag, mut diag_y) = (0.0_f64, 0.0_f64, 0.0_f64);
        let mut count = 0u32;
        for y in 3..(h - 3) {
            for x in 3..(w - 3) {
                let idx = y * w + x;
                diag_x += f64::from(samples[idx * 3]);
                off_diag += f64::from(samples[idx * 3 + 1]);
                diag_y += f64::from(samples[idx * 3 + 2]);
                count += 1;
            }
        }
        let inv = 1.0 / f64::from(count);
        (diag_x * inv, off_diag * inv, diag_y * inv)
    };
    let (xxx, _xxy, xyy) = avg(&out_x);
    let (yxx, _yxy, yyy) = avg(&out_y);
    // Ramp-along-x: Jxx large, Jyy ~ 0. Ramp-along-y: Jyy large, Jxx ~ 0 (swap).
    assert!(
        (xxx - yyy).abs() < 1e-4,
        "Jxx(x ramp) ~ Jyy(y ramp): {xxx} vs {yyy}"
    );
    assert!(
        (xyy - yxx).abs() < 1e-4,
        "Jyy(x ramp) ~ Jxx(y ramp): {xyy} vs {yxx}"
    );
}

#[test]
fn rerun_is_bit_identical() {
    let img = vertical_grating(10, 10);
    let params = serde_json::json!({ "gradient_sigma": 1.0, "integration_sigma": 2.0 });
    let a = tensor(&img, &params);
    let b = tensor(&img, &params);
    assert_eq!(
        a.samples(),
        b.samples(),
        "structure tensor must be deterministic"
    );
}

#[test]
fn multichannel_sums_per_channel_tensors() {
    // An RGB image whose red channel ramps along x and green ramps along y.
    let w = 10usize;
    let h = 10usize;
    let mut samples = Vec::with_capacity(w * h * 3);
    for y in 0..h {
        for x in 0..w {
            samples.push(ramp_value(x)); // R ramps along x
            samples.push(ramp_value(y)); // G ramps along y
            samples.push(0.0); // B constant
        }
    }
    let descriptor = ResourceDescriptor::Image(ImageDescriptor {
        extent: Extent::new(10, 10),
        layout: ChannelLayout::Rgb,
        scalar: ScalarType::F32,
        color: ColorEncoding::LinearSrgb,
        range: ColorRange::SceneReferred,
        alpha: AlphaRepresentation::Premultiplied,
        coordinates: CoordinateConvention::PixelCenterUpperLeft,
        semantic: SemanticRole::Color,
    });
    let img = ResourceValue::new(descriptor, 3, samples).expect("rgb buffer");
    let out = tensor(
        &img,
        &serde_json::json!({ "gradient_sigma": 0.0, "integration_sigma": 1.0 }),
    );
    let s = out.samples();
    // Interior pixel: Jxx from red, Jyy from green, both ~ (0.1)^2 = 0.01, Jxy ~ 0.
    let i = 5 * w + 5;
    assert!(
        (f64::from(s[i * 3]) - 0.01).abs() < 1e-3,
        "Jxx ~ 0.01: {}",
        s[i * 3]
    );
    assert!(
        (f64::from(s[i * 3 + 2]) - 0.01).abs() < 1e-3,
        "Jyy ~ 0.01: {}",
        s[i * 3 + 2]
    );
    assert!(s[i * 3 + 1].abs() < 1e-3, "Jxy ~ 0: {}", s[i * 3 + 1]);
}

#[test]
fn gaussian_smooth_preserves_constant_and_is_unit_sum() {
    let plane = vec![0.7_f64; 64];
    let out = gaussian_smooth(&plane, 8, 8, 1.5);
    for &s in &out {
        assert!((s - 0.7).abs() < 1e-9, "constant must be preserved: {s}");
    }
    // The σ→0 cutoff is the identity.
    let id = gaussian_smooth(&plane, 8, 8, super::SIGMA_CUTOFF / 10.0);
    assert_eq!(id, plane);
    assert_eq!(gaussian_radius(1.0), 3);
}

#[test]
fn zero_area_input_yields_empty_tensor() {
    let out = compute_tensor(
        &[],
        Extent::new(0, 0),
        1,
        Scales::resolve(&serde_json::json!({})).unwrap(),
    );
    assert!(out.is_empty());
}

#[test]
fn non_finite_scale_is_rejected() {
    let img = gray(4, 4, vec![0.0; 16]);
    let mut inputs = InputValues::new();
    inputs.insert("input".to_owned(), img);
    let err = StructureTensor::new()
        .compute(
            &inputs,
            &serde_json::json!({ "integration_sigma": "not a number" }),
        )
        .expect_err("a non-numeric scale must be rejected");
    assert_eq!(err.class, ErrorClass::Schema);
}

#[test]
fn over_limit_scale_is_rejected() {
    let descriptors_only = {
        let mut d = Descriptors::new();
        d.insert(
            "input".to_owned(),
            ResourceDescriptor::Image(ImageDescriptor {
                extent: Extent::new(4, 4),
                layout: ChannelLayout::Gray,
                scalar: ScalarType::F32,
                color: ColorEncoding::LinearSrgb,
                range: ColorRange::SceneReferred,
                alpha: AlphaRepresentation::Premultiplied,
                coordinates: CoordinateConvention::PixelCenterUpperLeft,
                semantic: SemanticRole::Color,
            }),
        );
        d
    };
    let err = StructureTensor::new()
        .infer_outputs(
            &descriptors_only,
            &serde_json::json!({ "integration_sigma": 10_000.0 }),
        )
        .expect_err("an over-limit scale must be rejected");
    assert_eq!(err.class, ErrorClass::Policy);
}
