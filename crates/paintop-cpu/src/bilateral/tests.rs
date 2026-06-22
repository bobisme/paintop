//! Verification suite for `filter.bilateral@1` (`OP_CATALOG` §8):
//!
//! - **schema/contract**: the manifest validates, agrees with its contract, and
//!   gates clean;
//! - **constant-image identity**: a flat image is reproduced exactly;
//! - **edge preservation**: a step edge guided by its own intensity stays a step
//!   (flats either side are smoothed, the boundary is not blurred across);
//! - **independent reference**: the weighted average matches a brute-force direct
//!   reference within tolerance;
//! - **large-range limit**: with a huge `range_sigma` the filter reduces to a
//!   (clamped, normalized) spatial Gaussian average;
//! - **determinism**: a rerun is bit-identical;
//! - **rejection**: missing / non-positive sigmas and an over-limit spatial sigma
//!   are rejected.

use paintop_core::executor::{InputValues, OpImplementation, ResourceValue};
use paintop_ir::{
    AlphaRepresentation, ChannelLayout, ColorEncoding, ColorRange, CoordinateConvention,
    ErrorClass, Extent, ImageDescriptor, ResourceDescriptor, ScalarType, SemanticRole,
};

use super::{Bilateral, BilateralRequest, FLAT_IDENTITY_TOLERANCE, REFERENCE_TOLERANCE};

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

/// Run the bilateral filter and recover the output.
fn run(value: &ResourceValue, params: &serde_json::Value) -> ResourceValue {
    let mut inputs = InputValues::new();
    inputs.insert("input".to_owned(), value.clone());
    let mut out = Bilateral::new()
        .compute(&inputs, params)
        .expect("bilateral computes");
    out.remove("output").expect("output port produced")
}

/// An independent, brute-force single-channel bilateral reference under clamp.
fn reference(
    plane: &[f64],
    width: usize,
    height: usize,
    spatial_sigma: f64,
    range_sigma: f64,
) -> Vec<f64> {
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "3*sigma is a small positive bounded radius"
    )]
    let radius = ((3.0 * spatial_sigma).ceil() as usize).max(1);
    let clamp_idx = |coord: usize, delta: usize, sub: bool, last: usize| -> usize {
        if sub {
            coord.saturating_sub(delta)
        } else {
            (coord + delta).min(last)
        }
    };
    let mut out = vec![0.0_f64; width * height];
    for cy in 0..height {
        for cx in 0..width {
            let centre = plane[cy * width + cx];
            let (mut wsum, mut vsum) = (0.0_f64, 0.0_f64);
            for dy in 0..=(2 * radius) {
                let sub_y = dy < radius;
                let off_y = if sub_y { radius - dy } else { dy - radius };
                let sy = clamp_idx(cy, off_y, sub_y, height - 1);
                for dx in 0..=(2 * radius) {
                    let sub_x = dx < radius;
                    let off_x = if sub_x { radius - dx } else { dx - radius };
                    let sx = clamp_idx(cx, off_x, sub_x, width - 1);
                    let neighbour = plane[sy * width + sx];
                    let dxs = i64::try_from(dx).unwrap_or(0) - i64::try_from(radius).unwrap_or(0);
                    let dys = i64::try_from(dy).unwrap_or(0) - i64::try_from(radius).unwrap_or(0);
                    #[allow(clippy::cast_precision_loss, reason = "small window offsets")]
                    let spatial_d2 = (dxs * dxs + dys * dys) as f64;
                    let spatial = (-spatial_d2 / (2.0 * spatial_sigma * spatial_sigma)).exp();
                    let rd = neighbour - centre;
                    let range = (-(rd * rd) / (2.0 * range_sigma * range_sigma)).exp();
                    let w = spatial * range;
                    wsum += w;
                    vsum += w * neighbour;
                }
            }
            out[cy * width + cx] = vsum / wsum;
        }
    }
    out
}

/// A box (top-hat) plain spatial-Gaussian average reference (range term ≡ 1).
fn spatial_only(plane: &[f64], width: usize, height: usize, spatial_sigma: f64) -> Vec<f64> {
    reference(plane, width, height, spatial_sigma, 1.0e9)
}

#[test]
fn manifest_validates_and_agrees_with_contract() {
    let manifest = Bilateral::manifest().expect("manifest");
    manifest.validate().expect("manifest valid");
    paintop_ir::check_contract_consistency(&manifest, &Bilateral::new())
        .expect("manifest agrees with contract");
    paintop_ir::verify_categories(&manifest, &manifest.test.verification)
        .expect("verification declarations gate clean");
    assert_eq!(manifest.id.to_string(), super::BILATERAL_OP_ID);
}

#[test]
fn constant_image_is_preserved_exactly() {
    let img = gray(9, 9, vec![0.6; 81]);
    let out = run(
        &img,
        &serde_json::json!({ "spatial_sigma": 2.0, "range_sigma": 0.1 }),
    );
    for &s in out.samples() {
        assert!(
            (s - 0.6).abs() < FLAT_IDENTITY_TOLERANCE,
            "constant not preserved: {s}"
        );
    }
}

#[test]
fn step_edge_is_preserved() {
    // A vertical step (left half 0.0, right half 1.0). A small range sigma keeps
    // the two sides apart, so the edge stays sharp.
    let w = 16usize;
    let h = 6usize;
    let mut samples = Vec::with_capacity(w * h);
    for _y in 0..h {
        for x in 0..w {
            samples.push(if x < w / 2 { 0.0 } else { 1.0 });
        }
    }
    let img = gray(16, 6, samples);
    let out = run(
        &img,
        &serde_json::json!({ "spatial_sigma": 2.0, "range_sigma": 0.05 }),
    );
    let s = out.samples();
    // Flats far from the edge are preserved; the jump across the boundary is large.
    assert!(f64::from(s[3 * w + 2]) < 0.02, "left flat preserved");
    assert!(f64::from(s[3 * w + (w - 3)]) > 0.98, "right flat preserved");
    let jump = f64::from(s[3 * w + (w / 2)]) - f64::from(s[3 * w + (w / 2 - 1)]);
    assert!(jump > 0.8, "step edge preserved (jump {jump})");
}

#[test]
fn matches_independent_reference() {
    let width = 9usize;
    let height = 7usize;
    let coord = |index: usize| f64::from(u32::try_from(index).unwrap_or(0));
    let mut samples = Vec::with_capacity(width * height);
    for y in 0..height {
        for x in 0..width {
            let v = coord(x)
                .mul_add(0.13, coord(y) * 0.07)
                .sin()
                .mul_add(0.5, 0.5);
            #[allow(clippy::cast_possible_truncation, reason = "bounded test sample")]
            samples.push(v as f32);
        }
    }
    let img = gray(9, 7, samples.clone());
    let spatial = 1.5;
    let range = 0.2;
    let out = run(
        &img,
        &serde_json::json!({ "spatial_sigma": spatial, "range_sigma": range }),
    );
    let plane: Vec<f64> = samples.iter().map(|&s| f64::from(s)).collect();
    let want = reference(&plane, width, height, spatial, range);
    for (idx, (got, expect)) in out.samples().iter().zip(want.iter()).enumerate() {
        assert!(
            (f64::from(*got) - expect).abs() < REFERENCE_TOLERANCE,
            "pixel {idx}: bilateral {got} != reference {expect}"
        );
    }
}

#[test]
fn large_range_sigma_reduces_to_spatial_gaussian() {
    let width = 8usize;
    let height = 8usize;
    let samples: Vec<f32> = (0..64u16).map(|v| f32::from(v) / 64.0).collect();
    let img = gray(8, 8, samples.clone());
    let spatial = 1.5;
    let out = run(
        &img,
        &serde_json::json!({ "spatial_sigma": spatial, "range_sigma": 1.0e9 }),
    );
    let plane: Vec<f64> = samples.iter().map(|&s| f64::from(s)).collect();
    let want = spatial_only(&plane, width, height, spatial);
    for (got, expect) in out.samples().iter().zip(want.iter()) {
        assert!(
            (f64::from(*got) - expect).abs() < REFERENCE_TOLERANCE,
            "large range sigma should be a spatial Gaussian: {got} vs {expect}"
        );
    }
}

#[test]
fn rerun_is_bit_identical() {
    let samples: Vec<f32> = (0..64u16).map(|v| f32::from(v) / 64.0).collect();
    let img = gray(8, 8, samples);
    let params = serde_json::json!({ "spatial_sigma": 2.0, "range_sigma": 0.3 });
    let a = run(&img, &params);
    let b = run(&img, &params);
    assert_eq!(a.samples(), b.samples(), "bilateral must be deterministic");
}

#[test]
fn missing_or_invalid_sigmas_are_rejected() {
    assert!(BilateralRequest::resolve(&serde_json::json!({ "range_sigma": 0.1 })).is_err());
    assert!(BilateralRequest::resolve(&serde_json::json!({ "spatial_sigma": 1.0 })).is_err());
    assert!(
        BilateralRequest::resolve(&serde_json::json!({ "spatial_sigma": 0.0, "range_sigma": 0.1 }))
            .is_err()
    );
    assert!(
        BilateralRequest::resolve(
            &serde_json::json!({ "spatial_sigma": 1.0, "range_sigma": -0.1 })
        )
        .is_err()
    );
}

#[test]
fn over_limit_spatial_sigma_is_rejected() {
    let img = gray(4, 4, vec![0.0; 16]);
    let mut inputs = InputValues::new();
    inputs.insert("input".to_owned(), img);
    let err = Bilateral::new()
        .compute(
            &inputs,
            &serde_json::json!({ "spatial_sigma": 1000.0, "range_sigma": 0.1 }),
        )
        .expect_err("an over-limit spatial sigma must be rejected");
    assert_eq!(err.class, ErrorClass::Policy);
}

#[test]
fn wrong_input_kind_is_rejected() {
    let mask = ResourceValue::new(
        ResourceDescriptor::Mask(paintop_ir::MaskDescriptor {
            extent: Extent::new(2, 2),
            scalar: ScalarType::F32,
            range: paintop_ir::ValidRange::Bounded { min: 0.0, max: 1.0 },
            meaning: paintop_ir::MaskMeaning::Coverage,
            coordinates: CoordinateConvention::PixelCenterUpperLeft,
        }),
        1,
        vec![0.0; 4],
    )
    .expect("mask value");
    let mut inputs = InputValues::new();
    inputs.insert("input".to_owned(), mask);
    let err = Bilateral::new()
        .compute(
            &inputs,
            &serde_json::json!({ "spatial_sigma": 1.0, "range_sigma": 0.1 }),
        )
        .expect_err("a Mask input must be rejected");
    assert_eq!(err.code, super::E_BILATERAL_INPUT);
}
