//! Verification suite for `filter.guided@1` (`OP_CATALOG` §8):
//!
//! - **schema/contract**: the manifest validates, agrees with its contract, and
//!   gates clean;
//! - **flat-input identity**: a constant input is reproduced exactly regardless
//!   of the guide;
//! - **flat-guide degeneracy**: a constant guide makes the output the input
//!   passed through the box mean twice (`a = 0`, `b = mean_p`, then `b` is
//!   box-averaged);
//! - **independent reference**: the per-window linear model matches a brute-force
//!   ridge-regression reference within tolerance;
//! - **edge preservation (self-guided)**: a step edge guided by itself stays a
//!   step (the edge transition is preserved, flats are smoothed);
//! - **broadcast guide**: a single-channel guide filters every input channel;
//! - **determinism**: a rerun is bit-identical;
//! - **rejection**: a missing / non-positive radius, a negative epsilon, and a
//!   mismatched extent are rejected.

use paintop_core::executor::{InputValues, OpImplementation, ResourceValue};
use paintop_ir::{
    AlphaRepresentation, ChannelLayout, ColorEncoding, ColorRange, CoordinateConvention,
    ErrorClass, Extent, ImageDescriptor, ResourceDescriptor, ScalarType, SemanticRole,
};

use super::{FLAT_IDENTITY_TOLERANCE, Guided, GuidedRequest, REFERENCE_TOLERANCE, box_mean};

/// Build a `channels`-layout image from a row-major interleaved sample list.
fn image(width: u32, height: u32, layout: ChannelLayout, samples: Vec<f32>) -> ResourceValue {
    let descriptor = ResourceDescriptor::Image(ImageDescriptor {
        extent: Extent::new(width, height),
        layout,
        scalar: ScalarType::F32,
        color: ColorEncoding::LinearSrgb,
        range: ColorRange::SceneReferred,
        alpha: AlphaRepresentation::Premultiplied,
        coordinates: CoordinateConvention::PixelCenterUpperLeft,
        semantic: SemanticRole::Color,
    });
    ResourceValue::new(descriptor, layout.channel_count(), samples)
        .expect("sample buffer matches descriptor")
}

/// A single-channel gray image.
fn gray(width: u32, height: u32, samples: Vec<f32>) -> ResourceValue {
    image(width, height, ChannelLayout::Gray, samples)
}

/// Run the guided filter and recover the output.
fn run(input: &ResourceValue, guide: &ResourceValue, params: &serde_json::Value) -> ResourceValue {
    let mut inputs = InputValues::new();
    inputs.insert("input".to_owned(), input.clone());
    inputs.insert("guide".to_owned(), guide.clone());
    let mut out = Guided::new()
        .compute(&inputs, params)
        .expect("guided filter computes");
    out.remove("output").expect("output port produced")
}

/// Sum a plane over the in-bounds intersection of the `(2r+1)` window centred at
/// `(cx, cy)`, returning `(sum, count)` (the shrinking-window boundary the op
/// uses — out-of-bounds taps are dropped, not replicated).
fn window_sum(
    plane: &[f64],
    width: usize,
    height: usize,
    cx: usize,
    cy: usize,
    radius: usize,
) -> (f64, f64) {
    let (mut sum, mut count) = (0.0_f64, 0.0_f64);
    let x0 = cx.saturating_sub(radius);
    let x1 = (cx + radius).min(width - 1);
    let y0 = cy.saturating_sub(radius);
    let y1 = (cy + radius).min(height - 1);
    for y in y0..=y1 {
        for x in x0..=x1 {
            sum += plane[y * width + x];
            count += 1.0;
        }
    }
    (sum, count)
}

/// An independent, brute-force single-channel guided-filter reference: for each
/// window centre fit `(a_k, b_k)` directly over the in-bounds window, then
/// average the per-window linear models over the in-bounds windows covering each
/// pixel. Matches the op's shrinking-window boundary.
fn reference(
    guide: &[f64],
    input: &[f64],
    width: usize,
    height: usize,
    radius: usize,
    eps: f64,
) -> Vec<f64> {
    let len = width * height;
    let mut coef_a = vec![0.0_f64; len];
    let mut coef_b = vec![0.0_f64; len];
    let prod: Vec<f64> = guide.iter().zip(input).map(|(gi, pi)| gi * pi).collect();
    let squares: Vec<f64> = guide.iter().map(|gi| gi * gi).collect();
    for ky in 0..height {
        for kx in 0..width {
            let (sum_guide, count) = window_sum(guide, width, height, kx, ky, radius);
            let (sum_input, _) = window_sum(input, width, height, kx, ky, radius);
            let (sum_prod, _) = window_sum(&prod, width, height, kx, ky, radius);
            let (sum_square, _) = window_sum(&squares, width, height, kx, ky, radius);
            let mean_guide = sum_guide / count;
            let mean_input = sum_input / count;
            let cov = mean_guide.mul_add(-mean_input, sum_prod / count);
            let var = mean_guide.mul_add(-mean_guide, sum_square / count);
            let denom = var + eps;
            let a_k = if denom > 0.0 { cov / denom } else { 0.0 };
            coef_a[ky * width + kx] = a_k;
            coef_b[ky * width + kx] = a_k.mul_add(-mean_guide, mean_input);
        }
    }
    // Average the linear models over the in-bounds windows covering each pixel.
    let mut result = vec![0.0_f64; len];
    for py in 0..height {
        for px in 0..width {
            let (sum_a, count) = window_sum(&coef_a, width, height, px, py, radius);
            let (sum_b, _) = window_sum(&coef_b, width, height, px, py, radius);
            let mean_a = sum_a / count;
            let mean_b = sum_b / count;
            result[py * width + px] = mean_a.mul_add(guide[py * width + px], mean_b);
        }
    }
    result
}

#[test]
fn manifest_validates_and_agrees_with_contract() {
    let manifest = Guided::manifest().expect("manifest");
    manifest.validate().expect("manifest valid");
    paintop_ir::check_contract_consistency(&manifest, &Guided::new())
        .expect("manifest agrees with contract");
    paintop_ir::verify_categories(&manifest, &manifest.test.verification)
        .expect("verification declarations gate clean");
    assert_eq!(manifest.id.to_string(), super::GUIDED_OP_ID);
}

#[test]
fn flat_input_is_preserved_exactly() {
    // A constant input -> a = 0, b = const, regardless of the guide.
    let input = gray(8, 8, vec![0.4; 64]);
    let guide = gray(8, 8, (0..64u16).map(|v| f32::from(v) / 64.0).collect());
    let out = run(
        &input,
        &guide,
        &serde_json::json!({ "radius": 2, "epsilon": 0.01 }),
    );
    for &s in out.samples() {
        assert!(
            (s - 0.4).abs() < FLAT_IDENTITY_TOLERANCE,
            "flat input not preserved: {s}"
        );
    }
}

#[test]
fn flat_guide_yields_box_mean_of_input() {
    // A constant guide -> var(I) = 0 -> a = 0, b = mean(p); output is the input's
    // box-mean.
    let input = gray(8, 8, (0..64u16).map(|v| f32::from(v) / 64.0).collect());
    let guide = gray(8, 8, vec![0.7; 64]);
    let out = run(
        &input,
        &guide,
        &serde_json::json!({ "radius": 1, "epsilon": 0.0 }),
    );
    // With a flat guide a=0, b=mean_p, and the output is mean_b = box_mean(b),
    // i.e. the input passed through the box mean twice (the coefficients are
    // themselves box-averaged in the final step).
    let p: Vec<f64> = input.samples().iter().map(|&s| f64::from(s)).collect();
    let expected = box_mean(&box_mean(&p, 8, 8, 1), 8, 8, 1);
    for (got, want) in out.samples().iter().zip(expected.iter()) {
        assert!(
            (f64::from(*got) - want).abs() < 1e-5,
            "flat-guide output {got} != double box mean {want}"
        );
    }
}

#[test]
fn matches_independent_reference() {
    // A ramp input and a noisy-ish guide; compare against the brute-force fit.
    let width = 9usize;
    let height = 7usize;
    let coord = |index: usize| f64::from(u32::try_from(index).unwrap_or(0));
    let mut p_samples = Vec::with_capacity(width * height);
    let mut i_samples = Vec::with_capacity(width * height);
    for y in 0..height {
        for x in 0..width {
            let pv = (coord(x) + coord(y)) * 0.05;
            let iv = coord(x)
                .mul_add(0.1, coord(y) * -0.03)
                .sin()
                .mul_add(0.5, 0.5);
            p_samples.push(narrow(pv));
            i_samples.push(narrow(iv));
        }
    }
    let input = gray(9, 7, p_samples.clone());
    let guide = gray(9, 7, i_samples.clone());
    let radius = 2usize;
    let eps = 0.02;
    let out = run(
        &input,
        &guide,
        &serde_json::json!({ "radius": radius, "epsilon": eps }),
    );

    let p: Vec<f64> = p_samples.iter().map(|&s| f64::from(s)).collect();
    let i_plane: Vec<f64> = i_samples.iter().map(|&s| f64::from(s)).collect();
    let want = reference(&i_plane, &p, width, height, radius, eps);
    for (idx, (got, expect)) in out.samples().iter().zip(want.iter()).enumerate() {
        assert!(
            (f64::from(*got) - expect).abs() < REFERENCE_TOLERANCE,
            "pixel {idx}: guided {got} != reference {expect}"
        );
    }
}

#[test]
fn self_guided_preserves_a_step_edge() {
    // A 1-D step (left half 0.0, right half 1.0), guided by itself: the step is
    // preserved (the means on either side stay near 0 and 1), while a uniform
    // smoother would blur the boundary.
    let w = 16usize;
    let h = 4usize;
    let mut samples = Vec::with_capacity(w * h);
    for _y in 0..h {
        for x in 0..w {
            samples.push(if x < w / 2 { 0.0 } else { 1.0 });
        }
    }
    let img = gray(16, 4, samples);
    let out = run(
        &img,
        &img,
        &serde_json::json!({ "radius": 2, "epsilon": 1e-6 }),
    );
    let s = out.samples();
    // Far from the edge the flat regions are preserved.
    let left = f64::from(s[2 * w + 2]);
    let right = f64::from(s[2 * w + (w - 3)]);
    assert!(left < 0.02, "left flat preserved: {left}");
    assert!(right > 0.98, "right flat preserved: {right}");
    // The edge stays sharp: the jump across the boundary is large.
    let jump = f64::from(s[2 * w + (w / 2)]) - f64::from(s[2 * w + (w / 2 - 1)]);
    assert!(jump > 0.5, "edge transition preserved (jump {jump})");
}

#[test]
fn single_channel_guide_broadcasts_over_rgb_input() {
    // An RGB input filtered by one gray guide: each channel is guided identically.
    let w = 6u32;
    let h = 6u32;
    let mut rgb = Vec::with_capacity((w * h * 3) as usize);
    for i in 0..(w * h) {
        let v = f32::from(u16::try_from(i).unwrap_or(0)) / 36.0;
        rgb.push(v);
        rgb.push(v * 0.5);
        rgb.push(0.2);
    }
    let input = image(w, h, ChannelLayout::Rgb, rgb);
    let guide_samples: Vec<f32> = (0..(w * h))
        .map(|i| f32::from(u16::try_from(i).unwrap_or(0)) / 36.0)
        .collect();
    let guide = gray(w, h, guide_samples);
    let out = run(
        &input,
        &guide,
        &serde_json::json!({ "radius": 1, "epsilon": 0.01 }),
    );
    assert_eq!(out.channels(), 3);
    assert_eq!(out.extent(), Extent::new(w, h));
}

#[test]
fn rerun_is_bit_identical() {
    let input = gray(8, 8, (0..64u16).map(|v| f32::from(v) / 64.0).collect());
    let guide = gray(8, 8, vec![0.3; 64]);
    let params = serde_json::json!({ "radius": 2, "epsilon": 0.05 });
    let a = run(&input, &guide, &params);
    let b = run(&input, &guide, &params);
    assert_eq!(
        a.samples(),
        b.samples(),
        "guided filter must be deterministic"
    );
}

#[test]
fn missing_radius_is_rejected() {
    let img = gray(4, 4, vec![0.0; 16]);
    let mut inputs = InputValues::new();
    inputs.insert("input".to_owned(), img.clone());
    inputs.insert("guide".to_owned(), img);
    let err = Guided::new()
        .compute(&inputs, &serde_json::json!({ "epsilon": 0.1 }))
        .expect_err("a missing radius must be rejected");
    assert_eq!(err.class, ErrorClass::Schema);
}

#[test]
fn zero_radius_and_negative_epsilon_are_rejected() {
    assert!(GuidedRequest::resolve(&serde_json::json!({ "radius": 0, "epsilon": 0.1 })).is_err());
    assert!(GuidedRequest::resolve(&serde_json::json!({ "radius": 2, "epsilon": -1.0 })).is_err());
}

#[test]
fn mismatched_extent_is_rejected() {
    let input = gray(4, 4, vec![0.0; 16]);
    let guide = gray(5, 5, vec![0.0; 25]);
    let mut inputs = InputValues::new();
    inputs.insert("input".to_owned(), input);
    inputs.insert("guide".to_owned(), guide);
    let err = Guided::new()
        .compute(&inputs, &serde_json::json!({ "radius": 1, "epsilon": 0.1 }))
        .expect_err("a mismatched extent must be rejected");
    assert_eq!(err.class, ErrorClass::Semantic);
}

/// Narrow an f64 test sample to the f32 storage type.
#[allow(
    clippy::cast_possible_truncation,
    reason = "test fixture samples are small bounded values stored as f32"
)]
fn narrow(v: f64) -> f32 {
    v as f32
}
