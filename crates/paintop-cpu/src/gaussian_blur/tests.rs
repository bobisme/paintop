//! Verification suite for `filter.gaussian_blur@1` (`OP_CATALOG` §8,
//! `AGENT_VERIFICATION` §3.4):
//!
//! - **schema/contract**: the manifest validates, agrees with its contract, gates
//!   clean, and the checked-in JSON matches the builder;
//! - **analytic fixtures**: the sampled kernel is positive and unit-sum; the
//!   blurred impulse's empirical variance matches `sigma^2` within a
//!   discretization bound;
//! - **property**: a constant image is preserved; the blur is isotropic under 90°
//!   rotation; the σ-semigroup `G_s1 * G_s2 ~ G_sqrt(s1^2 + s2^2)` holds within a
//!   bound; `sigma <= cutoff` is the exact identity;
//! - **differential**: the blur equals `filter.convolve` driven with the same
//!   Gaussian kernel, bit-for-bit;
//! - **rejection**: missing / non-positive / over-limit `sigma` and an unknown
//!   boundary mode are rejected.

use paintop_core::executor::{InputValues, OpImplementation, ResourceValue};
use paintop_ir::{
    AlphaRepresentation, ChannelLayout, ColorEncoding, ColorRange, CoordinateConvention,
    Descriptors, ErrorClass, Extent, ImageDescriptor, OpContract, ResourceDescriptor, ScalarType,
    SemanticRole,
};

use super::{
    GAUSSIAN_BLUR_OP_ID, GaussianBlur, SIGMA_CUTOFF, convolve_params, gaussian_kernel,
    kernel_radius,
};
use crate::convolve::Convolve;

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

/// Run the blur and recover the output resource.
fn blur(value: &ResourceValue, params: &serde_json::Value) -> ResourceValue {
    let mut inputs = InputValues::new();
    inputs.insert("input".to_owned(), value.clone());
    let mut out = GaussianBlur::new()
        .compute(&inputs, params)
        .expect("blur computes");
    out.remove("output").expect("output port produced")
}

/// The sum of a kernel object's `weights`.
fn kernel_sum(kernel: &serde_json::Value) -> f64 {
    kernel["weights"]
        .as_array()
        .expect("weights array")
        .iter()
        .map(|w| w.as_f64().expect("weight number"))
        .sum()
}

#[test]
fn manifest_validates_and_agrees_with_contract() {
    let manifest = GaussianBlur::manifest().expect("blur manifest");
    manifest.validate().expect("blur manifest valid");
    paintop_ir::check_contract_consistency(&manifest, &GaussianBlur::new())
        .expect("blur manifest agrees with contract");
    paintop_ir::verify_categories(&manifest, &manifest.test.verification)
        .expect("blur verification declarations gate clean");
    assert_eq!(manifest.id.to_string(), GAUSSIAN_BLUR_OP_ID);
}

#[test]
fn kernel_is_positive_and_unit_sum() {
    for sigma in [0.5, 1.0, 2.0, 3.5] {
        let kernel = gaussian_kernel(sigma);
        let weights = kernel["weights"].as_array().expect("weights");
        assert!(
            weights.iter().all(|w| w.as_f64().expect("num") > 0.0),
            "kernel must be strictly positive for sigma {sigma}"
        );
        let sum = kernel_sum(&kernel);
        assert!(
            (sum - 1.0).abs() < 1e-9,
            "kernel must be unit-sum for sigma {sigma}, got {sum}"
        );
    }
}

#[test]
fn kernel_radius_is_ceil_three_sigma() {
    assert_eq!(kernel_radius(1.0), 3);
    assert_eq!(kernel_radius(2.0), 6);
    assert_eq!(kernel_radius(0.5), 2); // ceil(1.5) = 2
    // σ→0 cutoff: radius 0.
    assert_eq!(kernel_radius(SIGMA_CUTOFF / 2.0), 0);
}

#[test]
fn constant_image_is_preserved() {
    let img = gray(9, 9, vec![0.7; 81]);
    let out = blur(&img, &serde_json::json!({ "sigma": 1.5, "mode": "clamp" }));
    assert_eq!(out.extent(), Extent::new(9, 9));
    for &s in out.samples() {
        assert!((s - 0.7).abs() < 1e-5, "constant not preserved: {s}");
    }
}

#[test]
fn sigma_below_cutoff_is_the_identity() {
    let img = gray(5, 5, (0..25u8).map(f32::from).collect());
    let out = blur(
        &img,
        &serde_json::json!({ "sigma": SIGMA_CUTOFF / 10.0, "mode": "clamp" }),
    );
    assert_eq!(out.samples(), img.samples());
}

#[test]
fn isotropic_under_ninety_degree_rotation() {
    // Blur commutes with a 90° rotation: rot(blur(x)) == blur(rot(x)). We test it
    // on an asymmetric image. The Gaussian kernel is rotationally symmetric, so
    // with a symmetric boundary (mirror) the two paths agree within rounding.
    let w = 5usize;
    let h = 5usize;
    let src: Vec<f32> = (0..u8::try_from(w * h).expect("fits u8"))
        .map(f32::from)
        .collect();
    let rot = |s: &[f32]| -> Vec<f32> {
        // 90° clockwise rotation of a w×h grid (square here).
        let mut out = vec![0.0f32; w * h];
        for y in 0..h {
            for x in 0..w {
                out[x * w + (h - 1 - y)] = s[y * w + x];
            }
        }
        out
    };
    let img = gray(5, 5, src.clone());
    let params = serde_json::json!({ "sigma": 1.0, "mode": "mirror" });
    let blurred = blur(&img, &params);
    let rot_then = rot(blurred.samples());

    let img_rot = gray(5, 5, rot(&src));
    let blur_of_rot = blur(&img_rot, &params);

    for (a, b) in rot_then.iter().zip(blur_of_rot.samples()) {
        assert!(
            (a - b).abs() < 1e-4,
            "blur is not rotation-isotropic: {a} != {b}"
        );
    }
}

#[test]
fn semigroup_holds_within_a_bound() {
    // G_s1 ∘ G_s2 ≈ G_sqrt(s1^2 + s2^2) on the interior, away from boundaries.
    // Use a large field with a single central impulse and mirror boundary.
    let n = 31usize;
    let mut src = vec![0.0f32; n * n];
    src[(n / 2) * n + (n / 2)] = 1.0;
    let img = gray(31, 31, src);
    let s1 = 1.5_f64;
    let s2 = 2.0_f64;
    let combined = s1.hypot(s2);

    let once = blur(
        &img,
        &serde_json::json!({ "sigma": s1, "mode": "constant", "value": [0.0] }),
    );
    let twice = blur(
        &once,
        &serde_json::json!({ "sigma": s2, "mode": "constant", "value": [0.0] }),
    );
    let direct = blur(
        &img,
        &serde_json::json!({ "sigma": combined, "mode": "constant", "value": [0.0] }),
    );

    // Compare on the central interior region where neither kernel touches the
    // edge. Max absolute error must be small.
    let margin = 12usize;
    let mut max_err = 0.0f32;
    for y in margin..(n - margin) {
        for x in margin..(n - margin) {
            let a = twice.samples()[y * n + x];
            let b = direct.samples()[y * n + x];
            max_err = max_err.max((a - b).abs());
        }
    }
    assert!(
        max_err < 2e-3,
        "semigroup discretization error too large: {max_err}"
    );
}

#[test]
fn blurred_impulse_variance_matches_sigma_squared() {
    // The empirical second moment of the blurred impulse equals sigma^2 within a
    // discretization bound. Single central impulse, big field, constant-zero
    // border so the kernel mass is conserved.
    let n = 41usize;
    let c = n / 2;
    let mut src = vec![0.0f32; n * n];
    src[c * n + c] = 1.0;
    let img = gray(41, 41, src);
    let sigma = 3.0_f64;
    let out = blur(
        &img,
        &serde_json::json!({ "sigma": sigma, "mode": "constant", "value": [0.0] }),
    );

    let ci = i32::try_from(c).expect("centre fits i32");
    let mut mass = 0.0_f64;
    let mut second = 0.0_f64;
    for y in 0..n {
        for x in 0..n {
            let w = f64::from(out.samples()[y * n + x]);
            let xi = i32::try_from(x).expect("x fits i32");
            let yi = i32::try_from(y).expect("y fits i32");
            let dx = f64::from(xi - ci);
            let dy = f64::from(yi - ci);
            mass += w;
            second += w * (dx * dx + dy * dy);
        }
    }
    // Mass conserved (unit-sum kernel) and variance ≈ 2*sigma^2 (sum over both
    // axes: E[dx^2 + dy^2] = 2 sigma^2).
    assert!((mass - 1.0).abs() < 1e-4, "mass not conserved: {mass}");
    let variance = second / mass;
    let expected = 2.0 * sigma * sigma;
    // The kernel is truncated at radius ceil(3*sigma), which clips the Gaussian
    // tails and biases the empirical variance slightly below 2*sigma^2; the bias
    // is ~1-2% for a 3-sigma cutoff, so allow a modest absolute tolerance.
    assert!(
        (variance - expected).abs() < 0.5,
        "impulse variance {variance} != expected {expected}"
    );
}

#[test]
fn differential_against_convolve_with_same_kernel() {
    // The blur must equal filter.convolve driven with the same Gaussian kernel,
    // bit-for-bit (it is literally that convolution).
    let img = gray(8, 8, (0..64u8).map(|n| f32::from(n) * 0.05).collect());
    let sigma = 1.7_f64;
    let mode = "clamp";

    let blurred = blur(&img, &serde_json::json!({ "sigma": sigma, "mode": mode }));

    let conv_params = convolve_params(sigma, mode, 1);
    let mut inputs = InputValues::new();
    inputs.insert("input".to_owned(), img);
    let mut conv_out = Convolve::new()
        .compute(&inputs, &conv_params)
        .expect("convolve computes");
    let convolved = conv_out.remove("output").expect("output");

    assert_eq!(
        blurred.samples(),
        convolved.samples(),
        "blur must equal convolve with the same kernel bit-for-bit"
    );
}

#[test]
fn infer_outputs_preserves_extent() {
    let descriptor = ResourceDescriptor::Image(ImageDescriptor {
        extent: Extent::new(12, 7),
        layout: ChannelLayout::Rgba,
        scalar: ScalarType::F32,
        color: ColorEncoding::Srgb,
        range: ColorRange::DisplayReferred,
        alpha: AlphaRepresentation::Straight,
        coordinates: CoordinateConvention::PixelCenterUpperLeft,
        semantic: SemanticRole::Color,
    });
    let mut inputs = Descriptors::new();
    inputs.insert("input".to_owned(), descriptor);
    let out = GaussianBlur::new()
        .infer_outputs(&inputs, &serde_json::json!({ "sigma": 2.0 }))
        .expect("infer");
    let ResourceDescriptor::Image(d) = out["output"] else {
        panic!("expected image");
    };
    assert_eq!(d.extent, Extent::new(12, 7));
    assert_eq!(d.layout, ChannelLayout::Rgba);
}

#[test]
fn missing_sigma_is_rejected() {
    let img = gray(3, 3, vec![0.0; 9]);
    let mut inputs = InputValues::new();
    inputs.insert("input".to_owned(), img);
    let err = GaussianBlur::new()
        .compute(&inputs, &serde_json::json!({ "mode": "clamp" }))
        .expect_err("missing sigma must fail");
    assert_eq!(err.code, super::E_BLUR_PARAM);
    assert_eq!(err.class, ErrorClass::Schema);
}

#[test]
fn negative_sigma_is_rejected() {
    let img = gray(3, 3, vec![0.0; 9]);
    let mut inputs = InputValues::new();
    inputs.insert("input".to_owned(), img);
    let err = GaussianBlur::new()
        .compute(&inputs, &serde_json::json!({ "sigma": -1.0 }))
        .expect_err("negative sigma must fail");
    assert_eq!(err.code, super::E_BLUR_PARAM);
}

#[test]
fn over_limit_sigma_is_rejected() {
    let img = gray(3, 3, vec![0.0; 9]);
    let mut inputs = InputValues::new();
    inputs.insert("input".to_owned(), img);
    let err = GaussianBlur::new()
        .compute(
            &inputs,
            &serde_json::json!({ "sigma": 10.0, "sigma_max": 4.0 }),
        )
        .expect_err("over-limit sigma must fail");
    assert_eq!(err.code, super::E_BLUR_PARAM);
    assert_eq!(err.class, ErrorClass::Policy);
}

#[test]
fn unknown_mode_is_rejected() {
    let img = gray(3, 3, vec![0.0; 9]);
    let mut inputs = InputValues::new();
    inputs.insert("input".to_owned(), img);
    let err = GaussianBlur::new()
        .compute(
            &inputs,
            &serde_json::json!({ "sigma": 1.0, "mode": "bogus" }),
        )
        .expect_err("unknown mode must fail");
    assert_eq!(err.code, super::E_BLUR_PARAM);
}

#[test]
fn checked_in_manifest_matches_builder() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("crates dir")
        .parent()
        .expect("repo root")
        .join("ops/manifests");
    let manifest = GaussianBlur::manifest().expect("blur manifest");
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
