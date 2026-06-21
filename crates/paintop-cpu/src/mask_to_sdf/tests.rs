//! Verification suite for `mask.to_sdf@1` (`AGENT_VERIFICATION` §3.7):
//!
//! - **schema/contract**: the manifest validates, agrees with the contract,
//!   gates clean, and declares the `negative-inside` sign;
//! - **analytic fixtures**: sign is correct inside/outside; the zero contour sits
//!   on the threshold boundary; a disk's and a rectangle's interior/exterior
//!   distances match the analytic signed distance away from the rasterized
//!   boundary;
//! - **property/metamorphic**: a single inside pixel gives the plain signed
//!   distance to that pixel; translation covariance; threshold monotonicity;
//! - **differential**: the field equals the one rebuilt from the brute-force EDT
//!   oracle sample-for-sample;
//! - **degenerate**: an empty / full partition yields `+∞` / `−∞`;
//! - **rejection**: an out-of-range threshold and a missing input are typed
//!   errors.

use paintop_core::executor::{InputValues, OpImplementation, ResourceValue};
use paintop_ir::{
    CoordinateConvention, Descriptors, ErrorClass, Extent, MaskDescriptor, MaskMeaning, OpContract,
    ResourceDescriptor, ScalarType, SdfSign, ValidRange, check_contract_consistency,
    verify_categories,
};

use crate::edt::{self, BinaryGrid};

use super::{MaskToSdf, OP_ID, signed_distance};

/// Build a coverage-mask value from explicit samples sized `w * h`.
fn mask(w: u32, h: u32, samples: Vec<f32>) -> ResourceValue {
    assert_eq!(samples.len(), (w * h) as usize, "sample count");
    let descriptor = ResourceDescriptor::Mask(MaskDescriptor {
        extent: Extent::new(w, h),
        scalar: ScalarType::F32,
        range: ValidRange::Bounded { min: 0.0, max: 1.0 },
        meaning: MaskMeaning::Coverage,
        coordinates: CoordinateConvention::PixelCenterUpperLeft,
    });
    ResourceValue::new(descriptor, 1, samples).expect("mask value")
}

/// Run the op on a mask with the given params and return the SDF samples.
fn to_sdf(m: &ResourceValue, params: &serde_json::Value) -> Vec<f32> {
    let mut inputs = InputValues::new();
    inputs.insert("mask".to_owned(), m.clone());
    let mut out = MaskToSdf::new()
        .compute(&inputs, params)
        .expect("to_sdf computes");
    out.remove("sdf").expect("sdf output").into_samples()
}

/// The index of `(x, y)` in a row-major `w`-wide buffer.
fn idx(x: u32, y: u32, w: u32) -> usize {
    (y as usize) * (w as usize) + (x as usize)
}

/// Assert two `f32`s are bit-identical (exact equality without tripping the
/// `float_cmp` lint — every SDF sample here is an exact integer, a root of one,
/// or a sentinel, so bit equality is the intended contract).
#[track_caller]
fn bits_eq(got: f32, want: f32, label: &str) {
    assert_eq!(got.to_bits(), want.to_bits(), "{label}: {got} != {want}");
}

// --- schema / contract -----------------------------------------------------

#[test]
fn manifest_validates_and_agrees_with_contract() {
    let manifest = MaskToSdf::manifest().expect("manifest");
    manifest.validate().expect("manifest valid");
    verify_categories(&manifest, &manifest.test.verification).expect("verification gates clean");
    assert_eq!(manifest.id.to_string(), OP_ID);
    check_contract_consistency(&manifest, &MaskToSdf::new()).expect("contract consistent");
}

#[test]
fn declares_negative_inside_sign() {
    let manifest = MaskToSdf::manifest().expect("manifest");
    let inputs = {
        let mut d = Descriptors::new();
        d.insert(
            "mask".to_owned(),
            ResourceDescriptor::Mask(MaskDescriptor {
                extent: Extent::new(4, 4),
                scalar: ScalarType::F32,
                range: ValidRange::Bounded { min: 0.0, max: 1.0 },
                meaning: MaskMeaning::Coverage,
                coordinates: CoordinateConvention::PixelCenterUpperLeft,
            }),
        );
        d
    };
    let _ = &manifest;
    let out = MaskToSdf::new()
        .infer_outputs(&inputs, &serde_json::json!({}))
        .expect("infer");
    let Some(ResourceDescriptor::SdfMask(sdf)) = out.get("sdf") else {
        panic!("sdf output");
    };
    assert_eq!(sdf.sign, SdfSign::NegativeInside);
}

// --- analytic fixtures -----------------------------------------------------

#[test]
fn sign_is_negative_inside_positive_outside() {
    // A 2x2 inside block centered in a 6x6 field.
    let w = 6;
    let h = 6;
    let mut samples = vec![0.0_f32; (w * h) as usize];
    for y in 2..4 {
        for x in 2..4 {
            samples[idx(x, y, w)] = 1.0;
        }
    }
    let field = to_sdf(&mask(w, h, samples), &serde_json::json!({}));
    // Inside pixels are strictly negative.
    assert!(field[idx(2, 2, w)] < 0.0, "inside must be negative");
    // Far outside pixels are strictly positive.
    assert!(field[idx(0, 0, w)] > 0.0, "outside must be positive");
}

#[test]
fn zero_contour_sits_on_threshold_boundary() {
    // A half-plane: left columns inside, right columns outside. The signed
    // distance at a boundary pixel and its neighbor straddle zero by exactly
    // +-0.5 (pixel-center distance to the boundary midway between centers... here
    // measured to the nearest opposite pixel, so |sdf| = 1 at the seam, but
    // monotone and sign-correct). We assert the seam is the sign change.
    let w = 4;
    let h = 1;
    // inside: x in {0,1}; outside: x in {2,3}
    let samples = vec![1.0, 1.0, 0.0, 0.0];
    let field = to_sdf(
        &mask(w, h, samples),
        &serde_json::json!({ "threshold": 0.5 }),
    );
    assert!(field[idx(1, 0, w)] < 0.0, "last inside negative");
    assert!(field[idx(2, 0, w)] > 0.0, "first outside positive");
    // Symmetric magnitude across the seam (1 pixel to the nearest opposite set).
    bits_eq(field[idx(1, 0, w)], -field[idx(2, 0, w)], "seam symmetry");
}

#[test]
fn disk_interior_matches_analytic_signed_distance() {
    // A rasterized disk of radius R at center c. Away from the 1px boundary
    // band, the signed distance to the disk boundary is |p - c| - R; the
    // rasterized EDT distance to the nearest opposite pixel approximates this to
    // within ~1px, so we check a loose analytic bound at a deep-interior and a
    // far-exterior sample where rasterization ambiguity is sub-pixel.
    let w = 31;
    let h = 31;
    let cx = 15.0_f64;
    let cy = 15.0_f64;
    let r = 8.0_f64;
    let mut samples = vec![0.0_f32; (w * h) as usize];
    for y in 0..h {
        for x in 0..w {
            let dx = f64::from(x) - cx;
            let dy = f64::from(y) - cy;
            if dx.mul_add(dx, dy * dy).sqrt() <= r {
                samples[idx(x, y, w)] = 1.0;
            }
        }
    }
    let field = to_sdf(&mask(w, h, samples), &serde_json::json!({}));
    // Deep interior at the center: analytic distance is -R; allow ~1.5px slack
    // for rasterization.
    let centre = f64::from(field[idx(15, 15, w)]);
    assert!(
        (centre - (-r)).abs() < 1.5,
        "centre sdf {centre} should be near -{r}"
    );
    // Far exterior corner: analytic distance is sqrt(15^2+15^2) - R.
    let corner = f64::from(field[idx(0, 0, w)]);
    let analytic = (15.0_f64).hypot(15.0) - r;
    assert!(
        (corner - analytic).abs() < 1.5,
        "corner sdf {corner} should be near {analytic}"
    );
}

#[test]
fn rectangle_interior_matches_chebyshev_free_distance() {
    // An axis-aligned filled rectangle; an interior pixel's signed distance is
    // the negative of its distance to the nearest edge, which for the EDT is the
    // distance to the nearest outside pixel (one past the edge).
    let w = 11;
    let h = 11;
    let mut samples = vec![0.0_f32; (w * h) as usize];
    for y in 2..9 {
        for x in 3..8 {
            samples[idx(x, y, w)] = 1.0;
        }
    }
    let field = to_sdf(&mask(w, h, samples), &serde_json::json!({}));
    // Pixel (5,5): nearest outside pixel is at x=8 (3 away) vs x=2 (3 away) vs
    // y=1 (4) / y=9 (4); horizontal edges are nearest. The nearest *outside*
    // pixel is at distance 3 (column 2 or 8), so the inside distance is 3 and the
    // sdf is -3.
    bits_eq(field[idx(5, 5, w)], -3.0, "rectangle interior");
}

// --- property / metamorphic ------------------------------------------------

#[test]
fn single_inside_pixel_is_plain_signed_distance() {
    // One inside pixel: outside distance is the euclidean distance to it; the
    // inside set is a single point so the only interior pixel has inside-distance
    // 0 (it borders outside immediately), so sdf(that pixel) = 0 - 0... actually
    // the inside pixel's nearest outside pixel is adjacent (distance 1), so its
    // sdf is -1 and every other pixel is +distance_to_that_pixel.
    let w = 5;
    let h = 5;
    let mut samples = vec![0.0_f32; (w * h) as usize];
    samples[idx(2, 2, w)] = 1.0;
    let field = to_sdf(&mask(w, h, samples), &serde_json::json!({}));
    // The lone inside pixel: nearest outside is 1 away -> sdf = -1.
    bits_eq(field[idx(2, 2, w)], -1.0, "lone inside pixel");
    // An outside pixel: sdf = +distance to the inside pixel.
    let expect = (2.0_f64).hypot(2.0);
    #[allow(clippy::cast_possible_truncation)]
    let expect = expect as f32;
    bits_eq(field[idx(0, 0, w)], expect, "outside distance");
}

#[test]
fn translation_covariance() {
    // Shifting the inside region by an integer offset shifts the field by the
    // same offset (with the same extent and zero-padding around it).
    let w = 8;
    let h = 8;
    let mut a = vec![0.0_f32; (w * h) as usize];
    let mut b = vec![0.0_f32; (w * h) as usize];
    for y in 1..3 {
        for x in 1..3 {
            a[idx(x, y, w)] = 1.0;
            b[idx(x + 2, y + 2, w)] = 1.0;
        }
    }
    let fa = to_sdf(&mask(w, h, a), &serde_json::json!({}));
    let fb = to_sdf(&mask(w, h, b), &serde_json::json!({}));
    // fa at (x,y) equals fb at (x+2, y+2) for the interior away from the field
    // edges (the EDT footprint stays inside the grid for these samples).
    for y in 0..4 {
        for x in 0..4 {
            bits_eq(
                fa[idx(x, y, w)],
                fb[idx(x + 2, y + 2, w)],
                &format!("translation covariance at ({x},{y})"),
            );
        }
    }
}

#[test]
fn threshold_selects_the_isocontour() {
    // A linear coverage ramp 0..1: a higher threshold shrinks the inside set, so
    // a fixed pixel that is inside at a low threshold can be outside at a high
    // one, flipping its sdf sign.
    let w = 5;
    let h = 1;
    let samples = vec![0.0, 0.25, 0.5, 0.75, 1.0];
    let low = to_sdf(
        &mask(w, h, samples.clone()),
        &serde_json::json!({ "threshold": 0.2 }),
    );
    let high = to_sdf(
        &mask(w, h, samples),
        &serde_json::json!({ "threshold": 0.8 }),
    );
    // Pixel x=2 (coverage 0.5): inside at t=0.2 (negative), outside at t=0.8.
    assert!(low[idx(2, 0, w)] < 0.0, "inside at low threshold");
    assert!(high[idx(2, 0, w)] > 0.0, "outside at high threshold");
}

// --- differential ----------------------------------------------------------

#[test]
fn matches_brute_force_edt_oracle() {
    // Rebuild the field independently from the brute-force EDT oracle and require
    // bit-for-bit agreement with the op on a structured small fixture (a ring).
    let w = 9;
    let h = 9;
    let mut samples = vec![0.0_f32; (w * h) as usize];
    for y in 0..h {
        for x in 0..w {
            let dx = f64::from(x) - 4.0;
            let dy = f64::from(y) - 4.0;
            let r = dx.hypot(dy);
            if (2.0..=3.5).contains(&r) {
                samples[idx(x, y, w)] = 1.0;
            }
        }
    }
    let field = to_sdf(&mask(w, h, samples.clone()), &serde_json::json!({}));

    let inside: Vec<bool> = samples.iter().map(|&c| c >= 0.5).collect();
    let outside: Vec<bool> = inside.iter().map(|&b| !b).collect();
    let ig = BinaryGrid::new(Extent::new(w, h), &inside).expect("inside grid");
    let og = BinaryGrid::new(Extent::new(w, h), &outside).expect("outside grid");
    let d_out = edt::distance(&edt::brute_force_sq(&ig));
    let d_in = edt::distance(&edt::brute_force_sq(&og));
    for (i, (&do_, &di_)) in d_out.iter().zip(d_in.iter()).enumerate() {
        let want = if do_.is_infinite() && !di_.is_infinite() {
            f32::INFINITY
        } else if !do_.is_infinite() && di_.is_infinite() {
            f32::NEG_INFINITY
        } else {
            do_ - di_
        };
        assert_eq!(
            field[i].to_bits(),
            want.to_bits(),
            "sample {i}: op {} vs oracle {want}",
            field[i]
        );
    }
}

// --- degenerate ------------------------------------------------------------

#[test]
fn empty_inside_is_positive_infinity() {
    let w = 3;
    let h = 3;
    let field = signed_distance(Extent::new(w, h), &[0.0; 9], 0.5).expect("field");
    assert!(field.iter().all(|s| *s == f32::INFINITY), "all +inf");
}

#[test]
fn full_inside_is_negative_infinity() {
    let w = 3;
    let h = 3;
    let field = signed_distance(Extent::new(w, h), &[1.0; 9], 0.5).expect("field");
    assert!(field.iter().all(|s| *s == f32::NEG_INFINITY), "all -inf");
}

// --- rejection -------------------------------------------------------------

#[test]
fn rejects_out_of_range_threshold() {
    let mut inputs = InputValues::new();
    inputs.insert("mask".to_owned(), mask(2, 2, vec![1.0, 0.0, 0.0, 1.0]));
    let err = MaskToSdf::new()
        .compute(&inputs, &serde_json::json!({ "threshold": 1.5 }))
        .expect_err("out-of-range threshold rejected");
    assert_eq!(err.class, ErrorClass::Schema);
}

#[test]
fn rejects_missing_mask() {
    let inputs = InputValues::new();
    let err = MaskToSdf::new()
        .compute(&inputs, &serde_json::json!({}))
        .expect_err("missing mask rejected");
    assert_eq!(err.class, ErrorClass::Reference);
}
