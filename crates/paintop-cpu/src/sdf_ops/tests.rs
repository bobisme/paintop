//! Verification suite for `sdf.offset@1` and `sdf.to_mask@1`
//! (`AGENT_VERIFICATION` §3.7):
//!
//! - **schema/contract**: both manifests validate, agree with their contracts,
//!   gate clean, and declare the right output kinds;
//! - **offset law**: `offset(offset(s, d1), d2) = offset(s, d1 + d2)` and
//!   `offset(s, 0) = s` bit-identically; `±∞` preserved;
//! - **reconstruction**: hard-step (`half_width = 0`) round-trips a hard mask
//!   exactly at the zero contour; smoothstep gives `0.5` on the contour and the
//!   declared feather width across the band; `±∞` map to full/empty coverage;
//! - **rejection**: a non-finite offset, negative half-width, unknown profile,
//!   wrong-kind / missing input are typed errors.

use paintop_core::executor::{InputValues, OpImplementation, ResourceValue};
use paintop_ir::{
    CoordinateConvention, ErrorClass, Extent, OpContract, ResourceDescriptor, ScalarType,
    SdfDescriptor, SdfSign, SdfUnits, check_contract_consistency, verify_categories,
};

use super::{OFFSET_OP_ID, SdfOffset, SdfToMask, TO_MASK_OP_ID, reconstruct_coverage};

/// Build an `SdfMask` value from explicit field samples sized `w * h`.
fn sdf(w: u32, h: u32, samples: Vec<f32>) -> ResourceValue {
    assert_eq!(samples.len(), (w * h) as usize, "sample count");
    let descriptor = ResourceDescriptor::SdfMask(SdfDescriptor {
        extent: Extent::new(w, h),
        scalar: ScalarType::F32,
        units: SdfUnits::Pixels,
        sign: SdfSign::NegativeInside,
        coordinates: CoordinateConvention::PixelCenterUpperLeft,
    });
    ResourceValue::new(descriptor, 1, samples).expect("sdf value")
}

/// Apply `sdf.offset` with `distance_px = d`, returning the offset field.
fn offset(s: &ResourceValue, d: f64) -> ResourceValue {
    let mut inputs = InputValues::new();
    inputs.insert("sdf".to_owned(), s.clone());
    let mut out = SdfOffset::new()
        .compute(&inputs, &serde_json::json!({ "distance_px": d }))
        .expect("offset computes");
    out.remove("sdf").expect("sdf output")
}

/// Apply `sdf.to_mask` with the given params, returning the coverage samples.
fn to_mask(s: &ResourceValue, params: &serde_json::Value) -> Vec<f32> {
    let mut inputs = InputValues::new();
    inputs.insert("sdf".to_owned(), s.clone());
    let mut out = SdfToMask::new()
        .compute(&inputs, params)
        .expect("to_mask computes");
    out.remove("mask").expect("mask output").into_samples()
}

/// Assert two `f32`s are bit-identical.
#[track_caller]
fn bits_eq(got: f32, want: f32, label: &str) {
    assert_eq!(got.to_bits(), want.to_bits(), "{label}: {got} != {want}");
}

// --- schema / contract -----------------------------------------------------

#[test]
fn manifests_validate_and_agree_with_contracts() {
    let off = SdfOffset::manifest().expect("offset manifest");
    off.validate().expect("offset valid");
    verify_categories(&off, &off.test.verification).expect("offset gates clean");
    assert_eq!(off.id.to_string(), OFFSET_OP_ID);
    check_contract_consistency(&off, &SdfOffset::new()).expect("offset contract");

    let tm = SdfToMask::manifest().expect("to_mask manifest");
    tm.validate().expect("to_mask valid");
    verify_categories(&tm, &tm.test.verification).expect("to_mask gates clean");
    assert_eq!(tm.id.to_string(), TO_MASK_OP_ID);
    check_contract_consistency(&tm, &SdfToMask::new()).expect("to_mask contract");
}

// --- offset law ------------------------------------------------------------

#[test]
fn offset_by_zero_is_identity() {
    let s = sdf(1, 4, vec![-2.0, -0.5, 0.5, 3.0]);
    let out = offset(&s, 0.0);
    for (a, b) in out.samples().iter().zip(s.samples()) {
        bits_eq(*a, *b, "zero offset identity");
    }
}

#[test]
fn offset_subtracts_the_distance() {
    let s = sdf(1, 3, vec![-1.0, 0.0, 2.0]);
    let out = offset(&s, 0.75);
    // phi' = phi - 0.75
    bits_eq(out.samples()[0], -1.75, "sample 0");
    bits_eq(out.samples()[1], -0.75, "sample 1");
    bits_eq(out.samples()[2], 1.25, "sample 2");
}

#[test]
fn offset_composition_law() {
    // offset(offset(s, d1), d2) == offset(s, d1 + d2), bit-for-bit, for distances
    // chosen so d1 + d2 is exact in f32 (no double rounding).
    let s = sdf(2, 2, vec![-3.0, -0.25, 0.5, 4.0]);
    let d1 = 1.5;
    let d2 = -0.75;
    let twice = offset(&offset(&s, d1), d2);
    let once = offset(&s, d1 + d2);
    for (a, b) in twice.samples().iter().zip(once.samples()) {
        bits_eq(*a, *b, "composition law");
    }
}

#[test]
fn offset_preserves_infinities() {
    let s = sdf(1, 2, vec![f32::INFINITY, f32::NEG_INFINITY]);
    let out = offset(&s, 2.0);
    assert!(
        out.samples()[0].is_infinite() && out.samples()[0] > 0.0,
        "+inf"
    );
    assert!(
        out.samples()[1].is_infinite() && out.samples()[1] < 0.0,
        "-inf"
    );
}

// --- reconstruction --------------------------------------------------------

#[test]
fn hard_step_round_trips_the_zero_contour() {
    // half_width = 0 (default): coverage is 1 for phi <= 0, 0 otherwise — the
    // hard mask the field's inside set describes.
    let s = sdf(1, 5, vec![-2.0, -0.5, 0.0, 0.5, 2.0]);
    let cov = to_mask(&s, &serde_json::json!({}));
    bits_eq(cov[0], 1.0, "deep inside");
    bits_eq(cov[1], 1.0, "near inside");
    bits_eq(cov[2], 1.0, "on contour (phi=0 is inside)");
    bits_eq(cov[3], 0.0, "near outside");
    bits_eq(cov[4], 0.0, "deep outside");
}

#[test]
fn smoothstep_is_half_on_the_contour() {
    let s = sdf(1, 1, vec![0.0]);
    let cov = to_mask(&s, &serde_json::json!({ "half_width_px": 2.0 }));
    bits_eq(cov[0], 0.5, "phi=0 -> 0.5");
}

#[test]
fn smoothstep_spans_the_feather_band() {
    // half_width = 2: coverage is 1 at phi <= -2, 0 at phi >= 2, monotone and
    // 0.5 at phi=0; the band is exactly 2*half_width = 4 pixels wide.
    let s = sdf(1, 5, vec![-2.0, -1.0, 0.0, 1.0, 2.0]);
    let cov = to_mask(&s, &serde_json::json!({ "half_width_px": 2.0 }));
    bits_eq(cov[0], 1.0, "phi=-2 -> 1 (band edge)");
    bits_eq(cov[4], 0.0, "phi=+2 -> 0 (band edge)");
    bits_eq(cov[2], 0.5, "phi=0 -> 0.5");
    // strictly monotone decreasing across the band
    assert!(cov[0] >= cov[1] && cov[1] >= cov[2] && cov[2] >= cov[3] && cov[3] >= cov[4]);
    assert!(cov[1] > cov[2] && cov[2] > cov[3], "strictly soft mid-band");
}

#[test]
fn reconstruction_maps_infinities() {
    // +inf (fully outside) -> 0, -inf (fully inside) -> 1, under a soft profile.
    assert_eq!(
        reconstruct_coverage(f32::INFINITY, 2.0).to_bits(),
        0.0f32.to_bits()
    );
    assert_eq!(
        reconstruct_coverage(f32::NEG_INFINITY, 2.0).to_bits(),
        1.0f32.to_bits()
    );
}

#[test]
fn feather_width_matches_half_width_param() {
    // The soft transition occupies exactly [-h, +h]: a sample just outside the
    // band is saturated, a sample inside is strictly between 0 and 1.
    let h = 3.0_f32;
    let s = sdf(1, 4, vec![-h - 0.1, -1.0, 1.0, h + 0.1]);
    let cov = to_mask(&s, &serde_json::json!({ "half_width_px": h }));
    bits_eq(cov[0], 1.0, "just past inner band edge -> saturated 1");
    bits_eq(cov[3], 0.0, "just past outer band edge -> saturated 0");
    assert!(cov[1] > 0.5 && cov[1] < 1.0, "inside band, partial");
    assert!(cov[2] > 0.0 && cov[2] < 0.5, "inside band, partial");
}

// --- rejection -------------------------------------------------------------

#[test]
fn offset_rejects_non_finite_distance() {
    let s = sdf(1, 1, vec![0.0]);
    let mut inputs = InputValues::new();
    inputs.insert("sdf".to_owned(), s);
    let err = SdfOffset::new()
        .compute(&inputs, &serde_json::json!({ "distance_px": "nope" }))
        .expect_err("non-numeric distance rejected");
    assert_eq!(err.class, ErrorClass::Schema);
}

#[test]
fn offset_requires_distance() {
    let s = sdf(1, 1, vec![0.0]);
    let mut inputs = InputValues::new();
    inputs.insert("sdf".to_owned(), s);
    let err = SdfOffset::new()
        .compute(&inputs, &serde_json::json!({}))
        .expect_err("missing distance rejected");
    assert_eq!(err.class, ErrorClass::Schema);
}

#[test]
fn to_mask_rejects_negative_half_width() {
    let s = sdf(1, 1, vec![0.0]);
    let mut inputs = InputValues::new();
    inputs.insert("sdf".to_owned(), s);
    let err = SdfToMask::new()
        .compute(&inputs, &serde_json::json!({ "half_width_px": -1.0 }))
        .expect_err("negative half-width rejected");
    assert_eq!(err.class, ErrorClass::Schema);
}

#[test]
fn to_mask_rejects_unknown_profile() {
    let s = sdf(1, 1, vec![0.0]);
    let mut inputs = InputValues::new();
    inputs.insert("sdf".to_owned(), s);
    let err = SdfToMask::new()
        .compute(&inputs, &serde_json::json!({ "profile": "linear" }))
        .expect_err("unknown profile rejected");
    assert_eq!(err.class, ErrorClass::Schema);
}

#[test]
fn ops_reject_missing_input() {
    let inputs = InputValues::new();
    let err = SdfOffset::new()
        .compute(&inputs, &serde_json::json!({ "distance_px": 1.0 }))
        .expect_err("missing sdf rejected");
    assert_eq!(err.class, ErrorClass::Reference);
    let err = SdfToMask::new()
        .compute(&inputs, &serde_json::json!({}))
        .expect_err("missing sdf rejected");
    assert_eq!(err.class, ErrorClass::Reference);
}

// --- contract output kinds -------------------------------------------------

#[test]
fn offset_outputs_sdf_to_mask_outputs_mask() {
    use paintop_ir::Descriptors;
    let mut d = Descriptors::new();
    d.insert(
        "sdf".to_owned(),
        ResourceDescriptor::SdfMask(SdfDescriptor {
            extent: Extent::new(2, 2),
            scalar: ScalarType::F32,
            units: SdfUnits::Pixels,
            sign: SdfSign::NegativeInside,
            coordinates: CoordinateConvention::PixelCenterUpperLeft,
        }),
    );
    let off = SdfOffset::new()
        .infer_outputs(&d, &serde_json::json!({ "distance_px": 0.0 }))
        .expect("offset infer");
    assert!(matches!(
        off.get("sdf"),
        Some(ResourceDescriptor::SdfMask(_))
    ));
    let tm = SdfToMask::new()
        .infer_outputs(&d, &serde_json::json!({}))
        .expect("to_mask infer");
    assert!(matches!(tm.get("mask"), Some(ResourceDescriptor::Mask(_))));
}
