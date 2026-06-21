//! Verification suite for `sdf.union@1`, `sdf.intersect@1`, `sdf.subtract@1`
//! (`AGENT_VERIFICATION` §3.7):
//!
//! - **schema/contract**: each manifest validates, agrees with its contract,
//!   gates clean, and declares the `negative-inside` output;
//! - **analytic fixtures**: `min`/`max`/`max(a, -b)` on hand-computed samples;
//! - **zero-contour agreement** (the headline acceptance): thresholding the SDF
//!   boolean at `φ < 0` reproduces the hard-mask boolean (`mask.union` /
//!   `mask.intersect` / `mask.subtract`) of the operand inside sets exactly, built
//!   end-to-end through `mask.to_sdf`;
//! - **algebra laws**: union/intersect commute; subtract obeys the difference law
//!   `(A − B) zero-set = A ∩ ¬B`;
//! - **gradient norm**: `|∇φ| ≈ 1` is preserved away from the new medial axis;
//! - **infinities**: degenerate `±∞` operands combine without `NaN`;
//! - **rejection**: extent mismatch / missing input are typed errors.

use paintop_core::executor::{InputValues, OpImplementation, ResourceValue};
use paintop_ir::{
    CoordinateConvention, Descriptors, ErrorClass, Extent, MaskDescriptor, MaskMeaning, OpContract,
    ResourceDescriptor, ScalarType, SdfDescriptor, SdfSign, SdfUnits, check_contract_consistency,
    verify_categories,
};

use crate::mask_algebra::BinaryMaskOp;
use crate::mask_to_sdf::MaskToSdf;

use super::{INTERSECT_OP_ID, SUBTRACT_OP_ID, SdfBooleanOp, UNION_OP_ID};

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

/// Build a coverage-mask value from explicit samples sized `w * h`.
fn mask(w: u32, h: u32, samples: Vec<f32>) -> ResourceValue {
    assert_eq!(samples.len(), (w * h) as usize, "sample count");
    let descriptor = ResourceDescriptor::Mask(MaskDescriptor {
        extent: Extent::new(w, h),
        scalar: ScalarType::F32,
        range: paintop_ir::ValidRange::Bounded { min: 0.0, max: 1.0 },
        meaning: MaskMeaning::Coverage,
        coordinates: CoordinateConvention::PixelCenterUpperLeft,
    });
    ResourceValue::new(descriptor, 1, samples).expect("mask value")
}

/// Apply a binary SDF op to two fields, returning the output samples.
fn apply(op: SdfBooleanOp, a: &ResourceValue, b: &ResourceValue) -> Vec<f32> {
    let mut inputs = InputValues::new();
    inputs.insert("a".to_owned(), a.clone());
    inputs.insert("b".to_owned(), b.clone());
    let mut out = op.compute(&inputs, &serde_json::json!({})).expect("sdf op");
    out.remove("sdf").expect("sdf output").into_samples()
}

/// Run `mask.to_sdf` on a hard mask, returning the field value.
fn to_sdf(m: &ResourceValue) -> ResourceValue {
    let mut inputs = InputValues::new();
    inputs.insert("mask".to_owned(), m.clone());
    let mut out = MaskToSdf::new()
        .compute(&inputs, &serde_json::json!({}))
        .expect("to_sdf");
    out.remove("sdf").expect("sdf output")
}

/// Apply a hard-mask boolean op, returning the output samples.
fn mask_bool(op: BinaryMaskOp, a: &ResourceValue, b: &ResourceValue) -> Vec<f32> {
    let mut inputs = InputValues::new();
    inputs.insert("a".to_owned(), a.clone());
    inputs.insert("b".to_owned(), b.clone());
    let mut out = op
        .compute(&inputs, &serde_json::json!({}))
        .expect("mask op");
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
    let cases = [
        (
            SdfBooleanOp::union_manifest().expect("union"),
            UNION_OP_ID,
            SdfBooleanOp::union(),
        ),
        (
            SdfBooleanOp::intersect_manifest().expect("intersect"),
            INTERSECT_OP_ID,
            SdfBooleanOp::intersect(),
        ),
        (
            SdfBooleanOp::subtract_manifest().expect("subtract"),
            SUBTRACT_OP_ID,
            SdfBooleanOp::subtract(),
        ),
    ];
    for (manifest, id, op) in cases {
        manifest.validate().expect("manifest valid");
        verify_categories(&manifest, &manifest.test.verification).expect("gates clean");
        assert_eq!(manifest.id.to_string(), id);
        check_contract_consistency(&manifest, &op).expect("contract consistent");
    }
}

// --- analytic fixtures -----------------------------------------------------

#[test]
fn union_intersect_subtract_fixtures() {
    let field_a = sdf(1, 3, vec![-2.0, 1.0, 0.5]);
    let field_b = sdf(1, 3, vec![1.0, -1.0, 0.5]);
    // union = min
    let uni = apply(SdfBooleanOp::union(), &field_a, &field_b);
    bits_eq(uni[0], -2.0, "union 0");
    bits_eq(uni[1], -1.0, "union 1");
    bits_eq(uni[2], 0.5, "union 2");
    // intersect = max
    let inter = apply(SdfBooleanOp::intersect(), &field_a, &field_b);
    bits_eq(inter[0], 1.0, "intersect 0");
    bits_eq(inter[1], 1.0, "intersect 1");
    bits_eq(inter[2], 0.5, "intersect 2");
    // subtract = max(a, -b)
    let diff = apply(SdfBooleanOp::subtract(), &field_a, &field_b);
    bits_eq(diff[0], -1.0, "subtract 0"); // max(-2, -1) = -1
    bits_eq(diff[1], 1.0, "subtract 1"); // max(1, 1) = 1
    bits_eq(diff[2], 0.5, "subtract 2"); // max(0.5, -0.5) = 0.5
}

// --- zero-contour agreement (headline acceptance) --------------------------

/// Two overlapping hard rectangles in a 9x9 field, as `(a, b)` masks.
fn overlapping_rects() -> (ResourceValue, ResourceValue) {
    let w = 9;
    let h = 9;
    let mut a = vec![0.0_f32; (w * h) as usize];
    let mut b = vec![0.0_f32; (w * h) as usize];
    for y in 1..6 {
        for x in 1..6 {
            a[(y * w + x) as usize] = 1.0;
        }
    }
    for y in 3..8 {
        for x in 3..8 {
            b[(y * w + x) as usize] = 1.0;
        }
    }
    (mask(w, h, a), mask(w, h, b))
}

/// The inside set of a field (`φ < 0`) as a 0/1 coverage buffer, matching the
/// hard-mask semantics.
fn inside_set(field: &[f32]) -> Vec<f32> {
    field
        .iter()
        .map(|&phi| if phi < 0.0 { 1.0 } else { 0.0 })
        .collect()
}

#[test]
fn union_zero_contour_matches_hard_mask_union() {
    let (ma, mb) = overlapping_rects();
    let sa = to_sdf(&ma);
    let sb = to_sdf(&mb);
    let sdf_union_inside = inside_set(&apply(SdfBooleanOp::union(), &sa, &sb));
    let hard_union = mask_bool(BinaryMaskOp::union(), &ma, &mb);
    assert_eq!(sdf_union_inside, hard_union, "union zero-contour agreement");
}

#[test]
fn intersect_zero_contour_matches_hard_mask_intersect() {
    let (ma, mb) = overlapping_rects();
    let sa = to_sdf(&ma);
    let sb = to_sdf(&mb);
    let sdf_inside = inside_set(&apply(SdfBooleanOp::intersect(), &sa, &sb));
    let hard = mask_bool(BinaryMaskOp::intersect(), &ma, &mb);
    assert_eq!(sdf_inside, hard, "intersect zero-contour agreement");
}

#[test]
fn subtract_zero_contour_matches_hard_mask_subtract() {
    let (ma, mb) = overlapping_rects();
    let sa = to_sdf(&ma);
    let sb = to_sdf(&mb);
    let sdf_inside = inside_set(&apply(SdfBooleanOp::subtract(), &sa, &sb));
    let hard = mask_bool(BinaryMaskOp::subtract(), &ma, &mb);
    assert_eq!(sdf_inside, hard, "subtract zero-contour agreement");
}

// --- algebra laws ----------------------------------------------------------

#[test]
fn union_and_intersect_commute() {
    let a = sdf(2, 2, vec![-1.0, 2.0, 0.0, 3.0]);
    let b = sdf(2, 2, vec![1.5, -2.0, 0.5, -0.5]);
    assert_eq!(
        apply(SdfBooleanOp::union(), &a, &b),
        apply(SdfBooleanOp::union(), &b, &a),
        "union commutes"
    );
    assert_eq!(
        apply(SdfBooleanOp::intersect(), &a, &b),
        apply(SdfBooleanOp::intersect(), &b, &a),
        "intersect commutes"
    );
}

#[test]
fn subtract_is_intersect_with_negated_b() {
    // A - B = A ∩ (¬B). With the negative-inside convention, ¬B is the field -φ_B,
    // so subtract(a, b) must equal intersect(a, negate(b)).
    let a = sdf(1, 4, vec![-1.0, 0.5, -2.0, 3.0]);
    let b = sdf(1, 4, vec![1.0, -0.5, 2.0, -3.0]);
    let neg_b = sdf(1, 4, b.samples().iter().map(|&x| -x).collect::<Vec<_>>());
    assert_eq!(
        apply(SdfBooleanOp::subtract(), &a, &b),
        apply(SdfBooleanOp::intersect(), &a, &neg_b),
        "A - B = A ∩ ¬B"
    );
}

// --- gradient norm ---------------------------------------------------------

#[test]
fn gradient_norm_is_preserved_away_from_medial_axis() {
    // Two disjoint half-plane fields: union of two unit-gradient fields keeps a
    // unit gradient everywhere except the seam where the nearer field switches.
    // Sample a 1-D union of two linear fields and check the central difference is
    // +-1 away from the single switch point.
    let n: u32 = 11;
    // field_a: phi increases left->right (boundary at x=2); field_b: phi
    // decreases (boundary at x=8). Both have |slope| = 1.
    let field_a: Vec<f32> = (0..n)
        .map(|x| f32::from(u16::try_from(x).unwrap()) - 2.0)
        .collect();
    let field_b: Vec<f32> = (0..n)
        .map(|x| 8.0 - f32::from(u16::try_from(x).unwrap()))
        .collect();
    let sa = sdf(n, 1, field_a);
    let sb = sdf(n, 1, field_b);
    let unioned = apply(SdfBooleanOp::union(), &sa, &sb);
    // central difference magnitude is 1 everywhere except at the medial axis
    // (the single crossover near the middle).
    let mut unit_count: u32 = 0;
    for x in 1..(n as usize) - 1 {
        let grad = (unioned[x + 1] - unioned[x - 1]) / 2.0;
        if (grad.abs() - 1.0).abs() < 1e-6 {
            unit_count += 1;
        }
    }
    // All but the one medial-axis sample have unit gradient.
    assert!(unit_count >= n - 3, "unit gradient away from medial axis");
}

// --- infinities ------------------------------------------------------------

#[test]
fn infinities_combine_without_nan() {
    let a = sdf(1, 3, vec![f32::NEG_INFINITY, f32::INFINITY, 0.0]);
    let b = sdf(1, 3, vec![f32::INFINITY, f32::NEG_INFINITY, f32::INFINITY]);
    for op in [
        SdfBooleanOp::union(),
        SdfBooleanOp::intersect(),
        SdfBooleanOp::subtract(),
    ] {
        for &s in &apply(op, &a, &b) {
            assert!(!s.is_nan(), "no NaN from infinity combination");
        }
    }
}

// --- rejection -------------------------------------------------------------

#[test]
fn rejects_extent_mismatch() {
    let a = sdf(2, 2, vec![0.0; 4]);
    let b = sdf(3, 1, vec![0.0; 3]);
    let mut inputs = InputValues::new();
    inputs.insert("a".to_owned(), a);
    inputs.insert("b".to_owned(), b);
    let err = SdfBooleanOp::union()
        .compute(&inputs, &serde_json::json!({}))
        .expect_err("extent mismatch rejected");
    assert_eq!(err.class, ErrorClass::Type);
}

#[test]
fn rejects_missing_input() {
    let mut inputs = InputValues::new();
    inputs.insert("a".to_owned(), sdf(1, 1, vec![0.0]));
    let err = SdfBooleanOp::intersect()
        .compute(&inputs, &serde_json::json!({}))
        .expect_err("missing b rejected");
    assert_eq!(err.class, ErrorClass::Reference);
}

#[test]
fn output_is_negative_inside() {
    let mut d = Descriptors::new();
    let desc = ResourceDescriptor::SdfMask(SdfDescriptor {
        extent: Extent::new(2, 2),
        scalar: ScalarType::F32,
        units: SdfUnits::Pixels,
        sign: SdfSign::NegativeInside,
        coordinates: CoordinateConvention::PixelCenterUpperLeft,
    });
    d.insert("a".to_owned(), desc);
    d.insert("b".to_owned(), desc);
    let out = SdfBooleanOp::union()
        .infer_outputs(&d, &serde_json::json!({}))
        .expect("infer");
    let Some(ResourceDescriptor::SdfMask(s)) = out.get("sdf") else {
        panic!("sdf output");
    };
    assert_eq!(s.sign, SdfSign::NegativeInside);
}
