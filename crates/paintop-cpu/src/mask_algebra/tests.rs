//! Verification suite for the mask boolean-algebra ops `mask.invert@1`,
//! `mask.union@1`, `mask.intersect@1`, `mask.subtract@1` (`OP_CATALOG` §4,
//! `AGENT_VERIFICATION` §2.5):
//!
//! - **schema/contract**: each manifest validates, agrees with its contract,
//!   gates clean, and the checked-in manifest matches the Rust builder;
//! - **analytic fixtures**: invert is `1 - a`, union/intersect/subtract are
//!   `max`/`min`/`min(a, 1 - b)` on hand-computed inputs;
//! - **hard-mask law suite** (§2.5): commutativity, associativity, idempotence,
//!   complement laws, De Morgan, `A - A = empty`, double-inverse — all
//!   bit-identical on `{0, 1}` masks;
//! - **soft-mask laws**: the lattice / De Morgan / double-inverse laws hold for
//!   arbitrary coverage, while the excluded-middle laws do **not** (the fuzzy
//!   distinction);
//! - **rejection**: an extent mismatch / missing input is a typed error.

use paintop_core::executor::{InputValues, OpImplementation, ResourceValue};
use paintop_ir::{
    CoordinateConvention, Descriptors, ErrorClass, Extent, MaskDescriptor, MaskMeaning, OpContract,
    ResourceDescriptor, ScalarType, ValidRange, check_contract_consistency, verify_categories,
};

use super::{BinaryMaskOp, INTERSECT_OP_ID, INVERT_OP_ID, InvertMask, SUBTRACT_OP_ID, UNION_OP_ID};

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

/// Apply the unary invert op to a mask, returning the output samples.
fn invert(a: &ResourceValue) -> Vec<f32> {
    let mut inputs = InputValues::new();
    inputs.insert("mask".to_owned(), a.clone());
    let mut out = InvertMask::new()
        .compute(&inputs, &serde_json::json!({}))
        .expect("invert computes");
    out.remove("mask").expect("mask").into_samples()
}

/// Apply a binary op to two masks, returning the output samples.
fn binary(op: BinaryMaskOp, a: &ResourceValue, b: &ResourceValue) -> Vec<f32> {
    let mut inputs = InputValues::new();
    inputs.insert("a".to_owned(), a.clone());
    inputs.insert("b".to_owned(), b.clone());
    let mut out = op
        .compute(&inputs, &serde_json::json!({}))
        .expect("binary computes");
    out.remove("mask").expect("mask").into_samples()
}

/// Assert two sample buffers are bit-identical.
fn assert_bits_eq(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len(), "length");
    for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        assert_eq!(a.to_bits(), e.to_bits(), "sample {i}: {a} != {e}");
    }
}

// --- schema / contract -----------------------------------------------------

#[test]
fn manifests_validate_and_agree_with_contracts() {
    let cases: [(paintop_ir::OperationManifest, &str); 4] = [
        (InvertMask::manifest().expect("invert"), INVERT_OP_ID),
        (BinaryMaskOp::union_manifest().expect("union"), UNION_OP_ID),
        (
            BinaryMaskOp::intersect_manifest().expect("intersect"),
            INTERSECT_OP_ID,
        ),
        (
            BinaryMaskOp::subtract_manifest().expect("subtract"),
            SUBTRACT_OP_ID,
        ),
    ];
    for (manifest, id) in cases {
        manifest.validate().expect("manifest valid");
        verify_categories(&manifest, &manifest.test.verification)
            .expect("verification declarations gate clean");
        assert_eq!(manifest.id.to_string(), id);
    }
    check_contract_consistency(&InvertMask::manifest().unwrap(), &InvertMask::new()).unwrap();
    check_contract_consistency(
        &BinaryMaskOp::union_manifest().unwrap(),
        &BinaryMaskOp::union(),
    )
    .unwrap();
    check_contract_consistency(
        &BinaryMaskOp::intersect_manifest().unwrap(),
        &BinaryMaskOp::intersect(),
    )
    .unwrap();
    check_contract_consistency(
        &BinaryMaskOp::subtract_manifest().unwrap(),
        &BinaryMaskOp::subtract(),
    )
    .unwrap();
}

// --- analytic fixtures -----------------------------------------------------

#[test]
fn invert_is_one_minus_a() {
    let a = mask(2, 2, vec![0.0, 1.0, 0.25, 0.75]);
    assert_bits_eq(&invert(&a), &[1.0, 0.0, 0.75, 0.25]);
}

#[test]
fn union_intersect_subtract_fixtures() {
    let a = mask(1, 4, vec![0.0, 1.0, 0.2, 0.8]);
    let b = mask(1, 4, vec![1.0, 0.0, 0.5, 0.3]);
    assert_bits_eq(
        &binary(BinaryMaskOp::union(), &a, &b),
        &[1.0, 1.0, 0.5, 0.8],
    );
    assert_bits_eq(
        &binary(BinaryMaskOp::intersect(), &a, &b),
        &[0.0, 0.0, 0.2, 0.3],
    );
    // a - b = min(a, 1 - b)
    assert_bits_eq(
        &binary(BinaryMaskOp::subtract(), &a, &b),
        &[0.0, 1.0, 0.2, 0.7],
    );
}

// --- hard-mask law suite (AGENT_VERIFICATION §2.5) -------------------------

/// A spread of hard `{0, 1}` masks to exercise the boolean laws over every
/// pairwise combination of cells.
fn hard_masks() -> (
    ResourceValue,
    ResourceValue,
    ResourceValue,
    ResourceValue,
    ResourceValue,
) {
    // Cover all 8 combinations of (a, b, c) bits across 8 pixels.
    let a = mask(1, 8, vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]);
    let b = mask(1, 8, vec![0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0]);
    let c = mask(1, 8, vec![0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]);
    let empty = mask(1, 8, vec![0.0; 8]);
    let full = mask(1, 8, vec![1.0; 8]);
    (a, b, c, empty, full)
}

#[test]
fn hard_commutativity() {
    let (a, b, ..) = hard_masks();
    assert_bits_eq(
        &binary(BinaryMaskOp::union(), &a, &b),
        &binary(BinaryMaskOp::union(), &b, &a),
    );
    assert_bits_eq(
        &binary(BinaryMaskOp::intersect(), &a, &b),
        &binary(BinaryMaskOp::intersect(), &b, &a),
    );
}

#[test]
fn hard_associativity() {
    let (a, b, c, ..) = hard_masks();
    for op in [BinaryMaskOp::union(), BinaryMaskOp::intersect()] {
        let ab = mask(1, 8, binary(op, &a, &b));
        let bc = mask(1, 8, binary(op, &b, &c));
        let left = binary(op, &ab, &c);
        let right = binary(op, &a, &bc);
        assert_bits_eq(&left, &right);
    }
}

#[test]
fn hard_idempotence() {
    let (a, ..) = hard_masks();
    assert_bits_eq(&binary(BinaryMaskOp::union(), &a, &a), a.samples());
    assert_bits_eq(&binary(BinaryMaskOp::intersect(), &a, &a), a.samples());
}

#[test]
fn hard_complement_laws() {
    let (a, _b, _c, empty, full) = hard_masks();
    let not_a = mask(1, 8, invert(&a));
    // A ∪ ¬A = full
    assert_bits_eq(&binary(BinaryMaskOp::union(), &a, &not_a), full.samples());
    // A ∩ ¬A = empty
    assert_bits_eq(
        &binary(BinaryMaskOp::intersect(), &a, &not_a),
        empty.samples(),
    );
}

#[test]
fn de_morgan_holds_for_all_coverage() {
    // De Morgan holds for the min-max algebra at arbitrary coverage, not only
    // hard masks: ¬(A ∪ B) = ¬A ∩ ¬B and ¬(A ∩ B) = ¬A ∪ ¬B.
    let a = mask(1, 5, vec![0.0, 0.3, 0.5, 0.9, 1.0]);
    let b = mask(1, 5, vec![1.0, 0.7, 0.5, 0.1, 0.0]);
    let not_a = mask(1, 5, invert(&a));
    let not_b = mask(1, 5, invert(&b));

    let lhs_union = invert(&mask(1, 5, binary(BinaryMaskOp::union(), &a, &b)));
    let rhs_union = binary(BinaryMaskOp::intersect(), &not_a, &not_b);
    assert_bits_eq(&lhs_union, &rhs_union);

    let lhs_inter = invert(&mask(1, 5, binary(BinaryMaskOp::intersect(), &a, &b)));
    let rhs_inter = binary(BinaryMaskOp::union(), &not_a, &not_b);
    assert_bits_eq(&lhs_inter, &rhs_inter);
}

#[test]
fn hard_subtract_self_is_empty() {
    let (a, _b, _c, empty, _full) = hard_masks();
    // A - A = ∅ on hard masks.
    assert_bits_eq(&binary(BinaryMaskOp::subtract(), &a, &a), empty.samples());
}

#[test]
fn double_inverse_is_identity() {
    // ¬¬a = a is exact for dyadic coverage values (every sample here round-trips
    // bit-identically under 1 - (1 - a)); it holds for hard masks too.
    let a = mask(1, 5, vec![0.0, 0.25, 0.5, 0.75, 1.0]);
    let once = mask(1, 5, invert(&a));
    assert_bits_eq(&invert(&once), a.samples());
}

#[test]
fn subtract_equals_intersect_not_b() {
    // a - b ≡ a ∩ ¬b for arbitrary coverage (definition of relative complement).
    let a = mask(1, 5, vec![0.0, 0.3, 0.5, 0.9, 1.0]);
    let b = mask(1, 5, vec![1.0, 0.7, 0.5, 0.1, 0.0]);
    let not_b = mask(1, 5, invert(&b));
    let lhs = binary(BinaryMaskOp::subtract(), &a, &b);
    let rhs = binary(BinaryMaskOp::intersect(), &a, &not_b);
    assert_bits_eq(&lhs, &rhs);
}

// --- soft-mask fuzzy distinction -------------------------------------------

#[test]
fn soft_excluded_middle_does_not_hold() {
    // The fuzzy algebra deliberately violates the excluded middle at a = 0.5:
    // A ∪ ¬A = 0.5 ≠ 1, A ∩ ¬A = 0.5 ≠ 0. This is what makes the soft algebra
    // fuzzy rather than crisp.
    let a = mask(1, 1, vec![0.5]);
    let not_a = mask(1, 1, invert(&a));
    assert_eq!(
        binary(BinaryMaskOp::union(), &a, &not_a)[0].to_bits(),
        0.5_f32.to_bits()
    );
    assert_eq!(
        binary(BinaryMaskOp::intersect(), &a, &not_a)[0].to_bits(),
        0.5_f32.to_bits()
    );
}

#[test]
fn output_is_bounded_and_finite() {
    let a = mask(1, 4, vec![0.0, 0.5, 1.0, 0.25]);
    let b = mask(1, 4, vec![1.0, 0.5, 0.0, 0.75]);
    for op in [
        BinaryMaskOp::union(),
        BinaryMaskOp::intersect(),
        BinaryMaskOp::subtract(),
    ] {
        for &s in &binary(op, &a, &b) {
            assert!(s.is_finite(), "must be finite: {s}");
            assert!((0.0..=1.0).contains(&s), "out of range: {s}");
        }
    }
    for &s in &invert(&a) {
        assert!((0.0..=1.0).contains(&s), "invert out of range: {s}");
    }
}

// --- rejection -------------------------------------------------------------

#[test]
fn binary_rejects_extent_mismatch() {
    let a = mask(2, 2, vec![0.0; 4]);
    let b = mask(3, 3, vec![0.0; 9]);
    let mut inputs = Descriptors::new();
    inputs.insert("a".to_owned(), *a.descriptor());
    inputs.insert("b".to_owned(), *b.descriptor());
    let err = BinaryMaskOp::union()
        .infer_outputs(&inputs, &serde_json::json!({}))
        .expect_err("extent mismatch must be rejected");
    assert_eq!(err.class, ErrorClass::Type);
    assert_eq!(err.code, super::E_ALGEBRA_SHAPE);
}

#[test]
fn missing_input_is_rejected() {
    let inputs = Descriptors::new();
    let err = InvertMask::new()
        .infer_outputs(&inputs, &serde_json::json!({}))
        .expect_err("missing input must be rejected");
    assert_eq!(err.class, ErrorClass::Reference);
    assert_eq!(err.code, super::E_ALGEBRA_INPUT);
}

/// The checked-in `ops/manifests/<id>.json` files must stay byte-identical to the
/// Rust manifest builders.
#[test]
fn checked_in_manifests_match_builders() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("crates dir")
        .parent()
        .expect("repo root")
        .join("ops/manifests");
    for manifest in [
        InvertMask::manifest().expect("invert"),
        BinaryMaskOp::union_manifest().expect("union"),
        BinaryMaskOp::intersect_manifest().expect("intersect"),
        BinaryMaskOp::subtract_manifest().expect("subtract"),
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
