//! Verification suite for `field.orientation@1` (`OP_CATALOG` §10.4):
//!
//! - **schema/contract**: the manifest validates, agrees with its contract, and
//!   gates clean;
//! - **analytic**: a tensor with a known dominant gradient recovers the expected
//!   orientation angle; the orientation is the *smaller*-eigenvalue eigenvector
//!   (perpendicular to the gradient);
//! - **coherence extremes**: a rank-1 tensor (clean edge) has coherence ~1; an
//!   isotropic tensor has coherence ~0 and a zero orientation vector;
//! - **sign convention**: the canonical representative has `ux >= 0`, and is a
//!   deterministic single-valued function of the tensor;
//! - **rotation covariance**: rotating the tensor rotates the orientation;
//! - **determinism**: a rerun is bit-identical.

use std::f64::consts::PI;

use paintop_core::executor::{InputValues, OpImplementation, ResourceValue};
use paintop_ir::{
    CoordinateConvention, Extent, FieldArity, FieldDescriptor, ResourceDescriptor, ScalarType,
    SemanticRole,
};

use super::{ORIENTATION_OP_ID, Orientation, analyze};

/// Wrap a flat (Jxx, Jxy, Jyy)-interleaved buffer as a Field3 tensor value.
fn tensor_field(width: u32, height: u32, samples: Vec<f32>) -> ResourceValue {
    let descriptor = ResourceDescriptor::Field3(FieldDescriptor {
        arity: FieldArity::Field3,
        extent: Extent::new(width, height),
        scalar: ScalarType::F32,
        semantic: SemanticRole::Material,
        coordinates: CoordinateConvention::PixelCenterUpperLeft,
        space: None,
        normalization: None,
        encoding: None,
    });
    ResourceValue::new(descriptor, 3, samples).expect("tensor buffer matches descriptor")
}

/// Run orientation and recover both outputs.
fn run(value: &ResourceValue) -> (ResourceValue, ResourceValue) {
    let mut inputs = InputValues::new();
    inputs.insert("tensor".to_owned(), value.clone());
    let mut out = Orientation::new()
        .compute(&inputs, &serde_json::json!({}))
        .expect("orientation computes");
    let orientation = out.remove("orientation").expect("orientation port");
    let coherence = out.remove("coherence").expect("coherence port");
    (orientation, coherence)
}

/// A rank-1 tensor `g gᵀ` for a unit gradient direction at angle `theta`.
fn rank1_tensor(theta: f64) -> (f64, f64, f64) {
    let gx = theta.cos();
    let gy = theta.sin();
    (gx * gx, gx * gy, gy * gy)
}

/// Narrow an f64 test value to the field's f32 storage type.
#[allow(
    clippy::cast_possible_truncation,
    reason = "test fixture tensor entries are small bounded values stored as f32"
)]
fn as_f32(v: f64) -> f32 {
    v as f32
}

/// A `width × height` Field3 filled with the same `(Jxx, Jxy, Jyy)` per pixel.
fn uniform_tensor(width: u32, height: u32, t: (f64, f64, f64)) -> ResourceValue {
    let n = (width as usize) * (height as usize);
    let mut samples = Vec::with_capacity(n * 3);
    for _ in 0..n {
        samples.push(as_f32(t.0));
        samples.push(as_f32(t.1));
        samples.push(as_f32(t.2));
    }
    tensor_field(width, height, samples)
}

#[test]
fn manifest_validates_and_agrees_with_contract() {
    let manifest = Orientation::manifest().expect("manifest");
    manifest.validate().expect("manifest valid");
    paintop_ir::check_contract_consistency(&manifest, &Orientation::new())
        .expect("manifest agrees with contract");
    paintop_ir::verify_categories(&manifest, &manifest.test.verification)
        .expect("verification declarations gate clean");
    assert_eq!(manifest.id.to_string(), ORIENTATION_OP_ID);
}

#[test]
fn orientation_is_perpendicular_to_a_known_gradient() {
    // Gradient along x (theta = 0): orientation (least variation) is along y.
    let (jxx, jxy, jyy) = rank1_tensor(0.0);
    let (ux, uy, coh) = analyze(jxx, jxy, jyy);
    // Orientation perpendicular to the x gradient -> +/- y axis; canonical sign
    // makes ux == 0, uy >= 0.
    assert!(ux.abs() < 1e-9, "ux should be ~0: {ux}");
    assert!((uy.abs() - 1.0).abs() < 1e-9, "|uy| should be 1: {uy}");
    assert!(uy >= 0.0, "canonical sign: ux==0 => uy>=0, got {uy}");
    assert!(
        (coh - 1.0).abs() < 1e-9,
        "rank-1 tensor coherence ~1: {coh}"
    );
}

#[test]
fn grating_angle_is_recovered() {
    // For a gradient at angle theta, orientation is at theta + 90deg (mod 180).
    for deg in [0.0_f64, 30.0, 45.0, 60.0, 120.0, 150.0] {
        let theta = deg.to_radians();
        let (jxx, jxy, jyy) = rank1_tensor(theta);
        let (ux, uy, _coh) = analyze(jxx, jxy, jyy);
        // The orientation direction's angle (mod PI) should equal theta + PI/2.
        let got = uy.atan2(ux).rem_euclid(PI);
        let want = (theta + PI / 2.0).rem_euclid(PI);
        let diff = (got - want).rem_euclid(PI);
        let diff = diff.min(PI - diff);
        assert!(diff < 1e-6, "deg {deg}: orientation angle {got} != {want}");
    }
}

#[test]
fn isotropic_tensor_has_zero_orientation_and_coherence() {
    // J = I: equal eigenvalues, no preferred direction.
    let (ux, uy, coh) = analyze(1.0, 0.0, 1.0);
    assert!(
        ux.abs() < 1e-12 && uy.abs() < 1e-12,
        "isotropic -> zero vector"
    );
    assert!(coh.abs() < 1e-12, "isotropic coherence ~0: {coh}");
}

#[test]
fn zero_tensor_is_degenerate() {
    let (ux, uy, coh) = analyze(0.0, 0.0, 0.0);
    assert_eq!((ux, uy, coh), (0.0, 0.0, 0.0));
}

#[test]
fn coherence_is_high_for_a_clean_edge_low_for_blob() {
    // Clean edge: rank-1 tensor -> coherence ~1.
    let (jxx, jxy, jyy) = rank1_tensor(0.3);
    let (_ux, _uy, edge) = analyze(jxx, jxy, jyy);
    assert!(edge > 0.99, "clean edge coherence ~1: {edge}");
    // Nearly isotropic: eigenvalues 1.0 and 0.9 -> low coherence.
    let (_a, _b, blob) = analyze(1.0, 0.0, 0.9);
    let expected = (0.1_f64 / 1.9_f64).powi(2);
    assert!(
        (blob - expected).abs() < 1e-9,
        "coherence {blob} != {expected}"
    );
    assert!(blob < 0.01, "near-isotropic coherence is small: {blob}");
}

#[test]
fn sign_convention_is_canonical_and_deterministic() {
    // Two tensors that are eigen-equivalent up to eigenvector sign must produce
    // the same canonical orientation. analyze is a pure function, so equal inputs
    // give equal outputs; verify ux >= 0 across a sweep.
    for deg in 0..180i32 {
        let theta = f64::from(deg).to_radians();
        let (jxx, jxy, jyy) = rank1_tensor(theta);
        let (ux, _uy, _c) = analyze(jxx, jxy, jyy);
        assert!(ux >= -1e-12, "canonical ux must be >= 0 at deg {deg}: {ux}");
    }
}

#[test]
fn output_descriptors_and_extents_are_correct() {
    let field = uniform_tensor(3, 4, rank1_tensor(0.5));
    let (orientation, coherence) = run(&field);
    assert_eq!(orientation.extent(), Extent::new(3, 4));
    assert_eq!(orientation.channels(), 2);
    assert!(matches!(
        orientation.descriptor(),
        ResourceDescriptor::Field2(_)
    ));
    assert_eq!(coherence.extent(), Extent::new(3, 4));
    assert_eq!(coherence.channels(), 1);
    assert!(matches!(
        coherence.descriptor(),
        ResourceDescriptor::Field1(_)
    ));
}

#[test]
fn rerun_is_bit_identical() {
    let field = uniform_tensor(4, 4, rank1_tensor(0.7));
    let (o1, c1) = run(&field);
    let (o2, c2) = run(&field);
    assert_eq!(o1.samples(), o2.samples());
    assert_eq!(c1.samples(), c2.samples());
}

#[test]
fn rotation_covariance_of_orientation() {
    // Rotating the gradient by alpha rotates the orientation by alpha (mod PI).
    let theta = 0.2;
    let alpha = 0.5;
    let t0 = rank1_tensor(theta);
    let t1 = rank1_tensor(theta + alpha);
    let (ux0, uy0, _) = analyze(t0.0, t0.1, t0.2);
    let (ux1, uy1, _) = analyze(t1.0, t1.1, t1.2);
    let ang0 = uy0.atan2(ux0).rem_euclid(PI);
    let ang1 = uy1.atan2(ux1).rem_euclid(PI);
    let got = (ang1 - ang0).rem_euclid(PI);
    let want = alpha.rem_euclid(PI);
    let diff = (got - want).rem_euclid(PI);
    let diff = diff.min(PI - diff);
    assert!(diff < 1e-6, "rotation covariance: delta {got} != {want}");
}

#[test]
fn wrong_input_kind_is_rejected() {
    use paintop_ir::{
        AlphaRepresentation, ChannelLayout, ColorEncoding, ColorRange, ImageDescriptor,
    };
    let img = ResourceValue::new(
        ResourceDescriptor::Image(ImageDescriptor {
            extent: Extent::new(2, 2),
            layout: ChannelLayout::Gray,
            scalar: ScalarType::F32,
            color: ColorEncoding::LinearSrgb,
            range: ColorRange::SceneReferred,
            alpha: AlphaRepresentation::Premultiplied,
            coordinates: CoordinateConvention::PixelCenterUpperLeft,
            semantic: SemanticRole::Color,
        }),
        1,
        vec![0.0; 4],
    )
    .expect("image");
    let mut inputs = InputValues::new();
    inputs.insert("tensor".to_owned(), img);
    let err = Orientation::new()
        .compute(&inputs, &serde_json::json!({}))
        .expect_err("a non-Field3 tensor must be rejected");
    assert_eq!(err.code, super::E_ORIENTATION_INPUT);
}
