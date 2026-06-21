//! Verification suite for the two MVP assertions `assert.no_change_outside_mask@1`
//! and `assert.finite@1` (`IR_SPEC` §13, `OP_CATALOG` §12, `AGENT_VERIFICATION`
//! §5.3):
//!
//! - **schema/contract**: each manifest validates, agrees with its contract, and
//!   its verification declarations gate clean; the checked-in `ops/manifests`
//!   JSON stays byte-identical to the Rust builder;
//! - **analytic fixtures**: a localized edit (change only where the mask allows)
//!   passes; an edit leaking a single pixel outside the mask fails with the worst
//!   pixel, the leaking count, and the outside delta; a NaN-injected resource
//!   fails `assert.finite` with the non-finite locations;
//! - **property/severity**: the verdict and metrics are independent of severity;
//!   `metric` severity records a failing verdict but does not fail the run; the
//!   comparison space is honored; `coverage_epsilon` widens the outside region.

use paintop_core::executor::{InputValues, OpImplementation, OutputValues, ResourceValue};
use paintop_ir::{
    AlphaRepresentation, AssertionOutcome, AssertionSeverity, ChannelLayout, ColorEncoding,
    ColorRange, CoordinateConvention, Extent, ImageDescriptor, MaskDescriptor, MaskMeaning, Report,
    ResourceDescriptor, ScalarType, SemanticRole, ValidRange, check_contract_consistency,
    verify_categories,
};

use super::{FINITE_OP_ID, Finite, NO_CHANGE_OUTSIDE_MASK_OP_ID, NoChangeOutsideMask};

/// Build an image [`ResourceValue`] of `channels` from an explicit sample buffer.
fn image_value(
    width: u32,
    height: u32,
    channels: u32,
    encoding: ColorEncoding,
    samples: Vec<f32>,
) -> ResourceValue {
    let layout = match channels {
        1 => ChannelLayout::Gray,
        2 => ChannelLayout::GrayA,
        3 => ChannelLayout::Rgb,
        _ => ChannelLayout::Rgba,
    };
    let descriptor = ResourceDescriptor::Image(ImageDescriptor {
        extent: Extent::new(width, height),
        layout,
        scalar: ScalarType::F32,
        color: encoding,
        range: ColorRange::SceneReferred,
        alpha: AlphaRepresentation::Straight,
        coordinates: CoordinateConvention::PixelCenterUpperLeft,
        semantic: SemanticRole::Color,
    });
    ResourceValue::new(descriptor, channels, samples).expect("sample buffer matches descriptor")
}

/// Build a coverage mask [`ResourceValue`] from an explicit `[0,1]` buffer.
fn mask_value(width: u32, height: u32, samples: Vec<f32>) -> ResourceValue {
    let descriptor = ResourceDescriptor::Mask(MaskDescriptor {
        extent: Extent::new(width, height),
        scalar: ScalarType::F32,
        range: ValidRange::Bounded { min: 0.0, max: 1.0 },
        meaning: MaskMeaning::Coverage,
        coordinates: CoordinateConvention::PixelCenterUpperLeft,
    });
    ResourceValue::new(descriptor, 1, samples).expect("mask buffer matches descriptor")
}

/// Run `assert.no_change_outside_mask` and recover its verdict.
fn run_no_change(
    before: &ResourceValue,
    after: &ResourceValue,
    allowed: &ResourceValue,
    params: &serde_json::Value,
) -> AssertionOutcome {
    let mut inputs = InputValues::new();
    inputs.insert("before".to_owned(), before.clone());
    inputs.insert("after".to_owned(), after.clone());
    inputs.insert("allowed".to_owned(), allowed.clone());
    let out: OutputValues = NoChangeOutsideMask::new()
        .compute(&inputs, params)
        .expect("assertion computes");
    report_outcome(&out)
}

/// Run `assert.finite` and recover its verdict.
fn run_finite(resource: &ResourceValue, params: &serde_json::Value) -> AssertionOutcome {
    let mut inputs = InputValues::new();
    inputs.insert("resource".to_owned(), resource.clone());
    let out: OutputValues = Finite::new()
        .compute(&inputs, params)
        .expect("assertion computes");
    report_outcome(&out)
}

/// Extract the assertion outcome from an op's `report` output.
fn report_outcome(out: &OutputValues) -> AssertionOutcome {
    let report: &Report = out
        .get("report")
        .expect("report port produced")
        .as_report()
        .expect("report payload present");
    report.assertion.clone().expect("assertion verdict present")
}

// --- schema / contract -----------------------------------------------------

#[test]
fn manifests_validate_and_agree_with_contracts() {
    let m = NoChangeOutsideMask::manifest().expect("manifest");
    m.validate().expect("valid");
    check_contract_consistency(&m, &NoChangeOutsideMask::new()).expect("agrees with contract");
    verify_categories(&m, &m.test.verification).expect("declarations gate clean");
    assert_eq!(m.id.to_string(), NO_CHANGE_OUTSIDE_MASK_OP_ID);

    let f = Finite::manifest().expect("manifest");
    f.validate().expect("valid");
    check_contract_consistency(&f, &Finite::new()).expect("agrees with contract");
    verify_categories(&f, &f.test.verification).expect("declarations gate clean");
    assert_eq!(f.id.to_string(), FINITE_OP_ID);
}

/// The checked-in `ops/manifests/<id>.json` files must stay byte-identical to the
/// Rust manifest builders, the source of truth.
#[test]
fn checked_in_manifests_match_builders() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("crates dir")
        .parent()
        .expect("repo root")
        .join("ops/manifests");
    for manifest in [
        NoChangeOutsideMask::manifest().expect("manifest"),
        Finite::manifest().expect("manifest"),
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

// --- no_change_outside_mask: analytic fixtures -----------------------------

#[test]
fn localized_edit_passes() {
    // 4x4 single channel. The mask allows the top-left 2x2; the edit changes only
    // those pixels. Nothing leaks outside, so the assertion passes.
    let before = image_value(4, 4, 1, ColorEncoding::LinearSrgb, vec![0.2_f32; 16]);
    let mut after_samples = vec![0.2_f32; 16];
    // Top-left 2x2 pixels at row-major indices 0,1,4,5.
    for i in [0, 1, 4, 5] {
        after_samples[i] = 0.9;
    }
    let after = image_value(4, 4, 1, ColorEncoding::LinearSrgb, after_samples);
    let mut mask_samples = vec![0.0_f32; 16];
    for i in [0, 1, 4, 5] {
        mask_samples[i] = 1.0;
    }
    let allowed = mask_value(4, 4, mask_samples);

    let outcome = run_no_change(&before, &after, &allowed, &serde_json::Value::Null);
    assert!(outcome.passed);
    assert_eq!(outcome.changed_pixels_outside, Some(0));
    assert_eq!(outcome.max_abs_delta_outside, Some(0.0));
    assert_eq!(outcome.worst_pixel, None);
    assert!(outcome.locations.is_empty());
}

#[test]
fn one_pixel_leak_fails_with_worst_pixel() {
    // The mask allows the top-left 2x2, but the edit also changes pixel (3, 3)
    // (row-major index 15) outside it: one leaking pixel.
    let before = image_value(4, 4, 1, ColorEncoding::LinearSrgb, vec![0.2_f32; 16]);
    let mut after_samples = vec![0.2_f32; 16];
    for i in [0, 1, 4, 5] {
        after_samples[i] = 0.9;
    }
    after_samples[15] = 0.5; // the leak: delta 0.3 at (3, 3)
    let after = image_value(4, 4, 1, ColorEncoding::LinearSrgb, after_samples);
    let mut mask_samples = vec![0.0_f32; 16];
    for i in [0, 1, 4, 5] {
        mask_samples[i] = 1.0;
    }
    let allowed = mask_value(4, 4, mask_samples);

    let outcome = run_no_change(
        &before,
        &after,
        &allowed,
        &serde_json::json!({"comparison_space": "encoded"}),
    );
    assert!(!outcome.passed);
    assert_eq!(outcome.changed_pixels_outside, Some(1));
    assert_eq!(outcome.worst_pixel, Some([3, 3]));
    assert_eq!(outcome.locations, vec![[3, 3]]);
    let max = outcome.max_abs_delta_outside.expect("max recorded");
    assert!((max - 0.3).abs() < 1e-6, "max outside delta {max}");
}

#[test]
fn worst_pixel_is_the_largest_outside_delta() {
    // Two leaks of different magnitude; the worst pixel must be the larger one.
    let before = image_value(4, 1, 1, ColorEncoding::LinearSrgb, vec![0.0_f32; 4]);
    // pixel 1 leaks by 0.2, pixel 3 leaks by 0.5.
    let after = image_value(4, 1, 1, ColorEncoding::LinearSrgb, vec![0.0, 0.2, 0.0, 0.5]);
    let allowed = mask_value(4, 1, vec![0.0_f32; 4]);

    let outcome = run_no_change(
        &before,
        &after,
        &allowed,
        &serde_json::json!({"comparison_space": "encoded"}),
    );
    assert!(!outcome.passed);
    assert_eq!(outcome.changed_pixels_outside, Some(2));
    assert_eq!(outcome.worst_pixel, Some([3, 0]));
}

#[test]
fn outside_threshold_tolerates_small_deltas() {
    // A 0.05 delta outside the mask is below a 0.1 threshold, so no leak.
    let before = image_value(2, 1, 1, ColorEncoding::LinearSrgb, vec![0.0, 0.0]);
    let after = image_value(2, 1, 1, ColorEncoding::LinearSrgb, vec![0.0, 0.05]);
    let allowed = mask_value(2, 1, vec![0.0, 0.0]);

    let outcome = run_no_change(
        &before,
        &after,
        &allowed,
        &serde_json::json!({"comparison_space": "encoded", "outside_threshold": 0.1}),
    );
    assert!(outcome.passed, "0.05 < 0.1 threshold");
    assert_eq!(outcome.changed_pixels_outside, Some(0));
    // The max delta is still measured even though it is under threshold.
    let max = outcome.max_abs_delta_outside.expect("max recorded");
    assert!((max - 0.05).abs() < 1e-6);
}

#[test]
fn coverage_epsilon_widens_the_outside_region() {
    // A pixel with coverage 0.3 is "inside" at epsilon 0 (only fully-uncovered is
    // outside) but "outside" at epsilon 0.5. The edit changes that pixel.
    let before = image_value(1, 1, 1, ColorEncoding::LinearSrgb, vec![0.0]);
    let after = image_value(1, 1, 1, ColorEncoding::LinearSrgb, vec![0.4]);
    let allowed = mask_value(1, 1, vec![0.3]);

    let inside = run_no_change(
        &before,
        &after,
        &allowed,
        &serde_json::json!({"comparison_space": "encoded"}),
    );
    assert!(
        inside.passed,
        "coverage 0.3 > epsilon 0 => inside => no leak"
    );

    let outside = run_no_change(
        &before,
        &after,
        &allowed,
        &serde_json::json!({"comparison_space": "encoded", "coverage_epsilon": 0.5}),
    );
    assert!(
        !outside.passed,
        "coverage 0.3 <= epsilon 0.5 => outside => leak"
    );
    assert_eq!(outside.changed_pixels_outside, Some(1));
}

#[test]
fn decoded_linear_space_measures_a_light_delta() {
    // An sRGB pair: a fixed encoded delta is a different light delta after decode.
    // Encoded delta 0.1 (0.4 -> 0.5) decodes to a smaller linear delta; assert the
    // outside max delta differs between the two spaces.
    let before = image_value(1, 1, 1, ColorEncoding::Srgb, vec![0.4]);
    let after = image_value(1, 1, 1, ColorEncoding::Srgb, vec![0.5]);
    let allowed = mask_value(1, 1, vec![0.0]);

    let encoded = run_no_change(
        &before,
        &after,
        &allowed,
        &serde_json::json!({"comparison_space": "encoded"}),
    );
    let linear = run_no_change(
        &before,
        &after,
        &allowed,
        &serde_json::json!({"comparison_space": "decoded-linear"}),
    );
    let enc = encoded.max_abs_delta_outside.expect("max");
    let lin = linear.max_abs_delta_outside.expect("max");
    assert!((enc - 0.1).abs() < 1e-6, "encoded delta {enc}");
    assert!(
        lin < enc,
        "decoded-linear delta {lin} should differ from encoded {enc}"
    );
}

// --- severity matrix -------------------------------------------------------

#[test]
fn severity_does_not_change_the_verdict_only_run_failing() {
    // The same leaking input under every severity yields the same passed=false
    // verdict and metrics; only `fails_run` differs.
    let before = image_value(2, 1, 1, ColorEncoding::LinearSrgb, vec![0.0, 0.0]);
    let after = image_value(2, 1, 1, ColorEncoding::LinearSrgb, vec![0.0, 0.5]);
    let allowed = mask_value(2, 1, vec![0.0, 0.0]);

    for (token, severity, fails) in [
        ("error", AssertionSeverity::Error, true),
        ("warning", AssertionSeverity::Warning, false),
        ("metric", AssertionSeverity::Metric, false),
    ] {
        let outcome = run_no_change(
            &before,
            &after,
            &allowed,
            &serde_json::json!({"comparison_space": "encoded", "severity": token}),
        );
        assert!(!outcome.passed, "{token}: verdict is still a failure");
        assert_eq!(outcome.severity, severity);
        assert_eq!(outcome.changed_pixels_outside, Some(1));
        assert_eq!(
            outcome.severity.fails_run(),
            fails,
            "{token}: fails_run mismatch"
        );
    }
}

// --- assert.finite ---------------------------------------------------------

#[test]
fn all_finite_resource_passes() {
    let img = image_value(3, 3, 1, ColorEncoding::LinearSrgb, vec![0.5_f32; 9]);
    let outcome = run_finite(&img, &serde_json::Value::Null);
    assert!(outcome.passed);
    assert_eq!(outcome.nonfinite_count, Some(0));
    assert_eq!(outcome.worst_pixel, None);
    assert!(outcome.locations.is_empty());
}

#[test]
fn nan_injection_fails_with_locations() {
    // A NaN at (1, 1) (index 4) and an Inf at (2, 2) (index 8) in a 3x3 field.
    let mut samples = vec![0.5_f32; 9];
    samples[4] = f32::NAN;
    samples[8] = f32::INFINITY;
    let img = image_value(3, 3, 1, ColorEncoding::LinearSrgb, samples);
    let outcome = run_finite(&img, &serde_json::Value::Null);
    assert!(!outcome.passed);
    assert_eq!(outcome.nonfinite_count, Some(2));
    // The worst (first, row-major) non-finite pixel is (1, 1).
    assert_eq!(outcome.worst_pixel, Some([1, 1]));
    assert_eq!(outcome.locations, vec![[1, 1], [2, 2]]);
}

#[test]
fn finite_severity_metric_records_failure_without_failing_run() {
    let mut samples = vec![0.0_f32; 4];
    samples[2] = f32::NAN;
    let img = image_value(2, 2, 1, ColorEncoding::LinearSrgb, samples);
    let outcome = run_finite(&img, &serde_json::json!({"severity": "metric"}));
    assert!(!outcome.passed);
    assert_eq!(outcome.severity, AssertionSeverity::Metric);
    assert!(!outcome.severity.fails_run(), "metric never fails the run");
    assert_eq!(outcome.nonfinite_count, Some(1));
}

// --- shape / error handling ------------------------------------------------

#[test]
fn mismatched_extent_is_a_shape_error() {
    let before = image_value(2, 2, 1, ColorEncoding::LinearSrgb, vec![0.0_f32; 4]);
    let after = image_value(3, 3, 1, ColorEncoding::LinearSrgb, vec![0.0_f32; 9]);
    let allowed = mask_value(2, 2, vec![0.0_f32; 4]);
    let mut inputs = InputValues::new();
    inputs.insert("before".to_owned(), before);
    inputs.insert("after".to_owned(), after);
    inputs.insert("allowed".to_owned(), allowed);
    let err = NoChangeOutsideMask::new()
        .compute(&inputs, &serde_json::Value::Null)
        .expect_err("extent mismatch rejected");
    assert_eq!(err.code, super::E_ASSERT_SHAPE);
}

#[test]
fn bad_comparison_space_is_a_param_error() {
    let before = image_value(1, 1, 1, ColorEncoding::LinearSrgb, vec![0.0]);
    let after = image_value(1, 1, 1, ColorEncoding::LinearSrgb, vec![0.0]);
    let allowed = mask_value(1, 1, vec![0.0]);
    let mut inputs = InputValues::new();
    inputs.insert("before".to_owned(), before);
    inputs.insert("after".to_owned(), after);
    inputs.insert("allowed".to_owned(), allowed);
    let err = NoChangeOutsideMask::new()
        .compute(&inputs, &serde_json::json!({"comparison_space": "bogus"}))
        .expect_err("bad comparison space rejected");
    assert_eq!(err.code, super::E_ASSERT_PARAM);
}

#[test]
fn does_not_mutate_inputs() {
    let before = image_value(2, 2, 1, ColorEncoding::LinearSrgb, vec![0.1, 0.2, 0.3, 0.4]);
    let after = image_value(2, 2, 1, ColorEncoding::LinearSrgb, vec![0.1, 0.9, 0.3, 0.4]);
    let allowed = mask_value(2, 2, vec![0.0; 4]);
    let before_copy = before.clone();
    let after_copy = after.clone();
    let _ = run_no_change(&before, &after, &allowed, &serde_json::Value::Null);
    assert_eq!(before, before_copy);
    assert_eq!(after, after_copy);
}
