//! Verification suite for `analyze.changed_bounds@1`, `assert.range@1`,
//! `assert.alpha_valid@1`, and `assert.changed_bounds@1` (`OP_CATALOG` §12,
//! `IR_SPEC` §13, `AGENT_VERIFICATION` §5.3):
//!
//! - **schema/contract**: each manifest validates, agrees with its contract,
//!   gates clean, and the checked-in manifest matches the Rust builder;
//! - **analytic fixtures**: `changed_bounds` is exact on a known injected delta
//!   and empty on an identity diff; `assert.range` fails with the worst value +
//!   count on an out-of-range fixture; `assert.alpha_valid` fails on `|C| > a`
//!   and on out-of-`[0,1]` alpha; `assert.changed_bounds` passes when change is
//!   contained and fails (with the escaping pixel) when it escapes;
//! - **severity**: a `metric`-severity failing assertion records its verdict
//!   without its `severity` ever being `Error`.

use paintop_core::executor::{InputValues, OpImplementation, OutputValues, ResourceValue};
use paintop_ir::{
    AlphaRepresentation, AssertionOutcome, AssertionSeverity, ChannelLayout, ColorEncoding,
    ColorRange, CoordinateConvention, Extent, ImageDescriptor, Rect, Report, ResourceDescriptor,
    ScalarType, SemanticRole, check_contract_consistency, verify_categories,
};

use super::{
    ALPHA_VALID_OP_ID, ASSERT_CHANGED_BOUNDS_OP_ID, AssertAlphaValid, AssertChangedBounds,
    AssertRange, CHANGED_BOUNDS_OP_ID, ChangedBounds, RANGE_OP_ID,
};

/// Build an image value with explicit color/alpha representation.
fn image_value(
    width: u32,
    height: u32,
    channels: u32,
    color: ColorEncoding,
    alpha: AlphaRepresentation,
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
        color,
        range: ColorRange::SceneReferred,
        alpha,
        coordinates: CoordinateConvention::PixelCenterUpperLeft,
        semantic: SemanticRole::Color,
    });
    ResourceValue::new(descriptor, channels, samples).expect("sample buffer matches descriptor")
}

/// A plain linear single/multi-channel image with straight alpha.
fn plain(width: u32, height: u32, channels: u32, samples: Vec<f32>) -> ResourceValue {
    image_value(
        width,
        height,
        channels,
        ColorEncoding::LinearSrgb,
        AlphaRepresentation::Straight,
        samples,
    )
}

/// Recover the assertion outcome from an op's output.
fn outcome(out: &OutputValues) -> AssertionOutcome {
    let report: &Report = out
        .get("report")
        .expect("report port")
        .as_report()
        .expect("report value");
    report.assertion.clone().expect("assertion verdict")
}

// --- schema / contract -----------------------------------------------------

#[test]
fn manifests_validate_and_agree_with_contracts() {
    let manifests = [
        (ChangedBounds::manifest().unwrap(), CHANGED_BOUNDS_OP_ID),
        (AssertRange::manifest().unwrap(), RANGE_OP_ID),
        (AssertAlphaValid::manifest().unwrap(), ALPHA_VALID_OP_ID),
        (
            AssertChangedBounds::manifest().unwrap(),
            ASSERT_CHANGED_BOUNDS_OP_ID,
        ),
    ];
    for (manifest, id) in manifests {
        manifest.validate().expect("manifest valid");
        verify_categories(&manifest, &manifest.test.verification)
            .expect("verification gates clean");
        assert_eq!(manifest.id.to_string(), id);
    }
    // Contract agreement (per concrete op type).
    check_contract_consistency(&ChangedBounds::manifest().unwrap(), &ChangedBounds::new()).unwrap();
    check_contract_consistency(&AssertRange::manifest().unwrap(), &AssertRange::new()).unwrap();
    check_contract_consistency(
        &AssertAlphaValid::manifest().unwrap(),
        &AssertAlphaValid::new(),
    )
    .unwrap();
    check_contract_consistency(
        &AssertChangedBounds::manifest().unwrap(),
        &AssertChangedBounds::new(),
    )
    .unwrap();
}

#[test]
fn checked_in_manifests_match_builders() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("crates dir")
        .parent()
        .expect("repo root")
        .join("ops/manifests");
    for manifest in [
        ChangedBounds::manifest().unwrap(),
        AssertRange::manifest().unwrap(),
        AssertAlphaValid::manifest().unwrap(),
        AssertChangedBounds::manifest().unwrap(),
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

// --- analyze.changed_bounds ------------------------------------------------

fn run_changed_bounds(diff: &ResourceValue, params: &serde_json::Value) -> AssertionOutcome {
    let mut inputs = InputValues::new();
    inputs.insert("diff".to_owned(), diff.clone());
    let out = ChangedBounds::new()
        .compute(&inputs, params)
        .expect("changed_bounds computes");
    outcome(&out)
}

#[test]
fn changed_bounds_empty_on_identity_diff() {
    // An all-zero diff field changed nothing: empty bounds, zero count.
    let diff = plain(4, 4, 1, vec![0.0; 16]);
    let v = run_changed_bounds(&diff, &serde_json::Value::Null);
    assert_eq!(v.changed_bounds, None);
    assert_eq!(v.violations, Some(0));
}

#[test]
fn changed_bounds_exact_on_injected_delta() {
    // A single changed pixel at (2, 1): tight bounds [2,3) x [1,2), count 1.
    let mut samples = vec![0.0_f32; 16];
    samples[4 + 2] = 0.9;
    let diff = plain(4, 4, 1, samples);
    let v = run_changed_bounds(&diff, &serde_json::Value::Null);
    assert_eq!(v.changed_bounds, Some(Rect::new(2, 1, 3, 2)));
    assert_eq!(v.violations, Some(1));
}

#[test]
fn changed_bounds_threshold_gates_the_region() {
    // Magnitudes 0.3 and 0.7; a threshold of 0.5 keeps only the 0.7 pixel.
    let diff = plain(2, 1, 1, vec![0.3, 0.7]);
    let v = run_changed_bounds(&diff, &serde_json::json!({"threshold": 0.5}));
    assert_eq!(v.changed_bounds, Some(Rect::new(1, 0, 2, 1)));
    assert_eq!(v.violations, Some(1));
}

// --- assert.range ----------------------------------------------------------

fn run_range(resource: &ResourceValue, params: &serde_json::Value) -> AssertionOutcome {
    let mut inputs = InputValues::new();
    inputs.insert("resource".to_owned(), resource.clone());
    let out = AssertRange::new()
        .compute(&inputs, params)
        .expect("range computes");
    outcome(&out)
}

#[test]
fn range_passes_when_all_in_bounds() {
    let img = plain(2, 2, 1, vec![0.0, 0.5, 1.0, 0.25]);
    let v = run_range(&img, &serde_json::json!({"min": 0.0, "max": 1.0}));
    assert!(v.passed);
    assert_eq!(v.violations, Some(0));
    assert_eq!(v.worst_value, None);
}

#[test]
fn range_fails_with_worst_value_and_count() {
    // Two out-of-range samples: 1.5 (excess 0.5) and -0.3 (excess 0.3). The
    // worst is the furthest-out value 1.5.
    let img = plain(2, 2, 1, vec![0.5, 1.5, -0.3, 0.25]);
    let v = run_range(&img, &serde_json::json!({"min": 0.0, "max": 1.0}));
    assert!(!v.passed);
    assert_eq!(v.violations, Some(2));
    assert_eq!(v.worst_value, Some(1.5));
    // 1.5 is at pixel index 1 -> (1, 0).
    assert_eq!(v.worst_pixel, Some([1, 0]));
}

#[test]
fn range_treats_nonfinite_as_out_of_range() {
    let img = plain(1, 1, 1, vec![f32::NAN]);
    let v = run_range(&img, &serde_json::json!({"min": 0.0, "max": 1.0}));
    assert!(!v.passed);
    assert_eq!(v.violations, Some(1));
}

#[test]
fn range_metric_severity_never_fails_but_records() {
    let img = plain(1, 1, 1, vec![2.0]);
    let v = run_range(
        &img,
        &serde_json::json!({"min": 0.0, "max": 1.0, "severity": "metric"}),
    );
    assert!(!v.passed);
    assert_eq!(v.severity, AssertionSeverity::Metric);
    assert!(!v.severity.fails_run());
}

// --- assert.alpha_valid ----------------------------------------------------

fn run_alpha(image: &ResourceValue, params: &serde_json::Value) -> AssertionOutcome {
    let mut inputs = InputValues::new();
    inputs.insert("image".to_owned(), image.clone());
    let out = AssertAlphaValid::new()
        .compute(&inputs, params)
        .expect("alpha_valid computes");
    outcome(&out)
}

#[test]
fn alpha_valid_passes_on_valid_premultiplied_image() {
    // RGBA premultiplied: every |C| <= alpha and alpha in [0,1].
    let img = image_value(
        1,
        1,
        4,
        ColorEncoding::LinearSrgb,
        AlphaRepresentation::Premultiplied,
        vec![0.3, 0.4, 0.5, 0.5],
    );
    let v = run_alpha(&img, &serde_json::Value::Null);
    assert!(v.passed);
    assert_eq!(v.violations, Some(0));
}

#[test]
fn alpha_valid_fails_when_color_exceeds_alpha() {
    // Premultiplied with a color channel 0.8 > alpha 0.5: excess 0.3.
    let img = image_value(
        1,
        1,
        4,
        ColorEncoding::LinearSrgb,
        AlphaRepresentation::Premultiplied,
        vec![0.8, 0.2, 0.2, 0.5],
    );
    let v = run_alpha(&img, &serde_json::Value::Null);
    assert!(!v.passed);
    assert_eq!(v.violations, Some(1));
    assert!((v.worst_value.expect("worst") - 0.3).abs() < 1e-6);
}

#[test]
fn alpha_valid_fails_on_out_of_unit_alpha() {
    // Straight alpha 1.5 is out of [0, 1] (excess 0.5). Color constraint not
    // applied for a straight-alpha image.
    let img = image_value(
        1,
        1,
        4,
        ColorEncoding::LinearSrgb,
        AlphaRepresentation::Straight,
        vec![0.9, 0.9, 0.9, 1.5],
    );
    let v = run_alpha(&img, &serde_json::Value::Null);
    assert!(!v.passed);
    assert_eq!(v.violations, Some(1));
    assert!((v.worst_value.expect("worst") - 0.5).abs() < 1e-6);
}

#[test]
fn alpha_valid_straight_alpha_ignores_color_constraint() {
    // Straight alpha: a color channel above alpha is fine (only the premult
    // constraint is alpha-gated).
    let img = image_value(
        1,
        1,
        4,
        ColorEncoding::LinearSrgb,
        AlphaRepresentation::Straight,
        vec![0.9, 0.0, 0.0, 0.2],
    );
    let v = run_alpha(&img, &serde_json::Value::Null);
    assert!(v.passed);
}

// --- assert.changed_bounds -------------------------------------------------

fn run_assert_bounds(
    before: &ResourceValue,
    after: &ResourceValue,
    params: &serde_json::Value,
) -> AssertionOutcome {
    let mut inputs = InputValues::new();
    inputs.insert("before".to_owned(), before.clone());
    inputs.insert("after".to_owned(), after.clone());
    let out = AssertChangedBounds::new()
        .compute(&inputs, params)
        .expect("assert.changed_bounds computes");
    outcome(&out)
}

#[test]
fn assert_changed_bounds_passes_when_change_is_contained() {
    // Change only pixel (1, 1) in a 4x4; expected box [1,1)..[3,3) contains it.
    let before = plain(4, 4, 1, vec![0.2_f32; 16]);
    let mut after_samples = vec![0.2_f32; 16];
    after_samples[4 + 1] = 0.9; // pixel (1, 1)
    let after = plain(4, 4, 1, after_samples);
    let v = run_assert_bounds(
        &before,
        &after,
        &serde_json::json!({"x0": 1, "y0": 1, "x1": 3, "y1": 3}),
    );
    assert!(v.passed);
    assert_eq!(v.violations, Some(0));
    assert_eq!(v.changed_bounds, Some(Rect::new(1, 1, 2, 2)));
    assert_eq!(v.expected_bounds, Some(Rect::new(1, 1, 3, 3)));
}

#[test]
fn assert_changed_bounds_fails_when_change_escapes_box() {
    // Change pixel (0, 0), which lies outside the expected box [1,1)..[3,3).
    let before = plain(4, 4, 1, vec![0.2_f32; 16]);
    let mut after_samples = vec![0.2_f32; 16];
    after_samples[0] = 0.9; // pixel (0, 0)
    let after = plain(4, 4, 1, after_samples);
    let v = run_assert_bounds(
        &before,
        &after,
        &serde_json::json!({"x0": 1, "y0": 1, "x1": 3, "y1": 3}),
    );
    assert!(!v.passed);
    assert_eq!(v.violations, Some(1));
    assert_eq!(v.worst_pixel, Some([0, 0]));
    assert_eq!(v.changed_bounds, Some(Rect::new(0, 0, 1, 1)));
}

#[test]
fn assert_changed_bounds_identity_passes_with_empty_bounds() {
    let before = plain(4, 4, 1, vec![0.2_f32; 16]);
    let v = run_assert_bounds(
        &before,
        &before,
        &serde_json::json!({"x0": 0, "y0": 0, "x1": 1, "y1": 1}),
    );
    assert!(v.passed);
    assert_eq!(v.changed_bounds, None);
    assert_eq!(v.violations, Some(0));
}

#[test]
fn assert_changed_bounds_metric_severity_never_fails_run() {
    let before = plain(2, 1, 1, vec![0.0, 0.0]);
    let after = plain(2, 1, 1, vec![0.0, 0.9]); // pixel (1, 0) changes
    let v = run_assert_bounds(
        &before,
        &after,
        &serde_json::json!({"x0": 0, "y0": 0, "x1": 1, "y1": 1, "severity": "metric"}),
    );
    assert!(!v.passed);
    assert_eq!(v.severity, AssertionSeverity::Metric);
    assert!(!v.severity.fails_run());
}

#[test]
fn assert_changed_bounds_rejects_ill_formed_box() {
    let before = plain(2, 2, 1, vec![0.0; 4]);
    let mut inputs = InputValues::new();
    inputs.insert("before".to_owned(), before.clone());
    inputs.insert("after".to_owned(), before);
    // x1 < x0 is ill-formed.
    assert!(
        AssertChangedBounds::new()
            .compute(
                &inputs,
                &serde_json::json!({"x0": 3, "y0": 0, "x1": 1, "y1": 1})
            )
            .is_err()
    );
}

#[test]
fn range_rejects_inverted_bounds() {
    let img = plain(1, 1, 1, vec![0.5]);
    let mut inputs = InputValues::new();
    inputs.insert("resource".to_owned(), img);
    assert!(
        AssertRange::new()
            .compute(&inputs, &serde_json::json!({"min": 1.0, "max": 0.0}))
            .is_err()
    );
}

#[test]
fn alpha_valid_rejects_image_without_alpha() {
    let img = plain(1, 1, 3, vec![0.1, 0.2, 0.3]);
    let mut inputs = InputValues::new();
    inputs.insert("image".to_owned(), img);
    assert!(
        AssertAlphaValid::new()
            .compute(&inputs, &serde_json::Value::Null)
            .is_err()
    );
}
