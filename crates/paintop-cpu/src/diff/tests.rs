//! Verification suite for `analyze.diff@1` (`OP_CATALOG` §12,
//! `AGENT_VERIFICATION` §2.6 differential metrics, §2.2 analytic fixtures, §2.4
//! property tests):
//!
//! - **schema/contract**: the manifest validates, agrees with its contract, and
//!   its verification declarations gate clean;
//! - **analytic fixtures**: identical inputs produce a zero diff, zero metrics,
//!   and empty changed bounds; a known injected constant delta produces exact
//!   max / mean / RMS error and a changed-pixel count and bounds matching the
//!   injected rectangle; the comparison space (encoded vs decoded-linear) is
//!   honored;
//! - **property/metamorphic**: the diff field is the absolute difference
//!   (symmetric in before/after), every diff sample stays finite for finite
//!   inputs, and the threshold strictly gates the changed count.

use paintop_core::executor::{InputValues, OpImplementation, OutputValues, ResourceValue};
use paintop_ir::{
    AlphaRepresentation, ChannelLayout, ColorEncoding, ColorRange, CoordinateConvention,
    DiffMetrics, Extent, ImageDescriptor, Rect, Report, ResourceDescriptor, ScalarType,
    SemanticRole, check_contract_consistency, verify_categories,
};

use super::{DIFF_OP_ID, Diff};

/// Build an image [`ResourceValue`] of `channels` from an explicit sample buffer
/// in the given color `encoding` (row-major, channel-interleaved).
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

/// Run the op's compute kernel and recover the diff field and the report.
fn run_diff(
    before: &ResourceValue,
    after: &ResourceValue,
    params: &serde_json::Value,
) -> (ResourceValue, Report) {
    let mut inputs = InputValues::new();
    inputs.insert("before".to_owned(), before.clone());
    inputs.insert("after".to_owned(), after.clone());
    let out: OutputValues = Diff::new()
        .compute(&inputs, params)
        .expect("analyze.diff computes");
    let diff = out.get("diff").expect("diff port produced").clone();
    let report = out
        .get("report")
        .expect("report port produced")
        .as_report()
        .expect("report payload present")
        .clone();
    (diff, report)
}

#[test]
fn manifest_validates_and_agrees_with_contract() {
    let manifest = Diff::manifest().expect("diff manifest");
    manifest.validate().expect("diff manifest valid");
    check_contract_consistency(&manifest, &Diff::new())
        .expect("diff manifest agrees with contract");
    verify_categories(&manifest, &manifest.test.verification)
        .expect("diff verification declarations gate clean");
    assert_eq!(manifest.id.to_string(), DIFF_OP_ID);
}

#[test]
fn identical_inputs_yield_zero_diff_and_empty_bounds() {
    let img = image_value(4, 4, 1, ColorEncoding::LinearSrgb, vec![0.25_f32; 16]);
    let (diff, report) = run_diff(&img, &img, &serde_json::Value::Null);

    // The diff field is all zero.
    assert!(diff.samples().iter().all(|&s| s == 0.0));
    // The diff output is retyped to raw-linear material.
    let ResourceDescriptor::Image(d) = diff.descriptor() else {
        panic!("diff is an image");
    };
    assert_eq!(d.color, ColorEncoding::RawLinear);
    assert_eq!(d.semantic, SemanticRole::Material);

    let metrics = report.diff.expect("report carries diff metrics");
    assert_eq!(metrics.max_abs_error.to_bits(), 0.0_f64.to_bits());
    assert_eq!(metrics.mean_abs_error.to_bits(), 0.0_f64.to_bits());
    assert_eq!(metrics.rms_error.to_bits(), 0.0_f64.to_bits());
    assert_eq!(metrics.changed_count, 0);
    assert_eq!(metrics.changed_bounds, None);
    assert!(report.all_finite);
}

#[test]
fn known_constant_delta_has_exact_metrics() {
    // 4x4 single channel; `after` = `before` + 0.5 everywhere. Every channel
    // diff is exactly 0.5: max = mean = rms = 0.5, all 16 pixels changed,
    // bounds cover the whole image.
    let before = image_value(4, 4, 1, ColorEncoding::LinearSrgb, vec![0.1_f32; 16]);
    let after = image_value(4, 4, 1, ColorEncoding::LinearSrgb, vec![0.6_f32; 16]);
    let (diff, report) = run_diff(
        &before,
        &after,
        &serde_json::json!({"comparison_space": "encoded"}),
    );

    assert!(diff.samples().iter().all(|&s| (s - 0.5).abs() < 1e-6));
    let metrics = report.diff.expect("metrics");
    assert!((metrics.max_abs_error - 0.5).abs() < 1e-6);
    assert!((metrics.mean_abs_error - 0.5).abs() < 1e-6);
    assert!((metrics.rms_error - 0.5).abs() < 1e-6);
    assert_eq!(metrics.changed_count, 16);
    assert_eq!(metrics.changed_bounds, Some(Rect::new(0, 0, 4, 4)));
}

#[test]
fn injected_rectangle_has_exact_changed_bounds() {
    // 4x4 single channel; inject a delta only at the single pixel (2, 1).
    let mut after_samples = vec![0.2_f32; 16];
    let idx = 4 + 2;
    after_samples[idx] = 0.9;
    let before = image_value(4, 4, 1, ColorEncoding::LinearSrgb, vec![0.2_f32; 16]);
    let after = image_value(4, 4, 1, ColorEncoding::LinearSrgb, after_samples);
    let (_, report) = run_diff(
        &before,
        &after,
        &serde_json::json!({"comparison_space": "encoded"}),
    );

    let metrics = report.diff.expect("metrics");
    assert!((metrics.max_abs_error - 0.7).abs() < 1e-6);
    assert_eq!(metrics.changed_count, 1);
    // Tight bounding box of the single changed pixel (2, 1): [2,3) x [1,2).
    assert_eq!(metrics.changed_bounds, Some(Rect::new(2, 1, 3, 2)));
    // Mean is the single 0.7 over 16 samples.
    assert!((metrics.mean_abs_error - 0.7 / 16.0).abs() < 1e-6);
}

#[test]
fn comparison_space_changes_metrics_for_srgb() {
    // A pair of sRGB-encoded constants. In `encoded` space the diff is the
    // stored-sample difference; in `decoded-linear` space it is the difference
    // of the sRGB-decoded linear values, which differ.
    let before = image_value(2, 2, 1, ColorEncoding::Srgb, vec![0.2_f32; 4]);
    let after = image_value(2, 2, 1, ColorEncoding::Srgb, vec![0.8_f32; 4]);

    let (_, encoded) = run_diff(
        &before,
        &after,
        &serde_json::json!({"comparison_space": "encoded"}),
    );
    let (_, linear) = run_diff(
        &before,
        &after,
        &serde_json::json!({"comparison_space": "decoded-linear"}),
    );

    let enc = encoded.diff.expect("encoded metrics").max_abs_error;
    let lin = linear.diff.expect("linear metrics").max_abs_error;
    assert!((enc - 0.6).abs() < 1e-6, "encoded diff is the stored delta");
    // Linear delta = srgb_decode(0.8) - srgb_decode(0.2); the sRGB curve
    // compresses this below the encoded 0.6 delta, so the two spaces disagree.
    let expected_lin = f64::from(super::srgb_decode(0.8)) - f64::from(super::srgb_decode(0.2));
    assert!(
        (lin - expected_lin).abs() < 1e-5,
        "linear delta {lin} vs {expected_lin}"
    );
    assert!(
        (lin - enc).abs() > 1e-3,
        "the comparison space changes the measured delta ({lin} vs {enc})",
    );
}

#[test]
fn default_comparison_space_is_decoded_linear() {
    // With no params, sRGB inputs are compared in decoded-linear (the default),
    // matching an explicit decoded-linear request.
    let before = image_value(2, 2, 1, ColorEncoding::Srgb, vec![0.2_f32; 4]);
    let after = image_value(2, 2, 1, ColorEncoding::Srgb, vec![0.8_f32; 4]);
    let (_, default) = run_diff(&before, &after, &serde_json::Value::Null);
    let (_, explicit) = run_diff(
        &before,
        &after,
        &serde_json::json!({"comparison_space": "decoded-linear"}),
    );
    assert_eq!(
        default.diff.expect("default").max_abs_error.to_bits(),
        explicit.diff.expect("explicit").max_abs_error.to_bits(),
    );
}

#[test]
fn threshold_gates_the_changed_count() {
    // Two pixels change by 0.3, one by 0.05. A threshold of 0.1 counts only the
    // two larger changes.
    let before = image_value(3, 1, 1, ColorEncoding::LinearSrgb, vec![0.0, 0.0, 0.0]);
    let after = image_value(3, 1, 1, ColorEncoding::LinearSrgb, vec![0.3, 0.05, 0.3]);
    let (_, report) = run_diff(
        &before,
        &after,
        &serde_json::json!({"comparison_space": "encoded", "threshold": 0.1}),
    );
    let metrics = report.diff.expect("metrics");
    assert!((metrics.threshold - 0.1).abs() < 1e-12);
    assert_eq!(metrics.changed_count, 2);
    // The changed pixels are at x = 0 and x = 2: bounds span [0,3) x [0,1).
    assert_eq!(metrics.changed_bounds, Some(Rect::new(0, 0, 3, 1)));
}

#[test]
fn diff_is_symmetric_in_before_and_after() {
    // |after − before| == |before − after|: swapping the inputs leaves the diff
    // field and the metrics unchanged.
    let a = image_value(
        3,
        2,
        4,
        ColorEncoding::LinearSrgb,
        (0..24_i16).map(|i| f32::from(i) / 24.0).collect(),
    );
    let b = image_value(
        3,
        2,
        4,
        ColorEncoding::LinearSrgb,
        (0..24_i16).map(|i| f32::from(23 - i) / 24.0).collect(),
    );
    let params = serde_json::json!({"comparison_space": "encoded"});
    let (diff_ab, report_ab) = run_diff(&a, &b, &params);
    let (diff_ba, report_ba) = run_diff(&b, &a, &params);
    assert_eq!(diff_ab.samples(), diff_ba.samples());
    assert_eq!(report_ab.diff, report_ba.diff);
}

#[test]
fn rms_dominates_mean_for_uneven_errors() {
    // For a non-constant error field, RMS >= mean (with equality only for a
    // constant field). Inject 0.0 and 1.0 errors.
    let before = image_value(2, 1, 1, ColorEncoding::LinearSrgb, vec![0.0, 0.0]);
    let after = image_value(2, 1, 1, ColorEncoding::LinearSrgb, vec![0.0, 1.0]);
    let (_, report) = run_diff(
        &before,
        &after,
        &serde_json::json!({"comparison_space": "encoded"}),
    );
    let m: DiffMetrics = report.diff.expect("metrics");
    // mean = 0.5, rms = sqrt((0 + 1)/2) = sqrt(0.5) ≈ 0.707.
    assert!((m.mean_abs_error - 0.5).abs() < 1e-9);
    assert!((m.rms_error - 0.5_f64.sqrt()).abs() < 1e-9);
    assert!(m.rms_error > m.mean_abs_error);
}

#[test]
fn nonfinite_input_is_flagged_and_excluded() {
    // A NaN in `after` yields a NaN diff: all_finite is false and the metric
    // reductions exclude the non-finite sample (mean over the one finite 0.5).
    let before = image_value(2, 1, 1, ColorEncoding::LinearSrgb, vec![0.0, 0.0]);
    let after = image_value(2, 1, 1, ColorEncoding::LinearSrgb, vec![0.5, f32::NAN]);
    let (_, report) = run_diff(
        &before,
        &after,
        &serde_json::json!({"comparison_space": "encoded"}),
    );
    assert!(!report.all_finite);
    let m = report.diff.expect("metrics");
    assert!((m.max_abs_error - 0.5).abs() < 1e-6);
    assert!((m.mean_abs_error - 0.5).abs() < 1e-6);
}

#[test]
fn unknown_comparison_space_is_rejected() {
    let before = image_value(1, 1, 1, ColorEncoding::LinearSrgb, vec![0.0]);
    let after = image_value(1, 1, 1, ColorEncoding::LinearSrgb, vec![0.0]);
    let mut inputs = InputValues::new();
    inputs.insert("before".to_owned(), before);
    inputs.insert("after".to_owned(), after);
    let err = Diff::new()
        .compute(
            &inputs,
            &serde_json::json!({"comparison_space": "perceptual"}),
        )
        .expect_err("an unknown comparison space is rejected");
    assert_eq!(err.code, super::E_DIFF_PARAM);
}

#[test]
fn mismatched_extent_is_rejected() {
    let before = image_value(2, 2, 1, ColorEncoding::LinearSrgb, vec![0.0; 4]);
    let after = image_value(3, 2, 1, ColorEncoding::LinearSrgb, vec![0.0; 6]);
    let mut inputs = InputValues::new();
    inputs.insert("before".to_owned(), before);
    inputs.insert("after".to_owned(), after);
    let err = Diff::new()
        .compute(&inputs, &serde_json::Value::Null)
        .expect_err("mismatched extents are rejected");
    assert_eq!(err.code, super::E_DIFF_SHAPE);
}

#[test]
fn diff_is_pure() {
    let before = image_value(2, 2, 1, ColorEncoding::LinearSrgb, vec![0.3; 4]);
    let after = image_value(2, 2, 1, ColorEncoding::LinearSrgb, vec![0.7; 4]);
    let before_copy = before.clone();
    let after_copy = after.clone();
    let _ = run_diff(&before, &after, &serde_json::Value::Null);
    assert_eq!(before, before_copy);
    assert_eq!(after, after_copy);
}

/// The checked-in `ops/manifests/<id>.json` file (read by `cargo xtask
/// verify-op`) must stay byte-identical to the Rust manifest builder, the source
/// of truth. Regenerate with `serde_json::to_string_pretty` if this fails.
#[test]
fn checked_in_manifest_matches_builder() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("crates dir")
        .parent()
        .expect("repo root")
        .join("ops/manifests");
    let manifest = Diff::manifest().expect("diff manifest");
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
