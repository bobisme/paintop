//! Tests for `assert.frequency_preserved@1`: passes when band energy is
//! preserved outside the edit, FAILS (with a band-energy delta) when a region is
//! over-blurred, and honours the mask window.

use super::{FREQUENCY_PRESERVED_OP_ID, FrequencyPreserved};
use crate::frequency::gaussian_blur_plane;
use paintop_core::executor::{InputValues, OpImplementation, ResourceValue};
use paintop_ir::{
    AlphaRepresentation, ChannelLayout, ColorEncoding, ColorRange, CoordinateConvention, Extent,
    ImageDescriptor, MaskDescriptor, MaskMeaning, OperationManifest, Report, ResourceDescriptor,
    ScalarType, SemanticRole, ValidRange,
};

fn gray_image(extent: Extent, samples: Vec<f32>) -> ResourceValue {
    let descriptor = ResourceDescriptor::Image(ImageDescriptor {
        extent,
        layout: ChannelLayout::Gray,
        scalar: ScalarType::F32,
        color: ColorEncoding::LinearSrgb,
        range: ColorRange::SceneReferred,
        alpha: AlphaRepresentation::Premultiplied,
        coordinates: CoordinateConvention::PixelCenterUpperLeft,
        semantic: SemanticRole::Color,
    });
    ResourceValue::new(descriptor, 1, samples).unwrap()
}

fn mask(extent: Extent, samples: Vec<f32>) -> ResourceValue {
    let descriptor = ResourceDescriptor::Mask(MaskDescriptor {
        extent,
        scalar: ScalarType::F32,
        range: ValidRange::Bounded { min: 0.0, max: 1.0 },
        meaning: MaskMeaning::Coverage,
        coordinates: CoordinateConvention::PixelCenterUpperLeft,
    });
    ResourceValue::new(descriptor, 1, samples).unwrap()
}

/// A deterministic high-frequency texture in `[0, 1)`.
fn texture(extent: Extent) -> Vec<f32> {
    let w = extent.width;
    (0..extent.width * extent.height)
        .map(|i| {
            let x = i % w;
            let y = i / w;
            if (x + y).is_multiple_of(2) { 0.9 } else { 0.1 }
        })
        .collect()
}

fn run(
    before: ResourceValue,
    after: ResourceValue,
    mask: Option<ResourceValue>,
    params: &serde_json::Value,
) -> Report {
    let mut inputs = InputValues::new();
    inputs.insert("before".to_owned(), before);
    inputs.insert("after".to_owned(), after);
    if let Some(m) = mask {
        inputs.insert("mask".to_owned(), m);
    }
    let out = FrequencyPreserved::new().compute(&inputs, params).unwrap();
    out.get("report").unwrap().as_report().unwrap().clone()
}

#[test]
fn manifest_validates_and_matches_id() {
    let m: OperationManifest = FrequencyPreserved::manifest().unwrap();
    m.validate().unwrap();
    assert_eq!(m.id.to_string(), FREQUENCY_PRESERVED_OP_ID);
}

#[test]
fn identical_images_pass_with_zero_delta() {
    let base = texture(Extent::new(8, 8));
    let report = run(
        gray_image(Extent::new(8, 8), base.clone()),
        gray_image(Extent::new(8, 8), base),
        None,
        &serde_json::json!({"bands": 3}),
    );
    let outcome = report.assertion.unwrap();
    assert!(outcome.passed, "identical images must preserve every band");
    assert!(outcome.worst_value.unwrap() < 1e-9);
}

#[test]
fn over_blurring_the_whole_image_fails_with_a_band_delta() {
    // The "after" image is a heavy blur of "before": high-frequency band energy
    // collapses, so the assertion fails and records a band-energy delta.
    let base = texture(Extent::new(16, 16));
    let blurred = gaussian_blur_plane(&base, Extent::new(16, 16), 1, 3.0);
    let report = run(
        gray_image(Extent::new(16, 16), base),
        gray_image(Extent::new(16, 16), blurred),
        None,
        &serde_json::json!({"bands": 3, "tolerance": 0.05}),
    );
    let outcome = report.assertion.unwrap();
    assert!(
        !outcome.passed,
        "an over-blurred image must fail preservation"
    );
    assert!(
        outcome.worst_value.unwrap() > 0.05,
        "the worst band delta {} must exceed the tolerance",
        outcome.worst_value.unwrap()
    );
    assert!(outcome.violations.unwrap() >= 1);
    // The worst band is recorded as a [band, 0] locator.
    assert!(outcome.worst_pixel.is_some());
}

#[test]
fn edit_inside_mask_does_not_fail_when_outside_is_preserved() {
    // The edit only touches the masked (left) region; the unmasked region is
    // identical. Checking the complement => preserved => pass, even though the
    // images differ overall.
    let extent = Extent::new(16, 16);
    let base = texture(extent);
    let mut after = base.clone();
    // Over-blur ONLY the left half by overwriting it with a constant.
    for (i, sample) in after.iter_mut().enumerate() {
        if (i % 16) < 8 {
            *sample = 0.5;
        }
    }
    // Mask marks the EDITED (left) region.
    let mask_samples: Vec<f32> = (0..16 * 16)
        .map(|i| if (i % 16) < 8 { 1.0 } else { 0.0 })
        .collect();
    let report = run(
        gray_image(extent, base),
        gray_image(extent, after),
        Some(mask(extent, mask_samples)),
        &serde_json::json!({"bands": 3, "tolerance": 0.05}),
    );
    let outcome = report.assertion.unwrap();
    assert!(
        outcome.passed,
        "an edit confined to the mask must preserve the complement; worst {}",
        outcome.worst_value.unwrap()
    );
}

#[test]
fn edit_leaking_outside_the_mask_fails() {
    // The edit blurs the whole image but the mask only covers the left half, so
    // the leak into the unmasked region trips the assertion.
    let extent = Extent::new(16, 16);
    let base = texture(extent);
    let after = gaussian_blur_plane(&base, extent, 1, 3.0);
    let mask_samples: Vec<f32> = (0..16 * 16)
        .map(|i| if (i % 16) < 8 { 1.0 } else { 0.0 })
        .collect();
    let report = run(
        gray_image(extent, base),
        gray_image(extent, after),
        Some(mask(extent, mask_samples)),
        &serde_json::json!({"bands": 3, "tolerance": 0.05}),
    );
    let outcome = report.assertion.unwrap();
    assert!(!outcome.passed, "a leak outside the mask must fail");
}

#[test]
fn severity_metric_records_but_never_fails_the_run() {
    let base = texture(Extent::new(16, 16));
    let after = gaussian_blur_plane(&base, Extent::new(16, 16), 1, 3.0);
    let report = run(
        gray_image(Extent::new(16, 16), base),
        gray_image(Extent::new(16, 16), after),
        None,
        &serde_json::json!({"bands": 3, "tolerance": 0.05, "severity": "metric"}),
    );
    let outcome = report.assertion.unwrap();
    assert!(!outcome.passed);
    assert!(
        !outcome.severity.fails_run(),
        "metric severity never fails the run"
    );
}

#[test]
fn rejects_mismatched_extents_and_missing_bands() {
    let a = gray_image(Extent::new(8, 8), vec![0.0; 64]);
    let b = gray_image(Extent::new(4, 4), vec![0.0; 16]);
    let mut inputs = InputValues::new();
    inputs.insert("before".to_owned(), a);
    inputs.insert("after".to_owned(), b);
    assert!(
        FrequencyPreserved::new()
            .compute(&inputs, &serde_json::json!({"bands": 2}))
            .is_err()
    );

    let a = gray_image(Extent::new(8, 8), vec![0.0; 64]);
    let b = gray_image(Extent::new(8, 8), vec![0.0; 64]);
    let mut inputs = InputValues::new();
    inputs.insert("before".to_owned(), a);
    inputs.insert("after".to_owned(), b);
    assert!(
        FrequencyPreserved::new()
            .compute(&inputs, &serde_json::json!({}))
            .is_err()
    );
}
