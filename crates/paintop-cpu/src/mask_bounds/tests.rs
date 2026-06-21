//! Verification suite for `mask.bounds@1` (`OP_CATALOG` §4):
//!
//! - **schema/contract**: the manifest validates, agrees with its contract,
//!   gates clean, and the checked-in manifest matches the Rust builder;
//! - **analytic bounds**: a rect mask reports the exact tight bounds and
//!   occupancy; an ellipse-shaped mask reports the analytic bounding box;
//! - **empty-mask behavior**: an all-zero mask reports `None` bounds and `0`
//!   occupancy without panicking; a zero-extent mask likewise;
//! - **occupancy correctness**: the count equals the number of positive-coverage
//!   pixels, including fractional (antialiased) coverage;
//! - **translation metamorphic**: translating the occupied region translates the
//!   reported bounds by the same offset.

use paintop_core::executor::{InputValues, OpImplementation, ResourceValue};
use paintop_ir::{
    CoordinateConvention, Descriptors, ErrorClass, Extent, MaskDescriptor, MaskMeaning, OpContract,
    Rect, Report, ResourceDescriptor, ScalarType, ValidRange, check_contract_consistency,
    verify_categories,
};

use super::{BOUNDS_OP_ID, MaskBounds};

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

/// Run `mask.bounds` and return the produced report.
fn bounds(value: &ResourceValue) -> Report {
    let mut inputs = InputValues::new();
    inputs.insert("mask".to_owned(), value.clone());
    let mut out = MaskBounds::new()
        .compute(&inputs, &serde_json::json!({}))
        .expect("bounds computes");
    out.remove("report")
        .expect("report")
        .as_report()
        .expect("report value")
        .clone()
}

/// A rect mask: coverage 1 inside the half-open pixel rect `[x0, x1) × [y0, y1)`.
fn rect_mask(w: u32, h: u32, x0: u32, y0: u32, x1: u32, y1: u32) -> ResourceValue {
    let mut samples = vec![0.0_f32; (w * h) as usize];
    for y in y0..y1 {
        for x in x0..x1 {
            samples[(y * w + x) as usize] = 1.0;
        }
    }
    mask(w, h, samples)
}

// --- schema / contract -----------------------------------------------------

#[test]
fn manifest_validates_and_agrees_with_contract() {
    let manifest = MaskBounds::manifest().expect("manifest");
    manifest.validate().expect("manifest valid");
    verify_categories(&manifest, &manifest.test.verification)
        .expect("verification declarations gate clean");
    assert_eq!(manifest.id.to_string(), BOUNDS_OP_ID);
    check_contract_consistency(&manifest, &MaskBounds::new()).unwrap();
}

// --- analytic bounds + occupancy -------------------------------------------

#[test]
fn rect_mask_reports_exact_tight_bounds() {
    // A 2..5 × 1..4 occupied block in a 8x6 mask.
    let m = rect_mask(8, 6, 2, 1, 5, 4);
    let report = bounds(&m);
    let diff = report.diff.expect("diff metrics present");
    assert_eq!(diff.changed_bounds, Some(Rect::new(2, 1, 5, 4)));
    // Occupancy = 3 columns * 3 rows = 9 pixels.
    assert_eq!(diff.changed_count, 9);
    assert_eq!(report.extent, Extent::new(8, 6));
    assert_eq!(report.channels, 1);
}

#[test]
fn fractional_coverage_counts_as_occupied() {
    // Any strictly-positive coverage is occupied, including antialiased edges.
    let m = mask(4, 1, vec![0.0, 0.25, 1.0, 0.0]);
    let report = bounds(&m);
    let diff = report.diff.expect("diff");
    assert_eq!(diff.changed_count, 2);
    assert_eq!(diff.changed_bounds, Some(Rect::new(1, 0, 3, 1)));
}

#[test]
fn single_pixel_bounds() {
    let mut samples = vec![0.0_f32; 25];
    samples[3 * 5 + 4] = 0.5; // pixel (4, 3)
    let report = bounds(&mask(5, 5, samples));
    let diff = report.diff.expect("diff");
    assert_eq!(diff.changed_bounds, Some(Rect::new(4, 3, 5, 4)));
    assert_eq!(diff.changed_count, 1);
}

// --- empty-mask behavior ---------------------------------------------------

#[test]
fn empty_mask_reports_empty_bounds_not_panic() {
    let m = mask(6, 4, vec![0.0; 24]);
    let report = bounds(&m);
    let diff = report.diff.expect("diff");
    assert_eq!(diff.changed_bounds, None);
    assert_eq!(diff.changed_count, 0);
    assert_eq!(report.extent, Extent::new(6, 4));
}

#[test]
fn zero_extent_mask_reports_empty_bounds() {
    let m = mask(0, 0, vec![]);
    let report = bounds(&m);
    let diff = report.diff.expect("diff");
    assert_eq!(diff.changed_bounds, None);
    assert_eq!(diff.changed_count, 0);
}

#[test]
fn full_mask_bounds_is_whole_extent() {
    let m = mask(3, 2, vec![1.0; 6]);
    let report = bounds(&m);
    let diff = report.diff.expect("diff");
    assert_eq!(diff.changed_bounds, Some(Rect::new(0, 0, 3, 2)));
    assert_eq!(diff.changed_count, 6);
}

// --- metamorphic: translation ----------------------------------------------

#[test]
fn translating_occupied_region_translates_bounds() {
    let base = rect_mask(10, 10, 1, 1, 3, 3);
    let shifted = rect_mask(10, 10, 4, 5, 6, 7);
    let b = bounds(&base).diff.expect("diff").changed_bounds.expect("b");
    let s = bounds(&shifted)
        .diff
        .expect("diff")
        .changed_bounds
        .expect("s");
    // The shifted region is the base translated by (+3, +4); the bounds shift by
    // the same offset and keep the same size.
    assert_eq!(s.x0 - b.x0, 3);
    assert_eq!(s.y0 - b.y0, 4);
    assert_eq!(s.width(), b.width());
    assert_eq!(s.height(), b.height());
}

// --- rejection -------------------------------------------------------------

#[test]
fn missing_input_is_rejected() {
    let inputs = Descriptors::new();
    let err = MaskBounds::new()
        .infer_outputs(&inputs, &serde_json::json!({}))
        .expect_err("missing input must be rejected");
    assert_eq!(err.class, ErrorClass::Reference);
    assert_eq!(err.code, super::E_BOUNDS_INPUT);
}

/// The checked-in `ops/manifests/mask.bounds@1.json` must stay byte-identical to
/// the Rust manifest builder.
#[test]
fn checked_in_manifest_matches_builder() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("crates dir")
        .parent()
        .expect("repo root")
        .join("ops/manifests");
    let manifest = MaskBounds::manifest().expect("manifest");
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
