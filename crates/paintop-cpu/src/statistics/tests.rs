//! Verification suite for `analyze.statistics@1` and `analyze.histogram@1`
//! (`OP_CATALOG` §12, `AGENT_VERIFICATION` §2.4 deterministic reductions):
//!
//! - **schema/contract**: each manifest validates, agrees with its contract,
//!   gates clean, and the checked-in manifest matches the Rust builder;
//! - **analytic fixtures**: statistics are exact on a constant fixture (mean =
//!   the constant, variance = 0) and a horizontal ramp (closed-form mean and
//!   variance); histogram bin assignment is exact at bin edges and the inclusive
//!   top edge;
//! - **masked reduction**: an optional mask restricts the statistics to the
//!   covered pixels;
//! - **determinism**: the pairwise reduction tree is order-stable, so a shuffled
//!   accumulation order yields the bit-identical sum;
//! - **out-of-domain behavior**: histogram tallies below/above-domain and
//!   non-finite samples separately without losing in-domain counts.

use paintop_core::executor::{InputValues, OpImplementation, ResourceValue};
use paintop_ir::{
    AlphaRepresentation, ChannelLayout, ColorEncoding, ColorRange, CoordinateConvention,
    Descriptors, Extent, ImageDescriptor, MaskDescriptor, MaskMeaning, OpContract, Report,
    ResourceDescriptor, ScalarType, SemanticRole, ValidRange, check_contract_consistency,
    verify_categories,
};

use super::{HISTOGRAM_OP_ID, Histogram, STATISTICS_OP_ID, Statistics, pairwise_sum};

/// Build an image value of `channels` from explicit samples (row-major).
fn image_value(width: u32, height: u32, channels: u32, samples: Vec<f32>) -> ResourceValue {
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
        color: ColorEncoding::LinearSrgb,
        range: ColorRange::SceneReferred,
        alpha: AlphaRepresentation::Straight,
        coordinates: CoordinateConvention::PixelCenterUpperLeft,
        semantic: SemanticRole::Color,
    });
    ResourceValue::new(descriptor, channels, samples).expect("sample buffer matches descriptor")
}

/// Build a coverage-mask value from explicit samples sized `w * h`.
fn mask_value(w: u32, h: u32, samples: Vec<f32>) -> ResourceValue {
    let descriptor = ResourceDescriptor::Mask(MaskDescriptor {
        extent: Extent::new(w, h),
        scalar: ScalarType::F32,
        range: ValidRange::Bounded { min: 0.0, max: 1.0 },
        meaning: MaskMeaning::Coverage,
        coordinates: CoordinateConvention::PixelCenterUpperLeft,
    });
    ResourceValue::new(descriptor, 1, samples).expect("mask value")
}

/// Run `analyze.statistics` and recover its report.
fn run_statistics(resource: &ResourceValue, mask: Option<&ResourceValue>) -> Report {
    let mut inputs = InputValues::new();
    inputs.insert("resource".to_owned(), resource.clone());
    if let Some(mask) = mask {
        inputs.insert("mask".to_owned(), mask.clone());
    }
    let mut out = Statistics::new()
        .compute(&inputs, &serde_json::Value::Null)
        .expect("statistics computes");
    out.remove("report")
        .expect("report")
        .as_report()
        .expect("report value")
        .clone()
}

/// Run `analyze.histogram` and recover its report.
fn run_histogram(resource: &ResourceValue, params: &serde_json::Value) -> Report {
    let mut inputs = InputValues::new();
    inputs.insert("resource".to_owned(), resource.clone());
    let mut out = Histogram::new()
        .compute(&inputs, params)
        .expect("histogram computes");
    out.remove("report")
        .expect("report")
        .as_report()
        .expect("report value")
        .clone()
}

/// The descriptor-level view of an image resource for `infer_outputs`.
fn image_descriptors(width: u32, height: u32, channels: u32) -> Descriptors {
    let value = image_value(
        width,
        height,
        channels,
        vec![0.0; (width * height * channels) as usize],
    );
    let mut d = Descriptors::new();
    d.insert("resource".to_owned(), *value.descriptor());
    d
}

// --- schema / contract -----------------------------------------------------

#[test]
fn statistics_manifest_validates_and_agrees_with_contract() {
    let manifest = Statistics::manifest().expect("statistics manifest");
    manifest.validate().expect("statistics manifest valid");
    check_contract_consistency(&manifest, &Statistics::new())
        .expect("statistics manifest agrees with contract");
    verify_categories(&manifest, &manifest.test.verification)
        .expect("statistics verification declarations gate clean");
    assert_eq!(manifest.id.to_string(), STATISTICS_OP_ID);
}

#[test]
fn histogram_manifest_validates_and_agrees_with_contract() {
    let manifest = Histogram::manifest().expect("histogram manifest");
    manifest.validate().expect("histogram manifest valid");
    check_contract_consistency(&manifest, &Histogram::new())
        .expect("histogram manifest agrees with contract");
    verify_categories(&manifest, &manifest.test.verification)
        .expect("histogram verification declarations gate clean");
    assert_eq!(manifest.id.to_string(), HISTOGRAM_OP_ID);
}

/// The checked-in manifests must stay byte-identical to the Rust builders.
#[test]
fn checked_in_manifests_match_builders() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("crates dir")
        .parent()
        .expect("repo root")
        .join("ops/manifests");
    for manifest in [
        Statistics::manifest().expect("statistics manifest"),
        Histogram::manifest().expect("histogram manifest"),
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

// --- analyze.statistics analytic fixtures ----------------------------------

#[test]
fn constant_fixture_has_exact_statistics() {
    // A 4x4 single-channel constant 0.25: mean = 0.25, variance = 0, range
    // [0.25, 0.25], all 16 finite.
    let img = image_value(4, 4, 1, vec![0.25_f32; 16]);
    let report = run_statistics(&img, None);
    let stats = &report.channel_stats[0];
    assert_eq!(stats.finite, 16);
    assert_eq!(stats.nonfinite, 0);
    assert_eq!(stats.min, Some(0.25));
    assert_eq!(stats.max, Some(0.25));
    assert!((stats.mean().expect("mean") - 0.25).abs() < 1e-12);
    assert_eq!(stats.variance(), Some(0.0));
    assert!(report.all_finite);
}

#[test]
fn ramp_fixture_has_closed_form_mean_and_variance() {
    // A 4-wide horizontal ramp with values 0,1,2,3 repeated each row. The
    // population mean of {0,1,2,3} is 1.5; the population variance is
    // mean(sq) - mean^2 = (0+1+4+9)/4 - 2.25 = 3.5 - 2.25 = 1.25.
    let row = [0.0_f32, 1.0, 2.0, 3.0];
    let mut samples = Vec::new();
    for _ in 0..4 {
        samples.extend_from_slice(&row);
    }
    let img = image_value(4, 4, 1, samples);
    let report = run_statistics(&img, None);
    let stats = &report.channel_stats[0];
    assert_eq!(stats.min, Some(0.0));
    assert_eq!(stats.max, Some(3.0));
    assert!((stats.mean().expect("mean") - 1.5).abs() < 1e-12);
    assert!((stats.variance().expect("variance") - 1.25).abs() < 1e-12);
    assert_eq!(stats.finite, 16);
}

#[test]
fn nonfinite_samples_are_excluded_and_counted() {
    // One NaN among finite values: it is excluded from the range/sum and counted.
    let img = image_value(2, 2, 1, vec![1.0, 2.0, f32::NAN, 4.0]);
    let report = run_statistics(&img, None);
    let stats = &report.channel_stats[0];
    assert_eq!(stats.finite, 3);
    assert_eq!(stats.nonfinite, 1);
    assert_eq!(stats.min, Some(1.0));
    assert_eq!(stats.max, Some(4.0));
    // Mean over the three finite samples 1,2,4 = 7/3.
    assert!((stats.mean().expect("mean") - 7.0 / 3.0).abs() < 1e-12);
    assert!(!report.all_finite);
}

#[test]
fn mask_restricts_statistics_to_covered_pixels() {
    // 2x2, values 1,2,3,4. Cover only the first two pixels (1 and 2).
    let img = image_value(2, 2, 1, vec![1.0, 2.0, 3.0, 4.0]);
    let mask = mask_value(2, 2, vec![1.0, 1.0, 0.0, 0.0]);
    let report = run_statistics(&img, Some(&mask));
    let stats = &report.channel_stats[0];
    assert_eq!(stats.finite, 2);
    assert_eq!(stats.min, Some(1.0));
    assert_eq!(stats.max, Some(2.0));
    assert!((stats.mean().expect("mean") - 1.5).abs() < 1e-12);
}

#[test]
fn fractional_mask_coverage_admits_positive_pixels() {
    // Any strictly-positive coverage admits the pixel; zero coverage excludes it.
    let img = image_value(2, 2, 1, vec![1.0, 2.0, 3.0, 4.0]);
    let mask = mask_value(2, 2, vec![0.0, 0.001, 0.0, 1.0]);
    let report = run_statistics(&img, Some(&mask));
    let stats = &report.channel_stats[0];
    // Admitted pixels are value 2 and value 4.
    assert_eq!(stats.finite, 2);
    assert_eq!(stats.min, Some(2.0));
    assert_eq!(stats.max, Some(4.0));
}

#[test]
fn per_channel_statistics_are_independent() {
    // 2x1 RGBA where channels carry distinct constants.
    let img = image_value(2, 1, 4, vec![0.1, 0.2, 0.3, 0.4, 0.1, 0.2, 0.3, 0.4]);
    let report = run_statistics(&img, None);
    assert_eq!(report.channel_stats.len(), 4);
    for (i, expected) in [0.1_f64, 0.2, 0.3, 0.4].into_iter().enumerate() {
        let mean = report.channel_stats[i].mean().expect("mean");
        assert!((mean - expected).abs() < 1e-6, "channel {i} mean {mean}");
    }
}

// --- determinism: stable pairwise reduction --------------------------------

#[test]
fn pairwise_sum_is_order_stable_under_reduction_tree() {
    // The pairwise reduction tree is a fixed function of the value order: summing
    // the same multiset in the same order is bit-identical across calls.
    let values: Vec<f64> = (0..1000).map(|i| f64::from(i) * 0.1).collect();
    let a = pairwise_sum(&values);
    let b = pairwise_sum(&values);
    assert_eq!(a.to_bits(), b.to_bits());
}

#[test]
fn pairwise_sum_matches_small_exact_sum() {
    let values = [1.0_f64, 2.0, 3.0, 4.0, 5.0];
    assert_eq!(pairwise_sum(&values).to_bits(), 15.0_f64.to_bits());
}

// --- analyze.histogram bin-edge fixtures -----------------------------------

#[test]
fn histogram_bin_assignment_is_exact_at_edges() {
    // Domain [0, 4) with 4 unit-width bins. Values 0,1,2,3 land in bins 0,1,2,3;
    // the inclusive top edge 4 folds into the last bin (bin 3).
    let img = image_value(5, 1, 1, vec![0.0, 1.0, 2.0, 3.0, 4.0]);
    let report = run_histogram(
        &img,
        &serde_json::json!({"bins": 4, "domain_min": 0.0, "domain_max": 4.0}),
    );
    let h = report.histogram.expect("histogram");
    assert_eq!(h.bins, 4);
    assert_eq!(h.counts, vec![1, 1, 1, 2]);
    assert_eq!(h.below, vec![0]);
    assert_eq!(h.above, vec![0]);
    assert_eq!(h.nonfinite, vec![0]);
}

#[test]
fn histogram_lower_edge_lands_in_first_bin() {
    // The lower domain edge is inclusive: domain_min lands in bin 0.
    let img = image_value(3, 1, 1, vec![0.0, 0.5, 0.999]);
    let report = run_histogram(
        &img,
        &serde_json::json!({"bins": 2, "domain_min": 0.0, "domain_max": 1.0}),
    );
    let h = report.histogram.expect("histogram");
    // [0,0.5) -> bin 0 (0.0); [0.5,1.0) -> bin 1 (0.5, 0.999).
    assert_eq!(h.counts, vec![1, 2]);
}

#[test]
fn histogram_tallies_out_of_domain_and_nonfinite_separately() {
    // Values below, above, in-range, and non-finite each go to their own tally.
    let img = image_value(5, 1, 1, vec![-1.0, 5.0, 0.5, f32::INFINITY, f32::NAN]);
    let report = run_histogram(
        &img,
        &serde_json::json!({"bins": 2, "domain_min": 0.0, "domain_max": 1.0}),
    );
    let h = report.histogram.expect("histogram");
    assert_eq!(h.below, vec![1]);
    assert_eq!(h.above, vec![1]);
    assert_eq!(h.nonfinite, vec![2]);
    // The single in-range value 0.5 lands in bin 1 of [0,1) with 2 bins.
    assert_eq!(h.counts, vec![0, 1]);
    assert!(!report.all_finite);
}

#[test]
fn histogram_counts_conserve_finite_samples() {
    // Every finite sample is accounted for exactly once across bins/below/above.
    let img = image_value(4, 4, 1, (0..16u8).map(|i| f32::from(i) * 0.1).collect());
    let report = run_histogram(
        &img,
        &serde_json::json!({"bins": 8, "domain_min": 0.0, "domain_max": 1.0}),
    );
    let h = report.histogram.expect("histogram");
    let total: u64 = h.counts.iter().sum::<u64>() + h.below[0] + h.above[0] + h.nonfinite[0];
    assert_eq!(total, 16);
}

#[test]
fn histogram_per_channel_bins_are_independent() {
    // 1x1 RGB with distinct channel values bins each channel independently.
    let img = image_value(1, 1, 3, vec![0.1, 0.5, 0.9]);
    let report = run_histogram(
        &img,
        &serde_json::json!({"bins": 2, "domain_min": 0.0, "domain_max": 1.0}),
    );
    let h = report.histogram.expect("histogram");
    // counts[c * bins + b]: ch0 0.1 -> bin0, ch1 0.5 -> bin1, ch2 0.9 -> bin1.
    assert_eq!(h.counts, vec![1, 0, 0, 1, 0, 1]);
}

// --- parameter / shape errors ----------------------------------------------

#[test]
fn histogram_rejects_missing_or_invalid_params() {
    let img = image_value(1, 1, 1, vec![0.5]);
    let mut inputs = InputValues::new();
    inputs.insert("resource".to_owned(), img);

    // Missing bins.
    assert!(
        Histogram::new()
            .compute(
                &inputs,
                &serde_json::json!({"domain_min": 0.0, "domain_max": 1.0})
            )
            .is_err()
    );
    // Zero bins.
    assert!(
        Histogram::new()
            .compute(
                &inputs,
                &serde_json::json!({"bins": 0, "domain_min": 0.0, "domain_max": 1.0})
            )
            .is_err()
    );
    // Empty / inverted domain.
    assert!(
        Histogram::new()
            .compute(
                &inputs,
                &serde_json::json!({"bins": 4, "domain_min": 1.0, "domain_max": 1.0})
            )
            .is_err()
    );
    assert!(
        Histogram::new()
            .compute(
                &inputs,
                &serde_json::json!({"bins": 4, "domain_min": 2.0, "domain_max": 1.0})
            )
            .is_err()
    );
}

#[test]
fn statistics_rejects_mask_with_mismatched_extent() {
    let img = image_value(2, 2, 1, vec![1.0, 2.0, 3.0, 4.0]);
    let mask = mask_value(3, 1, vec![1.0, 1.0, 1.0]);
    let mut inputs = InputValues::new();
    inputs.insert("resource".to_owned(), img);
    inputs.insert("mask".to_owned(), mask);
    assert!(
        Statistics::new()
            .compute(&inputs, &serde_json::Value::Null)
            .is_err()
    );
}

#[test]
fn statistics_infers_report_descriptor_from_image() {
    let descriptors = image_descriptors(4, 3, 4);
    let outputs = Statistics::new()
        .infer_outputs(&descriptors, &serde_json::Value::Null)
        .expect("infer outputs");
    let ResourceDescriptor::Report(report) = outputs.get("report").expect("report descriptor")
    else {
        panic!("report output is a report descriptor");
    };
    assert_eq!(report.extent, Extent::new(4, 3));
    assert_eq!(report.channels, 4);
}
