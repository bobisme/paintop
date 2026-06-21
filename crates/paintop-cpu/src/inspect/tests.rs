//! Verification suite for `image.inspect@1` (`OP_CATALOG` §1,
//! `AGENT_VERIFICATION` §2.2 analytic fixtures, §2.4 property tests):
//!
//! - **schema/contract**: the manifest validates, agrees with its contract, and
//!   its verification declarations gate clean;
//! - **analytic fixtures**: a constant fixture reports the known constant as its
//!   min/max/mean with zero non-finite samples; a horizontal ramp reports the
//!   ramp endpoints; a `NaN`/`Inf`-injected fixture flags the non-finite samples
//!   and excludes them from the range;
//! - **property/metamorphic**: the content hash is stable across repeated
//!   inspection, depends on the samples, is invariant to the particular `NaN`
//!   payload, and inspection is pure (the input value is untouched).

use paintop_core::executor::{InputValues, OpImplementation, ResourceValue};
use paintop_ir::{
    AlphaRepresentation, ChannelLayout, ColorEncoding, ColorRange, CoordinateConvention, Extent,
    ImageDescriptor, ResourceDescriptor, ScalarType, SemanticRole, check_contract_consistency,
    verify_categories,
};

use super::{INSPECT_OP_ID, Inspect, inspect_value};

/// Build an RGBA-ish image [`ResourceValue`] of `channels` from an explicit
/// sample buffer (row-major, channel-interleaved).
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
        alpha: AlphaRepresentation::Premultiplied,
        coordinates: CoordinateConvention::PixelCenterUpperLeft,
        semantic: SemanticRole::Color,
    });
    ResourceValue::new(descriptor, channels, samples).expect("sample buffer matches descriptor")
}

/// Run the op's compute kernel and recover the produced report.
fn compute_report(value: &ResourceValue) -> paintop_ir::Report {
    let mut inputs = InputValues::new();
    inputs.insert("image".to_owned(), value.clone());
    let out = Inspect::new()
        .compute(&inputs, &serde_json::Value::Null)
        .expect("inspect computes");
    out.get("report")
        .expect("report port produced")
        .as_report()
        .expect("report payload present")
        .clone()
}

#[test]
fn manifest_validates_and_agrees_with_contract() {
    let manifest = Inspect::manifest().expect("inspect manifest");
    manifest.validate().expect("inspect manifest valid");
    check_contract_consistency(&manifest, &Inspect::new())
        .expect("inspect manifest agrees with contract");
    verify_categories(&manifest, &manifest.test.verification)
        .expect("inspect verification declarations gate clean");
    assert_eq!(manifest.id.to_string(), INSPECT_OP_ID);
}

#[test]
fn constant_fixture_reports_known_stats() {
    // A 4x4 single-channel constant 0.25: min = max = mean = 0.25, all finite.
    let value = image_value(4, 4, 1, vec![0.25_f32; 16]);
    let report = inspect_value(&value);

    assert_eq!(report.extent, Extent::new(4, 4));
    assert_eq!(report.channels, 1);
    assert!(report.all_finite);
    assert_eq!(report.channel_stats.len(), 1);
    let c = report.channel_stats[0];
    assert_eq!(c.min, Some(0.25));
    assert_eq!(c.max, Some(0.25));
    assert_eq!(c.finite, 16);
    assert_eq!(c.nonfinite, 0);
    assert!((c.mean().expect("mean") - 0.25).abs() < 1e-12);
    // The compute kernel produces the identical report.
    assert_eq!(compute_report(&value), report);
}

#[test]
fn per_channel_stats_are_independent() {
    // 2x1 RGBA: channel c holds value (c+1)*0.1 in both pixels.
    let mut samples = Vec::new();
    for _ in 0..2 {
        for c in 0..4 {
            #[allow(clippy::cast_precision_loss, reason = "small channel index")]
            samples.push((c as f32 + 1.0) * 0.1);
        }
    }
    let value = image_value(2, 1, 4, samples);
    let report = inspect_value(&value);
    assert_eq!(report.channels, 4);
    for (c, stat) in report.channel_stats.iter().enumerate() {
        #[allow(clippy::cast_precision_loss, reason = "small channel index")]
        let expected = (c as f32 + 1.0) * 0.1;
        assert_eq!(stat.min, Some(expected));
        assert_eq!(stat.max, Some(expected));
        assert_eq!(stat.finite, 2);
    }
}

#[test]
fn ramp_fixture_reports_endpoints() {
    // A horizontal 0..1 ramp across width 5 (single channel, 1 row).
    let w = 5u32;
    let samples: Vec<f32> = (0..w)
        .map(|x| {
            #[allow(clippy::cast_precision_loss, reason = "small ramp width")]
            let v = x as f32 / (w as f32 - 1.0);
            v
        })
        .collect();
    let value = image_value(w, 1, 1, samples);
    let report = inspect_value(&value);
    let c = report.channel_stats[0];
    assert_eq!(c.min, Some(0.0));
    assert_eq!(c.max, Some(1.0));
    assert_eq!(c.finite, u64::from(w));
    assert!(c.all_finite());
    // Mean of an evenly spaced 0..1 ramp is 0.5.
    assert!((c.mean().expect("mean") - 0.5).abs() < 1e-6);
}

#[test]
fn nan_and_inf_are_flagged_and_excluded_from_range() {
    // Single channel: [1.0, NaN, -inf, +inf, 2.0]. Finite range is [1.0, 2.0];
    // three samples are non-finite.
    let samples = vec![1.0, f32::NAN, f32::NEG_INFINITY, f32::INFINITY, 2.0];
    let value = image_value(5, 1, 1, samples);
    let report = inspect_value(&value);

    assert!(!report.all_finite);
    let c = report.channel_stats[0];
    assert_eq!(c.min, Some(1.0));
    assert_eq!(c.max, Some(2.0));
    assert_eq!(c.finite, 2);
    assert_eq!(c.nonfinite, 3);
    assert!(!c.all_finite());
}

#[test]
fn all_nonfinite_channel_has_no_range() {
    let value = image_value(2, 1, 1, vec![f32::NAN, f32::INFINITY]);
    let report = inspect_value(&value);
    let c = report.channel_stats[0];
    assert_eq!(c.min, None);
    assert_eq!(c.max, None);
    assert_eq!(c.mean(), None);
    assert_eq!(c.finite, 0);
    assert_eq!(c.nonfinite, 2);
    assert!(!report.all_finite);
}

#[test]
fn content_hash_is_stable_and_algorithm_prefixed() {
    let value = image_value(3, 3, 1, (0..9_i16).map(|i| f32::from(i) / 9.0).collect());
    let a = inspect_value(&value).content_hash;
    let b = inspect_value(&value).content_hash;
    assert_eq!(a, b, "hashing the same samples twice is stable");
    assert!(a.starts_with("blake3:"), "{a}");
}

#[test]
fn content_hash_depends_on_samples() {
    let a = inspect_value(&image_value(2, 1, 1, vec![0.1, 0.2])).content_hash;
    let b = inspect_value(&image_value(2, 1, 1, vec![0.1, 0.3])).content_hash;
    assert_ne!(a, b, "a changed sample changes the content hash");
}

#[test]
fn content_hash_is_invariant_to_nan_payload() {
    // Two distinct NaN bit patterns must hash identically (logical content is
    // "not a number" in both).
    let nan_a = f32::from_bits(0x7fc0_0001);
    let nan_b = f32::from_bits(0x7fff_dead);
    assert!(nan_a.is_nan() && nan_b.is_nan());
    let a = inspect_value(&image_value(1, 1, 1, vec![nan_a])).content_hash;
    let b = inspect_value(&image_value(1, 1, 1, vec![nan_b])).content_hash;
    assert_eq!(a, b, "NaN payload must not affect the content hash");
}

#[test]
fn inspect_is_pure() {
    // The input value is untouched: a clone before inspection equals it after.
    let value = image_value(2, 2, 4, vec![0.3; 16]);
    let before = value.clone();
    let _ = inspect_value(&value);
    let _ = compute_report(&value);
    assert_eq!(value, before, "inspect must not mutate its input");
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
    let manifest = Inspect::manifest().expect("inspect manifest");
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
