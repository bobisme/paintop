//! Verification suite for `debug.materialize@1` (`OP_CATALOG` §1, `plan.md` §18,
//! `AGENT_VERIFICATION` §2.2 analytic fixtures, §2.4 property tests):
//!
//! - **schema/contract**: the manifest validates, agrees with its contract, and
//!   its verification declarations gate clean;
//! - **analytic fixtures**: a constant fixture and a ramp fixture round-trip the
//!   barrier with a bit-identical descriptor and sample buffer;
//! - **property/metamorphic**: the barrier is the *identity* — the output value
//!   equals the input value for arbitrary inputs (including `NaN`/`Inf` payloads),
//!   it is pure (the input is untouched), the type-level `infer_outputs` is the
//!   identity on the descriptor, and ROI is pointwise (a requested output region
//!   demands exactly the co-located input region) so downstream hashes are
//!   unchanged.

use paintop_core::executor::{InputValues, OpImplementation, ResourceValue};
use paintop_ir::{
    AlphaRepresentation, ChannelLayout, ColorEncoding, ColorRange, CoordinateConvention,
    Descriptors, Extent, ImageDescriptor, OpContract, OutputRegions, Rect, Report,
    ResourceDescriptor, ScalarType, SemanticRole, check_contract_consistency, verify_categories,
};

use super::{MATERIALIZE_OP_ID, Materialize};

/// Build an image [`ResourceValue`] of `channels` from an explicit sample buffer
/// (row-major, channel-interleaved).
fn image_value(width: u32, height: u32, channels: u32, samples: Vec<f32>) -> ResourceValue {
    ResourceValue::new(image_descriptor(width, height, channels), channels, samples)
        .expect("sample buffer matches descriptor")
}

/// Build an image descriptor with a layout matching `channels`.
fn image_descriptor(width: u32, height: u32, channels: u32) -> ResourceDescriptor {
    let layout = match channels {
        1 => ChannelLayout::Gray,
        2 => ChannelLayout::GrayA,
        3 => ChannelLayout::Rgb,
        _ => ChannelLayout::Rgba,
    };
    ResourceDescriptor::Image(ImageDescriptor {
        extent: Extent::new(width, height),
        layout,
        scalar: ScalarType::F32,
        color: ColorEncoding::Srgb,
        range: ColorRange::DisplayReferred,
        alpha: AlphaRepresentation::Straight,
        coordinates: CoordinateConvention::PixelCenterUpperLeft,
        semantic: SemanticRole::Color,
    })
}

/// Run the op's compute kernel and recover the produced `resource` value.
fn materialize_value(value: &ResourceValue) -> ResourceValue {
    let mut inputs = InputValues::new();
    inputs.insert("resource".to_owned(), value.clone());
    let out = Materialize::new()
        .compute(&inputs, &serde_json::Value::Null)
        .expect("materialize computes");
    out.get("resource").expect("resource port produced").clone()
}

#[test]
fn manifest_validates_and_agrees_with_contract() {
    let manifest = Materialize::manifest().expect("materialize manifest");
    manifest.validate().expect("materialize manifest valid");
    check_contract_consistency(&manifest, &Materialize::new())
        .expect("materialize manifest agrees with contract");
    verify_categories(&manifest, &manifest.test.verification)
        .expect("materialize verification declarations gate clean");
    assert_eq!(manifest.id.to_string(), MATERIALIZE_OP_ID);
}

#[test]
fn constant_fixture_round_trips_bit_identically() {
    // A 4x4 single-channel constant 0.25: the barrier returns it untouched.
    let value = image_value(4, 4, 1, vec![0.25_f32; 16]);
    let out = materialize_value(&value);
    assert_eq!(
        out, value,
        "the barrier is the identity on a constant fixture"
    );
    assert_eq!(out.descriptor(), value.descriptor());
    assert_eq!(out.samples(), value.samples());
}

#[test]
fn ramp_fixture_round_trips_bit_identically() {
    // A horizontal 0..1 ramp across width 5: every sample survives verbatim.
    let w = 5u32;
    let samples: Vec<f32> = (0..w)
        .map(|x| {
            #[allow(clippy::cast_precision_loss, reason = "small ramp width")]
            let v = x as f32 / (w as f32 - 1.0);
            v
        })
        .collect();
    let value = image_value(w, 1, 1, samples.clone());
    let out = materialize_value(&value);
    assert_eq!(out.samples(), samples.as_slice());
    assert_eq!(out, value);
}

#[test]
fn identity_preserves_nan_and_inf_bit_for_bit() {
    // A materialization barrier must not normalize a NaN payload or collapse an
    // infinity: the bits flow through verbatim.
    let nan = f32::from_bits(0x7fff_dead);
    let samples = vec![1.0, nan, f32::NEG_INFINITY, f32::INFINITY];
    let value = image_value(4, 1, 1, samples);
    let out = materialize_value(&value);

    let out_bits: Vec<u32> = out.samples().iter().map(|s| s.to_bits()).collect();
    let in_bits: Vec<u32> = value.samples().iter().map(|s| s.to_bits()).collect();
    assert_eq!(out_bits, in_bits, "every sample bit is preserved");
}

#[test]
fn identity_over_multichannel_image() {
    // 2x2 RGBA with distinct samples: the whole interleaved buffer is preserved.
    let samples: Vec<f32> = (0..16_i16).map(|i| f32::from(i) / 16.0).collect();
    let value = image_value(2, 2, 4, samples);
    let out = materialize_value(&value);
    assert_eq!(out, value);
    assert_eq!(out.channels(), 4);
    assert_eq!(out.extent(), Extent::new(2, 2));
}

#[test]
fn materialize_is_pure() {
    // The input value is untouched: a clone before equals it after.
    let value = image_value(3, 3, 4, vec![0.3; 36]);
    let before = value.clone();
    let _ = materialize_value(&value);
    assert_eq!(value, before, "materialize must not mutate its input");
}

#[test]
fn infer_outputs_is_the_identity_on_the_descriptor() {
    // The output descriptor is the input descriptor, unchanged (no retype).
    let descriptor = image_descriptor(2, 2, 4);
    let mut inputs = Descriptors::new();
    inputs.insert("resource".to_owned(), descriptor);
    let out = Materialize::new()
        .infer_outputs(&inputs, &serde_json::Value::Null)
        .expect("infer outputs");
    assert_eq!(
        out.get("resource"),
        Some(&descriptor),
        "the barrier preserves the resource type exactly"
    );
}

#[test]
fn required_inputs_is_pointwise() {
    // A requested output region demands exactly the co-located input region: the
    // barrier neither shrinks nor grows demand.
    let descriptor = image_descriptor(8, 8, 4);
    let mut inputs = Descriptors::new();
    inputs.insert("resource".to_owned(), descriptor);
    let mut requested = OutputRegions::new();
    let region = Rect::new(2, 3, 4, 5);
    requested.insert("resource".to_owned(), region);

    let demanded = Materialize::new()
        .required_inputs(&requested, &inputs, &serde_json::Value::Null)
        .expect("required inputs");
    assert_eq!(demanded.get("resource"), Some(&region));
}

#[test]
fn report_resource_round_trips() {
    // A non-raster resource (a Report) is materialized just as faithfully.
    let report = Report {
        extent: Extent::new(2, 2),
        channels: 0,
        channel_stats: Vec::new(),
        all_finite: true,
        content_hash: "blake3:deadbeef".to_owned(),
        diff: None,
        assertion: None,
        histogram: None,
        components: None,
    };
    let value = ResourceValue::report(report.clone());
    let out = materialize_value(&value);
    assert_eq!(out, value);
    assert_eq!(out.as_report(), Some(&report));
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
    let manifest = Materialize::manifest().expect("materialize manifest");
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
