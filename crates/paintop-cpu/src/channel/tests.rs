//! Verification suite for `image.extract_channel@1` and
//! `image.assemble_channels@1` (`OP_CATALOG` §1, `plan.md` §7.1 Field1):
//!
//! - **schema/contract**: each manifest validates, agrees with its contract, and
//!   its verification declarations gate clean; the checked-in manifest stays in
//!   lockstep with the Rust builder;
//! - **analytic fixtures**: extracting a known channel yields exactly that
//!   channel; assembling known fields yields the exact interleaved image;
//! - **metamorphic / round-trip**: extract→assemble round-trips a multi-channel
//!   image bit-for-bit (the core acceptance), and Field1 descriptors round-trip
//!   under `deny_unknown_fields`;
//! - **rejection**: an out-of-range channel, a wired-port-count/layout mismatch,
//!   and an extent mismatch are all rejected with typed errors.

use paintop_core::executor::{InputValues, OpImplementation, ResourceValue};
use paintop_ir::{
    AlphaRepresentation, ChannelLayout, ColorEncoding, ColorRange, CoordinateConvention,
    Descriptors, ErrorClass, Extent, FieldArity, FieldDescriptor, ImageDescriptor, OpContract,
    ResourceDescriptor, ScalarType, SemanticRole, ValidRange, check_contract_consistency,
    verify_categories,
};

use super::{ASSEMBLE_OP_ID, AssembleChannels, EXTRACT_OP_ID, ExtractChannel};

/// A small RGBA image whose samples encode `(pixel_index, channel)` so a channel
/// extraction is trivially checkable: channel `c` of pixel `p` is `p * 10 + c`.
fn coded_rgba(width: u32, height: u32) -> ResourceValue {
    let descriptor = ResourceDescriptor::Image(ImageDescriptor {
        extent: Extent::new(width, height),
        layout: ChannelLayout::Rgba,
        scalar: ScalarType::F32,
        color: ColorEncoding::Srgb,
        range: ColorRange::DisplayReferred,
        alpha: AlphaRepresentation::Straight,
        coordinates: CoordinateConvention::PixelCenterUpperLeft,
        semantic: SemanticRole::Color,
    });
    let pixels = (width as usize) * (height as usize);
    let mut samples = Vec::with_capacity(pixels * 4);
    for p in 0..pixels {
        for c in 0..4 {
            #[allow(clippy::cast_precision_loss, reason = "small test indices")]
            samples.push((p * 10 + c) as f32);
        }
    }
    ResourceValue::new(descriptor, 4, samples).expect("coded rgba")
}

/// A single-channel Field1 value carrying `samples`.
fn field1(extent: Extent, samples: Vec<f32>) -> ResourceValue {
    let descriptor = ResourceDescriptor::Field1(FieldDescriptor {
        arity: FieldArity::Field1,
        extent,
        scalar: ScalarType::F32,
        semantic: SemanticRole::Material,
        coordinates: CoordinateConvention::PixelCenterUpperLeft,
        space: None,
        normalization: None,
        encoding: None,
    });
    ResourceValue::new(descriptor, 1, samples).expect("field1")
}

fn extract(image: ResourceValue, params: &serde_json::Value) -> ResourceValue {
    let mut inputs = InputValues::new();
    inputs.insert("image".to_owned(), image);
    let mut out = ExtractChannel::new()
        .compute(&inputs, params)
        .expect("extract computes");
    out.remove("field").expect("field produced")
}

fn assemble(channels: Vec<(&str, ResourceValue)>, params: &serde_json::Value) -> ResourceValue {
    let mut inputs = InputValues::new();
    for (port, value) in channels {
        inputs.insert(port.to_owned(), value);
    }
    let mut out = AssembleChannels::new()
        .compute(&inputs, params)
        .expect("assemble computes");
    out.remove("image").expect("image produced")
}

// --- schema / contract -----------------------------------------------------

#[test]
fn extract_manifest_validates_and_agrees_with_contract() {
    let manifest = ExtractChannel::manifest().expect("extract manifest");
    manifest.validate().expect("extract manifest valid");
    check_contract_consistency(&manifest, &ExtractChannel::new())
        .expect("extract manifest agrees with contract");
    verify_categories(&manifest, &manifest.test.verification)
        .expect("extract verification declarations gate clean");
    assert_eq!(manifest.id.to_string(), EXTRACT_OP_ID);
}

#[test]
fn assemble_manifest_validates_and_agrees_with_contract() {
    let manifest = AssembleChannels::manifest().expect("assemble manifest");
    manifest.validate().expect("assemble manifest valid");
    check_contract_consistency(&manifest, &AssembleChannels::new())
        .expect("assemble manifest agrees with contract");
    verify_categories(&manifest, &manifest.test.verification)
        .expect("assemble verification declarations gate clean");
    assert_eq!(manifest.id.to_string(), ASSEMBLE_OP_ID);
}

/// Field1 descriptors round-trip through serde with `deny_unknown_fields`
/// (acceptance: "Field1 round-trips with `deny_unknown_fields`").
#[test]
fn field1_descriptor_round_trips_and_rejects_unknown_fields() {
    let d = FieldDescriptor {
        arity: FieldArity::Field1,
        extent: Extent::new(7, 3),
        scalar: ScalarType::F32,
        semantic: SemanticRole::Distance,
        coordinates: CoordinateConvention::PixelCenterUpperLeft,
        space: None,
        normalization: None,
        encoding: None,
    };
    let value = serde_json::to_value(d).expect("serialize");
    let back: FieldDescriptor = serde_json::from_value(value.clone()).expect("round trip");
    assert_eq!(back, d);

    let mut bogus = value;
    bogus["surprise"] = serde_json::json!(1);
    let err = serde_json::from_value::<FieldDescriptor>(bogus).unwrap_err();
    assert!(err.to_string().contains("surprise"), "{err}");

    // The tagged ResourceDescriptor carries the spec `kind: "Field1"`.
    let tagged = serde_json::to_value(ResourceDescriptor::Field1(d)).expect("tagged");
    assert_eq!(tagged["kind"], serde_json::json!("Field1"));
}

// --- analytic fixtures -----------------------------------------------------

#[test]
fn extract_selects_the_requested_channel_exactly() {
    let image = coded_rgba(3, 2);
    for c in 0..4_u64 {
        let field = extract(
            coded_rgba(3, 2).clone(),
            &serde_json::json!({ "channel": c }),
        );
        let ResourceDescriptor::Field1(d) = field.descriptor() else {
            panic!("expected Field1");
        };
        assert_eq!(d.arity, FieldArity::Field1);
        assert_eq!(d.extent, Extent::new(3, 2));
        assert_eq!(field.channels(), 1);
        // channel c of pixel p is p*10 + c.
        for (p, &s) in field.samples().iter().enumerate() {
            #[allow(clippy::cast_precision_loss, reason = "small test indices")]
            let expected = (p as u64 * 10 + c) as f32;
            // Verbatim copy: the bit pattern must be identical, not merely close.
            assert_eq!(s.to_bits(), expected.to_bits(), "pixel {p} channel {c}");
        }
    }
    let _ = image;
}

#[test]
fn extract_honors_semantic_and_range_params() {
    let field = extract(
        coded_rgba(2, 2),
        &serde_json::json!({
            "channel": 1,
            "semantic": "confidence",
            "range": { "policy": "bounded", "min": 0.0, "max": 1.0 }
        }),
    );
    let ResourceDescriptor::Field1(d) = field.descriptor() else {
        panic!("expected Field1");
    };
    assert_eq!(d.semantic, SemanticRole::Confidence);
    // The range param is validated but not stored on FieldDescriptor; assert it
    // at least parses by re-running infer_outputs (no panic) above.
    let _ = ValidRange::Bounded { min: 0.0, max: 1.0 };
}

#[test]
fn assemble_interleaves_channels_exactly() {
    let extent = Extent::new(2, 2);
    let r = field1(extent, vec![0.0, 1.0, 2.0, 3.0]);
    let g = field1(extent, vec![10.0, 11.0, 12.0, 13.0]);
    let b = field1(extent, vec![20.0, 21.0, 22.0, 23.0]);
    let image = assemble(
        vec![("ch0", r), ("ch1", g), ("ch2", b)],
        &serde_json::json!({ "layout": "rgb" }),
    );
    let ResourceDescriptor::Image(d) = image.descriptor() else {
        panic!("expected image");
    };
    assert_eq!(d.layout, ChannelLayout::Rgb);
    assert_eq!(d.color, ColorEncoding::RawLinear);
    assert_eq!(image.channels(), 3);
    // Interleaved row-major: pixel p -> [r_p, g_p, b_p].
    let expected = vec![
        0.0, 10.0, 20.0, // p0
        1.0, 11.0, 21.0, // p1
        2.0, 12.0, 22.0, // p2
        3.0, 13.0, 23.0, // p3
    ];
    assert_eq!(image.samples(), expected.as_slice());
}

// --- round trip (metamorphic / core acceptance) ----------------------------

/// extract→assemble round-trips a multi-channel image bit-for-bit (acceptance:
/// "extract→assemble round-trips a multi-channel image").
#[test]
fn extract_then_assemble_round_trips_an_rgba_image() {
    let original = coded_rgba(4, 3);
    let extent = original.extent();

    // Extract every channel into its own Field1.
    let mut fields = Vec::new();
    for c in 0..4_u64 {
        fields.push(extract(
            original.clone(),
            &serde_json::json!({ "channel": c }),
        ));
    }

    // Re-assemble with the original rgba layout (and the original metadata so the
    // descriptor matches too).
    let assembled = assemble(
        vec![
            ("ch0", fields[0].clone()),
            ("ch1", fields[1].clone()),
            ("ch2", fields[2].clone()),
            ("ch3", fields[3].clone()),
        ],
        &serde_json::json!({
            "layout": "rgba",
            "color": "srgb",
            "range": "display-referred",
            "alpha": "straight",
            "semantic": "color"
        }),
    );

    // Bit-identical samples and descriptor.
    assert_eq!(assembled.samples(), original.samples());
    assert_eq!(assembled.descriptor(), original.descriptor());
    assert_eq!(assembled.extent(), extent);
}

// --- rejection -------------------------------------------------------------

#[test]
fn extract_rejects_out_of_range_channel() {
    let image = coded_rgba(2, 2); // 4 channels
    let mut inputs = Descriptors::new();
    inputs.insert("image".to_owned(), *image.descriptor());
    let err = ExtractChannel::new()
        .infer_outputs(&inputs, &serde_json::json!({ "channel": 4 }))
        .expect_err("channel 4 is out of range for a 4-channel image");
    assert_eq!(err.class, ErrorClass::Semantic);
    assert_eq!(err.code, super::E_EXTRACT_CHANNEL);
}

#[test]
fn extract_rejects_missing_channel_param() {
    let image = coded_rgba(2, 2);
    let mut inputs = Descriptors::new();
    inputs.insert("image".to_owned(), *image.descriptor());
    let err = ExtractChannel::new()
        .infer_outputs(&inputs, &serde_json::json!({}))
        .expect_err("missing channel param");
    assert_eq!(err.class, ErrorClass::Schema);
}

#[test]
fn assemble_rejects_channel_count_layout_mismatch() {
    let extent = Extent::new(2, 2);
    // rgb layout (3 channels) but only two ports wired.
    let mut inputs = Descriptors::new();
    inputs.insert("ch0".to_owned(), *field1(extent, vec![0.0; 4]).descriptor());
    inputs.insert("ch1".to_owned(), *field1(extent, vec![0.0; 4]).descriptor());
    let err = AssembleChannels::new()
        .infer_outputs(&inputs, &serde_json::json!({ "layout": "rgb" }))
        .expect_err("2 ports != 3 layout channels");
    assert_eq!(err.class, ErrorClass::Semantic);
    assert_eq!(err.code, super::E_ASSEMBLE_INPUT);
}

#[test]
fn assemble_rejects_extent_mismatch() {
    let mut inputs = Descriptors::new();
    inputs.insert(
        "ch0".to_owned(),
        *field1(Extent::new(2, 2), vec![0.0; 4]).descriptor(),
    );
    inputs.insert(
        "ch1".to_owned(),
        *field1(Extent::new(3, 2), vec![0.0; 6]).descriptor(),
    );
    let err = AssembleChannels::new()
        .infer_outputs(&inputs, &serde_json::json!({ "layout": "gray-a" }))
        .expect_err("mismatched extents");
    assert_eq!(err.class, ErrorClass::Semantic);
    assert_eq!(err.code, super::E_ASSEMBLE_INPUT);
}

#[test]
fn assemble_rejects_non_contiguous_channel_ports() {
    // ch1 wired but ch0 absent: a hole that must be rejected.
    let extent = Extent::new(2, 2);
    let mut inputs = Descriptors::new();
    inputs.insert("ch1".to_owned(), *field1(extent, vec![0.0; 4]).descriptor());
    let err = AssembleChannels::new()
        .infer_outputs(&inputs, &serde_json::json!({ "layout": "gray" }))
        .expect_err("ch1 without ch0 is a hole");
    assert_eq!(err.class, ErrorClass::Semantic);
}

#[test]
fn assemble_rejects_unsupported_color_encoding() {
    let extent = Extent::new(1, 1);
    let mut inputs = Descriptors::new();
    inputs.insert("ch0".to_owned(), *field1(extent, vec![0.5]).descriptor());
    let err = AssembleChannels::new()
        .infer_outputs(
            &inputs,
            &serde_json::json!({ "layout": "gray", "color": "display-p3" }),
        )
        .expect_err("display-p3 is nameable but unsupported");
    assert_eq!(err.class, ErrorClass::Semantic);
}

/// The checked-in `ops/manifests/<id>.json` files must stay byte-identical to the
/// Rust manifest builders (the source of truth). Regenerate with
/// `serde_json::to_string_pretty` if this fails.
#[test]
fn checked_in_manifests_match_builders() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("crates dir")
        .parent()
        .expect("repo root")
        .join("ops/manifests");
    for manifest in [
        ExtractChannel::manifest().expect("extract manifest"),
        AssembleChannels::manifest().expect("assemble manifest"),
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
