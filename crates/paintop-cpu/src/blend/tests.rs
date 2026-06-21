//! Verification suite for `composite.blend@1` (`OP_CATALOG` §7) — a restricted,
//! exactly-pinned blend mode set with opacity + mask:
//!
//! - **schema/contract**: the manifest validates, agrees with its contract, and
//!   its declarations gate clean; the checked-in manifest matches the builder;
//! - **per-mode formula fixtures**: each mode matches its documented `B(s, d)` at
//!   full coverage (`opacity = 1`, `mask = 1`);
//! - **opacity = 0 identity** and **mask = 0 identity**: each is `dst` bit-exact;
//! - **commutativity**: the commutative modes (add, multiply, screen, darken,
//!   lighten, difference) give the same result with `src`/`dst` swapped;
//! - **finite**: outputs are finite for finite inputs;
//! - **rejection**: an unknown mode, an out-of-range opacity, nonlinear /
//!   straight-alpha inputs, and extent mismatches are rejected.

use paintop_core::executor::{InputValues, OpImplementation, ResourceValue};
use paintop_ir::{
    AlphaRepresentation, AssertionStatus, ChannelLayout, ColorEncoding, ColorRange,
    CoordinateConvention, Descriptors, ErrorClass, Extent, ImageDescriptor, MaskDescriptor,
    MaskMeaning, OpContract, OutputRegions, Rect, ResourceDescriptor, ScalarType, SemanticRole,
    ValidRange, check_contract_consistency, verify_categories,
};

use super::{BLEND_OP_ID, Blend, E_BLEND_INPUT, E_BLEND_PARAM, E_BLEND_SHAPE};

/// A premultiplied linear RGBA image descriptor of side `n`.
fn rgba_descriptor(n: u32) -> ImageDescriptor {
    ImageDescriptor {
        extent: Extent::new(n, n),
        layout: ChannelLayout::Rgba,
        scalar: ScalarType::F32,
        color: ColorEncoding::LinearSrgb,
        range: ColorRange::SceneReferred,
        alpha: AlphaRepresentation::Premultiplied,
        coordinates: CoordinateConvention::PixelCenterUpperLeft,
        semantic: SemanticRole::Color,
    }
}

/// A single-pixel premultiplied RGBA image `[r, g, b, a]`.
fn pixel_image(rgba: [f32; 4]) -> ResourceValue {
    ResourceValue::new(
        ResourceDescriptor::Image(rgba_descriptor(1)),
        4,
        rgba.to_vec(),
    )
    .expect("rgba pixel")
}

/// A premultiplied RGBA image of side `n` whose every sample is `fill`.
fn rgba_image(n: u32, fill: f32) -> ResourceValue {
    let len = (n as usize) * (n as usize) * 4;
    ResourceValue::new(
        ResourceDescriptor::Image(rgba_descriptor(n)),
        4,
        vec![fill; len],
    )
    .expect("rgba image")
}

/// A constant coverage mask of side `n`.
fn const_mask(n: u32, coverage: f32) -> ResourceValue {
    let descriptor = ResourceDescriptor::Mask(MaskDescriptor {
        extent: Extent::new(n, n),
        scalar: ScalarType::F32,
        range: ValidRange::Bounded { min: 0.0, max: 1.0 },
        meaning: MaskMeaning::Coverage,
        coordinates: CoordinateConvention::PixelCenterUpperLeft,
    });
    ResourceValue::new(descriptor, 1, vec![coverage; (n as usize) * (n as usize)]).expect("mask")
}

/// Blend `src` onto `dst` through `mask` with the given params, returning the
/// produced image.
fn blend(
    src: ResourceValue,
    dst: ResourceValue,
    mask: ResourceValue,
    params: &serde_json::Value,
) -> ResourceValue {
    let mut inputs = InputValues::new();
    inputs.insert("src".to_owned(), src);
    inputs.insert("dst".to_owned(), dst);
    inputs.insert("mask".to_owned(), mask);
    let mut out = Blend::new()
        .compute(&inputs, params)
        .expect("blend computes");
    out.remove("image").expect("image produced")
}

/// A full-coverage 1×1 blend of two premultiplied pixels under `mode`.
fn blend_one(mode: &str, src: [f32; 4], dst: [f32; 4]) -> [f32; 4] {
    let out = blend(
        pixel_image(src),
        pixel_image(dst),
        const_mask(1, 1.0),
        &serde_json::json!({ "mode": mode, "opacity": 1.0 }),
    );
    let s = out.samples();
    [s[0], s[1], s[2], s[3]]
}

fn assert_close(got: [f32; 4], expected: [f32; 4], label: &str) {
    for c in 0..4 {
        assert!(
            (got[c] - expected[c]).abs() < 1e-6,
            "{label}: channel {c} got {} expected {}",
            got[c],
            expected[c],
        );
    }
}

#[test]
fn manifest_validates_and_agrees_with_contract() {
    let manifest = Blend::manifest().expect("blend manifest");
    manifest.validate().expect("manifest valid");
    check_contract_consistency(&manifest, &Blend::new()).expect("contract agrees");
    verify_categories(&manifest, &manifest.test.verification).expect("declarations gate clean");
    assert_eq!(manifest.id.to_string(), BLEND_OP_ID);
}

/// Each mode matches its documented per-channel formula at full coverage.
#[test]
fn per_mode_formulas_match() {
    // src = half-opaque warm, dst = three-quarter cool, both premultiplied-plausible.
    let s = [0.4_f32, 0.2, 0.1, 0.5];
    let d = [0.1_f32, 0.3, 0.6, 0.75];

    // normal/over: B = s + d·(1 − αs); αs = 0.5 so factor 0.5. Precomputed
    // literals: [0.4+0.05, 0.2+0.15, 0.1+0.3, 0.5+0.375].
    assert_close(
        blend_one("normal", s, d),
        [0.45, 0.35, 0.4, 0.875],
        "normal",
    );
    // add: B = s + d.
    assert_close(blend_one("add", s, d), [0.5, 0.5, 0.7, 1.25], "add");
    // subtract: B = d − s.
    assert_close(
        blend_one("subtract", s, d),
        [d[0] - s[0], d[1] - s[1], d[2] - s[2], d[3] - s[3]],
        "subtract",
    );
    // multiply: B = s·d.
    assert_close(
        blend_one("multiply", s, d),
        [s[0] * d[0], s[1] * d[1], s[2] * d[2], s[3] * d[3]],
        "multiply",
    );
    // screen: B = s + d − s·d (fused as s·(−d) + (s + d)).
    assert_close(
        blend_one("screen", s, d),
        [
            s[0].mul_add(-d[0], s[0] + d[0]),
            s[1].mul_add(-d[1], s[1] + d[1]),
            s[2].mul_add(-d[2], s[2] + d[2]),
            s[3].mul_add(-d[3], s[3] + d[3]),
        ],
        "screen",
    );
    // darken / lighten.
    assert_close(
        blend_one("darken", s, d),
        [
            s[0].min(d[0]),
            s[1].min(d[1]),
            s[2].min(d[2]),
            s[3].min(d[3]),
        ],
        "darken",
    );
    assert_close(
        blend_one("lighten", s, d),
        [
            s[0].max(d[0]),
            s[1].max(d[1]),
            s[2].max(d[2]),
            s[3].max(d[3]),
        ],
        "lighten",
    );
    // difference: B = |s − d|.
    assert_close(
        blend_one("difference", s, d),
        [
            (s[0] - d[0]).abs(),
            (s[1] - d[1]).abs(),
            (s[2] - d[2]).abs(),
            (s[3] - d[3]).abs(),
        ],
        "difference",
    );
}

/// `over` is an alias for `normal`.
#[test]
fn over_aliases_normal() {
    let s = [0.4_f32, 0.2, 0.1, 0.5];
    let d = [0.1_f32, 0.3, 0.6, 0.75];
    let over = blend_one("over", s, d);
    let normal = blend_one("normal", s, d);
    for c in 0..4 {
        assert_eq!(over[c].to_bits(), normal[c].to_bits(), "channel {c}");
    }
}

/// `opacity = 0` is the identity on `dst`, bit-exactly, for every mode.
#[test]
fn opacity_zero_is_identity() {
    let d = [0.1_f32, 0.3, 0.6, 0.75];
    for mode in [
        "normal",
        "add",
        "subtract",
        "multiply",
        "screen",
        "darken",
        "lighten",
        "difference",
    ] {
        let out = blend(
            pixel_image([0.9, 0.8, 0.7, 1.0]),
            pixel_image(d),
            const_mask(1, 1.0),
            &serde_json::json!({ "mode": mode, "opacity": 0.0 }),
        );
        let s = out.samples();
        for c in 0..4 {
            assert_eq!(
                s[c].to_bits(),
                d[c].to_bits(),
                "mode {mode} not identity at opacity 0"
            );
        }
    }
}

/// `mask = 0` is the identity on `dst`, bit-exactly.
#[test]
fn mask_zero_is_identity() {
    let d = [0.1_f32, 0.3, 0.6, 0.75];
    let out = blend(
        pixel_image([0.9, 0.8, 0.7, 1.0]),
        pixel_image(d),
        const_mask(1, 0.0),
        &serde_json::json!({ "mode": "multiply", "opacity": 1.0 }),
    );
    let s = out.samples();
    for c in 0..4 {
        assert_eq!(s[c].to_bits(), d[c].to_bits(), "not identity at mask 0");
    }
}

/// The commutative modes give the same result with src and dst swapped.
///
/// At full coverage (`opacity = mask = 1`) the output is exactly `B(src, dst)`
/// (the mix-with-dst envelope is the identity), so a full-coverage blend isolates
/// the per-channel blend function: a commutative `B` satisfies
/// `B(a, b) == B(b, a)`.
#[test]
fn commutative_modes_are_commutative() {
    let a = [0.4_f32, 0.2, 0.1, 0.5];
    let b = [0.1_f32, 0.3, 0.6, 0.75];
    for mode in [
        "add",
        "multiply",
        "screen",
        "darken",
        "lighten",
        "difference",
    ] {
        assert_close(blend_one(mode, a, b), blend_one(mode, b, a), mode);
    }
}

/// Outputs are finite for finite inputs across modes.
#[test]
fn outputs_are_finite() {
    let out = blend(
        rgba_image(4, 0.3),
        rgba_image(4, 0.7),
        const_mask(4, 0.5),
        &serde_json::json!({ "mode": "screen", "opacity": 0.5 }),
    );
    for &v in out.samples() {
        assert!(v.is_finite(), "non-finite blend output {v}");
    }
}

/// `infer_outputs` returns the dst descriptor and `required_inputs` is pointwise.
#[test]
fn contract_infers_dst_and_pointwise_roi() {
    let mut inputs = Descriptors::new();
    inputs.insert(
        "src".to_owned(),
        ResourceDescriptor::Image(rgba_descriptor(8)),
    );
    inputs.insert(
        "dst".to_owned(),
        ResourceDescriptor::Image(rgba_descriptor(8)),
    );
    inputs.insert(
        "mask".to_owned(),
        ResourceDescriptor::Mask(MaskDescriptor {
            extent: Extent::new(8, 8),
            scalar: ScalarType::F32,
            range: ValidRange::Bounded { min: 0.0, max: 1.0 },
            meaning: MaskMeaning::Coverage,
            coordinates: CoordinateConvention::PixelCenterUpperLeft,
        }),
    );
    let params = serde_json::json!({ "mode": "add" });
    let outputs = Blend::new().infer_outputs(&inputs, &params).expect("infer");
    let ResourceDescriptor::Image(d) = outputs.get("image").expect("image out") else {
        panic!("expected image");
    };
    assert_eq!(d.extent, Extent::new(8, 8));

    let region = Rect::new(1, 1, 2, 2);
    let mut requested = OutputRegions::new();
    requested.insert("image".to_owned(), region);
    let regions = Blend::new()
        .required_inputs(&requested, &inputs, &params)
        .expect("required inputs");
    for port in ["src", "dst", "mask"] {
        assert_eq!(regions.get(port), Some(&region), "port {port} ROI");
    }
}

/// The postcondition asserts the result stays premultiplied linear.
#[test]
fn postcondition_checks_premultiplied_linear() {
    let mut inputs = Descriptors::new();
    inputs.insert(
        "src".to_owned(),
        ResourceDescriptor::Image(rgba_descriptor(4)),
    );
    inputs.insert(
        "dst".to_owned(),
        ResourceDescriptor::Image(rgba_descriptor(4)),
    );
    inputs.insert(
        "mask".to_owned(),
        ResourceDescriptor::Mask(MaskDescriptor {
            extent: Extent::new(4, 4),
            scalar: ScalarType::F32,
            range: ValidRange::Bounded { min: 0.0, max: 1.0 },
            meaning: MaskMeaning::Coverage,
            coordinates: CoordinateConvention::PixelCenterUpperLeft,
        }),
    );
    let params = serde_json::json!({ "mode": "normal" });
    let outputs = Blend::new().infer_outputs(&inputs, &params).expect("infer");
    let results = Blend::new()
        .validate_postconditions(&outputs, &params)
        .expect("postconditions");
    assert!(
        results.iter().all(|r| r.status == AssertionStatus::Pass),
        "all postconditions pass: {results:?}"
    );
}

/// An unknown / not-yet-supported mode (overlay) is rejected.
#[test]
fn unknown_mode_is_rejected() {
    let mut inputs = InputValues::new();
    inputs.insert("src".to_owned(), rgba_image(4, 0.3));
    inputs.insert("dst".to_owned(), rgba_image(4, 0.7));
    inputs.insert("mask".to_owned(), const_mask(4, 1.0));
    let err = Blend::new()
        .compute(&inputs, &serde_json::json!({ "mode": "overlay" }))
        .expect_err("overlay must be rejected");
    assert_eq!(err.class, ErrorClass::Schema);
    assert_eq!(err.code, E_BLEND_PARAM);
}

/// An out-of-range opacity is rejected.
#[test]
fn out_of_range_opacity_is_rejected() {
    let mut inputs = InputValues::new();
    inputs.insert("src".to_owned(), rgba_image(4, 0.3));
    inputs.insert("dst".to_owned(), rgba_image(4, 0.7));
    inputs.insert("mask".to_owned(), const_mask(4, 1.0));
    let err = Blend::new()
        .compute(
            &inputs,
            &serde_json::json!({ "mode": "add", "opacity": 1.5 }),
        )
        .expect_err("opacity > 1 must be rejected");
    assert_eq!(err.class, ErrorClass::Schema);
    assert_eq!(err.code, E_BLEND_PARAM);
}

/// A nonlinear (srgb) input is rejected.
#[test]
fn srgb_input_is_rejected() {
    let mut srgb = rgba_descriptor(4);
    srgb.color = ColorEncoding::Srgb;
    let src = ResourceValue::new(ResourceDescriptor::Image(srgb), 4, vec![0.5; 4 * 4 * 4])
        .expect("srgb image");
    let mut inputs = InputValues::new();
    inputs.insert("src".to_owned(), src);
    inputs.insert("dst".to_owned(), rgba_image(4, 0.0));
    inputs.insert("mask".to_owned(), const_mask(4, 1.0));
    let err = Blend::new()
        .compute(&inputs, &serde_json::json!({ "mode": "normal" }))
        .expect_err("srgb must be rejected");
    assert_eq!(err.class, ErrorClass::Semantic);
    assert_eq!(err.code, E_BLEND_SHAPE);
}

/// An extent mismatch between the ports is rejected.
#[test]
fn extent_mismatch_is_rejected() {
    let mut inputs = InputValues::new();
    inputs.insert("src".to_owned(), rgba_image(8, 0.0));
    inputs.insert("dst".to_owned(), rgba_image(4, 0.5));
    inputs.insert("mask".to_owned(), const_mask(4, 1.0));
    let err = Blend::new()
        .compute(&inputs, &serde_json::json!({ "mode": "add" }))
        .expect_err("extent mismatch must be rejected");
    assert_eq!(err.class, ErrorClass::Semantic);
    assert_eq!(err.code, E_BLEND_SHAPE);
}

/// A missing required port is a reference error.
#[test]
fn missing_port_is_rejected() {
    let mut inputs = InputValues::new();
    inputs.insert("src".to_owned(), rgba_image(4, 0.0));
    inputs.insert("dst".to_owned(), rgba_image(4, 0.5));
    let err = Blend::new()
        .compute(&inputs, &serde_json::json!({ "mode": "add" }))
        .expect_err("missing mask must be rejected");
    assert_eq!(err.class, ErrorClass::Reference);
    assert_eq!(err.code, E_BLEND_INPUT);
}

/// The checked-in `ops/manifests/<id>.json` must match the Rust builder.
#[test]
fn checked_in_manifest_matches_builder() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("crates dir")
        .parent()
        .expect("repo root")
        .join("ops/manifests");
    let manifest = Blend::manifest().expect("blend manifest");
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
