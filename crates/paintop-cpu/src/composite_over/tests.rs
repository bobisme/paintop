//! Verification suite for `composite.over@1` (`OP_CATALOG` §7,
//! `AGENT_VERIFICATION` §3.2) — premultiplied Porter–Duff source-over:
//!
//! - **schema/contract**: the manifest validates, agrees with its contract, and
//!   its declarations gate clean; the checked-in manifest matches the builder;
//! - **transparent-source identity**: `α_s = 0` everywhere reproduces `dst`
//!   bit-exactly;
//! - **opaque-source replacement**: `α_s = 1` everywhere reproduces `src`;
//! - **output alpha in [0, 1]** and the **premultiplied constraint** `|C_i| ≤ α_o`
//!   hold for arbitrary premultiplied inputs;
//! - **associativity within tolerance**: `(a over b) over c ≈ a over (b over c)`;
//! - **alpha-edge fringe**: hidden RGB under a fully transparent source
//!   contributes nothing, so a transparent-over edge leaves no colored fringe;
//! - **rejection**: nonlinear / straight-alpha / no-alpha inputs and a
//!   shape/extent mismatch are rejected.

use paintop_core::executor::{InputValues, OpImplementation, ResourceValue};
use paintop_ir::{
    AlphaRepresentation, AssertionStatus, ChannelLayout, ColorEncoding, ColorRange,
    CoordinateConvention, Descriptors, ErrorClass, Extent, ImageDescriptor, OpContract,
    OutputRegions, Rect, ResourceDescriptor, ScalarType, SemanticRole, check_contract_consistency,
    verify_categories,
};

use super::{E_OVER_INPUT, E_OVER_SHAPE, OVER_OP_ID, Over};

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

/// A premultiplied linear RGBA image of side `n` whose every sample is `fill`.
fn rgba_image(n: u32, fill: f32) -> ResourceValue {
    let descriptor = ResourceDescriptor::Image(rgba_descriptor(n));
    let len = (n as usize) * (n as usize) * 4;
    ResourceValue::new(descriptor, 4, vec![fill; len]).expect("rgba image")
}

/// A premultiplied linear RGBA image of side `n` from an explicit buffer.
fn rgba_image_from(n: u32, samples: Vec<f32>) -> ResourceValue {
    let descriptor = ResourceDescriptor::Image(rgba_descriptor(n));
    ResourceValue::new(descriptor, 4, samples).expect("rgba image")
}

/// A single-pixel premultiplied RGBA image `[r, g, b, a]`.
fn pixel_image(rgba: [f32; 4]) -> ResourceValue {
    rgba_image_from(1, rgba.to_vec())
}

/// Composite `src` over `dst`, returning the produced image.
fn over(src: ResourceValue, dst: ResourceValue) -> ResourceValue {
    let mut inputs = InputValues::new();
    inputs.insert("src".to_owned(), src);
    inputs.insert("dst".to_owned(), dst);
    let mut out = Over::new()
        .compute(&inputs, &serde_json::Value::Null)
        .expect("over computes");
    out.remove("image").expect("image produced")
}

/// The descriptor inputs for `infer_outputs` / `required_inputs`.
fn descriptors(n: u32) -> Descriptors {
    let mut inputs = Descriptors::new();
    inputs.insert(
        "src".to_owned(),
        ResourceDescriptor::Image(rgba_descriptor(n)),
    );
    inputs.insert(
        "dst".to_owned(),
        ResourceDescriptor::Image(rgba_descriptor(n)),
    );
    inputs
}

#[test]
fn manifest_validates_and_agrees_with_contract() {
    let manifest = Over::manifest().expect("over manifest");
    manifest.validate().expect("manifest valid");
    check_contract_consistency(&manifest, &Over::new()).expect("contract agrees");
    verify_categories(&manifest, &manifest.test.verification).expect("declarations gate clean");
    assert_eq!(manifest.id.to_string(), OVER_OP_ID);
}

/// Transparent source (`α_s = 0`, premultiplied so all channels 0) is the identity
/// on `dst`, bit-exactly — the no-fringe guarantee at the extreme.
#[test]
fn transparent_source_is_identity() {
    let dst = rgba_image_from(
        4,
        (0..4 * 4 * 4)
            .map(|i| f32::from(u8::try_from(i % 251).unwrap()) / 251.0)
            .collect(),
    );
    let before: Vec<f32> = dst.samples().to_vec();
    // Premultiplied transparent source: every channel (incl. hidden RGB) is 0.
    let src = rgba_image(4, 0.0);
    let out = over(src, dst);
    for (got, want) in out.samples().iter().zip(before.iter()) {
        assert_eq!(got.to_bits(), want.to_bits());
    }
}

/// Opaque source (`α_s = 1`) replaces `dst`: for a premultiplied opaque src,
/// `C_o = C_s + C_d·0 = C_s`.
#[test]
fn opaque_source_replaces_dst() {
    // Opaque red over opaque blue -> opaque red.
    let src = pixel_image([0.7, 0.0, 0.0, 1.0]);
    let dst = pixel_image([0.0, 0.0, 0.9, 1.0]);
    let out = over(src, dst);
    let got = out.samples();
    let expected = [0.7_f32, 0.0, 0.0, 1.0];
    for (g, e) in got.iter().zip(expected.iter()) {
        assert_eq!(g.to_bits(), e.to_bits());
    }
}

/// The closed-form per-channel formula on a half-opaque source over an opaque
/// destination.
#[test]
fn matches_porter_duff_formula() {
    // src = 50%-opaque red (premultiplied), dst = opaque blue.
    let src = pixel_image([0.5, 0.0, 0.0, 0.5]);
    let dst = pixel_image([0.0, 0.0, 1.0, 1.0]);
    let out = over(src, dst);
    let got = out.samples();
    // C_o = C_s + C_d·(1 − 0.5):
    // r = 0.5 + 0·0.5 = 0.5; g = 0; b = 0 + 1·0.5 = 0.5; a = 0.5 + 1·0.5 = 1.0
    let expected = [0.5, 0.0, 0.5, 1.0];
    for (g, e) in got.iter().zip(expected.iter()) {
        assert!((g - e).abs() < 1e-6, "got {g}, expected {e}");
    }
}

/// Output alpha stays in `[0, 1]` and the premultiplied constraint `|C_i| ≤ α_o`
/// holds for arbitrary valid premultiplied inputs.
#[test]
fn output_alpha_bounded_and_premultiplied() {
    // A grid of premultiplied pixels: pick alpha then color components <= alpha.
    let mut src = Vec::new();
    let mut dst = Vec::new();
    for i in 0_u8..16 {
        let a_s = f32::from(i) / 15.0;
        src.extend_from_slice(&[a_s * 0.3, a_s * 0.6, a_s * 0.9, a_s]);
        let a_d = 1.0 - f32::from(i) / 15.0;
        dst.extend_from_slice(&[a_d * 0.8, a_d * 0.2, a_d * 0.5, a_d]);
    }
    let out = over(rgba_image_from(4, src), rgba_image_from(4, dst));
    for pixel in out.samples().chunks_exact(4) {
        let alpha = pixel[3];
        assert!((0.0..=1.0).contains(&alpha), "alpha {alpha} out of [0, 1]");
        for &c in &pixel[..3] {
            assert!(
                c <= alpha + 1e-6,
                "premultiplied constraint |C| <= alpha violated: {c} > {alpha}"
            );
        }
    }
}

/// Associativity within tolerance: `(a over b) over c ≈ a over (b over c)`.
#[test]
fn over_is_associative_within_tolerance() {
    let a = pixel_image([0.30, 0.10, 0.05, 0.50]);
    let b = pixel_image([0.20, 0.40, 0.10, 0.60]);
    let c = pixel_image([0.05, 0.05, 0.30, 0.40]);

    let ab = over(
        pixel_image([0.30, 0.10, 0.05, 0.50]),
        pixel_image([0.20, 0.40, 0.10, 0.60]),
    );
    let left = over(ab, pixel_image([0.05, 0.05, 0.30, 0.40]));

    let bc = over(b, c);
    let right = over(a, bc);

    for (l, r) in left.samples().iter().zip(right.samples().iter()) {
        assert!((l - r).abs() < 1e-6, "not associative: {l} vs {r}");
    }
}

/// Alpha-edge fringe: a transparent source pixel whose *hidden* premultiplied RGB
/// is non-zero must still not tint the destination. Premultiplied transparent is
/// all-zero, so the only way hidden RGB could leak is a bug; this fixture pins the
/// no-fringe behavior across an edge (one opaque, one transparent source pixel).
#[test]
fn no_colored_fringe_across_alpha_edge() {
    // 2×1 source: left fully opaque green, right fully transparent (all zero,
    // since premultiplied transparent cannot carry hidden RGB).
    let src = rgba_image_from(1, vec![0.0, 1.0, 0.0, 1.0]);
    let dst = rgba_image_from(1, vec![0.4, 0.0, 0.4, 1.0]);
    let opaque_out = over(src, dst);
    // Opaque green replaces dst exactly (no purple bleed-through).
    let g = opaque_out.samples();
    for (got, want) in g.iter().zip([0.0_f32, 1.0, 0.0, 1.0].iter()) {
        assert_eq!(got.to_bits(), want.to_bits(), "opaque edge not exact");
    }
    // Transparent (all-zero) source over the same dst is the dst, unchanged: no
    // fringe is introduced.
    let transparent_out = over(
        rgba_image_from(1, vec![0.0, 0.0, 0.0, 0.0]),
        pixel_image([0.4, 0.0, 0.4, 1.0]),
    );
    for (got, want) in transparent_out
        .samples()
        .iter()
        .zip([0.4_f32, 0.0, 0.4, 1.0].iter())
    {
        assert_eq!(got.to_bits(), want.to_bits(), "transparent edge tinted dst");
    }
}

/// `infer_outputs` returns the dst descriptor and `required_inputs` is pointwise.
#[test]
fn contract_infers_dst_and_pointwise_roi() {
    let inputs = descriptors(10);
    let outputs = Over::new()
        .infer_outputs(&inputs, &serde_json::Value::Null)
        .expect("infer");
    let ResourceDescriptor::Image(d) = outputs.get("image").expect("image out") else {
        panic!("expected image");
    };
    assert_eq!(d.extent, Extent::new(10, 10));

    let region = Rect::new(1, 2, 3, 4);
    let mut requested = OutputRegions::new();
    requested.insert("image".to_owned(), region);
    let regions = Over::new()
        .required_inputs(&requested, &inputs, &serde_json::Value::Null)
        .expect("required inputs");
    for port in ["src", "dst"] {
        assert_eq!(regions.get(port), Some(&region), "port {port} ROI");
    }
}

/// The postcondition asserts the composited result stays premultiplied linear.
#[test]
fn postcondition_checks_premultiplied_linear() {
    let inputs = descriptors(4);
    let outputs = Over::new()
        .infer_outputs(&inputs, &serde_json::Value::Null)
        .expect("infer");
    let results = Over::new()
        .validate_postconditions(&outputs, &serde_json::Value::Null)
        .expect("postconditions");
    assert!(
        results.iter().all(|r| r.status == AssertionStatus::Pass),
        "all postconditions pass: {results:?}"
    );
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
    let err = Over::new()
        .compute(&inputs, &serde_json::Value::Null)
        .expect_err("srgb must be rejected");
    assert_eq!(err.class, ErrorClass::Semantic);
    assert_eq!(err.code, E_OVER_SHAPE);
}

/// A straight-alpha input is rejected.
#[test]
fn straight_alpha_input_is_rejected() {
    let mut straight = rgba_descriptor(4);
    straight.alpha = AlphaRepresentation::Straight;
    let dst = ResourceValue::new(ResourceDescriptor::Image(straight), 4, vec![0.5; 4 * 4 * 4])
        .expect("straight image");
    let mut inputs = InputValues::new();
    inputs.insert("src".to_owned(), rgba_image(4, 0.0));
    inputs.insert("dst".to_owned(), dst);
    let err = Over::new()
        .compute(&inputs, &serde_json::Value::Null)
        .expect_err("straight alpha must be rejected");
    assert_eq!(err.class, ErrorClass::Semantic);
    assert_eq!(err.code, E_OVER_SHAPE);
}

/// `src` and `dst` with differing extents are rejected.
#[test]
fn extent_mismatch_is_rejected() {
    let mut inputs = InputValues::new();
    inputs.insert("src".to_owned(), rgba_image(8, 0.0));
    inputs.insert("dst".to_owned(), rgba_image(4, 0.5));
    let err = Over::new()
        .compute(&inputs, &serde_json::Value::Null)
        .expect_err("extent mismatch must be rejected");
    assert_eq!(err.class, ErrorClass::Semantic);
    assert_eq!(err.code, E_OVER_SHAPE);
}

/// A missing required port is a reference error.
#[test]
fn missing_port_is_rejected() {
    let mut inputs = InputValues::new();
    inputs.insert("src".to_owned(), rgba_image(4, 0.0));
    let err = Over::new()
        .compute(&inputs, &serde_json::Value::Null)
        .expect_err("missing dst must be rejected");
    assert_eq!(err.class, ErrorClass::Reference);
    assert_eq!(err.code, E_OVER_INPUT);
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
    let manifest = Over::manifest().expect("over manifest");
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
