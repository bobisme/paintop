//! Verification suite for `image.create@1` (`OP_CATALOG` §1, `plan.md` §8.3):
//!
//! - **schema/contract**: the manifest validates, agrees with its contract, gates
//!   clean, and the checked-in manifest stays in lockstep with the Rust builder;
//! - **analytic fixtures**: the created image has the exact requested
//!   extent/layout/color/range/semantic and every sample equals the fill;
//! - **property / determinism**: two runs of the same request are bit-identical;
//! - **rejection**: a wrong-length fill, a non-finite fill, and an out-of-range
//!   fill under a display-referred policy are rejected with typed errors, while a
//!   scene-referred image admits an out-of-`[0,1]` color fill.

use paintop_core::executor::{InputValues, OpImplementation};
use paintop_ir::{
    AlphaRepresentation, ChannelLayout, ColorEncoding, ColorRange, Descriptors, ErrorClass, Extent,
    OpContract, ResourceDescriptor, ScalarType, SemanticRole, check_contract_consistency,
    verify_categories,
};

use super::{CREATE_OP_ID, CreateImage};

fn create(params: &serde_json::Value) -> paintop_core::executor::ResourceValue {
    let mut out = CreateImage::new()
        .compute(&InputValues::new(), params)
        .expect("create computes");
    out.remove("image").expect("image produced")
}

fn create_err(params: &serde_json::Value) -> paintop_ir::Error {
    CreateImage::new()
        .infer_outputs(&Descriptors::new(), params)
        .expect_err("expected a rejection")
}

#[test]
fn manifest_validates_and_agrees_with_contract() {
    let manifest = CreateImage::manifest().expect("create manifest");
    manifest.validate().expect("create manifest valid");
    check_contract_consistency(&manifest, &CreateImage::new())
        .expect("create manifest agrees with contract");
    verify_categories(&manifest, &manifest.test.verification)
        .expect("create verification declarations gate clean");
    assert_eq!(manifest.id.to_string(), CREATE_OP_ID);
}

#[test]
fn created_image_has_exact_descriptor_and_fill() {
    let image = create(&serde_json::json!({
        "width": 3,
        "height": 2,
        "layout": "rgba",
        "color": "linear-srgb",
        "range": "display-referred",
        "alpha": "premultiplied",
        "semantic": "color",
        "fill": [0.1, 0.2, 0.3, 1.0]
    }));
    let ResourceDescriptor::Image(d) = image.descriptor() else {
        panic!("expected image");
    };
    assert_eq!(d.extent, Extent::new(3, 2));
    assert_eq!(d.layout, ChannelLayout::Rgba);
    assert_eq!(d.scalar, ScalarType::F32);
    assert_eq!(d.color, ColorEncoding::LinearSrgb);
    assert_eq!(d.range, ColorRange::DisplayReferred);
    assert_eq!(d.alpha, AlphaRepresentation::Premultiplied);
    assert_eq!(d.semantic, SemanticRole::Color);
    assert_eq!(image.channels(), 4);
    assert_eq!(image.samples().len(), 3 * 2 * 4);
    // Every pixel is exactly the fill.
    for pixel in image.samples().chunks_exact(4) {
        assert_eq!(pixel, [0.1_f32, 0.2, 0.3, 1.0].as_slice());
    }
}

#[test]
fn default_metadata_is_applied_when_omitted() {
    let image = create(&serde_json::json!({
        "width": 1,
        "height": 1,
        "layout": "gray",
        "fill": [0.5]
    }));
    let ResourceDescriptor::Image(d) = image.descriptor() else {
        panic!("expected image");
    };
    // Defaults: srgb / display-referred / straight / color.
    assert_eq!(d.color, ColorEncoding::Srgb);
    assert_eq!(d.range, ColorRange::DisplayReferred);
    assert_eq!(d.alpha, AlphaRepresentation::Straight);
    assert_eq!(d.semantic, SemanticRole::Color);
}

#[test]
fn creation_is_deterministic() {
    let params = serde_json::json!({
        "width": 4, "height": 4, "layout": "rgb", "fill": [0.25, 0.5, 0.75]
    });
    let a = create(&params);
    let b = create(&params);
    assert_eq!(a.samples(), b.samples());
    assert_eq!(a.descriptor(), b.descriptor());
}

#[test]
fn scene_referred_admits_out_of_unit_color_fill() {
    // A scene-referred (HDR) image may carry color above 1.0; only finiteness is
    // required. Alpha still must stay in [0, 1].
    let image = create(&serde_json::json!({
        "width": 1, "height": 1, "layout": "rgba",
        "range": "scene-referred",
        "fill": [4.0, 2.0, 0.0, 1.0]
    }));
    let pixel: Vec<f32> = image.samples().to_vec();
    assert_eq!(pixel, vec![4.0, 2.0, 0.0, 1.0]);
}

// --- rejection -------------------------------------------------------------

#[test]
fn rejects_wrong_length_fill() {
    let err = create_err(&serde_json::json!({
        "width": 2, "height": 2, "layout": "rgb", "fill": [0.5, 0.5]
    }));
    assert_eq!(err.class, ErrorClass::Schema);
    assert_eq!(err.code, super::E_CREATE_FILL);
}

#[test]
fn rejects_non_finite_fill() {
    // JSON cannot carry NaN, but a string is a non-number and is rejected; an
    // explicit very-large-but-finite value passes, so use a non-number to force
    // the finiteness/number guard.
    let err = create_err(&serde_json::json!({
        "width": 1, "height": 1, "layout": "gray", "fill": ["x"]
    }));
    assert_eq!(err.class, ErrorClass::Schema);
    assert_eq!(err.code, super::E_CREATE_FILL);
}

#[test]
fn rejects_out_of_range_fill_under_display_referred_policy() {
    // Display-referred bounds color channels to [0, 1]; 1.5 is rejected, not
    // clamped (plan.md §8.3: clamping is never implicit).
    let err = create_err(&serde_json::json!({
        "width": 1, "height": 1, "layout": "rgb",
        "range": "display-referred",
        "fill": [1.5, 0.0, 0.0]
    }));
    assert_eq!(err.class, ErrorClass::Policy);
    assert_eq!(err.code, super::E_CREATE_FILL);
}

#[test]
fn rejects_out_of_range_alpha_even_when_scene_referred() {
    // Alpha is coverage in [0, 1] regardless of the color range.
    let err = create_err(&serde_json::json!({
        "width": 1, "height": 1, "layout": "rgba",
        "range": "scene-referred",
        "fill": [4.0, 0.0, 0.0, 2.0]
    }));
    assert_eq!(err.class, ErrorClass::Policy);
    assert_eq!(err.code, super::E_CREATE_FILL);
}

#[test]
fn rejects_unsupported_color_encoding() {
    let err = create_err(&serde_json::json!({
        "width": 1, "height": 1, "layout": "gray", "color": "icc", "fill": [0.5]
    }));
    assert_eq!(err.class, ErrorClass::Semantic);
}

#[test]
fn rejects_missing_required_params() {
    for params in [
        serde_json::json!({ "height": 1, "layout": "gray", "fill": [0.5] }),
        serde_json::json!({ "width": 1, "layout": "gray", "fill": [0.5] }),
        serde_json::json!({ "width": 1, "height": 1, "fill": [0.5] }),
        serde_json::json!({ "width": 1, "height": 1, "layout": "gray" }),
    ] {
        let err = create_err(&params);
        assert_eq!(err.class, ErrorClass::Schema, "params: {params}");
    }
}

/// The checked-in `ops/manifests/image.create@1.json` must stay byte-identical to
/// the Rust manifest builder (the source of truth).
#[test]
fn checked_in_manifest_matches_builder() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("crates dir")
        .parent()
        .expect("repo root")
        .join("ops/manifests");
    let manifest = CreateImage::manifest().expect("create manifest");
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
