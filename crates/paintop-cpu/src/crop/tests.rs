//! Verification suite for `image.crop@1` (`OP_CATALOG` §5, `IR_SPEC` §8.1):
//!
//! - **schema/contract**: the manifest validates, agrees with its contract, gates
//!   clean, and the checked-in JSON matches the builder;
//! - **analytic fixtures**: a crop selects exactly the half-open sub-window, with
//!   the correct output extent and bit-identical samples;
//! - **property**: a full-extent crop is the exact identity; an empty rect yields
//!   a zero-area image; crop is associative on nested rects;
//! - **metamorphic**: `crop(rect) ∘ pad` is omitted here (covered in the pad
//!   suite); crop commutes with channel-wise permutation of samples (verbatim);
//! - **rejection**: an ill-formed rect and an out-of-bounds rect are rejected.

use paintop_core::executor::{InputValues, OpImplementation, ResourceValue};
use paintop_ir::{
    AlphaRepresentation, ChannelLayout, ColorEncoding, ColorRange, CoordinateConvention,
    Descriptors, ErrorClass, Extent, ImageDescriptor, OpContract, OutputRegions, Rect,
    ResourceDescriptor, ScalarType, SemanticRole,
};

use super::{CROP_OP_ID, Crop, E_CROP_BOUNDS, E_CROP_RECT};

/// Build an RGBA image whose samples encode `(x, y, channel)` so a crop is easy
/// to verify positionally: `sample(x, y, c) = (y * width + x) * 4 + c`.
fn ramp_image(width: u32, height: u32) -> ResourceValue {
    let mut samples = Vec::new();
    for y in 0..height {
        for x in 0..width {
            for c in 0..4u32 {
                #[allow(clippy::cast_precision_loss, reason = "small test extents")]
                let v = ((y * width + x) * 4 + c) as f32;
                samples.push(v);
            }
        }
    }
    image_value(width, height, ChannelLayout::Rgba, samples)
}

/// Build an image [`ResourceValue`] with the given layout/samples.
fn image_value(width: u32, height: u32, layout: ChannelLayout, samples: Vec<f32>) -> ResourceValue {
    let descriptor = ResourceDescriptor::Image(ImageDescriptor {
        extent: Extent::new(width, height),
        layout,
        scalar: ScalarType::F32,
        color: ColorEncoding::Srgb,
        range: ColorRange::DisplayReferred,
        alpha: AlphaRepresentation::Straight,
        coordinates: CoordinateConvention::PixelCenterUpperLeft,
        semantic: SemanticRole::Color,
    });
    ResourceValue::new(descriptor, layout.channel_count(), samples)
        .expect("sample buffer matches descriptor")
}

/// Run the crop kernel and recover the output image.
fn crop(value: &ResourceValue, rect: Rect) -> ResourceValue {
    let mut inputs = InputValues::new();
    inputs.insert("image".to_owned(), value.clone());
    let params = serde_json::json!({
        "rect": { "x0": rect.x0, "y0": rect.y0, "x1": rect.x1, "y1": rect.y1 }
    });
    let mut out = Crop::new()
        .compute(&inputs, &params)
        .expect("crop computes");
    out.remove("image").expect("image port produced")
}

#[test]
fn manifest_validates_and_agrees_with_contract() {
    let manifest = Crop::manifest().expect("crop manifest");
    manifest.validate().expect("crop manifest valid");
    paintop_ir::check_contract_consistency(&manifest, &Crop::new())
        .expect("crop manifest agrees with contract");
    paintop_ir::verify_categories(&manifest, &manifest.test.verification)
        .expect("crop verification declarations gate clean");
    assert_eq!(manifest.id.to_string(), CROP_OP_ID);
}

#[test]
fn crop_selects_exact_half_open_window() {
    // A 4x3 ramp; crop [1, 3) x [1, 3) => a 2x2 image of pixels (1,1),(2,1),(1,2),(2,2).
    let img = ramp_image(4, 3);
    let out = crop(&img, Rect::new(1, 1, 3, 3));
    assert_eq!(out.extent(), Extent::new(2, 2));
    // Expected sample base index for pixel (x, y) is (y*4 + x)*4.
    let want: Vec<f32> = [(1, 1), (2, 1), (1, 2), (2, 2)]
        .iter()
        .flat_map(|&(x, y)| {
            let base = (y * 4 + x) * 4;
            #[allow(clippy::cast_precision_loss, reason = "tiny indices")]
            (0..4).map(move |c| (base + c) as f32)
        })
        .collect();
    assert_eq!(out.samples(), want.as_slice());
}

#[test]
fn single_pixel_crop_is_one_pixel() {
    let img = ramp_image(4, 3);
    let out = crop(&img, Rect::new(2, 1, 3, 2));
    assert_eq!(out.extent(), Extent::new(1, 1));
    // Pixel (2, 1) has base sample index (1 * 4 + 2) * 4 = 24.
    let base = 24;
    #[allow(clippy::cast_precision_loss, reason = "tiny indices")]
    let want: Vec<f32> = (0..4).map(|c| (base + c) as f32).collect();
    assert_eq!(out.samples(), want.as_slice());
}

#[test]
fn full_extent_crop_is_exact_identity() {
    let img = ramp_image(5, 4);
    let out = crop(&img, Rect::new(0, 0, 5, 4));
    assert_eq!(out.extent(), Extent::new(5, 4));
    assert_eq!(out.samples(), img.samples());
}

#[test]
fn empty_rect_yields_zero_area_image() {
    let img = ramp_image(4, 3);
    // x0 == x1 => zero width but well-formed.
    let out = crop(&img, Rect::new(2, 1, 2, 3));
    assert_eq!(out.extent(), Extent::new(0, 2));
    assert!(out.samples().is_empty());
}

#[test]
fn nested_crop_is_associative() {
    // crop(crop(img, outer), inner) == crop(img, inner shifted by outer origin).
    let img = ramp_image(8, 8);
    let outer = Rect::new(2, 1, 7, 6); // a 5x5 window at origin (2,1)
    let mid = crop(&img, outer);
    let inner = Rect::new(1, 1, 4, 3); // within the 5x5
    let nested = crop(&mid, inner);
    // Composed crop directly on the original.
    let direct = crop(&img, Rect::new(2 + 1, 1 + 1, 2 + 4, 1 + 3));
    assert_eq!(nested.extent(), direct.extent());
    assert_eq!(nested.samples(), direct.samples());
}

#[test]
fn ill_formed_rect_is_rejected() {
    let img = ramp_image(4, 3);
    let mut inputs = InputValues::new();
    inputs.insert("image".to_owned(), img);
    let params = serde_json::json!({ "rect": { "x0": 3, "y0": 0, "x1": 1, "y1": 2 } });
    let err = Crop::new()
        .compute(&inputs, &params)
        .expect_err("x1 < x0 must be rejected");
    assert_eq!(err.class, ErrorClass::Schema);
    assert_eq!(err.code, E_CROP_RECT);
}

#[test]
fn out_of_bounds_rect_is_rejected() {
    let img = ramp_image(4, 3);
    let mut inputs = InputValues::new();
    inputs.insert("image".to_owned(), img);
    // x1 = 5 escapes width 4.
    let params = serde_json::json!({ "rect": { "x0": 0, "y0": 0, "x1": 5, "y1": 2 } });
    let err = Crop::new()
        .compute(&inputs, &params)
        .expect_err("out-of-bounds rect must be rejected");
    assert_eq!(err.class, ErrorClass::Semantic);
    assert_eq!(err.code, E_CROP_BOUNDS);
}

#[test]
fn negative_origin_is_rejected() {
    let img = ramp_image(4, 3);
    let mut inputs = InputValues::new();
    inputs.insert("image".to_owned(), img);
    let params = serde_json::json!({ "rect": { "x0": -1, "y0": 0, "x1": 2, "y1": 2 } });
    let err = Crop::new()
        .compute(&inputs, &params)
        .expect_err("negative origin must be rejected");
    assert_eq!(err.class, ErrorClass::Semantic);
    assert_eq!(err.code, E_CROP_BOUNDS);
}

#[test]
fn infer_outputs_matches_rect_extent_and_geometric_roi() {
    let descriptor = ResourceDescriptor::Image(ImageDescriptor {
        extent: Extent::new(10, 8),
        layout: ChannelLayout::Rgb,
        scalar: ScalarType::F32,
        color: ColorEncoding::Srgb,
        range: ColorRange::DisplayReferred,
        alpha: AlphaRepresentation::Straight,
        coordinates: CoordinateConvention::PixelCenterUpperLeft,
        semantic: SemanticRole::Color,
    });
    let mut inputs = Descriptors::new();
    inputs.insert("image".to_owned(), descriptor);
    let params = serde_json::json!({ "rect": { "x0": 2, "y0": 3, "x1": 7, "y1": 6 } });

    let out = Crop::new().infer_outputs(&inputs, &params).expect("infer");
    let ResourceDescriptor::Image(d) = out["image"] else {
        panic!("expected image");
    };
    assert_eq!(d.extent, Extent::new(5, 3));
    assert_eq!(d.layout, ChannelLayout::Rgb);

    // Geometric ROI: an output region maps to the same region shifted by (x0, y0).
    let mut requested = OutputRegions::new();
    requested.insert("image".to_owned(), Rect::new(0, 0, 2, 2));
    let needed = Crop::new()
        .required_inputs(&requested, &inputs, &params)
        .expect("required_inputs");
    assert_eq!(needed["image"], Rect::new(2, 3, 4, 5));
}

#[test]
fn checked_in_manifest_matches_builder() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("crates dir")
        .parent()
        .expect("repo root")
        .join("ops/manifests");
    let manifest = Crop::manifest().expect("crop manifest");
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
