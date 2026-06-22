//! Tests for `repair.patch_synthesize@1`: outside-hole identity, anchored
//! gather, a repeated-texture fill fixture, and shape validation.
#![allow(
    clippy::cast_precision_loss,
    reason = "test fixtures use small exact integer samples and coordinates"
)]

use paintop_core::executor::{InputValues, OpImplementation, ResourceValue};
use paintop_ir::{
    AlphaRepresentation, ChannelLayout, ColorEncoding, ColorRange, CoordinateConvention, Extent,
    ImageDescriptor, MaskDescriptor, MaskMeaning, PatchFieldDescriptor, ResourceDescriptor,
    ScalarType, SemanticRole, ValidRange,
};

use super::{PATCH_SYNTHESIZE_OP_ID, PatchSynthesize};

fn gray_image(samples: Vec<f32>, w: u32, h: u32) -> ResourceValue {
    let descriptor = ResourceDescriptor::Image(ImageDescriptor {
        extent: Extent::new(w, h),
        layout: ChannelLayout::Gray,
        scalar: ScalarType::F32,
        color: ColorEncoding::LinearSrgb,
        range: ColorRange::SceneReferred,
        alpha: AlphaRepresentation::Premultiplied,
        coordinates: CoordinateConvention::PixelCenterUpperLeft,
        semantic: SemanticRole::Color,
    });
    ResourceValue::new(descriptor, 1, samples).expect("gray image")
}

fn mask(samples: Vec<f32>, w: u32, h: u32) -> ResourceValue {
    let descriptor = ResourceDescriptor::Mask(MaskDescriptor {
        extent: Extent::new(w, h),
        scalar: ScalarType::F32,
        range: ValidRange::Bounded { min: 0.0, max: 1.0 },
        meaning: MaskMeaning::Coverage,
        coordinates: CoordinateConvention::PixelCenterUpperLeft,
    });
    ResourceValue::new(descriptor, 1, samples).expect("mask")
}

/// A field over a `tw x th` target into a `sw x sh` source, packed
/// `(src_x, src_y, cost)` per pixel from a list of `(sx, sy)` anchors.
fn field(anchors: &[(u32, u32)], tw: u32, th: u32, sw: u32, sh: u32) -> ResourceValue {
    let descriptor = PatchFieldDescriptor {
        target_extent: Extent::new(tw, th),
        source_extent: Extent::new(sw, sh),
        radius: 0,
        coordinates: CoordinateConvention::PixelCenterUpperLeft,
    };
    let mut samples = Vec::with_capacity(anchors.len() * 3);
    for &(sx, sy) in anchors {
        samples.push(sx as f32);
        samples.push(sy as f32);
        samples.push(0.0);
    }
    ResourceValue::patch_field(descriptor, samples).expect("patch field")
}

fn compute(inputs: &InputValues) -> Vec<f32> {
    PatchSynthesize::new()
        .compute(inputs, &serde_json::json!({}))
        .expect("compute")
        .get("image")
        .expect("image output")
        .samples()
        .to_vec()
}

#[test]
fn manifest_declares_image_output_and_four_inputs() {
    let m = PatchSynthesize::manifest().expect("manifest");
    assert_eq!(m.id.to_string(), PATCH_SYNTHESIZE_OP_ID);
    assert_eq!(m.outputs.len(), 1);
    assert_eq!(m.outputs[0].name, "image");
    let in_names: Vec<&str> = m.inputs.iter().map(|i| i.name.as_str()).collect();
    assert_eq!(in_names, vec!["source", "target", "field", "hole"]);
}

#[test]
fn outside_hole_pixels_are_target_identity() {
    // 2x2 source and target. Hole only at top-left; the rest must equal target.
    let src = vec![100.0, 101.0, 102.0, 103.0];
    let tgt = vec![10.0, 11.0, 12.0, 13.0];
    let hole = vec![1.0, 0.0, 0.0, 0.0];
    // Field anchor for the hole pixel points at source (1,1) = 103.
    let anchors = [(1, 1), (0, 0), (0, 0), (0, 0)];
    let mut inputs = InputValues::new();
    inputs.insert("source".to_owned(), gray_image(src, 2, 2));
    inputs.insert("target".to_owned(), gray_image(tgt, 2, 2));
    inputs.insert("field".to_owned(), field(&anchors, 2, 2, 2, 2));
    inputs.insert("hole".to_owned(), mask(hole, 2, 2));
    let out = compute(&inputs);
    // Top-left filled from source (1,1)=103; the other three are target identity.
    assert_eq!(out, vec![103.0, 11.0, 12.0, 13.0]);
}

#[test]
fn hole_pixels_gather_from_the_field_anchor() {
    // A 3x1 source [7, 8, 9]; a 3x1 target all filled, each anchored to a
    // distinct source pixel in reverse.
    let src = vec![7.0, 8.0, 9.0];
    let tgt = vec![0.0, 0.0, 0.0];
    let hole = vec![1.0, 1.0, 1.0];
    let anchors = [(2, 0), (1, 0), (0, 0)];
    let mut inputs = InputValues::new();
    inputs.insert("source".to_owned(), gray_image(src, 3, 1));
    inputs.insert("target".to_owned(), gray_image(tgt, 3, 1));
    inputs.insert("field".to_owned(), field(&anchors, 3, 1, 3, 1));
    inputs.insert("hole".to_owned(), mask(hole, 3, 1));
    let out = compute(&inputs);
    assert_eq!(out, vec![9.0, 8.0, 7.0]);
}

#[test]
fn repeated_texture_fill_is_coherent() {
    // A 2x2 source with a repeated value 5 everywhere; whatever the field
    // anchors are, the fill is all 5 inside the hole.
    let src = vec![5.0, 5.0, 5.0, 5.0];
    let tgt = vec![0.0, 0.0, 0.0, 0.0];
    let hole = vec![1.0, 1.0, 1.0, 1.0];
    let anchors = [(0, 0), (1, 0), (0, 1), (1, 1)];
    let mut inputs = InputValues::new();
    inputs.insert("source".to_owned(), gray_image(src, 2, 2));
    inputs.insert("target".to_owned(), gray_image(tgt, 2, 2));
    inputs.insert("field".to_owned(), field(&anchors, 2, 2, 2, 2));
    inputs.insert("hole".to_owned(), mask(hole, 2, 2));
    let out = compute(&inputs);
    assert_eq!(out, vec![5.0; 4]);
}

#[test]
fn determinism_bit_identical_reruns() {
    let src = vec![1.0, 2.0, 3.0, 4.0];
    let tgt = vec![9.0, 9.0, 9.0, 9.0];
    let hole = vec![1.0, 0.0, 1.0, 0.0];
    let anchors = [(1, 1), (0, 0), (0, 1), (0, 0)];
    let mk = || {
        let mut inputs = InputValues::new();
        inputs.insert("source".to_owned(), gray_image(src.clone(), 2, 2));
        inputs.insert("target".to_owned(), gray_image(tgt.clone(), 2, 2));
        inputs.insert("field".to_owned(), field(&anchors, 2, 2, 2, 2));
        inputs.insert("hole".to_owned(), mask(hole.clone(), 2, 2));
        inputs
    };
    assert_eq!(compute(&mk()), compute(&mk()));
}

#[test]
fn mismatched_field_extent_is_rejected() {
    let src = vec![1.0, 2.0, 3.0, 4.0];
    let tgt = vec![0.0, 0.0, 0.0, 0.0];
    let hole = vec![1.0, 1.0, 1.0, 1.0];
    // The field claims a 1x1 target, but the target image is 2x2.
    let anchors = [(0, 0)];
    let mut inputs = InputValues::new();
    inputs.insert("source".to_owned(), gray_image(src, 2, 2));
    inputs.insert("target".to_owned(), gray_image(tgt, 2, 2));
    inputs.insert("field".to_owned(), field(&anchors, 1, 1, 2, 2));
    inputs.insert("hole".to_owned(), mask(hole, 2, 2));
    let err = PatchSynthesize::new()
        .compute(&inputs, &serde_json::json!({}))
        .unwrap_err();
    assert_eq!(err.code, super::E_PATCH_SYNTHESIZE_SHAPE);
}
