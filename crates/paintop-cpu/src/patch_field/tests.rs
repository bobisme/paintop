//! Tests for `repair.patch_field@1`: manifest/contract shape, parameter
//! validation, seeded determinism, the brute-force oracle differential, and the
//! convergence report.
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "test fixtures use small exact integer samples and coordinates"
)]

use paintop_core::executor::{InputValues, OpImplementation, ResourceValue};
use paintop_ir::{
    AlphaRepresentation, ChannelLayout, ColorEncoding, ColorRange, CoordinateConvention, Extent,
    ImageDescriptor, MaskDescriptor, MaskMeaning, ResourceDescriptor, ScalarType, SemanticRole,
    ValidRange,
};

use super::{PATCH_FIELD_OP_ID, PatchField, PatchFieldParams};
use crate::patchmatch::{PatchPlane, brute_force_nnf};

fn image_descriptor(extent: Extent, layout: ChannelLayout) -> ResourceDescriptor {
    ResourceDescriptor::Image(ImageDescriptor {
        extent,
        layout,
        scalar: ScalarType::F32,
        color: ColorEncoding::LinearSrgb,
        range: ColorRange::SceneReferred,
        alpha: AlphaRepresentation::Premultiplied,
        coordinates: CoordinateConvention::PixelCenterUpperLeft,
        semantic: SemanticRole::Color,
    })
}

fn gray_image(samples: Vec<f32>, w: u32, h: u32) -> ResourceValue {
    ResourceValue::new(
        image_descriptor(Extent::new(w, h), ChannelLayout::Gray),
        1,
        samples,
    )
    .expect("gray image")
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

fn params(radius: u64, iterations: u64, seed: u64) -> serde_json::Value {
    serde_json::json!({ "radius": radius, "iterations": iterations, "seed": seed })
}

#[test]
fn manifest_declares_field_and_report_outputs() {
    let m = PatchField::manifest().expect("manifest");
    assert_eq!(m.id.to_string(), PATCH_FIELD_OP_ID);
    let out_names: Vec<&str> = m.outputs.iter().map(|o| o.name.as_str()).collect();
    assert!(out_names.contains(&"field"));
    assert!(out_names.contains(&"report"));
    let in_names: Vec<&str> = m.inputs.iter().map(|i| i.name.as_str()).collect();
    assert!(in_names.contains(&"source"));
    assert!(in_names.contains(&"target"));
}

#[test]
fn params_reject_missing_and_out_of_range() {
    assert!(PatchFieldParams::resolve(&serde_json::json!({})).is_err());
    // Over-large radius.
    assert!(PatchFieldParams::resolve(&params(10_000, 8, 0)).is_err());
    // Zero iterations.
    assert!(PatchFieldParams::resolve(&params(1, 0, 0)).is_err());
    // A valid set resolves.
    let p = PatchFieldParams::resolve(&params(2, 8, 7)).expect("valid params");
    assert_eq!((p.radius, p.iterations, p.seed), (2, 8, 7));
}

#[test]
fn produces_a_patch_field_with_finite_samples() {
    let src: Vec<f32> = (0..16).map(|i| i as f32).collect();
    let tgt = vec![10.0, 3.0, 15.0, 6.0];
    let mut inputs = InputValues::new();
    inputs.insert("source".to_owned(), gray_image(src, 4, 4));
    inputs.insert("target".to_owned(), gray_image(tgt, 2, 2));
    let out = PatchField::new()
        .compute(&inputs, &params(0, 12, 7))
        .expect("compute");
    let field = out.get("field").expect("field output");
    assert_eq!(field.channels(), 3);
    assert_eq!(field.extent(), Extent::new(2, 2));
    assert!(field.samples().iter().all(|s| s.is_finite()));
    let report = out.get("report").expect("report output");
    let solver = report
        .as_report()
        .and_then(|r| r.solver.as_ref())
        .expect("solver data");
    assert_eq!(solver.kind, "patchmatch");
    assert!(!solver.residual_history.is_empty());
    assert_eq!(solver.iterations, Some(solver.steps));
}

#[test]
fn reruns_are_bit_identical_for_a_fixed_seed() {
    let src: Vec<f32> = (0..25).map(|i| (i % 9) as f32).collect();
    let tgt = vec![3.0, 5.0, 1.0, 6.0, 0.0, 2.0, 4.0, 5.0, 1.0];
    let mut inputs = InputValues::new();
    inputs.insert("source".to_owned(), gray_image(src, 5, 5));
    inputs.insert("target".to_owned(), gray_image(tgt, 3, 3));
    let a = PatchField::new()
        .compute(&inputs, &params(1, 10, 42))
        .expect("a");
    let b = PatchField::new()
        .compute(&inputs, &params(1, 10, 42))
        .expect("b");
    assert_eq!(
        a.get("field").unwrap().samples(),
        b.get("field").unwrap().samples(),
        "a fixed seed must produce a bit-identical field"
    );
}

#[test]
fn field_matches_the_brute_force_oracle_on_a_tiny_fixture() {
    // A gradient source whose values each appear once; with enough iterations
    // PatchMatch reaches the exact NNF.
    let src_vec: Vec<f32> = (0..16).map(|i| i as f32).collect();
    let tgt_vec = vec![10.0, 3.0, 15.0, 6.0];
    let mut inputs = InputValues::new();
    inputs.insert("source".to_owned(), gray_image(src_vec.clone(), 4, 4));
    inputs.insert("target".to_owned(), gray_image(tgt_vec.clone(), 2, 2));
    let out = PatchField::new()
        .compute(&inputs, &params(0, 32, 7))
        .expect("compute");
    let field = out.get("field").unwrap();

    let sp = PatchPlane::new(&src_vec, 4, 4, 1).unwrap();
    let tp = PatchPlane::new(&tgt_vec, 2, 2, 1).unwrap();
    let oracle = brute_force_nnf(&tp, &sp, 0, |_, _| true, |_, _| true);

    let samples = field.samples();
    for y in 0..2u32 {
        for x in 0..2u32 {
            let base = ((y as usize * 2) + x as usize) * 3;
            let (sx, sy) = (samples[base] as u32, samples[base + 1] as u32);
            let o = oracle.get(x, y).unwrap();
            assert_eq!(
                (sx, sy),
                (o.src_x, o.src_y),
                "field disagrees with the oracle at ({x},{y})"
            );
        }
    }
}

#[test]
fn target_mask_leaves_unmatched_pixels_at_identity() {
    let src = vec![0.0, 1.0, 2.0, 3.0];
    let tgt = vec![9.0, 9.0, 9.0, 9.0];
    // Match only the top-left target pixel.
    let tmask = vec![1.0, 0.0, 0.0, 0.0];
    let mut inputs = InputValues::new();
    inputs.insert("source".to_owned(), gray_image(src, 2, 2));
    inputs.insert("target".to_owned(), gray_image(tgt, 2, 2));
    inputs.insert("target_mask".to_owned(), mask(tmask, 2, 2));
    let out = PatchField::new()
        .compute(&inputs, &params(0, 6, 3))
        .expect("compute");
    let samples = out.get("field").unwrap().samples();
    // Bottom-right target (1,1) keeps its own coordinate: index (1*2 + 1)*3.
    let base = 3usize * 3;
    assert_eq!((samples[base] as u32, samples[base + 1] as u32), (1, 1));
    assert!((samples[base + 2]).abs() < 1.0e-6, "identity cost is zero");
}

#[test]
fn source_mask_restricts_eligible_anchors() {
    // The exact match for the target value (7) sits at source index 0, which the
    // mask excludes; the field must point elsewhere.
    let src = vec![7.0, 0.0, 0.0, 0.0];
    let tgt = vec![7.0];
    let smask = vec![0.0, 1.0, 1.0, 1.0]; // exclude anchor (0,0)
    let mut inputs = InputValues::new();
    inputs.insert("source".to_owned(), gray_image(src, 2, 2));
    inputs.insert("target".to_owned(), gray_image(tgt, 1, 1));
    inputs.insert("source_mask".to_owned(), mask(smask, 2, 2));
    let out = PatchField::new()
        .compute(&inputs, &params(0, 8, 1))
        .expect("compute");
    let samples = out.get("field").unwrap().samples();
    let (sx, sy) = (samples[0] as u32, samples[1] as u32);
    assert!(
        !(sx == 0 && sy == 0),
        "the excluded anchor (0,0) must not be chosen"
    );
}

#[test]
fn mismatched_mask_extent_is_rejected() {
    let src = vec![0.0, 1.0, 2.0, 3.0];
    let tgt = vec![0.0, 1.0, 2.0, 3.0];
    let mut inputs = InputValues::new();
    inputs.insert("source".to_owned(), gray_image(src, 2, 2));
    inputs.insert("target".to_owned(), gray_image(tgt, 2, 2));
    // A 1x1 mask against a 2x2 target is a type error.
    inputs.insert("target_mask".to_owned(), mask(vec![1.0], 1, 1));
    let err = PatchField::new()
        .compute(&inputs, &params(0, 4, 0))
        .unwrap_err();
    assert_eq!(err.code, super::E_PATCH_FIELD_INPUT);
}
