//! M4 patch-repair evidence fixtures (`OP_CATALOG` §10, `PatchMatch`, bn-2kv3).
//!
//! The consolidated *evidence* fixtures for the patch-repair family.
//!
//! They cover `repair.patch_field` (`PatchMatch` correspondence) and
//! `repair.patch_synthesize` (the fill gather), proving — through the
//! *registered* ops, not internal helpers — the three M4 acceptance guarantees
//! the gate must be able to see:
//!
//! 1. **Brute-force differential** — the seeded `PatchMatch` NNF matches an
//!    independent exact nearest-neighbour oracle on a tiny fixture.
//! 2. **Seeded determinism** — a fixed seed and scan order yield a bit-identical
//!    field (and a bit-identical synthesised fill) across reruns.
//! 3. **Report artifacts** — the field op's report carries the iterative-solver
//!    convergence metrics (iteration count, per-iteration cost history, stop
//!    reason) the M4 "solver exposes convergence" criterion requires.
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::float_cmp,
    reason = "fixtures use small exact integer samples and coordinates, and the \
              float equalities are exact integer-valued identity checks"
)]

use paintop_core::executor::{InputValues, OpImplementation, ResourceValue};
use paintop_cpu::patch_field::{PATCH_FIELD_OP_ID, PatchField};
use paintop_cpu::patch_synthesize::{PATCH_SYNTHESIZE_OP_ID, PatchSynthesize};
use paintop_cpu::patchmatch::{PatchPlane, brute_force_nnf};
use paintop_cpu::registry::operation_registry;
use paintop_ir::{
    AlphaRepresentation, ChannelLayout, ColorEncoding, ColorRange, CoordinateConvention, Extent,
    ImageDescriptor, MaskDescriptor, MaskMeaning, ResourceDescriptor, ScalarType, SemanticRole,
    SolverStopReason, ValidRange,
};

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

fn field_params(radius: u64, iterations: u64, seed: u64) -> serde_json::Value {
    serde_json::json!({ "radius": radius, "iterations": iterations, "seed": seed })
}

/// Both repair ops are wired into the registered op set under the `repair`
/// domain, with their `cpu.reference` implementations.
#[test]
fn patch_repair_ops_are_registered() {
    let reg = operation_registry().expect("op registry");
    for id in [PATCH_FIELD_OP_ID, PATCH_SYNTHESIZE_OP_ID] {
        let op_id = id.parse().expect("op id");
        assert!(reg.get(&op_id).is_ok(), "{id} not registered");
    }
}

/// Evidence 1: the seeded `PatchMatch` field matches the exact brute-force oracle
/// on a tiny gradient fixture whose every target value appears once in the
/// source.
#[test]
fn field_matches_brute_force_oracle() {
    let src_vec: Vec<f32> = (0..16).map(|i| i as f32).collect();
    let tgt_vec = vec![10.0, 3.0, 15.0, 6.0];
    let mut inputs = InputValues::new();
    inputs.insert("source".to_owned(), gray_image(src_vec.clone(), 4, 4));
    inputs.insert("target".to_owned(), gray_image(tgt_vec.clone(), 2, 2));
    let out = PatchField::new()
        .compute(&inputs, &field_params(0, 32, 7))
        .expect("patch_field");
    let field = out.get("field").expect("field");

    let sp = PatchPlane::new(&src_vec, 4, 4, 1).expect("source plane");
    let tp = PatchPlane::new(&tgt_vec, 2, 2, 1).expect("target plane");
    let oracle = brute_force_nnf(&tp, &sp, 0, |_, _| true, |_, _| true);

    let samples = field.samples();
    for y in 0..2u32 {
        for x in 0..2u32 {
            let base = ((y as usize * 2) + x as usize) * 3;
            let (sx, sy) = (samples[base] as u32, samples[base + 1] as u32);
            let o = oracle.get(x, y).expect("oracle cell");
            assert_eq!(
                (sx, sy),
                (o.src_x, o.src_y),
                "patch_field disagrees with the brute-force oracle at ({x},{y})"
            );
        }
    }
}

/// Evidence 2: the field is bit-identical across reruns for a fixed seed + scan
/// order, and the synthesised fill it drives is bit-identical too.
#[test]
fn field_and_fill_are_bit_identical_for_a_fixed_seed() {
    let src: Vec<f32> = (0..25).map(|i| (i % 9) as f32).collect();
    let tgt = vec![3.0, 5.0, 1.0, 6.0, 0.0, 2.0, 4.0, 5.0, 1.0];
    let hole = vec![1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0];

    let run = || {
        let mut inputs = InputValues::new();
        inputs.insert("source".to_owned(), gray_image(src.clone(), 5, 5));
        inputs.insert("target".to_owned(), gray_image(tgt.clone(), 3, 3));
        let field_out = PatchField::new()
            .compute(&inputs, &field_params(1, 12, 1234))
            .expect("patch_field");
        let field = field_out.get("field").expect("field").clone();

        let mut syn = InputValues::new();
        syn.insert("source".to_owned(), gray_image(src.clone(), 5, 5));
        syn.insert("target".to_owned(), gray_image(tgt.clone(), 3, 3));
        syn.insert("field".to_owned(), field.clone());
        syn.insert("hole".to_owned(), mask(hole.clone(), 3, 3));
        let fill_out = PatchSynthesize::new()
            .compute(&syn, &serde_json::json!({}))
            .expect("patch_synthesize");
        (
            field.samples().to_vec(),
            fill_out.get("image").expect("image").samples().to_vec(),
        )
    };

    let a = run();
    let b = run();
    assert_eq!(a.0, b.0, "the field must be bit-identical for a fixed seed");
    assert_eq!(a.1, b.1, "the fill must be bit-identical for a fixed seed");
}

/// Evidence 3: the field op's report artifact carries the iterative-solver
/// convergence metrics — a per-iteration cost history, the iteration count, and
/// a stop reason — so the M4 gate can audit convergence.
#[test]
fn report_artifact_carries_convergence_metrics() {
    let src: Vec<f32> = (0..16).map(|i| i as f32).collect();
    let tgt = vec![10.0, 3.0, 15.0, 6.0];
    let mut inputs = InputValues::new();
    inputs.insert("source".to_owned(), gray_image(src, 4, 4));
    inputs.insert("target".to_owned(), gray_image(tgt, 2, 2));
    let out = PatchField::new()
        .compute(&inputs, &field_params(0, 6, 7))
        .expect("patch_field");
    let report = out.get("report").expect("report output");
    let solver = report
        .as_report()
        .and_then(|r| r.solver.as_ref())
        .expect("solver data");

    assert_eq!(solver.kind, "patchmatch");
    assert_eq!(solver.iterations, Some(solver.steps));
    assert_eq!(solver.residual_history.len() as u32, solver.steps);
    assert!(solver.steps >= 1 && solver.steps <= 6);
    assert!(matches!(
        solver.stop_reason,
        Some(SolverStopReason::Stalled | SolverStopReason::MaxIterations)
    ));
    // The cost history is non-increasing (search only ever keeps improvements).
    for w in solver.residual_history.windows(2) {
        assert!(
            w[1] <= w[0] + 1.0e-9,
            "cost history rose: {:?}",
            solver.residual_history
        );
    }
}

/// Evidence 4: the synthesised fill honours the outside-hole identity and the
/// anchored gather — the two defining guarantees of the candidate fill — and
/// fills a uniform-texture hole coherently.
///
/// The source is a single uniform texture value (`5`), so *any* anchor the
/// `PatchField` chooses gathers `5`: the hole fills coherently regardless of the
/// search's choice. The unmasked target pixels (value `7`) must stay untouched
/// (outside-hole identity). This keeps the fixture's outcome analytic: the
/// expected image is fully determined, not a function of the stochastic search.
#[test]
fn synthesis_identity_and_uniform_texture_fill() {
    // A uniform-5 source; a target of 7; a hole over the middle column.
    let src = vec![5.0; 9];
    let tgt = vec![7.0; 9];
    let hole = vec![
        0.0, 1.0, 0.0, //
        0.0, 1.0, 0.0, //
        0.0, 1.0, 0.0,
    ];

    let mut field_inputs = InputValues::new();
    field_inputs.insert("source".to_owned(), gray_image(src.clone(), 3, 3));
    field_inputs.insert("target".to_owned(), gray_image(tgt.clone(), 3, 3));
    // A target_mask equal to the hole focuses the search on the hole pixels.
    field_inputs.insert("target_mask".to_owned(), mask(hole.clone(), 3, 3));
    let field_out = PatchField::new()
        .compute(&field_inputs, &field_params(0, 24, 5))
        .expect("patch_field");
    let field = field_out.get("field").expect("field").clone();

    let mut syn = InputValues::new();
    syn.insert("source".to_owned(), gray_image(src, 3, 3));
    syn.insert("target".to_owned(), gray_image(tgt, 3, 3));
    syn.insert("field".to_owned(), field);
    syn.insert("hole".to_owned(), mask(hole, 3, 3));
    let fill = PatchSynthesize::new()
        .compute(&syn, &serde_json::json!({}))
        .expect("patch_synthesize");
    let out = fill.get("image").expect("image").samples().to_vec();

    // Unmasked columns (0 and 2) stay at the target identity 7.
    for row in 0..3 {
        assert_eq!(out[row * 3], 7.0, "left column identity");
        assert_eq!(out[(row * 3) + 2], 7.0, "right column identity");
        // The middle (hole) column gathers the uniform source texture 5.
        assert_eq!(out[(row * 3) + 1], 5.0, "hole filled with coherent texture");
    }
}
