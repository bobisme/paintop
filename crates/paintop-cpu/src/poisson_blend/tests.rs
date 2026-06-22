//! Verification suite for `repair.poisson_blend@1` (`OP_CATALOG` §12):
//!
//! - **schema/contract**: the manifest validates and matches its contract;
//! - **boundary continuity**: outside the mask and at its edge the result equals
//!   the target (seamless: no visible seam);
//! - **gradient reproduction**: a uniform-gradient source over a uniform target
//!   reconstructs the source gradient inside the mask (up to the boundary offset);
//! - **determinism**: a rerun is bit-identical (M4 exit criterion 2);
//! - **report metrics**: the report carries `SolverData` with the iteration count,
//!   stop reason, and a residual history (M4 exit criterion 1).

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::needless_range_loop,
    clippy::float_cmp,
    clippy::suboptimal_flops,
    clippy::many_single_char_names,
    reason = "small analytic fixtures index by (col, row) and build coordinate \
              ramps narrowed to the op's f32 sample type; the grids are tiny so \
              the casts are exact and the explicit loops read clearly, and the \
              float equalities are exact analytic-fixture / determinism checks"
)]

use paintop_core::executor::{InputValues, OpImplementation, ResourceValue};
use paintop_ir::{
    AlphaRepresentation, ChannelLayout, ColorEncoding, ColorRange, CoordinateConvention, Extent,
    ImageDescriptor, MaskDescriptor, MaskMeaning, ResourceDescriptor, ScalarType, SemanticRole,
    SolverStopReason, ValidRange,
};

use super::{POISSON_BLEND_OP_ID, PoissonBlend};

/// A single-channel (Gray) image value of `extent` from row-major samples.
fn gray_image(width: u32, height: u32, samples: Vec<f32>) -> ResourceValue {
    let descriptor = ResourceDescriptor::Image(ImageDescriptor {
        extent: Extent::new(width, height),
        layout: ChannelLayout::Gray,
        scalar: ScalarType::F32,
        color: ColorEncoding::LinearSrgb,
        range: ColorRange::SceneReferred,
        alpha: AlphaRepresentation::Straight,
        coordinates: CoordinateConvention::PixelCenterUpperLeft,
        semantic: SemanticRole::Color,
    });
    ResourceValue::new(descriptor, 1, samples).expect("gray image value")
}

/// A coverage mask value of `extent` from row-major samples.
fn mask_value(width: u32, height: u32, samples: Vec<f32>) -> ResourceValue {
    let descriptor = ResourceDescriptor::Mask(MaskDescriptor {
        extent: Extent::new(width, height),
        scalar: ScalarType::F32,
        range: ValidRange::Bounded { min: 0.0, max: 1.0 },
        meaning: MaskMeaning::Coverage,
        coordinates: CoordinateConvention::PixelCenterUpperLeft,
    });
    ResourceValue::new(descriptor, 1, samples).expect("mask value")
}

/// A centred square mask: 1 inside `[lo, hi)` on both axes, 0 elsewhere.
fn square_mask(width: usize, height: usize, lo: usize, hi: usize) -> Vec<f32> {
    let mut m = vec![0.0_f32; width * height];
    for row in lo..hi {
        for col in lo..hi {
            m[row * width + col] = 1.0;
        }
    }
    m
}

/// Run the op and return (candidate samples, report value).
fn run(
    width: u32,
    height: u32,
    source: Vec<f32>,
    target: Vec<f32>,
    mask: Vec<f32>,
    params: &serde_json::Value,
) -> (Vec<f32>, ResourceValue) {
    let mut inputs = InputValues::new();
    inputs.insert("source".to_owned(), gray_image(width, height, source));
    inputs.insert("target".to_owned(), gray_image(width, height, target));
    inputs.insert("mask".to_owned(), mask_value(width, height, mask));
    let mut out = PoissonBlend::new()
        .compute(&inputs, params)
        .expect("blend computes");
    let candidate = out.remove("candidate").expect("candidate output");
    let report = out.remove("report").expect("report output");
    (candidate.samples().to_vec(), report)
}

#[test]
fn manifest_matches_contract() {
    let manifest = PoissonBlend::manifest().expect("manifest builds");
    assert_eq!(manifest.id.to_string(), POISSON_BLEND_OP_ID);
    assert_eq!(manifest.inputs.len(), 3);
    assert_eq!(manifest.outputs.len(), 2);
}

#[test]
fn boundary_pixels_keep_the_target() {
    // Outside the masked square the candidate must equal the target bit-for-bit.
    let (w, h) = (16, 16);
    let source = vec![0.9_f32; w * h];
    let target = vec![0.1_f32; w * h];
    let mask = square_mask(w, h, 5, 11);
    let (candidate, _report) = run(
        w as u32,
        h as u32,
        source,
        target.clone(),
        mask.clone(),
        &serde_json::json!({}),
    );
    for idx in 0..w * h {
        if mask[idx] <= 0.5 {
            assert_eq!(
                candidate[idx], target[idx],
                "outside the mask the candidate must equal the target at {idx}"
            );
        }
    }
}

#[test]
fn uniform_source_over_uniform_target_yields_target_constant() {
    // A flat source (zero gradient everywhere) blended into a flat target must
    // reconstruct the target constant inside the mask: the gradient to match is
    // zero, and the boundary pins the constant.
    let (w, h) = (20, 20);
    let source = vec![0.7_f32; w * h];
    let target = vec![0.25_f32; w * h];
    let mask = square_mask(w, h, 6, 14);
    let (candidate, report) = run(
        w as u32,
        h as u32,
        source,
        target,
        mask.clone(),
        &serde_json::json!({"tolerance": 1e-8, "max_iterations": 5000}),
    );
    for idx in 0..w * h {
        if mask[idx] > 0.5 {
            assert!(
                (candidate[idx] - 0.25).abs() < 1e-3,
                "flat-source blend interior {idx} = {} should equal the target 0.25",
                candidate[idx]
            );
        }
    }
    let report = report.as_report().expect("report payload");
    let solver = report.solver.as_ref().expect("solver data");
    assert_eq!(solver.kind, "poisson");
    assert!(
        solver.converged.unwrap_or(false),
        "the flat blend converges"
    );
}

#[test]
fn linear_gradient_source_is_reproduced_inside_mask() {
    // A source with a constant horizontal gradient f(x) = x is harmonic, so the
    // blended interior must reproduce the gradient: consecutive interior columns
    // differ by ~1 along a row (the source's gradient), independent of the target.
    let (w, h) = (24, 24);
    let mut source = vec![0.0_f32; w * h];
    for row in 0..h {
        for col in 0..w {
            source[row * w + col] = col as f32;
        }
    }
    // A target equal to the source on the boundary so the blend is the source
    // itself: with matching boundary and harmonic source the result is the source.
    let target = source.clone();
    let mask = square_mask(w, h, 6, 18);
    let (candidate, _report) = run(
        w as u32,
        h as u32,
        source.clone(),
        target,
        mask.clone(),
        &serde_json::json!({"tolerance": 1e-9, "max_iterations": 8000}),
    );
    for row in 7..17 {
        for col in 7..17 {
            let idx = row * w + col;
            if mask[idx] > 0.5 {
                let dx = candidate[idx] - candidate[idx - 1];
                assert!(
                    (dx - 1.0).abs() < 5e-2,
                    "interior horizontal gradient at ({col},{row}) = {dx}, expected ~1",
                );
            }
        }
    }
}

#[test]
fn reruns_are_bit_identical() {
    let (w, h) = (18, 14);
    let mut source = vec![0.0_f32; w * h];
    for i in 0..w * h {
        source[i] = ((i * 7) % 13) as f32 / 13.0;
    }
    let target = vec![0.3_f32; w * h];
    let mask = square_mask(w, h, 4, 12);
    let p = serde_json::json!({"max_iterations": 200});
    let (a, _) = run(
        w as u32,
        h as u32,
        source.clone(),
        target.clone(),
        mask.clone(),
        &p,
    );
    let (b, _) = run(w as u32, h as u32, source, target, mask, &p);
    assert_eq!(a, b, "the blended candidate must be bit-identical on rerun");
}

#[test]
fn report_carries_iterative_metrics() {
    let (w, h) = (16, 16);
    // A quadratic source (non-zero Laplacian) so the guidance is non-trivial and
    // the solver has real work to do — the residual starts above the tolerance.
    let mut source = vec![0.0_f32; w * h];
    for row in 0..h {
        for col in 0..w {
            let x = col as f32;
            let y = row as f32;
            source[row * w + col] = 0.01 * (x * x + y * y);
        }
    }
    let target = vec![0.2_f32; w * h];
    let mask = square_mask(w, h, 4, 12);
    let (_c, report) = run(
        w as u32,
        h as u32,
        source,
        target,
        mask,
        &serde_json::json!({"max_iterations": 3, "tolerance": 1e-12}),
    );
    let report = report.as_report().expect("report payload");
    let solver = report.solver.as_ref().expect("solver data");
    assert_eq!(solver.iterations, Some(3));
    assert_eq!(solver.stop_reason, Some(SolverStopReason::MaxIterations));
    assert_eq!(solver.converged, Some(false));
    assert_eq!(
        solver.residual_history.len(),
        3,
        "one residual recorded per sweep"
    );
    assert!(solver.tolerance.unwrap() > 0.0);
    assert!(solver.final_residual.unwrap().is_finite());
}

#[test]
fn mismatched_extents_are_rejected() {
    let mut inputs = InputValues::new();
    inputs.insert("source".to_owned(), gray_image(8, 8, vec![0.0; 64]));
    inputs.insert("target".to_owned(), gray_image(8, 8, vec![0.0; 64]));
    inputs.insert("mask".to_owned(), mask_value(4, 4, vec![0.0; 16]));
    let result = PoissonBlend::new().compute(&inputs, &serde_json::json!({}));
    assert!(result.is_err(), "a mask-extent mismatch must be rejected");
}

#[test]
fn invalid_tolerance_is_rejected() {
    let (w, h) = (8, 8);
    let mut inputs = InputValues::new();
    inputs.insert("source".to_owned(), gray_image(w, h, vec![0.0; 64]));
    inputs.insert("target".to_owned(), gray_image(w, h, vec![0.0; 64]));
    inputs.insert("mask".to_owned(), mask_value(w, h, vec![0.0; 64]));
    let result = PoissonBlend::new().compute(&inputs, &serde_json::json!({"tolerance": 2.0}));
    assert!(result.is_err(), "a tolerance >= 1 must be rejected");
}
