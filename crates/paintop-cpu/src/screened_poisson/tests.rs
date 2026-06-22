//! Verification suite for `repair.screened_poisson@1` (`OP_CATALOG` §12):
//!
//! - **schema/contract**: the manifest validates and matches its contract;
//! - **lambda = 0 is pure Poisson**: with a flat anchor and a gradient guidance,
//!   the interior reconstructs the guidance gradients (the seamless-clone limit);
//! - **large lambda snaps to the anchor**: the interior collapses onto the anchor
//!   field, ignoring the guidance;
//! - **monotone lambda**: a larger lambda moves the interior closer to the anchor;
//! - **boundary continuity / determinism / report metrics**;
//! - **invalid lambda is rejected**.

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
    ValidRange,
};

use super::{SCREENED_POISSON_OP_ID, ScreenedPoisson};

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

fn square_mask(width: usize, height: usize, lo: usize, hi: usize) -> Vec<f32> {
    let mut m = vec![0.0_f32; width * height];
    for row in lo..hi {
        for col in lo..hi {
            m[row * width + col] = 1.0;
        }
    }
    m
}

fn run(
    width: u32,
    height: u32,
    guidance: Vec<f32>,
    anchor: Vec<f32>,
    mask: Vec<f32>,
    params: &serde_json::Value,
) -> (Vec<f32>, ResourceValue) {
    let mut inputs = InputValues::new();
    inputs.insert("guidance".to_owned(), gray_image(width, height, guidance));
    inputs.insert("anchor".to_owned(), gray_image(width, height, anchor));
    inputs.insert("mask".to_owned(), mask_value(width, height, mask));
    let mut out = ScreenedPoisson::new()
        .compute(&inputs, params)
        .expect("screened poisson computes");
    let candidate = out.remove("candidate").expect("candidate output");
    let report = out.remove("report").expect("report output");
    (candidate.samples().to_vec(), report)
}

/// A horizontal-ramp guidance field f(x) = x (harmonic: zero Laplacian).
fn ramp(width: usize, height: usize) -> Vec<f32> {
    let mut g = vec![0.0_f32; width * height];
    for row in 0..height {
        for col in 0..width {
            g[row * width + col] = col as f32;
        }
    }
    g
}

/// A quadratic-bump guidance field f(x,y) = 0.02·(x²+y²) with a *non-zero*
/// constant Laplacian (Δf = 0.08), so the pure-Poisson interior bows away from a
/// flat anchor and the screened term has a real effect to measure.
fn bump(width: usize, height: usize) -> Vec<f32> {
    let mut g = vec![0.0_f32; width * height];
    for row in 0..height {
        for col in 0..width {
            let x = col as f32;
            let y = row as f32;
            g[row * width + col] = 0.02 * (x * x + y * y);
        }
    }
    g
}

#[test]
fn manifest_matches_contract() {
    let manifest = ScreenedPoisson::manifest().expect("manifest builds");
    assert_eq!(manifest.id.to_string(), SCREENED_POISSON_OP_ID);
    assert_eq!(manifest.inputs.len(), 3);
    assert_eq!(manifest.outputs.len(), 2);
    // The lambda param is declared.
    assert!(manifest.params.iter().any(|p| p.name == "lambda"));
}

#[test]
fn lambda_zero_recovers_guidance_gradient() {
    // lambda = 0 is pure Poisson: with a ramp guidance and a boundary equal to
    // that ramp, the interior reproduces the ramp gradient (~1 per column).
    let (w, h) = (24, 24);
    let guidance = ramp(w, h);
    let anchor = guidance.clone();
    let mask = square_mask(w, h, 6, 18);
    let (candidate, _report) = run(
        w as u32,
        h as u32,
        guidance,
        anchor,
        mask.clone(),
        &serde_json::json!({"lambda": 0.0, "tolerance": 1e-9, "max_iterations": 8000}),
    );
    for row in 7..17 {
        for col in 7..17 {
            let idx = row * w + col;
            if mask[idx] > 0.5 {
                let dx = candidate[idx] - candidate[idx - 1];
                assert!(
                    (dx - 1.0).abs() < 5e-2,
                    "lambda=0 interior gradient at ({col},{row}) = {dx}, expected ~1"
                );
            }
        }
    }
}

#[test]
fn large_lambda_snaps_to_anchor() {
    // A huge lambda makes the data term dominate: the interior collapses onto the
    // (flat) anchor regardless of the (curved) bump guidance.
    let (w, h) = (20, 20);
    let guidance = bump(w, h);
    let anchor = vec![0.4_f32; w * h];
    let mask = square_mask(w, h, 6, 14);
    let (candidate, _report) = run(
        w as u32,
        h as u32,
        guidance,
        anchor,
        mask.clone(),
        &serde_json::json!({"lambda": 1.0e6, "tolerance": 1e-9, "max_iterations": 8000}),
    );
    for idx in 0..w * h {
        if mask[idx] > 0.5 {
            assert!(
                (candidate[idx] - 0.4).abs() < 1e-2,
                "large lambda interior {idx} = {} should snap to anchor 0.4",
                candidate[idx]
            );
        }
    }
}

#[test]
fn larger_lambda_moves_closer_to_anchor() {
    // Monotone: increasing lambda pulls the interior centre nearer the anchor.
    // A bump guidance (non-zero Laplacian) bows the pure-Poisson interior away
    // from the flat anchor, so lambda has a measurable pull.
    let (w, h) = (20, 20);
    let guidance = bump(w, h);
    let anchor = vec![0.5_f32; w * h];
    let mask = square_mask(w, h, 6, 14);
    let centre = (h / 2) * w + (w / 2);

    let dist = |lambda: f64| -> f32 {
        let (candidate, _r) = run(
            w as u32,
            h as u32,
            guidance.clone(),
            anchor.clone(),
            mask.clone(),
            &serde_json::json!({"lambda": lambda, "tolerance": 1e-8, "max_iterations": 8000}),
        );
        (candidate[centre] - 0.5).abs()
    };

    let small = dist(0.01);
    let large = dist(10.0);
    assert!(
        large < small,
        "a larger lambda must move the interior closer to the anchor: \
         small-lambda dist {small}, large-lambda dist {large}"
    );
}

#[test]
fn boundary_pixels_keep_the_anchor() {
    let (w, h) = (16, 16);
    let guidance = ramp(w, h);
    let anchor = vec![0.3_f32; w * h];
    let mask = square_mask(w, h, 5, 11);
    let (candidate, _report) = run(
        w as u32,
        h as u32,
        guidance,
        anchor.clone(),
        mask.clone(),
        &serde_json::json!({"lambda": 5.0}),
    );
    for idx in 0..w * h {
        if mask[idx] <= 0.5 {
            assert_eq!(
                candidate[idx], anchor[idx],
                "outside the mask the candidate must equal the anchor at {idx}"
            );
        }
    }
}

#[test]
fn reruns_are_bit_identical() {
    let (w, h) = (18, 14);
    let guidance = ramp(w, h);
    let mut anchor = vec![0.0_f32; w * h];
    for i in 0..w * h {
        anchor[i] = ((i * 3) % 11) as f32 / 11.0;
    }
    let mask = square_mask(w, h, 4, 12);
    let p = serde_json::json!({"lambda": 2.0, "max_iterations": 200});
    let (a, _) = run(
        w as u32,
        h as u32,
        guidance.clone(),
        anchor.clone(),
        mask.clone(),
        &p,
    );
    let (b, _) = run(w as u32, h as u32, guidance, anchor, mask, &p);
    assert_eq!(a, b, "the reconstruction must be bit-identical on rerun");
}

#[test]
fn report_carries_screened_solver_metrics() {
    let (w, h) = (16, 16);
    // A bump guidance (non-zero Laplacian) keeps the initial residual above the
    // tolerance so the solver does real iterations.
    let guidance = bump(w, h);
    let anchor = vec![0.2_f32; w * h];
    let mask = square_mask(w, h, 4, 12);
    let (_c, report) = run(
        w as u32,
        h as u32,
        guidance,
        anchor,
        mask,
        &serde_json::json!({"lambda": 1.0, "max_iterations": 3, "tolerance": 1e-12}),
    );
    let report = report.as_report().expect("report payload");
    let solver = report.solver.as_ref().expect("solver data");
    assert_eq!(solver.kind, "screened-poisson");
    assert_eq!(solver.iterations, Some(3));
    assert_eq!(solver.residual_history.len(), 3);
    assert!(solver.tolerance.unwrap() > 0.0);
}

#[test]
fn negative_lambda_is_rejected() {
    let (w, h) = (8, 8);
    let mut inputs = InputValues::new();
    inputs.insert("guidance".to_owned(), gray_image(w, h, vec![0.0; 64]));
    inputs.insert("anchor".to_owned(), gray_image(w, h, vec![0.0; 64]));
    inputs.insert("mask".to_owned(), mask_value(w, h, vec![0.0; 64]));
    let result = ScreenedPoisson::new().compute(&inputs, &serde_json::json!({"lambda": -1.0}));
    assert!(result.is_err(), "a negative lambda must be rejected");
}
