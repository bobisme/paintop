//! Verification suite for `optimize.local@1` (`plan.md` §1428; `ALIEN_OPS` §7):
//!
//! - **schema/contract**: the manifest validates and matches its contract;
//! - **known minimum**: with `smooth_weight = 0` and a synthetic target, the
//!   masked interior converges to the target (the analytic minimizer);
//! - **objective trajectory + stop reason** are recorded in the report's
//!   `SolverData`;
//! - **determinism**: a rerun with the same seed/schedule is bit-identical;
//! - **locality**: pixels outside the mask never change;
//! - **stop rules**: a tiny cap stops at the cap;
//! - **disable switch**: `enabled = false` returns a typed policy error;
//! - **invalid params** (bad weight, step, tolerance, degenerate objective) are
//!   rejected.

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::needless_range_loop,
    reason = "small analytic fixtures index by (col, row) and build coordinate \
              ramps narrowed to the op's f32 sample type; the grids are tiny so \
              the casts are exact and the explicit loops read clearly"
)]

use paintop_core::executor::{InputValues, OpImplementation, ResourceValue};
use paintop_ir::{
    AlphaRepresentation, ChannelLayout, ColorEncoding, ColorRange, CoordinateConvention,
    Descriptors, ErrorClass, Extent, ImageDescriptor, MaskDescriptor, MaskMeaning, OpContract,
    OutputRegions, ResourceDescriptor, ScalarType, SemanticRole, SolverStopReason, ValidRange,
};

use super::{E_OPTIMIZE_DISABLED, LOCAL_OPTIMIZE_OP_ID, LocalOptimize, MAX_ITERATIONS_LIMIT};

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
    init: Vec<f32>,
    target: Vec<f32>,
    mask: Vec<f32>,
    params: &serde_json::Value,
) -> Result<(Vec<f32>, ResourceValue), paintop_ir::Error> {
    let mut inputs = InputValues::new();
    inputs.insert("init".to_owned(), gray_image(width, height, init));
    inputs.insert("target".to_owned(), gray_image(width, height, target));
    inputs.insert("mask".to_owned(), mask_value(width, height, mask));
    let mut out = LocalOptimize::new().compute(&inputs, params)?;
    let candidate = out.remove("candidate").expect("candidate output");
    let report = out.remove("report").expect("report output");
    Ok((candidate.samples().to_vec(), report))
}

fn solver(report: &ResourceValue) -> paintop_ir::SolverData {
    report
        .as_report()
        .expect("report value")
        .solver
        .clone()
        .expect("solver data")
}

#[test]
fn manifest_matches_the_contract() {
    let manifest = LocalOptimize::manifest().expect("manifest builds");
    assert_eq!(manifest.id.to_string(), LOCAL_OPTIMIZE_OP_ID);
    let op = LocalOptimize::new();
    assert_eq!(manifest.inputs.len(), op.declared_inputs().len());
    assert_eq!(manifest.outputs.len(), op.declared_outputs().len());
}

#[test]
fn converges_to_the_known_minimum() {
    let (w, h) = (8_u32, 8_u32);
    let mask = square_mask(8, 8, 2, 6);
    let init = vec![0.0_f32; 64];
    let target = vec![0.7_f32; 64];
    let params = serde_json::json!({ "tolerance": 1e-7, "max_iterations": 4000 });

    let (out, report) = run(w, h, init, target, mask, &params).expect("optimize runs");
    // Inside the mask, the optimized field reaches the target.
    for row in 2..6 {
        for col in 2..6 {
            let v = out[row * 8 + col];
            assert!((v - 0.7).abs() < 1e-2, "free pixel {col},{row} = {v}");
        }
    }
    let data = solver(&report);
    assert_eq!(data.kind, "local-optimizer");
    assert_eq!(data.converged, Some(true));
    assert_eq!(data.stop_reason, Some(SolverStopReason::Converged));
    assert!(
        !data.residual_history.is_empty(),
        "objective trajectory recorded"
    );
    // The trajectory decays monotonically.
    for pair in data.residual_history.windows(2) {
        assert!(
            pair[1] <= pair[0] + 1e-9,
            "non-monotone trajectory: {pair:?}"
        );
    }
    assert_eq!(data.tolerance, Some(1e-7));
}

#[test]
fn pixels_outside_the_mask_are_unchanged() {
    let (w, h) = (6_u32, 6_u32);
    let mask = square_mask(6, 6, 2, 4);
    let init: Vec<f32> = (0..36).map(|i| i as f32 * 0.01).collect();
    let target = vec![1.0_f32; 36];
    let params = serde_json::json!({});

    let (out, _) = run(w, h, init.clone(), target, mask, &params).expect("optimize runs");
    for idx in 0..36 {
        let row = idx / 6;
        let col = idx % 6;
        let inside = (2..4).contains(&row) && (2..4).contains(&col);
        if !inside {
            assert!(
                (out[idx] - init[idx]).abs() < f32::EPSILON,
                "outside pixel {idx} changed"
            );
        }
    }
}

#[test]
fn reruns_are_bit_identical() {
    let (w, h) = (10_u32, 10_u32);
    let mask = square_mask(10, 10, 1, 9);
    let init: Vec<f32> = (0..100).map(|i| (i as f32 * 0.1).sin()).collect();
    let target: Vec<f32> = (0..100).map(|i| (i as f32 * 0.07).cos()).collect();
    let params = serde_json::json!({ "seed": 42, "max_iterations": 300 });

    let (a, ra) = run(w, h, init.clone(), target.clone(), mask.clone(), &params).expect("a");
    let (b, rb) = run(w, h, init, target, mask, &params).expect("b");
    assert_eq!(a, b, "candidate must be bit-identical on rerun");
    assert_eq!(solver(&ra).residual_history, solver(&rb).residual_history);
}

#[test]
fn the_seed_is_recorded_in_the_report() {
    let (w, h) = (5_u32, 5_u32);
    let mask = square_mask(5, 5, 1, 4);
    let init = vec![0.0_f32; 25];
    let target = vec![0.5_f32; 25];
    let params = serde_json::json!({ "seed": 9, "max_iterations": 50 });
    let (_, report) = run(w, h, init, target, mask, &params).expect("runs");
    // The schedule seed is carried in the stability_number field as the run identity.
    assert!((solver(&report).stability_number - 9.0).abs() < f64::EPSILON);
}

#[test]
fn tiny_cap_stops_at_the_cap() {
    let (w, h) = (8_u32, 8_u32);
    let mask = square_mask(8, 8, 1, 7);
    let init = vec![0.0_f32; 64];
    let target = vec![1.0_f32; 64];
    let params = serde_json::json!({ "max_iterations": 3, "tolerance": 1e-12, "step": 0.05 });
    let (_, report) = run(w, h, init, target, mask, &params).expect("runs");
    let data = solver(&report);
    assert_eq!(data.iterations, Some(3));
    assert_eq!(data.stop_reason, Some(SolverStopReason::MaxIterations));
    assert_eq!(data.converged, Some(false));
}

#[test]
fn disabled_returns_a_policy_error() {
    let (w, h) = (4_u32, 4_u32);
    let mask = square_mask(4, 4, 1, 3);
    let init = vec![0.0_f32; 16];
    let target = vec![1.0_f32; 16];
    let params = serde_json::json!({ "enabled": false });
    let err = run(w, h, init, target, mask, &params).expect_err("must refuse to run");
    assert_eq!(err.class, ErrorClass::Policy);
    assert_eq!(err.code, E_OPTIMIZE_DISABLED);
}

#[test]
fn invalid_params_are_rejected() {
    let (w, h) = (4_u32, 4_u32);
    let mask = square_mask(4, 4, 1, 3);
    let init = vec![0.0_f32; 16];
    let target = vec![1.0_f32; 16];

    for bad in [
        serde_json::json!({ "data_weight": -1.0 }),
        serde_json::json!({ "smooth_weight": -0.1 }),
        serde_json::json!({ "data_weight": 0.0, "smooth_weight": 0.0 }),
        serde_json::json!({ "step": 0.0 }),
        serde_json::json!({ "step": -1.0 }),
        serde_json::json!({ "tolerance": 0.0 }),
        serde_json::json!({ "tolerance": 1.0 }),
        serde_json::json!({ "max_iterations": 0 }),
        serde_json::json!({ "max_iterations": MAX_ITERATIONS_LIMIT + 1 }),
    ] {
        let err = run(w, h, init.clone(), target.clone(), mask.clone(), &bad)
            .expect_err("invalid params must be rejected");
        assert_eq!(err.class, ErrorClass::Schema, "for {bad}");
    }
}

#[test]
fn infer_outputs_validates_and_describes() {
    let op = LocalOptimize::new();
    let mut inputs = Descriptors::new();
    inputs.insert(
        "init".to_owned(),
        ResourceDescriptor::Image(ImageDescriptor {
            extent: Extent::new(4, 4),
            layout: ChannelLayout::Gray,
            scalar: ScalarType::F32,
            color: ColorEncoding::LinearSrgb,
            range: ColorRange::SceneReferred,
            alpha: AlphaRepresentation::Straight,
            coordinates: CoordinateConvention::PixelCenterUpperLeft,
            semantic: SemanticRole::Color,
        }),
    );
    inputs.insert(
        "target".to_owned(),
        ResourceDescriptor::Image(ImageDescriptor {
            extent: Extent::new(4, 4),
            layout: ChannelLayout::Gray,
            scalar: ScalarType::F32,
            color: ColorEncoding::LinearSrgb,
            range: ColorRange::SceneReferred,
            alpha: AlphaRepresentation::Straight,
            coordinates: CoordinateConvention::PixelCenterUpperLeft,
            semantic: SemanticRole::Color,
        }),
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
    let out = op
        .infer_outputs(&inputs, &serde_json::json!({}))
        .expect("infer outputs");
    assert!(matches!(
        out.get("candidate"),
        Some(ResourceDescriptor::Image(_))
    ));
    assert!(matches!(
        out.get("report"),
        Some(ResourceDescriptor::Report(_))
    ));

    // required_inputs covers the full extent of each port.
    let regions = op
        .required_inputs(&OutputRegions::new(), &inputs, &serde_json::json!({}))
        .expect("required inputs");
    assert_eq!(regions.len(), 3);
}
