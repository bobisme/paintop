//! Verification suite for `field.reaction_diffusion@1` (`OP_CATALOG` §11):
//!
//! - **schema/contract**: the manifest validates and matches its contract;
//! - **seeded determinism**: a rerun with the same seed and step count is
//!   bit-identical (M4 exit criterion 2);
//! - **stability guard**: an unstable `(Du, Dv, dt)` is rejected, not integrated
//!   into a `NaN`;
//! - **report metrics**: the report carries `SolverData` with the step count, the
//!   stability number/limit, and a residual history of the right length;
//! - **pattern statistics**: a known Turing parameter set drives the `v` field
//!   away from its uniform initial state into bounded, structured values.

use paintop_core::executor::{InputValues, OpImplementation, ResourceValue};
use paintop_ir::{
    AlphaRepresentation, ChannelLayout, ColorEncoding, ColorRange, CoordinateConvention, Extent,
    ImageDescriptor, ResourceDescriptor, ScalarType, SemanticRole,
};

use super::{REACTION_DIFFUSION_OP_ID, ReactionDiffusion};

/// A blank image of the given extent, used only for its extent.
fn extent_image(width: u32, height: u32) -> ResourceValue {
    let descriptor = ResourceDescriptor::Image(ImageDescriptor {
        extent: Extent::new(width, height),
        layout: ChannelLayout::Rgba,
        scalar: ScalarType::F32,
        color: ColorEncoding::LinearSrgb,
        range: ColorRange::SceneReferred,
        alpha: AlphaRepresentation::Straight,
        coordinates: CoordinateConvention::PixelCenterUpperLeft,
        semantic: SemanticRole::Color,
    });
    let samples = vec![0.0_f32; (width * height * 4) as usize];
    ResourceValue::new(descriptor, 4, samples).expect("image value")
}

/// Run the solver and return (u, v, report-value).
fn run(width: u32, height: u32, params: &serde_json::Value) -> (Vec<f32>, Vec<f32>, ResourceValue) {
    let mut inputs = InputValues::new();
    inputs.insert("extent_from".to_owned(), extent_image(width, height));
    let mut out = ReactionDiffusion::new()
        .compute(&inputs, params)
        .expect("solver computes");
    let u = out.remove("u").expect("u output");
    let v = out.remove("v").expect("v output");
    let report = out.remove("report").expect("report output");
    (u.samples().to_vec(), v.samples().to_vec(), report)
}

#[test]
fn manifest_matches_contract() {
    let manifest = ReactionDiffusion::manifest().expect("manifest builds");
    assert_eq!(manifest.id.to_string(), REACTION_DIFFUSION_OP_ID);
    assert_eq!(manifest.outputs.len(), 3);
}

#[test]
fn reruns_are_bit_identical() {
    let params = serde_json::json!({"steps": 50, "seed": 7});
    let (u1, v1, _) = run(32, 32, &params);
    let (u2, v2, _) = run(32, 32, &params);
    assert_eq!(u1, u2, "u must be bit-identical on rerun");
    assert_eq!(v1, v2, "v must be bit-identical on rerun");
}

#[test]
fn a_different_seed_changes_the_result() {
    let (_, v_a, _) = run(32, 32, &serde_json::json!({"steps": 80, "seed": 1}));
    let (_, v_b, _) = run(32, 32, &serde_json::json!({"steps": 80, "seed": 2}));
    assert_ne!(v_a, v_b, "a different seed must change the v field");
}

#[test]
fn unstable_step_is_rejected() {
    // Du = 0.4, dt = 1 => stability number 0.4*1*4 = 1.6 > 1: must be rejected.
    let mut inputs = InputValues::new();
    inputs.insert("extent_from".to_owned(), extent_image(8, 8));
    let result = ReactionDiffusion::new().compute(
        &inputs,
        &serde_json::json!({"diffusion_u": 0.4, "dt": 1.0, "steps": 10}),
    );
    assert!(
        result.is_err(),
        "an explicitly unstable request must be rejected"
    );
}

#[test]
fn marginally_stable_step_is_accepted() {
    // Du = 0.25, dt = 1 => 0.25*1*4 = 1.0 == limit: accepted.
    let mut inputs = InputValues::new();
    inputs.insert("extent_from".to_owned(), extent_image(8, 8));
    let result = ReactionDiffusion::new().compute(
        &inputs,
        &serde_json::json!({"diffusion_u": 0.25, "diffusion_v": 0.1, "dt": 1.0, "steps": 5}),
    );
    assert!(
        result.is_ok(),
        "a step at the stability limit must be accepted"
    );
}

#[test]
fn report_carries_solver_metrics() {
    let steps = 40_u32;
    let (_, _, report) = run(24, 24, &serde_json::json!({"steps": steps, "seed": 3}));
    let report = report.as_report().expect("report payload");
    let solver = report.solver.as_ref().expect("solver data present");
    assert_eq!(solver.kind, "gray-scott");
    assert_eq!(solver.steps, steps);
    assert_eq!(
        solver.residual_history.len(),
        steps as usize,
        "the residual history must record every step"
    );
    assert!(
        solver.stable,
        "the default parameters are within the stability bound"
    );
    assert!(
        solver.stability_number <= solver.stability_limit,
        "the realized stability number must respect the limit"
    );
    assert!(
        solver
            .residual_history
            .iter()
            .all(|r| r.is_finite() && *r >= 0.0),
        "every residual must be a finite non-negative L2 norm"
    );
    assert!(
        solver.total_energy.is_finite(),
        "the total energy must be finite"
    );
}

#[test]
fn solution_stays_finite_and_bounded() {
    let (u, v, _) = run(32, 32, &serde_json::json!({"steps": 120, "seed": 5}));
    for &s in u.iter().chain(&v) {
        assert!(s.is_finite(), "every sample must be finite");
        // Gray-Scott concentrations stay in a small band around [0, 1]; allow a
        // generous bound to catch a divergence without over-fitting the dynamics.
        assert!(
            (-0.5..=1.5).contains(&s),
            "sample {s} drifted out of the expected band"
        );
    }
}

#[test]
fn turing_parameters_break_the_uniform_state() {
    // The seeded perturbation under a known spot-forming parameter set must drive
    // the v field away from its (near-)uniform initial background: the spatial
    // variance after many steps must be meaningfully non-zero.
    let (_, v, _) = run(
        48,
        48,
        &serde_json::json!({
            "steps": 400, "seed": 11,
            "feed": 0.035, "kill": 0.065,
            "diffusion_u": 0.16, "diffusion_v": 0.08, "dt": 1.0
        }),
    );
    let count = f64::from(u32::try_from(v.len()).expect("pixel count fits u32"));
    let mean = v.iter().map(|&s| f64::from(s)).sum::<f64>() / count;
    let variance = v
        .iter()
        .map(|&s| {
            let d = f64::from(s) - mean;
            d * d
        })
        .sum::<f64>()
        / count;
    assert!(
        variance > 1e-5,
        "a Turing parameter set must produce spatial structure, got variance {variance}"
    );
}
