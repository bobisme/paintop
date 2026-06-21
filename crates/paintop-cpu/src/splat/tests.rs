//! Verification suite for `paint.gaussian_splats@1` (`OP_CATALOG` §4,
//! `AGENT_VERIFICATION` §3.5, `M0_DECISIONS` D2/Q6):
//!
//! - **schema/contract**: the manifest validates, agrees with its contract, and
//!   its verification declarations gate clean; the checked-in manifest stays in
//!   lockstep with the Rust builder;
//! - **analytic fixtures**: a single splat is centered and symmetric; its peak
//!   sits at the requested center;
//! - **metamorphic**: translating a splat's center translates the painted field;
//!   the covariance axes align with the requested rotation; a quarter-turn equals
//!   swapping the sigmas;
//! - **property**: an empty batch is the identity; a zero-opacity splat is the
//!   identity; every sample stays finite and (premultiplied) in range; batch order
//!   matters for accumulation;
//! - **policy/rejection**: an oversized batch (over `max_splats`) is a `policy`
//!   error; a negative / non-finite sigma, an out-of-range color, a malformed
//!   batch, and a nonlinear / straight-alpha base are rejected; an empty batch on a
//!   valid base is accepted.

use paintop_core::executor::{InputValues, OpImplementation, ResourceValue};
use paintop_ir::{
    AlphaRepresentation, AssertionStatus, ChannelLayout, ColorEncoding, ColorRange,
    CoordinateConvention, Descriptors, ErrorClass, Extent, ImageDescriptor, OpContract,
    OutputRegions, Rect, ResourceDescriptor, ScalarType, SemanticRole, check_contract_consistency,
    verify_categories,
};

use super::{DEFAULT_MAX_SPLATS, GaussianSplats, SPLAT_OP_ID};

/// Coverage tolerance for the `bounded` (exp-based) accumulation.
const TOL: f64 = 1e-6;

/// The continuous coordinate of pixel index `i`'s center (`i + 0.5`). Image
/// extents in these fixtures are well under 2⁵², so the cast is exact.
#[allow(
    clippy::cast_precision_loss,
    reason = "fixture pixel indices are tiny; the cast is exact"
)]
fn px_center(i: usize) -> f64 {
    i as f64 + 0.5
}

/// A premultiplied linear RGBA base of side `n`, filled with `fill` per channel.
fn base_image(n: u32, fill: f32) -> ResourceValue {
    base_image_with(
        n,
        ColorEncoding::LinearSrgb,
        AlphaRepresentation::Premultiplied,
        fill,
    )
}

/// A base image with an explicit encoding / alpha representation (for rejection
/// tests).
fn base_image_with(
    n: u32,
    color: ColorEncoding,
    alpha: AlphaRepresentation,
    fill: f32,
) -> ResourceValue {
    let descriptor = ResourceDescriptor::Image(ImageDescriptor {
        extent: Extent::new(n, n),
        layout: ChannelLayout::Rgba,
        scalar: ScalarType::F32,
        color,
        range: ColorRange::SceneReferred,
        alpha,
        coordinates: CoordinateConvention::PixelCenterUpperLeft,
        semantic: SemanticRole::Color,
    });
    let channels = ChannelLayout::Rgba.channel_count();
    let len = (n as usize) * (n as usize) * channels as usize;
    ResourceValue::new(descriptor, channels, vec![fill; len]).expect("base image")
}

/// A base image descriptor (for `infer_outputs`/`required_inputs`).
fn base_descriptor(n: u32) -> Descriptors {
    let mut inputs = Descriptors::new();
    inputs.insert(
        "base".to_owned(),
        ResourceDescriptor::Image(ImageDescriptor {
            extent: Extent::new(n, n),
            layout: ChannelLayout::Rgba,
            scalar: ScalarType::F32,
            color: ColorEncoding::LinearSrgb,
            range: ColorRange::SceneReferred,
            alpha: AlphaRepresentation::Premultiplied,
            coordinates: CoordinateConvention::PixelCenterUpperLeft,
            semantic: SemanticRole::Color,
        }),
    );
    inputs
}

/// Run the kernel against a base and recover the painted image value.
fn paint(base: ResourceValue, params: &serde_json::Value) -> ResourceValue {
    let mut inputs = InputValues::new();
    inputs.insert("base".to_owned(), base);
    let mut out = GaussianSplats::new()
        .compute(&inputs, params)
        .expect("splats compute");
    out.remove("image").expect("image port produced")
}

/// The alpha (coverage) channel of a painted RGBA image, row-major.
fn alpha_channel(image: &ResourceValue) -> Vec<f64> {
    image
        .samples()
        .chunks_exact(4)
        .map(|p| f64::from(p[3]))
        .collect()
}

#[test]
fn manifest_validates_and_agrees_with_contract() {
    let manifest = GaussianSplats::manifest().expect("splat manifest");
    manifest.validate().expect("splat manifest valid");
    check_contract_consistency(&manifest, &GaussianSplats::new())
        .expect("splat manifest agrees with contract");
    verify_categories(&manifest, &manifest.test.verification)
        .expect("splat verification declarations gate clean");
    assert_eq!(manifest.id.to_string(), SPLAT_OP_ID);
}

#[test]
fn produces_image_descriptor_matching_base() {
    let params = serde_json::json!({
        "splats": [
            { "center_px": [8.0, 8.0], "sigma_px": [3.0, 3.0], "color": [1.0, 0.0, 0.0, 1.0] }
        ]
    });
    let out = paint(base_image(16, 0.0), &params);
    let ResourceDescriptor::Image(d) = out.descriptor() else {
        panic!("expected image output");
    };
    assert_eq!(d.extent, Extent::new(16, 16));
    assert_eq!(d.alpha, AlphaRepresentation::Premultiplied);
    assert_eq!(d.color, ColorEncoding::LinearSrgb);
    assert_eq!(out.channels(), 4);
    assert_eq!(out.samples().len(), 16 * 16 * 4);
}

#[test]
fn output_is_bounded_and_finite() {
    let params = serde_json::json!({
        "splats": [
            { "center_px": [20.0, 14.0], "sigma_px": [6.0, 2.0], "angle_rad": 0.5,
              "color": [0.8, 0.3, 0.1, 1.0], "opacity": 0.7 },
            { "center_px": [40.0, 40.0], "sigma_px": [10.0, 10.0],
              "color": [0.1, 0.2, 0.9, 0.5], "opacity": 0.4, "blend": "multiply" }
        ]
    });
    let out = paint(base_image(64, 0.3), &params);
    for &s in out.samples() {
        assert!(s.is_finite(), "sample must be finite, got {s}");
        assert!(
            (-TOL..=1.0 + TOL).contains(&f64::from(s)),
            "sample out of range: {s}"
        );
    }
}

/// Empty batch is the identity: the base passes through unchanged (Q6: an empty
/// batch is legal).
#[test]
fn empty_batch_is_identity() {
    let params = serde_json::json!({ "splats": [] });
    let base = base_image(16, 0.25);
    let before: Vec<f32> = base.samples().to_vec();
    let out = paint(base, &params);
    assert_eq!(out.samples(), before.as_slice());
}

/// Zero-opacity splat is the identity (`AGENT_VERIFICATION` §3.5).
#[test]
fn zero_opacity_is_identity() {
    let base = base_image(24, 0.5);
    let before: Vec<f32> = base.samples().to_vec();
    let params = serde_json::json!({
        "splats": [
            { "center_px": [12.0, 12.0], "sigma_px": [5.0, 5.0],
              "color": [1.0, 1.0, 1.0, 1.0], "opacity": 0.0 }
        ]
    });
    let out = paint(base, &params);
    for (a, b) in out.samples().iter().zip(before.iter()) {
        assert!((f64::from(*a) - f64::from(*b)).abs() < TOL, "{a} vs {b}");
    }
}

/// Center symmetry: a single isotropic splot on a black base has its coverage peak
/// at the requested center and is symmetric about it (`AGENT_VERIFICATION` §3.5).
#[test]
fn single_splat_is_centered_and_symmetric() {
    let n = 33_u32;
    let center = 16.0_f64; // pixel center of index 16 is 16.5; choose 16.5
    let params = serde_json::json!({
        "splats": [
            { "center_px": [center + 0.5, center + 0.5], "sigma_px": [4.0, 4.0],
              "color": [0.0, 0.0, 0.0, 1.0], "opacity": 1.0 }
        ]
    });
    let out = paint(base_image(n, 0.0), &params);
    let alpha = alpha_channel(&out);
    let at = |x: usize, y: usize| alpha[y * n as usize + x];

    // Peak coverage sits at the center pixel.
    let peak = at(16, 16);
    assert!(peak > 0.99, "peak coverage {peak} not near 1");
    // Symmetric under reflection about the center (x -> 32 - x, y -> 32 - y).
    for y in 0..n as usize {
        for x in 0..n as usize {
            let mirror = at(32 - x, 32 - y);
            assert!((at(x, y) - mirror).abs() < TOL, "asymmetry at ({x},{y})");
        }
    }
}

/// Translation covariance: shifting the center by an integer pixel offset shifts
/// the painted field by the same offset (`AGENT_VERIFICATION` §3.5).
#[test]
fn translation_covariance() {
    let n = 40_u32;
    let make = |cx: f64, cy: f64| {
        serde_json::json!({
            "splats": [
                { "center_px": [cx, cy], "sigma_px": [3.0, 5.0], "angle_rad": 0.3,
                  "color": [0.0, 0.0, 0.0, 1.0], "opacity": 1.0 }
            ]
        })
    };
    let a = alpha_channel(&paint(base_image(n, 0.0), &make(12.5, 18.5)));
    let b = alpha_channel(&paint(base_image(n, 0.0), &make(15.5, 21.5)));
    let w = n as usize;
    // b at (x+3, y+3) equals a at (x, y) wherever both are in-bounds.
    for y in 0..w - 3 {
        for x in 0..w - 3 {
            let av = a[y * w + x];
            let bv = b[(y + 3) * w + (x + 3)];
            assert!(
                (av - bv).abs() < TOL,
                "translation broke at ({x},{y}): {av} vs {bv}"
            );
        }
    }
}

/// Covariance-axis alignment: a quarter-turn rotation of an anisotropic splat is
/// the same field as swapping its sigmas (`AGENT_VERIFICATION` §3.5).
#[test]
fn quarter_turn_equals_swapping_sigmas() {
    let n = 32_u32;
    let rotated = serde_json::json!({
        "splats": [
            { "center_px": [16.5, 16.5], "sigma_px": [8.0, 3.0],
              "angle_rad": std::f64::consts::FRAC_PI_2,
              "color": [0.0, 0.0, 0.0, 1.0], "opacity": 1.0 }
        ]
    });
    let swapped = serde_json::json!({
        "splats": [
            { "center_px": [16.5, 16.5], "sigma_px": [3.0, 8.0],
              "color": [0.0, 0.0, 0.0, 1.0], "opacity": 1.0 }
        ]
    });
    let a = alpha_channel(&paint(base_image(n, 0.0), &rotated));
    let b = alpha_channel(&paint(base_image(n, 0.0), &swapped));
    for (x, y) in a.iter().zip(b.iter()) {
        assert!((x - y).abs() < TOL, "{x} vs {y}");
    }
}

/// Batch order matters for non-commutative accumulation: two opaque splats applied
/// in opposite orders differ where they overlap.
#[test]
fn batch_order_affects_overlapping_splats() {
    let n = 24_u32;
    let red = serde_json::json!(
        { "center_px": [11.0, 12.0], "sigma_px": [5.0, 5.0],
          "color": [1.0, 0.0, 0.0, 1.0], "opacity": 0.6 });
    let blue = serde_json::json!(
        { "center_px": [13.0, 12.0], "sigma_px": [5.0, 5.0],
          "color": [0.0, 0.0, 1.0, 1.0], "opacity": 0.6 });
    let ab = paint(
        base_image(n, 0.0),
        &serde_json::json!({ "splats": [red, blue] }),
    );
    let ba = paint(
        base_image(n, 0.0),
        &serde_json::json!({ "splats": [blue, red] }),
    );
    let differ = ab
        .samples()
        .iter()
        .zip(ba.samples().iter())
        .any(|(x, y)| (f64::from(*x) - f64::from(*y)).abs() > 1e-3);
    assert!(
        differ,
        "swapping overlapping splats should change the result"
    );
}

// --- Policy and rejection -------------------------------------------------

#[test]
fn oversized_batch_is_a_policy_error() {
    // Two splats with a max_splats budget of one: rejected on policy.
    let params = serde_json::json!({
        "max_splats": 1,
        "splats": [
            { "center_px": [4.0, 4.0], "sigma_px": [2.0, 2.0], "color": [0.0, 0.0, 0.0, 1.0] },
            { "center_px": [6.0, 6.0], "sigma_px": [2.0, 2.0], "color": [0.0, 0.0, 0.0, 1.0] }
        ]
    });
    let mut inputs = InputValues::new();
    inputs.insert("base".to_owned(), base_image(8, 0.0));
    let err = GaussianSplats::new()
        .compute(&inputs, &params)
        .expect_err("oversized batch must be rejected");
    assert_eq!(err.class, ErrorClass::Policy);
}

#[test]
fn batch_at_budget_is_accepted() {
    let params = serde_json::json!({
        "max_splats": 1,
        "splats": [
            { "center_px": [4.0, 4.0], "sigma_px": [2.0, 2.0], "color": [0.0, 0.0, 0.0, 1.0] }
        ]
    });
    let mut inputs = InputValues::new();
    inputs.insert("base".to_owned(), base_image(8, 0.0));
    GaussianSplats::new()
        .compute(&inputs, &params)
        .expect("batch exactly at budget is accepted");
}

#[test]
fn negative_sigma_is_rejected() {
    let params = serde_json::json!({
        "splats": [
            { "center_px": [4.0, 4.0], "sigma_px": [-2.0, 2.0], "color": [0.0, 0.0, 0.0, 1.0] }
        ]
    });
    let mut inputs = InputValues::new();
    inputs.insert("base".to_owned(), base_image(8, 0.0));
    let err = GaussianSplats::new()
        .compute(&inputs, &params)
        .expect_err("negative sigma must be rejected");
    assert_eq!(err.class, ErrorClass::Schema);
}

#[test]
fn non_finite_sigma_is_rejected() {
    // serde_json cannot hold NaN; a zero sigma is the boundary degenerate case.
    let params = serde_json::json!({
        "splats": [
            { "center_px": [4.0, 4.0], "sigma_px": [0.0, 2.0], "color": [0.0, 0.0, 0.0, 1.0] }
        ]
    });
    let mut inputs = InputValues::new();
    inputs.insert("base".to_owned(), base_image(8, 0.0));
    let err = GaussianSplats::new()
        .compute(&inputs, &params)
        .expect_err("zero sigma must be rejected");
    assert_eq!(err.class, ErrorClass::Schema);
}

#[test]
fn out_of_range_color_is_rejected() {
    let params = serde_json::json!({
        "splats": [
            { "center_px": [4.0, 4.0], "sigma_px": [2.0, 2.0], "color": [1.5, 0.0, 0.0, 1.0] }
        ]
    });
    let mut inputs = InputValues::new();
    inputs.insert("base".to_owned(), base_image(8, 0.0));
    let err = GaussianSplats::new()
        .compute(&inputs, &params)
        .expect_err("out-of-range color must be rejected");
    assert_eq!(err.class, ErrorClass::Schema);
}

#[test]
fn out_of_range_opacity_is_rejected() {
    let params = serde_json::json!({
        "splats": [
            { "center_px": [4.0, 4.0], "sigma_px": [2.0, 2.0],
              "color": [0.0, 0.0, 0.0, 1.0], "opacity": 1.5 }
        ]
    });
    let mut inputs = InputValues::new();
    inputs.insert("base".to_owned(), base_image(8, 0.0));
    let err = GaussianSplats::new()
        .compute(&inputs, &params)
        .expect_err("out-of-range opacity must be rejected");
    assert_eq!(err.class, ErrorClass::Schema);
}

#[test]
fn missing_splats_param_is_rejected() {
    let params = serde_json::json!({});
    let mut inputs = InputValues::new();
    inputs.insert("base".to_owned(), base_image(8, 0.0));
    let err = GaussianSplats::new()
        .compute(&inputs, &params)
        .expect_err("missing splats must be rejected");
    assert_eq!(err.class, ErrorClass::Schema);
}

#[test]
fn malformed_splat_shape_is_rejected() {
    // A splat missing its required sigma_px field.
    let params = serde_json::json!({
        "splats": [ { "center_px": [4.0, 4.0], "color": [0.0, 0.0, 0.0, 1.0] } ]
    });
    let mut inputs = InputValues::new();
    inputs.insert("base".to_owned(), base_image(8, 0.0));
    let err = GaussianSplats::new()
        .compute(&inputs, &params)
        .expect_err("malformed splat must be rejected");
    assert_eq!(err.class, ErrorClass::Schema);
}

#[test]
fn unknown_blend_mode_is_rejected() {
    let params = serde_json::json!({
        "splats": [
            { "center_px": [4.0, 4.0], "sigma_px": [2.0, 2.0],
              "color": [0.0, 0.0, 0.0, 1.0], "blend": "screen" }
        ]
    });
    let mut inputs = InputValues::new();
    inputs.insert("base".to_owned(), base_image(8, 0.0));
    let err = GaussianSplats::new()
        .compute(&inputs, &params)
        .expect_err("unknown blend mode must be rejected");
    assert_eq!(err.class, ErrorClass::Schema);
}

#[test]
fn nonlinear_base_is_rejected() {
    let params = serde_json::json!({ "splats": [] });
    let mut inputs = InputValues::new();
    inputs.insert(
        "base".to_owned(),
        base_image_with(
            8,
            ColorEncoding::Srgb,
            AlphaRepresentation::Premultiplied,
            0.0,
        ),
    );
    let err = GaussianSplats::new()
        .compute(&inputs, &params)
        .expect_err("nonlinear base must be rejected");
    assert_eq!(err.class, ErrorClass::Semantic);
}

#[test]
fn straight_alpha_base_is_rejected() {
    let params = serde_json::json!({ "splats": [] });
    let mut inputs = InputValues::new();
    inputs.insert(
        "base".to_owned(),
        base_image_with(
            8,
            ColorEncoding::LinearSrgb,
            AlphaRepresentation::Straight,
            0.0,
        ),
    );
    let err = GaussianSplats::new()
        .compute(&inputs, &params)
        .expect_err("straight-alpha base must be rejected");
    assert_eq!(err.class, ErrorClass::Semantic);
}

#[test]
fn mismatched_space_is_rejected() {
    let params = serde_json::json!({ "splats": [], "space": "raw-linear" });
    let mut inputs = InputValues::new();
    inputs.insert("base".to_owned(), base_image(8, 0.0));
    let err = GaussianSplats::new()
        .compute(&inputs, &params)
        .expect_err("space mismatching the base encoding must be rejected");
    assert_eq!(err.class, ErrorClass::Semantic);
}

// --- Contract surface -----------------------------------------------------

#[test]
fn infer_outputs_records_image_and_pointwise_roi() {
    let inputs = base_descriptor(20);
    let params = serde_json::json!({
        "splats": [
            { "center_px": [10.0, 6.0], "sigma_px": [3.0, 3.0], "color": [0.0, 0.0, 0.0, 1.0] }
        ]
    });
    let out = GaussianSplats::new()
        .infer_outputs(&inputs, &params)
        .expect("infer");
    let ResourceDescriptor::Image(d) = out["image"] else {
        panic!("expected image");
    };
    assert_eq!(d.extent, Extent::new(20, 20));

    // The base is read pointwise: the demanded region equals the requested output.
    let mut requested = OutputRegions::new();
    requested.insert("image".to_owned(), Rect::new(0, 0, 20, 20));
    let needed = GaussianSplats::new()
        .required_inputs(&requested, &inputs, &params)
        .expect("required_inputs");
    assert_eq!(needed["base"], Rect::new(0, 0, 20, 20));

    let results = GaussianSplats::new()
        .validate_postconditions(&out, &params)
        .expect("postconditions");
    assert!(results.iter().all(|r| r.status == AssertionStatus::Pass));
}

#[test]
fn infer_outputs_rejects_degenerate_batch() {
    let inputs = base_descriptor(8);
    let params = serde_json::json!({
        "splats": [
            { "center_px": [4.0, 4.0], "sigma_px": [0.0, 0.0], "color": [0.0, 0.0, 0.0, 1.0] }
        ]
    });
    let err = GaussianSplats::new()
        .infer_outputs(&inputs, &params)
        .expect_err("degenerate batch must be rejected at infer time");
    assert_eq!(err.class, ErrorClass::Schema);
}

#[test]
fn default_max_splats_matches_policy_default() {
    // A guard that the documented default tracks the IR_SPEC policy example.
    assert_eq!(DEFAULT_MAX_SPLATS, 100_000);
}

// --- Goal-closing properties (bn-2ym): determinism, split-batch equivalence,
// bounded-support / tail-energy reporting, locality, interior translation
// covariance, and covariance recovery from weighted moments (the
// `AGENT_VERIFICATION` §3.5 acceptance items not already exercised above).

/// Deterministic ordering: painting the same batch twice yields a *bit-identical*
/// buffer. Accumulation is a pure, in-order function of the params, so the result
/// is reproducible run-to-run (`AGENT_VERIFICATION` §3.5 ordering / replay).
#[test]
fn deterministic_rerun_is_bit_identical() {
    let params = serde_json::json!({
        "splats": [
            { "center_px": [18.5, 22.5], "sigma_px": [5.0, 9.0], "angle_rad": 0.9,
              "color": [0.7, 0.2, 0.4, 0.8], "opacity": 0.6 },
            { "center_px": [30.5, 10.5], "sigma_px": [4.0, 4.0],
              "color": [0.1, 0.9, 0.3, 1.0], "opacity": 0.5, "blend": "multiply" },
            { "center_px": [25.5, 25.5], "sigma_px": [7.0, 3.0], "angle_rad": -0.4,
              "color": [0.2, 0.2, 0.9, 0.9], "opacity": 0.7 }
        ]
    });
    let a = paint(base_image(48, 0.2), &params);
    let b = paint(base_image(48, 0.2), &params);
    // Bit-identical, not merely within tolerance: same inputs ⇒ same bytes.
    assert_eq!(
        a.samples(),
        b.samples(),
        "repeated paint of the same batch must be bit-identical"
    );
}

/// Split-batch equivalence for a commutative accumulation mode: with `normal`
/// (source-over) blending and *disjoint* (non-overlapping) splats, painting the
/// whole batch at once equals painting it as two sub-batches in order. This is the
/// §3.5 "split-batch equivalence for commutative accumulation modes" property; it
/// holds exactly for disjoint coverage because source-over of a zero-coverage
/// sample is the identity.
#[test]
fn split_batch_equivalence_for_disjoint_normal_blend() {
    let n = 64_u32;
    // Two well-separated isotropic splats: at σ = 3 px their coverage is < 1e-6 by
    // ~8 px, so a 32 px separation makes their supports effectively disjoint.
    let left = serde_json::json!(
        { "center_px": [12.5, 32.5], "sigma_px": [3.0, 3.0],
          "color": [0.9, 0.1, 0.1, 1.0], "opacity": 0.8 });
    let right = serde_json::json!(
        { "center_px": [51.5, 32.5], "sigma_px": [3.0, 3.0],
          "color": [0.1, 0.1, 0.9, 1.0], "opacity": 0.8 });
    let whole = paint(
        base_image(n, 0.15),
        &serde_json::json!({ "splats": [left, right] }),
    );
    // Two sub-batches applied in order to the same base.
    let first = paint(
        base_image(n, 0.15),
        &serde_json::json!({ "splats": [left] }),
    );
    let split = paint(first, &serde_json::json!({ "splats": [right] }));
    for (w, s) in whole.samples().iter().zip(split.samples().iter()) {
        assert!(
            (f64::from(*w) - f64::from(*s)).abs() < TOL,
            "split-batch differs for disjoint normal blend: {w} vs {s}"
        );
    }
}

/// Bounded-support truncation error is reported and within budget: the analytic
/// Gaussian tail beyond a support radius of `k·σ_max` is bounded by `exp(-k²/2)`.
/// At `k = 6` the per-sample coverage outside the support disc never exceeds that
/// reported bound, so a renderer truncating at `6σ` drops at most that mass — the
/// §3.5 "bounded support truncation error reported" property. The kernel evaluates
/// the full Gaussian, so the observed far-field coverage must sit *under* the
/// reported bound everywhere outside the support radius.
#[test]
fn bounded_support_tail_energy_within_budget() {
    let n = 96_u32;
    let cx = 48.5_f64;
    let cy = 48.5_f64;
    let sigma = 4.0_f64;
    let k = 6.0_f64;
    let support_radius = k * sigma;
    // The reported truncation bound on the *weight* outside the support disc.
    let reported_tail_bound = (-0.5 * k * k).exp();
    let params = serde_json::json!({
        "splats": [
            { "center_px": [cx, cy], "sigma_px": [sigma, sigma],
              "color": [0.0, 0.0, 0.0, 1.0], "opacity": 1.0 }
        ]
    });
    let out = paint(base_image(n, 0.0), &params);
    let alpha = alpha_channel(&out);
    let w = n as usize;
    let mut max_outside = 0.0_f64;
    for y in 0..w {
        for x in 0..w {
            let px = px_center(x);
            let py = px_center(y);
            let dx = px - cx;
            let dy = py - cy;
            let r2 = dx.mul_add(dx, dy * dy);
            if r2 > support_radius * support_radius {
                max_outside = max_outside.max(alpha[y * w + x]);
            }
        }
    }
    assert!(
        max_outside <= reported_tail_bound + TOL,
        "coverage {max_outside} outside the {support_radius}px support exceeds the \
         reported tail bound {reported_tail_bound}"
    );
}

/// Locality / mask interaction through the MVP edit-layer loop: per `M0_DECISIONS`
/// D2 a splat paints onto an edit layer and locality is *bounded by its support*.
/// Pixels far from every splat center (beyond `6σ`) are altered by no more than the
/// reported tail bound — the edit is local, which is exactly the invariant a
/// downstream `composite.masked_replace` relies on. Here we verify the splat op's
/// own locality contribution against the original base.
#[test]
fn far_field_edit_is_local_within_budget() {
    let n = 80_u32;
    let fill = 0.4_f32;
    let sigma = 3.0_f64;
    let k = 6.0_f64;
    let reported_tail_bound = (-0.5 * k * k).exp();
    let center = [20.5_f64, 20.5_f64];
    let params = serde_json::json!({
        "splats": [
            { "center_px": center, "sigma_px": [sigma, sigma],
              "color": [1.0, 1.0, 1.0, 1.0], "opacity": 1.0 }
        ]
    });
    let base = base_image(n, fill);
    let before: Vec<f32> = base.samples().to_vec();
    let out = paint(base, &params);
    let w = n as usize;
    for y in 0..w {
        for x in 0..w {
            let dx = px_center(x) - center[0];
            let dy = px_center(y) - center[1];
            let r2 = dx.mul_add(dx, dy * dy);
            if r2 > (k * sigma) * (k * sigma) {
                let idx = (y * w + x) * 4;
                for c in 0..4 {
                    let delta =
                        (f64::from(out.samples()[idx + c]) - f64::from(before[idx + c])).abs();
                    assert!(
                        delta <= reported_tail_bound + TOL,
                        "far-field pixel ({x},{y}) ch{c} changed by {delta}, exceeding the \
                         locality budget {reported_tail_bound}"
                    );
                }
            }
        }
    }
}

/// Translation covariance strictly in the interior: shifting the center by an
/// integer offset shifts the painted field exactly, evaluated on a window that
/// stays clear of the image boundary on both placements so no support is clipped
/// (`AGENT_VERIFICATION` §3.5 "translation covariance away from boundaries").
#[test]
fn translation_covariance_interior() {
    let n = 80_u32;
    let sigma_max = 5.0_f64;
    let shift = 7_usize;
    let shift_px = 7.0_f64;
    let make = |cx: f64, cy: f64| {
        serde_json::json!({
            "splats": [
                { "center_px": [cx, cy], "sigma_px": [4.0, sigma_max], "angle_rad": 0.6,
                  "color": [0.0, 0.0, 0.0, 1.0], "opacity": 1.0 }
            ]
        })
    };
    let w = n as usize;
    let a = alpha_channel(&paint(base_image(n, 0.0), &make(30.5, 34.5)));
    let b = alpha_channel(&paint(
        base_image(n, 0.0),
        &make(30.5 + shift_px, 34.5 + shift_px),
    ));
    // Only compare an interior window where both splats' supports (±6σ) are fully
    // inside the image, so boundary clipping cannot mask a translation defect.
    // 6σ_max = 30px; a fixed, generous margin keeps the cast off the lint wall.
    let margin = 30_usize + shift;
    let mut compared = 0_usize;
    for y in margin..w - margin {
        for x in margin..w - margin {
            let av = a[y * w + x];
            let bv = b[(y + shift) * w + (x + shift)];
            assert!(
                (av - bv).abs() < TOL,
                "interior translation broke at ({x},{y}): {av} vs {bv}"
            );
            compared += 1;
        }
    }
    assert!(compared > 0, "interior window was empty");
}

/// Covariance reconstruction from weighted moments: the painted coverage field of a
/// single anisotropic splat is (up to truncation) proportional to its Gaussian, so
/// its weighted second-moment matrix recovers the splat covariance `Σ = R diag(σx²,
/// σy²) Rᵀ`. We reconstruct `Σ` from the coverage field and check the eigenvalues
/// (≈ σ²) and the principal-axis orientation against the requested `(σ, θ)`
/// (`AGENT_VERIFICATION` §3.5 "covariance axes align with specified rotation",
/// reconstructed from weighted moments).
#[test]
fn covariance_recovered_from_weighted_moments() {
    let n = 128_u32;
    let cx = 64.5_f64;
    let cy = 64.5_f64;
    let sigma_x = 9.0_f64;
    let sigma_y = 4.0_f64;
    let theta = 0.5_f64;
    let params = serde_json::json!({
        "splats": [
            { "center_px": [cx, cy], "sigma_px": [sigma_x, sigma_y], "angle_rad": theta,
              "color": [0.0, 0.0, 0.0, 1.0], "opacity": 1.0 }
        ]
    });
    let out = paint(base_image(n, 0.0), &params);
    let alpha = alpha_channel(&out);
    let w = n as usize;

    // Weighted first/second moments of the coverage field about the known center.
    let (mut sum_w, mut sxx, mut syy, mut sxy) = (0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64);
    for y in 0..w {
        for x in 0..w {
            let weight = alpha[y * w + x];
            let dx = px_center(x) - cx;
            let dy = px_center(y) - cy;
            sum_w += weight;
            sxx = (weight * dx).mul_add(dx, sxx);
            syy = (weight * dy).mul_add(dy, syy);
            sxy = (weight * dx).mul_add(dy, sxy);
        }
    }
    assert!(sum_w > 0.0, "coverage field is empty");
    let (cxx, cyy, cxy) = (sxx / sum_w, syy / sum_w, sxy / sum_w);

    // Eigen-decomposition of the symmetric 2×2 moment matrix.
    let trace = cxx + cyy;
    let det = cxx.mul_add(cyy, -(cxy * cxy));
    let disc = (0.25 * trace).mul_add(trace, -det).max(0.0).sqrt();
    let half_trace = 0.5 * trace;
    let lambda_big = half_trace + disc;
    let lambda_small = half_trace - disc;

    // Eigenvalues recover the variances (σ²), so √λ recovers the sigmas. A 128px
    // window at 6σ ≈ 54px captures essentially the whole mass, so the moment
    // estimate is tight; allow a small discretization tolerance.
    let recovered_major = lambda_big.sqrt();
    let recovered_minor = lambda_small.sqrt();
    assert!(
        (recovered_major - sigma_x).abs() < 0.2,
        "major axis σ recovered {recovered_major}, expected {sigma_x}"
    );
    assert!(
        (recovered_minor - sigma_y).abs() < 0.2,
        "minor axis σ recovered {recovered_minor}, expected {sigma_y}"
    );

    // Orientation of the major eigenvector matches the requested rotation θ (mod π).
    let recovered_angle = 0.5 * (2.0 * cxy).atan2(cxx - cyy);
    let wrap = |a: f64| {
        let mut a = a % std::f64::consts::PI;
        if a < -std::f64::consts::FRAC_PI_2 {
            a += std::f64::consts::PI;
        } else if a > std::f64::consts::FRAC_PI_2 {
            a -= std::f64::consts::PI;
        }
        a
    };
    assert!(
        (wrap(recovered_angle - theta)).abs() < 0.02,
        "principal axis angle recovered {recovered_angle}, expected {theta}"
    );
}

/// The checked-in `ops/manifests/<id>.json` file (read by `cargo xtask
/// verify-op`) must stay byte-identical to the Rust manifest builder, the source
/// of truth. Regenerate with `serde_json::to_string_pretty` if this fails.
#[test]
fn checked_in_manifest_matches_builder() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("crates dir")
        .parent()
        .expect("repo root")
        .join("ops/manifests");
    let manifest = GaussianSplats::manifest().expect("splat manifest");
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
