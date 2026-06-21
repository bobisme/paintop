//! Cross-backend differential + benchmark for the separable Gaussian (bn-u6g;
//! `plan.md` §12.2).
//!
//! `filter.gaussian_blur@1` carries its `cpu.reference` direct-convolution oracle
//! (an `O(r²)` 2-D sampled Gaussian) plus a `cpu.optimized` **separable** backend
//! (two `O(r)` 1-D passes). The two are *the same kernel*: the reference's 2-D
//! sum over the `(2r+1)²` square factorizes, so the separable product is
//! algebraically identical, differing only by f64 reassociation across the two
//! passes — exactly what the op's [`Bounded`](paintop_ir::DeterminismTier::Bounded)
//! tier permits, and why `gaussian_blur` was declared bounded in M1.
//!
//! This suite drives the cluster-1 differential harness
//! ([`paintop_testkit::differential`]) against the *real* op registries built by
//! [`paintop_cpu::registry`], comparing the separable backend to the oracle:
//!
//! * across a sigma sweep (small sub-pixel → large), so both the σ→0 identity and
//!   the large-radius regime are covered;
//! * across every boundary mode (clamp / mirror / wrap / constant / transparent),
//!   so the per-axis boundary handling is proven to reproduce the oracle;
//! * for both an RGBA image and a single-channel field, exercising the alpha /
//!   gray channel paths;
//!
//! plus a benchmark proving the separable backend is **meaningfully faster for
//! large sigma**, and a determinism rerun proving a fixed backend is bit-identical
//! across runs.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "an integration test crate building exact fixtures from integer indices"
)]

use std::time::Instant;

use paintop_core::executor::value::ResourceValue;
use paintop_core::executor::{BackendId, ImplRegistry, InputValues, OpImplementation};
use paintop_cpu::gaussian_blur::{GaussianBlur, GaussianBlurOptimized};
use paintop_cpu::registry::{implementation_registry, operation_registry};
use paintop_ir::{
    AlphaRepresentation, ChannelLayout, ColorEncoding, ColorRange, CoordinateConvention,
    DeterminismTier, Extent, FieldArity, FieldDescriptor, ImageDescriptor, OpId, OperationRegistry,
    ResourceDescriptor, ScalarType, SemanticRole,
};
use paintop_testkit::differential::{
    AllAvailable, DifferentialReport, OpInvocation, differential_check,
};

const GAUSSIAN_BLUR_OP_ID: &str = "filter.gaussian_blur@1";

/// A deterministic pseudo-random buffer in `[0, 1)` (xorshift64*), so the inputs
/// are non-degenerate (a spatially varying image, not a constant) without a `rand`
/// dependency.
fn fill(len: usize, seed: u64) -> Vec<f32> {
    let mut state = seed | 1;
    let mut out = Vec::with_capacity(len);
    for _ in 0..len {
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        let r = state.wrapping_mul(0x2545_f491_4f6c_dd1d);
        let mantissa = u16::try_from((r >> 40) & 0xFFFF).unwrap_or(0);
        out.push(f32::from(mantissa) / f32::from(u16::MAX));
    }
    out
}

/// An RGBA linear-sRGB color image value of `extent`.
fn rgba(extent: Extent, samples: Vec<f32>) -> ResourceValue {
    let descriptor = ResourceDescriptor::Image(ImageDescriptor {
        extent,
        layout: ChannelLayout::Rgba,
        scalar: ScalarType::F32,
        color: ColorEncoding::LinearSrgb,
        range: ColorRange::SceneReferred,
        alpha: AlphaRepresentation::Premultiplied,
        coordinates: CoordinateConvention::PixelCenterUpperLeft,
        semantic: SemanticRole::Color,
    });
    ResourceValue::new(descriptor, 4, samples).expect("sized rgba")
}

/// A single-channel `Field1` value of `extent` (the blur's other supported kind).
fn field1(extent: Extent, samples: Vec<f32>) -> ResourceValue {
    let descriptor = ResourceDescriptor::Field1(FieldDescriptor {
        arity: FieldArity::Field1,
        extent,
        scalar: ScalarType::F32,
        semantic: SemanticRole::Material,
        coordinates: CoordinateConvention::PixelCenterUpperLeft,
        space: None,
        normalization: None,
        encoding: None,
    });
    ResourceValue::new(descriptor, 1, samples).expect("sized field1")
}

/// Run the differential harness for the Gaussian with the given input/params,
/// treating every backend as available (the separable backend is pure-CPU).
fn check(
    manifests: &OperationRegistry,
    impls: &ImplRegistry,
    input: ResourceValue,
    params: &serde_json::Value,
) -> DifferentialReport {
    let op: OpId = GAUSSIAN_BLUR_OP_ID.parse().expect("op");
    let mut inputs = InputValues::new();
    inputs.insert("input".to_owned(), input);
    let invocation = OpInvocation {
        inputs: &inputs,
        params,
        output_port: "output",
    };
    differential_check(&op, manifests, impls, &invocation, &AllAvailable, None)
        .expect("differential check runs")
}

/// Assert the optimized (separable) backend was actually exercised (not silently
/// skipped) and matched the oracle within the op's bounded tier tolerance.
fn assert_optimized_passed(report: &DifferentialReport) {
    let optimized = BackendId::new("cpu", "optimized");
    let (_, outcome) = report
        .backends
        .iter()
        .find(|(b, _)| b == &optimized)
        .expect("cpu.optimized separable backend present in the report");
    assert!(
        outcome.is_pass(),
        "separable Gaussian diverged from the oracle (tolerance {:?}): {:?}",
        report.tolerance,
        outcome.error_map(),
    );
    assert!(report.all_pass(), "every backend passed");
}

#[test]
fn separable_matches_reference_across_sigmas_and_modes() {
    let manifests = operation_registry().expect("ops");
    let impls = implementation_registry().expect("impls");
    assert_eq!(
        manifests
            .get(&GAUSSIAN_BLUR_OP_ID.parse::<OpId>().unwrap())
            .unwrap()
            .determinism,
        DeterminismTier::Bounded,
        "gaussian_blur is bounded-tier, so the separable backend is held to the bounded envelope"
    );

    let extent = Extent::new(24, 18);
    // A spatially varying RGBA image so the blur actually mixes neighbours; a flat
    // image would pass trivially.
    let samples = fill((extent.width * extent.height * 4) as usize, 0x6A5);

    // A sigma sweep: sub-cutoff identity, a small radius, and large radii where the
    // separable win is largest.
    for &sigma in &[0.0005_f64, 0.4, 0.8, 1.5, 3.0, 6.0, 9.0] {
        for mode in ["clamp", "mirror", "wrap", "constant", "transparent"] {
            let params = serde_json::json!({ "sigma": sigma, "mode": mode });
            let report = check(&manifests, &impls, rgba(extent, samples.clone()), &params);
            assert_optimized_passed(&report);
            let map = report.backends[0].1.error_map().expect("error map");
            // The separable backend is well inside the bounded envelope for every
            // sigma/mode (the kernels are algebraically identical).
            assert!(
                map.max_abs <= report.tolerance.max_abs,
                "sigma {sigma} mode {mode}: max_abs {} exceeds bounded envelope {}",
                map.max_abs,
                report.tolerance.max_abs
            );
        }
    }
}

#[test]
fn separable_matches_reference_for_a_single_channel_field() {
    let manifests = operation_registry().expect("ops");
    let impls = implementation_registry().expect("impls");

    let extent = Extent::new(20, 20);
    let samples = fill((extent.width * extent.height) as usize, 0x7B6);
    for &sigma in &[1.0_f64, 4.0, 8.0] {
        for mode in ["clamp", "mirror", "wrap"] {
            let params = serde_json::json!({ "sigma": sigma, "mode": mode });
            let report = check(&manifests, &impls, field1(extent, samples.clone()), &params);
            assert_optimized_passed(&report);
        }
    }
}

#[test]
fn separable_default_mode_matches_reference() {
    // No explicit mode => clamp default on both backends.
    let manifests = operation_registry().expect("ops");
    let impls = implementation_registry().expect("impls");
    let extent = Extent::new(16, 16);
    let samples = fill((extent.width * extent.height * 4) as usize, 0x8C7);
    let params = serde_json::json!({ "sigma": 2.5 });
    let report = check(&manifests, &impls, rgba(extent, samples), &params);
    assert_optimized_passed(&report);
}

#[test]
fn separable_is_deterministic_on_reruns() {
    // A fixed backend is bit-identical across runs (the determinism requirement).
    let extent = Extent::new(18, 14);
    let samples = fill((extent.width * extent.height * 4) as usize, 0x9D8);
    let input = rgba(extent, samples);
    let params = serde_json::json!({ "sigma": 4.0, "mode": "mirror" });

    let mut inputs = InputValues::new();
    inputs.insert("input".to_owned(), input);
    let opt = GaussianBlurOptimized::new();
    let first = opt
        .compute(&inputs, &params)
        .expect("first run")
        .remove("output")
        .expect("output");
    for _ in 0..4 {
        let again = opt
            .compute(&inputs, &params)
            .expect("rerun")
            .remove("output")
            .expect("output");
        assert_eq!(
            first.samples(),
            again.samples(),
            "the separable backend must be bit-identical across reruns"
        );
    }
}

/// Time a closure as the median of repeated runs (nanoseconds per run), with a
/// short untimed warmup so steady-state throughput is measured.
fn time_median<F: FnMut()>(repeats: usize, mut f: F) -> f64 {
    for _ in 0..2 {
        f();
    }
    let mut samples = Vec::with_capacity(repeats);
    for _ in 0..repeats {
        let start = Instant::now();
        f();
        samples.push(start.elapsed().as_nanos() as f64);
    }
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    samples[samples.len() / 2]
}

#[test]
fn separable_is_faster_than_reference_for_large_sigma() {
    // For a large sigma the reference is O(r²) per pixel and the separable is O(r),
    // so the separable should be substantially faster. Use a real image so the
    // per-pixel kernel cost dominates the measurement.
    let extent = Extent::new(96, 96);
    let samples = fill((extent.width * extent.height * 4) as usize, 0xBE9);
    let input = rgba(extent, samples);
    let mut inputs = InputValues::new();
    inputs.insert("input".to_owned(), input);

    // sigma 12 => radius 36 => reference kernel 73x73 (5329 taps) vs separable
    // 2*73 = 146 taps: a >30x tap-count reduction.
    let params = serde_json::json!({ "sigma": 12.0, "mode": "clamp" });

    let reference = GaussianBlur::new();
    let optimized = GaussianBlurOptimized::new();

    let ref_ns = time_median(5, || {
        let _ = std::hint::black_box(reference.compute(&inputs, &params).expect("ref"));
    });
    let opt_ns = time_median(5, || {
        let _ = std::hint::black_box(optimized.compute(&inputs, &params).expect("opt"));
    });

    let speedup = ref_ns / opt_ns;
    // The asymptotic win is ~r/2; demand a conservative, machine-robust 3x so the
    // gate proves a real speedup (not a tie) without being brittle on slow CI.
    assert!(
        speedup >= 3.0,
        "separable Gaussian was not meaningfully faster for large sigma: \
         reference {ref_ns:.0}ns vs separable {opt_ns:.0}ns (speedup {speedup:.2}x)"
    );
}
