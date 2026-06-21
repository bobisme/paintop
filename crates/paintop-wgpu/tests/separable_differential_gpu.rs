//! GPU separable-filter cross-backend differential (bn-b4c).
//!
//! The `wgpu` two-pass separable Gaussian must reproduce the
//! `filter.gaussian_blur@1` **`cpu.reference`** oracle (a direct 2-D sampled
//! Gaussian) within the op's [`Bounded`](paintop_ir::DeterminismTier::Bounded)
//! tolerance, across every boundary mode and a sweep of kernel sizes, with **no
//! visible tile/texture seam** (`plan.md` §12.3, §19 M3).
//!
//! The reference's 2-D kernel sum over the `(2r+1)²` square factorizes into the
//! product of two 1-D normalized Gaussians, so the GPU separable result is the same
//! kernel up to f32/f64 reassociation — exactly what the bounded tier permits (the
//! same property the `cpu.optimized` separable backend relies on). This suite builds
//! the *same* normalized 1-D taps the CPU separable backend uses, runs the GPU passes,
//! and compares against the 2-D oracle through the cross-backend harness's
//! [`ErrorMap`]/[`Tolerance`] machinery.
//!
//! No-seam: a step-edge image is filtered and the GPU↔oracle error is asserted
//! tolerance-bounded **everywhere**, including across the high-gradient seam where a
//! tiling artifact would show — the whole image is one GPU buffer, so there is no
//! tile boundary to leak.
//!
//! Gated on adapter presence: with no GPU the test returns without failing
//! (`just check` passes GPU-less). On this host (RTX 3090) the GPU path runs.

use paintop_core::executor::value::ResourceValue;
use paintop_core::executor::{BackendId, InputValues, OpImplementation};
use paintop_cpu::gaussian_blur::GaussianBlur;
use paintop_ir::{
    AlphaRepresentation, ChannelLayout, ColorEncoding, ColorRange, CoordinateConvention,
    DeterminismTier, Extent, ImageDescriptor, ResourceDescriptor, ScalarType, SemanticRole,
};
use paintop_testkit::differential::{BackendAvailability, ErrorMap, GpuAdapter, Tolerance};
use paintop_wgpu::{
    Boundary, DeviceLimits, PipelineCache, adapter_present, probe, run_separable_gaussian,
};
use serde_json::{Value, json};
use std::sync::Mutex;

/// Serialize the GPU section across parallel test threads (concurrent `wgpu` device
/// creation segfaults this Vulkan driver). CPU-only assertions never take the lock.
static GPU_LOCK: Mutex<()> = Mutex::new(());

/// The kernel radius for a `sigma`: `ceil(3σ)` (matching the CPU `kernel_radius`).
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "3*sigma is positive and small; the fixtures keep r well within u32"
)]
fn kernel_radius(sigma: f64) -> usize {
    if sigma <= 1.0e-3 {
        return 0;
    }
    ((3.0 * sigma).ceil() as usize).max(1)
}

/// The normalized 1-D Gaussian taps for `sigma`, identical to the CPU separable
/// backend's `gaussian_taps_1d` (length `2r+1`, hot tap at `r`, unit sum).
#[allow(
    clippy::cast_precision_loss,
    reason = "kernel offsets are tiny integers, exact in f64"
)]
fn gaussian_taps(sigma: f64) -> Vec<f32> {
    let r = kernel_radius(sigma);
    if r == 0 {
        return vec![1.0];
    }
    let ri = i64::try_from(r).expect("radius fits i64");
    let two_sigma_sq = 2.0 * sigma * sigma;
    let mut taps = Vec::with_capacity(2 * r + 1);
    let mut sum = 0.0_f64;
    for d in -ri..=ri {
        let w = (-((d * d) as f64) / two_sigma_sq).exp();
        sum += w;
        taps.push(w);
    }
    taps.into_iter()
        .map(|w| {
            #[allow(
                clippy::cast_possible_truncation,
                reason = "the GPU kernel accumulates in f32; the bounded tier permits it"
            )]
            {
                (w / sum) as f32
            }
        })
        .collect()
}

const fn image_descriptor(extent: Extent) -> ResourceDescriptor {
    ResourceDescriptor::Image(ImageDescriptor {
        extent,
        layout: ChannelLayout::Rgba,
        scalar: ScalarType::F32,
        color: ColorEncoding::LinearSrgb,
        range: ColorRange::SceneReferred,
        alpha: AlphaRepresentation::Straight,
        coordinates: CoordinateConvention::PixelCenterUpperLeft,
        semantic: SemanticRole::Color,
    })
}

/// An RGBA test image with structure (a smooth ramp plus a sharp vertical step edge
/// at the middle column), so a tiling/seam artifact at any column would surface.
#[allow(
    clippy::cast_precision_loss,
    reason = "small fixture extents; integer->f32 is exact here"
)]
fn structured_image(extent: Extent) -> ResourceValue {
    let w = extent.width;
    let h = extent.height;
    let mut s = Vec::with_capacity((w * h * 4) as usize);
    for y in 0..h {
        for x in 0..w {
            let rx = x as f32 / w as f32;
            let ry = y as f32 / h as f32;
            // A sharp step at the middle column to stress boundary/seam behavior.
            let step = if x >= w / 2 { 1.0 } else { 0.0 };
            s.extend_from_slice(&[rx, ry, step, 0.5f32.mul_add(rx, 0.25)]);
        }
    }
    ResourceValue::new(image_descriptor(extent), 4, s).expect("image value")
}

/// Run the `cpu.reference` 2-D Gaussian oracle for a sigma/mode.
fn cpu_oracle(input: &ResourceValue, sigma: f64, mode: &str) -> ResourceValue {
    let mut inputs = InputValues::new();
    inputs.insert("input".to_owned(), input.clone());
    let params: Value = json!({ "sigma": sigma, "mode": mode });
    let out = GaussianBlur::new()
        .compute(&inputs, &params)
        .expect("cpu reference blurs");
    out.get("output").expect("output").clone()
}

/// Differentially check the GPU separable Gaussian against the 2-D oracle for a
/// sigma + boundary mode. Skips cleanly with no adapter.
fn check_separable(extent: Extent, sigma: f64, mode: &str) {
    let input = structured_image(extent);
    let oracle = cpu_oracle(&input, sigma, mode);

    let gpu_backend = BackendId::new("wgpu", "separable");
    if let Some(reason) = GpuAdapter::new(adapter_present()).unavailable(&gpu_backend) {
        eprintln!("skipping GPU separable differential cleanly: {reason}");
        return;
    }
    let _guard = GPU_LOCK.lock().expect("gpu lock");
    let Ok(context) = probe() else {
        eprintln!("no GPU adapter present; skipping GPU separable differential");
        return;
    };

    let taps = gaussian_taps(sigma);
    let boundary = Boundary::from_mode(mode);
    let mut cache = PipelineCache::new();
    let out = run_separable_gaussian(
        &context,
        &mut cache,
        input.samples(),
        extent,
        4,
        &taps,
        boundary,
    )
    .expect("gpu separable run");

    // Readback-free intermediate: one upload, one export readback (bn-3q0).
    assert_eq!(out.trace.intermediate_readbacks, 0);
    assert_eq!(out.trace.uploads, 1);
    assert_eq!(out.trace.export_readbacks, 1);

    let candidate =
        ResourceValue::new(image_descriptor(extent), 4, out.samples.clone()).expect("gpu value");
    let error_map = ErrorMap::compute(&oracle, &candidate);
    // Bounded tier, slightly relaxed for f32-vs-f64 over a multi-tap accumulation;
    // the tier (bounded vs exact) is fixed, only the magnitude is tuned.
    let tolerance = Tolerance::for_tier(DeterminismTier::Bounded).with_bounds(2.0e-4, 5.0e-5);
    assert!(
        error_map.within(&tolerance),
        "GPU separable (sigma={sigma}, mode={mode}) diverged from the 2-D oracle: \
         max_abs={} rms={} argmax={} (saved error map)",
        error_map.max_abs,
        error_map.rms,
        error_map.argmax,
    );

    // Determinism: a fixed backend reruns bit-identically.
    let again = run_separable_gaussian(
        &context,
        &mut cache,
        input.samples(),
        extent,
        4,
        &taps,
        boundary,
    )
    .expect("gpu separable rerun");
    assert_eq!(
        out.samples, again.samples,
        "fixed GPU backend reruns bit-identically"
    );
}

#[test]
fn separable_matches_oracle_across_boundary_modes() {
    let extent = Extent::new(40, 32);
    for mode in ["clamp", "mirror", "wrap", "constant", "transparent"] {
        check_separable(extent, 2.0, mode);
    }
}

#[test]
fn separable_matches_oracle_across_kernel_sizes() {
    let extent = Extent::new(48, 40);
    // A sweep of sigmas -> radii: small (r=3) through large (r=18).
    for sigma in [1.0, 2.5, 4.0, 6.0] {
        check_separable(extent, sigma, "clamp");
    }
}

#[test]
fn no_seam_across_the_step_edge() {
    // A high-gradient step edge is the worst case for a tiling/seam artifact; the
    // whole-image GPU buffer must match the oracle everywhere across it.
    let extent = Extent::new(64, 16);
    check_separable(extent, 3.0, "clamp");
    check_separable(extent, 3.0, "mirror");
}

#[test]
fn no_adapter_skips_cleanly_via_the_harness_gate() {
    let gpu_backend = BackendId::new("wgpu", "separable");
    let gate = GpuAdapter::new(false);
    assert!(
        gate.unavailable(&gpu_backend).is_some(),
        "with no adapter the wgpu separable backend is skipped cleanly"
    );
    assert!(
        gate.unavailable(&BackendId::new("cpu", "reference"))
            .is_none()
    );
}

#[test]
fn sigma_zero_identity_when_adapter_present() {
    let extent = Extent::new(16, 16);
    let input = structured_image(extent);
    if !adapter_present() {
        eprintln!("no GPU adapter; skipping sigma-zero GPU identity check");
        return;
    }
    let _guard = GPU_LOCK.lock().expect("gpu lock");
    let Ok(context) = probe() else { return };
    let _limits = DeviceLimits::of_context(&context);
    // A single unit tap is the identity (the sigma->0 cutoff).
    let mut cache = PipelineCache::new();
    let out = run_separable_gaussian(
        &context,
        &mut cache,
        input.samples(),
        extent,
        4,
        &[1.0],
        Boundary::Clamp,
    )
    .expect("identity run");
    assert_eq!(out.samples, input.samples(), "unit tap is the identity");
    assert_eq!(out.trace.uploads, 0, "no dispatch for the identity");
}
