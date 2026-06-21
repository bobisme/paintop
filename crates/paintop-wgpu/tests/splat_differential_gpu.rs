//! GPU gaussian-splat cross-backend differential + fallback verification (bn-3qa).
//!
//! Closes the loop on the GPU splat kernel (`plan.md` §12.3, §19 M3):
//!
//! 1. **Differential** — the `wgpu` splat kernel reproduces the `paint.gaussian_splats@1`
//!    CPU reference within the op's bounded tier tolerance, compared through the
//!    cross-backend harness's [`ErrorMap`]/[`Tolerance`] machinery (the same
//!    `cpu.reference`-oracle comparison every other backend is held to). Covered for a
//!    **small** batch (a handful of mixed-blend splats) and a **large** batch (many
//!    overlapping splats), since the large batch exercises the per-pixel array-order
//!    accumulation under heavy overlap.
//! 2. **Saved error map on mismatch** — a divergence beyond tolerance fails with the
//!    [`ErrorMap`]'s `max_abs` / `rms` / `argmax` localized in the panic message
//!    (the saved triage artifact `AGENT_VERIFICATION` §3 requires).
//! 3. **Fallback** — with no adapter the harness gate reports the `wgpu` backend
//!    unavailable and the differential **skips cleanly** (the GPU-less CI path); the
//!    forced-no-adapter probe yields a typed unavailable, not a crash; and the empty
//!    batch is the GPU identity (the base passes through).
//!
//! Every GPU assertion is gated on adapter presence: with no GPU the test returns
//! without failing (`just check` passes GPU-less). On this host (RTX 3090) the GPU
//! path actually runs.

use paintop_core::executor::value::ResourceValue;
use paintop_core::executor::{BackendId, InputValues, OpImplementation};
use paintop_cpu::splat::GaussianSplats;
use paintop_ir::{
    AlphaRepresentation, ChannelLayout, ColorEncoding, ColorRange, CoordinateConvention,
    DeterminismTier, Extent, ImageDescriptor, ResourceDescriptor, ScalarType, SemanticRole,
};
use paintop_testkit::differential::{BackendAvailability, ErrorMap, GpuAdapter, Tolerance};
use paintop_wgpu::{
    GpuBlend, GpuSplat, PipelineCache, SplatBatchLayout, adapter_present, probe, probe_forced,
    run_splats,
};
use serde_json::{Value, json};
use std::sync::Mutex;

/// A process-wide lock serializing GPU work across the (parallel) test threads.
///
/// `wgpu` adapter/device acquisition and dispatch are not safe to drive from many
/// test threads concurrently against this Vulkan driver (concurrent device creation
/// segfaults inside the driver). The harness already probes once per test; this lock
/// makes the GPU section single-flight so the suite is deterministic and crash-free,
/// without changing any op semantics. CPU-only tests never take the lock.
static GPU_LOCK: Mutex<()> = Mutex::new(());

/// A premultiplied-linear RGBA descriptor (the splat op's required base layout).
const fn descriptor(extent: Extent) -> ResourceDescriptor {
    ResourceDescriptor::Image(ImageDescriptor {
        extent,
        layout: ChannelLayout::Rgba,
        scalar: ScalarType::F32,
        color: ColorEncoding::LinearSrgb,
        range: ColorRange::SceneReferred,
        alpha: AlphaRepresentation::Premultiplied,
        coordinates: CoordinateConvention::PixelCenterUpperLeft,
        semantic: SemanticRole::Color,
    })
}

/// A premultiplied RGBA base of `extent`, filled with a mid-gray so over/multiply
/// blends have something to composite against.
fn base_value(extent: Extent) -> ResourceValue {
    let n = usize::try_from(extent.width * extent.height * 4).expect("sample count");
    let mut s = vec![0.0_f32; n];
    for px in s.chunks_exact_mut(4) {
        // Premultiplied gray at alpha 1: rgb == coverage.
        px.copy_from_slice(&[0.2, 0.2, 0.2, 1.0]);
    }
    ResourceValue::new(descriptor(extent), 4, s).expect("base value")
}

/// A finite JSON number coerced to `f32`. The fixtures are single-precision-clean
/// literals, so the narrowing is exact for the values used here.
#[allow(
    clippy::cast_possible_truncation,
    reason = "fixture splat params are single-precision literals; the narrowing is exact"
)]
fn num(v: &Value) -> f32 {
    v.as_f64().expect("finite number") as f32
}

/// Convert one JSON splat object into the GPU record, mirroring the op's resolve
/// (the `hardness -> exponent` map is `p = 1 + hardness*7`).
fn gpu_splat(v: &Value) -> GpuSplat {
    let arr2 = |k: &str| -> [f32; 2] {
        let a = v.get(k).and_then(Value::as_array).expect("pair");
        [num(&a[0]), num(&a[1])]
    };
    let color = {
        let a = v.get("color").and_then(Value::as_array).expect("color");
        [num(&a[0]), num(&a[1]), num(&a[2]), num(&a[3])]
    };
    let angle = v.get("angle_rad").map_or(0.0, num);
    let opacity = v.get("opacity").map_or(1.0, num);
    let hardness = v.get("hardness").map_or(0.0, num);
    let blend = GpuBlend::parse(v.get("blend").and_then(Value::as_str)).expect("blend");
    GpuSplat {
        center: arr2("center_px"),
        sigma: arr2("sigma_px"),
        angle_rad: angle,
        color,
        opacity,
        exponent: hardness.mul_add(7.0, 1.0),
        blend,
    }
}

/// Run the CPU oracle for `params` on `base`, returning the painted `image` value.
fn cpu_oracle(base: &ResourceValue, params: &Value) -> ResourceValue {
    let mut inputs = InputValues::new();
    inputs.insert("base".to_owned(), base.clone());
    let out = GaussianSplats::new()
        .compute(&inputs, params)
        .expect("cpu reference paints the batch");
    out.get("image").expect("image output").clone()
}

/// The shared differential body: build the oracle + GPU result for a batch and
/// compare through the harness error map at the op's bounded tolerance. Skips
/// cleanly when no adapter is present.
fn run_differential(extent: Extent, splats_json: &[Value]) {
    let base = base_value(extent);
    let params = json!({ "splats": splats_json });
    let oracle = cpu_oracle(&base, &params);

    // Gate exactly as the harness would: a `wgpu` backend with no adapter skips.
    let gpu_backend = BackendId::new("wgpu", "splat");
    let availability = GpuAdapter::new(adapter_present());
    if let Some(reason) = availability.unavailable(&gpu_backend) {
        eprintln!("skipping GPU splat differential cleanly: {reason}");
        return;
    }
    // Serialize the GPU section across parallel test threads (see GPU_LOCK).
    let _guard = GPU_LOCK.lock().expect("gpu lock");
    let Ok(context) = probe() else {
        eprintln!("no GPU adapter present; skipping GPU splat differential");
        return;
    };

    let gpu_records: Vec<GpuSplat> = oracle_records(&params);
    let limits = paintop_wgpu::DeviceLimits::of_context(&context);
    let batch = SplatBatchLayout::build(&gpu_records, &limits).expect("gpu batch layout");

    let mut cache = PipelineCache::new();
    let first = run_splats(&context, &mut cache, &batch, base.samples(), extent).expect("gpu run");

    // No intermediate readback: one upload, one export readback.
    assert_eq!(first.trace.intermediate_readbacks, 0);
    assert_eq!(first.trace.uploads, 1);
    assert_eq!(first.trace.export_readbacks, 1);

    let candidate =
        ResourceValue::new(descriptor(extent), 4, first.samples.clone()).expect("gpu value");
    let error_map = ErrorMap::compute(&oracle, &candidate);
    let tolerance = Tolerance::for_tier(DeterminismTier::Bounded);
    assert!(
        error_map.within(&tolerance),
        "GPU splat diverged from the CPU reference beyond tolerance: \
         max_abs={} rms={} argmax={} (saved error map)",
        error_map.max_abs,
        error_map.rms,
        error_map.argmax,
    );

    // Determinism: a fixed backend on a fixed device reruns bit-identically.
    let second =
        run_splats(&context, &mut cache, &batch, base.samples(), extent).expect("gpu rerun");
    assert_eq!(
        first.samples, second.samples,
        "a fixed GPU backend reruns bit-identically"
    );
    // The pipeline cache served a hit on the rerun (one compile, one reuse).
    assert_eq!(cache.misses(), 1);
    assert_eq!(cache.hits(), 1);
}

/// Map the params' `splats` array into GPU records.
fn oracle_records(params: &Value) -> Vec<GpuSplat> {
    params
        .get("splats")
        .and_then(Value::as_array)
        .expect("splats array")
        .iter()
        .map(gpu_splat)
        .collect()
}

#[test]
fn small_mixed_blend_batch_matches_cpu_reference() {
    let extent = Extent::new(32, 32);
    let splats = vec![
        json!({ "center_px": [10.0, 10.0], "sigma_px": [4.0, 2.0], "angle_rad": 0.6,
                "color": [1.0, 0.2, 0.1, 0.9], "opacity": 0.8, "blend": "normal" }),
        json!({ "center_px": [20.0, 14.0], "sigma_px": [3.0, 3.0],
                "color": [0.1, 0.9, 0.3, 0.7], "opacity": 1.0, "blend": "add" }),
        json!({ "center_px": [16.0, 22.0], "sigma_px": [5.0, 2.5], "angle_rad": -0.4,
                "color": [0.2, 0.3, 1.0, 0.6], "opacity": 0.9, "blend": "screen",
                "hardness": 0.5 }),
        json!({ "center_px": [8.0, 24.0], "sigma_px": [2.0, 6.0],
                "color": [0.8, 0.8, 0.1, 0.8], "opacity": 0.7, "blend": "multiply" }),
        json!({ "center_px": [25.0, 25.0], "sigma_px": [3.5, 3.5],
                "color": [0.5, 0.5, 0.5, 1.0], "opacity": 0.6, "blend": "lighten" }),
    ];
    run_differential(extent, &splats);
}

#[test]
fn large_overlapping_batch_matches_cpu_reference() {
    // Many overlapping splats stress the per-pixel array-order accumulation: every
    // pixel composites a deep stack in order.
    let extent = Extent::new(48, 48);
    let mut splats = Vec::new();
    for i in 0..64_u16 {
        let fx = f32::from(i % 8) * 6.0 + 4.0;
        let fy = f32::from(i / 8) * 6.0 + 4.0;
        let t = f32::from(i) / 64.0;
        let blend = ["normal", "add", "screen", "multiply"][usize::from(i % 4)];
        splats.push(json!({
            "center_px": [fx, fy],
            "sigma_px": [2.0f32.mul_add(t, 2.5), 2.0 + t],
            "angle_rad": t * 3.0,
            "color": [t, 1.0 - t, 0.5, 0.4f32.mul_add(t, 0.4)],
            "opacity": 0.4f32.mul_add(t, 0.5),
            "blend": blend
        }));
    }
    run_differential(extent, &splats);
}

#[test]
fn no_adapter_skips_cleanly_via_the_harness_gate() {
    // The forced-no-adapter gate is the same one the harness uses; with it reporting
    // unavailable, a GPU splat differential skips rather than fails (GPU-less CI).
    let gpu_backend = BackendId::new("wgpu", "splat");
    let gate = GpuAdapter::new(false);
    assert!(
        gate.unavailable(&gpu_backend).is_some(),
        "with no adapter the wgpu backend is skipped cleanly"
    );
    // A CPU backend is never gated by the GPU probe.
    assert!(
        gate.unavailable(&BackendId::new("cpu", "reference"))
            .is_none()
    );
}

#[test]
fn forced_no_adapter_probe_is_a_clean_unavailable_not_a_crash() {
    // The fallback path is exercisable even on a host WITH a GPU.
    let outcome = probe_forced(true);
    let err = outcome.expect_err("forced probe yields unavailable");
    assert!(err.reason.contains("forced"), "{}", err.reason);
}

#[test]
fn empty_batch_is_the_gpu_identity_when_an_adapter_is_present() {
    let extent = Extent::new(16, 16);
    let base = base_value(extent);
    if !adapter_present() {
        eprintln!("no GPU adapter; skipping empty-batch GPU identity check");
        return;
    }
    let _guard = GPU_LOCK.lock().expect("gpu lock");
    let Ok(context) = probe() else {
        return;
    };
    let batch = SplatBatchLayout::build(&[], &paintop_wgpu::DeviceLimits::of_context(&context))
        .expect("empty batch");
    let mut cache = PipelineCache::new();
    let out = run_splats(&context, &mut cache, &batch, base.samples(), extent).expect("gpu run");
    // The base passes through unchanged (the identity), with no dispatch.
    assert_eq!(out.samples, base.samples());
    assert_eq!(out.trace.uploads, 0);
    assert_eq!(out.trace.export_readbacks, 0);
}
