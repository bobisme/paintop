//! No-unplanned-readback verification on a real GPU chain (bn-3q0).
//!
//! Drives a fully GPU-compatible chain — fused pointwise → separable filter →
//! gaussian splat — on the live adapter and asserts the readback policy
//! (`plan.md` §12.3, §19 M3 exit criterion 2):
//!
//! 1. Each GPU stage is individually **readback-free for its intermediates**: it
//!    uploads its input once and reads its output back once, with zero intermediate
//!    round trips (the property each `run_*` reports in its [`ExecutionTrace`]).
//! 2. Composed GPU-resident — intermediates stay on the device — the whole chain
//!    shows **zero unplanned readbacks** and exactly **one** declared export
//!    ([`ChainTrace::verify(1)`]).
//! 3. Inserting a host-side `debug.materialize@1` barrier introduces **exactly one**
//!    expected readback: the chain now verifies at `2` (export + one barrier) and the
//!    strict no-barrier expectation correctly fails.
//!
//! Gated on adapter presence: with no GPU the test returns without failing
//! (`just check` passes GPU-less). On this host (RTX 3090) the GPU path runs.

use paintop_ir::{Extent, RoiCategory};
use paintop_testkit::differential::{BackendAvailability, GpuAdapter};
use paintop_wgpu::{
    Boundary, ChainStage, ChainTrace, FusionCandidate, PipelineCache, ReadbackViolation,
    ResourceFormat, adapter_present, gpu_stage, plan_fusion, probe, run_fused,
    run_separable_gaussian, run_splats,
};
use paintop_wgpu::{GpuBlend, GpuSplat, SplatBatchLayout};
use serde_json::json;
use std::sync::Mutex;

/// Serialize the GPU section across parallel test threads (concurrent `wgpu` device
/// creation segfaults this Vulkan driver).
static GPU_LOCK: Mutex<()> = Mutex::new(());

const EXTENT: Extent = Extent::new(24, 24);

/// A varied RGBA input (premultiplied-friendly values in `[0, 1]`).
fn input_samples() -> Vec<f32> {
    let pixels = EXTENT.width * EXTENT.height;
    let mut s = Vec::with_capacity((pixels * 4) as usize);
    for i in 0..pixels {
        let t = f32::from(u16::try_from(i % 256).expect("small")) / 256.0;
        s.extend_from_slice(&[t, 1.0 - t, 0.5 * t, 1.0]);
    }
    s
}

/// A small pointwise chain (gain + bias) that fuses into one GPU kernel.
fn pointwise_chain() -> Vec<FusionCandidate> {
    let f = ResourceFormat::f32(4);
    vec![
        FusionCandidate {
            op: "color.adjust@1".parse().expect("op"),
            params: json!({ "exposure_ev": 0.3 }),
            roi: RoiCategory::Pointwise,
            format: f,
        },
        FusionCandidate {
            op: "color.bias@1".parse().expect("op"),
            params: json!({ "bias": 0.05 }),
            roi: RoiCategory::Pointwise,
            format: f,
        },
    ]
}

/// A small splat batch to paint at the end of the chain.
fn splat_batch() -> Vec<GpuSplat> {
    vec![
        GpuSplat {
            center: [8.0, 8.0],
            sigma: [3.0, 2.0],
            angle_rad: 0.4,
            color: [1.0, 0.3, 0.2, 0.8],
            opacity: 0.9,
            exponent: 1.0,
            blend: GpuBlend::Normal,
        },
        GpuSplat {
            center: [16.0, 16.0],
            sigma: [2.5, 2.5],
            angle_rad: 0.0,
            color: [0.2, 0.8, 0.4, 0.7],
            opacity: 0.8,
            exponent: 1.0,
            blend: GpuBlend::Add,
        },
    ]
}

#[test]
fn fully_gpu_chain_is_readback_free_except_the_declared_export() {
    let gpu_backend = paintop_core::executor::BackendId::new("wgpu", "pointwise");
    if let Some(reason) = GpuAdapter::new(adapter_present()).unavailable(&gpu_backend) {
        eprintln!("skipping GPU readback-trace verification cleanly: {reason}");
        return;
    }
    let _guard = GPU_LOCK.lock().expect("gpu lock");
    let Ok(context) = probe() else {
        eprintln!("no GPU adapter present; skipping GPU readback-trace verification");
        return;
    };
    let limits = paintop_wgpu::DeviceLimits::of_context(&context);

    // ---- Stage 1: fused pointwise. ----
    let runs = plan_fusion(&pointwise_chain());
    assert_eq!(runs.len(), 1, "the pointwise chain fuses into one run");
    let samples = input_samples();
    let mut cache = PipelineCache::new();
    let pw = run_fused(&context, &mut cache, &runs[0], &samples, true).expect("pointwise run");
    assert!(
        pw.trace.is_readback_free(),
        "fused pointwise stage is readback-free: {:?}",
        pw.trace
    );

    // ---- Stage 2: separable gaussian filter (fed the pointwise output). ----
    let taps = gaussian_taps(2.0);
    let mut sep_cache = PipelineCache::new();
    let filt = run_separable_gaussian(
        &context,
        &mut sep_cache,
        &pw.samples,
        EXTENT,
        4,
        &taps,
        Boundary::Clamp,
    )
    .expect("separable run");
    assert_eq!(filt.trace.intermediate_readbacks, 0);
    assert_eq!(filt.trace.uploads, 1);
    assert_eq!(filt.trace.export_readbacks, 1);

    // ---- Stage 3: gaussian splat (fed the filtered output). ----
    let batch = SplatBatchLayout::build(&splat_batch(), &limits).expect("batch");
    let mut splat_cache = PipelineCache::new();
    let painted =
        run_splats(&context, &mut splat_cache, &batch, &filt.samples, EXTENT).expect("splat run");
    assert_eq!(painted.trace.intermediate_readbacks, 0);
    assert_eq!(painted.trace.export_readbacks, 1);

    // ---- Compose the chain GPU-resident: intermediates stay on the device, only the
    // terminal splat exports. The two earlier stages are recorded as resident (their
    // standalone export is elided when fused into a resident chain). ----
    let resident = |stages: u32| {
        gpu_stage(paintop_wgpu::ExecutionTrace {
            uploads: 1,
            intermediate_readbacks: 0,
            export_readbacks: 0,
            stages,
        })
    };
    let chain = [
        resident(pw.trace.stages),
        resident(filt.trace.stages),
        gpu_stage(painted.trace), // terminal: one export
    ];
    let trace = ChainTrace::build(&chain);
    assert_eq!(trace.unplanned(), 0, "no unplanned intermediate readbacks");
    assert_eq!(trace.materialize_readbacks(), 0);
    trace
        .verify(1)
        .expect("a fully GPU-resident chain reads back only at the declared export");

    // ---- Insert a host-side debug.materialize between the filter and the splat:
    // exactly one extra expected readback. ----
    let chain_mat = [
        resident(pw.trace.stages),
        resident(filt.trace.stages),
        ChainStage::Materialize,
        gpu_stage(painted.trace),
    ];
    let trace_mat = ChainTrace::build(&chain_mat);
    assert_eq!(
        trace_mat.unplanned(),
        0,
        "the barrier is planned, not a leak"
    );
    assert_eq!(
        trace_mat.materialize_readbacks(),
        1,
        "exactly one debug.materialize readback"
    );
    trace_mat
        .verify(2)
        .expect("export + exactly one debug.materialize barrier");
    // The strict no-barrier expectation now fails (the planned count changed).
    assert!(matches!(
        trace_mat.verify(1),
        Err(ReadbackViolation::PlannedMismatch {
            expected: 1,
            actual: 2
        })
    ));
}

#[test]
fn no_adapter_skips_cleanly_via_the_harness_gate() {
    // The readback-trace verification is CI-gated on adapter presence: with no
    // adapter the wgpu backend is skipped cleanly (GPU-less CI path).
    let gpu_backend = paintop_core::executor::BackendId::new("wgpu", "pointwise");
    let gate = GpuAdapter::new(false);
    assert!(gate.unavailable(&gpu_backend).is_some());
}

/// The normalized 1-D Gaussian taps for `sigma` (same as the CPU separable backend).
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "tiny integer kernel offsets; exact in f64, narrowed for the f32 kernel"
)]
fn gaussian_taps(sigma: f64) -> Vec<f32> {
    let r = ((3.0 * sigma).ceil() as usize).max(1);
    let ri = i64::try_from(r).expect("radius fits i64");
    let two_sigma_sq = 2.0 * sigma * sigma;
    let mut taps = Vec::with_capacity(2 * r + 1);
    let mut sum = 0.0_f64;
    for d in -ri..=ri {
        let w = (-((d * d) as f64) / two_sigma_sq).exp();
        sum += w;
        taps.push(w);
    }
    taps.into_iter().map(|w| (w / sum) as f32).collect()
}
