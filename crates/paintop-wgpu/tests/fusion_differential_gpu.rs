//! Fused pointwise differential + no-readback + cache-hit verification (bn-3eb).
//!
//! Closes the loop on GPU pointwise fusion (`plan.md` §13.2, §19 M3):
//!
//! 1. **Differential** — the fused GPU kernel reproduces the CPU reference for the
//!    same chain within the op's tier tolerance, compared through the cross-backend
//!    harness's [`ErrorMap`]/[`Tolerance`] machinery (the same `cpu.reference`-oracle
//!    comparison every other backend is held to, `plan.md` §12.1).
//! 2. **No intermediate readback** — a fully GPU-compatible chain performs exactly
//!    one upload and one final export readback, with **zero** intermediate readbacks
//!    (`plan.md` §19 M3 criterion: "no unplanned readback in a fully GPU-compatible
//!    chain").
//! 3. **Pipeline cache hit** — running the same fused run twice compiles its pipeline
//!    once and reuses it on the second dispatch (the normalized-key cache from
//!    bn-2vi/bn-t2v actually serves a hit).
//!
//! Every GPU assertion is gated on adapter presence: with no GPU the harness skips
//! the `wgpu` backend cleanly and the test returns without failing (`just check`
//! passes GPU-less). On this host (RTX 3090) the GPU path actually runs.

use paintop_core::executor::BackendId;
use paintop_core::executor::value::ResourceValue;
use paintop_ir::{
    AlphaRepresentation, ChannelLayout, ColorEncoding, ColorRange, CoordinateConvention,
    DeterminismTier, Extent, ImageDescriptor, ResourceDescriptor, ScalarType, SemanticRole,
};
use paintop_testkit::differential::{BackendAvailability, ErrorMap, GpuAdapter, Tolerance};
use paintop_wgpu::{
    CacheOutcome, FusionCandidate, PipelineCache, ResourceFormat, adapter_present, plan_fusion,
    probe, run_fused,
};
use serde_json::json;

const EXTENT: Extent = Extent::new(4, 4);
const CHANNELS: u32 = 4;

const fn descriptor() -> ResourceDescriptor {
    ResourceDescriptor::Image(ImageDescriptor {
        extent: EXTENT,
        layout: ChannelLayout::Rgba,
        scalar: ScalarType::F32,
        color: ColorEncoding::LinearSrgb,
        range: ColorRange::SceneReferred,
        alpha: AlphaRepresentation::Straight,
        coordinates: CoordinateConvention::PixelCenterUpperLeft,
        semantic: SemanticRole::Color,
    })
}

fn value(samples: Vec<f32>) -> ResourceValue {
    ResourceValue::new(descriptor(), CHANNELS, samples).expect("sized image")
}

/// A varied RGBA test image (16 pixels) so every channel and a spread of alphas are
/// exercised.
fn input_samples() -> Vec<f32> {
    let pixels = EXTENT.width * EXTENT.height;
    let mut s = Vec::with_capacity((pixels * CHANNELS) as usize);
    for i in 0..pixels {
        // `i < 16`, so `u16`->`f32` is exact; `t` spreads the test values across [0,1).
        let t = f32::from(u16::try_from(i).expect("small index")) / 16.0;
        s.extend_from_slice(&[t, 1.0 - t, 0.5 * t, 0.5_f32.mul_add(t, 0.25)]);
    }
    s
}

/// A 4-stage RGBA pointwise chain: gain, saturation, bias, premultiply — enough
/// distinct stages that fusion meaningfully elides intermediates.
fn chain() -> Vec<FusionCandidate> {
    let f = ResourceFormat::f32(4);
    vec![
        FusionCandidate {
            op: "color.adjust@1".parse().expect("op"),
            params: json!({ "exposure_ev": 0.5, "saturation": -0.3 }),
            roi: paintop_ir::RoiCategory::Pointwise,
            format: f,
        },
        FusionCandidate {
            op: "color.bias@1".parse().expect("op"),
            params: json!({ "bias": 0.02 }),
            roi: paintop_ir::RoiCategory::Pointwise,
            format: f,
        },
        FusionCandidate {
            op: "alpha.premultiply@1".parse().expect("op"),
            params: json!({}),
            roi: paintop_ir::RoiCategory::Pointwise,
            format: f,
        },
    ]
}

#[test]
fn fused_gpu_matches_cpu_reference_no_readback_and_caches() {
    let candidates = chain();
    let runs = plan_fusion(&candidates);
    assert_eq!(runs.len(), 1, "the whole chain fuses into one run");
    let run = &runs[0];
    // color.adjust(exposure+saturation) -> 2 stages, +bias, +premultiply = 4 stages.
    assert_eq!(run.stages.len(), 4);

    let samples = input_samples();
    let oracle = value(run.eval_buffer(&samples, true));

    // Gate exactly as the harness would: a `wgpu` backend with no adapter skips.
    let gpu_backend = BackendId::new("wgpu", "pointwise");
    let availability = GpuAdapter::new(adapter_present());
    if let Some(reason) = availability.unavailable(&gpu_backend) {
        eprintln!("skipping GPU fusion differential cleanly: {reason}");
        return;
    }

    let Ok(context) = probe() else {
        eprintln!("no GPU adapter present; skipping GPU fusion differential");
        return;
    };

    // ---- Differential + no-readback ----
    let mut cache = PipelineCache::new();
    let first = run_fused(&context, &mut cache, run, &samples, true).expect("first GPU run");

    // No intermediate readback in a fully GPU-compatible chain (one upload, one
    // export readback, zero intermediates).
    assert!(
        first.trace.is_readback_free(),
        "fused chain must be readback-free: {:?}",
        first.trace
    );
    assert_eq!(first.trace.intermediate_readbacks, 0);
    assert_eq!(first.trace.uploads, 1);
    assert_eq!(first.trace.export_readbacks, 1);

    // Compare against the oracle through the harness's error-map machinery, at the
    // tolerance the op's tier (bounded) implies — never a per-test ad-hoc epsilon.
    let candidate = value(first.samples.clone());
    let error_map = ErrorMap::compute(&oracle, &candidate);
    let tolerance = Tolerance::for_tier(DeterminismTier::Bounded);
    assert!(
        error_map.within(&tolerance),
        "fused GPU diverged from the CPU reference beyond tolerance: \
         max_abs={} rms={} argmax={}",
        error_map.max_abs,
        error_map.rms,
        error_map.argmax,
    );

    // ---- Pipeline cache hit ----
    // A second run of the same fused chain must reuse the compiled pipeline: the
    // cache served exactly one compile (miss) and now one reuse (hit).
    assert_eq!(cache.misses(), 1, "first run compiled the pipeline");
    assert_eq!(cache.hits(), 0, "first run was a pure miss");
    let second = run_fused(&context, &mut cache, run, &samples, true).expect("second GPU run");
    assert_eq!(cache.misses(), 1, "no recompile on the second run");
    assert_eq!(cache.hits(), 1, "the second run hit the pipeline cache");
    assert_eq!(cache.len(), 1, "one distinct pipeline for one fused key");

    // Determinism: a fixed backend on a fixed device is bit-identical across reruns.
    assert_eq!(
        first.samples, second.samples,
        "a fixed GPU backend reruns bit-identically"
    );
}

#[test]
fn cache_outcome_reports_hit_on_reuse_of_a_fused_key() {
    // The cache-hit property is also exercised GPU-lessly against the run's normalized
    // key, so the cache semantics are verified even with no adapter.
    let candidates = chain();
    let run = &plan_fusion(&candidates)[0];
    let key = run.pipeline_key().expect("key");

    let mut cache = PipelineCache::<u32>::new();
    let (_, first) = cache
        .get_or_compile::<_, std::convert::Infallible>(key.clone(), || Ok(7))
        .expect("compile");
    let (_, second) = cache
        .get_or_compile::<_, std::convert::Infallible>(key, || {
            panic!("must not recompile a cached fused key")
        })
        .expect("hit");
    assert_eq!(first, CacheOutcome::Miss);
    assert_eq!(second, CacheOutcome::Hit);
    assert_eq!(cache.hits(), 1);
}

#[test]
fn no_adapter_skips_cleanly_via_the_harness_gate() {
    // The forced-no-adapter gate is the same one the harness uses; with it reporting
    // unavailable, a fused-GPU differential skips rather than fails — the GPU-less CI
    // path. (Run unconditionally; it never touches the GPU.)
    let gpu_backend = BackendId::new("wgpu", "pointwise");
    let gate = GpuAdapter::new(false);
    assert!(
        gate.unavailable(&gpu_backend).is_some(),
        "with no adapter, the wgpu backend is skipped cleanly"
    );
    // A CPU backend is never gated by the GPU probe.
    assert!(
        gate.unavailable(&BackendId::new("cpu", "reference"))
            .is_none()
    );
}
