//! GPU-resident fused-pointwise pipeline test (bn-125).
//!
//! Proves a *multi-op* pointwise chain runs as **one** `wgpu` compute pipeline over
//! GPU storage buffers and reproduces the CPU reference, and that the CPU fallback is
//! always available.
//!
//! Gated on adapter presence: with no GPU the GPU assertions skip cleanly (`just
//! check` passes GPU-less, `plan.md` §19 M3 criterion 4); the CPU-fallback assertion
//! runs unconditionally. On this host (RTX 3090) the GPU path actually executes.

use paintop_ir::RoiCategory;
use paintop_wgpu::{FusionCandidate, PipelineCache, ResourceFormat, plan_fusion, probe, run_fused};
use serde_json::json;

/// A 3-stage RGBA pointwise chain: exposure gain, additive bias, premultiply.
fn chain() -> Vec<FusionCandidate> {
    vec![
        FusionCandidate {
            op: "color.gain@1".parse().expect("op"),
            params: json!({ "gain": 1.5 }),
            roi: RoiCategory::Pointwise,
            format: ResourceFormat::f32(4),
        },
        FusionCandidate {
            op: "color.bias@1".parse().expect("op"),
            params: json!({ "bias": 0.05 }),
            roi: RoiCategory::Pointwise,
            format: ResourceFormat::f32(4),
        },
        FusionCandidate {
            op: "alpha.premultiply@1".parse().expect("op"),
            params: json!({}),
            roi: RoiCategory::Pointwise,
            format: ResourceFormat::f32(4),
        },
    ]
}

/// 4 RGBA pixels of varied color/alpha.
fn input() -> Vec<f32> {
    vec![
        0.10, 0.20, 0.30, 1.00, // opaque
        0.50, 0.40, 0.30, 0.50, // half alpha
        0.00, 0.90, 0.10, 0.25, // low alpha
        0.80, 0.80, 0.80, 0.00, // transparent
    ]
}

#[test]
fn a_multi_op_chain_runs_as_one_gpu_pipeline_matching_the_cpu_reference() {
    let candidates = chain();
    let runs = plan_fusion(&candidates);
    assert_eq!(runs.len(), 1, "the whole chain fuses into one run");
    let run = &runs[0];
    assert_eq!(run.node_count(), 3, "three source nodes, one kernel");

    let samples = input();
    // The CPU reference for the fused chain (the oracle the GPU must match).
    let reference = run.eval_buffer(&samples, true);

    let Ok(context) = probe() else {
        eprintln!("no GPU adapter present; skipping GPU fused-pipeline execution");
        // The CPU fallback path is still exercised: the reference is a real result.
        assert_eq!(reference.len(), samples.len());
        return;
    };

    let mut cache = PipelineCache::new();
    let output = run_fused(&context, &mut cache, run, &samples, true).expect("fused GPU run");

    // The whole chain ran in one dispatch: one upload, zero intermediate readbacks.
    assert!(output.trace.is_readback_free(), "{:?}", output.trace);
    assert_eq!(output.trace.stages, 3);
    assert_eq!(output.samples.len(), samples.len());

    // The GPU result matches the CPU reference within a tight bounded envelope (the
    // chain's ops are bounded-tier; the only slack is f32 reassociation).
    for (i, (gpu, cpu)) in output.samples.iter().zip(reference.iter()).enumerate() {
        assert!(
            (gpu - cpu).abs() <= 1.0e-5,
            "sample {i}: gpu={gpu} cpu={cpu}"
        );
    }
}

#[test]
fn cpu_fallback_is_always_available_without_a_gpu() {
    // The eval_buffer fallback never needs an adapter and is the same semantics the
    // GPU kernel implements: a GPU-less host still produces the correct result.
    let candidates = chain();
    let run = &plan_fusion(&candidates)[0];
    let samples = input();
    let out = run.eval_buffer(&samples, true);
    assert_eq!(out.len(), samples.len());

    // Pixel 0 is opaque (alpha 1): gain 1.5, bias +0.05, premultiply by 1 -> unchanged
    // by premultiply. Channel 0: 0.10*1.5 + 0.05 = 0.20.
    assert!((out[0] - 0.20).abs() < 1e-6, "{}", out[0]);
    // Pixel 1 alpha 0.5: channel 0 = (0.50*1.5 + 0.05) * 0.5 = 0.80 * 0.5 = 0.40.
    assert!((out[4] - 0.40).abs() < 1e-6, "{}", out[4]);
    // Alpha channels themselves are never modified by these stages.
    assert!((out[3] - 1.00).abs() < 1e-6);
    assert!((out[7] - 0.50).abs() < 1e-6);
}
