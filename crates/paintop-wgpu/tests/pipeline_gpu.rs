//! GPU-resident pipeline-cache reuse test (bn-2vi).
//!
//! Proves the cache compiles a **real** `wgpu::ComputePipeline` once for a fused
//! key and reuses it on a second lookup with the same key — the GPU-resident
//! counterpart to the GPU-less reuse unit tests in `gpu::pipeline`.
//!
//! Gated on adapter presence: with no GPU this skips cleanly (`just check` passes
//! GPU-less), per `plan.md` §19 M3 criterion 4. On this host (RTX 3090) it runs and
//! actually compiles a shader.

use paintop_wgpu::{
    CacheOutcome, FusedExpr, FusedStage, PipelineCache, PipelineKey, ResourceFormat, probe,
};
use serde_json::json;

/// A trivial valid compute shader so the pipeline actually compiles on the device.
const TRIVIAL_WGSL: &str = r"
@compute @workgroup_size(1)
fn main() {}
";

fn fused() -> FusedExpr {
    FusedExpr::new()
        .with(FusedStage::new(
            "color.adjust@1".parse().expect("op"),
            json!({ "gain": 1.25 }),
        ))
        .with(FusedStage::new(
            "alpha.premultiply@1".parse().expect("op"),
            json!({}),
        ))
}

#[test]
fn real_pipeline_is_compiled_once_and_reused_on_a_hit() {
    let Ok(context) = probe() else {
        eprintln!("no GPU adapter present; skipping GPU pipeline-cache reuse test");
        return;
    };
    let device = context.device();

    let key = PipelineKey::derive(&fused(), ResourceFormat::f32(4)).expect("key");
    let mut cache = PipelineCache::<wgpu::ComputePipeline>::new();

    let compile = || -> Result<wgpu::ComputePipeline, std::convert::Infallible> {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("paintop-wgpu test shader"),
            source: wgpu::ShaderSource::Wgsl(TRIVIAL_WGSL.into()),
        });
        Ok(
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("paintop-wgpu test pipeline"),
                layout: None,
                module: &module,
                entry_point: Some("main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            }),
        )
    };

    // First lookup: miss -> compiles a real pipeline.
    let (p1, o1) = cache
        .get_or_compile(key.clone(), compile)
        .expect("compile pipeline");
    assert_eq!(o1, CacheOutcome::Miss);

    // Second lookup with the SAME key: hit -> reuses the same compiled pipeline,
    // and the compile closure (which would build a new module) is never invoked.
    let (p2, o2) = cache
        .get_or_compile::<_, std::convert::Infallible>(key, || {
            panic!("must not recompile on a cache hit")
        })
        .expect("hit");
    assert_eq!(o2, CacheOutcome::Hit);
    assert!(
        std::sync::Arc::ptr_eq(&p1, &p2),
        "the same cached pipeline Arc is reused"
    );
    assert_eq!(cache.len(), 1);
    assert_eq!(cache.hits(), 1);
    assert_eq!(cache.misses(), 1);
}

#[test]
fn distinct_fused_keys_compile_distinct_real_pipelines() {
    let Ok(context) = probe() else {
        eprintln!("no GPU adapter present; skipping GPU distinct-key test");
        return;
    };
    let device = context.device();

    let k_rgba = PipelineKey::derive(&fused(), ResourceFormat::f32(4)).expect("key");
    let k_r = PipelineKey::derive(&fused(), ResourceFormat::f32(1)).expect("key");
    assert_ne!(k_rgba, k_r, "different formats must key apart");

    let mut cache = PipelineCache::<wgpu::ComputePipeline>::new();
    let compile = || -> Result<wgpu::ComputePipeline, std::convert::Infallible> {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(TRIVIAL_WGSL.into()),
        });
        Ok(
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: None,
                layout: None,
                module: &module,
                entry_point: Some("main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            }),
        )
    };

    let (_, o1) = cache.get_or_compile(k_rgba, compile).expect("c1");
    let (_, o2) = cache.get_or_compile(k_r, compile).expect("c2");
    assert_eq!(o1, CacheOutcome::Miss);
    assert_eq!(o2, CacheOutcome::Miss);
    assert_eq!(cache.len(), 2, "distinct keys cache distinct pipelines");
}
