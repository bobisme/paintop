//! `paintop-wgpu`: the GPU (`wgpu`) op backends for paintop (`plan.md` §6, §12.3).
//!
//! This crate is the `wgpu` half of paintop's backend strategy. The
//! `cpu.reference` oracle in `paintop-cpu` remains the **semantic authority**
//! (`plan.md` §12.1); every kernel here must reproduce it within the op's declared
//! tolerance, validated through the cross-backend differential harness in
//! `paintop-testkit`. This crate adds **faster backends**, never new ops.
//!
//! # Build- and run-time GPU boundary (`plan.md` §19 M3 criterion 4)
//!
//! `wgpu` is an unconditional *build* dependency: this crate always compiles and
//! links, on any host, **without a GPU present** — linking `wgpu` requires no
//! adapter. A GPU is only needed at *run time*, and even then its absence is a
//! first-class, recoverable state, never a build error and never a panic:
//!
//! * Acquiring a device is fallible and explicit ([`gpu::probe::probe`]). On a host
//!   with no compatible adapter it returns a typed
//!   [`GpuUnavailable`], which a caller turns into a
//!   clean `cpu.reference` fallback or an explicit
//!   [`E_GPU_UNAVAILABLE`] unsupported error.
//! * The differential harness gates every `wgpu` backend on adapter presence
//!   (`paintop_testkit::differential::GpuAdapter`), so `just check` passes
//!   GPU-less: GPU-requiring tests **skip cleanly** rather than fail.
//!
//! There is deliberately **no build feature that strips `wgpu` out**: a
//! conditionally-absent backend would let a plan silently change semantics
//! depending on how the crate was compiled. The boundary is a *runtime* probe with
//! a typed unavailable state, not a compile-time toggle — so the same binary
//! behaves identically (clean fallback / explicit error) whether or not a GPU is
//! attached.
//!
//! # Determinism
//!
//! A GPU kernel is still bound by paintop's determinism contract (`plan.md` §1):
//! a fixed backend on a fixed device yields bit-identical reruns, and the result
//! must land within the op's tier tolerance of the oracle. Pipelines are cached by
//! the *normalized fused expression + resource format* so the same logical work
//! reuses the same compiled artifact ([`gpu::pipeline`]).

pub mod gpu;

pub use gpu::{
    GpuBackend, WGPU_BACKEND,
    error::{E_GPU_DISPATCH_INVALID, E_GPU_UNAVAILABLE, GpuError},
    fusion::{
        FusedRun, FusionBreak, FusionCandidate, MATERIALIZE_OP, PointwiseStage, is_supported_op,
        plan_fusion, stage_kinds,
    },
    pipeline::{
        CacheOutcome, FusedExpr, FusedStage, PipelineCache, PipelineKey, ResourceFormat,
        ScalarFormat,
    },
    pointwise::{ExecutionTrace, FusedOutput, POINTWISE_WORKGROUP, run_fused},
    probe::{AdapterIdentity, GpuContext, GpuUnavailable, adapter_present, probe, probe_forced},
    readback::{
        ChainStage, ChainTrace, ReadbackEvent, ReadbackReason, ReadbackViolation, gpu_stage,
    },
    resource::{DeviceLimits, Dispatch, StorageBufferSpec, StorageTextureSpec, WorkgroupSize},
    separable::{Boundary, SeparableOutput, run_separable_gaussian},
    splat::{
        GpuBlend, GpuSplat, SPLAT_STRIDE_BYTES, SPLAT_WORDS, SplatBatchLayout, SplatTargetLayout,
    },
    splat_kernel::{SplatOutput, run_splats},
};
