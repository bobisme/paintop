//! The `wgpu` fused-pointwise compute pipeline (`plan.md` §12.3, §13.2; bn-125).
//!
//! A [`FusedRun`] collapses an order-sensitive chain of
//! pointwise color/alpha stages into one kernel. This module *runs* that kernel on
//! the GPU: it generates the run's WGSL ([`FusedRun::wgsl_shader`]), compiles it into
//! a cached [`wgpu::ComputePipeline`] (keyed by the run's normalized
//! [`PipelineKey`] + format, so identical work reuses
//! one compiled artifact), uploads the input samples as a single storage buffer,
//! dispatches one invocation per pixel, and reads the result back **once** at the
//! end.
//!
//! # Readback-free fusion
//!
//! The whole stage chain executes inside one dispatch over GPU-resident storage
//! buffers, so there is exactly **one** host→GPU upload and **one** GPU→host readback
//! for the entire fused run — never a per-stage round trip (`plan.md` §13.2; the
//! no-intermediate-readback property bn-3eb asserts). A run's
//! [`ExecutionTrace`] records those counts so a caller (and the bn-3eb test) can
//! assert the chain produced no intermediate readbacks.
//!
//! # Fallback
//!
//! GPU presence is a runtime fact. [`run_fused`] needs a live
//! [`GpuContext`]; a caller with no adapter runs
//! [`FusedRun::eval_buffer`](super::fusion::FusedRun::eval_buffer) — the same
//! semantics on the CPU — so a GPU-less host still produces the correct result
//! (`plan.md` §19 M3 criterion 4). The two paths share the per-stage
//! [`eval_pixel`](super::fusion::PointwiseStage::eval_pixel) /
//! [`wgsl_body`](super::fusion::PointwiseStage::wgsl_body) lowering, kept in lock-step
//! so the GPU result lands within the op's tolerance of the CPU oracle.

use std::sync::Arc;

use wgpu::util::DeviceExt;

use super::error::GpuError;
use super::fusion::FusedRun;
use super::pipeline::{PipelineCache, PipelineKey};
use super::probe::GpuContext;
use super::resource::{DeviceLimits, Dispatch, StorageBufferSpec, WorkgroupSize};

/// The fixed workgroup size (invocations along X) the fused pointwise kernel
/// dispatches with. A 1-D grid over pixels; 64 is a safe, widely-supported size.
pub const POINTWISE_WORKGROUP: u32 = 64;

/// A count of the host↔GPU transfers a fused run performed, for the no-readback
/// assertion (`plan.md` §13.2; bn-3eb).
///
/// A correctly fused chain uploads its input once and reads its output back once,
/// regardless of how many stages it fuses — the intermediate values never leave the
/// GPU. `intermediate_readbacks` is therefore always `0`; it is recorded explicitly
/// so a test can assert the property rather than infer it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExecutionTrace {
    /// Host→GPU uploads of input data (exactly 1 for a non-empty run).
    pub uploads: u32,
    /// GPU→host readbacks of *intermediate* results — always 0 for a fused chain.
    pub intermediate_readbacks: u32,
    /// GPU→host readbacks of the final exported output (exactly 1).
    pub export_readbacks: u32,
    /// The number of fused stages executed in the single dispatch.
    pub stages: u32,
}

impl ExecutionTrace {
    /// Whether the run was readback-free for its intermediates (the bn-3eb property):
    /// one upload, zero intermediate readbacks, one final export readback.
    #[must_use]
    pub const fn is_readback_free(&self) -> bool {
        self.intermediate_readbacks == 0 && self.uploads == 1 && self.export_readbacks == 1
    }
}

/// The output of a fused GPU run: the transformed samples and the transfer trace.
#[derive(Debug, Clone, PartialEq)]
pub struct FusedOutput {
    /// The result samples, same length/layout as the input.
    pub samples: Vec<f32>,
    /// The host↔GPU transfer trace (for the no-readback assertion).
    pub trace: ExecutionTrace,
}

/// Run a fused pointwise `run` on the GPU over `samples`, reusing a compiled pipeline
/// from `cache` (compiling on the first sight of the run's key).
///
/// `samples` is a row-major, channel-interleaved `f32` buffer with `run.channels()`
/// channels per pixel; `has_alpha` whether the trailing channel is alpha. The whole
/// stage chain runs in one dispatch over storage buffers, so the input is uploaded
/// once and the output read back once — no intermediate leaves the GPU
/// ([`ExecutionTrace::is_readback_free`]).
///
/// # Errors
/// - [`GpuError::DispatchInvalid`] if the buffer length is not a whole number of
///   pixels, is empty, or exceeds the device's storage/dispatch limits.
/// - a shader-compilation error is surfaced via `wgpu`'s validation; the device's
///   error scope is checked so a malformed generated shader fails explicitly rather
///   than producing a wrong result.
pub fn run_fused(
    context: &GpuContext,
    cache: &mut PipelineCache<wgpu::ComputePipeline>,
    run: &FusedRun,
    samples: &[f32],
    has_alpha: bool,
) -> Result<FusedOutput, GpuError> {
    let stride = run.channels();
    if stride == 0 {
        return Err(GpuError::DispatchInvalid {
            reason: "fused run has zero channels".to_owned(),
        });
    }
    if samples.is_empty() || !samples.len().is_multiple_of(stride) {
        return Err(GpuError::DispatchInvalid {
            reason: format!(
                "sample buffer length {} is not a positive multiple of the {stride}-channel stride",
                samples.len()
            ),
        });
    }
    let pixel_count = samples.len() / stride;

    let limits = DeviceLimits::of_context(context);
    // Validate the storage buffer and the 1-D dispatch before any submission.
    StorageBufferSpec::new(samples.len() as u64).validate(&limits)?;
    let dispatch = pixel_dispatch(pixel_count, &limits)?;

    let device = context.device();
    let queue = context.queue();

    // Compile (or reuse) the pipeline for this run's normalized key.
    let key = run.pipeline_key().map_err(|e| GpuError::DispatchInvalid {
        reason: format!("failed to derive pipeline key: {e}"),
    })?;
    let pipeline = compile_or_reuse(device, cache, run, has_alpha, &key)?;

    // Upload input (1 upload), allocate output + uniform. Samples are serialized to
    // an owned native-endian byte buffer (no pointer casting — the crate forbids
    // `unsafe`); `read_back` decodes with the matching `from_ne_bytes`.
    let input_bytes = to_byte_vec(samples);
    let input_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("paintop fused input"),
        contents: &input_bytes,
        usage: wgpu::BufferUsages::STORAGE,
    });
    // The output mirrors the input's byte length exactly (same element count).
    let output_size = input_bytes.len() as u64;
    let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("paintop fused output"),
        size: output_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let pixel_count_u32 = u32::try_from(pixel_count).map_err(|_| GpuError::DispatchInvalid {
        reason: format!("pixel count {pixel_count} exceeds u32"),
    })?;
    let params_bytes = pixel_count_u32.to_ne_bytes();
    let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("paintop fused params"),
        contents: &params_bytes,
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("paintop fused bind group"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: input_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: output_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: params_buffer.as_entire_binding(),
            },
        ],
    });

    // A staging buffer for the single final readback.
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("paintop fused staging"),
        size: output_size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("paintop fused pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        let (gx, gy, gz) = dispatch.groups();
        pass.dispatch_workgroups(gx, gy, gz);
    }
    encoder.copy_buffer_to_buffer(&output_buffer, 0, &staging, 0, output_size);
    queue.submit(std::iter::once(encoder.finish()));

    // The single export readback: map, poll to completion, copy out.
    let samples = read_back(device, &staging, samples.len())?;

    Ok(FusedOutput {
        samples,
        trace: ExecutionTrace {
            uploads: 1,
            intermediate_readbacks: 0,
            export_readbacks: 1,
            stages: u32::try_from(run.stages.len()).unwrap_or(u32::MAX),
        },
    })
}

/// Compile the run's WGSL into a pipeline, or reuse the cached one for its key.
///
/// The device's validation error scope is pushed around the compile so a malformed
/// generated shader surfaces as an explicit [`GpuError::DispatchInvalid`] rather than
/// a panic or a silently-broken pipeline.
fn compile_or_reuse(
    device: &wgpu::Device,
    cache: &mut PipelineCache<wgpu::ComputePipeline>,
    run: &FusedRun,
    has_alpha: bool,
    key: &PipelineKey,
) -> Result<Arc<wgpu::ComputePipeline>, GpuError> {
    let source = run.wgsl_shader(has_alpha, POINTWISE_WORKGROUP);
    let (pipeline, _outcome) = cache.get_or_compile(key.clone(), || {
        device.push_error_scope(wgpu::ErrorFilter::Validation);
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("paintop fused pointwise shader"),
            source: wgpu::ShaderSource::Wgsl(source.clone().into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("paintop fused pointwise pipeline"),
            layout: None,
            module: &module,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        if let Some(err) = pollster::block_on(device.pop_error_scope()) {
            return Err(GpuError::DispatchInvalid {
                reason: format!("fused pointwise shader failed to compile: {err}"),
            });
        }
        Ok(pipeline)
    })?;
    Ok(pipeline)
}

/// Build and validate the 1-D pixel dispatch (one invocation per pixel).
fn pixel_dispatch(pixel_count: usize, limits: &DeviceLimits) -> Result<Dispatch, GpuError> {
    let width = u32::try_from(pixel_count).map_err(|_| GpuError::DispatchInvalid {
        reason: format!("pixel count {pixel_count} exceeds u32 dispatch width"),
    })?;
    // The resource model validates a 2D extent; model the linear pixel array as a
    // width×1 strip and dispatch a (workgroup×1) tile over it.
    Dispatch::for_extent(
        paintop_ir::Extent::new(width, 1),
        WorkgroupSize::new(POINTWISE_WORKGROUP, 1, 1),
        limits,
    )
}

/// Map the staging buffer and copy its `len` `f32`s back to the host (the single
/// export readback).
fn read_back(
    device: &wgpu::Device,
    staging: &wgpu::Buffer,
    len: usize,
) -> Result<Vec<f32>, GpuError> {
    let slice = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |res| {
        let _ = tx.send(res);
    });
    // Drive the device until the map callback fires.
    let _ = device.poll(wgpu::PollType::wait_indefinitely());
    match rx.recv() {
        Ok(Ok(())) => {}
        Ok(Err(e)) => {
            return Err(GpuError::DispatchInvalid {
                reason: format!("failed to map fused output for readback: {e}"),
            });
        }
        Err(e) => {
            return Err(GpuError::DispatchInvalid {
                reason: format!("fused output readback channel closed: {e}"),
            });
        }
    }
    let data = slice.get_mapped_range();
    let mut out = Vec::with_capacity(len);
    for chunk in data.chunks_exact(std::mem::size_of::<f32>()).take(len) {
        out.push(f32::from_ne_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    drop(data);
    staging.unmap();
    Ok(out)
}

/// Serialize an `f32` slice to an owned native-endian byte buffer for upload.
///
/// `f32::to_ne_bytes` round-trips through [`read_back`]'s `from_ne_bytes`, so the
/// upload and readback encodings agree. Building an owned `Vec<u8>` (rather than
/// reinterpreting the slice's bytes) keeps the code free of `unsafe`, which the crate
/// forbids.
fn to_byte_vec(samples: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(std::mem::size_of_val(samples));
    for &s in samples {
        bytes.extend_from_slice(&s.to_ne_bytes());
    }
    bytes
}
