//! The `wgpu` Gaussian-splat kernel with deterministic draw order (`plan.md`
//! §12.3; bn-eji).
//!
//! Runs a [`SplatBatchLayout`] on the GPU with the **exact** semantics of the CPU
//! oracle (`crates/paintop-cpu/src/splat.rs`): each splat is an oriented
//! super-Gaussian dab with anisotropic σ, rotation θ, straight color, opacity, and
//! one of five premultiplied blend modes, accumulated **in array order** over a
//! premultiplied-linear RGBA base.
//!
//! # Deterministic draw order
//!
//! The CPU reference composites splat `k` over the result of splats `0..k` at every
//! pixel, in array order. The GPU kernel reproduces this **without any cross-thread
//! ordering hazard** by dispatching *one invocation per pixel*: each invocation owns
//! its pixel's accumulator and loops the whole batch sequentially in array order, so
//! the per-pixel accumulation order is identical to the oracle's and there is no
//! atomic / blend-order race. A fixed batch on a fixed device therefore reruns
//! bit-identically (the determinism contract), and the result lands within the op's
//! bounded tolerance of the `f64` oracle (the kernel computes in `f32`; the op is
//! [`Bounded`](paintop_ir::DeterminismTier::Bounded)).
//!
//! Each invocation culls with the same conservative support box the oracle uses
//! (`GpuSplat::support_box`) — a per-pixel point-in-box test — so a splat whose
//! support does not cover the pixel is skipped exactly where its weight is `0.0`,
//! the bit-exact identity for every blend mode.
//!
//! # Fallback
//!
//! [`run_splats`] needs a live [`GpuContext`]. A caller with no adapter runs the CPU
//! reference (`paint.gaussian_splats@1`'s `cpu.reference` kernel), so a GPU-less host
//! still produces the correct result (`plan.md` §19 M3 criterion 4). The differential
//! harness gates the GPU backend on adapter presence (bn-3qa).

use super::error::GpuError;
use super::pipeline::{FusedExpr, FusedStage, PipelineCache, PipelineKey, ResourceFormat};
use super::pointwise::ExecutionTrace;
use super::probe::GpuContext;
use super::resource::{DeviceLimits, Dispatch, WorkgroupSize};
use super::splat::{SPLAT_WORDS, SplatBatchLayout, SplatTargetLayout};

use paintop_ir::Extent;
use serde_json::json;

/// The 2-D workgroup tile the splat kernel dispatches with (one invocation per
/// pixel). 8×8 is the conventional image tile (`WorkgroupSize::default`).
const SPLAT_WORKGROUP: WorkgroupSize = WorkgroupSize { x: 8, y: 8, z: 1 };

/// The painted output of a GPU splat run: the RGBA `f32` samples + the transfer
/// trace.
#[derive(Debug, Clone, PartialEq)]
pub struct SplatOutput {
    /// The painted RGBA samples, row-major, 4 channels per pixel (same length as the
    /// base).
    pub samples: Vec<f32>,
    /// The host↔GPU transfer trace (for the no-readback assertion, bn-3q0).
    pub trace: ExecutionTrace,
}

/// The pipeline-cache key for the splat kernel.
///
/// The splat kernel's generated WGSL is a pure function of the RGBA `f32` format
/// (the source is otherwise fixed — the batch and canvas are *data*, not shape), so
/// it keys on a single canonical `paint.gaussian_splats@1` fused expression on the
/// RGBA format. Two splat runs share the one compiled pipeline regardless of batch
/// contents, exactly the reuse the cache is for.
///
/// # Errors
/// Propagates the canonicalization error from [`PipelineKey::derive`] (unreachable
/// for the fixed canonical params here).
fn splat_pipeline_key() -> Result<PipelineKey, paintop_ir::Error> {
    let expr = FusedExpr::new().with(FusedStage::new(
        "paint.gaussian_splats@1".parse()?,
        json!({ "kernel": "wgpu.splat" }),
    ));
    PipelineKey::derive(&expr, ResourceFormat::f32(4))
}

/// Paint `batch` onto `base` on the GPU, reproducing the CPU oracle's array-order
/// accumulation deterministically.
///
/// `base` is a row-major premultiplied-linear RGBA `f32` buffer of
/// `extent.width * extent.height * 4` samples; the output has the same shape. The
/// whole batch is uploaded once and the result read back once (an
/// [`ExecutionTrace`] records the counts for bn-3q0). An **empty** batch is the
/// identity — the base is returned unchanged with no dispatch.
///
/// # Errors
/// - [`GpuError::DispatchInvalid`] if the base length is not `w·h·4`, the extent is
///   zero, the batch/target exceeds the device's storage/dispatch limits, or a splat
///   is degenerate (validated by [`SplatBatchLayout::build`]).
/// - a shader-compilation error is surfaced via `wgpu`'s validation error scope, so
///   a malformed generated shader fails explicitly rather than producing a wrong
///   result.
#[allow(
    clippy::too_many_lines,
    reason = "one cohesive GPU dispatch: validate, upload base+batch+params, bind, \
              dispatch, read back — splitting it would scatter the resource lifetimes"
)]
pub fn run_splats(
    context: &GpuContext,
    cache: &mut PipelineCache<wgpu::ComputePipeline>,
    batch: &SplatBatchLayout,
    base: &[f32],
    extent: Extent,
) -> Result<SplatOutput, GpuError> {
    let target = SplatTargetLayout::new(extent);
    let expected = target.sample_count()?;
    if base.len() as u64 != expected {
        return Err(GpuError::DispatchInvalid {
            reason: format!(
                "splat base buffer length {} != w*h*4 = {expected} for {}x{}",
                base.len(),
                extent.width,
                extent.height
            ),
        });
    }

    let limits = DeviceLimits::of_context(context);
    target.validate(&limits)?;

    // An empty batch is the identity: no dispatch, the base passes through (matching
    // the CPU op). The trace still records the (absent) upload/readback honestly.
    if batch.is_empty() {
        return Ok(SplatOutput {
            samples: base.to_vec(),
            trace: ExecutionTrace {
                uploads: 0,
                intermediate_readbacks: 0,
                export_readbacks: 0,
                stages: 0,
            },
        });
    }

    let dispatch = Dispatch::for_extent(extent, SPLAT_WORKGROUP, &limits)?;

    let device = context.device();
    let queue = context.queue();

    let key = splat_pipeline_key().map_err(|e| GpuError::DispatchInvalid {
        reason: format!("failed to derive splat pipeline key: {e}"),
    })?;
    let pipeline = compile_or_reuse(device, cache, &key)?;

    // ---- Uploads (1 upload of base + batch + params, counted as one host->GPU
    // transfer of the run's inputs). ----
    let base_bytes = to_byte_vec(base);
    let base_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("paintop splat base"),
        size: base_bytes.len() as u64,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    queue.write_buffer(&base_buffer, 0, &base_bytes);

    let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("paintop splat output"),
        size: base_bytes.len() as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let splat_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("paintop splat batch"),
        size: batch.bytes().len() as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&splat_buffer, 0, batch.bytes());

    // Params: width, height, splat_count (3 u32s).
    let params = [extent.width, extent.height, batch.count()];
    let mut params_bytes = Vec::with_capacity(params.len() * 4);
    for p in params {
        params_bytes.extend_from_slice(&p.to_ne_bytes());
    }
    let params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("paintop splat params"),
        size: params_bytes.len() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&params_buffer, 0, &params_bytes);

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("paintop splat bind group"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: base_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: output_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: splat_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: params_buffer.as_entire_binding(),
            },
        ],
    });

    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("paintop splat staging"),
        size: base_bytes.len() as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("paintop splat pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        let (gx, gy, gz) = dispatch.groups();
        pass.dispatch_workgroups(gx, gy, gz);
    }
    encoder.copy_buffer_to_buffer(&output_buffer, 0, &staging, 0, base_bytes.len() as u64);
    queue.submit(std::iter::once(encoder.finish()));

    let samples = read_back(device, &staging, base.len())?;

    Ok(SplatOutput {
        samples,
        trace: ExecutionTrace {
            uploads: 1,
            intermediate_readbacks: 0,
            export_readbacks: 1,
            stages: 1,
        },
    })
}

/// Compile the splat WGSL into a pipeline, or reuse the cached one for its key.
///
/// The device's validation error scope is pushed around the compile so a malformed
/// shader surfaces as an explicit [`GpuError::DispatchInvalid`] rather than a panic
/// or a silently-broken pipeline.
fn compile_or_reuse(
    device: &wgpu::Device,
    cache: &mut PipelineCache<wgpu::ComputePipeline>,
    key: &PipelineKey,
) -> Result<std::sync::Arc<wgpu::ComputePipeline>, GpuError> {
    let source = splat_wgsl();
    let (pipeline, _outcome) = cache.get_or_compile(key.clone(), || {
        device.push_error_scope(wgpu::ErrorFilter::Validation);
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("paintop splat shader"),
            source: wgpu::ShaderSource::Wgsl(source.clone().into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("paintop splat pipeline"),
            layout: None,
            module: &module,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        if let Some(err) = pollster::block_on(device.pop_error_scope()) {
            return Err(GpuError::DispatchInvalid {
                reason: format!("splat shader failed to compile: {err}"),
            });
        }
        Ok(pipeline)
    })?;
    Ok(pipeline)
}

/// The fixed splat WGSL compute shader.
///
/// One invocation per pixel. The pixel seeds its premultiplied RGBA accumulator from
/// the base, then loops the batch **in array order**, culling per-pixel against each
/// splat's conservative support box and compositing the splat's weighted
/// premultiplied source with the matching blend mode. The per-pixel sequential loop
/// reproduces the oracle's accumulation order with no cross-thread hazard.
///
/// The math mirrors `crates/paintop-cpu/src/splat.rs` `weight` + `composite`:
/// rotate the offset into the splat's local frame, form the Mahalanobis distance,
/// remap by the super-Gaussian exponent (`m^p`, with the `p == 1` fast path), and
/// `opacity * exp(-0.5 * shaped)`. The support box uses the same `K = sqrt(1500)`
/// σ-multiple and ±1px widening as the host [`super::splat`] layer.
fn splat_wgsl() -> String {
    // `SPLAT_WORDS` is the stride into the flat splat storage buffer; baked in so the
    // shader and the host serializer (`GpuSplat::to_words`) agree on the layout.
    let stride = SPLAT_WORDS;
    format!(
        r"// Generated GPU gaussian-splat kernel (deterministic array-order accumulation).
struct Params {{ width: u32, height: u32, splat_count: u32 }};

@group(0) @binding(0) var<storage, read> base: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<storage, read> splats: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

const STRIDE: u32 = {stride}u;
// K = sqrt(1500): the conservative support-box sigma multiple (matches the oracle).
const SUPPORT_K: f32 = 38.729833462074170f;

fn splat_f(i: u32, off: u32) -> f32 {{ return splats[i * STRIDE + off]; }}
fn splat_u(i: u32, off: u32) -> u32 {{ return bitcast<u32>(splats[i * STRIDE + off]); }}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let px = gid.x;
    let py = gid.y;
    if (px >= params.width || py >= params.height) {{ return; }}
    let pidx = (py * params.width + px) * 4u;

    // Seed the premultiplied RGBA accumulator from the base pixel.
    var acc = vec4<f32>(base[pidx], base[pidx + 1u], base[pidx + 2u], base[pidx + 3u]);

    // Pixel center in continuous coordinates (pixel i covers i + 0.5).
    let sx = f32(px) + 0.5f;
    let sy = f32(py) + 0.5f;

    // Accumulate every splat IN ARRAY ORDER (matches the CPU reference exactly).
    for (var i: u32 = 0u; i < params.splat_count; i = i + 1u) {{
        let cx = splat_f(i, 0u);
        let cy = splat_f(i, 1u);
        let sig_x = splat_f(i, 2u);
        let sig_y = splat_f(i, 3u);
        let angle = splat_f(i, 4u);
        let color = vec4<f32>(splat_f(i, 5u), splat_f(i, 6u), splat_f(i, 7u), splat_f(i, 8u));
        let opacity = splat_f(i, 9u);
        let exponent = splat_f(i, 10u);
        let blend = splat_u(i, 11u);

        // ---- Per-pixel support-box cull (bit-exact identity outside the box). ----
        let cosv = cos(angle);
        let sinv = sin(angle);
        let cs = cosv * sig_x;
        let sc = sinv * sig_y;
        let ss = sinv * sig_x;
        let cc = cosv * sig_y;
        let var_xx = cs * cs + sc * sc;
        let var_yy = ss * ss + cc * cc;
        let hx = SUPPORT_K * sqrt(var_xx);
        let hy = SUPPORT_K * sqrt(var_yy);
        if (sx < cx - hx - 1.0f || sx > cx + hx + 1.0f ||
            sy < cy - hy - 1.0f || sy > cy + hy + 1.0f) {{
            continue;
        }}

        // ---- Gaussian weight (matches `Splat::weight`). ----
        let off_x = sx - cx;
        let off_y = sy - cy;
        let local_u = sinv * off_y + cosv * off_x;
        let local_v = cosv * off_y - sinv * off_x;
        let nu = local_u / sig_x;
        let nv = local_v / sig_y;
        let maha = nu * nu + nv * nv;
        var shaped = maha;
        if (exponent != 1.0f) {{ shaped = pow(maha, exponent); }}
        let weight = opacity * exp(-0.5f * shaped);

        // ---- Composite (matches `Splat::composite`, premultiplied). ----
        let pre = color.a * weight;
        let src = vec4<f32>(color.r * pre, color.g * pre, color.b * pre, color.a * weight);
        let src_a = color.a * weight;
        let inv = 1.0f - src_a;

        if (blend == 0u) {{
            // Normal: source-over.
            acc = vec4<f32>(acc.r * inv + src.r, acc.g * inv + src.g,
                            acc.b * inv + src.b, acc.a * inv + src.a);
        }} else if (blend == 1u) {{
            // Multiply.
            acc = vec4<f32>(acc.r * inv + acc.r * color.r * src_a,
                            acc.g * inv + acc.g * color.g * src_a,
                            acc.b * inv + acc.b * color.b * src_a,
                            acc.a * inv + src.a);
        }} else if (blend == 2u) {{
            // Add: s + d.
            acc = acc + src;
        }} else if (blend == 3u) {{
            // Screen: s + d - s*d.
            acc = src + acc - src * acc;
        }} else {{
            // Lighten: max(s, d).
            acc = max(src, acc);
        }}
    }}

    output[pidx] = acc.r;
    output[pidx + 1u] = acc.g;
    output[pidx + 2u] = acc.b;
    output[pidx + 3u] = acc.a;
}}
",
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
    let _ = device.poll(wgpu::PollType::wait_indefinitely());
    match rx.recv() {
        Ok(Ok(())) => {}
        Ok(Err(e)) => {
            return Err(GpuError::DispatchInvalid {
                reason: format!("failed to map splat output for readback: {e}"),
            });
        }
        Err(e) => {
            return Err(GpuError::DispatchInvalid {
                reason: format!("splat output readback channel closed: {e}"),
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

/// Serialize an `f32` slice to an owned native-endian byte buffer for upload
/// (no pointer casting; the crate forbids `unsafe`).
fn to_byte_vec(samples: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(std::mem::size_of_val(samples));
    for &s in samples {
        bytes.extend_from_slice(&s.to_ne_bytes());
    }
    bytes
}

#[cfg(test)]
mod tests {
    use super::{SPLAT_WORKGROUP, splat_pipeline_key, splat_wgsl};

    #[test]
    fn wgsl_is_a_single_per_pixel_kernel() {
        let src = splat_wgsl();
        assert_eq!(src.matches("@compute").count(), 1);
        assert_eq!(src.matches("fn main").count(), 1);
        // Branches on all five blend tags.
        assert!(src.contains("blend == 0u"));
        assert!(src.contains("blend == 1u"));
        assert!(src.contains("blend == 2u"));
        assert!(src.contains("blend == 3u"));
        // The array-order accumulation loop is present.
        assert!(src.contains("i < params.splat_count"));
    }

    #[test]
    fn pipeline_key_is_stable_and_content_addressed() {
        let k1 = splat_pipeline_key().expect("key");
        let k2 = splat_pipeline_key().expect("key");
        assert_eq!(k1, k2);
        assert!(k1.as_str().starts_with("blake3:"));
    }

    #[test]
    fn workgroup_is_the_8x8_image_tile() {
        assert_eq!(
            (SPLAT_WORKGROUP.x, SPLAT_WORKGROUP.y, SPLAT_WORKGROUP.z),
            (8, 8, 1)
        );
    }
}
