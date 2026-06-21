//! GPU separable filters: the `wgpu` two-pass separable Gaussian for
//! `filter.gaussian_blur@1` / `filter.convolve@1` (`plan.md` §12.3; bn-b4c).
//!
//! The CPU oracle's direct 2-D sampled Gaussian factorizes into two 1-D passes (a
//! horizontal then a vertical convolution with the same normalized taps), exactly as
//! the `cpu.optimized` separable backend does (`crates/paintop-cpu/src/gaussian_blur.rs`).
//! This module runs those two passes on the GPU: each pass is one compute dispatch
//! reading the input texture and the 1-D tap buffer, writing the intermediate; the
//! second pass consumes that intermediate **without a host readback**, so a fully
//! GPU-resident filter chain incurs one upload and one final export readback
//! (bn-3q0).
//!
//! # Boundary parity (no seam)
//!
//! Each pass resolves an out-of-bounds tap with [`Boundary::source_index`], a
//! bit-faithful port of the oracle's per-axis `source_index`
//! (clamp / mirror / wrap / constant). Because the whole image is one GPU buffer —
//! never tiled — there is no tile/texture seam by construction: every output sample
//! reads from the full input plane exactly as the single-image CPU reference does.
//! The GPU result lands within the op's [`Bounded`](paintop_ir::DeterminismTier::Bounded)
//! tolerance of the `f64` oracle (the kernel accumulates in `f32`).
//!
//! # Fallback
//!
//! [`run_separable_gaussian`] needs a live [`GpuContext`]; a caller with no adapter
//! runs the `cpu.reference` / `cpu.optimized` separable backend, so a GPU-less host
//! still produces the correct result (`plan.md` §19 M3 criterion 4). The differential
//! harness gates this backend on adapter presence.

use super::error::GpuError;
use super::pipeline::{FusedExpr, FusedStage, PipelineCache, PipelineKey, ResourceFormat};
use super::pointwise::ExecutionTrace;
use super::probe::GpuContext;
use super::resource::{DeviceLimits, Dispatch, StorageBufferSpec, WorkgroupSize};

use paintop_ir::Extent;
use serde_json::json;

/// The 2-D workgroup tile each separable pass dispatches with (one invocation per
/// sample).
const SEPARABLE_WORKGROUP: WorkgroupSize = WorkgroupSize { x: 8, y: 8, z: 1 };

/// The per-axis boundary policy.
///
/// Mirrors the CPU oracle's `source_index`
/// (`crates/paintop-cpu/src/convolve.rs` / `gaussian_blur.rs`) exactly, so the GPU
/// passes reproduce the reference's boundary handling.
///
/// The integer tag is the wire contract between the host and the WGSL kernel (which
/// branches on it). The `constant` and `transparent` modes both blur against an
/// all-zero border (the only sensible default for a normalized smoothing kernel), so
/// they share the [`Constant`](Self::Constant) tag.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum Boundary {
    /// A fixed all-zero border for any out-of-bounds tap (`constant` / `transparent`).
    Constant = 0,
    /// Replicate the nearest edge sample.
    Clamp = 1,
    /// Whole-sample mirror across the edge (edge not repeated).
    Mirror = 2,
    /// Periodic (toroidal) tiling.
    Wrap = 3,
}

impl Boundary {
    /// The `u32` tag the WGSL kernel branches on.
    #[must_use]
    pub const fn tag(self) -> u32 {
        self as u32
    }

    /// Map the boundary `mode` token onto the per-axis policy, matching the CPU
    /// `BlurBoundary::from_mode`. An unrecognized token defaults to `clamp` (the
    /// op's documented default for any other valid token).
    #[must_use]
    pub fn from_mode(mode: &str) -> Self {
        match mode {
            "mirror" => Self::Mirror,
            "wrap" => Self::Wrap,
            "constant" | "transparent" => Self::Constant,
            _ => Self::Clamp,
        }
    }

    /// Resolve a 1-D `coord` to a source index in `[0, n)`, or `None` for the
    /// all-zero constant border. Bit-faithful to the oracle's `source_index` per
    /// axis (`n >= 1`). Used by the host-side parity check; the WGSL kernel applies
    /// the same logic.
    #[must_use]
    pub fn source_index(self, coord: i64, n: i64) -> Option<i64> {
        if coord >= 0 && coord < n {
            return Some(coord);
        }
        match self {
            Self::Constant => None,
            Self::Clamp => Some(coord.clamp(0, n - 1)),
            Self::Wrap => Some(coord.rem_euclid(n)),
            Self::Mirror => {
                if n == 1 {
                    Some(0)
                } else {
                    let period = 2 * (n - 1);
                    let m = coord.rem_euclid(period);
                    Some(if m < n { m } else { period - m })
                }
            }
        }
    }
}

/// The output of a GPU separable filter run: the filtered samples + the transfer
/// trace.
#[derive(Debug, Clone, PartialEq)]
pub struct SeparableOutput {
    /// The filtered samples, same length/layout as the input.
    pub samples: Vec<f32>,
    /// The host↔GPU transfer trace (one upload, one export readback; zero
    /// intermediate readbacks — the inter-pass buffer never leaves the GPU).
    pub trace: ExecutionTrace,
}

/// The pipeline-cache key for the separable pass on a given format. The WGSL is a
/// pure function of the channel count, so one compiled pipeline serves every
/// separable run on that format (both passes share it; the axis/boundary/taps are
/// uniform/storage *data*, not shape).
///
/// # Errors
/// Propagates the canonicalization error from [`PipelineKey::derive`] (unreachable
/// for the fixed canonical params here).
fn separable_pipeline_key(channels: u32) -> Result<PipelineKey, paintop_ir::Error> {
    let expr = FusedExpr::new().with(FusedStage::new(
        "filter.gaussian_blur@1".parse()?,
        json!({ "kernel": "wgpu.separable" }),
    ));
    PipelineKey::derive(&expr, ResourceFormat::f32(channels))
}

/// Run a separable Gaussian over `samples` on the GPU as two 1-D passes (horizontal
/// then vertical), reproducing the CPU oracle within the op's bounded tolerance.
///
/// `samples` is a row-major, `channels`-interleaved `f32` buffer of `extent`; `taps`
/// is the normalized 1-D Gaussian (length `2r+1`, hot tap at index `r`), the same
/// taps the `cpu.optimized` separable backend builds. The two passes run over a pair
/// of GPU-resident ping-scratch buffers; the intermediate never returns to the host, so
/// the run uploads once and reads back once ([`ExecutionTrace`]).
///
/// A `taps` of length 1 (the σ→0 identity) is a pass-through: the input is returned
/// unchanged with no dispatch.
///
/// # Errors
/// - [`GpuError::DispatchInvalid`] if the buffer length is not `w·h·channels`, the
///   extent/channels are zero, the taps are empty or even-length, or a buffer/dispatch
///   exceeds the device limits.
/// - a shader-compilation error surfaces via `wgpu`'s validation error scope.
#[allow(
    clippy::too_many_lines,
    reason = "one cohesive two-pass GPU run: validate, upload, dispatch H then V over \
              ping-scratch buffers, read back once — splitting it scatters the buffer lifetimes"
)]
pub fn run_separable_gaussian(
    context: &GpuContext,
    cache: &mut PipelineCache<wgpu::ComputePipeline>,
    samples: &[f32],
    extent: Extent,
    channels: u32,
    taps: &[f32],
    boundary: Boundary,
) -> Result<SeparableOutput, GpuError> {
    if channels == 0 || extent.width == 0 || extent.height == 0 {
        return Err(GpuError::DispatchInvalid {
            reason: format!(
                "separable filter needs a non-empty extent and channels, got {}x{}x{channels}",
                extent.width, extent.height
            ),
        });
    }
    let expected = u64::from(extent.width) * u64::from(extent.height) * u64::from(channels);
    if samples.len() as u64 != expected {
        return Err(GpuError::DispatchInvalid {
            reason: format!(
                "separable input length {} != w*h*channels = {expected}",
                samples.len()
            ),
        });
    }
    if taps.is_empty() || taps.len().is_multiple_of(2) {
        return Err(GpuError::DispatchInvalid {
            reason: format!(
                "separable taps must be a non-empty odd length, got {}",
                taps.len()
            ),
        });
    }
    // The σ→0 identity (a single unit tap) is a pass-through; no dispatch.
    if taps.len() == 1 {
        return Ok(SeparableOutput {
            samples: samples.to_vec(),
            trace: ExecutionTrace {
                uploads: 0,
                intermediate_readbacks: 0,
                export_readbacks: 0,
                stages: 0,
            },
        });
    }
    let radius = u32::try_from((taps.len() - 1) / 2).map_err(|_| GpuError::DispatchInvalid {
        reason: "separable tap radius exceeds u32".to_owned(),
    })?;

    let limits = DeviceLimits::of_context(context);
    StorageBufferSpec::new(samples.len() as u64).validate(&limits)?;
    StorageBufferSpec::new(taps.len() as u64).validate(&limits)?;
    let dispatch = Dispatch::for_extent(extent, SEPARABLE_WORKGROUP, &limits)?;

    let device = context.device();
    let queue = context.queue();

    let key = separable_pipeline_key(channels).map_err(|e| GpuError::DispatchInvalid {
        reason: format!("failed to derive separable pipeline key: {e}"),
    })?;
    let pipeline = compile_or_reuse(device, cache, channels, &key)?;

    // ---- Buffers: ping (input), scratch (intermediate/output), taps, params. ----
    let byte_len = (samples.len() * 4) as u64;
    let input_bytes = to_byte_vec(samples);
    // `ping` holds the input, then the vertical pass writes the final result back
    // into it, which is copied to staging — so it needs COPY_DST (upload) and
    // COPY_SRC (final copy). `scratch` only ever holds the GPU-resident intermediate.
    let ping = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("paintop separable ping"),
        size: byte_len,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    queue.write_buffer(&ping, 0, &input_bytes);
    let scratch = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("paintop separable scratch"),
        size: byte_len,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });

    let taps_bytes = to_byte_vec(taps);
    let taps_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("paintop separable taps"),
        size: taps_bytes.len() as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&taps_buffer, 0, &taps_bytes);

    // Params for one pass: width, height, channels, radius, boundary, horizontal.
    let make_params = |horizontal: u32| -> Vec<u8> {
        let words = [
            extent.width,
            extent.height,
            channels,
            radius,
            boundary.tag(),
            horizontal,
        ];
        let mut b = Vec::with_capacity(words.len() * 4);
        for w in words {
            b.extend_from_slice(&w.to_ne_bytes());
        }
        b
    };
    let params_h = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("paintop separable params H"),
        size: 24,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&params_h, 0, &make_params(1));
    let params_v = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("paintop separable params V"),
        size: 24,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&params_v, 0, &make_params(0));

    let layout = pipeline.get_bind_group_layout(0);
    // Pass 1 (horizontal): ping -> scratch.
    let bind_h = make_bind_group(device, &layout, &ping, &scratch, &taps_buffer, &params_h);
    // Pass 2 (vertical): scratch -> ping (reuse ping as the output target).
    let bind_v = make_bind_group(device, &layout, &scratch, &ping, &taps_buffer, &params_v);

    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("paintop separable staging"),
        size: byte_len,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    let (gx, gy, gz) = dispatch.groups();
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("paintop separable H"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_h, &[]);
        pass.dispatch_workgroups(gx, gy, gz);
    }
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("paintop separable V"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_v, &[]);
        pass.dispatch_workgroups(gx, gy, gz);
    }
    // The vertical pass wrote into `ping`; copy that to the staging buffer.
    encoder.copy_buffer_to_buffer(&ping, 0, &staging, 0, byte_len);
    queue.submit(std::iter::once(encoder.finish()));

    let out = read_back(device, &staging, samples.len())?;

    Ok(SeparableOutput {
        samples: out,
        trace: ExecutionTrace {
            uploads: 1,
            // The horizontal pass's result stays GPU-resident for the vertical pass.
            intermediate_readbacks: 0,
            export_readbacks: 1,
            stages: 2,
        },
    })
}

/// Build a separable-pass bind group: `src` (read), `dst` (`read_write`), taps, params.
fn make_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    src: &wgpu::Buffer,
    dst: &wgpu::Buffer,
    taps: &wgpu::Buffer,
    params: &wgpu::Buffer,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("paintop separable bind group"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: src.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: dst.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: taps.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: params.as_entire_binding(),
            },
        ],
    })
}

/// Compile the separable WGSL into a pipeline, or reuse the cached one for its key.
fn compile_or_reuse(
    device: &wgpu::Device,
    cache: &mut PipelineCache<wgpu::ComputePipeline>,
    channels: u32,
    key: &PipelineKey,
) -> Result<std::sync::Arc<wgpu::ComputePipeline>, GpuError> {
    let source = separable_wgsl(channels);
    let (pipeline, _outcome) = cache.get_or_compile(key.clone(), || {
        device.push_error_scope(wgpu::ErrorFilter::Validation);
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("paintop separable shader"),
            source: wgpu::ShaderSource::Wgsl(source.clone().into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("paintop separable pipeline"),
            layout: None,
            module: &module,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        if let Some(err) = pollster::block_on(device.pop_error_scope()) {
            return Err(GpuError::DispatchInvalid {
                reason: format!("separable shader failed to compile: {err}"),
            });
        }
        Ok(pipeline)
    })?;
    Ok(pipeline)
}

/// The separable 1-D convolution WGSL, specialized to `channels`.
///
/// One invocation per sample. For output `(x, y)` it slides the `2r+1` taps along
/// the selected axis (`horizontal`), accumulating `taps[k] * src(boundary(coord))`
/// per channel — the same fixed-order accumulation the oracle's per-axis pass
/// performs. The boundary index map (`source_index`) branches on the `boundary` tag,
/// matching [`Boundary::source_index`] bit-for-bit.
fn separable_wgsl(channels: u32) -> String {
    format!(
        r"// Generated GPU separable 1-D convolution pass.
struct Params {{ width: u32, height: u32, channels: u32, radius: u32, boundary: u32, horizontal: u32 }};

@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@group(0) @binding(2) var<storage, read> taps: array<f32>;
@group(0) @binding(3) var<uniform> p: Params;

const CHANNELS: u32 = {channels}u;

// A euclidean modulo `a mod m` in `[0, m)` computed with add/subtract loops rather
// than the `%` operator: some naga/driver targets miscompile signed `i32 %` as an
// unsigned remainder for negative operands, so we avoid it entirely. The overhang is
// bounded by the tap radius, so the loops run a small, bounded number of times.
fn emod(a: i32, m: i32) -> i32 {{
    var v = a;
    loop {{
        if (v >= 0) {{ break; }}
        v = v + m;
    }}
    loop {{
        if (v < m) {{ break; }}
        v = v - m;
    }}
    return v;
}}

// Mirror (whole-sample reflection, edge not repeated): fold `coord` into
// `[0, 2(n-1))`, then reflect the second half back.
fn mirror_index(coord: i32, n: i32) -> i32 {{
    if (n == 1) {{ return 0; }}
    let period = 2 * (n - 1);
    let m = emod(coord, period);
    return select(m, period - m, m >= n);
}}

// Resolve a 1-D coord to a source index in [0, n), or -1 for the constant border.
fn source_index(coord: i32, n: i32, boundary: u32) -> i32 {{
    if (coord >= 0 && coord < n) {{ return coord; }}
    switch (boundary) {{
        case 0u: {{ return -1; }}                   // Constant: all-zero border.
        case 1u: {{ return clamp(coord, 0, n - 1); }}  // Clamp.
        case 3u: {{ return emod(coord, n); }}          // Wrap (toroidal).
        default: {{ return mirror_index(coord, n); }}  // Mirror.
    }}
}}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let x = gid.x;
    let y = gid.y;
    if (x >= p.width || y >= p.height) {{ return; }}

    let r = i32(p.radius);
    let horizontal = p.horizontal != 0u;
    let axis_len = select(i32(p.height), i32(p.width), horizontal);
    let pos = select(i32(y), i32(x), horizontal);
    let base = (y * p.width + x) * CHANNELS;

    for (var ch: u32 = 0u; ch < CHANNELS; ch = ch + 1u) {{
        var acc: f32 = 0.0;
        for (var k: i32 = 0; k <= 2 * r; k = k + 1) {{
            let w = taps[u32(k)];
            if (w == 0.0) {{ continue; }}
            let coord = pos + (k - r);
            let idx = source_index(coord, axis_len, p.boundary);
            if (idx < 0) {{ continue; }}             // constant border contributes 0.
            var sidx: u32;
            if (horizontal) {{
                sidx = (y * p.width + u32(idx)) * CHANNELS + ch;
            }} else {{
                sidx = (u32(idx) * p.width + x) * CHANNELS + ch;
            }}
            acc = acc + w * src[sidx];
        }}
        dst[base + ch] = acc;
    }}
}}
",
    )
}

/// Map the staging buffer and copy its `len` `f32`s back to the host.
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
                reason: format!("failed to map separable output for readback: {e}"),
            });
        }
        Err(e) => {
            return Err(GpuError::DispatchInvalid {
                reason: format!("separable output readback channel closed: {e}"),
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

/// Serialize an `f32` slice to an owned native-endian byte buffer (no `unsafe`).
fn to_byte_vec(samples: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(std::mem::size_of_val(samples));
    for &s in samples {
        bytes.extend_from_slice(&s.to_ne_bytes());
    }
    bytes
}

#[cfg(test)]
mod tests {
    use super::{Boundary, SEPARABLE_WORKGROUP, separable_pipeline_key, separable_wgsl};

    #[test]
    fn boundary_source_index_matches_oracle_semantics() {
        // Clamp.
        assert_eq!(Boundary::Clamp.source_index(-2, 5), Some(0));
        assert_eq!(Boundary::Clamp.source_index(7, 5), Some(4));
        // Wrap (toroidal).
        assert_eq!(Boundary::Wrap.source_index(-1, 5), Some(4));
        assert_eq!(Boundary::Wrap.source_index(6, 5), Some(1));
        // Constant: out-of-bounds -> None (zero border).
        assert_eq!(Boundary::Constant.source_index(-1, 5), None);
        assert_eq!(Boundary::Constant.source_index(5, 5), None);
        // Mirror (whole-sample, edge not repeated).
        assert_eq!(Boundary::Mirror.source_index(-1, 5), Some(1));
        assert_eq!(Boundary::Mirror.source_index(5, 5), Some(3));
        // In-bounds is identity for every mode.
        for b in [
            Boundary::Clamp,
            Boundary::Wrap,
            Boundary::Constant,
            Boundary::Mirror,
        ] {
            assert_eq!(b.source_index(2, 5), Some(2));
        }
    }

    #[test]
    fn from_mode_maps_tokens_like_the_cpu_backend() {
        assert_eq!(Boundary::from_mode("mirror"), Boundary::Mirror);
        assert_eq!(Boundary::from_mode("wrap"), Boundary::Wrap);
        assert_eq!(Boundary::from_mode("constant"), Boundary::Constant);
        assert_eq!(Boundary::from_mode("transparent"), Boundary::Constant);
        assert_eq!(Boundary::from_mode("clamp"), Boundary::Clamp);
        // Any other valid token defaults to clamp.
        assert_eq!(Boundary::from_mode("anything"), Boundary::Clamp);
    }

    #[test]
    fn boundary_tags_are_stable() {
        assert_eq!(Boundary::Constant.tag(), 0);
        assert_eq!(Boundary::Clamp.tag(), 1);
        assert_eq!(Boundary::Mirror.tag(), 2);
        assert_eq!(Boundary::Wrap.tag(), 3);
    }

    #[test]
    fn wgsl_is_a_single_pass_kernel_specialized_to_channels() {
        let rgba = separable_wgsl(4);
        assert_eq!(rgba.matches("@compute").count(), 1);
        assert_eq!(rgba.matches("fn main").count(), 1);
        assert!(rgba.contains("const CHANNELS: u32 = 4u;"));
        let gray = separable_wgsl(1);
        assert!(gray.contains("const CHANNELS: u32 = 1u;"));
        // Branches on every boundary tag via the source_index switch, and uses the
        // `%`-free euclidean modulo helper (a naga/driver target miscompiles signed
        // `i32 %` on negative operands).
        assert!(rgba.contains("switch (boundary)"));
        assert!(rgba.contains("case 0u"));
        assert!(rgba.contains("case 1u"));
        assert!(rgba.contains("case 3u"));
        assert!(rgba.contains("fn emod("));
        assert!(rgba.contains("fn mirror_index("));
    }

    #[test]
    fn pipeline_key_separates_by_channel_count() {
        let k4 = separable_pipeline_key(4).expect("key");
        let k1 = separable_pipeline_key(1).expect("key");
        assert_ne!(k4, k1, "channel count is part of the key");
        assert_eq!(k4, separable_pipeline_key(4).expect("key"));
    }

    #[test]
    fn workgroup_is_the_8x8_tile() {
        assert_eq!(
            (
                SEPARABLE_WORKGROUP.x,
                SEPARABLE_WORKGROUP.y,
                SEPARABLE_WORKGROUP.z
            ),
            (8, 8, 1)
        );
    }
}
