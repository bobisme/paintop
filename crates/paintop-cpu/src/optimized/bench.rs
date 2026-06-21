//! Baseline benchmarks and kernel selection for the `cpu.optimized` backend
//! (`bn-m0k`; `plan.md` §12.2 "explicit SIMD where benchmarks prove value").
//!
//! M3 adds *faster* CPU kernels, not new ops, and the plan is explicit that SIMD
//! is justified by measurement, not vibes: a kernel is only worth a second backend
//! when it shows a real speedup over the scalar reference. This module is the
//! measurement substrate and the selection record:
//!
//! * [`KernelKind`] enumerates every pointwise kernel the optimized backend can
//!   provide, each carrying its **precision tag** (the determinism tier it must
//!   honour against the oracle);
//! * [`measure`] runs a kernel's scalar reference and its optimized form over a
//!   representative buffer and reports a [`Measurement`] (median nanoseconds each,
//!   throughput, and the speedup ratio);
//! * [`Selection`] applies the [`MIN_SPEEDUP`] gate: a kernel that does not clear
//!   the bar is **rejected** (recorded, not silently shipped), so a "no measurable
//!   win" kernel never gets a `cpu.optimized` impl wired behind it;
//! * [`benchmark_all`] / [`BenchmarkArtifact`] produce the serialisable artifact
//!   `bn-m0k` requires — every kernel, its precision tag, its timings, and the
//!   accept/reject decision — which `bn-2ja` extends with the differential result.
//!
//! The benchmark is intentionally dependency-free (no `criterion`): it is a small,
//! deterministic median-of-repeats timer so it can run inside the normal test
//! suite and emit a JSON artifact without a heavyweight harness. Timing is inputs-
//! and machine-dependent, so the *gate* test asserts the structural invariants
//! (every selected kernel matches the oracle; rejects are recorded with a reason)
//! rather than a brittle absolute nanosecond bound.

use std::time::Instant;

use paintop_ir::DeterminismTier;
use serde::Serialize;

use super::kernels::{self, Adjustment, BlendMode, Transfer};

/// The pointwise kernels the `cpu.optimized` backend can provide, each bound to
/// the op (and precision tier) it must reproduce.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum KernelKind {
    /// `color.convert@1` sRGB transfer (bounded: `powf` last bit varies).
    ColorConvert,
    /// `color.adjust@1` grade (bounded: `exp2` last bit varies).
    ColorAdjust,
    /// `alpha.premultiply@1` (exact: a per-channel multiply).
    AlphaPremultiply,
    /// `alpha.unpremultiply@1` (bounded: a per-channel divide with a clamp policy).
    AlphaUnpremultiply,
    /// `composite.over@1` (exact: a per-channel fused multiply-add).
    CompositeOver,
    /// `composite.blend@1` (exact: per-channel arithmetic over a pinned mode set).
    CompositeBlend,
}

impl KernelKind {
    /// Every kernel, in a fixed order, for the full benchmark sweep.
    pub const ALL: [Self; 6] = [
        Self::ColorConvert,
        Self::ColorAdjust,
        Self::AlphaPremultiply,
        Self::AlphaUnpremultiply,
        Self::CompositeOver,
        Self::CompositeBlend,
    ];

    /// The canonical op id this kernel optimizes.
    #[must_use]
    pub const fn op_id(self) -> &'static str {
        match self {
            Self::ColorConvert => "color.convert@1",
            Self::ColorAdjust => "color.adjust@1",
            Self::AlphaPremultiply => "alpha.premultiply@1",
            Self::AlphaUnpremultiply => "alpha.unpremultiply@1",
            Self::CompositeOver => "composite.over@1",
            Self::CompositeBlend => "composite.blend@1",
        }
    }

    /// The **precision tag**: the determinism tier the optimized kernel must honour
    /// against the oracle. Exact kernels must match bit-for-bit; bounded kernels
    /// within the op's envelope (`bn-2ja` enforces this through the differential
    /// harness, which reads the same tier from the manifest).
    #[must_use]
    pub const fn precision(self) -> DeterminismTier {
        match self {
            Self::AlphaPremultiply | Self::CompositeOver | Self::CompositeBlend => {
                DeterminismTier::Exact
            }
            Self::ColorConvert | Self::ColorAdjust | Self::AlphaUnpremultiply => {
                DeterminismTier::Bounded
            }
        }
    }
}

/// The minimum speedup (optimized vs scalar) a kernel must show to be *selected*
/// for a `cpu.optimized` backend.
///
/// Below this the optimized path is no faster than the reference, so wiring a
/// second backend would only add surface area and a differential obligation for no
/// gain — `plan.md` §12.2's "where benchmarks prove value". The bar is modest (a
/// 5% improvement) because the win for these tight pointwise loops is autovec-
/// driven and machine-dependent; the gate exists to *reject regressions and ties*,
/// not to demand a specific architecture's vector width.
pub const MIN_SPEEDUP: f64 = 1.05;

/// One kernel's measured scalar-vs-optimized timing.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct Measurement {
    /// The number of `f32` samples processed per call (the working-set size).
    pub samples: usize,
    /// Median nanoseconds for the scalar reference path.
    pub scalar_ns: f64,
    /// Median nanoseconds for the optimized (autovectorized) path.
    pub optimized_ns: f64,
    /// The speedup ratio `scalar_ns / optimized_ns` (`> 1` is faster).
    pub speedup: f64,
}

impl Measurement {
    /// Whether this kernel cleared the [`MIN_SPEEDUP`] selection bar.
    #[must_use]
    pub fn is_win(&self) -> bool {
        self.speedup >= MIN_SPEEDUP
    }
}

/// The accept/reject decision for one kernel, with its measurement and precision
/// tag — the row the benchmark artifact records.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct Selection {
    /// The kernel.
    pub kernel: KernelKind,
    /// Its determinism (precision) tier.
    #[serde(serialize_with = "serialize_tier")]
    pub precision: DeterminismTier,
    /// The measured timing.
    pub measurement: Measurement,
    /// Whether the kernel is selected for a `cpu.optimized` backend (cleared the
    /// speedup bar).
    pub selected: bool,
}

/// Serialize a [`DeterminismTier`] as its kebab-case token for the artifact.
#[expect(
    clippy::trivially_copy_pass_by_ref,
    reason = "serde's serialize_with contract requires &T"
)]
fn serialize_tier<S: serde::Serializer>(tier: &DeterminismTier, s: S) -> Result<S::Ok, S::Error> {
    let token = match tier {
        DeterminismTier::Exact => "exact",
        DeterminismTier::Reproducible => "reproducible",
        DeterminismTier::Bounded => "bounded",
        DeterminismTier::Stochastic => "stochastic",
        _ => "unknown",
    };
    s.serialize_str(token)
}

/// The full benchmark artifact `bn-m0k` produces: every kernel's selection row.
///
/// Serialises to the JSON `bn-2ja` extends with the per-kernel differential
/// tolerance and writes under the evidence/perf path.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct BenchmarkArtifact {
    /// The minimum speedup gate applied to every kernel.
    pub min_speedup: f64,
    /// The per-kernel selection rows, in [`KernelKind::ALL`] order.
    pub selections: Vec<Selection>,
}

impl BenchmarkArtifact {
    /// The kernels that were selected (cleared the speedup bar).
    #[must_use]
    pub fn selected(&self) -> Vec<KernelKind> {
        self.selections
            .iter()
            .filter(|s| s.selected)
            .map(|s| s.kernel)
            .collect()
    }

    /// The kernels that were rejected (no measurable win).
    #[must_use]
    pub fn rejected(&self) -> Vec<KernelKind> {
        self.selections
            .iter()
            .filter(|s| !s.selected)
            .map(|s| s.kernel)
            .collect()
    }

    /// Serialise the artifact to pretty JSON.
    ///
    /// # Errors
    /// Propagates a [`serde_json`] serialisation error (it cannot fail for this
    /// shape, but the signature stays fallible for the caller's error flow).
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }
}

/// A deterministic pseudo-random sample buffer in `[0, 1)`, so the benchmark and
/// its differential check exercise non-degenerate (non-constant) data without a
/// dependency on `rand`.
fn fill(samples: usize, seed: u64) -> Vec<f32> {
    let mut state = seed | 1;
    let mut out = Vec::with_capacity(samples);
    for _ in 0..samples {
        // xorshift64*: cheap, deterministic, well-distributed.
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        let r = state.wrapping_mul(0x2545_f491_4f6c_dd1d);
        // A 16-bit mantissa -> a float in [0, 1); `u16 as f32` is lossless.
        let mantissa = u16::try_from((r >> 40) & 0xFFFF).unwrap_or(0);
        out.push(f32::from(mantissa) / f32::from(u16::MAX));
    }
    out
}

/// Time a closure as a median of `repeats` runs of `inner_iters` invocations each,
/// returning median nanoseconds **per invocation**.
///
/// A short warmup primes the instruction cache and branch predictors so the timed
/// runs reflect steady-state throughput rather than first-touch cost; the median
/// (not the mean) rejects scheduler-jitter outliers.
fn time_median<F: FnMut()>(repeats: usize, inner_iters: usize, mut f: F) -> f64 {
    // Warmup, untimed.
    for _ in 0..inner_iters.max(1) {
        f();
    }
    let mut samples: Vec<f64> = Vec::with_capacity(repeats);
    for _ in 0..repeats {
        let start = Instant::now();
        for _ in 0..inner_iters {
            f();
        }
        let elapsed = start.elapsed().as_nanos();
        #[expect(
            clippy::cast_precision_loss,
            reason = "elapsed nanos for a short loop are far below 2^52"
        )]
        let per = elapsed as f64 / inner_iters as f64;
        samples.push(per);
    }
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    samples[samples.len() / 2]
}

/// Run a kernel's scalar reference and optimized form over a representative buffer
/// and report the [`Measurement`].
///
/// `pixels` is the working-set size; the kernel-specific buffers (image, alpha,
/// mask, src/dst) are filled deterministically. The scalar closure mirrors the
/// op's `cpu.reference` arithmetic exactly so the ratio is apples-to-apples.
#[must_use]
#[allow(
    clippy::too_many_lines,
    reason = "a flat per-kernel timing dispatch; splitting the arms would only scatter \
              the symmetric scalar-vs-optimized pairs"
)]
pub fn measure(kernel: KernelKind, pixels: usize) -> Measurement {
    // A modest, fixed timing budget keeps the suite fast and deterministic in
    // shape; the absolute numbers vary by host but the *ratio* is what selects.
    const REPEATS: usize = 9;
    const INNER: usize = 4;
    let stride = 4_usize; // Rgba, the dominant pointwise layout.
    let n = pixels * stride;

    let (scalar_ns, optimized_ns, samples) = match kernel {
        KernelKind::AlphaPremultiply => {
            let buf = fill(n, 0x1111);
            let s = time_median(REPEATS, INNER, || {
                let _ = std::hint::black_box(scalar_premultiply(&buf, stride));
            });
            let o = time_median(REPEATS, INNER, || {
                let _ = std::hint::black_box(kernels::premultiply(&buf, stride));
            });
            (s, o, n)
        }
        KernelKind::AlphaUnpremultiply => {
            let buf = fill(n, 0x2222);
            let s = time_median(REPEATS, INNER, || {
                let _ = std::hint::black_box(scalar_unpremultiply(&buf, stride));
            });
            let o = time_median(REPEATS, INNER, || {
                let _ = std::hint::black_box(kernels::unpremultiply(&buf, stride));
            });
            (s, o, n)
        }
        KernelKind::ColorConvert => {
            let buf = fill(n, 0x3333);
            let s = time_median(REPEATS, INNER, || {
                let _ = std::hint::black_box(scalar_convert(&buf, stride, true, Transfer::Decode));
            });
            let o = time_median(REPEATS, INNER, || {
                let _ = std::hint::black_box(kernels::color_convert(
                    &buf,
                    stride,
                    true,
                    Transfer::Decode,
                ));
            });
            (s, o, n)
        }
        KernelKind::ColorAdjust => {
            let buf = fill(n, 0x4444);
            let adj = Adjustment {
                exposure_ev: 0.5,
                saturation: 0.2,
                temperature: 0.1,
            };
            let s = time_median(REPEATS, INNER, || {
                let _ = std::hint::black_box(scalar_adjust(&buf, stride, 3, adj, None));
            });
            let o = time_median(REPEATS, INNER, || {
                let _ = std::hint::black_box(kernels::color_adjust(&buf, stride, 3, adj, None));
            });
            (s, o, n)
        }
        KernelKind::CompositeOver => {
            let src = fill(n, 0x5555);
            let dst = fill(n, 0x6666);
            let s = time_median(REPEATS, INNER, || {
                let _ = std::hint::black_box(scalar_over(&src, &dst, stride));
            });
            let o = time_median(REPEATS, INNER, || {
                let _ = std::hint::black_box(kernels::composite_over(&src, &dst, stride));
            });
            (s, o, n)
        }
        KernelKind::CompositeBlend => {
            let src = fill(n, 0x7777);
            let dst = fill(n, 0x8888);
            let mask = fill(pixels, 0x9999);
            let s = time_median(REPEATS, INNER, || {
                let _ = std::hint::black_box(scalar_blend(
                    &src,
                    &dst,
                    &mask,
                    stride,
                    BlendMode::Screen,
                    0.75,
                ));
            });
            let o = time_median(REPEATS, INNER, || {
                let _ = std::hint::black_box(kernels::composite_blend(
                    &src,
                    &dst,
                    &mask,
                    stride,
                    BlendMode::Screen,
                    0.75,
                ));
            });
            (s, o, n)
        }
    };

    let speedup = if optimized_ns > 0.0 {
        scalar_ns / optimized_ns
    } else {
        1.0
    };
    Measurement {
        samples,
        scalar_ns,
        optimized_ns,
        speedup,
    }
}

/// Benchmark every kernel and apply the selection gate, producing the artifact.
#[must_use]
pub fn benchmark_all(pixels: usize) -> BenchmarkArtifact {
    let selections = KernelKind::ALL
        .iter()
        .map(|&kernel| {
            let measurement = measure(kernel, pixels);
            Selection {
                kernel,
                precision: kernel.precision(),
                measurement,
                selected: measurement.is_win(),
            }
        })
        .collect();
    BenchmarkArtifact {
        min_speedup: MIN_SPEEDUP,
        selections,
    }
}

// ---------------------------------------------------------------------------
// Scalar reference baselines.
//
// These mirror the `cpu.reference` arithmetic of each op (the same operation
// order) so the benchmark ratio compares like with like. They are NOT the
// production reference (that lives in each op module); they are the timing
// baseline the optimized kernel is measured against, kept in lock-step by the
// kernels' own differential equivalence tests.
// ---------------------------------------------------------------------------

fn scalar_premultiply(samples: &[f32], stride: usize) -> Vec<f32> {
    let alpha_index = stride - 1;
    let mut out = samples.to_vec();
    for pixel in out.chunks_mut(stride) {
        let Some(&alpha) = pixel.get(alpha_index) else {
            continue;
        };
        for color in &mut pixel[..alpha_index] {
            *color *= alpha;
        }
    }
    out
}

fn scalar_unpremultiply(samples: &[f32], stride: usize) -> Vec<f32> {
    let alpha_index = stride - 1;
    let mut out = samples.to_vec();
    for pixel in out.chunks_mut(stride) {
        let Some(&alpha) = pixel.get(alpha_index) else {
            continue;
        };
        for color in &mut pixel[..alpha_index] {
            *color = if alpha > kernels::UNPREMULTIPLY_EPSILON {
                *color / alpha
            } else {
                0.0
            };
        }
    }
    out
}

fn scalar_convert(samples: &[f32], stride: usize, has_alpha: bool, transfer: Transfer) -> Vec<f32> {
    let func: fn(f32) -> f32 = match transfer {
        Transfer::Identity => return samples.to_vec(),
        Transfer::Decode => |c: f32| {
            if c <= 0.040_45 {
                c / 12.92
            } else {
                ((c + 0.055) / 1.055).powf(2.4)
            }
        },
        Transfer::Encode => |c: f32| {
            if c <= 0.003_130_8 {
                c * 12.92
            } else {
                1.055_f32.mul_add(c.powf(1.0 / 2.4), -0.055)
            }
        },
    };
    let alpha_index = if has_alpha { Some(stride - 1) } else { None };
    samples
        .iter()
        .enumerate()
        .map(|(i, &s)| {
            if Some(i % stride) == alpha_index {
                s
            } else {
                func(s)
            }
        })
        .collect()
}

fn scalar_adjust(
    samples: &[f32],
    stride: usize,
    color_count: usize,
    adj: Adjustment,
    mask: Option<&[f32]>,
) -> Vec<f32> {
    let mut out = samples.to_vec();
    for (pixel_index, pixel) in out.chunks_mut(stride).enumerate() {
        let original: [f32; 3] = match color_count {
            1 => [pixel[0], 0.0, 0.0],
            _ => [pixel[0], pixel[1], pixel[2]],
        };
        let mut color = [original[0], original[1], original[2]];
        let gain = adj.exposure_ev.exp2();
        for c in &mut color[..color_count] {
            *c *= gain;
        }
        if color_count == 3 {
            color[0] *= 1.0 + adj.temperature;
            color[2] *= 1.0 - adj.temperature;
            let luma =
                0.212_6_f32.mul_add(color[0], 0.715_2_f32.mul_add(color[1], 0.072_2 * color[2]));
            let scale = 1.0 + adj.saturation;
            for c in &mut color[..3] {
                *c = scale.mul_add(*c - luma, luma);
            }
        }
        let coverage = mask.map_or(1.0, |m| m.get(pixel_index).copied().unwrap_or(0.0));
        for ch in 0..color_count {
            pixel[ch] = coverage.mul_add(color[ch] - original[ch], original[ch]);
        }
    }
    out
}

fn scalar_over(src: &[f32], dst: &[f32], stride: usize) -> Vec<f32> {
    let alpha_index = stride - 1;
    let mut out = Vec::with_capacity(dst.len());
    for (src_px, dst_px) in src.chunks_exact(stride).zip(dst.chunks_exact(stride)) {
        let inv_alpha_s = 1.0 - src_px[alpha_index];
        for (&c_s, &c_d) in src_px.iter().zip(dst_px.iter()) {
            out.push(c_d.mul_add(inv_alpha_s, c_s));
        }
    }
    out
}

fn scalar_blend(
    src: &[f32],
    dst: &[f32],
    mask: &[f32],
    stride: usize,
    mode: BlendMode,
    opacity: f32,
) -> Vec<f32> {
    let alpha_index = stride - 1;
    let mut out = Vec::with_capacity(dst.len());
    for ((src_px, dst_px), &coverage) in src
        .chunks_exact(stride)
        .zip(dst.chunks_exact(stride))
        .zip(mask.iter())
    {
        let k = opacity * coverage;
        let inv_alpha_s = 1.0 - src_px[alpha_index];
        for (&s, &d) in src_px.iter().zip(dst_px.iter()) {
            let sample = if k.to_bits() == 0.0_f32.to_bits() {
                d
            } else {
                let blended = mode.blend_channel(s, d, inv_alpha_s);
                k.mul_add(blended - d, d)
            };
            out.push(sample);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    // Exact-tier kernels are compared bit-for-bit against the scalar baseline; the
    // float equality here is intentional.
    #![allow(clippy::float_cmp)]
    use super::Adjustment;
    use super::{
        BlendMode, KernelKind, MIN_SPEEDUP, Transfer, benchmark_all, kernels, scalar_adjust,
        scalar_blend, scalar_convert, scalar_over, scalar_premultiply, scalar_unpremultiply,
    };
    use paintop_ir::DeterminismTier;

    /// The optimized kernels must compute the same result the scalar baselines do
    /// (exact tiers bit-for-bit), so the benchmark ratio is honest and the
    /// production differential (bn-2ja) has the same equivalence guarantee.
    #[test]
    fn optimized_matches_scalar_baseline() {
        let stride = 4;
        let n = 64 * stride;
        let buf = super::fill(n, 0xABCD);

        // Exact kernels: bit-for-bit.
        assert_eq!(
            kernels::premultiply(&buf, stride),
            scalar_premultiply(&buf, stride)
        );
        let src = super::fill(n, 0x1);
        let dst = super::fill(n, 0x2);
        assert_eq!(
            kernels::composite_over(&src, &dst, stride),
            scalar_over(&src, &dst, stride)
        );
        let mask = super::fill(64, 0x3);
        assert_eq!(
            kernels::composite_blend(&src, &dst, &mask, stride, BlendMode::Screen, 0.75),
            scalar_blend(&src, &dst, &mask, stride, BlendMode::Screen, 0.75)
        );

        // Bounded kernels: within a tight envelope.
        let opt = kernels::unpremultiply(&buf, stride);
        let scl = scalar_unpremultiply(&buf, stride);
        for (a, b) in opt.iter().zip(scl.iter()) {
            assert!((a - b).abs() <= 1e-6, "unpremultiply {a} vs {b}");
        }
        let opt = kernels::color_convert(&buf, stride, true, Transfer::Decode);
        let scl = scalar_convert(&buf, stride, true, Transfer::Decode);
        for (a, b) in opt.iter().zip(scl.iter()) {
            assert!((a - b).abs() <= 1e-6, "convert {a} vs {b}");
        }
        let adj = Adjustment {
            exposure_ev: 0.5,
            saturation: 0.2,
            temperature: 0.1,
        };
        let opt = kernels::color_adjust(&buf, stride, 3, adj, None);
        let scl = scalar_adjust(&buf, stride, 3, adj, None);
        for (a, b) in opt.iter().zip(scl.iter()) {
            assert!((a - b).abs() <= 1e-5, "adjust {a} vs {b}");
        }
    }

    #[test]
    fn precision_tags_match_op_tiers() {
        // The exact ops are pinned exact; the transcendental ops are bounded.
        assert_eq!(
            KernelKind::AlphaPremultiply.precision(),
            DeterminismTier::Exact
        );
        assert_eq!(
            KernelKind::CompositeOver.precision(),
            DeterminismTier::Exact
        );
        assert_eq!(
            KernelKind::CompositeBlend.precision(),
            DeterminismTier::Exact
        );
        assert_eq!(
            KernelKind::ColorConvert.precision(),
            DeterminismTier::Bounded
        );
        assert_eq!(
            KernelKind::ColorAdjust.precision(),
            DeterminismTier::Bounded
        );
        assert_eq!(
            KernelKind::AlphaUnpremultiply.precision(),
            DeterminismTier::Bounded
        );
    }

    #[test]
    fn artifact_records_every_kernel_with_a_decision() {
        // A real (if small) working set so the timings are not noise-dominated.
        let artifact = benchmark_all(4_096);
        assert_eq!(artifact.selections.len(), KernelKind::ALL.len());
        assert!((artifact.min_speedup - MIN_SPEEDUP).abs() < 1e-12);
        // Every kernel has an explicit accept/reject decision and a serialisable
        // row; the union of selected+rejected is the whole set.
        let total = artifact.selected().len() + artifact.rejected().len();
        assert_eq!(total, KernelKind::ALL.len());
        // The artifact serialises (the perf evidence shape bn-2ja extends).
        let json = artifact.to_json().expect("artifact json");
        assert!(json.contains("min_speedup"));
        assert!(json.contains("speedup"));
        assert!(json.contains("precision"));
    }

    #[test]
    fn selection_gate_rejects_a_non_win() {
        // A measurement whose optimized path is no faster is rejected by the gate.
        let m = super::Measurement {
            samples: 1,
            scalar_ns: 100.0,
            optimized_ns: 100.0,
            speedup: 1.0,
        };
        assert!(!m.is_win(), "a tie does not clear the bar");
        let m2 = super::Measurement {
            speedup: MIN_SPEEDUP,
            ..m
        };
        assert!(m2.is_win(), "exactly the bar clears");
    }
}
