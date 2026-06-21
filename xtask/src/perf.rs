//! The performance-baseline harness driver (`bn-7k0`; `plan.md` §19 M3 exit
//! criterion 3).
//!
//! This is the *measurement* half of the perf baseline: it owns the
//! `paintop-cpu` / `paintop-wgpu` kernels (which the artifact crate
//! [`paintop_testkit::perf`] cannot depend on), runs them over a fixed sweep of
//! working-set sizes, and emits a [`PerfBaseline`] artifact spanning
//! `(op, backend, size, throughput)`. It then optionally compares against a
//! checked-in reference baseline at a configurable relative threshold and flags
//! regressions — **without** hard-failing on absolute wall-clock (the comparison
//! is purely relative and machine-tolerant; see the artifact crate's docs).
//!
//! # Backends measured
//!
//! * `cpu.reference` and `cpu.optimized` — the pointwise kernel sweep, timed
//!   through [`paintop_cpu::optimized::bench::measure`] (its scalar baseline is
//!   the reference path, its optimized form the optimized path), so a single
//!   measurement yields both backends' rows for free and the speedup is implicit
//!   in the throughput ratio.
//! * `wgpu.separable` — the two-pass GPU Gaussian, timed end-to-end (upload +
//!   dispatch + readback) **only when a GPU adapter is present**; with no adapter
//!   the GPU rows are skipped cleanly (the artifact still records the CPU rows),
//!   exactly mirroring the differential harness's GPU gating.

use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result, bail};
use paintop_cpu::optimized::bench::{KernelKind, measure};
use paintop_ir::Extent;
use paintop_testkit::perf::{PerfBaseline, PerfRow, RegressionReport, check_regressions};

/// The working-set sizes (in pixels per side, square) the sweep measures. A small
/// cache-resident size and a near-image size, so a regression in either the hot
/// inner loop or the memory-bound regime is visible.
const CPU_SIZES_PX: &[u32] = &[256, 1024];

/// The GPU separable sweep sizes (square edge length). The GPU path has a fixed
/// per-dispatch cost, so a too-small image is launch-bound noise; these are large
/// enough to measure steady-state throughput.
const GPU_SIZES_PX: &[u32] = &[512, 2048];

/// The σ (in pixels) the GPU separable Gaussian is benchmarked at — a mid-size
/// kernel so the two passes do real work.
const GPU_SIGMA: f64 = 4.0;

/// Timed repeats per GPU size; the median rejects scheduler/queue jitter.
const GPU_REPEATS: usize = 5;

/// Options for the `perf-baseline` driver.
#[derive(Debug, Clone)]
pub struct PerfOptions {
    /// Where to write the emitted baseline artifact JSON.
    pub out: PathBuf,
    /// An optional reference baseline to compare against; when present, a
    /// regression beyond `threshold` causes a non-zero exit (the CI gate).
    pub baseline: Option<PathBuf>,
    /// The fractional throughput-drop slack before a row is flagged a regression
    /// (e.g. `0.25` allows a 25% drop on noisy CI hardware).
    pub threshold: f64,
    /// A machine/runner identity recorded in the artifact so a comparison only
    /// runs against the same machine class.
    pub machine: String,
    /// Skip the GPU sweep even when an adapter is present (for a CPU-only
    /// artifact or to keep a run hermetic).
    pub no_gpu: bool,
}

/// Run the perf-baseline sweep, write the artifact, and (when a reference
/// baseline is supplied) check for regressions.
///
/// # Errors
/// Fails if the artifact cannot be written, the reference baseline cannot be read
/// or parsed, or a regression beyond the threshold is detected.
pub fn run(opts: &PerfOptions) -> Result<()> {
    let profile = if cfg!(debug_assertions) {
        "debug"
    } else {
        "release"
    };
    if profile == "debug" {
        eprintln!(
            "perf-baseline: WARNING running under a debug build; throughput is not \
             representative. Re-run with `--release` for a real baseline."
        );
    }

    let mut rows = measure_cpu_rows();
    if opts.no_gpu {
        eprintln!("perf-baseline: GPU sweep disabled (--no-gpu)");
    } else {
        match measure_gpu_rows() {
            Some(mut gpu) => rows.append(&mut gpu),
            None => eprintln!(
                "perf-baseline: no GPU adapter present; GPU rows skipped (CPU rows recorded)"
            ),
        }
    }

    let baseline = PerfBaseline::new(opts.machine.clone(), profile, rows);
    let json = baseline
        .to_json()
        .context("serialize perf baseline artifact")?;
    if let Some(parent) = opts.out.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("create artifact dir {}", parent.display()))?;
    }
    std::fs::write(&opts.out, format!("{json}\n"))
        .with_context(|| format!("write perf baseline to {}", opts.out.display()))?;
    eprintln!(
        "perf-baseline: wrote {} rows to {}",
        baseline.rows.len(),
        opts.out.display()
    );
    for row in &baseline.rows {
        eprintln!(
            "  {:<24} {:<16} {:>8}px  {:>10.1} Mpx/s",
            row.op, row.backend, row.size_px, row.throughput_mpps
        );
    }

    if let Some(ref_path) = &opts.baseline {
        let report = compare(&baseline, ref_path, opts.threshold)?;
        report_and_gate(&report)?;
    }

    Ok(())
}

/// Compare a freshly-measured baseline against a checked-in reference at the
/// threshold, returning the regression report.
fn compare(current: &PerfBaseline, ref_path: &Path, threshold: f64) -> Result<RegressionReport> {
    let bytes = std::fs::read_to_string(ref_path)
        .with_context(|| format!("read reference baseline {}", ref_path.display()))?;
    let reference = PerfBaseline::from_json(&bytes)
        .with_context(|| format!("parse reference baseline {}", ref_path.display()))?;
    if reference.machine != current.machine {
        eprintln!(
            "perf-baseline: NOTE reference machine {:?} != current {:?}; relative \
             comparison may be noisy across machine classes",
            reference.machine, current.machine
        );
    }
    Ok(check_regressions(current, &reference, threshold))
}

/// Log the regression report and fail the process if any row regressed.
fn report_and_gate(report: &RegressionReport) -> Result<()> {
    eprintln!(
        "perf-baseline: regression check at threshold {:.0}% slack",
        report.threshold * 100.0
    );
    for check in &report.checks {
        let ratio = check
            .ratio
            .map_or_else(|| "n/a".to_string(), |r| format!("{r:.2}x"));
        eprintln!(
            "  {:<24} {:<16} {:>8}px  {:>6} [{:?}]",
            check.op, check.backend, check.size_px, ratio, check.verdict
        );
    }
    let regressions = report.regressions();
    if regressions.is_empty() {
        eprintln!("perf-baseline: OK no regression beyond threshold");
        Ok(())
    } else {
        for r in &regressions {
            eprintln!(
                "  REGRESSION {} / {} @ {}px: {:.2}x of baseline",
                r.op,
                r.backend,
                r.size_px,
                r.ratio.unwrap_or(0.0)
            );
        }
        bail!(
            "perf-baseline: {} row(s) regressed beyond the threshold",
            regressions.len()
        )
    }
}

/// Measure every pointwise kernel at every CPU sweep size, emitting a
/// `cpu.reference` row (the scalar baseline) and a `cpu.optimized` row (the
/// vectorized form) per `(op, size)`.
fn measure_cpu_rows() -> Vec<PerfRow> {
    measure_cpu_rows_for(CPU_SIZES_PX)
}

/// The size-parameterized core of [`measure_cpu_rows`], so a unit test can sweep
/// a tiny working set (the real driver uses the full [`CPU_SIZES_PX`], which is
/// far too slow for the debug `just test` gate).
fn measure_cpu_rows_for(sizes: &[u32]) -> Vec<PerfRow> {
    let mut rows = Vec::new();
    for &edge in sizes {
        let pixels_usize = (edge as usize) * (edge as usize);
        let pixels = pixels_usize as u64;
        for kernel in KernelKind::ALL {
            let m = measure(kernel, pixels_usize);
            let op = kernel.op_id();
            rows.push(PerfRow::new(op, "cpu.reference", pixels, m.scalar_ns));
            rows.push(PerfRow::new(op, "cpu.optimized", pixels, m.optimized_ns));
        }
    }
    rows
}

/// Measure the `wgpu.separable` Gaussian end-to-end at every GPU sweep size, or
/// `None` when no adapter is present.
fn measure_gpu_rows() -> Option<Vec<PerfRow>> {
    use paintop_wgpu::{Boundary, PipelineCache, probe, run_separable_gaussian};

    let context = probe().ok()?;
    let mut cache: PipelineCache<wgpu::ComputePipeline> = PipelineCache::new();
    let taps = gaussian_taps(GPU_SIGMA);
    let channels = 4_u32;

    let mut rows = Vec::new();
    for &edge in GPU_SIZES_PX {
        let extent = Extent::new(edge, edge);
        let n = (edge as usize) * (edge as usize) * (channels as usize);
        let samples = deterministic_fill(n, 0xBEEF ^ u64::from(edge));

        // Warmup (prime the pipeline cache + device queue), untimed.
        let warm = run_separable_gaussian(
            &context,
            &mut cache,
            &samples,
            extent,
            channels,
            &taps,
            Boundary::Clamp,
        );
        if warm.is_err() {
            // A dispatch error on this size — record nothing for it rather than
            // crash; the CPU rows still stand.
            continue;
        }

        let mut times = Vec::with_capacity(GPU_REPEATS);
        let mut ok = true;
        for _ in 0..GPU_REPEATS {
            let start = Instant::now();
            let out = run_separable_gaussian(
                &context,
                &mut cache,
                &samples,
                extent,
                channels,
                &taps,
                Boundary::Clamp,
            );
            if out.is_err() {
                ok = false;
                break;
            }
            times.push(start.elapsed().as_nanos());
        }
        if !ok || times.is_empty() {
            continue;
        }
        times.sort_unstable();
        #[expect(
            clippy::cast_precision_loss,
            reason = "median nanos for a single GPU dispatch are far below 2^52"
        )]
        let median_ns = times[times.len() / 2] as f64;
        let pixels = u64::from(edge) * u64::from(edge);
        rows.push(PerfRow::new(
            "filter.gaussian_blur@1",
            "wgpu.separable",
            pixels,
            median_ns,
        ));
    }
    Some(rows)
}

/// Build a normalized odd-length 1-D Gaussian tap array for `sigma` (the same
/// shape the GPU separable backend expects).
fn gaussian_taps(sigma: f64) -> Vec<f32> {
    if sigma <= 0.0 {
        return vec![1.0];
    }
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "radius is a small positive integer from a bounded sigma"
    )]
    let radius = (sigma * 3.0).ceil() as usize;
    let len = 2 * radius + 1;
    let mut taps = Vec::with_capacity(len);
    let two_sigma_sq = 2.0 * sigma * sigma;
    let mut sum = 0.0_f64;
    for i in 0..len {
        #[expect(
            clippy::cast_possible_wrap,
            clippy::cast_precision_loss,
            reason = "len and radius are small bounded indices"
        )]
        let x = (i as isize - radius as isize) as f64;
        let w = (-(x * x) / two_sigma_sq).exp();
        sum += w;
        taps.push(w);
    }
    taps.into_iter()
        .map(|w| {
            #[expect(
                clippy::cast_possible_truncation,
                reason = "normalized Gaussian weights are well within f32 range"
            )]
            let v = (w / sum) as f32;
            v
        })
        .collect()
}

/// A deterministic pseudo-random `[0,1)` sample buffer (xorshift64*), so the GPU
/// sweep exercises non-degenerate data without a `rand` dependency or any
/// run-to-run variation.
fn deterministic_fill(n: usize, seed: u64) -> Vec<f32> {
    let mut state = seed | 1;
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        let r = state.wrapping_mul(0x2545_f491_4f6c_dd1d);
        let mantissa = u16::try_from((r >> 40) & 0xFFFF).unwrap_or(0);
        out.push(f32::from(mantissa) / f32::from(u16::MAX));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::{deterministic_fill, gaussian_taps, measure_cpu_rows_for};

    #[test]
    fn cpu_rows_cover_every_kernel_on_both_backends() {
        // A tiny 16px sweep so the debug `just test` gate stays fast; the full
        // CPU_SIZES_PX sweep is only run by the release driver.
        let sizes = [16_u32];
        let rows = measure_cpu_rows_for(&sizes);
        // 6 kernels * 1 size * 2 backends.
        assert_eq!(rows.len(), 6 * sizes.len() * 2);
        let backends: std::collections::BTreeSet<_> =
            rows.iter().map(|r| r.backend.as_str()).collect();
        assert!(backends.contains("cpu.reference"));
        assert!(backends.contains("cpu.optimized"));
        // Every row has a finite, non-negative throughput.
        for r in &rows {
            assert!(
                r.throughput_mpps >= 0.0 && r.throughput_mpps.is_finite(),
                "{r:?}"
            );
        }
    }

    #[test]
    fn gaussian_taps_are_normalized_and_odd() {
        let taps = gaussian_taps(2.5);
        assert_eq!(taps.len() % 2, 1, "odd length");
        let sum: f32 = taps.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "taps sum to 1, got {sum}");
        // sigma <= 0 is the identity single tap.
        assert_eq!(gaussian_taps(0.0), vec![1.0]);
    }

    #[test]
    fn fill_is_deterministic() {
        assert_eq!(deterministic_fill(32, 7), deterministic_fill(32, 7));
        assert_ne!(deterministic_fill(32, 7), deterministic_fill(32, 8));
        for v in deterministic_fill(64, 1) {
            assert!((0.0..1.0).contains(&v));
        }
    }
}
