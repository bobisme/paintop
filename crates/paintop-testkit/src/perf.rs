//! Performance-baseline artifacts and machine-tolerant regression checking
//! (`bn-7k0`; `plan.md` §19 M3 exit criterion 3).
//!
//! M3 adds *faster* backends — an optimized CPU path and a `wgpu` path — and the
//! milestone's third exit criterion is that performance baselines are **checked
//! into CI artifacts**. This module is the artifact shape and the comparison
//! logic; the [`xtask`](../../../xtask) `perf-baseline` command drives the actual
//! measurement (it owns the `paintop-cpu` / `paintop-wgpu` kernels) and emits a
//! [`PerfBaseline`] through these types.
//!
//! # Why machine-tolerant
//!
//! Absolute wall-clock is meaningless across CI runners (a shared cloud VM is
//! 3–10× slower and far noisier than a dev box), so this harness does **not**
//! hard-fail on an absolute nanosecond or throughput bound. It records the
//! numbers and a *relative* regression threshold: a row is flagged only when its
//! throughput drops below `baseline × (1 − threshold)` against a checked-in
//! reference captured on the **same** machine class. The CI job compares against
//! its own previously-recorded baseline (an artifact it carries forward), so the
//! comparison is always like-for-like; a fresh machine with no baseline simply
//! records one and reports `NoBaseline` rather than failing.
//!
//! # The row identity
//!
//! Each [`PerfRow`] is keyed by `(op, backend, size_px)` — the same axes the bone
//! calls for (op, backend, size, throughput). The throughput is megapixels per
//! second (`Mpx/s`), a size-normalized rate so rows at different working-set
//! sizes are directly comparable and a regression shows up as a rate drop, not a
//! raw-time change that merely tracks the buffer size.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

/// One measured `(op, backend, size)` performance row.
///
/// The key triple `(op, backend, size_px)` uniquely identifies the row within a
/// baseline; [`throughput_mpps`](Self::throughput_mpps) is the size-normalized
/// rate the regression check compares.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PerfRow {
    /// The canonical op id this row measures, e.g. `composite.over@1`.
    pub op: String,
    /// The backend that produced the measurement, e.g. `cpu.optimized`,
    /// `cpu.reference`, `wgpu.pointwise`.
    pub backend: String,
    /// The working-set size in pixels (the image area the kernel processed).
    pub size_px: u64,
    /// Median nanoseconds per call over the timed sweep (host- and load-
    /// dependent; recorded for provenance, not asserted absolutely).
    pub ns_per_call: f64,
    /// Throughput in **megapixels per second** — the size-normalized rate the
    /// regression check compares. Higher is faster.
    pub throughput_mpps: f64,
}

impl PerfRow {
    /// Build a row, computing the megapixels-per-second throughput from the
    /// per-call time and the pixel count.
    ///
    /// A zero or non-finite time yields a `0.0` throughput rather than an
    /// infinity, so a degenerate measurement can never masquerade as the fastest.
    #[must_use]
    pub fn new(
        op: impl Into<String>,
        backend: impl Into<String>,
        size_px: u64,
        ns_per_call: f64,
    ) -> Self {
        let throughput_mpps = if ns_per_call > 0.0 && ns_per_call.is_finite() {
            // pixels / nanosecond == megapixels / millisecond == … work it out:
            // (size_px pixels / ns_per_call ns) * (1e9 ns / s) / (1e6 px / Mpx)
            #[expect(
                clippy::cast_precision_loss,
                reason = "pixel counts for benchmarked images are far below 2^52"
            )]
            let px = size_px as f64;
            px / ns_per_call * 1.0e9 / 1.0e6
        } else {
            0.0
        };
        Self {
            op: op.into(),
            backend: backend.into(),
            size_px,
            ns_per_call,
            throughput_mpps,
        }
    }

    /// The `(op, backend, size_px)` identity used to match a current row against
    /// its baseline.
    #[must_use]
    pub fn key(&self) -> RowKey {
        RowKey {
            op: self.op.clone(),
            backend: self.backend.clone(),
            size_px: self.size_px,
        }
    }
}

/// The `(op, backend, size)` identity that matches a current row to its baseline.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RowKey {
    /// The op id.
    pub op: String,
    /// The backend id.
    pub backend: String,
    /// The working-set size in pixels.
    pub size_px: u64,
}

impl std::fmt::Display for RowKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}/{}@{}px", self.op, self.backend, self.size_px)
    }
}

/// A full performance baseline: every measured row plus the provenance that makes
/// a comparison like-for-like.
///
/// The `machine` / `profile` provenance fields are recorded so a comparison only
/// ever runs against a baseline captured under the same conditions (the CI job
/// keys its stored baseline on the runner identity). They are not asserted by
/// this crate — they are evidence for the gate and the human reader.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PerfBaseline {
    /// Artifact schema version, so a format change is detectable rather than
    /// silently mis-parsed.
    pub schema: u32,
    /// A free-form machine/runner identity (e.g. the CI runner label or a host
    /// tag) so a baseline is only compared against the same machine class.
    pub machine: String,
    /// The build profile the sweep ran under (`debug` / `release`). Throughput is
    /// only meaningful in `release`, so the profile is recorded and the gate can
    /// require it.
    pub profile: String,
    /// The measured rows, sorted by `(op, backend, size_px)` for a stable,
    /// diffable artifact.
    pub rows: Vec<PerfRow>,
}

/// The current artifact schema version.
pub const PERF_SCHEMA: u32 = 1;

impl PerfBaseline {
    /// Build a baseline from measured rows, sorting them into the canonical
    /// `(op, backend, size_px)` order so the serialized artifact is stable and
    /// diff-friendly across runs.
    #[must_use]
    pub fn new(machine: impl Into<String>, profile: impl Into<String>, rows: Vec<PerfRow>) -> Self {
        let mut rows = rows;
        rows.sort_by(|a, b| {
            (a.op.as_str(), a.backend.as_str(), a.size_px).cmp(&(
                b.op.as_str(),
                b.backend.as_str(),
                b.size_px,
            ))
        });
        Self {
            schema: PERF_SCHEMA,
            machine: machine.into(),
            profile: profile.into(),
            rows,
        }
    }

    /// Serialize the baseline to pretty JSON (the on-disk artifact form).
    ///
    /// # Errors
    /// Propagates a [`serde_json`] serialization error.
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// Parse a baseline from its JSON artifact form.
    ///
    /// # Errors
    /// Propagates a [`serde_json`] deserialization error (including the
    /// `deny_unknown_fields` rejection of a drifted schema).
    pub fn from_json(s: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(s)
    }

    /// Index the rows by their `(op, backend, size)` key for lookup.
    #[must_use]
    fn by_key(&self) -> BTreeMap<RowKey, &PerfRow> {
        self.rows.iter().map(|r| (r.key(), r)).collect()
    }
}

/// The verdict for a single row when checked against a baseline.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum RowVerdict {
    /// The row's throughput held within the allowed slack (or improved).
    Ok,
    /// The row regressed: throughput dropped below `baseline × (1 − threshold)`.
    Regressed,
    /// No matching baseline row exists (first run, or a newly-added kernel). This
    /// is **not** a failure — it records a new baseline point.
    NoBaseline,
}

/// One row's regression-check result.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RowCheck {
    /// The op id.
    pub op: String,
    /// The backend id.
    pub backend: String,
    /// The working-set size in pixels.
    pub size_px: u64,
    /// The current run's throughput (Mpx/s).
    pub current_mpps: f64,
    /// The baseline throughput (Mpx/s), absent when there is no baseline row.
    pub baseline_mpps: Option<f64>,
    /// The ratio `current / baseline` (`1.0` == on par, `< 1.0` == slower),
    /// absent when there is no baseline.
    pub ratio: Option<f64>,
    /// The verdict.
    pub verdict: RowVerdict,
}

/// The full regression report: a per-row verdict plus the slack threshold and an
/// overall pass/fail.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RegressionReport {
    /// The fractional slack allowed before a row is flagged (e.g. `0.20` allows a
    /// 20% throughput drop before failing).
    pub threshold: f64,
    /// The per-row checks, in baseline order.
    pub checks: Vec<RowCheck>,
}

impl RegressionReport {
    /// The rows that regressed beyond the threshold.
    #[must_use]
    pub fn regressions(&self) -> Vec<&RowCheck> {
        self.checks
            .iter()
            .filter(|c| c.verdict == RowVerdict::Regressed)
            .collect()
    }

    /// Whether the report is clean (no regression beyond the threshold). Missing
    /// baselines do **not** fail the report — they only record new points.
    #[must_use]
    pub fn is_clean(&self) -> bool {
        self.regressions().is_empty()
    }

    /// Serialize to pretty JSON.
    ///
    /// # Errors
    /// Propagates a [`serde_json`] serialization error.
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }
}

/// Compare a current baseline against a reference baseline at a relative slack
/// `threshold`, returning a per-row verdict.
///
/// A row is `Regressed` when its throughput drops below
/// `baseline × (1 − threshold)`; an improvement or a within-slack drop is `Ok`; a
/// current row with no matching baseline is `NoBaseline` (recorded, not failed).
/// The comparison is purely relative, so it is **machine-tolerant** as long as
/// the two baselines were captured on the same machine class (the caller's
/// responsibility — the CI job keys its stored baseline on the runner identity).
///
/// `threshold` is clamped to `[0, 1)`; a value `>= 1` would allow throughput to
/// fall to zero, which is never intended.
#[must_use]
pub fn check_regressions(
    current: &PerfBaseline,
    baseline: &PerfBaseline,
    threshold: f64,
) -> RegressionReport {
    let threshold = threshold.clamp(0.0, 0.999);
    let base = baseline.by_key();
    let checks = current
        .rows
        .iter()
        .map(|row| {
            let key = row.key();
            match base.get(&key) {
                Some(b) if b.throughput_mpps > 0.0 => {
                    let ratio = row.throughput_mpps / b.throughput_mpps;
                    let verdict = if ratio < (1.0 - threshold) {
                        RowVerdict::Regressed
                    } else {
                        RowVerdict::Ok
                    };
                    RowCheck {
                        op: row.op.clone(),
                        backend: row.backend.clone(),
                        size_px: row.size_px,
                        current_mpps: row.throughput_mpps,
                        baseline_mpps: Some(b.throughput_mpps),
                        ratio: Some(ratio),
                        verdict,
                    }
                }
                _ => RowCheck {
                    op: row.op.clone(),
                    backend: row.backend.clone(),
                    size_px: row.size_px,
                    current_mpps: row.throughput_mpps,
                    baseline_mpps: None,
                    ratio: None,
                    verdict: RowVerdict::NoBaseline,
                },
            }
        })
        .collect();
    RegressionReport { threshold, checks }
}

#[cfg(test)]
mod tests {
    // The zero-throughput assertions below compare against an exact `0.0` the code
    // sets literally (a degenerate-time sentinel), so bit-equality is intended.
    #![allow(clippy::float_cmp)]
    use super::{PERF_SCHEMA, PerfBaseline, PerfRow, RowVerdict, check_regressions};

    fn row(op: &str, backend: &str, size: u64, ns: f64) -> PerfRow {
        PerfRow::new(op, backend, size, ns)
    }

    #[test]
    fn throughput_is_size_normalized_megapixels_per_second() {
        // 1_000_000 px in 1_000_000 ns == 1 px/ns == 1000 Mpx/s.
        let r = PerfRow::new("composite.over@1", "cpu.optimized", 1_000_000, 1_000_000.0);
        assert!((r.throughput_mpps - 1000.0).abs() < 1e-6, "{r:?}");
    }

    #[test]
    fn degenerate_time_yields_zero_throughput_not_infinity() {
        let r = PerfRow::new("x@1", "cpu.optimized", 1024, 0.0);
        assert_eq!(r.throughput_mpps, 0.0);
        let r = PerfRow::new("x@1", "cpu.optimized", 1024, f64::NAN);
        assert_eq!(r.throughput_mpps, 0.0);
    }

    #[test]
    fn baseline_sorts_rows_for_a_stable_artifact() {
        let rows = vec![
            row("b@1", "cpu.optimized", 16, 10.0),
            row("a@1", "wgpu.pointwise", 16, 10.0),
            row("a@1", "cpu.optimized", 32, 10.0),
            row("a@1", "cpu.optimized", 16, 10.0),
        ];
        let base = PerfBaseline::new("test-host", "release", rows);
        let keys: Vec<_> = base
            .rows
            .iter()
            .map(|r| (r.op.as_str(), r.backend.as_str(), r.size_px))
            .collect();
        assert_eq!(
            keys,
            vec![
                ("a@1", "cpu.optimized", 16),
                ("a@1", "cpu.optimized", 32),
                ("a@1", "wgpu.pointwise", 16),
                ("b@1", "cpu.optimized", 16),
            ]
        );
        assert_eq!(base.schema, PERF_SCHEMA);
    }

    #[test]
    fn json_round_trips() {
        let base = PerfBaseline::new(
            "host",
            "release",
            vec![row("composite.over@1", "cpu.optimized", 1024, 100.0)],
        );
        let json = base.to_json().expect("serialize");
        let back = PerfBaseline::from_json(&json).expect("deserialize");
        assert_eq!(base, back);
    }

    #[test]
    fn unknown_field_in_baseline_is_rejected() {
        let json = r#"{"schema":1,"machine":"h","profile":"release","rows":[],"bogus":true}"#;
        assert!(PerfBaseline::from_json(json).is_err());
    }

    #[test]
    fn a_within_slack_drop_is_ok_a_larger_drop_regresses() {
        // Baseline: 1000 Mpx/s. Current at 850 (15% drop) with a 20% threshold is OK;
        // current at 750 (25% drop) regresses.
        let baseline = PerfBaseline::new(
            "h",
            "release",
            vec![row("op@1", "cpu.optimized", 1_000_000, 1_000_000.0)], // 1000 Mpx/s
        );
        // ns for 850 Mpx/s over 1e6 px: t = px/ (mpps*1e6/1e9) = 1e6 / (850*1e3) ns/px... recompute:
        // throughput = px/ns*1000 => ns = px*1000/throughput = 1e6*1000/850.
        let cur_ok = PerfBaseline::new(
            "h",
            "release",
            vec![row(
                "op@1",
                "cpu.optimized",
                1_000_000,
                1_000_000.0 * 1000.0 / 850.0,
            )],
        );
        let report = check_regressions(&cur_ok, &baseline, 0.20);
        assert!(report.is_clean(), "{:?}", report.checks);
        assert_eq!(report.checks[0].verdict, RowVerdict::Ok);

        let cur_bad = PerfBaseline::new(
            "h",
            "release",
            vec![row(
                "op@1",
                "cpu.optimized",
                1_000_000,
                1_000_000.0 * 1000.0 / 750.0,
            )],
        );
        let report = check_regressions(&cur_bad, &baseline, 0.20);
        assert!(!report.is_clean());
        assert_eq!(report.regressions().len(), 1);
        assert_eq!(report.regressions()[0].verdict, RowVerdict::Regressed);
    }

    #[test]
    fn an_improvement_is_ok() {
        let baseline = PerfBaseline::new(
            "h",
            "release",
            vec![row("op@1", "cpu.optimized", 1_000_000, 2_000_000.0)],
        );
        let faster = PerfBaseline::new(
            "h",
            "release",
            vec![row("op@1", "cpu.optimized", 1_000_000, 1_000_000.0)],
        );
        let report = check_regressions(&faster, &baseline, 0.20);
        assert!(report.is_clean());
        assert_eq!(report.checks[0].verdict, RowVerdict::Ok);
        let ratio = report.checks[0].ratio.expect("ratio");
        assert!(ratio > 1.5, "improvement ratio {ratio}");
    }

    #[test]
    fn a_missing_baseline_row_records_not_fails() {
        let baseline = PerfBaseline::new("h", "release", vec![]);
        let current = PerfBaseline::new(
            "h",
            "release",
            vec![row("new.op@1", "wgpu.pointwise", 4096, 500.0)],
        );
        let report = check_regressions(&current, &baseline, 0.20);
        // A new row is NOT a regression — the report stays clean.
        assert!(report.is_clean());
        assert_eq!(report.checks[0].verdict, RowVerdict::NoBaseline);
        assert!(report.checks[0].baseline_mpps.is_none());
    }

    #[test]
    fn threshold_is_clamped_so_zero_throughput_still_regresses() {
        let baseline = PerfBaseline::new(
            "h",
            "release",
            vec![row("op@1", "cpu.optimized", 1_000, 1.0)],
        );
        // A near-dead current row (huge ns -> tiny throughput) regresses even at an
        // absurd threshold, because the clamp keeps the bar above zero.
        let current = PerfBaseline::new(
            "h",
            "release",
            vec![row("op@1", "cpu.optimized", 1_000, 1.0e15)],
        );
        let report = check_regressions(&current, &baseline, 5.0);
        assert!(!report.is_clean());
    }

    #[test]
    fn report_json_round_trips() {
        let baseline = PerfBaseline::new(
            "h",
            "release",
            vec![row("op@1", "cpu.optimized", 1024, 100.0)],
        );
        let current = baseline.clone();
        let report = check_regressions(&current, &baseline, 0.20);
        let json = report.to_json().expect("serialize report");
        assert!(json.contains("threshold"));
        assert!(json.contains("verdict"));
    }
}
