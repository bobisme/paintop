//! The two analysis-reduction operations (`OP_CATALOG` §12,
//! `AGENT_VERIFICATION` §2.4).
//!
//! `analyze.statistics@1` reduces a resource (and an optional mask) to a
//! per-channel statistics [`Report`]; `analyze.histogram@1` bins an image/field
//! over an explicit domain into a per-channel [`HistogramData`] report.
//!
//! Both ops are *pure* (they never mutate an input) and
//! [`Exact`](DeterminismTier::Exact): every reduction runs in a fixed order so
//! the report is a deterministic function of the inputs and parameters.
//!
//! # `analyze.statistics@1`
//!
//! Reduce a resource to its per-channel finite statistics — count, min, max,
//! sum, mean, and population variance — over the samples an optional coverage
//! `mask` admits. A pixel contributes when its mask coverage is strictly
//! positive (the whole image when no mask is supplied); non-finite samples are
//! excluded from the extrema/sum/sum-of-squares and counted separately, so one
//! `NaN` cannot poison a channel's range while still being flagged.
//!
//! The per-channel `sum` / `sum_sq` reductions use a **stable pairwise merge
//! tree** rather than a running scalar accumulation: the admitted values are
//! summed by recursively halving the index range and adding the two halves. The
//! tree is a fixed function of the value order, so the result is bit-identical
//! across runs, and the pairwise shape both bounds rounding error and is the
//! deterministic-reduction primitive M2's tiled reductions reuse.
//!
//! # `analyze.histogram@1`
//!
//! Bin an image/field's samples into `bins` equal-width bins over the explicit,
//! half-open-with-inclusive-top domain `[domain_min, domain_max]`. A finite
//! sample `v` in range lands in bin `floor((v - domain_min) / width)`, with the
//! top edge `domain_max` folded into the last bin; finite samples below / above
//! the domain and non-finite samples are counted separately (`below` / `above`
//! / `nonfinite`). The domain and bin count are required parameters — the op
//! never guesses a range — so the assignment is fully explicit and exact.

use paintop_core::executor::{InputValues, OpImplementation, OutputValues, ResourceValue};
use paintop_ir::{
    AssertionResult, ChannelStats, Descriptors, DeterminismTier, Error, ErrorClass, ErrorContext,
    Extent, HistogramData, ImplId, InputRegions, InputSpec, OpContract, OperationManifest,
    OutputDescriptors, OutputRegions, OutputSpec, ParamSpec, ParamType, Rect, Report,
    ReportDescriptor, ResourceDescriptor, ResourceKind, Result, RoiCategory, RoiPolicy,
    TestMetadata,
};

/// The canonical id of the statistics operation.
pub const STATISTICS_OP_ID: &str = "analyze.statistics@1";

/// The canonical id of the histogram operation.
pub const HISTOGRAM_OP_ID: &str = "analyze.histogram@1";

/// A required input port was absent or carried the wrong resource kind.
pub const E_STATS_INPUT: &str = "E_STATS_INPUT";

/// The optional `mask` input disagrees with the resource's extent.
pub const E_STATS_SHAPE: &str = "E_STATS_SHAPE";

/// A parameter was missing, the wrong type, or out of range.
pub const E_STATS_PARAM: &str = "E_STATS_PARAM";

/// The largest bin count a histogram may request, bounding the report size so a
/// pathological `bins` cannot exhaust memory.
const MAX_BINS: u64 = 1 << 20;

// ---------------------------------------------------------------------------
// Shared input helpers
// ---------------------------------------------------------------------------

/// Read a required input *descriptor* port, erroring if absent.
fn require_descriptor<'a>(
    inputs: &'a Descriptors,
    op: &str,
    port: &str,
) -> Result<&'a ResourceDescriptor> {
    inputs.get(port).ok_or_else(|| {
        Error::new(
            ErrorClass::Reference,
            E_STATS_INPUT,
            format!("{op} requires a `{port}` input"),
        )
    })
}

/// Read a required input *value* port, erroring if absent.
fn require_value<'a>(
    inputs: &'a InputValues,
    op: &str,
    port: &str,
) -> std::result::Result<&'a ResourceValue, Error> {
    inputs.get(port).ok_or_else(|| {
        Error::new(
            ErrorClass::Reference,
            E_STATS_INPUT,
            format!("{op} requires a `{port}` input value"),
        )
    })
}

/// The single `report` output descriptor map for a reduction op.
fn report_output(extent: Extent, channels: u32) -> OutputDescriptors {
    let mut out = OutputDescriptors::new();
    out.insert(
        "report".to_owned(),
        ResourceDescriptor::Report(ReportDescriptor { extent, channels }),
    );
    out
}

/// Wrap a report as the single `report` output of a reduction op.
fn single_report(report: Report) -> OutputValues {
    let mut out = OutputValues::new();
    out.insert("report".to_owned(), ResourceValue::report(report));
    out
}

/// Full-domain input regions for the named ports present in `inputs`.
fn full_domain_regions(inputs: &Descriptors, ports: &[&str]) -> InputRegions {
    let mut regions = InputRegions::new();
    for &port in ports {
        if let Some(input) = inputs.get(port) {
            let extent = input.extent();
            regions.insert(
                port.to_owned(),
                Rect::new(0, 0, i64::from(extent.width), i64::from(extent.height)),
            );
        }
    }
    regions
}

/// Postcondition: the op produced a `report` output.
fn produces_report(outputs: &OutputDescriptors) -> Vec<AssertionResult> {
    let produced = matches!(outputs.get("report"), Some(ResourceDescriptor::Report(_)));
    vec![if produced {
        AssertionResult::pass("produces_report")
    } else {
        AssertionResult::fail("produces_report", "no `report` output produced")
    }]
}

/// The mandatory `cpu.reference@1` oracle implementation id.
fn reference_impl() -> Result<ImplId> {
    ImplId::new("cpu", "reference", 1)
}

/// The verification declarations shared by the reduction ops. Each is an exact,
/// single-reference reduction, so differential and perceptual do not apply
/// (derived not-applicable); every other category is covered by the
/// analytic-fixture and property tests in this module.
fn reduction_test_metadata() -> TestMetadata {
    use paintop_ir::{CategoryStatus, VerificationCategory, VerificationDeclarations};
    let mut decls = VerificationDeclarations::new();
    for category in [
        VerificationCategory::BuildHygiene,
        VerificationCategory::SchemaContract,
        VerificationCategory::AnalyticFixtures,
        VerificationCategory::PropertyTests,
        VerificationCategory::Metamorphic,
        VerificationCategory::Goldens,
        VerificationCategory::Fuzzing,
        VerificationCategory::Performance,
    ] {
        decls = decls.with(category, CategoryStatus::Covered);
    }
    TestMetadata {
        has_analytic_reference: true,
        has_property_tests: true,
        golden_fixtures: vec![],
        not_applicable_reason: String::new(),
        verification: decls,
    }
}

// ---------------------------------------------------------------------------
// Stable pairwise reduction
// ---------------------------------------------------------------------------

/// The stable pairwise (`f64`) sum of a slice: recursively split the slice in
/// half and add the two halves.
///
/// The tree is a fixed function of the slice's order, so the result is
/// bit-identical across runs; the pairwise shape also bounds the accumulated
/// rounding error (O(log n) rather than O(n) for a running scalar sum). This is
/// the deterministic-reduction primitive `analyze.statistics` uses and M2's
/// tiled reductions reuse.
#[must_use]
fn pairwise_sum(values: &[f64]) -> f64 {
    // A small base case keeps the recursion shallow without changing the value
    // for the leaf blocks (a left-to-right sum of <= 8 terms).
    const BLOCK: usize = 8;
    if values.len() <= BLOCK {
        let mut acc = 0.0_f64;
        for &v in values {
            acc += v;
        }
        return acc;
    }
    let mid = values.len() / 2;
    pairwise_sum(&values[..mid]) + pairwise_sum(&values[mid..])
}

// ---------------------------------------------------------------------------
// analyze.statistics@1
// ---------------------------------------------------------------------------

/// Reduce `samples` (interleaved `channels`-wide, row-major) to per-channel
/// [`ChannelStats`], admitting only pixels whose `mask` coverage is strictly
/// positive. When `mask` is `None` every pixel is admitted.
///
/// The min/max are running extrema over the admitted finite samples; the `sum`
/// and `sum_sq` are stable pairwise reductions over the per-channel admitted
/// finite values, so the mean and variance are deterministic.
#[must_use]
fn masked_statistics(samples: &[f32], channels: u32, mask: Option<&[f32]>) -> Vec<ChannelStats> {
    let channel_count = channels as usize;
    if channel_count == 0 {
        return Vec::new();
    }

    // Per-channel finite values (for the pairwise sums) and the running extrema /
    // counts gathered in a single fixed-order pass.
    let mut values: Vec<Vec<f64>> = vec![Vec::new(); channel_count];
    let mut min = vec![None::<f32>; channel_count];
    let mut max = vec![None::<f32>; channel_count];
    let mut finite = vec![0_u64; channel_count];
    let mut nonfinite = vec![0_u64; channel_count];

    for (pixel_index, pixel) in samples.chunks_exact(channel_count).enumerate() {
        // A mask shorter than the pixel count treats the missing tail as fully
        // uncovered (excluded), so a degenerate mask never spuriously admits. A
        // pixel is admitted only when its coverage is strictly positive (a NaN
        // coverage compares false, so it is excluded).
        if let Some(mask) = mask {
            let coverage = mask.get(pixel_index).copied().unwrap_or(0.0);
            if coverage <= 0.0 || coverage.is_nan() {
                continue;
            }
        }
        for (channel, &sample) in pixel.iter().enumerate() {
            if sample.is_finite() {
                min[channel] = Some(min[channel].map_or(sample, |m| m.min(sample)));
                max[channel] = Some(max[channel].map_or(sample, |m| m.max(sample)));
                values[channel].push(f64::from(sample));
                finite[channel] += 1;
            } else {
                nonfinite[channel] += 1;
            }
        }
    }

    (0..channel_count)
        .map(|c| {
            let sum = pairwise_sum(&values[c]);
            let squares: Vec<f64> = values[c].iter().map(|&v| v * v).collect();
            let sum_sq = pairwise_sum(&squares);
            ChannelStats {
                min: min[c],
                max: max[c],
                sum,
                sum_sq,
                finite: finite[c],
                nonfinite: nonfinite[c],
            }
        })
        .collect()
}

/// The `analyze.statistics@1` operation.
#[derive(Debug, Clone, Copy, Default)]
pub struct Statistics;

impl Statistics {
    /// Construct the statistics operation.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    /// The declared manifest for `analyze.statistics@1`.
    ///
    /// # Errors
    /// Propagates the [`schema`](ErrorClass::Schema) error if the hard-coded op /
    /// impl ids are somehow invalid (they are not).
    pub fn manifest() -> Result<OperationManifest> {
        Ok(OperationManifest {
            id: STATISTICS_OP_ID.parse()?,
            impl_version: 1,
            summary: "Reduce a resource to its per-channel finite statistics (count, min, max, \
                      sum, mean, population variance) over an optional coverage mask, using a \
                      stable pairwise reduction tree so the sums are deterministic."
                .to_owned(),
            determinism: DeterminismTier::Exact,
            roi: RoiPolicy {
                // The report reduces over every admitted sample.
                category: RoiCategory::FullDomain,
                halo_px: None,
            },
            inputs: vec![
                InputSpec {
                    name: "resource".to_owned(),
                    kind: ResourceKind::Image,
                    required: true,
                    doc: "The resource whose per-channel statistics are computed.".to_owned(),
                },
                InputSpec {
                    name: "mask".to_owned(),
                    kind: ResourceKind::Mask,
                    required: false,
                    doc: "An optional coverage mask: only strictly-positive-coverage pixels \
                          contribute (the whole resource when absent)."
                        .to_owned(),
                },
            ],
            outputs: vec![OutputSpec {
                name: "report".to_owned(),
                kind: ResourceKind::Report,
                doc: "The statistics report: per-channel count, min, max, sum, mean, and variance."
                    .to_owned(),
            }],
            params: vec![],
            implementations: vec![reference_impl()?],
            test: reduction_test_metadata(),
        })
    }
}

impl OpContract for Statistics {
    fn declared_inputs(&self) -> Vec<(String, ResourceKind)> {
        vec![
            ("resource".to_owned(), ResourceKind::Image),
            ("mask".to_owned(), ResourceKind::Mask),
        ]
    }

    fn declared_outputs(&self) -> Vec<(String, ResourceKind)> {
        vec![("report".to_owned(), ResourceKind::Report)]
    }

    fn infer_outputs(
        &self,
        inputs: &Descriptors,
        _params: &serde_json::Value,
    ) -> Result<OutputDescriptors> {
        let op = STATISTICS_OP_ID;
        let resource = require_descriptor(inputs, op, "resource")?;
        let extent = resource.extent();
        // The report's channel count is structurally known only for an image; the
        // execution kernel fills it from the concrete value otherwise.
        let channels = match resource {
            ResourceDescriptor::Image(d) => d.layout.channel_count(),
            _ => 0,
        };
        // An optional mask, if present, must match the resource extent.
        if let Some(mask) = inputs.get("mask")
            && mask.extent() != extent
        {
            return Err(shape_mismatch(
                op,
                "the `mask` must share the resource's extent",
                format!("mask {:?} vs resource {:?}", mask.extent(), extent),
            ));
        }
        Ok(report_output(extent, channels))
    }

    fn required_inputs(
        &self,
        _requested_outputs: &OutputRegions,
        inputs: &Descriptors,
        _params: &serde_json::Value,
    ) -> Result<InputRegions> {
        Ok(full_domain_regions(inputs, &["resource", "mask"]))
    }

    fn validate_postconditions(
        &self,
        outputs: &OutputDescriptors,
        _params: &serde_json::Value,
    ) -> Result<Vec<AssertionResult>> {
        Ok(produces_report(outputs))
    }
}

impl OpImplementation for Statistics {
    fn compute(
        &self,
        inputs: &InputValues,
        _params: &serde_json::Value,
    ) -> std::result::Result<OutputValues, Error> {
        let op = STATISTICS_OP_ID;
        let resource = require_value(inputs, op, "resource")?;
        let extent = resource.extent();
        let channels = resource.channels();

        let mask_samples = match inputs.get("mask") {
            Some(mask) => {
                if !matches!(mask.descriptor(), ResourceDescriptor::Mask(_)) {
                    return Err(Error::new(
                        ErrorClass::Type,
                        E_STATS_INPUT,
                        format!("{op} `mask` input must be a mask resource"),
                    ));
                }
                if mask.extent() != extent {
                    return Err(shape_mismatch(
                        op,
                        "the `mask` must share the resource's extent",
                        format!("mask {:?} vs resource {:?}", mask.extent(), extent),
                    ));
                }
                Some(mask.samples())
            }
            None => None,
        };

        let channel_stats = masked_statistics(resource.samples(), channels, mask_samples);
        let all_finite = channel_stats.iter().all(ChannelStats::all_finite);
        let report = Report {
            extent,
            channels,
            channel_stats,
            all_finite,
            content_hash: String::new(),
            diff: None,
            assertion: None,
            histogram: None,
            components: None,
        };
        Ok(single_report(report))
    }
}

// ---------------------------------------------------------------------------
// analyze.histogram@1
// ---------------------------------------------------------------------------

/// The resolved histogram parameters: the bin count and the value domain.
#[derive(Debug, Clone, Copy)]
struct HistogramParams {
    bins: u32,
    domain_min: f64,
    domain_max: f64,
}

impl HistogramParams {
    /// Resolve and validate the `bins`, `domain_min`, and `domain_max`
    /// parameters: `bins` is a positive integer at most [`MAX_BINS`] and the
    /// domain is a finite, strictly-ordered interval.
    fn resolve(params: &serde_json::Value) -> Result<Self> {
        let bins = require_u64(params, "bins")?;
        if bins == 0 || bins > MAX_BINS {
            return Err(Error::new(
                ErrorClass::Schema,
                E_STATS_PARAM,
                format!("analyze.histogram `bins` must be in 1..={MAX_BINS}"),
            )
            .with_context(ErrorContext::default().with_actual(bins.to_string())));
        }
        let domain_min = require_finite(params, "domain_min")?;
        let domain_max = require_finite(params, "domain_max")?;
        // Both edges are finite (checked above), so the negation is exact.
        if domain_max <= domain_min {
            return Err(Error::new(
                ErrorClass::Schema,
                E_STATS_PARAM,
                "analyze.histogram requires `domain_max` > `domain_min`".to_owned(),
            )
            .with_context(
                ErrorContext::default().with_actual(format!("min {domain_min}, max {domain_max}")),
            ));
        }
        #[allow(
            clippy::cast_possible_truncation,
            reason = "bins is validated <= MAX_BINS (2^20), well within u32"
        )]
        let bins = bins as u32;
        Ok(Self {
            bins,
            domain_min,
            domain_max,
        })
    }

    /// The bin index a finite, in-range sample `v` falls in. The upper domain
    /// edge folds into the last bin (`bins - 1`).
    #[must_use]
    fn bin_of(&self, v: f64) -> usize {
        let width = (self.domain_max - self.domain_min) / f64::from(self.bins);
        #[allow(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "v is in [domain_min, domain_max], so the quotient is in [0, bins]; \
                      clamped to bins-1"
        )]
        let raw = ((v - self.domain_min) / width) as usize;
        raw.min(self.bins as usize - 1)
    }
}

/// Resolve a required positive-integer parameter.
fn require_u64(params: &serde_json::Value, name: &str) -> Result<u64> {
    let value = params.get(name).ok_or_else(|| {
        Error::new(
            ErrorClass::Schema,
            E_STATS_PARAM,
            format!("analyze.histogram requires a `{name}` parameter"),
        )
    })?;
    value.as_u64().ok_or_else(|| {
        Error::new(
            ErrorClass::Schema,
            E_STATS_PARAM,
            format!("analyze.histogram `{name}` must be a non-negative integer"),
        )
        .with_context(ErrorContext::default().with_actual(value.to_string()))
    })
}

/// Resolve a required finite floating-point parameter.
fn require_finite(params: &serde_json::Value, name: &str) -> Result<f64> {
    let value = params.get(name).ok_or_else(|| {
        Error::new(
            ErrorClass::Schema,
            E_STATS_PARAM,
            format!("analyze.histogram requires a `{name}` parameter"),
        )
    })?;
    let v = value.as_f64().ok_or_else(|| {
        Error::new(
            ErrorClass::Schema,
            E_STATS_PARAM,
            format!("analyze.histogram `{name}` must be a finite number"),
        )
    })?;
    if v.is_finite() {
        Ok(v)
    } else {
        Err(Error::new(
            ErrorClass::Schema,
            E_STATS_PARAM,
            format!("analyze.histogram `{name}` must be a finite number"),
        )
        .with_context(ErrorContext::default().with_actual(v.to_string())))
    }
}

/// Bin `samples` (interleaved `channels`-wide, row-major) into a per-channel
/// [`HistogramData`] over the resolved domain. Finite samples below / above the
/// domain and non-finite samples are tallied separately.
#[must_use]
fn histogram_of(samples: &[f32], channels: u32, params: HistogramParams) -> HistogramData {
    let channel_count = channels as usize;
    let bins = params.bins as usize;
    let mut counts = vec![0_u64; channel_count.saturating_mul(bins)];
    let mut below = vec![0_u64; channel_count];
    let mut above = vec![0_u64; channel_count];
    let mut nonfinite = vec![0_u64; channel_count];

    if channel_count != 0 {
        for pixel in samples.chunks_exact(channel_count) {
            for (channel, &sample) in pixel.iter().enumerate() {
                if !sample.is_finite() {
                    nonfinite[channel] += 1;
                    continue;
                }
                let v = f64::from(sample);
                if v < params.domain_min {
                    below[channel] += 1;
                } else if v > params.domain_max {
                    above[channel] += 1;
                } else {
                    let bin = params.bin_of(v);
                    counts[channel * bins + bin] += 1;
                }
            }
        }
    }

    HistogramData {
        channels,
        bins: params.bins,
        domain_min: params.domain_min,
        domain_max: params.domain_max,
        counts,
        below,
        above,
        nonfinite,
    }
}

/// The `analyze.histogram@1` operation.
#[derive(Debug, Clone, Copy, Default)]
pub struct Histogram;

impl Histogram {
    /// Construct the histogram operation.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    /// The declared manifest for `analyze.histogram@1`.
    ///
    /// # Errors
    /// Propagates the [`schema`](ErrorClass::Schema) error if the hard-coded op /
    /// impl ids are somehow invalid (they are not).
    pub fn manifest() -> Result<OperationManifest> {
        Ok(OperationManifest {
            id: HISTOGRAM_OP_ID.parse()?,
            impl_version: 1,
            summary: "Bin an image/field's samples into per-channel equal-width bins over an \
                      explicit value domain. The domain and bin count are required, so the bin \
                      assignment is fully explicit; out-of-domain and non-finite samples are \
                      tallied separately."
                .to_owned(),
            determinism: DeterminismTier::Exact,
            roi: RoiPolicy {
                category: RoiCategory::FullDomain,
                halo_px: None,
            },
            inputs: vec![InputSpec {
                name: "resource".to_owned(),
                kind: ResourceKind::Image,
                required: true,
                doc: "The image/field whose samples are binned.".to_owned(),
            }],
            outputs: vec![OutputSpec {
                name: "report".to_owned(),
                kind: ResourceKind::Report,
                doc: "The histogram report: per-channel bin counts plus below/above/non-finite \
                      tallies."
                    .to_owned(),
            }],
            params: vec![
                ParamSpec {
                    name: "bins".to_owned(),
                    ty: ParamType::Integer,
                    unit: None,
                    required: true,
                    default: None,
                    choices: vec![],
                    doc: "The number of equal-width bins per channel (1..=2^20).".to_owned(),
                },
                ParamSpec {
                    name: "domain_min".to_owned(),
                    ty: ParamType::Float,
                    unit: None,
                    required: true,
                    default: None,
                    choices: vec![],
                    doc: "The inclusive lower edge of the value domain.".to_owned(),
                },
                ParamSpec {
                    name: "domain_max".to_owned(),
                    ty: ParamType::Float,
                    unit: None,
                    required: true,
                    default: None,
                    choices: vec![],
                    doc: "The inclusive upper edge of the value domain (> domain_min).".to_owned(),
                },
            ],
            implementations: vec![reference_impl()?],
            test: reduction_test_metadata(),
        })
    }
}

impl OpContract for Histogram {
    fn declared_inputs(&self) -> Vec<(String, ResourceKind)> {
        vec![("resource".to_owned(), ResourceKind::Image)]
    }

    fn declared_outputs(&self) -> Vec<(String, ResourceKind)> {
        vec![("report".to_owned(), ResourceKind::Report)]
    }

    fn infer_outputs(
        &self,
        inputs: &Descriptors,
        params: &serde_json::Value,
    ) -> Result<OutputDescriptors> {
        // Resolve params so a malformed domain / bin count is a graph-time error.
        let _ = HistogramParams::resolve(params)?;
        let op = HISTOGRAM_OP_ID;
        let resource = require_descriptor(inputs, op, "resource")?;
        let channels = match resource {
            ResourceDescriptor::Image(d) => d.layout.channel_count(),
            _ => 0,
        };
        Ok(report_output(resource.extent(), channels))
    }

    fn required_inputs(
        &self,
        _requested_outputs: &OutputRegions,
        inputs: &Descriptors,
        _params: &serde_json::Value,
    ) -> Result<InputRegions> {
        Ok(full_domain_regions(inputs, &["resource"]))
    }

    fn validate_postconditions(
        &self,
        outputs: &OutputDescriptors,
        _params: &serde_json::Value,
    ) -> Result<Vec<AssertionResult>> {
        Ok(produces_report(outputs))
    }
}

impl OpImplementation for Histogram {
    fn compute(
        &self,
        inputs: &InputValues,
        params: &serde_json::Value,
    ) -> std::result::Result<OutputValues, Error> {
        let op = HISTOGRAM_OP_ID;
        let resolved = HistogramParams::resolve(params)?;
        let resource = require_value(inputs, op, "resource")?;
        let extent = resource.extent();
        let channels = resource.channels();

        let histogram = histogram_of(resource.samples(), channels, resolved);
        let report = Report {
            extent,
            channels,
            channel_stats: Vec::new(),
            all_finite: histogram.nonfinite.iter().all(|&n| n == 0),
            content_hash: String::new(),
            diff: None,
            assertion: None,
            histogram: Some(histogram),
            components: None,
        };
        Ok(single_report(report))
    }
}

/// Build a shape-mismatch [`semantic`](ErrorClass::Semantic) error.
fn shape_mismatch(op: &str, detail: &str, actual: String) -> Error {
    Error::new(
        ErrorClass::Semantic,
        E_STATS_SHAPE,
        format!("{op}: {detail}"),
    )
    .with_context(ErrorContext::default().with_actual(actual))
}

#[cfg(test)]
mod tests;
