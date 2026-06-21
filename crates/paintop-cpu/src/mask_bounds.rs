//! The `mask.bounds@1` operation: a coverage [`Mask`](ResourceKind::Mask) → a
//! structured [`Report`] of the mask's tight nonzero bounds and occupancy
//! (`OP_CATALOG` §4).
//!
//! `mask.bounds` is the agent's geometry-analysis primitive over masks: it reads
//! a coverage mask and reports
//!
//! - the **tight bounding box** of the mask's nonzero (occupied) region — the
//!   smallest axis-aligned [`Rect`] containing every pixel whose coverage is
//!   strictly positive — carried in the report's
//!   [`changed_bounds`](paintop_ir::DiffMetrics::changed_bounds); and
//! - the **occupancy**: the count of strictly-positive-coverage pixels, carried
//!   in [`changed_count`](paintop_ir::DiffMetrics::changed_count).
//!
//! The bounds reuse the report's `diff` block (`DiffMetrics`), whose
//! `changed_bounds`/`changed_count` are exactly "the tight bounds and count of
//! the pixels exceeding a threshold" — here the threshold is `0`, so an
//! *occupied* pixel is any pixel with positive coverage. The block's
//! `max_abs_error`/`mean_abs_error`/`rms_error` carry the coverage extremum/mean/
//! RMS for completeness.
//!
//! # Empty-mask behavior (defined, not a panic)
//!
//! An all-zero (empty) mask has **no** occupied pixels: the report's
//! `changed_count` is `0` and `changed_bounds` is `None` (a well-defined "empty
//! bounds", omitted on serialization), rather than an ill-formed rect or a
//! panic. A zero-extent mask (`0 × 0`) likewise reports empty bounds.
//!
//! # Determinism
//!
//! The op is [`Exact`](DeterminismTier::Exact): the bounds and occupancy are a
//! fixed-order row-major reduction over the coverage samples (`> 0` is a total
//! predicate on finite samples; a `NaN` sample is treated as *not* occupied), so
//! the report is a deterministic function of the mask.

use paintop_core::executor::{InputValues, OpImplementation, OutputValues, ResourceValue};
use paintop_ir::{
    AssertionResult, Descriptors, DeterminismTier, DiffMetrics, Error, ErrorClass, Extent, ImplId,
    InputRegions, InputSpec, OpContract, OperationManifest, OutputDescriptors, OutputRegions,
    OutputSpec, Rect, Report, ReportDescriptor, ResourceDescriptor, ResourceKind, Result,
    RoiCategory, RoiPolicy, TestMetadata,
};

/// The canonical id of the mask-bounds operation.
pub const BOUNDS_OP_ID: &str = "mask.bounds@1";

/// The `mask` input to bound was absent or carried a non-mask descriptor.
pub const E_BOUNDS_INPUT: &str = "E_BOUNDS_INPUT";

/// The reduction over a mask's coverage samples: the tight bounds of the occupied
/// (positive-coverage) pixels and the occupancy count.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Occupancy {
    /// The tight bounds of occupied pixels, or `None` when the mask is empty.
    bounds: Option<Rect>,
    /// The number of occupied (positive-coverage) pixels.
    count: u64,
    /// The maximum finite coverage sample (`0` for an empty/all-NaN mask).
    max: f64,
    /// The sum of finite coverage samples (used to derive the mean).
    sum: f64,
    /// The number of finite samples.
    finite: u64,
}

impl Occupancy {
    /// Reduce a coverage sample buffer (one sample per pixel, row-major) over an
    /// `extent` into the occupancy/bounds summary.
    #[must_use]
    fn measure(samples: &[f32], extent: Extent) -> Self {
        let mut bounds: Option<Rect> = None;
        let mut count: u64 = 0;
        let mut max = 0.0_f64;
        let mut sum = 0.0_f64;
        let mut finite: u64 = 0;
        let width = u64::from(extent.width).max(1);
        for (index, &sample) in samples.iter().enumerate() {
            if sample.is_finite() {
                finite += 1;
                let value = f64::from(sample);
                sum += value;
                if value > max {
                    max = value;
                }
                if value > 0.0 {
                    count += 1;
                    let index = u64::try_from(index).unwrap_or(u64::MAX);
                    let x = i64::try_from(index % width).unwrap_or(i64::MAX);
                    let y = i64::try_from(index / width).unwrap_or(i64::MAX);
                    let cell = Rect::new(x, y, x + 1, y + 1);
                    bounds = Some(bounds.map_or(cell, |b| b.union(cell)));
                }
            }
        }
        Self {
            bounds,
            count,
            max,
            sum,
            finite,
        }
    }

    /// The mean of the finite coverage samples (`0` when there are none).
    #[must_use]
    fn mean(&self) -> f64 {
        if self.finite == 0 {
            0.0
        } else {
            #[allow(
                clippy::cast_precision_loss,
                reason = "finite is a sample count; f64 mantissa covers realistic mask sizes"
            )]
            let denom = self.finite as f64;
            self.sum / denom
        }
    }

    /// Project the summary into the report's `diff` metrics block: the bounds and
    /// occupancy plus the coverage extremum/mean for completeness.
    #[must_use]
    fn into_metrics(self) -> DiffMetrics {
        DiffMetrics {
            max_abs_error: self.max,
            mean_abs_error: self.mean(),
            rms_error: 0.0,
            threshold: 0.0,
            changed_count: self.count,
            changed_bounds: self.bounds,
        }
    }
}

/// Build the bounds [`Report`] for a coverage mask value: its extent, the single
/// coverage channel's finite statistics and content hash (via the shared
/// inspection summary), and the occupancy/bounds in the `diff` block.
#[must_use]
fn bounds_report(value: &ResourceValue) -> Report {
    // Reuse the canonical inspection summary for the channel stats, all-finite
    // flag, and stable content hash, then attach the occupancy/bounds metrics.
    let mut report = crate::inspect::inspect_value(value);
    let occupancy = Occupancy::measure(value.samples(), value.extent());
    report.diff = Some(occupancy.into_metrics());
    report
}

/// The `mask.bounds@1` operation: a coverage mask → a bounds/occupancy report.
#[derive(Debug, Clone, Copy, Default)]
pub struct MaskBounds;

impl MaskBounds {
    /// Construct the bounds operation.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    /// The declared manifest for `mask.bounds@1`.
    ///
    /// # Errors
    /// Propagates the [`schema`](ErrorClass::Schema) error if the hard-coded op /
    /// impl ids are somehow invalid (they are not).
    pub fn manifest() -> Result<OperationManifest> {
        Ok(OperationManifest {
            id: BOUNDS_OP_ID.parse()?,
            impl_version: 1,
            summary: "Report the tight bounding box and occupancy (count of positive-coverage \
                      pixels) of a coverage Mask's nonzero region; an empty mask reports empty \
                      bounds and zero occupancy."
                .to_owned(),
            determinism: DeterminismTier::Exact,
            roi: RoiPolicy {
                // A global bound reads every sample.
                category: RoiCategory::FullDomain,
                halo_px: None,
            },
            inputs: vec![InputSpec {
                name: "mask".to_owned(),
                kind: ResourceKind::Mask,
                required: true,
                doc: "The coverage mask whose nonzero bounds and occupancy are reported."
                    .to_owned(),
            }],
            outputs: vec![OutputSpec {
                name: "report".to_owned(),
                kind: ResourceKind::Report,
                doc: "The bounds report: the tight nonzero bounding box (changed_bounds) and \
                      occupancy (changed_count)."
                    .to_owned(),
            }],
            params: vec![],
            implementations: vec![reference_impl()?],
            test: bounds_test_metadata(),
        })
    }
}

/// Read the `mask` input descriptor, erroring if absent or not a mask.
fn mask_descriptor(inputs: &Descriptors) -> Result<&paintop_ir::MaskDescriptor> {
    match inputs.get("mask") {
        Some(ResourceDescriptor::Mask(d)) => Ok(d),
        Some(_) => Err(Error::new(
            ErrorClass::Type,
            E_BOUNDS_INPUT,
            "mask.bounds input `mask` must be a Mask".to_owned(),
        )),
        None => Err(Error::new(
            ErrorClass::Reference,
            E_BOUNDS_INPUT,
            "mask.bounds requires a `mask` input".to_owned(),
        )),
    }
}

impl OpContract for MaskBounds {
    fn declared_inputs(&self) -> Vec<(String, ResourceKind)> {
        vec![("mask".to_owned(), ResourceKind::Mask)]
    }

    fn declared_outputs(&self) -> Vec<(String, ResourceKind)> {
        vec![("report".to_owned(), ResourceKind::Report)]
    }

    fn infer_outputs(
        &self,
        inputs: &Descriptors,
        _params: &serde_json::Value,
    ) -> Result<OutputDescriptors> {
        let mask = mask_descriptor(inputs)?;
        let mut out = OutputDescriptors::new();
        out.insert(
            "report".to_owned(),
            ResourceDescriptor::Report(ReportDescriptor {
                extent: mask.extent,
                // A coverage mask is single-channel.
                channels: 1,
            }),
        );
        Ok(out)
    }

    fn required_inputs(
        &self,
        _requested_outputs: &OutputRegions,
        inputs: &Descriptors,
        _params: &serde_json::Value,
    ) -> Result<InputRegions> {
        // A global bound depends on every sample: demand the full domain.
        let mut regions = InputRegions::new();
        if let Some(mask) = inputs.get("mask") {
            let extent = mask.extent();
            regions.insert(
                "mask".to_owned(),
                Rect::new(0, 0, i64::from(extent.width), i64::from(extent.height)),
            );
        }
        Ok(regions)
    }

    fn validate_postconditions(
        &self,
        outputs: &OutputDescriptors,
        _params: &serde_json::Value,
    ) -> Result<Vec<AssertionResult>> {
        let produced = matches!(outputs.get("report"), Some(ResourceDescriptor::Report(_)));
        Ok(vec![if produced {
            AssertionResult::pass("produces_report")
        } else {
            AssertionResult::fail("produces_report", "no `report` output produced")
        }])
    }
}

impl OpImplementation for MaskBounds {
    fn compute(
        &self,
        inputs: &InputValues,
        _params: &serde_json::Value,
    ) -> std::result::Result<OutputValues, Error> {
        let mask = inputs.get("mask").ok_or_else(|| {
            Error::new(
                ErrorClass::Reference,
                E_BOUNDS_INPUT,
                "mask.bounds requires a `mask` input value".to_owned(),
            )
        })?;
        if !matches!(mask.descriptor(), ResourceDescriptor::Mask(_)) {
            return Err(Error::new(
                ErrorClass::Type,
                E_BOUNDS_INPUT,
                "mask.bounds input `mask` must be a Mask".to_owned(),
            ));
        }
        let report = bounds_report(mask);
        let mut out = OutputValues::new();
        out.insert("report".to_owned(), ResourceValue::report(report));
        Ok(out)
    }
}

/// The mandatory `cpu.reference@1` oracle implementation id.
fn reference_impl() -> Result<ImplId> {
    ImplId::new("cpu", "reference", 1)
}

/// Verification declarations for `mask.bounds@1`: an exact, single-reference
/// analysis op. Differential and perceptual do not apply; every other category is
/// covered by this module's analytic-bounds, empty-mask, and property tests.
fn bounds_test_metadata() -> TestMetadata {
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

#[cfg(test)]
mod tests;
