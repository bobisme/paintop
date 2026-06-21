//! The `image.inspect@1` operation: any resource → a structured [`Report`]
//! (`OP_CATALOG` §1, `M0_DECISIONS` D2).
//!
//! `image.inspect` is the agent's first analysis primitive: it reads a resource
//! and produces a [`Report`] carrying the resource's extent, per-channel finite
//! statistics (min/max/sum/mean, finite and non-finite counts), and a stable
//! **content hash** of the samples. It is *pure* — it never mutates its input —
//! and *exact*: the report is a deterministic function of the input samples.
//!
//! # Finite vs. non-finite samples
//!
//! Image math can produce `NaN`/`±∞` (a divide-by-zero, an unpremultiply of a
//! zero-alpha pixel). The report must surface those without letting one poisoned
//! sample destroy the whole channel's range, so every per-channel extremum and
//! the sum are computed over the channel's **finite** samples only, while the
//! non-finite samples are *counted* in [`ChannelStats::nonfinite`]. A channel of
//! all-`NaN` samples therefore reports `min`/`max`/`mean` of `None` and a
//! `nonfinite` count equal to its sample count, rather than a meaningless range.
//!
//! # Content hash
//!
//! The content hash is computed by the canonical hashing module
//! ([`paintop_ir::hash`], domain [`Content`](paintop_ir::HashDomain::Content))
//! over a fixed byte encoding of the resource: its extent and channel count
//! followed by every sample's IEEE-754 little-endian bits, with **every** `NaN`
//! normalized to one canonical quiet-`NaN` pattern. Normalizing `NaN` keeps the
//! hash a function of the resource's *logical* content (two computations that
//! both produce "not a number" hash identically regardless of the bit payload
//! the FPU happened to leave behind), so the hash is stable across runs and
//! machines.

use paintop_core::executor::{InputValues, OpImplementation, OutputValues, ResourceValue};
use paintop_ir::{
    AssertionResult, ChannelStats, Descriptors, DeterminismTier, Error, ErrorClass, HashDomain,
    ImplId, InputRegions, InputSpec, OpContract, OperationManifest, OutputDescriptors,
    OutputRegions, OutputSpec, Rect, Report, ReportDescriptor, ResourceDescriptor, ResourceKind,
    Result, RoiCategory, RoiPolicy, TestMetadata, hash_canonical_bytes,
};

/// The canonical id of the inspect operation.
pub const INSPECT_OP_ID: &str = "image.inspect@1";

/// The `image` input to inspect was absent or carried no sample buffer.
pub const E_INSPECT_INPUT: &str = "E_INSPECT_INPUT";

/// The canonical quiet-`NaN` bit pattern every `NaN` sample is normalized to
/// before hashing, so the content hash depends only on a sample's *value* (a
/// finite number, an infinity, or "not a number") and never on a particular
/// `NaN` payload the hardware produced.
const CANONICAL_NAN_BITS: u32 = 0x7fc0_0000;

/// Compute the per-channel finite statistics of `samples`, interleaved
/// `channels`-wide in row-major order.
///
/// Each returned [`ChannelStats`] aggregates only the channel's finite samples
/// for its extrema and sum; non-finite samples are counted separately. An empty
/// or zero-channel resource yields one all-zero entry per channel.
#[must_use]
fn channel_statistics(samples: &[f32], channels: u32) -> Vec<ChannelStats> {
    let channel_count = channels as usize;
    if channel_count == 0 {
        return Vec::new();
    }
    let mut stats = vec![
        ChannelStats {
            min: None,
            max: None,
            sum: 0.0,
            sum_sq: 0.0,
            finite: 0,
            nonfinite: 0,
        };
        channel_count
    ];
    for (i, &sample) in samples.iter().enumerate() {
        let channel = i % channel_count;
        let entry = &mut stats[channel];
        if sample.is_finite() {
            entry.min = Some(entry.min.map_or(sample, |m| m.min(sample)));
            entry.max = Some(entry.max.map_or(sample, |m| m.max(sample)));
            let value = f64::from(sample);
            entry.sum += value;
            entry.sum_sq = value.mul_add(value, entry.sum_sq);
            entry.finite += 1;
        } else {
            entry.nonfinite += 1;
        }
    }
    stats
}

/// The fixed byte encoding of a resource hashed for its content hash: extent,
/// channel count, then every sample's `NaN`-normalized IEEE-754 little-endian
/// bits.
#[must_use]
fn content_bytes(extent: paintop_ir::Extent, channels: u32, samples: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(16 + samples.len() * 4);
    bytes.extend_from_slice(&extent.width.to_le_bytes());
    bytes.extend_from_slice(&extent.height.to_le_bytes());
    bytes.extend_from_slice(&channels.to_le_bytes());
    for &sample in samples {
        // Normalize every NaN to one canonical pattern so the hash tracks the
        // sample's logical value, not the FPU's NaN payload.
        let bits = if sample.is_nan() {
            CANONICAL_NAN_BITS
        } else {
            sample.to_bits()
        };
        bytes.extend_from_slice(&bits.to_le_bytes());
    }
    bytes
}

/// Build the [`Report`] for a resource value: its extent, per-channel finite
/// statistics, and content hash.
#[must_use]
pub fn inspect_value(value: &ResourceValue) -> Report {
    let extent = value.extent();
    let channels = value.channels();
    let samples = value.samples();
    let channel_stats = channel_statistics(samples, channels);
    let all_finite = channel_stats.iter().all(ChannelStats::all_finite);
    let content_hash = hash_canonical_bytes(
        HashDomain::Content,
        &content_bytes(extent, channels, samples),
    )
    .to_string();
    Report {
        extent,
        channels,
        channel_stats,
        all_finite,
        content_hash,
        diff: None,
        assertion: None,
        histogram: None,
    }
}

/// The `image.inspect@1` operation: a resource → a structured [`Report`].
#[derive(Debug, Clone, Copy, Default)]
pub struct Inspect;

impl Inspect {
    /// Construct the inspect operation.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    /// The declared manifest for `image.inspect@1`.
    ///
    /// # Errors
    /// Propagates the [`schema`](ErrorClass::Schema) error if the hard-coded op /
    /// impl ids are somehow invalid (they are not).
    pub fn manifest() -> Result<OperationManifest> {
        Ok(OperationManifest {
            id: INSPECT_OP_ID.parse()?,
            impl_version: 1,
            summary: "Summarize a resource into a Report: extent, per-channel ranges, \
                      finite-value statistics, and a stable content hash."
                .to_owned(),
            determinism: DeterminismTier::Exact,
            roi: RoiPolicy {
                // Inspection reads every sample to compute global statistics.
                category: RoiCategory::FullDomain,
                halo_px: None,
            },
            inputs: vec![InputSpec {
                name: "image".to_owned(),
                kind: ResourceKind::Image,
                required: true,
                doc: "The resource to inspect.".to_owned(),
            }],
            outputs: vec![OutputSpec {
                name: "report".to_owned(),
                kind: ResourceKind::Report,
                doc: "The analysis report: extent, ranges, finite stats, content hash.".to_owned(),
            }],
            params: vec![],
            implementations: vec![reference_impl()?],
            test: inspect_test_metadata(),
        })
    }
}

impl OpContract for Inspect {
    fn declared_inputs(&self) -> Vec<(String, ResourceKind)> {
        vec![("image".to_owned(), ResourceKind::Image)]
    }

    fn declared_outputs(&self) -> Vec<(String, ResourceKind)> {
        vec![("report".to_owned(), ResourceKind::Report)]
    }

    fn infer_outputs(
        &self,
        inputs: &Descriptors,
        _params: &serde_json::Value,
    ) -> Result<OutputDescriptors> {
        // The report summarizes the input resource, so its descriptor records the
        // input's extent and channel count. The statistical payload is produced
        // only at execution (it depends on the samples).
        let input = inputs.get("image").ok_or_else(|| {
            Error::new(
                ErrorClass::Reference,
                E_INSPECT_INPUT,
                "image.inspect requires an `image` input".to_owned(),
            )
        })?;
        let mut out = OutputDescriptors::new();
        out.insert(
            "report".to_owned(),
            ResourceDescriptor::Report(ReportDescriptor {
                extent: input.extent(),
                channels: channel_count_of(input),
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
        // A global statistic depends on every sample: inspect demands the input's
        // full domain regardless of which part of the (point-sized) report is
        // requested.
        let mut regions = InputRegions::new();
        if let Some(input) = inputs.get("image") {
            let extent = input.extent();
            regions.insert(
                "image".to_owned(),
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

impl OpImplementation for Inspect {
    fn compute(
        &self,
        inputs: &InputValues,
        _params: &serde_json::Value,
    ) -> std::result::Result<OutputValues, Error> {
        let image = inputs.get("image").ok_or_else(|| {
            Error::new(
                ErrorClass::Reference,
                E_INSPECT_INPUT,
                "image.inspect requires an `image` input value".to_owned(),
            )
        })?;
        let report = inspect_value(image);
        let mut out = OutputValues::new();
        out.insert("report".to_owned(), ResourceValue::report(report));
        Ok(out)
    }
}

/// The channel count of a resource descriptor, used to shape the report.
///
/// Only an image's channel count is structurally known from its layout at the
/// descriptor level; for other kinds the report's channel count is filled in at
/// execution from the concrete value, so a conservative `0` is used here.
const fn channel_count_of(descriptor: &ResourceDescriptor) -> u32 {
    match descriptor {
        ResourceDescriptor::Image(d) => d.layout.channel_count(),
        _ => 0,
    }
}

/// The mandatory `cpu.reference@1` oracle implementation id.
fn reference_impl() -> Result<ImplId> {
    ImplId::new("cpu", "reference", 1)
}

/// The verification declarations for `image.inspect@1`. It is an exact,
/// single-reference op, so differential and perceptual do not apply; every other
/// category is covered by the analytic-fixture and property tests in this module.
fn inspect_test_metadata() -> TestMetadata {
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
