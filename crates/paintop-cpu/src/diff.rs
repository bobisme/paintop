//! The `analyze.diff@1` operation: `before` + `after` → a difference field and a
//! report (`OP_CATALOG` §12, `AGENT_VERIFICATION` §2.6, `IR_SPEC` §12).
//!
//! `analyze.diff` is the agent's core *evidence* primitive: it measures the
//! whole-image change an edit produced. It takes a `before` image and an `after`
//! image of the same extent and layout and produces two outputs:
//!
//! - a **diff field** (`diff`): the per-pixel, per-channel **absolute** difference
//!   `|after − before|`, the heatmap the evidence bundle's `diffs/` consumes; and
//! - a **report** (`report`): the [`DiffMetrics`] reduction — maximum, mean, and
//!   root-mean-square absolute error, the count of *changed* pixels (those with a
//!   channel exceeding the `threshold`), and the tight bounding box of those
//!   changed pixels (the *changed bounds*).
//!
//! It is *pure* (it never mutates an input) and [`Exact`](DeterminismTier::Exact):
//! the difference is a per-channel `f32` subtraction, and every reduction is a
//! fixed-order accumulation in `f64`, so the report is a deterministic function
//! of the inputs and the comparison space.
//!
//! # Comparison space
//!
//! Two images can be compared either in the representation they are stored in
//! (`encoded`) or in linear light (`decoded-linear`). The `comparison_space`
//! parameter makes the choice **explicit** (`IR_SPEC` §12): an `srgb`-encoded
//! pair compared in `decoded-linear` is sRGB-decoded to linear light *before*
//! the difference, so the metrics measure a physically meaningful light delta
//! rather than a perceptual-curve delta; compared as `encoded`, the difference is
//! taken on the stored samples verbatim. Linear inputs (`linear-srgb`,
//! `raw-linear`) decode to themselves, so the two spaces agree for them.
//!
//! # Finiteness
//!
//! Each diff sample is `|after − before|`; the report records whether every diff
//! sample is finite ([`Report::all_finite`]). A non-finite input sample yields a
//! non-finite diff and is excluded from the metric reductions (which aggregate
//! finite samples only), so one poisoned sample never silently corrupts the
//! reported error while still being flagged through `all_finite`.

use paintop_core::executor::{InputValues, OpImplementation, OutputValues, ResourceValue};
use paintop_ir::{
    AlphaRepresentation, AssertionResult, ChannelStats, ColorEncoding, ColorRange,
    CoordinateConvention, Descriptors, DeterminismTier, DiffMetrics, Error, ErrorClass,
    ErrorContext, Extent, HashDomain, ImageDescriptor, ImplId, InputRegions, InputSpec, OpContract,
    OperationManifest, OutputDescriptors, OutputRegions, OutputSpec, ParamSpec, ParamType,
    ParamUnit, Rect, Report, ResourceDescriptor, ResourceKind, Result, RoiCategory, RoiPolicy,
    ScalarType, SemanticRole, TestMetadata, hash_canonical_bytes,
};

/// The canonical id of the diff operation.
pub const DIFF_OP_ID: &str = "analyze.diff@1";

/// A required input port (`before` / `after`) was absent or carried a
/// non-image descriptor.
pub const E_DIFF_INPUT: &str = "E_DIFF_INPUT";

/// The `before` and `after` images disagree on extent or channel layout, so they
/// cannot be differenced channel-for-channel.
pub const E_DIFF_SHAPE: &str = "E_DIFF_SHAPE";

/// The `comparison_space` parameter was not a known comparison-space token.
pub const E_DIFF_PARAM: &str = "E_DIFF_PARAM";

/// The op produced a sample buffer whose length disagrees with its descriptor.
pub const E_DIFF_BUFFER: &str = "E_DIFF_BUFFER";

/// The canonical quiet-`NaN` bit pattern every `NaN` diff sample is normalized
/// to before hashing, so the diff field's content hash depends only on a
/// sample's logical value and never on a particular `NaN` payload.
const CANONICAL_NAN_BITS: u32 = 0x7fc0_0000;

/// The space the `before`/`after` pair is compared in (`IR_SPEC` §12).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ComparisonSpace {
    /// Compare the stored samples verbatim, without decoding.
    Encoded,
    /// Decode each input to linear light before differencing.
    DecodedLinear,
}

impl ComparisonSpace {
    /// The `comparison_space` token for the [`DecodedLinear`](Self::DecodedLinear)
    /// space.
    const DECODED_LINEAR: &'static str = "decoded-linear";
    /// The `comparison_space` token for the [`Encoded`](Self::Encoded) space.
    const ENCODED: &'static str = "encoded";

    /// Resolve the `comparison_space` parameter, defaulting to
    /// [`DecodedLinear`](Self::DecodedLinear) when absent (the physically
    /// meaningful default the manifest declares).
    fn resolve(params: &serde_json::Value) -> Result<Self> {
        let Some(value) = params.get("comparison_space") else {
            return Ok(Self::DecodedLinear);
        };
        match value.as_str() {
            Some(Self::DECODED_LINEAR) => Ok(Self::DecodedLinear),
            Some(Self::ENCODED) => Ok(Self::Encoded),
            other => Err(Error::new(
                ErrorClass::Schema,
                E_DIFF_PARAM,
                format!(
                    "analyze.diff `comparison_space` must be `{}` or `{}`",
                    Self::DECODED_LINEAR,
                    Self::ENCODED,
                ),
            )
            .with_context(
                ErrorContext::default()
                    .with_actual(other.unwrap_or("<non-string>").to_owned())
                    .with_expected(format!("{} | {}", Self::DECODED_LINEAR, Self::ENCODED)),
            )),
        }
    }
}

/// Resolve the `threshold` parameter: the (non-negative) absolute error a pixel
/// must strictly exceed to count as *changed*. Defaults to `0.0` (any nonzero
/// change counts) when absent.
fn resolve_threshold(params: &serde_json::Value) -> Result<f64> {
    let Some(value) = params.get("threshold") else {
        return Ok(0.0);
    };
    let threshold = value.as_f64().ok_or_else(|| {
        Error::new(
            ErrorClass::Schema,
            E_DIFF_PARAM,
            "analyze.diff `threshold` must be a finite non-negative number".to_owned(),
        )
    })?;
    if !threshold.is_finite() || threshold < 0.0 {
        return Err(Error::new(
            ErrorClass::Schema,
            E_DIFF_PARAM,
            "analyze.diff `threshold` must be a finite non-negative number".to_owned(),
        )
        .with_context(ErrorContext::default().with_actual(threshold.to_string())));
    }
    Ok(threshold)
}

/// The sRGB decode knot (`srgb -> linear`): below this the function is linear.
const SRGB_DECODE_KNOT: f32 = 0.040_45;

/// Decode one sRGB-encoded sample to linear light (IEC 61966-2-1), matching
/// `color.convert@1`'s transfer so the two ops agree on linear values.
#[must_use]
fn srgb_decode(c: f32) -> f32 {
    if c <= SRGB_DECODE_KNOT {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

/// Whether the input encoding must be sRGB-decoded for the requested comparison
/// space: only an `srgb`-encoded input compared in `decoded-linear` is decoded;
/// linear encodings decode to themselves and `encoded` never decodes.
const fn decodes(space: ComparisonSpace, color: ColorEncoding) -> bool {
    matches!(space, ComparisonSpace::DecodedLinear) && matches!(color, ColorEncoding::Srgb)
}

/// Validate that `before` and `after` may be differenced and return the diff
/// output descriptor.
///
/// The two inputs must agree on extent and channel layout (so the difference is
/// channel-for-channel). The diff output keeps their extent and layout but is
/// retyped to a `raw-linear`, straight-alpha **material** field: an absolute
/// difference magnitude is no longer displayable, premultiplied color.
fn check_and_retarget(
    before: &ImageDescriptor,
    after: &ImageDescriptor,
) -> Result<ImageDescriptor> {
    if before.extent != after.extent {
        return Err(shape_mismatch(
            "the `before` and `after` images must share an extent",
            format!("before {:?} vs after {:?}", before.extent, after.extent),
        ));
    }
    if before.layout != after.layout {
        return Err(shape_mismatch(
            "the `before` and `after` images must share a channel layout",
            format!("before {:?} vs after {:?}", before.layout, after.layout),
        ));
    }
    Ok(ImageDescriptor {
        extent: before.extent,
        layout: before.layout,
        scalar: ScalarType::F32,
        color: ColorEncoding::RawLinear,
        range: ColorRange::SceneReferred,
        alpha: AlphaRepresentation::Straight,
        coordinates: CoordinateConvention::PixelCenterUpperLeft,
        semantic: SemanticRole::Material,
    })
}

/// Build a shape-mismatch [`semantic`](ErrorClass::Semantic) error.
fn shape_mismatch(detail: &str, actual: String) -> Error {
    Error::new(
        ErrorClass::Semantic,
        E_DIFF_SHAPE,
        format!("analyze.diff: {detail}"),
    )
    .with_context(ErrorContext::default().with_actual(actual))
}

/// Map a sample to the comparison space: sRGB-decoded when `decode` is set, else
/// passed through.
#[must_use]
fn to_space(sample: f32, decode: bool) -> f32 {
    if decode { srgb_decode(sample) } else { sample }
}

/// The per-pixel, per-channel absolute difference `|after − before|` in the
/// requested comparison space, interleaved `channels`-wide in row-major order.
#[must_use]
fn diff_samples(
    before: &[f32],
    after: &[f32],
    decode_before: bool,
    decode_after: bool,
) -> Vec<f32> {
    before
        .iter()
        .zip(after.iter())
        .map(|(&b, &a)| (to_space(a, decode_after) - to_space(b, decode_before)).abs())
        .collect()
}

/// Compute the per-channel finite statistics of the diff field (reused for the
/// report), interleaved `channels`-wide in row-major order.
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
            finite: 0,
            nonfinite: 0,
        };
        channel_count
    ];
    for (i, &sample) in samples.iter().enumerate() {
        let entry = &mut stats[i % channel_count];
        if sample.is_finite() {
            entry.min = Some(entry.min.map_or(sample, |m| m.min(sample)));
            entry.max = Some(entry.max.map_or(sample, |m| m.max(sample)));
            entry.sum += f64::from(sample);
            entry.finite += 1;
        } else {
            entry.nonfinite += 1;
        }
    }
    stats
}

/// Reduce the diff field to its [`DiffMetrics`]: the max / mean / RMS finite
/// absolute error, the count of *changed* pixels (a pixel with any channel whose
/// finite diff strictly exceeds `threshold`), and their tight bounding box.
///
/// All reductions accumulate finite samples only, in a fixed row-major order, in
/// `f64`, so they are deterministic. `width`/`height` shape the changed-bounds
/// scan; `channels` is the interleaved sample count per pixel.
#[must_use]
fn diff_metrics(samples: &[f32], extent: Extent, channels: u32, threshold: f64) -> DiffMetrics {
    let mut max_abs = 0.0_f64;
    let mut sum_abs = 0.0_f64;
    let mut sum_sq = 0.0_f64;
    let mut finite_count: u64 = 0;
    let mut changed_count: u64 = 0;
    let mut bounds: Option<Rect> = None;

    let channel_count = channels as usize;
    let width = extent.width;
    if channel_count != 0 {
        for (pixel_index, pixel) in samples.chunks_exact(channel_count).enumerate() {
            let mut pixel_changed = false;
            for &sample in pixel {
                if sample.is_finite() {
                    let abs = f64::from(sample);
                    if abs > max_abs {
                        max_abs = abs;
                    }
                    sum_abs += abs;
                    sum_sq = abs.mul_add(abs, sum_sq);
                    finite_count += 1;
                    if abs > threshold {
                        pixel_changed = true;
                    }
                }
            }
            if pixel_changed {
                changed_count += 1;
                // pixel_index < width*height, and width > 0 (an empty extent
                // yields no pixels, so this branch is unreachable for width 0).
                let index = u64::try_from(pixel_index).unwrap_or(u64::MAX);
                let stride = u64::from(width).max(1);
                let x = i64::try_from(index % stride).unwrap_or(i64::MAX);
                let y = i64::try_from(index / stride).unwrap_or(i64::MAX);
                let cell = Rect::new(x, y, x + 1, y + 1);
                bounds = Some(bounds.map_or(cell, |b| b.union(cell)));
            }
        }
    }

    #[allow(
        clippy::cast_precision_loss,
        reason = "finite_count is a sample count; f64 mantissa covers realistic image sizes"
    )]
    let denom = finite_count as f64;
    let (mean_abs, rms) = if finite_count == 0 {
        (0.0, 0.0)
    } else {
        (sum_abs / denom, (sum_sq / denom).sqrt())
    };

    DiffMetrics {
        max_abs_error: max_abs,
        mean_abs_error: mean_abs,
        rms_error: rms,
        threshold,
        changed_count,
        changed_bounds: bounds,
    }
}

/// The fixed byte encoding of the diff field hashed for the report's content
/// hash: extent, channel count, then every diff sample's `NaN`-normalized
/// IEEE-754 little-endian bits.
#[must_use]
fn content_bytes(extent: Extent, channels: u32, samples: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(16 + samples.len() * 4);
    bytes.extend_from_slice(&extent.width.to_le_bytes());
    bytes.extend_from_slice(&extent.height.to_le_bytes());
    bytes.extend_from_slice(&channels.to_le_bytes());
    for &sample in samples {
        let bits = if sample.is_nan() {
            CANONICAL_NAN_BITS
        } else {
            sample.to_bits()
        };
        bytes.extend_from_slice(&bits.to_le_bytes());
    }
    bytes
}

/// Build the [`Report`] summarizing a diff field: its extent, per-channel finite
/// statistics, the [`DiffMetrics`] reduction, the all-finite flag, and a stable
/// content hash of the diff samples.
#[must_use]
fn diff_report(samples: &[f32], extent: Extent, channels: u32, threshold: f64) -> Report {
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
        diff: Some(diff_metrics(samples, extent, channels, threshold)),
        assertion: None,
    }
}

/// Read a required image input *descriptor* port, erroring if absent or the
/// wrong kind.
fn image_descriptor<'a>(inputs: &'a Descriptors, port: &str) -> Result<&'a ImageDescriptor> {
    let resource = inputs.get(port).ok_or_else(|| {
        Error::new(
            ErrorClass::Reference,
            E_DIFF_INPUT,
            format!("analyze.diff requires a `{port}` input"),
        )
    })?;
    let ResourceDescriptor::Image(descriptor) = resource else {
        return Err(Error::new(
            ErrorClass::Type,
            E_DIFF_INPUT,
            format!("analyze.diff `{port}` input must be an image resource"),
        ));
    };
    Ok(descriptor)
}

/// Read a required input *value* port, erroring if absent.
fn input_value<'a>(
    inputs: &'a InputValues,
    port: &str,
) -> std::result::Result<&'a ResourceValue, Error> {
    inputs.get(port).ok_or_else(|| {
        Error::new(
            ErrorClass::Reference,
            E_DIFF_INPUT,
            format!("analyze.diff requires a `{port}` input value"),
        )
    })
}

/// The `analyze.diff@1` operation: `before` + `after` → a `diff` field and a
/// `report`.
#[derive(Debug, Clone, Copy, Default)]
pub struct Diff;

impl Diff {
    /// Construct the diff operation.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    /// The declared manifest for `analyze.diff@1`.
    ///
    /// # Errors
    /// Propagates the [`schema`](ErrorClass::Schema) error if the hard-coded op /
    /// impl ids are somehow invalid (they are not).
    pub fn manifest() -> Result<OperationManifest> {
        Ok(OperationManifest {
            id: DIFF_OP_ID.parse()?,
            impl_version: 1,
            summary: "Difference a before and after image into a per-pixel absolute-difference \
                      field and a report (max / mean / RMS error, changed-pixel count and \
                      bounds) in an explicit comparison space (encoded or decoded-linear). \
                      Identical inputs yield a zero diff with empty changed bounds."
                .to_owned(),
            // |after − before| per channel plus fixed-order f64 reductions: a
            // deterministic function of the inputs and comparison space.
            determinism: DeterminismTier::Exact,
            roi: RoiPolicy {
                // The report reduces over every diff sample.
                category: RoiCategory::FullDomain,
                halo_px: None,
            },
            inputs: vec![
                InputSpec {
                    name: "before".to_owned(),
                    kind: ResourceKind::Image,
                    required: true,
                    doc: "The reference image the change is measured against.".to_owned(),
                },
                InputSpec {
                    name: "after".to_owned(),
                    kind: ResourceKind::Image,
                    required: true,
                    doc: "The edited image whose change from `before` is measured.".to_owned(),
                },
            ],
            outputs: vec![
                OutputSpec {
                    name: "diff".to_owned(),
                    kind: ResourceKind::Image,
                    doc: "The per-pixel, per-channel absolute-difference field |after − before| \
                          (the evidence heatmap), retyped raw-linear material."
                        .to_owned(),
                },
                OutputSpec {
                    name: "report".to_owned(),
                    kind: ResourceKind::Report,
                    doc: "The difference report: per-channel stats plus max / mean / RMS error, \
                          changed-pixel count and bounds."
                        .to_owned(),
                },
            ],
            params: vec![
                ParamSpec {
                    name: "comparison_space".to_owned(),
                    ty: ParamType::String,
                    unit: None,
                    required: false,
                    default: Some(serde_json::Value::String(
                        ComparisonSpace::DECODED_LINEAR.to_owned(),
                    )),
                    choices: vec![
                        ComparisonSpace::ENCODED.to_owned(),
                        ComparisonSpace::DECODED_LINEAR.to_owned(),
                    ],
                    doc: "The space the pair is compared in: `encoded` (stored samples) or \
                          `decoded-linear` (sRGB-decoded to linear light first)."
                        .to_owned(),
                },
                ParamSpec {
                    name: "threshold".to_owned(),
                    ty: ParamType::Float,
                    unit: Some(ParamUnit::Ratio),
                    required: false,
                    default: Some(serde_json::json!(0.0)),
                    choices: vec![],
                    doc: "The absolute error a pixel must strictly exceed to count as changed \
                          (drives changed_count / changed_bounds)."
                        .to_owned(),
                },
            ],
            implementations: vec![reference_impl()?],
            test: diff_test_metadata(),
        })
    }
}

impl OpContract for Diff {
    fn declared_inputs(&self) -> Vec<(String, ResourceKind)> {
        vec![
            ("before".to_owned(), ResourceKind::Image),
            ("after".to_owned(), ResourceKind::Image),
        ]
    }

    fn declared_outputs(&self) -> Vec<(String, ResourceKind)> {
        vec![
            ("diff".to_owned(), ResourceKind::Image),
            ("report".to_owned(), ResourceKind::Report),
        ]
    }

    fn infer_outputs(
        &self,
        inputs: &Descriptors,
        params: &serde_json::Value,
    ) -> Result<OutputDescriptors> {
        // Resolve the params so a malformed comparison space / threshold is a
        // graph-time error, not a run-time surprise.
        let _ = ComparisonSpace::resolve(params)?;
        let _ = resolve_threshold(params)?;

        let before = image_descriptor(inputs, "before")?;
        let after = image_descriptor(inputs, "after")?;
        let diff_descriptor = check_and_retarget(before, after)?;

        let mut out = OutputDescriptors::new();
        out.insert(
            "diff".to_owned(),
            ResourceDescriptor::Image(diff_descriptor),
        );
        out.insert(
            "report".to_owned(),
            ResourceDescriptor::Report(paintop_ir::ReportDescriptor {
                extent: diff_descriptor.extent,
                channels: diff_descriptor.layout.channel_count(),
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
        // The report reduces over every diff sample, so the op demands both
        // inputs' full domain regardless of which output region is requested.
        let mut regions = InputRegions::new();
        for port in ["before", "after"] {
            if let Some(input) = inputs.get(port) {
                let extent = input.extent();
                regions.insert(
                    port.to_owned(),
                    Rect::new(0, 0, i64::from(extent.width), i64::from(extent.height)),
                );
            }
        }
        Ok(regions)
    }

    fn validate_postconditions(
        &self,
        outputs: &OutputDescriptors,
        _params: &serde_json::Value,
    ) -> Result<Vec<AssertionResult>> {
        let produces_diff = matches!(outputs.get("diff"), Some(ResourceDescriptor::Image(_)));
        let produces_report = matches!(outputs.get("report"), Some(ResourceDescriptor::Report(_)));
        Ok(vec![
            if produces_diff {
                AssertionResult::pass("produces_diff_field")
            } else {
                AssertionResult::fail("produces_diff_field", "no `diff` output produced")
            },
            if produces_report {
                AssertionResult::pass("produces_report")
            } else {
                AssertionResult::fail("produces_report", "no `report` output produced")
            },
        ])
    }
}

impl OpImplementation for Diff {
    fn compute(
        &self,
        inputs: &InputValues,
        params: &serde_json::Value,
    ) -> std::result::Result<OutputValues, Error> {
        let space = ComparisonSpace::resolve(params)?;
        let threshold = resolve_threshold(params)?;

        let before = input_value(inputs, "before")?;
        let after = input_value(inputs, "after")?;

        let ResourceDescriptor::Image(before_descriptor) = before.descriptor() else {
            return Err(input_type_error("before"));
        };
        let ResourceDescriptor::Image(after_descriptor) = after.descriptor() else {
            return Err(input_type_error("after"));
        };
        let diff_descriptor = check_and_retarget(before_descriptor, after_descriptor)?;

        let decode_before = decodes(space, before_descriptor.color);
        let decode_after = decodes(space, after_descriptor.color);
        let samples = diff_samples(
            before.samples(),
            after.samples(),
            decode_before,
            decode_after,
        );

        let channels = before.channels();
        let extent = diff_descriptor.extent;
        let report = diff_report(&samples, extent, channels, threshold);

        let diff_value = ResourceValue::new(
            ResourceDescriptor::Image(diff_descriptor),
            channels,
            samples,
        )
        .map_err(|actual| {
            Error::new(
                ErrorClass::Execution,
                E_DIFF_BUFFER,
                format!("analyze.diff produced a diff buffer of unexpected length {actual}"),
            )
        })?;

        let mut out = OutputValues::new();
        out.insert("diff".to_owned(), diff_value);
        out.insert("report".to_owned(), ResourceValue::report(report));
        Ok(out)
    }
}

/// The wrong-resource-kind error for an image input port.
fn input_type_error(port: &str) -> Error {
    Error::new(
        ErrorClass::Type,
        E_DIFF_INPUT,
        format!("analyze.diff `{port}` input must be an image resource"),
    )
}

/// The mandatory `cpu.reference@1` oracle implementation id.
fn reference_impl() -> Result<ImplId> {
    ImplId::new("cpu", "reference", 1)
}

/// The verification declarations for `analyze.diff@1` (`OP_CATALOG` §12,
/// `AGENT_VERIFICATION` §2.6). It is an exact, single-reference reduction
/// verified by identity (zero diff / empty changed bounds), known-delta exact
/// metrics, and comparison-space analytic fixtures plus metamorphic properties
/// (anti-symmetry of the magnitude, additivity of the changed-count). Both
/// differential (one implementation) and perceptual (exact, no quality metric)
/// do not apply and are derived not-applicable; every other category is covered.
fn diff_test_metadata() -> TestMetadata {
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
