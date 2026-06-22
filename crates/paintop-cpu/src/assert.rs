//! The two MVP assertion operations: `assert.no_change_outside_mask@1` and
//! `assert.finite@1` (`IR_SPEC` §13, `OP_CATALOG` §12, `AGENT_VERIFICATION`
//! §5.3).
//!
//! Assertions are *ordinary typed nodes*: each reads its inputs, evaluates a
//! predicate, and produces a [`Report`] whose [`assertion`](Report::assertion)
//! block records the verdict ([`passed`](paintop_ir::AssertionOutcome::passed)),
//! the explicit [`severity`](paintop_ir::AssertionSeverity) that decides whether a
//! failure fails the run (exit class 6), and the failure evidence (worst pixel,
//! offending-pixel locations, and the assertion-specific metrics). They are
//! *pure* (they never mutate an input) and [`Exact`](DeterminismTier::Exact):
//! every reduction runs in a fixed row-major order, so the verdict is a
//! deterministic function of the inputs and parameters.
//!
//! # `assert.no_change_outside_mask@1`
//!
//! The core safety assertion. Given a `before` image, an `after` image, and an
//! `allowed` coverage mask of the same extent, it checks that every pixel
//! *outside* the allowed region (coverage `<= coverage_epsilon`) changed by no
//! more than `outside_threshold` in the chosen `comparison_space`. A leak — even
//! a single pixel exceeding the threshold outside the mask — fails the assertion
//! and records the worst leaking pixel, the count of leaking pixels, the maximum
//! outside delta, and a capped list of leaking pixel locations (the inputs to the
//! evidence bundle's outside-diff artifact and minimal replay, `AGENT_VERIFICATION`
//! §5.3).
//!
//! # `assert.finite@1`
//!
//! The finiteness guard. Given a `resource`, it checks that every sample is
//! finite; any `NaN`/`±∞` fails the assertion and records the non-finite count,
//! the first such pixel (the worst pixel), and a capped list of non-finite pixel
//! locations.
//!
//! # Severity
//!
//! Severity is *explicit* (`IR_SPEC` §13): `error` (the default) fails the run on
//! a violation, `warning` retains the output but marks the evidence, and `metric`
//! never fails the run. The assertion still evaluates the predicate and records
//! the same metrics under every severity; only the *run-failing* consequence
//! differs, and that consequence is decided by the evidence/CLI layer from the
//! recorded [`severity`](paintop_ir::AssertionSeverity) and `passed` flag.

use paintop_core::executor::{InputValues, OpImplementation, OutputValues, ResourceValue};
use paintop_ir::{
    AssertionOutcome, AssertionResult, AssertionSeverity, Descriptors, DeterminismTier, Error,
    ErrorClass, ErrorContext, Extent, ImplId, InputRegions, InputSpec, MaskDescriptor, OpContract,
    OperationManifest, OutputDescriptors, OutputRegions, OutputSpec, ParamSpec, ParamType, Rect,
    Report, ReportDescriptor, ResourceDescriptor, ResourceKind, Result, RoiCategory, RoiPolicy,
    TestMetadata,
};

/// The canonical id of the no-change-outside-mask assertion.
pub const NO_CHANGE_OUTSIDE_MASK_OP_ID: &str = "assert.no_change_outside_mask@1";

/// The canonical id of the finiteness assertion.
pub const FINITE_OP_ID: &str = "assert.finite@1";

/// A required input port was absent or carried the wrong resource kind.
pub const E_ASSERT_INPUT: &str = "E_ASSERT_INPUT";

/// The inputs disagree on extent or channel layout, so they cannot be compared
/// sample-for-sample.
pub const E_ASSERT_SHAPE: &str = "E_ASSERT_SHAPE";

/// A parameter was not a known token / not a finite non-negative number.
pub const E_ASSERT_PARAM: &str = "E_ASSERT_PARAM";

/// The maximum number of offending-pixel locations recorded in an assertion's
/// `locations` list. The full count is always reported separately; this only
/// caps the explicit coordinate list so a pathological all-failing input cannot
/// produce an unbounded report.
const MAX_LOCATIONS: usize = 256;

/// The sRGB decode knot (`srgb -> linear`): below this the function is linear.
const SRGB_DECODE_KNOT: f32 = 0.040_45;

/// Decode one sRGB-encoded sample to linear light (IEC 61966-2-1), matching
/// `color.convert@1` / `analyze.diff@1` so the assertion agrees on linear values.
#[must_use]
fn srgb_decode(c: f32) -> f32 {
    if c <= SRGB_DECODE_KNOT {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

/// The space the `before`/`after` pair is compared in (`IR_SPEC` §12 / §13).
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
    /// [`DecodedLinear`](Self::DecodedLinear) when absent.
    fn resolve(params: &serde_json::Value) -> Result<Self> {
        let Some(value) = params.get("comparison_space") else {
            return Ok(Self::DecodedLinear);
        };
        match value.as_str() {
            Some(Self::DECODED_LINEAR) => Ok(Self::DecodedLinear),
            Some(Self::ENCODED) => Ok(Self::Encoded),
            other => Err(Error::new(
                ErrorClass::Schema,
                E_ASSERT_PARAM,
                format!(
                    "assert.no_change_outside_mask `comparison_space` must be `{}` or `{}`",
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

/// The severity token for an assertion (`error` / `warning` / `metric`),
/// defaulting to `error` when absent (the strict default per `IR_SPEC` §13).
fn resolve_severity(params: &serde_json::Value) -> Result<AssertionSeverity> {
    let Some(value) = params.get("severity") else {
        return Ok(AssertionSeverity::Error);
    };
    match value.as_str() {
        Some("error") => Ok(AssertionSeverity::Error),
        Some("warning") => Ok(AssertionSeverity::Warning),
        Some("metric") => Ok(AssertionSeverity::Metric),
        other => Err(Error::new(
            ErrorClass::Schema,
            E_ASSERT_PARAM,
            "assertion `severity` must be `error`, `warning`, or `metric`".to_owned(),
        )
        .with_context(
            ErrorContext::default()
                .with_actual(other.unwrap_or("<non-string>").to_owned())
                .with_expected("error | warning | metric".to_owned()),
        )),
    }
}

/// Resolve a finite, non-negative `f64` parameter, defaulting to `0.0` when
/// absent. Used for `outside_threshold` and `coverage_epsilon`.
fn resolve_non_negative(params: &serde_json::Value, name: &str) -> Result<f64> {
    let Some(value) = params.get(name) else {
        return Ok(0.0);
    };
    let v = value.as_f64().ok_or_else(|| {
        Error::new(
            ErrorClass::Schema,
            E_ASSERT_PARAM,
            format!("assert.no_change_outside_mask `{name}` must be a finite non-negative number"),
        )
    })?;
    if !v.is_finite() || v < 0.0 {
        return Err(Error::new(
            ErrorClass::Schema,
            E_ASSERT_PARAM,
            format!("assert.no_change_outside_mask `{name}` must be a finite non-negative number"),
        )
        .with_context(ErrorContext::default().with_actual(v.to_string())));
    }
    Ok(v)
}

/// Map a sample to the comparison space: sRGB-decoded when `decode` is set.
#[must_use]
fn to_space(sample: f32, decode: bool) -> f32 {
    if decode { srgb_decode(sample) } else { sample }
}

/// Build a shape-mismatch [`semantic`](ErrorClass::Semantic) error.
fn shape_mismatch(op: &str, detail: &str, actual: String) -> Error {
    Error::new(
        ErrorClass::Semantic,
        E_ASSERT_SHAPE,
        format!("{op}: {detail}"),
    )
    .with_context(ErrorContext::default().with_actual(actual))
}

/// The `(x, y)` pixel coordinate of a row-major pixel index for a given width.
#[must_use]
fn pixel_coord(index: usize, width: u32) -> [i64; 2] {
    let stride = u64::from(width).max(1);
    let i = u64::try_from(index).unwrap_or(u64::MAX);
    let x = i64::try_from(i % stride).unwrap_or(i64::MAX);
    let y = i64::try_from(i / stride).unwrap_or(i64::MAX);
    [x, y]
}

/// Read a required image input *descriptor* port.
fn image_descriptor<'a>(
    inputs: &'a Descriptors,
    op: &str,
    port: &str,
) -> Result<&'a paintop_ir::ImageDescriptor> {
    let resource = inputs.get(port).ok_or_else(|| {
        Error::new(
            ErrorClass::Reference,
            E_ASSERT_INPUT,
            format!("{op} requires a `{port}` input"),
        )
    })?;
    let ResourceDescriptor::Image(descriptor) = resource else {
        return Err(Error::new(
            ErrorClass::Type,
            E_ASSERT_INPUT,
            format!("{op} `{port}` input must be an image resource"),
        ));
    };
    Ok(descriptor)
}

/// Read a required mask input *descriptor* port.
fn mask_descriptor<'a>(
    inputs: &'a Descriptors,
    op: &str,
    port: &str,
) -> Result<&'a MaskDescriptor> {
    let resource = inputs.get(port).ok_or_else(|| {
        Error::new(
            ErrorClass::Reference,
            E_ASSERT_INPUT,
            format!("{op} requires a `{port}` input"),
        )
    })?;
    let ResourceDescriptor::Mask(descriptor) = resource else {
        return Err(Error::new(
            ErrorClass::Type,
            E_ASSERT_INPUT,
            format!("{op} `{port}` input must be a mask resource"),
        ));
    };
    Ok(descriptor)
}

/// Read a required input *value* port.
fn input_value<'a>(
    inputs: &'a InputValues,
    op: &str,
    port: &str,
) -> std::result::Result<&'a ResourceValue, Error> {
    inputs.get(port).ok_or_else(|| {
        Error::new(
            ErrorClass::Reference,
            E_ASSERT_INPUT,
            format!("{op} requires a `{port}` input value"),
        )
    })
}

/// The `Report` descriptor for an assertion: it summarizes the (point-sized)
/// verdict, so it carries the asserted resource's extent and zero channels (a
/// report carries no raster).
const fn report_descriptor(extent: Extent) -> ReportDescriptor {
    ReportDescriptor {
        extent,
        channels: 0,
    }
}

/// The single `report` output descriptor map for an assertion op.
fn report_output(extent: Extent) -> OutputDescriptors {
    let mut out = OutputDescriptors::new();
    out.insert(
        "report".to_owned(),
        ResourceDescriptor::Report(report_descriptor(extent)),
    );
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

/// A `severity` parameter spec shared by both assertion ops.
fn severity_param() -> ParamSpec {
    ParamSpec {
        name: "severity".to_owned(),
        ty: ParamType::String,
        unit: None,
        required: false,
        default: Some(serde_json::Value::String("error".to_owned())),
        choices: vec![
            "error".to_owned(),
            "warning".to_owned(),
            "metric".to_owned(),
        ],
        doc: "How a violation affects the run: `error` fails the run, `warning` marks the \
              evidence, `metric` never fails."
            .to_owned(),
    }
}

/// The verification declarations shared by the assertion ops. Each is an exact,
/// single-reference op, so differential and perceptual do not apply (derived
/// not-applicable); every other category is covered by the analytic-fixture and
/// property tests in this module.
fn assert_test_metadata() -> TestMetadata {
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
// assert.no_change_outside_mask@1
// ---------------------------------------------------------------------------

/// The accumulated evidence of the no-change-outside-mask scan.
#[derive(Debug, Default)]
struct OutsideScan {
    /// The number of pixels that changed beyond the threshold outside the mask.
    changed_outside: u64,
    /// The maximum absolute outside delta seen (over every channel).
    max_outside: f64,
    /// The worst leaking pixel (largest outside delta), if any.
    worst: Option<[i64; 2]>,
    /// The worst delta backing `worst`, for the running maximum.
    worst_delta: f64,
    /// A capped list of leaking pixel locations.
    locations: Vec<[i64; 2]>,
}

/// The raster inputs of a no-change-outside-mask scan: the row-major,
/// channel-interleaved `before` / `after` sample buffers, the per-pixel coverage
/// `mask`, the interleaved channel count, and the row width.
struct ScanInputs<'a> {
    before: &'a [f32],
    after: &'a [f32],
    mask: &'a [f32],
    channels: usize,
    width: u32,
}

/// The decision parameters of a no-change-outside-mask scan: whether to
/// sRGB-decode, the leak threshold a pixel must strictly exceed, and the maximum
/// coverage at which a pixel still counts as *outside*.
struct ScanParams {
    decode: bool,
    outside_threshold: f64,
    coverage_epsilon: f64,
}

/// Evaluate the no-change-outside-mask predicate over the whole image, returning
/// the scan evidence. A pixel is *outside* when its mask coverage is at most
/// `coverage_epsilon`; it *leaks* when any channel's absolute delta (in the
/// comparison space) strictly exceeds `outside_threshold`.
fn scan_outside(inputs: &ScanInputs, params: &ScanParams) -> OutsideScan {
    let mut scan = OutsideScan::default();
    if inputs.channels == 0 {
        return scan;
    }
    for (pixel_index, (b_px, a_px)) in inputs
        .before
        .chunks_exact(inputs.channels)
        .zip(inputs.after.chunks_exact(inputs.channels))
        .enumerate()
    {
        // Outside = coverage at or below epsilon. A mask shorter than the pixel
        // count treats the missing tail as fully covered (inside), so it can
        // never spuriously flag a leak.
        let coverage = inputs.mask.get(pixel_index).copied().unwrap_or(1.0);
        if f64::from(coverage) > params.coverage_epsilon {
            continue;
        }
        let mut pixel_delta = 0.0_f64;
        for (&b, &a) in b_px.iter().zip(a_px.iter()) {
            let delta = f64::from((to_space(a, params.decode) - to_space(b, params.decode)).abs());
            if delta > pixel_delta {
                pixel_delta = delta;
            }
        }
        if pixel_delta > scan.max_outside {
            scan.max_outside = pixel_delta;
        }
        if pixel_delta > params.outside_threshold {
            scan.changed_outside += 1;
            if scan.locations.len() < MAX_LOCATIONS {
                scan.locations.push(pixel_coord(pixel_index, inputs.width));
            }
            if scan.worst.is_none() || pixel_delta > scan.worst_delta {
                scan.worst = Some(pixel_coord(pixel_index, inputs.width));
                scan.worst_delta = pixel_delta;
            }
        }
    }
    scan
}

/// The `assert.no_change_outside_mask@1` operation.
#[derive(Debug, Clone, Copy, Default)]
pub struct NoChangeOutsideMask;

impl NoChangeOutsideMask {
    /// Construct the operation.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    /// The declared manifest for `assert.no_change_outside_mask@1`.
    ///
    /// # Errors
    /// Propagates the [`schema`](ErrorClass::Schema) error if the hard-coded op /
    /// impl ids are somehow invalid (they are not).
    pub fn manifest() -> Result<OperationManifest> {
        Ok(OperationManifest {
            id: NO_CHANGE_OUTSIDE_MASK_OP_ID.parse()?,
            impl_version: 1,
            summary: "Assert that an edit changed nothing outside an allowed coverage mask: every \
                      pixel outside the mask must change by no more than outside_threshold in the \
                      chosen comparison space. A leak fails the assertion and records the worst \
                      leaking pixel, the leaking-pixel count, and the maximum outside delta."
                .to_owned(),
            // Per-channel absolute delta plus fixed-order reductions: a
            // deterministic function of the inputs and parameters.
            determinism: DeterminismTier::Exact,
            roi: RoiPolicy {
                // The verdict reduces over every pixel.
                category: RoiCategory::FullDomain,
                halo_px: None,
            },
            inputs: vec![
                InputSpec {
                    name: "before".to_owned(),
                    kind: ResourceKind::Image,
                    required: true,
                    doc: "The reference image the edit is measured against.".to_owned(),
                },
                InputSpec {
                    name: "after".to_owned(),
                    kind: ResourceKind::Image,
                    required: true,
                    doc: "The edited image whose change from `before` is checked.".to_owned(),
                },
                InputSpec {
                    name: "allowed".to_owned(),
                    kind: ResourceKind::Mask,
                    required: true,
                    doc: "The coverage mask of the region the edit is authorized to change."
                        .to_owned(),
                },
            ],
            outputs: vec![OutputSpec {
                name: "report".to_owned(),
                kind: ResourceKind::Report,
                doc: "The assertion verdict: pass/fail plus the outside metrics (max delta, \
                      leaking-pixel count, worst pixel, locations)."
                    .to_owned(),
            }],
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
                    name: "outside_threshold".to_owned(),
                    ty: ParamType::Float,
                    unit: Some(paintop_ir::ParamUnit::Ratio),
                    required: false,
                    default: Some(serde_json::json!(0.0)),
                    choices: vec![],
                    doc: "The absolute delta a pixel outside the mask must strictly exceed to \
                          count as a leak."
                        .to_owned(),
                },
                ParamSpec {
                    name: "coverage_epsilon".to_owned(),
                    ty: ParamType::Float,
                    unit: Some(paintop_ir::ParamUnit::Ratio),
                    required: false,
                    default: Some(serde_json::json!(0.0)),
                    choices: vec![],
                    doc: "The maximum mask coverage at which a pixel still counts as `outside` \
                          (defaults to 0: only fully-uncovered pixels are outside)."
                        .to_owned(),
                },
                severity_param(),
            ],
            implementations: vec![reference_impl()?],
            test: assert_test_metadata(),
        })
    }
}

impl OpContract for NoChangeOutsideMask {
    fn declared_inputs(&self) -> Vec<(String, ResourceKind)> {
        vec![
            ("before".to_owned(), ResourceKind::Image),
            ("after".to_owned(), ResourceKind::Image),
            ("allowed".to_owned(), ResourceKind::Mask),
        ]
    }

    fn declared_outputs(&self) -> Vec<(String, ResourceKind)> {
        vec![("report".to_owned(), ResourceKind::Report)]
    }

    fn infer_outputs(
        &self,
        inputs: &Descriptors,
        params: &serde_json::Value,
    ) -> Result<OutputDescriptors> {
        // Resolve params so a malformed token is a graph-time error.
        let _ = ComparisonSpace::resolve(params)?;
        let _ = resolve_severity(params)?;
        let _ = resolve_non_negative(params, "outside_threshold")?;
        let _ = resolve_non_negative(params, "coverage_epsilon")?;

        let op = NO_CHANGE_OUTSIDE_MASK_OP_ID;
        let before = image_descriptor(inputs, op, "before")?;
        let after = image_descriptor(inputs, op, "after")?;
        let allowed = mask_descriptor(inputs, op, "allowed")?;

        if before.extent != after.extent {
            return Err(shape_mismatch(
                op,
                "the `before` and `after` images must share an extent",
                format!("before {:?} vs after {:?}", before.extent, after.extent),
            ));
        }
        if before.layout != after.layout {
            return Err(shape_mismatch(
                op,
                "the `before` and `after` images must share a channel layout",
                format!("before {:?} vs after {:?}", before.layout, after.layout),
            ));
        }
        if allowed.extent != before.extent {
            return Err(shape_mismatch(
                op,
                "the `allowed` mask must share the images' extent",
                format!("mask {:?} vs image {:?}", allowed.extent, before.extent),
            ));
        }
        Ok(report_output(before.extent))
    }

    fn required_inputs(
        &self,
        _requested_outputs: &OutputRegions,
        inputs: &Descriptors,
        _params: &serde_json::Value,
    ) -> Result<InputRegions> {
        Ok(full_domain_regions(inputs, &["before", "after", "allowed"]))
    }

    fn validate_postconditions(
        &self,
        outputs: &OutputDescriptors,
        _params: &serde_json::Value,
    ) -> Result<Vec<AssertionResult>> {
        Ok(produces_report(outputs))
    }
}

impl OpImplementation for NoChangeOutsideMask {
    fn compute(
        &self,
        inputs: &InputValues,
        params: &serde_json::Value,
    ) -> std::result::Result<OutputValues, Error> {
        let op = NO_CHANGE_OUTSIDE_MASK_OP_ID;
        let space = ComparisonSpace::resolve(params)?;
        let severity = resolve_severity(params)?;
        let outside_threshold = resolve_non_negative(params, "outside_threshold")?;
        let coverage_epsilon = resolve_non_negative(params, "coverage_epsilon")?;

        let before = input_value(inputs, op, "before")?;
        let after = input_value(inputs, op, "after")?;
        let allowed = input_value(inputs, op, "allowed")?;

        let ResourceDescriptor::Image(before_desc) = before.descriptor() else {
            return Err(input_type_error(op, "before", "image"));
        };
        let ResourceDescriptor::Image(after_desc) = after.descriptor() else {
            return Err(input_type_error(op, "after", "image"));
        };
        if before_desc.extent != after_desc.extent || before.channels() != after.channels() {
            return Err(shape_mismatch(
                op,
                "the `before` and `after` images must share an extent and channel count",
                format!(
                    "before {:?}x{} vs after {:?}x{}",
                    before_desc.extent,
                    before.channels(),
                    after_desc.extent,
                    after.channels(),
                ),
            ));
        }
        if !matches!(allowed.descriptor(), ResourceDescriptor::Mask(_)) {
            return Err(input_type_error(op, "allowed", "mask"));
        }
        if allowed.extent() != before_desc.extent {
            return Err(shape_mismatch(
                op,
                "the `allowed` mask must share the images' extent",
                format!(
                    "mask {:?} vs image {:?}",
                    allowed.extent(),
                    before_desc.extent
                ),
            ));
        }

        // Decoding is symmetric for the pair: both samples are mapped the same
        // way, so an sRGB pair compared in decoded-linear measures a light delta.
        let decode = matches!(space, ComparisonSpace::DecodedLinear)
            && matches!(before_desc.color, paintop_ir::ColorEncoding::Srgb);
        let channels = before.channels() as usize;
        let scan = scan_outside(
            &ScanInputs {
                before: before.samples(),
                after: after.samples(),
                mask: allowed.samples(),
                channels,
                width: before_desc.extent.width,
            },
            &ScanParams {
                decode,
                outside_threshold,
                coverage_epsilon,
            },
        );

        let passed = scan.changed_outside == 0;
        let outcome = AssertionOutcome {
            assertion: op.to_owned(),
            passed,
            severity,
            max_abs_delta_outside: Some(scan.max_outside),
            changed_pixels_outside: Some(scan.changed_outside),
            nonfinite_count: None,
            worst_pixel: scan.worst,
            locations: scan.locations,
            violations: None,
            worst_value: None,
            changed_bounds: None,
            expected_bounds: None,
        };
        let report = assertion_report(before_desc.extent, outcome);
        Ok(single_report(report))
    }
}

// ---------------------------------------------------------------------------
// assert.finite@1
// ---------------------------------------------------------------------------

/// The `assert.finite@1` operation.
#[derive(Debug, Clone, Copy, Default)]
pub struct Finite;

impl Finite {
    /// Construct the operation.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    /// The declared manifest for `assert.finite@1`.
    ///
    /// # Errors
    /// Propagates the [`schema`](ErrorClass::Schema) error if the hard-coded op /
    /// impl ids are somehow invalid (they are not).
    pub fn manifest() -> Result<OperationManifest> {
        Ok(OperationManifest {
            id: FINITE_OP_ID.parse()?,
            impl_version: 1,
            summary: "Assert that every sample of a resource is finite. Any NaN/Inf fails the \
                      assertion and records the non-finite count, the worst (first) non-finite \
                      pixel, and a capped list of non-finite pixel locations."
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
                doc: "The resource whose samples must all be finite.".to_owned(),
            }],
            outputs: vec![OutputSpec {
                name: "report".to_owned(),
                kind: ResourceKind::Report,
                doc: "The assertion verdict: pass/fail plus the non-finite count, worst pixel, \
                      and locations."
                    .to_owned(),
            }],
            params: vec![severity_param()],
            implementations: vec![reference_impl()?],
            test: assert_test_metadata(),
        })
    }
}

impl OpContract for Finite {
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
        let _ = resolve_severity(params)?;
        let input = inputs.get("resource").ok_or_else(|| {
            Error::new(
                ErrorClass::Reference,
                E_ASSERT_INPUT,
                "assert.finite requires a `resource` input".to_owned(),
            )
        })?;
        Ok(report_output(input.extent()))
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

impl OpImplementation for Finite {
    fn compute(
        &self,
        inputs: &InputValues,
        params: &serde_json::Value,
    ) -> std::result::Result<OutputValues, Error> {
        let op = FINITE_OP_ID;
        let severity = resolve_severity(params)?;
        let resource = input_value(inputs, op, "resource")?;

        let extent = resource.extent();
        let channels = resource.channels() as usize;
        let width = extent.width;

        let mut nonfinite: u64 = 0;
        let mut worst: Option<[i64; 2]> = None;
        let mut locations: Vec<[i64; 2]> = Vec::new();

        if channels != 0 {
            for (pixel_index, pixel) in resource.samples().chunks_exact(channels).enumerate() {
                if pixel.iter().any(|s| !s.is_finite()) {
                    nonfinite += 1;
                    let coord = pixel_coord(pixel_index, width);
                    if worst.is_none() {
                        worst = Some(coord);
                    }
                    if locations.len() < MAX_LOCATIONS {
                        locations.push(coord);
                    }
                }
            }
        }

        let passed = nonfinite == 0;
        let outcome = AssertionOutcome {
            assertion: op.to_owned(),
            passed,
            severity,
            max_abs_delta_outside: None,
            changed_pixels_outside: None,
            nonfinite_count: Some(nonfinite),
            worst_pixel: worst,
            locations,
            violations: None,
            worst_value: None,
            changed_bounds: None,
            expected_bounds: None,
        };
        Ok(single_report(assertion_report(extent, outcome)))
    }
}

// ---------------------------------------------------------------------------
// Shared output helpers
// ---------------------------------------------------------------------------

/// Build the assertion [`Report`] carrying `outcome`. The report's channel
/// statistics are empty (an assertion summarizes a verdict, not a raster) and its
/// `all_finite` mirrors the verdict for a finiteness assertion, else `true`.
#[must_use]
fn assertion_report(extent: Extent, outcome: AssertionOutcome) -> Report {
    // For a finiteness assertion, surface the all-finite flag on the report too,
    // matching the inspection-report convention; for the locality assertion no
    // finiteness claim is made, so it stays `true`.
    let all_finite = outcome.nonfinite_count.is_none_or(|n| n == 0);
    Report {
        extent,
        channels: 0,
        channel_stats: Vec::new(),
        all_finite,
        content_hash: String::new(),
        diff: None,
        assertion: Some(outcome),
        histogram: None,
        components: None,
        frequency_energy: None,
        solver: None,
    }
}

/// Wrap a report as the single `report` output of an assertion op.
fn single_report(report: Report) -> OutputValues {
    let mut out = OutputValues::new();
    out.insert("report".to_owned(), ResourceValue::report(report));
    out
}

/// The wrong-resource-kind error for an input port.
fn input_type_error(op: &str, port: &str, kind: &str) -> Error {
    Error::new(
        ErrorClass::Type,
        E_ASSERT_INPUT,
        format!("{op} `{port}` input must be a {kind} resource"),
    )
}

#[cfg(test)]
mod tests;
