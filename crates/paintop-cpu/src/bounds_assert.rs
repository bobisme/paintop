//! The changed-bounds analysis op and the remaining P0 assertions
//! (`OP_CATALOG` §12, `IR_SPEC` §13, `AGENT_VERIFICATION` §5.3).
//!
//! `analyze.changed_bounds@1` reduces a difference field and a threshold to the
//! tight bounding box of the *changed* region (the pixels whose magnitude
//! exceeds the threshold), reporting empty bounds on an identity (all-quiet)
//! diff. The three assertions are ordinary typed nodes whose verdict maps a
//! failure to exit class 6 through the bundle's severity handling:
//!
//! - `assert.range@1`: every sample of a field/image lies within `[min, max]`;
//!   a failure records the out-of-range count, the worst (furthest-out) value,
//!   and its pixel.
//! - `assert.alpha_valid@1`: an RGBA image's alpha lies in `[0, 1]` and (for a
//!   premultiplied image) every color channel satisfies `|C| <= α`; a failure
//!   records the invalid-pixel count, the worst constraint excess, and its
//!   pixel.
//! - `assert.changed_bounds@1`: the region a `before -> after` edit changed
//!   (beyond a threshold) is contained in an expected box; a failure records
//!   the actual changed bounds, the expected box, and the count/worst of the
//!   escaping pixels.
//!
//! Every op is *pure* and [`Exact`](DeterminismTier::Exact): each reduction runs
//! in a fixed row-major order, so the verdict is a deterministic function of the
//! inputs and parameters.

use paintop_core::executor::{InputValues, OpImplementation, OutputValues, ResourceValue};
use paintop_ir::{
    AssertionOutcome, AssertionResult, AssertionSeverity, ColorEncoding, Descriptors,
    DeterminismTier, Error, ErrorClass, ErrorContext, Extent, ImageDescriptor, ImplId,
    InputRegions, InputSpec, OpContract, OperationManifest, OutputDescriptors, OutputRegions,
    OutputSpec, ParamSpec, ParamType, ParamUnit, Rect, Report, ReportDescriptor,
    ResourceDescriptor, ResourceKind, Result, RoiCategory, RoiPolicy, TestMetadata,
};

/// The canonical id of the changed-bounds analysis op.
pub const CHANGED_BOUNDS_OP_ID: &str = "analyze.changed_bounds@1";

/// The canonical id of the range assertion.
pub const RANGE_OP_ID: &str = "assert.range@1";

/// The canonical id of the alpha-validity assertion.
pub const ALPHA_VALID_OP_ID: &str = "assert.alpha_valid@1";

/// The canonical id of the changed-bounds assertion.
pub const ASSERT_CHANGED_BOUNDS_OP_ID: &str = "assert.changed_bounds@1";

/// A required input port was absent or carried the wrong resource kind.
pub const E_BOUNDS_INPUT: &str = "E_BOUNDS_INPUT";

/// The inputs disagree on extent or channel layout.
pub const E_BOUNDS_SHAPE: &str = "E_BOUNDS_SHAPE";

/// A parameter was missing, the wrong type, or out of range.
pub const E_BOUNDS_PARAM: &str = "E_BOUNDS_PARAM";

/// The maximum number of offending-pixel locations recorded in an assertion's
/// `locations` list (the full count is always reported separately).
const MAX_LOCATIONS: usize = 256;

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

/// The `(x, y)` pixel coordinate of a row-major pixel index for a given width.
#[must_use]
fn pixel_coord(index: usize, width: u32) -> [i64; 2] {
    let stride = u64::from(width).max(1);
    let i = u64::try_from(index).unwrap_or(u64::MAX);
    let x = i64::try_from(i % stride).unwrap_or(i64::MAX);
    let y = i64::try_from(i / stride).unwrap_or(i64::MAX);
    [x, y]
}

/// The single-pixel cell rect of a row-major pixel index for a given width.
#[must_use]
fn pixel_cell(index: usize, width: u32) -> Rect {
    let [x, y] = pixel_coord(index, width);
    Rect::new(x, y, x + 1, y + 1)
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
            E_BOUNDS_INPUT,
            format!("{op} requires a `{port}` input value"),
        )
    })
}

/// Read a required image input *descriptor* port.
fn image_descriptor<'a>(
    inputs: &'a Descriptors,
    op: &str,
    port: &str,
) -> Result<&'a ImageDescriptor> {
    let resource = inputs.get(port).ok_or_else(|| {
        Error::new(
            ErrorClass::Reference,
            E_BOUNDS_INPUT,
            format!("{op} requires a `{port}` input"),
        )
    })?;
    let ResourceDescriptor::Image(descriptor) = resource else {
        return Err(Error::new(
            ErrorClass::Type,
            E_BOUNDS_INPUT,
            format!("{op} `{port}` input must be an image resource"),
        ));
    };
    Ok(descriptor)
}

/// Build a shape-mismatch [`semantic`](ErrorClass::Semantic) error.
fn shape_mismatch(op: &str, detail: &str, actual: String) -> Error {
    Error::new(
        ErrorClass::Semantic,
        E_BOUNDS_SHAPE,
        format!("{op}: {detail}"),
    )
    .with_context(ErrorContext::default().with_actual(actual))
}

/// The wrong-resource-kind error for an input port.
fn input_type_error(op: &str, port: &str, kind: &str) -> Error {
    Error::new(
        ErrorClass::Type,
        E_BOUNDS_INPUT,
        format!("{op} `{port}` input must be a {kind} resource"),
    )
}

/// The `Report` descriptor for a point-sized report carrying `extent`.
const fn report_descriptor(extent: Extent) -> ReportDescriptor {
    ReportDescriptor {
        extent,
        channels: 0,
    }
}

/// The single `report` output descriptor map for a report-producing op.
fn report_output(extent: Extent) -> OutputDescriptors {
    let mut out = OutputDescriptors::new();
    out.insert(
        "report".to_owned(),
        ResourceDescriptor::Report(report_descriptor(extent)),
    );
    out
}

/// Wrap a report as the single `report` output.
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

/// A `severity` parameter spec shared by the assertion ops.
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

/// A `threshold` parameter spec (the magnitude a sample must strictly exceed to
/// count as changed), defaulting to `0.0`.
fn threshold_param(doc: &str) -> ParamSpec {
    ParamSpec {
        name: "threshold".to_owned(),
        ty: ParamType::Float,
        unit: Some(ParamUnit::Ratio),
        required: false,
        default: Some(serde_json::json!(0.0)),
        choices: vec![],
        doc: doc.to_owned(),
    }
}

/// Resolve the `severity` token, defaulting to `error`.
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
            E_BOUNDS_PARAM,
            "assertion `severity` must be `error`, `warning`, or `metric`".to_owned(),
        )
        .with_context(
            ErrorContext::default()
                .with_actual(other.unwrap_or("<non-string>").to_owned())
                .with_expected("error | warning | metric".to_owned()),
        )),
    }
}

/// Resolve a finite, non-negative `threshold`, defaulting to `0.0`.
fn resolve_threshold(op: &str, params: &serde_json::Value) -> Result<f64> {
    let Some(value) = params.get("threshold") else {
        return Ok(0.0);
    };
    let v = value.as_f64().ok_or_else(|| {
        Error::new(
            ErrorClass::Schema,
            E_BOUNDS_PARAM,
            format!("{op} `threshold` must be a finite non-negative number"),
        )
    })?;
    if v.is_finite() && v >= 0.0 {
        Ok(v)
    } else {
        Err(Error::new(
            ErrorClass::Schema,
            E_BOUNDS_PARAM,
            format!("{op} `threshold` must be a finite non-negative number"),
        )
        .with_context(ErrorContext::default().with_actual(v.to_string())))
    }
}

/// Resolve a required finite floating-point parameter.
fn require_finite(op: &str, params: &serde_json::Value, name: &str) -> Result<f64> {
    let value = params.get(name).ok_or_else(|| {
        Error::new(
            ErrorClass::Schema,
            E_BOUNDS_PARAM,
            format!("{op} requires a `{name}` parameter"),
        )
    })?;
    let v = value.as_f64().ok_or_else(|| {
        Error::new(
            ErrorClass::Schema,
            E_BOUNDS_PARAM,
            format!("{op} `{name}` must be a finite number"),
        )
    })?;
    if v.is_finite() {
        Ok(v)
    } else {
        Err(Error::new(
            ErrorClass::Schema,
            E_BOUNDS_PARAM,
            format!("{op} `{name}` must be a finite number"),
        )
        .with_context(ErrorContext::default().with_actual(v.to_string())))
    }
}

/// Resolve a required integer parameter.
fn require_i64(op: &str, params: &serde_json::Value, name: &str) -> Result<i64> {
    let value = params.get(name).ok_or_else(|| {
        Error::new(
            ErrorClass::Schema,
            E_BOUNDS_PARAM,
            format!("{op} requires a `{name}` parameter"),
        )
    })?;
    value.as_i64().ok_or_else(|| {
        Error::new(
            ErrorClass::Schema,
            E_BOUNDS_PARAM,
            format!("{op} `{name}` must be an integer"),
        )
        .with_context(ErrorContext::default().with_actual(value.to_string()))
    })
}

/// Build a report carrying an assertion verdict (no raster, empty stats).
#[must_use]
const fn assertion_report(extent: Extent, outcome: AssertionOutcome) -> Report {
    Report {
        extent,
        channels: 0,
        channel_stats: Vec::new(),
        all_finite: true,
        content_hash: String::new(),
        diff: None,
        assertion: Some(outcome),
        histogram: None,
        components: None,
    }
}

/// The verification declarations shared by these ops: exact, single-reference
/// reductions, so differential and perceptual do not apply (derived
/// not-applicable); every other category is covered by this module's tests.
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

/// Scan a per-pixel magnitude field for the tight bounds of the *changed*
/// pixels (any channel whose finite magnitude strictly exceeds `threshold`),
/// returning the bounds (empty when none changed) and the changed-pixel count.
fn changed_region(
    samples: &[f32],
    channels: usize,
    width: u32,
    threshold: f64,
) -> (Option<Rect>, u64) {
    let mut bounds: Option<Rect> = None;
    let mut count: u64 = 0;
    if channels == 0 {
        return (None, 0);
    }
    for (pixel_index, pixel) in samples.chunks_exact(channels).enumerate() {
        let changed = pixel
            .iter()
            .any(|&s| s.is_finite() && f64::from(s.abs()) > threshold);
        if changed {
            count += 1;
            let cell = pixel_cell(pixel_index, width);
            bounds = Some(bounds.map_or(cell, |b| b.union(cell)));
        }
    }
    (bounds, count)
}

// ---------------------------------------------------------------------------
// analyze.changed_bounds@1
// ---------------------------------------------------------------------------

/// The `analyze.changed_bounds@1` operation.
#[derive(Debug, Clone, Copy, Default)]
pub struct ChangedBounds;

impl ChangedBounds {
    /// Construct the operation.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    /// The declared manifest for `analyze.changed_bounds@1`.
    ///
    /// # Errors
    /// Propagates the [`schema`](ErrorClass::Schema) error if the hard-coded op /
    /// impl ids are somehow invalid (they are not).
    pub fn manifest() -> Result<OperationManifest> {
        Ok(OperationManifest {
            id: CHANGED_BOUNDS_OP_ID.parse()?,
            impl_version: 1,
            summary: "Reduce a difference field and a threshold to the tight bounding box of the \
                      changed region (pixels whose magnitude exceeds the threshold) and the \
                      changed-pixel count. An all-quiet diff reports empty bounds."
                .to_owned(),
            determinism: DeterminismTier::Exact,
            roi: RoiPolicy {
                category: RoiCategory::FullDomain,
                halo_px: None,
            },
            inputs: vec![InputSpec {
                name: "diff".to_owned(),
                kind: ResourceKind::Image,
                required: true,
                doc: "The per-pixel difference magnitude field (e.g. analyze.diff's `diff`)."
                    .to_owned(),
            }],
            outputs: vec![OutputSpec {
                name: "report".to_owned(),
                kind: ResourceKind::Report,
                doc: "The bounds report: the changed-pixel count and the tight changed bounds \
                      (empty when nothing changed)."
                    .to_owned(),
            }],
            params: vec![threshold_param(
                "The magnitude a pixel must strictly exceed to count as changed.",
            )],
            implementations: vec![reference_impl()?],
            test: bounds_test_metadata(),
        })
    }
}

impl OpContract for ChangedBounds {
    fn declared_inputs(&self) -> Vec<(String, ResourceKind)> {
        vec![("diff".to_owned(), ResourceKind::Image)]
    }

    fn declared_outputs(&self) -> Vec<(String, ResourceKind)> {
        vec![("report".to_owned(), ResourceKind::Report)]
    }

    fn infer_outputs(
        &self,
        inputs: &Descriptors,
        params: &serde_json::Value,
    ) -> Result<OutputDescriptors> {
        let _ = resolve_threshold(CHANGED_BOUNDS_OP_ID, params)?;
        let diff = image_descriptor(inputs, CHANGED_BOUNDS_OP_ID, "diff")?;
        Ok(report_output(diff.extent))
    }

    fn required_inputs(
        &self,
        _requested_outputs: &OutputRegions,
        inputs: &Descriptors,
        _params: &serde_json::Value,
    ) -> Result<InputRegions> {
        Ok(full_domain_regions(inputs, &["diff"]))
    }

    fn validate_postconditions(
        &self,
        outputs: &OutputDescriptors,
        _params: &serde_json::Value,
    ) -> Result<Vec<AssertionResult>> {
        Ok(produces_report(outputs))
    }
}

impl OpImplementation for ChangedBounds {
    fn compute(
        &self,
        inputs: &InputValues,
        params: &serde_json::Value,
    ) -> std::result::Result<OutputValues, Error> {
        let op = CHANGED_BOUNDS_OP_ID;
        let threshold = resolve_threshold(op, params)?;
        let diff = require_value(inputs, op, "diff")?;
        if !matches!(diff.descriptor(), ResourceDescriptor::Image(_)) {
            return Err(input_type_error(op, "diff", "image"));
        }
        let extent = diff.extent();
        let channels = diff.channels() as usize;
        let (bounds, count) = changed_region(diff.samples(), channels, extent.width, threshold);

        // changed_bounds is not an assertion; it reports the bounds as a diff-less
        // verdict-less report. We carry the bounds through the assertion outcome
        // block's `changed_bounds` field with `passed = true` so the bundle and
        // downstream ops read a single, uniform shape.
        let mut outcome = AssertionOutcome::new(op, true, AssertionSeverity::Metric);
        outcome.changed_bounds = bounds;
        outcome.violations = Some(count);
        Ok(single_report(assertion_report(extent, outcome)))
    }
}

// ---------------------------------------------------------------------------
// assert.range@1
// ---------------------------------------------------------------------------

/// The `assert.range@1` operation.
#[derive(Debug, Clone, Copy, Default)]
pub struct AssertRange;

impl AssertRange {
    /// Construct the operation.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    /// The declared manifest for `assert.range@1`.
    ///
    /// # Errors
    /// Propagates the [`schema`](ErrorClass::Schema) error if the hard-coded op /
    /// impl ids are somehow invalid (they are not).
    pub fn manifest() -> Result<OperationManifest> {
        Ok(OperationManifest {
            id: RANGE_OP_ID.parse()?,
            impl_version: 1,
            summary: "Assert that every finite sample of a field/image lies within [min, max]. A \
                      violation fails the assertion and records the out-of-range count, the worst \
                      (furthest-out) value, and its pixel."
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
                doc: "The field/image whose samples must lie within [min, max].".to_owned(),
            }],
            outputs: vec![OutputSpec {
                name: "report".to_owned(),
                kind: ResourceKind::Report,
                doc: "The assertion verdict: pass/fail plus the out-of-range count, worst value, \
                      and worst pixel."
                    .to_owned(),
            }],
            params: vec![
                ParamSpec {
                    name: "min".to_owned(),
                    ty: ParamType::Float,
                    unit: None,
                    required: true,
                    default: None,
                    choices: vec![],
                    doc: "The inclusive lower bound every sample must meet.".to_owned(),
                },
                ParamSpec {
                    name: "max".to_owned(),
                    ty: ParamType::Float,
                    unit: None,
                    required: true,
                    default: None,
                    choices: vec![],
                    doc: "The inclusive upper bound every sample must meet (>= min).".to_owned(),
                },
                severity_param(),
            ],
            implementations: vec![reference_impl()?],
            test: bounds_test_metadata(),
        })
    }
}

/// Resolve and validate the `[min, max]` range parameters.
fn resolve_range(params: &serde_json::Value) -> Result<(f64, f64)> {
    let min = require_finite(RANGE_OP_ID, params, "min")?;
    let max = require_finite(RANGE_OP_ID, params, "max")?;
    if max < min {
        return Err(Error::new(
            ErrorClass::Schema,
            E_BOUNDS_PARAM,
            "assert.range requires `max` >= `min`".to_owned(),
        )
        .with_context(ErrorContext::default().with_actual(format!("min {min}, max {max}"))));
    }
    Ok((min, max))
}

impl OpContract for AssertRange {
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
        let _ = resolve_range(params)?;
        let resource = inputs.get("resource").ok_or_else(|| {
            Error::new(
                ErrorClass::Reference,
                E_BOUNDS_INPUT,
                "assert.range requires a `resource` input".to_owned(),
            )
        })?;
        Ok(report_output(resource.extent()))
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

impl OpImplementation for AssertRange {
    fn compute(
        &self,
        inputs: &InputValues,
        params: &serde_json::Value,
    ) -> std::result::Result<OutputValues, Error> {
        let op = RANGE_OP_ID;
        let severity = resolve_severity(params)?;
        let (min, max) = resolve_range(params)?;
        let resource = require_value(inputs, op, "resource")?;
        let extent = resource.extent();
        let channels = resource.channels() as usize;
        let width = extent.width;

        let mut violations: u64 = 0;
        let mut worst_value: Option<f64> = None;
        let mut worst_excess = 0.0_f64;
        let mut worst_pixel: Option<[i64; 2]> = None;
        let mut locations: Vec<[i64; 2]> = Vec::new();

        if channels != 0 {
            for (pixel_index, pixel) in resource.samples().chunks_exact(channels).enumerate() {
                let mut pixel_bad = false;
                for &sample in pixel {
                    // A non-finite sample is out of any finite range.
                    let value = f64::from(sample);
                    let excess = if !sample.is_finite() {
                        f64::INFINITY
                    } else if value < min {
                        min - value
                    } else if value > max {
                        value - max
                    } else {
                        continue;
                    };
                    pixel_bad = true;
                    if worst_pixel.is_none() || excess > worst_excess {
                        worst_excess = excess;
                        worst_value = Some(value);
                        worst_pixel = Some(pixel_coord(pixel_index, width));
                    }
                }
                if pixel_bad {
                    violations += 1;
                    if locations.len() < MAX_LOCATIONS {
                        locations.push(pixel_coord(pixel_index, width));
                    }
                }
            }
        }

        let passed = violations == 0;
        let mut outcome = AssertionOutcome::new(op, passed, severity);
        outcome.violations = Some(violations);
        outcome.worst_value = worst_value;
        outcome.worst_pixel = worst_pixel;
        outcome.locations = locations;
        Ok(single_report(assertion_report(extent, outcome)))
    }
}

// ---------------------------------------------------------------------------
// assert.alpha_valid@1
// ---------------------------------------------------------------------------

/// The `assert.alpha_valid@1` operation.
#[derive(Debug, Clone, Copy, Default)]
pub struct AssertAlphaValid;

impl AssertAlphaValid {
    /// Construct the operation.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    /// The declared manifest for `assert.alpha_valid@1`.
    ///
    /// # Errors
    /// Propagates the [`schema`](ErrorClass::Schema) error if the hard-coded op /
    /// impl ids are somehow invalid (they are not).
    pub fn manifest() -> Result<OperationManifest> {
        Ok(OperationManifest {
            id: ALPHA_VALID_OP_ID.parse()?,
            impl_version: 1,
            summary: "Assert that an RGBA image's alpha lies in [0, 1] and (for a premultiplied \
                      image) every color channel satisfies |C| <= alpha. A violation fails the \
                      assertion and records the invalid-pixel count, worst constraint excess, and \
                      its pixel."
                .to_owned(),
            determinism: DeterminismTier::Exact,
            roi: RoiPolicy {
                category: RoiCategory::FullDomain,
                halo_px: None,
            },
            inputs: vec![InputSpec {
                name: "image".to_owned(),
                kind: ResourceKind::Image,
                required: true,
                doc: "The RGBA image whose alpha and premultiplied-color constraint are checked."
                    .to_owned(),
            }],
            outputs: vec![OutputSpec {
                name: "report".to_owned(),
                kind: ResourceKind::Report,
                doc: "The assertion verdict: pass/fail plus the invalid-pixel count, worst \
                      constraint excess, and worst pixel."
                    .to_owned(),
            }],
            params: vec![severity_param()],
            implementations: vec![reference_impl()?],
            test: bounds_test_metadata(),
        })
    }
}

impl OpContract for AssertAlphaValid {
    fn declared_inputs(&self) -> Vec<(String, ResourceKind)> {
        vec![("image".to_owned(), ResourceKind::Image)]
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
        let image = image_descriptor(inputs, ALPHA_VALID_OP_ID, "image")?;
        if !image.layout.has_alpha() {
            return Err(Error::new(
                ErrorClass::Semantic,
                E_BOUNDS_SHAPE,
                "assert.alpha_valid requires an image with an alpha channel".to_owned(),
            ));
        }
        Ok(report_output(image.extent))
    }

    fn required_inputs(
        &self,
        _requested_outputs: &OutputRegions,
        inputs: &Descriptors,
        _params: &serde_json::Value,
    ) -> Result<InputRegions> {
        Ok(full_domain_regions(inputs, &["image"]))
    }

    fn validate_postconditions(
        &self,
        outputs: &OutputDescriptors,
        _params: &serde_json::Value,
    ) -> Result<Vec<AssertionResult>> {
        Ok(produces_report(outputs))
    }
}

impl OpImplementation for AssertAlphaValid {
    fn compute(
        &self,
        inputs: &InputValues,
        params: &serde_json::Value,
    ) -> std::result::Result<OutputValues, Error> {
        let op = ALPHA_VALID_OP_ID;
        let severity = resolve_severity(params)?;
        let image = require_value(inputs, op, "image")?;
        let ResourceDescriptor::Image(desc) = image.descriptor() else {
            return Err(input_type_error(op, "image", "image"));
        };
        if !desc.layout.has_alpha() {
            return Err(Error::new(
                ErrorClass::Semantic,
                E_BOUNDS_SHAPE,
                "assert.alpha_valid requires an image with an alpha channel".to_owned(),
            ));
        }
        let extent = desc.extent;
        let premultiplied = matches!(desc.alpha, paintop_ir::AlphaRepresentation::Premultiplied);
        let channels = image.channels() as usize;
        // The alpha channel is the last interleaved channel of an alpha layout.
        let alpha_index = channels.saturating_sub(1);
        let width = extent.width;

        let mut violations: u64 = 0;
        let mut worst_value: Option<f64> = None;
        let mut worst_excess = 0.0_f64;
        let mut worst_pixel: Option<[i64; 2]> = None;
        let mut locations: Vec<[i64; 2]> = Vec::new();

        if channels != 0 {
            for (pixel_index, pixel) in image.samples().chunks_exact(channels).enumerate() {
                let alpha = pixel[alpha_index];
                let mut pixel_excess: Option<f64> = None;
                // Alpha must be a finite value in [0, 1].
                let alpha_excess = if !alpha.is_finite() {
                    Some(f64::INFINITY)
                } else if alpha < 0.0 {
                    Some(f64::from(-alpha))
                } else if alpha > 1.0 {
                    Some(f64::from(alpha) - 1.0)
                } else {
                    None
                };
                if let Some(e) = alpha_excess {
                    pixel_excess = Some(pixel_excess.map_or(e, |p: f64| p.max(e)));
                }
                // For a premultiplied image, each color channel must satisfy
                // |C| <= alpha (a finite alpha; a non-finite alpha already failed).
                if premultiplied && alpha.is_finite() {
                    for &c in &pixel[..alpha_index] {
                        let excess = if c.is_finite() {
                            f64::from(c.abs() - alpha).max(0.0)
                        } else {
                            f64::INFINITY
                        };
                        if excess > 0.0 {
                            pixel_excess = Some(pixel_excess.map_or(excess, |p| p.max(excess)));
                        }
                    }
                }
                if let Some(excess) = pixel_excess {
                    violations += 1;
                    if locations.len() < MAX_LOCATIONS {
                        locations.push(pixel_coord(pixel_index, width));
                    }
                    if worst_pixel.is_none() || excess > worst_excess {
                        worst_excess = excess;
                        worst_value = Some(excess);
                        worst_pixel = Some(pixel_coord(pixel_index, width));
                    }
                }
            }
        }

        let passed = violations == 0;
        let mut outcome = AssertionOutcome::new(op, passed, severity);
        outcome.violations = Some(violations);
        outcome.worst_value = worst_value;
        outcome.worst_pixel = worst_pixel;
        outcome.locations = locations;
        Ok(single_report(assertion_report(extent, outcome)))
    }
}

// ---------------------------------------------------------------------------
// assert.changed_bounds@1
// ---------------------------------------------------------------------------

/// The `assert.changed_bounds@1` operation.
#[derive(Debug, Clone, Copy, Default)]
pub struct AssertChangedBounds;

impl AssertChangedBounds {
    /// Construct the operation.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    /// The declared manifest for `assert.changed_bounds@1`.
    ///
    /// # Errors
    /// Propagates the [`schema`](ErrorClass::Schema) error if the hard-coded op /
    /// impl ids are somehow invalid (they are not).
    pub fn manifest() -> Result<OperationManifest> {
        Ok(OperationManifest {
            id: ASSERT_CHANGED_BOUNDS_OP_ID.parse()?,
            impl_version: 1,
            summary: "Assert that the region a before -> after edit changed (beyond a threshold) \
                      is contained in an expected box. Change escaping the box fails the \
                      assertion and records the actual changed bounds, the expected box, and the \
                      escaping-pixel count and worst pixel."
                .to_owned(),
            determinism: DeterminismTier::Exact,
            roi: RoiPolicy {
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
                    doc: "The edited image whose change from `before` is checked.".to_owned(),
                },
            ],
            outputs: vec![OutputSpec {
                name: "report".to_owned(),
                kind: ResourceKind::Report,
                doc: "The assertion verdict: pass/fail plus the actual changed bounds, the \
                      expected box, and the escaping-pixel count and worst pixel."
                    .to_owned(),
            }],
            params: vec![
                ParamSpec {
                    name: "bounds".to_owned(),
                    ty: ParamType::Integer,
                    unit: Some(ParamUnit::Pixels),
                    required: true,
                    default: None,
                    choices: vec![],
                    doc: "The expected box as the four pixel edges [x0, y0, x1, y1] (passed as \
                          the params `x0`, `y0`, `x1`, `y1`)."
                        .to_owned(),
                },
                threshold_param(
                    "The absolute delta a pixel must strictly exceed to count as changed.",
                ),
                severity_param(),
            ],
            implementations: vec![reference_impl()?],
            test: bounds_test_metadata(),
        })
    }
}

/// Resolve the expected box from the `x0`/`y0`/`x1`/`y1` integer params.
fn resolve_expected_bounds(params: &serde_json::Value) -> Result<Rect> {
    let x0 = require_i64(ASSERT_CHANGED_BOUNDS_OP_ID, params, "x0")?;
    let y0 = require_i64(ASSERT_CHANGED_BOUNDS_OP_ID, params, "y0")?;
    let x1 = require_i64(ASSERT_CHANGED_BOUNDS_OP_ID, params, "x1")?;
    let y1 = require_i64(ASSERT_CHANGED_BOUNDS_OP_ID, params, "y1")?;
    let rect = Rect::new(x0, y0, x1, y1);
    if rect.is_valid() {
        Ok(rect)
    } else {
        Err(Error::new(
            ErrorClass::Schema,
            E_BOUNDS_PARAM,
            "assert.changed_bounds expected box must be well-formed (x1 >= x0, y1 >= y0)"
                .to_owned(),
        )
        .with_context(ErrorContext::default().with_actual(format!("{rect:?}"))))
    }
}

/// The sRGB decode knot (`srgb -> linear`): below this the function is linear.
const SRGB_DECODE_KNOT: f32 = 0.040_45;

/// Decode one sRGB-encoded sample to linear light (matching analyze.diff).
#[must_use]
fn srgb_decode(c: f32) -> f32 {
    if c <= SRGB_DECODE_KNOT {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

impl OpContract for AssertChangedBounds {
    fn declared_inputs(&self) -> Vec<(String, ResourceKind)> {
        vec![
            ("before".to_owned(), ResourceKind::Image),
            ("after".to_owned(), ResourceKind::Image),
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
        let _ = resolve_severity(params)?;
        let _ = resolve_threshold(ASSERT_CHANGED_BOUNDS_OP_ID, params)?;
        let _ = resolve_expected_bounds(params)?;
        let op = ASSERT_CHANGED_BOUNDS_OP_ID;
        let before = image_descriptor(inputs, op, "before")?;
        let after = image_descriptor(inputs, op, "after")?;
        if before.extent != after.extent || before.layout != after.layout {
            return Err(shape_mismatch(
                op,
                "the `before` and `after` images must share an extent and channel layout",
                format!(
                    "before {:?}/{:?} vs after {:?}/{:?}",
                    before.extent, before.layout, after.extent, after.layout
                ),
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
        Ok(full_domain_regions(inputs, &["before", "after"]))
    }

    fn validate_postconditions(
        &self,
        outputs: &OutputDescriptors,
        _params: &serde_json::Value,
    ) -> Result<Vec<AssertionResult>> {
        Ok(produces_report(outputs))
    }
}

impl OpImplementation for AssertChangedBounds {
    fn compute(
        &self,
        inputs: &InputValues,
        params: &serde_json::Value,
    ) -> std::result::Result<OutputValues, Error> {
        let op = ASSERT_CHANGED_BOUNDS_OP_ID;
        let severity = resolve_severity(params)?;
        let threshold = resolve_threshold(op, params)?;
        let expected = resolve_expected_bounds(params)?;

        let before = require_value(inputs, op, "before")?;
        let after = require_value(inputs, op, "after")?;
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

        // Compare in decoded-linear when the inputs are sRGB-encoded, matching
        // analyze.diff's default comparison space.
        let decode = matches!(before_desc.color, ColorEncoding::Srgb);
        let extent = before_desc.extent;
        let channels = before.channels() as usize;
        let width = extent.width;

        let mut changed_bounds: Option<Rect> = None;
        let mut escaped: u64 = 0;
        let mut worst_pixel: Option<[i64; 2]> = None;
        let mut worst_delta = 0.0_f64;
        let mut locations: Vec<[i64; 2]> = Vec::new();

        if channels != 0 {
            for (pixel_index, (b_px, a_px)) in before
                .samples()
                .chunks_exact(channels)
                .zip(after.samples().chunks_exact(channels))
                .enumerate()
            {
                let mut pixel_delta = 0.0_f64;
                for (&b, &a) in b_px.iter().zip(a_px.iter()) {
                    let bv = if decode { srgb_decode(b) } else { b };
                    let av = if decode { srgb_decode(a) } else { a };
                    let delta = f64::from((av - bv).abs());
                    if delta > pixel_delta {
                        pixel_delta = delta;
                    }
                }
                if pixel_delta > threshold {
                    let [x, y] = pixel_coord(pixel_index, width);
                    let cell = Rect::new(x, y, x + 1, y + 1);
                    changed_bounds = Some(changed_bounds.map_or(cell, |b| b.union(cell)));
                    // A changed pixel that lies outside the expected box escapes.
                    if !expected.contains(x, y) {
                        escaped += 1;
                        if locations.len() < MAX_LOCATIONS {
                            locations.push([x, y]);
                        }
                        if worst_pixel.is_none() || pixel_delta > worst_delta {
                            worst_delta = pixel_delta;
                            worst_pixel = Some([x, y]);
                        }
                    }
                }
            }
        }

        let passed = escaped == 0;
        let mut outcome = AssertionOutcome::new(op, passed, severity);
        outcome.changed_bounds = changed_bounds;
        outcome.expected_bounds = Some(expected);
        outcome.violations = Some(escaped);
        outcome.worst_pixel = worst_pixel;
        outcome.locations = locations;
        Ok(single_report(assertion_report(extent, outcome)))
    }
}

#[cfg(test)]
mod tests;
