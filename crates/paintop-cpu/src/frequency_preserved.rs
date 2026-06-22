//! The `assert.frequency_preserved@1` operation (`OP_CATALOG` §14).
//!
//! Verifies that an edit preserved the image's band energy *outside* the edited
//! region: a touch-up that over-blurs (or sharpens) a region destroys
//! high-frequency texture, which shows up as a per-band energy delta. This is
//! the texture-preservation assertion the M4 "known/bounded synthetic fixtures"
//! exit criterion builds on.
//!
//! # Window / band policy (explicit)
//!
//! - **before / after**: two equal-extent Image/`Field1` resources (the edit's
//!   input and output).
//! - **mask**: optional. When present it marks the *edited* region; preservation
//!   is checked over its **complement** (the untouched area). When absent, the
//!   whole image is checked. The checked region is windowed by zeroing the other
//!   pixels before the decomposition, the same policy as
//!   `analyze.frequency_energy`.
//! - **bands / decomposition / phase**: the same pyramid policy as
//!   `analyze.frequency_energy`.
//! - **tolerance**: the maximum allowed *relative* per-band energy change
//!   `|E_after − E_before| / (E_before + epsilon)`. The assertion FAILS (records
//!   the worst band's delta) when any band exceeds it.
//!
//! The op is [`Exact`](DeterminismTier::Exact): every reduction is a fixed-order
//! stable pairwise `f64` sum, so the verdict is bit-identical across runs.

use paintop_core::executor::{InputValues, OpImplementation, OutputValues, ResourceValue};
use paintop_ir::{
    AssertionOutcome, AssertionResult, AssertionSeverity, Descriptors, DeterminismTier, Error,
    ErrorClass, ErrorContext, ImplId, InputRegions, InputSpec, OpContract, OperationManifest,
    OutputDescriptors, OutputRegions, OutputSpec, ParamSpec, ParamType, Rect, Report,
    ReportDescriptor, ResourceDescriptor, ResourceKind, Result, RoiCategory, RoiPolicy,
    TestMetadata,
};

use crate::frequency_energy::{FrequencyEnergyParams, frequency_energy_of};

/// The canonical id of the frequency-preservation assertion.
pub const FREQUENCY_PRESERVED_OP_ID: &str = "assert.frequency_preserved@1";

/// A required input port was absent or carried the wrong kind.
pub const E_FREQ_PRESERVED_INPUT: &str = "E_FREQ_PRESERVED_INPUT";

/// A parameter was missing, the wrong type, or out of range.
pub const E_FREQ_PRESERVED_PARAM: &str = "E_FREQ_PRESERVED_PARAM";

/// The inputs disagree on extent / channel count.
pub const E_FREQ_PRESERVED_SHAPE: &str = "E_FREQ_PRESERVED_SHAPE";

/// The small denominator floor that keeps the relative delta finite for an
/// (almost) zero-energy band.
const ENERGY_EPSILON: f64 = 1.0e-9;

/// The channel count of a supported Image/`Field1` descriptor.
fn input_channels(descriptor: &ResourceDescriptor) -> Result<u32> {
    match descriptor {
        ResourceDescriptor::Image(d) => Ok(d.layout.channel_count()),
        ResourceDescriptor::Field1(_) => Ok(1),
        _ => Err(Error::new(
            ErrorClass::Type,
            E_FREQ_PRESERVED_INPUT,
            "assert.frequency_preserved inputs must be Image or Field1 resources".to_owned(),
        )),
    }
}

/// Resolve the maximum allowed relative per-band energy change, defaulting to a
/// strict `0.05` (5%).
fn resolve_tolerance(params: &serde_json::Value) -> Result<f64> {
    let Some(value) = params.get("tolerance") else {
        return Ok(0.05);
    };
    let v = value.as_f64().ok_or_else(|| {
        Error::new(
            ErrorClass::Schema,
            E_FREQ_PRESERVED_PARAM,
            "assert.frequency_preserved `tolerance` must be a finite non-negative number"
                .to_owned(),
        )
    })?;
    if !v.is_finite() || v < 0.0 {
        return Err(Error::new(
            ErrorClass::Schema,
            E_FREQ_PRESERVED_PARAM,
            "assert.frequency_preserved `tolerance` must be a finite non-negative number"
                .to_owned(),
        )
        .with_context(ErrorContext::default().with_actual(v.to_string())));
    }
    Ok(v)
}

/// The severity token, defaulting to `error`.
fn resolve_severity(params: &serde_json::Value) -> Result<AssertionSeverity> {
    match params.get("severity").and_then(serde_json::Value::as_str) {
        None | Some("error") => Ok(AssertionSeverity::Error),
        Some("warning") => Ok(AssertionSeverity::Warning),
        Some("metric") => Ok(AssertionSeverity::Metric),
        Some(other) => Err(Error::new(
            ErrorClass::Schema,
            E_FREQ_PRESERVED_PARAM,
            "assert.frequency_preserved `severity` must be `error`, `warning`, or `metric`"
                .to_owned(),
        )
        .with_context(ErrorContext::default().with_actual(other.to_owned()))),
    }
}

/// Window the base to the *checked* region (the complement of the edited mask):
/// keep a pixel when its mask coverage is `<= 0` (untouched), zero it otherwise.
/// With no mask the whole image is kept.
#[must_use]
fn window_complement(base: &[f32], channels: u32, mask: Option<&[f32]>) -> Vec<f32> {
    let Some(mask) = mask else {
        return base.to_vec();
    };
    let ch = channels.max(1) as usize;
    let mut out = base.to_vec();
    for (pixel_index, pixel) in out.chunks_exact_mut(ch).enumerate() {
        let coverage = mask.get(pixel_index).copied().unwrap_or(0.0);
        // Inside the edited region (coverage > 0) is excluded from the check.
        if coverage > 0.0 {
            for v in pixel {
                *v = 0.0;
            }
        }
    }
    out
}

/// The worst (band index, relative delta) across the per-band energies of two
/// decompositions.
fn worst_band_delta(before: &[f64], after: &[f64]) -> (usize, f64) {
    let mut worst_band = 0usize;
    let mut worst_delta = 0.0_f64;
    for (band, (&eb, &ea)) in before.iter().zip(after.iter()).enumerate() {
        let delta = (ea - eb).abs() / (eb + ENERGY_EPSILON);
        if delta > worst_delta {
            worst_delta = delta;
            worst_band = band;
        }
    }
    (worst_band, worst_delta)
}

/// The `assert.frequency_preserved@1` operation.
#[derive(Debug, Clone, Copy, Default)]
pub struct FrequencyPreserved;

impl FrequencyPreserved {
    /// Construct the frequency-preservation assertion.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    /// The declared manifest for `assert.frequency_preserved@1`.
    ///
    /// # Errors
    /// Propagates the [`schema`](ErrorClass::Schema) error if the hard-coded op /
    /// impl ids are somehow invalid (they are not).
    pub fn manifest() -> Result<OperationManifest> {
        Ok(OperationManifest {
            id: FREQUENCY_PRESERVED_OP_ID.parse()?,
            impl_version: 1,
            summary:
                "Assert an edit preserved per-band frequency energy outside the edited region \
                      (within a relative tolerance); FAILS with the worst band's energy delta when \
                      a region is over-blurred or sharpened."
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
                    doc: "The edit's input Image/Field1.".to_owned(),
                },
                InputSpec {
                    name: "after".to_owned(),
                    kind: ResourceKind::Image,
                    required: true,
                    doc: "The edit's output Image/Field1 (same extent as `before`).".to_owned(),
                },
                InputSpec {
                    name: "mask".to_owned(),
                    kind: ResourceKind::Mask,
                    required: false,
                    doc: "An optional coverage mask marking the EDITED region; preservation is \
                          checked over its complement (the whole image when absent)."
                        .to_owned(),
                },
            ],
            outputs: vec![OutputSpec {
                name: "report".to_owned(),
                kind: ResourceKind::Report,
                doc: "The assertion report: per-band before/after energy and the verdict."
                    .to_owned(),
            }],
            params: vec![
                ParamSpec {
                    name: "bands".to_owned(),
                    ty: ParamType::Integer,
                    unit: None,
                    required: true,
                    default: None,
                    choices: vec![],
                    doc: "The number of bands (= pyramid levels), finest first.".to_owned(),
                },
                ParamSpec {
                    name: "tolerance".to_owned(),
                    ty: ParamType::Float,
                    unit: None,
                    required: false,
                    default: Some(serde_json::json!(0.05)),
                    choices: vec![],
                    doc: "The maximum allowed relative per-band energy change.".to_owned(),
                },
                ParamSpec {
                    name: "decomposition".to_owned(),
                    ty: ParamType::String,
                    unit: None,
                    required: false,
                    default: Some(serde_json::json!("laplacian")),
                    choices: vec!["laplacian".to_owned(), "gaussian".to_owned()],
                    doc: "The decomposition to compare over.".to_owned(),
                },
                ParamSpec {
                    name: "phase".to_owned(),
                    ty: ParamType::String,
                    unit: None,
                    required: false,
                    default: Some(serde_json::json!("floor")),
                    choices: vec!["floor".to_owned(), "ceil".to_owned()],
                    doc: "The odd-size rounding rule for the pyramid level extents.".to_owned(),
                },
                ParamSpec {
                    name: "severity".to_owned(),
                    ty: ParamType::String,
                    unit: None,
                    required: false,
                    default: Some(serde_json::json!("error")),
                    choices: vec![
                        "error".to_owned(),
                        "warning".to_owned(),
                        "metric".to_owned(),
                    ],
                    doc: "Whether a failure fails the run (error), warns, or is a metric."
                        .to_owned(),
                },
            ],
            implementations: vec![reference_impl()?],
            test: preserved_test_metadata(),
        })
    }
}

impl OpContract for FrequencyPreserved {
    fn declared_inputs(&self) -> Vec<(String, ResourceKind)> {
        vec![
            ("before".to_owned(), ResourceKind::Image),
            ("after".to_owned(), ResourceKind::Image),
            ("mask".to_owned(), ResourceKind::Mask),
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
        let before = require_descriptor(inputs, "before")?;
        let after = require_descriptor(inputs, "after")?;
        let channels = input_channels(before)?;
        if after.extent() != before.extent() {
            return Err(shape_mismatch(
                "`before` and `after` must share an extent",
                format!("{:?} vs {:?}", before.extent(), after.extent()),
            ));
        }
        let _ = resolve_params(params)?;
        if let Some(mask) = inputs.get("mask")
            && mask.extent() != before.extent()
        {
            return Err(shape_mismatch(
                "the `mask` must share the input extent",
                format!("mask {:?} vs input {:?}", mask.extent(), before.extent()),
            ));
        }
        let mut out = OutputDescriptors::new();
        out.insert(
            "report".to_owned(),
            ResourceDescriptor::Report(ReportDescriptor {
                extent: before.extent(),
                channels,
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
        let mut regions = InputRegions::new();
        for port in ["before", "after", "mask"] {
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
        Ok(vec![match outputs.get("report") {
            Some(ResourceDescriptor::Report(_)) => AssertionResult::pass("produces_report"),
            _ => AssertionResult::fail("produces_report", "no `report` output produced"),
        }])
    }
}

impl OpImplementation for FrequencyPreserved {
    fn compute(
        &self,
        inputs: &InputValues,
        params: &serde_json::Value,
    ) -> std::result::Result<OutputValues, Error> {
        let before = require_value(inputs, "before")?;
        let after = require_value(inputs, "after")?;
        let extent = before.extent();
        let channels = before.channels();
        if after.extent() != extent || after.channels() != channels {
            return Err(shape_mismatch(
                "`before` and `after` must share extent and channel count",
                format!(
                    "{:?}/{} vs {:?}/{}",
                    extent,
                    channels,
                    after.extent(),
                    after.channels()
                ),
            ));
        }
        let (energy_params, tolerance, severity) = resolve_params(params)?;

        let mask_samples = match inputs.get("mask") {
            Some(mask) => {
                if !matches!(mask.descriptor(), ResourceDescriptor::Mask(_)) {
                    return Err(Error::new(
                        ErrorClass::Type,
                        E_FREQ_PRESERVED_INPUT,
                        "assert.frequency_preserved `mask` input must be a mask resource"
                            .to_owned(),
                    ));
                }
                if mask.extent() != extent {
                    return Err(shape_mismatch(
                        "the `mask` must share the input extent",
                        format!("mask {:?} vs input {:?}", mask.extent(), extent),
                    ));
                }
                Some(mask.samples())
            }
            None => None,
        };

        // Window both images to the checked region (the complement of the edit),
        // then compare per-band energy.
        let before_win = window_complement(before.samples(), channels, mask_samples);
        let after_win = window_complement(after.samples(), channels, mask_samples);
        let before_energy = frequency_energy_of(&before_win, extent, channels, energy_params);
        let after_energy = frequency_energy_of(&after_win, extent, channels, energy_params);

        let (worst_band, worst_delta) =
            worst_band_delta(&before_energy.band_energy, &after_energy.band_energy);
        let passed = worst_delta <= tolerance;

        let mut outcome = AssertionOutcome::new(FREQUENCY_PRESERVED_OP_ID, passed, severity);
        // The worst band's relative energy delta and its band index.
        outcome.worst_value = Some(worst_delta);
        if !passed {
            outcome.violations = Some(count_violating_bands(
                &before_energy.band_energy,
                &after_energy.band_energy,
                tolerance,
            ));
            // Record the worst band as a [band_index, 0] locator so the failure
            // artifact names the offending band.
            outcome.worst_pixel = Some([i64::try_from(worst_band).unwrap_or(i64::MAX), 0]);
        }

        let report = Report {
            extent,
            channels,
            channel_stats: Vec::new(),
            all_finite: true,
            content_hash: String::new(),
            diff: None,
            assertion: Some(outcome),
            histogram: None,
            components: None,
            // Attach the *after* per-band energy so a consumer can inspect the
            // measured bands alongside the verdict.
            frequency_energy: Some(after_energy),
            solver: None,
        };
        let mut out = OutputValues::new();
        out.insert("report".to_owned(), ResourceValue::report(report));
        Ok(out)
    }
}

/// The number of bands whose relative energy delta exceeds `tolerance`.
fn count_violating_bands(before: &[f64], after: &[f64], tolerance: f64) -> u64 {
    let count = before
        .iter()
        .zip(after.iter())
        .filter(|&(&eb, &ea)| (ea - eb).abs() / (eb + ENERGY_EPSILON) > tolerance)
        .count();
    count as u64
}

/// Resolve the energy/pyramid params plus the tolerance and severity, mapping
/// `bands` onto the pyramid `levels` (shared with `analyze.frequency_energy`).
fn resolve_params(
    params: &serde_json::Value,
) -> Result<(FrequencyEnergyParams, f64, AssertionSeverity)> {
    // Require `bands` explicitly so the assertion never guesses a band count.
    if params.get("bands").is_none() {
        return Err(Error::new(
            ErrorClass::Schema,
            E_FREQ_PRESERVED_PARAM,
            "assert.frequency_preserved requires a `bands` parameter".to_owned(),
        ));
    }
    let energy_params = FrequencyEnergyParams::resolve(params)
        .map_err(|e| Error::new(ErrorClass::Schema, E_FREQ_PRESERVED_PARAM, e.message))?;
    let tolerance = resolve_tolerance(params)?;
    let severity = resolve_severity(params)?;
    Ok((energy_params, tolerance, severity))
}

/// Read a required input descriptor port.
fn require_descriptor<'a>(inputs: &'a Descriptors, port: &str) -> Result<&'a ResourceDescriptor> {
    inputs.get(port).ok_or_else(|| {
        Error::new(
            ErrorClass::Reference,
            E_FREQ_PRESERVED_INPUT,
            format!("assert.frequency_preserved requires a `{port}` input"),
        )
    })
}

/// Read a required input value port.
fn require_value<'a>(
    inputs: &'a InputValues,
    port: &str,
) -> std::result::Result<&'a ResourceValue, Error> {
    inputs.get(port).ok_or_else(|| {
        Error::new(
            ErrorClass::Reference,
            E_FREQ_PRESERVED_INPUT,
            format!("assert.frequency_preserved requires a `{port}` input value"),
        )
    })
}

/// Build a shape-mismatch [`semantic`](ErrorClass::Semantic) error.
fn shape_mismatch(detail: &str, actual: String) -> Error {
    Error::new(
        ErrorClass::Semantic,
        E_FREQ_PRESERVED_SHAPE,
        format!("assert.frequency_preserved: {detail}"),
    )
    .with_context(ErrorContext::default().with_actual(actual))
}

/// The mandatory `cpu.reference@1` oracle implementation id.
fn reference_impl() -> Result<ImplId> {
    ImplId::new("cpu", "reference", 1)
}

/// Verification declarations: an exact, single-reference assertion. Differential
/// and perceptual do not apply; analytic fixtures (preserved => pass,
/// over-blurred => fail with a band delta) and property tests cover the rest.
fn preserved_test_metadata() -> TestMetadata {
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
