//! The `analyze.frequency_energy@1` operation (`OP_CATALOG` §13).
//!
//! Reduces an Image/`Field1` to its per-band frequency energy: it builds a
//! multi-resolution decomposition (a Gaussian pyramid's per-level low-pass
//! planes or a Laplacian pyramid's band-pass residuals) and reports the
//! sum-of-squares energy of each band, finest first, plus the total. This is the
//! texture-preservation measurement the M4 "known/bounded synthetic fixtures"
//! exit criterion builds on, and the metric `assert.frequency_preserved`
//! compares against.
//!
//! # Band / window policy (explicit)
//!
//! - **decomposition**: `laplacian` (default) reports band-pass energy per
//!   level (the coarsest level is the low-pass); `gaussian` reports the energy
//!   of each blurred level.
//! - **window**: an optional coverage `mask` restricts the analysis to the
//!   masked region by zeroing the unmasked base pixels *before* the
//!   decomposition, so the reported energy is the energy of the masked content.
//!   The mask must share the input extent.
//!
//! Every reduction is a fixed-order stable pairwise `f64` sum, so the report is
//! [`Exact`](DeterminismTier::Exact) and bit-identical across runs.

use paintop_core::executor::{InputValues, OpImplementation, OutputValues, ResourceValue};
use paintop_ir::{
    AssertionResult, Descriptors, DeterminismTier, Error, ErrorClass, ErrorContext, Extent,
    FrequencyEnergyData, ImplId, InputRegions, InputSpec, OpContract, OperationManifest,
    OutputDescriptors, OutputRegions, OutputSpec, ParamSpec, ParamType, Rect, Report,
    ReportDescriptor, ResourceDescriptor, ResourceKind, Result, RoiCategory, RoiPolicy,
    TestMetadata,
};

use crate::frequency::level_extents;
use crate::gaussian_pyramid::{PyramidParams, build_pyramid_samples};
use crate::laplacian::build_laplacian_samples;

/// The canonical id of the frequency-energy analysis operation.
pub const FREQUENCY_ENERGY_OP_ID: &str = "analyze.frequency_energy@1";

/// A required input port was absent or carried the wrong kind.
pub const E_FREQ_ENERGY_INPUT: &str = "E_FREQ_ENERGY_INPUT";

/// A parameter was missing, the wrong type, or out of range.
pub const E_FREQ_ENERGY_PARAM: &str = "E_FREQ_ENERGY_PARAM";

/// The optional `mask` disagrees with the input extent.
pub const E_FREQ_ENERGY_SHAPE: &str = "E_FREQ_ENERGY_SHAPE";

/// The decomposition a frequency-energy analysis reports over.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Decomposition {
    /// Per-level low-pass energy (Gaussian pyramid).
    Gaussian,
    /// Per-band band-pass energy (Laplacian pyramid); the coarsest level is the
    /// low-pass.
    Laplacian,
}

impl Decomposition {
    /// The stable wire token for this decomposition.
    const fn token(self) -> &'static str {
        match self {
            Self::Gaussian => "gaussian",
            Self::Laplacian => "laplacian",
        }
    }
}

/// The stable pairwise (`f64`) sum of a slice — the deterministic reduction
/// primitive shared with `analyze.statistics`.
#[must_use]
fn pairwise_sum(values: &[f64]) -> f64 {
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

/// The sum-of-squares energy of a band plane, as a stable pairwise `f64` sum.
#[must_use]
fn band_energy(plane: &[f32]) -> f64 {
    let squares: Vec<f64> = plane
        .iter()
        .map(|&v| {
            let x = f64::from(v);
            x * x
        })
        .collect();
    pairwise_sum(&squares)
}

/// The resolved frequency-energy parameters.
#[derive(Debug, Clone, Copy)]
pub struct FrequencyEnergyParams {
    /// The decomposition to report over.
    pub decomposition: Decomposition,
    /// The pyramid build parameters (levels, sigma, phase).
    pub pyramid: PyramidParams,
}

impl FrequencyEnergyParams {
    /// Resolve `bands` (alias for `levels`), `decomposition`, `sigma`, `phase`.
    ///
    /// # Errors
    /// Returns a [`schema`](ErrorClass::Schema) error for a missing/ill-typed
    /// parameter or an unknown decomposition/phase token.
    pub fn resolve(params: &serde_json::Value) -> Result<Self> {
        // `bands` is the public name; it maps onto the pyramid's `levels`.
        let mut pyramid_params = params.clone();
        if let Some(bands) = params.get("bands")
            && let Some(obj) = pyramid_params.as_object_mut()
        {
            obj.insert("levels".to_owned(), bands.clone());
        }
        if pyramid_params.get("levels").is_none() {
            return Err(Error::new(
                ErrorClass::Schema,
                E_FREQ_ENERGY_PARAM,
                "analyze.frequency_energy requires a `bands` parameter".to_owned(),
            ));
        }
        let pyramid = PyramidParams::resolve(&pyramid_params).map_err(|e| {
            Error::new(ErrorClass::Schema, E_FREQ_ENERGY_PARAM, e.message)
                .with_context(ErrorContext::default().with_actual(params.to_string()))
        })?;

        let decomposition = match params
            .get("decomposition")
            .and_then(serde_json::Value::as_str)
        {
            None | Some("laplacian") => Decomposition::Laplacian,
            Some("gaussian") => Decomposition::Gaussian,
            Some(other) => {
                return Err(Error::new(
                    ErrorClass::Schema,
                    E_FREQ_ENERGY_PARAM,
                    "analyze.frequency_energy `decomposition` must be `laplacian` or `gaussian`"
                        .to_owned(),
                )
                .with_context(ErrorContext::default().with_actual(other.to_owned())));
            }
        };

        Ok(Self {
            decomposition,
            pyramid,
        })
    }
}

/// The channel count of a supported Image/`Field1` descriptor.
fn input_channels(descriptor: &ResourceDescriptor) -> Result<u32> {
    match descriptor {
        ResourceDescriptor::Image(d) => Ok(d.layout.channel_count()),
        ResourceDescriptor::Field1(_) => Ok(1),
        _ => Err(Error::new(
            ErrorClass::Type,
            E_FREQ_ENERGY_INPUT,
            "analyze.frequency_energy `resource` must be an Image or Field1 resource".to_owned(),
        )),
    }
}

/// Zero the base samples outside the masked region (coverage strictly positive),
/// returning the windowed base. When `mask` is `None` the base is returned
/// unchanged.
#[must_use]
fn apply_window(base: &[f32], channels: u32, mask: Option<&[f32]>) -> Vec<f32> {
    let Some(mask) = mask else {
        return base.to_vec();
    };
    let ch = channels.max(1) as usize;
    let mut out = base.to_vec();
    for (pixel_index, pixel) in out.chunks_exact_mut(ch).enumerate() {
        let coverage = mask.get(pixel_index).copied().unwrap_or(0.0);
        if coverage <= 0.0 || coverage.is_nan() {
            for v in pixel {
                *v = 0.0;
            }
        }
    }
    out
}

/// Compute the per-band energy summary for a windowed base.
#[must_use]
pub fn frequency_energy_of(
    base: &[f32],
    extent: Extent,
    channels: u32,
    params: FrequencyEnergyParams,
) -> FrequencyEnergyData {
    let pyramid = params.pyramid;
    let buffer = match params.decomposition {
        Decomposition::Gaussian => build_pyramid_samples(base, extent, channels, pyramid),
        Decomposition::Laplacian => build_laplacian_samples(base, extent, channels, pyramid),
    };
    let extents = level_extents(extent, pyramid.levels, pyramid.phase);
    let mut band_energy_vec = Vec::with_capacity(extents.len());
    let mut band_pixels = Vec::with_capacity(extents.len());
    let mut offset = 0usize;
    for e in &extents {
        let n = (e.width as usize) * (e.height as usize) * (channels as usize);
        let plane = &buffer[offset..offset + n];
        band_energy_vec.push(band_energy(plane));
        band_pixels.push(n as u64);
        offset += n;
    }
    let total_energy = pairwise_sum(&band_energy_vec);
    FrequencyEnergyData {
        decomposition: params.decomposition.token().to_owned(),
        bands: pyramid.levels,
        band_energy: band_energy_vec,
        band_pixels,
        total_energy,
    }
}

/// The `analyze.frequency_energy@1` operation.
#[derive(Debug, Clone, Copy, Default)]
pub struct FrequencyEnergy;

impl FrequencyEnergy {
    /// Construct the frequency-energy analysis operation.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    /// The declared manifest for `analyze.frequency_energy@1`.
    ///
    /// # Errors
    /// Propagates the [`schema`](ErrorClass::Schema) error if the hard-coded op /
    /// impl ids are somehow invalid (they are not).
    pub fn manifest() -> Result<OperationManifest> {
        Ok(OperationManifest {
            id: FREQUENCY_ENERGY_OP_ID.parse()?,
            impl_version: 1,
            summary: "Report the per-band frequency energy (sum of squares) of an Image/Field1 \
                      under a Laplacian or Gaussian decomposition, optionally windowed by a mask. \
                      The texture-preservation measurement assert.frequency_preserved compares."
                .to_owned(),
            determinism: DeterminismTier::Exact,
            roi: RoiPolicy {
                category: RoiCategory::FullDomain,
                halo_px: None,
            },
            inputs: vec![
                InputSpec {
                    name: "resource".to_owned(),
                    kind: ResourceKind::Image,
                    required: true,
                    doc: "The Image or Field1 whose per-band energy is reported.".to_owned(),
                },
                InputSpec {
                    name: "mask".to_owned(),
                    kind: ResourceKind::Mask,
                    required: false,
                    doc: "An optional coverage mask: energy is computed over the masked region \
                          (unmasked base pixels are zeroed before decomposition)."
                        .to_owned(),
                },
            ],
            outputs: vec![OutputSpec {
                name: "report".to_owned(),
                kind: ResourceKind::Report,
                doc: "The per-band frequency-energy report.".to_owned(),
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
                    name: "decomposition".to_owned(),
                    ty: ParamType::String,
                    unit: None,
                    required: false,
                    default: Some(serde_json::json!("laplacian")),
                    choices: vec!["laplacian".to_owned(), "gaussian".to_owned()],
                    doc: "The decomposition to report over (band-pass vs per-level low-pass)."
                        .to_owned(),
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
            ],
            implementations: vec![reference_impl()?],
            test: energy_test_metadata(),
        })
    }
}

impl OpContract for FrequencyEnergy {
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
        params: &serde_json::Value,
    ) -> Result<OutputDescriptors> {
        let resource = inputs.get("resource").ok_or_else(|| {
            Error::new(
                ErrorClass::Reference,
                E_FREQ_ENERGY_INPUT,
                "analyze.frequency_energy requires a `resource` input".to_owned(),
            )
        })?;
        let channels = input_channels(resource)?;
        let _ = FrequencyEnergyParams::resolve(params)?;
        if let Some(mask) = inputs.get("mask")
            && mask.extent() != resource.extent()
        {
            return Err(shape_mismatch(
                "the `mask` must share the resource extent",
                format!(
                    "mask {:?} vs resource {:?}",
                    mask.extent(),
                    resource.extent()
                ),
            ));
        }
        let mut out = OutputDescriptors::new();
        out.insert(
            "report".to_owned(),
            ResourceDescriptor::Report(ReportDescriptor {
                extent: resource.extent(),
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
        for port in ["resource", "mask"] {
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

impl OpImplementation for FrequencyEnergy {
    fn compute(
        &self,
        inputs: &InputValues,
        params: &serde_json::Value,
    ) -> std::result::Result<OutputValues, Error> {
        let resource = inputs.get("resource").ok_or_else(|| {
            Error::new(
                ErrorClass::Reference,
                E_FREQ_ENERGY_INPUT,
                "analyze.frequency_energy requires a `resource` value".to_owned(),
            )
        })?;
        let channels = resource.channels();
        let extent = resource.extent();
        let resolved = FrequencyEnergyParams::resolve(params)?;

        let mask_samples = match inputs.get("mask") {
            Some(mask) => {
                if !matches!(mask.descriptor(), ResourceDescriptor::Mask(_)) {
                    return Err(Error::new(
                        ErrorClass::Type,
                        E_FREQ_ENERGY_INPUT,
                        "analyze.frequency_energy `mask` input must be a mask resource".to_owned(),
                    ));
                }
                if mask.extent() != extent {
                    return Err(shape_mismatch(
                        "the `mask` must share the resource extent",
                        format!("mask {:?} vs resource {:?}", mask.extent(), extent),
                    ));
                }
                Some(mask.samples())
            }
            None => None,
        };

        let windowed = apply_window(resource.samples(), channels, mask_samples);
        let energy = frequency_energy_of(&windowed, extent, channels, resolved);
        let report = Report {
            extent,
            channels,
            channel_stats: Vec::new(),
            all_finite: true,
            content_hash: String::new(),
            diff: None,
            assertion: None,
            histogram: None,
            components: None,
            frequency_energy: Some(energy),
            solver: None,
        };
        let mut out = OutputValues::new();
        out.insert("report".to_owned(), ResourceValue::report(report));
        Ok(out)
    }
}

/// Build a shape-mismatch [`semantic`](ErrorClass::Semantic) error.
fn shape_mismatch(detail: &str, actual: String) -> Error {
    Error::new(
        ErrorClass::Semantic,
        E_FREQ_ENERGY_SHAPE,
        format!("analyze.frequency_energy: {detail}"),
    )
    .with_context(ErrorContext::default().with_actual(actual))
}

/// The mandatory `cpu.reference@1` oracle implementation id.
fn reference_impl() -> Result<ImplId> {
    ImplId::new("cpu", "reference", 1)
}

/// Verification declarations: an exact, single-reference reduction. Differential
/// and perceptual do not apply; analytic fixtures (a single-frequency band gets
/// its energy in the matching band) and property tests cover the rest.
fn energy_test_metadata() -> TestMetadata {
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
