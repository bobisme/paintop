//! The `frequency.bandpass@1` operation (`OP_CATALOG` §9).
//!
//! `frequency.bandpass` filters a real Image/`Field1` plane by a radial
//! frequency response: it forward-transforms the plane ([`crate::dft`]),
//! multiplies every bin by a declared band response, and inverse-transforms back
//! to the same spatial kind. The result attenuates (or isolates) a declared
//! frequency band.
//!
//! # Boundary / window policy (declared, never implicit)
//!
//! - **Boundary**: the DFT is inherently *periodic* — the plane is treated as
//!   one tile of an infinite toroidal tiling ([`BoundaryMode::Wrap`]). This is
//!   the only boundary a frequency-domain filter admits; it is recorded
//!   explicitly so a consumer knows edge wrap-around, not zero-padding, governs
//!   the response.
//! - **Window** (the band response shape): the multiplicative response on a
//!   bin of normalized radial frequency `f` (cycles-per-pixel, DC `0`, per-axis
//!   Nyquist `0.5`) is either an **ideal** brick-wall (`1` inside `[low, high]`,
//!   `0` outside) or a soft **gaussian** roll-off centered on the band. The
//!   choice and the `[low, high]` cutoffs are explicit parameters.
//! - **Mode**: `band-pass` keeps the band and attenuates the rest; `band-stop`
//!   is its complement (`1 − response`).
//!
//! # Determinism
//!
//! The forward/response/inverse chain is a fixed-order `f64` computation, so the
//! filter is bit-identical across reruns ([`Bounded`](DeterminismTier::Bounded),
//! agreeing with an alternate FFT backend only within the transform bound). The
//! response weight is a pure function of the bin's radial frequency, so it is
//! tiling-invariant (no sequential RNG, the M4 reproducible-noise criterion's
//! spirit applied to the deterministic response).

use paintop_core::executor::{InputValues, OpImplementation, OutputValues, ResourceValue};
use paintop_ir::{
    AssertionResult, BoundaryMode, Descriptors, DeterminismTier, Error, ErrorClass, ErrorContext,
    Extent, ImplId, InputRegions, InputSpec, OpContract, OperationManifest, OutputDescriptors,
    OutputRegions, OutputSpec, ParamSpec, ParamType, Rect, ResourceDescriptor, ResourceKind,
    Result, RoiCategory, RoiPolicy, TestMetadata,
};

use crate::dft::{forward_real, inverse_real, radial_frequency};

/// The canonical id of the band-pass operation.
pub const BANDPASS_OP_ID: &str = "frequency.bandpass@1";

/// The `input` was absent or carried an unsupported descriptor.
pub const E_BANDPASS_INPUT: &str = "E_BANDPASS_INPUT";

/// A parameter was missing, the wrong type, or out of range.
pub const E_BANDPASS_PARAM: &str = "E_BANDPASS_PARAM";

/// The maximum radial frequency a bin can take: a corner bin reaches
/// `sqrt(0.5² + 0.5²)` cycles-per-pixel. Cutoffs are bounded by this.
const MAX_RADIAL: f64 = std::f64::consts::SQRT_2 * 0.5;

/// The window (response shape) of a band response.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Window {
    /// A brick-wall response: `1` strictly inside `[low, high]`, `0` outside.
    Ideal,
    /// A soft Gaussian response centered on the band midpoint, with a standard
    /// deviation of half the band half-width (so the band edges sit at `1σ`).
    Gaussian,
}

/// Whether the band is kept (`Pass`) or rejected (`Stop`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    /// Keep the band, attenuate the rest.
    Pass,
    /// Attenuate the band, keep the rest (the complement response).
    Stop,
}

/// The resolved band-response parameters.
#[derive(Debug, Clone, Copy)]
pub struct BandParams {
    /// The lower cutoff (cycles-per-pixel), `>= 0`.
    pub low: f64,
    /// The upper cutoff (cycles-per-pixel), `>= low`.
    pub high: f64,
    /// The response window shape.
    pub window: Window,
    /// Pass or stop the band.
    pub mode: Mode,
}

impl BandParams {
    /// Resolve and validate `low`, `high`, `window`, and `mode` from the param
    /// object.
    ///
    /// # Errors
    /// Returns a [`schema`](ErrorClass::Schema) error for a missing/ill-typed
    /// parameter, a non-finite or negative cutoff, `high < low`, an out-of-range
    /// cutoff, or an unknown window/mode token.
    pub fn resolve(params: &serde_json::Value) -> Result<Self> {
        let low = cutoff(params, "low")?;
        let high = cutoff(params, "high")?;
        if high < low {
            return Err(param_err("`high` must be >= `low`").with_context(
                ErrorContext::default().with_actual(format!("low={low} high={high}")),
            ));
        }
        let window = match params.get("window").and_then(serde_json::Value::as_str) {
            None | Some("ideal") => Window::Ideal,
            Some("gaussian") => Window::Gaussian,
            Some(other) => {
                return Err(param_err("`window` must be `ideal` or `gaussian`")
                    .with_context(ErrorContext::default().with_actual(other.to_owned())));
            }
        };
        let mode = match params.get("mode").and_then(serde_json::Value::as_str) {
            None | Some("band-pass") => Mode::Pass,
            Some("band-stop") => Mode::Stop,
            Some(other) => {
                return Err(param_err("`mode` must be `band-pass` or `band-stop`")
                    .with_context(ErrorContext::default().with_actual(other.to_owned())));
            }
        };
        Ok(Self {
            low,
            high,
            window,
            mode,
        })
    }

    /// The multiplicative response weight for a bin of normalized radial
    /// frequency `f` (cycles-per-pixel) under this band.
    #[must_use]
    pub fn response(&self, f: f64) -> f64 {
        let base = match self.window {
            Window::Ideal => {
                if f >= self.low && f <= self.high {
                    1.0
                } else {
                    0.0
                }
            }
            Window::Gaussian => {
                let center = 0.5 * (self.low + self.high);
                let half = 0.5 * (self.high - self.low);
                if half <= 0.0 {
                    // A zero-width gaussian band degenerates to a single-bin
                    // impulse: respond only exactly at the center.
                    if (f - center).abs() < 1e-12 { 1.0 } else { 0.0 }
                } else {
                    let z = (f - center) / half;
                    (-0.5 * z * z).exp()
                }
            }
        };
        match self.mode {
            Mode::Pass => base,
            Mode::Stop => 1.0 - base,
        }
    }
}

/// Parse and validate a cutoff parameter named `name`.
fn cutoff(params: &serde_json::Value, name: &str) -> Result<f64> {
    let v = params
        .get(name)
        .ok_or_else(|| param_err(&format!("requires a `{name}` cutoff parameter")))?
        .as_f64()
        .ok_or_else(|| param_err(&format!("`{name}` must be a number")))?;
    if !v.is_finite() || v < 0.0 {
        return Err(
            param_err(&format!("`{name}` must be a finite, non-negative number"))
                .with_context(ErrorContext::default().with_actual(v.to_string())),
        );
    }
    if v > MAX_RADIAL + 1e-9 {
        return Err(param_err(&format!("`{name}` must not exceed {MAX_RADIAL}"))
            .with_context(ErrorContext::default().with_actual(v.to_string())));
    }
    Ok(v)
}

/// Build a schema [`Error`] for a malformed band-pass parameter.
fn param_err(detail: &str) -> Error {
    Error::new(
        ErrorClass::Schema,
        E_BANDPASS_PARAM,
        format!("frequency.bandpass {detail}"),
    )
}

/// The interleaved channel count of a supported Image/`Field1` descriptor.
fn input_channels(descriptor: &ResourceDescriptor) -> Result<u32> {
    match descriptor {
        ResourceDescriptor::Image(d) => Ok(d.layout.channel_count()),
        ResourceDescriptor::Field1(_) => Ok(1),
        _ => Err(Error::new(
            ErrorClass::Type,
            E_BANDPASS_INPUT,
            "frequency.bandpass `input` must be an Image or Field1 resource".to_owned(),
        )),
    }
}

/// Apply the band response to a `channels`-interleaved real spatial plane and
/// inverse-transform back to the spatial domain.
///
/// The plane is forward-transformed, every bin scaled by the band response of
/// its radial frequency, and inverse-transformed (real part). Pure DC (the mean)
/// is preserved iff the band admits `f = 0`.
#[must_use]
pub fn apply_band(samples: &[f32], extent: Extent, channels: u32, band: BandParams) -> Vec<f32> {
    let width = extent.width as usize;
    let height = extent.height as usize;
    let ch = channels as usize;
    if width == 0 || height == 0 || ch == 0 {
        return Vec::new();
    }
    let mut spectrum = forward_real(samples, width, height, ch);
    for ky in 0..height {
        for kx in 0..width {
            let f = radial_frequency(kx, ky, width, height);
            let w = band.response(f);
            let pixel = ky * width + kx;
            for c in 0..ch {
                let base = (pixel * ch + c) * 2;
                #[allow(
                    clippy::cast_possible_truncation,
                    reason = "scale in f64, store the op's f32 spectrum component"
                )]
                {
                    spectrum[base] = (f64::from(spectrum[base]) * w) as f32;
                    spectrum[base + 1] = (f64::from(spectrum[base + 1]) * w) as f32;
                }
            }
        }
    }
    inverse_real(&spectrum, width, height, ch)
}

/// The `frequency.bandpass@1` operation.
#[derive(Debug, Clone, Copy, Default)]
pub struct Bandpass;

impl Bandpass {
    /// Construct the band-pass operation.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    /// The declared manifest for `frequency.bandpass@1`.
    ///
    /// # Errors
    /// Propagates the schema error if the hard-coded op/impl ids are invalid.
    pub fn manifest() -> Result<OperationManifest> {
        Ok(OperationManifest {
            id: BANDPASS_OP_ID.parse()?,
            impl_version: 1,
            summary: "Filter a real Image/Field1 plane by a radial frequency band: forward DFT, \
                      multiply by an ideal or gaussian band response over [low, high] \
                      cycles-per-pixel (band-pass or band-stop), inverse DFT. Periodic (wrap) \
                      boundary."
                .to_owned(),
            determinism: DeterminismTier::Bounded,
            roi: RoiPolicy {
                category: RoiCategory::FullDomain,
                halo_px: None,
            },
            inputs: vec![InputSpec {
                name: "input".to_owned(),
                kind: ResourceKind::Image,
                required: true,
                doc: "The real Image or Field1 plane to filter.".to_owned(),
            }],
            outputs: vec![OutputSpec {
                name: "output".to_owned(),
                kind: ResourceKind::Image,
                doc: "The band-filtered plane, same kind and extent as the input.".to_owned(),
            }],
            params: band_params_spec(),
            implementations: vec![reference_impl()?],
            test: bandpass_test_metadata(),
        })
    }
}

impl OpContract for Bandpass {
    fn declared_inputs(&self) -> Vec<(String, ResourceKind)> {
        vec![("input".to_owned(), ResourceKind::Image)]
    }

    fn declared_outputs(&self) -> Vec<(String, ResourceKind)> {
        vec![("output".to_owned(), ResourceKind::Image)]
    }

    fn infer_outputs(
        &self,
        inputs: &Descriptors,
        params: &serde_json::Value,
    ) -> Result<OutputDescriptors> {
        let input = inputs.get("input").ok_or_else(|| {
            Error::new(
                ErrorClass::Reference,
                E_BANDPASS_INPUT,
                "frequency.bandpass requires an `input` resource".to_owned(),
            )
        })?;
        // Validate the input kind and the band parameters early.
        let _ = input_channels(input)?;
        let _ = BandParams::resolve(params)?;
        // The output is the same typed resource as the input (same extent/kind).
        let mut out = OutputDescriptors::new();
        out.insert("output".to_owned(), *input);
        Ok(out)
    }

    fn required_inputs(
        &self,
        _requested_outputs: &OutputRegions,
        inputs: &Descriptors,
        _params: &serde_json::Value,
    ) -> Result<InputRegions> {
        let mut regions = InputRegions::new();
        if let Some(input) = inputs.get("input") {
            let extent = input.extent();
            regions.insert(
                "input".to_owned(),
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
        Ok(vec![match outputs.get("output") {
            Some(ResourceDescriptor::Image(_) | ResourceDescriptor::Field1(_)) => {
                AssertionResult::pass("produces_output")
            }
            _ => AssertionResult::fail("produces_output", "no filtered `output` produced"),
        }])
    }
}

impl OpImplementation for Bandpass {
    fn compute(
        &self,
        inputs: &InputValues,
        params: &serde_json::Value,
    ) -> std::result::Result<OutputValues, Error> {
        let input = inputs.get("input").ok_or_else(|| {
            Error::new(
                ErrorClass::Reference,
                E_BANDPASS_INPUT,
                "frequency.bandpass requires an `input` value".to_owned(),
            )
        })?;
        let _ = input_channels(input.descriptor())?;
        let band = BandParams::resolve(params)?;
        let channels = input.channels();
        let extent = input.extent();
        let samples = apply_band(input.samples(), extent, channels, band);
        let value =
            ResourceValue::new(*input.descriptor(), channels, samples).map_err(|actual| {
                Error::new(
                    ErrorClass::Execution,
                    E_BANDPASS_INPUT,
                    format!("frequency.bandpass produced a buffer of unexpected length {actual}"),
                )
            })?;
        let mut out = OutputValues::new();
        out.insert("output".to_owned(), value);
        Ok(out)
    }
}

/// The declared boundary policy of a frequency-domain filter: periodic wrap.
/// Recorded so the policy is explicit (the module docs reference it).
pub const BANDPASS_BOUNDARY: BoundaryMode = BoundaryMode::Wrap;

/// The `low`/`high`/`window`/`mode` parameter spec for the band-pass op.
fn band_params_spec() -> Vec<ParamSpec> {
    vec![
        ParamSpec {
            name: "low".to_owned(),
            ty: ParamType::Float,
            unit: None,
            required: true,
            default: None,
            choices: vec![],
            doc: "The lower band cutoff in cycles-per-pixel (DC 0, per-axis Nyquist 0.5)."
                .to_owned(),
        },
        ParamSpec {
            name: "high".to_owned(),
            ty: ParamType::Float,
            unit: None,
            required: true,
            default: None,
            choices: vec![],
            doc: "The upper band cutoff in cycles-per-pixel (>= low).".to_owned(),
        },
        ParamSpec {
            name: "window".to_owned(),
            ty: ParamType::String,
            unit: None,
            required: false,
            default: Some(serde_json::json!("ideal")),
            choices: vec!["ideal".to_owned(), "gaussian".to_owned()],
            doc: "The band response shape: an ideal brick-wall or a soft gaussian roll-off."
                .to_owned(),
        },
        ParamSpec {
            name: "mode".to_owned(),
            ty: ParamType::String,
            unit: None,
            required: false,
            default: Some(serde_json::json!("band-pass")),
            choices: vec!["band-pass".to_owned(), "band-stop".to_owned()],
            doc: "Keep the band (band-pass) or reject it (band-stop, the complement).".to_owned(),
        },
    ]
}

/// The mandatory `cpu.reference@1` oracle implementation id.
fn reference_impl() -> Result<ImplId> {
    ImplId::new("cpu", "reference", 1)
}

/// Verification declarations for `frequency.bandpass@1`: a bounded filter with
/// analytic fixtures (a sinusoid attenuated by a band-stop straddling its
/// frequency; a low-pass band preserving the DC mean; a full-band identity) and
/// property tests (determinism, response monotonic complement). Differential
/// applies in principle (single reference today → not-applicable with a reason);
/// perceptual does not apply — correctness is the band-attenuation property set.
fn bandpass_test_metadata() -> TestMetadata {
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
    decls = decls.with(
        VerificationCategory::Perceptual,
        CategoryStatus::not_applicable(
            "the band-pass filter is verified by the declared-band attenuation, DC preservation, \
             and full-band identity fixtures, not a perceptual metric",
        ),
    );
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
