//! The `filter.gaussian_blur@1` operation: an isotropic Gaussian blur built on
//! the direct convolution reference.
//!
//! Refs: `OP_CATALOG` §8, `AGENT_VERIFICATION` §3.4.
//!
//! `filter.gaussian_blur` samples a normalized 2-D Gaussian of standard deviation
//! `sigma` into a finite kernel and convolves the input with it through the same
//! direct-convolution oracle as [`crate::convolve`] (the separable/FFT fast paths
//! are M3; this is the semantic reference). It is the canonical blur the rest of
//! the system is differentially checked against.
//!
//! # Kernel construction
//!
//! For a requested `sigma > 0` the kernel is square with **radius**
//! `r = ceil(3 * sigma)` (the declared halo) and side `2r + 1`. Tap `(dx, dy)`
//! (offsets in `[-r, r]`) gets weight `exp(-(dx² + dy²) / (2σ²))`; the whole
//! kernel is then divided by its sum, so the kernel is **positive** and
//! **unit-sum** by construction — a constant image is preserved exactly (to
//! rounding) and the kernel is isotropic, hence invariant under 90° rotation.
//!
//! # σ→0 cutoff policy
//!
//! A Gaussian narrower than the sampling grid cannot be represented; below a
//! fixed cutoff `sigma <= SIGMA_CUTOFF` the op is the **identity** (a radius-0,
//! single-tap unit kernel). This makes the σ→0 limit well defined and exact
//! rather than an ever-shrinking-but-nonzero blur.
//!
//! # Boundary
//!
//! The boundary `mode` is forwarded verbatim to the convolution
//! (clamp/mirror/wrap/constant/transparent), defaulting to `clamp`.
//!
//! # Determinism
//!
//! The reference kernel is a fixed-order `f64` accumulation; alternate
//! (separable / FFT) implementations agree only within a discretization bound, so
//! the op declares [`Bounded`](DeterminismTier::Bounded). Correctness is the
//! analytic Gaussian property set (unit-sum, constant preservation, rotational
//! isotropy, the σ-semigroup, the impulse-variance match, and the σ→0 identity),
//! not a perceptual metric.

use paintop_core::executor::{InputValues, OpImplementation, OutputValues};
use paintop_ir::{
    AssertionResult, Descriptors, DeterminismTier, Error, ErrorClass, ErrorContext, ImplId,
    InputRegions, InputSpec, OpContract, OperationManifest, OutputDescriptors, OutputRegions,
    OutputSpec, ParamSpec, ParamType, ParamUnit, ResourceDescriptor, ResourceKind, Result,
    RoiCategory, RoiPolicy, TestMetadata,
};

use crate::convolve::{Convolve, ConvolveRequest};

/// The canonical id of the Gaussian-blur operation.
pub const GAUSSIAN_BLUR_OP_ID: &str = "filter.gaussian_blur@1";

/// The `input` was absent or carried an unsupported descriptor.
pub const E_BLUR_INPUT: &str = "E_BLUR_INPUT";

/// The `sigma` / `mode` parameters were missing or malformed.
pub const E_BLUR_PARAM: &str = "E_BLUR_PARAM";

/// Below this `sigma` the blur is the identity (the σ→0 cutoff policy). A
/// Gaussian this narrow is sub-pixel and indistinguishable from a delta on the
/// sampling grid.
pub const SIGMA_CUTOFF: f64 = 1.0e-3;

/// The default upper bound on `sigma` (`OP_CATALOG` §8 `sigma_max_default`),
/// keeping kernels finite for an unbounded request.
pub const SIGMA_MAX_DEFAULT: f64 = 512.0;

/// Parse and validate the required, positive, finite `sigma`.
fn sigma_param(params: &serde_json::Value) -> Result<f64> {
    let value = params.get("sigma").ok_or_else(|| {
        Error::new(
            ErrorClass::Schema,
            E_BLUR_PARAM,
            "filter.gaussian_blur requires a `sigma` parameter".to_owned(),
        )
    })?;
    let sigma = value.as_f64().ok_or_else(|| {
        Error::new(
            ErrorClass::Schema,
            E_BLUR_PARAM,
            "filter.gaussian_blur `sigma` must be a number".to_owned(),
        )
        .with_context(ErrorContext::default().with_actual(value.to_string()))
    })?;
    if !sigma.is_finite() || sigma < 0.0 {
        return Err(Error::new(
            ErrorClass::Schema,
            E_BLUR_PARAM,
            format!(
                "filter.gaussian_blur `sigma` must be a finite, non-negative number, got {sigma}"
            ),
        ));
    }
    let sigma_max = params.get("sigma_max").map_or(Ok(SIGMA_MAX_DEFAULT), |v| {
        v.as_f64().ok_or_else(|| {
            Error::new(
                ErrorClass::Schema,
                E_BLUR_PARAM,
                "filter.gaussian_blur `sigma_max` must be a number".to_owned(),
            )
            .with_context(ErrorContext::default().with_actual(v.to_string()))
        })
    })?;
    if sigma > sigma_max {
        return Err(Error::new(
            ErrorClass::Policy,
            E_BLUR_PARAM,
            format!("filter.gaussian_blur `sigma` {sigma} exceeds the limit {sigma_max}"),
        )
        .with_context(
            ErrorContext::default()
                .with_actual(sigma.to_string())
                .with_expected(format!("<= {sigma_max}")),
        ));
    }
    Ok(sigma)
}

/// Parse the optional boundary `mode`, defaulting to `clamp`; the token is
/// validated by reusing the convolution's mode vocabulary downstream.
fn mode_param(params: &serde_json::Value) -> String {
    params
        .get("mode")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("clamp")
        .to_owned()
}

/// The kernel radius for a `sigma`: `ceil(3σ)`, or `0` under the σ→0 cutoff.
fn kernel_radius(sigma: f64) -> u32 {
    if sigma <= SIGMA_CUTOFF {
        return 0;
    }
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "3*sigma is positive and bounded by 3*sigma_max, well within u32"
    )]
    let r = (3.0 * sigma).ceil() as u32;
    r.max(1)
}

/// Build the normalized, row-major Gaussian kernel object for `sigma`, suitable
/// as the `kernel` param of [`Convolve`]. Returns the side and the kernel JSON.
fn gaussian_kernel(sigma: f64) -> serde_json::Value {
    let r = kernel_radius(sigma);
    let side = 2 * r + 1;
    if r == 0 {
        // σ→0 cutoff: a 1x1 unit kernel (the identity).
        return serde_json::json!({
            "width": 1, "height": 1, "origin_x": 0, "origin_y": 0, "weights": [1.0]
        });
    }
    let ri = i64::from(r);
    let two_sigma_sq = 2.0 * sigma * sigma;
    let mut weights = Vec::with_capacity((side * side) as usize);
    let mut sum = 0.0_f64;
    for dy in -ri..=ri {
        for dx in -ri..=ri {
            #[allow(
                clippy::cast_precision_loss,
                reason = "dx, dy are small kernel offsets bounded by 3*sigma_max"
            )]
            let r2 = (dx * dx + dy * dy) as f64;
            let w = (-r2 / two_sigma_sq).exp();
            sum += w;
            weights.push(w);
        }
    }
    // Normalize to unit sum so the kernel preserves a constant.
    for w in &mut weights {
        *w /= sum;
    }
    serde_json::json!({
        "width": side, "height": side, "origin_x": r, "origin_y": r, "weights": weights
    })
}

/// Build the full convolution params (kernel + boundary mode + per-channel
/// constant) for a blur request, so the blur delegates to [`Convolve`].
fn convolve_params(sigma: f64, mode: &str, channels: u32) -> serde_json::Value {
    let mut params = serde_json::json!({
        "kernel": gaussian_kernel(sigma),
        "mode": mode,
    });
    // For `constant` mode, blur against an all-zero border (the only sensible
    // default for a normalized smoothing kernel).
    if mode == "constant"
        && let Some(obj) = params.as_object_mut()
    {
        obj.insert(
            "value".to_owned(),
            serde_json::Value::Array(vec![serde_json::json!(0.0); channels as usize]),
        );
    }
    params
}

/// The `filter.gaussian_blur@1` operation.
#[derive(Debug, Clone, Copy, Default)]
pub struct GaussianBlur;

impl GaussianBlur {
    /// Construct the Gaussian-blur operation.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    /// The declared manifest for `filter.gaussian_blur@1`.
    ///
    /// # Errors
    /// Propagates the [`schema`](ErrorClass::Schema) error if the hard-coded op /
    /// impl ids are somehow invalid (they are not).
    pub fn manifest() -> Result<OperationManifest> {
        Ok(OperationManifest {
            id: GAUSSIAN_BLUR_OP_ID.parse()?,
            impl_version: 1,
            summary:
                "Isotropic Gaussian blur: a normalized sampled Gaussian (radius ceil(3*sigma), \
                      sigma->0 identity cutoff) convolved through the direct convolution reference."
                    .to_owned(),
            determinism: DeterminismTier::Bounded,
            roi: RoiPolicy {
                category: RoiCategory::Geometric,
                halo_px: None,
            },
            inputs: vec![InputSpec {
                name: "input".to_owned(),
                kind: ResourceKind::Image,
                required: true,
                doc: "The Image or Field1 to blur (each channel blurred independently).".to_owned(),
            }],
            outputs: vec![OutputSpec {
                name: "output".to_owned(),
                kind: ResourceKind::Image,
                doc: "The blurred Image or Field1 (same kind and extent as the input).".to_owned(),
            }],
            params: vec![
                ParamSpec {
                    name: "sigma".to_owned(),
                    ty: ParamType::Float,
                    unit: Some(ParamUnit::Pixels),
                    required: true,
                    default: None,
                    choices: vec![],
                    doc:
                        "The Gaussian standard deviation in pixels; sigma <= 1e-3 is the identity."
                            .to_owned(),
                },
                ParamSpec {
                    name: "mode".to_owned(),
                    ty: ParamType::String,
                    unit: None,
                    required: false,
                    default: Some(serde_json::json!("clamp")),
                    choices: vec![
                        "constant".to_owned(),
                        "transparent".to_owned(),
                        "clamp".to_owned(),
                        "mirror".to_owned(),
                        "wrap".to_owned(),
                    ],
                    doc: "The boundary mode forwarded to the convolution.".to_owned(),
                },
                ParamSpec {
                    name: "sigma_max".to_owned(),
                    ty: ParamType::Float,
                    unit: Some(ParamUnit::Pixels),
                    required: false,
                    default: Some(serde_json::json!(SIGMA_MAX_DEFAULT)),
                    choices: vec![],
                    doc: "Upper bound on sigma; a larger request is rejected.".to_owned(),
                },
            ],
            implementations: vec![reference_impl()?],
            test: blur_test_metadata(),
        })
    }
}

/// Resolve the blur params and channel count for an input, delegating to the
/// convolution to build the full request.
fn build_convolve_params(params: &serde_json::Value, channels: u32) -> Result<serde_json::Value> {
    let sigma = sigma_param(params)?;
    let mode = mode_param(params);
    let conv = convolve_params(sigma, &mode, channels);
    // Validate the forwarded mode by parsing it through the convolution's
    // resolver, so an unknown boundary mode is rejected here with the blur's
    // error code rather than surfacing only at compute time.
    ConvolveRequest::resolve(&conv, channels).map_err(|e| {
        Error::new(
            ErrorClass::Schema,
            E_BLUR_PARAM,
            format!(
                "filter.gaussian_blur boundary mode is invalid: {}",
                e.message
            ),
        )
    })?;
    Ok(conv)
}

/// The interleaved channel count of a supported input descriptor.
fn input_channels(descriptor: &ResourceDescriptor) -> Result<u32> {
    match descriptor {
        ResourceDescriptor::Image(d) => Ok(d.layout.channel_count()),
        ResourceDescriptor::Field1(_) => Ok(1),
        _ => Err(Error::new(
            ErrorClass::Type,
            E_BLUR_INPUT,
            "filter.gaussian_blur `input` must be an Image or Field1 resource".to_owned(),
        )),
    }
}

impl OpContract for GaussianBlur {
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
                E_BLUR_INPUT,
                "filter.gaussian_blur requires an `input` resource".to_owned(),
            )
        })?;
        let channels = input_channels(input)?;
        // Validate sigma/mode up front (same extent in, same extent out).
        let _ = build_convolve_params(params, channels)?;
        let mut out = OutputDescriptors::new();
        out.insert("output".to_owned(), *input);
        Ok(out)
    }

    fn required_inputs(
        &self,
        requested_outputs: &OutputRegions,
        inputs: &Descriptors,
        params: &serde_json::Value,
    ) -> Result<InputRegions> {
        // Delegate the ROI footprint to the underlying convolution with the
        // Gaussian kernel, so the declared halo (ceil(3*sigma)) is honoured.
        let input = inputs.get("input").ok_or_else(|| {
            Error::new(
                ErrorClass::Reference,
                E_BLUR_INPUT,
                "filter.gaussian_blur requires an `input` resource".to_owned(),
            )
        })?;
        let channels = input_channels(input)?;
        let conv = build_convolve_params(params, channels)?;
        Convolve::new().required_inputs(requested_outputs, inputs, &conv)
    }

    fn validate_postconditions(
        &self,
        outputs: &OutputDescriptors,
        _params: &serde_json::Value,
    ) -> Result<Vec<AssertionResult>> {
        Ok(vec![match outputs.get("output") {
            Some(ResourceDescriptor::Image(_) | ResourceDescriptor::Field1(_)) => {
                AssertionResult::pass("produces_blurred")
            }
            _ => AssertionResult::fail("produces_blurred", "no `output` Image/Field1 produced"),
        }])
    }
}

impl OpImplementation for GaussianBlur {
    fn compute(
        &self,
        inputs: &InputValues,
        params: &serde_json::Value,
    ) -> std::result::Result<OutputValues, Error> {
        let input = inputs.get("input").ok_or_else(|| {
            Error::new(
                ErrorClass::Reference,
                E_BLUR_INPUT,
                "filter.gaussian_blur requires an `input` value".to_owned(),
            )
        })?;
        let channels = input.channels();
        let conv = build_convolve_params(params, channels)?;
        Convolve::new().compute(inputs, &conv)
    }
}

/// The mandatory `cpu.reference@1` oracle implementation id.
fn reference_impl() -> Result<ImplId> {
    ImplId::new("cpu", "reference", 1)
}

/// Verification declarations for `filter.gaussian_blur@1`: a bounded,
/// single-reference neighbourhood op. Differential does not apply (one
/// implementation). Perceptual is not applicable: correctness is the analytic
/// Gaussian property set (unit-sum positive kernel, constant preservation, 90°
/// isotropy, the σ-semigroup, the blurred-impulse variance match, and the σ→0
/// identity), plus a differential check against `filter.convolve` with the same
/// kernel — not a perceptual-quality metric.
fn blur_test_metadata() -> TestMetadata {
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
            "filter.gaussian_blur convolves a normalized sampled Gaussian verified by analytic \
             properties (unit-sum positive kernel, constant preservation, 90-degree isotropy, the \
             sigma-semigroup G_s1*G_s2 ~ G_sqrt(s1^2+s2^2), the blurred-impulse variance match, \
             and the sigma->0 identity cutoff) and a differential check against filter.convolve \
             with the same kernel; there is no perceptual-quality metric to apply",
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
