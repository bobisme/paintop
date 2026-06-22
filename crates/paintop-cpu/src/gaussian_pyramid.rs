//! The `frequency.gaussian_pyramid@1` operation (`OP_CATALOG` §13).
//!
//! Builds a multi-resolution Gaussian pyramid from an Image or `Field1`: level
//! `0` is the input verbatim, and each deeper level is the previous level
//! smoothed by a fixed Gaussian and decimated 2:1 under the declared
//! [`PyramidPhase`] odd-size rounding rule. The output is a single
//! [`Pyramid`](paintop_ir::ResourceDescriptor::Pyramid) resource whose levels
//! are stored finest-first in one concatenated buffer.
//!
//! # Convention
//!
//! - **Downsample**: factor-2 per level ([`DownsampleFactor::Half`]), the only
//!   factor M4 defines.
//! - **Pre-blur**: a fixed separable Gaussian of σ
//!   ([`DEFAULT_PYRAMID_SIGMA`], a documented override) with a clamp boundary —
//!   the anti-alias low-pass that precedes every decimation, so the pyramid does
//!   not alias.
//! - **Phase**: the odd-size rounding rule (`floor`, the classical convention,
//!   or `ceil`) is an explicit parameter; level extents are exactly those
//!   [`PyramidDescriptor::level_extent`] derives.
//!
//! # Determinism
//!
//! Every level is a fixed-order `f64` separable blur followed by a deterministic
//! even-sample decimation, so the pyramid is bit-identical across reruns on a
//! fixed backend (the M4 reproducibility criterion). The op is
//! [`Bounded`](DeterminismTier::Bounded): an alternate (e.g. separable-on-GPU)
//! backend agrees only within the Gaussian discretization bound.

use paintop_core::executor::{InputValues, OpImplementation, OutputValues, ResourceValue};
use paintop_ir::{
    AssertionResult, Descriptors, DeterminismTier, DownsampleFactor, Error, ErrorClass,
    ErrorContext, Extent, ImplId, InputRegions, InputSpec, MAX_PYRAMID_LEVELS, OpContract,
    OperationManifest, OutputDescriptors, OutputRegions, OutputSpec, ParamSpec, ParamType,
    ParamUnit, PyramidDescriptor, PyramidPhase, Rect, ResourceDescriptor, ResourceKind, Result,
    RoiCategory, RoiPolicy, ScalarType, TestMetadata,
};

use crate::frequency::{DEFAULT_PYRAMID_SIGMA, child_extent, downsample};

/// The canonical id of the Gaussian-pyramid operation.
pub const GAUSSIAN_PYRAMID_OP_ID: &str = "frequency.gaussian_pyramid@1";

/// The `input` was absent or carried an unsupported descriptor.
pub const E_PYRAMID_INPUT: &str = "E_PYRAMID_INPUT";

/// A parameter was missing, the wrong type, or out of range.
pub const E_PYRAMID_PARAM: &str = "E_PYRAMID_PARAM";

/// The upper bound on the pre-blur `sigma`, keeping the smoothing kernel finite.
pub const SIGMA_MAX: f64 = 64.0;

/// The interleaved channel count of a supported input descriptor (Image or
/// `Field1`).
fn input_channels(descriptor: &ResourceDescriptor) -> Result<u32> {
    match descriptor {
        ResourceDescriptor::Image(d) => Ok(d.layout.channel_count()),
        ResourceDescriptor::Field1(_) => Ok(1),
        _ => Err(Error::new(
            ErrorClass::Type,
            E_PYRAMID_INPUT,
            "frequency.gaussian_pyramid `input` must be an Image or Field1 resource".to_owned(),
        )),
    }
}

/// The resolved pyramid build parameters.
#[derive(Debug, Clone, Copy)]
pub struct PyramidParams {
    /// The number of levels, level 0 the base (`1..=MAX_PYRAMID_LEVELS`).
    pub levels: u32,
    /// The pre-decimation Gaussian standard deviation in pixels.
    pub sigma: f64,
    /// The odd-size rounding rule.
    pub phase: PyramidPhase,
}

impl PyramidParams {
    /// Resolve and validate `levels`, `sigma`, and `phase` from the param object.
    ///
    /// # Errors
    /// Returns a [`schema`](ErrorClass::Schema) error for a missing/ill-typed
    /// parameter, a zero or over-deep level count, a non-finite or negative
    /// sigma, an over-large sigma, or an unknown phase token.
    pub fn resolve(params: &serde_json::Value) -> Result<Self> {
        let levels = params
            .get("levels")
            .ok_or_else(|| param_err("requires a `levels` parameter"))?
            .as_u64()
            .ok_or_else(|| param_err("`levels` must be a positive integer"))?;
        if levels == 0 || levels > u64::from(MAX_PYRAMID_LEVELS) {
            return Err(
                param_err(&format!("`levels` must be in 1..={MAX_PYRAMID_LEVELS}"))
                    .with_context(ErrorContext::default().with_actual(levels.to_string())),
            );
        }
        #[allow(
            clippy::cast_possible_truncation,
            reason = "levels is validated <= MAX_PYRAMID_LEVELS, well within u32"
        )]
        let levels = levels as u32;

        let sigma = match params.get("sigma") {
            None => DEFAULT_PYRAMID_SIGMA,
            Some(v) => v
                .as_f64()
                .ok_or_else(|| param_err("`sigma` must be a number"))?,
        };
        if !sigma.is_finite() || sigma < 0.0 {
            return Err(param_err("`sigma` must be a finite, non-negative number")
                .with_context(ErrorContext::default().with_actual(sigma.to_string())));
        }
        if sigma > SIGMA_MAX {
            return Err(param_err(&format!("`sigma` must not exceed {SIGMA_MAX}"))
                .with_context(ErrorContext::default().with_actual(sigma.to_string())));
        }

        let phase = match params.get("phase").and_then(serde_json::Value::as_str) {
            None | Some("floor") => PyramidPhase::Floor,
            Some("ceil") => PyramidPhase::Ceil,
            Some(other) => {
                return Err(param_err("`phase` must be `floor` or `ceil`")
                    .with_context(ErrorContext::default().with_actual(other.to_owned())));
            }
        };

        Ok(Self {
            levels,
            sigma,
            phase,
        })
    }
}

/// Build a schema [`Error`] for a malformed pyramid parameter.
fn param_err(detail: &str) -> Error {
    Error::new(
        ErrorClass::Schema,
        E_PYRAMID_PARAM,
        format!("frequency.gaussian_pyramid {detail}"),
    )
}

/// The [`PyramidDescriptor`] a build with `params` produces from an input of
/// `extent`/`channels`/`scalar`.
const fn pyramid_descriptor(
    extent: Extent,
    channels: u32,
    params: PyramidParams,
) -> PyramidDescriptor {
    PyramidDescriptor {
        base_extent: extent,
        levels: params.levels,
        channels,
        scalar: ScalarType::F32,
        factor: DownsampleFactor::Half,
        phase: params.phase,
        coordinates: paintop_ir::CoordinateConvention::PixelCenterUpperLeft,
    }
}

/// Build the concatenated, finest-first pyramid sample buffer from a base plane.
///
/// Level 0 is the base samples verbatim; each deeper level is the previous
/// level's [`downsample`] (pre-blur + 2:1 decimation) under the phase. The
/// running extents follow [`child_extent`], matching the descriptor's derived
/// chain exactly.
#[must_use]
pub fn build_pyramid_samples(
    base: &[f32],
    extent: Extent,
    channels: u32,
    params: PyramidParams,
) -> Vec<f32> {
    let mut out = Vec::new();
    out.extend_from_slice(base);
    let mut level_samples = base.to_vec();
    let mut level_extent = extent;
    for _ in 1..params.levels {
        let child = downsample(
            &level_samples,
            level_extent,
            channels,
            params.sigma,
            params.phase,
        );
        out.extend_from_slice(&child);
        level_samples = child;
        level_extent = child_extent(level_extent, params.phase);
    }
    out
}

/// The `frequency.gaussian_pyramid@1` operation.
#[derive(Debug, Clone, Copy, Default)]
pub struct GaussianPyramid;

impl GaussianPyramid {
    /// Construct the Gaussian-pyramid operation.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    /// The declared manifest for `frequency.gaussian_pyramid@1`.
    ///
    /// # Errors
    /// Propagates the [`schema`](ErrorClass::Schema) error if the hard-coded op /
    /// impl ids are somehow invalid (they are not).
    pub fn manifest() -> Result<OperationManifest> {
        Ok(OperationManifest {
            id: GAUSSIAN_PYRAMID_OP_ID.parse()?,
            impl_version: 1,
            summary: "Build a multi-resolution Gaussian pyramid: level 0 is the input, each \
                      deeper level a fixed separable Gaussian pre-blur decimated 2:1 under the \
                      declared floor/ceil odd-size phase. Output is a single Pyramid resource."
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
                doc: "The Image or Field1 base (level 0) of the pyramid.".to_owned(),
            }],
            outputs: vec![OutputSpec {
                name: "pyramid".to_owned(),
                kind: ResourceKind::Pyramid,
                doc: "The multi-resolution Gaussian pyramid (level 0 = input).".to_owned(),
            }],
            params: vec![
                ParamSpec {
                    name: "levels".to_owned(),
                    ty: ParamType::Integer,
                    unit: None,
                    required: true,
                    default: None,
                    choices: vec![],
                    doc: format!(
                        "The number of levels, level 0 the base (1..={MAX_PYRAMID_LEVELS})."
                    ),
                },
                ParamSpec {
                    name: "sigma".to_owned(),
                    ty: ParamType::Float,
                    unit: Some(ParamUnit::Pixels),
                    required: false,
                    default: Some(serde_json::json!(DEFAULT_PYRAMID_SIGMA)),
                    choices: vec![],
                    doc: "The pre-decimation Gaussian standard deviation (anti-alias low-pass)."
                        .to_owned(),
                },
                ParamSpec {
                    name: "phase".to_owned(),
                    ty: ParamType::String,
                    unit: None,
                    required: false,
                    default: Some(serde_json::json!("floor")),
                    choices: vec!["floor".to_owned(), "ceil".to_owned()],
                    doc: "The odd-size rounding rule mapping a parent extent to its child."
                        .to_owned(),
                },
            ],
            implementations: vec![reference_impl()?],
            test: pyramid_test_metadata(),
        })
    }
}

impl OpContract for GaussianPyramid {
    fn declared_inputs(&self) -> Vec<(String, ResourceKind)> {
        vec![("input".to_owned(), ResourceKind::Image)]
    }

    fn declared_outputs(&self) -> Vec<(String, ResourceKind)> {
        vec![("pyramid".to_owned(), ResourceKind::Pyramid)]
    }

    fn infer_outputs(
        &self,
        inputs: &Descriptors,
        params: &serde_json::Value,
    ) -> Result<OutputDescriptors> {
        let input = inputs.get("input").ok_or_else(|| {
            Error::new(
                ErrorClass::Reference,
                E_PYRAMID_INPUT,
                "frequency.gaussian_pyramid requires an `input` resource".to_owned(),
            )
        })?;
        let channels = input_channels(input)?;
        let resolved = PyramidParams::resolve(params)?;
        let descriptor = pyramid_descriptor(input.extent(), channels, resolved);
        // Validate the derived level chain (overflow, degenerate sizes).
        descriptor.validate()?;
        let mut out = OutputDescriptors::new();
        out.insert(
            "pyramid".to_owned(),
            ResourceDescriptor::Pyramid(descriptor),
        );
        Ok(out)
    }

    fn required_inputs(
        &self,
        _requested_outputs: &OutputRegions,
        inputs: &Descriptors,
        _params: &serde_json::Value,
    ) -> Result<InputRegions> {
        // The pyramid reads the whole base to build every level.
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
        Ok(vec![match outputs.get("pyramid") {
            Some(ResourceDescriptor::Pyramid(_)) => AssertionResult::pass("produces_pyramid"),
            _ => AssertionResult::fail("produces_pyramid", "no `pyramid` output produced"),
        }])
    }
}

impl OpImplementation for GaussianPyramid {
    fn compute(
        &self,
        inputs: &InputValues,
        params: &serde_json::Value,
    ) -> std::result::Result<OutputValues, Error> {
        let input = inputs.get("input").ok_or_else(|| {
            Error::new(
                ErrorClass::Reference,
                E_PYRAMID_INPUT,
                "frequency.gaussian_pyramid requires an `input` value".to_owned(),
            )
        })?;
        let channels = input.channels();
        let extent = input.extent();
        let resolved = PyramidParams::resolve(params)?;
        let descriptor = pyramid_descriptor(extent, channels, resolved);
        descriptor.validate()?;

        let samples = build_pyramid_samples(input.samples(), extent, channels, resolved);
        let value = ResourceValue::pyramid(descriptor, samples).map_err(|actual| {
            Error::new(
                ErrorClass::Execution,
                E_PYRAMID_INPUT,
                format!(
                    "frequency.gaussian_pyramid produced a pyramid buffer of unexpected length \
                     {actual}"
                ),
            )
        })?;
        let mut out = OutputValues::new();
        out.insert("pyramid".to_owned(), value);
        Ok(out)
    }
}

/// The mandatory `cpu.reference@1` oracle implementation id.
fn reference_impl() -> Result<ImplId> {
    ImplId::new("cpu", "reference", 1)
}

/// Verification declarations for `frequency.gaussian_pyramid@1`: a bounded
/// multi-resolution op with analytic fixtures (constant-preservation, derived
/// level extents, the impulse/step convention) and property tests
/// (determinism, level-extent agreement). Differential applies (a single
/// reference today, so it is declared not-applicable with a reason); perceptual
/// does not apply — correctness is the level-extent / constant-preservation
/// property set, not a perceptual metric.
fn pyramid_test_metadata() -> TestMetadata {
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
            "the Gaussian pyramid is verified by the level-extent convention, constant \
             preservation, and the deterministic blur+decimate chain, not a perceptual metric",
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
