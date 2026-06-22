//! The `filter.structure_tensor@1` operation: the per-pixel **structure tensor**.
//!
//! Computes the structure tensor of an `Image` or scalar `Field1` at a chosen
//! gradient scale and smoothing scale (`OP_CATALOG` §8, §10.4; `plan.md` §1428).
//!
//! The structure tensor `J` summarizes the local first-order geometry of a scalar
//! signal. For a scalar plane `f` with gradient `∇f = (f_x, f_y)` the raw tensor
//! at a pixel is the outer product
//!
//! ```text
//! J0 = ∇f ∇fᵀ = [ f_x²    f_x·f_y ]
//!               [ f_x·f_y f_y²    ]
//! ```
//!
//! and the *structure tensor* is that outer product smoothed over a neighbourhood,
//! `J = G_ρ * J0`, with `ρ` the **smoothing scale** (`integration_sigma`). The
//! gradient itself is taken after an optional pre-smoothing at the **gradient
//! scale** (`gradient_sigma`), so the two scales are independent (Weickert's
//! construction): the gradient scale sets the feature size the gradient responds
//! to; the smoothing scale sets the window the orientation is averaged over.
//!
//! # Output
//!
//! The op produces a three-component [`Field3`](paintop_ir::ResourceKind::Field3)
//! carrying the **symmetric** tensor's three independent entries per pixel, in the
//! fixed order `(Jxx, Jxy, Jyy)`. A symmetric 2×2 tensor has exactly these three
//! degrees of freedom, so `Field3` is its natural lossless carrier; the consumer
//! ([`field.orientation`](crate::orientation)) reconstructs `[[Jxx, Jxy],[Jxy,
//! Jyy]]` from it.
//!
//! # Multi-channel input
//!
//! For a multi-channel `Image` the per-channel raw tensors are **summed** (the
//! standard colour structure tensor, Di Zenzo): `J0 = Σ_c ∇f_c ∇f_cᵀ`. This is
//! the rotationally-covariant generalization of the scalar tensor and reduces to
//! it for a single channel. Alpha, if present, participates like any other
//! channel (the op operates on the raw stored samples, matching
//! [`filter.convolve`](crate::convolve)).
//!
//! # Gradient operator
//!
//! The gradient uses the **central difference** `f_x(x) = (f(x+1) − f(x−1)) / 2`
//! (and likewise `f_y`), under a `clamp` boundary. Central differences are the
//! smallest exactly-antisymmetric, rotation-pair-consistent gradient stencil, so a
//! 90° image rotation rotates the tensor consistently (the metamorphic check). The
//! optional gradient-scale and the smoothing-scale Gaussians are the same
//! normalized separable Gaussian as [`filter.gaussian_blur`](crate::gaussian_blur)
//! (radius `ceil(3σ)`, σ→0 identity cutoff), applied per plane.
//!
//! # Determinism
//!
//! Every stage is a fixed-order `f64` accumulation rounded once to `f32`: the
//! per-channel gradient, the outer-product sum, and the two separable Gaussian
//! passes all run in a pinned order, so the op is bit-identical on reruns of a
//! fixed backend. Because the Gaussian smoothing reassociates an `f64` sum the
//! op declares [`Bounded`](DeterminismTier::Bounded) (it agrees with an
//! independent direct-convolution reference only within a discretization bound),
//! but it is *exactly reproducible* against itself.

use paintop_core::executor::{InputValues, OpImplementation, OutputValues, ResourceValue};
use paintop_ir::{
    AssertionResult, CoordinateConvention, Descriptors, DeterminismTier, Error, ErrorClass,
    ErrorContext, Extent, FieldArity, FieldDescriptor, ImplId, InputRegions, InputSpec, OpContract,
    OperationManifest, OutputDescriptors, OutputRegions, OutputSpec, ParamSpec, ParamType,
    ParamUnit, Rect, ResourceDescriptor, ResourceKind, Result, RoiCategory, RoiPolicy, ScalarType,
    SemanticRole, TestMetadata,
};

/// The canonical id of the structure-tensor operation.
pub const STRUCTURE_TENSOR_OP_ID: &str = "filter.structure_tensor@1";

/// The `input` was absent or carried an unsupported descriptor.
pub const E_STRUCTURE_TENSOR_INPUT: &str = "E_STRUCTURE_TENSOR_INPUT";

/// A scale parameter was missing, malformed, or out of range.
pub const E_STRUCTURE_TENSOR_PARAM: &str = "E_STRUCTURE_TENSOR_PARAM";

/// Below this σ a Gaussian is sub-pixel and the smoothing stage is the identity
/// (matching [`filter.gaussian_blur`](crate::gaussian_blur)'s cutoff).
pub const SIGMA_CUTOFF: f64 = 1.0e-3;

/// The upper bound on either scale, keeping the separable kernels finite.
pub const SIGMA_MAX: f64 = 256.0;

/// The two scales of a structure-tensor request.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Scales {
    /// Pre-smoothing applied before differentiation (the gradient scale).
    gradient_sigma: f64,
    /// Smoothing applied to the outer-product components (the integration scale).
    integration_sigma: f64,
}

impl Scales {
    /// Parse and validate both scales.
    fn resolve(params: &serde_json::Value) -> Result<Self> {
        let gradient_sigma = optional_sigma(params, "gradient_sigma", 0.0)?;
        let integration_sigma = optional_sigma(params, "integration_sigma", 2.0)?;
        Ok(Self {
            gradient_sigma,
            integration_sigma,
        })
    }
}

/// Parse an optional, finite, non-negative, bounded σ param, defaulting when
/// absent.
fn optional_sigma(params: &serde_json::Value, name: &str, default: f64) -> Result<f64> {
    let Some(value) = params.get(name) else {
        return Ok(default);
    };
    let sigma = value.as_f64().ok_or_else(|| {
        Error::new(
            ErrorClass::Schema,
            E_STRUCTURE_TENSOR_PARAM,
            format!("filter.structure_tensor `{name}` must be a number"),
        )
        .with_context(ErrorContext::default().with_actual(value.to_string()))
    })?;
    if !sigma.is_finite() || sigma < 0.0 {
        return Err(Error::new(
            ErrorClass::Schema,
            E_STRUCTURE_TENSOR_PARAM,
            format!(
                "filter.structure_tensor `{name}` must be finite and non-negative, got {sigma}"
            ),
        ));
    }
    if sigma > SIGMA_MAX {
        return Err(Error::new(
            ErrorClass::Policy,
            E_STRUCTURE_TENSOR_PARAM,
            format!("filter.structure_tensor `{name}` {sigma} exceeds the limit {SIGMA_MAX}"),
        )
        .with_context(
            ErrorContext::default()
                .with_actual(sigma.to_string())
                .with_expected(format!("<= {SIGMA_MAX}")),
        ));
    }
    Ok(sigma)
}

/// The `filter.structure_tensor@1` operation.
#[derive(Debug, Clone, Copy, Default)]
pub struct StructureTensor;

impl StructureTensor {
    /// Construct the structure-tensor operation.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    /// The declared manifest for `filter.structure_tensor@1`.
    ///
    /// # Errors
    /// Propagates the [`schema`](ErrorClass::Schema) error if the hard-coded op /
    /// impl ids are somehow invalid (they are not).
    pub fn manifest() -> Result<OperationManifest> {
        Ok(OperationManifest {
            id: STRUCTURE_TENSOR_OP_ID.parse()?,
            impl_version: 1,
            summary: "Per-pixel structure tensor of an Image or Field1: central-difference \
                      gradients at a gradient scale, the per-channel outer-product summed, then \
                      Gaussian-smoothed at an integration scale; output is the symmetric tensor's \
                      (Jxx, Jxy, Jyy) as a Field3."
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
                doc: "The Image or Field1 to analyze (channels summed for the colour tensor)."
                    .to_owned(),
            }],
            outputs: vec![OutputSpec {
                name: "tensor".to_owned(),
                kind: ResourceKind::Field3,
                doc: "The structure tensor's three independent entries (Jxx, Jxy, Jyy) per pixel."
                    .to_owned(),
            }],
            params: vec![
                ParamSpec {
                    name: "gradient_sigma".to_owned(),
                    ty: ParamType::Float,
                    unit: Some(ParamUnit::Pixels),
                    required: false,
                    default: Some(serde_json::json!(0.0)),
                    choices: vec![],
                    doc: "Pre-smoothing applied before differentiation; 0 differentiates the raw \
                          input."
                        .to_owned(),
                },
                ParamSpec {
                    name: "integration_sigma".to_owned(),
                    ty: ParamType::Float,
                    unit: Some(ParamUnit::Pixels),
                    required: false,
                    default: Some(serde_json::json!(2.0)),
                    choices: vec![],
                    doc: "Gaussian smoothing applied to the outer-product components (the window \
                          orientation is averaged over)."
                        .to_owned(),
                },
            ],
            implementations: vec![reference_impl()?],
            test: structure_tensor_test_metadata(),
        })
    }
}

/// The supported input descriptor's extent and interleaved channel count.
fn input_extent_channels(descriptor: &ResourceDescriptor) -> Result<(Extent, u32)> {
    match descriptor {
        ResourceDescriptor::Image(d) => Ok((d.extent, d.layout.channel_count())),
        ResourceDescriptor::Field1(d) => Ok((d.extent, 1)),
        _ => Err(Error::new(
            ErrorClass::Type,
            E_STRUCTURE_TENSOR_INPUT,
            "filter.structure_tensor `input` must be an Image or Field1 resource".to_owned(),
        )),
    }
}

/// The `Field3` descriptor a structure tensor produces for `extent`: a generic
/// 2-vector-derived feature field (the symmetric tensor's three entries),
/// unconstrained.
const fn tensor_descriptor(extent: Extent) -> FieldDescriptor {
    FieldDescriptor {
        arity: FieldArity::Field3,
        extent,
        scalar: ScalarType::F32,
        semantic: SemanticRole::Material,
        coordinates: CoordinateConvention::PixelCenterUpperLeft,
        space: None,
        normalization: None,
        encoding: None,
    }
}

impl OpContract for StructureTensor {
    fn declared_inputs(&self) -> Vec<(String, ResourceKind)> {
        vec![("input".to_owned(), ResourceKind::Image)]
    }

    fn declared_outputs(&self) -> Vec<(String, ResourceKind)> {
        vec![("tensor".to_owned(), ResourceKind::Field3)]
    }

    fn infer_outputs(
        &self,
        inputs: &Descriptors,
        params: &serde_json::Value,
    ) -> Result<OutputDescriptors> {
        let input = inputs.get("input").ok_or_else(|| {
            Error::new(
                ErrorClass::Reference,
                E_STRUCTURE_TENSOR_INPUT,
                "filter.structure_tensor requires an `input` resource".to_owned(),
            )
        })?;
        let (extent, _channels) = input_extent_channels(input)?;
        // Validate the scales up front so a bad request fails type-checking.
        Scales::resolve(params)?;
        let mut out = OutputDescriptors::new();
        out.insert(
            "tensor".to_owned(),
            ResourceDescriptor::Field3(tensor_descriptor(extent)),
        );
        Ok(out)
    }

    fn required_inputs(
        &self,
        requested_outputs: &OutputRegions,
        inputs: &Descriptors,
        params: &serde_json::Value,
    ) -> Result<InputRegions> {
        // A neighbourhood op: each output pixel reads a window of radius
        // (gradient halo + 1 for the central difference) + integration halo. Under
        // the clamp boundary a border tap can reference an arbitrary edge sample,
        // so demand the whole input plane (intersected with the dilated window).
        let input = inputs.get("input").ok_or_else(|| {
            Error::new(
                ErrorClass::Reference,
                E_STRUCTURE_TENSOR_INPUT,
                "filter.structure_tensor requires an `input` resource".to_owned(),
            )
        })?;
        let (extent, _channels) = input_extent_channels(input)?;
        let scales = Scales::resolve(params)?;
        let halo = i64::from(total_halo(scales));
        let mut regions = InputRegions::new();
        if let Some(region) = requested_outputs.get("tensor") {
            let full = Rect::new(0, 0, i64::from(extent.width), i64::from(extent.height));
            let dilated = Rect::new(
                region.x0 - halo,
                region.y0 - halo,
                region.x1 + halo,
                region.y1 + halo,
            );
            regions.insert("input".to_owned(), dilated.intersect(full));
        }
        Ok(regions)
    }

    fn validate_postconditions(
        &self,
        outputs: &OutputDescriptors,
        _params: &serde_json::Value,
    ) -> Result<Vec<AssertionResult>> {
        Ok(vec![match outputs.get("tensor") {
            Some(ResourceDescriptor::Field3(_)) => AssertionResult::pass("produces_tensor"),
            _ => AssertionResult::fail("produces_tensor", "no `tensor` Field3 produced"),
        }])
    }
}

/// The pixel halo a request's two scales add: the gradient pre-smoothing radius,
/// `+1` for the central-difference stencil, plus the integration radius.
fn total_halo(scales: Scales) -> u32 {
    gaussian_radius(scales.gradient_sigma)
        .saturating_add(1)
        .saturating_add(gaussian_radius(scales.integration_sigma))
}

impl OpImplementation for StructureTensor {
    fn compute(
        &self,
        inputs: &InputValues,
        params: &serde_json::Value,
    ) -> std::result::Result<OutputValues, Error> {
        let input = inputs.get("input").ok_or_else(|| {
            Error::new(
                ErrorClass::Reference,
                E_STRUCTURE_TENSOR_INPUT,
                "filter.structure_tensor requires an `input` value".to_owned(),
            )
        })?;
        let (extent, channels) = input_extent_channels(input.descriptor())?;
        let scales = Scales::resolve(params)?;

        let samples = compute_tensor(input.samples(), extent, channels, scales);
        let descriptor = ResourceDescriptor::Field3(tensor_descriptor(extent));
        let value = ResourceValue::new(descriptor, 3, samples).map_err(|actual| {
            Error::new(
                ErrorClass::Execution,
                E_STRUCTURE_TENSOR_INPUT,
                format!(
                    "filter.structure_tensor produced a tensor buffer of unexpected length {actual}"
                ),
            )
        })?;
        let mut out = OutputValues::new();
        out.insert("tensor".to_owned(), value);
        Ok(out)
    }
}

/// Compute the `(Jxx, Jxy, Jyy)` Field3 buffer (row-major, 3-interleaved) for an
/// interleaved `samples` plane of the given extent and channel count.
pub(crate) fn compute_tensor(
    samples: &[f32],
    extent: Extent,
    channels: u32,
    scales: Scales,
) -> Vec<f32> {
    let w = extent.width as usize;
    let h = extent.height as usize;
    let n = w * h;
    let ch = channels as usize;
    if w == 0 || h == 0 || ch == 0 {
        return Vec::new();
    }

    // Per-channel: extract the plane, pre-smooth at the gradient scale, take
    // central-difference gradients, and accumulate the raw outer-product sum.
    let mut jxx = vec![0.0_f64; n];
    let mut jxy = vec![0.0_f64; n];
    let mut jyy = vec![0.0_f64; n];
    for c in 0..ch {
        let plane = extract_plane(samples, n, ch, c);
        let smoothed = gaussian_smooth(&plane, w, h, scales.gradient_sigma);
        for y in 0..h {
            for x in 0..w {
                let i = y * w + x;
                let fx = central_diff_x(&smoothed, w, h, x, y);
                let fy = central_diff_y(&smoothed, w, h, x, y);
                jxx[i] = fx.mul_add(fx, jxx[i]);
                jxy[i] = fx.mul_add(fy, jxy[i]);
                jyy[i] = fy.mul_add(fy, jyy[i]);
            }
        }
    }

    // Smooth each tensor component at the integration scale.
    let jxx = gaussian_smooth(&jxx, w, h, scales.integration_sigma);
    let jxy = gaussian_smooth(&jxy, w, h, scales.integration_sigma);
    let jyy = gaussian_smooth(&jyy, w, h, scales.integration_sigma);

    // Interleave (Jxx, Jxy, Jyy) into the Field3 buffer.
    let mut out = vec![0.0_f32; n * 3];
    for i in 0..n {
        #[allow(
            clippy::cast_possible_truncation,
            reason = "tensor entries accumulated in f64, stored as the field's f32 sample type"
        )]
        {
            out[i * 3] = jxx[i] as f32;
            out[i * 3 + 1] = jxy[i] as f32;
            out[i * 3 + 2] = jyy[i] as f32;
        }
    }
    out
}

/// Extract channel `c` of an interleaved `samples` plane into a contiguous f64
/// plane of `n` pixels.
fn extract_plane(samples: &[f32], n: usize, channels: usize, c: usize) -> Vec<f64> {
    let mut plane = Vec::with_capacity(n);
    for i in 0..n {
        plane.push(f64::from(samples[i * channels + c]));
    }
    plane
}

/// Clamp a 1-D coordinate into `[0, n)` (replicate-edge boundary), as a `usize`.
fn clamp_index(coord: i64, n: usize) -> usize {
    let last = n.saturating_sub(1);
    let last_i = i64::try_from(last).unwrap_or(i64::MAX);
    let clamped = coord.clamp(0, last_i);
    usize::try_from(clamped).unwrap_or(0)
}

/// The clamped sample of an f64 plane at `(x, y)` (replicate-edge boundary).
fn at(plane: &[f64], w: usize, h: usize, x: i64, y: i64) -> f64 {
    let xi = clamp_index(x, w);
    let yi = clamp_index(y, h);
    plane[yi * w + xi]
}

/// The central-difference x-gradient `(f(x+1) − f(x−1)) / 2` under clamp.
fn central_diff_x(plane: &[f64], w: usize, h: usize, x: usize, y: usize) -> f64 {
    let xi = i64::try_from(x).unwrap_or(0);
    let yi = i64::try_from(y).unwrap_or(0);
    (at(plane, w, h, xi + 1, yi) - at(plane, w, h, xi - 1, yi)) * 0.5
}

/// The central-difference y-gradient `(f(y+1) − f(y−1)) / 2` under clamp.
fn central_diff_y(plane: &[f64], w: usize, h: usize, x: usize, y: usize) -> f64 {
    let xi = i64::try_from(x).unwrap_or(0);
    let yi = i64::try_from(y).unwrap_or(0);
    (at(plane, w, h, xi, yi + 1) - at(plane, w, h, xi, yi - 1)) * 0.5
}

/// The Gaussian radius `ceil(3σ)` for a scale, or `0` under the σ→0 cutoff.
pub(crate) fn gaussian_radius(sigma: f64) -> u32 {
    if sigma <= SIGMA_CUTOFF {
        return 0;
    }
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "3*sigma is positive and bounded by 3*SIGMA_MAX, well within u32"
    )]
    let r = (3.0 * sigma).ceil() as u32;
    r.max(1)
}

/// The normalized 1-D Gaussian taps for `sigma`, indexed `[-r, r] → [0, 2r]`, and
/// the radius `r`. Identical construction to [`filter.gaussian_blur`] so the two
/// agree.
fn gaussian_taps(sigma: f64) -> (Vec<f64>, usize) {
    let r = gaussian_radius(sigma) as usize;
    if r == 0 {
        return (vec![1.0], 0);
    }
    let two_sigma_sq = 2.0 * sigma * sigma;
    let mut taps = Vec::with_capacity(2 * r + 1);
    let mut sum = 0.0_f64;
    let ri = i64::try_from(r).unwrap_or(i64::MAX);
    for d in -ri..=ri {
        #[allow(
            clippy::cast_precision_loss,
            reason = "d is a small kernel offset bounded by 3*SIGMA_MAX"
        )]
        let dd = (d * d) as f64;
        let w = (-dd / two_sigma_sq).exp();
        sum += w;
        taps.push(w);
    }
    for t in &mut taps {
        *t /= sum;
    }
    (taps, r)
}

/// Smooth an f64 plane with a separable normalized Gaussian of standard deviation
/// `sigma` under a clamp boundary (the σ→0 cutoff is the identity).
pub(crate) fn gaussian_smooth(plane: &[f64], w: usize, h: usize, sigma: f64) -> Vec<f64> {
    let (taps, r) = gaussian_taps(sigma);
    if r == 0 {
        return plane.to_vec();
    }
    let ri = i64::try_from(r).unwrap_or(i64::MAX);
    // Horizontal pass.
    let mut tmp = vec![0.0_f64; plane.len()];
    for y in 0..h {
        let yi = i64::try_from(y).unwrap_or(0);
        for x in 0..w {
            let xi = i64::try_from(x).unwrap_or(0);
            let mut acc = 0.0_f64;
            for (k, &t) in taps.iter().enumerate() {
                let cx = xi + (i64::try_from(k).unwrap_or(0) - ri);
                acc = t.mul_add(at(plane, w, h, cx, yi), acc);
            }
            tmp[y * w + x] = acc;
        }
    }
    // Vertical pass.
    let mut out = vec![0.0_f64; plane.len()];
    for y in 0..h {
        let yi = i64::try_from(y).unwrap_or(0);
        for x in 0..w {
            let xi = i64::try_from(x).unwrap_or(0);
            let mut acc = 0.0_f64;
            for (k, &t) in taps.iter().enumerate() {
                let cy = yi + (i64::try_from(k).unwrap_or(0) - ri);
                acc = t.mul_add(at(&tmp, w, h, xi, cy), acc);
            }
            out[y * w + x] = acc;
        }
    }
    out
}

/// The mandatory `cpu.reference@1` oracle implementation id.
fn reference_impl() -> Result<ImplId> {
    ImplId::new("cpu", "reference", 1)
}

/// Verification declarations for `filter.structure_tensor@1`: a bounded,
/// single-reference neighbourhood op. Differential does not apply (one
/// implementation). Perceptual is not applicable: correctness is the analytic
/// tensor property set (the tensor of an oriented grating points across the
/// stripes; a constant or isotropic field yields a near-zero / isotropic tensor;
/// 90° rotation covariance of the (Jxx, Jxy, Jyy) entries), not a perceptual
/// metric.
fn structure_tensor_test_metadata() -> TestMetadata {
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
            "filter.structure_tensor is a closed-form local-geometry summary verified by analytic \
             tensor properties (oriented-grating tensor direction, constant/isotropic degeneracy, \
             90-degree rotation covariance of the (Jxx, Jxy, Jyy) entries); there is no \
             perceptual-quality metric to apply",
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
