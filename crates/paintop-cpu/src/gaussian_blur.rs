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

use paintop_core::executor::{InputValues, OpImplementation, OutputValues, ResourceValue};
use paintop_ir::{
    AssertionResult, Descriptors, DeterminismTier, Error, ErrorClass, ErrorContext, Extent, ImplId,
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
            implementations: vec![reference_impl()?, optimized_impl()?],
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

// ---------------------------------------------------------------------------
// The `cpu.optimized` separable Gaussian (bn-u6g, `plan.md` §12.2).
// ---------------------------------------------------------------------------

/// The mandatory `cpu.optimized@1` backend implementation id.
fn optimized_impl() -> Result<ImplId> {
    ImplId::new("cpu", "optimized", 1)
}

/// The per-axis boundary index policy, mirroring the direct convolution oracle's
/// [`source_index`](crate::convolve) exactly so the separable passes reproduce the
/// reference's boundary handling within the bounded tolerance.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BlurBoundary {
    /// A fixed constant for any out-of-bounds tap (the `constant`/`transparent`
    /// modes; the Gaussian always uses an all-zero constant).
    Constant,
    /// Replicate the nearest edge sample.
    Clamp,
    /// Whole-sample mirror across the edge (edge not repeated).
    Mirror,
    /// Periodic (toroidal) tiling.
    Wrap,
}

impl BlurBoundary {
    /// Map the boundary `mode` token onto the per-axis policy. The Gaussian's
    /// `constant` and `transparent` modes both blur against an all-zero border, so
    /// they share the [`Constant`](Self::Constant) arm.
    fn from_mode(mode: &str) -> Self {
        match mode {
            "mirror" => Self::Mirror,
            "wrap" => Self::Wrap,
            "constant" | "transparent" => Self::Constant,
            // `clamp` is the documented default for every other (valid) token.
            _ => Self::Clamp,
        }
    }

    /// Resolve an out-of-or-in-bounds 1-D `coord` to a source index in `[0, n)`,
    /// or `None` when the constant border applies. Identical in behaviour to the
    /// oracle's `source_index` per axis (`n >= 1`).
    fn source_index(self, coord: i64, n: i64) -> Option<i64> {
        if coord >= 0 && coord < n {
            return Some(coord);
        }
        match self {
            Self::Constant => None,
            Self::Clamp => Some(coord.clamp(0, n - 1)),
            Self::Wrap => Some(coord.rem_euclid(n)),
            Self::Mirror => {
                if n == 1 {
                    Some(0)
                } else {
                    let period = 2 * (n - 1);
                    let m = coord.rem_euclid(period);
                    Some(if m < n { m } else { period - m })
                }
            }
        }
    }
}

/// The normalized 1-D Gaussian taps for a `sigma`, indexed `[-r, r]` → `[0, 2r]`,
/// and the radius `r = ceil(3σ)` (`0` under the σ→0 cutoff).
///
/// The taps are `g(d) = exp(-d² / 2σ²)` normalized so `Σ g = 1`. Because the
/// reference's 2-D kernel sum over the `(2r+1)²` square **factorizes** —
/// `ΣΣ exp(-(dx²+dy²)/2σ²) = (Σ exp(-d²/2σ²))²` — the separable product
/// `g(dx)·g(dy)` is *algebraically the same kernel* the reference normalizes, so
/// the two-pass result matches the direct convolution to within f64 reassociation
/// (the bounded tier), not a different blur.
fn gaussian_taps_1d(sigma: f64) -> (Vec<f64>, u32) {
    let r = kernel_radius(sigma);
    if r == 0 {
        return (vec![1.0], 0);
    }
    let ri = i64::from(r);
    let two_sigma_sq = 2.0 * sigma * sigma;
    let mut taps = Vec::with_capacity((2 * r + 1) as usize);
    let mut sum = 0.0_f64;
    for d in -ri..=ri {
        #[allow(
            clippy::cast_precision_loss,
            reason = "d is a small kernel offset bounded by 3*sigma_max"
        )]
        let w = (-(d * d) as f64 / two_sigma_sq).exp();
        sum += w;
        taps.push(w);
    }
    for w in &mut taps {
        *w /= sum;
    }
    (taps, r)
}

/// Convolve one axis of a `channels`-interleaved `f32` plane with the 1-D Gaussian
/// `taps` (radius `r`, hot tap at index `r`), under the per-axis boundary policy.
///
/// `horizontal` selects the axis: when true the kernel slides along x (the inner,
/// stride-`channels` direction); when false along y (stride `width*channels`). Each
/// output sample is an f64 accumulation of `taps[k] * src(boundary(coord))`, rounded
/// once to f32 — the same fixed-order-per-axis accumulation the oracle performs over
/// the full 2-D tap set, just factored into two passes.
#[allow(
    clippy::too_many_arguments,
    reason = "a flat per-axis separable pass over an interleaved plane; the geometry \
              (extent, channels, taps, radius, boundary, axis) is irreducible"
)]
fn blur_axis(
    src: &[f32],
    width: usize,
    height: usize,
    channels: usize,
    taps: &[f64],
    r: u32,
    boundary: BlurBoundary,
    horizontal: bool,
) -> Vec<f32> {
    let mut out = vec![0.0_f32; src.len()];
    let ri = i64::from(r);
    // The along-axis length, as the i64 the boundary index map operates in. Image
    // extents are u32-bounded, so the conversion is exact.
    let axis_len = i64::from(u32::try_from(if horizontal { width } else { height }).unwrap_or(0));
    for y in 0..height {
        for x in 0..width {
            let base = (y * width + x) * channels;
            // The hot pixel's along-axis position (exact: bounded by the extent).
            let pos = i64::from(u32::try_from(if horizontal { x } else { y }).unwrap_or(0));
            for ch in 0..channels {
                let mut acc = 0.0_f64;
                for (k, &w) in taps.iter().enumerate() {
                    if w == 0.0 {
                        continue;
                    }
                    // Tap k lands at offset (k - r) from the hot pixel.
                    let coord = pos + (i64::from(u32::try_from(k).unwrap_or(0)) - ri);
                    let sample = boundary.source_index(coord, axis_len).map_or(
                        // The constant border: the Gaussian blurs against zero.
                        0.0,
                        |idx| {
                            // `source_index` returns an index in `[0, axis_len)`.
                            let idx = usize::try_from(idx).unwrap_or(0);
                            let src_base = if horizontal {
                                (y * width + idx) * channels + ch
                            } else {
                                (idx * width + x) * channels + ch
                            };
                            f64::from(src[src_base])
                        },
                    );
                    acc = w.mul_add(sample, acc);
                }
                #[allow(
                    clippy::cast_possible_truncation,
                    reason = "accumulate in f64 then store the op's f32 sample type"
                )]
                {
                    out[base + ch] = acc as f32;
                }
            }
        }
    }
    out
}

/// The separable two-pass Gaussian over a `channels`-interleaved plane: a
/// horizontal pass followed by a vertical pass with the same 1-D taps.
///
/// The two passes commute with the boundary index map (each axis applies it
/// independently), so `V(H(src))` reproduces the oracle's single 2-D sum
/// `(V∘H)(src)` to within f64 reassociation. Cost is `O(r)` per pixel per axis
/// (`2r+1` taps each) versus the reference's `O(r²)` (`(2r+1)²` taps), the win that
/// grows with sigma.
fn separable_blur(
    samples: &[f32],
    extent: Extent,
    channels: u32,
    sigma: f64,
    boundary: BlurBoundary,
) -> Vec<f32> {
    let width = extent.width as usize;
    let height = extent.height as usize;
    let ch = channels as usize;
    if width == 0 || height == 0 || ch == 0 {
        return Vec::new();
    }
    let (taps, r) = gaussian_taps_1d(sigma);
    if r == 0 {
        // The σ→0 identity: a single unit tap is a pass-through.
        return samples.to_vec();
    }
    let horizontal = blur_axis(samples, width, height, ch, &taps, r, boundary, true);
    blur_axis(&horizontal, width, height, ch, &taps, r, boundary, false)
}

/// The `cpu.optimized` separable Gaussian backend of `filter.gaussian_blur@1`
/// (bn-u6g; `plan.md` §12.2).
///
/// Computes the identical normalized Gaussian as the
/// [`GaussianBlur`] reference, but as two `O(r)` separable passes instead of one
/// `O(r²)` direct 2-D convolution, validated against the reference oracle within
/// the op's bounded tolerance by the cross-backend differential harness. Faster for
/// large sigma; deterministic and bit-identical on reruns (a fixed-order f64
/// accumulation per axis).
#[derive(Debug, Clone, Copy, Default)]
pub struct GaussianBlurOptimized;

impl GaussianBlurOptimized {
    /// Construct the optimized separable Gaussian backend.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }
}

impl OpImplementation for GaussianBlurOptimized {
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
        // Reuse the reference's param validation (sigma bounds, mode vocabulary) so
        // the optimized backend rejects exactly what the oracle rejects, then build
        // the separable plan from the validated sigma/mode.
        let _ = build_convolve_params(params, channels)?;
        let sigma = sigma_param(params)?;
        let mode = mode_param(params);
        let boundary = BlurBoundary::from_mode(&mode);

        let descriptor = *input.descriptor();
        // The Gaussian preserves the extent (no `valid` mode), so the output frame
        // matches the input.
        let extent = input.extent();
        let samples = separable_blur(input.samples(), extent, channels, sigma, boundary);

        let value = ResourceValue::new(descriptor, channels, samples).map_err(|actual| {
            Error::new(
                ErrorClass::Execution,
                E_BLUR_INPUT,
                format!(
                    "filter.gaussian_blur (optimized) produced a sample buffer of unexpected \
                     length {actual}"
                ),
            )
        })?;
        let mut out = OutputValues::new();
        out.insert("output".to_owned(), value);
        Ok(out)
    }
}

/// The mandatory `cpu.reference@1` oracle implementation id.
fn reference_impl() -> Result<ImplId> {
    ImplId::new("cpu", "reference", 1)
}

/// Verification declarations for `filter.gaussian_blur@1`: a bounded
/// neighbourhood op carrying its `cpu.reference` direct-convolution oracle plus a
/// `cpu.optimized` separable backend (M3 bn-u6g). Differential **applies** and is
/// covered: the cross-backend harness validates the separable result against the
/// oracle within the op's bounded tolerance. Perceptual is not applicable:
/// correctness is the analytic Gaussian property set (unit-sum positive kernel,
/// constant preservation, 90° isotropy, the σ-semigroup, the blurred-impulse
/// variance match, and the σ→0 identity), plus the differential checks against
/// `filter.convolve` and the separable backend — not a perceptual-quality metric.
fn blur_test_metadata() -> TestMetadata {
    use paintop_ir::{CategoryStatus, VerificationCategory, VerificationDeclarations};
    let mut decls = VerificationDeclarations::new();
    for category in [
        VerificationCategory::BuildHygiene,
        VerificationCategory::SchemaContract,
        VerificationCategory::AnalyticFixtures,
        VerificationCategory::PropertyTests,
        VerificationCategory::Metamorphic,
        VerificationCategory::Differential,
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
