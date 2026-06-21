//! The `image.resize@1` operation: separable resampling to a target extent with
//! a fixed-definition filter (`OP_CATALOG` §5, `AGENT_VERIFICATION` §3.8).
//!
//! `image.resize` maps an input image of extent `W × H` to a target extent
//! `W' × H'` by **separable resampling**: the rows are resampled along x, then the
//! columns along y (the two passes commute up to floating-point rounding for the
//! linear kernels). Four fixed-definition filters are offered, each with a
//! declared support and halo:
//!
//! | filter     | support (taps) | halo (px) | kernel |
//! |------------|----------------|-----------|--------|
//! | `nearest`  | 1              | 0         | box / round-to-nearest |
//! | `bilinear` | 2              | 1         | triangle (linear) |
//! | `bicubic`  | 4              | 2         | Catmull–Rom (cubic, `a = -0.5`) |
//! | `lanczos`  | 6              | 3         | Lanczos-3 (`sinc · sinc`, 3 lobes) |
//!
//! # Coordinate convention
//!
//! Pixels are sampled at their centers (`PixelCenterUpperLeft`). An output pixel
//! center `i` maps to the continuous source coordinate
//!
//! ```text
//! src(i) = (i + 0.5) · (in_size / out_size) − 0.5
//! ```
//!
//! the standard half-pixel-corrected mapping. With this convention an
//! **identity** resize (`out_size == in_size`) maps every center to itself and is
//! the exact identity; a pure integer up/down by a whole factor lands kernel taps
//! on input centers as expected.
//!
//! # Boundary
//!
//! Source taps that fall outside `[0, in_size)` are resolved by **edge clamp**
//! (replicate the nearest edge sample). This keeps a normalized kernel
//! constant-preserving at the border (a constant image resizes to the same
//! constant) without inventing energy.
//!
//! # Requirements
//!
//! The target extent must be non-zero on both axes (a zero-size target is
//! rejected — there is no pixel to define). A zero-area *input* cannot be resampled
//! to a non-empty output and is likewise rejected. The op preserves the input
//! descriptor's layout, color encoding, range, alpha representation, and semantic
//! role; only the extent changes.
//!
//! # Determinism
//!
//! [`Bounded`](DeterminismTier::Bounded): the filter weights and separable
//! accumulation are floating-point, so equality is asserted within a tolerance
//! rather than bit-exactly — except the identity-scale fast path, which copies the
//! input verbatim (bit-exact), and `nearest`, which selects a single input sample
//! per output (exact).

use paintop_core::executor::{InputValues, OpImplementation, OutputValues, ResourceValue};
use paintop_ir::{
    AssertionResult, Descriptors, DeterminismTier, Error, ErrorClass, ErrorContext, Extent,
    ImageDescriptor, ImplId, InputRegions, InputSpec, OpContract, OperationManifest,
    OutputDescriptors, OutputRegions, OutputSpec, ParamSpec, ParamType, ParamUnit, Rect,
    ResourceDescriptor, ResourceKind, Result, RoiCategory, RoiPolicy, TestMetadata,
};

/// The canonical id of the image-resize operation.
pub const RESIZE_OP_ID: &str = "image.resize@1";

/// The `image` input was absent or carried a non-image descriptor.
pub const E_RESIZE_INPUT: &str = "E_RESIZE_INPUT";

/// A target-size (`width` / `height`) param was missing, malformed, or zero.
pub const E_RESIZE_SIZE: &str = "E_RESIZE_SIZE";

/// The `filter` parameter was missing or not a known resampler.
pub const E_RESIZE_FILTER: &str = "E_RESIZE_FILTER";

/// The input image was zero-area and cannot be resampled to a non-empty output.
pub const E_RESIZE_EMPTY: &str = "E_RESIZE_EMPTY";

/// The Lanczos lobe count `a` (Lanczos-3): a fixed three-lobe window.
const LANCZOS_A: f64 = 3.0;

/// The Catmull–Rom cubic parameter (`a = -0.5`), the fixed bicubic definition.
const BICUBIC_A: f64 = -0.5;

/// A fixed-definition separable resampling filter.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Filter {
    /// Round-to-nearest source sample (1 tap, halo 0). Exact.
    Nearest,
    /// Linear / triangle (2 taps, halo 1).
    Bilinear,
    /// Catmull–Rom cubic (4 taps, halo 2).
    Bicubic,
    /// Lanczos-3 (6 taps, halo 3).
    Lanczos,
}

impl Filter {
    /// Parse the filter from its wire token.
    fn from_token(token: &str) -> Option<Self> {
        match token {
            "nearest" => Some(Self::Nearest),
            "bilinear" => Some(Self::Bilinear),
            "bicubic" => Some(Self::Bicubic),
            "lanczos" => Some(Self::Lanczos),
            _ => None,
        }
    }

    /// The half-width of the kernel support in source pixels (the declared halo).
    pub(crate) const fn halo(self) -> u32 {
        match self {
            Self::Nearest => 0,
            Self::Bilinear => 1,
            Self::Bicubic => 2,
            Self::Lanczos => 3,
        }
    }

    /// Evaluate the (continuous) reconstruction kernel at offset `t` (in source
    /// pixels) from a tap center. The kernel is normalized so a constant input
    /// reproduces the constant (the per-output weights are renormalized at use).
    fn weight(self, t: f64) -> f64 {
        let x = t.abs();
        match self {
            // Nearest uses a separate integer path; the box kernel is width 1.
            Self::Nearest => f64::from(u8::from(x < 0.5)),
            Self::Bilinear => {
                if x < 1.0 {
                    1.0 - x
                } else {
                    0.0
                }
            }
            Self::Bicubic => cubic_weight(x, BICUBIC_A),
            Self::Lanczos => lanczos_weight(t, LANCZOS_A),
        }
    }
}

/// The Keys cubic convolution kernel with parameter `a` (`a = -0.5` is
/// Catmull–Rom), evaluated at `|t| = x`.
fn cubic_weight(x: f64, a: f64) -> f64 {
    if x < 1.0 {
        // (a+2)x^3 - (a+3)x^2 + 1, via Horner with fused multiply-adds.
        let inner = (a + 2.0).mul_add(x, -(a + 3.0));
        inner.mul_add(x * x, 1.0)
    } else if x < 2.0 {
        // a x^3 - 5a x^2 + 8a x - 4a, via Horner.
        let h = a.mul_add(x, -5.0 * a);
        let h = h.mul_add(x, 8.0 * a);
        h.mul_add(x, -4.0 * a)
    } else {
        0.0
    }
}

/// The Lanczos window of lobe count `a`, evaluated at offset `t` (source pixels).
fn lanczos_weight(t: f64, a: f64) -> f64 {
    if t.abs() < f64::EPSILON {
        return 1.0;
    }
    if t.abs() >= a {
        return 0.0;
    }
    let pt = std::f64::consts::PI * t;
    (pt.sin() / pt) * ((pt / a).sin() / (pt / a))
}

/// A resolved resize request: the target extent and the chosen filter.
#[derive(Debug, Clone, Copy)]
struct ResizeRequest {
    target: Extent,
    filter: Filter,
}

impl ResizeRequest {
    /// Parse and validate the resize params.
    fn resolve(params: &serde_json::Value) -> Result<Self> {
        let width = size_param(params, "width")?;
        let height = size_param(params, "height")?;
        let filter = filter_param(params)?;
        Ok(Self {
            target: Extent::new(width, height),
            filter,
        })
    }
}

/// Parse a required non-zero `u32` target-size param.
fn size_param(params: &serde_json::Value, name: &str) -> Result<u32> {
    let value = params.get(name).ok_or_else(|| {
        Error::new(
            ErrorClass::Schema,
            E_RESIZE_SIZE,
            format!("image.resize requires an integer `{name}` parameter"),
        )
    })?;
    let n = value.as_u64().ok_or_else(|| {
        Error::new(
            ErrorClass::Schema,
            E_RESIZE_SIZE,
            format!("image.resize `{name}` must be a non-negative integer"),
        )
        .with_context(ErrorContext::default().with_actual(value.to_string()))
    })?;
    if n == 0 {
        return Err(Error::new(
            ErrorClass::Policy,
            E_RESIZE_SIZE,
            format!("image.resize `{name}` must be non-zero; there is no pixel to define"),
        ));
    }
    u32::try_from(n).map_err(|_| {
        Error::new(
            ErrorClass::Schema,
            E_RESIZE_SIZE,
            format!("image.resize `{name}` value {n} does not fit in u32"),
        )
    })
}

/// Parse the required `filter` param.
fn filter_param(params: &serde_json::Value) -> Result<Filter> {
    let value = params.get("filter").ok_or_else(|| {
        Error::new(
            ErrorClass::Schema,
            E_RESIZE_FILTER,
            "image.resize requires a `filter` parameter (nearest | bilinear | bicubic | lanczos)"
                .to_owned(),
        )
    })?;
    let token = value.as_str().ok_or_else(|| {
        Error::new(
            ErrorClass::Schema,
            E_RESIZE_FILTER,
            "image.resize `filter` must be a string".to_owned(),
        )
        .with_context(ErrorContext::default().with_actual(value.to_string()))
    })?;
    Filter::from_token(token).ok_or_else(|| {
        Error::new(
            ErrorClass::Schema,
            E_RESIZE_FILTER,
            format!("image.resize `filter` is not a known resampler: {token}"),
        )
        .with_context(
            ErrorContext::default()
                .with_actual(token.to_owned())
                .with_expected("nearest | bilinear | bicubic | lanczos"),
        )
    })
}

/// One output sample's resampling plan along an axis: the source tap indices
/// (already edge-clamped) and their normalized weights.
struct AxisTaps {
    /// Source sample indices, edge-clamped into `[0, n)`.
    indices: Vec<usize>,
    /// Weights aligned with `indices`, normalized to sum to 1.
    weights: Vec<f64>,
}

/// The source coordinate of output center `i` under the half-pixel-corrected
/// mapping `src = (i + 0.5)·scale − 0.5`, where `scale = in / out`.
fn source_coord(i: usize, scale: f64) -> f64 {
    #[allow(
        clippy::cast_precision_loss,
        reason = "image dimensions fit f64 exactly"
    )]
    let center = i as f64 + 0.5;
    center.mul_add(scale, -0.5)
}

/// Build the per-output tap plan for one axis: for output index `i`, the source
/// taps and normalized weights under `filter`, with edge clamping.
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    reason = "image dimensions fit f64; floored source coords are within the i64 tap range"
)]
fn axis_plan(out_size: usize, in_size: usize, filter: Filter) -> Vec<AxisTaps> {
    let scale = in_size as f64 / out_size as f64;
    let halo = i64::from(filter.halo());
    let n = i64::try_from(in_size).unwrap_or(i64::MAX);

    let mut plans = Vec::with_capacity(out_size);
    for i in 0..out_size {
        let src = source_coord(i, scale);
        let mut indices = Vec::new();
        let mut weights = Vec::new();

        if filter == Filter::Nearest {
            // Round half up to the nearest source center, then clamp.
            let nearest = (src + 0.5).floor() as i64;
            let clamped = nearest.clamp(0, n - 1);
            indices.push(usize::try_from(clamped).unwrap_or(0));
            weights.push(1.0);
        } else {
            // The tap window covers the kernel support around `src`.
            let center = src.floor() as i64;
            let first = center - halo + 1;
            let last = center + halo;
            for tap in first..=last {
                let t = src - tap as f64;
                let w = filter.weight(t);
                if w == 0.0 {
                    continue;
                }
                let clamped = tap.clamp(0, n - 1);
                indices.push(usize::try_from(clamped).unwrap_or(0));
                weights.push(w);
            }
            normalize(&mut weights);
        }
        plans.push(AxisTaps { indices, weights });
    }
    plans
}

/// Normalize a weight vector to sum to 1 (a no-op for an empty / zero-sum vector,
/// which cannot occur for the supported kernels at any offset).
fn normalize(weights: &mut [f64]) {
    let sum: f64 = weights.iter().sum();
    if sum.abs() > f64::EPSILON {
        for w in weights.iter_mut() {
            *w /= sum;
        }
    }
}

/// Resample `samples` (row-major, interleaved, `src` extent) to `target` extent
/// with `filter`, edge-clamped, via two separable passes (x then y).
fn resample(
    samples: &[f32],
    src: Extent,
    target: Extent,
    filter: Filter,
    channels: u32,
) -> Vec<f32> {
    let stride = channels as usize;
    let in_w = src.width as usize;
    let in_h = src.height as usize;
    let out_w = target.width as usize;
    let out_h = target.height as usize;

    // Pass 1: resample along x into an (out_w × in_h) intermediate buffer.
    let x_plans = axis_plan(out_w, in_w, filter);
    let mut horiz = vec![0.0f32; out_w * in_h * stride];
    for y in 0..in_h {
        let in_row = y * in_w * stride;
        let out_row = y * out_w * stride;
        for (ox, plan) in x_plans.iter().enumerate() {
            for c in 0..stride {
                let mut acc = 0.0f64;
                for (idx, w) in plan.indices.iter().zip(plan.weights.iter()) {
                    acc += w * f64::from(samples[in_row + idx * stride + c]);
                }
                #[allow(
                    clippy::cast_possible_truncation,
                    reason = "accumulated in f64, stored back to the f32 sample type"
                )]
                {
                    horiz[out_row + ox * stride + c] = acc as f32;
                }
            }
        }
    }

    // Pass 2: resample along y into the (out_w × out_h) output.
    let y_plans = axis_plan(out_h, in_h, filter);
    let mut out = vec![0.0f32; out_w * out_h * stride];
    for (oy, plan) in y_plans.iter().enumerate() {
        for ox in 0..out_w {
            for c in 0..stride {
                let mut acc = 0.0f64;
                for (idx, w) in plan.indices.iter().zip(plan.weights.iter()) {
                    acc += w * f64::from(horiz[(idx * out_w + ox) * stride + c]);
                }
                #[allow(
                    clippy::cast_possible_truncation,
                    reason = "accumulated in f64, stored back to the f32 sample type"
                )]
                {
                    out[(oy * out_w + ox) * stride + c] = acc as f32;
                }
            }
        }
    }
    out
}

/// The `image.resize@1` operation.
#[derive(Debug, Clone, Copy, Default)]
pub struct Resize;

impl Resize {
    /// Construct the image-resize operation.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    /// The declared manifest for `image.resize@1`.
    ///
    /// # Errors
    /// Propagates the [`schema`](ErrorClass::Schema) error if the hard-coded op /
    /// impl ids are somehow invalid (they are not).
    pub fn manifest() -> Result<OperationManifest> {
        let size_param = |name: &str, doc: &str| ParamSpec {
            name: name.to_owned(),
            ty: ParamType::Integer,
            unit: Some(ParamUnit::Pixels),
            required: true,
            default: None,
            choices: vec![],
            doc: doc.to_owned(),
        };
        Ok(OperationManifest {
            id: RESIZE_OP_ID.parse()?,
            impl_version: 1,
            summary: "Separable resampling to a target extent with a fixed-definition filter \
                      (nearest / bilinear / bicubic Catmull-Rom / Lanczos-3); half-pixel-corrected \
                      pixel-center mapping, edge-clamped boundary."
                .to_owned(),
            determinism: DeterminismTier::Bounded,
            roi: RoiPolicy {
                category: RoiCategory::Geometric,
                halo_px: None,
            },
            inputs: vec![InputSpec {
                name: "image".to_owned(),
                kind: ResourceKind::Image,
                required: true,
                doc: "The image to resample.".to_owned(),
            }],
            outputs: vec![OutputSpec {
                name: "image".to_owned(),
                kind: ResourceKind::Image,
                doc: "The resampled image (target extent; all other descriptor fields preserved)."
                    .to_owned(),
            }],
            params: vec![
                size_param("width", "Target width in pixels (must be non-zero)."),
                size_param("height", "Target height in pixels (must be non-zero)."),
                ParamSpec {
                    name: "filter".to_owned(),
                    ty: ParamType::String,
                    unit: None,
                    required: true,
                    default: None,
                    choices: vec![
                        "nearest".to_owned(),
                        "bilinear".to_owned(),
                        "bicubic".to_owned(),
                        "lanczos".to_owned(),
                    ],
                    doc: "The fixed-definition resampling filter: nearest (1 tap), bilinear (2), \
                          bicubic Catmull-Rom (4), or Lanczos-3 (6)."
                        .to_owned(),
                },
            ],
            implementations: vec![reference_impl()?],
            test: resize_test_metadata(),
        })
    }
}

impl OpContract for Resize {
    fn declared_inputs(&self) -> Vec<(String, ResourceKind)> {
        vec![("image".to_owned(), ResourceKind::Image)]
    }

    fn declared_outputs(&self) -> Vec<(String, ResourceKind)> {
        vec![("image".to_owned(), ResourceKind::Image)]
    }

    fn infer_outputs(
        &self,
        inputs: &Descriptors,
        params: &serde_json::Value,
    ) -> Result<OutputDescriptors> {
        let descriptor = image_descriptor(inputs)?;
        let request = ResizeRequest::resolve(params)?;
        // A non-empty target needs a non-empty source to resample from.
        if descriptor.extent.width == 0 || descriptor.extent.height == 0 {
            return Err(empty_input_error());
        }
        let mut out_desc = *descriptor;
        out_desc.extent = request.target;
        let mut out = OutputDescriptors::new();
        out.insert("image".to_owned(), ResourceDescriptor::Image(out_desc));
        Ok(out)
    }

    fn required_inputs(
        &self,
        requested_outputs: &OutputRegions,
        inputs: &Descriptors,
        params: &serde_json::Value,
    ) -> Result<InputRegions> {
        // Geometric: an output region maps to its source footprint dilated by the
        // filter halo. The mapping is conservative — the full input is always a
        // safe demand — so we map the requested window's source span and dilate.
        let descriptor = image_descriptor(inputs)?;
        let request = ResizeRequest::resolve(params)?;
        let mut regions = InputRegions::new();
        if let Some(r) = requested_outputs.get("image") {
            let region = source_footprint(*r, descriptor.extent, request);
            regions.insert("image".to_owned(), region);
        }
        Ok(regions)
    }

    fn validate_postconditions(
        &self,
        outputs: &OutputDescriptors,
        _params: &serde_json::Value,
    ) -> Result<Vec<AssertionResult>> {
        let Some(ResourceDescriptor::Image(_)) = outputs.get("image") else {
            return Ok(vec![AssertionResult::fail(
                "produces_image",
                "no `image` output produced",
            )]);
        };
        Ok(vec![AssertionResult::pass("produces_image")])
    }
}

/// The source footprint (input region) needed for an output region `r`, under the
/// half-pixel mapping plus the filter halo, clamped to the input extent.
fn source_footprint(r: Rect, src: Extent, request: ResizeRequest) -> Rect {
    #[allow(clippy::cast_precision_loss, reason = "image dimensions fit f64")]
    let sx = f64::from(src.width) / f64::from(request.target.width);
    #[allow(clippy::cast_precision_loss, reason = "image dimensions fit f64")]
    let sy = f64::from(src.height) / f64::from(request.target.height);
    let halo = i64::from(request.filter.halo());
    let w = i64::from(src.width);
    let h = i64::from(src.height);

    let map = |coord: i64, scale: f64| -> i64 {
        #[allow(
            clippy::cast_possible_truncation,
            clippy::cast_precision_loss,
            reason = "mapped source coordinate floored to an integer tap"
        )]
        let v = (coord as f64)
            .mul_add(scale, 0.5f64.mul_add(scale, -0.5))
            .floor() as i64;
        v
    };
    let x0 = (map(r.x0, sx) - halo).clamp(0, w);
    let x1 = (map(r.x1.saturating_sub(1), sx) + halo + 1).clamp(0, w);
    let y0 = (map(r.y0, sy) - halo).clamp(0, h);
    let y1 = (map(r.y1.saturating_sub(1), sy) + halo + 1).clamp(0, h);
    Rect::new(x0, y0, x1, y1)
}

impl OpImplementation for Resize {
    fn compute(
        &self,
        inputs: &InputValues,
        params: &serde_json::Value,
    ) -> std::result::Result<OutputValues, Error> {
        let image = inputs.get("image").ok_or_else(|| {
            Error::new(
                ErrorClass::Reference,
                E_RESIZE_INPUT,
                "image.resize requires an `image` input value".to_owned(),
            )
        })?;
        let ResourceDescriptor::Image(descriptor) = image.descriptor() else {
            return Err(Error::new(
                ErrorClass::Type,
                E_RESIZE_INPUT,
                "image.resize `image` input must be an image resource".to_owned(),
            ));
        };
        let request = ResizeRequest::resolve(params)?;
        if descriptor.extent.width == 0 || descriptor.extent.height == 0 {
            return Err(empty_input_error());
        }

        // Identity scale is the exact (bit-for-bit) identity: skip resampling so no
        // floating-point perturbation touches the samples.
        let samples = if request.target == descriptor.extent {
            image.samples().to_vec()
        } else {
            resample(
                image.samples(),
                descriptor.extent,
                request.target,
                request.filter,
                image.channels(),
            )
        };

        let mut out_desc = *descriptor;
        out_desc.extent = request.target;
        let value = ResourceValue::new(
            ResourceDescriptor::Image(out_desc),
            image.channels(),
            samples,
        )
        .map_err(|actual| {
            Error::new(
                ErrorClass::Execution,
                E_RESIZE_INPUT,
                format!("image.resize produced a sample buffer of unexpected length {actual}"),
            )
        })?;

        let mut out = OutputValues::new();
        out.insert("image".to_owned(), value);
        Ok(out)
    }
}

/// Extract the required `image` input descriptor, erroring if absent or non-image.
fn image_descriptor(inputs: &Descriptors) -> Result<&ImageDescriptor> {
    let image = inputs.get("image").ok_or_else(|| {
        Error::new(
            ErrorClass::Reference,
            E_RESIZE_INPUT,
            "image.resize requires an `image` input".to_owned(),
        )
    })?;
    let ResourceDescriptor::Image(descriptor) = image else {
        return Err(Error::new(
            ErrorClass::Type,
            E_RESIZE_INPUT,
            "image.resize `image` input must be an image resource".to_owned(),
        ));
    };
    Ok(descriptor)
}

/// Build the zero-area-input rejection error.
fn empty_input_error() -> Error {
    Error::new(
        ErrorClass::Policy,
        E_RESIZE_EMPTY,
        "image.resize cannot resample a zero-area input to a non-empty output".to_owned(),
    )
}

/// The mandatory `cpu.reference@1` oracle implementation id.
fn reference_impl() -> Result<ImplId> {
    ImplId::new("cpu", "reference", 1)
}

/// Verification declarations for `image.resize@1`: a single-reference, bounded,
/// geometric resampler. Differential does not apply (one implementation).
/// Perceptual is not applicable: the resamplers are closed-form, fixed-definition
/// kernels verified by analytic value tables, the kernel algebra (identity scale,
/// nearest exactness, constant preservation, tiling independence), and a
/// band-limited round-trip tolerance — not a perceptual-quality metric.
fn resize_test_metadata() -> TestMetadata {
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
            "image.resize applies fixed-definition resampling kernels verified by analytic value \
             tables and kernel algebra (identity-scale identity, nearest integer-translation \
             exactness, constant preservation, tiling independence, band-limited round-trip \
             tolerance); there is no perceptual-quality metric to apply",
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
