//! The `color.adjust@1` operation: pointwise exposure / saturation / temperature
//! color grading in linear light (`OP_CATALOG` §2, `AGENT_VERIFICATION` §3.1).
//!
//! `color.adjust` applies three explicit, independent sub-adjustments to the
//! **color** channels of a linear-light image, in a fixed order
//! (exposure → temperature → saturation). Each sub-parameter defaults to its
//! identity value, so an all-default request is the exact identity. An optional
//! `mask` input gates the adjustment per pixel: the output is the input linearly
//! blended toward the fully-adjusted value by the mask coverage, so an empty
//! (all-zero) mask is the identity and a full (all-one) mask equals the unmasked
//! adjustment.
//!
//! # Semantics
//!
//! Let `R, G, B` be a pixel's linear-light color channels (a `Gray` image has a
//! single channel treated as all three; alpha, when present, is never touched).
//!
//! - **`exposure_ev`** (`e`, EV / stops): a multiplicative exposure in linear
//!   light, `x ↦ x · 2^e`. This is the only sub-adjustment when `saturation` and
//!   `temperature` are zero, so exposures compose additively in unclamped linear
//!   light: `E_a(E_b(x)) = E_{a+b}(x)` (`AGENT_VERIFICATION` §3.1).
//! - **`temperature`** (`t`, in `[-1, 1]`): a symmetric warm/cool tilt scaling
//!   the red and blue channels in opposite directions, `R ↦ R·(1+t)`,
//!   `B ↦ B·(1−t)`, leaving green fixed. `t = 0` is the identity; a positive `t`
//!   warms (more red, less blue).
//! - **`saturation`** (`s`): a blend of each channel toward the pixel's
//!   Rec. 709 linear luminance `Y = 0.2126·R + 0.7152·G + 0.0722·B`,
//!   `c ↦ Y + (1+s)·(c − Y)`. `s = 0` is the identity; `s = −1` is fully
//!   desaturated (grayscale); `s > 0` increases saturation.
//!
//! The op is **pointwise** (each output sample depends only on the co-located
//! input sample and, when present, the co-located mask coverage) and
//! **bounded**-determinism: exposure uses `exp2`, whose last bit is not
//! guaranteed identical across platforms, so equality is asserted within a
//! tolerance rather than bit-exactly.
//!
//! # Linear light
//!
//! The adjustment math is only meaningful in linear light, so the op rejects an
//! input image whose color encoding is the sRGB display transfer function
//! (`srgb`) with a [`semantic`](ErrorClass::Semantic) error; the agent must
//! `color.convert` into `linear-srgb` first. A `linear-srgb` or `raw-linear`
//! image is accepted.

use paintop_core::executor::{InputValues, OpImplementation, OutputValues, ResourceValue};
use paintop_ir::{
    AssertionResult, ColorEncoding, Descriptors, DeterminismTier, Error, ErrorClass, ErrorContext,
    ImageDescriptor, ImplId, InputRegions, InputSpec, OpContract, OperationManifest,
    OutputDescriptors, OutputRegions, OutputSpec, ParamSpec, ParamType, ParamUnit,
    ResourceDescriptor, ResourceKind, Result, RoiCategory, RoiPolicy, TestMetadata,
};

/// The canonical id of the color-adjust operation.
pub const ADJUST_OP_ID: &str = "color.adjust@1";

/// The `image` input to adjust was absent or carried a non-image descriptor.
pub const E_ADJUST_INPUT: &str = "E_ADJUST_INPUT";

/// A sub-parameter (`exposure_ev`, `saturation`, `temperature`) was the wrong
/// type or held a non-finite value.
pub const E_ADJUST_PARAM: &str = "E_ADJUST_PARAM";

/// The optional `mask` input did not match the image extent, or was not a mask.
pub const E_ADJUST_MASK: &str = "E_ADJUST_MASK";

/// The input image was in a non-linear (`srgb`) color encoding the adjustment
/// math cannot operate on.
pub const E_ADJUST_NONLINEAR: &str = "E_ADJUST_NONLINEAR";

/// Rec. 709 linear-luminance weight for the red channel.
const LUMA_R: f32 = 0.212_6;
/// Rec. 709 linear-luminance weight for the green channel.
const LUMA_G: f32 = 0.715_2;
/// Rec. 709 linear-luminance weight for the blue channel.
const LUMA_B: f32 = 0.072_2;

/// The resolved sub-adjustments of a `color.adjust` request.
///
/// Each field carries the parsed sub-parameter; all default to their identity
/// value (`0.0`), so a request with no params is [`is_identity`](Self::is_identity).
#[derive(Debug, Clone, Copy, PartialEq)]
struct Adjustment {
    /// Exposure in EV / stops (`x ↦ x·2^e`).
    exposure_ev: f32,
    /// Saturation blend toward luminance (`0` identity).
    saturation: f32,
    /// Warm/cool channel tilt (`0` identity).
    temperature: f32,
}

impl Adjustment {
    /// Parse the optional sub-parameters, defaulting each to its identity value.
    ///
    /// # Errors
    /// [`schema`](ErrorClass::Schema) / [`E_ADJUST_PARAM`] if any present
    /// sub-parameter is non-numeric or non-finite.
    fn resolve(params: &serde_json::Value) -> Result<Self> {
        Ok(Self {
            exposure_ev: optional_finite(params, "exposure_ev")?,
            saturation: optional_finite(params, "saturation")?,
            temperature: optional_finite(params, "temperature")?,
        })
    }

    /// Whether this is the identity adjustment (every sub-parameter at its
    /// identity value).
    fn is_identity(self) -> bool {
        self.exposure_ev == 0.0 && self.saturation == 0.0 && self.temperature == 0.0
    }

    /// Apply the fixed-order adjustment (exposure → temperature → saturation) to
    /// one pixel's color channels in place.
    ///
    /// `rgb` holds the channel values: a `Gray` pixel passes a 1-slice (its
    /// single channel is its own luminance), an `Rgb`/`Rgba` pixel a 3-slice.
    fn apply(self, rgb: &mut [f32]) {
        // Exposure: a uniform linear-light gain on every color channel.
        let gain = self.exposure_ev.exp2();
        for c in rgb.iter_mut() {
            *c *= gain;
        }

        // Temperature: a symmetric warm/cool tilt on red and blue. A single-channel
        // (gray) pixel has no chroma to tilt, so it is left unchanged.
        if rgb.len() == 3 {
            rgb[0] *= 1.0 + self.temperature;
            rgb[2] *= 1.0 - self.temperature;
        }

        // Saturation: blend each channel toward the pixel luminance. For a gray
        // pixel the single channel *is* its luminance, so this is the identity.
        let luma = if rgb.len() == 3 {
            LUMA_R.mul_add(rgb[0], LUMA_G.mul_add(rgb[1], LUMA_B * rgb[2]))
        } else {
            // A single channel is its own luminance: the blend is a no-op.
            return;
        };
        let scale = 1.0 + self.saturation;
        for c in rgb.iter_mut() {
            *c = scale.mul_add(*c - luma, luma);
        }
    }
}

/// Read an optional finite `f32` sub-parameter, defaulting to `0.0` when absent.
///
/// # Errors
/// [`schema`](ErrorClass::Schema) / [`E_ADJUST_PARAM`] if the value is present
/// but not a finite number.
fn optional_finite(params: &serde_json::Value, name: &str) -> Result<f32> {
    let Some(value) = params.get(name) else {
        return Ok(0.0);
    };
    let n = value.as_f64().ok_or_else(|| {
        Error::new(
            ErrorClass::Schema,
            E_ADJUST_PARAM,
            format!("color.adjust `{name}` must be a number"),
        )
        .with_context(ErrorContext::default().with_actual(value.to_string()))
    })?;
    if n.is_finite() {
        #[allow(
            clippy::cast_possible_truncation,
            reason = "sub-parameters are intentionally single-precision grading controls"
        )]
        Ok(n as f32)
    } else {
        Err(Error::new(
            ErrorClass::Schema,
            E_ADJUST_PARAM,
            format!("color.adjust `{name}` must be finite"),
        )
        .with_context(ErrorContext::default().with_actual(value.to_string())))
    }
}

/// Reject a `srgb`-encoded input: the adjustment math is only meaningful in
/// linear light.
fn require_linear(color: ColorEncoding) -> Result<()> {
    if color == ColorEncoding::Srgb {
        return Err(Error::new(
            ErrorClass::Semantic,
            E_ADJUST_NONLINEAR,
            "color.adjust operates in linear light; convert the image to `linear-srgb` first"
                .to_owned(),
        )
        .with_context(
            ErrorContext::default()
                .with_actual(format!("{color:?}"))
                .with_expected("linear-srgb | raw-linear"),
        ));
    }
    Ok(())
}

/// The interleaved color channels of one pixel (everything but a trailing alpha).
///
/// `channels` is the per-pixel sample count and `has_alpha` whether the last is
/// alpha. Returns the color-channel count (`channels` minus any alpha).
const fn color_channel_count(channels: u32, has_alpha: bool) -> usize {
    let c = channels as usize;
    if has_alpha { c.saturating_sub(1) } else { c }
}

/// Apply the adjustment to an image's interleaved samples, optionally gated by a
/// per-pixel mask coverage.
///
/// Color channels are adjusted; a trailing alpha channel passes through. When a
/// `mask` is given, each pixel's output is the input linearly blended toward the
/// adjusted value by the co-located coverage (`out = in + cov·(adj − in)`), so a
/// zero coverage is the identity and a unit coverage the full adjustment.
fn adjust_samples(
    samples: &[f32],
    channels: u32,
    has_alpha: bool,
    adjustment: Adjustment,
    mask: Option<&[f32]>,
) -> Vec<f32> {
    let stride = channels as usize;
    let color_count = color_channel_count(channels, has_alpha);
    if stride == 0 || color_count == 0 {
        return samples.to_vec();
    }

    let mut out = samples.to_vec();
    for (pixel_index, pixel) in out.chunks_mut(stride).enumerate() {
        let original: [f32; 3] = match color_count {
            1 => [pixel[0], 0.0, 0.0],
            _ => [pixel[0], pixel[1], pixel[2]],
        };

        let mut color = [original[0], original[1], original[2]];
        adjustment.apply(&mut color[..color_count]);

        // Gate by mask coverage when present: blend input -> adjusted.
        let coverage = mask.map_or(1.0, |m| m.get(pixel_index).copied().unwrap_or(0.0));
        for ch in 0..color_count {
            let adjusted = coverage.mul_add(color[ch] - original[ch], original[ch]);
            pixel[ch] = adjusted;
        }
    }
    out
}

/// The `color.adjust@1` operation: a linear-light color `Image` (+ optional
/// coverage `Mask`) → a graded color `Image`.
#[derive(Debug, Clone, Copy, Default)]
pub struct Adjust;

impl Adjust {
    /// Construct the color-adjust operation.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    /// The declared manifest for `color.adjust@1`.
    ///
    /// # Errors
    /// Propagates the [`schema`](ErrorClass::Schema) error if the hard-coded op /
    /// impl ids are somehow invalid (they are not).
    pub fn manifest() -> Result<OperationManifest> {
        Ok(OperationManifest {
            id: ADJUST_OP_ID.parse()?,
            impl_version: 1,
            summary: "Pointwise linear-light color grading: explicit exposure (EV), saturation, \
                      and temperature sub-adjustments, optionally gated by a coverage mask."
                .to_owned(),
            determinism: DeterminismTier::Bounded,
            roi: RoiPolicy {
                category: RoiCategory::Pointwise,
                halo_px: None,
            },
            inputs: vec![
                InputSpec {
                    name: "image".to_owned(),
                    kind: ResourceKind::Image,
                    required: true,
                    doc: "The linear-light color image to grade.".to_owned(),
                },
                InputSpec {
                    name: "mask".to_owned(),
                    kind: ResourceKind::Mask,
                    required: false,
                    doc: "Optional coverage mask gating the adjustment per pixel; absent applies \
                          the adjustment everywhere."
                        .to_owned(),
                },
            ],
            outputs: vec![OutputSpec {
                name: "image".to_owned(),
                kind: ResourceKind::Image,
                doc: "The graded image (same extent/layout/encoding as the input).".to_owned(),
            }],
            params: vec![
                ParamSpec {
                    name: "exposure_ev".to_owned(),
                    ty: ParamType::Float,
                    unit: Some(ParamUnit::Ev),
                    required: false,
                    default: Some(serde_json::json!(0.0)),
                    choices: vec![],
                    doc: "Linear-light exposure in EV / stops (x -> x * 2^ev); 0 is the identity."
                        .to_owned(),
                },
                ParamSpec {
                    name: "saturation".to_owned(),
                    ty: ParamType::Float,
                    unit: None,
                    required: false,
                    default: Some(serde_json::json!(0.0)),
                    choices: vec![],
                    doc: "Saturation blend toward luminance; 0 is the identity, -1 fully \
                          desaturates."
                        .to_owned(),
                },
                ParamSpec {
                    name: "temperature".to_owned(),
                    ty: ParamType::Float,
                    unit: None,
                    required: false,
                    default: Some(serde_json::json!(0.0)),
                    choices: vec![],
                    doc: "Warm/cool tilt scaling red up and blue down (positive warms); 0 is the \
                          identity."
                        .to_owned(),
                },
            ],
            implementations: vec![reference_impl()?, optimized_impl()?],
            test: adjust_test_metadata(),
        })
    }
}

impl OpContract for Adjust {
    fn declared_inputs(&self) -> Vec<(String, ResourceKind)> {
        vec![
            ("image".to_owned(), ResourceKind::Image),
            ("mask".to_owned(), ResourceKind::Mask),
        ]
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
        require_linear(descriptor.color)?;
        // Validate the params at infer time so a malformed request fails on the
        // type-checking pass, before any pixels are touched.
        Adjustment::resolve(params)?;
        // When a mask is wired, it must size to the image so the per-pixel gate is
        // well defined.
        check_mask_extent(inputs, descriptor)?;

        // A pointwise grade preserves the image descriptor exactly.
        let mut out = OutputDescriptors::new();
        out.insert("image".to_owned(), ResourceDescriptor::Image(*descriptor));
        Ok(out)
    }

    fn required_inputs(
        &self,
        requested_outputs: &OutputRegions,
        inputs: &Descriptors,
        _params: &serde_json::Value,
    ) -> Result<InputRegions> {
        // Pointwise: each output sample needs exactly the co-located image (and,
        // when wired, mask) sample.
        let mut regions = InputRegions::new();
        if let Some(region) = requested_outputs.get("image") {
            regions.insert("image".to_owned(), *region);
            if inputs.contains_key("mask") {
                regions.insert("mask".to_owned(), *region);
            }
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

/// The compute backend serving `color.adjust`: the scalar reference oracle or the
/// autovectorization-friendly `cpu.optimized` kernel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Backend {
    /// The scalar reference oracle ([`adjust_samples`]).
    Reference,
    /// The `cpu.optimized` grade kernel ([`crate::optimized::kernels`]).
    Optimized,
}

/// Apply the grade with the selected backend; both compute the same fixed-order
/// adjustment (exposure -> temperature -> saturation) with the same mask blend.
fn apply_backend(
    backend: Backend,
    samples: &[f32],
    channels: u32,
    has_alpha: bool,
    adjustment: Adjustment,
    mask: Option<&[f32]>,
) -> Vec<f32> {
    match backend {
        Backend::Reference => adjust_samples(samples, channels, has_alpha, adjustment, mask),
        Backend::Optimized => {
            let stride = channels as usize;
            let color_count = color_channel_count(channels, has_alpha);
            crate::optimized::kernels::color_adjust(
                samples,
                stride,
                color_count,
                crate::optimized::kernels::Adjustment {
                    exposure_ev: adjustment.exposure_ev,
                    saturation: adjustment.saturation,
                    temperature: adjustment.temperature,
                },
                mask,
            )
        }
    }
}

/// Shared compute for both backends: validate the image/mask/params, then grade.
fn compute_backend(
    backend: Backend,
    inputs: &InputValues,
    params: &serde_json::Value,
) -> std::result::Result<OutputValues, Error> {
    let image = inputs.get("image").ok_or_else(|| {
        Error::new(
            ErrorClass::Reference,
            E_ADJUST_INPUT,
            "color.adjust requires an `image` input value".to_owned(),
        )
    })?;
    let ResourceDescriptor::Image(descriptor) = image.descriptor() else {
        return Err(Error::new(
            ErrorClass::Type,
            E_ADJUST_INPUT,
            "color.adjust `image` input must be an image resource".to_owned(),
        ));
    };
    require_linear(descriptor.color)?;

    let adjustment = Adjustment::resolve(params)?;

    // An optional mask gates the grade per pixel; it must size to the image.
    let mask_samples = match inputs.get("mask") {
        None => None,
        Some(mask_value) => {
            let ResourceDescriptor::Mask(mask_desc) = mask_value.descriptor() else {
                return Err(Error::new(
                    ErrorClass::Type,
                    E_ADJUST_MASK,
                    "color.adjust `mask` input must be a mask resource".to_owned(),
                ));
            };
            if mask_desc.extent != descriptor.extent {
                return Err(mask_extent_error(mask_desc.extent, descriptor.extent));
            }
            Some(mask_value.samples())
        }
    };

    // A pure-identity request with no mask is a verbatim passthrough, avoiding
    // any floating-point perturbation of the samples — on either backend.
    let samples = if adjustment.is_identity() && mask_samples.is_none() {
        image.samples().to_vec()
    } else {
        apply_backend(
            backend,
            image.samples(),
            image.channels(),
            descriptor.layout.has_alpha(),
            adjustment,
            mask_samples,
        )
    };

    let value = ResourceValue::new(
        ResourceDescriptor::Image(*descriptor),
        image.channels(),
        samples,
    )
    .map_err(|actual| {
        Error::new(
            ErrorClass::Execution,
            E_ADJUST_INPUT,
            format!("color.adjust produced a sample buffer of unexpected length {actual}"),
        )
    })?;

    let mut out = OutputValues::new();
    out.insert("image".to_owned(), value);
    Ok(out)
}

impl OpImplementation for Adjust {
    fn compute(
        &self,
        inputs: &InputValues,
        params: &serde_json::Value,
    ) -> std::result::Result<OutputValues, Error> {
        compute_backend(Backend::Reference, inputs, params)
    }
}

/// The `cpu.optimized@1` backend for `color.adjust@1`.
///
/// It applies the same fixed-order exposure/temperature/saturation grade as the
/// oracle, computed by the autovectorization-friendly kernel. `color.adjust` is
/// [`Bounded`](DeterminismTier::Bounded) (the `exp2` last bit varies), and the
/// kernel mirrors the reference operation order, so the result stays within the
/// op's envelope (the differential harness enforces it).
#[derive(Debug, Clone, Copy, Default)]
pub struct AdjustOptimized;

impl AdjustOptimized {
    /// Construct the optimized adjust backend.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }
}

impl OpImplementation for AdjustOptimized {
    fn compute(
        &self,
        inputs: &InputValues,
        params: &serde_json::Value,
    ) -> std::result::Result<OutputValues, Error> {
        compute_backend(Backend::Optimized, inputs, params)
    }
}

/// Extract the required `image` input descriptor, erroring if absent or non-image.
fn image_descriptor(inputs: &Descriptors) -> Result<&ImageDescriptor> {
    let image = inputs.get("image").ok_or_else(|| {
        Error::new(
            ErrorClass::Reference,
            E_ADJUST_INPUT,
            "color.adjust requires an `image` input".to_owned(),
        )
    })?;
    let ResourceDescriptor::Image(descriptor) = image else {
        return Err(Error::new(
            ErrorClass::Type,
            E_ADJUST_INPUT,
            "color.adjust `image` input must be an image resource".to_owned(),
        ));
    };
    Ok(descriptor)
}

/// If a `mask` descriptor is wired, verify it sizes to the image extent.
fn check_mask_extent(inputs: &Descriptors, image: &ImageDescriptor) -> Result<()> {
    let Some(mask) = inputs.get("mask") else {
        return Ok(());
    };
    let ResourceDescriptor::Mask(mask_desc) = mask else {
        return Err(Error::new(
            ErrorClass::Type,
            E_ADJUST_MASK,
            "color.adjust `mask` input must be a mask resource".to_owned(),
        ));
    };
    if mask_desc.extent == image.extent {
        Ok(())
    } else {
        Err(mask_extent_error(mask_desc.extent, image.extent))
    }
}

/// Build the mask/image extent-mismatch error.
fn mask_extent_error(mask: paintop_ir::Extent, image: paintop_ir::Extent) -> Error {
    Error::new(
        ErrorClass::Semantic,
        E_ADJUST_MASK,
        "color.adjust `mask` extent must match the `image` extent".to_owned(),
    )
    .with_context(
        ErrorContext::default()
            .with_actual(format!("{}x{}", mask.width, mask.height))
            .with_expected(format!("{}x{}", image.width, image.height)),
    )
}

/// The mandatory `cpu.reference@1` oracle implementation id.
fn reference_impl() -> Result<ImplId> {
    ImplId::new("cpu", "reference", 1)
}

/// The `cpu.optimized@1` autovectorized backend implementation id.
fn optimized_impl() -> Result<ImplId> {
    ImplId::new("cpu", "optimized", 1)
}

/// The verification declarations for `color.adjust@1`: a single-reference,
/// bounded, pointwise grading op. Differential does not apply (one
/// implementation). Perceptual is not applicable: the adjustment is a closed-form
/// numeric transform verified by analytic value tables and algebraic properties
/// (exposure composition, masked identity), not a perceptual-quality comparison.
/// Every other applicable category is covered by this module's analytic,
/// property, and metamorphic tests.
fn adjust_test_metadata() -> TestMetadata {
    use paintop_ir::{CategoryStatus, VerificationCategory, VerificationDeclarations};
    let mut decls = VerificationDeclarations::new();
    for category in [
        VerificationCategory::BuildHygiene,
        VerificationCategory::SchemaContract,
        VerificationCategory::AnalyticFixtures,
        VerificationCategory::PropertyTests,
        VerificationCategory::Metamorphic,
        // The op now exposes a cpu.optimized backend, so differential testing
        // applies: the cross-backend harness validates it against the oracle.
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
            "color.adjust is a closed-form numeric color transform verified by analytic value \
             tables and algebraic properties (exposure composition, zero-adjustment identity, \
             masked identity); there is no perceptual-quality metric to apply",
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
