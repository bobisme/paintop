//! The `composite.blend@1` operation: a restricted, exactly-pinned set of blend
//! modes applied with an opacity and a coverage mask (`OP_CATALOG` §7).
//!
//! `composite.blend` blends a source image `src` onto a destination image `dst`
//! under one of a **restricted** mode set, then mixes the blended result back into
//! `dst` by an effective coverage `k = opacity · mask`:
//!
//! ```text
//! out = dst + k · ( B_mode(src, dst) − dst )      (per channel)
//! ```
//!
//! so `k = 0` (opacity `0` *or* mask `0`) is the identity on `dst`, and `k = 1`
//! is the pure blend. The per-channel blend functions `B_mode` operate directly on
//! the **premultiplied linear** samples (`s = src`, `d = dst`), which is what makes
//! each mode an exact, closed-form arithmetic identity:
//!
//! | mode              | `B(s, d)`        | commutative |
//! |-------------------|------------------|:-----------:|
//! | `normal` / `over` | `s + d·(1 − αs)` | no          |
//! | `add`             | `s + d`          | yes         |
//! | `subtract`        | `d − s`          | no          |
//! | `multiply`        | `s · d`          | yes         |
//! | `screen`          | `s + d − s·d`    | yes         |
//! | `darken`          | `min(s, d)`      | yes         |
//! | `lighten`         | `max(s, d)`      | yes         |
//! | `difference`      | `|s − d|`        | yes         |
//!
//! For `normal`/`over` the alpha channel uses the premultiplied over update
//! `αo = αs + αd·(1 − αs)`; every other mode applies its arithmetic to the alpha
//! channel identically to the color channels. `overlay` and `soft-light` are
//! deliberately **omitted** until their exact premultiplied semantics are pinned.
//!
//! # Color space & determinism
//!
//! All modes are defined on premultiplied **linear** light: a nonlinear (`srgb`)
//! input, a straight-alpha input, or an image without alpha is rejected with a
//! [`semantic`](ErrorClass::Semantic) error. Every mode is a per-channel `f32`
//! arithmetic expression with the `k ∈ {0, 1}` mix extremes handled exactly, so the
//! op is [`Exact`](DeterminismTier::Exact). The op is **pointwise**.

use paintop_core::executor::{InputValues, OpImplementation, OutputValues, ResourceValue};
use paintop_ir::{
    AlphaRepresentation, AssertionResult, ColorEncoding, Descriptors, DeterminismTier, Error,
    ErrorClass, ErrorContext, Extent, ImageDescriptor, ImplId, InputRegions, InputSpec, OpContract,
    OperationManifest, OutputDescriptors, OutputRegions, OutputSpec, ParamSpec, ParamType,
    ResourceDescriptor, ResourceKind, Result, RoiCategory, RoiPolicy, TestMetadata,
};

/// The canonical id of the restricted-blend operation.
pub const BLEND_OP_ID: &str = "composite.blend@1";

/// A required input port (`src`, `dst`, or `mask`) was absent or carried the wrong
/// resource kind.
pub const E_BLEND_INPUT: &str = "E_BLEND_INPUT";

/// The `src` / `dst` images are in a representation this op cannot blend (nonlinear
/// encoding, straight alpha, or no alpha channel), or the ports disagree on
/// extent / layout.
pub const E_BLEND_SHAPE: &str = "E_BLEND_SHAPE";

/// The `mode` param was missing or not one of the restricted mode tokens, or the
/// `opacity` param was missing / non-finite / out of `[0, 1]`.
pub const E_BLEND_PARAM: &str = "E_BLEND_PARAM";

/// The op produced a sample buffer whose length disagrees with its descriptor.
pub const E_BLEND_BUFFER: &str = "E_BLEND_BUFFER";

/// The restricted, exactly-pinned blend modes (`OP_CATALOG` §7). `overlay` and
/// `soft-light` are intentionally absent until their semantics are pinned.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Mode {
    /// Premultiplied source-over.
    Normal,
    /// `s + d`.
    Add,
    /// `d − s`.
    Subtract,
    /// `s · d`.
    Multiply,
    /// `s + d − s·d`.
    Screen,
    /// `min(s, d)`.
    Darken,
    /// `max(s, d)`.
    Lighten,
    /// `|s − d|`.
    Difference,
}

impl Mode {
    /// Resolve the `mode` string token to a restricted [`Mode`].
    fn parse(token: &str) -> Result<Self> {
        match token {
            "normal" | "over" => Ok(Self::Normal),
            "add" => Ok(Self::Add),
            "subtract" => Ok(Self::Subtract),
            "multiply" => Ok(Self::Multiply),
            "screen" => Ok(Self::Screen),
            "darken" => Ok(Self::Darken),
            "lighten" => Ok(Self::Lighten),
            "difference" => Ok(Self::Difference),
            other => Err(Error::new(
                ErrorClass::Schema,
                E_BLEND_PARAM,
                format!(
                    "composite.blend `mode` must be one of normal/over, add, subtract, multiply, \
                     screen, darken, lighten, difference; got `{other}` (overlay/soft-light are \
                     not yet supported)"
                ),
            )
            .with_context(ErrorContext::default().with_actual(other))),
        }
    }

    /// The kebab/lower-case mode choices for the manifest, in declaration order.
    fn choices() -> Vec<String> {
        [
            "normal",
            "over",
            "add",
            "subtract",
            "multiply",
            "screen",
            "darken",
            "lighten",
            "difference",
        ]
        .iter()
        .map(|s| (*s).to_owned())
        .collect()
    }

    /// Map this restricted mode to the optimized kernel's identical mode set, so
    /// the optimized backend computes the exact same per-channel arithmetic.
    const fn to_kernel(self) -> crate::optimized::kernels::BlendMode {
        use crate::optimized::kernels::BlendMode as K;
        match self {
            Self::Normal => K::Normal,
            Self::Add => K::Add,
            Self::Subtract => K::Subtract,
            Self::Multiply => K::Multiply,
            Self::Screen => K::Screen,
            Self::Darken => K::Darken,
            Self::Lighten => K::Lighten,
            Self::Difference => K::Difference,
        }
    }

    /// The per-color-channel blend value `B(s, d)`. `inv_alpha_s = 1 − αs` is the
    /// source-over factor for `Normal` (ignored by every other mode).
    fn blend_channel(self, s: f32, d: f32, inv_alpha_s: f32) -> f32 {
        match self {
            Self::Normal => d.mul_add(inv_alpha_s, s),
            Self::Add => s + d,
            Self::Subtract => d - s,
            Self::Multiply => s * d,
            // s + d − s·d, as the fused form s·(−d) + (s + d).
            Self::Screen => s.mul_add(-d, s + d),
            Self::Darken => s.min(d),
            Self::Lighten => s.max(d),
            Self::Difference => (s - d).abs(),
        }
    }
}

/// The resolved blend params: the mode and the validated opacity in `[0, 1]`.
#[derive(Debug, Clone, Copy)]
struct BlendParams {
    mode: Mode,
    opacity: f32,
}

impl BlendParams {
    /// Parse and validate the `mode` and `opacity` params.
    fn resolve(params: &serde_json::Value) -> Result<Self> {
        let token = params
            .get("mode")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| {
                Error::new(
                    ErrorClass::Schema,
                    E_BLEND_PARAM,
                    "composite.blend requires a string `mode` parameter".to_owned(),
                )
            })?;
        let mode = Mode::parse(token)?;

        let opacity = params.get("opacity").map_or(Ok(1.0), |value| {
            let n = value.as_f64().ok_or_else(|| {
                Error::new(
                    ErrorClass::Schema,
                    E_BLEND_PARAM,
                    "composite.blend `opacity` must be a number".to_owned(),
                )
                .with_context(ErrorContext::default().with_actual(value.to_string()))
            })?;
            if !(n.is_finite() && (0.0..=1.0).contains(&n)) {
                return Err(Error::new(
                    ErrorClass::Schema,
                    E_BLEND_PARAM,
                    format!("composite.blend `opacity` must be a finite value in [0, 1], got {n}"),
                ));
            }
            #[allow(
                clippy::cast_possible_truncation,
                reason = "opacity in [0, 1] stored as the f32 sample type"
            )]
            Ok(n as f32)
        })?;

        Ok(Self { mode, opacity })
    }
}

/// Validate that a `src` / `dst` image may be blended in premultiplied linear
/// light: linear-encoded, premultiplied, with an alpha channel.
fn check_color_image(descriptor: &ImageDescriptor, port: &str) -> Result<()> {
    if descriptor.color == ColorEncoding::Srgb {
        return Err(shape_error(format!(
            "composite.blend requires linear-light color; the `{port}` input is `srgb`-encoded. \
             Insert a color.convert to linear-srgb first."
        )));
    }
    if !descriptor.layout.has_alpha() {
        return Err(shape_error(format!(
            "composite.blend requires the `{port}` image to have an alpha channel (GrayA or Rgba)"
        )));
    }
    if descriptor.alpha != AlphaRepresentation::Premultiplied {
        return Err(shape_error(format!(
            "composite.blend blends in premultiplied space; premultiply the `{port}` image first"
        )));
    }
    Ok(())
}

/// Validate the three ports together and return the output descriptor (the `dst`
/// descriptor: blending onto `dst` never changes its type).
fn check_and_retarget(
    src: &ImageDescriptor,
    dst: &ImageDescriptor,
    mask_extent: Extent,
) -> Result<ImageDescriptor> {
    check_color_image(src, "src")?;
    check_color_image(dst, "dst")?;
    if src.extent != dst.extent {
        return Err(shape_error(format!(
            "composite.blend: the `src` and `dst` images must share an extent (src {:?} vs dst {:?})",
            src.extent, dst.extent
        )));
    }
    if src.layout != dst.layout {
        return Err(shape_error(format!(
            "composite.blend: the `src` and `dst` images must share a channel layout (src {:?} vs \
             dst {:?})",
            src.layout, dst.layout
        )));
    }
    if mask_extent != dst.extent {
        return Err(shape_error(format!(
            "composite.blend: the `mask` must share the images' extent (mask {mask_extent:?} vs \
             image {:?})",
            dst.extent
        )));
    }
    Ok(*dst)
}

/// Build a shape [`semantic`](ErrorClass::Semantic) error.
fn shape_error(detail: String) -> Error {
    Error::new(ErrorClass::Semantic, E_BLEND_SHAPE, detail)
}

/// Blend `src` onto `dst` through `mask` at `opacity` using `mode`, per pixel and
/// channel.
///
/// `channels` is the interleaved color+alpha sample count per pixel; the alpha
/// channel is the last of each pixel and the mask is one coverage sample per pixel.
/// The output is `dst + k·(B_mode(src, dst) − dst)` with `k = opacity·mask`; at
/// `k == 0` the `dst` sample is returned verbatim (the identity guarantee).
fn blend(src: &[f32], dst: &[f32], mask: &[f32], channels: u32, params: BlendParams) -> Vec<f32> {
    let channel_count = channels as usize;
    if channel_count == 0 {
        return dst.to_vec();
    }
    let alpha_index = channel_count - 1;
    let mut out = Vec::with_capacity(dst.len());
    for ((src_pixel, dst_pixel), &coverage) in src
        .chunks_exact(channel_count)
        .zip(dst.chunks_exact(channel_count))
        .zip(mask.iter())
    {
        let k = params.opacity * coverage;
        let inv_alpha_s = 1.0 - src_pixel[alpha_index];
        for (&s, &d) in src_pixel.iter().zip(dst_pixel.iter()) {
            // k == 0 (opacity 0 or mask 0): the dst passes through verbatim — the
            // identity guarantee, matched by bit pattern (clippy-clean exact eq).
            let sample = if k.to_bits() == 0.0_f32.to_bits() {
                d
            } else {
                let blended = params.mode.blend_channel(s, d, inv_alpha_s);
                // dst + k·(blended − dst): a single deterministic f32 multiply-add.
                k.mul_add(blended - d, d)
            };
            out.push(sample);
        }
    }
    out
}

/// Read a required image port's descriptor.
fn image_descriptor<'a>(inputs: &'a Descriptors, port: &str) -> Result<&'a ImageDescriptor> {
    let resource = inputs.get(port).ok_or_else(|| {
        Error::new(
            ErrorClass::Reference,
            E_BLEND_INPUT,
            format!("composite.blend requires a `{port}` input"),
        )
    })?;
    let ResourceDescriptor::Image(descriptor) = resource else {
        return Err(Error::new(
            ErrorClass::Type,
            E_BLEND_INPUT,
            format!("composite.blend `{port}` input must be an image resource"),
        ));
    };
    Ok(descriptor)
}

/// Read the `mask` port's extent.
fn mask_extent(inputs: &Descriptors) -> Result<Extent> {
    let resource = inputs.get("mask").ok_or_else(|| {
        Error::new(
            ErrorClass::Reference,
            E_BLEND_INPUT,
            "composite.blend requires a `mask` input".to_owned(),
        )
    })?;
    let ResourceDescriptor::Mask(mask) = resource else {
        return Err(Error::new(
            ErrorClass::Type,
            E_BLEND_INPUT,
            "composite.blend `mask` input must be a mask resource".to_owned(),
        ));
    };
    Ok(mask.extent)
}

/// The `composite.blend@1` operation: `src` + `dst` + `mask` → the blended
/// `image`.
#[derive(Debug, Clone, Copy, Default)]
pub struct Blend;

impl Blend {
    /// Construct the restricted-blend operation.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    /// The declared manifest for `composite.blend@1`.
    ///
    /// # Errors
    /// Propagates the [`schema`](ErrorClass::Schema) error if the hard-coded op /
    /// impl ids are somehow invalid (they are not).
    pub fn manifest() -> Result<OperationManifest> {
        Ok(OperationManifest {
            id: BLEND_OP_ID.parse()?,
            impl_version: 1,
            summary: "Blend a source image onto a destination through a coverage mask at an \
                      opacity, over a restricted exactly-pinned mode set (normal/over, add, \
                      subtract, multiply, screen, darken, lighten, difference) in premultiplied \
                      linear light; out = dst + opacity·mask·(B_mode(src, dst) − dst)."
                .to_owned(),
            determinism: DeterminismTier::Exact,
            roi: RoiPolicy {
                category: RoiCategory::Pointwise,
                halo_px: None,
            },
            inputs: vec![
                InputSpec {
                    name: "src".to_owned(),
                    kind: ResourceKind::Image,
                    required: true,
                    doc: "The premultiplied linear-light source image being blended.".to_owned(),
                },
                InputSpec {
                    name: "dst".to_owned(),
                    kind: ResourceKind::Image,
                    required: true,
                    doc: "The premultiplied linear-light destination image blended onto."
                        .to_owned(),
                },
                InputSpec {
                    name: "mask".to_owned(),
                    kind: ResourceKind::Mask,
                    required: true,
                    doc:
                        "The coverage mask in [0, 1] modulating the blend; a full (all-ones) mask \
                          blends everywhere."
                            .to_owned(),
                },
            ],
            outputs: vec![OutputSpec {
                name: "image".to_owned(),
                kind: ResourceKind::Image,
                doc: "The blended image (the dst descriptor; same extent/layout).".to_owned(),
            }],
            params: vec![
                ParamSpec {
                    name: "mode".to_owned(),
                    ty: ParamType::String,
                    unit: None,
                    required: true,
                    default: None,
                    choices: Mode::choices(),
                    doc: "The blend mode; one of the restricted exactly-pinned set.".to_owned(),
                },
                ParamSpec {
                    name: "opacity".to_owned(),
                    ty: ParamType::Float,
                    unit: None,
                    required: false,
                    default: Some(serde_json::json!(1.0)),
                    choices: vec![],
                    doc: "The global blend opacity in [0, 1]; 0 is the identity on dst.".to_owned(),
                },
            ],
            implementations: vec![reference_impl()?, optimized_impl()?],
            test: blend_test_metadata(),
        })
    }
}

impl OpContract for Blend {
    fn declared_inputs(&self) -> Vec<(String, ResourceKind)> {
        vec![
            ("src".to_owned(), ResourceKind::Image),
            ("dst".to_owned(), ResourceKind::Image),
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
        let src = image_descriptor(inputs, "src")?;
        let dst = image_descriptor(inputs, "dst")?;
        let extent = mask_extent(inputs)?;
        // Validate params at infer time so a bad mode/opacity fails type-checking.
        BlendParams::resolve(params)?;
        let out_descriptor = check_and_retarget(src, dst, extent)?;

        let mut out = OutputDescriptors::new();
        out.insert(
            "image".to_owned(),
            ResourceDescriptor::Image(out_descriptor),
        );
        Ok(out)
    }

    fn required_inputs(
        &self,
        requested_outputs: &OutputRegions,
        _inputs: &Descriptors,
        _params: &serde_json::Value,
    ) -> Result<InputRegions> {
        // Pointwise: each output pixel reads the co-located src, dst, and mask.
        let mut regions = InputRegions::new();
        if let Some(region) = requested_outputs.get("image") {
            for port in ["src", "dst", "mask"] {
                regions.insert(port.to_owned(), *region);
            }
        }
        Ok(regions)
    }

    fn validate_postconditions(
        &self,
        outputs: &OutputDescriptors,
        _params: &serde_json::Value,
    ) -> Result<Vec<AssertionResult>> {
        let Some(ResourceDescriptor::Image(image)) = outputs.get("image") else {
            return Ok(vec![AssertionResult::fail(
                "produces_image",
                "no `image` output produced",
            )]);
        };
        let mut results = vec![AssertionResult::pass("produces_image")];
        results.push(if image.alpha == AlphaRepresentation::Premultiplied {
            AssertionResult::pass("stays_premultiplied")
        } else {
            AssertionResult::fail(
                "stays_premultiplied",
                format!("output alpha {:?} is not Premultiplied", image.alpha),
            )
        });
        results.push(if image.color.is_linear_light() {
            AssertionResult::pass("stays_linear")
        } else {
            AssertionResult::fail(
                "stays_linear",
                format!("output encoding {:?} is not linear", image.color),
            )
        });
        Ok(results)
    }
}

/// The compute backend serving `composite.blend`: the scalar reference oracle or
/// the autovectorization-friendly `cpu.optimized` kernel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Backend {
    /// The scalar reference oracle ([`blend`]).
    Reference,
    /// The `cpu.optimized` blend kernel ([`crate::optimized::kernels`]).
    Optimized,
}

/// Shared compute for both backends: validate the three ports + params, then blend.
fn compute_backend(
    backend: Backend,
    inputs: &InputValues,
    params: &serde_json::Value,
) -> std::result::Result<OutputValues, Error> {
    let src = input_value(inputs, "src")?;
    let dst = input_value(inputs, "dst")?;
    let mask = input_value(inputs, "mask")?;

    let ResourceDescriptor::Image(src_descriptor) = src.descriptor() else {
        return Err(input_type_error("src"));
    };
    let ResourceDescriptor::Image(dst_descriptor) = dst.descriptor() else {
        return Err(input_type_error("dst"));
    };
    let ResourceDescriptor::Mask(mask_descriptor) = mask.descriptor() else {
        return Err(Error::new(
            ErrorClass::Type,
            E_BLEND_INPUT,
            "composite.blend `mask` input must be a mask resource".to_owned(),
        ));
    };

    let blend_params = BlendParams::resolve(params)?;
    let out_descriptor =
        check_and_retarget(src_descriptor, dst_descriptor, mask_descriptor.extent)?;
    let samples = match backend {
        Backend::Reference => blend(
            src.samples(),
            dst.samples(),
            mask.samples(),
            dst.channels(),
            blend_params,
        ),
        Backend::Optimized => crate::optimized::kernels::composite_blend(
            src.samples(),
            dst.samples(),
            mask.samples(),
            dst.channels() as usize,
            blend_params.mode.to_kernel(),
            blend_params.opacity,
        ),
    };

    let value = ResourceValue::new(
        ResourceDescriptor::Image(out_descriptor),
        dst.channels(),
        samples,
    )
    .map_err(|actual| {
        Error::new(
            ErrorClass::Execution,
            E_BLEND_BUFFER,
            format!("composite.blend produced a sample buffer of unexpected length {actual}"),
        )
    })?;

    let mut out = OutputValues::new();
    out.insert("image".to_owned(), value);
    Ok(out)
}

impl OpImplementation for Blend {
    fn compute(
        &self,
        inputs: &InputValues,
        params: &serde_json::Value,
    ) -> std::result::Result<OutputValues, Error> {
        compute_backend(Backend::Reference, inputs, params)
    }
}

/// The `cpu.optimized@1` backend for `composite.blend@1`.
///
/// It computes the same `out = dst + opacity*mask*(B_mode(src, dst) - dst)` over
/// the same restricted, exactly-pinned mode set as the oracle, via the
/// autovectorization-friendly kernel, with the identical `k == 0` verbatim-`dst`
/// identity. `composite.blend` is [`Exact`](DeterminismTier::Exact), so the
/// optimized result is **bit-identical** to the reference (the differential harness
/// enforces it).
#[derive(Debug, Clone, Copy, Default)]
pub struct BlendOptimized;

impl BlendOptimized {
    /// Construct the optimized blend backend.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }
}

impl OpImplementation for BlendOptimized {
    fn compute(
        &self,
        inputs: &InputValues,
        params: &serde_json::Value,
    ) -> std::result::Result<OutputValues, Error> {
        compute_backend(Backend::Optimized, inputs, params)
    }
}

/// Read a required input *value* port, erroring if absent.
fn input_value<'a>(
    inputs: &'a InputValues,
    port: &str,
) -> std::result::Result<&'a ResourceValue, Error> {
    inputs.get(port).ok_or_else(|| {
        Error::new(
            ErrorClass::Reference,
            E_BLEND_INPUT,
            format!("composite.blend requires a `{port}` input value"),
        )
    })
}

/// The wrong-resource-kind error for a color image port.
fn input_type_error(port: &str) -> Error {
    Error::new(
        ErrorClass::Type,
        E_BLEND_INPUT,
        format!("composite.blend `{port}` input must be an image resource"),
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

/// The verification declarations for `composite.blend@1`: an exact,
/// single-reference per-channel blend over a restricted, exactly-pinned mode set.
/// Differential does not apply (one implementation). Perceptual is not applicable:
/// each mode is a closed-form arithmetic identity verified by per-mode analytic
/// fixtures, opacity/mask identity, and commutativity properties, with no
/// perceptual-quality metric.
fn blend_test_metadata() -> TestMetadata {
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
            "composite.blend is a bit-exact per-channel blend over a restricted mode set verified \
             by per-mode formula fixtures, opacity/mask identity, and commutativity properties; \
             there is no perceptual-quality metric to apply",
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
