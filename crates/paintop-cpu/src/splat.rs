//! The `paint.gaussian_splats@1` operation: paint an inline batch of anisotropic
//! Gaussian splats onto a base image (`OP_CATALOG` §4, `AGENT_VERIFICATION` §3.5,
//! `M0_DECISIONS` D2/Q6, `IR_SPEC` §20).
//!
//! Each splat is an oriented 2-D Gaussian — a center `μ`, anisotropic standard
//! deviations `σ = (σx, σy)`, and a rotation `θ` of its covariance axes — carrying
//! a premultiplied-ready color and a scalar `opacity`. The Gaussian weight
//!
//! ```text
//! w(p) = opacity * exp(-½ (p - μ)^T Σ^{-1} (p - μ))
//! ```
//!
//! (with `Σ = R diag(σx², σy²) R^T`) modulates the splat's coverage. Splats are
//! accumulated **in array order** onto the `base` image, each compositing *over*
//! the running result. Per `M0_DECISIONS` D2 the op paints onto an edit layer
//! (here: the base, returned as the `image` output); locality against the original
//! is enforced downstream by `composite.masked_replace@1`, not inside this op.
//!
//! # Batch policy (Q6)
//!
//! Splat batches are inline in `params` and bounded by `policy.resources.max_splats`.
//! That bound is threaded to the op as the optional `max_splats` param
//! (the normalizer fills it from the plan policy; absent it defaults to
//! [`DEFAULT_MAX_SPLATS`]). A batch exceeding the bound is rejected with a
//! [`policy`](ErrorClass::Policy) error *before* any pixel is touched. An **empty**
//! batch is legal and is the identity (the base passes through unchanged). The op
//! is designed so a future large-batch path (a `splats` input port backed by a
//! content-addressed blob) is purely additive — no CAS infrastructure is built now.
//!
//! # Color space
//!
//! The op paints in a linear-light premultiplied space: the `base` must be
//! `Premultiplied` linear color with an alpha channel, and the optional `space`
//! param, when present, must name the base's linear color encoding. A nonlinear
//! (`srgb`) base, a straight-alpha base, or an image without alpha is rejected with
//! a [`semantic`](ErrorClass::Semantic) error rather than producing wrong color.
//!
//! # Blend modes
//!
//! Two blend modes are supported per splat: `normal` (source-over) and `multiply`.
//! Both operate on premultiplied color; the splat's premultiplied color is
//! `color.rgb * color.a` scaled by the spatial weight, and the splat alpha is
//! `color.a * weight`.
//!
//! # Determinism
//!
//! The op is `bounded`: the Gaussian uses `exp`, whose last bit is not guaranteed
//! identical across platforms, so accumulated coverage is asserted within a
//! tolerance rather than bit-exactly. The geometry and accumulation order are an
//! exact function of the params.
//!
//! # Rejected requests
//!
//! - A non-positive (or non-finite) `sigma_px` is a degenerate splat with a
//!   singular covariance and is rejected as [`schema`](ErrorClass::Schema).
//! - A `color` or `opacity` outside `[0, 1]` (or non-finite) is rejected as
//!   [`schema`](ErrorClass::Schema): coverage and color stay bounded.
//! - A batch larger than `max_splats` is rejected as [`policy`](ErrorClass::Policy).

use paintop_core::executor::{InputValues, OpImplementation, OutputValues, ResourceValue};
use paintop_ir::{
    AlphaRepresentation, AssertionResult, ColorEncoding, Descriptors, DeterminismTier, Error,
    ErrorClass, ErrorContext, Extent, ImageDescriptor, ImplId, InputRegions, InputSpec, OpContract,
    OperationManifest, OutputDescriptors, OutputRegions, OutputSpec, ParamSpec, ParamType,
    ResourceDescriptor, ResourceKind, Result, RoiCategory, RoiPolicy, TestMetadata,
};

/// The canonical id of the Gaussian-splat paint operation.
pub const SPLAT_OP_ID: &str = "paint.gaussian_splats@1";

/// The `base` input was absent or carried a non-image descriptor.
pub const E_SPLAT_INPUT: &str = "E_SPLAT_INPUT";

/// A splat-batch parameter (`splats`, a splat field, `space`, `blend`) was
/// missing, the wrong shape, non-finite, out of range, or degenerate.
pub const E_SPLAT_PARAM: &str = "E_SPLAT_PARAM";

/// The `base` image is in a representation this op cannot paint onto (nonlinear
/// encoding, straight alpha, or no alpha channel).
pub const E_SPLAT_BASE: &str = "E_SPLAT_BASE";

/// The inline splat batch exceeds the policy bound `max_splats`.
pub const E_SPLAT_BUDGET: &str = "E_SPLAT_BUDGET";

/// The op produced a sample buffer whose length disagrees with its descriptor.
pub const E_SPLAT_BUFFER: &str = "E_SPLAT_BUFFER";

/// The default inline-batch bound when the plan policy does not supply one
/// (`IR_SPEC` §16 policy example).
pub const DEFAULT_MAX_SPLATS: u64 = 100_000;

/// The blend mode this op composites a splat with. Both operate on premultiplied
/// color.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BlendMode {
    /// Source-over: the splat composites over the running result.
    Normal,
    /// Multiply: the splat modulates the running result toward its color.
    Multiply,
}

impl BlendMode {
    /// The token a blend mode is named by in JSON.
    const NORMAL: &'static str = "normal";
    /// The token for the multiply mode.
    const MULTIPLY: &'static str = "multiply";

    /// Parse a blend-mode token, defaulting to [`Normal`](Self::Normal) when
    /// absent.
    fn parse(value: Option<&serde_json::Value>) -> Result<Self> {
        let Some(value) = value else {
            return Ok(Self::Normal);
        };
        let token = value
            .as_str()
            .ok_or_else(|| param_error("`blend` must be a string", "blend", value))?;
        match token {
            Self::NORMAL => Ok(Self::Normal),
            Self::MULTIPLY => Ok(Self::Multiply),
            other => Err(Error::new(
                ErrorClass::Schema,
                E_SPLAT_PARAM,
                format!(
                    "paint.gaussian_splats only supports `blend: normal | multiply`, got `{other}`"
                ),
            )),
        }
    }
}

/// The resolved geometry and appearance of one Gaussian splat, all in physical
/// pixels (the angle in radians, color and opacity in `[0, 1]`).
#[derive(Debug, Clone, Copy, PartialEq)]
struct Splat {
    /// Center x in pixel coordinates.
    cx: f64,
    /// Center y in pixel coordinates.
    cy: f64,
    /// Standard deviation along the (pre-rotation) x axis, strictly positive.
    sigma_x: f64,
    /// Standard deviation along the (pre-rotation) y axis, strictly positive.
    sigma_y: f64,
    /// Rotation of the covariance axes, radians (counter-clockwise).
    angle_rad: f64,
    /// Straight color `[r, g, b, a]` in `[0, 1]`.
    color: [f64; 4],
    /// Scalar opacity in `[0, 1]` modulating the Gaussian weight.
    opacity: f64,
    /// How this splat composites with the running result.
    blend: BlendMode,
}

impl Splat {
    /// Parse and validate one splat object.
    ///
    /// # Errors
    /// [`schema`](ErrorClass::Schema) / [`E_SPLAT_PARAM`] if a field is missing,
    /// the wrong shape, non-finite, degenerate (a non-positive `sigma_px`), or out
    /// of the `[0, 1]` range for color / opacity.
    fn resolve(index: usize, value: &serde_json::Value) -> Result<Self> {
        let object = value.as_object().ok_or_else(|| {
            param_error(
                "each splat must be an object",
                &splat_field(index, ""),
                value,
            )
        })?;
        let get = |field: &str| object.get(field);

        let (cx, cy) = pair(get("center_px"), &splat_field(index, "center_px"))?;
        let (sigma_x, sigma_y) = pair(get("sigma_px"), &splat_field(index, "sigma_px"))?;
        for (name, s) in [("sigma_px.x", sigma_x), ("sigma_px.y", sigma_y)] {
            if !(s.is_finite() && s > 0.0) {
                return Err(degenerate(&splat_field(index, name), s));
            }
        }

        let angle_rad = match get("angle_rad") {
            None => 0.0,
            Some(v) => finite(v, &splat_field(index, "angle_rad"))?,
        };

        let color = color4(get("color"), &splat_field(index, "color"))?;

        let opacity = match get("opacity") {
            None => 1.0,
            Some(v) => unit(v, &splat_field(index, "opacity"))?,
        };

        let blend = BlendMode::parse(get("blend"))?;

        Ok(Self {
            cx,
            cy,
            sigma_x,
            sigma_y,
            angle_rad,
            color,
            opacity,
            blend,
        })
    }

    /// The Gaussian spatial weight in `[0, opacity]` at a sample `(x, y)`.
    ///
    /// Evaluates `opacity * exp(-½ (p-μ)^T Σ^{-1} (p-μ))` in the splat's local
    /// (axis-aligned) frame, where the precision matrix is diagonal `diag(1/σx²,
    /// 1/σy²)`.
    #[must_use]
    fn weight(&self, sample_x: f64, sample_y: f64) -> f64 {
        let off_x = sample_x - self.cx;
        let off_y = sample_y - self.cy;
        let (sin, cos) = self.angle_rad.sin_cos();
        // Rotate the offset into the covariance's axis-aligned local frame.
        let local_u = sin.mul_add(off_y, cos * off_x);
        let local_v = cos.mul_add(off_y, -sin * off_x);
        let nu = local_u / self.sigma_x;
        let nv = local_v / self.sigma_y;
        let mahalanobis = nu.mul_add(nu, nv * nv);
        self.opacity * (-0.5 * mahalanobis).exp()
    }
}

/// A premultiplied RGBA pixel as `f64` for accumulation.
#[derive(Debug, Clone, Copy)]
struct Pixel {
    /// Premultiplied red.
    r: f64,
    /// Premultiplied green.
    g: f64,
    /// Premultiplied blue.
    b: f64,
    /// Coverage / alpha.
    a: f64,
}

impl Splat {
    /// Composite this splat at coverage `weight` over a running premultiplied
    /// `dst` pixel, returning the new pixel.
    #[must_use]
    fn composite(&self, dst: Pixel, weight: f64) -> Pixel {
        // The splat's premultiplied source for this sample: straight color
        // premultiplied by its own alpha, then scaled by the spatial coverage.
        let src_a = self.color[3] * weight;
        let pre = self.color[3] * weight;
        let src = Pixel {
            r: self.color[0] * pre,
            g: self.color[1] * pre,
            b: self.color[2] * pre,
            a: src_a,
        };
        match self.blend {
            BlendMode::Normal => {
                // Source-over with premultiplied color: out = src + dst*(1 - src_a).
                let inv = 1.0 - src_a;
                Pixel {
                    r: dst.r.mul_add(inv, src.r),
                    g: dst.g.mul_add(inv, src.g),
                    b: dst.b.mul_add(inv, src.b),
                    a: dst.a.mul_add(inv, src.a),
                }
            }
            BlendMode::Multiply => {
                // Multiply blends the *un-premultiplied* colors, then re-applies
                // coverage as a source-over of the multiplied result. With
                // coverage `c = src_a`, lerp dst toward dst*color by c on each
                // channel; alpha follows source-over so coverage only grows.
                let inv = 1.0 - src_a;
                Pixel {
                    r: dst.r.mul_add(inv, dst.r * self.color[0] * src_a),
                    g: dst.g.mul_add(inv, dst.g * self.color[1] * src_a),
                    b: dst.b.mul_add(inv, dst.b * self.color[2] * src_a),
                    a: dst.a.mul_add(inv, src.a),
                }
            }
        }
    }
}

/// The resolved request: the validated batch and its declared paint space.
#[derive(Debug, Clone, PartialEq)]
struct SplatRequest {
    /// The validated splats, in accumulation order.
    splats: Vec<Splat>,
}

impl SplatRequest {
    /// Parse and validate the whole splat-paint request from the resolved params,
    /// enforcing the `max_splats` budget and the `space` constraint against the
    /// base encoding.
    ///
    /// # Errors
    /// [`schema`](ErrorClass::Schema) for a malformed batch or splat, or a `space`
    /// that disagrees with the linear base encoding; [`policy`](ErrorClass::Policy)
    /// if the batch exceeds `max_splats`.
    fn resolve(params: &serde_json::Value, base: &ImageDescriptor) -> Result<Self> {
        check_base(base)?;
        check_space(params, base)?;

        let max_splats = max_splats(params)?;
        let splats_value = params
            .get("splats")
            .ok_or_else(|| param_error("missing required parameter", "splats", &NULL))?;
        let array = splats_value.as_array().ok_or_else(|| {
            param_error(
                "`splats` must be an array of splat objects",
                "splats",
                splats_value,
            )
        })?;

        // Enforce the inline-batch budget before resolving any element, so an
        // oversized request fails on `max_splats` rather than on the first bad
        // splat it happens to contain.
        let count = array.len() as u64;
        if count > max_splats {
            return Err(Error::new(
                ErrorClass::Policy,
                E_SPLAT_BUDGET,
                format!(
                    "paint.gaussian_splats batch of {count} splats exceeds policy max_splats \
                     {max_splats}"
                ),
            )
            .with_context(
                ErrorContext::default()
                    .with_actual(count.to_string())
                    .with_expected(format!("<= {max_splats}")),
            ));
        }

        let mut splats = Vec::with_capacity(array.len());
        for (index, value) in array.iter().enumerate() {
            splats.push(Splat::resolve(index, value)?);
        }
        Ok(Self { splats })
    }

    /// Paint the batch onto a premultiplied RGBA base, returning the row-major
    /// interleaved `f32` result. `channels` must be 4 (RGBA); other layouts are
    /// rejected upstream by [`check_base`].
    #[must_use]
    fn paint(&self, base: &[f32], extent: Extent) -> Vec<f32> {
        let width = extent.width as usize;
        let height = extent.height as usize;
        let mut out = Vec::with_capacity(base.len());
        for j in 0..height {
            #[allow(clippy::cast_precision_loss, reason = "pixel index well within f64")]
            let y = j as f64 + 0.5;
            for i in 0..width {
                #[allow(clippy::cast_precision_loss, reason = "pixel index well within f64")]
                let x = i as f64 + 0.5;
                let base_index = (j * width + i) * 4;
                let mut pixel = Pixel {
                    r: f64::from(base[base_index]),
                    g: f64::from(base[base_index + 1]),
                    b: f64::from(base[base_index + 2]),
                    a: f64::from(base[base_index + 3]),
                };
                for splat in &self.splats {
                    let weight = splat.weight(x, y);
                    pixel = splat.composite(pixel, weight);
                }
                #[allow(
                    clippy::cast_possible_truncation,
                    reason = "premultiplied accumulation stored as the image's f32"
                )]
                {
                    out.push(pixel.r as f32);
                    out.push(pixel.g as f32);
                    out.push(pixel.b as f32);
                    out.push(pixel.a as f32);
                }
            }
        }
        out
    }
}

/// The `null` JSON value, reused for missing-field error contexts.
const NULL: serde_json::Value = serde_json::Value::Null;

/// The dotted path of a splat-batch field, e.g. `splats[2].sigma_px`.
fn splat_field(index: usize, field: &str) -> String {
    if field.is_empty() {
        format!("splats[{index}]")
    } else {
        format!("splats[{index}].{field}")
    }
}

/// Read the `max_splats` budget from params, defaulting to [`DEFAULT_MAX_SPLATS`].
fn max_splats(params: &serde_json::Value) -> Result<u64> {
    let Some(value) = params.get("max_splats") else {
        return Ok(DEFAULT_MAX_SPLATS);
    };
    let n = value
        .as_u64()
        .ok_or_else(|| param_error("must be a non-negative integer", "max_splats", value))?;
    Ok(n)
}

/// Validate that the `base` image may be painted onto: linear, premultiplied,
/// with an alpha channel.
fn check_base(base: &ImageDescriptor) -> Result<()> {
    if base.color == ColorEncoding::Srgb {
        return Err(Error::new(
            ErrorClass::Semantic,
            E_SPLAT_BASE,
            "paint.gaussian_splats requires linear-light color; the base is `srgb`-encoded. \
             Insert a color.convert to linear-srgb first."
                .to_owned(),
        )
        .with_context(
            ErrorContext::default()
                .with_actual("srgb")
                .with_expected("linear-srgb | raw-linear"),
        ));
    }
    if !base.layout.has_alpha() {
        return Err(Error::new(
            ErrorClass::Semantic,
            E_SPLAT_BASE,
            "paint.gaussian_splats requires a base image with an alpha channel (Rgba)".to_owned(),
        ));
    }
    if base.alpha != AlphaRepresentation::Premultiplied {
        return Err(Error::new(
            ErrorClass::Semantic,
            E_SPLAT_BASE,
            "paint.gaussian_splats paints in premultiplied space; premultiply the base first"
                .to_owned(),
        )
        .with_context(
            ErrorContext::default()
                .with_actual(format!("{:?}", base.alpha))
                .with_expected("Premultiplied"),
        ));
    }
    Ok(())
}

/// Validate the optional `space` param: when present it must name the base's
/// linear color encoding.
fn check_space(params: &serde_json::Value, base: &ImageDescriptor) -> Result<()> {
    let Some(value) = params.get("space") else {
        return Ok(());
    };
    let token = value
        .as_str()
        .ok_or_else(|| param_error("`space` must be a string", "space", value))?;
    // `check_base` has already rejected the `Srgb` encoding, so only the two
    // linear encodings reach here; the wildcard keeps the match total against the
    // non-exhaustive enum and maps the (unreachable) `Srgb` to its own token.
    let expected = if base.color == ColorEncoding::RawLinear {
        "raw-linear"
    } else {
        "linear-srgb"
    };
    if token != expected {
        return Err(Error::new(
            ErrorClass::Semantic,
            E_SPLAT_PARAM,
            format!(
                "paint.gaussian_splats `space` `{token}` does not match the base encoding \
                 `{expected}`"
            ),
        )
        .with_context(
            ErrorContext::default()
                .with_actual(token.to_owned())
                .with_expected(expected.to_owned()),
        ));
    }
    Ok(())
}

/// Extract a required `[x, y]` numeric pair from an optional value.
fn pair(value: Option<&serde_json::Value>, name: &str) -> Result<(f64, f64)> {
    let value = value.ok_or_else(|| param_error("missing required field", name, &NULL))?;
    let array = value
        .as_array()
        .ok_or_else(|| param_error("must be a [x, y] array", name, value))?;
    if array.len() != 2 {
        return Err(param_error("must have exactly two elements", name, value));
    }
    Ok((finite(&array[0], name)?, finite(&array[1], name)?))
}

/// Extract a required `[r, g, b, a]` color, each component finite and in `[0, 1]`.
fn color4(value: Option<&serde_json::Value>, name: &str) -> Result<[f64; 4]> {
    let value = value.ok_or_else(|| param_error("missing required field", name, &NULL))?;
    let array = value
        .as_array()
        .ok_or_else(|| param_error("must be a [r, g, b, a] array", name, value))?;
    if array.len() != 4 {
        return Err(param_error(
            "must have exactly four components",
            name,
            value,
        ));
    }
    let mut color = [0.0_f64; 4];
    for (slot, component) in color.iter_mut().zip(array.iter()) {
        *slot = unit(component, name)?;
    }
    Ok(color)
}

/// Coerce a JSON value to a finite `f64`, erroring on a non-number or `NaN`/∞.
fn finite(value: &serde_json::Value, name: &str) -> Result<f64> {
    let n = value
        .as_f64()
        .ok_or_else(|| param_error("must be a number", name, value))?;
    if n.is_finite() {
        Ok(n)
    } else {
        Err(param_error("must be finite", name, value))
    }
}

/// Coerce a JSON value to a finite `f64` in `[0, 1]`.
fn unit(value: &serde_json::Value, name: &str) -> Result<f64> {
    let n = finite(value, name)?;
    if (0.0..=1.0).contains(&n) {
        Ok(n)
    } else {
        Err(param_error("must be in [0, 1]", name, value))
    }
}

/// Build a [`schema`](ErrorClass::Schema) param error carrying the offending
/// value.
fn param_error(detail: &str, name: &str, value: &serde_json::Value) -> Error {
    Error::new(
        ErrorClass::Schema,
        E_SPLAT_PARAM,
        format!("paint.gaussian_splats parameter `{name}`: {detail}"),
    )
    .with_context(ErrorContext::default().with_actual(value.to_string()))
}

/// Build the degenerate-sigma error.
fn degenerate(name: &str, value: f64) -> Error {
    Error::new(
        ErrorClass::Schema,
        E_SPLAT_PARAM,
        format!("paint.gaussian_splats `{name}` must be a strictly positive sigma, got {value}"),
    )
    .with_context(
        ErrorContext::default()
            .with_actual(value.to_string())
            .with_expected("a finite sigma > 0"),
    )
}

/// The `paint.gaussian_splats@1` operation: a base `Image` + an inline splat
/// batch → the painted `Image`.
#[derive(Debug, Clone, Copy, Default)]
pub struct GaussianSplats;

impl GaussianSplats {
    /// Construct the Gaussian-splat operation.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    /// The declared manifest for `paint.gaussian_splats@1`.
    ///
    /// # Errors
    /// Propagates the [`schema`](ErrorClass::Schema) error if the hard-coded op /
    /// impl ids are somehow invalid (they are not).
    pub fn manifest() -> Result<OperationManifest> {
        Ok(OperationManifest {
            id: SPLAT_OP_ID.parse()?,
            impl_version: 1,
            summary: "Paint an inline batch of anisotropic Gaussian splats (center, sigma, angle, \
                      color, opacity, blend) onto a premultiplied linear base image, accumulated \
                      in array order; the batch is bounded by policy.resources.max_splats."
                .to_owned(),
            determinism: DeterminismTier::Bounded,
            roi: RoiPolicy {
                category: RoiCategory::Pointwise,
                halo_px: None,
            },
            inputs: vec![InputSpec {
                name: "base".to_owned(),
                kind: ResourceKind::Image,
                required: true,
                doc: "The premultiplied linear-light base image the splats paint onto.".to_owned(),
            }],
            outputs: vec![OutputSpec {
                name: "image".to_owned(),
                kind: ResourceKind::Image,
                doc: "The base image with the splat batch painted on (same extent/layout)."
                    .to_owned(),
            }],
            params: vec![
                ParamSpec {
                    name: "splats".to_owned(),
                    ty: ParamType::Json,
                    unit: None,
                    required: true,
                    default: None,
                    choices: vec![],
                    doc: "The inline batch of splat objects { center_px, sigma_px, angle_rad?, \
                          color, opacity?, blend? }; accumulated in array order. May be empty."
                        .to_owned(),
                },
                ParamSpec {
                    name: "space".to_owned(),
                    ty: ParamType::String,
                    unit: None,
                    required: false,
                    default: None,
                    choices: vec!["linear-srgb".to_owned(), "raw-linear".to_owned()],
                    doc: "Optional paint color space; when present must name the base's linear \
                          encoding."
                        .to_owned(),
                },
                ParamSpec {
                    name: "max_splats".to_owned(),
                    ty: ParamType::Integer,
                    unit: None,
                    required: false,
                    default: Some(serde_json::json!(DEFAULT_MAX_SPLATS)),
                    choices: vec![],
                    doc: "The inline-batch bound from policy.resources.max_splats; a larger batch \
                          is rejected with a policy error."
                        .to_owned(),
                },
            ],
            implementations: vec![reference_impl()?],
            test: splat_test_metadata(),
        })
    }
}

impl OpContract for GaussianSplats {
    fn declared_inputs(&self) -> Vec<(String, ResourceKind)> {
        vec![("base".to_owned(), ResourceKind::Image)]
    }

    fn declared_outputs(&self) -> Vec<(String, ResourceKind)> {
        vec![("image".to_owned(), ResourceKind::Image)]
    }

    fn infer_outputs(
        &self,
        inputs: &Descriptors,
        params: &serde_json::Value,
    ) -> Result<OutputDescriptors> {
        let descriptor = base_descriptor(inputs)?;
        // Validate the whole batch at infer time so a degenerate / oversized
        // request fails on the type-checking pass, before any pixels are touched.
        SplatRequest::resolve(params, &descriptor)?;

        let mut out = OutputDescriptors::new();
        // The painted image keeps the base's descriptor exactly: same extent,
        // layout, encoding, premultiplied alpha, coordinates.
        out.insert("image".to_owned(), ResourceDescriptor::Image(descriptor));
        Ok(out)
    }

    fn required_inputs(
        &self,
        requested_outputs: &OutputRegions,
        _inputs: &Descriptors,
        _params: &serde_json::Value,
    ) -> Result<InputRegions> {
        // Each output pixel composites the splats over the co-located base pixel:
        // a pointwise dependency on the base (the splat field itself is generated,
        // not read from an input).
        let mut regions = InputRegions::new();
        if let Some(region) = requested_outputs.get("image") {
            regions.insert("base".to_owned(), *region);
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
        // The painted layer stays premultiplied linear color: a future edit that
        // changed the output representation is caught here.
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

impl OpImplementation for GaussianSplats {
    fn compute(
        &self,
        inputs: &InputValues,
        params: &serde_json::Value,
    ) -> std::result::Result<OutputValues, Error> {
        let base = inputs.get("base").ok_or_else(|| {
            Error::new(
                ErrorClass::Reference,
                E_SPLAT_INPUT,
                "paint.gaussian_splats requires a `base` input value".to_owned(),
            )
        })?;
        let ResourceDescriptor::Image(descriptor) = base.descriptor() else {
            return Err(Error::new(
                ErrorClass::Type,
                E_SPLAT_INPUT,
                "paint.gaussian_splats `base` input must be an image resource".to_owned(),
            ));
        };

        let request = SplatRequest::resolve(params, descriptor)?;
        let extent = descriptor.extent;
        let samples = request.paint(base.samples(), extent);

        let value = ResourceValue::new(
            ResourceDescriptor::Image(*descriptor),
            base.channels(),
            samples,
        )
        .map_err(|actual| {
            Error::new(
                ErrorClass::Execution,
                E_SPLAT_BUFFER,
                format!(
                    "paint.gaussian_splats produced a sample buffer of unexpected length {actual}"
                ),
            )
        })?;

        let mut out = OutputValues::new();
        out.insert("image".to_owned(), value);
        Ok(out)
    }
}

/// The base image descriptor from the `base` input port.
fn base_descriptor(inputs: &Descriptors) -> Result<ImageDescriptor> {
    let base = inputs.get("base").ok_or_else(|| {
        Error::new(
            ErrorClass::Reference,
            E_SPLAT_INPUT,
            "paint.gaussian_splats requires a `base` input".to_owned(),
        )
    })?;
    let ResourceDescriptor::Image(descriptor) = base else {
        return Err(Error::new(
            ErrorClass::Type,
            E_SPLAT_INPUT,
            "paint.gaussian_splats `base` input must be an image resource".to_owned(),
        ));
    };
    Ok(*descriptor)
}

/// The mandatory `cpu.reference@1` oracle implementation id.
fn reference_impl() -> Result<ImplId> {
    ImplId::new("cpu", "reference", 1)
}

/// The verification declarations for `paint.gaussian_splats@1`: an analytic,
/// bounded, single-reference accumulator (`AGENT_VERIFICATION` §3.5). Differential
/// does not apply (one implementation). Perceptual is not applicable: the splat
/// field is a closed-form Gaussian accumulation verified by center symmetry,
/// covariance-axis alignment, translation covariance, zero-opacity identity, batch
/// order/blend semantics, and the inline-batch budget — there is no
/// perceptual-quality metric. Every other applicable category is covered by the
/// fixtures and property tests in this module.
fn splat_test_metadata() -> TestMetadata {
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
            "paint.gaussian_splats accumulates a closed-form Gaussian field verified by center \
             symmetry, covariance-axis alignment, translation covariance, zero-opacity identity, \
             batch-order/blend semantics, and the inline-batch budget; there is no \
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
