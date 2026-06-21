//! The `mask.ellipse@1` operation: an analytic rotated-ellipse coverage mask.
//!
//! See `OP_CATALOG` §3, `AGENT_VERIFICATION` §3.6, `M0_DECISIONS` D1, and
//! `IR_SPEC` §20.
//!
//! `mask.ellipse` rasterizes a rotated ellipse into a coverage mask in `[0, 1]`.
//! The mask's extent is copied from the `extent_from` input (D3: references live
//! under `in`, so the resource whose size the mask matches is an input port, not
//! a param). The geometry is the center, the two semi-axis radii, and a rotation
//! `angle_rad`; the edge is feathered **analytically** from the ellipse's
//! implicit quadratic — there is no signed-distance-field infrastructure (D1: the
//! first vertical slice feathers by a physical pixel radius directly on this op,
//! and exact-EDT + the SDF calculus land in the next slice).
//!
//! # Coverage model
//!
//! Each pixel is sampled at its center: under
//! [`CoordinateConvention::PixelCenterUpperLeft`](paintop_ir::CoordinateConvention)
//! pixel `(i, j)` has center `(i + 0.5, j + 0.5)`. The sample is rotated into the
//! ellipse's local frame and evaluated against the implicit form
//!
//! ```text
//! Q(u, v) = (u / rx)^2 + (v / ry)^2,   f = sqrt(Q)
//! ```
//!
//! where `f` is the *normalized radius* (`f = 1` exactly on the boundary). A
//! first-order **analytic signed distance** to the `f = 1` contour, in physical
//! pixels, is
//!
//! ```text
//! sd = (f - 1) / |grad f|,
//! |grad f| = sqrt( (u / rx^2)^2 + (v / ry^2)^2 ) / f
//! ```
//!
//! (`sd < 0` inside, `> 0` outside). This is the standard implicit-surface
//! distance estimate; it needs only the quadratic and its gradient — no distance
//! field. Coverage is then a `smoothstep` feather of half-width `h`
//! (`edge.half_width_px`) about the boundary:
//!
//! ```text
//! t        = clamp((sd + h) / (2h), 0, 1)
//! coverage = 1 - (3 t^2 - 2 t^3)
//! ```
//!
//! so coverage is `1` for `sd <= -h` (fully inside), `0` for `sd >= +h` (fully
//! outside), and falls monotonically through `0.5` at the boundary. The soft
//! transition therefore spans exactly `2h` physical pixels along the boundary
//! normal — "feather by a physical pixel radius" of `h`. With `h = 0` the edge is
//! a hard, half-open coverage (`sd <= 0` is inside): a pixel center exactly on the
//! boundary is covered, matching the half-open pixel convention.
//!
//! # Determinism
//!
//! The op is `bounded`: the feather and the normalized radius use `sqrt`, whose
//! last bit is not guaranteed identical across platforms, so coverage is asserted
//! within a tolerance rather than bit-exactly. The geometry — center, radii,
//! angle — is otherwise an exact function of the params and extent.
//!
//! # Rejected requests
//!
//! - A non-positive (or non-finite) radius is a degenerate ellipse with no
//!   interior and is rejected as [`schema`](ErrorClass::Schema).
//! - A non-finite center, angle, or feather half-width is rejected likewise: the
//!   coverage must never contain `NaN`.

use paintop_core::executor::{InputValues, OpImplementation, OutputValues, ResourceValue};
use paintop_ir::{
    AssertionResult, CoordinateConvention, Descriptors, DeterminismTier, Error, ErrorClass,
    ErrorContext, Extent, ImplId, InputRegions, InputSpec, MaskDescriptor, MaskMeaning, OpContract,
    OperationManifest, OutputDescriptors, OutputRegions, OutputSpec, ParamSpec, ParamType,
    ParamUnit, Rect, ResourceDescriptor, ResourceKind, Result, RoiCategory, RoiPolicy, ScalarType,
    TestMetadata, ValidRange,
};

/// The canonical id of the ellipse-mask operation.
pub const ELLIPSE_OP_ID: &str = "mask.ellipse@1";

/// The `extent_from` input was absent or carried no descriptor to size the mask.
pub const E_ELLIPSE_INPUT: &str = "E_ELLIPSE_INPUT";

/// A geometry parameter (`center_px`, `radii_px`, `angle_rad`, `edge`) was
/// missing, the wrong shape, or held a non-finite / degenerate value.
pub const E_ELLIPSE_PARAM: &str = "E_ELLIPSE_PARAM";

/// The only antialias mode this op supports: an analytic coverage evaluated from
/// the implicit quadratic (`M0_DECISIONS` D1).
const ANTIALIAS_ANALYTIC: &str = "analytic";

/// The only edge profile this op supports: a `smoothstep` feather.
const EDGE_SMOOTHSTEP: &str = "smoothstep";

/// The resolved geometry of an ellipse-mask request: center, semi-axis radii,
/// rotation, and the feather half-width — all in physical pixels (the angle in
/// radians).
#[derive(Debug, Clone, Copy, PartialEq)]
struct EllipseGeometry {
    /// Center x in pixel coordinates.
    cx: f64,
    /// Center y in pixel coordinates.
    cy: f64,
    /// Semi-axis radius along the (pre-rotation) x axis, strictly positive.
    rx: f64,
    /// Semi-axis radius along the (pre-rotation) y axis, strictly positive.
    ry: f64,
    /// Rotation of the ellipse's local frame, radians (counter-clockwise).
    angle_rad: f64,
    /// The smoothstep feather half-width in physical pixels; `0` is a hard edge.
    half_width_px: f64,
}

impl EllipseGeometry {
    /// Parse and validate the ellipse geometry from the resolved params.
    ///
    /// # Errors
    /// [`schema`](ErrorClass::Schema) / [`E_ELLIPSE_PARAM`] if a required param is
    /// missing, the wrong shape, non-finite, or describes a degenerate ellipse
    /// (a non-positive radius), or if an unsupported `antialias` / `edge.profile`
    /// is requested.
    fn resolve(params: &serde_json::Value) -> Result<Self> {
        let (cx, cy) = pair_param(params, "center_px")?;
        let (rx, ry) = pair_param(params, "radii_px")?;

        // A radius must be a strictly-positive, finite extent: a zero or negative
        // semi-axis is a degenerate ellipse with no interior, rejected rather
        // than rasterized to an empty/NaN mask.
        for (name, r) in [("radii_px.x", rx), ("radii_px.y", ry)] {
            if !(r.is_finite() && r > 0.0) {
                return Err(degenerate(name, r));
            }
        }

        let angle_rad = optional_finite(params, "angle_rad", 0.0)?;

        // `antialias` is optional and defaults to the only supported mode; any
        // other token is rejected rather than silently treated as analytic.
        if let Some(value) = params.get("antialias") {
            let token = value
                .as_str()
                .ok_or_else(|| param_error("`antialias` must be a string", "antialias", value))?;
            if token != ANTIALIAS_ANALYTIC {
                return Err(Error::new(
                    ErrorClass::Schema,
                    E_ELLIPSE_PARAM,
                    format!("mask.ellipse only supports `antialias: analytic`, got `{token}`"),
                ));
            }
        }

        let half_width_px = parse_edge(params)?;

        Ok(Self {
            cx,
            cy,
            rx,
            ry,
            angle_rad,
            half_width_px,
        })
    }

    /// The analytic coverage in `[0, 1]` at a sample point `(x, y)` in pixel
    /// coordinates.
    ///
    /// Evaluates the implicit quadratic in the ellipse's local frame, derives the
    /// first-order signed distance to the boundary, and feathers it with a
    /// `smoothstep` of half-width `half_width_px`.
    #[must_use]
    fn coverage(&self, sample_x: f64, sample_y: f64) -> f64 {
        // Translate to the ellipse center, then rotate by -angle into the local
        // axis-aligned frame.
        let off_x = sample_x - self.cx;
        let off_y = sample_y - self.cy;
        let (sin, cos) = self.angle_rad.sin_cos();
        let local_u = sin.mul_add(off_y, cos * off_x);
        let local_v = cos.mul_add(off_y, -sin * off_x);

        // Normalized-radius implicit form: f = 1 on the boundary, < 1 inside.
        let norm_x = local_u / self.rx;
        let norm_y = local_v / self.ry;
        let quadratic = norm_x.mul_add(norm_x, norm_y * norm_y);
        let normalized_radius = quadratic.sqrt();

        // First-order analytic signed distance to the f = 1 contour, in pixels:
        // sd = (f - 1) / |grad f|. At the exact center f = 0 and the gradient is
        // undefined, but the center is unambiguously deep inside -> full coverage.
        if normalized_radius == 0.0 {
            return 1.0;
        }
        // |grad f| = sqrt((u/rx^2)^2 + (v/ry^2)^2) / f, so
        // (f - 1)/|grad f| = (f - 1) * f / sqrt(...).
        let grad_x = local_u / (self.rx * self.rx);
        let grad_y = local_v / (self.ry * self.ry);
        let grad = grad_x.hypot(grad_y);
        if grad == 0.0 {
            // Unreachable for f != 0, but guard against a 0/0: treat as inside.
            return 1.0;
        }
        let signed_distance = (normalized_radius - 1.0) * normalized_radius / grad;

        feather(signed_distance, self.half_width_px)
    }
}

/// The `smoothstep` feather: coverage `1` for `sd <= -h`, `0` for `sd >= +h`, a
/// smooth monotone fall through `0.5` at the boundary. `h <= 0` is a hard,
/// half-open edge (`sd <= 0` is covered).
#[must_use]
fn feather(sd: f64, half_width_px: f64) -> f64 {
    if half_width_px <= 0.0 {
        // Hard, half-open: a sample exactly on the boundary (sd == 0) is inside.
        return if sd <= 0.0 { 1.0 } else { 0.0 };
    }
    // Map sd in [-h, +h] to t in [0, 1]; outside the band clamps to a fixed end.
    let t = ((sd + half_width_px) / (2.0 * half_width_px)).clamp(0.0, 1.0);
    // 1 - smoothstep(t): coverage is 1 inside (t = 0) and 0 outside (t = 1).
    let smoothstep = t * t * 2.0f64.mul_add(-t, 3.0);
    1.0 - smoothstep
}

/// Extract a required `[x, y]` numeric-pair parameter.
fn pair_param(params: &serde_json::Value, name: &str) -> Result<(f64, f64)> {
    let value = params
        .get(name)
        .ok_or_else(|| param_error("missing required parameter", name, &serde_json::Value::Null))?;
    let array = value
        .as_array()
        .ok_or_else(|| param_error("must be a [x, y] array", name, value))?;
    if array.len() != 2 {
        return Err(param_error("must have exactly two elements", name, value));
    }
    let x = finite_number(&array[0], name)?;
    let y = finite_number(&array[1], name)?;
    Ok((x, y))
}

/// Read an optional finite float parameter, defaulting when absent.
fn optional_finite(params: &serde_json::Value, name: &str, default: f64) -> Result<f64> {
    params
        .get(name)
        .map_or(Ok(default), |value| finite_number(value, name))
}

/// Coerce a JSON value to a finite `f64`, erroring on a non-number or a
/// `NaN`/infinity.
fn finite_number(value: &serde_json::Value, name: &str) -> Result<f64> {
    let n = value
        .as_f64()
        .ok_or_else(|| param_error("must be a number", name, value))?;
    if n.is_finite() {
        Ok(n)
    } else {
        Err(param_error("must be finite", name, value))
    }
}

/// Parse the optional `edge` object, returning the feather half-width in pixels.
///
/// An absent `edge` is a hard edge (`half_width_px = 0`). A present `edge` must
/// be `{ "profile": "smoothstep", "half_width_px": N }` with a finite,
/// non-negative `N`.
fn parse_edge(params: &serde_json::Value) -> Result<f64> {
    let Some(edge) = params.get("edge") else {
        return Ok(0.0);
    };
    let object = edge
        .as_object()
        .ok_or_else(|| param_error("`edge` must be an object", "edge", edge))?;

    let profile = object
        .get("profile")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| param_error("`edge.profile` must be a string", "edge.profile", edge))?;
    if profile != EDGE_SMOOTHSTEP {
        return Err(Error::new(
            ErrorClass::Schema,
            E_ELLIPSE_PARAM,
            format!("mask.ellipse only supports `edge.profile: smoothstep`, got `{profile}`"),
        ));
    }

    let half = object.get("half_width_px").ok_or_else(|| {
        param_error(
            "`edge.half_width_px` is required",
            "edge.half_width_px",
            edge,
        )
    })?;
    let half_width_px = finite_number(half, "edge.half_width_px")?;
    if half_width_px < 0.0 {
        return Err(param_error(
            "must be non-negative",
            "edge.half_width_px",
            half,
        ));
    }
    Ok(half_width_px)
}

/// Build a [`schema`](ErrorClass::Schema) param error carrying the offending
/// value.
fn param_error(detail: &str, name: &str, value: &serde_json::Value) -> Error {
    Error::new(
        ErrorClass::Schema,
        E_ELLIPSE_PARAM,
        format!("mask.ellipse parameter `{name}`: {detail}"),
    )
    .with_context(ErrorContext::default().with_actual(value.to_string()))
}

/// Build the degenerate-radius error.
fn degenerate(name: &str, value: f64) -> Error {
    Error::new(
        ErrorClass::Schema,
        E_ELLIPSE_PARAM,
        format!("mask.ellipse `{name}` must be a strictly positive radius, got {value}"),
    )
    .with_context(
        ErrorContext::default()
            .with_actual(value.to_string())
            .with_expected("a finite radius > 0"),
    )
}

/// The mask descriptor produced for an extent: a coverage mask in `[0, 1]`,
/// `f32`, sharing the input's coordinate convention.
const fn mask_descriptor(extent: Extent) -> MaskDescriptor {
    MaskDescriptor {
        extent,
        scalar: ScalarType::F32,
        range: ValidRange::Bounded { min: 0.0, max: 1.0 },
        meaning: MaskMeaning::Coverage,
        coordinates: CoordinateConvention::PixelCenterUpperLeft,
    }
}

/// The `mask.ellipse@1` operation: an `extent_from` resource → a coverage
/// `Mask`.
#[derive(Debug, Clone, Copy, Default)]
pub struct EllipseMask;

impl EllipseMask {
    /// Construct the ellipse-mask operation.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    /// The declared manifest for `mask.ellipse@1`.
    ///
    /// # Errors
    /// Propagates the [`schema`](ErrorClass::Schema) error if the hard-coded op /
    /// impl ids are somehow invalid (they are not).
    pub fn manifest() -> Result<OperationManifest> {
        Ok(OperationManifest {
            id: ELLIPSE_OP_ID.parse()?,
            impl_version: 1,
            summary: "Rasterize a rotated ellipse into a coverage Mask with an analytic \
                      soft-edge feather (smoothstep, half_width_px) evaluated from the implicit \
                      quadratic."
                .to_owned(),
            determinism: DeterminismTier::Bounded,
            roi: RoiPolicy {
                category: RoiCategory::Pointwise,
                halo_px: None,
            },
            inputs: vec![InputSpec {
                name: "extent_from".to_owned(),
                kind: ResourceKind::Image,
                required: true,
                doc: "The resource whose pixel extent the produced mask matches.".to_owned(),
            }],
            outputs: vec![OutputSpec {
                name: "mask".to_owned(),
                kind: ResourceKind::Mask,
                doc: "The ellipse coverage mask in [0, 1].".to_owned(),
            }],
            params: vec![
                ParamSpec {
                    name: "center_px".to_owned(),
                    ty: ParamType::Json,
                    unit: Some(ParamUnit::Pixels),
                    required: true,
                    default: None,
                    choices: vec![],
                    doc: "The ellipse center [x, y] in pixel coordinates.".to_owned(),
                },
                ParamSpec {
                    name: "radii_px".to_owned(),
                    ty: ParamType::Json,
                    unit: Some(ParamUnit::Pixels),
                    required: true,
                    default: None,
                    choices: vec![],
                    doc: "The semi-axis radii [rx, ry] in pixels; both strictly positive."
                        .to_owned(),
                },
                ParamSpec {
                    name: "angle_rad".to_owned(),
                    ty: ParamType::Float,
                    unit: Some(ParamUnit::Radians),
                    required: false,
                    default: Some(serde_json::json!(0.0)),
                    choices: vec![],
                    doc: "Counter-clockwise rotation of the ellipse, in radians.".to_owned(),
                },
                ParamSpec {
                    name: "antialias".to_owned(),
                    ty: ParamType::String,
                    unit: None,
                    required: false,
                    default: Some(serde_json::json!(ANTIALIAS_ANALYTIC)),
                    choices: vec![ANTIALIAS_ANALYTIC.to_owned()],
                    doc: "The antialias mode; only analytic coverage is supported.".to_owned(),
                },
                ParamSpec {
                    name: "edge".to_owned(),
                    ty: ParamType::Json,
                    unit: None,
                    required: false,
                    default: None,
                    choices: vec![],
                    doc: "Optional soft edge { profile: smoothstep, half_width_px: N } feathering \
                          the boundary by N physical pixels; absent is a hard edge."
                        .to_owned(),
                },
            ],
            implementations: vec![reference_impl()?],
            test: ellipse_test_metadata(),
        })
    }
}

impl OpContract for EllipseMask {
    fn declared_inputs(&self) -> Vec<(String, ResourceKind)> {
        vec![("extent_from".to_owned(), ResourceKind::Image)]
    }

    fn declared_outputs(&self) -> Vec<(String, ResourceKind)> {
        vec![("mask".to_owned(), ResourceKind::Mask)]
    }

    fn infer_outputs(
        &self,
        inputs: &Descriptors,
        params: &serde_json::Value,
    ) -> Result<OutputDescriptors> {
        let extent = extent_of(inputs)?;
        // Validate the geometry at infer time so a degenerate request fails on the
        // type-checking pass, before any pixels are touched.
        EllipseGeometry::resolve(params)?;

        let mut out = OutputDescriptors::new();
        out.insert(
            "mask".to_owned(),
            ResourceDescriptor::Mask(mask_descriptor(extent)),
        );
        Ok(out)
    }

    fn required_inputs(
        &self,
        _requested_outputs: &OutputRegions,
        inputs: &Descriptors,
        _params: &serde_json::Value,
    ) -> Result<InputRegions> {
        // The mask reads no input *samples* — only the `extent_from` size — so no
        // input pixels are demanded. An empty region for the port records that
        // honestly (the generator depends on geometry, not on input content).
        let mut regions = InputRegions::new();
        if inputs.contains_key("extent_from") {
            regions.insert("extent_from".to_owned(), Rect::new(0, 0, 0, 0));
        }
        Ok(regions)
    }

    fn validate_postconditions(
        &self,
        outputs: &OutputDescriptors,
        _params: &serde_json::Value,
    ) -> Result<Vec<AssertionResult>> {
        let Some(ResourceDescriptor::Mask(mask)) = outputs.get("mask") else {
            return Ok(vec![AssertionResult::fail(
                "produces_mask",
                "no `mask` output produced",
            )]);
        };
        let mut results = vec![AssertionResult::pass("produces_mask")];
        // Coverage is, by construction, a bounded [0, 1] mask: record that the
        // declared range agrees so a future edit that changed it is caught.
        let unit_range = ValidRange::Bounded { min: 0.0, max: 1.0 };
        results.push(if mask.range == unit_range {
            AssertionResult::pass("coverage_in_unit_range")
        } else {
            AssertionResult::fail(
                "coverage_in_unit_range",
                format!(
                    "mask range {:?} is not the coverage range [0, 1]",
                    mask.range
                ),
            )
        });
        Ok(results)
    }
}

impl OpImplementation for EllipseMask {
    fn compute(
        &self,
        inputs: &InputValues,
        params: &serde_json::Value,
    ) -> std::result::Result<OutputValues, Error> {
        let source = inputs.get("extent_from").ok_or_else(|| {
            Error::new(
                ErrorClass::Reference,
                E_ELLIPSE_INPUT,
                "mask.ellipse requires an `extent_from` input value".to_owned(),
            )
        })?;
        let extent = source.extent();
        let ellipse = EllipseGeometry::resolve(params)?;

        let samples = rasterize(&ellipse, extent);
        let value = ResourceValue::new(
            ResourceDescriptor::Mask(mask_descriptor(extent)),
            1,
            samples,
        )
        .map_err(|actual| {
            Error::new(
                ErrorClass::Execution,
                E_ELLIPSE_INPUT,
                format!("mask.ellipse produced a sample buffer of unexpected length {actual}"),
            )
        })?;

        let mut out = OutputValues::new();
        out.insert("mask".to_owned(), value);
        Ok(out)
    }
}

/// Rasterize the ellipse's coverage into a row-major, single-channel `f32`
/// buffer of `extent.width * extent.height` samples, one per pixel center.
#[must_use]
fn rasterize(ellipse: &EllipseGeometry, extent: Extent) -> Vec<f32> {
    let width = extent.width as usize;
    let height = extent.height as usize;
    let mut samples = Vec::with_capacity(width.saturating_mul(height));
    for j in 0..height {
        // Pixel-center convention: pixel (i, j) is sampled at (i + 0.5, j + 0.5).
        #[allow(clippy::cast_precision_loss, reason = "pixel index well within f64")]
        let y = j as f64 + 0.5;
        for i in 0..width {
            #[allow(clippy::cast_precision_loss, reason = "pixel index well within f64")]
            let x = i as f64 + 0.5;
            let coverage = ellipse.coverage(x, y);
            #[allow(
                clippy::cast_possible_truncation,
                reason = "coverage is a bounded [0, 1] f64 stored as the mask's f32"
            )]
            samples.push(coverage as f32);
        }
    }
    samples
}

/// The pixel extent of the `extent_from` input descriptor.
fn extent_of(inputs: &Descriptors) -> Result<Extent> {
    let source = inputs.get("extent_from").ok_or_else(|| {
        Error::new(
            ErrorClass::Reference,
            E_ELLIPSE_INPUT,
            "mask.ellipse requires an `extent_from` input".to_owned(),
        )
    })?;
    Ok(source.extent())
}

/// The mandatory `cpu.reference@1` oracle implementation id.
fn reference_impl() -> Result<ImplId> {
    ImplId::new("cpu", "reference", 1)
}

/// The verification declarations for `mask.ellipse@1`: an analytic, bounded,
/// single-reference rasterizer. Differential does not apply (one implementation).
/// Perceptual is not applicable: coverage is a closed-form geometric quantity
/// checked against analytic area, rotation/translation covariance, and a measured
/// feather width — there is no perceptual-quality metric. Every other applicable
/// category is covered by the analytic-area, covariance, feather-width,
/// half-open-convention and degenerate-rejection tests in this module.
fn ellipse_test_metadata() -> TestMetadata {
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
            "mask.ellipse produces analytic geometric coverage verified by analytic area \
             convergence, rotation/translation covariance, and a measured feather width; there \
             is no perceptual-quality metric to apply",
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
