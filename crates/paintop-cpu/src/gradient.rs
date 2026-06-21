//! The `paint.linear_gradient@1` and `paint.radial_gradient@1` operations:
//! parameterized color gradients with explicit stop interpolation
//! (`OP_CATALOG` §6).
//!
//! Both ops synthesize a color `Image` sized from an `extent_from` input by
//! evaluating a 1-D gradient parameter `t ∈ [0, 1]` at every pixel center and
//! interpolating an explicit, sorted list of color **stops** at that `t`:
//!
//! - **linear**: `t = clamp( dot(P − a, b − a) / |b − a|², 0, 1 )` for endpoints
//!   `a = start_px`, `b = end_px`. `t = 0` at `a`, `t = 1` at `b`, and the value
//!   is constant along lines perpendicular to `b − a`. A degenerate axis
//!   (`a == b`) has no direction and is rejected.
//! - **radial**: `t = clamp( |P − c| / r, 0, 1 )` for center `c = center_px` and
//!   radius `r = radius_px`. `t = 0` at the center, `t = 1` at radius `r`. A
//!   non-positive (or non-finite) radius is degenerate and rejected.
//!
//! # Stops and interpolation space
//!
//! Stops are `{ "position": p, "color": [c0, c1, …] }` with `p ∈ [0, 1]` and one
//! color component per output channel. They must be sorted by non-decreasing
//! position with a stop at `0` and a stop at `1` (so the whole `[0, 1]` range is
//! covered without implicit extrapolation). At a `t` exactly equal to a stop
//! position the output is that stop's color **bit-exactly**; between two stops the
//! output is the per-channel linear interpolation
//! `c = c_lo + f·(c_hi − c_lo)`, `f = (t − p_lo)/(p_hi − p_lo)`, evaluated in the
//! output image's declared **color space** (the interpolation happens directly on
//! the stored samples, so the declared `color` encoding *is* the interpolation
//! space). Each stop color is range-checked against the image's valid-range
//! policy exactly as `image.create` does (clamping is never implicit).
//!
//! # Determinism
//!
//! The parameter `t` uses a division (and, for radial, a `hypot`/`sqrt`), so the
//! gradient is [`Bounded`](DeterminismTier::Bounded): coverage is asserted within
//! a tolerance rather than bit-exactly. The exception is the **stop-exactness**
//! guarantee at `t == position`, which returns the stop color verbatim.

use paintop_core::executor::{InputValues, OpImplementation, OutputValues, ResourceValue};
use paintop_ir::{
    AlphaRepresentation, AssertionResult, ChannelLayout, ColorEncoding, ColorRange,
    CoordinateConvention, Descriptors, DeterminismTier, Error, ErrorClass, ErrorContext, Extent,
    ImageDescriptor, ImplId, InputRegions, InputSpec, OpContract, OperationManifest,
    OutputDescriptors, OutputRegions, OutputSpec, ParamSpec, ParamType, ParamUnit,
    RequestedColorEncoding, ResourceDescriptor, ResourceKind, Result, RoiCategory, RoiPolicy,
    ScalarType, SemanticRole, TestMetadata,
};

/// The canonical id of the linear-gradient operation.
pub const LINEAR_GRADIENT_OP_ID: &str = "paint.linear_gradient@1";

/// The canonical id of the radial-gradient operation.
pub const RADIAL_GRADIENT_OP_ID: &str = "paint.radial_gradient@1";

/// The `extent_from` input was absent or carried no descriptor to size the image.
pub const E_GRADIENT_INPUT: &str = "E_GRADIENT_INPUT";

/// A geometry parameter (`start_px`/`end_px`, `center_px`/`radius_px`) was
/// missing, the wrong shape, non-finite, or describes a degenerate gradient.
pub const E_GRADIENT_PARAM: &str = "E_GRADIENT_PARAM";

/// The `stops` array was missing, malformed, unsorted, did not span `[0, 1]`, or a
/// stop color was the wrong length / out of the valid range.
pub const E_GRADIENT_STOPS: &str = "E_GRADIENT_STOPS";

/// The op produced a sample buffer whose length disagrees with its descriptor.
pub const E_GRADIENT_BUFFER: &str = "E_GRADIENT_BUFFER";

/// One color stop: a position in `[0, 1]` and a per-channel color.
#[derive(Debug, Clone, PartialEq)]
struct Stop {
    /// The gradient parameter (`f32`, the sample type) at which this stop applies.
    position: f32,
    /// The per-channel color of this stop (one component per output channel).
    color: Vec<f32>,
}

/// A parsed, validated stop list: sorted, spanning `[0, 1]`, with a fixed channel
/// count.
#[derive(Debug, Clone, PartialEq)]
struct Stops {
    stops: Vec<Stop>,
}

impl Stops {
    /// Parse and validate the `stops` array against the output image descriptor.
    ///
    /// # Errors
    /// [`schema`](ErrorClass::Schema) / [`E_GRADIENT_STOPS`] if the array is
    /// missing, empty, has a malformed entry, has a position outside `[0, 1]`, is
    /// not non-decreasing in position, does not begin at `0` and end at `1`, or a
    /// stop color is the wrong length or violates the image's range policy.
    fn resolve(params: &serde_json::Value, descriptor: &ImageDescriptor) -> Result<Self> {
        let value = params.get("stops").ok_or_else(|| {
            Error::new(
                ErrorClass::Schema,
                E_GRADIENT_STOPS,
                "a gradient requires a `stops` array".to_owned(),
            )
        })?;
        let array = value.as_array().ok_or_else(|| {
            stops_error(
                "`stops` must be an array of { position, color } entries",
                value,
            )
        })?;
        if array.len() < 2 {
            return Err(stops_error(
                "`stops` must have at least two entries (one at 0 and one at 1)",
                value,
            ));
        }

        let channels = descriptor.layout.channel_count() as usize;
        let mut stops = Vec::with_capacity(array.len());
        let mut previous_position = f32::NEG_INFINITY;
        for (index, entry) in array.iter().enumerate() {
            let object = entry.as_object().ok_or_else(|| {
                stops_error(&format!("`stops[{index}]` must be an object"), entry)
            })?;
            let raw = object
                .get("position")
                .and_then(serde_json::Value::as_f64)
                .ok_or_else(|| {
                    stops_error(
                        &format!("`stops[{index}].position` must be a number"),
                        entry,
                    )
                })?;
            if !(raw.is_finite() && (0.0..=1.0).contains(&raw)) {
                return Err(stops_error(
                    &format!("`stops[{index}].position` must be a finite value in [0, 1]"),
                    entry,
                ));
            }
            #[allow(
                clippy::cast_possible_truncation,
                reason = "position in [0, 1] stored as the f32 sample type"
            )]
            let position = raw as f32;
            if position < previous_position {
                return Err(stops_error(
                    &format!("`stops[{index}].position` must be >= the previous stop (sorted)"),
                    entry,
                ));
            }
            previous_position = position;

            let color = parse_stop_color(object.get("color"), index, channels, descriptor)?;
            stops.push(Stop { position, color });
        }

        // The stops must span the whole [0, 1] range so no implicit extrapolation
        // is needed at the parameter extremes. Compared by bit pattern (the
        // clippy-clean exact f32 equality).
        let spans = stops
            .first()
            .is_some_and(|s| s.position.to_bits() == 0.0_f32.to_bits())
            && stops
                .last()
                .is_some_and(|s| s.position.to_bits() == 1.0_f32.to_bits());
        if !spans {
            return Err(stops_error(
                "`stops` must begin at position 0 and end at position 1",
                value,
            ));
        }

        Ok(Self { stops })
    }

    /// Interpolate the per-channel color at gradient parameter `t ∈ [0, 1]`.
    ///
    /// At a `t` exactly equal to a stop position the stop color is returned
    /// verbatim (the stop-exactness guarantee); otherwise the result is the
    /// per-channel linear interpolation of the bracketing stops.
    #[must_use]
    fn color_at(&self, t: f32) -> Vec<f32> {
        // Find the first stop whose position is >= t. Positions are sorted.
        let upper = self
            .stops
            .iter()
            .position(|s| s.position >= t)
            .unwrap_or(self.stops.len() - 1);
        // Exact stop hit (incl. t <= first or t >= last): return verbatim.
        if upper == 0 || self.stops[upper].position.to_bits() == t.to_bits() {
            return self.stops[upper].color.clone();
        }
        let lo = &self.stops[upper - 1];
        let hi = &self.stops[upper];
        // p_hi > p_lo here (a zero-width interval would have been an exact hit on
        // the lower stop above), so the divide is well-defined.
        let f = (t - lo.position) / (hi.position - lo.position);
        lo.color
            .iter()
            .zip(hi.color.iter())
            .map(|(&c_lo, &c_hi)| f.mul_add(c_hi - c_lo, c_lo))
            .collect()
    }
}

/// Parse and range-check a single stop's per-channel color array.
fn parse_stop_color(
    value: Option<&serde_json::Value>,
    index: usize,
    channels: usize,
    descriptor: &ImageDescriptor,
) -> Result<Vec<f32>> {
    let value = value.ok_or_else(|| {
        stops_error(
            &format!("`stops[{index}].color` is required"),
            &serde_json::Value::Null,
        )
    })?;
    let array = value
        .as_array()
        .ok_or_else(|| stops_error(&format!("`stops[{index}].color` must be an array"), value))?;
    if array.len() != channels {
        return Err(stops_error(
            &format!(
                "`stops[{index}].color` has {} components but the layout has {channels} channels",
                array.len()
            ),
            value,
        ));
    }
    let has_alpha = descriptor.layout.has_alpha();
    let mut color = Vec::with_capacity(channels);
    for (channel, component) in array.iter().enumerate() {
        let n = component.as_f64().ok_or_else(|| {
            stops_error(
                &format!("`stops[{index}].color[{channel}]` must be a number"),
                component,
            )
        })?;
        if !n.is_finite() {
            return Err(stops_error(
                &format!("`stops[{index}].color[{channel}]` must be finite"),
                component,
            ));
        }
        let is_alpha = has_alpha && channel == channels - 1;
        let bounded = is_alpha || matches!(descriptor.range, ColorRange::DisplayReferred);
        if bounded && !(0.0..=1.0).contains(&n) {
            return Err(stops_error(
                &format!(
                    "`stops[{index}].color[{channel}]` = {n} is out of the [0, 1] range; clamping \
                     is never implicit"
                ),
                component,
            ));
        }
        #[allow(
            clippy::cast_possible_truncation,
            reason = "stop color stored as the image's f32 sample type"
        )]
        color.push(n as f32);
    }
    Ok(color)
}

/// Build a [`schema`](ErrorClass::Schema) stops error carrying the offending
/// value.
fn stops_error(detail: &str, value: &serde_json::Value) -> Error {
    Error::new(ErrorClass::Schema, E_GRADIENT_STOPS, detail.to_owned())
        .with_context(ErrorContext::default().with_actual(value.to_string()))
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

/// Build a [`schema`](ErrorClass::Schema) param error carrying the offending
/// value.
fn param_error(detail: &str, name: &str, value: &serde_json::Value) -> Error {
    Error::new(
        ErrorClass::Schema,
        E_GRADIENT_PARAM,
        format!("gradient parameter `{name}`: {detail}"),
    )
    .with_context(ErrorContext::default().with_actual(value.to_string()))
}

/// Parse the optional output descriptor params (`layout`, `color`, `range`,
/// `alpha`) into the image descriptor for an `extent`. Defaults match a
/// straight-alpha sRGB display-referred RGBA color image.
fn output_descriptor(params: &serde_json::Value, extent: Extent) -> Result<ImageDescriptor> {
    let layout = match params.get("layout") {
        None => ChannelLayout::Rgba,
        Some(value) => serde_json::from_value(value.clone())
            .map_err(|e| descriptor_error("layout", value, &e))?,
    };
    let color = match params.get("color") {
        None => ColorEncoding::Srgb,
        Some(value) => {
            let requested: RequestedColorEncoding = serde_json::from_value(value.clone())
                .map_err(|e| descriptor_error("color", value, &e))?;
            requested.resolve()?
        }
    };
    let range = match params.get("range") {
        None => ColorRange::DisplayReferred,
        Some(value) => serde_json::from_value(value.clone())
            .map_err(|e| descriptor_error("range", value, &e))?,
    };
    let alpha = match params.get("alpha") {
        None => AlphaRepresentation::Straight,
        Some(value) => serde_json::from_value(value.clone())
            .map_err(|e| descriptor_error("alpha", value, &e))?,
    };
    let descriptor = ImageDescriptor {
        extent,
        layout,
        scalar: ScalarType::F32,
        color,
        range,
        alpha,
        coordinates: CoordinateConvention::PixelCenterUpperLeft,
        semantic: SemanticRole::Color,
    };
    descriptor
        .extent
        .byte_count(layout.channel_count(), ScalarType::F32)?;
    Ok(descriptor)
}

/// Build a [`schema`](ErrorClass::Schema) descriptor-param error.
fn descriptor_error(name: &str, value: &serde_json::Value, source: &serde_json::Error) -> Error {
    Error::new(
        ErrorClass::Schema,
        E_GRADIENT_PARAM,
        format!("gradient `{name}` is not a recognized value: {source}"),
    )
    .with_context(ErrorContext::default().with_actual(value.to_string()))
}

/// The pixel extent of the `extent_from` input descriptor.
fn extent_of(inputs: &Descriptors) -> Result<Extent> {
    let source = inputs.get("extent_from").ok_or_else(|| {
        Error::new(
            ErrorClass::Reference,
            E_GRADIENT_INPUT,
            "a gradient requires an `extent_from` input".to_owned(),
        )
    })?;
    Ok(source.extent())
}

/// The shared output-descriptor parameter specs (layout/color/range/alpha) plus a
/// `stops` array, used by both gradient manifests.
fn shared_descriptor_params() -> Vec<ParamSpec> {
    let string_param =
        |name: &str, default: serde_json::Value, choices: Vec<String>, doc: &str| ParamSpec {
            name: name.to_owned(),
            ty: ParamType::String,
            unit: None,
            required: false,
            default: Some(default),
            choices,
            doc: doc.to_owned(),
        };
    vec![
        string_param(
            "layout",
            serde_json::json!("rgba"),
            vec![
                "gray".to_owned(),
                "gray-a".to_owned(),
                "rgb".to_owned(),
                "rgba".to_owned(),
            ],
            "The channel layout of the produced image; stop colors carry one component per channel.",
        ),
        string_param(
            "color",
            serde_json::json!("srgb"),
            vec![
                "srgb".to_owned(),
                "linear-srgb".to_owned(),
                "raw-linear".to_owned(),
            ],
            "The color encoding of the produced image; interpolation happens in this space.",
        ),
        string_param(
            "range",
            serde_json::json!("display-referred"),
            vec!["display-referred".to_owned(), "scene-referred".to_owned()],
            "The reference-light range policy; display-referred bounds color stops to [0, 1].",
        ),
        string_param(
            "alpha",
            serde_json::json!("straight"),
            vec!["premultiplied".to_owned(), "straight".to_owned()],
            "The alpha representation of the produced image.",
        ),
        ParamSpec {
            name: "stops".to_owned(),
            ty: ParamType::Json,
            unit: None,
            required: true,
            default: None,
            choices: vec![],
            doc: "The sorted color stops [{ position: p in [0,1], color: [per-channel] }, …], \
                  spanning position 0 to 1."
                .to_owned(),
        },
    ]
}

/// The shared `extent_from` input spec.
fn extent_input() -> InputSpec {
    InputSpec {
        name: "extent_from".to_owned(),
        kind: ResourceKind::Image,
        required: true,
        doc: "The resource whose pixel extent the produced gradient image matches.".to_owned(),
    }
}

/// The shared `image` output spec.
fn image_output() -> OutputSpec {
    OutputSpec {
        name: "image".to_owned(),
        kind: ResourceKind::Image,
        doc: "The produced gradient image.".to_owned(),
    }
}

/// The shared `required_inputs`: the generator reads no input samples, only the
/// `extent_from` size, so it demands an empty region.
fn extent_only_regions(inputs: &Descriptors) -> InputRegions {
    let mut regions = InputRegions::new();
    if inputs.contains_key("extent_from") {
        regions.insert("extent_from".to_owned(), paintop_ir::Rect::new(0, 0, 0, 0));
    }
    regions
}

/// The shared postcondition: an `image` output is produced.
fn image_postconditions(outputs: &OutputDescriptors) -> Vec<AssertionResult> {
    match outputs.get("image") {
        Some(ResourceDescriptor::Image(_)) => vec![AssertionResult::pass("produces_image")],
        _ => vec![AssertionResult::fail(
            "produces_image",
            "no `image` output produced",
        )],
    }
}

/// The mandatory `cpu.reference@1` oracle implementation id.
fn reference_impl() -> Result<ImplId> {
    ImplId::new("cpu", "reference", 1)
}

/// The shared verification declarations for both gradient ops: bounded,
/// single-reference, analytic generators. Differential does not apply. Perceptual
/// is not applicable: a gradient is a closed-form interpolation verified by
/// stop-exactness, monotonicity, covariance, and degenerate-rejection tests, not a
/// perceptual metric.
fn gradient_test_metadata(perceptual_reason: &str) -> TestMetadata {
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
        CategoryStatus::not_applicable(perceptual_reason.to_owned()),
    );
    TestMetadata {
        has_analytic_reference: true,
        has_property_tests: true,
        golden_fixtures: vec![],
        not_applicable_reason: String::new(),
        verification: decls,
    }
}

/// Rasterize a gradient into a row-major interleaved `f32` buffer by evaluating
/// `parameter(x, y)` and interpolating `stops` at each pixel center.
fn rasterize(
    extent: Extent,
    channels: usize,
    stops: &Stops,
    parameter: impl Fn(f64, f64) -> f64,
) -> Vec<f32> {
    let width = extent.width as usize;
    let height = extent.height as usize;
    let mut samples = Vec::with_capacity(width.saturating_mul(height).saturating_mul(channels));
    for j in 0..height {
        #[allow(clippy::cast_precision_loss, reason = "pixel index well within f64")]
        let y = j as f64 + 0.5;
        for i in 0..width {
            #[allow(clippy::cast_precision_loss, reason = "pixel index well within f64")]
            let x = i as f64 + 0.5;
            let t = parameter(x, y).clamp(0.0, 1.0);
            #[allow(
                clippy::cast_possible_truncation,
                reason = "t is a bounded [0, 1] f64 stored as f32"
            )]
            let color = stops.color_at(t as f32);
            samples.extend_from_slice(&color);
        }
    }
    samples
}

// ---------------------------------------------------------------------------
// Linear gradient.
// ---------------------------------------------------------------------------

/// The resolved geometry of a linear gradient: two endpoints in pixel
/// coordinates, with a precomputed axis and its squared length.
#[derive(Debug, Clone, Copy, PartialEq)]
struct LinearGeometry {
    ax: f64,
    ay: f64,
    dx: f64,
    dy: f64,
    len_sq: f64,
}

impl LinearGeometry {
    /// Parse and validate the linear endpoints, rejecting a degenerate (zero
    /// length) axis.
    fn resolve(params: &serde_json::Value) -> Result<Self> {
        let (ax, ay) = pair_param(params, "start_px")?;
        let (bx, by) = pair_param(params, "end_px")?;
        let dx = bx - ax;
        let dy = by - ay;
        let len_sq = dx.mul_add(dx, dy * dy);
        if !(len_sq.is_finite() && len_sq > 0.0) {
            return Err(Error::new(
                ErrorClass::Schema,
                E_GRADIENT_PARAM,
                "paint.linear_gradient: `start_px` and `end_px` must differ (a zero-length axis \
                 has no gradient direction)"
                    .to_owned(),
            )
            .with_context(
                ErrorContext::default()
                    .with_actual(format!("start ({ax}, {ay}) end ({bx}, {by})"))
                    .with_expected("two distinct endpoints"),
            ));
        }
        Ok(Self {
            ax,
            ay,
            dx,
            dy,
            len_sq,
        })
    }

    /// The gradient parameter (pre-clamp) at sample `(x, y)`: the projection of
    /// `P − a` onto the axis, normalized by its squared length.
    fn parameter(&self, x: f64, y: f64) -> f64 {
        let px = x - self.ax;
        let py = y - self.ay;
        px.mul_add(self.dx, py * self.dy) / self.len_sq
    }
}

/// The `paint.linear_gradient@1` operation.
#[derive(Debug, Clone, Copy, Default)]
pub struct LinearGradient;

impl LinearGradient {
    /// Construct the linear-gradient operation.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    /// The declared manifest for `paint.linear_gradient@1`.
    ///
    /// # Errors
    /// Propagates the [`schema`](ErrorClass::Schema) error if the hard-coded op /
    /// impl ids are somehow invalid (they are not).
    pub fn manifest() -> Result<OperationManifest> {
        let mut params = vec![
            ParamSpec {
                name: "start_px".to_owned(),
                ty: ParamType::Json,
                unit: Some(ParamUnit::Pixels),
                required: true,
                default: None,
                choices: vec![],
                doc: "The gradient start endpoint [x, y] in pixel coordinates (t = 0).".to_owned(),
            },
            ParamSpec {
                name: "end_px".to_owned(),
                ty: ParamType::Json,
                unit: Some(ParamUnit::Pixels),
                required: true,
                default: None,
                choices: vec![],
                doc: "The gradient end endpoint [x, y] in pixel coordinates (t = 1).".to_owned(),
            },
        ];
        params.extend(shared_descriptor_params());
        Ok(OperationManifest {
            id: LINEAR_GRADIENT_OP_ID.parse()?,
            impl_version: 1,
            summary: "Synthesize a linear color gradient between two endpoints with explicit \
                      sorted color stops; t = clamp(dot(P − start, end − start) / |end − start|², \
                      0, 1) interpolated per channel in the declared color space."
                .to_owned(),
            determinism: DeterminismTier::Bounded,
            roi: RoiPolicy {
                category: RoiCategory::Pointwise,
                halo_px: None,
            },
            inputs: vec![extent_input()],
            outputs: vec![image_output()],
            params,
            implementations: vec![reference_impl()?],
            test: gradient_test_metadata(
                "paint.linear_gradient is a closed-form per-channel interpolation verified by \
                 stop-exactness, monotonic parameterization, and translation covariance; there is \
                 no perceptual-quality metric to apply",
            ),
        })
    }
}

impl OpContract for LinearGradient {
    fn declared_inputs(&self) -> Vec<(String, ResourceKind)> {
        vec![("extent_from".to_owned(), ResourceKind::Image)]
    }

    fn declared_outputs(&self) -> Vec<(String, ResourceKind)> {
        vec![("image".to_owned(), ResourceKind::Image)]
    }

    fn infer_outputs(
        &self,
        inputs: &Descriptors,
        params: &serde_json::Value,
    ) -> Result<OutputDescriptors> {
        let extent = extent_of(inputs)?;
        let descriptor = output_descriptor(params, extent)?;
        // Validate geometry and stops at infer time so a bad request fails the
        // type-checking pass before any pixels are touched.
        LinearGeometry::resolve(params)?;
        Stops::resolve(params, &descriptor)?;

        let mut out = OutputDescriptors::new();
        out.insert("image".to_owned(), ResourceDescriptor::Image(descriptor));
        Ok(out)
    }

    fn required_inputs(
        &self,
        _requested_outputs: &OutputRegions,
        inputs: &Descriptors,
        _params: &serde_json::Value,
    ) -> Result<InputRegions> {
        Ok(extent_only_regions(inputs))
    }

    fn validate_postconditions(
        &self,
        outputs: &OutputDescriptors,
        _params: &serde_json::Value,
    ) -> Result<Vec<AssertionResult>> {
        Ok(image_postconditions(outputs))
    }
}

impl OpImplementation for LinearGradient {
    fn compute(
        &self,
        inputs: &InputValues,
        params: &serde_json::Value,
    ) -> std::result::Result<OutputValues, Error> {
        let source = inputs.get("extent_from").ok_or_else(|| {
            Error::new(
                ErrorClass::Reference,
                E_GRADIENT_INPUT,
                "paint.linear_gradient requires an `extent_from` input value".to_owned(),
            )
        })?;
        let extent = source.extent();
        let descriptor = output_descriptor(params, extent)?;
        let geometry = LinearGeometry::resolve(params)?;
        let stops = Stops::resolve(params, &descriptor)?;
        let channels = descriptor.layout.channel_count();

        let samples = rasterize(extent, channels as usize, &stops, |x, y| {
            geometry.parameter(x, y)
        });
        produce_image(descriptor, channels, samples)
    }
}

// ---------------------------------------------------------------------------
// Radial gradient.
// ---------------------------------------------------------------------------

/// The resolved geometry of a radial gradient: center and a strictly-positive
/// radius in pixel coordinates.
#[derive(Debug, Clone, Copy, PartialEq)]
struct RadialGeometry {
    cx: f64,
    cy: f64,
    radius: f64,
}

impl RadialGeometry {
    /// Parse and validate the center and radius, rejecting a non-positive radius.
    fn resolve(params: &serde_json::Value) -> Result<Self> {
        let (cx, cy) = pair_param(params, "center_px")?;
        let radius = finite_number(
            params.get("radius_px").ok_or_else(|| {
                param_error(
                    "missing required parameter",
                    "radius_px",
                    &serde_json::Value::Null,
                )
            })?,
            "radius_px",
        )?;
        if radius <= 0.0 {
            return Err(Error::new(
                ErrorClass::Schema,
                E_GRADIENT_PARAM,
                format!(
                    "paint.radial_gradient: `radius_px` must be strictly positive, got {radius} \
                     (a zero radius is a degenerate gradient)"
                ),
            )
            .with_context(
                ErrorContext::default()
                    .with_actual(radius.to_string())
                    .with_expected("a finite radius > 0"),
            ));
        }
        Ok(Self { cx, cy, radius })
    }

    /// The gradient parameter (pre-clamp) at sample `(x, y)`: the distance from the
    /// center normalized by the radius.
    fn parameter(&self, x: f64, y: f64) -> f64 {
        (x - self.cx).hypot(y - self.cy) / self.radius
    }
}

/// The `paint.radial_gradient@1` operation.
#[derive(Debug, Clone, Copy, Default)]
pub struct RadialGradient;

impl RadialGradient {
    /// Construct the radial-gradient operation.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    /// The declared manifest for `paint.radial_gradient@1`.
    ///
    /// # Errors
    /// Propagates the [`schema`](ErrorClass::Schema) error if the hard-coded op /
    /// impl ids are somehow invalid (they are not).
    pub fn manifest() -> Result<OperationManifest> {
        let mut params = vec![
            ParamSpec {
                name: "center_px".to_owned(),
                ty: ParamType::Json,
                unit: Some(ParamUnit::Pixels),
                required: true,
                default: None,
                choices: vec![],
                doc: "The gradient center [x, y] in pixel coordinates (t = 0).".to_owned(),
            },
            ParamSpec {
                name: "radius_px".to_owned(),
                ty: ParamType::Float,
                unit: Some(ParamUnit::Pixels),
                required: true,
                default: None,
                choices: vec![],
                doc: "The gradient radius in pixels (t = 1 at this distance); strictly positive."
                    .to_owned(),
            },
        ];
        params.extend(shared_descriptor_params());
        Ok(OperationManifest {
            id: RADIAL_GRADIENT_OP_ID.parse()?,
            impl_version: 1,
            summary:
                "Synthesize a radial color gradient about a center with explicit sorted color \
                      stops; t = clamp(|P − center| / radius, 0, 1) interpolated per channel in \
                      the declared color space. A non-positive radius is rejected."
                    .to_owned(),
            determinism: DeterminismTier::Bounded,
            roi: RoiPolicy {
                category: RoiCategory::Pointwise,
                halo_px: None,
            },
            inputs: vec![extent_input()],
            outputs: vec![image_output()],
            params,
            implementations: vec![reference_impl()?],
            test: gradient_test_metadata(
                "paint.radial_gradient is a closed-form per-channel interpolation verified by \
                 stop-exactness, monotonic radial parameterization, and degenerate-radius \
                 rejection; there is no perceptual-quality metric to apply",
            ),
        })
    }
}

impl OpContract for RadialGradient {
    fn declared_inputs(&self) -> Vec<(String, ResourceKind)> {
        vec![("extent_from".to_owned(), ResourceKind::Image)]
    }

    fn declared_outputs(&self) -> Vec<(String, ResourceKind)> {
        vec![("image".to_owned(), ResourceKind::Image)]
    }

    fn infer_outputs(
        &self,
        inputs: &Descriptors,
        params: &serde_json::Value,
    ) -> Result<OutputDescriptors> {
        let extent = extent_of(inputs)?;
        let descriptor = output_descriptor(params, extent)?;
        RadialGeometry::resolve(params)?;
        Stops::resolve(params, &descriptor)?;

        let mut out = OutputDescriptors::new();
        out.insert("image".to_owned(), ResourceDescriptor::Image(descriptor));
        Ok(out)
    }

    fn required_inputs(
        &self,
        _requested_outputs: &OutputRegions,
        inputs: &Descriptors,
        _params: &serde_json::Value,
    ) -> Result<InputRegions> {
        Ok(extent_only_regions(inputs))
    }

    fn validate_postconditions(
        &self,
        outputs: &OutputDescriptors,
        _params: &serde_json::Value,
    ) -> Result<Vec<AssertionResult>> {
        Ok(image_postconditions(outputs))
    }
}

impl OpImplementation for RadialGradient {
    fn compute(
        &self,
        inputs: &InputValues,
        params: &serde_json::Value,
    ) -> std::result::Result<OutputValues, Error> {
        let source = inputs.get("extent_from").ok_or_else(|| {
            Error::new(
                ErrorClass::Reference,
                E_GRADIENT_INPUT,
                "paint.radial_gradient requires an `extent_from` input value".to_owned(),
            )
        })?;
        let extent = source.extent();
        let descriptor = output_descriptor(params, extent)?;
        let geometry = RadialGeometry::resolve(params)?;
        let stops = Stops::resolve(params, &descriptor)?;
        let channels = descriptor.layout.channel_count();

        let samples = rasterize(extent, channels as usize, &stops, |x, y| {
            geometry.parameter(x, y)
        });
        produce_image(descriptor, channels, samples)
    }
}

/// Wrap a rasterized sample buffer into the `image` output value.
fn produce_image(
    descriptor: ImageDescriptor,
    channels: u32,
    samples: Vec<f32>,
) -> std::result::Result<OutputValues, Error> {
    let value = ResourceValue::new(ResourceDescriptor::Image(descriptor), channels, samples)
        .map_err(|actual| {
            Error::new(
                ErrorClass::Execution,
                E_GRADIENT_BUFFER,
                format!("a gradient produced a sample buffer of unexpected length {actual}"),
            )
        })?;
    let mut out = OutputValues::new();
    out.insert("image".to_owned(), value);
    Ok(out)
}

#[cfg(test)]
mod tests;
