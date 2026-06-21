//! The `image.create@1` operation: a descriptor + fill value → a synthesized
//! `Image` (`OP_CATALOG` §1, `plan.md` §8.3).
//!
//! `image.create` is the in-graph image source: it manufactures a constant image
//! from an explicit descriptor (extent, channel layout, color/range/alpha/semantic
//! metadata) and a per-channel `fill`. Unlike `io.decode_image` — whose extent and
//! format come from a file header only known at execution — every property of a
//! created image is fixed by its params, so the concrete output descriptor is
//! inferred at type-check time and the op is fully deterministic.
//!
//! # Fill and the valid-range policy
//!
//! The `fill` is a per-channel array whose length must equal the layout's channel
//! count. Each fill component is written verbatim to every pixel of that channel.
//! The fill must respect the image's declared range policy (`plan.md` §8.3:
//! clamping is never implicit): a [`display-referred`](ColorRange::DisplayReferred)
//! image bounds its color channels to `[0, 1]`, so an out-of-range fill is
//! **rejected** rather than silently clamped; a
//! [`scene-referred`](ColorRange::SceneReferred) image only requires the fill be
//! finite (no `NaN`/`±∞`). The alpha channel, when present, is always coverage in
//! `[0, 1]` regardless of the color range.
//!
//! # Determinism
//!
//! The op is [`Exact`](DeterminismTier::Exact): the output is a closed-form
//! function of the params (a constant raster), bit-identical on every run and
//! machine.

use paintop_core::executor::{InputValues, OpImplementation, OutputValues, ResourceValue};
use paintop_ir::{
    AlphaRepresentation, AssertionResult, ChannelLayout, ColorEncoding, ColorRange,
    CoordinateConvention, Descriptors, DeterminismTier, Error, ErrorClass, ErrorContext, Extent,
    ImageDescriptor, ImplId, InputRegions, OpContract, OperationManifest, OutputDescriptors,
    OutputRegions, OutputSpec, ParamSpec, ParamType, ParamUnit, RequestedColorEncoding,
    ResourceDescriptor, ResourceKind, Result, RoiCategory, RoiPolicy, ScalarType, SemanticRole,
    TestMetadata,
};

/// The canonical id of the image-create operation.
pub const CREATE_OP_ID: &str = "image.create@1";

/// A descriptor parameter (`width`, `height`, `layout`, `color`, …) was missing
/// or malformed.
pub const E_CREATE_PARAM: &str = "E_CREATE_PARAM";

/// The `fill` array was the wrong length, non-finite, or violated the image's
/// declared valid-range policy.
pub const E_CREATE_FILL: &str = "E_CREATE_FILL";

/// The resolved creation request: the output image descriptor plus the validated
/// per-channel fill.
#[derive(Debug, Clone)]
struct CreateRequest {
    descriptor: ImageDescriptor,
    fill: Vec<f32>,
}

impl CreateRequest {
    /// Parse and validate every creation param, returning the settled descriptor
    /// and per-channel fill.
    fn resolve(params: &serde_json::Value) -> Result<Self> {
        let width = u32_param(params, "width")?;
        let height = u32_param(params, "height")?;
        let layout = layout_param(params)?;
        let color = color_param(params)?;
        let range = range_param(params)?;
        let alpha = alpha_param(params)?;
        let semantic = semantic_param(params)?;

        let descriptor = ImageDescriptor {
            extent: Extent::new(width, height),
            layout,
            scalar: ScalarType::F32,
            color,
            range,
            alpha,
            coordinates: CoordinateConvention::PixelCenterUpperLeft,
            semantic,
        };
        // Reject an extent whose pixel/byte product overflows before allocating.
        descriptor
            .extent
            .byte_count(layout.channel_count(), ScalarType::F32)?;

        let fill = fill_param(params, layout, range)?;
        Ok(Self { descriptor, fill })
    }

    /// The interleaved sample buffer realizing this request: the fill tiled over
    /// every pixel in row-major, channel-interleaved order.
    fn samples(&self) -> Vec<f32> {
        let pixels = (self.descriptor.extent.width as usize)
            .saturating_mul(self.descriptor.extent.height as usize);
        let mut samples = Vec::with_capacity(pixels.saturating_mul(self.fill.len()));
        for _ in 0..pixels {
            samples.extend_from_slice(&self.fill);
        }
        samples
    }
}

/// Parse a required non-negative `u32` integer param.
fn u32_param(params: &serde_json::Value, name: &str) -> Result<u32> {
    let value = params.get(name).ok_or_else(|| {
        Error::new(
            ErrorClass::Schema,
            E_CREATE_PARAM,
            format!("image.create requires an integer `{name}` parameter"),
        )
    })?;
    let n = value.as_u64().ok_or_else(|| {
        Error::new(
            ErrorClass::Schema,
            E_CREATE_PARAM,
            format!("image.create `{name}` must be a non-negative integer"),
        )
        .with_context(ErrorContext::default().with_actual(value.to_string()))
    })?;
    u32::try_from(n).map_err(|_| {
        Error::new(
            ErrorClass::Schema,
            E_CREATE_PARAM,
            format!("image.create `{name}` value {n} does not fit in u32"),
        )
    })
}

/// Parse the required `layout` param into a [`ChannelLayout`].
fn layout_param(params: &serde_json::Value) -> Result<ChannelLayout> {
    let value = params.get("layout").ok_or_else(|| {
        Error::new(
            ErrorClass::Schema,
            E_CREATE_PARAM,
            "image.create requires a `layout` parameter".to_owned(),
        )
    })?;
    serde_json::from_value(value.clone()).map_err(|e| {
        Error::new(
            ErrorClass::Schema,
            E_CREATE_PARAM,
            format!("image.create `layout` is not a known channel layout: {e}"),
        )
        .with_context(
            ErrorContext::default()
                .with_actual(value.to_string())
                .with_expected("gray | gray-a | rgb | rgba"),
        )
    })
}

/// Parse the optional `color` param through [`RequestedColorEncoding`] so a
/// nameable-but-unsupported encoding is rejected, defaulting to
/// [`ColorEncoding::Srgb`].
fn color_param(params: &serde_json::Value) -> Result<ColorEncoding> {
    params
        .get("color")
        .map_or(Ok(ColorEncoding::Srgb), |value| {
            let requested: RequestedColorEncoding =
                serde_json::from_value(value.clone()).map_err(|e| {
                    Error::new(
                        ErrorClass::Schema,
                        E_CREATE_PARAM,
                        format!("image.create `color` is not a known color encoding: {e}"),
                    )
                    .with_context(ErrorContext::default().with_actual(value.to_string()))
                })?;
            requested.resolve()
        })
}

/// Parse an optional `range` policy param, defaulting to display-referred.
fn range_param(params: &serde_json::Value) -> Result<ColorRange> {
    params
        .get("range")
        .map_or(Ok(ColorRange::DisplayReferred), |value| {
            serde_json::from_value(value.clone()).map_err(|e| enum_param_error("range", value, &e))
        })
}

/// Parse an optional `alpha` representation param, defaulting to straight.
fn alpha_param(params: &serde_json::Value) -> Result<AlphaRepresentation> {
    params
        .get("alpha")
        .map_or(Ok(AlphaRepresentation::Straight), |value| {
            serde_json::from_value(value.clone()).map_err(|e| enum_param_error("alpha", value, &e))
        })
}

/// Parse an optional `semantic` role param, defaulting to color.
fn semantic_param(params: &serde_json::Value) -> Result<SemanticRole> {
    params
        .get("semantic")
        .map_or(Ok(SemanticRole::Color), |value| {
            serde_json::from_value(value.clone())
                .map_err(|e| enum_param_error("semantic", value, &e))
        })
}

/// Build the schema error for an unrecognized enum param token.
fn enum_param_error(name: &str, value: &serde_json::Value, source: &serde_json::Error) -> Error {
    Error::new(
        ErrorClass::Schema,
        E_CREATE_PARAM,
        format!("image.create `{name}` is not a recognized value: {source}"),
    )
    .with_context(ErrorContext::default().with_actual(value.to_string()))
}

/// Parse and range-check the per-channel `fill` array against the layout and
/// range policy.
fn fill_param(
    params: &serde_json::Value,
    layout: ChannelLayout,
    range: ColorRange,
) -> Result<Vec<f32>> {
    let value = params.get("fill").ok_or_else(|| {
        Error::new(
            ErrorClass::Schema,
            E_CREATE_FILL,
            "image.create requires a per-channel `fill` array".to_owned(),
        )
    })?;
    let array = value.as_array().ok_or_else(|| {
        Error::new(
            ErrorClass::Schema,
            E_CREATE_FILL,
            "image.create `fill` must be an array of per-channel values".to_owned(),
        )
        .with_context(ErrorContext::default().with_actual(value.to_string()))
    })?;
    let channels = layout.channel_count() as usize;
    if array.len() != channels {
        return Err(Error::new(
            ErrorClass::Schema,
            E_CREATE_FILL,
            format!(
                "image.create `fill` has {} components but the {layout:?} layout has {channels} channels",
                array.len()
            ),
        )
        .with_context(
            ErrorContext::default()
                .with_actual(array.len().to_string())
                .with_expected(channels.to_string()),
        ));
    }

    let has_alpha = layout.has_alpha();
    let mut fill = Vec::with_capacity(channels);
    for (index, component) in array.iter().enumerate() {
        let n = component.as_f64().ok_or_else(|| {
            Error::new(
                ErrorClass::Schema,
                E_CREATE_FILL,
                format!("image.create `fill[{index}]` must be a number"),
            )
            .with_context(ErrorContext::default().with_actual(component.to_string()))
        })?;
        if !n.is_finite() {
            return Err(Error::new(
                ErrorClass::Schema,
                E_CREATE_FILL,
                format!("image.create `fill[{index}]` must be finite, got {n}"),
            ));
        }
        // The alpha channel (the last channel of a *-A layout) is coverage in
        // [0, 1] regardless of the color range; color channels follow the range
        // policy. A display-referred color channel is bounded to [0, 1]; a
        // scene-referred one only needs to be finite (already checked).
        let is_alpha = has_alpha && index == channels - 1;
        let bounded = is_alpha || matches!(range, ColorRange::DisplayReferred);
        if bounded && !(0.0..=1.0).contains(&n) {
            let what = if is_alpha {
                "alpha coverage"
            } else {
                "display-referred color"
            };
            return Err(Error::new(
                ErrorClass::Policy,
                E_CREATE_FILL,
                format!(
                    "image.create `fill[{index}]` = {n} is out of the {what} range [0, 1]; \
                     clamping is never implicit"
                ),
            )
            .with_context(
                ErrorContext::default()
                    .with_actual(n.to_string())
                    .with_expected("[0, 1]"),
            ));
        }
        #[allow(
            clippy::cast_possible_truncation,
            reason = "fill stored as the image's f32 sample type"
        )]
        fill.push(n as f32);
    }
    Ok(fill)
}

/// The `image.create@1` operation: a descriptor + fill → a synthesized `Image`.
#[derive(Debug, Clone, Copy, Default)]
pub struct CreateImage;

impl CreateImage {
    /// Construct the image-create operation.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    /// The declared manifest for `image.create@1`.
    ///
    /// # Errors
    /// Propagates the [`schema`](ErrorClass::Schema) error if the hard-coded op /
    /// impl ids are somehow invalid (they are not).
    pub fn manifest() -> Result<OperationManifest> {
        Ok(OperationManifest {
            id: CREATE_OP_ID.parse()?,
            impl_version: 1,
            summary: "Synthesize a constant Image from an explicit descriptor (extent, layout, \
                      color/range/alpha/semantic) and a per-channel fill, range-checked against \
                      the declared valid-range policy."
                .to_owned(),
            determinism: DeterminismTier::Exact,
            roi: RoiPolicy {
                category: RoiCategory::Pointwise,
                halo_px: None,
            },
            inputs: vec![],
            outputs: vec![OutputSpec {
                name: "image".to_owned(),
                kind: ResourceKind::Image,
                doc: "The synthesized constant image.".to_owned(),
            }],
            params: create_params(),
            implementations: vec![reference_impl()?],
            test: create_test_metadata(),
        })
    }
}

impl OpContract for CreateImage {
    fn declared_inputs(&self) -> Vec<(String, ResourceKind)> {
        vec![]
    }

    fn declared_outputs(&self) -> Vec<(String, ResourceKind)> {
        vec![("image".to_owned(), ResourceKind::Image)]
    }

    fn infer_outputs(
        &self,
        _inputs: &Descriptors,
        params: &serde_json::Value,
    ) -> Result<OutputDescriptors> {
        // Every property is fixed by the params, so the concrete descriptor is
        // known at type-check time (validating the fill in the process).
        let request = CreateRequest::resolve(params)?;
        let mut out = OutputDescriptors::new();
        out.insert(
            "image".to_owned(),
            ResourceDescriptor::Image(request.descriptor),
        );
        Ok(out)
    }

    fn required_inputs(
        &self,
        _requested_outputs: &OutputRegions,
        _inputs: &Descriptors,
        _params: &serde_json::Value,
    ) -> Result<InputRegions> {
        // A source op has no input ports, hence no required input regions.
        Ok(InputRegions::new())
    }

    fn validate_postconditions(
        &self,
        outputs: &OutputDescriptors,
        params: &serde_json::Value,
    ) -> Result<Vec<AssertionResult>> {
        let Some(ResourceDescriptor::Image(image)) = outputs.get("image") else {
            return Ok(vec![AssertionResult::fail(
                "produces_image",
                "no `image` output produced",
            )]);
        };
        let request = CreateRequest::resolve(params)?;
        Ok(vec![if *image == request.descriptor {
            AssertionResult::pass("matches_requested_descriptor")
        } else {
            AssertionResult::fail(
                "matches_requested_descriptor",
                "created image descriptor does not match the request",
            )
        }])
    }
}

impl OpImplementation for CreateImage {
    fn compute(
        &self,
        _inputs: &InputValues,
        params: &serde_json::Value,
    ) -> std::result::Result<OutputValues, Error> {
        let request = CreateRequest::resolve(params)?;
        let channels = request.descriptor.layout.channel_count();
        let samples = request.samples();
        let value = ResourceValue::new(
            ResourceDescriptor::Image(request.descriptor),
            channels,
            samples,
        )
        .map_err(|actual| {
            Error::new(
                ErrorClass::Execution,
                E_CREATE_FILL,
                format!("image.create produced an image buffer of unexpected length {actual}"),
            )
        })?;

        let mut out = OutputValues::new();
        out.insert("image".to_owned(), value);
        Ok(out)
    }
}

/// The declared parameter list for `image.create@1`.
fn create_params() -> Vec<ParamSpec> {
    let extent_param = |name: &str, doc: &str| ParamSpec {
        name: name.to_owned(),
        ty: ParamType::Integer,
        unit: Some(ParamUnit::Pixels),
        required: true,
        default: None,
        choices: vec![],
        doc: doc.to_owned(),
    };
    let string_param = |name: &str,
                        required: bool,
                        default: Option<serde_json::Value>,
                        choices: Vec<String>,
                        doc: &str| ParamSpec {
        name: name.to_owned(),
        ty: ParamType::String,
        unit: None,
        required,
        default,
        choices,
        doc: doc.to_owned(),
    };
    vec![
        extent_param("width", "The image width in pixels."),
        extent_param("height", "The image height in pixels."),
        string_param(
            "layout",
            true,
            None,
            vec![
                "gray".to_owned(),
                "gray-a".to_owned(),
                "rgb".to_owned(),
                "rgba".to_owned(),
            ],
            "The channel layout of the created image.",
        ),
        ParamSpec {
            name: "fill".to_owned(),
            ty: ParamType::Json,
            unit: None,
            required: true,
            default: None,
            choices: vec![],
            doc: "The per-channel constant fill value; one component per layout channel, each \
                  respecting the declared valid range."
                .to_owned(),
        },
        string_param(
            "color",
            false,
            Some(serde_json::json!("srgb")),
            vec![
                "srgb".to_owned(),
                "linear-srgb".to_owned(),
                "raw-linear".to_owned(),
            ],
            "The color transfer encoding of the created image.",
        ),
        string_param(
            "range",
            false,
            Some(serde_json::json!("display-referred")),
            vec!["display-referred".to_owned(), "scene-referred".to_owned()],
            "The reference-light range policy; display-referred bounds color channels to [0, 1].",
        ),
        string_param(
            "alpha",
            false,
            Some(serde_json::json!("straight")),
            vec!["premultiplied".to_owned(), "straight".to_owned()],
            "The alpha representation of the created image.",
        ),
        string_param(
            "semantic",
            false,
            Some(serde_json::json!("color")),
            semantic_choices(),
            "The semantic role of the created image.",
        ),
    ]
}

/// The kebab-case `semantic` role choices, matching [`SemanticRole`]'s wire
/// tokens.
fn semantic_choices() -> Vec<String> {
    [
        "color",
        "material",
        "normal",
        "depth",
        "confidence",
        "distance",
        "flow",
    ]
    .iter()
    .map(|s| (*s).to_owned())
    .collect()
}

/// The mandatory `cpu.reference@1` oracle implementation id.
fn reference_impl() -> Result<ImplId> {
    ImplId::new("cpu", "reference", 1)
}

/// Verification declarations for `image.create@1`: an exact, single-reference,
/// constant source. Differential and perceptual do not apply; every other
/// applicable category is covered by this module's fixture, property, and
/// rejection tests.
fn create_test_metadata() -> TestMetadata {
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
            "image.create synthesizes a constant raster whose every sample equals the requested \
             fill; correctness is verified by exact descriptor/fill fixtures and range-policy \
             rejection, not a perceptual metric",
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
