//! The `image.extract_channel@1` / `image.assemble_channels@1` operations
//! (`OP_CATALOG` §1, `plan.md` §7.1 `Field1`, §19 M1).
//!
//! These two ops are the bridge between a packed, channel-interleaved `Image` and
//! the typed scalar-field substrate (`Field1`) that convolution, distance fields,
//! and other per-scalar math operate on. They are mutual inverses on a chosen
//! channel set: extracting every channel of an image and re-assembling them with
//! the original layout reproduces the image exactly.
//!
//! # `image.extract_channel@1`
//!
//! Pulls one channel (selected by a 0-based `channel` index) out of an `Image`
//! into a single-channel `Field1`. The op is **exact** and **pointwise**: output
//! sample `(x, y)` is input sample `(x, y, channel)`, copied verbatim (`NaN`
//! payloads and all). The produced field is `raw-linear` material data: a bare
//! scalar with no color transfer function, which is exactly what a scalar-field
//! consumer wants. Its `semantic` role and valid `range` are explicit params so a
//! plan can name what the extracted scalar *means* (a confidence in `[0, 1]`, a
//! material roughness, an unbounded depth) rather than guessing.
//!
//! A `channel` index outside the image's channel count is a typed
//! [`semantic`](ErrorClass::Semantic) error — the op never silently clamps to a
//! neighbouring channel.
//!
//! # `image.assemble_channels@1`
//!
//! Packs up to four `Field1` inputs (`ch0`..`ch3`) into an interleaved `Image`
//! with a requested channel `layout`. The op is **exact** and **pointwise**:
//! output sample `(x, y, k)` is `ch{k}`'s sample `(x, y)`. The number of wired
//! channel ports must equal the layout's channel count, and every wired field
//! must share one extent; either mismatch is rejected with a typed error so a
//! malformed assembly can never produce a ragged or mis-sized raster. The output
//! image's color/range/alpha/semantic metadata is taken from explicit params (so
//! the assembled image is honestly typed), defaulting to a `raw-linear`,
//! display-referred material image.

use paintop_core::executor::{InputValues, OpImplementation, OutputValues, ResourceValue};
use paintop_ir::{
    AlphaRepresentation, AssertionResult, ChannelLayout, ColorEncoding, ColorRange,
    CoordinateConvention, Descriptors, DeterminismTier, Error, ErrorClass, ErrorContext, Extent,
    FieldArity, FieldDescriptor, ImageDescriptor, ImplId, InputRegions, InputSpec, OpContract,
    OperationManifest, OutputDescriptors, OutputRegions, OutputSpec, ParamSpec, ParamType,
    RequestedColorEncoding, ResourceDescriptor, ResourceKind, Result, RoiCategory, RoiPolicy,
    ScalarType, SemanticRole, TestMetadata, ValidRange,
};

/// The canonical id of the channel-extract operation.
pub const EXTRACT_OP_ID: &str = "image.extract_channel@1";
/// The canonical id of the channel-assemble operation.
pub const ASSEMBLE_OP_ID: &str = "image.assemble_channels@1";

/// The `image` input to extract from was absent or carried a non-image descriptor.
pub const E_EXTRACT_INPUT: &str = "E_EXTRACT_INPUT";
/// The `channel` index was missing, malformed, or out of the image's channel range.
pub const E_EXTRACT_CHANNEL: &str = "E_EXTRACT_CHANNEL";
/// A channel input to assemble was absent or carried a non-`Field1` descriptor,
/// the wired-port count disagreed with the layout, or the fields disagreed on
/// extent.
pub const E_ASSEMBLE_INPUT: &str = "E_ASSEMBLE_INPUT";
/// A required `image.assemble_channels` parameter was missing or malformed.
pub const E_ASSEMBLE_PARAM: &str = "E_ASSEMBLE_PARAM";

/// The fixed channel input-port names for `image.assemble_channels`, in packing
/// order. A layout of `n` channels wires exactly `ch0..ch{n-1}`.
const ASSEMBLE_PORTS: [&str; 4] = ["ch0", "ch1", "ch2", "ch3"];

// ---------------------------------------------------------------------------
// image.extract_channel@1
// ---------------------------------------------------------------------------

/// Parse the required, non-negative integer `channel` index.
fn channel_index(params: &serde_json::Value) -> Result<u32> {
    let value = params.get("channel").ok_or_else(|| {
        Error::new(
            ErrorClass::Schema,
            E_EXTRACT_CHANNEL,
            "image.extract_channel requires an integer `channel` index".to_owned(),
        )
    })?;
    let index = value.as_u64().ok_or_else(|| {
        Error::new(
            ErrorClass::Schema,
            E_EXTRACT_CHANNEL,
            "image.extract_channel `channel` must be a non-negative integer".to_owned(),
        )
        .with_context(ErrorContext::default().with_actual(value.to_string()))
    })?;
    u32::try_from(index).map_err(|_| {
        Error::new(
            ErrorClass::Schema,
            E_EXTRACT_CHANNEL,
            format!("image.extract_channel `channel` index {index} does not fit in u32"),
        )
    })
}

/// Parse the optional `semantic` role for the extracted field, defaulting to
/// [`SemanticRole::Material`].
fn extracted_semantic(params: &serde_json::Value) -> Result<SemanticRole> {
    parse_semantic(params, SemanticRole::Material, E_EXTRACT_CHANNEL)
}

/// Parse the optional `range` policy for the extracted field, defaulting to
/// [`ValidRange::Unbounded`].
fn extracted_range(params: &serde_json::Value) -> Result<ValidRange> {
    params
        .get("range")
        .map_or(Ok(ValidRange::Unbounded), |value| {
            serde_json::from_value(value.clone()).map_err(|e| {
                Error::new(
                    ErrorClass::Schema,
                    E_EXTRACT_CHANNEL,
                    format!("image.extract_channel `range` is not a valid range policy: {e}"),
                )
                .with_context(ErrorContext::default().with_actual(value.to_string()))
            })
        })
}

/// The `Field1` descriptor an extraction produces for `extent`.
const fn extracted_field_descriptor(extent: Extent, semantic: SemanticRole) -> FieldDescriptor {
    FieldDescriptor {
        arity: FieldArity::Field1,
        extent,
        scalar: ScalarType::F32,
        semantic,
        coordinates: CoordinateConvention::PixelCenterUpperLeft,
        space: None,
        normalization: None,
        encoding: None,
    }
}

/// Resolve and validate the extraction against the input image descriptor,
/// returning the produced `Field1` descriptor.
fn extract_infer(
    descriptor: &ImageDescriptor,
    params: &serde_json::Value,
) -> Result<FieldDescriptor> {
    let channel = channel_index(params)?;
    let channels = descriptor.layout.channel_count();
    if channel >= channels {
        return Err(Error::new(
            ErrorClass::Semantic,
            E_EXTRACT_CHANNEL,
            format!(
                "image.extract_channel `channel` {channel} is out of range for a {channels}-channel image"
            ),
        )
        .with_context(
            ErrorContext::default()
                .with_actual(channel.to_string())
                .with_expected(format!("0..{channels}")),
        ));
    }
    // Validate the optional descriptive params up front so an infer-time pass
    // catches a malformed range/semantic before any pixels are touched.
    let semantic = extracted_semantic(params)?;
    let _range = extracted_range(params)?;
    Ok(extracted_field_descriptor(descriptor.extent, semantic))
}

/// The `image.extract_channel@1` operation: an `Image` → a single channel as a
/// `Field1`.
#[derive(Debug, Clone, Copy, Default)]
pub struct ExtractChannel;

impl ExtractChannel {
    /// Construct the extract-channel operation.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    /// The declared manifest for `image.extract_channel@1`.
    ///
    /// # Errors
    /// Propagates the [`schema`](ErrorClass::Schema) error if the hard-coded op /
    /// impl ids are somehow invalid (they are not).
    pub fn manifest() -> Result<OperationManifest> {
        Ok(OperationManifest {
            id: EXTRACT_OP_ID.parse()?,
            impl_version: 1,
            summary: "Extract one channel (by 0-based index) of an Image into a single-channel \
                      Field1 scalar field, copied verbatim, with explicit semantic role and \
                      valid range."
                .to_owned(),
            determinism: DeterminismTier::Exact,
            roi: RoiPolicy {
                category: RoiCategory::Pointwise,
                halo_px: None,
            },
            inputs: vec![InputSpec {
                name: "image".to_owned(),
                kind: ResourceKind::Image,
                required: true,
                doc: "The image to extract a channel from.".to_owned(),
            }],
            outputs: vec![OutputSpec {
                name: "field".to_owned(),
                kind: ResourceKind::Field1,
                doc: "The extracted channel as a single-channel scalar field.".to_owned(),
            }],
            params: vec![
                ParamSpec {
                    name: "channel".to_owned(),
                    ty: ParamType::Integer,
                    unit: None,
                    required: true,
                    default: None,
                    choices: vec![],
                    doc: "The 0-based channel index to extract.".to_owned(),
                },
                ParamSpec {
                    name: "semantic".to_owned(),
                    ty: ParamType::String,
                    unit: None,
                    required: false,
                    default: Some(serde_json::json!("material")),
                    choices: semantic_choices(),
                    doc: "The semantic role of the extracted scalar field.".to_owned(),
                },
                ParamSpec {
                    name: "range".to_owned(),
                    ty: ParamType::Json,
                    unit: None,
                    required: false,
                    default: None,
                    choices: vec![],
                    doc: "The valid-range policy of the extracted field; defaults to unbounded."
                        .to_owned(),
                },
            ],
            implementations: vec![reference_impl()?],
            test: exact_pointwise_test_metadata(
                "image.extract_channel copies one channel verbatim into a Field1; correctness is \
                 verified by an exact channel-selection fixture and a round-trip with \
                 image.assemble_channels, not a perceptual metric",
            ),
        })
    }
}

impl OpContract for ExtractChannel {
    fn declared_inputs(&self) -> Vec<(String, ResourceKind)> {
        vec![("image".to_owned(), ResourceKind::Image)]
    }

    fn declared_outputs(&self) -> Vec<(String, ResourceKind)> {
        vec![("field".to_owned(), ResourceKind::Field1)]
    }

    fn infer_outputs(
        &self,
        inputs: &Descriptors,
        params: &serde_json::Value,
    ) -> Result<OutputDescriptors> {
        let descriptor = extract_image_descriptor(inputs)?;
        let field = extract_infer(&descriptor, params)?;
        let mut out = OutputDescriptors::new();
        out.insert("field".to_owned(), ResourceDescriptor::Field1(field));
        Ok(out)
    }

    fn required_inputs(
        &self,
        requested_outputs: &OutputRegions,
        _inputs: &Descriptors,
        _params: &serde_json::Value,
    ) -> Result<InputRegions> {
        // Pointwise: each output sample needs exactly the co-located input pixel.
        let mut regions = InputRegions::new();
        if let Some(region) = requested_outputs.get("field") {
            regions.insert("image".to_owned(), *region);
        }
        Ok(regions)
    }

    fn validate_postconditions(
        &self,
        outputs: &OutputDescriptors,
        _params: &serde_json::Value,
    ) -> Result<Vec<AssertionResult>> {
        Ok(vec![match outputs.get("field") {
            Some(ResourceDescriptor::Field1(_)) => AssertionResult::pass("produces_field1"),
            _ => AssertionResult::fail("produces_field1", "no `field` Field1 output produced"),
        }])
    }
}

impl OpImplementation for ExtractChannel {
    fn compute(
        &self,
        inputs: &InputValues,
        params: &serde_json::Value,
    ) -> std::result::Result<OutputValues, Error> {
        let image = inputs.get("image").ok_or_else(|| {
            Error::new(
                ErrorClass::Reference,
                E_EXTRACT_INPUT,
                "image.extract_channel requires an `image` input value".to_owned(),
            )
        })?;
        let ResourceDescriptor::Image(descriptor) = image.descriptor() else {
            return Err(Error::new(
                ErrorClass::Type,
                E_EXTRACT_INPUT,
                "image.extract_channel `image` input must be an image resource".to_owned(),
            ));
        };
        let field = extract_infer(descriptor, params)?;
        let channel = channel_index(params)?;
        let channels = image.channels() as usize;
        let channel = channel as usize;

        // Gather every `channel`-th interleaved sample, in row-major pixel order.
        let samples: Vec<f32> = image
            .samples()
            .chunks_exact(channels.max(1))
            .map(|pixel| pixel[channel])
            .collect();

        let value = ResourceValue::new(ResourceDescriptor::Field1(field), 1, samples).map_err(
            |actual| {
                Error::new(
                    ErrorClass::Execution,
                    E_EXTRACT_INPUT,
                    format!(
                        "image.extract_channel produced a field buffer of unexpected length {actual}"
                    ),
                )
            },
        )?;

        let mut out = OutputValues::new();
        out.insert("field".to_owned(), value);
        Ok(out)
    }
}

/// Read the `image` input descriptor, erroring if absent or not an image.
fn extract_image_descriptor(inputs: &Descriptors) -> Result<ImageDescriptor> {
    let resource = inputs.get("image").ok_or_else(|| {
        Error::new(
            ErrorClass::Reference,
            E_EXTRACT_INPUT,
            "image.extract_channel requires an `image` input".to_owned(),
        )
    })?;
    let ResourceDescriptor::Image(descriptor) = resource else {
        return Err(Error::new(
            ErrorClass::Type,
            E_EXTRACT_INPUT,
            "image.extract_channel `image` input must be an image resource".to_owned(),
        ));
    };
    Ok(*descriptor)
}

// ---------------------------------------------------------------------------
// image.assemble_channels@1
// ---------------------------------------------------------------------------

/// The resolved output-image typing for an assembly: the channel layout and the
/// color/range/alpha/semantic metadata stamped onto the produced image.
#[derive(Debug, Clone, Copy)]
struct AssembleTyping {
    layout: ChannelLayout,
    color: ColorEncoding,
    range: ColorRange,
    alpha: AlphaRepresentation,
    semantic: SemanticRole,
}

impl AssembleTyping {
    /// Parse and validate the output-image typing params.
    fn resolve(params: &serde_json::Value) -> Result<Self> {
        let layout = parse_layout(params)?;
        let color = parse_color(params)?;
        let range = parse_color_range(params)?;
        let alpha = parse_alpha(params)?;
        let semantic = parse_semantic(params, SemanticRole::Material, E_ASSEMBLE_PARAM)?;
        Ok(Self {
            layout,
            color,
            range,
            alpha,
            semantic,
        })
    }

    /// The image descriptor an assembly with this typing produces for `extent`.
    const fn image_descriptor(self, extent: Extent) -> ImageDescriptor {
        ImageDescriptor {
            extent,
            layout: self.layout,
            scalar: ScalarType::F32,
            color: self.color,
            range: self.range,
            alpha: self.alpha,
            coordinates: CoordinateConvention::PixelCenterUpperLeft,
            semantic: self.semantic,
        }
    }
}

/// Parse the required `layout` param into a [`ChannelLayout`].
fn parse_layout(params: &serde_json::Value) -> Result<ChannelLayout> {
    let value = params.get("layout").ok_or_else(|| {
        Error::new(
            ErrorClass::Schema,
            E_ASSEMBLE_PARAM,
            "image.assemble_channels requires a `layout` parameter".to_owned(),
        )
    })?;
    serde_json::from_value(value.clone()).map_err(|e| {
        Error::new(
            ErrorClass::Schema,
            E_ASSEMBLE_PARAM,
            format!("image.assemble_channels `layout` is not a known channel layout: {e}"),
        )
        .with_context(
            ErrorContext::default()
                .with_actual(value.to_string())
                .with_expected("gray | gray-a | rgb | rgba"),
        )
    })
}

/// Parse the optional `color` param through [`RequestedColorEncoding`] so a
/// nameable-but-unsupported encoding (`display-p3`, `icc`) is rejected, not
/// silently approximated. Defaults to [`ColorEncoding::RawLinear`].
fn parse_color(params: &serde_json::Value) -> Result<ColorEncoding> {
    let Some(value) = params.get("color") else {
        return Ok(ColorEncoding::RawLinear);
    };
    let requested: RequestedColorEncoding = serde_json::from_value(value.clone()).map_err(|e| {
        Error::new(
            ErrorClass::Schema,
            E_ASSEMBLE_PARAM,
            format!("image.assemble_channels `color` is not a known color encoding: {e}"),
        )
        .with_context(ErrorContext::default().with_actual(value.to_string()))
    })?;
    requested.resolve()
}

/// The descriptors of every wired channel port, in `ch0..ch3` order, with
/// `None` for an unwired optional port.
fn assemble_channel_descriptors(inputs: &Descriptors) -> Result<Vec<FieldDescriptor>> {
    let mut fields = Vec::with_capacity(ASSEMBLE_PORTS.len());
    let mut saw_gap = false;
    for port in ASSEMBLE_PORTS {
        match inputs.get(port) {
            None => saw_gap = true,
            Some(resource) => {
                // Channel ports must be wired contiguously from ch0: a hole (ch1
                // wired but ch0 absent) is a malformed assembly, rejected rather
                // than packing channels into the wrong slots.
                if saw_gap {
                    return Err(Error::new(
                        ErrorClass::Semantic,
                        E_ASSEMBLE_INPUT,
                        format!(
                            "image.assemble_channels `{port}` is wired but an earlier channel \
                             port is not; channel ports must be wired contiguously from ch0"
                        ),
                    ));
                }
                let ResourceDescriptor::Field1(field) = resource else {
                    return Err(Error::new(
                        ErrorClass::Type,
                        E_ASSEMBLE_INPUT,
                        format!("image.assemble_channels `{port}` input must be a Field1 resource"),
                    ));
                };
                fields.push(*field);
            }
        }
    }
    Ok(fields)
}

/// Validate that the wired channel count matches the layout and the fields share
/// one extent, returning that common extent.
fn assemble_validate(fields: &[FieldDescriptor], layout: ChannelLayout) -> Result<Extent> {
    let wired = u32::try_from(fields.len()).unwrap_or(u32::MAX);
    let expected = layout.channel_count();
    if wired != expected {
        return Err(Error::new(
            ErrorClass::Semantic,
            E_ASSEMBLE_INPUT,
            format!(
                "image.assemble_channels layout has {expected} channels but {wired} channel \
                 ports are wired"
            ),
        )
        .with_context(
            ErrorContext::default()
                .with_actual(wired.to_string())
                .with_expected(expected.to_string()),
        ));
    }
    let Some(first) = fields.first() else {
        // A zero-channel layout is impossible (ChannelLayout has >= 1 channel), so
        // this is unreachable; report a typed error rather than panic.
        return Err(Error::new(
            ErrorClass::Semantic,
            E_ASSEMBLE_INPUT,
            "image.assemble_channels requires at least one channel".to_owned(),
        ));
    };
    let extent = first.extent;
    for field in &fields[1..] {
        if field.extent != extent {
            return Err(Error::new(
                ErrorClass::Semantic,
                E_ASSEMBLE_INPUT,
                "image.assemble_channels channel fields must share one extent".to_owned(),
            )
            .with_context(
                ErrorContext::default()
                    .with_actual(format!("{:?}", field.extent))
                    .with_expected(format!("{extent:?}")),
            ));
        }
    }
    Ok(extent)
}

/// The `image.assemble_channels@1` operation: up to four `Field1` inputs → an
/// interleaved `Image`.
#[derive(Debug, Clone, Copy, Default)]
pub struct AssembleChannels;

impl AssembleChannels {
    /// Construct the assemble-channels operation.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    /// The declared manifest for `image.assemble_channels@1`.
    ///
    /// # Errors
    /// Propagates the [`schema`](ErrorClass::Schema) error if the hard-coded op /
    /// impl ids are somehow invalid (they are not).
    pub fn manifest() -> Result<OperationManifest> {
        let channel_port = |name: &str, required: bool, ordinal: &str| InputSpec {
            name: name.to_owned(),
            kind: ResourceKind::Field1,
            required,
            doc: format!("The {ordinal} channel (Field1) to pack into the image."),
        };
        Ok(OperationManifest {
            id: ASSEMBLE_OP_ID.parse()?,
            impl_version: 1,
            summary: "Pack up to four Field1 scalar fields into an interleaved Image with a \
                      requested channel layout; channel count and extent are validated."
                .to_owned(),
            determinism: DeterminismTier::Exact,
            roi: RoiPolicy {
                category: RoiCategory::Pointwise,
                halo_px: None,
            },
            inputs: vec![
                channel_port("ch0", true, "first"),
                channel_port("ch1", false, "second"),
                channel_port("ch2", false, "third"),
                channel_port("ch3", false, "fourth"),
            ],
            outputs: vec![OutputSpec {
                name: "image".to_owned(),
                kind: ResourceKind::Image,
                doc: "The assembled interleaved image.".to_owned(),
            }],
            params: vec![
                ParamSpec {
                    name: "layout".to_owned(),
                    ty: ParamType::String,
                    unit: None,
                    required: true,
                    default: None,
                    choices: vec![
                        "gray".to_owned(),
                        "gray-a".to_owned(),
                        "rgb".to_owned(),
                        "rgba".to_owned(),
                    ],
                    doc: "The channel layout of the assembled image; its channel count must equal \
                          the number of wired channel ports."
                        .to_owned(),
                },
                ParamSpec {
                    name: "color".to_owned(),
                    ty: ParamType::String,
                    unit: None,
                    required: false,
                    default: Some(serde_json::json!("raw-linear")),
                    choices: vec![
                        "srgb".to_owned(),
                        "linear-srgb".to_owned(),
                        "raw-linear".to_owned(),
                    ],
                    doc: "The color transfer encoding of the assembled image.".to_owned(),
                },
                ParamSpec {
                    name: "range".to_owned(),
                    ty: ParamType::String,
                    unit: None,
                    required: false,
                    default: Some(serde_json::json!("display-referred")),
                    choices: vec!["display-referred".to_owned(), "scene-referred".to_owned()],
                    doc: "The reference-light range of the assembled image.".to_owned(),
                },
                ParamSpec {
                    name: "alpha".to_owned(),
                    ty: ParamType::String,
                    unit: None,
                    required: false,
                    default: Some(serde_json::json!("straight")),
                    choices: vec!["premultiplied".to_owned(), "straight".to_owned()],
                    doc: "The alpha representation of the assembled image.".to_owned(),
                },
                ParamSpec {
                    name: "semantic".to_owned(),
                    ty: ParamType::String,
                    unit: None,
                    required: false,
                    default: Some(serde_json::json!("material")),
                    choices: semantic_choices(),
                    doc: "The semantic role of the assembled image.".to_owned(),
                },
            ],
            implementations: vec![reference_impl()?],
            test: exact_pointwise_test_metadata(
                "image.assemble_channels interleaves scalar fields verbatim into an image; \
                 correctness is verified by an exact assembly fixture and a round-trip with \
                 image.extract_channel, not a perceptual metric",
            ),
        })
    }
}

impl OpContract for AssembleChannels {
    fn declared_inputs(&self) -> Vec<(String, ResourceKind)> {
        ASSEMBLE_PORTS
            .iter()
            .map(|p| ((*p).to_owned(), ResourceKind::Field1))
            .collect()
    }

    fn declared_outputs(&self) -> Vec<(String, ResourceKind)> {
        vec![("image".to_owned(), ResourceKind::Image)]
    }

    fn infer_outputs(
        &self,
        inputs: &Descriptors,
        params: &serde_json::Value,
    ) -> Result<OutputDescriptors> {
        let typing = AssembleTyping::resolve(params)?;
        let fields = assemble_channel_descriptors(inputs)?;
        let extent = assemble_validate(&fields, typing.layout)?;
        let mut out = OutputDescriptors::new();
        out.insert(
            "image".to_owned(),
            ResourceDescriptor::Image(typing.image_descriptor(extent)),
        );
        Ok(out)
    }

    fn required_inputs(
        &self,
        requested_outputs: &OutputRegions,
        inputs: &Descriptors,
        _params: &serde_json::Value,
    ) -> Result<InputRegions> {
        // Pointwise: each output pixel needs the co-located sample of every wired
        // channel field.
        let mut regions = InputRegions::new();
        if let Some(region) = requested_outputs.get("image") {
            for port in ASSEMBLE_PORTS {
                if inputs.contains_key(port) {
                    regions.insert(port.to_owned(), *region);
                }
            }
        }
        Ok(regions)
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
        let typing = AssembleTyping::resolve(params)?;
        Ok(vec![if image.layout == typing.layout {
            AssertionResult::pass("layout_matches_request")
        } else {
            AssertionResult::fail(
                "layout_matches_request",
                format!(
                    "assembled layout {:?} does not match requested {:?}",
                    image.layout, typing.layout
                ),
            )
        }])
    }
}

impl OpImplementation for AssembleChannels {
    fn compute(
        &self,
        inputs: &InputValues,
        params: &serde_json::Value,
    ) -> std::result::Result<OutputValues, Error> {
        let typing = AssembleTyping::resolve(params)?;

        // Collect the wired channel field *values* in ch0..ch3 order, enforcing
        // the same contiguity rule as the descriptor pass.
        let mut channels: Vec<&ResourceValue> = Vec::with_capacity(ASSEMBLE_PORTS.len());
        let mut saw_gap = false;
        for port in ASSEMBLE_PORTS {
            match inputs.get(port) {
                None => saw_gap = true,
                Some(value) => {
                    if saw_gap {
                        return Err(Error::new(
                            ErrorClass::Semantic,
                            E_ASSEMBLE_INPUT,
                            format!(
                                "image.assemble_channels `{port}` is wired but an earlier channel \
                                 port is not"
                            ),
                        ));
                    }
                    let ResourceDescriptor::Field1(_) = value.descriptor() else {
                        return Err(Error::new(
                            ErrorClass::Type,
                            E_ASSEMBLE_INPUT,
                            format!(
                                "image.assemble_channels `{port}` input must be a Field1 resource"
                            ),
                        ));
                    };
                    channels.push(value);
                }
            }
        }

        let descriptors: Vec<FieldDescriptor> = channels
            .iter()
            .filter_map(|v| match v.descriptor() {
                ResourceDescriptor::Field1(f) => Some(*f),
                _ => None,
            })
            .collect();
        let extent = assemble_validate(&descriptors, typing.layout)?;

        let channel_count = typing.layout.channel_count() as usize;
        let pixels = (extent.width as usize).saturating_mul(extent.height as usize);
        let mut samples = vec![0.0_f32; pixels.saturating_mul(channel_count)];
        for (k, channel) in channels.iter().enumerate() {
            let field = channel.samples();
            for (pixel, &s) in field.iter().enumerate() {
                samples[pixel * channel_count + k] = s;
            }
        }

        let value = ResourceValue::new(
            ResourceDescriptor::Image(typing.image_descriptor(extent)),
            typing.layout.channel_count(),
            samples,
        )
        .map_err(|actual| {
            Error::new(
                ErrorClass::Execution,
                E_ASSEMBLE_INPUT,
                format!(
                    "image.assemble_channels produced an image buffer of unexpected length {actual}"
                ),
            )
        })?;

        let mut out = OutputValues::new();
        out.insert("image".to_owned(), value);
        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// shared helpers
// ---------------------------------------------------------------------------

/// Parse an optional `range` color-range param, defaulting to display-referred.
fn parse_color_range(params: &serde_json::Value) -> Result<ColorRange> {
    params
        .get("range")
        .map_or(Ok(ColorRange::DisplayReferred), |value| {
            serde_json::from_value(value.clone())
                .map_err(|e| enum_param_error("range", value, &e, E_ASSEMBLE_PARAM))
        })
}

/// Parse an optional `alpha` representation param, defaulting to straight.
fn parse_alpha(params: &serde_json::Value) -> Result<AlphaRepresentation> {
    params
        .get("alpha")
        .map_or(Ok(AlphaRepresentation::Straight), |value| {
            serde_json::from_value(value.clone())
                .map_err(|e| enum_param_error("alpha", value, &e, E_ASSEMBLE_PARAM))
        })
}

/// Parse an optional `semantic` role param, defaulting to `default` and erroring
/// (code `code`) on an unknown token.
fn parse_semantic(
    params: &serde_json::Value,
    default: SemanticRole,
    code: &'static str,
) -> Result<SemanticRole> {
    params.get("semantic").map_or(Ok(default), |value| {
        serde_json::from_value(value.clone())
            .map_err(|e| enum_param_error("semantic", value, &e, code))
    })
}

/// Build the schema error for an unrecognized enum param token.
fn enum_param_error(
    name: &str,
    value: &serde_json::Value,
    source: &serde_json::Error,
    code: &'static str,
) -> Error {
    Error::new(
        ErrorClass::Schema,
        code,
        format!("`{name}` is not a recognized value: {source}"),
    )
    .with_context(ErrorContext::default().with_actual(value.to_string()))
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

/// Verification declarations for an exact, single-reference, pointwise op:
/// differential and perceptual do not apply (single reference / bit-exact), and
/// every other applicable category is covered by this module's tests. `reason`
/// justifies the perceptual skip.
fn exact_pointwise_test_metadata(reason: &str) -> TestMetadata {
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
    // Perceptual is not applicable to an `exact` op, but declaring it explicitly
    // with a reason keeps the report self-documenting.
    decls = decls.with(
        VerificationCategory::Perceptual,
        CategoryStatus::not_applicable(reason.to_owned()),
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
