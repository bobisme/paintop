//! The `image.pad@1` operation: enlarge an image with margins filled per a
//! boundary mode (`OP_CATALOG` §5, `IR_SPEC` §8.4).
//!
//! `image.pad` adds `left`/`right`/`top`/`bottom` margins (in pixels) around an
//! input image, producing a larger image whose interior is the verbatim input and
//! whose new border samples are synthesized from the input under a **boundary
//! mode**:
//!
//! - **`constant`** — every margin sample is a fixed per-channel `value`
//!   (default `0`).
//! - **`clamp`** — each margin sample copies the nearest input edge sample
//!   (edge replication).
//! - **`mirror`** — reflect across the edge *without* repeating the edge sample
//!   (`gfedcb|abcde|dcba`-style half-sample mirror).
//! - **`wrap`** — tile the image periodically (toroidal).
//!
//! # Negative margins (crop policy)
//!
//! A negative margin removes rows/columns from that side rather than adding them:
//! `image.pad` with all-negative margins is exactly `image.crop` of the shrunken
//! interior, so the crop/pad calculus is one continuous family. A margin may not
//! remove more than the extent on its axis (the two opposing margins must leave a
//! non-negative resulting extent); such a request is rejected.
//!
//! # Determinism
//!
//! The op is [`Exact`](DeterminismTier::Exact): every output sample is either a
//! verbatim copy of an input sample (interior, clamp, mirror, wrap) or the exact
//! constant `value`, so the result is bit-identical on every run.

use paintop_core::executor::{InputValues, OpImplementation, OutputValues, ResourceValue};
use paintop_ir::{
    AssertionResult, Descriptors, DeterminismTier, Error, ErrorClass, ErrorContext, Extent,
    ImageDescriptor, ImplId, InputRegions, InputSpec, OpContract, OperationManifest,
    OutputDescriptors, OutputRegions, OutputSpec, ParamSpec, ParamType, ParamUnit, Rect,
    ResourceDescriptor, ResourceKind, Result, RoiCategory, RoiPolicy, TestMetadata,
};

/// The canonical id of the image-pad operation.
pub const PAD_OP_ID: &str = "image.pad@1";

/// The `image` input was absent or carried a non-image descriptor.
pub const E_PAD_INPUT: &str = "E_PAD_INPUT";

/// A margin / boundary-mode / value parameter was missing or malformed.
pub const E_PAD_PARAM: &str = "E_PAD_PARAM";

/// The resulting extent would be negative (a margin removed more than its axis).
pub const E_PAD_EXTENT: &str = "E_PAD_EXTENT";

/// The `value` array length did not match the image's channel count.
pub const E_PAD_VALUE: &str = "E_PAD_VALUE";

/// How the synthesized margin samples are derived from the input edge.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BoundaryMode {
    /// A fixed per-channel constant.
    Constant,
    /// Replicate the nearest edge sample.
    Clamp,
    /// Half-sample reflection across the edge (edge not repeated).
    Mirror,
    /// Periodic (toroidal) tiling.
    Wrap,
}

impl BoundaryMode {
    /// Parse the boundary mode from its wire token.
    fn from_token(token: &str) -> Option<Self> {
        match token {
            "constant" => Some(Self::Constant),
            "clamp" => Some(Self::Clamp),
            "mirror" => Some(Self::Mirror),
            "wrap" => Some(Self::Wrap),
            _ => None,
        }
    }
}

/// The four signed margins, one per side, in pixels.
#[derive(Debug, Clone, Copy)]
struct Margins {
    left: i64,
    right: i64,
    top: i64,
    bottom: i64,
}

/// A fully-resolved pad request: margins, mode, and (for `constant`) the
/// per-channel fill value.
#[derive(Debug, Clone)]
struct PadRequest {
    margins: Margins,
    mode: BoundaryMode,
    value: Vec<f32>,
}

impl PadRequest {
    /// Parse and validate every pad param against the input descriptor.
    fn resolve(params: &serde_json::Value, descriptor: &ImageDescriptor) -> Result<Self> {
        let margins = Margins {
            left: i64_param(params, "left")?,
            right: i64_param(params, "right")?,
            top: i64_param(params, "top")?,
            bottom: i64_param(params, "bottom")?,
        };
        let mode = mode_param(params)?;
        let channels = descriptor.layout.channel_count() as usize;
        let value = value_param(params, channels)?;
        Ok(Self {
            margins,
            mode,
            value,
        })
    }

    /// The output extent after applying the (possibly negative) margins.
    ///
    /// # Errors
    /// [`policy`](ErrorClass::Policy) / [`E_PAD_EXTENT`] if either axis would
    /// become negative.
    fn output_extent(&self, src: Extent) -> Result<Extent> {
        let w = i64::from(src.width) + self.margins.left + self.margins.right;
        let h = i64::from(src.height) + self.margins.top + self.margins.bottom;
        if w < 0 || h < 0 {
            return Err(Error::new(
                ErrorClass::Policy,
                E_PAD_EXTENT,
                "image.pad margins remove more than the image extent on an axis".to_owned(),
            )
            .with_context(
                ErrorContext::default()
                    .with_actual(format!("{w}x{h}"))
                    .with_expected("non-negative on both axes"),
            ));
        }
        #[allow(
            clippy::cast_sign_loss,
            clippy::cast_possible_truncation,
            reason = "w, h are checked non-negative; padded extents stay within u32 for valid plans"
        )]
        Ok(Extent::new(w as u32, h as u32))
    }
}

/// Parse a required signed integer margin param (defaulting to `0` when absent).
fn i64_param(params: &serde_json::Value, name: &str) -> Result<i64> {
    let Some(value) = params.get(name) else {
        return Ok(0);
    };
    value.as_i64().ok_or_else(|| {
        Error::new(
            ErrorClass::Schema,
            E_PAD_PARAM,
            format!("image.pad `{name}` must be an integer number of pixels"),
        )
        .with_context(ErrorContext::default().with_actual(value.to_string()))
    })
}

/// Parse the optional `mode` param, defaulting to `constant`.
fn mode_param(params: &serde_json::Value) -> Result<BoundaryMode> {
    let Some(value) = params.get("mode") else {
        return Ok(BoundaryMode::Constant);
    };
    let token = value.as_str().ok_or_else(|| {
        Error::new(
            ErrorClass::Schema,
            E_PAD_PARAM,
            "image.pad `mode` must be a string boundary mode".to_owned(),
        )
        .with_context(ErrorContext::default().with_actual(value.to_string()))
    })?;
    BoundaryMode::from_token(token).ok_or_else(|| {
        Error::new(
            ErrorClass::Schema,
            E_PAD_PARAM,
            format!("image.pad `mode` is not a known boundary mode: {token}"),
        )
        .with_context(
            ErrorContext::default()
                .with_actual(token.to_owned())
                .with_expected("constant | clamp | mirror | wrap"),
        )
    })
}

/// Parse the optional per-channel `value` array, defaulting to all-zero.
fn value_param(params: &serde_json::Value, channels: usize) -> Result<Vec<f32>> {
    let Some(value) = params.get("value") else {
        return Ok(vec![0.0; channels]);
    };
    let array = value.as_array().ok_or_else(|| {
        Error::new(
            ErrorClass::Schema,
            E_PAD_VALUE,
            "image.pad `value` must be a per-channel array".to_owned(),
        )
        .with_context(ErrorContext::default().with_actual(value.to_string()))
    })?;
    if array.len() != channels {
        return Err(Error::new(
            ErrorClass::Schema,
            E_PAD_VALUE,
            format!(
                "image.pad `value` has {} components but the image has {channels} channels",
                array.len()
            ),
        )
        .with_context(
            ErrorContext::default()
                .with_actual(array.len().to_string())
                .with_expected(channels.to_string()),
        ));
    }
    let mut out = Vec::with_capacity(channels);
    for (i, component) in array.iter().enumerate() {
        let n = component.as_f64().ok_or_else(|| {
            Error::new(
                ErrorClass::Schema,
                E_PAD_VALUE,
                format!("image.pad `value[{i}]` must be a number"),
            )
            .with_context(ErrorContext::default().with_actual(component.to_string()))
        })?;
        if !n.is_finite() {
            return Err(Error::new(
                ErrorClass::Schema,
                E_PAD_VALUE,
                format!("image.pad `value[{i}]` must be finite, got {n}"),
            ));
        }
        #[allow(
            clippy::cast_possible_truncation,
            reason = "fill stored as the image's f32 sample type"
        )]
        out.push(n as f32);
    }
    Ok(out)
}

/// Map an output coordinate on one axis to a source index under the boundary
/// mode, given the source length `n` (`n >= 1`) and the leading margin `lead`.
///
/// Returns `Some(src_index)` to copy `src[src_index]`, or `None` to use the
/// constant value (only when `mode == Constant` and the coordinate is outside the
/// interior).
fn source_index(out_coord: i64, lead: i64, n: i64, mode: BoundaryMode) -> Option<i64> {
    let c = out_coord - lead; // coordinate in source space (may be < 0 or >= n)
    if c >= 0 && c < n {
        return Some(c);
    }
    match mode {
        BoundaryMode::Constant => None,
        BoundaryMode::Clamp => Some(c.clamp(0, n - 1)),
        BoundaryMode::Wrap => Some(c.rem_euclid(n)),
        BoundaryMode::Mirror => Some(mirror_index(c, n)),
    }
}

/// Whole-sample mirror (`reflect`): fold `c` into `[0, n)` reflecting across the
/// edge sample *without* repeating it, using a period-`2(n-1)` triangle wave
/// (`...d c | a b c d | c b a`). For `n == 1` every index maps to `0`.
const fn mirror_index(c: i64, n: i64) -> i64 {
    if n == 1 {
        return 0;
    }
    let period = 2 * (n - 1);
    let m = c.rem_euclid(period);
    if m < n { m } else { period - m }
}

/// Build the padded interleaved sample buffer.
///
/// `src` is the source extent and `samples` its row-major interleaved buffer.
/// Negative margins remove the leading/trailing rows/columns; positive margins add
/// border samples per `mode`.
fn pad_samples(
    samples: &[f32],
    src: Extent,
    out: Extent,
    request: &PadRequest,
    channels: u32,
) -> Vec<f32> {
    let stride = channels as usize;
    let src_w = i64::from(src.width);
    let src_h = i64::from(src.height);
    let out_w = out.width as usize;
    let out_h = out.height as usize;
    let lead_x = request.margins.left;
    let lead_y = request.margins.top;

    let mut buf = Vec::with_capacity(out_w.saturating_mul(out_h).saturating_mul(stride));
    for oy in 0..out_h {
        #[allow(
            clippy::cast_possible_wrap,
            reason = "out_h fits i64 for valid extents"
        )]
        let sy = source_index(oy as i64, lead_y, src_h, request.mode);
        for ox in 0..out_w {
            #[allow(
                clippy::cast_possible_wrap,
                reason = "out_w fits i64 for valid extents"
            )]
            let sx = source_index(ox as i64, lead_x, src_w, request.mode);
            match (sx, sy) {
                (Some(x), Some(y)) => {
                    // source_index returns indices within [0, n), so each fits a
                    // usize; the fallback never triggers for a valid request.
                    let xi = usize::try_from(x).unwrap_or(0);
                    let yi = usize::try_from(y).unwrap_or(0);
                    let base = (yi * (src.width as usize) + xi) * stride;
                    buf.extend_from_slice(&samples[base..base + stride]);
                }
                _ => buf.extend_from_slice(&request.value),
            }
        }
    }
    buf
}

/// The `image.pad@1` operation: an image + margins/mode/value → an enlarged image.
#[derive(Debug, Clone, Copy, Default)]
pub struct Pad;

impl Pad {
    /// Construct the image-pad operation.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    /// The declared manifest for `image.pad@1`.
    ///
    /// # Errors
    /// Propagates the [`schema`](ErrorClass::Schema) error if the hard-coded op /
    /// impl ids are somehow invalid (they are not).
    pub fn manifest() -> Result<OperationManifest> {
        let margin_param = |name: &str, doc: &str| ParamSpec {
            name: name.to_owned(),
            ty: ParamType::Integer,
            unit: Some(ParamUnit::Pixels),
            required: false,
            default: Some(serde_json::json!(0)),
            choices: vec![],
            doc: doc.to_owned(),
        };
        Ok(OperationManifest {
            id: PAD_OP_ID.parse()?,
            impl_version: 1,
            summary: "Enlarge an image with left/right/top/bottom margins filled per a boundary \
                      mode (constant/clamp/mirror/wrap); negative margins crop that side."
                .to_owned(),
            determinism: DeterminismTier::Exact,
            roi: RoiPolicy {
                category: RoiCategory::Geometric,
                halo_px: None,
            },
            inputs: vec![InputSpec {
                name: "image".to_owned(),
                kind: ResourceKind::Image,
                required: true,
                doc: "The image to pad.".to_owned(),
            }],
            outputs: vec![OutputSpec {
                name: "image".to_owned(),
                kind: ResourceKind::Image,
                doc: "The padded image (extent grown by the margins; all other descriptor fields \
                      preserved)."
                    .to_owned(),
            }],
            params: vec![
                margin_param("left", "Pixels to add on the left (negative crops)."),
                margin_param("right", "Pixels to add on the right (negative crops)."),
                margin_param("top", "Pixels to add on the top (negative crops)."),
                margin_param("bottom", "Pixels to add on the bottom (negative crops)."),
                ParamSpec {
                    name: "mode".to_owned(),
                    ty: ParamType::String,
                    unit: None,
                    required: false,
                    default: Some(serde_json::json!("constant")),
                    choices: vec![
                        "constant".to_owned(),
                        "clamp".to_owned(),
                        "mirror".to_owned(),
                        "wrap".to_owned(),
                    ],
                    doc: "How border samples are synthesized from the input edge.".to_owned(),
                },
                ParamSpec {
                    name: "value".to_owned(),
                    ty: ParamType::Json,
                    unit: None,
                    required: false,
                    default: None,
                    choices: vec![],
                    doc: "Per-channel constant fill for `constant` mode; defaults to all-zero."
                        .to_owned(),
                },
            ],
            implementations: vec![reference_impl()?],
            test: pad_test_metadata(),
        })
    }
}

impl OpContract for Pad {
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
        let request = PadRequest::resolve(params, descriptor)?;
        let extent = request.output_extent(descriptor.extent)?;

        let mut out_desc = *descriptor;
        out_desc.extent = extent;
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
        // Geometric / full-domain on the border: a margin sample under clamp /
        // mirror / wrap can reference an arbitrary input column or row, so demand
        // the whole input plane intersected with the requested window's source
        // mapping. Conservatively, the full input is required when any non-constant
        // mode is in play; under constant the interior maps 1:1 (shifted).
        let descriptor = image_descriptor(inputs)?;
        let request = PadRequest::resolve(params, descriptor)?;
        let mut regions = InputRegions::new();
        if let Some(region) = requested_outputs.get("image") {
            let full = Rect::new(
                0,
                0,
                i64::from(descriptor.extent.width),
                i64::from(descriptor.extent.height),
            );
            let mapped = if request.mode == BoundaryMode::Constant {
                // Interior maps by -lead; clamp the shifted window to the input.
                let shifted = Rect::new(
                    region.x0 - request.margins.left,
                    region.y0 - request.margins.top,
                    region.x1 - request.margins.left,
                    region.y1 - request.margins.top,
                );
                shifted.intersect(full)
            } else {
                full
            };
            regions.insert("image".to_owned(), mapped);
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

impl OpImplementation for Pad {
    fn compute(
        &self,
        inputs: &InputValues,
        params: &serde_json::Value,
    ) -> std::result::Result<OutputValues, Error> {
        let image = inputs.get("image").ok_or_else(|| {
            Error::new(
                ErrorClass::Reference,
                E_PAD_INPUT,
                "image.pad requires an `image` input value".to_owned(),
            )
        })?;
        let ResourceDescriptor::Image(descriptor) = image.descriptor() else {
            return Err(Error::new(
                ErrorClass::Type,
                E_PAD_INPUT,
                "image.pad `image` input must be an image resource".to_owned(),
            ));
        };
        let request = PadRequest::resolve(params, descriptor)?;
        let out_extent = request.output_extent(descriptor.extent)?;

        // Clamp/mirror/wrap of a zero-area input have no edge to sample; only
        // constant (or an all-zero output) is well defined. Guard against it.
        let samples = if descriptor.extent.width == 0 || descriptor.extent.height == 0 {
            zero_area_pad(&request, out_extent, image.channels())?
        } else {
            pad_samples(
                image.samples(),
                descriptor.extent,
                out_extent,
                &request,
                image.channels(),
            )
        };

        let mut out_desc = *descriptor;
        out_desc.extent = out_extent;
        let value = ResourceValue::new(
            ResourceDescriptor::Image(out_desc),
            image.channels(),
            samples,
        )
        .map_err(|actual| {
            Error::new(
                ErrorClass::Execution,
                E_PAD_INPUT,
                format!("image.pad produced a sample buffer of unexpected length {actual}"),
            )
        })?;

        let mut out = OutputValues::new();
        out.insert("image".to_owned(), value);
        Ok(out)
    }
}

/// Pad a zero-area input. Only `constant` mode (the constant fills the whole
/// output) is well defined; the non-constant modes have no edge to replicate and
/// are rejected unless the output is also empty.
fn zero_area_pad(request: &PadRequest, out: Extent, channels: u32) -> Result<Vec<f32>> {
    let pixels = (out.width as usize).saturating_mul(out.height as usize);
    if pixels == 0 {
        return Ok(Vec::new());
    }
    if request.mode != BoundaryMode::Constant {
        return Err(Error::new(
            ErrorClass::Semantic,
            E_PAD_PARAM,
            "image.pad cannot synthesize clamp/mirror/wrap borders from a zero-area input"
                .to_owned(),
        ));
    }
    let stride = channels as usize;
    let mut buf = Vec::with_capacity(pixels * stride);
    for _ in 0..pixels {
        buf.extend_from_slice(&request.value);
    }
    Ok(buf)
}

/// Extract the required `image` input descriptor, erroring if absent or non-image.
fn image_descriptor(inputs: &Descriptors) -> Result<&ImageDescriptor> {
    let image = inputs.get("image").ok_or_else(|| {
        Error::new(
            ErrorClass::Reference,
            E_PAD_INPUT,
            "image.pad requires an `image` input".to_owned(),
        )
    })?;
    let ResourceDescriptor::Image(descriptor) = image else {
        return Err(Error::new(
            ErrorClass::Type,
            E_PAD_INPUT,
            "image.pad `image` input must be an image resource".to_owned(),
        ));
    };
    Ok(descriptor)
}

/// The mandatory `cpu.reference@1` oracle implementation id.
fn reference_impl() -> Result<ImplId> {
    ImplId::new("cpu", "reference", 1)
}

/// Verification declarations for `image.pad@1`: an exact, single-reference,
/// geometric op. Differential does not apply (one implementation). Perceptual is
/// not applicable: every output sample is a verbatim copy or the exact constant,
/// verified by per-mode boundary fixtures and the crop/pad round-trip, not a
/// perceptual metric.
fn pad_test_metadata() -> TestMetadata {
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
            "image.pad copies the interior verbatim and synthesizes borders by exact \
             clamp/mirror/wrap/constant rules; correctness is verified by per-mode boundary \
             fixtures and the crop/pad round-trip identity, not a perceptual-quality metric",
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
