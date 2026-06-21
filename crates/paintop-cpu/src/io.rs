//! The `io.decode_image@1` / `io.encode_image@1` operations: PNG decode →
//! [`ResourceValue`] and [`ResourceValue`] → PNG export (`OP_CATALOG` §1,
//! `plan.md` §17.2, `IR_SPEC` §4).
//!
//! The `image` crate (0.25) is used **only** as a PNG codec: its decoded bytes
//! are converted into paintop's resource types immediately and never escape this
//! module. Decoding runs behind explicit, enforced [`DecodeLimits`] — a maximum
//! decoded-pixel count, maximum per-axis dimensions, and a decompression-bomb
//! allocation ceiling — so a malicious or malformed PNG is rejected with a typed
//! [`asset`](paintop_ir::ErrorClass::Asset) / [`policy`](paintop_ir::ErrorClass::Policy)
//! error rather than exhausting memory. Encoding is lossless and writes its file
//! **atomically** (write to a sibling temp file, then rename), so a crash mid-write
//! never leaves a torn output.
//!
//! # Scalar model
//!
//! The M0 [`ResourceValue`] buffer is `f32` (`paintop_core::executor::value`).
//! PNG import/export is 8-bit, so a decoded sample `b: u8` is stored normalized as
//! `b as f32 / 255.0` and re-quantized on encode by rounding `clamp(v, 0, 1) *
//! 255` to the nearest integer. The descriptor records [`ScalarType::U8`] so the
//! logical content is exactly the 8-bit raster; the *round trip*
//! decode∘encode∘decode is byte-exact (the `decode_then_encode_round_trips`
//! test). 16-bit PNGs are out of the M0 scalar set ([`ScalarType`] has no `u16`)
//! and are rejected rather than silently truncated.
//!
//! # Color / alpha metadata
//!
//! A decoded PNG carries no reliable color-management metadata in the M0 codec
//! path, so it is typed with the import defaults: [`ColorEncoding::Srgb`] (PNG's
//! conventional display encoding), [`AlphaRepresentation::Straight`] (PNG stores
//! unassociated alpha), [`ColorRange::DisplayReferred`], and
//! [`SemanticRole::Color`]. The channel layout is taken from the PNG color type
//! and is preserved exactly across a round trip (the
//! `round_trip_preserves_layout_and_alpha` test).

use std::path::{Path, PathBuf};

use image::codecs::png::{CompressionType, FilterType, PngDecoder, PngEncoder};
use image::{ColorType, ExtendedColorType, ImageDecoder, ImageEncoder, ImageError};
use paintop_core::executor::{InputValues, OpImplementation, OutputValues, ResourceValue};
use paintop_ir::{
    AlphaRepresentation, ChannelLayout, ColorEncoding, ColorRange, CoordinateConvention,
    Descriptors, DeterminismTier, Error, ErrorClass, ErrorContext, Extent, ImageDescriptor, ImplId,
    InputRegions, InputSpec, OpContract, OperationManifest, OutputDescriptors, OutputRegions,
    OutputSpec, ParamSpec, ParamType, ResourceDescriptor, ResourceKind, Result, RoiCategory,
    RoiPolicy, ScalarType, SemanticRole, TestMetadata,
};

/// The canonical id of the decode operation.
pub const DECODE_OP_ID: &str = "io.decode_image@1";
/// The canonical id of the encode operation.
pub const ENCODE_OP_ID: &str = "io.encode_image@1";

/// A decoded PNG exceeded the configured [`DecodeLimits`] (pixels, dimensions, or
/// the decompression-bomb allocation ceiling).
pub const E_DECODE_LIMIT_EXCEEDED: &str = "E_DECODE_LIMIT_EXCEEDED";
/// A PNG was truncated, malformed, or otherwise undecodable.
pub const E_DECODE_MALFORMED: &str = "E_DECODE_MALFORMED";
/// A PNG used a pixel format outside the M0 import set (e.g. 16-bit channels).
pub const E_DECODE_UNSUPPORTED_FORMAT: &str = "E_DECODE_UNSUPPORTED_FORMAT";
/// A required `path` parameter was missing or not a string.
pub const E_IO_PATH_PARAM: &str = "E_IO_PATH_PARAM";
/// A filesystem read/write/rename failed.
pub const E_IO_FILESYSTEM: &str = "E_IO_FILESYSTEM";
/// The `image` input to the encoder was absent or had a non-image descriptor.
pub const E_ENCODE_INPUT: &str = "E_ENCODE_INPUT";

/// The enforced decode budget (`plan.md` §17.2: maximum decoded pixels, maximum
/// dimensions, decompression-bomb guard, integer-overflow checks).
///
/// The defaults mirror the `IR_SPEC` §4 input-declaration limits example
/// (`8192 × 8192`, `67_108_864` pixels) and add a `max_alloc` ceiling the codec
/// enforces while inflating, so a small file that claims a vast canvas is
/// rejected before the allocation rather than after.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DecodeLimits {
    /// Maximum image width in pixels.
    pub max_width: u32,
    /// Maximum image height in pixels.
    pub max_height: u32,
    /// Maximum decoded pixel count (`width * height`).
    pub max_pixels: u64,
    /// Maximum bytes the codec may allocate while decoding (the
    /// decompression-bomb ceiling).
    pub max_alloc_bytes: u64,
}

impl DecodeLimits {
    /// The `IR_SPEC` §4 default budget: `8192 × 8192`, `67_108_864` pixels, and a
    /// 512 MiB allocation ceiling.
    pub const DEFAULT: Self = Self {
        max_width: 8192,
        max_height: 8192,
        max_pixels: 67_108_864,
        max_alloc_bytes: 512 * 1024 * 1024,
    };
}

impl Default for DecodeLimits {
    fn default() -> Self {
        Self::DEFAULT
    }
}

/// Build a [`policy`](ErrorClass::Policy) limit-exceeded error.
fn limit_error(message: impl Into<String>, actual: impl Into<String>) -> Error {
    Error::new(ErrorClass::Policy, E_DECODE_LIMIT_EXCEEDED, message)
        .with_context(ErrorContext::default().with_actual(actual))
}

/// Build an [`asset`](ErrorClass::Asset) malformed-input error.
fn malformed_error(message: impl Into<String>) -> Error {
    Error::new(ErrorClass::Asset, E_DECODE_MALFORMED, message)
}

/// Decode a PNG byte stream into a row-major `f32`-normalized [`ResourceValue`],
/// enforcing `limits`.
///
/// The dimensions are checked *before* any pixel buffer is allocated, and the
/// codec's own allocation ceiling is set from `limits.max_alloc_bytes`, so a
/// decompression bomb (a tiny file claiming a huge canvas) is refused at the
/// header rather than after inflating. The pixel/byte arithmetic is overflow
/// checked via [`Extent::pixel_count`] / [`Extent::byte_count`].
///
/// # Errors
/// - [`policy`](ErrorClass::Policy) / [`E_DECODE_LIMIT_EXCEEDED`] if the image's
///   dimensions or pixel count exceed `limits`, or the codec hits its allocation
///   ceiling.
/// - [`asset`](ErrorClass::Asset) / [`E_DECODE_MALFORMED`] if the bytes are not a
///   decodable PNG (truncated, bad CRC, malformed chunks).
/// - [`asset`](ErrorClass::Asset) / [`E_DECODE_UNSUPPORTED_FORMAT`] if the PNG
///   uses a pixel format outside the M0 8-bit import set.
pub fn decode_png(bytes: &[u8], limits: &DecodeLimits) -> Result<ResourceValue> {
    let codec_limits = {
        let mut l = image::Limits::no_limits();
        l.max_image_width = Some(limits.max_width);
        l.max_image_height = Some(limits.max_height);
        l.max_alloc = Some(limits.max_alloc_bytes);
        l
    };

    let decoder = PngDecoder::with_limits(std::io::Cursor::new(bytes), codec_limits)
        .map_err(map_decode_error)?;

    let (width, height) = decoder.dimensions();
    let layout = layout_for(decoder.color_type())?;
    let extent = Extent::new(width, height);

    // Enforce the per-axis and pixel-count budget with overflow-checked
    // arithmetic before allocating the sample buffer.
    if width > limits.max_width || height > limits.max_height {
        return Err(limit_error(
            format!(
                "decoded image {width}x{height} exceeds the maximum dimensions {}x{}",
                limits.max_width, limits.max_height
            ),
            format!("{width}x{height}"),
        ));
    }
    let pixels = extent.pixel_count()?;
    if pixels > limits.max_pixels {
        return Err(limit_error(
            format!(
                "decoded image has {pixels} pixels, exceeding the maximum {}",
                limits.max_pixels
            ),
            pixels.to_string(),
        ));
    }

    let channels = layout.channel_count();
    let byte_count = extent.byte_count(channels, ScalarType::U8)?;

    // The decompression-bomb guard: refuse to allocate the decoded raster if it
    // would exceed the allocation ceiling, *before* the buffer is reserved. This
    // does not rely on the codec's internal `max_alloc` (which does not cover the
    // caller-provided output buffer), so a tiny file claiming a vast canvas is
    // rejected deterministically.
    if byte_count > limits.max_alloc_bytes {
        return Err(limit_error(
            format!(
                "decoded raster would allocate {byte_count} bytes, exceeding the ceiling {}",
                limits.max_alloc_bytes
            ),
            byte_count.to_string(),
        ));
    }

    let buffer_len = usize::try_from(byte_count).map_err(|_| {
        limit_error(
            "decoded image byte count does not fit in addressable memory",
            byte_count.to_string(),
        )
    })?;

    let mut raw = vec![0u8; buffer_len];
    decoder.read_image(&mut raw).map_err(map_decode_error)?;

    // Normalize the 8-bit samples into the f32 resource buffer.
    let samples: Vec<f32> = raw.iter().map(|&b| f32::from(b) / 255.0).collect();

    let descriptor = ResourceDescriptor::Image(decoded_image_descriptor(extent, layout));
    ResourceValue::new(descriptor, channels, samples).map_err(|actual| {
        malformed_error(format!("decoded sample buffer length {actual} mismatch"))
    })
}

/// The descriptor of a freshly decoded PNG: the channel layout from the file,
/// import-default color/alpha metadata, and `u8` scalar storage.
const fn decoded_image_descriptor(extent: Extent, layout: ChannelLayout) -> ImageDescriptor {
    ImageDescriptor {
        extent,
        layout,
        scalar: ScalarType::U8,
        color: ColorEncoding::Srgb,
        range: ColorRange::DisplayReferred,
        alpha: AlphaRepresentation::Straight,
        coordinates: CoordinateConvention::PixelCenterUpperLeft,
        semantic: SemanticRole::Color,
    }
}

/// Map an `image` [`ColorType`] to a paintop [`ChannelLayout`], rejecting any
/// format outside the M0 8-bit import set.
fn layout_for(color: ColorType) -> Result<ChannelLayout> {
    match color {
        ColorType::L8 => Ok(ChannelLayout::Gray),
        ColorType::La8 => Ok(ChannelLayout::GrayA),
        ColorType::Rgb8 => Ok(ChannelLayout::Rgb),
        ColorType::Rgba8 => Ok(ChannelLayout::Rgba),
        other => Err(Error::new(
            ErrorClass::Asset,
            E_DECODE_UNSUPPORTED_FORMAT,
            format!("PNG color type {other:?} is outside the M0 8-bit import set"),
        )),
    }
}

/// Classify an `image` decode error into the paintop taxonomy.
///
/// A codec limit failure is a [`policy`](ErrorClass::Policy) budget violation; any
/// other decode failure (truncation, bad CRC, malformed chunk) is an
/// [`asset`](ErrorClass::Asset) malformed-input error.
fn map_decode_error(err: ImageError) -> Error {
    match err {
        ImageError::Limits(e) => limit_error(format!("PNG decode exceeded a codec limit: {e}"), ""),
        other => malformed_error(format!("PNG is malformed or truncated: {other}")),
    }
}

/// The `ExtendedColorType` to encode a layout as.
///
/// # Errors
/// Returns an [`export`](ErrorClass::Export) error for a layout outside the M0
/// 8-bit export set (`ChannelLayout` is `#[non_exhaustive]`).
fn extended_color_for(layout: ChannelLayout) -> Result<ExtendedColorType> {
    match layout {
        ChannelLayout::Gray => Ok(ExtendedColorType::L8),
        ChannelLayout::GrayA => Ok(ExtendedColorType::La8),
        ChannelLayout::Rgb => Ok(ExtendedColorType::Rgb8),
        ChannelLayout::Rgba => Ok(ExtendedColorType::Rgba8),
        other => Err(Error::new(
            ErrorClass::Export,
            "E_ENCODE_UNSUPPORTED_LAYOUT",
            format!("channel layout {other:?} is outside the M0 8-bit export set"),
        )),
    }
}

/// Quantize a normalized `f32` sample to a `u8`, rounding to nearest and clamping
/// out-of-range values into `[0, 255]`. Non-finite samples clamp to `0`.
fn quantize(sample: f32) -> u8 {
    if sample.is_nan() {
        return 0;
    }
    // `round` gives nearest-integer; the clamped, scaled value is in `[0, 255]`,
    // so the cast to `u8` is exact (provably no truncation or sign loss).
    let rounded = (sample.clamp(0.0, 1.0) * 255.0).round();
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "rounded is in [0.0, 255.0]; the cast is exact"
    )]
    let byte = rounded as u8;
    byte
}

/// Encode a [`ResourceValue`] as lossless PNG bytes.
///
/// The value's descriptor must be an [`ResourceDescriptor::Image`]; its channel
/// layout selects the PNG color type and its samples are quantized back to 8 bit.
/// Encoding is deterministic: a fixed [`CompressionType`] and [`FilterType`] are
/// used so identical inputs produce byte-identical files.
///
/// # Errors
/// - [`asset`](ErrorClass::Asset) / [`E_ENCODE_INPUT`] if the descriptor is not an
///   image.
/// - [`export`](ErrorClass::Export) / `E_ENCODE_FAILED` if the codec rejects the
///   buffer.
pub fn encode_png(value: &ResourceValue) -> Result<Vec<u8>> {
    let ResourceDescriptor::Image(image) = value.descriptor() else {
        return Err(Error::new(
            ErrorClass::Asset,
            E_ENCODE_INPUT,
            "io.encode_image requires an image resource".to_owned(),
        ));
    };
    let layout = image.layout;
    let extent = image.extent;

    let raw: Vec<u8> = value.samples().iter().map(|&s| quantize(s)).collect();

    let mut out = Vec::new();
    // A fixed compression/filter pair pins the byte output so encoding is
    // deterministic (the same image always yields the same file).
    let encoder =
        PngEncoder::new_with_quality(&mut out, CompressionType::Default, FilterType::Adaptive);
    encoder
        .write_image(
            &raw,
            extent.width,
            extent.height,
            extended_color_for(layout)?,
        )
        .map_err(|e| {
            Error::new(
                ErrorClass::Export,
                "E_ENCODE_FAILED",
                format!("PNG encode failed: {e}"),
            )
        })?;
    Ok(out)
}

/// Atomically write `bytes` to `path`.
///
/// Writes to a sibling temp file in the same directory, flushes it, then renames
/// it over `path`. A rename within a directory is atomic on POSIX, so a reader
/// never observes a partially written file.
///
/// # Errors
/// Returns an [`export`](ErrorClass::Export) / [`E_IO_FILESYSTEM`] error if any
/// filesystem step (create, write, sync, rename) fails.
pub fn atomic_write(path: &Path, bytes: &[u8]) -> Result<()> {
    use std::io::Write as _;

    let fs_error = |stage: &str, e: &std::io::Error| {
        Error::new(
            ErrorClass::Export,
            E_IO_FILESYSTEM,
            format!("failed to {stage} {}: {e}", path.display()),
        )
    };

    // The temp file is a sibling of the target (same directory), so the final
    // rename stays within one filesystem and is atomic.
    let tmp = temp_sibling(path);

    {
        let mut file = std::fs::File::create(&tmp).map_err(|e| fs_error("create", &e))?;
        file.write_all(bytes).map_err(|e| fs_error("write", &e))?;
        file.sync_all().map_err(|e| fs_error("sync", &e))?;
    }

    if let Err(e) = std::fs::rename(&tmp, path) {
        // Best-effort cleanup of the temp file on a failed rename.
        let _ = std::fs::remove_file(&tmp);
        return Err(fs_error("rename into place", &e));
    }
    Ok(())
}

/// Build the temp-file path the atomic write stages through: the target path with
/// a `.paintop-tmp` suffix appended, keeping it in the same directory so the
/// rename is atomic.
fn temp_sibling(path: &Path) -> PathBuf {
    let mut name = path.as_os_str().to_owned();
    name.push(".paintop-tmp");
    PathBuf::from(name)
}

/// Extract the required `path` string parameter from a node's resolved params.
fn path_param(params: &serde_json::Value) -> Result<PathBuf> {
    params
        .get("path")
        .and_then(serde_json::Value::as_str)
        .map(PathBuf::from)
        .ok_or_else(|| {
            Error::new(
                ErrorClass::Schema,
                E_IO_PATH_PARAM,
                "io operation requires a string `path` parameter".to_owned(),
            )
        })
}

// ---------------------------------------------------------------------------
// io.decode_image@1
// ---------------------------------------------------------------------------

/// The `io.decode_image@1` operation: PNG file → `Image`.
#[derive(Debug, Clone, Copy, Default)]
pub struct DecodeImage {
    /// The enforced decode budget.
    pub limits: DecodeLimits,
}

impl DecodeImage {
    /// A decoder with the default [`DecodeLimits`].
    #[must_use]
    pub const fn new() -> Self {
        Self {
            limits: DecodeLimits::DEFAULT,
        }
    }

    /// The declared manifest for `io.decode_image@1`.
    ///
    /// # Errors
    /// Propagates the [`schema`](ErrorClass::Schema) error if the hard-coded op /
    /// impl ids are somehow invalid (they are not).
    pub fn manifest() -> Result<OperationManifest> {
        Ok(OperationManifest {
            id: DECODE_OP_ID.parse()?,
            impl_version: 1,
            summary: "Decode a PNG file into an Image under enforced decode limits.".to_owned(),
            determinism: DeterminismTier::Exact,
            roi: RoiPolicy {
                category: RoiCategory::FullDomain,
                halo_px: None,
            },
            inputs: vec![],
            outputs: vec![image_port("image", "The decoded image.")],
            params: vec![ParamSpec {
                name: "path".to_owned(),
                ty: ParamType::String,
                unit: None,
                required: true,
                default: None,
                choices: vec![],
                doc: "Path to the PNG file to decode, under the input root.".to_owned(),
            }],
            implementations: vec![reference_impl()?],
            test: io_test_metadata(),
        })
    }
}

impl OpContract for DecodeImage {
    fn declared_inputs(&self) -> Vec<(String, ResourceKind)> {
        vec![]
    }

    fn declared_outputs(&self) -> Vec<(String, ResourceKind)> {
        vec![("image".to_owned(), ResourceKind::Image)]
    }

    fn infer_outputs(
        &self,
        _inputs: &Descriptors,
        _params: &serde_json::Value,
    ) -> Result<OutputDescriptors> {
        // The decoded extent and layout are known only at execution (they come
        // from the file header), so the descriptor is inferred as a placeholder
        // import-typed image; the executor produces the concrete value.
        let mut out = OutputDescriptors::new();
        out.insert(
            "image".to_owned(),
            ResourceDescriptor::Image(decoded_image_descriptor(
                Extent::new(0, 0),
                ChannelLayout::Rgba,
            )),
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
        _params: &serde_json::Value,
    ) -> Result<Vec<paintop_ir::AssertionResult>> {
        use paintop_ir::AssertionResult;
        let produced = outputs.contains_key("image");
        Ok(vec![if produced {
            AssertionResult::pass("decodes_to_image")
        } else {
            AssertionResult::fail("decodes_to_image", "no `image` output produced")
        }])
    }
}

impl OpImplementation for DecodeImage {
    fn compute(
        &self,
        _inputs: &InputValues,
        params: &serde_json::Value,
    ) -> std::result::Result<OutputValues, Error> {
        let path = path_param(params)?;
        let bytes = std::fs::read(&path).map_err(|e| {
            Error::new(
                ErrorClass::Asset,
                E_IO_FILESYSTEM,
                format!("failed to read {}: {e}", path.display()),
            )
        })?;
        let value = decode_png(&bytes, &self.limits)?;
        let mut out = OutputValues::new();
        out.insert("image".to_owned(), value);
        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// io.encode_image@1
// ---------------------------------------------------------------------------

/// The `io.encode_image@1` operation: `Image` → PNG file export.
///
/// Encode is a materialization barrier (`OP_CATALOG` §1): it writes the image to
/// disk and passes the same image through as its `image` output so downstream
/// consumers see the exact bytes that were written.
#[derive(Debug, Clone, Copy, Default)]
pub struct EncodeImage;

impl EncodeImage {
    /// Construct the encoder.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    /// The declared manifest for `io.encode_image@1`.
    ///
    /// # Errors
    /// Propagates the [`schema`](ErrorClass::Schema) error if the hard-coded op /
    /// impl ids are somehow invalid (they are not).
    pub fn manifest() -> Result<OperationManifest> {
        Ok(OperationManifest {
            id: ENCODE_OP_ID.parse()?,
            impl_version: 1,
            summary: "Encode an Image to a PNG file with an atomic, lossless write.".to_owned(),
            determinism: DeterminismTier::Exact,
            roi: RoiPolicy {
                category: RoiCategory::FullDomain,
                halo_px: None,
            },
            inputs: vec![InputSpec {
                name: "image".to_owned(),
                kind: ResourceKind::Image,
                required: true,
                doc: "The image to encode.".to_owned(),
            }],
            outputs: vec![image_port("image", "The encoded image, passed through.")],
            params: vec![ParamSpec {
                name: "path".to_owned(),
                ty: ParamType::String,
                unit: None,
                required: true,
                default: None,
                choices: vec![],
                doc: "Destination PNG path, under the output root.".to_owned(),
            }],
            implementations: vec![reference_impl()?],
            test: io_test_metadata(),
        })
    }
}

impl OpContract for EncodeImage {
    fn declared_inputs(&self) -> Vec<(String, ResourceKind)> {
        vec![("image".to_owned(), ResourceKind::Image)]
    }

    fn declared_outputs(&self) -> Vec<(String, ResourceKind)> {
        vec![("image".to_owned(), ResourceKind::Image)]
    }

    fn infer_outputs(
        &self,
        inputs: &Descriptors,
        _params: &serde_json::Value,
    ) -> Result<OutputDescriptors> {
        // Encode is a pass-through barrier: the output descriptor equals the
        // input image's descriptor exactly.
        let image = inputs.get("image").ok_or_else(|| {
            Error::new(
                ErrorClass::Reference,
                E_ENCODE_INPUT,
                "io.encode_image requires an `image` input".to_owned(),
            )
        })?;
        if !matches!(image, ResourceDescriptor::Image(_)) {
            return Err(Error::new(
                ErrorClass::Type,
                E_ENCODE_INPUT,
                "io.encode_image `image` input must be an image resource".to_owned(),
            ));
        }
        let mut out = OutputDescriptors::new();
        out.insert("image".to_owned(), *image);
        Ok(out)
    }

    fn required_inputs(
        &self,
        requested_outputs: &OutputRegions,
        _inputs: &Descriptors,
        _params: &serde_json::Value,
    ) -> Result<InputRegions> {
        // A pass-through barrier needs exactly the requested region; if none is
        // requested it needs nothing.
        let mut regions = InputRegions::new();
        if let Some(region) = requested_outputs.get("image") {
            regions.insert("image".to_owned(), *region);
        }
        Ok(regions)
    }

    fn validate_postconditions(
        &self,
        outputs: &OutputDescriptors,
        _params: &serde_json::Value,
    ) -> Result<Vec<paintop_ir::AssertionResult>> {
        use paintop_ir::AssertionResult;
        let passed = outputs.contains_key("image");
        Ok(vec![if passed {
            AssertionResult::pass("passes_image_through")
        } else {
            AssertionResult::fail("passes_image_through", "no `image` output produced")
        }])
    }
}

impl OpImplementation for EncodeImage {
    fn compute(
        &self,
        inputs: &InputValues,
        params: &serde_json::Value,
    ) -> std::result::Result<OutputValues, Error> {
        let path = path_param(params)?;
        let image = inputs.get("image").ok_or_else(|| {
            Error::new(
                ErrorClass::Reference,
                E_ENCODE_INPUT,
                "io.encode_image requires an `image` input value".to_owned(),
            )
        })?;
        let bytes = encode_png(image)?;
        atomic_write(&path, &bytes)?;
        let mut out = OutputValues::new();
        out.insert("image".to_owned(), image.clone());
        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// Shared manifest helpers
// ---------------------------------------------------------------------------

/// A standard image output port.
fn image_port(name: &str, doc: &str) -> OutputSpec {
    OutputSpec {
        name: name.to_owned(),
        kind: ResourceKind::Image,
        doc: doc.to_owned(),
    }
}

/// The mandatory `cpu.reference@1` oracle implementation id.
fn reference_impl() -> Result<ImplId> {
    ImplId::new("cpu", "reference", 1)
}

/// The verification declarations shared by both io ops: every applicable
/// category is covered (round-trip property, malformed-PNG fuzz seeds,
/// metadata-preservation goldens, differential-vs-`image` decode).
fn io_test_metadata() -> TestMetadata {
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
