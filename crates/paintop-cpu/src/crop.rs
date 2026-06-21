//! The `image.crop@1` operation: an exact pixel selection on half-open bounds
//! (`OP_CATALOG` §5, `IR_SPEC` §8.1).
//!
//! `image.crop` selects the sub-rectangle `[x0, x1) × [y0, y1)` of an input image
//! and returns it verbatim as a new, smaller image. The rect is **half-open** in
//! the input's `PixelCenterUpperLeft` pixel space: `Rect::new(0, 0, w, h)` copies
//! exactly the `w × h` pixels with top-left corner `(0, 0)`. The output extent is
//! the rect's `width × height` and every other descriptor field (layout, color,
//! range, alpha, semantic) is preserved.
//!
//! # Bounds policy
//!
//! The crop rect must be well-formed (`x1 >= x0`, `y1 >= y0`) and lie **fully
//! within** the input extent (`0 <= x0`, `x1 <= width`, likewise for `y`). A rect
//! that escapes the input is rejected rather than silently clamped or
//! zero-extended: cropping never invents samples. An empty rect (zero width or
//! height) is accepted and yields a `0`-area image, the identity element of the
//! crop/pad calculus.
//!
//! # Determinism
//!
//! The op is [`Exact`](DeterminismTier::Exact): the output samples are a verbatim
//! copy of a contiguous sub-window of the input, bit-identical on every run.

use paintop_core::executor::{InputValues, OpImplementation, OutputValues, ResourceValue};
use paintop_ir::{
    AssertionResult, Descriptors, DeterminismTier, Error, ErrorClass, ErrorContext,
    ImageDescriptor, ImplId, InputRegions, InputSpec, OpContract, OperationManifest,
    OutputDescriptors, OutputRegions, OutputSpec, ParamSpec, ParamType, ParamUnit, Rect,
    ResourceDescriptor, ResourceKind, Result, RoiCategory, RoiPolicy, TestMetadata,
};

/// The canonical id of the image-crop operation.
pub const CROP_OP_ID: &str = "image.crop@1";

/// The `image` input was absent or carried a non-image descriptor.
pub const E_CROP_INPUT: &str = "E_CROP_INPUT";

/// The `rect` parameter was missing, malformed, or not a well-formed half-open
/// rectangle.
pub const E_CROP_RECT: &str = "E_CROP_RECT";

/// The crop rect escaped the input image extent (cropping never invents samples).
pub const E_CROP_BOUNDS: &str = "E_CROP_BOUNDS";

/// Parse the required `rect` param into a well-formed half-open [`Rect`].
///
/// # Errors
/// [`schema`](ErrorClass::Schema) / [`E_CROP_RECT`] if the param is absent, not an
/// object with integer `x0,y0,x1,y1`, or ill-formed (`x1 < x0` or `y1 < y0`).
fn rect_param(params: &serde_json::Value) -> Result<Rect> {
    let value = params.get("rect").ok_or_else(|| {
        Error::new(
            ErrorClass::Schema,
            E_CROP_RECT,
            "image.crop requires a `rect` parameter (x0, y0, x1, y1)".to_owned(),
        )
    })?;
    let rect: Rect = serde_json::from_value(value.clone()).map_err(|e| {
        Error::new(
            ErrorClass::Schema,
            E_CROP_RECT,
            format!("image.crop `rect` is not a half-open rectangle: {e}"),
        )
        .with_context(ErrorContext::default().with_actual(value.to_string()))
    })?;
    if !rect.is_valid() {
        return Err(Error::new(
            ErrorClass::Schema,
            E_CROP_RECT,
            format!(
                "image.crop `rect` is ill-formed: x1 ({}) >= x0 ({}) and y1 ({}) >= y0 ({}) required",
                rect.x1, rect.x0, rect.y1, rect.y0
            ),
        ));
    }
    Ok(rect)
}

/// Verify the crop rect lies fully inside the `extent` and return it.
///
/// # Errors
/// [`semantic`](ErrorClass::Semantic) / [`E_CROP_BOUNDS`] if any edge escapes the
/// input bounds.
fn check_bounds(rect: Rect, extent: paintop_ir::Extent) -> Result<()> {
    let w = i64::from(extent.width);
    let h = i64::from(extent.height);
    if rect.x0 < 0 || rect.y0 < 0 || rect.x1 > w || rect.y1 > h {
        return Err(Error::new(
            ErrorClass::Semantic,
            E_CROP_BOUNDS,
            "image.crop `rect` must lie fully within the input extent; \
             cropping never invents samples"
                .to_owned(),
        )
        .with_context(
            ErrorContext::default()
                .with_actual(format!(
                    "[{}, {}) x [{}, {})",
                    rect.x0, rect.x1, rect.y0, rect.y1
                ))
                .with_expected(format!("within {}x{}", extent.width, extent.height)),
        ));
    }
    Ok(())
}

/// Copy the half-open sub-window `rect` out of the interleaved `samples`.
///
/// `samples` is row-major, channel-interleaved over the `src` extent; `rect` is
/// assumed already bounds-checked. Returns the interleaved samples of the cropped
/// `rect.width() × rect.height()` image.
fn crop_samples(samples: &[f32], src: paintop_ir::Extent, rect: Rect, channels: u32) -> Vec<f32> {
    let stride = channels as usize;
    let src_w = src.width as usize;
    // rect is bounds-checked to lie within the non-negative extent, so each edge
    // fits a usize; the fallback never triggers for a valid request.
    let x0 = usize::try_from(rect.x0).unwrap_or(0);
    let y0 = usize::try_from(rect.y0).unwrap_or(0);
    let out_w = usize::try_from(rect.width()).unwrap_or(0);
    let out_h = usize::try_from(rect.height()).unwrap_or(0);
    let mut out = Vec::with_capacity(out_w.saturating_mul(out_h).saturating_mul(stride));
    for row in 0..out_h {
        let src_row = y0 + row;
        let start = (src_row * src_w + x0) * stride;
        let end = start + out_w * stride;
        out.extend_from_slice(&samples[start..end]);
    }
    out
}

/// The `image.crop@1` operation: an image + a half-open rect → the cropped image.
#[derive(Debug, Clone, Copy, Default)]
pub struct Crop;

impl Crop {
    /// Construct the image-crop operation.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    /// The declared manifest for `image.crop@1`.
    ///
    /// # Errors
    /// Propagates the [`schema`](ErrorClass::Schema) error if the hard-coded op /
    /// impl ids are somehow invalid (they are not).
    pub fn manifest() -> Result<OperationManifest> {
        Ok(OperationManifest {
            id: CROP_OP_ID.parse()?,
            impl_version: 1,
            summary:
                "Exact pixel selection of a half-open rectangle [x0, x1) x [y0, y1) within the \
                      input image; the rect must lie fully inside the input extent."
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
                doc: "The image to crop.".to_owned(),
            }],
            outputs: vec![OutputSpec {
                name: "image".to_owned(),
                kind: ResourceKind::Image,
                doc: "The cropped image (extent = rect width x height; all other descriptor \
                      fields preserved)."
                    .to_owned(),
            }],
            params: vec![ParamSpec {
                name: "rect".to_owned(),
                ty: ParamType::Json,
                unit: Some(ParamUnit::Pixels),
                required: true,
                default: None,
                choices: vec![],
                doc: "The half-open crop rectangle {x0, y0, x1, y1} in input pixel space; upper \
                      bounds are exclusive and must lie within the input extent."
                    .to_owned(),
            }],
            implementations: vec![reference_impl()?],
            test: crop_test_metadata(),
        })
    }
}

impl OpContract for Crop {
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
        let rect = rect_param(params)?;
        check_bounds(rect, descriptor.extent)?;

        // The cropped image shares every descriptor field but its extent.
        let mut out_desc = *descriptor;
        #[allow(
            clippy::cast_sign_loss,
            clippy::cast_possible_truncation,
            reason = "rect is bounds-checked within the u32 extent, so width/height fit u32"
        )]
        {
            out_desc.extent = paintop_ir::Extent::new(rect.width() as u32, rect.height() as u32);
        }
        let mut out = OutputDescriptors::new();
        out.insert("image".to_owned(), ResourceDescriptor::Image(out_desc));
        Ok(out)
    }

    fn required_inputs(
        &self,
        requested_outputs: &OutputRegions,
        _inputs: &Descriptors,
        params: &serde_json::Value,
    ) -> Result<InputRegions> {
        // Geometric: an output region R (in cropped-image space) maps back to the
        // input region R translated by the crop origin (x0, y0).
        let rect = rect_param(params)?;
        let mut regions = InputRegions::new();
        if let Some(region) = requested_outputs.get("image") {
            let shifted = Rect::new(
                region.x0 + rect.x0,
                region.y0 + rect.y0,
                region.x1 + rect.x0,
                region.y1 + rect.y0,
            );
            regions.insert("image".to_owned(), shifted);
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

impl OpImplementation for Crop {
    fn compute(
        &self,
        inputs: &InputValues,
        params: &serde_json::Value,
    ) -> std::result::Result<OutputValues, Error> {
        let image = inputs.get("image").ok_or_else(|| {
            Error::new(
                ErrorClass::Reference,
                E_CROP_INPUT,
                "image.crop requires an `image` input value".to_owned(),
            )
        })?;
        let ResourceDescriptor::Image(descriptor) = image.descriptor() else {
            return Err(Error::new(
                ErrorClass::Type,
                E_CROP_INPUT,
                "image.crop `image` input must be an image resource".to_owned(),
            ));
        };
        let rect = rect_param(params)?;
        check_bounds(rect, descriptor.extent)?;

        let samples = crop_samples(image.samples(), descriptor.extent, rect, image.channels());

        let mut out_desc = *descriptor;
        #[allow(
            clippy::cast_sign_loss,
            clippy::cast_possible_truncation,
            reason = "rect is bounds-checked within the u32 extent"
        )]
        {
            out_desc.extent = paintop_ir::Extent::new(rect.width() as u32, rect.height() as u32);
        }

        let value = ResourceValue::new(
            ResourceDescriptor::Image(out_desc),
            image.channels(),
            samples,
        )
        .map_err(|actual| {
            Error::new(
                ErrorClass::Execution,
                E_CROP_INPUT,
                format!("image.crop produced a sample buffer of unexpected length {actual}"),
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
            E_CROP_INPUT,
            "image.crop requires an `image` input".to_owned(),
        )
    })?;
    let ResourceDescriptor::Image(descriptor) = image else {
        return Err(Error::new(
            ErrorClass::Type,
            E_CROP_INPUT,
            "image.crop `image` input must be an image resource".to_owned(),
        ));
    };
    Ok(descriptor)
}

/// The mandatory `cpu.reference@1` oracle implementation id.
fn reference_impl() -> Result<ImplId> {
    ImplId::new("cpu", "reference", 1)
}

/// Verification declarations for `image.crop@1`: an exact, single-reference,
/// geometric selection. Differential does not apply (one implementation).
/// Perceptual is not applicable: a crop is a verbatim sub-window copy verified by
/// exact pixel-selection fixtures and the crop/pad round-trip, not a perceptual
/// metric.
fn crop_test_metadata() -> TestMetadata {
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
            "image.crop is an exact verbatim sub-window copy verified by pixel-selection fixtures, \
             half-open boundary cases, and the crop/pad round-trip identity; there is no \
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
