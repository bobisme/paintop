//! The `paint.fill@1` operation: a typed scalar/color constant painted into an
//! image through a coverage mask (`OP_CATALOG` §6).
//!
//! `paint.fill` writes a per-channel constant `value` into an `image` wherever a
//! coverage `mask` selects it, blending toward the existing sample everywhere
//! else:
//!
//! ```text
//! out = base + m · (value − base)   (per channel)
//! ```
//!
//! It is the masked, constant-source twin of
//! [`composite.masked_replace`](crate::composite): where `masked_replace` blends
//! a whole *edited image* over a base, `paint.fill` blends a single typed
//! **constant** (one component per channel) over the base. Where the mask is `0`
//! the output is the base bit-for-bit (the no-change-outside-mask invariant); where
//! it is `1` the output is exactly the fill `value`.
//!
//! # Typed value and the valid-range policy
//!
//! The `value` is a per-channel array whose length must equal the image's channel
//! count, so a scalar (single-channel) field takes a one-element value and a color
//! image takes one component per color/alpha channel. Each component must respect
//! the image's declared valid-range policy exactly as `image.create` does
//! (`plan.md` §8.3, clamping is never implicit): a
//! [`display-referred`](ColorRange::DisplayReferred) color channel is bounded to
//! `[0, 1]` and an out-of-range fill is **rejected**; a
//! [`scene-referred`](ColorRange::SceneReferred) color channel need only be finite.
//! The alpha channel, when present, is always coverage in `[0, 1]`.
//!
//! # Determinism
//!
//! The blend is `base + m · (value − base)`, a single per-channel `f32`
//! multiply-add with the `m ∈ {0, 1}` extremes returned verbatim, so the op is
//! [`Exact`](DeterminismTier::Exact): bit-identical to its reference on every run
//! and machine.
//!
//! # Geometry
//!
//! The op is **pointwise**: every output sample depends only on the co-located
//! input sample and mask coverage. The `mask` must share the image's extent; a
//! mismatch is a [`semantic`](ErrorClass::Semantic) error.

use paintop_core::executor::{InputValues, OpImplementation, OutputValues, ResourceValue};
use paintop_ir::{
    AssertionResult, ColorRange, Descriptors, DeterminismTier, Error, ErrorClass, ErrorContext,
    Extent, ImageDescriptor, ImplId, InputRegions, InputSpec, OpContract, OperationManifest,
    OutputDescriptors, OutputRegions, OutputSpec, ParamSpec, ParamType, ResourceDescriptor,
    ResourceKind, Result, RoiCategory, RoiPolicy, TestMetadata,
};

/// The canonical id of the masked-fill operation.
pub const FILL_OP_ID: &str = "paint.fill@1";

/// A required input port (`image` or `mask`) was absent or carried the wrong
/// resource kind.
pub const E_FILL_INPUT: &str = "E_FILL_INPUT";

/// The `value` array was the wrong length, non-finite, or violated the image's
/// declared valid-range policy.
pub const E_FILL_VALUE: &str = "E_FILL_VALUE";

/// The `mask` extent disagrees with the `image` extent, or the op produced a
/// sample buffer whose length disagrees with its descriptor.
pub const E_FILL_SHAPE: &str = "E_FILL_SHAPE";

/// Parse and range-check the per-channel `value` array against the image's channel
/// layout and range policy.
///
/// The check mirrors `image.create`'s fill policy exactly: every component must be
/// finite; an alpha component, and every color component of a
/// [`display-referred`](ColorRange::DisplayReferred) image, must additionally lie
/// in `[0, 1]` (clamping is never implicit).
fn parse_value(params: &serde_json::Value, descriptor: &ImageDescriptor) -> Result<Vec<f32>> {
    let value = params.get("value").ok_or_else(|| {
        Error::new(
            ErrorClass::Schema,
            E_FILL_VALUE,
            "paint.fill requires a per-channel `value` array".to_owned(),
        )
    })?;
    let array = value.as_array().ok_or_else(|| {
        Error::new(
            ErrorClass::Schema,
            E_FILL_VALUE,
            "paint.fill `value` must be an array of per-channel components".to_owned(),
        )
        .with_context(ErrorContext::default().with_actual(value.to_string()))
    })?;
    let channels = descriptor.layout.channel_count() as usize;
    if array.len() != channels {
        return Err(Error::new(
            ErrorClass::Schema,
            E_FILL_VALUE,
            format!(
                "paint.fill `value` has {} components but the {:?} layout has {channels} channels",
                array.len(),
                descriptor.layout,
            ),
        )
        .with_context(
            ErrorContext::default()
                .with_actual(array.len().to_string())
                .with_expected(channels.to_string()),
        ));
    }

    let has_alpha = descriptor.layout.has_alpha();
    let mut fill = Vec::with_capacity(channels);
    for (index, component) in array.iter().enumerate() {
        let n = component.as_f64().ok_or_else(|| {
            Error::new(
                ErrorClass::Schema,
                E_FILL_VALUE,
                format!("paint.fill `value[{index}]` must be a number"),
            )
            .with_context(ErrorContext::default().with_actual(component.to_string()))
        })?;
        if !n.is_finite() {
            return Err(Error::new(
                ErrorClass::Schema,
                E_FILL_VALUE,
                format!("paint.fill `value[{index}]` must be finite, got {n}"),
            ));
        }
        // The alpha channel is coverage in [0, 1] regardless of the color range;
        // color channels follow the range policy. A display-referred color channel
        // is bounded to [0, 1]; a scene-referred one need only be finite.
        let is_alpha = has_alpha && index == channels - 1;
        let bounded = is_alpha || matches!(descriptor.range, ColorRange::DisplayReferred);
        if bounded && !(0.0..=1.0).contains(&n) {
            let what = if is_alpha {
                "alpha coverage"
            } else {
                "display-referred color"
            };
            return Err(Error::new(
                ErrorClass::Policy,
                E_FILL_VALUE,
                format!(
                    "paint.fill `value[{index}]` = {n} is out of the {what} range [0, 1]; \
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
            reason = "value stored as the image's f32 sample type"
        )]
        fill.push(n as f32);
    }
    Ok(fill)
}

/// Paint the per-channel `fill` constant into `base` through `mask`, per pixel and
/// channel.
///
/// `channels` is the interleaved sample count per pixel; the mask is one coverage
/// sample per pixel and modulates every channel of that pixel identically. The
/// blend is `out = base + m·(fill − base)` with the `m ∈ {0, 1}` extremes returned
/// bit-exactly (the no-change-outside-mask guarantee).
fn paint(base: &[f32], mask: &[f32], fill: &[f32], channels: u32) -> Vec<f32> {
    let channel_count = channels as usize;
    if channel_count == 0 {
        return base.to_vec();
    }
    let mut out = Vec::with_capacity(base.len());
    for (base_pixel, &coverage) in base.chunks_exact(channel_count).zip(mask.iter()) {
        for (&b, &f) in base_pixel.iter().zip(fill.iter()) {
            // Bit-exact extremes: m == 0 keeps `base`, m == 1 takes the fill,
            // untouched by arithmetic — the safety-critical locality invariant.
            // Compared by bit pattern so the exact constants are matched without a
            // tolerance (`to_bits` is the clippy-clean form of an exact `f32` eq).
            let sample = if coverage.to_bits() == 0.0_f32.to_bits() {
                b
            } else if coverage.to_bits() == 1.0_f32.to_bits() {
                f
            } else {
                // base + m·(fill − base): a single deterministic f32 multiply-add.
                coverage.mul_add(f - b, b)
            };
            out.push(sample);
        }
    }
    out
}

/// Read the required `image` port's descriptor.
fn image_descriptor(inputs: &Descriptors) -> Result<&ImageDescriptor> {
    let resource = inputs.get("image").ok_or_else(|| {
        Error::new(
            ErrorClass::Reference,
            E_FILL_INPUT,
            "paint.fill requires an `image` input".to_owned(),
        )
    })?;
    let ResourceDescriptor::Image(descriptor) = resource else {
        return Err(Error::new(
            ErrorClass::Type,
            E_FILL_INPUT,
            "paint.fill `image` input must be an image resource".to_owned(),
        ));
    };
    Ok(descriptor)
}

/// Read the `mask` port's extent.
fn mask_extent(inputs: &Descriptors) -> Result<Extent> {
    let resource = inputs.get("mask").ok_or_else(|| {
        Error::new(
            ErrorClass::Reference,
            E_FILL_INPUT,
            "paint.fill requires a `mask` input".to_owned(),
        )
    })?;
    let ResourceDescriptor::Mask(mask) = resource else {
        return Err(Error::new(
            ErrorClass::Type,
            E_FILL_INPUT,
            "paint.fill `mask` input must be a mask resource".to_owned(),
        ));
    };
    Ok(mask.extent)
}

/// Require the `mask` extent to match the `image` extent.
fn check_extent(image: Extent, mask: Extent) -> Result<()> {
    if image == mask {
        return Ok(());
    }
    Err(Error::new(
        ErrorClass::Semantic,
        E_FILL_SHAPE,
        "paint.fill: the `mask` must share the `image` extent".to_owned(),
    )
    .with_context(ErrorContext::default().with_actual(format!("mask {mask:?} vs image {image:?}"))))
}

/// The `paint.fill@1` operation: an `image` + `mask` + typed `value` → the
/// masked-filled `image`.
#[derive(Debug, Clone, Copy, Default)]
pub struct Fill;

impl Fill {
    /// Construct the masked-fill operation.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    /// The declared manifest for `paint.fill@1`.
    ///
    /// # Errors
    /// Propagates the [`schema`](ErrorClass::Schema) error if the hard-coded op /
    /// impl ids are somehow invalid (they are not).
    pub fn manifest() -> Result<OperationManifest> {
        Ok(OperationManifest {
            id: FILL_OP_ID.parse()?,
            impl_version: 1,
            summary: "Paint a typed per-channel constant value into an image through a coverage \
                      mask (out = base + m·(value − base)); where the mask is 0 the output is the \
                      base bit-for-bit, where it is 1 the output is the value. The value is \
                      range-checked against the image's valid-range policy."
                .to_owned(),
            // base + m·(value − base) is a per-channel f32 multiply-add with the
            // m ∈ {0, 1} extremes returned verbatim: bit-exact for the scalar
            // reference.
            determinism: DeterminismTier::Exact,
            roi: RoiPolicy {
                category: RoiCategory::Pointwise,
                halo_px: None,
            },
            inputs: vec![
                InputSpec {
                    name: "image".to_owned(),
                    kind: ResourceKind::Image,
                    required: true,
                    doc: "The image painted into; kept where the mask is 0.".to_owned(),
                },
                InputSpec {
                    name: "mask".to_owned(),
                    kind: ResourceKind::Mask,
                    required: true,
                    doc:
                        "The coverage mask in [0, 1] selecting the fill, co-located with the image."
                            .to_owned(),
                },
            ],
            outputs: vec![OutputSpec {
                name: "image".to_owned(),
                kind: ResourceKind::Image,
                doc: "The masked-filled image (same descriptor as the input image).".to_owned(),
            }],
            params: vec![ParamSpec {
                name: "value".to_owned(),
                ty: ParamType::Json,
                unit: None,
                required: true,
                default: None,
                doc: "The per-channel constant fill value; one component per layout channel, each \
                      respecting the image's declared valid range."
                    .to_owned(),
                choices: vec![],
            }],
            implementations: vec![reference_impl()?],
            test: fill_test_metadata(),
        })
    }
}

impl OpContract for Fill {
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
        let image = image_descriptor(inputs)?;
        let extent = mask_extent(inputs)?;
        check_extent(image.extent, extent)?;
        // Validate the typed value at infer time so a bad fill fails type-checking
        // before any pixels are touched.
        parse_value(params, image)?;

        let mut out = OutputDescriptors::new();
        out.insert("image".to_owned(), ResourceDescriptor::Image(*image));
        Ok(out)
    }

    fn required_inputs(
        &self,
        requested_outputs: &OutputRegions,
        _inputs: &Descriptors,
        _params: &serde_json::Value,
    ) -> Result<InputRegions> {
        // Pointwise: each output pixel reads the co-located image and mask sample,
        // so both ports share the requested output region.
        let mut regions = InputRegions::new();
        if let Some(region) = requested_outputs.get("image") {
            for port in ["image", "mask"] {
                regions.insert(port.to_owned(), *region);
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
        let mut results = vec![AssertionResult::pass("produces_image")];
        // The typed value must be valid for the output image's layout/range, so a
        // postcondition pass guarantees the painted constant was in range.
        results.push(match parse_value(params, image) {
            Ok(_) => AssertionResult::pass("value_in_range"),
            Err(e) => AssertionResult::fail("value_in_range", e.message),
        });
        Ok(results)
    }
}

impl OpImplementation for Fill {
    fn compute(
        &self,
        inputs: &InputValues,
        params: &serde_json::Value,
    ) -> std::result::Result<OutputValues, Error> {
        let image = inputs.get("image").ok_or_else(|| {
            Error::new(
                ErrorClass::Reference,
                E_FILL_INPUT,
                "paint.fill requires an `image` input value".to_owned(),
            )
        })?;
        let mask = inputs.get("mask").ok_or_else(|| {
            Error::new(
                ErrorClass::Reference,
                E_FILL_INPUT,
                "paint.fill requires a `mask` input value".to_owned(),
            )
        })?;

        let ResourceDescriptor::Image(descriptor) = image.descriptor() else {
            return Err(Error::new(
                ErrorClass::Type,
                E_FILL_INPUT,
                "paint.fill `image` input must be an image resource".to_owned(),
            ));
        };
        let ResourceDescriptor::Mask(mask_descriptor) = mask.descriptor() else {
            return Err(Error::new(
                ErrorClass::Type,
                E_FILL_INPUT,
                "paint.fill `mask` input must be a mask resource".to_owned(),
            ));
        };

        check_extent(descriptor.extent, mask_descriptor.extent)?;
        let fill = parse_value(params, descriptor)?;
        let samples = paint(image.samples(), mask.samples(), &fill, image.channels());

        let value = ResourceValue::new(
            ResourceDescriptor::Image(*descriptor),
            image.channels(),
            samples,
        )
        .map_err(|actual| {
            Error::new(
                ErrorClass::Execution,
                E_FILL_SHAPE,
                format!("paint.fill produced a sample buffer of unexpected length {actual}"),
            )
        })?;

        let mut out = OutputValues::new();
        out.insert("image".to_owned(), value);
        Ok(out)
    }
}

/// The mandatory `cpu.reference@1` oracle implementation id.
fn reference_impl() -> Result<ImplId> {
    ImplId::new("cpu", "reference", 1)
}

/// The verification declarations for `paint.fill@1`: an exact, single-reference,
/// pointwise masked blend of a typed constant. Differential does not apply (one
/// implementation). Perceptual is not applicable: the fill is a bit-exact
/// per-channel multiply-add, pinned by mask-extreme and lerp fixtures, with no
/// perceptual-quality metric. Every other applicable category is covered by the
/// mask-extreme, outside-region bit-identity, soft-mask lerp, type-correctness and
/// range-rejection tests in this module.
fn fill_test_metadata() -> TestMetadata {
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
            "paint.fill is a bit-exact per-channel masked blend of a typed constant verified by \
             mask-extreme exactness, outside-region bit-identity, and soft-mask lerp fixtures; \
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
