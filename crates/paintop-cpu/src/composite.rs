//! The `composite.masked_replace@1` operation: the **single MVP authorization
//! boundary** (`OP_CATALOG` §7, `IR_SPEC` §9.1, `M0_DECISIONS` D2).
//!
//! `composite.masked_replace` composites an `edited` image over a `base` image
//! through a coverage `mask`, in premultiplied linear light:
//!
//! ```text
//! out = m · edited + (1 − m) · base
//! ```
//!
//! This is the *one* place in the MVP where an edited layer is allowed to replace
//! the original (`M0_DECISIONS` D2): every other op paints onto an edit layer, and
//! locality against the original is enforced **here** by the mask. It is exactly
//! the operation `assert.no_change_outside_mask@1` checks against — wherever the
//! mask is `0`, the output is the `base`, bit-for-bit (`IR_SPEC` §9.1).
//!
//! # Exactness at the mask extremes
//!
//! The blend is evaluated as `out = base + m · (edited − base)`, the
//! fused-multiply-add form, with two **bit-exact** boundary guarantees:
//!
//! - where `m == 0`, the output sample is the `base` sample, returned verbatim
//!   (no arithmetic touches it) — this is the safety-critical
//!   no-change-outside-mask invariant;
//! - where `m == 1`, the output sample is the `edited` sample, returned verbatim.
//!
//! Between the extremes the blend is a single per-channel multiply-add on the
//! declared `f32` scalar, a deterministic, bit-exact function of the inputs — so
//! the op is [`Exact`](DeterminismTier::Exact), not merely bounded.
//!
//! # Premultiplied-correct compositing
//!
//! The blend is linear in the premultiplied channels, so it is only correct on
//! premultiplied linear color. Both `edited` and `base` must therefore be linear
//! (`raw-linear` / `linear-srgb`), premultiplied, and carry an alpha channel; a
//! nonlinear (`srgb`) input, a straight-alpha input, or an image without alpha is
//! rejected with a [`semantic`](ErrorClass::Semantic) error rather than producing
//! wrong color. The `mask` is co-located coverage in `[0, 1]`.
//!
//! # Geometry
//!
//! The op is **pointwise**: every output sample depends only on the co-located
//! `edited`, `base`, and `mask` samples. `edited` and `base` must share extent and
//! channel layout, and the `mask` must share their extent; a mismatch is a
//! [`semantic`](ErrorClass::Semantic) error.

use paintop_core::executor::{InputValues, OpImplementation, OutputValues, ResourceValue};
use paintop_ir::{
    AlphaRepresentation, AssertionResult, ColorEncoding, Descriptors, DeterminismTier, Error,
    ErrorClass, ErrorContext, Extent, ImageDescriptor, ImplId, InputRegions, InputSpec, OpContract,
    OperationManifest, OutputDescriptors, OutputRegions, OutputSpec, ResourceDescriptor,
    ResourceKind, Result, RoiCategory, RoiPolicy, TestMetadata,
};

/// The canonical id of the masked-replace authorization op.
pub const MASKED_REPLACE_OP_ID: &str = "composite.masked_replace@1";

/// A required input port (`edited`, `base`, or `mask`) was absent or carried the
/// wrong resource kind.
pub const E_MASKED_REPLACE_INPUT: &str = "E_MASKED_REPLACE_INPUT";

/// The `edited` / `base` images are in a representation this op cannot composite
/// (nonlinear encoding, straight alpha, or no alpha channel), or the three ports
/// disagree on extent / layout.
pub const E_MASKED_REPLACE_SHAPE: &str = "E_MASKED_REPLACE_SHAPE";

/// The op produced a sample buffer whose length disagrees with its descriptor.
pub const E_MASKED_REPLACE_BUFFER: &str = "E_MASKED_REPLACE_BUFFER";

/// Validate that an `edited` / `base` image may be composited in premultiplied
/// linear light: linear-encoded, premultiplied, with an alpha channel.
fn check_color_image(descriptor: &ImageDescriptor, port: &str) -> Result<()> {
    if descriptor.color == ColorEncoding::Srgb {
        return Err(Error::new(
            ErrorClass::Semantic,
            E_MASKED_REPLACE_SHAPE,
            format!(
                "composite.masked_replace requires linear-light color; the `{port}` input is \
                 `srgb`-encoded. Insert a color.convert to linear-srgb first."
            ),
        )
        .with_context(
            ErrorContext::default()
                .with_actual("srgb")
                .with_expected("linear-srgb | raw-linear"),
        ));
    }
    if !descriptor.layout.has_alpha() {
        return Err(Error::new(
            ErrorClass::Semantic,
            E_MASKED_REPLACE_SHAPE,
            format!(
                "composite.masked_replace requires the `{port}` image to have an alpha channel \
                 (GrayA or Rgba)"
            ),
        ));
    }
    if descriptor.alpha != AlphaRepresentation::Premultiplied {
        return Err(Error::new(
            ErrorClass::Semantic,
            E_MASKED_REPLACE_SHAPE,
            format!(
                "composite.masked_replace composites in premultiplied space; premultiply the \
                 `{port}` image first"
            ),
        )
        .with_context(
            ErrorContext::default()
                .with_actual(format!("{:?}", descriptor.alpha))
                .with_expected("Premultiplied"),
        ));
    }
    Ok(())
}

/// Validate the three ports together and return the output image descriptor.
///
/// `edited` and `base` must agree on extent and channel layout (so the blend is
/// channel-for-channel), and the `mask` must share their extent. The output keeps
/// the `base` descriptor exactly: compositing onto `base` never changes its type.
///
/// The check is shared by `infer_outputs` and the kernel so a graph and a run
/// never disagree on what is rejected.
fn check_and_retarget(
    edited: &ImageDescriptor,
    base: &ImageDescriptor,
    mask_extent: Extent,
) -> Result<ImageDescriptor> {
    check_color_image(edited, "edited")?;
    check_color_image(base, "base")?;

    if edited.extent != base.extent {
        return Err(shape_mismatch(
            "the `edited` and `base` images must share an extent",
            format!("edited {:?} vs base {:?}", edited.extent, base.extent),
        ));
    }
    if edited.layout != base.layout {
        return Err(shape_mismatch(
            "the `edited` and `base` images must share a channel layout",
            format!("edited {:?} vs base {:?}", edited.layout, base.layout),
        ));
    }
    if mask_extent != base.extent {
        return Err(shape_mismatch(
            "the `mask` must share the images' extent",
            format!("mask {mask_extent:?} vs image {:?}", base.extent),
        ));
    }
    Ok(*base)
}

/// Build a shape-mismatch [`semantic`](ErrorClass::Semantic) error.
fn shape_mismatch(detail: &str, actual: String) -> Error {
    Error::new(
        ErrorClass::Semantic,
        E_MASKED_REPLACE_SHAPE,
        format!("composite.masked_replace: {detail}"),
    )
    .with_context(ErrorContext::default().with_actual(actual))
}

/// Composite `edited` over `base` through `mask`, per pixel and channel.
///
/// `channels` is the interleaved color+alpha sample count per pixel; the mask is
/// one coverage sample per pixel and modulates every channel of that pixel
/// identically. The blend is `out = base + m·(edited − base)` with the `m ∈ {0, 1}`
/// extremes returned bit-exactly (the no-change-outside-mask guarantee).
fn blend(edited: &[f32], base: &[f32], mask: &[f32], channels: u32) -> Vec<f32> {
    let channel_count = channels as usize;
    if channel_count == 0 {
        return base.to_vec();
    }
    let mut out = Vec::with_capacity(base.len());
    let base_pixels = base.chunks_exact(channel_count);
    let edited_pixels = edited.chunks_exact(channel_count);
    for ((base_pixel, edited_pixel), &coverage) in base_pixels.zip(edited_pixels).zip(mask.iter()) {
        for (&b, &e) in base_pixel.iter().zip(edited_pixel.iter()) {
            // Bit-exact extremes: m == 0 keeps `base`, m == 1 takes `edited`,
            // untouched by arithmetic — the safety-critical locality invariant.
            // Compared by bit pattern so the exact constants are matched without a
            // tolerance (`to_bits` is the clippy-clean form of an exact `f32` eq).
            let sample = if coverage.to_bits() == 0.0_f32.to_bits() {
                b
            } else if coverage.to_bits() == 1.0_f32.to_bits() {
                e
            } else {
                // m · edited + (1 − m) · base, as the fused multiply-add form
                // base + m·(edited − base): a single deterministic f32 op.
                coverage.mul_add(e - b, b)
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
            E_MASKED_REPLACE_INPUT,
            format!("composite.masked_replace requires a `{port}` input"),
        )
    })?;
    let ResourceDescriptor::Image(descriptor) = resource else {
        return Err(Error::new(
            ErrorClass::Type,
            E_MASKED_REPLACE_INPUT,
            format!("composite.masked_replace `{port}` input must be an image resource"),
        ));
    };
    Ok(descriptor)
}

/// Read the `mask` port's extent.
fn mask_extent(inputs: &Descriptors) -> Result<Extent> {
    let resource = inputs.get("mask").ok_or_else(|| {
        Error::new(
            ErrorClass::Reference,
            E_MASKED_REPLACE_INPUT,
            "composite.masked_replace requires a `mask` input".to_owned(),
        )
    })?;
    let ResourceDescriptor::Mask(mask) = resource else {
        return Err(Error::new(
            ErrorClass::Type,
            E_MASKED_REPLACE_INPUT,
            "composite.masked_replace `mask` input must be a mask resource".to_owned(),
        ));
    };
    Ok(mask.extent)
}

/// The `composite.masked_replace@1` operation: `edited` + `base` + `mask` → the
/// composited `image`.
#[derive(Debug, Clone, Copy, Default)]
pub struct MaskedReplace;

impl MaskedReplace {
    /// Construct the masked-replace operation.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    /// The declared manifest for `composite.masked_replace@1`.
    ///
    /// # Errors
    /// Propagates the [`schema`](ErrorClass::Schema) error if the hard-coded op /
    /// impl ids are somehow invalid (they are not).
    pub fn manifest() -> Result<OperationManifest> {
        Ok(OperationManifest {
            id: MASKED_REPLACE_OP_ID.parse()?,
            impl_version: 1,
            summary: "Composite an edited image over a base image through a coverage mask in \
                      premultiplied linear light (out = m·edited + (1−m)·base); the single MVP \
                      authorization boundary. Where the mask is 0 the output is the base \
                      bit-for-bit; where it is 1, the edited. Rejected on nonlinear (srgb) or \
                      straight-alpha input."
                .to_owned(),
            // base + m·(edited − base) is a per-channel f32 multiply-add with the
            // m ∈ {0, 1} extremes returned verbatim: bit-exact for the declared
            // scalar/backend.
            determinism: DeterminismTier::Exact,
            roi: RoiPolicy {
                category: RoiCategory::Pointwise,
                halo_px: None,
            },
            inputs: vec![
                InputSpec {
                    name: "edited".to_owned(),
                    kind: ResourceKind::Image,
                    required: true,
                    doc: "The premultiplied linear-light edited image, kept where the mask is 1."
                        .to_owned(),
                },
                InputSpec {
                    name: "base".to_owned(),
                    kind: ResourceKind::Image,
                    required: true,
                    doc: "The premultiplied linear-light base image, kept where the mask is 0."
                        .to_owned(),
                },
                InputSpec {
                    name: "mask".to_owned(),
                    kind: ResourceKind::Mask,
                    required: true,
                    doc: "The coverage mask in [0, 1] selecting edited over base, co-located with \
                          the images."
                        .to_owned(),
                },
            ],
            outputs: vec![OutputSpec {
                name: "image".to_owned(),
                kind: ResourceKind::Image,
                doc: "The composited image (the base descriptor; same extent/layout).".to_owned(),
            }],
            params: vec![],
            implementations: vec![reference_impl()?],
            test: masked_replace_test_metadata(),
        })
    }
}

impl OpContract for MaskedReplace {
    fn declared_inputs(&self) -> Vec<(String, ResourceKind)> {
        vec![
            ("edited".to_owned(), ResourceKind::Image),
            ("base".to_owned(), ResourceKind::Image),
            ("mask".to_owned(), ResourceKind::Mask),
        ]
    }

    fn declared_outputs(&self) -> Vec<(String, ResourceKind)> {
        vec![("image".to_owned(), ResourceKind::Image)]
    }

    fn infer_outputs(
        &self,
        inputs: &Descriptors,
        _params: &serde_json::Value,
    ) -> Result<OutputDescriptors> {
        let edited = image_descriptor(inputs, "edited")?;
        let base = image_descriptor(inputs, "base")?;
        let extent = mask_extent(inputs)?;
        let out_descriptor = check_and_retarget(edited, base, extent)?;

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
        // Pointwise: each output pixel reads the co-located edited, base, and mask
        // sample, so every port shares the requested output region.
        let mut regions = InputRegions::new();
        if let Some(region) = requested_outputs.get("image") {
            for port in ["edited", "base", "mask"] {
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
        // The composited result stays premultiplied linear color: a future edit
        // that changed the output representation is caught here.
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

impl OpImplementation for MaskedReplace {
    fn compute(
        &self,
        inputs: &InputValues,
        _params: &serde_json::Value,
    ) -> std::result::Result<OutputValues, Error> {
        let edited = input_value(inputs, "edited")?;
        let base = input_value(inputs, "base")?;
        let mask = input_value(inputs, "mask")?;

        let ResourceDescriptor::Image(edited_descriptor) = edited.descriptor() else {
            return Err(input_type_error("edited"));
        };
        let ResourceDescriptor::Image(base_descriptor) = base.descriptor() else {
            return Err(input_type_error("base"));
        };
        let ResourceDescriptor::Mask(mask_descriptor) = mask.descriptor() else {
            return Err(Error::new(
                ErrorClass::Type,
                E_MASKED_REPLACE_INPUT,
                "composite.masked_replace `mask` input must be a mask resource".to_owned(),
            ));
        };

        let out_descriptor =
            check_and_retarget(edited_descriptor, base_descriptor, mask_descriptor.extent)?;
        let samples = blend(
            edited.samples(),
            base.samples(),
            mask.samples(),
            base.channels(),
        );

        let value = ResourceValue::new(
            ResourceDescriptor::Image(out_descriptor),
            base.channels(),
            samples,
        )
        .map_err(|actual| {
            Error::new(
                ErrorClass::Execution,
                E_MASKED_REPLACE_BUFFER,
                format!(
                    "composite.masked_replace produced a sample buffer of unexpected length {actual}"
                ),
            )
        })?;

        let mut out = OutputValues::new();
        out.insert("image".to_owned(), value);
        Ok(out)
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
            E_MASKED_REPLACE_INPUT,
            format!("composite.masked_replace requires a `{port}` input value"),
        )
    })
}

/// The wrong-resource-kind error for a color image port.
fn input_type_error(port: &str) -> Error {
    Error::new(
        ErrorClass::Type,
        E_MASKED_REPLACE_INPUT,
        format!("composite.masked_replace `{port}` input must be an image resource"),
    )
}

/// The mandatory `cpu.reference@1` oracle implementation id.
fn reference_impl() -> Result<ImplId> {
    ImplId::new("cpu", "reference", 1)
}

/// The verification declarations for `composite.masked_replace@1`
/// (`AGENT_VERIFICATION` §3.x, the authorization primitive). The op is an exact,
/// single-reference per-channel blend verified by analytic mask-extreme fixtures
/// (m = 0 → base, m = 1 → edited), the safety-critical outside-region bit-identity
/// property, a soft-mask blend-monotonicity property, and premultiplied
/// compositing correctness. Differential does not apply (one implementation).
/// Perceptual is not applicable: the blend is bit-exact, pinned by numeric
/// goldens, with no perceptual-quality metric to apply.
fn masked_replace_test_metadata() -> TestMetadata {
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
            "composite.masked_replace is a bit-exact per-channel premultiplied blend verified by \
             mask-extreme exactness, outside-region bit-identity, and soft-mask blend properties; \
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
