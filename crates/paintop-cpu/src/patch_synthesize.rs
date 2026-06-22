//! The `repair.patch_synthesize@1` operation (`OP_CATALOG` §10, patch fill).
//!
//! Synthesises a candidate fill image from a precomputed
//! [`PatchField`](paintop_ir::ResourceDescriptor::PatchField): for every target
//! pixel inside the `hole` mask it copies the `source` pixel at the field's
//! anchor coordinate for that pixel; every pixel outside the hole is the `target`
//! pixel verbatim. It is the synthesis half of the `PatchMatch` inpainting pair —
//! `repair.patch_field` finds the correspondences, this op gathers colour through
//! them.
//!
//! # Determinism
//!
//! The op is a pure per-pixel gather with no floating-point arithmetic — every
//! output sample is copied from an input sample — so it is
//! [`Exact`](DeterminismTier::Exact): bit-identical on every backend. Its
//! defining guarantees are *outside-hole identity* (untouched pixels equal the
//! target) and *anchored gather* (hole pixels equal the source at the field
//! anchor).

use paintop_core::executor::{InputValues, OpImplementation, OutputValues, ResourceValue};
use paintop_ir::{
    AssertionResult, Descriptors, DeterminismTier, Error, ErrorClass, Extent, ImageDescriptor,
    ImplId, InputRegions, InputSpec, OpContract, OperationManifest, OutputDescriptors,
    OutputRegions, OutputSpec, PATCH_FIELD_CHANNELS, PatchFieldDescriptor, Rect,
    ResourceDescriptor, ResourceKind, Result, RoiCategory, RoiPolicy, TestMetadata,
};

/// The canonical id of the patch-synthesis operation.
pub const PATCH_SYNTHESIZE_OP_ID: &str = "repair.patch_synthesize@1";

/// A required input was absent or carried an unsupported descriptor.
pub const E_PATCH_SYNTHESIZE_INPUT: &str = "E_PATCH_SYNTHESIZE_INPUT";

/// The inputs disagreed on shape (extent or channel count).
pub const E_PATCH_SYNTHESIZE_SHAPE: &str = "E_PATCH_SYNTHESIZE_SHAPE";

/// Read a required image port's descriptor.
fn image_descriptor<'a>(inputs: &'a Descriptors, port: &str) -> Result<&'a ImageDescriptor> {
    let resource = inputs.get(port).ok_or_else(|| {
        Error::new(
            ErrorClass::Reference,
            E_PATCH_SYNTHESIZE_INPUT,
            format!("repair.patch_synthesize requires a `{port}` input"),
        )
    })?;
    match resource {
        ResourceDescriptor::Image(d) => Ok(d),
        _ => Err(Error::new(
            ErrorClass::Type,
            E_PATCH_SYNTHESIZE_INPUT,
            format!("repair.patch_synthesize `{port}` must be an Image resource"),
        )),
    }
}

/// Read the `field` port's [`PatchFieldDescriptor`].
fn field_descriptor(inputs: &Descriptors) -> Result<&PatchFieldDescriptor> {
    let resource = inputs.get("field").ok_or_else(|| {
        Error::new(
            ErrorClass::Reference,
            E_PATCH_SYNTHESIZE_INPUT,
            "repair.patch_synthesize requires a `field` input".to_owned(),
        )
    })?;
    match resource {
        ResourceDescriptor::PatchField(d) => Ok(d),
        _ => Err(Error::new(
            ErrorClass::Type,
            E_PATCH_SYNTHESIZE_INPUT,
            "repair.patch_synthesize `field` must be a PatchField resource".to_owned(),
        )),
    }
}

/// Check that the source/target images and the field agree on shape, and return
/// the output (= target) descriptor.
fn check_shapes(
    source: &ImageDescriptor,
    target: &ImageDescriptor,
    field: &PatchFieldDescriptor,
    hole: Extent,
) -> Result<()> {
    if source.layout.channel_count() != target.layout.channel_count() {
        return Err(shape_err(
            "source and target must have the same channel count",
        ));
    }
    if field.target_extent != target.extent {
        return Err(shape_err(
            "the field target extent must match the target image extent",
        ));
    }
    if field.source_extent != source.extent {
        return Err(shape_err(
            "the field source extent must match the source image extent",
        ));
    }
    if hole != target.extent {
        return Err(shape_err(
            "the hole mask extent must match the target image",
        ));
    }
    Ok(())
}

/// Build a shape-mismatch error.
fn shape_err(detail: &str) -> Error {
    Error::new(
        ErrorClass::Semantic,
        E_PATCH_SYNTHESIZE_SHAPE,
        format!("repair.patch_synthesize: {detail}"),
    )
}

/// The `repair.patch_synthesize@1` operation.
#[derive(Debug, Clone, Copy, Default)]
pub struct PatchSynthesize;

impl PatchSynthesize {
    /// Construct the patch-synthesis operation.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    /// The declared manifest for `repair.patch_synthesize@1`.
    ///
    /// # Errors
    /// Propagates the [`schema`](ErrorClass::Schema) error if the hard-coded op /
    /// impl ids are somehow invalid (they are not).
    pub fn manifest() -> Result<OperationManifest> {
        Ok(OperationManifest {
            id: PATCH_SYNTHESIZE_OP_ID.parse()?,
            impl_version: 1,
            summary: "Synthesise a candidate fill image from a PatchField: inside the hole mask, \
                      gather each pixel from the source at the field's anchor; outside the hole, \
                      copy the target verbatim (outside-hole identity)."
                .to_owned(),
            determinism: DeterminismTier::Exact,
            roi: RoiPolicy {
                category: RoiCategory::FullDomain,
                halo_px: None,
            },
            inputs: vec![
                InputSpec {
                    name: "source".to_owned(),
                    kind: ResourceKind::Image,
                    required: true,
                    doc: "The source image patches are gathered from.".to_owned(),
                },
                InputSpec {
                    name: "target".to_owned(),
                    kind: ResourceKind::Image,
                    required: true,
                    doc: "The target image; pixels outside the hole are copied verbatim."
                        .to_owned(),
                },
                InputSpec {
                    name: "field".to_owned(),
                    kind: ResourceKind::PatchField,
                    required: true,
                    doc: "The patch correspondence field mapping each target pixel to a source \
                          anchor."
                        .to_owned(),
                },
                InputSpec {
                    name: "hole".to_owned(),
                    kind: ResourceKind::Mask,
                    required: true,
                    doc: "The coverage mask selecting the region to fill (> 0.5 is filled from \
                          the source)."
                        .to_owned(),
                },
            ],
            outputs: vec![OutputSpec {
                name: "image".to_owned(),
                kind: ResourceKind::Image,
                doc: "The candidate fill: source-gathered inside the hole, target outside."
                    .to_owned(),
            }],
            params: vec![],
            implementations: vec![reference_impl()?],
            test: patch_synthesize_test_metadata(),
        })
    }
}

impl OpContract for PatchSynthesize {
    fn declared_inputs(&self) -> Vec<(String, ResourceKind)> {
        vec![
            ("source".to_owned(), ResourceKind::Image),
            ("target".to_owned(), ResourceKind::Image),
            ("field".to_owned(), ResourceKind::PatchField),
            ("hole".to_owned(), ResourceKind::Mask),
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
        let source = image_descriptor(inputs, "source")?;
        let target = image_descriptor(inputs, "target")?;
        let field = field_descriptor(inputs)?;
        let hole = inputs.get("hole").ok_or_else(|| {
            Error::new(
                ErrorClass::Reference,
                E_PATCH_SYNTHESIZE_INPUT,
                "repair.patch_synthesize requires a `hole` input".to_owned(),
            )
        })?;
        check_shapes(source, target, field, hole.extent())?;

        let mut out = OutputDescriptors::new();
        out.insert("image".to_owned(), ResourceDescriptor::Image(*target));
        Ok(out)
    }

    fn required_inputs(
        &self,
        _requested_outputs: &OutputRegions,
        inputs: &Descriptors,
        _params: &serde_json::Value,
    ) -> Result<InputRegions> {
        // A gather reads the whole source (any anchor may be referenced) and the
        // whole target/field/hole.
        let mut regions = InputRegions::new();
        for port in ["source", "target", "field", "hole"] {
            if let Some(resource) = inputs.get(port) {
                let e = resource.extent();
                regions.insert(
                    port.to_owned(),
                    Rect::new(0, 0, i64::from(e.width), i64::from(e.height)),
                );
            }
        }
        Ok(regions)
    }

    fn validate_postconditions(
        &self,
        outputs: &OutputDescriptors,
        _params: &serde_json::Value,
    ) -> Result<Vec<AssertionResult>> {
        Ok(vec![match outputs.get("image") {
            Some(ResourceDescriptor::Image(_)) => AssertionResult::pass("produces_image"),
            _ => AssertionResult::fail("produces_image", "no `image` output produced"),
        }])
    }
}

impl OpImplementation for PatchSynthesize {
    fn compute(
        &self,
        inputs: &InputValues,
        _params: &serde_json::Value,
    ) -> std::result::Result<OutputValues, Error> {
        let source = input_value(inputs, "source")?;
        let target = input_value(inputs, "target")?;
        let field = input_value(inputs, "field")?;
        let hole = input_value(inputs, "hole")?;

        let ResourceDescriptor::Image(target_desc) = *target.descriptor() else {
            return Err(input_type_error("target"));
        };
        let ResourceDescriptor::Image(source_desc) = source.descriptor() else {
            return Err(input_type_error("source"));
        };
        let ResourceDescriptor::PatchField(field_desc) = field.descriptor() else {
            return Err(input_type_error("field"));
        };
        check_shapes(source_desc, &target_desc, field_desc, hole.extent())?;

        let channels = target.channels() as usize;
        let extent = target.extent();
        let (sw, sh) = (source.extent().width, source.extent().height);
        let src = source.samples();
        let tgt = target.samples();
        let field_samples = field.samples();
        let hole_samples = hole.samples();

        let pixel_count = (extent.width as usize) * (extent.height as usize);
        let mut out = vec![0.0_f32; pixel_count * channels];
        for p in 0..pixel_count {
            let filled = hole_samples.get(p).is_some_and(|&m| m > 0.5);
            if filled {
                // Read this pixel's source anchor from the packed field.
                let base = p * PATCH_FIELD_CHANNELS as usize;
                let sx = coord(field_samples.get(base).copied().unwrap_or(0.0), sw);
                let sy = coord(field_samples.get(base + 1).copied().unwrap_or(0.0), sh);
                let s_off = ((sy * sw as usize) + sx) * channels;
                for c in 0..channels {
                    out[(p * channels) + c] = src.get(s_off + c).copied().unwrap_or(0.0);
                }
            } else {
                // Outside-hole identity: copy the target pixel verbatim.
                for c in 0..channels {
                    out[(p * channels) + c] = tgt.get((p * channels) + c).copied().unwrap_or(0.0);
                }
            }
        }

        let value = ResourceValue::new(
            ResourceDescriptor::Image(target_desc),
            target.channels(),
            out,
        )
        .map_err(|actual| {
            Error::new(
                ErrorClass::Execution,
                E_PATCH_SYNTHESIZE_INPUT,
                format!(
                    "repair.patch_synthesize produced an image buffer of unexpected length \
                         {actual}"
                ),
            )
        })?;
        let mut outputs = OutputValues::new();
        outputs.insert("image".to_owned(), value);
        Ok(outputs)
    }
}

/// Convert a packed `f32` source coordinate to a clamped, in-bounds pixel index.
/// A field anchor is always a small non-negative integer, but the value is
/// clamped defensively so a malformed field can never index out of bounds.
fn coord(value: f32, dim: u32) -> usize {
    if !value.is_finite() || value < 0.0 {
        return 0;
    }
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "value is finite and non-negative; the result is clamped below the extent"
    )]
    let idx = value as usize;
    idx.min(dim.saturating_sub(1) as usize)
}

/// Read a required input value by port.
fn input_value<'a>(
    inputs: &'a InputValues,
    port: &str,
) -> std::result::Result<&'a ResourceValue, Error> {
    inputs.get(port).ok_or_else(|| {
        Error::new(
            ErrorClass::Reference,
            E_PATCH_SYNTHESIZE_INPUT,
            format!("repair.patch_synthesize requires a `{port}` value"),
        )
    })
}

/// Build a type error for a port carrying the wrong resource kind.
fn input_type_error(port: &str) -> Error {
    Error::new(
        ErrorClass::Type,
        E_PATCH_SYNTHESIZE_INPUT,
        format!("repair.patch_synthesize `{port}` carried the wrong resource kind"),
    )
}

/// The mandatory `cpu.reference@1` oracle implementation id.
fn reference_impl() -> Result<ImplId> {
    ImplId::new("cpu", "reference", 1)
}

/// Verification declarations for `repair.patch_synthesize@1`: an exact per-pixel
/// gather verified by analytic fixtures (outside-hole identity, anchored gather,
/// repeated-texture fill) and property tests. Differential across backends is not
/// applicable (a single reference, and the op is exact); perceptual does not
/// apply — correctness is exact sample equality, not a perceptual metric.
fn patch_synthesize_test_metadata() -> TestMetadata {
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
        VerificationCategory::Differential,
        CategoryStatus::not_applicable(
            "patch_synthesize is an exact per-pixel gather with a single cpu.reference \
             implementation; its analytic fixtures pin the exact output",
        ),
    );
    decls = decls.with(
        VerificationCategory::Perceptual,
        CategoryStatus::not_applicable(
            "the fill is verified by exact sample equality (outside-hole identity, anchored \
             gather), not a perceptual metric",
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
