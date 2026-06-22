//! The `frequency.laplacian_split@1` and `frequency.recombine@1` operations
//! (`OP_CATALOG` §13).
//!
//! `frequency.laplacian_split` decomposes an Image/`Field1` into a **Laplacian
//! pyramid**: every level `l < levels − 1` is the band-pass residual
//! `G_l − upsample(G_{l+1})` of the Gaussian pyramid, and the coarsest level
//! keeps the low-pass `G_{levels-1}` verbatim. `frequency.recombine` is its
//! exact inverse: it telescopes the bands back into the full-resolution image.
//!
//! # Reconstruction contract
//!
//! `recombine(laplacian_split(x))` reconstructs `x` up to f32 rounding: the
//! coarsest band is the full low-pass and each finer band adds back exactly the
//! gap to its upsampled child, so the sum telescopes to `G_0 = x`. The module's
//! tests assert this on bounded fixtures, including odd extents and the
//! single-level case.
//!
//! # Determinism
//!
//! Both ops are fixed-order `f64` reductions over a deterministic blur /
//! up/downsample lattice, so they are bit-identical across reruns
//! ([`Bounded`](DeterminismTier::Bounded), since the blur agrees with an
//! alternate backend only within the discretization bound).

use paintop_core::executor::{InputValues, OpImplementation, OutputValues, ResourceValue};
use paintop_ir::{
    AlphaRepresentation, AssertionResult, ChannelLayout, ColorEncoding, ColorRange,
    CoordinateConvention, Descriptors, DeterminismTier, DownsampleFactor, Error, ErrorClass,
    Extent, FieldArity, FieldDescriptor, ImageDescriptor, ImplId, InputRegions, InputSpec,
    OpContract, OperationManifest, OutputDescriptors, OutputRegions, OutputSpec, ParamSpec,
    ParamType, PyramidDescriptor, Rect, ResourceDescriptor, ResourceKind, Result, RoiCategory,
    RoiPolicy, ScalarType, SemanticRole, TestMetadata,
};

use crate::frequency::{laplacian_recombine, laplacian_split, level_extents};
use crate::gaussian_pyramid::{E_PYRAMID_INPUT, PyramidParams, build_pyramid_samples};

/// The canonical id of the Laplacian-split operation.
pub const LAPLACIAN_SPLIT_OP_ID: &str = "frequency.laplacian_split@1";

/// The canonical id of the recombine operation.
pub const RECOMBINE_OP_ID: &str = "frequency.recombine@1";

/// The interleaved channel count of a supported Image/`Field1` descriptor.
fn input_channels(descriptor: &ResourceDescriptor) -> Result<u32> {
    match descriptor {
        ResourceDescriptor::Image(d) => Ok(d.layout.channel_count()),
        ResourceDescriptor::Field1(_) => Ok(1),
        _ => Err(Error::new(
            ErrorClass::Type,
            E_PYRAMID_INPUT,
            "frequency.laplacian_split `input` must be an Image or Field1 resource".to_owned(),
        )),
    }
}

/// Build the Laplacian pyramid descriptor for an `extent`/`channels` input.
const fn pyramid_descriptor(
    extent: Extent,
    channels: u32,
    params: PyramidParams,
) -> PyramidDescriptor {
    PyramidDescriptor {
        base_extent: extent,
        levels: params.levels,
        channels,
        scalar: ScalarType::F32,
        factor: DownsampleFactor::Half,
        phase: params.phase,
        coordinates: CoordinateConvention::PixelCenterUpperLeft,
    }
}

/// Slice a finest-first concatenated pyramid buffer into its per-level planes,
/// using the derived per-level extents.
fn slice_levels<'a>(buffer: &'a [f32], extents: &[Extent], channels: u32) -> Vec<&'a [f32]> {
    let mut planes = Vec::with_capacity(extents.len());
    let mut offset = 0usize;
    for e in extents {
        let n = (e.width as usize) * (e.height as usize) * (channels as usize);
        planes.push(&buffer[offset..offset + n]);
        offset += n;
    }
    planes
}

/// Build the Laplacian band buffer from a base plane (build the Gaussian
/// pyramid, then split it).
#[must_use]
pub fn build_laplacian_samples(
    base: &[f32],
    extent: Extent,
    channels: u32,
    params: PyramidParams,
) -> Vec<f32> {
    let gaussian = build_pyramid_samples(base, extent, channels, params);
    let extents = level_extents(extent, params.levels, params.phase);
    let levels = slice_levels(&gaussian, &extents, channels);
    laplacian_split(&levels, &extents, channels)
}

// ---------------------------------------------------------------------------
// frequency.laplacian_split@1
// ---------------------------------------------------------------------------

/// The `frequency.laplacian_split@1` operation.
#[derive(Debug, Clone, Copy, Default)]
pub struct LaplacianSplit;

impl LaplacianSplit {
    /// Construct the Laplacian-split operation.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    /// The declared manifest for `frequency.laplacian_split@1`.
    ///
    /// # Errors
    /// Propagates the [`schema`](ErrorClass::Schema) error if the hard-coded op /
    /// impl ids are somehow invalid (they are not).
    pub fn manifest() -> Result<OperationManifest> {
        Ok(OperationManifest {
            id: LAPLACIAN_SPLIT_OP_ID.parse()?,
            impl_version: 1,
            summary:
                "Decompose an Image/Field1 into a Laplacian pyramid: each level the band-pass \
                      residual G_l - upsample(G_{l+1}), the coarsest level the low-pass. Exactly \
                      invertible by frequency.recombine."
                    .to_owned(),
            determinism: DeterminismTier::Bounded,
            roi: RoiPolicy {
                category: RoiCategory::FullDomain,
                halo_px: None,
            },
            inputs: vec![InputSpec {
                name: "input".to_owned(),
                kind: ResourceKind::Image,
                required: true,
                doc: "The Image or Field1 to decompose (level 0 of the Gaussian pyramid)."
                    .to_owned(),
            }],
            outputs: vec![OutputSpec {
                name: "pyramid".to_owned(),
                kind: ResourceKind::Pyramid,
                doc: "The Laplacian (band-pass) pyramid; the coarsest level is the low-pass."
                    .to_owned(),
            }],
            params: pyramid_params_spec(),
            implementations: vec![reference_impl()?],
            test: split_test_metadata(),
        })
    }
}

impl OpContract for LaplacianSplit {
    fn declared_inputs(&self) -> Vec<(String, ResourceKind)> {
        vec![("input".to_owned(), ResourceKind::Image)]
    }

    fn declared_outputs(&self) -> Vec<(String, ResourceKind)> {
        vec![("pyramid".to_owned(), ResourceKind::Pyramid)]
    }

    fn infer_outputs(
        &self,
        inputs: &Descriptors,
        params: &serde_json::Value,
    ) -> Result<OutputDescriptors> {
        let input = inputs.get("input").ok_or_else(|| {
            Error::new(
                ErrorClass::Reference,
                E_PYRAMID_INPUT,
                "frequency.laplacian_split requires an `input` resource".to_owned(),
            )
        })?;
        let channels = input_channels(input)?;
        let resolved = PyramidParams::resolve(params)?;
        let descriptor = pyramid_descriptor(input.extent(), channels, resolved);
        descriptor.validate()?;
        let mut out = OutputDescriptors::new();
        out.insert(
            "pyramid".to_owned(),
            ResourceDescriptor::Pyramid(descriptor),
        );
        Ok(out)
    }

    fn required_inputs(
        &self,
        _requested_outputs: &OutputRegions,
        inputs: &Descriptors,
        _params: &serde_json::Value,
    ) -> Result<InputRegions> {
        Ok(full_input_region(inputs, "input"))
    }

    fn validate_postconditions(
        &self,
        outputs: &OutputDescriptors,
        _params: &serde_json::Value,
    ) -> Result<Vec<AssertionResult>> {
        Ok(vec![match outputs.get("pyramid") {
            Some(ResourceDescriptor::Pyramid(_)) => AssertionResult::pass("produces_pyramid"),
            _ => AssertionResult::fail("produces_pyramid", "no `pyramid` output produced"),
        }])
    }
}

impl OpImplementation for LaplacianSplit {
    fn compute(
        &self,
        inputs: &InputValues,
        params: &serde_json::Value,
    ) -> std::result::Result<OutputValues, Error> {
        let input = inputs.get("input").ok_or_else(|| {
            Error::new(
                ErrorClass::Reference,
                E_PYRAMID_INPUT,
                "frequency.laplacian_split requires an `input` value".to_owned(),
            )
        })?;
        let channels = input.channels();
        let extent = input.extent();
        let resolved = PyramidParams::resolve(params)?;
        let descriptor = pyramid_descriptor(extent, channels, resolved);
        descriptor.validate()?;

        let samples = build_laplacian_samples(input.samples(), extent, channels, resolved);
        let value = ResourceValue::pyramid(descriptor, samples).map_err(|actual| {
            Error::new(
                ErrorClass::Execution,
                E_PYRAMID_INPUT,
                format!(
                    "frequency.laplacian_split produced a pyramid buffer of unexpected length \
                     {actual}"
                ),
            )
        })?;
        let mut out = OutputValues::new();
        out.insert("pyramid".to_owned(), value);
        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// frequency.recombine@1
// ---------------------------------------------------------------------------

/// The `frequency.recombine@1` operation.
#[derive(Debug, Clone, Copy, Default)]
pub struct Recombine;

impl Recombine {
    /// Construct the recombine operation.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    /// The declared manifest for `frequency.recombine@1`.
    ///
    /// # Errors
    /// Propagates the [`schema`](ErrorClass::Schema) error if the hard-coded op /
    /// impl ids are somehow invalid (they are not).
    pub fn manifest() -> Result<OperationManifest> {
        Ok(OperationManifest {
            id: RECOMBINE_OP_ID.parse()?,
            impl_version: 1,
            summary: "Reconstruct a full-resolution Image/Field1 from a Laplacian pyramid by \
                      telescoping the bands: recon_l = L_l + upsample(recon_{l+1}). The exact \
                      inverse of frequency.laplacian_split."
                .to_owned(),
            determinism: DeterminismTier::Bounded,
            roi: RoiPolicy {
                category: RoiCategory::FullDomain,
                halo_px: None,
            },
            inputs: vec![InputSpec {
                name: "pyramid".to_owned(),
                kind: ResourceKind::Pyramid,
                required: true,
                doc: "The Laplacian pyramid to reconstruct (as produced by laplacian_split)."
                    .to_owned(),
            }],
            outputs: vec![OutputSpec {
                name: "image".to_owned(),
                kind: ResourceKind::Image,
                doc: "The reconstructed full-resolution Image (level-0 extent).".to_owned(),
            }],
            params: vec![],
            implementations: vec![reference_impl()?],
            test: recombine_test_metadata(),
        })
    }
}

/// The reconstructed Image descriptor at a pyramid's level-0 extent.
const fn reconstructed_image(d: &PyramidDescriptor) -> ResourceDescriptor {
    let layout = match d.channels {
        1 => ChannelLayout::Gray,
        2 => ChannelLayout::GrayA,
        3 => ChannelLayout::Rgb,
        _ => ChannelLayout::Rgba,
    };
    ResourceDescriptor::Image(ImageDescriptor {
        extent: d.base_extent,
        layout,
        scalar: ScalarType::F32,
        color: ColorEncoding::LinearSrgb,
        range: ColorRange::SceneReferred,
        alpha: AlphaRepresentation::Premultiplied,
        coordinates: d.coordinates,
        semantic: SemanticRole::Color,
    })
}

/// The reconstructed `Field1` descriptor at a single-channel pyramid's level-0
/// extent (used when the pyramid carries one channel of scalar field data).
const fn reconstructed_field(d: &PyramidDescriptor) -> ResourceDescriptor {
    ResourceDescriptor::Field1(FieldDescriptor {
        arity: FieldArity::Field1,
        extent: d.base_extent,
        scalar: ScalarType::F32,
        semantic: SemanticRole::Distance,
        coordinates: d.coordinates,
        space: None,
        normalization: None,
        encoding: None,
    })
}

impl OpContract for Recombine {
    fn declared_inputs(&self) -> Vec<(String, ResourceKind)> {
        vec![("pyramid".to_owned(), ResourceKind::Pyramid)]
    }

    fn declared_outputs(&self) -> Vec<(String, ResourceKind)> {
        vec![("image".to_owned(), ResourceKind::Image)]
    }

    fn infer_outputs(
        &self,
        inputs: &Descriptors,
        _params: &serde_json::Value,
    ) -> Result<OutputDescriptors> {
        let pyramid = inputs.get("pyramid").ok_or_else(|| {
            Error::new(
                ErrorClass::Reference,
                E_PYRAMID_INPUT,
                "frequency.recombine requires a `pyramid` resource".to_owned(),
            )
        })?;
        let ResourceDescriptor::Pyramid(d) = pyramid else {
            return Err(Error::new(
                ErrorClass::Type,
                E_PYRAMID_INPUT,
                "frequency.recombine `pyramid` input must be a Pyramid resource".to_owned(),
            ));
        };
        d.validate()?;
        let mut out = OutputDescriptors::new();
        out.insert("image".to_owned(), reconstructed_image(d));
        Ok(out)
    }

    fn required_inputs(
        &self,
        _requested_outputs: &OutputRegions,
        inputs: &Descriptors,
        _params: &serde_json::Value,
    ) -> Result<InputRegions> {
        Ok(full_input_region(inputs, "pyramid"))
    }

    fn validate_postconditions(
        &self,
        outputs: &OutputDescriptors,
        _params: &serde_json::Value,
    ) -> Result<Vec<AssertionResult>> {
        Ok(vec![match outputs.get("image") {
            Some(ResourceDescriptor::Image(_) | ResourceDescriptor::Field1(_)) => {
                AssertionResult::pass("produces_image")
            }
            _ => AssertionResult::fail("produces_image", "no reconstructed `image` produced"),
        }])
    }
}

impl OpImplementation for Recombine {
    fn compute(
        &self,
        inputs: &InputValues,
        _params: &serde_json::Value,
    ) -> std::result::Result<OutputValues, Error> {
        let pyramid = inputs.get("pyramid").ok_or_else(|| {
            Error::new(
                ErrorClass::Reference,
                E_PYRAMID_INPUT,
                "frequency.recombine requires a `pyramid` value".to_owned(),
            )
        })?;
        let ResourceDescriptor::Pyramid(d) = *pyramid.descriptor() else {
            return Err(Error::new(
                ErrorClass::Type,
                E_PYRAMID_INPUT,
                "frequency.recombine `pyramid` input must be a Pyramid resource".to_owned(),
            ));
        };
        d.validate()?;

        let extents = level_extents(d.base_extent, d.levels, d.phase);
        // Recover each level's band slice from the pyramid value.
        let mut bands: Vec<&[f32]> = Vec::with_capacity(d.levels as usize);
        for level in 0..d.levels {
            let band = pyramid.pyramid_level(level).ok_or_else(|| {
                Error::new(
                    ErrorClass::Execution,
                    E_PYRAMID_INPUT,
                    format!("frequency.recombine could not slice pyramid level {level}"),
                )
            })?;
            bands.push(band);
        }
        let samples = laplacian_recombine(&bands, &extents, d.channels);

        // The reconstructed resource keeps the pyramid's channel count; a
        // single-channel pyramid reconstructs to a Field1, others to an Image.
        let descriptor = if d.channels == 1 {
            reconstructed_field(&d)
        } else {
            reconstructed_image(&d)
        };
        let value = ResourceValue::new(descriptor, d.channels, samples).map_err(|actual| {
            Error::new(
                ErrorClass::Execution,
                E_PYRAMID_INPUT,
                format!("frequency.recombine produced a buffer of unexpected length {actual}"),
            )
        })?;
        let mut out = OutputValues::new();
        out.insert("image".to_owned(), value);
        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

/// The shared `levels`/`sigma`/`phase` parameter spec for the split op (the
/// same vocabulary `frequency.gaussian_pyramid` uses).
fn pyramid_params_spec() -> Vec<ParamSpec> {
    use paintop_ir::MAX_PYRAMID_LEVELS;
    use paintop_ir::ParamUnit;
    vec![
        ParamSpec {
            name: "levels".to_owned(),
            ty: ParamType::Integer,
            unit: None,
            required: true,
            default: None,
            choices: vec![],
            doc: format!("The number of levels, level 0 the base (1..={MAX_PYRAMID_LEVELS})."),
        },
        ParamSpec {
            name: "sigma".to_owned(),
            ty: ParamType::Float,
            unit: Some(ParamUnit::Pixels),
            required: false,
            default: Some(serde_json::json!(crate::frequency::DEFAULT_PYRAMID_SIGMA)),
            choices: vec![],
            doc: "The pre-decimation Gaussian standard deviation (anti-alias low-pass).".to_owned(),
        },
        ParamSpec {
            name: "phase".to_owned(),
            ty: ParamType::String,
            unit: None,
            required: false,
            default: Some(serde_json::json!("floor")),
            choices: vec!["floor".to_owned(), "ceil".to_owned()],
            doc: "The odd-size rounding rule mapping a parent extent to its child.".to_owned(),
        },
    ]
}

/// The full-domain input region map for a single named port.
fn full_input_region(inputs: &Descriptors, port: &str) -> InputRegions {
    let mut regions = InputRegions::new();
    if let Some(input) = inputs.get(port) {
        let extent = input.extent();
        regions.insert(
            port.to_owned(),
            Rect::new(0, 0, i64::from(extent.width), i64::from(extent.height)),
        );
    }
    regions
}

/// The mandatory `cpu.reference@1` oracle implementation id.
fn reference_impl() -> Result<ImplId> {
    ImplId::new("cpu", "reference", 1)
}

/// Verification declarations shared by the split/recombine ops: bounded
/// multi-resolution ops with analytic fixtures (the reconstruction round-trip,
/// the single-level low-pass, odd extents) and property tests (determinism,
/// reconstruction tolerance). Perceptual does not apply — correctness is the
/// exact-inverse reconstruction property, not a perceptual metric.
fn split_test_metadata() -> TestMetadata {
    band_test_metadata(
        "the Laplacian split is verified by the split/recombine reconstruction round-trip and the \
         band-extent convention, not a perceptual metric",
    )
}

fn recombine_test_metadata() -> TestMetadata {
    band_test_metadata(
        "recombine is verified as the exact inverse of laplacian_split (reconstruction within \
         f32 tolerance), not a perceptual metric",
    )
}

fn band_test_metadata(perceptual_reason: &str) -> TestMetadata {
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
        CategoryStatus::not_applicable(perceptual_reason),
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
