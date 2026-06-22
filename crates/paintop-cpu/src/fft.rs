//! The `frequency.fft2@1` and `frequency.ifft2@1` operations (`OP_CATALOG` §9).
//!
//! `frequency.fft2` transforms a real Image/`Field1` spatial plane into a typed
//! complex [`Spectrum`](paintop_ir::ResourceDescriptor::Spectrum); `frequency.ifft2`
//! is its inverse, reconstructing the real spatial plane (a `Field1` for a
//! single channel, an Image otherwise).
//!
//! # Round-trip contract
//!
//! `ifft2(fft2(x))` reconstructs `x` up to floating-point rounding: the forward
//! transform carries no scale and the inverse carries the full `1/(W·H)` scale
//! (see [`crate::dft`]). The reconstruction's tiny imaginary residual is
//! discarded — the source is real — so the op is
//! [`Bounded`](DeterminismTier::Bounded): bit-identical across reruns on a fixed
//! backend, but agreeing with an alternate FFT backend only within the
//! transform's floating-point bound.

use paintop_core::executor::{InputValues, OpImplementation, OutputValues, ResourceValue};
use paintop_ir::{
    AlphaRepresentation, AssertionResult, ChannelLayout, ColorEncoding, ColorRange,
    CoordinateConvention, Descriptors, DeterminismTier, Error, ErrorClass, Extent, FieldArity,
    FieldDescriptor, ImageDescriptor, ImplId, InputRegions, InputSpec, OpContract,
    OperationManifest, OutputDescriptors, OutputRegions, OutputSpec, Rect, ResourceDescriptor,
    ResourceKind, Result, RoiCategory, RoiPolicy, ScalarType, SemanticRole, SpectrumDescriptor,
    TestMetadata,
};

use crate::dft::{forward_real, inverse_real};

/// The canonical id of the forward-FFT operation.
pub const FFT2_OP_ID: &str = "frequency.fft2@1";

/// The canonical id of the inverse-FFT operation.
pub const IFFT2_OP_ID: &str = "frequency.ifft2@1";

/// The `input`/`spectrum` was absent or carried an unsupported descriptor.
pub const E_FFT_INPUT: &str = "E_FFT_INPUT";

/// The interleaved channel count of a supported Image/`Field1` descriptor.
fn input_channels(descriptor: &ResourceDescriptor) -> Result<u32> {
    match descriptor {
        ResourceDescriptor::Image(d) => Ok(d.layout.channel_count()),
        ResourceDescriptor::Field1(_) => Ok(1),
        _ => Err(Error::new(
            ErrorClass::Type,
            E_FFT_INPUT,
            "frequency.fft2 `input` must be an Image or Field1 resource".to_owned(),
        )),
    }
}

/// The spectrum descriptor for a real plane of `extent`/`channels`.
const fn spectrum_descriptor(extent: Extent, channels: u32) -> SpectrumDescriptor {
    SpectrumDescriptor {
        extent,
        channels,
        scalar: ScalarType::F32,
        coordinates: CoordinateConvention::PixelCenterUpperLeft,
    }
}

/// The reconstructed Image descriptor at a spectrum's spatial extent.
const fn reconstructed_image(d: &SpectrumDescriptor) -> ResourceDescriptor {
    let layout = match d.channels {
        1 => ChannelLayout::Gray,
        2 => ChannelLayout::GrayA,
        3 => ChannelLayout::Rgb,
        _ => ChannelLayout::Rgba,
    };
    ResourceDescriptor::Image(ImageDescriptor {
        extent: d.extent,
        layout,
        scalar: ScalarType::F32,
        color: ColorEncoding::LinearSrgb,
        range: ColorRange::SceneReferred,
        alpha: AlphaRepresentation::Premultiplied,
        coordinates: d.coordinates,
        semantic: SemanticRole::Color,
    })
}

/// The reconstructed `Field1` descriptor at a single-channel spectrum's extent.
const fn reconstructed_field(d: &SpectrumDescriptor) -> ResourceDescriptor {
    ResourceDescriptor::Field1(FieldDescriptor {
        arity: FieldArity::Field1,
        extent: d.extent,
        scalar: ScalarType::F32,
        semantic: SemanticRole::Distance,
        coordinates: d.coordinates,
        space: None,
        normalization: None,
        encoding: None,
    })
}

/// The mandatory `cpu.reference@1` oracle implementation id.
///
/// # Errors
/// Propagates a schema error if the hard-coded impl id is invalid (it is not).
fn reference_impl() -> Result<ImplId> {
    ImplId::new("cpu", "reference", 1)
}

// ---------------------------------------------------------------------------
// frequency.fft2@1
// ---------------------------------------------------------------------------

/// The `frequency.fft2@1` operation.
#[derive(Debug, Clone, Copy, Default)]
pub struct Fft2;

impl Fft2 {
    /// Construct the forward-FFT operation.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    /// The declared manifest for `frequency.fft2@1`.
    ///
    /// # Errors
    /// Propagates the schema error if the hard-coded op/impl ids are invalid.
    pub fn manifest() -> Result<OperationManifest> {
        Ok(OperationManifest {
            id: FFT2_OP_ID.parse()?,
            impl_version: 1,
            summary: "Forward 2-D DFT of a real Image/Field1 plane into a typed complex Spectrum \
                      (non-normalized forward transform, DC at the array origin, channels \
                      transformed independently)."
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
                doc: "The real Image or Field1 spatial plane to transform.".to_owned(),
            }],
            outputs: vec![OutputSpec {
                name: "spectrum".to_owned(),
                kind: ResourceKind::Spectrum,
                doc: "The complex frequency spectrum (interleaved real/imaginary per bin)."
                    .to_owned(),
            }],
            params: vec![],
            implementations: vec![reference_impl()?],
            test: fft_test_metadata(
                "the forward DFT is verified by the fft2->ifft2 reconstruction round trip, the \
                 constant-plane pure-DC fixture, and the known-sinusoid spectral-peak fixture, \
                 not a perceptual metric",
            ),
        })
    }
}

impl OpContract for Fft2 {
    fn declared_inputs(&self) -> Vec<(String, ResourceKind)> {
        vec![("input".to_owned(), ResourceKind::Image)]
    }

    fn declared_outputs(&self) -> Vec<(String, ResourceKind)> {
        vec![("spectrum".to_owned(), ResourceKind::Spectrum)]
    }

    fn infer_outputs(
        &self,
        inputs: &Descriptors,
        _params: &serde_json::Value,
    ) -> Result<OutputDescriptors> {
        let input = inputs.get("input").ok_or_else(|| {
            Error::new(
                ErrorClass::Reference,
                E_FFT_INPUT,
                "frequency.fft2 requires an `input` resource".to_owned(),
            )
        })?;
        let channels = input_channels(input)?;
        let descriptor = spectrum_descriptor(input.extent(), channels);
        descriptor.validate()?;
        let mut out = OutputDescriptors::new();
        out.insert(
            "spectrum".to_owned(),
            ResourceDescriptor::Spectrum(descriptor),
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
        Ok(vec![match outputs.get("spectrum") {
            Some(ResourceDescriptor::Spectrum(_)) => AssertionResult::pass("produces_spectrum"),
            _ => AssertionResult::fail("produces_spectrum", "no `spectrum` output produced"),
        }])
    }
}

impl OpImplementation for Fft2 {
    fn compute(
        &self,
        inputs: &InputValues,
        _params: &serde_json::Value,
    ) -> std::result::Result<OutputValues, Error> {
        let input = inputs.get("input").ok_or_else(|| {
            Error::new(
                ErrorClass::Reference,
                E_FFT_INPUT,
                "frequency.fft2 requires an `input` value".to_owned(),
            )
        })?;
        let channels = input.channels();
        let extent = input.extent();
        let descriptor = spectrum_descriptor(extent, channels);
        descriptor.validate()?;
        let samples = forward_real(
            input.samples(),
            extent.width as usize,
            extent.height as usize,
            channels as usize,
        );
        let value = ResourceValue::spectrum(descriptor, samples).map_err(|actual| {
            Error::new(
                ErrorClass::Execution,
                E_FFT_INPUT,
                format!("frequency.fft2 produced a spectrum buffer of unexpected length {actual}"),
            )
        })?;
        let mut out = OutputValues::new();
        out.insert("spectrum".to_owned(), value);
        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// frequency.ifft2@1
// ---------------------------------------------------------------------------

/// The `frequency.ifft2@1` operation.
#[derive(Debug, Clone, Copy, Default)]
pub struct Ifft2;

impl Ifft2 {
    /// Construct the inverse-FFT operation.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    /// The declared manifest for `frequency.ifft2@1`.
    ///
    /// # Errors
    /// Propagates the schema error if the hard-coded op/impl ids are invalid.
    pub fn manifest() -> Result<OperationManifest> {
        Ok(OperationManifest {
            id: IFFT2_OP_ID.parse()?,
            impl_version: 1,
            summary: "Inverse 2-D DFT of a complex Spectrum back to a real spatial plane \
                      (Field1 for one channel, Image otherwise; full 1/(W*H) normalization, \
                      imaginary residual discarded)."
                .to_owned(),
            determinism: DeterminismTier::Bounded,
            roi: RoiPolicy {
                category: RoiCategory::FullDomain,
                halo_px: None,
            },
            inputs: vec![InputSpec {
                name: "spectrum".to_owned(),
                kind: ResourceKind::Spectrum,
                required: true,
                doc: "The complex frequency spectrum to invert.".to_owned(),
            }],
            outputs: vec![OutputSpec {
                name: "image".to_owned(),
                kind: ResourceKind::Image,
                doc: "The reconstructed real spatial plane (Field1 for a single channel)."
                    .to_owned(),
            }],
            params: vec![],
            implementations: vec![reference_impl()?],
            test: fft_test_metadata(
                "the inverse DFT is verified as the exact inverse of fft2 (reconstruction round \
                 trip within tolerance) and the DC-only-spectrum constant-plane fixture, not a \
                 perceptual metric",
            ),
        })
    }
}

impl OpContract for Ifft2 {
    fn declared_inputs(&self) -> Vec<(String, ResourceKind)> {
        vec![("spectrum".to_owned(), ResourceKind::Spectrum)]
    }

    fn declared_outputs(&self) -> Vec<(String, ResourceKind)> {
        vec![("image".to_owned(), ResourceKind::Image)]
    }

    fn infer_outputs(
        &self,
        inputs: &Descriptors,
        _params: &serde_json::Value,
    ) -> Result<OutputDescriptors> {
        let spectrum = inputs.get("spectrum").ok_or_else(|| {
            Error::new(
                ErrorClass::Reference,
                E_FFT_INPUT,
                "frequency.ifft2 requires a `spectrum` resource".to_owned(),
            )
        })?;
        let ResourceDescriptor::Spectrum(d) = spectrum else {
            return Err(Error::new(
                ErrorClass::Type,
                E_FFT_INPUT,
                "frequency.ifft2 `spectrum` input must be a Spectrum resource".to_owned(),
            ));
        };
        d.validate()?;
        let descriptor = if d.channels == 1 {
            reconstructed_field(d)
        } else {
            reconstructed_image(d)
        };
        let mut out = OutputDescriptors::new();
        out.insert("image".to_owned(), descriptor);
        Ok(out)
    }

    fn required_inputs(
        &self,
        _requested_outputs: &OutputRegions,
        inputs: &Descriptors,
        _params: &serde_json::Value,
    ) -> Result<InputRegions> {
        Ok(full_input_region(inputs, "spectrum"))
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

impl OpImplementation for Ifft2 {
    fn compute(
        &self,
        inputs: &InputValues,
        _params: &serde_json::Value,
    ) -> std::result::Result<OutputValues, Error> {
        let spectrum = inputs.get("spectrum").ok_or_else(|| {
            Error::new(
                ErrorClass::Reference,
                E_FFT_INPUT,
                "frequency.ifft2 requires a `spectrum` value".to_owned(),
            )
        })?;
        let ResourceDescriptor::Spectrum(d) = *spectrum.descriptor() else {
            return Err(Error::new(
                ErrorClass::Type,
                E_FFT_INPUT,
                "frequency.ifft2 `spectrum` input must be a Spectrum resource".to_owned(),
            ));
        };
        d.validate()?;
        let samples = inverse_real(
            spectrum.samples(),
            d.extent.width as usize,
            d.extent.height as usize,
            d.channels as usize,
        );
        let descriptor = if d.channels == 1 {
            reconstructed_field(&d)
        } else {
            reconstructed_image(&d)
        };
        let value = ResourceValue::new(descriptor, d.channels, samples).map_err(|actual| {
            Error::new(
                ErrorClass::Execution,
                E_FFT_INPUT,
                format!("frequency.ifft2 produced a buffer of unexpected length {actual}"),
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

/// Verification declarations shared by `fft2`/`ifft2`: bounded transforms with
/// analytic fixtures (round-trip, pure-DC constant, known-sinusoid peak) and
/// property tests (determinism, channel independence). Differential applies in
/// principle (a single reference today, so declared not-applicable with a
/// reason); perceptual does not apply — correctness is the reconstruction /
/// spectral-peak property set, not a perceptual metric.
fn fft_test_metadata(perceptual_reason: &str) -> TestMetadata {
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

#[cfg(test)]
mod tests;
