//! The `image.flip@1` operation: an exact axis reflection (`OP_CATALOG` §5,
//! `AGENT_VERIFICATION` §2.5).
//!
//! `image.flip` mirrors an image about the horizontal axis (`vertical` flip,
//! top↔bottom), the vertical axis (`horizontal` flip, left↔right), or both. It is
//! a pure integer pixel remap — no resampling, no interpolation — so every output
//! sample is a verbatim copy of exactly one input sample and the op is **exact**.
//!
//! # Axis
//!
//! - **`horizontal`** — reflect left↔right: `out(x, y) = in(W-1-x, y)`.
//! - **`vertical`** — reflect top↔bottom: `out(x, y) = in(x, H-1-y)`.
//! - **`both`** — both reflections (equivalently a 180° rotation):
//!   `out(x, y) = in(W-1-x, H-1-y)`.
//!
//! The extent is preserved. A flip is an **involution**: applying the same flip
//! twice is the exact identity, the keystone metamorphic property (§2.5
//! reflection covariance) this op and the metamorphic harness rely on.
//!
//! # Determinism
//!
//! [`Exact`](DeterminismTier::Exact): a bijective integer remap copying samples
//! verbatim, bit-identical on every run.

use paintop_core::executor::{InputValues, OpImplementation, OutputValues, ResourceValue};
use paintop_ir::{
    AssertionResult, Descriptors, DeterminismTier, Error, ErrorClass, ErrorContext, Extent,
    ImageDescriptor, ImplId, InputRegions, InputSpec, OpContract, OperationManifest,
    OutputDescriptors, OutputRegions, OutputSpec, ParamSpec, ParamType, Rect, ResourceDescriptor,
    ResourceKind, Result, RoiCategory, RoiPolicy, TestMetadata,
};

/// The canonical id of the image-flip operation.
pub const FLIP_OP_ID: &str = "image.flip@1";

/// The `image` input was absent or carried a non-image descriptor.
pub const E_FLIP_INPUT: &str = "E_FLIP_INPUT";

/// The `axis` parameter was missing or not one of the known axes.
pub const E_FLIP_AXIS: &str = "E_FLIP_AXIS";

/// Which axis (or axes) the reflection mirrors about.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Axis {
    /// Reflect left↔right (about the vertical axis).
    Horizontal,
    /// Reflect top↔bottom (about the horizontal axis).
    Vertical,
    /// Reflect both (a 180° turn).
    Both,
}

impl Axis {
    /// Parse the axis from its wire token.
    fn from_token(token: &str) -> Option<Self> {
        match token {
            "horizontal" => Some(Self::Horizontal),
            "vertical" => Some(Self::Vertical),
            "both" => Some(Self::Both),
            _ => None,
        }
    }

    /// Whether the horizontal (left↔right) reflection is active.
    const fn flips_x(self) -> bool {
        matches!(self, Self::Horizontal | Self::Both)
    }

    /// Whether the vertical (top↔bottom) reflection is active.
    const fn flips_y(self) -> bool {
        matches!(self, Self::Vertical | Self::Both)
    }
}

/// Parse the required `axis` param.
fn axis_param(params: &serde_json::Value) -> Result<Axis> {
    let value = params.get("axis").ok_or_else(|| {
        Error::new(
            ErrorClass::Schema,
            E_FLIP_AXIS,
            "image.flip requires an `axis` parameter (horizontal | vertical | both)".to_owned(),
        )
    })?;
    let token = value.as_str().ok_or_else(|| {
        Error::new(
            ErrorClass::Schema,
            E_FLIP_AXIS,
            "image.flip `axis` must be a string".to_owned(),
        )
        .with_context(ErrorContext::default().with_actual(value.to_string()))
    })?;
    Axis::from_token(token).ok_or_else(|| {
        Error::new(
            ErrorClass::Schema,
            E_FLIP_AXIS,
            format!("image.flip `axis` is not a known axis: {token}"),
        )
        .with_context(
            ErrorContext::default()
                .with_actual(token.to_owned())
                .with_expected("horizontal | vertical | both"),
        )
    })
}

/// Remap the interleaved `samples` of an `extent` image under the flip `axis`.
fn flip_samples(samples: &[f32], extent: Extent, axis: Axis, channels: u32) -> Vec<f32> {
    let stride = channels as usize;
    let w = extent.width as usize;
    let h = extent.height as usize;
    let mut out = vec![0.0; samples.len()];
    for y in 0..h {
        let sy = if axis.flips_y() { h - 1 - y } else { y };
        for x in 0..w {
            let sx = if axis.flips_x() { w - 1 - x } else { x };
            let dst = (y * w + x) * stride;
            let src = (sy * w + sx) * stride;
            out[dst..dst + stride].copy_from_slice(&samples[src..src + stride]);
        }
    }
    out
}

/// The `image.flip@1` operation: an image + axis → the reflected image.
#[derive(Debug, Clone, Copy, Default)]
pub struct Flip;

impl Flip {
    /// Construct the image-flip operation.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    /// The declared manifest for `image.flip@1`.
    ///
    /// # Errors
    /// Propagates the [`schema`](ErrorClass::Schema) error if the hard-coded op /
    /// impl ids are somehow invalid (they are not).
    pub fn manifest() -> Result<OperationManifest> {
        Ok(OperationManifest {
            id: FLIP_OP_ID.parse()?,
            impl_version: 1,
            summary: "Exact axis reflection (horizontal / vertical / both) as a bijective integer \
                      pixel remap; no resampling. Involution: flipping twice is the identity."
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
                doc: "The image to flip.".to_owned(),
            }],
            outputs: vec![OutputSpec {
                name: "image".to_owned(),
                kind: ResourceKind::Image,
                doc: "The reflected image (same extent and descriptor as the input).".to_owned(),
            }],
            params: vec![ParamSpec {
                name: "axis".to_owned(),
                ty: ParamType::String,
                unit: None,
                required: true,
                default: None,
                choices: vec![
                    "horizontal".to_owned(),
                    "vertical".to_owned(),
                    "both".to_owned(),
                ],
                doc:
                    "Reflection axis: horizontal (left<->right), vertical (top<->bottom), or both."
                        .to_owned(),
            }],
            implementations: vec![reference_impl()?],
            test: flip_test_metadata(),
        })
    }
}

impl OpContract for Flip {
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
        // Validate the axis at infer time so a bad request fails type-checking.
        axis_param(params)?;
        // A flip preserves the descriptor exactly (extent unchanged).
        let mut out = OutputDescriptors::new();
        out.insert("image".to_owned(), ResourceDescriptor::Image(*descriptor));
        Ok(out)
    }

    fn required_inputs(
        &self,
        requested_outputs: &OutputRegions,
        inputs: &Descriptors,
        params: &serde_json::Value,
    ) -> Result<InputRegions> {
        // Geometric: the output region reflects to the mirror-image input region.
        let descriptor = image_descriptor(inputs)?;
        let axis = axis_param(params)?;
        let w = i64::from(descriptor.extent.width);
        let h = i64::from(descriptor.extent.height);
        let mut regions = InputRegions::new();
        if let Some(r) = requested_outputs.get("image") {
            let (x0, x1) = if axis.flips_x() {
                (w - r.x1, w - r.x0)
            } else {
                (r.x0, r.x1)
            };
            let (y0, y1) = if axis.flips_y() {
                (h - r.y1, h - r.y0)
            } else {
                (r.y0, r.y1)
            };
            regions.insert("image".to_owned(), Rect::new(x0, y0, x1, y1));
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

impl OpImplementation for Flip {
    fn compute(
        &self,
        inputs: &InputValues,
        params: &serde_json::Value,
    ) -> std::result::Result<OutputValues, Error> {
        let image = inputs.get("image").ok_or_else(|| {
            Error::new(
                ErrorClass::Reference,
                E_FLIP_INPUT,
                "image.flip requires an `image` input value".to_owned(),
            )
        })?;
        let ResourceDescriptor::Image(descriptor) = image.descriptor() else {
            return Err(Error::new(
                ErrorClass::Type,
                E_FLIP_INPUT,
                "image.flip `image` input must be an image resource".to_owned(),
            ));
        };
        let axis = axis_param(params)?;
        let samples = flip_samples(image.samples(), descriptor.extent, axis, image.channels());

        let value = ResourceValue::new(
            ResourceDescriptor::Image(*descriptor),
            image.channels(),
            samples,
        )
        .map_err(|actual| {
            Error::new(
                ErrorClass::Execution,
                E_FLIP_INPUT,
                format!("image.flip produced a sample buffer of unexpected length {actual}"),
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
            E_FLIP_INPUT,
            "image.flip requires an `image` input".to_owned(),
        )
    })?;
    let ResourceDescriptor::Image(descriptor) = image else {
        return Err(Error::new(
            ErrorClass::Type,
            E_FLIP_INPUT,
            "image.flip `image` input must be an image resource".to_owned(),
        ));
    };
    Ok(descriptor)
}

/// The mandatory `cpu.reference@1` oracle implementation id.
fn reference_impl() -> Result<ImplId> {
    ImplId::new("cpu", "reference", 1)
}

/// Verification declarations for `image.flip@1`: an exact, single-reference,
/// geometric integer remap. Differential does not apply (one implementation).
/// Perceptual is not applicable: a flip copies samples verbatim and is verified by
/// exact correspondence fixtures and the double-flip involution, not a perceptual
/// metric.
fn flip_test_metadata() -> TestMetadata {
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
            "image.flip is a bijective integer pixel remap copying samples verbatim; correctness \
             is verified by exact pixel-correspondence fixtures and the double-flip involution \
             identity, not a perceptual-quality metric",
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
