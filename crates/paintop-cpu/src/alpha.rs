//! The `alpha.premultiply@1` and `alpha.unpremultiply@1` operations.
//!
//! They convert a color image between *straight* (unassociated) and
//! *premultiplied* (associated) alpha, **in linear light** (`OP_CATALOG` §2,
//! `plan.md` §8.2, `AGENT_VERIFICATION` §3.2).
//!
//! Compositing happens with premultiplied alpha; unassociated alpha is converted
//! to premultiplied *explicitly* at graph boundaries (`plan.md` §8.2). These two
//! ops are that boundary: they are mutual inverses for every pixel whose alpha is
//! above an explicit epsilon.
//!
//! # Semantics
//!
//! For a color sample `C` and coverage `α`:
//!
//! ```text
//! premultiply   (straight -> premultiplied):   C' = C * α
//! unpremultiply (premultiplied -> straight):   C  = C' / α   (α > ε)
//! ```
//!
//! Both are **pointwise**: each output sample depends only on the co-located input
//! sample. The alpha channel itself is carried through unchanged — it is coverage,
//! not color. Only the color channels are scaled.
//!
//! ## Linear-light rule
//!
//! Premultiplication is a linear-light operation: scaling sRGB-encoded values by
//! coverage is meaningless (the encoding is nonlinear), so a premultiply request
//! on a `srgb`-encoded image is **rejected** with a [`semantic`](ErrorClass::Semantic)
//! error rather than silently producing wrong color (`plan.md` §8.2, the §8 rule).
//! The same rule applies to `unpremultiply`: it only un-associates linear color.
//!
//! ## Hidden RGB and the near-zero-alpha policy
//!
//! A *straight* image may carry non-zero "hidden" color beneath a transparent
//! pixel (`α = 0`). Premultiplying it yields `C' = C * 0 = 0`: the hidden color is
//! intentionally collapsed to black under zero coverage, which is exactly what
//! prevents a **colored fringe** when the premultiplied image is composited or
//! exported (`AGENT_VERIFICATION` §3.2). This is not "silently discarding" color:
//! premultiplied black-under-zero-coverage *is* the correct representation of an
//! invisible pixel.
//!
//! Unpremultiply is the inverse only where `α > ε`. At or below `ε` the division
//! `C' / α` is numerically unrecoverable (a premultiplied transparent pixel stores
//! `C' = 0`, so the original straight color is genuinely lost). The explicit
//! policy here is to **leave the color channels at zero** for `α <= ε` rather than
//! divide by a near-zero number and manufacture an enormous, arbitrary color. The
//! epsilon is a fixed, documented constant ([`UNPREMULTIPLY_EPSILON`]).

use paintop_core::executor::{InputValues, OpImplementation, OutputValues, ResourceValue};
use paintop_ir::{
    AlphaRepresentation, AssertionResult, ColorEncoding, Descriptors, DeterminismTier, Error,
    ErrorClass, ErrorContext, ImageDescriptor, ImplId, InputRegions, InputSpec, OpContract,
    OperationManifest, OutputDescriptors, OutputRegions, OutputSpec, ResourceDescriptor,
    ResourceKind, Result, RoiCategory, RoiPolicy, TestMetadata,
};

/// The canonical id of the premultiply operation.
pub const PREMULTIPLY_OP_ID: &str = "alpha.premultiply@1";

/// The canonical id of the unpremultiply operation.
pub const UNPREMULTIPLY_OP_ID: &str = "alpha.unpremultiply@1";

/// The `image` input was absent or carried a non-image descriptor.
pub const E_ALPHA_INPUT: &str = "E_ALPHA_INPUT";

/// The input image has no alpha channel, so there is nothing to (un)associate.
pub const E_ALPHA_NO_ALPHA: &str = "E_ALPHA_NO_ALPHA";

/// The input image is already in the representation the op would produce, or is
/// encoded in a nonlinear (display) color space where premultiplication is not
/// defined.
pub const E_ALPHA_REPRESENTATION: &str = "E_ALPHA_REPRESENTATION";

/// The op produced a sample buffer whose length disagrees with its descriptor.
pub const E_ALPHA_BUFFER: &str = "E_ALPHA_BUFFER";

/// The near-zero-alpha policy threshold for [`Unpremultiply`].
///
/// At or below this coverage the original straight color is numerically
/// unrecoverable, so the color channels are left at zero rather than divided by a
/// near-zero alpha.
pub const UNPREMULTIPLY_EPSILON: f32 = 1.0e-6;

/// The direction an alpha op moves the representation in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Direction {
    /// `straight -> premultiplied` (`C' = C * α`).
    Premultiply,
    /// `premultiplied -> straight` (`C = C' / α`).
    Unpremultiply,
}

impl Direction {
    /// The alpha representation the input must already be in.
    const fn from(self) -> AlphaRepresentation {
        match self {
            Self::Premultiply => AlphaRepresentation::Straight,
            Self::Unpremultiply => AlphaRepresentation::Premultiplied,
        }
    }

    /// The alpha representation the output records.
    const fn to(self) -> AlphaRepresentation {
        match self {
            Self::Premultiply => AlphaRepresentation::Premultiplied,
            Self::Unpremultiply => AlphaRepresentation::Straight,
        }
    }

    /// The op id this direction declares.
    #[cfg(test)]
    const fn op_id(self) -> &'static str {
        match self {
            Self::Premultiply => PREMULTIPLY_OP_ID,
            Self::Unpremultiply => UNPREMULTIPLY_OP_ID,
        }
    }

    /// A short verb for diagnostics.
    const fn verb(self) -> &'static str {
        match self {
            Self::Premultiply => "alpha.premultiply",
            Self::Unpremultiply => "alpha.unpremultiply",
        }
    }
}

/// Validate that an image descriptor may be (un)premultiplied in this direction,
/// returning the resulting output descriptor.
///
/// The check is shared by `infer_outputs` and the compute kernel so a graph and a
/// run never disagree on what is rejected.
fn check_and_retarget(
    descriptor: &ImageDescriptor,
    direction: Direction,
) -> Result<ImageDescriptor> {
    // Premultiplication is only defined in linear light: scaling a nonlinear sRGB
    // value by coverage is meaningless. Reject display-encoded input outright.
    if descriptor.color == ColorEncoding::Srgb {
        return Err(Error::new(
            ErrorClass::Semantic,
            E_ALPHA_REPRESENTATION,
            format!(
                "{} requires linear-light color; the input is `srgb`-encoded. \
                 Insert a color.convert to linear-srgb first.",
                direction.verb()
            ),
        )
        .with_context(
            ErrorContext::default()
                .with_actual("srgb")
                .with_expected("linear-srgb | raw-linear"),
        ));
    }

    // There must be an alpha channel to associate with the color.
    if !descriptor.layout.has_alpha() {
        return Err(Error::new(
            ErrorClass::Semantic,
            E_ALPHA_NO_ALPHA,
            format!(
                "{} requires an image with an alpha channel (GrayA or Rgba)",
                direction.verb()
            ),
        ));
    }

    // The input must already be in the source representation, so the op can never
    // double-premultiply (or double-divide) silently.
    if descriptor.alpha != direction.from() {
        return Err(Error::new(
            ErrorClass::Semantic,
            E_ALPHA_REPRESENTATION,
            format!(
                "{} expects {:?} alpha but the input is {:?}",
                direction.verb(),
                direction.from(),
                descriptor.alpha
            ),
        )
        .with_context(
            ErrorContext::default()
                .with_actual(format!("{:?}", descriptor.alpha))
                .with_expected(format!("{:?}", direction.from())),
        ));
    }

    let mut out = *descriptor;
    out.alpha = direction.to();
    Ok(out)
}

/// Apply the per-pixel (un)premultiply to an interleaved color+alpha buffer.
///
/// `channels` is the interleaved sample count per pixel; the last channel is the
/// alpha (coverage) and is passed through unchanged. The color channels are scaled
/// by alpha (premultiply) or divided by it where `α > ε` (unpremultiply).
fn apply(samples: &[f32], channels: u32, direction: Direction) -> Vec<f32> {
    let channel_count = channels as usize;
    if channel_count == 0 {
        return samples.to_vec();
    }
    let alpha_index = channel_count - 1;
    let mut out = samples.to_vec();
    for pixel in out.chunks_mut(channel_count) {
        // `chunks_mut` yields exactly `channel_count`-wide slices for a buffer
        // whose length is a multiple of `channel_count` (guaranteed by the
        // descriptor's length invariant); a short final chunk simply has no alpha
        // to read and is left untouched.
        let Some(&alpha) = pixel.get(alpha_index) else {
            continue;
        };
        for color in &mut pixel[..alpha_index] {
            *color = match direction {
                Direction::Premultiply => *color * alpha,
                Direction::Unpremultiply => {
                    if alpha > UNPREMULTIPLY_EPSILON {
                        *color / alpha
                    } else {
                        // Near-zero coverage: the straight color is unrecoverable.
                        // Leave it at zero rather than divide by ~0.
                        0.0
                    }
                }
            };
        }
    }
    out
}

/// Shared `infer_outputs` for both directions.
fn infer(direction: Direction, inputs: &Descriptors) -> Result<OutputDescriptors> {
    let image = inputs.get("image").ok_or_else(|| {
        Error::new(
            ErrorClass::Reference,
            E_ALPHA_INPUT,
            format!("{} requires an `image` input", direction.verb()),
        )
    })?;
    let ResourceDescriptor::Image(descriptor) = image else {
        return Err(Error::new(
            ErrorClass::Type,
            E_ALPHA_INPUT,
            format!(
                "{} `image` input must be an image resource",
                direction.verb()
            ),
        ));
    };
    let out_descriptor = check_and_retarget(descriptor, direction)?;
    let mut out = OutputDescriptors::new();
    out.insert(
        "image".to_owned(),
        ResourceDescriptor::Image(out_descriptor),
    );
    Ok(out)
}

/// Shared pointwise ROI: each output sample needs exactly the co-located input.
fn pointwise_inputs(requested_outputs: &OutputRegions) -> InputRegions {
    let mut regions = InputRegions::new();
    if let Some(region) = requested_outputs.get("image") {
        regions.insert("image".to_owned(), *region);
    }
    regions
}

/// Shared postcondition: the output records the target alpha representation.
fn target_postcondition(direction: Direction, outputs: &OutputDescriptors) -> Vec<AssertionResult> {
    let Some(ResourceDescriptor::Image(out)) = outputs.get("image") else {
        return vec![AssertionResult::fail(
            "produces_image",
            "no `image` output produced",
        )];
    };
    vec![if out.alpha == direction.to() {
        AssertionResult::pass("records_target_alpha")
    } else {
        AssertionResult::fail(
            "records_target_alpha",
            format!(
                "output alpha {:?} does not match target {:?}",
                out.alpha,
                direction.to()
            ),
        )
    }]
}

/// Shared compute kernel for both directions.
fn compute(direction: Direction, inputs: &InputValues) -> std::result::Result<OutputValues, Error> {
    let image = inputs.get("image").ok_or_else(|| {
        Error::new(
            ErrorClass::Reference,
            E_ALPHA_INPUT,
            format!("{} requires an `image` input value", direction.verb()),
        )
    })?;
    let ResourceDescriptor::Image(descriptor) = image.descriptor() else {
        return Err(Error::new(
            ErrorClass::Type,
            E_ALPHA_INPUT,
            format!(
                "{} `image` input must be an image resource",
                direction.verb()
            ),
        ));
    };

    let out_descriptor = check_and_retarget(descriptor, direction)?;
    let samples = apply(image.samples(), image.channels(), direction);

    let value = ResourceValue::new(
        ResourceDescriptor::Image(out_descriptor),
        image.channels(),
        samples,
    )
    .map_err(|actual| {
        Error::new(
            ErrorClass::Execution,
            E_ALPHA_BUFFER,
            format!(
                "{} produced a sample buffer of unexpected length {actual}",
                direction.verb()
            ),
        )
    })?;

    let mut out = OutputValues::new();
    out.insert("image".to_owned(), value);
    Ok(out)
}

/// Build a manifest shared by both directions (they differ only in id, summary,
/// determinism tier, and the verification rationale).
fn manifest_for(
    op_id: &str,
    summary: &str,
    determinism: DeterminismTier,
    test: TestMetadata,
) -> Result<OperationManifest> {
    Ok(OperationManifest {
        id: op_id.parse()?,
        impl_version: 1,
        summary: summary.to_owned(),
        determinism,
        roi: RoiPolicy {
            category: RoiCategory::Pointwise,
            halo_px: None,
        },
        inputs: vec![InputSpec {
            name: "image".to_owned(),
            kind: ResourceKind::Image,
            required: true,
            doc: "The linear-light color image with an alpha channel.".to_owned(),
        }],
        outputs: vec![OutputSpec {
            name: "image".to_owned(),
            kind: ResourceKind::Image,
            doc: "The image with its alpha representation flipped (same extent/layout).".to_owned(),
        }],
        params: vec![],
        implementations: vec![reference_impl()?],
        test,
    })
}

/// The mandatory `cpu.reference@1` oracle implementation id.
fn reference_impl() -> Result<ImplId> {
    ImplId::new("cpu", "reference", 1)
}

/// The verification declarations shared by both alpha ops: pointwise, single
/// reference implementation, no perceptual metric. Differential does not apply
/// (one implementation). Perceptual is not applicable: these are closed-form
/// arithmetic transforms verified by an analytic round-trip and the hidden-RGB
/// fringe fixture, not a perceptual-quality comparison.
fn alpha_test_metadata(rationale: &str) -> TestMetadata {
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
        CategoryStatus::not_applicable(rationale),
    );
    TestMetadata {
        has_analytic_reference: true,
        has_property_tests: true,
        golden_fixtures: vec![],
        not_applicable_reason: String::new(),
        verification: decls,
    }
}

/// The `alpha.premultiply@1` operation: straight color → premultiplied color.
#[derive(Debug, Clone, Copy, Default)]
pub struct Premultiply;

impl Premultiply {
    /// Construct the premultiply operation.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    /// The declared manifest for `alpha.premultiply@1`.
    ///
    /// # Errors
    /// Propagates the [`schema`](ErrorClass::Schema) error if the hard-coded op /
    /// impl ids are somehow invalid (they are not).
    pub fn manifest() -> Result<OperationManifest> {
        manifest_for(
            PREMULTIPLY_OP_ID,
            "Premultiply a straight-alpha linear color image (C' = C * alpha); alpha passes \
             through. Rejected on nonlinear (srgb) input.",
            // C' = C * alpha is a bit-exact multiply for the declared scalar.
            DeterminismTier::Exact,
            alpha_test_metadata(
                "alpha.premultiply is a closed-form per-channel multiply by coverage, verified by \
                 an analytic round-trip and a hidden-RGB fringe fixture; there is no \
                 perceptual-quality metric to apply",
            ),
        )
    }
}

impl OpContract for Premultiply {
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
        infer(Direction::Premultiply, inputs)
    }

    fn required_inputs(
        &self,
        requested_outputs: &OutputRegions,
        _inputs: &Descriptors,
        _params: &serde_json::Value,
    ) -> Result<InputRegions> {
        Ok(pointwise_inputs(requested_outputs))
    }

    fn validate_postconditions(
        &self,
        outputs: &OutputDescriptors,
        _params: &serde_json::Value,
    ) -> Result<Vec<AssertionResult>> {
        Ok(target_postcondition(Direction::Premultiply, outputs))
    }
}

impl OpImplementation for Premultiply {
    fn compute(
        &self,
        inputs: &InputValues,
        _params: &serde_json::Value,
    ) -> std::result::Result<OutputValues, Error> {
        compute(Direction::Premultiply, inputs)
    }
}

/// The `alpha.unpremultiply@1` operation: premultiplied color → straight color.
#[derive(Debug, Clone, Copy, Default)]
pub struct Unpremultiply;

impl Unpremultiply {
    /// Construct the unpremultiply operation.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    /// The declared manifest for `alpha.unpremultiply@1`.
    ///
    /// # Errors
    /// Propagates the [`schema`](ErrorClass::Schema) error if the hard-coded op /
    /// impl ids are somehow invalid (they are not).
    pub fn manifest() -> Result<OperationManifest> {
        manifest_for(
            UNPREMULTIPLY_OP_ID,
            "Unpremultiply a premultiplied linear color image (C = C' / alpha where alpha > eps); \
             alpha passes through. Rejected on nonlinear (srgb) input.",
            // The near-zero-alpha policy makes this irreversible below epsilon, so
            // it is declared bounded rather than exact (OP_CATALOG §2).
            DeterminismTier::Bounded,
            alpha_test_metadata(
                "alpha.unpremultiply is a closed-form per-channel divide by coverage with an \
                 explicit near-zero-alpha policy, verified by an analytic round-trip and a \
                 hidden-RGB fringe fixture; there is no perceptual-quality metric to apply",
            ),
        )
    }
}

impl OpContract for Unpremultiply {
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
        infer(Direction::Unpremultiply, inputs)
    }

    fn required_inputs(
        &self,
        requested_outputs: &OutputRegions,
        _inputs: &Descriptors,
        _params: &serde_json::Value,
    ) -> Result<InputRegions> {
        Ok(pointwise_inputs(requested_outputs))
    }

    fn validate_postconditions(
        &self,
        outputs: &OutputDescriptors,
        _params: &serde_json::Value,
    ) -> Result<Vec<AssertionResult>> {
        Ok(target_postcondition(Direction::Unpremultiply, outputs))
    }
}

impl OpImplementation for Unpremultiply {
    fn compute(
        &self,
        inputs: &InputValues,
        _params: &serde_json::Value,
    ) -> std::result::Result<OutputValues, Error> {
        compute(Direction::Unpremultiply, inputs)
    }
}

#[cfg(test)]
mod tests;
