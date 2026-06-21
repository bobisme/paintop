//! The `color.convert@1` operation: transfer-function decode/encode between
//! `srgb` and `linear-srgb` (`OP_CATALOG` §2, `plan.md` §8.2,
//! `AGENT_VERIFICATION` §2.5).
//!
//! `color.convert` re-encodes a color image's *transfer function* between the
//! sRGB display encoding and linear-light sRGB, with **explicit** source and
//! destination encodings (the `from` / `to` params). It is the boundary op the
//! agent uses to move an image into the linear space color math happens in, and
//! back out for display.
//!
//! # Semantics
//!
//! The conversion applies the standard sRGB electro-optical transfer function
//! (IEC 61966-2-1) to the **color** channels only; an alpha channel, when
//! present, is passed through untouched (alpha is a coverage value, not a color,
//! so it carries no transfer function). The piecewise transfer functions are:
//!
//! ```text
//! decode (srgb -> linear):   c <= 0.04045 ? c/12.92 : ((c+0.055)/1.055)^2.4
//! encode (linear -> srgb):   c <= 0.0031308 ? c*12.92 : 1.055*c^(1/2.4) - 0.055
//! ```
//!
//! Both branches are continuous at the knot and the pair is a mutual inverse, so
//! `encode ∘ decode` (and `decode ∘ encode`) round-trips to within floating-point
//! tolerance. A `from == to` conversion is the identity. The op is **pointwise**
//! (each output sample depends only on the co-located input sample) and
//! **bounded** determinism: the encode/decode use `powf`, whose last bit is not
//! guaranteed identical across platforms, so equality is asserted within a
//! tolerance rather than bit-exactly.
//!
//! # Rejected requests
//!
//! - `display-p3` and `icc` are *nameable* (so a plan can request them) but
//!   **unsupported**: there is no backend, so a request is rejected with a
//!   [`semantic`](ErrorClass::Semantic) error rather than silently approximated
//!   (`plan.md` §8.2).
//! - `raw-linear` is material scalar data with no color transfer function;
//!   converting it to/from a color encoding (or vice versa) is a category error
//!   and is rejected as [`semantic`](ErrorClass::Semantic). A `raw-linear ->
//!   raw-linear` request is a no-op identity and is allowed.

use paintop_core::executor::{InputValues, OpImplementation, OutputValues, ResourceValue};
use paintop_ir::{
    AssertionResult, ColorEncoding, Descriptors, DeterminismTier, Error, ErrorClass, ErrorContext,
    ImageDescriptor, ImplId, InputRegions, InputSpec, OpContract, OperationManifest,
    OutputDescriptors, OutputRegions, OutputSpec, ParamSpec, ParamType, RequestedColorEncoding,
    ResourceDescriptor, ResourceKind, Result, RoiCategory, RoiPolicy, TestMetadata,
};

/// The canonical id of the color-convert operation.
pub const CONVERT_OP_ID: &str = "color.convert@1";

/// A required `from` / `to` encoding parameter was missing or not a recognized
/// encoding token.
pub const E_CONVERT_PARAM: &str = "E_CONVERT_PARAM";

/// The `image` input to convert was absent or carried a non-image descriptor.
pub const E_CONVERT_INPUT: &str = "E_CONVERT_INPUT";

/// A conversion mixed a color transfer function with `raw-linear` material data,
/// or otherwise requested a transfer the op cannot honor.
pub const E_CONVERT_UNSUPPORTED: &str = "E_CONVERT_UNSUPPORTED";

/// The sRGB decode knot (`srgb -> linear`): below this the function is linear.
const DECODE_KNOT: f32 = 0.040_45;
/// The sRGB encode knot (`linear -> srgb`): below this the function is linear.
const ENCODE_KNOT: f32 = 0.003_130_8;

/// Decode one sRGB-encoded sample to linear light (IEC 61966-2-1).
#[must_use]
fn srgb_decode(c: f32) -> f32 {
    if c <= DECODE_KNOT {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

/// Encode one linear-light sample to the sRGB transfer function
/// (IEC 61966-2-1).
#[must_use]
fn srgb_encode(c: f32) -> f32 {
    if c <= ENCODE_KNOT {
        c * 12.92
    } else {
        1.055_f32.mul_add(c.powf(1.0 / 2.4), -0.055)
    }
}

/// The two supported color transfer encodings this op converts between, plus the
/// `raw-linear` material passthrough.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Encoding {
    /// The sRGB display transfer function.
    Srgb,
    /// Linear-light sRGB primaries.
    LinearSrgb,
    /// Raw linear material data (no color transfer function).
    RawLinear,
}

impl Encoding {
    /// Resolve a string param token to a supported [`Encoding`].
    ///
    /// The token is parsed through [`RequestedColorEncoding`] so the *nameable
    /// but unsupported* encodings (`display-p3`, `icc`) are rejected with the
    /// central [`semantic`](ErrorClass::Semantic) taxonomy rather than treated as
    /// "unknown token".
    fn parse(token: &str, side: &str) -> Result<Self> {
        let requested: RequestedColorEncoding =
            serde_json::from_value(serde_json::Value::String(token.to_owned())).map_err(|_| {
                Error::new(
                    ErrorClass::Schema,
                    E_CONVERT_PARAM,
                    format!("color.convert `{side}` is not a known color encoding: `{token}`"),
                )
                .with_context(
                    ErrorContext::default()
                        .with_actual(token)
                        .with_expected("srgb | linear-srgb | raw-linear"),
                )
            })?;
        // `resolve` rejects display-p3 / icc as semantic E_UNSUPPORTED_COLOR_ENCODING.
        match requested.resolve()? {
            ColorEncoding::Srgb => Ok(Self::Srgb),
            ColorEncoding::LinearSrgb => Ok(Self::LinearSrgb),
            ColorEncoding::RawLinear => Ok(Self::RawLinear),
            // `ColorEncoding` is `#[non_exhaustive]`: a future supported encoding
            // this op has no transfer function for is rejected rather than
            // silently mishandled.
            other => Err(Error::new(
                ErrorClass::Semantic,
                E_CONVERT_UNSUPPORTED,
                format!("color.convert `{side}` encoding {other:?} has no supported transfer"),
            )),
        }
    }

    /// The settled [`ColorEncoding`] this resolves to, recorded on the output
    /// descriptor.
    const fn settled(self) -> ColorEncoding {
        match self {
            Self::Srgb => ColorEncoding::Srgb,
            Self::LinearSrgb => ColorEncoding::LinearSrgb,
            Self::RawLinear => ColorEncoding::RawLinear,
        }
    }
}

/// A resolved `from -> to` conversion: the parsed encodings and the per-sample
/// transform they imply.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Conversion {
    from: Encoding,
    to: Encoding,
}

impl Conversion {
    /// Parse the `from` / `to` params and reject any unsupported pairing.
    ///
    /// # Errors
    /// - [`schema`](ErrorClass::Schema) / [`E_CONVERT_PARAM`] if a param is
    ///   missing or not a known encoding token.
    /// - [`semantic`](ErrorClass::Semantic) / `E_UNSUPPORTED_COLOR_ENCODING` for a
    ///   `display-p3` / `icc` request.
    /// - [`semantic`](ErrorClass::Semantic) / [`E_CONVERT_UNSUPPORTED`] when the
    ///   conversion crosses `raw-linear` and a color encoding (a category error).
    fn resolve(params: &serde_json::Value) -> Result<Self> {
        let from = Encoding::parse(&string_param(params, "from")?, "from")?;
        let to = Encoding::parse(&string_param(params, "to")?, "to")?;

        // raw-linear is material data with no transfer function. It may only be
        // "converted" to itself (an identity passthrough); mixing it with a color
        // encoding is a semantic category error, not a transfer.
        let mixes_raw_linear = (from == Encoding::RawLinear) != (to == Encoding::RawLinear);
        if mixes_raw_linear {
            return Err(Error::new(
                ErrorClass::Semantic,
                E_CONVERT_UNSUPPORTED,
                "color.convert cannot apply a transfer function between `raw-linear` material \
                 data and a color encoding; raw-linear has no color transfer function"
                    .to_owned(),
            )
            .with_context(
                ErrorContext::default()
                    .with_actual(format!("{from:?} -> {to:?}"))
                    .with_expected("srgb <-> linear-srgb, or an identity conversion"),
            ));
        }

        Ok(Self { from, to })
    }

    /// The per-color-channel transform for this conversion: `None` for an
    /// identity (`from == to`), else the decode/encode function.
    fn transform(self) -> Option<fn(f32) -> f32> {
        match (self.from, self.to) {
            (Encoding::Srgb, Encoding::LinearSrgb) => Some(srgb_decode),
            (Encoding::LinearSrgb, Encoding::Srgb) => Some(srgb_encode),
            // from == to (incl. raw-linear -> raw-linear): identity.
            _ => None,
        }
    }
}

/// Extract a required string parameter, erroring if absent or non-string.
fn string_param(params: &serde_json::Value, name: &str) -> Result<String> {
    params
        .get(name)
        .and_then(serde_json::Value::as_str)
        .map(str::to_owned)
        .ok_or_else(|| {
            Error::new(
                ErrorClass::Schema,
                E_CONVERT_PARAM,
                format!("color.convert requires a string `{name}` encoding parameter"),
            )
        })
}

/// Apply a transfer function to an image's interleaved samples, transforming
/// only the color channels and passing any alpha channel through.
///
/// `channels` is the interleaved sample count per pixel and `has_alpha` whether
/// the last channel is alpha (skipped by the transfer function). A `None`
/// `transform` is an identity conversion (`from == to`): every sample passes
/// through unchanged.
#[must_use]
fn convert_samples(
    samples: &[f32],
    channels: u32,
    has_alpha: bool,
    transform: Option<fn(f32) -> f32>,
) -> Vec<f32> {
    let channel_count = channels as usize;
    let Some(transform) = transform else {
        // Identity conversion: samples pass through unchanged.
        return samples.to_vec();
    };
    if channel_count == 0 {
        return samples.to_vec();
    }
    let alpha_index = if has_alpha {
        Some(channel_count - 1)
    } else {
        None
    };
    samples
        .iter()
        .enumerate()
        .map(|(i, &s)| {
            if Some(i % channel_count) == alpha_index {
                s
            } else {
                transform(s)
            }
        })
        .collect()
}

/// The `color.convert@1` operation: a color `Image` → a re-encoded color
/// `Image`.
#[derive(Debug, Clone, Copy, Default)]
pub struct Convert;

impl Convert {
    /// Construct the convert operation.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    /// The declared manifest for `color.convert@1`.
    ///
    /// # Errors
    /// Propagates the [`schema`](ErrorClass::Schema) error if the hard-coded op /
    /// impl ids are somehow invalid (they are not).
    pub fn manifest() -> Result<OperationManifest> {
        Ok(OperationManifest {
            id: CONVERT_OP_ID.parse()?,
            impl_version: 1,
            summary: "Decode/encode the color transfer function between srgb and linear-srgb \
                      with explicit source and destination encodings."
                .to_owned(),
            determinism: DeterminismTier::Bounded,
            roi: RoiPolicy {
                category: RoiCategory::Pointwise,
                halo_px: None,
            },
            inputs: vec![InputSpec {
                name: "image".to_owned(),
                kind: ResourceKind::Image,
                required: true,
                doc: "The color image to re-encode.".to_owned(),
            }],
            outputs: vec![OutputSpec {
                name: "image".to_owned(),
                kind: ResourceKind::Image,
                doc: "The re-encoded image (same extent/layout, target color encoding).".to_owned(),
            }],
            params: vec![
                encoding_param("from", "The source color encoding of the input image."),
                encoding_param("to", "The destination color encoding to produce."),
            ],
            implementations: vec![reference_impl()?, optimized_impl()?],
            test: convert_test_metadata(),
        })
    }
}

impl OpContract for Convert {
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
        let image = inputs.get("image").ok_or_else(|| {
            Error::new(
                ErrorClass::Reference,
                E_CONVERT_INPUT,
                "color.convert requires an `image` input".to_owned(),
            )
        })?;
        let ResourceDescriptor::Image(descriptor) = image else {
            return Err(Error::new(
                ErrorClass::Type,
                E_CONVERT_INPUT,
                "color.convert `image` input must be an image resource".to_owned(),
            ));
        };

        let conversion = Conversion::resolve(params)?;
        // The `from` declaration must agree with the input's recorded encoding,
        // so a plan cannot mislabel an already-decoded image and silently
        // double-decode it.
        if descriptor.color != conversion.from.settled() {
            return Err(Error::new(
                ErrorClass::Semantic,
                E_CONVERT_UNSUPPORTED,
                format!(
                    "color.convert `from` is {:?} but the input image is encoded as {:?}",
                    conversion.from.settled(),
                    descriptor.color
                ),
            )
            .with_context(
                ErrorContext::default()
                    .with_actual(format!("{:?}", descriptor.color))
                    .with_expected(format!("{:?}", conversion.from.settled())),
            ));
        }

        let mut out_descriptor: ImageDescriptor = *descriptor;
        out_descriptor.color = conversion.to.settled();

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
        // Pointwise: each output sample needs exactly the co-located input sample.
        let mut regions = InputRegions::new();
        if let Some(region) = requested_outputs.get("image") {
            regions.insert("image".to_owned(), *region);
        }
        Ok(regions)
    }

    fn validate_postconditions(
        &self,
        outputs: &OutputDescriptors,
        params: &serde_json::Value,
    ) -> Result<Vec<AssertionResult>> {
        let Some(ResourceDescriptor::Image(out)) = outputs.get("image") else {
            return Ok(vec![AssertionResult::fail(
                "produces_image",
                "no `image` output produced",
            )]);
        };
        // The output records the requested target encoding (the op's contract).
        let conversion = Conversion::resolve(params)?;
        let encoded_to_target = out.color == conversion.to.settled();
        Ok(vec![if encoded_to_target {
            AssertionResult::pass("encodes_to_target")
        } else {
            AssertionResult::fail(
                "encodes_to_target",
                format!(
                    "output encoding {:?} does not match requested target {:?}",
                    out.color,
                    conversion.to.settled()
                ),
            )
        }])
    }
}

/// The compute backend serving `color.convert`: the scalar reference oracle or the
/// autovectorization-friendly `cpu.optimized` kernel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Backend {
    /// The scalar reference oracle ([`convert_samples`]).
    Reference,
    /// The `cpu.optimized` transfer kernel ([`crate::optimized::kernels`]).
    Optimized,
}

/// Map a resolved conversion to the optimized kernel's transfer direction.
const fn kernel_transfer(conversion: Conversion) -> crate::optimized::kernels::Transfer {
    use crate::optimized::kernels::Transfer;
    match (conversion.from, conversion.to) {
        (Encoding::Srgb, Encoding::LinearSrgb) => Transfer::Decode,
        (Encoding::LinearSrgb, Encoding::Srgb) => Transfer::Encode,
        // from == to (incl. raw-linear -> raw-linear): identity passthrough.
        _ => Transfer::Identity,
    }
}

/// Shared compute for both backends: resolve and validate the conversion, then
/// transform the samples with the selected backend.
fn compute_backend(
    backend: Backend,
    inputs: &InputValues,
    params: &serde_json::Value,
) -> std::result::Result<OutputValues, Error> {
    let image = inputs.get("image").ok_or_else(|| {
        Error::new(
            ErrorClass::Reference,
            E_CONVERT_INPUT,
            "color.convert requires an `image` input value".to_owned(),
        )
    })?;
    let ResourceDescriptor::Image(descriptor) = image.descriptor() else {
        return Err(Error::new(
            ErrorClass::Type,
            E_CONVERT_INPUT,
            "color.convert `image` input must be an image resource".to_owned(),
        ));
    };

    let conversion = Conversion::resolve(params)?;
    if descriptor.color != conversion.from.settled() {
        return Err(Error::new(
            ErrorClass::Semantic,
            E_CONVERT_UNSUPPORTED,
            format!(
                "color.convert `from` is {:?} but the input image is encoded as {:?}",
                conversion.from.settled(),
                descriptor.color
            ),
        ));
    }

    let mut out_descriptor: ImageDescriptor = *descriptor;
    out_descriptor.color = conversion.to.settled();

    let has_alpha = descriptor.layout.has_alpha();
    let samples = match backend {
        Backend::Reference => convert_samples(
            image.samples(),
            image.channels(),
            has_alpha,
            conversion.transform(),
        ),
        Backend::Optimized => crate::optimized::kernels::color_convert(
            image.samples(),
            image.channels() as usize,
            has_alpha,
            kernel_transfer(conversion),
        ),
    };

    let value = ResourceValue::new(
        ResourceDescriptor::Image(out_descriptor),
        image.channels(),
        samples,
    )
    .map_err(|actual| {
        Error::new(
            ErrorClass::Execution,
            E_CONVERT_INPUT,
            format!("color.convert produced a sample buffer of unexpected length {actual}"),
        )
    })?;

    let mut out = OutputValues::new();
    out.insert("image".to_owned(), value);
    Ok(out)
}

impl OpImplementation for Convert {
    fn compute(
        &self,
        inputs: &InputValues,
        params: &serde_json::Value,
    ) -> std::result::Result<OutputValues, Error> {
        compute_backend(Backend::Reference, inputs, params)
    }
}

/// The `cpu.optimized@1` backend for `color.convert@1`.
///
/// It applies the same sRGB transfer function as the oracle via the
/// autovectorization-friendly kernel. `color.convert` is
/// [`Bounded`](DeterminismTier::Bounded) (the `powf` last bit varies), and the
/// kernel uses the identical transfer expressions, so the result stays within the
/// op's envelope (the differential harness enforces it).
#[derive(Debug, Clone, Copy, Default)]
pub struct ConvertOptimized;

impl ConvertOptimized {
    /// Construct the optimized convert backend.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }
}

impl OpImplementation for ConvertOptimized {
    fn compute(
        &self,
        inputs: &InputValues,
        params: &serde_json::Value,
    ) -> std::result::Result<OutputValues, Error> {
        compute_backend(Backend::Optimized, inputs, params)
    }
}

/// A `from` / `to` color-encoding string parameter, drawn from the supported
/// encoding choices.
fn encoding_param(name: &str, doc: &str) -> ParamSpec {
    ParamSpec {
        name: name.to_owned(),
        ty: ParamType::String,
        unit: None,
        required: true,
        default: None,
        choices: vec![
            "srgb".to_owned(),
            "linear-srgb".to_owned(),
            "raw-linear".to_owned(),
        ],
        doc: doc.to_owned(),
    }
}

/// The mandatory `cpu.reference@1` oracle implementation id.
fn reference_impl() -> Result<ImplId> {
    ImplId::new("cpu", "reference", 1)
}

/// The `cpu.optimized@1` autovectorized backend implementation id.
fn optimized_impl() -> Result<ImplId> {
    ImplId::new("cpu", "optimized", 1)
}

/// The verification declarations for `color.convert@1`: a single-reference,
/// bounded, pointwise transfer-function op. Differential does not apply (one
/// implementation). Perceptual is not applicable: the conversion is a closed-form
/// numeric transfer function checked against an analytic value table and a
/// tolerance round-trip, not a perceptual-quality comparison. Every other
/// applicable category is covered by the analytic-table, round-trip, monotonicity
/// and rejection tests in this module.
fn convert_test_metadata() -> TestMetadata {
    use paintop_ir::{CategoryStatus, VerificationCategory, VerificationDeclarations};
    let mut decls = VerificationDeclarations::new();
    for category in [
        VerificationCategory::BuildHygiene,
        VerificationCategory::SchemaContract,
        VerificationCategory::AnalyticFixtures,
        VerificationCategory::PropertyTests,
        VerificationCategory::Metamorphic,
        // The op now exposes a cpu.optimized backend, so differential testing
        // applies: the cross-backend harness validates it against the oracle.
        VerificationCategory::Differential,
        VerificationCategory::Goldens,
        VerificationCategory::Fuzzing,
        VerificationCategory::Performance,
    ] {
        decls = decls.with(category, CategoryStatus::Covered);
    }
    decls = decls.with(
        VerificationCategory::Perceptual,
        CategoryStatus::not_applicable(
            "color.convert is a closed-form numeric transfer function verified by an analytic \
             value table and a tolerance round-trip; there is no perceptual-quality metric to \
             apply",
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
