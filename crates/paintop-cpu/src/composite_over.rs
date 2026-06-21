//! The `composite.over@1` operation: premultiplied linear Porter–Duff *over*
//! (`OP_CATALOG` §7; `AGENT_VERIFICATION` §3.2).
//!
//! `composite.over` composites a source image `src` over a destination image
//! `dst` in **premultiplied** linear light, the standard Porter–Duff source-over:
//!
//! ```text
//! C_o = C_s + C_d · (1 − α_s)      (per color channel)
//! α_o = α_s + α_d · (1 − α_s)
//! ```
//!
//! Because both inputs are premultiplied, the color and alpha updates share the
//! single factor `(1 − α_s)`, so the op is associative (within floating-point
//! tolerance) and produces no colored fringe along alpha edges — the hidden RGB
//! under a fully transparent source contributes nothing. A fully transparent
//! source (`α_s = 0`) is the identity on `dst`; a fully opaque source (`α_s = 1`)
//! replaces `dst` entirely.
//!
//! # Premultiplied-correct compositing
//!
//! The blend is linear in the premultiplied channels, so it is only correct on
//! premultiplied linear color with an alpha channel. A nonlinear (`srgb`) input, a
//! straight-alpha input, or an image without alpha is rejected with a
//! [`semantic`](ErrorClass::Semantic) error rather than producing wrong color.
//!
//! # Geometry & determinism
//!
//! The op is **pointwise**: every output sample depends only on the co-located
//! `src` and `dst` samples (and that pixel's source alpha). `src` and `dst` must
//! share extent and channel layout. The blend is a per-channel `f32` multiply-add,
//! a deterministic, bit-exact function of the inputs, so the op is
//! [`Exact`](DeterminismTier::Exact).

use paintop_core::executor::{InputValues, OpImplementation, OutputValues, ResourceValue};
use paintop_ir::{
    AlphaRepresentation, AssertionResult, ColorEncoding, Descriptors, DeterminismTier, Error,
    ErrorClass, ErrorContext, ImageDescriptor, ImplId, InputRegions, InputSpec, OpContract,
    OperationManifest, OutputDescriptors, OutputRegions, OutputSpec, ResourceDescriptor,
    ResourceKind, Result, RoiCategory, RoiPolicy, TestMetadata,
};

/// The canonical id of the premultiplied-over operation.
pub const OVER_OP_ID: &str = "composite.over@1";

/// A required input port (`src` or `dst`) was absent or carried the wrong
/// resource kind.
pub const E_OVER_INPUT: &str = "E_OVER_INPUT";

/// The `src` / `dst` images are in a representation this op cannot composite
/// (nonlinear encoding, straight alpha, or no alpha channel), or the two ports
/// disagree on extent / layout.
pub const E_OVER_SHAPE: &str = "E_OVER_SHAPE";

/// The op produced a sample buffer whose length disagrees with its descriptor.
pub const E_OVER_BUFFER: &str = "E_OVER_BUFFER";

/// Validate that a `src` / `dst` image may be composited in premultiplied linear
/// light: linear-encoded, premultiplied, with an alpha channel.
fn check_color_image(descriptor: &ImageDescriptor, port: &str) -> Result<()> {
    if descriptor.color == ColorEncoding::Srgb {
        return Err(Error::new(
            ErrorClass::Semantic,
            E_OVER_SHAPE,
            format!(
                "composite.over requires linear-light color; the `{port}` input is `srgb`-encoded. \
                 Insert a color.convert to linear-srgb first."
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
            E_OVER_SHAPE,
            format!(
                "composite.over requires the `{port}` image to have an alpha channel (GrayA or Rgba)"
            ),
        ));
    }
    if descriptor.alpha != AlphaRepresentation::Premultiplied {
        return Err(Error::new(
            ErrorClass::Semantic,
            E_OVER_SHAPE,
            format!(
                "composite.over composites in premultiplied space; premultiply the `{port}` image \
                 first"
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

/// Validate the two ports together and return the output image descriptor (the
/// `dst` descriptor: compositing onto `dst` never changes its type).
fn check_and_retarget(src: &ImageDescriptor, dst: &ImageDescriptor) -> Result<ImageDescriptor> {
    check_color_image(src, "src")?;
    check_color_image(dst, "dst")?;
    if src.extent != dst.extent {
        return Err(shape_mismatch(
            "the `src` and `dst` images must share an extent",
            format!("src {:?} vs dst {:?}", src.extent, dst.extent),
        ));
    }
    if src.layout != dst.layout {
        return Err(shape_mismatch(
            "the `src` and `dst` images must share a channel layout",
            format!("src {:?} vs dst {:?}", src.layout, dst.layout),
        ));
    }
    Ok(*dst)
}

/// Build a shape-mismatch [`semantic`](ErrorClass::Semantic) error.
fn shape_mismatch(detail: &str, actual: String) -> Error {
    Error::new(
        ErrorClass::Semantic,
        E_OVER_SHAPE,
        format!("composite.over: {detail}"),
    )
    .with_context(ErrorContext::default().with_actual(actual))
}

/// Composite `src` over `dst` per pixel and channel, premultiplied source-over.
///
/// `channels` is the interleaved color+alpha sample count per pixel; the alpha
/// channel is the last of each pixel. For every channel (color *and* alpha) the
/// update is `out = c_s + c_d·(1 − α_s)`, sharing the single source-alpha factor.
fn over(src: &[f32], dst: &[f32], channels: u32) -> Vec<f32> {
    let channel_count = channels as usize;
    if channel_count == 0 {
        return dst.to_vec();
    }
    let alpha_index = channel_count - 1;
    let mut out = Vec::with_capacity(dst.len());
    for (src_pixel, dst_pixel) in src
        .chunks_exact(channel_count)
        .zip(dst.chunks_exact(channel_count))
    {
        let inv_alpha_s = 1.0 - src_pixel[alpha_index];
        for (&c_s, &c_d) in src_pixel.iter().zip(dst_pixel.iter()) {
            // C_o = C_s + C_d·(1 − α_s); the alpha channel uses the same form
            // (α_o = α_s + α_d·(1 − α_s)). A fused multiply-add: one deterministic
            // f32 op per channel.
            out.push(c_d.mul_add(inv_alpha_s, c_s));
        }
    }
    out
}

/// Read a required image port's descriptor.
fn image_descriptor<'a>(inputs: &'a Descriptors, port: &str) -> Result<&'a ImageDescriptor> {
    let resource = inputs.get(port).ok_or_else(|| {
        Error::new(
            ErrorClass::Reference,
            E_OVER_INPUT,
            format!("composite.over requires a `{port}` input"),
        )
    })?;
    let ResourceDescriptor::Image(descriptor) = resource else {
        return Err(Error::new(
            ErrorClass::Type,
            E_OVER_INPUT,
            format!("composite.over `{port}` input must be an image resource"),
        ));
    };
    Ok(descriptor)
}

/// The `composite.over@1` operation: `src` + `dst` → the composited `image`.
#[derive(Debug, Clone, Copy, Default)]
pub struct Over;

impl Over {
    /// Construct the premultiplied-over operation.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    /// The declared manifest for `composite.over@1`.
    ///
    /// # Errors
    /// Propagates the [`schema`](ErrorClass::Schema) error if the hard-coded op /
    /// impl ids are somehow invalid (they are not).
    pub fn manifest() -> Result<OperationManifest> {
        Ok(OperationManifest {
            id: OVER_OP_ID.parse()?,
            impl_version: 1,
            summary: "Composite a source image over a destination image in premultiplied linear \
                      light (Porter–Duff over: C_o = C_s + C_d·(1−α_s), α_o = α_s + α_d·(1−α_s)). \
                      Rejected on nonlinear (srgb), straight-alpha, or no-alpha input."
                .to_owned(),
            // Per-channel f32 multiply-add: bit-exact for the scalar reference.
            determinism: DeterminismTier::Exact,
            roi: RoiPolicy {
                category: RoiCategory::Pointwise,
                halo_px: None,
            },
            inputs: vec![
                InputSpec {
                    name: "src".to_owned(),
                    kind: ResourceKind::Image,
                    required: true,
                    doc: "The premultiplied linear-light source image, composited over dst."
                        .to_owned(),
                },
                InputSpec {
                    name: "dst".to_owned(),
                    kind: ResourceKind::Image,
                    required: true,
                    doc: "The premultiplied linear-light destination image, composited under src."
                        .to_owned(),
                },
            ],
            outputs: vec![OutputSpec {
                name: "image".to_owned(),
                kind: ResourceKind::Image,
                doc: "The composited image (the dst descriptor; same extent/layout).".to_owned(),
            }],
            params: vec![],
            implementations: vec![reference_impl()?, optimized_impl()?],
            test: over_test_metadata(),
        })
    }
}

impl OpContract for Over {
    fn declared_inputs(&self) -> Vec<(String, ResourceKind)> {
        vec![
            ("src".to_owned(), ResourceKind::Image),
            ("dst".to_owned(), ResourceKind::Image),
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
        let src = image_descriptor(inputs, "src")?;
        let dst = image_descriptor(inputs, "dst")?;
        let out_descriptor = check_and_retarget(src, dst)?;

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
        // Pointwise: each output pixel reads the co-located src and dst sample.
        let mut regions = InputRegions::new();
        if let Some(region) = requested_outputs.get("image") {
            for port in ["src", "dst"] {
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

/// The compute backend serving `composite.over`: the scalar reference oracle or
/// the autovectorization-friendly `cpu.optimized` kernel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Backend {
    /// The scalar reference oracle ([`over`]).
    Reference,
    /// The `cpu.optimized` source-over kernel ([`crate::optimized::kernels`]).
    Optimized,
}

/// Shared compute for both backends: validate the two ports, then composite.
fn compute_backend(
    backend: Backend,
    inputs: &InputValues,
) -> std::result::Result<OutputValues, Error> {
    let src = input_value(inputs, "src")?;
    let dst = input_value(inputs, "dst")?;

    let ResourceDescriptor::Image(src_descriptor) = src.descriptor() else {
        return Err(input_type_error("src"));
    };
    let ResourceDescriptor::Image(dst_descriptor) = dst.descriptor() else {
        return Err(input_type_error("dst"));
    };

    let out_descriptor = check_and_retarget(src_descriptor, dst_descriptor)?;
    let samples = match backend {
        Backend::Reference => over(src.samples(), dst.samples(), dst.channels()),
        Backend::Optimized => crate::optimized::kernels::composite_over(
            src.samples(),
            dst.samples(),
            dst.channels() as usize,
        ),
    };

    let value = ResourceValue::new(
        ResourceDescriptor::Image(out_descriptor),
        dst.channels(),
        samples,
    )
    .map_err(|actual| {
        Error::new(
            ErrorClass::Execution,
            E_OVER_BUFFER,
            format!("composite.over produced a sample buffer of unexpected length {actual}"),
        )
    })?;

    let mut out = OutputValues::new();
    out.insert("image".to_owned(), value);
    Ok(out)
}

impl OpImplementation for Over {
    fn compute(
        &self,
        inputs: &InputValues,
        _params: &serde_json::Value,
    ) -> std::result::Result<OutputValues, Error> {
        compute_backend(Backend::Reference, inputs)
    }
}

/// The `cpu.optimized@1` backend for `composite.over@1`.
///
/// It computes the same per-channel premultiplied source-over
/// `C_o = C_s + C_d * (1 - alpha_s)` as the oracle via the autovectorization-
/// friendly kernel, using the identical fused multiply-add.
/// `composite.over` is [`Exact`](DeterminismTier::Exact), so the optimized result
/// is **bit-identical** to the reference (the differential harness enforces it).
#[derive(Debug, Clone, Copy, Default)]
pub struct OverOptimized;

impl OverOptimized {
    /// Construct the optimized source-over backend.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }
}

impl OpImplementation for OverOptimized {
    fn compute(
        &self,
        inputs: &InputValues,
        _params: &serde_json::Value,
    ) -> std::result::Result<OutputValues, Error> {
        compute_backend(Backend::Optimized, inputs)
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
            E_OVER_INPUT,
            format!("composite.over requires a `{port}` input value"),
        )
    })
}

/// The wrong-resource-kind error for a color image port.
fn input_type_error(port: &str) -> Error {
    Error::new(
        ErrorClass::Type,
        E_OVER_INPUT,
        format!("composite.over `{port}` input must be an image resource"),
    )
}

/// The mandatory `cpu.reference@1` oracle implementation id.
fn reference_impl() -> Result<ImplId> {
    ImplId::new("cpu", "reference", 1)
}

/// The `cpu.optimized@1` autovectorized backend implementation id.
fn optimized_impl() -> Result<ImplId> {
    ImplId::new("cpu", "optimized", 1)
}

/// The verification declarations for `composite.over@1` (`AGENT_VERIFICATION`
/// §3.2). The op is an exact, single-reference per-channel premultiplied blend
/// verified by the §3.2 property set: transparent-source identity, opaque-source
/// replacement, output alpha in [0, 1], the premultiplied constraint, associativity
/// within tolerance, and the alpha-edge fringe fixture. Differential does not apply
/// (one implementation). Perceptual is not applicable: the blend is bit-exact,
/// pinned by numeric goldens, with no perceptual-quality metric.
fn over_test_metadata() -> TestMetadata {
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
            "composite.over is a bit-exact per-channel premultiplied source-over verified by the \
             AGENT_VERIFICATION §3.2 property set (transparent/opaque identities, premultiplied \
             constraint, associativity, alpha-edge fringe); there is no perceptual-quality metric \
             to apply",
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
