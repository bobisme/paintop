//! The `image.rotate90@1` operation: an exact quarter-turn rotation
//! (`OP_CATALOG` §5, `AGENT_VERIFICATION` §2.5).
//!
//! `image.rotate90` rotates an image by a multiple of 90° **clockwise** as a pure
//! integer pixel remap — no resampling. The `turns` parameter is the number of
//! clockwise quarter-turns, taken modulo 4 (so `turns = 0` and `turns = 4` are the
//! identity, `turns = -1` is one turn counter-clockwise). Every output sample is a
//! verbatim copy of exactly one input sample, so the op is **exact**.
//!
//! # Geometry (clockwise, `PixelCenterUpperLeft`)
//!
//! For an input of extent `W × H`:
//!
//! - **1 turn** (90° CW): output extent `H × W`, `out(x, y) = in(y, H-1-x)`.
//! - **2 turns** (180°): output extent `W × H`, `out(x, y) = in(W-1-x, H-1-y)`.
//! - **3 turns** (270° CW): output extent `H × W`, `out(x, y) = in(W-1-y, x)`.
//!
//! Odd turns transpose the extent (swap width and height). Four turns return the
//! exact identity — the keystone metamorphic property (§2.5 rotation covariance)
//! this op and the metamorphic harness rely on.
//!
//! # Determinism
//!
//! [`Exact`](DeterminismTier::Exact): a bijective integer remap copying samples
//! verbatim, bit-identical on every run.

use paintop_core::executor::{InputValues, OpImplementation, OutputValues, ResourceValue};
use paintop_ir::{
    AssertionResult, Descriptors, DeterminismTier, Error, ErrorClass, ErrorContext, Extent,
    ImageDescriptor, ImplId, InputRegions, InputSpec, OpContract, OperationManifest,
    OutputDescriptors, OutputRegions, OutputSpec, ParamSpec, ParamType, ResourceDescriptor,
    ResourceKind, Result, RoiCategory, RoiPolicy, TestMetadata,
};

/// The canonical id of the image-rotate90 operation.
pub const ROTATE90_OP_ID: &str = "image.rotate90@1";

/// The `image` input was absent or carried a non-image descriptor.
pub const E_ROTATE_INPUT: &str = "E_ROTATE_INPUT";

/// The `turns` parameter was missing or not an integer.
pub const E_ROTATE_TURNS: &str = "E_ROTATE_TURNS";

/// Parse the required `turns` param and normalize it to `0..=3` clockwise turns.
fn turns_param(params: &serde_json::Value) -> Result<u32> {
    let value = params.get("turns").ok_or_else(|| {
        Error::new(
            ErrorClass::Schema,
            E_ROTATE_TURNS,
            "image.rotate90 requires an integer `turns` parameter".to_owned(),
        )
    })?;
    let n = value.as_i64().ok_or_else(|| {
        Error::new(
            ErrorClass::Schema,
            E_ROTATE_TURNS,
            "image.rotate90 `turns` must be an integer".to_owned(),
        )
        .with_context(ErrorContext::default().with_actual(value.to_string()))
    })?;
    #[allow(
        clippy::cast_sign_loss,
        clippy::cast_possible_truncation,
        reason = "rem_euclid(4) yields a value in 0..=3"
    )]
    Ok(n.rem_euclid(4) as u32)
}

/// The output extent after applying `turns` clockwise quarter-turns: odd turns
/// transpose width and height.
const fn rotated_extent(src: Extent, turns: u32) -> Extent {
    if turns % 2 == 1 {
        Extent::new(src.height, src.width)
    } else {
        src
    }
}

/// Map an output cell `(x, y)` (in the rotated `out_w × out_h` space) back to its
/// source cell in the `src_w × src_h` input, for `turns` clockwise quarter-turns.
const fn source_cell(x: usize, y: usize, src_w: usize, src_h: usize, turns: u32) -> (usize, usize) {
    match turns {
        // 90 CW: out(x, y) = in(y, H-1-x).
        1 => (y, src_h - 1 - x),
        // 180: out(x, y) = in(W-1-x, H-1-y).
        2 => (src_w - 1 - x, src_h - 1 - y),
        // 270 CW: out(x, y) = in(W-1-y, x).
        3 => (src_w - 1 - y, x),
        // 0 turns: identity.
        _ => (x, y),
    }
}

/// Remap the interleaved `samples` of a `src` image under `turns` clockwise
/// quarter-turns into a buffer for the rotated extent.
fn rotate_samples(samples: &[f32], src: Extent, turns: u32, channels: u32) -> Vec<f32> {
    let stride = channels as usize;
    let src_w = src.width as usize;
    let src_h = src.height as usize;
    let out = rotated_extent(src, turns);
    let out_w = out.width as usize;
    let out_h = out.height as usize;
    let mut buf = vec![0.0; samples.len()];
    for y in 0..out_h {
        for x in 0..out_w {
            let (sx, sy) = source_cell(x, y, src_w, src_h, turns);
            let dst = (y * out_w + x) * stride;
            let s = (sy * src_w + sx) * stride;
            buf[dst..dst + stride].copy_from_slice(&samples[s..s + stride]);
        }
    }
    buf
}

/// The `image.rotate90@1` operation: an image + turns → the rotated image.
#[derive(Debug, Clone, Copy, Default)]
pub struct Rotate90;

impl Rotate90 {
    /// Construct the image-rotate90 operation.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    /// The declared manifest for `image.rotate90@1`.
    ///
    /// # Errors
    /// Propagates the [`schema`](ErrorClass::Schema) error if the hard-coded op /
    /// impl ids are somehow invalid (they are not).
    pub fn manifest() -> Result<OperationManifest> {
        Ok(OperationManifest {
            id: ROTATE90_OP_ID.parse()?,
            impl_version: 1,
            summary:
                "Exact quarter-turn rotation (turns * 90 deg clockwise, mod 4) as a bijective \
                      integer pixel remap; no resampling. Four turns is the identity; odd turns \
                      transpose the extent."
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
                doc: "The image to rotate.".to_owned(),
            }],
            outputs: vec![OutputSpec {
                name: "image".to_owned(),
                kind: ResourceKind::Image,
                doc: "The rotated image (extent transposed for odd turns; all other descriptor \
                      fields preserved)."
                    .to_owned(),
            }],
            params: vec![ParamSpec {
                name: "turns".to_owned(),
                ty: ParamType::Integer,
                unit: None,
                required: true,
                default: None,
                choices: vec![],
                doc: "Number of 90-degree clockwise quarter-turns, taken modulo 4 (negative turns \
                      rotate counter-clockwise)."
                    .to_owned(),
            }],
            implementations: vec![reference_impl()?],
            test: rotate_test_metadata(),
        })
    }
}

impl OpContract for Rotate90 {
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
        let turns = turns_param(params)?;
        let mut out_desc = *descriptor;
        out_desc.extent = rotated_extent(descriptor.extent, turns);
        let mut out = OutputDescriptors::new();
        out.insert("image".to_owned(), ResourceDescriptor::Image(out_desc));
        Ok(out)
    }

    fn required_inputs(
        &self,
        requested_outputs: &OutputRegions,
        inputs: &Descriptors,
        params: &serde_json::Value,
    ) -> Result<InputRegions> {
        // Geometric: a rotation maps an output region to a rotated input region.
        // Conservatively, demand the whole input plane (the rotation is a global
        // remap), which is always correct and keeps the ROI math simple for an
        // exact whole-image transform.
        let descriptor = image_descriptor(inputs)?;
        let _ = turns_param(params)?;
        let mut regions = InputRegions::new();
        if requested_outputs.contains_key("image") {
            regions.insert(
                "image".to_owned(),
                paintop_ir::Rect::new(
                    0,
                    0,
                    i64::from(descriptor.extent.width),
                    i64::from(descriptor.extent.height),
                ),
            );
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

impl OpImplementation for Rotate90 {
    fn compute(
        &self,
        inputs: &InputValues,
        params: &serde_json::Value,
    ) -> std::result::Result<OutputValues, Error> {
        let image = inputs.get("image").ok_or_else(|| {
            Error::new(
                ErrorClass::Reference,
                E_ROTATE_INPUT,
                "image.rotate90 requires an `image` input value".to_owned(),
            )
        })?;
        let ResourceDescriptor::Image(descriptor) = image.descriptor() else {
            return Err(Error::new(
                ErrorClass::Type,
                E_ROTATE_INPUT,
                "image.rotate90 `image` input must be an image resource".to_owned(),
            ));
        };
        let turns = turns_param(params)?;
        let samples = rotate_samples(image.samples(), descriptor.extent, turns, image.channels());

        let mut out_desc = *descriptor;
        out_desc.extent = rotated_extent(descriptor.extent, turns);
        let value = ResourceValue::new(
            ResourceDescriptor::Image(out_desc),
            image.channels(),
            samples,
        )
        .map_err(|actual| {
            Error::new(
                ErrorClass::Execution,
                E_ROTATE_INPUT,
                format!("image.rotate90 produced a sample buffer of unexpected length {actual}"),
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
            E_ROTATE_INPUT,
            "image.rotate90 requires an `image` input".to_owned(),
        )
    })?;
    let ResourceDescriptor::Image(descriptor) = image else {
        return Err(Error::new(
            ErrorClass::Type,
            E_ROTATE_INPUT,
            "image.rotate90 `image` input must be an image resource".to_owned(),
        ));
    };
    Ok(descriptor)
}

/// The mandatory `cpu.reference@1` oracle implementation id.
fn reference_impl() -> Result<ImplId> {
    ImplId::new("cpu", "reference", 1)
}

/// Verification declarations for `image.rotate90@1`: an exact, single-reference,
/// geometric integer remap. Differential does not apply (one implementation).
/// Perceptual is not applicable: the rotation copies samples verbatim and is
/// verified by exact-correspondence fixtures and the four-turn identity, not a
/// perceptual metric.
fn rotate_test_metadata() -> TestMetadata {
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
            "image.rotate90 is a bijective integer pixel remap copying samples verbatim; \
             correctness is verified by exact pixel-correspondence fixtures and the four-turn \
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
