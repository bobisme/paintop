//! The `mask.to_sdf@1` operation: a coverage/selection `Mask` → a signed
//! distance field `SdfMask` (`OP_CATALOG` §4, `IR_SPEC` §7.4, `ALIEN_OPS` §2).
//!
//! # What it computes
//!
//! Given a mask and a **contour threshold** `t ∈ [0, 1]`, the op partitions the
//! grid into an *inside* set `S = { p : coverage(p) ≥ t }` and its complement,
//! then builds the signed distance to the boundary between them using the exact
//! Euclidean distance transform ([`crate::edt`]) run over both sets:
//!
//! ```text
//! sdf(p) = D_outside(p) − D_inside(p)
//!        = sqrt(D²_S(p)) − sqrt(D²_{\overline S}(p))
//! ```
//!
//! where `D²_S` is the squared distance to the nearest *inside* pixel and
//! `D²_{\overline S}` the squared distance to the nearest *outside* pixel. This is
//! `0` on the boundary, strictly **negative inside** the region, and strictly
//! positive outside — the project's mandatory `negative-inside` sign convention
//! (`IR_SPEC` §7.4), which is recorded explicitly in the produced
//! [`SdfDescriptor`] and never left implicit.
//!
//! ## Why a threshold is mandatory for soft masks
//!
//! A *hard* mask (every sample exactly `0` or `1`) has one unambiguous boundary,
//! so any threshold in `(0, 1)` partitions it identically. A *soft* coverage map
//! does **not** have a unique signed distance field — the contour at coverage
//! `0.3` and the contour at `0.7` are different curves — so this op never guesses:
//! the contour threshold is an explicit param (default `0.5`, the natural
//! half-coverage isocontour), and the partition is `coverage ≥ threshold`. The op
//! does not anti-alias or interpolate the contour to sub-pixel precision; the
//! distance is measured between pixel centers (`PixelCenterUpperLeft`), exact in
//! the EDT sense, which is the "rasterized-boundary ambiguity" the acceptance
//! criteria scope analytic agreement away from.
//!
//! ## Degenerate partitions
//!
//! - **Empty inside** (no pixel meets the threshold): every pixel is outside, so
//!   `D_inside = +∞` everywhere; the field is `+∞` (there is no boundary to be
//!   signed-distant from). Represented as [`f32::INFINITY`].
//! - **Full inside** (every pixel meets the threshold): symmetrically `−∞`.
//!
//! These two are the only fields with no zero contour, and the op preserves the
//! `±∞` sentinel rather than fabricating a finite distance.
//!
//! # Determinism
//!
//! [`DeterminismTier::Exact`]: the partition is a pointwise threshold comparison,
//! the EDT is the exact integer squared-distance transform (bit-identical across
//! platforms, `crate::edt`), and the only rounding is the per-sample `sqrt` and
//! the final subtraction, both IEEE-754 and order-independent. A tile boundary
//! cannot change the result because the EDT footprint is the whole domain
//! ([`RoiCategory::FullDomain`]).

use paintop_core::executor::{InputValues, OpImplementation, OutputValues, ResourceValue};
use paintop_ir::{
    AssertionResult, CoordinateConvention, Descriptors, DeterminismTier, Error, ErrorClass, Extent,
    ImplId, InputRegions, InputSpec, OpContract, OperationManifest, OutputDescriptors,
    OutputRegions, OutputSpec, ParamSpec, ParamType, Rect, ResourceDescriptor, ResourceKind,
    Result, RoiCategory, RoiPolicy, ScalarType, SdfDescriptor, SdfSign, SdfUnits, TestMetadata,
};

use crate::edt::{self, BinaryGrid};

/// The canonical id of the mask → signed-distance-field operation.
pub const OP_ID: &str = "mask.to_sdf@1";

/// The default contour threshold: the half-coverage isocontour.
pub const DEFAULT_THRESHOLD: f64 = 0.5;

/// A required mask input was absent or was not a mask.
pub const E_TO_SDF_INPUT: &str = "E_TO_SDF_INPUT";
/// The `threshold` parameter was malformed or out of the `[0, 1]` range.
pub const E_TO_SDF_PARAM: &str = "E_TO_SDF_PARAM";
/// The produced SDF buffer had an unexpected length, or the grid could not be
/// formed from the input mask.
pub const E_TO_SDF_BUFFER: &str = "E_TO_SDF_BUFFER";

/// The `mask.to_sdf@1` operation (zero-sized; all state is in the params).
#[derive(Debug, Clone, Copy, Default)]
pub struct MaskToSdf;

impl MaskToSdf {
    /// Construct the op.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    /// The declared manifest for `mask.to_sdf@1`.
    ///
    /// # Errors
    /// Propagates a [`schema`](ErrorClass::Schema) error if the hard-coded op or
    /// impl ids are invalid (they are not).
    pub fn manifest() -> Result<OperationManifest> {
        Ok(OperationManifest {
            id: OP_ID.parse()?,
            impl_version: 1,
            summary: "Convert a coverage/selection Mask to a signed distance field (SdfMask) in \
                      physical pixels via the exact Euclidean distance transform; the \
                      `negative-inside` sign convention is explicit and the contour threshold \
                      that defines the boundary (coverage >= threshold) is a required-for-soft \
                      param (default 0.5)."
                .to_owned(),
            determinism: DeterminismTier::Exact,
            roi: RoiPolicy {
                category: RoiCategory::FullDomain,
                halo_px: None,
            },
            inputs: vec![InputSpec {
                name: "mask".to_owned(),
                kind: ResourceKind::Mask,
                required: true,
                doc: "The coverage or selection mask whose `coverage >= threshold` region is \
                      the inside set of the produced field."
                    .to_owned(),
            }],
            outputs: vec![OutputSpec {
                name: "sdf".to_owned(),
                kind: ResourceKind::SdfMask,
                doc: "The signed distance to the threshold contour, in physical pixels, \
                      negative inside and positive outside."
                    .to_owned(),
            }],
            params: vec![ParamSpec {
                name: "threshold".to_owned(),
                ty: ParamType::Float,
                unit: None,
                required: false,
                default: Some(serde_json::json!(DEFAULT_THRESHOLD)),
                choices: vec![],
                doc: "The contour level in [0, 1]: a pixel is inside iff its coverage is >= \
                      threshold. A soft mask has no unique SDF, so this is the explicit \
                      boundary choice; default 0.5 is the half-coverage isocontour."
                    .to_owned(),
            }],
            implementations: vec![reference_impl()?],
            test: to_sdf_test_metadata(),
        })
    }
}

impl OpContract for MaskToSdf {
    fn declared_inputs(&self) -> Vec<(String, ResourceKind)> {
        vec![("mask".to_owned(), ResourceKind::Mask)]
    }
    fn declared_outputs(&self) -> Vec<(String, ResourceKind)> {
        vec![("sdf".to_owned(), ResourceKind::SdfMask)]
    }
    fn infer_outputs(
        &self,
        inputs: &Descriptors,
        _params: &serde_json::Value,
    ) -> Result<OutputDescriptors> {
        let extent = mask_extent_of(inputs, OP_ID)?;
        let mut out = OutputDescriptors::new();
        out.insert(
            "sdf".to_owned(),
            ResourceDescriptor::SdfMask(sdf_descriptor(extent)),
        );
        Ok(out)
    }
    fn required_inputs(
        &self,
        _requested_outputs: &OutputRegions,
        inputs: &Descriptors,
        _params: &serde_json::Value,
    ) -> Result<InputRegions> {
        // The EDT propagates distance information across the whole connected
        // domain, so producing any output region needs the entire input mask.
        let mut regions = InputRegions::new();
        if let Some(d) = inputs.get("mask") {
            let extent = d.extent();
            regions.insert(
                "mask".to_owned(),
                Rect::new(0, 0, i64::from(extent.width), i64::from(extent.height)),
            );
        }
        Ok(regions)
    }
    fn validate_postconditions(
        &self,
        outputs: &OutputDescriptors,
        _params: &serde_json::Value,
    ) -> Result<Vec<AssertionResult>> {
        Ok(sdf_postcondition(outputs))
    }
}

impl OpImplementation for MaskToSdf {
    fn compute(
        &self,
        inputs: &InputValues,
        params: &serde_json::Value,
    ) -> std::result::Result<OutputValues, Error> {
        let mask = inputs.get("mask").ok_or_else(|| {
            Error::new(
                ErrorClass::Reference,
                E_TO_SDF_INPUT,
                format!("{OP_ID} requires a `mask` input value"),
            )
        })?;
        let threshold = resolve_threshold(params)?;
        let samples = signed_distance(mask.extent(), mask.samples(), threshold)?;
        finish_sdf(mask.extent(), samples)
    }
}

/// Compute the signed distance field from a coverage buffer and threshold.
///
/// The inside set is `coverage >= threshold`. Returns the row-major field with
/// the `negative-inside` convention; `±∞` for a degenerate empty/full partition.
///
/// # Errors
/// [`E_TO_SDF_BUFFER`] if the binary grid cannot be formed (a buffer length that
/// disagrees with the extent — a caller bug surfaced as a typed error, not a
/// panic).
fn signed_distance(extent: Extent, coverage: &[f32], threshold: f32) -> Result<Vec<f32>> {
    // Partition: inside iff coverage >= threshold.
    let inside: Vec<bool> = coverage.iter().map(|&c| c >= threshold).collect();
    let outside: Vec<bool> = inside.iter().map(|&b| !b).collect();

    let inside_grid = BinaryGrid::new(extent, &inside).map_err(grid_error)?;
    let outside_grid = BinaryGrid::new(extent, &outside).map_err(grid_error)?;

    // D_outside(p): distance to the nearest inside pixel (0 inside, grows
    // outward). D_inside(p): distance to the nearest outside pixel (0 outside,
    // grows inward). sdf = D_outside - D_inside is 0 on the boundary, negative
    // strictly inside, positive strictly outside.
    let dist_to_inside = edt::distance(&edt::transform_sq(&inside_grid));
    let dist_to_outside = edt::distance(&edt::transform_sq(&outside_grid));

    let field: Vec<f32> = dist_to_inside
        .iter()
        .zip(dist_to_outside.iter())
        .map(|(&d_out, &d_in)| signed_sample(d_out, d_in))
        .collect();
    Ok(field)
}

/// Combine the two unsigned distances into one signed sample under the
/// `negative-inside` convention, handling the `±∞` degenerate partitions.
///
/// `d_out` is the distance to the nearest inside pixel (finite outside,
/// `0` inside, `+∞` when there is no inside pixel at all); `d_in` is the distance
/// to the nearest outside pixel. `d_out - d_in` would be `∞ - ∞ = NaN` in the
/// fully-empty/fully-full cases, so those are special-cased to the correct
/// signed infinity.
fn signed_sample(d_out: f32, d_in: f32) -> f32 {
    match (d_out.is_infinite(), d_in.is_infinite()) {
        // No outside pixel anywhere: everything is inside, distance is -inf.
        (false, true) => f32::NEG_INFINITY,
        // No inside pixel anywhere (or a zero-area grid where both are +inf):
        // everything is outside, distance is +inf.
        (true, _) => f32::INFINITY,
        // The normal case: signed distance to the boundary.
        (false, false) => d_out - d_in,
    }
}

/// Resolve and validate the `threshold` param to an `f32` in `[0, 1]`.
fn resolve_threshold(params: &serde_json::Value) -> Result<f32> {
    let value = match params.get("threshold") {
        None | Some(serde_json::Value::Null) => DEFAULT_THRESHOLD,
        Some(v) => v.as_f64().ok_or_else(|| {
            Error::new(
                ErrorClass::Schema,
                E_TO_SDF_PARAM,
                format!("{OP_ID} `threshold` must be a number"),
            )
        })?,
    };
    if !value.is_finite() || !(0.0..=1.0).contains(&value) {
        return Err(Error::new(
            ErrorClass::Schema,
            E_TO_SDF_PARAM,
            format!("{OP_ID} `threshold` must be a finite value in [0, 1], got {value}"),
        ));
    }
    #[allow(
        clippy::cast_possible_truncation,
        reason = "threshold compared against f32 coverage; f64->f32 narrowing of a [0,1] value is exact enough and the comparison is the contract"
    )]
    Ok(value as f32)
}

/// Map an EDT grid-shape error to a typed buffer error.
fn grid_error(err: edt::GridShapeError) -> Error {
    Error::new(
        ErrorClass::Execution,
        E_TO_SDF_BUFFER,
        format!("{OP_ID} could not form a binary grid: {err}"),
    )
}

/// Wrap a signed-distance buffer as an `SdfMask` output value.
fn finish_sdf(extent: Extent, samples: Vec<f32>) -> std::result::Result<OutputValues, Error> {
    let value = ResourceValue::new(
        ResourceDescriptor::SdfMask(sdf_descriptor(extent)),
        1,
        samples,
    )
    .map_err(|actual| {
        Error::new(
            ErrorClass::Execution,
            E_TO_SDF_BUFFER,
            format!("{OP_ID} produced an SDF buffer of unexpected length {actual}"),
        )
    })?;
    let mut out = OutputValues::new();
    out.insert("sdf".to_owned(), value);
    Ok(out)
}

/// The `negative-inside`, pixel-unit SDF descriptor for `extent`.
const fn sdf_descriptor(extent: Extent) -> SdfDescriptor {
    SdfDescriptor {
        extent,
        scalar: ScalarType::F32,
        units: SdfUnits::Pixels,
        sign: SdfSign::NegativeInside,
        coordinates: CoordinateConvention::PixelCenterUpperLeft,
    }
}

/// The extent of the `mask` input descriptor, or a typed error if absent / wrong
/// kind.
fn mask_extent_of(inputs: &Descriptors, op: &str) -> Result<Extent> {
    match inputs.get("mask") {
        Some(ResourceDescriptor::Mask(d)) => Ok(d.extent),
        Some(_) => Err(Error::new(
            ErrorClass::Type,
            E_TO_SDF_INPUT,
            format!("{op} input `mask` must be a Mask"),
        )),
        None => Err(Error::new(
            ErrorClass::Reference,
            E_TO_SDF_INPUT,
            format!("{op} requires a `mask` input"),
        )),
    }
}

/// The postcondition: the op produces an `sdf` output that is a `negative-inside`
/// pixel-unit signed distance field.
fn sdf_postcondition(outputs: &OutputDescriptors) -> Vec<AssertionResult> {
    let Some(ResourceDescriptor::SdfMask(sdf)) = outputs.get("sdf") else {
        return vec![AssertionResult::fail(
            "produces_sdf",
            "no `sdf` output produced",
        )];
    };
    vec![
        AssertionResult::pass("produces_sdf"),
        if sdf.sign == SdfSign::NegativeInside {
            AssertionResult::pass("negative_inside_sign")
        } else {
            AssertionResult::fail(
                "negative_inside_sign",
                format!(
                    "sdf sign {:?} is not the mandatory negative-inside",
                    sdf.sign
                ),
            )
        },
        if sdf.units == SdfUnits::Pixels {
            AssertionResult::pass("pixel_units")
        } else {
            AssertionResult::fail("pixel_units", "sdf units are not physical pixels")
        },
    ]
}

/// The mandatory `cpu.reference@1` oracle implementation id.
fn reference_impl() -> Result<ImplId> {
    ImplId::new("cpu", "reference", 1)
}

/// Verification declarations for `mask.to_sdf@1`: an exact, single-reference op.
/// Perceptual does not apply (it is bit-exact); every other category is covered
/// by this module's analytic circle/rect fixtures, sign/property tests, and the
/// brute-force EDT differential.
fn to_sdf_test_metadata() -> TestMetadata {
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
            "mask.to_sdf is an exact distance transform verified by analytic circle/rectangle \
             distances, sign correctness, and a brute-force EDT differential; there is no \
             perceptual-quality metric"
                .to_owned(),
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
