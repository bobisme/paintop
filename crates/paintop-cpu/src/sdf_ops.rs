//! The signed-distance-field reconstruction and offset ops `sdf.to_mask@1` and
//! `sdf.offset@1` (`OP_CATALOG` §4, `ALIEN_OPS` §2.1).
//!
//! These are the inverse and the displacement of [`crate::mask_to_sdf`]: one
//! turns a signed distance field back into a coverage `Mask`, the other shifts
//! the zero contour to grow or shrink the region without ever rasterizing it.
//!
//! # `sdf.offset@1` — grow / shrink
//!
//! Offsetting a field by a signed physical distance `d` is the single
//! subtraction
//!
//! ```text
//! φ'(p) = φ(p) − d
//! ```
//!
//! Under the project's `negative-inside` convention (`IR_SPEC` §7.4) a pixel is
//! inside iff `φ < 0`, so subtracting a **positive** `d` pushes more of the field
//! below zero — the region **grows** by `d` pixels — and a negative `d` shrinks
//! it. Because `φ` is a true Euclidean distance field (unit gradient), the offset
//! field is again a valid distance field away from any newly-created medial axis,
//! and the operation composes exactly:
//! `offset(offset(φ, d₁), d₂) = offset(φ, d₁ + d₂)`, with `offset(φ, 0) = φ`
//! bit-identically. The `±∞` sentinels of a degenerate (empty / full) field are
//! preserved (`±∞ − d = ±∞`).
//!
//! # `sdf.to_mask@1` — reconstruction
//!
//! Reconstruction maps the signed distance to a coverage value through a
//! **profile** with an explicit physical feather half-width `h ≥ 0`:
//!
//! ```text
//! coverage(p) = smoothstep( clamp( (h − φ(p)) / (2h), 0, 1 ) )
//! ```
//!
//! the Hermite `3t² − 2t³` smoothstep. This is `1` deep inside (`φ ≤ −h`), `0.5`
//! exactly on the zero contour (`φ = 0`), and `0` outside the band (`φ ≥ h`), so
//! the soft edge spans `2h` physical pixels centered on the contour — the
//! `half_width_px` the manifest advertises. A half-width of `0` is the hard step
//! (`coverage = 1` iff `φ ≤ 0`), which round-trips a hard mask through
//! `mask.to_sdf → sdf.to_mask` exactly at the zero contour. A `+∞` field is fully
//! outside (coverage `0`); a `−∞` field is fully inside (coverage `1`).
//!
//! # Determinism
//!
//! Both ops are [`DeterminismTier::Exact`]: every output sample is a closed-form
//! IEEE-754 function of the corresponding input sample (a subtraction, or a
//! clamp-and-smoothstep), pointwise and order-independent, so a tile boundary
//! never changes a sample and the result is bit-identical across platforms.

use paintop_core::executor::{InputValues, OpImplementation, OutputValues, ResourceValue};
use paintop_ir::{
    AssertionResult, CoordinateConvention, Descriptors, DeterminismTier, Error, ErrorClass, Extent,
    ImplId, InputRegions, InputSpec, MaskDescriptor, MaskMeaning, OpContract, OperationManifest,
    OutputDescriptors, OutputRegions, OutputSpec, ParamSpec, ParamType, ParamUnit, Rect,
    ResourceDescriptor, ResourceKind, Result, RoiCategory, RoiPolicy, ScalarType, SdfDescriptor,
    SdfSign, SdfUnits, TestMetadata, ValidRange,
};

/// The canonical id of the SDF reconstruction (→ coverage mask) operation.
pub const TO_MASK_OP_ID: &str = "sdf.to_mask@1";
/// The canonical id of the SDF grow/shrink (offset) operation.
pub const OFFSET_OP_ID: &str = "sdf.offset@1";

/// A required SDF input was absent or was not an `SdfMask`.
pub const E_SDF_INPUT: &str = "E_SDF_INPUT";
/// A reconstruction/offset parameter was malformed or out of range.
pub const E_SDF_PARAM: &str = "E_SDF_PARAM";
/// A produced buffer had an unexpected length.
pub const E_SDF_BUFFER: &str = "E_SDF_BUFFER";

// ---------------------------------------------------------------------------
// shared descriptor helpers
// ---------------------------------------------------------------------------

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

/// The coverage-mask descriptor produced by reconstruction.
const fn mask_descriptor(extent: Extent) -> MaskDescriptor {
    MaskDescriptor {
        extent,
        scalar: ScalarType::F32,
        range: ValidRange::Bounded { min: 0.0, max: 1.0 },
        meaning: MaskMeaning::Coverage,
        coordinates: CoordinateConvention::PixelCenterUpperLeft,
    }
}

/// The extent of the `sdf` input descriptor, or a typed error if absent / wrong.
fn sdf_extent_of(inputs: &Descriptors, op: &str) -> Result<Extent> {
    match inputs.get("sdf") {
        Some(ResourceDescriptor::SdfMask(d)) => Ok(d.extent),
        Some(_) => Err(Error::new(
            ErrorClass::Type,
            E_SDF_INPUT,
            format!("{op} input `sdf` must be an SdfMask"),
        )),
        None => Err(Error::new(
            ErrorClass::Reference,
            E_SDF_INPUT,
            format!("{op} requires an `sdf` input"),
        )),
    }
}

/// The `sdf` input value, or a typed error if absent.
fn sdf_value_of<'a>(
    inputs: &'a InputValues,
    op: &str,
) -> std::result::Result<&'a ResourceValue, Error> {
    inputs.get("sdf").ok_or_else(|| {
        Error::new(
            ErrorClass::Reference,
            E_SDF_INPUT,
            format!("{op} requires an `sdf` input value"),
        )
    })
}

/// The whole-domain input region: the EDT-derived field has already propagated
/// across the whole domain, so any output region depends on the full input.
fn full_sdf_region(inputs: &Descriptors, regions: &mut InputRegions) {
    if let Some(d) = inputs.get("sdf") {
        let extent = d.extent();
        regions.insert(
            "sdf".to_owned(),
            Rect::new(0, 0, i64::from(extent.width), i64::from(extent.height)),
        );
    }
}

/// The mandatory `cpu.reference@1` oracle implementation id.
fn reference_impl() -> Result<ImplId> {
    ImplId::new("cpu", "reference", 1)
}

/// Read an optional finite float param, defaulting when absent/null.
fn optional_finite(params: &serde_json::Value, name: &str, default: f64, op: &str) -> Result<f64> {
    let value = match params.get(name) {
        None | Some(serde_json::Value::Null) => return Ok(default),
        Some(v) => v.as_f64().ok_or_else(|| {
            Error::new(
                ErrorClass::Schema,
                E_SDF_PARAM,
                format!("{op} `{name}` must be a number"),
            )
        })?,
    };
    if value.is_finite() {
        Ok(value)
    } else {
        Err(Error::new(
            ErrorClass::Schema,
            E_SDF_PARAM,
            format!("{op} `{name}` must be finite, got {value}"),
        ))
    }
}

// ---------------------------------------------------------------------------
// sdf.offset@1
// ---------------------------------------------------------------------------

/// The `sdf.offset@1` operation: grow/shrink a field by a signed distance.
#[derive(Debug, Clone, Copy, Default)]
pub struct SdfOffset;

impl SdfOffset {
    /// Construct the offset op.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    /// The declared manifest for `sdf.offset@1`.
    ///
    /// # Errors
    /// Propagates a [`schema`](ErrorClass::Schema) error if the hard-coded op /
    /// impl ids are invalid (they are not).
    pub fn manifest() -> Result<OperationManifest> {
        Ok(OperationManifest {
            id: OFFSET_OP_ID.parse()?,
            impl_version: 1,
            summary: "Grow or shrink a signed distance field by a signed physical distance: \
                      phi' = phi - distance_px. Positive distance grows the (negative-inside) \
                      region, negative shrinks it; the field stays a valid distance field away \
                      from new medial axes and the op composes exactly."
                .to_owned(),
            determinism: DeterminismTier::Exact,
            roi: RoiPolicy {
                category: RoiCategory::Pointwise,
                halo_px: None,
            },
            inputs: vec![sdf_input_port("The signed distance field to offset.")],
            outputs: vec![OutputSpec {
                name: "sdf".to_owned(),
                kind: ResourceKind::SdfMask,
                doc: "The offset field phi - distance_px (same extent and sign convention)."
                    .to_owned(),
            }],
            params: vec![ParamSpec {
                name: "distance_px".to_owned(),
                ty: ParamType::Float,
                unit: Some(ParamUnit::Pixels),
                required: true,
                default: None,
                choices: vec![],
                doc: "The signed physical pixel distance to offset by; positive grows the \
                      region, negative shrinks it."
                    .to_owned(),
            }],
            implementations: vec![reference_impl()?],
            test: sdf_test_metadata(
                "sdf.offset is the exact closed-form phi - d verified by the composition law, \
                 the zero-distance identity, and infinity preservation; there is no perceptual \
                 metric",
            ),
        })
    }
}

impl OpContract for SdfOffset {
    fn declared_inputs(&self) -> Vec<(String, ResourceKind)> {
        vec![("sdf".to_owned(), ResourceKind::SdfMask)]
    }
    fn declared_outputs(&self) -> Vec<(String, ResourceKind)> {
        vec![("sdf".to_owned(), ResourceKind::SdfMask)]
    }
    fn infer_outputs(
        &self,
        inputs: &Descriptors,
        _params: &serde_json::Value,
    ) -> Result<OutputDescriptors> {
        let extent = sdf_extent_of(inputs, OFFSET_OP_ID)?;
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
        let mut regions = InputRegions::new();
        full_sdf_region(inputs, &mut regions);
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

impl OpImplementation for SdfOffset {
    fn compute(
        &self,
        inputs: &InputValues,
        params: &serde_json::Value,
    ) -> std::result::Result<OutputValues, Error> {
        let sdf = sdf_value_of(inputs, OFFSET_OP_ID)?;
        let distance = require_finite(params, "distance_px", OFFSET_OP_ID)?;
        #[allow(
            clippy::cast_possible_truncation,
            reason = "distance is in the same f32 pixel units as the field; the narrowing is the contract"
        )]
        let d = distance as f32;
        // phi' = phi - d; +-inf - d = +-inf (finite d), preserving sentinels.
        let samples: Vec<f32> = sdf.samples().iter().map(|&phi| phi - d).collect();
        finish_sdf(sdf.extent(), samples)
    }
}

/// Read a required finite float param.
fn require_finite(params: &serde_json::Value, name: &str, op: &str) -> Result<f64> {
    let value = params
        .get(name)
        .and_then(serde_json::Value::as_f64)
        .ok_or_else(|| {
            Error::new(
                ErrorClass::Schema,
                E_SDF_PARAM,
                format!("{op} requires a numeric `{name}` parameter"),
            )
        })?;
    if value.is_finite() {
        Ok(value)
    } else {
        Err(Error::new(
            ErrorClass::Schema,
            E_SDF_PARAM,
            format!("{op} `{name}` must be finite, got {value}"),
        ))
    }
}

// ---------------------------------------------------------------------------
// sdf.to_mask@1
// ---------------------------------------------------------------------------

/// The `sdf.to_mask@1` operation: reconstruct a coverage mask from a field.
#[derive(Debug, Clone, Copy, Default)]
pub struct SdfToMask;

impl SdfToMask {
    /// Construct the reconstruction op.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    /// The declared manifest for `sdf.to_mask@1`.
    ///
    /// # Errors
    /// Propagates a [`schema`](ErrorClass::Schema) error if the hard-coded op /
    /// impl ids are invalid (they are not).
    pub fn manifest() -> Result<OperationManifest> {
        Ok(OperationManifest {
            id: TO_MASK_OP_ID.parse()?,
            impl_version: 1,
            summary: "Reconstruct a coverage Mask from a signed distance field with a smoothstep \
                      profile over a physical feather half-width: coverage is 1 deep inside, 0.5 \
                      on the zero contour, 0 outside the 2*half_width_px band. half_width_px = 0 \
                      is a hard step that round-trips a hard mask exactly."
                .to_owned(),
            determinism: DeterminismTier::Exact,
            roi: RoiPolicy {
                category: RoiCategory::Pointwise,
                halo_px: None,
            },
            inputs: vec![sdf_input_port("The signed distance field to reconstruct.")],
            outputs: vec![OutputSpec {
                name: "mask".to_owned(),
                kind: ResourceKind::Mask,
                doc: "The reconstructed coverage mask in [0, 1].".to_owned(),
            }],
            params: vec![
                ParamSpec {
                    name: "profile".to_owned(),
                    ty: ParamType::String,
                    unit: None,
                    required: false,
                    default: Some(serde_json::json!("smoothstep")),
                    choices: vec!["smoothstep".to_owned()],
                    doc: "The reconstruction profile across the feather band; only the Hermite \
                          smoothstep is supported."
                        .to_owned(),
                },
                ParamSpec {
                    name: "half_width_px".to_owned(),
                    ty: ParamType::Float,
                    unit: Some(ParamUnit::Pixels),
                    required: false,
                    default: Some(serde_json::json!(0.0)),
                    choices: vec![],
                    doc: "Half the physical feather width in pixels; the soft edge spans \
                          2*half_width_px centered on the zero contour. 0 is a hard step. Must \
                          be >= 0."
                        .to_owned(),
                },
            ],
            implementations: vec![reference_impl()?],
            test: sdf_test_metadata(
                "sdf.to_mask is the exact closed-form clamp+smoothstep verified by the contour \
                 round-trip, a measured feather width, and the hard-step identity; there is no \
                 perceptual metric",
            ),
        })
    }
}

impl OpContract for SdfToMask {
    fn declared_inputs(&self) -> Vec<(String, ResourceKind)> {
        vec![("sdf".to_owned(), ResourceKind::SdfMask)]
    }
    fn declared_outputs(&self) -> Vec<(String, ResourceKind)> {
        vec![("mask".to_owned(), ResourceKind::Mask)]
    }
    fn infer_outputs(
        &self,
        inputs: &Descriptors,
        _params: &serde_json::Value,
    ) -> Result<OutputDescriptors> {
        let extent = sdf_extent_of(inputs, TO_MASK_OP_ID)?;
        let mut out = OutputDescriptors::new();
        out.insert(
            "mask".to_owned(),
            ResourceDescriptor::Mask(mask_descriptor(extent)),
        );
        Ok(out)
    }
    fn required_inputs(
        &self,
        _requested_outputs: &OutputRegions,
        inputs: &Descriptors,
        _params: &serde_json::Value,
    ) -> Result<InputRegions> {
        let mut regions = InputRegions::new();
        full_sdf_region(inputs, &mut regions);
        Ok(regions)
    }
    fn validate_postconditions(
        &self,
        outputs: &OutputDescriptors,
        _params: &serde_json::Value,
    ) -> Result<Vec<AssertionResult>> {
        Ok(mask_postcondition(outputs))
    }
}

impl OpImplementation for SdfToMask {
    fn compute(
        &self,
        inputs: &InputValues,
        params: &serde_json::Value,
    ) -> std::result::Result<OutputValues, Error> {
        let sdf = sdf_value_of(inputs, TO_MASK_OP_ID)?;
        require_smoothstep(params)?;
        let half_width = optional_finite(params, "half_width_px", 0.0, TO_MASK_OP_ID)?;
        if half_width < 0.0 {
            return Err(Error::new(
                ErrorClass::Schema,
                E_SDF_PARAM,
                format!("{TO_MASK_OP_ID} `half_width_px` must be >= 0, got {half_width}"),
            ));
        }
        #[allow(
            clippy::cast_possible_truncation,
            reason = "half-width is in f32 pixel units matching the field; the narrowing is the contract"
        )]
        let h = half_width as f32;
        let samples: Vec<f32> = sdf
            .samples()
            .iter()
            .map(|&phi| reconstruct_coverage(phi, h))
            .collect();
        finish_mask(sdf.extent(), samples)
    }
}

/// Validate that the `profile` param, if present, is the supported smoothstep.
fn require_smoothstep(params: &serde_json::Value) -> Result<()> {
    match params.get("profile") {
        None | Some(serde_json::Value::Null) => Ok(()),
        Some(serde_json::Value::String(s)) if s == "smoothstep" => Ok(()),
        Some(other) => Err(Error::new(
            ErrorClass::Schema,
            E_SDF_PARAM,
            format!("{TO_MASK_OP_ID} `profile` must be \"smoothstep\", got {other}"),
        )),
    }
}

/// Reconstruct a single coverage sample from a signed distance `phi` and a
/// feather half-width `h` (negative-inside convention).
///
/// `h == 0` is a hard step (`1` iff `phi <= 0`). For `h > 0` the coverage is the
/// smoothstep of `t = clamp((h - phi) / 2h, 0, 1)`: `1` at `phi <= -h`, `0.5` at
/// `phi == 0`, `0` at `phi >= h`. `±∞` map to the fully-inside / fully-outside
/// limits.
fn reconstruct_coverage(phi: f32, h: f32) -> f32 {
    if h <= 0.0 {
        // Hard step at the zero contour: inside (phi <= 0) is full coverage.
        return if phi <= 0.0 { 1.0 } else { 0.0 };
    }
    if phi.is_infinite() {
        return if phi < 0.0 { 1.0 } else { 0.0 };
    }
    let t = ((h - phi) / (2.0 * h)).clamp(0.0, 1.0);
    smoothstep(t)
}

/// The Hermite smoothstep `3t² − 2t³` on an already-clamped `t ∈ [0, 1]`.
fn smoothstep(t: f32) -> f32 {
    t * t * 2.0f32.mul_add(-t, 3.0)
}

// ---------------------------------------------------------------------------
// shared finishers / postconditions
// ---------------------------------------------------------------------------

/// The shared `sdf` input port declaration.
fn sdf_input_port(doc: &str) -> InputSpec {
    InputSpec {
        name: "sdf".to_owned(),
        kind: ResourceKind::SdfMask,
        required: true,
        doc: doc.to_owned(),
    }
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
            E_SDF_BUFFER,
            format!("{OFFSET_OP_ID} produced an SDF buffer of unexpected length {actual}"),
        )
    })?;
    let mut out = OutputValues::new();
    out.insert("sdf".to_owned(), value);
    Ok(out)
}

/// Wrap a coverage buffer as a `Mask` output value.
fn finish_mask(extent: Extent, samples: Vec<f32>) -> std::result::Result<OutputValues, Error> {
    let value = ResourceValue::new(
        ResourceDescriptor::Mask(mask_descriptor(extent)),
        1,
        samples,
    )
    .map_err(|actual| {
        Error::new(
            ErrorClass::Execution,
            E_SDF_BUFFER,
            format!("{TO_MASK_OP_ID} produced a mask buffer of unexpected length {actual}"),
        )
    })?;
    let mut out = OutputValues::new();
    out.insert("mask".to_owned(), value);
    Ok(out)
}

/// The postcondition for an op producing an `sdf` `negative-inside` field.
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
                "sdf output is not negative-inside".to_owned(),
            )
        },
    ]
}

/// The postcondition for an op producing a `[0, 1]` coverage `mask`.
fn mask_postcondition(outputs: &OutputDescriptors) -> Vec<AssertionResult> {
    let Some(ResourceDescriptor::Mask(mask)) = outputs.get("mask") else {
        return vec![AssertionResult::fail(
            "produces_mask",
            "no `mask` output produced",
        )];
    };
    let unit_range = ValidRange::Bounded { min: 0.0, max: 1.0 };
    vec![
        AssertionResult::pass("produces_mask"),
        if mask.range == unit_range {
            AssertionResult::pass("coverage_in_unit_range")
        } else {
            AssertionResult::fail(
                "coverage_in_unit_range",
                "mask range is not the coverage range [0, 1]".to_owned(),
            )
        },
    ]
}

/// Verification declarations for an exact, single-reference SDF op: perceptual
/// does not apply; every other applicable category is covered by this module's
/// analytic fixtures, law/property, and metamorphic tests.
fn sdf_test_metadata(perceptual_reason: &str) -> TestMetadata {
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
