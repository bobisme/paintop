//! The signed-distance-field boolean algebra `sdf.union@1`, `sdf.intersect@1`,
//! and `sdf.subtract@1` (`OP_CATALOG` §4, `ALIEN_OPS` §2.1).
//!
//! Constructive-solid-geometry boolean operations expressed directly on signed
//! distance fields under the project's `negative-inside` convention
//! (`IR_SPEC` §7.4), where a pixel is *inside* a region iff its field value is
//! `< 0`:
//!
//! - **`sdf.union`** — `A ∪ B = min(φ_A, φ_B)`: a pixel is inside the union iff it
//!   is inside *either* operand, i.e. iff `min(φ_A, φ_B) < 0`.
//! - **`sdf.intersect`** — `A ∩ B = max(φ_A, φ_B)`: inside iff inside *both*.
//! - **`sdf.subtract`** — `A − B = max(φ_A, −φ_B)`: inside `A` and *outside* `B`
//!   (negating `φ_B` flips its inside/outside, then intersect).
//!
//! # Exactness vs. the true distance
//!
//! `min`/`max` of two distance fields is the exact signed distance only away from
//! the new medial axes the boolean introduces; near a freshly-created concave
//! corner the combined field is a conservative *bound* on the true distance, not
//! the true distance (this is the standard CSG-on-SDF caveat). What **is** exact —
//! and what these ops promise — is the **zero contour**: thresholding the result
//! at `φ < 0` reproduces the hard-mask boolean algebra
//! (`mask.union`/`intersect`/`subtract`) of the operands' inside sets exactly.
//! The acceptance suite checks that zero-contour agreement, plus commutativity of
//! union/intersect and the difference law of subtract, and that `|∇φ| ≈ 1` is
//! preserved away from the new medial axes.
//!
//! # Infinities
//!
//! A degenerate operand carries `±∞` (an empty or full field). `min`/`max`
//! propagate these correctly (`min(x, −∞) = −∞`, `max(x, +∞) = +∞`), and
//! `subtract` negates `φ_B` first (`−(+∞) = −∞`), so the algebra is closed over
//! the degenerate fields with no `NaN`.
//!
//! # Determinism
//!
//! [`DeterminismTier::Exact`]: each output sample is `min`/`max`/`−` of the
//! corresponding input samples — IEEE-754 operations that are bit-identical
//! across platforms and pointwise, so a tile boundary never changes a sample.

use paintop_core::executor::{InputValues, OpImplementation, OutputValues, ResourceValue};
use paintop_ir::{
    AssertionResult, CoordinateConvention, Descriptors, DeterminismTier, Error, ErrorClass,
    ErrorContext, Extent, ImplId, InputRegions, InputSpec, OpContract, OperationManifest,
    OutputDescriptors, OutputRegions, OutputSpec, Rect, ResourceDescriptor, ResourceKind, Result,
    RoiCategory, RoiPolicy, ScalarType, SdfDescriptor, SdfSign, SdfUnits, TestMetadata,
};

/// The canonical id of the SDF union (CSG join) operation.
pub const UNION_OP_ID: &str = "sdf.union@1";
/// The canonical id of the SDF intersect (CSG meet) operation.
pub const INTERSECT_OP_ID: &str = "sdf.intersect@1";
/// The canonical id of the SDF subtract (CSG difference) operation.
pub const SUBTRACT_OP_ID: &str = "sdf.subtract@1";

/// A required SDF input was absent or was not an `SdfMask`.
pub const E_SDF_ALGEBRA_INPUT: &str = "E_SDF_ALGEBRA_INPUT";
/// Two SDF inputs disagree on extent.
pub const E_SDF_ALGEBRA_SHAPE: &str = "E_SDF_ALGEBRA_SHAPE";
/// A produced SDF buffer had an unexpected length.
pub const E_SDF_ALGEBRA_BUFFER: &str = "E_SDF_ALGEBRA_BUFFER";

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

/// The pointwise kernel a binary SDF boolean op applies sample-for-sample.
#[derive(Debug, Clone, Copy)]
enum SdfKernel {
    /// `A ∪ B = min(φ_A, φ_B)`.
    Union,
    /// `A ∩ B = max(φ_A, φ_B)`.
    Intersect,
    /// `A − B = max(φ_A, −φ_B)`.
    Subtract,
}

impl SdfKernel {
    /// Apply the kernel to a single pair of signed-distance samples.
    #[must_use]
    fn apply(self, a: f32, b: f32) -> f32 {
        match self {
            Self::Union => a.min(b),
            Self::Intersect => a.max(b),
            Self::Subtract => a.max(-b),
        }
    }
}

/// A binary SDF-algebra operation parameterized by its pointwise kernel.
#[derive(Debug, Clone, Copy)]
pub struct SdfBooleanOp {
    id: &'static str,
    kernel: SdfKernel,
}

impl SdfBooleanOp {
    /// The `sdf.union@1` operation.
    #[must_use]
    pub const fn union() -> Self {
        Self {
            id: UNION_OP_ID,
            kernel: SdfKernel::Union,
        }
    }

    /// The `sdf.intersect@1` operation.
    #[must_use]
    pub const fn intersect() -> Self {
        Self {
            id: INTERSECT_OP_ID,
            kernel: SdfKernel::Intersect,
        }
    }

    /// The `sdf.subtract@1` operation.
    #[must_use]
    pub const fn subtract() -> Self {
        Self {
            id: SUBTRACT_OP_ID,
            kernel: SdfKernel::Subtract,
        }
    }

    /// The declared manifest for `sdf.union@1`.
    ///
    /// # Errors
    /// Propagates a [`schema`](ErrorClass::Schema) error if the op id is invalid.
    pub fn union_manifest() -> Result<OperationManifest> {
        Self::union().manifest(
            "Union (CSG join) of two signed distance fields: phi = min(phi_a, phi_b). The zero \
             contour is the union of the operands' inside sets; exact away from new medial axes.",
            "The union field min(phi_a, phi_b) (negative-inside).",
            "sdf.union is the exact closed-form min verified by zero-contour agreement with the \
             hard-mask union, commutativity, and gradient-norm preservation; there is no \
             perceptual metric",
        )
    }

    /// The declared manifest for `sdf.intersect@1`.
    ///
    /// # Errors
    /// Propagates a [`schema`](ErrorClass::Schema) error if the op id is invalid.
    pub fn intersect_manifest() -> Result<OperationManifest> {
        Self::intersect().manifest(
            "Intersection (CSG meet) of two signed distance fields: phi = max(phi_a, phi_b). The \
             zero contour is the intersection of the operands' inside sets.",
            "The intersection field max(phi_a, phi_b) (negative-inside).",
            "sdf.intersect is the exact closed-form max verified by zero-contour agreement with \
             the hard-mask intersection, commutativity, and gradient-norm preservation; there is \
             no perceptual metric",
        )
    }

    /// The declared manifest for `sdf.subtract@1`.
    ///
    /// # Errors
    /// Propagates a [`schema`](ErrorClass::Schema) error if the op id is invalid.
    pub fn subtract_manifest() -> Result<OperationManifest> {
        Self::subtract().manifest(
            "Difference (CSG subtraction) of two signed distance fields: phi = max(phi_a, \
             -phi_b), i.e. inside A and outside B. The zero contour is the relative complement \
             of the operands' inside sets.",
            "The difference field max(phi_a, -phi_b) (negative-inside).",
            "sdf.subtract is the exact closed-form max(a, -b) verified by zero-contour agreement \
             with the hard-mask subtraction and the difference law; there is no perceptual metric",
        )
    }

    /// Build the manifest for this binary op.
    fn manifest(
        self,
        summary: &str,
        out_doc: &str,
        perceptual_reason: &str,
    ) -> Result<OperationManifest> {
        Ok(OperationManifest {
            id: self.id.parse()?,
            impl_version: 1,
            summary: summary.to_owned(),
            determinism: DeterminismTier::Exact,
            roi: RoiPolicy {
                category: RoiCategory::Pointwise,
                halo_px: None,
            },
            inputs: vec![
                sdf_input_port("a", "The left-hand signed distance field."),
                sdf_input_port(
                    "b",
                    "The right-hand signed distance field (same extent as `a`).",
                ),
            ],
            outputs: vec![OutputSpec {
                name: "sdf".to_owned(),
                kind: ResourceKind::SdfMask,
                doc: out_doc.to_owned(),
            }],
            params: vec![],
            implementations: vec![reference_impl()?],
            test: sdf_algebra_test_metadata(perceptual_reason),
        })
    }
}

impl OpContract for SdfBooleanOp {
    fn declared_inputs(&self) -> Vec<(String, ResourceKind)> {
        vec![
            ("a".to_owned(), ResourceKind::SdfMask),
            ("b".to_owned(), ResourceKind::SdfMask),
        ]
    }
    fn declared_outputs(&self) -> Vec<(String, ResourceKind)> {
        vec![("sdf".to_owned(), ResourceKind::SdfMask)]
    }
    fn infer_outputs(
        &self,
        inputs: &Descriptors,
        _params: &serde_json::Value,
    ) -> Result<OutputDescriptors> {
        let a = sdf_extent_of(inputs, "a", self.id)?;
        let b = sdf_extent_of(inputs, "b", self.id)?;
        require_same_extent(a, b, self.id)?;
        let mut out = OutputDescriptors::new();
        out.insert(
            "sdf".to_owned(),
            ResourceDescriptor::SdfMask(sdf_descriptor(a)),
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
        full_region(inputs, "a", &mut regions);
        full_region(inputs, "b", &mut regions);
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

impl OpImplementation for SdfBooleanOp {
    fn compute(
        &self,
        inputs: &InputValues,
        _params: &serde_json::Value,
    ) -> std::result::Result<OutputValues, Error> {
        let a = sdf_value_of(inputs, "a", self.id)?;
        let b = sdf_value_of(inputs, "b", self.id)?;
        require_same_extent(a.extent(), b.extent(), self.id)?;
        let kernel = self.kernel;
        let samples: Vec<f32> = a
            .samples()
            .iter()
            .zip(b.samples().iter())
            .map(|(&x, &y)| kernel.apply(x, y))
            .collect();
        finish_sdf(a.extent(), samples, self.id)
    }
}

// ---------------------------------------------------------------------------
// shared
// ---------------------------------------------------------------------------

/// A named `SdfMask` input port declaration.
fn sdf_input_port(name: &str, doc: &str) -> InputSpec {
    InputSpec {
        name: name.to_owned(),
        kind: ResourceKind::SdfMask,
        required: true,
        doc: doc.to_owned(),
    }
}

/// The extent of a named `SdfMask` input descriptor, or a typed error.
fn sdf_extent_of(inputs: &Descriptors, port: &str, op: &str) -> Result<Extent> {
    match inputs.get(port) {
        Some(ResourceDescriptor::SdfMask(d)) => Ok(d.extent),
        Some(_) => Err(Error::new(
            ErrorClass::Type,
            E_SDF_ALGEBRA_INPUT,
            format!("{op} input `{port}` must be an SdfMask"),
        )),
        None => Err(Error::new(
            ErrorClass::Reference,
            E_SDF_ALGEBRA_INPUT,
            format!("{op} requires an `{port}` SdfMask input"),
        )),
    }
}

/// The value of a named SDF input, or a typed error if absent.
fn sdf_value_of<'a>(
    inputs: &'a InputValues,
    port: &str,
    op: &str,
) -> std::result::Result<&'a ResourceValue, Error> {
    inputs.get(port).ok_or_else(|| {
        Error::new(
            ErrorClass::Reference,
            E_SDF_ALGEBRA_INPUT,
            format!("{op} requires an `{port}` SdfMask input value"),
        )
    })
}

/// Require two extents to be equal, raising a typed shape error otherwise.
fn require_same_extent(a: Extent, b: Extent, op: &str) -> std::result::Result<(), Error> {
    if a == b {
        Ok(())
    } else {
        Err(Error::new(
            ErrorClass::Type,
            E_SDF_ALGEBRA_SHAPE,
            format!("{op} requires both fields to share an extent"),
        )
        .with_context(ErrorContext::default().with_actual(format!("a {a:?} vs b {b:?}"))))
    }
}

/// The full-domain input region a binary SDF op demands of `port`, if present.
fn full_region(inputs: &Descriptors, port: &str, regions: &mut InputRegions) {
    if let Some(d) = inputs.get(port) {
        let extent = d.extent();
        regions.insert(
            port.to_owned(),
            Rect::new(0, 0, i64::from(extent.width), i64::from(extent.height)),
        );
    }
}

/// Wrap a signed-distance buffer as an `SdfMask` output value.
fn finish_sdf(
    extent: Extent,
    samples: Vec<f32>,
    op: &str,
) -> std::result::Result<OutputValues, Error> {
    let value = ResourceValue::new(
        ResourceDescriptor::SdfMask(sdf_descriptor(extent)),
        1,
        samples,
    )
    .map_err(|actual| {
        Error::new(
            ErrorClass::Execution,
            E_SDF_ALGEBRA_BUFFER,
            format!("{op} produced an SDF buffer of unexpected length {actual}"),
        )
    })?;
    let mut out = OutputValues::new();
    out.insert("sdf".to_owned(), value);
    Ok(out)
}

/// The postcondition: a `negative-inside` `sdf` output.
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

/// The mandatory `cpu.reference@1` oracle implementation id.
fn reference_impl() -> Result<ImplId> {
    ImplId::new("cpu", "reference", 1)
}

/// Verification declarations for an exact, single-reference SDF algebra op.
fn sdf_algebra_test_metadata(perceptual_reason: &str) -> TestMetadata {
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
