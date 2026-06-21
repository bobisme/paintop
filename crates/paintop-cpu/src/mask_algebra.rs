//! The mask boolean-algebra operations `mask.invert@1`, `mask.union@1`,
//! `mask.intersect@1`, and `mask.subtract@1` (`OP_CATALOG` §4,
//! `AGENT_VERIFICATION` §2.5).
//!
//! These ops form a **fuzzy** boolean algebra over coverage masks in `[0, 1]`.
//! The chosen algebra is the standard Zadeh / Gödel min-max algebra, which is the
//! unique pointwise-monotone extension of crisp `{0, 1}` boolean logic that the
//! coverage masks demand:
//!
//! - **`mask.invert`** — complement: `¬a = 1 − a`.
//! - **`mask.union`** — join: `a ∪ b = max(a, b)`.
//! - **`mask.intersect`** — meet: `a ∩ b = min(a, b)`.
//! - **`mask.subtract`** — relative complement: `a − b = a ∩ ¬b = min(a, 1 − b)`.
//!
//! # Hard vs. soft (fuzzy) algebra
//!
//! A *hard* mask is one whose every sample is exactly `0` or `1`. On hard masks
//! the min-max algebra collapses to crisp boolean logic, so the full
//! `AGENT_VERIFICATION` §2.5 law suite holds **exactly** (bit-identically):
//! commutativity, associativity, idempotence (`A ∪ A = A`, `A ∩ A = A`), the
//! complement laws (`A ∪ ¬A = full`, `A ∩ ¬A = empty`), De Morgan
//! (`¬(A ∪ B) = ¬A ∩ ¬B`), `A − A = ∅`, and the double-inverse `¬¬A = A`.
//!
//! On *soft* masks only the laws the min-max algebra actually defines hold. The
//! min-max algebra is a bounded distributive lattice with an involutive
//! order-reversing complement, so the following hold for **all** coverage values:
//! commutativity, associativity, idempotence, De Morgan, double-inverse, and the
//! absorption/distributive lattice laws. The complement laws (`A ∪ ¬A = full`,
//! `A ∩ ¬A = empty`) and `A − A = ∅` are the *excluded-middle* laws; they do
//! **not** hold for a soft mask (e.g. `max(0.5, 0.5) = 0.5 ≠ 1`), which is the
//! defining feature of a fuzzy — rather than crisp — algebra.
//!
//! # Determinism
//!
//! Every op is [`Exact`](DeterminismTier::Exact): each output sample is a
//! closed-form function (`1 − x`, `min`, `max`) of the corresponding input
//! sample(s), using only IEEE-754 subtraction and `min`/`max`, which are
//! bit-identical across platforms. The result is pointwise, so a tile boundary
//! never changes a sample.

use paintop_core::executor::{InputValues, OpImplementation, OutputValues, ResourceValue};
use paintop_ir::{
    AssertionResult, CoordinateConvention, Descriptors, DeterminismTier, Error, ErrorClass,
    ErrorContext, Extent, ImplId, InputRegions, InputSpec, MaskDescriptor, MaskMeaning, OpContract,
    OperationManifest, OutputDescriptors, OutputRegions, OutputSpec, Rect, ResourceDescriptor,
    ResourceKind, Result, RoiCategory, RoiPolicy, ScalarType, TestMetadata, ValidRange,
};

/// The canonical id of the mask-complement operation.
pub const INVERT_OP_ID: &str = "mask.invert@1";
/// The canonical id of the mask-union (join) operation.
pub const UNION_OP_ID: &str = "mask.union@1";
/// The canonical id of the mask-intersect (meet) operation.
pub const INTERSECT_OP_ID: &str = "mask.intersect@1";
/// The canonical id of the mask-subtract (relative complement) operation.
pub const SUBTRACT_OP_ID: &str = "mask.subtract@1";

/// A required mask input was absent or carried no sample buffer.
pub const E_ALGEBRA_INPUT: &str = "E_ALGEBRA_INPUT";
/// Two mask inputs to a binary op disagree on extent.
pub const E_ALGEBRA_SHAPE: &str = "E_ALGEBRA_SHAPE";
/// The produced mask buffer had an unexpected length.
pub const E_ALGEBRA_BUFFER: &str = "E_ALGEBRA_BUFFER";

/// The coverage mask descriptor produced for `extent`.
const fn mask_descriptor(extent: Extent) -> MaskDescriptor {
    MaskDescriptor {
        extent,
        scalar: ScalarType::F32,
        range: ValidRange::Bounded { min: 0.0, max: 1.0 },
        meaning: MaskMeaning::Coverage,
        coordinates: CoordinateConvention::PixelCenterUpperLeft,
    }
}

/// A mask input port declaration.
fn mask_input_port(name: &str, doc: &str) -> InputSpec {
    InputSpec {
        name: name.to_owned(),
        kind: ResourceKind::Mask,
        required: true,
        doc: doc.to_owned(),
    }
}

/// The `mask` output port shared by every algebra op.
fn mask_output_port(doc: &str) -> OutputSpec {
    OutputSpec {
        name: "mask".to_owned(),
        kind: ResourceKind::Mask,
        doc: doc.to_owned(),
    }
}

/// The descriptor of a named mask input, or a typed [`reference`](ErrorClass::Reference)
/// error if it is absent or not a mask.
fn mask_descriptor_of<'a>(
    inputs: &'a Descriptors,
    port: &str,
    op: &str,
) -> Result<&'a MaskDescriptor> {
    match inputs.get(port) {
        Some(ResourceDescriptor::Mask(d)) => Ok(d),
        Some(_) => Err(Error::new(
            ErrorClass::Type,
            E_ALGEBRA_INPUT,
            format!("{op} input `{port}` must be a Mask"),
        )),
        None => Err(Error::new(
            ErrorClass::Reference,
            E_ALGEBRA_INPUT,
            format!("{op} requires a `{port}` mask input"),
        )),
    }
}

/// The value of a named mask input, or a typed error if it is absent.
fn mask_value_of<'a>(
    inputs: &'a InputValues,
    port: &str,
    op: &str,
) -> std::result::Result<&'a ResourceValue, Error> {
    inputs.get(port).ok_or_else(|| {
        Error::new(
            ErrorClass::Reference,
            E_ALGEBRA_INPUT,
            format!("{op} requires a `{port}` mask input value"),
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
            E_ALGEBRA_SHAPE,
            format!("{op} requires both masks to share an extent"),
        )
        .with_context(ErrorContext::default().with_actual(format!("a {a:?} vs b {b:?}"))))
    }
}

/// Wrap a coverage sample buffer as a mask value, mapping a length mismatch to a
/// typed buffer error.
fn finish_mask(
    extent: Extent,
    samples: Vec<f32>,
    op: &str,
) -> std::result::Result<OutputValues, Error> {
    let value = ResourceValue::new(
        ResourceDescriptor::Mask(mask_descriptor(extent)),
        1,
        samples,
    )
    .map_err(|actual| {
        Error::new(
            ErrorClass::Execution,
            E_ALGEBRA_BUFFER,
            format!("{op} produced a mask buffer of unexpected length {actual}"),
        )
    })?;
    let mut out = OutputValues::new();
    out.insert("mask".to_owned(), value);
    Ok(out)
}

/// The full-domain input region a binary algebra op demands of `port`, if present.
fn full_region(inputs: &Descriptors, port: &str, regions: &mut InputRegions) {
    if let Some(d) = inputs.get(port) {
        let extent = d.extent();
        regions.insert(
            port.to_owned(),
            Rect::new(0, 0, i64::from(extent.width), i64::from(extent.height)),
        );
    }
}

/// The coverage postcondition shared by the algebra ops: a `[0, 1]` mask output.
fn coverage_postcondition(outputs: &OutputDescriptors) -> Vec<AssertionResult> {
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
                format!(
                    "mask range {:?} is not the coverage range [0, 1]",
                    mask.range
                ),
            )
        },
    ]
}

// ---------------------------------------------------------------------------
// mask.invert@1 (unary)
// ---------------------------------------------------------------------------

/// The `mask.invert@1` operation: a coverage mask → its complement `1 − a`.
#[derive(Debug, Clone, Copy, Default)]
pub struct InvertMask;

impl InvertMask {
    /// Construct the invert operation.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    /// The declared manifest for `mask.invert@1`.
    ///
    /// # Errors
    /// Propagates the [`schema`](ErrorClass::Schema) error if the hard-coded op /
    /// impl ids are somehow invalid (they are not).
    pub fn manifest() -> Result<OperationManifest> {
        Ok(OperationManifest {
            id: INVERT_OP_ID.parse()?,
            impl_version: 1,
            summary: "Complement a coverage Mask: each output sample is 1 - a, the fuzzy \
                      (min-max algebra) complement."
                .to_owned(),
            determinism: DeterminismTier::Exact,
            roi: RoiPolicy {
                category: RoiCategory::Pointwise,
                halo_px: None,
            },
            inputs: vec![mask_input_port("mask", "The coverage mask to complement.")],
            outputs: vec![mask_output_port("The complemented mask 1 - a in [0, 1].")],
            params: vec![],
            implementations: vec![reference_impl()?],
            test: algebra_test_metadata(
                "the complement is an exact closed-form 1 - a verified by fixtures, the \
                 double-inverse property, and analytic equality; there is no perceptual metric",
            ),
        })
    }
}

impl OpContract for InvertMask {
    fn declared_inputs(&self) -> Vec<(String, ResourceKind)> {
        vec![("mask".to_owned(), ResourceKind::Mask)]
    }
    fn declared_outputs(&self) -> Vec<(String, ResourceKind)> {
        vec![("mask".to_owned(), ResourceKind::Mask)]
    }
    fn infer_outputs(
        &self,
        inputs: &Descriptors,
        _params: &serde_json::Value,
    ) -> Result<OutputDescriptors> {
        let extent = mask_descriptor_of(inputs, "mask", INVERT_OP_ID)?.extent;
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
        full_region(inputs, "mask", &mut regions);
        Ok(regions)
    }
    fn validate_postconditions(
        &self,
        outputs: &OutputDescriptors,
        _params: &serde_json::Value,
    ) -> Result<Vec<AssertionResult>> {
        Ok(coverage_postcondition(outputs))
    }
}

impl OpImplementation for InvertMask {
    fn compute(
        &self,
        inputs: &InputValues,
        _params: &serde_json::Value,
    ) -> std::result::Result<OutputValues, Error> {
        let mask = mask_value_of(inputs, "mask", INVERT_OP_ID)?;
        let samples: Vec<f32> = mask.samples().iter().map(|&a| 1.0 - a).collect();
        finish_mask(mask.extent(), samples, INVERT_OP_ID)
    }
}

// ---------------------------------------------------------------------------
// binary ops: mask.union@1 / mask.intersect@1 / mask.subtract@1
// ---------------------------------------------------------------------------

/// The pointwise kernel a binary algebra op applies sample-for-sample.
#[derive(Debug, Clone, Copy)]
enum BinaryKernel {
    /// `a ∪ b = max(a, b)`.
    Union,
    /// `a ∩ b = min(a, b)`.
    Intersect,
    /// `a − b = min(a, 1 − b)`.
    Subtract,
}

impl BinaryKernel {
    /// Apply the kernel to a single pair of coverage samples.
    #[must_use]
    fn apply(self, a: f32, b: f32) -> f32 {
        match self {
            Self::Union => a.max(b),
            Self::Intersect => a.min(b),
            Self::Subtract => a.min(1.0 - b),
        }
    }
}

/// A binary mask-algebra operation parameterized by its pointwise kernel.
#[derive(Debug, Clone, Copy)]
pub struct BinaryMaskOp {
    id: &'static str,
    kernel: BinaryKernel,
}

impl BinaryMaskOp {
    /// The `mask.union@1` operation.
    #[must_use]
    pub const fn union() -> Self {
        Self {
            id: UNION_OP_ID,
            kernel: BinaryKernel::Union,
        }
    }

    /// The `mask.intersect@1` operation.
    #[must_use]
    pub const fn intersect() -> Self {
        Self {
            id: INTERSECT_OP_ID,
            kernel: BinaryKernel::Intersect,
        }
    }

    /// The `mask.subtract@1` operation.
    #[must_use]
    pub const fn subtract() -> Self {
        Self {
            id: SUBTRACT_OP_ID,
            kernel: BinaryKernel::Subtract,
        }
    }

    /// The declared manifest for `mask.union@1`.
    ///
    /// # Errors
    /// Propagates a [`schema`](ErrorClass::Schema) error if the op id is invalid.
    pub fn union_manifest() -> Result<OperationManifest> {
        Self::union().manifest(
            "Union (join) of two coverage Masks: each output sample is max(a, b), the fuzzy \
             (min-max algebra) join.",
            "The union mask max(a, b) in [0, 1].",
            "union is the exact closed-form max(a, b) verified by fixtures and the lattice / \
             De Morgan / idempotence laws; there is no perceptual metric",
        )
    }

    /// The declared manifest for `mask.intersect@1`.
    ///
    /// # Errors
    /// Propagates a [`schema`](ErrorClass::Schema) error if the op id is invalid.
    pub fn intersect_manifest() -> Result<OperationManifest> {
        Self::intersect().manifest(
            "Intersection (meet) of two coverage Masks: each output sample is min(a, b), the \
             fuzzy (min-max algebra) meet.",
            "The intersection mask min(a, b) in [0, 1].",
            "intersection is the exact closed-form min(a, b) verified by fixtures and the \
             lattice / De Morgan / idempotence laws; there is no perceptual metric",
        )
    }

    /// The declared manifest for `mask.subtract@1`.
    ///
    /// # Errors
    /// Propagates a [`schema`](ErrorClass::Schema) error if the op id is invalid.
    pub fn subtract_manifest() -> Result<OperationManifest> {
        Self::subtract().manifest(
            "Relative complement of two coverage Masks: a - b = min(a, 1 - b), the fuzzy \
             (min-max algebra) difference a intersect not-b.",
            "The difference mask min(a, 1 - b) in [0, 1].",
            "subtract is the exact closed-form min(a, 1 - b) verified by fixtures and the \
             A - A = empty (hard) law; there is no perceptual metric",
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
                mask_input_port("a", "The left-hand coverage mask."),
                mask_input_port("b", "The right-hand coverage mask (same extent as `a`)."),
            ],
            outputs: vec![mask_output_port(out_doc)],
            params: vec![],
            implementations: vec![reference_impl()?],
            test: algebra_test_metadata(perceptual_reason),
        })
    }
}

impl OpContract for BinaryMaskOp {
    fn declared_inputs(&self) -> Vec<(String, ResourceKind)> {
        vec![
            ("a".to_owned(), ResourceKind::Mask),
            ("b".to_owned(), ResourceKind::Mask),
        ]
    }
    fn declared_outputs(&self) -> Vec<(String, ResourceKind)> {
        vec![("mask".to_owned(), ResourceKind::Mask)]
    }
    fn infer_outputs(
        &self,
        inputs: &Descriptors,
        _params: &serde_json::Value,
    ) -> Result<OutputDescriptors> {
        let a = mask_descriptor_of(inputs, "a", self.id)?.extent;
        let b = mask_descriptor_of(inputs, "b", self.id)?.extent;
        require_same_extent(a, b, self.id)?;
        let mut out = OutputDescriptors::new();
        out.insert(
            "mask".to_owned(),
            ResourceDescriptor::Mask(mask_descriptor(a)),
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
        Ok(coverage_postcondition(outputs))
    }
}

impl OpImplementation for BinaryMaskOp {
    fn compute(
        &self,
        inputs: &InputValues,
        _params: &serde_json::Value,
    ) -> std::result::Result<OutputValues, Error> {
        let a = mask_value_of(inputs, "a", self.id)?;
        let b = mask_value_of(inputs, "b", self.id)?;
        require_same_extent(a.extent(), b.extent(), self.id)?;
        let kernel = self.kernel;
        let samples: Vec<f32> = a
            .samples()
            .iter()
            .zip(b.samples().iter())
            .map(|(&x, &y)| kernel.apply(x, y))
            .collect();
        finish_mask(a.extent(), samples, self.id)
    }
}

// ---------------------------------------------------------------------------
// shared
// ---------------------------------------------------------------------------

/// The mandatory `cpu.reference@1` oracle implementation id.
fn reference_impl() -> Result<ImplId> {
    ImplId::new("cpu", "reference", 1)
}

/// Verification declarations for an exact, single-reference mask algebra op:
/// perceptual does not apply; every other applicable category is covered by this
/// module's exact fixtures, law/property, and metamorphic tests.
fn algebra_test_metadata(perceptual_reason: &str) -> TestMetadata {
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
