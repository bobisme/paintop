//! The primitive mask constructors `mask.empty@1`, `mask.full@1`, and
//! `mask.rect@1` (`OP_CATALOG` §3, `AGENT_VERIFICATION` §3.6, `IR_SPEC` §8.1).
//!
//! These three ops manufacture a coverage [`Mask`](ResourceKind::Mask) whose
//! pixel extent is copied from an `extent_from` input (D3: the resource a mask is
//! sized to is an input port, not a param). They are the base cases of the mask
//! calculus:
//!
//! - **`mask.empty`** — every sample is `0` (no coverage).
//! - **`mask.full`** — every sample is `1` (full coverage).
//! - **`mask.rect`** — an axis-aligned rectangle with **analytic antialiasing**:
//!   each pixel's coverage is the exact area of the intersection of its unit cell
//!   with the requested half-open rect, in `[0, 1]`.
//!
//! # The rectangle's coverage model
//!
//! `mask.rect` takes the rect in the project's half-open pixel convention
//! (`plan.md` §8.1): the rect covers the continuous region `[x0, x1) × [y0, y1)`.
//! Pixel cell `(i, j)` occupies the continuous square `[i, i+1) × [j, j+1)`, so
//! its coverage is
//!
//! ```text
//! coverage(i, j) = overlap_x * overlap_y
//! overlap_x      = clamp(min(x1, i+1) - max(x0, i), 0, 1)
//! overlap_y      = clamp(min(y1, j+1) - max(y0, j), 0, 1)
//! ```
//!
//! the exact fractional area the rect covers of that cell. For an
//! **integer-aligned** rect every overlap is exactly `0` or `1`, which reproduces
//! the half-open convention precisely: pixel column `x1` is excluded, and the
//! column just left of it is the last fully-covered column. For a fractional rect
//! the boundary pixels carry the exact partial area (analytic antialiasing).
//! Summed over every pixel, the coverage equals the rect's analytic area
//! `(x1 - x0) * (y1 - y0)` (clipped to the image), so the rendered area converges
//! to — in fact equals — the analytic area.
//!
//! # Determinism
//!
//! All three ops are [`Exact`](DeterminismTier::Exact): the coverage is a
//! closed-form function of integer pixel indices and the (finite) rect bounds
//! using only `min`/`max`/subtraction/multiplication, which are bit-identical
//! IEEE-754 operations across platforms. The coverage depends only on the pixel's
//! own cell, so a tile boundary never changes a sample.

use paintop_core::executor::{InputValues, OpImplementation, OutputValues, ResourceValue};
use paintop_ir::{
    AssertionResult, CoordinateConvention, Descriptors, DeterminismTier, Error, ErrorClass,
    ErrorContext, Extent, ImplId, InputRegions, InputSpec, MaskDescriptor, MaskMeaning, OpContract,
    OperationManifest, OutputDescriptors, OutputRegions, OutputSpec, ParamSpec, ParamType,
    ParamUnit, Rect, ResourceDescriptor, ResourceKind, Result, RoiCategory, RoiPolicy, ScalarType,
    TestMetadata, ValidRange,
};

/// The canonical id of the empty-mask operation.
pub const EMPTY_OP_ID: &str = "mask.empty@1";
/// The canonical id of the full-mask operation.
pub const FULL_OP_ID: &str = "mask.full@1";
/// The canonical id of the rectangle-mask operation.
pub const RECT_OP_ID: &str = "mask.rect@1";

/// The `extent_from` input was absent or carried no descriptor to size the mask.
pub const E_MASK_INPUT: &str = "E_MASK_INPUT";
/// A `mask.rect` geometry parameter was missing, the wrong shape, or non-finite.
pub const E_RECT_PARAM: &str = "E_RECT_PARAM";

/// The mask descriptor produced for `extent`: a coverage mask in `[0, 1]`, `f32`,
/// sharing the project's pixel convention.
const fn mask_descriptor(extent: Extent) -> MaskDescriptor {
    MaskDescriptor {
        extent,
        scalar: ScalarType::F32,
        range: ValidRange::Bounded { min: 0.0, max: 1.0 },
        meaning: MaskMeaning::Coverage,
        coordinates: CoordinateConvention::PixelCenterUpperLeft,
    }
}

/// The pixel extent of a named extent-source input descriptor.
fn extent_of(inputs: &Descriptors, op: &str) -> Result<Extent> {
    let source = inputs.get("extent_from").ok_or_else(|| {
        Error::new(
            ErrorClass::Reference,
            E_MASK_INPUT,
            format!("{op} requires an `extent_from` input"),
        )
    })?;
    Ok(source.extent())
}

/// The pixel extent of a named extent-source input value.
fn extent_of_value(inputs: &InputValues, op: &str) -> std::result::Result<Extent, Error> {
    let source = inputs.get("extent_from").ok_or_else(|| {
        Error::new(
            ErrorClass::Reference,
            E_MASK_INPUT,
            format!("{op} requires an `extent_from` input value"),
        )
    })?;
    Ok(source.extent())
}

/// Build a constant-coverage mask value (`fill` everywhere) for `extent`.
fn constant_mask(extent: Extent, fill: f32, op: &str) -> std::result::Result<ResourceValue, Error> {
    let pixels = (extent.width as usize).saturating_mul(extent.height as usize);
    ResourceValue::new(
        ResourceDescriptor::Mask(mask_descriptor(extent)),
        1,
        vec![fill; pixels],
    )
    .map_err(|actual| {
        Error::new(
            ErrorClass::Execution,
            E_MASK_INPUT,
            format!("{op} produced a mask buffer of unexpected length {actual}"),
        )
    })
}

/// The `extent_from` input port shared by every primitive mask op.
fn extent_from_port() -> InputSpec {
    InputSpec {
        name: "extent_from".to_owned(),
        kind: ResourceKind::Image,
        required: true,
        doc: "The resource whose pixel extent the produced mask matches.".to_owned(),
    }
}

/// The `mask` output port shared by every primitive mask op.
fn mask_output_port(doc: &str) -> OutputSpec {
    OutputSpec {
        name: "mask".to_owned(),
        kind: ResourceKind::Mask,
        doc: doc.to_owned(),
    }
}

/// The shared `extent_from` → `mask` declared input/output ports.
fn declared_extent_inputs() -> Vec<(String, ResourceKind)> {
    vec![("extent_from".to_owned(), ResourceKind::Image)]
}

/// The shared `mask` declared output port.
fn declared_mask_outputs() -> Vec<(String, ResourceKind)> {
    vec![("mask".to_owned(), ResourceKind::Mask)]
}

/// The empty `extent_from` region: a generator reads no input *samples*, only the
/// extent, so it demands no input pixels.
fn no_input_regions(inputs: &Descriptors) -> InputRegions {
    let mut regions = InputRegions::new();
    if inputs.contains_key("extent_from") {
        regions.insert("extent_from".to_owned(), Rect::new(0, 0, 0, 0));
    }
    regions
}

/// The coverage postcondition shared by the mask ops: the output is a `[0, 1]`
/// coverage mask.
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
// mask.empty@1 / mask.full@1
// ---------------------------------------------------------------------------

/// The `mask.empty@1` operation: an `extent_from` resource → an all-zero coverage
/// `Mask`.
#[derive(Debug, Clone, Copy, Default)]
pub struct EmptyMask;

/// The `mask.full@1` operation: an `extent_from` resource → an all-one coverage
/// `Mask`.
#[derive(Debug, Clone, Copy, Default)]
pub struct FullMask;

impl EmptyMask {
    /// Construct the empty-mask operation.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    /// The declared manifest for `mask.empty@1`.
    ///
    /// # Errors
    /// Propagates the [`schema`](ErrorClass::Schema) error if the hard-coded op /
    /// impl ids are somehow invalid (they are not).
    pub fn manifest() -> Result<OperationManifest> {
        constant_mask_manifest(
            EMPTY_OP_ID,
            "Produce an all-zero (no coverage) Mask sized from the `extent_from` input.",
            "The all-zero coverage mask.",
        )
    }
}

impl FullMask {
    /// Construct the full-mask operation.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    /// The declared manifest for `mask.full@1`.
    ///
    /// # Errors
    /// Propagates the [`schema`](ErrorClass::Schema) error if the hard-coded op /
    /// impl ids are somehow invalid (they are not).
    pub fn manifest() -> Result<OperationManifest> {
        constant_mask_manifest(
            FULL_OP_ID,
            "Produce an all-one (full coverage) Mask sized from the `extent_from` input.",
            "The all-one coverage mask.",
        )
    }
}

/// Build the manifest for a constant (`empty`/`full`) mask op.
fn constant_mask_manifest(id: &str, summary: &str, out_doc: &str) -> Result<OperationManifest> {
    Ok(OperationManifest {
        id: id.parse()?,
        impl_version: 1,
        summary: summary.to_owned(),
        determinism: DeterminismTier::Exact,
        roi: RoiPolicy {
            category: RoiCategory::Pointwise,
            halo_px: None,
        },
        inputs: vec![extent_from_port()],
        outputs: vec![mask_output_port(out_doc)],
        params: vec![],
        implementations: vec![reference_impl()?],
        test: mask_test_metadata(
            "produces a constant coverage mask whose every sample is exactly 0 or 1; correctness \
             is an exact all-zero/all-one fixture, not a perceptual metric",
        ),
    })
}

impl OpContract for EmptyMask {
    fn declared_inputs(&self) -> Vec<(String, ResourceKind)> {
        declared_extent_inputs()
    }
    fn declared_outputs(&self) -> Vec<(String, ResourceKind)> {
        declared_mask_outputs()
    }
    fn infer_outputs(
        &self,
        inputs: &Descriptors,
        _params: &serde_json::Value,
    ) -> Result<OutputDescriptors> {
        constant_infer(inputs, EMPTY_OP_ID)
    }
    fn required_inputs(
        &self,
        _requested_outputs: &OutputRegions,
        inputs: &Descriptors,
        _params: &serde_json::Value,
    ) -> Result<InputRegions> {
        Ok(no_input_regions(inputs))
    }
    fn validate_postconditions(
        &self,
        outputs: &OutputDescriptors,
        _params: &serde_json::Value,
    ) -> Result<Vec<AssertionResult>> {
        Ok(coverage_postcondition(outputs))
    }
}

impl OpContract for FullMask {
    fn declared_inputs(&self) -> Vec<(String, ResourceKind)> {
        declared_extent_inputs()
    }
    fn declared_outputs(&self) -> Vec<(String, ResourceKind)> {
        declared_mask_outputs()
    }
    fn infer_outputs(
        &self,
        inputs: &Descriptors,
        _params: &serde_json::Value,
    ) -> Result<OutputDescriptors> {
        constant_infer(inputs, FULL_OP_ID)
    }
    fn required_inputs(
        &self,
        _requested_outputs: &OutputRegions,
        inputs: &Descriptors,
        _params: &serde_json::Value,
    ) -> Result<InputRegions> {
        Ok(no_input_regions(inputs))
    }
    fn validate_postconditions(
        &self,
        outputs: &OutputDescriptors,
        _params: &serde_json::Value,
    ) -> Result<Vec<AssertionResult>> {
        Ok(coverage_postcondition(outputs))
    }
}

/// Infer the coverage-mask descriptor for a constant mask op.
fn constant_infer(inputs: &Descriptors, op: &str) -> Result<OutputDescriptors> {
    let extent = extent_of(inputs, op)?;
    let mut out = OutputDescriptors::new();
    out.insert(
        "mask".to_owned(),
        ResourceDescriptor::Mask(mask_descriptor(extent)),
    );
    Ok(out)
}

impl OpImplementation for EmptyMask {
    fn compute(
        &self,
        inputs: &InputValues,
        _params: &serde_json::Value,
    ) -> std::result::Result<OutputValues, Error> {
        let extent = extent_of_value(inputs, EMPTY_OP_ID)?;
        let mut out = OutputValues::new();
        out.insert("mask".to_owned(), constant_mask(extent, 0.0, EMPTY_OP_ID)?);
        Ok(out)
    }
}

impl OpImplementation for FullMask {
    fn compute(
        &self,
        inputs: &InputValues,
        _params: &serde_json::Value,
    ) -> std::result::Result<OutputValues, Error> {
        let extent = extent_of_value(inputs, FULL_OP_ID)?;
        let mut out = OutputValues::new();
        out.insert("mask".to_owned(), constant_mask(extent, 1.0, FULL_OP_ID)?);
        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// mask.rect@1
// ---------------------------------------------------------------------------

/// The resolved half-open rectangle of a `mask.rect` request, in continuous pixel
/// coordinates.
#[derive(Debug, Clone, Copy, PartialEq)]
struct RectGeometry {
    x0: f64,
    y0: f64,
    x1: f64,
    y1: f64,
}

impl RectGeometry {
    /// Parse and validate the rect geometry from the resolved params.
    ///
    /// # Errors
    /// [`schema`](ErrorClass::Schema) / [`E_RECT_PARAM`] if `rect` is missing, the
    /// wrong shape, or holds a non-finite bound. An ill-formed rect (`x1 < x0`) is
    /// accepted and yields an empty (zero-coverage) mask, matching [`Rect`]'s
    /// saturating semantics, rather than being rejected.
    fn resolve(params: &serde_json::Value) -> Result<Self> {
        let value = params.get("rect").ok_or_else(|| {
            Error::new(
                ErrorClass::Schema,
                E_RECT_PARAM,
                "mask.rect requires a `rect` [x0, y0, x1, y1] parameter".to_owned(),
            )
        })?;
        let array = value
            .as_array()
            .ok_or_else(|| param_error("`rect` must be a [x0, y0, x1, y1] array", value))?;
        if array.len() != 4 {
            return Err(param_error("`rect` must have exactly four elements", value));
        }
        let x0 = finite(&array[0])?;
        let y0 = finite(&array[1])?;
        let x1 = finite(&array[2])?;
        let y1 = finite(&array[3])?;
        Ok(Self { x0, y0, x1, y1 })
    }

    /// The analytic coverage in `[0, 1]` of pixel cell `(i, j)`: the area of the
    /// intersection of the unit cell `[i, i+1) × [j, j+1)` with the rect.
    #[must_use]
    fn coverage(&self, i: u32, j: u32) -> f64 {
        let cell_left = f64::from(i);
        let cell_top = f64::from(j);
        let overlap_x = (self.x1.min(cell_left + 1.0) - self.x0.max(cell_left)).clamp(0.0, 1.0);
        let overlap_y = (self.y1.min(cell_top + 1.0) - self.y0.max(cell_top)).clamp(0.0, 1.0);
        overlap_x * overlap_y
    }
}

/// Coerce a JSON value to a finite `f64`.
fn finite(value: &serde_json::Value) -> Result<f64> {
    let n = value
        .as_f64()
        .ok_or_else(|| param_error("`rect` bounds must be numbers", value))?;
    if n.is_finite() {
        Ok(n)
    } else {
        Err(param_error("`rect` bounds must be finite", value))
    }
}

/// Build a [`schema`](ErrorClass::Schema) rect-param error carrying the offending
/// value.
fn param_error(detail: &str, value: &serde_json::Value) -> Error {
    Error::new(
        ErrorClass::Schema,
        E_RECT_PARAM,
        format!("mask.rect parameter: {detail}"),
    )
    .with_context(ErrorContext::default().with_actual(value.to_string()))
}

/// The `mask.rect@1` operation: an `extent_from` resource → an analytically
/// antialiased rectangle coverage `Mask`.
#[derive(Debug, Clone, Copy, Default)]
pub struct RectMask;

impl RectMask {
    /// Construct the rectangle-mask operation.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    /// The declared manifest for `mask.rect@1`.
    ///
    /// # Errors
    /// Propagates the [`schema`](ErrorClass::Schema) error if the hard-coded op /
    /// impl ids are somehow invalid (they are not).
    pub fn manifest() -> Result<OperationManifest> {
        Ok(OperationManifest {
            id: RECT_OP_ID.parse()?,
            impl_version: 1,
            summary: "Rasterize an axis-aligned half-open rectangle into a coverage Mask with \
                      analytic antialiasing (each pixel's coverage is its exact fractional area \
                      overlap with the rect)."
                .to_owned(),
            determinism: DeterminismTier::Exact,
            roi: RoiPolicy {
                category: RoiCategory::Pointwise,
                halo_px: None,
            },
            inputs: vec![extent_from_port()],
            outputs: vec![mask_output_port("The rectangle coverage mask in [0, 1].")],
            params: vec![ParamSpec {
                name: "rect".to_owned(),
                ty: ParamType::Json,
                unit: Some(ParamUnit::Pixels),
                required: true,
                default: None,
                choices: vec![],
                doc: "The half-open rectangle [x0, y0, x1, y1] in pixel coordinates; covers \
                      [x0, x1) x [y0, y1)."
                    .to_owned(),
            }],
            implementations: vec![reference_impl()?],
            test: mask_test_metadata(
                "produces analytic rectangle coverage verified by exact half-open fixtures and \
                 analytic area equality; there is no perceptual-quality metric to apply",
            ),
        })
    }
}

impl OpContract for RectMask {
    fn declared_inputs(&self) -> Vec<(String, ResourceKind)> {
        declared_extent_inputs()
    }

    fn declared_outputs(&self) -> Vec<(String, ResourceKind)> {
        declared_mask_outputs()
    }

    fn infer_outputs(
        &self,
        inputs: &Descriptors,
        params: &serde_json::Value,
    ) -> Result<OutputDescriptors> {
        let extent = extent_of(inputs, RECT_OP_ID)?;
        // Validate the geometry at infer time so a malformed request fails on the
        // type-checking pass.
        RectGeometry::resolve(params)?;
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
        Ok(no_input_regions(inputs))
    }

    fn validate_postconditions(
        &self,
        outputs: &OutputDescriptors,
        _params: &serde_json::Value,
    ) -> Result<Vec<AssertionResult>> {
        Ok(coverage_postcondition(outputs))
    }
}

impl OpImplementation for RectMask {
    fn compute(
        &self,
        inputs: &InputValues,
        params: &serde_json::Value,
    ) -> std::result::Result<OutputValues, Error> {
        let extent = extent_of_value(inputs, RECT_OP_ID)?;
        let rect = RectGeometry::resolve(params)?;
        let samples = rasterize_rect(&rect, extent);
        let value = ResourceValue::new(
            ResourceDescriptor::Mask(mask_descriptor(extent)),
            1,
            samples,
        )
        .map_err(|actual| {
            Error::new(
                ErrorClass::Execution,
                E_MASK_INPUT,
                format!("mask.rect produced a mask buffer of unexpected length {actual}"),
            )
        })?;
        let mut out = OutputValues::new();
        out.insert("mask".to_owned(), value);
        Ok(out)
    }
}

/// Rasterize the rect's analytic coverage into a row-major, single-channel `f32`
/// buffer of `extent.width * extent.height` samples.
#[must_use]
fn rasterize_rect(rect: &RectGeometry, extent: Extent) -> Vec<f32> {
    let width = extent.width as usize;
    let height = extent.height as usize;
    let mut samples = Vec::with_capacity(width.saturating_mul(height));
    for j in 0..extent.height {
        for i in 0..extent.width {
            #[allow(
                clippy::cast_possible_truncation,
                reason = "coverage is a bounded [0, 1] f64 stored as the mask's f32"
            )]
            samples.push(rect.coverage(i, j) as f32);
        }
    }
    samples
}

// ---------------------------------------------------------------------------
// shared
// ---------------------------------------------------------------------------

/// The mandatory `cpu.reference@1` oracle implementation id.
fn reference_impl() -> Result<ImplId> {
    ImplId::new("cpu", "reference", 1)
}

/// Verification declarations for an exact, single-reference mask generator:
/// differential and perceptual do not apply; every other applicable category is
/// covered by this module's exact fixtures, area, and property tests.
fn mask_test_metadata(perceptual_reason: &str) -> TestMetadata {
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
