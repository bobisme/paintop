//! The `mask.polygon@1` operation.
//!
//! It rasterizes a polygon into a coverage [`Mask`](ResourceKind::Mask) with an
//! explicit fill rule and supersampled antialiasing (`OP_CATALOG` §3,
//! `AGENT_VERIFICATION` §3.6, §2.9).
//!
//! `mask.polygon` takes the mask's pixel extent from an `extent_from` input
//! (D3: the resource a mask is sized to is an input port, not a param), a list of
//! polygon vertices `points` in pixel coordinates, and an explicit `fill_rule`:
//!
//! - **`nonzero`** — a point is inside when the signed winding number of the
//!   polygon around it is nonzero (the default, robust to self-intersection in
//!   the "wrap" sense);
//! - **`even-odd`** — a point is inside when a ray from it crosses the polygon an
//!   odd number of times (the classic parity rule, which carves out
//!   self-overlapping regions).
//!
//! # Coverage model (supersampled, half-open)
//!
//! Each pixel cell `(i, j)` occupies the continuous square `[i, i+1) × [j, j+1)`.
//! Its coverage is the fraction of a fixed `S × S` grid of subsample points
//! inside the polygon under the chosen fill rule. The subsample for `(s, t)` is
//! at the cell-relative offset `((s + 0.5) / S, (t + 0.5) / S)`, a grid that is
//! **symmetric about the pixel centre**, so a 90° rotation or an axis reflection
//! of the polygon maps subsamples to subsamples and the coverage is covariant.
//!
//! The inside test is a half-open ray cast to `+x`: an edge `(p0, p1)` is counted
//! only when `(p0.y <= y) != (p1.y <= y)`. This *half-open in y* convention makes
//! a shared vertex between two edges count exactly once (no double count, no
//! gap), and automatically ignores horizontal edges (whose endpoints compare
//! equal), so a **degenerate or self-touching edge never divides by zero and the
//! coverage is always finite** (`AGENT_VERIFICATION` §2.9): the crossing slope is
//! computed only when the two endpoints straddle the scanline, which guarantees
//! `p1.y − p0.y ≠ 0`.
//!
//! # Area convergence
//!
//! Summed over the raster, the coverage approaches the polygon's analytic area as
//! the raster resolution increases (each pixel's `S × S` estimate refines toward
//! its true fractional overlap, and the per-pixel quantisation error shrinks with
//! the cell size), so a polygon rendered at successively higher resolutions
//! converges to its analytic area (`AGENT_VERIFICATION` §3.6).
//!
//! # Determinism
//!
//! The op is [`Exact`](DeterminismTier::Exact): coverage is a fixed-order count
//! over a fixed subsample grid using only comparisons, subtraction, multiply, and
//! divide on finite inputs, bit-identical across platforms, and depends only on
//! the pixel's own cell, so a tile boundary never changes a sample.

use paintop_core::executor::{InputValues, OpImplementation, OutputValues, ResourceValue};
use paintop_ir::{
    AssertionResult, CoordinateConvention, Descriptors, DeterminismTier, Error, ErrorClass,
    ErrorContext, Extent, ImplId, InputRegions, InputSpec, MaskDescriptor, MaskMeaning, OpContract,
    OperationManifest, OutputDescriptors, OutputRegions, OutputSpec, ParamSpec, ParamType,
    ParamUnit, Rect, ResourceDescriptor, ResourceKind, Result, RoiCategory, RoiPolicy, ScalarType,
    TestMetadata, ValidRange,
};

/// The canonical id of the polygon-mask operation.
pub const POLYGON_OP_ID: &str = "mask.polygon@1";

/// The `extent_from` input was absent or carried no descriptor to size the mask.
pub const E_POLYGON_INPUT: &str = "E_POLYGON_INPUT";
/// A `mask.polygon` geometry parameter was missing, the wrong shape, or
/// non-finite, or the fill rule was unknown.
pub const E_POLYGON_PARAM: &str = "E_POLYGON_PARAM";

/// The `nonzero` fill-rule token.
const FILL_NONZERO: &str = "nonzero";
/// The `even-odd` fill-rule token.
const FILL_EVEN_ODD: &str = "even-odd";

/// The per-axis supersampling factor: each pixel is estimated from `S × S`
/// subsamples. Four is a fixed, symmetric grid giving 16 coverage levels — enough
/// for the area-convergence and rotation-covariance gates while staying exact.
const SUPERSAMPLE: u32 = 4;

/// The polygon fill rule.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FillRule {
    /// Inside when the winding number is nonzero.
    NonZero,
    /// Inside when the crossing parity is odd.
    EvenOdd,
}

impl FillRule {
    /// Resolve the `fill_rule` parameter, defaulting to `nonzero`.
    fn resolve(params: &serde_json::Value) -> Result<Self> {
        let Some(value) = params.get("fill_rule") else {
            return Ok(Self::NonZero);
        };
        match value.as_str() {
            Some(FILL_NONZERO) => Ok(Self::NonZero),
            Some(FILL_EVEN_ODD) => Ok(Self::EvenOdd),
            _ => Err(Error::new(
                ErrorClass::Schema,
                E_POLYGON_PARAM,
                format!("mask.polygon `fill_rule` must be `{FILL_NONZERO}` or `{FILL_EVEN_ODD}`"),
            )
            .with_context(ErrorContext::default().with_actual(value.to_string()))),
        }
    }
}

/// A resolved polygon: its vertices and the chosen fill rule.
#[derive(Debug, Clone, PartialEq)]
struct Polygon {
    points: Vec<(f64, f64)>,
    fill_rule: FillRule,
}

impl Polygon {
    /// Parse and validate the polygon geometry and fill rule from the resolved
    /// params.
    ///
    /// # Errors
    /// [`schema`](ErrorClass::Schema) / [`E_POLYGON_PARAM`] if `points` is
    /// missing, not an array of `[x, y]` pairs, or holds a non-finite coordinate,
    /// or if `fill_rule` is an unknown token. A polygon with fewer than three
    /// vertices is accepted and rasterizes to an empty (zero-coverage) mask rather
    /// than being rejected (a degenerate but well-defined request).
    fn resolve(params: &serde_json::Value) -> Result<Self> {
        let value = params.get("points").ok_or_else(|| {
            Error::new(
                ErrorClass::Schema,
                E_POLYGON_PARAM,
                "mask.polygon requires a `points` array of [x, y] vertices".to_owned(),
            )
        })?;
        let array = value
            .as_array()
            .ok_or_else(|| param_error("`points` must be an array of [x, y] pairs", value))?;
        let mut points = Vec::with_capacity(array.len());
        for vertex in array {
            let pair = vertex
                .as_array()
                .ok_or_else(|| param_error("each vertex must be a [x, y] array", vertex))?;
            if pair.len() != 2 {
                return Err(param_error(
                    "each vertex must have exactly two elements",
                    vertex,
                ));
            }
            points.push((finite(&pair[0])?, finite(&pair[1])?));
        }
        Ok(Self {
            points,
            fill_rule: FillRule::resolve(params)?,
        })
    }

    /// Whether the point `(x, y)` is inside the polygon under the fill rule.
    ///
    /// A half-open ray cast to `+x`: an edge is counted only when its endpoints
    /// straddle the scanline `y` in the half-open sense `(p0.y <= y) != (p1.y <=
    /// y)`, which counts a shared vertex once, ignores horizontal edges, and never
    /// divides by zero.
    #[must_use]
    fn contains(&self, x: f64, y: f64) -> bool {
        let n = self.points.len();
        if n < 3 {
            return false;
        }
        let mut winding: i32 = 0;
        let mut parity = false;
        for i in 0..n {
            let (x0, y0) = self.points[i];
            let (x1, y1) = self.points[(i + 1) % n];
            let below0 = y0 <= y;
            let below1 = y1 <= y;
            if below0 == below1 {
                // Endpoints on the same side of the scanline (incl. horizontal
                // edges): no crossing, and the divisor below is never formed.
                continue;
            }
            // The edge straddles y; y1 - y0 != 0 here, so this is finite.
            let t = (y - y0) / (y1 - y0);
            let cross_x = t.mul_add(x1 - x0, x0);
            if cross_x > x {
                // The crossing is to the right of the sample point.
                parity = !parity;
                if below0 {
                    // Edge going upward (y increasing through the scanline).
                    winding += 1;
                } else {
                    winding -= 1;
                }
            }
        }
        match self.fill_rule {
            FillRule::NonZero => winding != 0,
            FillRule::EvenOdd => parity,
        }
    }
}

/// Coerce a JSON value to a finite `f64`.
fn finite(value: &serde_json::Value) -> Result<f64> {
    let n = value
        .as_f64()
        .ok_or_else(|| param_error("coordinates must be numbers", value))?;
    if n.is_finite() {
        Ok(n)
    } else {
        Err(param_error("coordinates must be finite", value))
    }
}

/// Build a [`schema`](ErrorClass::Schema) polygon-param error carrying the
/// offending value.
fn param_error(detail: &str, value: &serde_json::Value) -> Error {
    Error::new(
        ErrorClass::Schema,
        E_POLYGON_PARAM,
        format!("mask.polygon parameter: {detail}"),
    )
    .with_context(ErrorContext::default().with_actual(value.to_string()))
}

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

/// The pixel extent of the `extent_from` input descriptor.
fn extent_of(inputs: &Descriptors) -> Result<Extent> {
    let source = inputs.get("extent_from").ok_or_else(|| {
        Error::new(
            ErrorClass::Reference,
            E_POLYGON_INPUT,
            "mask.polygon requires an `extent_from` input".to_owned(),
        )
    })?;
    Ok(source.extent())
}

/// The pixel extent of the `extent_from` input value.
fn extent_of_value(inputs: &InputValues) -> std::result::Result<Extent, Error> {
    let source = inputs.get("extent_from").ok_or_else(|| {
        Error::new(
            ErrorClass::Reference,
            E_POLYGON_INPUT,
            "mask.polygon requires an `extent_from` input value".to_owned(),
        )
    })?;
    Ok(source.extent())
}

/// Rasterize the polygon into a row-major, single-channel `f32` coverage buffer.
#[must_use]
fn rasterize(polygon: &Polygon, extent: Extent) -> Vec<f32> {
    let width = extent.width as usize;
    let height = extent.height as usize;
    let mut samples = Vec::with_capacity(width.saturating_mul(height));
    let step = 1.0 / f64::from(SUPERSAMPLE);
    let total = f64::from(SUPERSAMPLE * SUPERSAMPLE);
    for j in 0..extent.height {
        for i in 0..extent.width {
            let mut inside = 0u32;
            for t in 0..SUPERSAMPLE {
                let sy = (f64::from(t) + 0.5).mul_add(step, f64::from(j));
                for s in 0..SUPERSAMPLE {
                    let sx = (f64::from(s) + 0.5).mul_add(step, f64::from(i));
                    if polygon.contains(sx, sy) {
                        inside += 1;
                    }
                }
            }
            let coverage = f64::from(inside) / total;
            #[allow(
                clippy::cast_possible_truncation,
                reason = "coverage is a bounded [0, 1] f64 stored as the mask's f32"
            )]
            samples.push(coverage as f32);
        }
    }
    samples
}

/// The `mask.polygon@1` operation.
#[derive(Debug, Clone, Copy, Default)]
pub struct PolygonMask;

impl PolygonMask {
    /// Construct the polygon-mask operation.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    /// The declared manifest for `mask.polygon@1`.
    ///
    /// # Errors
    /// Propagates the [`schema`](ErrorClass::Schema) error if the hard-coded op /
    /// impl ids are somehow invalid (they are not).
    pub fn manifest() -> Result<OperationManifest> {
        Ok(OperationManifest {
            id: POLYGON_OP_ID.parse()?,
            impl_version: 1,
            summary: "Rasterize a polygon into a coverage Mask under an explicit fill rule \
                      (nonzero winding / even-odd) with supersampled antialiasing; degenerate \
                      and self-touching edges produce finite coverage."
                .to_owned(),
            determinism: DeterminismTier::Exact,
            roi: RoiPolicy {
                category: RoiCategory::Pointwise,
                halo_px: None,
            },
            inputs: vec![InputSpec {
                name: "extent_from".to_owned(),
                kind: ResourceKind::Image,
                required: true,
                doc: "The resource whose pixel extent the produced mask matches.".to_owned(),
            }],
            outputs: vec![OutputSpec {
                name: "mask".to_owned(),
                kind: ResourceKind::Mask,
                doc: "The polygon coverage mask in [0, 1].".to_owned(),
            }],
            params: vec![
                ParamSpec {
                    name: "points".to_owned(),
                    ty: ParamType::Json,
                    unit: Some(ParamUnit::Pixels),
                    required: true,
                    default: None,
                    choices: vec![],
                    doc: "The polygon vertices as an array of [x, y] pixel coordinates, in \
                          path order (the closing edge from the last to the first vertex is \
                          implicit)."
                        .to_owned(),
                },
                ParamSpec {
                    name: "fill_rule".to_owned(),
                    ty: ParamType::String,
                    unit: None,
                    required: false,
                    default: Some(serde_json::Value::String(FILL_NONZERO.to_owned())),
                    choices: vec![FILL_NONZERO.to_owned(), FILL_EVEN_ODD.to_owned()],
                    doc: "The fill rule: `nonzero` winding (default) or `even-odd` parity."
                        .to_owned(),
                },
            ],
            implementations: vec![reference_impl()?],
            test: polygon_test_metadata(),
        })
    }
}

impl OpContract for PolygonMask {
    fn declared_inputs(&self) -> Vec<(String, ResourceKind)> {
        vec![("extent_from".to_owned(), ResourceKind::Image)]
    }

    fn declared_outputs(&self) -> Vec<(String, ResourceKind)> {
        vec![("mask".to_owned(), ResourceKind::Mask)]
    }

    fn infer_outputs(
        &self,
        inputs: &Descriptors,
        params: &serde_json::Value,
    ) -> Result<OutputDescriptors> {
        let extent = extent_of(inputs)?;
        // Validate geometry at infer time so a malformed request fails on the
        // type-checking pass.
        Polygon::resolve(params)?;
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
        // A generator reads no input samples, only the extent.
        let mut regions = InputRegions::new();
        if inputs.contains_key("extent_from") {
            regions.insert("extent_from".to_owned(), Rect::new(0, 0, 0, 0));
        }
        Ok(regions)
    }

    fn validate_postconditions(
        &self,
        outputs: &OutputDescriptors,
        _params: &serde_json::Value,
    ) -> Result<Vec<AssertionResult>> {
        let Some(ResourceDescriptor::Mask(mask)) = outputs.get("mask") else {
            return Ok(vec![AssertionResult::fail(
                "produces_mask",
                "no `mask` output produced",
            )]);
        };
        let unit_range = ValidRange::Bounded { min: 0.0, max: 1.0 };
        Ok(vec![
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
        ])
    }
}

impl OpImplementation for PolygonMask {
    fn compute(
        &self,
        inputs: &InputValues,
        params: &serde_json::Value,
    ) -> std::result::Result<OutputValues, Error> {
        let extent = extent_of_value(inputs)?;
        let polygon = Polygon::resolve(params)?;
        let samples = rasterize(&polygon, extent);
        let value = ResourceValue::new(
            ResourceDescriptor::Mask(mask_descriptor(extent)),
            1,
            samples,
        )
        .map_err(|actual| {
            Error::new(
                ErrorClass::Execution,
                E_POLYGON_INPUT,
                format!("mask.polygon produced a mask buffer of unexpected length {actual}"),
            )
        })?;
        let mut out = OutputValues::new();
        out.insert("mask".to_owned(), value);
        Ok(out)
    }
}

/// The mandatory `cpu.reference@1` oracle implementation id.
fn reference_impl() -> Result<ImplId> {
    ImplId::new("cpu", "reference", 1)
}

/// Verification declarations for `mask.polygon@1`: an exact, single-reference
/// generator. Differential and perceptual do not apply; every other applicable
/// category is covered by this module's fixtures, area-convergence, fill-rule,
/// degenerate-edge, and rotation-metamorphic tests.
fn polygon_test_metadata() -> TestMetadata {
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
            "polygon coverage is an exact supersampled count verified by analytic-area \
             convergence, fill-rule, degenerate-edge, and rotation-covariance tests; there is \
             no perceptual-quality metric to apply"
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
