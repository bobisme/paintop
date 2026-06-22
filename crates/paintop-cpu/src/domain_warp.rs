//! The `field.domain_warp@1` operation: **displacement-field resampling**
//! (`OP_CATALOG` §11; `plan.md` §1428).
//!
//! `field.domain_warp@1` warps a `source` raster (an `Image` or a scalar/vector
//! `Field`) by a per-pixel **displacement** [`Field2`](paintop_ir::ResourceKind::Field2):
//! the output pixel `(x, y)` is the source resampled at the displaced position
//!
//! ```text
//! sx = (x + 0.5) + dx(x, y)
//! sy = (y + 0.5) + dy(x, y)
//! ```
//!
//! under the fixed [`PixelCenterUpperLeft`](paintop_ir::CoordinateConvention)
//! convention (pixel `(x, y)` has center `(x + 0.5, y + 0.5)`). The displacement
//! is in **physical pixels**, a *forward* lookup offset (where to read *from*).
//!
//! # Resampling contract (reused from M1 `image.resize`)
//!
//! The lookup reuses the same two ingredients as the M1 resize sampler:
//!
//! - the **half-pixel pixel-center mapping** — a continuous coordinate `c` reads
//!   between integer pixel centers, with pixel center `i` at `i + 0.5`;
//! - the **`bilinear` / `nearest` reconstruction kernels** — `nearest` rounds to
//!   the closest source center; `bilinear` blends the four surrounding centers
//!   with triangle weights.
//!
//! Out-of-bounds taps are resolved by an explicit **boundary mode** drawn from the
//! same `constant | transparent | clamp | mirror | wrap` vocabulary the
//! neighbourhood filters use (`filter.convolve`), so the warp's edge behaviour is
//! the project's one boundary contract, not a private convention.
//!
//! # Properties (the verification anchors)
//!
//! - **Zero displacement is the identity.** With `dx = dy = 0` every output pixel
//!   reads its own center `(x + 0.5, y + 0.5)`, which under bilinear/nearest is the
//!   source pixel verbatim — `warp(src, 0) == src` bit-for-bit.
//! - **A constant displacement is a translation.** With `dx = a`, `dy = b`
//!   constant, the warp is a uniform shift by `(a, b)`; an *integer* shift equals
//!   `image.pad`/`crop` translation under the matching boundary, and a fractional
//!   shift is the bilinear-interpolated translate.
//! - **Round-trip.** Warping by `d` then by the negated, resampled `-d` returns
//!   the source within the bilinear reconstruction tolerance away from edges.
//!
//! # Determinism
//!
//! The op is [`Bounded`](DeterminismTier::Bounded): the displaced coordinate and
//! the bilinear blend are a fixed-order `f64` evaluation rounded once to `f32`, so
//! it is bit-identical on reruns of a fixed backend, but the float blend is
//! asserted within tolerance against an independent reference rather than
//! bit-exactly across platforms. With `nearest` and an integer displacement the
//! result is exact.

use paintop_core::executor::{InputValues, OpImplementation, OutputValues, ResourceValue};
use paintop_ir::{
    AssertionResult, CoordinateConvention, Descriptors, DeterminismTier, Error, ErrorClass,
    ErrorContext, Extent, FieldArity, ImplId, InputRegions, InputSpec, OpContract,
    OperationManifest, OutputDescriptors, OutputRegions, OutputSpec, ParamSpec, ParamType, Rect,
    ResourceDescriptor, ResourceKind, Result, RoiCategory, RoiPolicy,
};

use crate::convolve::{self, BoundaryMode};

/// The canonical id of the domain-warp operation.
pub const DOMAIN_WARP_OP_ID: &str = "field.domain_warp@1";

/// The `source` or `displacement` input was absent or carried an unsupported
/// descriptor.
pub const E_DOMAIN_WARP_INPUT: &str = "E_DOMAIN_WARP_INPUT";

/// A `filter` / `boundary` parameter was missing or malformed.
pub const E_DOMAIN_WARP_PARAM: &str = "E_DOMAIN_WARP_PARAM";

/// The execution buffer length disagreed with the declared extent.
pub const E_DOMAIN_WARP_BUFFER: &str = "E_DOMAIN_WARP_BUFFER";

/// The reconstruction filter the warp samples the source with.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum WarpFilter {
    /// Round-to-nearest source center (exact for integer displacements).
    Nearest,
    /// Bilinear blend of the four surrounding source centers.
    Bilinear,
}

impl WarpFilter {
    /// Parse the filter from its wire token (the M1 resize vocabulary subset the
    /// warp supports).
    fn from_token(token: &str) -> Option<Self> {
        match token {
            "nearest" => Some(Self::Nearest),
            "bilinear" => Some(Self::Bilinear),
            _ => None,
        }
    }
}

/// The resolved warp request: the reconstruction filter, the boundary mode, and
/// the out-of-bounds constant value used by the `constant` mode.
#[derive(Debug, Clone, Copy)]
struct WarpRequest {
    filter: WarpFilter,
    boundary: BoundaryMode,
    constant: f32,
}

impl WarpRequest {
    /// Resolve the `filter` and `boundary` params (both optional, defaulting to
    /// `bilinear` and `clamp`).
    fn resolve(params: &serde_json::Value) -> Result<Self> {
        let filter = match params.get("filter") {
            None => WarpFilter::Bilinear,
            Some(value) => {
                let token = value
                    .as_str()
                    .ok_or_else(|| param_error("`filter` must be a string", "filter", value))?;
                WarpFilter::from_token(token).ok_or_else(|| {
                    param_error("must be `nearest` or `bilinear`", "filter", value)
                })?
            }
        };
        let (boundary, constant) = boundary_param(params)?;
        Ok(Self {
            filter,
            boundary,
            constant,
        })
    }
}

/// Parse the optional `boundary` object `{ mode, value? }`, defaulting to
/// `clamp` with a `0.0` constant. The `value` is only meaningful for the
/// `constant` mode.
fn boundary_param(params: &serde_json::Value) -> Result<(BoundaryMode, f32)> {
    let Some(boundary) = params.get("boundary") else {
        return Ok((BoundaryMode::Clamp, 0.0));
    };
    let object = boundary
        .as_object()
        .ok_or_else(|| param_error("`boundary` must be an object", "boundary", boundary))?;
    let mode = object
        .get("mode")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| {
            param_error(
                "`boundary.mode` must be a string",
                "boundary.mode",
                boundary,
            )
        })?;
    if mode == "valid" {
        return Err(param_error(
            "`valid` is not a warp boundary mode (a warp reads arbitrary positions)",
            "boundary.mode",
            boundary,
        ));
    }
    let parsed = BoundaryMode::from_token(mode).ok_or_else(|| {
        param_error(
            "is not a known boundary mode (constant | transparent | clamp | mirror | wrap)",
            "boundary.mode",
            boundary,
        )
    })?;
    let constant = match object.get("value") {
        None => 0.0_f32,
        Some(v) => {
            let n = v.as_f64().ok_or_else(|| {
                param_error("`boundary.value` must be a number", "boundary.value", v)
            })?;
            if !n.is_finite() {
                return Err(param_error(
                    "`boundary.value` must be finite",
                    "boundary.value",
                    v,
                ));
            }
            #[allow(
                clippy::cast_possible_truncation,
                reason = "the out-of-bounds constant is stored as the f32 sample type"
            )]
            {
                n as f32
            }
        }
    };
    Ok((parsed, constant))
}

/// Build a [`schema`](ErrorClass::Schema) param error carrying the offending
/// value.
fn param_error(detail: &str, name: &str, value: &serde_json::Value) -> Error {
    Error::new(
        ErrorClass::Schema,
        E_DOMAIN_WARP_PARAM,
        format!("field.domain_warp parameter `{name}`: {detail}"),
    )
    .with_context(ErrorContext::default().with_actual(value.to_string()))
}

/// The component count of a `source` descriptor the warp can carry, or an error
/// if the source is an unsupported resource kind.
fn source_channels(descriptor: &ResourceDescriptor) -> Result<u32> {
    match descriptor {
        ResourceDescriptor::Image(d) => Ok(d.layout.channel_count()),
        ResourceDescriptor::Mask(_) | ResourceDescriptor::SdfMask(_) => Ok(1),
        ResourceDescriptor::Field1(_) => Ok(FieldArity::Field1.component_count()),
        ResourceDescriptor::Field2(_) => Ok(FieldArity::Field2.component_count()),
        ResourceDescriptor::Field3(_) => Ok(FieldArity::Field3.component_count()),
        other => Err(Error::new(
            ErrorClass::Type,
            E_DOMAIN_WARP_INPUT,
            format!(
                "field.domain_warp `source` must be an Image, Mask, Sdf, or Field; got {:?}",
                other.kind()
            ),
        )),
    }
}

/// The displacement input's extent, requiring a [`Field2`].
fn displacement_extent(descriptor: &ResourceDescriptor) -> Result<Extent> {
    match descriptor {
        ResourceDescriptor::Field2(d) => Ok(d.extent),
        other => Err(Error::new(
            ErrorClass::Type,
            E_DOMAIN_WARP_INPUT,
            format!(
                "field.domain_warp `displacement` must be a Field2 (dx, dy); got {:?}",
                other.kind()
            ),
        )),
    }
}

/// The `field.domain_warp@1` operation.
#[derive(Debug, Clone, Copy, Default)]
pub struct DomainWarp;

impl DomainWarp {
    /// Construct the domain-warp operation.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    /// The declared manifest for `field.domain_warp@1`.
    ///
    /// # Errors
    /// Propagates the [`schema`](ErrorClass::Schema) error if the hard-coded ids
    /// are invalid (they are not).
    pub fn manifest() -> Result<OperationManifest> {
        Ok(OperationManifest {
            id: DOMAIN_WARP_OP_ID.parse()?,
            impl_version: 1,
            summary: "Warp a source Image/Field by a per-pixel Field2 displacement: output (x, y) \
                      reads the source at (x + 0.5 + dx, y + 0.5 + dy) with the M1 \
                      bilinear/nearest sampler and an explicit boundary mode. Zero displacement \
                      is identity; a constant displacement is a translation."
                .to_owned(),
            determinism: DeterminismTier::Bounded,
            roi: RoiPolicy {
                // The warp reads an unbounded neighbourhood (any displacement can
                // point anywhere), so the honest footprint is the whole source.
                category: RoiCategory::FullDomain,
                halo_px: None,
            },
            inputs: vec![
                InputSpec {
                    name: "source".to_owned(),
                    kind: ResourceKind::Image,
                    required: true,
                    doc: "The raster to warp (an Image, Mask, Sdf, or Field). Its descriptor and \
                          extent are preserved on the output."
                        .to_owned(),
                },
                InputSpec {
                    name: "displacement".to_owned(),
                    kind: ResourceKind::Field2,
                    required: true,
                    doc: "The per-pixel (dx, dy) lookup offset in physical pixels; must share the \
                          source extent."
                        .to_owned(),
                },
            ],
            outputs: vec![OutputSpec {
                name: "warped".to_owned(),
                kind: ResourceKind::Image,
                doc: "The warped raster, same descriptor/extent as `source`.".to_owned(),
            }],
            params: vec![
                ParamSpec {
                    name: "filter".to_owned(),
                    ty: ParamType::String,
                    unit: None,
                    required: false,
                    default: Some(serde_json::json!("bilinear")),
                    choices: vec!["nearest".to_owned(), "bilinear".to_owned()],
                    doc: "The reconstruction filter (M1 resize vocabulary): nearest or bilinear."
                        .to_owned(),
                },
                ParamSpec {
                    name: "boundary".to_owned(),
                    ty: ParamType::Json,
                    unit: None,
                    required: false,
                    default: None,
                    choices: vec![],
                    doc: "Out-of-bounds policy { mode: constant|transparent|clamp|mirror|wrap, \
                          value?: N }; absent is clamp."
                        .to_owned(),
                },
            ],
            implementations: vec![reference_impl()?],
            test: warp_test_metadata(),
        })
    }
}

impl OpContract for DomainWarp {
    fn declared_inputs(&self) -> Vec<(String, ResourceKind)> {
        vec![
            ("source".to_owned(), ResourceKind::Image),
            ("displacement".to_owned(), ResourceKind::Field2),
        ]
    }

    fn declared_outputs(&self) -> Vec<(String, ResourceKind)> {
        vec![("warped".to_owned(), ResourceKind::Image)]
    }

    fn infer_outputs(
        &self,
        inputs: &Descriptors,
        params: &serde_json::Value,
    ) -> Result<OutputDescriptors> {
        let source = inputs
            .get("source")
            .ok_or_else(|| missing_input("source"))?;
        let displacement = inputs
            .get("displacement")
            .ok_or_else(|| missing_input("displacement"))?;
        // Validate params and shapes at infer time so a malformed request fails on
        // the type-checking pass.
        WarpRequest::resolve(params)?;
        source_channels(source)?;
        let disp_extent = displacement_extent(displacement)?;
        if source.extent() != disp_extent {
            return Err(extent_mismatch(source.extent(), disp_extent));
        }
        // The warp preserves the source's descriptor (and thus its extent and
        // semantics) verbatim.
        let mut out = OutputDescriptors::new();
        out.insert("warped".to_owned(), *source);
        Ok(out)
    }

    fn required_inputs(
        &self,
        requested_outputs: &OutputRegions,
        inputs: &Descriptors,
        _params: &serde_json::Value,
    ) -> Result<InputRegions> {
        // A warp can read any source pixel (the displacement is unbounded), so a
        // requested output region demands the whole source and the matching
        // displacement region. This is the honest dependency for a global op.
        let mut regions = InputRegions::new();
        if let Some(source) = inputs.get("source") {
            let extent = source.extent();
            let full = Rect::new(0, 0, i64::from(extent.width), i64::from(extent.height));
            regions.insert("source".to_owned(), full);
            // The displacement is read pointwise: only the requested output region.
            let needed = requested_outputs
                .get("warped")
                .map_or(full, |r| r.intersect(full));
            regions.insert("displacement".to_owned(), needed);
        }
        Ok(regions)
    }

    fn validate_postconditions(
        &self,
        outputs: &OutputDescriptors,
        _params: &serde_json::Value,
    ) -> Result<Vec<AssertionResult>> {
        Ok(vec![if outputs.contains_key("warped") {
            AssertionResult::pass("produces_warped")
        } else {
            AssertionResult::fail("produces_warped", "no `warped` output produced")
        }])
    }
}

impl OpImplementation for DomainWarp {
    fn compute(
        &self,
        inputs: &InputValues,
        params: &serde_json::Value,
    ) -> std::result::Result<OutputValues, Error> {
        let source = inputs
            .get("source")
            .ok_or_else(|| missing_input("source"))?;
        let displacement = inputs
            .get("displacement")
            .ok_or_else(|| missing_input("displacement"))?;
        let request = WarpRequest::resolve(params)?;

        let extent = source.extent();
        let disp_extent = displacement.extent();
        if extent != disp_extent {
            return Err(extent_mismatch(extent, disp_extent));
        }
        let channels = source_channels(source.descriptor())?;
        let disp = displacement.samples();
        let src = source.samples();

        let warped = warp_samples(src, extent, channels, disp, request);
        let value =
            ResourceValue::new(*source.descriptor(), channels, warped).map_err(buffer_error)?;
        let mut out = OutputValues::new();
        out.insert("warped".to_owned(), value);
        Ok(out)
    }
}

/// Resample `src` (row-major, `channels`-interleaved, `extent`) at every output
/// pixel's displaced position, returning the warped buffer.
fn warp_samples(
    src: &[f32],
    extent: Extent,
    channels: u32,
    disp: &[f32],
    request: WarpRequest,
) -> Vec<f32> {
    let w = extent.width as usize;
    let h = extent.height as usize;
    let stride = channels as usize;
    let mut out = vec![0.0_f32; w * h * stride];

    for y in 0..h {
        for x in 0..w {
            let pixel = y * w + x;
            let dx = f64::from(disp[pixel * 2]);
            let dy = f64::from(disp[pixel * 2 + 1]);
            // The source position this output reads from, in continuous pixel
            // coordinates (pixel center convention).
            let (cx, cy) = CoordinateConvention::PixelCenterUpperLeft
                .pixel_center(u32::try_from(x).unwrap_or(0), u32::try_from(y).unwrap_or(0));
            let sx = cx + dx;
            let sy = cy + dy;
            for c in 0..stride {
                let value = sample(src, extent, channels, c, sx, sy, request);
                #[allow(
                    clippy::cast_possible_truncation,
                    reason = "blend accumulated in f64, stored once as the f32 sample"
                )]
                {
                    out[pixel * stride + c] = value as f32;
                }
            }
        }
    }
    out
}

/// Sample channel `c` of `src` at continuous source coordinate `(sx, sy)` under
/// the request's filter and boundary mode.
fn sample(
    src: &[f32],
    extent: Extent,
    channels: u32,
    c: usize,
    sx: f64,
    sy: f64,
    request: WarpRequest,
) -> f64 {
    let n_x = i64::from(extent.width);
    let n_y = i64::from(extent.height);
    let stride = channels as usize;
    let w = extent.width as usize;

    // Fetch a single source pixel center `(ix, iy)` (integer indices), resolving
    // out-of-bounds reads through the boundary mode.
    let fetch = |ix: i64, iy: i64| -> f64 {
        let mapped_x = convolve::source_index(ix, n_x, request.boundary);
        let mapped_y = convolve::source_index(iy, n_y, request.boundary);
        match (mapped_x, mapped_y) {
            (Some(mx), Some(my)) => {
                // `source_index` returns indices already clamped into `[0, n)`, so
                // the conversion to `usize` cannot lose a value.
                let row = usize::try_from(my).unwrap_or(0);
                let col = usize::try_from(mx).unwrap_or(0);
                let idx = (row * w + col) * stride + c;
                f64::from(src[idx])
            }
            // Out of bounds under constant/transparent: the constant value.
            _ => f64::from(request.constant),
        }
    };

    match request.filter {
        WarpFilter::Nearest => {
            // Round the continuous center to the nearest integer pixel center:
            // center `i` sits at `i + 0.5`, so `floor(coord)` is the nearest cell.
            #[allow(
                clippy::cast_possible_truncation,
                reason = "floored source coordinate to its integer pixel cell; extents are bounded \
                          far below i64 range"
            )]
            let (ix, iy) = (sx.floor() as i64, sy.floor() as i64);
            fetch(ix, iy)
        }
        WarpFilter::Bilinear => {
            // Convert center-coordinate to a lattice of pixel centers: the cell to
            // the lower-left has center index `floor(coord - 0.5)`, fractional
            // weight `coord - 0.5 - that index`.
            let gx = sx - 0.5;
            let gy = sy - 0.5;
            let x0 = gx.floor();
            let y0 = gy.floor();
            let tx = gx - x0;
            let ty = gy - y0;
            #[allow(
                clippy::cast_possible_truncation,
                reason = "floored source coordinate to its integer pixel cell; extents are bounded \
                          far below i64 range"
            )]
            let (ix0, iy0) = (x0 as i64, y0 as i64);

            let v00 = fetch(ix0, iy0);
            let v10 = fetch(ix0 + 1, iy0);
            let v01 = fetch(ix0, iy0 + 1);
            let v11 = fetch(ix0 + 1, iy0 + 1);
            let top = (v10 - v00).mul_add(tx, v00);
            let bottom = (v11 - v01).mul_add(tx, v01);
            (bottom - top).mul_add(ty, top)
        }
    }
}

/// The missing-input reference error for a named port.
fn missing_input(port: &str) -> Error {
    Error::new(
        ErrorClass::Reference,
        E_DOMAIN_WARP_INPUT,
        format!("field.domain_warp requires a `{port}` input"),
    )
}

/// The source/displacement extent-mismatch error.
fn extent_mismatch(source: Extent, displacement: Extent) -> Error {
    Error::new(
        ErrorClass::Type,
        E_DOMAIN_WARP_INPUT,
        format!(
            "field.domain_warp `displacement` extent {}x{} must match the `source` extent {}x{}",
            displacement.width, displacement.height, source.width, source.height
        ),
    )
}

/// A buffer-length-mismatch execution error.
fn buffer_error(actual: usize) -> Error {
    Error::new(
        ErrorClass::Execution,
        E_DOMAIN_WARP_BUFFER,
        format!("field.domain_warp produced a buffer of unexpected length {actual}"),
    )
}

/// The mandatory `cpu.reference@1` oracle implementation id.
fn reference_impl() -> Result<ImplId> {
    ImplId::new("cpu", "reference", 1)
}

/// Verification declarations for `field.domain_warp@1`: a bounded resampling op
/// verified by analytic properties (zero-displacement identity, constant-
/// displacement translation equivalence, round-trip, boundary honoring). No
/// perceptual metric applies — correctness is the resampling property set.
fn warp_test_metadata() -> paintop_ir::TestMetadata {
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
            "field.domain_warp is a displacement-field resampler verified by analytic properties \
             (zero-displacement identity, constant-displacement integer/fractional translation \
             equivalence, warp/inverse-warp round-trip within the bilinear tolerance, boundary-mode \
             honoring); there is no perceptual-quality metric to apply",
        ),
    );
    paintop_ir::TestMetadata {
        has_analytic_reference: true,
        has_property_tests: true,
        golden_fixtures: vec![],
        not_applicable_reason: String::new(),
        verification: decls,
    }
}

#[cfg(test)]
mod tests;
