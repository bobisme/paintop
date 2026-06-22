//! The `filter.bilateral@1` operation: the **bilateral filter** (Tomasi &
//! Manduchi 1998).
//!
//! An edge-preserving smoother whose weights combine a spatial Gaussian with a
//! range (intensity) Gaussian (`OP_CATALOG` §8).
//!
//! Each output pixel is a normalized weighted average of its spatial neighbours,
//! where a neighbour `q` of the centre `p` contributes weight
//!
//! ```text
//! w(p, q) = exp( −‖p − q‖² / (2 σ_s²) ) · exp( −‖I_p − I_q‖² / (2 σ_r²) )
//! ```
//!
//! the product of a **spatial** Gaussian (`spatial_sigma`, in pixels) and a
//! **range** Gaussian (`range_sigma`, in intensity units). The range distance
//! `‖I_p − I_q‖²` is the sum of squared per-channel differences, so on a
//! multi-channel image the filter respects colour edges jointly (not per channel).
//! The output is `Σ_q w(p,q)·I_q / Σ_q w(p,q)`, computed per channel with the
//! **same** shared weights.
//!
//! Because the range term down-weights neighbours across an intensity edge, flat
//! regions are smoothed (every neighbour is similar → weight ≈ spatial Gaussian →
//! a near-Gaussian blur) while edges are preserved (neighbours on the far side of
//! a step contribute almost nothing). A **constant** image is reproduced exactly
//! (all range terms are `1`, the weights are the normalized spatial Gaussian, and
//! a normalized average of a constant is that constant).
//!
//! # Window
//!
//! The spatial window is the square of **radius** `r = ceil(3 · σ_s)` (the
//! declared halo), under a `clamp` boundary — a tap overhanging the edge reads the
//! nearest in-bounds sample. This is the exact, direct reference: a brute-force
//! double loop over the window per output pixel, summing in `f64`. (There is no
//! separable fast path; the range weight is not separable.)
//!
//! # Determinism
//!
//! Every output sample is a fixed-order `f64` accumulation of the same taps on
//! every run, rounded once to `f32`. The op is bit-identical on reruns and
//! declares [`Bounded`](DeterminismTier::Bounded) (the `exp`/divide agree with an
//! independent reference only within a discretization bound).
//!
//! # Tolerance contract (M4 edge-aware gate)
//!
//! The op is verified against an *independent* brute-force reference (a direct
//! window sum written separately from the production kernel) and against its
//! analytic identities:
//!
//! - **constant-image identity**: a flat image is reproduced to within
//!   [`FLAT_IDENTITY_TOLERANCE`] (`1e-5`), the `f32` rounding floor — every range
//!   weight is `1`, the weights are the normalized spatial Gaussian, and a
//!   normalized average of a constant is that constant.
//! - **reference differential**: every output sample agrees with the independent
//!   reference to within [`REFERENCE_TOLERANCE`] (`1e-4`), the bounded-tier
//!   discretization budget; the same tolerance bounds the large-`range_sigma`
//!   reduction to a pure spatial Gaussian.
//! - **edge preservation**: a step edge keeps a cross-boundary jump `> 0.8` under
//!   a small `range_sigma` while the flats either side are smoothed — the
//!   edge-aware property the M4 gate asserts.
//!
//! These declared tolerances are the op's bounded-tier contract; the
//! `cargo xtask verify-op filter.bilateral@1` report records the covered
//! categories (analytic-fixtures, property-tests, metamorphic) carrying this
//! evidence.

use paintop_core::executor::{InputValues, OpImplementation, OutputValues, ResourceValue};
use paintop_ir::{
    AssertionResult, Descriptors, DeterminismTier, Error, ErrorClass, ErrorContext, Extent, ImplId,
    InputRegions, InputSpec, OpContract, OperationManifest, OutputDescriptors, OutputRegions,
    OutputSpec, ParamSpec, ParamType, ParamUnit, Rect, ResourceDescriptor, ResourceKind, Result,
    RoiCategory, RoiPolicy, TestMetadata,
};

/// The canonical id of the bilateral-filter operation.
pub const BILATERAL_OP_ID: &str = "filter.bilateral@1";

/// The `input` was absent or carried an unsupported descriptor.
pub const E_BILATERAL_INPUT: &str = "E_BILATERAL_INPUT";

/// A `spatial_sigma` / `range_sigma` parameter was missing, malformed, or out of
/// range.
pub const E_BILATERAL_PARAM: &str = "E_BILATERAL_PARAM";

/// The largest `spatial_sigma` accepted, keeping the window finite.
pub const SPATIAL_SIGMA_MAX: f64 = 64.0;

/// The declared bounded-tier tolerance against an independent reference.
///
/// `OP_CATALOG` §8, the M4 edge-aware gate. The reference sums the window in a
/// different order from the production kernel; `1e-4` is the discretization
/// budget that gap — and the large-`range_sigma` Gaussian reduction — stays
/// within.
pub const REFERENCE_TOLERANCE: f64 = 1.0e-4;

/// The declared constant-image identity tolerance: a flat image has all range
/// weights `1`, so the output equals the input up to the `f32` storage floor.
pub const FLAT_IDENTITY_TOLERANCE: f32 = 1.0e-5;

/// A resolved bilateral request: the two sigmas (both strictly positive).
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct BilateralRequest {
    spatial_sigma: f64,
    range_sigma: f64,
}

impl BilateralRequest {
    /// Parse and validate both sigmas: finite, strictly positive, spatial bounded.
    fn resolve(params: &serde_json::Value) -> Result<Self> {
        let spatial_sigma = positive_sigma(params, "spatial_sigma")?;
        if spatial_sigma > SPATIAL_SIGMA_MAX {
            return Err(Error::new(
                ErrorClass::Policy,
                E_BILATERAL_PARAM,
                format!(
                    "filter.bilateral `spatial_sigma` {spatial_sigma} exceeds the limit \
                     {SPATIAL_SIGMA_MAX}"
                ),
            )
            .with_context(
                ErrorContext::default()
                    .with_actual(spatial_sigma.to_string())
                    .with_expected(format!("<= {SPATIAL_SIGMA_MAX}")),
            ));
        }
        let range_sigma = positive_sigma(params, "range_sigma")?;
        Ok(Self {
            spatial_sigma,
            range_sigma,
        })
    }

    /// The spatial window radius `ceil(3 σ_s)` (at least 1).
    fn radius(self) -> u32 {
        #[allow(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "3*sigma is positive and bounded by 3*SPATIAL_SIGMA_MAX, well within u32"
        )]
        let r = (3.0 * self.spatial_sigma).ceil() as u32;
        r.max(1)
    }
}

/// Parse a required, finite, strictly-positive sigma param.
fn positive_sigma(params: &serde_json::Value, name: &str) -> Result<f64> {
    let value = params.get(name).ok_or_else(|| {
        Error::new(
            ErrorClass::Schema,
            E_BILATERAL_PARAM,
            format!("filter.bilateral requires a `{name}` parameter"),
        )
    })?;
    let sigma = value.as_f64().ok_or_else(|| {
        Error::new(
            ErrorClass::Schema,
            E_BILATERAL_PARAM,
            format!("filter.bilateral `{name}` must be a number"),
        )
        .with_context(ErrorContext::default().with_actual(value.to_string()))
    })?;
    if !sigma.is_finite() || sigma <= 0.0 {
        return Err(Error::new(
            ErrorClass::Schema,
            E_BILATERAL_PARAM,
            format!("filter.bilateral `{name}` must be finite and strictly positive, got {sigma}"),
        ));
    }
    Ok(sigma)
}

/// The `filter.bilateral@1` operation.
#[derive(Debug, Clone, Copy, Default)]
pub struct Bilateral;

impl Bilateral {
    /// Construct the bilateral-filter operation.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    /// The declared manifest for `filter.bilateral@1`.
    ///
    /// # Errors
    /// Propagates the [`schema`](ErrorClass::Schema) error if the hard-coded op /
    /// impl ids are somehow invalid (they are not).
    pub fn manifest() -> Result<OperationManifest> {
        Ok(OperationManifest {
            id: BILATERAL_OP_ID.parse()?,
            impl_version: 1,
            summary: "Bilateral filter (Tomasi & Manduchi): edge-preserving weighted average with \
                      a spatial Gaussian (spatial_sigma) times a range Gaussian (range_sigma) over \
                      the joint per-channel intensity distance; exact direct reference."
                .to_owned(),
            determinism: DeterminismTier::Bounded,
            roi: RoiPolicy {
                category: RoiCategory::Geometric,
                halo_px: None,
            },
            inputs: vec![InputSpec {
                name: "input".to_owned(),
                kind: ResourceKind::Image,
                required: true,
                doc: "The Image or Field1 to filter (range distance is the joint per-channel \
                      intensity difference)."
                    .to_owned(),
            }],
            outputs: vec![OutputSpec {
                name: "output".to_owned(),
                kind: ResourceKind::Image,
                doc: "The bilaterally-filtered result (same kind and extent as the input)."
                    .to_owned(),
            }],
            params: vec![
                ParamSpec {
                    name: "spatial_sigma".to_owned(),
                    ty: ParamType::Float,
                    unit: Some(ParamUnit::Pixels),
                    required: true,
                    default: None,
                    choices: vec![],
                    doc: "The spatial Gaussian standard deviation in pixels (window radius is \
                          ceil(3*spatial_sigma)); strictly positive."
                        .to_owned(),
                },
                ParamSpec {
                    name: "range_sigma".to_owned(),
                    ty: ParamType::Float,
                    unit: None,
                    required: true,
                    default: None,
                    choices: vec![],
                    doc: "The range (intensity) Gaussian standard deviation; smaller values \
                          preserve sharper edges. Strictly positive."
                        .to_owned(),
                },
            ],
            implementations: vec![reference_impl()?],
            test: bilateral_test_metadata(),
        })
    }
}

/// The supported input descriptor's extent and interleaved channel count.
fn extent_channels(descriptor: &ResourceDescriptor) -> Result<(Extent, u32)> {
    match descriptor {
        ResourceDescriptor::Image(d) => Ok((d.extent, d.layout.channel_count())),
        ResourceDescriptor::Field1(d) => Ok((d.extent, 1)),
        _ => Err(Error::new(
            ErrorClass::Type,
            E_BILATERAL_INPUT,
            "filter.bilateral `input` must be an Image or Field1 resource".to_owned(),
        )),
    }
}

/// A missing-input reference error.
fn missing_input() -> Error {
    Error::new(
        ErrorClass::Reference,
        E_BILATERAL_INPUT,
        "filter.bilateral requires an `input` resource".to_owned(),
    )
}

impl OpContract for Bilateral {
    fn declared_inputs(&self) -> Vec<(String, ResourceKind)> {
        vec![("input".to_owned(), ResourceKind::Image)]
    }

    fn declared_outputs(&self) -> Vec<(String, ResourceKind)> {
        vec![("output".to_owned(), ResourceKind::Image)]
    }

    fn infer_outputs(
        &self,
        inputs: &Descriptors,
        params: &serde_json::Value,
    ) -> Result<OutputDescriptors> {
        let input = inputs.get("input").ok_or_else(missing_input)?;
        extent_channels(input)?;
        BilateralRequest::resolve(params)?;
        let mut out = OutputDescriptors::new();
        out.insert("output".to_owned(), *input);
        Ok(out)
    }

    fn required_inputs(
        &self,
        requested_outputs: &OutputRegions,
        inputs: &Descriptors,
        params: &serde_json::Value,
    ) -> Result<InputRegions> {
        // A neighbourhood op: each output reads a window of the spatial radius.
        // Under clamp a border tap can land on an arbitrary edge sample, so demand
        // the dilated window clipped to the plane.
        let input = inputs.get("input").ok_or_else(missing_input)?;
        let (extent, _channels) = extent_channels(input)?;
        let request = BilateralRequest::resolve(params)?;
        let halo = i64::from(request.radius());
        let mut regions = InputRegions::new();
        if let Some(region) = requested_outputs.get("output") {
            let full = Rect::new(0, 0, i64::from(extent.width), i64::from(extent.height));
            let dilated = Rect::new(
                region.x0 - halo,
                region.y0 - halo,
                region.x1 + halo,
                region.y1 + halo,
            )
            .intersect(full);
            regions.insert("input".to_owned(), dilated);
        }
        Ok(regions)
    }

    fn validate_postconditions(
        &self,
        outputs: &OutputDescriptors,
        _params: &serde_json::Value,
    ) -> Result<Vec<AssertionResult>> {
        Ok(vec![match outputs.get("output") {
            Some(ResourceDescriptor::Image(_) | ResourceDescriptor::Field1(_)) => {
                AssertionResult::pass("produces_filtered")
            }
            _ => AssertionResult::fail("produces_filtered", "no `output` Image/Field1 produced"),
        }])
    }
}

impl OpImplementation for Bilateral {
    fn compute(
        &self,
        inputs: &InputValues,
        params: &serde_json::Value,
    ) -> std::result::Result<OutputValues, Error> {
        let input = inputs.get("input").ok_or_else(missing_input)?;
        let (extent, channels) = extent_channels(input.descriptor())?;
        let request = BilateralRequest::resolve(params)?;

        let samples = bilateral_filter(input.samples(), extent, channels, request);
        let descriptor = *input.descriptor();
        let value = ResourceValue::new(descriptor, channels, samples).map_err(|actual| {
            Error::new(
                ErrorClass::Execution,
                E_BILATERAL_INPUT,
                format!("filter.bilateral produced a sample buffer of unexpected length {actual}"),
            )
        })?;
        let mut out = OutputValues::new();
        out.insert("output".to_owned(), value);
        Ok(out)
    }
}

/// The direct bilateral filter over an interleaved plane: for each output pixel a
/// normalized weighted average of its spatial window, with weights the product of
/// the spatial and range Gaussians (shared across channels).
pub(crate) fn bilateral_filter(
    samples: &[f32],
    extent: Extent,
    channels: u32,
    request: BilateralRequest,
) -> Vec<f32> {
    let width = extent.width as usize;
    let height = extent.height as usize;
    let ch = channels as usize;
    if width == 0 || height == 0 || ch == 0 {
        return Vec::new();
    }
    let radius = i64::from(request.radius());
    let two_spatial_sq = 2.0 * request.spatial_sigma * request.spatial_sigma;
    let two_range_sq = 2.0 * request.range_sigma * request.range_sigma;
    let last_x = i64::try_from(width).unwrap_or(i64::MAX) - 1;
    let last_y = i64::try_from(height).unwrap_or(i64::MAX) - 1;

    let mut out = vec![0.0_f32; samples.len()];
    for y in 0..height {
        let yi = i64::try_from(y).unwrap_or(0);
        for x in 0..width {
            let xi = i64::try_from(x).unwrap_or(0);
            let centre = (y * width + x) * ch;
            // Accumulate the per-channel weighted sums and the shared weight sum.
            let mut acc = vec![0.0_f64; ch];
            let mut weight_sum = 0.0_f64;
            for dy in -radius..=radius {
                let sy = (yi + dy).clamp(0, last_y);
                let syu = usize::try_from(sy).unwrap_or(0);
                for dx in -radius..=radius {
                    let sx = (xi + dx).clamp(0, last_x);
                    let sxu = usize::try_from(sx).unwrap_or(0);
                    let neighbour = (syu * width + sxu) * ch;

                    // Spatial term: distance in *requested* offset space (so the
                    // clamp does not collapse border weights), exact for in-bounds.
                    #[allow(
                        clippy::cast_precision_loss,
                        reason = "dx, dy are small window offsets bounded by 3*SPATIAL_SIGMA_MAX"
                    )]
                    let spatial_d2 = (dx * dx + dy * dy) as f64;
                    let spatial = (-spatial_d2 / two_spatial_sq).exp();

                    // Range term: the joint per-channel squared intensity distance.
                    let mut range_d2 = 0.0_f64;
                    for c in 0..ch {
                        let diff =
                            f64::from(samples[neighbour + c]) - f64::from(samples[centre + c]);
                        range_d2 = diff.mul_add(diff, range_d2);
                    }
                    let range = (-range_d2 / two_range_sq).exp();

                    let w = spatial * range;
                    weight_sum += w;
                    for c in 0..ch {
                        acc[c] = w.mul_add(f64::from(samples[neighbour + c]), acc[c]);
                    }
                }
            }
            // Normalize (weight_sum > 0 always — the centre tap has weight 1).
            for c in 0..ch {
                #[allow(
                    clippy::cast_possible_truncation,
                    reason = "weighted average accumulated in f64, stored as the f32 sample type"
                )]
                {
                    out[centre + c] = (acc[c] / weight_sum) as f32;
                }
            }
        }
    }
    out
}

/// The mandatory `cpu.reference@1` oracle implementation id.
fn reference_impl() -> Result<ImplId> {
    ImplId::new("cpu", "reference", 1)
}

/// Verification declarations for `filter.bilateral@1`: a bounded, single-
/// reference neighbourhood op. Differential does not apply (one implementation).
/// Perceptual is not applicable: correctness is the analytic bilateral property
/// set (constant-image identity, flat-region smoothing, step-edge preservation,
/// the range-sigma → spatial-Gaussian limit, the closed-form weighted average
/// against an independent reference), not a perceptual metric.
fn bilateral_test_metadata() -> TestMetadata {
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
            "filter.bilateral is a closed-form edge-preserving weighted average verified by \
             analytic properties (constant-image identity, flat-region smoothing, step-edge \
             preservation, the large-range-sigma Gaussian limit, the weighted average against an \
             independent direct reference); there is no perceptual-quality metric to apply",
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
