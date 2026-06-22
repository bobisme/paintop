//! The `filter.guided@1` operation: the **guided filter** (He, Sun & Tang 2013).
//!
//! An edge-preserving smoother that filters an `input` under the structure of a
//! `guide` image via a per-window **local linear model** (`OP_CATALOG` §8).
//!
//! The guided filter assumes the output `q` is a *local linear transform* of the
//! guide `I`: in a square window `ω_k` of radius `r` centred at `k`,
//! `q_i = a_k · I_i + b_k` for all `i ∈ ω_k`. The coefficients `(a_k, b_k)` are
//! the ridge-regression fit of the input `p` against the guide over that window:
//!
//! ```text
//! a_k = ( mean_k(I·p) − mean_k(I)·mean_k(p) ) / ( var_k(I) + ε )
//! b_k = mean_k(p) − a_k · mean_k(I)
//! ```
//!
//! with `ε` (`epsilon`) the regularization that bounds `a_k` (and so controls how
//! aggressively flat regions are smoothed). A pixel belongs to every window that
//! covers it, so the final output averages the linear models of all such windows:
//!
//! ```text
//! q_i = mean over windows ∋ i of (a_k · I_i + b_k) = ā_i · I_i + b̄_i
//! ```
//!
//! where `ā`, `b̄` are the box-mean of the per-window coefficients. This is the
//! exact `O(N)` reference formulation: every window mean is the **box mean** of
//! radius `r` over the in-bounds window intersection (the shrinking-window
//! boundary — a border pixel averages the smaller neighbourhood it has, rather
//! than replicating edge samples), so the filter preserves a constant input
//! (the linear fit is `a = 0`, `b = mean(p)`) and, when `guide == input`, reduces
//! to the classic self-guided edge-preserving smoother.
//!
//! # Channels
//!
//! Both `input` and `guide` are filtered per channel: input channel `c` is guided
//! by guide channel `c`. A **single-channel guide** is broadcast across all input
//! channels (the common "grayscale guide for a colour input" case). The two
//! resources must share one pixel extent.
//!
//! # Identity / flat cases
//!
//! - A **flat input** (constant `p`) yields `a_k = 0`, `b_k = p`, so the output is
//!   that constant exactly (to rounding) — the documented flat-identity property.
//! - A **flat guide** (`var(I) = 0`) gives `a_k = 0`, `b_k = mean_k(p)`: with no
//!   structure to follow the output is `b̄ = mean(mean(p))`, the input passed
//!   through the box mean twice (the coefficients are themselves box-averaged in
//!   the final step) — the correct degenerate "just smooth it" behaviour.
//!
//! # Determinism
//!
//! Every box mean is a fixed-order `f64` accumulation rounded once to `f32`; the
//! coefficient solve is a closed-form divide. The op is bit-identical on reruns
//! and declares [`Bounded`](DeterminismTier::Bounded) (the divides agree with an
//! independent reference only within a discretization bound).
//!
//! # Tolerance contract (M4 edge-aware gate)
//!
//! The op is verified against an *independent* brute-force reference — a direct
//! per-window ridge-regression fit with the same shrinking-window boundary,
//! structurally different from the production integral-image path — and against
//! its analytic identities:
//!
//! - **flat-input identity**: a constant input is reproduced to within
//!   [`FLAT_IDENTITY_TOLERANCE`] (`1e-5`), the `f32` rounding floor — `a = 0`,
//!   `b = const`, so this is exact up to storage precision.
//! - **reference differential**: every output sample agrees with the independent
//!   reference to within [`REFERENCE_TOLERANCE`] (`1e-4`), the bounded-tier
//!   discretization budget for the two divergent summation orders.
//! - **edge preservation**: a self-guided step edge keeps a cross-boundary jump
//!   `> 0.5` while the flats either side relax toward their local mean — the
//!   edge-aware property the M4 gate asserts.
//!
//! These declared tolerances are the op's bounded-tier contract; the
//! `cargo xtask verify-op filter.guided@1` report records the covered categories
//! (analytic-fixtures, property-tests, metamorphic) that carry this evidence.

use paintop_core::executor::{InputValues, OpImplementation, OutputValues, ResourceValue};
use paintop_ir::{
    AssertionResult, Descriptors, DeterminismTier, Error, ErrorClass, ErrorContext, Extent, ImplId,
    InputRegions, InputSpec, OpContract, OperationManifest, OutputDescriptors, OutputRegions,
    OutputSpec, ParamSpec, ParamType, ParamUnit, Rect, ResourceDescriptor, ResourceKind, Result,
    RoiCategory, RoiPolicy, TestMetadata,
};

/// The canonical id of the guided-filter operation.
pub const GUIDED_OP_ID: &str = "filter.guided@1";

/// The `input` or `guide` was absent or carried an unsupported descriptor.
pub const E_GUIDED_INPUT: &str = "E_GUIDED_INPUT";

/// A `radius` / `epsilon` parameter was missing, malformed, or out of range.
pub const E_GUIDED_PARAM: &str = "E_GUIDED_PARAM";

/// The largest box radius accepted, keeping the window sums bounded.
pub const RADIUS_MAX: u32 = 256;

/// The declared bounded-tier tolerance against an independent reference.
///
/// `OP_CATALOG` §8, the M4 edge-aware gate. The production integral-image path
/// and the direct per-window fit reassociate the same `f64` sums differently;
/// `1e-4` is the discretization budget that gap stays within.
pub const REFERENCE_TOLERANCE: f64 = 1.0e-4;

/// The declared flat-input identity tolerance: a constant input fits `a = 0`,
/// `b = const`, so the output equals the input up to the `f32` storage floor.
pub const FLAT_IDENTITY_TOLERANCE: f32 = 1.0e-5;

/// A resolved guided-filter request: the box radius and the regularization.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct GuidedRequest {
    radius: u32,
    epsilon: f64,
}

impl GuidedRequest {
    /// Parse and validate the `radius` (>= 1) and `epsilon` (>= 0) params.
    fn resolve(params: &serde_json::Value) -> Result<Self> {
        let radius = radius_param(params)?;
        let epsilon = epsilon_param(params)?;
        Ok(Self { radius, epsilon })
    }
}

/// Parse the required, positive, bounded integer `radius`.
fn radius_param(params: &serde_json::Value) -> Result<u32> {
    let value = params.get("radius").ok_or_else(|| {
        Error::new(
            ErrorClass::Schema,
            E_GUIDED_PARAM,
            "filter.guided requires a `radius` parameter".to_owned(),
        )
    })?;
    let raw = value.as_u64().ok_or_else(|| {
        Error::new(
            ErrorClass::Schema,
            E_GUIDED_PARAM,
            "filter.guided `radius` must be a non-negative integer".to_owned(),
        )
        .with_context(ErrorContext::default().with_actual(value.to_string()))
    })?;
    let radius = u32::try_from(raw).map_err(|_| {
        Error::new(
            ErrorClass::Schema,
            E_GUIDED_PARAM,
            format!("filter.guided `radius` ({raw}) does not fit in u32"),
        )
    })?;
    if radius == 0 {
        return Err(Error::new(
            ErrorClass::Schema,
            E_GUIDED_PARAM,
            "filter.guided `radius` must be >= 1 (a radius-0 window has no neighbourhood)"
                .to_owned(),
        ));
    }
    if radius > RADIUS_MAX {
        return Err(Error::new(
            ErrorClass::Policy,
            E_GUIDED_PARAM,
            format!("filter.guided `radius` {radius} exceeds the limit {RADIUS_MAX}"),
        )
        .with_context(
            ErrorContext::default()
                .with_actual(radius.to_string())
                .with_expected(format!("<= {RADIUS_MAX}")),
        ));
    }
    Ok(radius)
}

/// Parse the required, finite, non-negative `epsilon`.
fn epsilon_param(params: &serde_json::Value) -> Result<f64> {
    let value = params.get("epsilon").ok_or_else(|| {
        Error::new(
            ErrorClass::Schema,
            E_GUIDED_PARAM,
            "filter.guided requires an `epsilon` parameter".to_owned(),
        )
    })?;
    let epsilon = value.as_f64().ok_or_else(|| {
        Error::new(
            ErrorClass::Schema,
            E_GUIDED_PARAM,
            "filter.guided `epsilon` must be a number".to_owned(),
        )
        .with_context(ErrorContext::default().with_actual(value.to_string()))
    })?;
    if !epsilon.is_finite() || epsilon < 0.0 {
        return Err(Error::new(
            ErrorClass::Schema,
            E_GUIDED_PARAM,
            format!("filter.guided `epsilon` must be finite and non-negative, got {epsilon}"),
        ));
    }
    Ok(epsilon)
}

/// The `filter.guided@1` operation.
#[derive(Debug, Clone, Copy, Default)]
pub struct Guided;

impl Guided {
    /// Construct the guided-filter operation.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    /// The declared manifest for `filter.guided@1`.
    ///
    /// # Errors
    /// Propagates the [`schema`](ErrorClass::Schema) error if the hard-coded op /
    /// impl ids are somehow invalid (they are not).
    pub fn manifest() -> Result<OperationManifest> {
        Ok(OperationManifest {
            id: GUIDED_OP_ID.parse()?,
            impl_version: 1,
            summary: "Guided filter (He et al.): filter an input under a guide's structure via a \
                      per-window local linear model q = a*I + b fitted by ridge regression \
                      (box radius, epsilon); edge-preserving, flat input preserved exactly."
                .to_owned(),
            determinism: DeterminismTier::Bounded,
            roi: RoiPolicy {
                category: RoiCategory::Geometric,
                halo_px: None,
            },
            inputs: vec![
                InputSpec {
                    name: "input".to_owned(),
                    kind: ResourceKind::Image,
                    required: true,
                    doc: "The Image or Field1 to filter (per channel).".to_owned(),
                },
                InputSpec {
                    name: "guide".to_owned(),
                    kind: ResourceKind::Image,
                    required: true,
                    doc: "The guide Image or Field1 whose structure the filter follows; a \
                          single-channel guide is broadcast across the input's channels."
                        .to_owned(),
                },
            ],
            outputs: vec![OutputSpec {
                name: "output".to_owned(),
                kind: ResourceKind::Image,
                doc: "The guided-filtered result (same kind and extent as the input).".to_owned(),
            }],
            params: vec![
                ParamSpec {
                    name: "radius".to_owned(),
                    ty: ParamType::Integer,
                    unit: Some(ParamUnit::Pixels),
                    required: true,
                    default: None,
                    choices: vec![],
                    doc: "The box-window radius (the window side is 2*radius + 1); >= 1."
                        .to_owned(),
                },
                ParamSpec {
                    name: "epsilon".to_owned(),
                    ty: ParamType::Float,
                    unit: None,
                    required: true,
                    default: None,
                    choices: vec![],
                    doc: "The ridge regularization bounding the linear coefficient; larger \
                          epsilon smooths flatter regions. Non-negative."
                        .to_owned(),
                },
            ],
            implementations: vec![reference_impl()?],
            test: guided_test_metadata(),
        })
    }
}

/// The supported descriptor's extent and interleaved channel count.
fn extent_channels(descriptor: &ResourceDescriptor, role: &str) -> Result<(Extent, u32)> {
    match descriptor {
        ResourceDescriptor::Image(d) => Ok((d.extent, d.layout.channel_count())),
        ResourceDescriptor::Field1(d) => Ok((d.extent, 1)),
        _ => Err(Error::new(
            ErrorClass::Type,
            E_GUIDED_INPUT,
            format!("filter.guided `{role}` must be an Image or Field1 resource"),
        )),
    }
}

/// Validate the `input`/`guide` pairing: same extent, and the guide channel count
/// either matches the input or is `1` (broadcast). Returns the input extent and
/// channel count and whether the guide is broadcast.
fn validate_pair(
    input: &ResourceDescriptor,
    guide: &ResourceDescriptor,
) -> Result<(Extent, u32, bool)> {
    let (in_extent, in_channels) = extent_channels(input, "input")?;
    let (guide_extent, guide_channels) = extent_channels(guide, "guide")?;
    if in_extent != guide_extent {
        return Err(Error::new(
            ErrorClass::Semantic,
            E_GUIDED_INPUT,
            format!(
                "filter.guided `input` ({}x{}) and `guide` ({}x{}) must share one extent",
                in_extent.width, in_extent.height, guide_extent.width, guide_extent.height
            ),
        ));
    }
    if guide_channels != in_channels && guide_channels != 1 {
        return Err(Error::new(
            ErrorClass::Semantic,
            E_GUIDED_INPUT,
            format!(
                "filter.guided `guide` has {guide_channels} channels; expected {in_channels} \
                 (per-channel) or 1 (broadcast)"
            ),
        ));
    }
    Ok((
        in_extent,
        in_channels,
        guide_channels == 1 && in_channels != 1,
    ))
}

impl OpContract for Guided {
    fn declared_inputs(&self) -> Vec<(String, ResourceKind)> {
        vec![
            ("input".to_owned(), ResourceKind::Image),
            ("guide".to_owned(), ResourceKind::Image),
        ]
    }

    fn declared_outputs(&self) -> Vec<(String, ResourceKind)> {
        vec![("output".to_owned(), ResourceKind::Image)]
    }

    fn infer_outputs(
        &self,
        inputs: &Descriptors,
        params: &serde_json::Value,
    ) -> Result<OutputDescriptors> {
        let input = inputs.get("input").ok_or_else(|| missing("input"))?;
        let guide = inputs.get("guide").ok_or_else(|| missing("guide"))?;
        validate_pair(input, guide)?;
        GuidedRequest::resolve(params)?;
        let mut out = OutputDescriptors::new();
        // Same kind and extent as the input (the filter preserves the frame).
        out.insert("output".to_owned(), *input);
        Ok(out)
    }

    fn required_inputs(
        &self,
        requested_outputs: &OutputRegions,
        inputs: &Descriptors,
        params: &serde_json::Value,
    ) -> Result<InputRegions> {
        // A neighbourhood op: the output at a pixel reads box means over windows
        // covering it, so the read footprint dilates by 2*radius (a pixel's value
        // depends on coefficients of windows up to radius away, each of which
        // averages samples up to radius away). Under clamp a border read can land
        // on an arbitrary edge sample, so demand the dilated window clipped to the
        // plane for both ports.
        let input = inputs.get("input").ok_or_else(|| missing("input"))?;
        let guide = inputs.get("guide").ok_or_else(|| missing("guide"))?;
        let (extent, _channels, _broadcast) = validate_pair(input, guide)?;
        let request = GuidedRequest::resolve(params)?;
        let halo = i64::from(request.radius) * 2;
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
            regions.insert("guide".to_owned(), dilated);
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

/// A missing-input reference error for port `name`.
fn missing(name: &str) -> Error {
    Error::new(
        ErrorClass::Reference,
        E_GUIDED_INPUT,
        format!("filter.guided requires a `{name}` resource"),
    )
}

impl OpImplementation for Guided {
    fn compute(
        &self,
        inputs: &InputValues,
        params: &serde_json::Value,
    ) -> std::result::Result<OutputValues, Error> {
        let input = inputs.get("input").ok_or_else(|| missing("input"))?;
        let guide = inputs.get("guide").ok_or_else(|| missing("guide"))?;
        let (extent, in_channels, _broadcast) =
            validate_pair(input.descriptor(), guide.descriptor())?;
        let request = GuidedRequest::resolve(params)?;
        let guide_channels = guide.channels();

        let samples = guided_filter(
            input.samples(),
            guide.samples(),
            extent,
            in_channels,
            guide_channels,
            request,
        );

        let descriptor = *input.descriptor();
        let value = ResourceValue::new(descriptor, in_channels, samples).map_err(|actual| {
            Error::new(
                ErrorClass::Execution,
                E_GUIDED_INPUT,
                format!("filter.guided produced a sample buffer of unexpected length {actual}"),
            )
        })?;
        let mut out = OutputValues::new();
        out.insert("output".to_owned(), value);
        Ok(out)
    }
}

/// The guided filter over an interleaved plane: filter each `input` channel under
/// the matching (or broadcast) `guide` channel and re-interleave.
pub(crate) fn guided_filter(
    input: &[f32],
    guide: &[f32],
    extent: Extent,
    in_channels: u32,
    guide_channels: u32,
    request: GuidedRequest,
) -> Vec<f32> {
    let width = extent.width as usize;
    let height = extent.height as usize;
    let len = width * height;
    let in_ch = in_channels as usize;
    let guide_ch = guide_channels as usize;
    if len == 0 || in_ch == 0 {
        return Vec::new();
    }
    let mut out = vec![0.0_f32; len * in_ch];
    for channel in 0..in_ch {
        // Per-channel guide index: broadcast a single-channel guide.
        let guide_index = if guide_ch == 1 { 0 } else { channel };
        let input_plane = extract(input, len, in_ch, channel);
        let guide_plane = extract(guide, len, guide_ch, guide_index);
        let filtered = guided_channel(&guide_plane, &input_plane, width, height, request);
        for px in 0..len {
            #[allow(
                clippy::cast_possible_truncation,
                reason = "guided output accumulated in f64, stored as the f32 sample type"
            )]
            {
                out[px * in_ch + channel] = filtered[px] as f32;
            }
        }
    }
    out
}

/// Extract channel `c` of an interleaved plane into a contiguous f64 plane.
fn extract(samples: &[f32], n: usize, channels: usize, c: usize) -> Vec<f64> {
    let mut plane = Vec::with_capacity(n);
    for i in 0..n {
        plane.push(f64::from(samples[i * channels + c]));
    }
    plane
}

/// The single-channel guided filter: guide `guide`, input `input`, box radius and
/// epsilon from `request`. Returns the filtered plane.
fn guided_channel(
    guide: &[f64],
    input: &[f64],
    width: usize,
    height: usize,
    request: GuidedRequest,
) -> Vec<f64> {
    let radius = request.radius;
    let eps = request.epsilon;
    let mean_guide = box_mean(guide, width, height, radius);
    let mean_input = box_mean(input, width, height, radius);
    let prod: Vec<f64> = guide.iter().zip(input).map(|(gi, pi)| gi * pi).collect();
    let mean_prod = box_mean(&prod, width, height, radius);
    let squares: Vec<f64> = guide.iter().map(|gi| gi * gi).collect();
    let mean_square = box_mean(&squares, width, height, radius);

    let len = width * height;
    let mut coef_a = vec![0.0_f64; len];
    let mut coef_b = vec![0.0_f64; len];
    for px in 0..len {
        let cov = mean_guide[px].mul_add(-mean_input[px], mean_prod[px]);
        let var = mean_guide[px].mul_add(-mean_guide[px], mean_square[px]);
        let denom = var + eps;
        // A zero denominator means a flat guide window with no regularization:
        // there is no structure to follow, so the linear coefficient is 0 and the
        // model is the local mean of the input (a = 0, b = mean_input).
        let a_k = if denom > 0.0 { cov / denom } else { 0.0 };
        coef_a[px] = a_k;
        coef_b[px] = a_k.mul_add(-mean_guide[px], mean_input[px]);
    }
    let mean_a = box_mean(&coef_a, width, height, radius);
    let mean_b = box_mean(&coef_b, width, height, radius);

    let mut result = vec![0.0_f64; len];
    for px in 0..len {
        result[px] = mean_a[px].mul_add(guide[px], mean_b[px]);
    }
    result
}

/// The box mean of `plane` over the in-bounds intersection of a `(2r+1)×(2r+1)`
/// window (a fixed-order f64 accumulation).
///
/// The window is clipped to the image rather than replicating edge samples, so a
/// border pixel averages the smaller in-bounds neighbourhood it actually has
/// (the standard guided-filter "shrinking window" boundary). Uses a summed-area
/// (integral image) table so each window mean is `O(1)`, keeping the whole filter
/// `O(N)` independent of the radius.
pub(crate) fn box_mean(plane: &[f64], w: usize, h: usize, r: u32) -> Vec<f64> {
    if w == 0 || h == 0 {
        return Vec::new();
    }
    let ri = i64::from(r);
    // Integral image with a zero top/left border: sat[(y+1)*(w+1) + (x+1)] is the
    // sum of plane[0..=y, 0..=x].
    let sw = w + 1;
    let sh = h + 1;
    let mut sat = vec![0.0_f64; sw * sh];
    for y in 0..h {
        let mut row_sum = 0.0_f64;
        for x in 0..w {
            row_sum += plane[y * w + x];
            sat[(y + 1) * sw + (x + 1)] = sat[y * sw + (x + 1)] + row_sum;
        }
    }
    // A clamped window sum reads the integral image at clamped corner indices.
    let last_x = i64::try_from(w).unwrap_or(i64::MAX);
    let last_y = i64::try_from(h).unwrap_or(i64::MAX);
    let clamp_x = |v: i64| -> usize { usize::try_from(v.clamp(0, last_x)).unwrap_or(0) };
    let clamp_y = |v: i64| -> usize { usize::try_from(v.clamp(0, last_y)).unwrap_or(0) };

    let mut out = vec![0.0_f64; w * h];
    for y in 0..h {
        let yi = i64::try_from(y).unwrap_or(0);
        // Half-open window rows [y0, y1) in plane coords, clamped to [0, h].
        let y0 = clamp_y(yi - ri);
        let y1 = clamp_y(yi + ri + 1);
        for x in 0..w {
            let xi = i64::try_from(x).unwrap_or(0);
            let x0 = clamp_x(xi - ri);
            let x1 = clamp_x(xi + ri + 1);
            let sum = sat[y1 * sw + x1] - sat[y0 * sw + x1] - sat[y1 * sw + x0] + sat[y0 * sw + x0];
            #[allow(
                clippy::cast_precision_loss,
                reason = "window area is bounded by (2*RADIUS_MAX+1)^2, exact in f64"
            )]
            let area = ((x1 - x0) * (y1 - y0)) as f64;
            out[y * w + x] = if area > 0.0 { sum / area } else { 0.0 };
        }
    }
    out
}

/// The mandatory `cpu.reference@1` oracle implementation id.
fn reference_impl() -> Result<ImplId> {
    ImplId::new("cpu", "reference", 1)
}

/// Verification declarations for `filter.guided@1`: a bounded, single-reference
/// neighbourhood op. Differential does not apply (one implementation).
/// Perceptual is not applicable: correctness is the analytic guided-filter
/// property set (flat-input identity, flat-guide box-mean degeneracy,
/// self-guided edge preservation, the closed-form linear model against a small
/// independent reference), not a perceptual metric.
fn guided_test_metadata() -> TestMetadata {
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
            "filter.guided is a closed-form local-linear-model filter verified by analytic \
             properties (flat-input identity, flat-guide box-mean degeneracy, self-guided edge \
             preservation, the per-window ridge fit against an independent box-statistics \
             reference); there is no perceptual-quality metric to apply",
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
