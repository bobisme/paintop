//! The `filter.convolve@1` operation: direct 2-D correlation of an `Image` or
//! `Field1` against a small, explicit kernel under a boundary mode.
//!
//! Refs: `OP_CATALOG` §8, `AGENT_VERIFICATION` §3.3, `IR_SPEC` §8.4.
//!
//! `filter.convolve` is the **semantic oracle** for every later optimized,
//! separable, or FFT convolution path. It is written for clarity, not speed: a
//! direct quadruple loop over output pixels and kernel taps, summing in `f64` and
//! storing `f32`. Each channel is filtered independently; alpha (if present) is
//! filtered like any other channel, so the op operates on whatever
//! premultiplication state its input declares.
//!
//! # Kernel
//!
//! The kernel is an explicit param object:
//!
//! ```json
//! { "width": 3, "height": 3, "origin_x": 1, "origin_y": 1,
//!   "weights": [0,0,0, 0,1,0, 0,0,0] }
//! ```
//!
//! `weights` is a row-major array of exactly `width * height` finite numbers.
//! `origin_x`/`origin_y` name the tap that lands on the output pixel (the kernel
//! "hot spot"); they default to the geometric centre `(width/2, height/2)`. The
//! op is **correlation**: output sample `o(x, y)` sums `w(kx, ky) * src(x + kx -
//! origin_x, y + ky - origin_y)` over the kernel. (A true convolution is this with
//! a flipped kernel; the caller flips if they want convolution, so the impulse
//! response is exactly the kernel — `AGENT_VERIFICATION` §3.3.)
//!
//! # Boundary modes
//!
//! - **`constant`** — out-of-bounds samples take a fixed per-channel `value`
//!   (default `0`).
//! - **`transparent`** — out-of-bounds samples are zero on every channel (the
//!   premultiplied-alpha "nothing is there" convention); equivalent to a
//!   constant of all-zero.
//! - **`clamp`** — replicate the nearest edge sample.
//! - **`mirror`** — half-sample reflection across the edge (edge not repeated).
//! - **`wrap`** — periodic (toroidal) tiling.
//! - **`valid`** — *valid-only*: the output is shrunk so the kernel never reads
//!   out of bounds. The output extent is `(W - width + 1, H - height + 1)` and
//!   output pixel `(x, y)` is centred on input pixel `(x + origin_x, y +
//!   origin_y)`. A kernel larger than the input on either axis yields an empty
//!   extent on that axis.
//!
//! The boundary mode is a normalized param and therefore part of the op hash, so
//! two plans that differ only in boundary mode are distinct cache entries
//! (`AGENT_VERIFICATION` §3.3).
//!
//! # Determinism
//!
//! Each output sample is a fixed-order `f64` accumulation of the same taps on
//! every run, rounded once to `f32`; the op is
//! [`Exact`](DeterminismTier::Exact) — bit-identical to its own reference on every
//! run for a given input/param set.

use paintop_core::executor::{InputValues, OpImplementation, OutputValues, ResourceValue};
use paintop_ir::{
    AssertionResult, Descriptors, DeterminismTier, Error, ErrorClass, ErrorContext, Extent,
    FieldDescriptor, ImageDescriptor, ImplId, InputRegions, InputSpec, OpContract,
    OperationManifest, OutputDescriptors, OutputRegions, OutputSpec, ParamSpec, ParamType, Rect,
    ResourceDescriptor, ResourceKind, Result, RoiCategory, RoiPolicy, TestMetadata,
};

/// The canonical id of the convolution operation.
pub const CONVOLVE_OP_ID: &str = "filter.convolve@1";

/// The `image`/`field` input was absent or carried an unsupported descriptor.
pub const E_CONVOLVE_INPUT: &str = "E_CONVOLVE_INPUT";

/// The `kernel` param was missing, malformed, or shaped inconsistently.
pub const E_CONVOLVE_KERNEL: &str = "E_CONVOLVE_KERNEL";

/// The `mode` / `value` boundary parameters were missing or malformed.
pub const E_CONVOLVE_PARAM: &str = "E_CONVOLVE_PARAM";

/// How out-of-bounds taps are resolved when the kernel overhangs the input edge.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BoundaryMode {
    /// A fixed per-channel constant.
    Constant,
    /// Zero on every channel (premultiplied "nothing"); a constant-zero alias.
    Transparent,
    /// Replicate the nearest edge sample.
    Clamp,
    /// Half-sample reflection across the edge (edge not repeated).
    Mirror,
    /// Periodic (toroidal) tiling.
    Wrap,
    /// Valid-only: shrink the output so the kernel never overhangs.
    Valid,
}

impl BoundaryMode {
    /// Parse the boundary mode from its wire token.
    fn from_token(token: &str) -> Option<Self> {
        match token {
            "constant" => Some(Self::Constant),
            "transparent" => Some(Self::Transparent),
            "clamp" => Some(Self::Clamp),
            "mirror" => Some(Self::Mirror),
            "wrap" => Some(Self::Wrap),
            "valid" => Some(Self::Valid),
            _ => None,
        }
    }
}

/// A validated convolution kernel: shape, origin, and row-major weights.
#[derive(Debug, Clone)]
pub(crate) struct Kernel {
    width: u32,
    height: u32,
    origin_x: u32,
    origin_y: u32,
    weights: Vec<f64>,
}

impl Kernel {
    /// Parse and validate the `kernel` param object.
    fn parse(params: &serde_json::Value) -> Result<Self> {
        let kernel = params.get("kernel").ok_or_else(|| {
            Error::new(
                ErrorClass::Schema,
                E_CONVOLVE_KERNEL,
                "filter.convolve requires a `kernel` object".to_owned(),
            )
        })?;
        if !kernel.is_object() {
            return Err(Error::new(
                ErrorClass::Schema,
                E_CONVOLVE_KERNEL,
                "filter.convolve `kernel` must be an object with width/height/weights".to_owned(),
            )
            .with_context(ErrorContext::default().with_actual(kernel.to_string())));
        }
        let width = u32_field(kernel, "width")?;
        let height = u32_field(kernel, "height")?;
        if width == 0 || height == 0 {
            return Err(Error::new(
                ErrorClass::Schema,
                E_CONVOLVE_KERNEL,
                format!("filter.convolve `kernel` must have non-zero extent, got {width}x{height}"),
            ));
        }
        let origin_x = optional_u32_field(kernel, "origin_x", width / 2)?;
        let origin_y = optional_u32_field(kernel, "origin_y", height / 2)?;
        if origin_x >= width || origin_y >= height {
            return Err(Error::new(
                ErrorClass::Schema,
                E_CONVOLVE_KERNEL,
                format!(
                    "filter.convolve `kernel` origin ({origin_x},{origin_y}) is outside the \
                     {width}x{height} kernel"
                ),
            )
            .with_context(
                ErrorContext::default()
                    .with_actual(format!("({origin_x},{origin_y})"))
                    .with_expected(format!("0..{width}, 0..{height}")),
            ));
        }
        let weights = weights_field(kernel, width, height)?;
        Ok(Self {
            width,
            height,
            origin_x,
            origin_y,
            weights,
        })
    }
}

/// Read a required, positive `u32` kernel field.
fn u32_field(kernel: &serde_json::Value, name: &str) -> Result<u32> {
    let value = kernel.get(name).ok_or_else(|| {
        Error::new(
            ErrorClass::Schema,
            E_CONVOLVE_KERNEL,
            format!("filter.convolve `kernel.{name}` is required"),
        )
    })?;
    let n = value.as_u64().ok_or_else(|| {
        Error::new(
            ErrorClass::Schema,
            E_CONVOLVE_KERNEL,
            format!("filter.convolve `kernel.{name}` must be a non-negative integer"),
        )
        .with_context(ErrorContext::default().with_actual(value.to_string()))
    })?;
    u32::try_from(n).map_err(|_| {
        Error::new(
            ErrorClass::Schema,
            E_CONVOLVE_KERNEL,
            format!("filter.convolve `kernel.{name}` ({n}) does not fit in u32"),
        )
    })
}

/// Read an optional `u32` kernel field, defaulting when absent.
fn optional_u32_field(kernel: &serde_json::Value, name: &str, default: u32) -> Result<u32> {
    if kernel.get(name).is_none() {
        return Ok(default);
    }
    u32_field(kernel, name)
}

/// Read and validate the row-major `weights` array (`width * height` finite numbers).
fn weights_field(kernel: &serde_json::Value, width: u32, height: u32) -> Result<Vec<f64>> {
    let value = kernel.get("weights").ok_or_else(|| {
        Error::new(
            ErrorClass::Schema,
            E_CONVOLVE_KERNEL,
            "filter.convolve `kernel.weights` is required".to_owned(),
        )
    })?;
    let array = value.as_array().ok_or_else(|| {
        Error::new(
            ErrorClass::Schema,
            E_CONVOLVE_KERNEL,
            "filter.convolve `kernel.weights` must be a row-major array".to_owned(),
        )
        .with_context(ErrorContext::default().with_actual(value.to_string()))
    })?;
    let expected = (width as usize) * (height as usize);
    if array.len() != expected {
        return Err(Error::new(
            ErrorClass::Schema,
            E_CONVOLVE_KERNEL,
            format!(
                "filter.convolve `kernel.weights` has {} entries but {width}x{height} needs {expected}",
                array.len()
            ),
        )
        .with_context(
            ErrorContext::default()
                .with_actual(array.len().to_string())
                .with_expected(expected.to_string()),
        ));
    }
    let mut out = Vec::with_capacity(expected);
    for (i, entry) in array.iter().enumerate() {
        let n = entry.as_f64().ok_or_else(|| {
            Error::new(
                ErrorClass::Schema,
                E_CONVOLVE_KERNEL,
                format!("filter.convolve `kernel.weights[{i}]` must be a number"),
            )
            .with_context(ErrorContext::default().with_actual(entry.to_string()))
        })?;
        if !n.is_finite() {
            return Err(Error::new(
                ErrorClass::Schema,
                E_CONVOLVE_KERNEL,
                format!("filter.convolve `kernel.weights[{i}]` must be finite, got {n}"),
            ));
        }
        out.push(n);
    }
    Ok(out)
}

/// A fully-resolved convolution request.
#[derive(Debug, Clone)]
pub(crate) struct ConvolveRequest {
    kernel: Kernel,
    mode: BoundaryMode,
    value: Vec<f64>,
}

impl ConvolveRequest {
    /// Parse and validate every param against the input's channel count.
    pub(crate) fn resolve(params: &serde_json::Value, channels: u32) -> Result<Self> {
        let kernel = Kernel::parse(params)?;
        let mode = mode_param(params)?;
        let value = value_param(params, channels as usize)?;
        Ok(Self {
            kernel,
            mode,
            value,
        })
    }

    /// The output extent the request produces for a source extent.
    ///
    /// Identical to the source except under `valid`, which shrinks both axes by
    /// `kernel - 1` (saturating to zero).
    fn output_extent(&self, src: Extent) -> Extent {
        if self.mode == BoundaryMode::Valid {
            let w = src.width.saturating_sub(self.kernel.width - 1);
            let h = src.height.saturating_sub(self.kernel.height - 1);
            Extent::new(w, h)
        } else {
            src
        }
    }
}

/// Parse the optional `mode` param, defaulting to `clamp` (the most common
/// neighbourhood-filter boundary).
fn mode_param(params: &serde_json::Value) -> Result<BoundaryMode> {
    let Some(value) = params.get("mode") else {
        return Ok(BoundaryMode::Clamp);
    };
    let token = value.as_str().ok_or_else(|| {
        Error::new(
            ErrorClass::Schema,
            E_CONVOLVE_PARAM,
            "filter.convolve `mode` must be a string boundary mode".to_owned(),
        )
        .with_context(ErrorContext::default().with_actual(value.to_string()))
    })?;
    BoundaryMode::from_token(token).ok_or_else(|| {
        Error::new(
            ErrorClass::Schema,
            E_CONVOLVE_PARAM,
            format!("filter.convolve `mode` is not a known boundary mode: {token}"),
        )
        .with_context(
            ErrorContext::default()
                .with_actual(token.to_owned())
                .with_expected("constant | transparent | clamp | mirror | wrap | valid"),
        )
    })
}

/// Parse the optional per-channel `value` array (for `constant` mode), defaulting
/// to all-zero.
fn value_param(params: &serde_json::Value, channels: usize) -> Result<Vec<f64>> {
    let Some(value) = params.get("value") else {
        return Ok(vec![0.0; channels]);
    };
    let array = value.as_array().ok_or_else(|| {
        Error::new(
            ErrorClass::Schema,
            E_CONVOLVE_PARAM,
            "filter.convolve `value` must be a per-channel array".to_owned(),
        )
        .with_context(ErrorContext::default().with_actual(value.to_string()))
    })?;
    if array.len() != channels {
        return Err(Error::new(
            ErrorClass::Schema,
            E_CONVOLVE_PARAM,
            format!(
                "filter.convolve `value` has {} components but the input has {channels} channels",
                array.len()
            ),
        )
        .with_context(
            ErrorContext::default()
                .with_actual(array.len().to_string())
                .with_expected(channels.to_string()),
        ));
    }
    let mut out = Vec::with_capacity(channels);
    for (i, component) in array.iter().enumerate() {
        let n = component.as_f64().ok_or_else(|| {
            Error::new(
                ErrorClass::Schema,
                E_CONVOLVE_PARAM,
                format!("filter.convolve `value[{i}]` must be a number"),
            )
            .with_context(ErrorContext::default().with_actual(component.to_string()))
        })?;
        if !n.is_finite() {
            return Err(Error::new(
                ErrorClass::Schema,
                E_CONVOLVE_PARAM,
                format!("filter.convolve `value[{i}]` must be finite, got {n}"),
            ));
        }
        out.push(n);
    }
    Ok(out)
}

/// Map an output coordinate on one axis to a source index under an edge-extension
/// boundary mode, given the source length `n` (`n >= 1`).
///
/// Returns `Some(src_index)` to read `src[src_index]`, or `None` to use the
/// out-of-bounds constant (only when the coordinate is genuinely outside and the
/// mode is `constant`/`transparent`).
fn source_index(coord: i64, n: i64, mode: BoundaryMode) -> Option<i64> {
    if coord >= 0 && coord < n {
        return Some(coord);
    }
    match mode {
        BoundaryMode::Constant | BoundaryMode::Transparent => None,
        // `Valid` never reads out of bounds, so its arm is unreachable in
        // practice; folding it onto `Clamp` is a harmless defensive fallback.
        BoundaryMode::Clamp | BoundaryMode::Valid => Some(coord.clamp(0, n - 1)),
        BoundaryMode::Wrap => Some(coord.rem_euclid(n)),
        BoundaryMode::Mirror => Some(mirror_index(coord, n)),
    }
}

/// Whole-sample mirror (`reflect`): fold `c` into `[0, n)` reflecting across the
/// edge sample without repeating it (period `2(n-1)`). For `n == 1` every index
/// maps to `0`.
const fn mirror_index(c: i64, n: i64) -> i64 {
    if n == 1 {
        return 0;
    }
    let period = 2 * (n - 1);
    let m = c.rem_euclid(period);
    if m < n { m } else { period - m }
}

/// The direct convolution oracle: filter `samples` (row-major, `channels`-
/// interleaved, extent `src`) by `request`, producing the output buffer for
/// `out`.
pub(crate) fn convolve_samples(
    samples: &[f32],
    src: Extent,
    out: Extent,
    request: &ConvolveRequest,
    channels: u32,
) -> Vec<f32> {
    let stride = channels as usize;
    let src_w = i64::from(src.width);
    let src_h = i64::from(src.height);
    let out_w = out.width as usize;
    let out_h = out.height as usize;
    let kw = i64::from(request.kernel.width);
    let kh = i64::from(request.kernel.height);
    let ox = i64::from(request.kernel.origin_x);
    let oy = i64::from(request.kernel.origin_y);
    // Under `valid` the output pixel (x, y) maps to input pixel (x + ox, y + oy);
    // under the edge modes it maps to (x, y) itself.
    let (shift_x, shift_y) = if request.mode == BoundaryMode::Valid {
        (ox, oy)
    } else {
        (0, 0)
    };

    let mut buf = vec![0.0_f32; out_w.saturating_mul(out_h).saturating_mul(stride)];
    for oy_px in 0..out_h {
        for ox_px in 0..out_w {
            #[allow(
                clippy::cast_possible_wrap,
                reason = "output extents fit i64 for valid plans"
            )]
            let cx = ox_px as i64 + shift_x;
            #[allow(
                clippy::cast_possible_wrap,
                reason = "output extents fit i64 for valid plans"
            )]
            let cy = oy_px as i64 + shift_y;
            let dst_base = (oy_px * out_w + ox_px) * stride;
            for ch in 0..stride {
                let mut acc = 0.0_f64;
                let mut tap = 0usize;
                for ky in 0..kh {
                    let sy = cy + ky - oy;
                    let syi = source_index(sy, src_h, request.mode);
                    for kx in 0..kw {
                        let w = request.kernel.weights[tap];
                        tap += 1;
                        if w == 0.0 {
                            continue;
                        }
                        let sx = cx + kx - ox;
                        let sxi = source_index(sx, src_w, request.mode);
                        let sample = match (sxi, syi) {
                            (Some(x), Some(y)) => {
                                let xi = usize::try_from(x).unwrap_or(0);
                                let yi = usize::try_from(y).unwrap_or(0);
                                let base = (yi * (src.width as usize) + xi) * stride + ch;
                                f64::from(samples[base])
                            }
                            // Out of bounds under constant/transparent.
                            _ => request.value[ch],
                        };
                        acc = w.mul_add(sample, acc);
                    }
                }
                #[allow(
                    clippy::cast_possible_truncation,
                    reason = "reference accumulates in f64 then stores the op's f32 sample type"
                )]
                {
                    buf[dst_base + ch] = acc as f32;
                }
            }
        }
    }
    buf
}

/// The `filter.convolve@1` operation.
#[derive(Debug, Clone, Copy, Default)]
pub struct Convolve;

impl Convolve {
    /// Construct the convolution operation.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    /// The declared manifest for `filter.convolve@1`.
    ///
    /// # Errors
    /// Propagates the [`schema`](ErrorClass::Schema) error if the hard-coded op /
    /// impl ids are somehow invalid (they are not).
    pub fn manifest() -> Result<OperationManifest> {
        Ok(OperationManifest {
            id: CONVOLVE_OP_ID.parse()?,
            impl_version: 1,
            summary:
                "Direct 2-D correlation of an Image or Field1 against a small explicit kernel \
                      under a boundary mode (constant/transparent/clamp/mirror/wrap/valid); the \
                      reference oracle for all optimized convolution paths."
                    .to_owned(),
            determinism: DeterminismTier::Exact,
            roi: RoiPolicy {
                category: RoiCategory::Geometric,
                halo_px: None,
            },
            inputs: vec![InputSpec {
                name: "input".to_owned(),
                kind: ResourceKind::Image,
                required: true,
                doc: "The Image or Field1 to filter (each channel filtered independently)."
                    .to_owned(),
            }],
            outputs: vec![OutputSpec {
                name: "output".to_owned(),
                kind: ResourceKind::Image,
                doc: "The filtered Image or Field1 (same kind as the input; extent shrinks under \
                      `valid`)."
                    .to_owned(),
            }],
            params: vec![
                ParamSpec {
                    name: "kernel".to_owned(),
                    ty: ParamType::Json,
                    unit: None,
                    required: true,
                    default: None,
                    choices: vec![],
                    doc: "The kernel: { width, height, origin_x?, origin_y?, weights[] } with a \
                          row-major weights array of width*height finite numbers; origin defaults \
                          to the geometric centre."
                        .to_owned(),
                },
                ParamSpec {
                    name: "mode".to_owned(),
                    ty: ParamType::String,
                    unit: None,
                    required: false,
                    default: Some(serde_json::json!("clamp")),
                    choices: vec![
                        "constant".to_owned(),
                        "transparent".to_owned(),
                        "clamp".to_owned(),
                        "mirror".to_owned(),
                        "wrap".to_owned(),
                        "valid".to_owned(),
                    ],
                    doc: "How taps that overhang the input edge are resolved.".to_owned(),
                },
                ParamSpec {
                    name: "value".to_owned(),
                    ty: ParamType::Json,
                    unit: None,
                    required: false,
                    default: None,
                    choices: vec![],
                    doc: "Per-channel constant for `constant` mode; defaults to all-zero."
                        .to_owned(),
                },
            ],
            implementations: vec![reference_impl()?],
            test: convolve_test_metadata(),
        })
    }
}

/// Either supported input kind, carrying its channel count and the inferred
/// output descriptor builder.
enum ConvInput<'a> {
    Image(&'a ImageDescriptor),
    Field1(&'a FieldDescriptor),
}

impl ConvInput<'_> {
    /// The interleaved channel count.
    const fn channels(&self) -> u32 {
        match self {
            Self::Image(d) => d.layout.channel_count(),
            Self::Field1(_) => 1,
        }
    }

    /// The source extent.
    const fn extent(&self) -> Extent {
        match self {
            Self::Image(d) => d.extent,
            Self::Field1(d) => d.extent,
        }
    }

    /// The output descriptor for a (possibly shrunk) extent.
    const fn output_descriptor(&self, extent: Extent) -> ResourceDescriptor {
        match self {
            Self::Image(d) => {
                let mut out = **d;
                out.extent = extent;
                ResourceDescriptor::Image(out)
            }
            Self::Field1(d) => {
                let mut out = **d;
                out.extent = extent;
                ResourceDescriptor::Field1(out)
            }
        }
    }
}

/// Extract the supported `input` descriptor (Image or Field1).
fn conv_input(inputs: &Descriptors) -> Result<ConvInput<'_>> {
    let input = inputs.get("input").ok_or_else(|| {
        Error::new(
            ErrorClass::Reference,
            E_CONVOLVE_INPUT,
            "filter.convolve requires an `input` resource".to_owned(),
        )
    })?;
    match input {
        ResourceDescriptor::Image(d) => Ok(ConvInput::Image(d)),
        ResourceDescriptor::Field1(d) => Ok(ConvInput::Field1(d)),
        _ => Err(Error::new(
            ErrorClass::Type,
            E_CONVOLVE_INPUT,
            "filter.convolve `input` must be an Image or Field1 resource".to_owned(),
        )),
    }
}

impl OpContract for Convolve {
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
        let input = conv_input(inputs)?;
        let request = ConvolveRequest::resolve(params, input.channels())?;
        let extent = request.output_extent(input.extent());
        let mut out = OutputDescriptors::new();
        out.insert("output".to_owned(), input.output_descriptor(extent));
        Ok(out)
    }

    fn required_inputs(
        &self,
        requested_outputs: &OutputRegions,
        inputs: &Descriptors,
        params: &serde_json::Value,
    ) -> Result<InputRegions> {
        // A neighbourhood filter: each output sample reads a kernel-sized window.
        // Under the edge-extension modes a border tap can reference an arbitrary
        // edge sample, so demand the whole input plane intersected with the
        // dilated, (for `valid`) shifted request window.
        let input = conv_input(inputs)?;
        let request = ConvolveRequest::resolve(params, input.channels())?;
        let mut regions = InputRegions::new();
        if let Some(region) = requested_outputs.get("output") {
            let src = input.extent();
            let full = Rect::new(0, 0, i64::from(src.width), i64::from(src.height));
            let ox = i64::from(request.kernel.origin_x);
            let oy = i64::from(request.kernel.origin_y);
            let kw = i64::from(request.kernel.width);
            let kh = i64::from(request.kernel.height);
            let (shift_x, shift_y) = if request.mode == BoundaryMode::Valid {
                (ox, oy)
            } else {
                (0, 0)
            };
            // Output pixel x maps to centre (x + shift). The window spans
            // [centre - origin, centre + (k - 1 - origin)].
            let dilated = Rect::new(
                region.x0 + shift_x - ox,
                region.y0 + shift_y - oy,
                region.x1 + shift_x + (kw - 1 - ox),
                region.y1 + shift_y + (kh - 1 - oy),
            );
            regions.insert("input".to_owned(), dilated.intersect(full));
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

impl OpImplementation for Convolve {
    fn compute(
        &self,
        inputs: &InputValues,
        params: &serde_json::Value,
    ) -> std::result::Result<OutputValues, Error> {
        let input = inputs.get("input").ok_or_else(|| {
            Error::new(
                ErrorClass::Reference,
                E_CONVOLVE_INPUT,
                "filter.convolve requires an `input` value".to_owned(),
            )
        })?;
        let (src_extent, out_descriptor_for) = match input.descriptor() {
            ResourceDescriptor::Image(d) => (d.extent, OutKind::Image(*d)),
            ResourceDescriptor::Field1(d) => (d.extent, OutKind::Field1(*d)),
            _ => {
                return Err(Error::new(
                    ErrorClass::Type,
                    E_CONVOLVE_INPUT,
                    "filter.convolve `input` must be an Image or Field1 resource".to_owned(),
                ));
            }
        };
        let channels = input.channels();
        let request = ConvolveRequest::resolve(params, channels)?;
        let out_extent = request.output_extent(src_extent);

        let samples = if src_extent.width == 0 || src_extent.height == 0 || channels == 0 {
            // A zero-area input convolves to a zero-area output; nothing to read.
            Vec::new()
        } else {
            convolve_samples(input.samples(), src_extent, out_extent, &request, channels)
        };

        let descriptor = out_descriptor_for.with_extent(out_extent);
        let value = ResourceValue::new(descriptor, channels, samples).map_err(|actual| {
            Error::new(
                ErrorClass::Execution,
                E_CONVOLVE_INPUT,
                format!("filter.convolve produced a sample buffer of unexpected length {actual}"),
            )
        })?;

        let mut out = OutputValues::new();
        out.insert("output".to_owned(), value);
        Ok(out)
    }
}

/// The output descriptor kind, captured so the kernel can rebuild it with a new
/// extent without re-borrowing the input.
enum OutKind {
    Image(ImageDescriptor),
    Field1(FieldDescriptor),
}

impl OutKind {
    /// Rebuild the descriptor with the output extent.
    const fn with_extent(self, extent: Extent) -> ResourceDescriptor {
        match self {
            Self::Image(mut d) => {
                d.extent = extent;
                ResourceDescriptor::Image(d)
            }
            Self::Field1(mut d) => {
                d.extent = extent;
                ResourceDescriptor::Field1(d)
            }
        }
    }
}

/// The mandatory `cpu.reference@1` oracle implementation id.
fn reference_impl() -> Result<ImplId> {
    ImplId::new("cpu", "reference", 1)
}

/// Verification declarations for `filter.convolve@1`: an exact, single-reference
/// neighbourhood op. Differential does not apply (one implementation); perceptual
/// does not apply (exact tier — correctness is the analytic impulse/linearity/
/// translation property set, not a perceptual metric).
fn convolve_test_metadata() -> TestMetadata {
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
