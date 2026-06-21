//! GPU splat-batch resource layout + per-splat support bounds (`plan.md` §12.3;
//! bn-1gs).
//!
//! `paint.gaussian_splats@1` paints an inline batch of anisotropic Gaussian splats
//! onto a premultiplied-linear base (`crates/paintop-cpu/src/splat.rs` is the
//! semantic oracle). To run that batch on the GPU the kernel needs the batch as a
//! flat storage buffer of fixed-stride records, plus, per splat, the conservative
//! axis-aligned support box the CPU reference culls with — so the GPU kernel
//! evaluates exactly the same nonzero-weight pixels in the same array order.
//!
//! This module owns the **host-side** layout contract, kept deliberately `wgpu`-free
//! so it is exhaustively unit-testable on a GPU-less host:
//!
//! * [`GpuSplat`] — the packed per-splat record (`center`, anisotropic `sigma`,
//!   `angle`, straight `color`, `opacity`, `blend` mode, super-Gaussian `exponent`),
//!   mirroring the resolved CPU [`Splat`](../../../paintop_cpu/index.html) one-to-one;
//! * [`GpuBlend`] — the blend-mode enum, with the *same* integer tags the WGSL kernel
//!   branches on (bn-eji);
//! * [`SplatBatchLayout`] — the std430-compatible serialization of a whole batch to a
//!   native-endian `f32`/`u32` byte buffer (no pointer casting; the crate forbids
//!   `unsafe`), plus the dispatch/buffer validation against [`DeviceLimits`];
//! * [`support_box`](GpuSplat::support_box) — the per-splat conservative support box,
//!   reproducing the oracle's [bbox culling] **bit-for-bit** (the same
//!   `K = √1500` σ-multiple, the same ±1px widening, the same canvas clamp), so the
//!   GPU and CPU paths cull identically.
//!
//! # Validation parity with the CPU oracle
//!
//! [`GpuSplat::validate`] rejects exactly what the CPU `Splat::resolve` rejects
//! numerically — a non-finite or non-positive `sigma`, a non-finite center/angle, an
//! out-of-`[0,1]` color/opacity — so a batch that reaches the GPU layout has already
//! passed the oracle's semantic gate. The batch-size bound (`max_splats`) and the
//! base-encoding checks stay in the op contract (`paintop-cpu`); this layer is the
//! *resource* contract: huge, tiny, off-canvas, and rotated anisotropic splats all
//! produce a well-formed, dispatchable record or an explicit
//! [`GpuError::DispatchInvalid`].
//!
//! [bbox culling]: ../../../paintop_cpu/splat/index.html

use paintop_ir::Extent;

use super::error::GpuError;
use super::resource::{DeviceLimits, StorageBufferSpec};

/// The Mahalanobis-distance cutoff (`K²`, in σ²-units) of the conservative support
/// box, identical to the CPU oracle's `SUPPORT_MAHALANOBIS`
/// (`crates/paintop-cpu/src/splat.rs`).
///
/// At `m = K²` the Gaussian weight `opacity·exp(−½·m)` is already exactly `0.0` in
/// `f64`, so the box of the `m = K²` ellipse encloses every nonzero-weight pixel.
/// Pinned identical to the oracle so GPU and CPU cull the *same* pixels.
const SUPPORT_MAHALANOBIS: f64 = 1500.0;

/// `K = √SUPPORT_MAHALANOBIS`, the σ-multiple of the conservative support box —
/// the same constant the CPU oracle uses.
const SUPPORT_K: f64 = 38.729_833_462_074_17; // √1500

/// Pin the GPU support constants to the oracle's so the two backends can never drift
/// apart (the same invariant `paintop-cpu` asserts).
const _: () = {
    assert!(SUPPORT_K * SUPPORT_K >= SUPPORT_MAHALANOBIS - 1e-6);
    assert!(SUPPORT_K * SUPPORT_K <= SUPPORT_MAHALANOBIS + 1e-6);
    assert!(SUPPORT_MAHALANOBIS > 1490.4);
};

/// The number of `f32`/`u32` words in one serialized [`GpuSplat`] record.
///
/// Laid out as a flat run of 4-byte words so the WGSL kernel reads it as a
/// `array<f32>` / `array<u32>` storage buffer with a fixed stride:
///
/// | words | field            | type        |
/// |-------|------------------|-------------|
/// | 0..2  | `center` (x, y)  | `f32`×2     |
/// | 2..4  | `sigma` (x, y)   | `f32`×2     |
/// | 4     | `angle_rad`      | `f32`       |
/// | 5..9  | `color` (r,g,b,a)| `f32`×4     |
/// | 9     | `opacity`        | `f32`       |
/// | 10    | `exponent`       | `f32`       |
/// | 11    | `blend`          | `u32` (tag) |
///
/// 12 words = 48 bytes, a multiple of 16 so it satisfies std430's 16-byte structure
/// alignment with no tail padding.
pub const SPLAT_WORDS: usize = 12;

/// The byte stride of one serialized splat record (`SPLAT_WORDS * 4`).
pub const SPLAT_STRIDE_BYTES: usize = SPLAT_WORDS * 4;

/// The blend mode a splat composites with, tagged identically on the host and in the
/// WGSL kernel (`crates/paintop-cpu/src/splat.rs` `BlendMode`).
///
/// The integer tag is the wire contract between the host layout and the GPU shader:
/// the kernel branches on this `u32`, so the host and shader must agree on every
/// value. The order matches the CPU `BlendMode` declaration order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum GpuBlend {
    /// Source-over: the splat composites over the running result.
    Normal = 0,
    /// Multiply: the splat modulates the running result toward its color.
    Multiply = 1,
    /// Additive: `s + d` per channel (premultiplied).
    Add = 2,
    /// Screen: `s + d − s·d` per channel (premultiplied).
    Screen = 3,
    /// Lighten: `max(s, d)` per channel (premultiplied).
    Lighten = 4,
}

impl GpuBlend {
    /// The `u32` tag the WGSL kernel branches on.
    #[must_use]
    pub const fn tag(self) -> u32 {
        self as u32
    }

    /// The blend-mode token as it appears in JSON (matching the CPU op).
    #[must_use]
    pub const fn token(self) -> &'static str {
        match self {
            Self::Normal => "normal",
            Self::Multiply => "multiply",
            Self::Add => "add",
            Self::Screen => "screen",
            Self::Lighten => "lighten",
        }
    }

    /// Parse a blend-mode token, defaulting to [`Normal`](Self::Normal) when `None`.
    ///
    /// # Errors
    /// [`GpuError::DispatchInvalid`] if the token is not one of the five supported
    /// modes — the same vocabulary the CPU op accepts.
    pub fn parse(token: Option<&str>) -> Result<Self, GpuError> {
        match token {
            None | Some("normal") => Ok(Self::Normal),
            Some("multiply") => Ok(Self::Multiply),
            Some("add") => Ok(Self::Add),
            Some("screen") => Ok(Self::Screen),
            Some("lighten") => Ok(Self::Lighten),
            Some(other) => Err(GpuError::DispatchInvalid {
                reason: format!(
                    "splat blend mode `{other}` is not one of normal|multiply|add|screen|lighten"
                ),
            }),
        }
    }
}

/// One resolved Gaussian splat in the GPU layout, in physical pixels (angle in
/// radians, color/opacity in `[0, 1]`).
///
/// A one-to-one mirror of the CPU oracle's resolved `Splat`
/// (`crates/paintop-cpu/src/splat.rs`): the same geometry, the same super-Gaussian
/// `exponent` (`p = 1 + hardness·7`), the same blend tag. Held as `f32` because the
/// GPU kernel computes in `f32`; the host validation runs in `f64`/`f32` exactly as
/// the oracle parses, so a record that validates here is one the oracle accepts.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GpuSplat {
    /// Center `(x, y)` in pixel coordinates.
    pub center: [f32; 2],
    /// Anisotropic standard deviations `(σx, σy)`, strictly positive.
    pub sigma: [f32; 2],
    /// Rotation of the covariance axes, radians (counter-clockwise).
    pub angle_rad: f32,
    /// Straight color `[r, g, b, a]`, each in `[0, 1]`.
    pub color: [f32; 4],
    /// Scalar opacity in `[0, 1]` modulating the Gaussian weight.
    pub opacity: f32,
    /// The super-Gaussian falloff exponent `p ≥ 1` (`1.0` is the pure Gaussian).
    pub exponent: f32,
    /// How this splat composites with the running result.
    pub blend: GpuBlend,
}

impl GpuSplat {
    /// Validate this record against the oracle's numeric admissibility rules.
    ///
    /// Rejects exactly what the CPU `Splat::resolve` rejects per element: a
    /// non-finite center/angle, a non-finite or non-positive `sigma`, a non-finite or
    /// out-of-`[0,1]` color/opacity, a non-finite or `< 1` `exponent`. (The batch
    /// budget and base-encoding checks stay in the op contract.)
    ///
    /// # Errors
    /// [`GpuError::DispatchInvalid`] describing the first offending field, so a
    /// degenerate splat fails explicitly before any GPU work rather than producing a
    /// singular covariance / out-of-range color on the device.
    pub fn validate(&self) -> Result<(), GpuError> {
        let bad = |reason: String| GpuError::DispatchInvalid { reason };
        for (name, v) in [("center.x", self.center[0]), ("center.y", self.center[1])] {
            if !v.is_finite() {
                return Err(bad(format!("splat {name} must be finite, got {v}")));
            }
        }
        if !self.angle_rad.is_finite() {
            return Err(bad(format!(
                "splat angle_rad must be finite, got {}",
                self.angle_rad
            )));
        }
        for (name, s) in [("sigma.x", self.sigma[0]), ("sigma.y", self.sigma[1])] {
            if !(s.is_finite() && s > 0.0) {
                return Err(bad(format!(
                    "splat {name} must be a strictly positive sigma, got {s}"
                )));
            }
        }
        for (i, c) in self.color.iter().enumerate() {
            if !(c.is_finite() && (0.0..=1.0).contains(c)) {
                return Err(bad(format!("splat color[{i}] must be in [0, 1], got {c}")));
            }
        }
        if !(self.opacity.is_finite() && (0.0..=1.0).contains(&self.opacity)) {
            return Err(bad(format!(
                "splat opacity must be in [0, 1], got {}",
                self.opacity
            )));
        }
        if !(self.exponent.is_finite() && self.exponent >= 1.0) {
            return Err(bad(format!(
                "splat exponent must be a finite value >= 1, got {}",
                self.exponent
            )));
        }
        Ok(())
    }

    /// The axis-aligned half-extents `(hx, hy)` (in pixels) of the conservative
    /// support box, computed exactly as the CPU oracle's `support_half_extents`.
    ///
    /// For the rotated anisotropic precision the level set `m = K²` is an ellipse
    /// whose bounding box has half-extents `K·√Σxx` and `K·√Σyy`, with
    /// `Σxx = (σx·cosθ)² + (σy·sinθ)²` and `Σyy = (σx·sinθ)² + (σy·cosθ)²`. Computed
    /// in `f64` (the same precision as the oracle) so the box matches bit-for-bit.
    #[must_use]
    pub fn support_half_extents(&self) -> (f64, f64) {
        let angle = f64::from(self.angle_rad);
        let (sin, cos) = angle.sin_cos();
        let sx = f64::from(self.sigma[0]);
        let sy = f64::from(self.sigma[1]);
        let cs = cos * sx;
        let sc = sin * sy;
        let ss = sin * sx;
        let cc = cos * sy;
        let var_xx = cs.mul_add(cs, sc * sc);
        let var_yy = ss.mul_add(ss, cc * cc);
        (SUPPORT_K * var_xx.sqrt(), SUPPORT_K * var_yy.sqrt())
    }

    /// The conservative half-open pixel bounding box `[x0, x1) × [y0, y1)` of this
    /// splat's support, clamped to a `width × height` canvas.
    ///
    /// Reproduces the oracle's `support_box` exactly: the support spans continuous
    /// `[c − h, c + h]`, widened by one pixel each side for the −0.5 sample offset and
    /// f64 rounding, then `floor`/`ceil`-clamped into `[0, limit]`. An entirely
    /// off-canvas support yields an empty box (`x0 == x1` or `y0 == y1`) — the splat
    /// then contributes nothing, exactly as on the CPU. A non-finite center/half
    /// (which `validate` rejects) conservatively yields the full axis.
    #[must_use]
    pub fn support_box(&self, width: u32, height: u32) -> (u32, u32, u32, u32) {
        let (hx, hy) = self.support_half_extents();
        let span = |center: f64, half: f64, limit: u32| -> (u32, u32) {
            if !(center.is_finite() && half.is_finite()) {
                return (0, limit);
            }
            let lo = center - half - 1.0;
            let hi = center + half + 1.0;
            let limit_f = f64::from(limit);
            #[allow(
                clippy::cast_possible_truncation,
                clippy::cast_sign_loss,
                reason = "bounds are clamped into [0, limit], a small pixel count"
            )]
            {
                let lo_px = lo.floor().clamp(0.0, limit_f) as u32;
                let hi_px = (hi.ceil().clamp(0.0, limit_f) as u32).max(lo_px);
                (lo_px, hi_px)
            }
        };
        let (x0, x1) = span(f64::from(self.center[0]), hx, width);
        let (y0, y1) = span(f64::from(self.center[1]), hy, height);
        (x0, y0, x1, y1)
    }

    /// Serialize this record into a `[f32/u32; SPLAT_WORDS]` word array, in the layout
    /// the WGSL kernel reads (see [`SPLAT_WORDS`]).
    ///
    /// The blend tag is stored as a `u32` reinterpreted into the `f32` slot via
    /// `to_bits`/`from_bits` round-trip on the GPU side; here it is kept as raw bits
    /// so the host serializer can write it with `to_ne_bytes` like every other word.
    #[must_use]
    const fn to_words(self) -> [u32; SPLAT_WORDS] {
        [
            self.center[0].to_bits(),
            self.center[1].to_bits(),
            self.sigma[0].to_bits(),
            self.sigma[1].to_bits(),
            self.angle_rad.to_bits(),
            self.color[0].to_bits(),
            self.color[1].to_bits(),
            self.color[2].to_bits(),
            self.color[3].to_bits(),
            self.opacity.to_bits(),
            self.exponent.to_bits(),
            self.blend.tag(),
        ]
    }
}

/// The serialized layout of a whole splat batch for GPU upload.
///
/// Carries the flat native-endian byte buffer (each splat is [`SPLAT_STRIDE_BYTES`]
/// of `f32`/`u32` words) plus the splat count, ready to bind as a `read` storage
/// buffer. Built via [`build`](Self::build), which validates every splat and the
/// total buffer size against the device limits *before* any GPU submission.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SplatBatchLayout {
    /// The number of splats in the batch.
    count: u32,
    /// The flat native-endian byte buffer (count × [`SPLAT_STRIDE_BYTES`]).
    bytes: Vec<u8>,
}

impl SplatBatchLayout {
    /// Build and validate the GPU layout for a resolved splat batch.
    ///
    /// Validates each splat ([`GpuSplat::validate`]) and the total storage-buffer size
    /// against `limits`. An **empty** batch is legal (it is the identity, exactly as
    /// the CPU op) and yields a zero-count, empty-bytes layout; the caller short-
    /// circuits to the base passthrough rather than dispatching an empty buffer.
    ///
    /// # Errors
    /// [`GpuError::DispatchInvalid`] if a splat is degenerate, the splat count
    /// overflows `u32`, or the batch buffer exceeds the device's storage-binding /
    /// buffer limits or overflows.
    pub fn build(splats: &[GpuSplat], limits: &DeviceLimits) -> Result<Self, GpuError> {
        let count = u32::try_from(splats.len()).map_err(|_| GpuError::DispatchInvalid {
            reason: format!("splat batch of {} exceeds u32 count", splats.len()),
        })?;
        if splats.is_empty() {
            return Ok(Self {
                count: 0,
                bytes: Vec::new(),
            });
        }
        // The flat f32/u32 element count of the buffer, overflow-checked, then
        // validated against the device's storage limits via the resource model.
        let words =
            splats
                .len()
                .checked_mul(SPLAT_WORDS)
                .ok_or_else(|| GpuError::DispatchInvalid {
                    reason: format!(
                        "splat batch of {} * {SPLAT_WORDS} words overflows usize",
                        splats.len()
                    ),
                })?;
        StorageBufferSpec::new(words as u64).validate(limits)?;

        let mut bytes = Vec::with_capacity(words * 4);
        for splat in splats {
            splat.validate()?;
            for word in splat.to_words() {
                bytes.extend_from_slice(&word.to_ne_bytes());
            }
        }
        Ok(Self { count, bytes })
    }

    /// The number of splats in the batch.
    #[must_use]
    pub const fn count(&self) -> u32 {
        self.count
    }

    /// Whether the batch is empty (the identity passthrough).
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// The flat native-endian byte buffer, ready to bind as a `read` storage buffer.
    #[must_use]
    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }
}

/// The base-image resource layout a splat batch paints onto: an RGBA `f32` raster.
///
/// `paint.gaussian_splats` paints in premultiplied-linear RGBA (the op contract
/// requires a 4-channel alpha image), so the GPU base/output buffers are always
/// `width × height × 4` `f32`. This validates that shape against the device limits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SplatTargetLayout {
    /// The canvas extent.
    pub extent: Extent,
}

impl SplatTargetLayout {
    /// A target layout for a canvas extent.
    #[must_use]
    pub const fn new(extent: Extent) -> Self {
        Self { extent }
    }

    /// The number of `f32` samples in the base/output buffer (`w·h·4`),
    /// overflow-checked.
    ///
    /// # Errors
    /// [`GpuError::DispatchInvalid`] if `w·h·4` overflows `u64`.
    pub fn sample_count(&self) -> Result<u64, GpuError> {
        u64::from(self.extent.width)
            .checked_mul(u64::from(self.extent.height))
            .and_then(|n| n.checked_mul(4))
            .ok_or_else(|| GpuError::DispatchInvalid {
                reason: format!(
                    "splat target {}x{} * 4 channels overflows u64",
                    self.extent.width, self.extent.height
                ),
            })
    }

    /// Validate the base/output storage buffers against `limits`.
    ///
    /// # Errors
    /// [`GpuError::DispatchInvalid`] if the extent is zero in either axis, or the
    /// `w·h·4` sample buffer exceeds the device's storage limits or overflows.
    pub fn validate(&self, limits: &DeviceLimits) -> Result<(), GpuError> {
        if self.extent.width == 0 || self.extent.height == 0 {
            return Err(GpuError::DispatchInvalid {
                reason: format!(
                    "zero-sized splat target {}x{}",
                    self.extent.width, self.extent.height
                ),
            });
        }
        let samples = self.sample_count()?;
        StorageBufferSpec::new(samples).validate(limits)
    }
}

#[cfg(test)]
mod tests {
    use super::{GpuBlend, GpuSplat, SPLAT_STRIDE_BYTES, SPLAT_WORDS, SplatBatchLayout};
    use crate::gpu::resource::DeviceLimits;
    use paintop_ir::Extent;

    const LIMITS: DeviceLimits = DeviceLimits {
        max_texture_dimension_2d: 8192,
        max_storage_buffer_binding_size: 128 << 20,
        max_buffer_size: 256 << 20,
        max_workgroup_size_x: 256,
        max_workgroup_size_y: 256,
        max_workgroup_size_z: 64,
        max_invocations_per_workgroup: 256,
        max_workgroups_per_dimension: 65535,
    };

    fn splat() -> GpuSplat {
        GpuSplat {
            center: [10.0, 12.0],
            sigma: [3.0, 2.0],
            angle_rad: 0.0,
            color: [1.0, 0.5, 0.25, 0.8],
            opacity: 0.9,
            exponent: 1.0,
            blend: GpuBlend::Normal,
        }
    }

    #[test]
    fn blend_tags_match_cpu_declaration_order() {
        assert_eq!(GpuBlend::Normal.tag(), 0);
        assert_eq!(GpuBlend::Multiply.tag(), 1);
        assert_eq!(GpuBlend::Add.tag(), 2);
        assert_eq!(GpuBlend::Screen.tag(), 3);
        assert_eq!(GpuBlend::Lighten.tag(), 4);
    }

    #[test]
    fn blend_parse_matches_op_vocabulary() {
        assert_eq!(GpuBlend::parse(None).unwrap(), GpuBlend::Normal);
        assert_eq!(GpuBlend::parse(Some("normal")).unwrap(), GpuBlend::Normal);
        assert_eq!(GpuBlend::parse(Some("lighten")).unwrap(), GpuBlend::Lighten);
        assert!(GpuBlend::parse(Some("overlay")).is_err());
    }

    #[test]
    fn valid_splat_passes_and_serializes_to_stride() {
        let layout = SplatBatchLayout::build(&[splat()], &LIMITS).expect("layout");
        assert_eq!(layout.count(), 1);
        assert_eq!(layout.bytes().len(), SPLAT_STRIDE_BYTES);
        assert_eq!(SPLAT_STRIDE_BYTES, SPLAT_WORDS * 4);
        // 48 bytes is a multiple of 16 (std430 structure alignment).
        assert_eq!(SPLAT_STRIDE_BYTES % 16, 0);
    }

    #[test]
    fn empty_batch_is_the_identity_layout() {
        let layout = SplatBatchLayout::build(&[], &LIMITS).expect("empty");
        assert_eq!(layout.count(), 0);
        assert!(layout.is_empty());
        assert!(layout.bytes().is_empty());
    }

    #[test]
    fn degenerate_sigma_is_rejected() {
        let mut s = splat();
        s.sigma = [0.0, 2.0];
        assert!(SplatBatchLayout::build(&[s], &LIMITS).is_err());
        s.sigma = [-1.0, 2.0];
        assert!(SplatBatchLayout::build(&[s], &LIMITS).is_err());
        s.sigma = [f32::NAN, 2.0];
        assert!(SplatBatchLayout::build(&[s], &LIMITS).is_err());
    }

    #[test]
    fn out_of_range_color_and_opacity_are_rejected() {
        let mut s = splat();
        s.color = [1.5, 0.0, 0.0, 1.0];
        assert!(SplatBatchLayout::build(&[s], &LIMITS).is_err());
        let mut s = splat();
        s.opacity = -0.1;
        assert!(SplatBatchLayout::build(&[s], &LIMITS).is_err());
    }

    #[test]
    fn non_finite_center_or_angle_is_rejected() {
        let mut s = splat();
        s.center = [f32::INFINITY, 0.0];
        assert!(SplatBatchLayout::build(&[s], &LIMITS).is_err());
        let mut s = splat();
        s.angle_rad = f32::NAN;
        assert!(SplatBatchLayout::build(&[s], &LIMITS).is_err());
    }

    #[test]
    fn exponent_below_one_is_rejected() {
        let mut s = splat();
        s.exponent = 0.5;
        assert!(SplatBatchLayout::build(&[s], &LIMITS).is_err());
    }

    #[test]
    fn tiny_splat_has_a_nonempty_clamped_support_box() {
        // A tiny isotropic splat near the origin: the box is small but nonempty and
        // clamped at 0 on the lower edge.
        let mut s = splat();
        s.center = [1.0, 1.0];
        s.sigma = [0.5, 0.5];
        let (x0, y0, x1, y1) = s.support_box(64, 64);
        assert!(x1 > x0 && y1 > y0, "tiny splat still covers ≥1 pixel");
        assert_eq!(x0, 0, "support clamps to the canvas origin");
        assert_eq!(y0, 0);
    }

    #[test]
    fn huge_splat_support_box_clamps_to_canvas() {
        // A huge sigma's support exceeds the canvas; the box clamps to [0, w)×[0, h).
        let mut s = splat();
        s.center = [32.0, 32.0];
        s.sigma = [50.0, 50.0];
        let (x0, y0, x1, y1) = s.support_box(64, 64);
        assert_eq!((x0, y0, x1, y1), (0, 0, 64, 64));
    }

    #[test]
    fn off_canvas_splat_has_an_empty_support_box() {
        // A splat far off the right edge: its support never intersects the canvas.
        let mut s = splat();
        s.center = [1000.0, 32.0];
        s.sigma = [1.0, 1.0];
        let (x0, _y0, x1, _y1) = s.support_box(64, 64);
        assert_eq!(x0, 64, "off-canvas support clamps to the far edge");
        assert_eq!(x1, 64);
        assert_eq!(x0, x1, "an empty box: nothing to paint");
    }

    #[test]
    fn rotated_anisotropic_support_box_is_wider_on_the_major_axis() {
        // An anisotropic splat at 0 rad has hx≈K·σx, hy≈K·σy; rotating 90° swaps the
        // axis-aligned extents. The box geometry follows the rotated covariance.
        let mut s = splat();
        s.center = [200.0, 200.0];
        s.sigma = [6.0, 2.0];
        s.angle_rad = 0.0;
        let (x0, y0, x1, y1) = s.support_box(400, 400);
        let width_unrotated = x1 - x0;
        let height_unrotated = y1 - y0;
        assert!(
            width_unrotated > height_unrotated,
            "wider along x for σx > σy at 0 rad"
        );

        s.angle_rad = std::f32::consts::FRAC_PI_2;
        let (x0, y0, x1, y1) = s.support_box(400, 400);
        let width_rotated = x1 - x0;
        let height_rotated = y1 - y0;
        assert!(
            height_rotated > width_rotated,
            "a quarter turn swaps the major axis to y"
        );
    }

    #[test]
    fn oversized_batch_buffer_is_rejected() {
        // A binding-size-tight device rejects a batch whose buffer exceeds the limit.
        let tight = DeviceLimits {
            max_storage_buffer_binding_size: SPLAT_STRIDE_BYTES as u64, // room for 1 splat
            ..LIMITS
        };
        let batch = vec![splat(), splat()];
        assert!(SplatBatchLayout::build(&batch, &tight).is_err());
        // One splat still fits.
        assert!(SplatBatchLayout::build(&[splat()], &tight).is_ok());
    }

    #[test]
    fn target_layout_validates_extent_and_rejects_zero() {
        use super::SplatTargetLayout;
        let t = SplatTargetLayout::new(Extent::new(64, 64));
        assert_eq!(t.sample_count().expect("sample count"), 64 * 64 * 4);
        assert!(t.validate(&LIMITS).is_ok());
        let zero = SplatTargetLayout::new(Extent::new(0, 16));
        assert!(zero.validate(&LIMITS).is_err());
    }
}
