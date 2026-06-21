//! Autovectorization-friendly pointwise kernels for the `cpu.optimized` backend
//! (`plan.md` §12.2: "vectorized pointwise kernels", "explicit SIMD where
//! benchmarks prove value").
//!
//! Every kernel here is **safe** Rust (the crate forbids `unsafe`): there are no
//! raw SIMD intrinsics. Instead each kernel is written in the shape the compiler
//! autovectorizes well — a tight, fixed-stride loop over a flat `f32` slice with
//! the inner per-channel arithmetic expressed branch-free wherever the reference
//! semantics allow. With `-C target-feature` / the default `x86-64-v3`-ish codegen
//! the backend emits packed SSE/AVX float ops for these loops; on a target without
//! vector units they degrade to the same scalar arithmetic, never to a different
//! *result*.
//!
//! ## Numeric contract
//!
//! Each kernel reproduces its scalar reference's arithmetic **exactly the same
//! way**: the same operation order, the same fused multiply-adds, the same
//! near-zero-alpha policy. For the [`Exact`](paintop_ir::DeterminismTier::Exact)
//! ops (`alpha.premultiply`, `composite.over`, `composite.blend`) this yields
//! bit-identical output to the oracle; for the
//! [`Bounded`](paintop_ir::DeterminismTier::Bounded) ops (`color.convert`,
//! `color.adjust`, `alpha.unpremultiply`) it stays within the op's declared
//! envelope. The cross-backend differential harness proves both (`bn-2ja`).
//!
//! These functions take and return owned sample buffers so they slot directly
//! behind an [`OpImplementation`](paintop_core::executor::OpImplementation) without
//! the caller reasoning about aliasing.

/// The near-zero-alpha threshold for [`unpremultiply`], identical to the scalar
/// reference's `UNPREMULTIPLY_EPSILON` so the two backends agree pixel-for-pixel
/// on which pixels are clamped to zero.
pub const UNPREMULTIPLY_EPSILON: f32 = 1.0e-6;

/// Rec. 709 linear-luminance weights, identical to the scalar `color.adjust`.
const LUMA_R: f32 = 0.212_6;
const LUMA_G: f32 = 0.715_2;
const LUMA_B: f32 = 0.072_2;

/// The sRGB decode knot (`srgb -> linear`).
const DECODE_KNOT: f32 = 0.040_45;
/// The sRGB encode knot (`linear -> srgb`).
const ENCODE_KNOT: f32 = 0.003_130_8;

/// Premultiply: scale every color channel by the trailing alpha (`C' = C * a`),
/// alpha passing through.
///
/// `stride` is the interleaved sample count per pixel; the alpha is the last
/// channel. The loop is fixed-stride so the compiler can vectorize the per-pixel
/// color multiplies. A `stride` of zero (or a buffer that is not a whole number of
/// pixels) returns the input verbatim, matching the scalar fallback.
#[must_use]
pub fn premultiply(samples: &[f32], stride: usize) -> Vec<f32> {
    if stride == 0 {
        return samples.to_vec();
    }
    let alpha_index = stride - 1;
    let mut out = samples.to_vec();
    for pixel in out.chunks_exact_mut(stride) {
        let alpha = pixel[alpha_index];
        for color in &mut pixel[..alpha_index] {
            *color *= alpha;
        }
    }
    out
}

/// Unpremultiply: divide every color channel by the trailing alpha where it
/// exceeds [`UNPREMULTIPLY_EPSILON`], else clamp the color to zero; alpha passes
/// through.
///
/// The per-channel select is written branch-free (a multiply by a `0.0`/`1.0`
/// gate) so it stays vectorizable: a per-pixel `gate` is computed once, and each
/// color channel becomes `gate * (c / a)`. When `a <= eps` the gate is `0.0`,
/// reproducing the scalar reference's "leave color at zero" policy exactly,
/// including the same division being skipped (the gate multiply annihilates it).
#[must_use]
pub fn unpremultiply(samples: &[f32], stride: usize) -> Vec<f32> {
    if stride == 0 {
        return samples.to_vec();
    }
    let alpha_index = stride - 1;
    let mut out = samples.to_vec();
    for pixel in out.chunks_exact_mut(stride) {
        let alpha = pixel[alpha_index];
        // Recoverable iff alpha is above the epsilon; otherwise the straight color
        // is lost and the reference leaves it at zero.
        let recoverable = alpha > UNPREMULTIPLY_EPSILON;
        // Divide by a guarded alpha (>= eps so it can never be a near-zero / zero
        // divide), exactly as the scalar reference's `c / alpha`, then gate the
        // unrecoverable pixels to zero. Using the same division op (not a
        // reciprocal-multiply) keeps the optimized result bit-identical to the
        // reference for every recoverable pixel.
        let divisor = if recoverable { alpha } else { 1.0 };
        let gate = if recoverable { 1.0 } else { 0.0 };
        for color in &mut pixel[..alpha_index] {
            *color = gate * (*color / divisor);
        }
    }
    out
}

/// Decode one sRGB-encoded sample to linear light (IEC 61966-2-1), identical to
/// the scalar reference.
#[inline]
#[must_use]
fn srgb_decode(c: f32) -> f32 {
    if c <= DECODE_KNOT {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

/// Encode one linear-light sample to the sRGB transfer function, identical to the
/// scalar reference.
#[inline]
#[must_use]
fn srgb_encode(c: f32) -> f32 {
    if c <= ENCODE_KNOT {
        c * 12.92
    } else {
        1.055_f32.mul_add(c.powf(1.0 / 2.4), -0.055)
    }
}

/// The transfer direction a [`color_convert`] applies.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Transfer {
    /// Identity (`from == to`): samples pass through unchanged.
    Identity,
    /// `srgb -> linear-srgb`.
    Decode,
    /// `linear-srgb -> srgb`.
    Encode,
}

/// Apply an sRGB transfer function to the color channels, passing a trailing alpha
/// through.
///
/// `stride` is the per-pixel sample count and `has_alpha` whether the last channel
/// is alpha (skipped). The `powf` transfer is not auto-vectorized (it is a libm
/// call), but the surrounding traversal and the alpha-skip predicate are kept flat
/// so the *non*-transfer work vectorizes and the kernel stays a single tight pass.
/// An [`Transfer::Identity`] conversion clones the input.
#[must_use]
pub fn color_convert(
    samples: &[f32],
    stride: usize,
    has_alpha: bool,
    transfer: Transfer,
) -> Vec<f32> {
    let func: fn(f32) -> f32 = match transfer {
        Transfer::Identity => return samples.to_vec(),
        Transfer::Decode => srgb_decode,
        Transfer::Encode => srgb_encode,
    };
    if stride == 0 {
        return samples.to_vec();
    }
    let alpha_index = if has_alpha { Some(stride - 1) } else { None };
    let mut out = samples.to_vec();
    // The descriptor guarantees the buffer is a whole number of pixels, so
    // `chunks_exact_mut` covers every sample; the alpha-skip predicate keeps the
    // traversal flat and the colour transfer the only per-channel work.
    for pixel in out.chunks_exact_mut(stride) {
        for (idx, sample) in pixel.iter_mut().enumerate() {
            if Some(idx) != alpha_index {
                *sample = func(*sample);
            }
        }
    }
    out
}

/// The resolved, identity-defaulted `color.adjust` sub-parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Adjustment {
    /// Exposure in EV / stops (`x -> x * 2^ev`).
    pub exposure_ev: f32,
    /// Saturation blend toward luminance (`0` identity).
    pub saturation: f32,
    /// Warm/cool channel tilt (`0` identity).
    pub temperature: f32,
}

/// Apply the fixed-order `color.adjust` grade (exposure -> temperature ->
/// saturation), optionally gated by a per-pixel coverage mask.
///
/// Mirrors the scalar reference operation order exactly. `color_count` is the
/// number of leading color channels (`stride` minus any trailing alpha). The
/// per-pixel work is branchless apart from the 1-vs-3 channel split decided once
/// per pixel, keeping the dominant 3-channel path vectorizable across pixels.
#[must_use]
pub fn color_adjust(
    samples: &[f32],
    stride: usize,
    color_count: usize,
    adj: Adjustment,
    mask: Option<&[f32]>,
) -> Vec<f32> {
    if stride == 0 || color_count == 0 {
        return samples.to_vec();
    }
    let gain = adj.exposure_ev.exp2();
    let warm = 1.0 + adj.temperature;
    let cool = 1.0 - adj.temperature;
    let sat = 1.0 + adj.saturation;

    let mut out = samples.to_vec();
    for (pixel_index, pixel) in out.chunks_exact_mut(stride).enumerate() {
        let original = [
            pixel[0],
            if color_count >= 2 { pixel[1] } else { 0.0 },
            if color_count >= 3 { pixel[2] } else { 0.0 },
        ];

        // Exposure on every color channel.
        let mut color = [original[0] * gain, original[1] * gain, original[2] * gain];

        if color_count == 3 {
            // Temperature tilt on red and blue.
            color[0] *= warm;
            color[2] *= cool;
            // Saturation blend toward luminance.
            let luma = LUMA_R.mul_add(color[0], LUMA_G.mul_add(color[1], LUMA_B * color[2]));
            color[0] = sat.mul_add(color[0] - luma, luma);
            color[1] = sat.mul_add(color[1] - luma, luma);
            color[2] = sat.mul_add(color[2] - luma, luma);
        }
        // A single (gray) color channel is its own luminance: temperature and
        // saturation are the identity, so only the exposure gain applies.

        let coverage = mask.map_or(1.0, |m| m.get(pixel_index).copied().unwrap_or(0.0));
        for ch in 0..color_count {
            pixel[ch] = coverage.mul_add(color[ch] - original[ch], original[ch]);
        }
    }
    out
}

/// Composite `src` over `dst` in premultiplied linear light, per channel:
/// `out = c_s + c_d * (1 - a_s)` (the alpha channel uses the same form).
///
/// `stride` is the interleaved color+alpha sample count per pixel. The per-pixel
/// `(1 - a_s)` factor is hoisted once, then the channel loop is a flat fused
/// multiply-add the compiler vectorizes. Bit-identical to the scalar reference's
/// `c_d.mul_add(inv_alpha_s, c_s)`.
#[must_use]
pub fn composite_over(src: &[f32], dst: &[f32], stride: usize) -> Vec<f32> {
    if stride == 0 {
        return dst.to_vec();
    }
    let alpha_index = stride - 1;
    let mut out = vec![0.0_f32; dst.len()];
    for ((src_px, dst_px), out_px) in src
        .chunks_exact(stride)
        .zip(dst.chunks_exact(stride))
        .zip(out.chunks_exact_mut(stride))
    {
        let inv_alpha_s = 1.0 - src_px[alpha_index];
        for ((&c_s, &c_d), o) in src_px.iter().zip(dst_px.iter()).zip(out_px.iter_mut()) {
            *o = c_d.mul_add(inv_alpha_s, c_s);
        }
    }
    out
}

/// The restricted, exactly-pinned blend modes, identical to the scalar reference.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlendMode {
    /// Premultiplied source-over (`s + d*(1 - a_s)`).
    Normal,
    /// `s + d`.
    Add,
    /// `d - s`.
    Subtract,
    /// `s * d`.
    Multiply,
    /// `s + d - s*d`.
    Screen,
    /// `min(s, d)`.
    Darken,
    /// `max(s, d)`.
    Lighten,
    /// `|s - d|`.
    Difference,
}

impl BlendMode {
    /// The per-channel blend value `B(s, d)`, identical fused form to the scalar
    /// reference.
    #[inline]
    #[must_use]
    pub(super) fn blend_channel(self, s: f32, d: f32, inv_alpha_s: f32) -> f32 {
        match self {
            Self::Normal => d.mul_add(inv_alpha_s, s),
            Self::Add => s + d,
            Self::Subtract => d - s,
            Self::Multiply => s * d,
            Self::Screen => s.mul_add(-d, s + d),
            Self::Darken => s.min(d),
            Self::Lighten => s.max(d),
            Self::Difference => (s - d).abs(),
        }
    }
}

/// Blend `src` onto `dst` through `mask` at `opacity` using `mode`.
///
/// The output is `out = dst + k*(B_mode(src, dst) - dst)` with `k = opacity *
/// coverage`, and the `k == 0` pixel passes `dst` through verbatim (the identity
/// guarantee, matched by bit pattern exactly as the scalar reference does).
#[must_use]
pub fn composite_blend(
    src: &[f32],
    dst: &[f32],
    mask: &[f32],
    stride: usize,
    mode: BlendMode,
    opacity: f32,
) -> Vec<f32> {
    if stride == 0 {
        return dst.to_vec();
    }
    let alpha_index = stride - 1;
    let mut out = vec![0.0_f32; dst.len()];
    for (((src_px, dst_px), &coverage), out_px) in src
        .chunks_exact(stride)
        .zip(dst.chunks_exact(stride))
        .zip(mask.iter())
        .zip(out.chunks_exact_mut(stride))
    {
        let k = opacity * coverage;
        let inv_alpha_s = 1.0 - src_px[alpha_index];
        let identity = k.to_bits() == 0.0_f32.to_bits();
        for ((&s, &d), o) in src_px.iter().zip(dst_px.iter()).zip(out_px.iter_mut()) {
            *o = if identity {
                d
            } else {
                let blended = mode.blend_channel(s, d, inv_alpha_s);
                k.mul_add(blended - d, d)
            };
        }
    }
    out
}

#[cfg(test)]
mod tests {
    // Exact-tier kernels are asserted bit-for-bit against hand-computed values, so
    // direct float equality is intentional here.
    #![allow(clippy::float_cmp)]
    use super::{
        Adjustment, BlendMode, Transfer, color_adjust, color_convert, composite_blend,
        composite_over, premultiply, unpremultiply,
    };

    #[test]
    fn premultiply_scales_color_passes_alpha() {
        // One Rgba pixel: color (0.2, 0.4, 0.6), alpha 0.5.
        let out = premultiply(&[0.2, 0.4, 0.6, 0.5], 4);
        assert_eq!(out, vec![0.1, 0.2, 0.3, 0.5]);
    }

    #[test]
    fn unpremultiply_clamps_near_zero_alpha() {
        // Recoverable pixel divides; transparent pixel clamps color to zero.
        let out = unpremultiply(&[0.1, 0.2, 0.3, 0.5, 0.9, 0.9, 0.9, 0.0], 4);
        assert!((out[0] - 0.2).abs() < 1e-6);
        assert!((out[1] - 0.4).abs() < 1e-6);
        assert!((out[2] - 0.6).abs() < 1e-6);
        assert_eq!(out[3], 0.5);
        assert_eq!(&out[4..7], &[0.0, 0.0, 0.0]);
        assert_eq!(out[7], 0.0);
    }

    #[test]
    fn convert_identity_is_passthrough() {
        let s = [0.1, 0.2, 0.3, 0.4];
        assert_eq!(color_convert(&s, 4, true, Transfer::Identity), s.to_vec());
    }

    #[test]
    fn convert_decode_skips_alpha() {
        // alpha (last) must pass through unchanged even though it is > knot.
        let out = color_convert(&[0.5, 0.5, 0.5, 0.5], 4, true, Transfer::Decode);
        assert_eq!(out[3], 0.5, "alpha passes through the transfer");
        assert!(out[0] < 0.5, "color decoded toward linear");
    }

    #[test]
    fn adjust_identity_is_passthrough() {
        let s = [0.2, 0.4, 0.6, 1.0];
        let adj = Adjustment {
            exposure_ev: 0.0,
            saturation: 0.0,
            temperature: 0.0,
        };
        let out = color_adjust(&s, 4, 3, adj, None);
        for (a, b) in out.iter().zip(s.iter()) {
            assert!((a - b).abs() < 1e-6, "{a} vs {b}");
        }
    }

    #[test]
    fn adjust_exposure_doubles_color() {
        let adj = Adjustment {
            exposure_ev: 1.0,
            saturation: 0.0,
            temperature: 0.0,
        };
        // 1 EV = x2; with neutral saturation/temperature the gray-equal pixel keeps
        // ratios so each channel doubles.
        let out = color_adjust(&[0.1, 0.1, 0.1, 1.0], 4, 3, adj, None);
        for c in &out[..3] {
            assert!((c - 0.2).abs() < 1e-5, "{c}");
        }
    }

    #[test]
    fn over_transparent_source_is_identity() {
        // src alpha 0 => out == dst.
        let src = [0.0, 0.0, 0.0, 0.0];
        let dst = [0.3, 0.4, 0.5, 0.6];
        assert_eq!(composite_over(&src, &dst, 4), dst.to_vec());
    }

    #[test]
    fn over_opaque_source_replaces() {
        // src alpha 1 (premultiplied) => out == src.
        let src = [0.3, 0.4, 0.5, 1.0];
        let dst = [0.9, 0.9, 0.9, 1.0];
        assert_eq!(composite_over(&src, &dst, 4), src.to_vec());
    }

    #[test]
    fn blend_zero_opacity_is_identity() {
        let src = [0.3, 0.4, 0.5, 0.6];
        let dst = [0.1, 0.2, 0.3, 0.4];
        let out = composite_blend(&src, &dst, &[1.0], 4, BlendMode::Add, 0.0);
        assert_eq!(out, dst.to_vec());
    }

    #[test]
    fn blend_add_full_is_sum() {
        let src = [0.3, 0.4, 0.5, 0.6];
        let dst = [0.1, 0.2, 0.3, 0.4];
        let out = composite_blend(&src, &dst, &[1.0], 4, BlendMode::Add, 1.0);
        assert_eq!(out, vec![0.4, 0.6, 0.8, 1.0]);
    }
}
