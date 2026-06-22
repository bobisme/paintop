//! Shared multi-resolution pyramid primitives for the `frequency` domain
//! (`OP_CATALOG` §13).
//!
//! `frequency.gaussian_pyramid`, `frequency.laplacian_split`, and
//! `frequency.recombine` all build on the same three deterministic kernels:
//!
//! - [`gaussian_blur_plane`]: a separable, fixed-order `f64` Gaussian smoothing
//!   of a channel-interleaved plane with a clamp (replicate) boundary;
//! - [`downsample`]: a 2:1 decimation that, per the [`PyramidPhase`] rounding
//!   rule, keeps every other sample (the classical pick-even-samples
//!   convention) after a smoothing pre-blur;
//! - [`upsample`]: the dyadic inverse — nearest-parent expansion to a parent
//!   extent, the operation `frequency.recombine` uses to reconstruct a level.
//!
//! Every reduction runs in a single fixed order, so the kernels are
//! bit-identical across runs (the M4 determinism criterion). The smoothing
//! kernel reuses the same `exp(-d²/2σ²)` normalized taps as
//! `filter.gaussian_blur`, so the pyramid's blur is the same analytic Gaussian
//! verified elsewhere.

use paintop_ir::{Extent, PyramidPhase};

/// The fixed Gaussian standard deviation, in base-level pixels, of the
/// pyramid's pre-decimation smoothing kernel.
///
/// A σ of 1.0 gives the classical 5-tap binomial-like low-pass that suppresses
/// the frequencies above the child Nyquist before 2:1 decimation, so the
/// downsample does not alias. It is a fixed convention (not a free parameter)
/// so the pyramid's level chain is a deterministic function of the input alone;
/// `frequency.gaussian_pyramid` exposes it as a documented default that a plan
/// may override.
pub const DEFAULT_PYRAMID_SIGMA: f64 = 1.0;

/// The kernel radius for a smoothing `sigma`: `ceil(3σ)`, at least 1 for any
/// positive sigma, `0` under the sub-pixel cutoff (the identity).
#[must_use]
pub fn blur_radius(sigma: f64) -> u32 {
    if sigma <= 1.0e-3 {
        return 0;
    }
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "3*sigma is positive and bounded by the op's sigma_max, well within u32"
    )]
    let r = (3.0 * sigma).ceil() as u32;
    r.max(1)
}

/// The normalized 1-D Gaussian taps for `sigma`, indexed `[-r, r]` → `[0, 2r]`,
/// and the radius `r`. Identical construction to `filter.gaussian_blur`.
#[must_use]
fn gaussian_taps(sigma: f64) -> (Vec<f64>, u32) {
    let r = blur_radius(sigma);
    if r == 0 {
        return (vec![1.0], 0);
    }
    let ri = i64::from(r);
    let two_sigma_sq = 2.0 * sigma * sigma;
    let mut taps = Vec::with_capacity((2 * r + 1) as usize);
    let mut sum = 0.0_f64;
    for d in -ri..=ri {
        #[allow(
            clippy::cast_precision_loss,
            reason = "d is a small kernel offset bounded by 3*sigma_max"
        )]
        let w = (-((d * d) as f64) / two_sigma_sq).exp();
        sum += w;
        taps.push(w);
    }
    for w in &mut taps {
        *w /= sum;
    }
    (taps, r)
}

/// Replicate (clamp) an out-of-range 1-D coordinate to `[0, n)`.
const fn clamp_index(coord: i64, n: i64) -> i64 {
    if coord < 0 {
        0
    } else if coord >= n {
        n - 1
    } else {
        coord
    }
}

/// Convolve one axis of a `channels`-interleaved plane with `taps` (radius `r`,
/// hot tap at index `r`) under a clamp boundary, accumulating in `f64`.
fn blur_axis(
    src: &[f32],
    width: usize,
    height: usize,
    channels: usize,
    taps: &[f64],
    r: u32,
    horizontal: bool,
) -> Vec<f32> {
    let mut out = vec![0.0_f32; src.len()];
    let ri = i64::from(r);
    let axis_len = i64::from(u32::try_from(if horizontal { width } else { height }).unwrap_or(0));
    for y in 0..height {
        for x in 0..width {
            let base = (y * width + x) * channels;
            let pos = i64::from(u32::try_from(if horizontal { x } else { y }).unwrap_or(0));
            for ch in 0..channels {
                let mut acc = 0.0_f64;
                for (k, &w) in taps.iter().enumerate() {
                    if w == 0.0 {
                        continue;
                    }
                    let coord = pos + (i64::from(u32::try_from(k).unwrap_or(0)) - ri);
                    let idx = usize::try_from(clamp_index(coord, axis_len)).unwrap_or(0);
                    let src_base = if horizontal {
                        (y * width + idx) * channels + ch
                    } else {
                        (idx * width + x) * channels + ch
                    };
                    acc = w.mul_add(f64::from(src[src_base]), acc);
                }
                #[allow(
                    clippy::cast_possible_truncation,
                    reason = "accumulate in f64, store the op's f32 sample type"
                )]
                {
                    out[base + ch] = acc as f32;
                }
            }
        }
    }
    out
}

/// Separable Gaussian smoothing of a `channels`-interleaved plane at `extent`
/// with a clamp boundary; the σ→0 cutoff is the identity.
#[must_use]
pub fn gaussian_blur_plane(samples: &[f32], extent: Extent, channels: u32, sigma: f64) -> Vec<f32> {
    let width = extent.width as usize;
    let height = extent.height as usize;
    let ch = channels as usize;
    if width == 0 || height == 0 || ch == 0 {
        return Vec::new();
    }
    let (taps, r) = gaussian_taps(sigma);
    if r == 0 {
        return samples.to_vec();
    }
    let h = blur_axis(samples, width, height, ch, &taps, r, true);
    blur_axis(&h, width, height, ch, &taps, r, false)
}

/// The child extent of `parent` under one dyadic step of `phase` (the same rule
/// [`PyramidDescriptor::level_extent`](paintop_ir::PyramidDescriptor::level_extent)
/// derives).
#[must_use]
pub const fn child_extent(parent: Extent, phase: PyramidPhase) -> Extent {
    Extent::new(
        phase.child_axis(parent.width, 2),
        phase.child_axis(parent.height, 2),
    )
}

/// Pre-blur `samples` then decimate it 2:1 to the child extent, keeping the
/// even-indexed samples (`src[2*i]`), the classical subsampling phase.
///
/// The pre-blur (σ = `sigma`) is the anti-alias low-pass; the decimation then
/// picks one in every two samples along each axis. With [`PyramidPhase::Floor`]
/// an odd parent axis `n` yields `n/2` children covering indices `0,2,…,2(n/2-1)`;
/// with [`PyramidPhase::Ceil`] it yields `ceil(n/2)` children, the last sourced
/// from the final (clamped) parent column/row.
#[must_use]
pub fn downsample(
    samples: &[f32],
    parent: Extent,
    channels: u32,
    sigma: f64,
    phase: PyramidPhase,
) -> Vec<f32> {
    let blurred = gaussian_blur_plane(samples, parent, channels, sigma);
    let pw = parent.width as usize;
    let ch = channels as usize;
    let child = child_extent(parent, phase);
    let cw = child.width as usize;
    let cheight = child.height as usize;
    let mut out = vec![0.0_f32; cw * cheight * ch];
    let last_parent_x = parent.width.saturating_sub(1) as usize;
    let last_parent_y = parent.height.saturating_sub(1) as usize;
    for cy in 0..cheight {
        let sy = (cy * 2).min(last_parent_y);
        for cx in 0..cw {
            let sx = (cx * 2).min(last_parent_x);
            let src_base = (sy * pw + sx) * ch;
            let dst_base = (cy * cw + cx) * ch;
            out[dst_base..dst_base + ch].copy_from_slice(&blurred[src_base..src_base + ch]);
        }
    }
    out
}

/// Expand `samples` from a child extent up to `parent` by nearest-parent
/// replication: parent sample `(x, y)` reads child `(x/2, y/2)`.
///
/// This is the dyadic inverse of [`downsample`]'s decimation lattice — every
/// parent pixel maps to the child pixel it was decimated toward — so a Laplacian
/// `parent − upsample(child)` residual reconstructs exactly under
/// `child` plus the residual. The expansion never reads out of the child's
/// derived extent.
#[must_use]
pub fn upsample(samples: &[f32], child: Extent, parent: Extent, channels: u32) -> Vec<f32> {
    let cw = child.width as usize;
    let last_child_x = child.width.saturating_sub(1) as usize;
    let last_child_y = child.height.saturating_sub(1) as usize;
    let ch = channels as usize;
    let pw = parent.width as usize;
    let phgt = parent.height as usize;
    let mut out = vec![0.0_f32; pw * phgt * ch];
    for py in 0..phgt {
        let sy = (py / 2).min(last_child_y);
        for px in 0..pw {
            let sx = (px / 2).min(last_child_x);
            let src_base = (sy * cw + sx) * ch;
            let dst_base = (py * pw + px) * ch;
            out[dst_base..dst_base + ch].copy_from_slice(&samples[src_base..src_base + ch]);
        }
    }
    out
}

/// The per-level extents of a dyadic pyramid: level 0 = `base`, each deeper
/// level the [`child_extent`] of the previous under `phase`, for `levels`
/// entries.
#[must_use]
pub fn level_extents(base: Extent, levels: u32, phase: PyramidPhase) -> Vec<Extent> {
    let mut extents = Vec::with_capacity(levels as usize);
    let mut extent = base;
    for l in 0..levels {
        extents.push(extent);
        if l + 1 < levels {
            extent = child_extent(extent, phase);
        }
    }
    extents
}

/// Subtract two equal-length planes elementwise (`a − b`), in a fixed order.
#[must_use]
fn subtract_planes(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| {
            #[allow(
                clippy::cast_possible_truncation,
                reason = "compute the residual in f64, store the op's f32 sample type"
            )]
            let d = (f64::from(x) - f64::from(y)) as f32;
            d
        })
        .collect()
}

/// Add two equal-length planes elementwise (`a + b`), in a fixed order.
#[must_use]
fn add_planes(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| {
            #[allow(
                clippy::cast_possible_truncation,
                reason = "compute the sum in f64, store the op's f32 sample type"
            )]
            let s = (f64::from(x) + f64::from(y)) as f32;
            s
        })
        .collect()
}

/// Split a Gaussian pyramid (its per-level sample planes) into a Laplacian
/// pyramid, returned as the finest-first concatenated band buffer.
///
/// For every level `l < levels − 1` the band is the *high-pass* residual
/// `G_l − upsample(G_{l+1})`; the coarsest level keeps the *low-pass* `G_{l-1}`
/// verbatim, so the pyramid is losslessly invertible. The `extents` are the
/// pyramid's derived per-level extents (from [`level_extents`]); `channels` is
/// constant across levels.
///
/// The split telescopes with [`laplacian_recombine`]: because the coarsest band
/// is the full low-pass and each finer band adds back exactly the gap to its
/// upsampled child, the recombination reconstructs the original level-0 plane up
/// to f32 rounding.
#[must_use]
pub fn laplacian_split(gaussian_levels: &[&[f32]], extents: &[Extent], channels: u32) -> Vec<f32> {
    let levels = extents.len();
    let mut out = Vec::new();
    for l in 0..levels {
        if l + 1 < levels {
            // High-pass band: G_l − upsample(G_{l+1}) at level l's extent.
            let up = upsample(gaussian_levels[l + 1], extents[l + 1], extents[l], channels);
            out.extend(subtract_planes(gaussian_levels[l], &up));
        } else {
            // Coarsest level: the low-pass residual is the Gaussian level itself.
            out.extend_from_slice(gaussian_levels[l]);
        }
    }
    out
}

/// Reconstruct the level-0 plane of an image from a Laplacian pyramid's
/// finest-first concatenated band buffer.
///
/// Starting from the coarsest low-pass band, each finer level is rebuilt as
/// `L_l + upsample(recon_{l+1})`, the exact inverse of [`laplacian_split`]. The
/// returned plane is the full-resolution reconstruction at `extents[0]`.
#[must_use]
pub fn laplacian_recombine(bands: &[&[f32]], extents: &[Extent], channels: u32) -> Vec<f32> {
    let levels = extents.len();
    debug_assert!(levels >= 1, "a pyramid has at least one level");
    // The coarsest band is the low-pass; start the reconstruction there.
    let mut recon = bands[levels - 1].to_vec();
    let mut recon_extent = extents[levels - 1];
    for l in (0..levels - 1).rev() {
        let up = upsample(&recon, recon_extent, extents[l], channels);
        recon = add_planes(bands[l], &up);
        recon_extent = extents[l];
    }
    recon
}

#[cfg(test)]
mod tests {
    use super::{
        child_extent, downsample, gaussian_blur_plane, laplacian_recombine, laplacian_split,
        level_extents, upsample,
    };
    use paintop_ir::{Extent, PyramidPhase};

    /// Build the per-level Gaussian planes of a single-channel base for testing.
    fn gaussian_levels(
        base: &[f32],
        extents: &[Extent],
        sigma: f64,
        phase: PyramidPhase,
    ) -> Vec<Vec<f32>> {
        let mut levels = vec![base.to_vec()];
        for win in extents.windows(2) {
            let prev = levels.last().expect("at least the base level");
            levels.push(downsample(prev, win[0], 1, sigma, phase));
        }
        levels
    }

    #[test]
    fn split_then_recombine_reconstructs_within_tolerance() {
        // A non-trivial gradient-ish base reconstructs to itself through the
        // Laplacian split/recombine telescope (exact up to f32 rounding).
        let extents = level_extents(Extent::new(16, 16), 4, PyramidPhase::Floor);
        let base: Vec<f32> = (0..16u16 * 16)
            .map(|i| f32::from(i.wrapping_mul(53) % 97) / 97.0)
            .collect();
        let levels = gaussian_levels(&base, &extents, 1.0, PyramidPhase::Floor);
        let level_refs: Vec<&[f32]> = levels.iter().map(Vec::as_slice).collect();
        let bands = laplacian_split(&level_refs, &extents, 1);

        // Slice the band buffer back into per-level planes.
        let mut band_refs: Vec<&[f32]> = Vec::new();
        let mut offset = 0usize;
        for e in &extents {
            let n = (e.width * e.height) as usize;
            band_refs.push(&bands[offset..offset + n]);
            offset += n;
        }
        let recon = laplacian_recombine(&band_refs, &extents, 1);
        assert_eq!(recon.len(), base.len());
        for (r, b) in recon.iter().zip(base.iter()) {
            assert!((r - b).abs() < 1e-5, "reconstruction drift {r} vs {b}");
        }
    }

    #[test]
    fn single_level_split_is_the_base_lowpass() {
        // A 1-level pyramid: the only band is the low-pass (= the base), and
        // recombine returns it verbatim.
        let extents = level_extents(Extent::new(4, 3), 1, PyramidPhase::Floor);
        let base: Vec<f32> = (0..12u16).map(|i| f32::from(i) / 12.0).collect();
        let level_refs: Vec<&[f32]> = vec![base.as_slice()];
        let bands = laplacian_split(&level_refs, &extents, 1);
        assert_eq!(bands, base);
        let recon = laplacian_recombine(&[bands.as_slice()], &extents, 1);
        assert_eq!(recon, base);
    }

    #[test]
    fn odd_extent_split_recombine_round_trips() {
        // Odd base under floor and ceil phases both telescope exactly.
        for phase in [PyramidPhase::Floor, PyramidPhase::Ceil] {
            let extents = level_extents(Extent::new(7, 5), 3, phase);
            let base: Vec<f32> = (0..7u16 * 5).map(|i| f32::from(i % 11) / 11.0).collect();
            let levels = gaussian_levels(&base, &extents, 1.0, phase);
            let level_refs: Vec<&[f32]> = levels.iter().map(Vec::as_slice).collect();
            let bands = laplacian_split(&level_refs, &extents, 1);
            let mut band_refs: Vec<&[f32]> = Vec::new();
            let mut offset = 0usize;
            for e in &extents {
                let n = (e.width * e.height) as usize;
                band_refs.push(&bands[offset..offset + n]);
                offset += n;
            }
            let recon = laplacian_recombine(&band_refs, &extents, 1);
            for (r, b) in recon.iter().zip(base.iter()) {
                assert!((r - b).abs() < 1e-5, "phase {phase:?}: {r} vs {b}");
            }
        }
    }

    #[test]
    fn blur_preserves_a_constant_plane() {
        let samples = vec![0.5_f32; 6 * 6];
        let out = gaussian_blur_plane(&samples, Extent::new(6, 6), 1, 1.0);
        for v in out {
            assert!((v - 0.5).abs() < 1e-6, "constant not preserved: {v}");
        }
    }

    #[test]
    fn downsample_halves_even_extents() {
        let samples = vec![1.0_f32; 8 * 8];
        let out = downsample(&samples, Extent::new(8, 8), 1, 1.0, PyramidPhase::Floor);
        assert_eq!(out.len(), 4 * 4);
        // A constant plane stays constant through blur+decimate.
        for v in &out {
            assert!((v - 1.0).abs() < 1e-6, "{v}");
        }
    }

    #[test]
    fn downsample_floor_vs_ceil_on_odd_extent() {
        let samples = vec![0.0_f32; 5 * 5];
        let floor = downsample(&samples, Extent::new(5, 5), 1, 1.0, PyramidPhase::Floor);
        assert_eq!(floor.len(), 2 * 2);
        let ceil = downsample(&samples, Extent::new(5, 5), 1, 1.0, PyramidPhase::Ceil);
        assert_eq!(ceil.len(), 3 * 3);
        assert_eq!(
            child_extent(Extent::new(5, 5), PyramidPhase::Ceil),
            Extent::new(3, 3)
        );
    }

    #[test]
    fn upsample_is_nearest_parent_expansion() {
        // A 2x2 child with distinct values expands to a 4x4 parent block-wise.
        let child = vec![1.0_f32, 2.0, 3.0, 4.0];
        let up = upsample(&child, Extent::new(2, 2), Extent::new(4, 4), 1);
        assert_eq!(up.len(), 16);
        // Row 0 and 1 read child row 0: [1 1 2 2].
        assert_eq!(&up[0..4], &[1.0, 1.0, 2.0, 2.0]);
        assert_eq!(&up[4..8], &[1.0, 1.0, 2.0, 2.0]);
        // Row 2 and 3 read child row 1: [3 3 4 4].
        assert_eq!(&up[8..12], &[3.0, 3.0, 4.0, 4.0]);
    }
}
