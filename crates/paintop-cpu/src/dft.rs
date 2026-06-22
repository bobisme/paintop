//! Shared deterministic complex-DFT primitives for the `frequency` domain
//! (`OP_CATALOG` §9).
//!
//! `frequency.fft2`, `frequency.ifft2`, and `frequency.bandpass` all build on a
//! single fixed-order, channel-interleaved 2-D discrete Fourier transform of a
//! real (spatial) or complex (spectral) plane.
//!
//! # Conventions (declared, never implicit)
//!
//! - **Normalization**: the forward transform is the *non-normalized* DFT
//!   `X[k] = Σ_n x[n] · exp(-2πi·k·n / N)`; the inverse carries the full
//!   `1/N` scale `x[n] = (1/N) Σ_k X[k] · exp(+2πi·k·n / N)`. A forward-then-
//!   inverse round trip therefore reconstructs the input up to floating-point
//!   rounding, with no extra scale to remember.
//! - **Layout**: the spectrum is *not* `fftshift`-ed — bin `(0, 0)` is the DC
//!   (zero-frequency) term at the array origin, the natural DFT layout.
//! - **2-D order**: every row is transformed first (along `x`), then every
//!   column (along `y`); each 1-D pass runs in a single fixed index order, so
//!   the transform is bit-identical across reruns on a fixed backend (the M4
//!   determinism criterion). Accumulation is in `f64`.
//!
//! # Algorithm
//!
//! A length-`n` 1-D transform uses an iterative radix-2 Cooley–Tukey FFT when
//! `n` is a power of two, and an exact direct `O(n²)` DFT otherwise. Both
//! evaluate the same mathematical transform with the same twiddle convention, so
//! the result is independent of which path a given axis length takes — a
//! `12×12` plane (non-power-of-two) and a `16×16` plane round-trip identically.

use std::f64::consts::PI;

/// A complex sample, real and imaginary parts in `f64` for accumulation.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Complex {
    /// Real part.
    pub re: f64,
    /// Imaginary part.
    pub im: f64,
}

impl Complex {
    /// Construct a complex number from its real and imaginary parts.
    #[must_use]
    pub const fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }

    /// Complex addition.
    #[must_use]
    fn add(self, other: Self) -> Self {
        Self::new(self.re + other.re, self.im + other.im)
    }

    /// Complex subtraction.
    #[must_use]
    fn sub(self, other: Self) -> Self {
        Self::new(self.re - other.re, self.im - other.im)
    }

    /// Complex multiplication.
    #[must_use]
    fn mul(self, other: Self) -> Self {
        Self::new(
            self.re.mul_add(other.re, -(self.im * other.im)),
            self.re.mul_add(other.im, self.im * other.re),
        )
    }

    /// The squared magnitude `re² + im²`.
    #[must_use]
    pub fn norm_sq(self) -> f64 {
        self.re.mul_add(self.re, self.im * self.im)
    }
}

/// The sign of the exponent's argument: `-1` for the forward transform,
/// `+1` for the inverse. Kept as a plain `f64` so the twiddle construction is a
/// single multiply.
type Direction = f64;

/// The forward transform direction (`exp(-2πi …)`).
pub const FORWARD: Direction = -1.0;

/// The inverse transform direction (`exp(+2πi …)`), *without* the `1/N` scale
/// (the caller applies the scale once over the whole 2-D inverse).
pub const INVERSE: Direction = 1.0;

/// In-place iterative radix-2 Cooley–Tukey FFT of `data` (length a power of
/// two), with twiddle sign `dir`. No normalization is applied.
#[allow(
    clippy::many_single_char_names,
    reason = "n/i/j/k/u/v/w are the conventional Cooley-Tukey butterfly variable names"
)]
fn fft_radix2(data: &mut [Complex], dir: Direction) {
    let n = data.len();
    if n <= 1 {
        return;
    }
    // Bit-reversal permutation.
    let mut j = 0usize;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if i < j {
            data.swap(i, j);
        }
    }
    // Butterfly stages.
    let mut len = 2usize;
    while len <= n {
        let half = len / 2;
        #[allow(
            clippy::cast_precision_loss,
            reason = "len is a power-of-two FFT stage size, exactly representable in f64"
        )]
        let theta = dir * 2.0 * PI / (len as f64);
        let wstep = Complex::new(theta.cos(), theta.sin());
        let mut start = 0usize;
        while start < n {
            let mut w = Complex::new(1.0, 0.0);
            for k in 0..half {
                let u = data[start + k];
                let v = data[start + k + half].mul(w);
                data[start + k] = u.add(v);
                data[start + k + half] = u.sub(v);
                w = w.mul(wstep);
            }
            start += len;
        }
        len <<= 1;
    }
}

/// Exact direct `O(n²)` DFT of `src` into `dst`, with twiddle sign `dir`. Used
/// for non-power-of-two lengths; evaluates the same transform as
/// [`fft_radix2`].
fn dft_direct(src: &[Complex], dst: &mut [Complex], dir: Direction) {
    let n = src.len();
    #[allow(
        clippy::cast_precision_loss,
        reason = "n is a small axis length bounded by the op's max-extent policy"
    )]
    let nf = n as f64;
    for (k, out) in dst.iter_mut().enumerate() {
        let mut acc = Complex::new(0.0, 0.0);
        for (m, &s) in src.iter().enumerate() {
            #[allow(
                clippy::cast_precision_loss,
                reason = "k*m is bounded by n^2 with n a small axis length"
            )]
            let angle = dir * 2.0 * PI * ((k * m) as f64) / nf;
            let tw = Complex::new(angle.cos(), angle.sin());
            acc = acc.add(s.mul(tw));
        }
        *out = acc;
    }
}

/// Transform one 1-D line of length `n`, choosing radix-2 when `n` is a power of
/// two and the exact direct DFT otherwise. No normalization is applied.
fn fft_line(line: &mut [Complex], scratch: &mut [Complex], dir: Direction) {
    if line.len().is_power_of_two() {
        fft_radix2(line, dir);
    } else {
        dft_direct(line, scratch, dir);
        line.copy_from_slice(scratch);
    }
}

/// The 2-D transform of one channel plane `data` (`width × height`, row-major),
/// transforming rows then columns with twiddle sign `dir`. No normalization is
/// applied; the caller scales the inverse.
fn fft2_plane(data: &mut [Complex], width: usize, height: usize, dir: Direction) {
    if width == 0 || height == 0 {
        return;
    }
    // Rows (along x).
    let mut scratch = vec![Complex::default(); width.max(height)];
    for y in 0..height {
        let row = &mut data[y * width..(y + 1) * width];
        fft_line(row, &mut scratch[..width], dir);
    }
    // Columns (along y): gather, transform, scatter — a fixed-order pass.
    let mut col = vec![Complex::default(); height];
    for x in 0..width {
        for (y, c) in col.iter_mut().enumerate() {
            *c = data[y * width + x];
        }
        fft_line(&mut col, &mut scratch[..height], dir);
        for (y, &c) in col.iter().enumerate() {
            data[y * width + x] = c;
        }
    }
}

/// The forward 2-D DFT of a *real* `channels`-interleaved spatial plane.
///
/// The result is packed as the interleaved real/imaginary spectrum buffer
/// (`spectrum[2·(idx·channels + c)] = re`, `+1 = im`).
/// `samples` is the row-major, channel-interleaved spatial buffer (length
/// `width·height·channels`); the returned buffer has length
/// `width·height·channels·2`. Each channel is transformed independently.
#[must_use]
pub fn forward_real(samples: &[f32], width: usize, height: usize, channels: usize) -> Vec<f32> {
    let pixels = width * height;
    let mut out = vec![0.0_f32; pixels * channels * 2];
    let mut plane = vec![Complex::default(); pixels];
    for c in 0..channels {
        for (idx, p) in plane.iter_mut().enumerate() {
            *p = Complex::new(f64::from(samples[idx * channels + c]), 0.0);
        }
        fft2_plane(&mut plane, width, height, FORWARD);
        for (idx, p) in plane.iter().enumerate() {
            let base = (idx * channels + c) * 2;
            #[allow(
                clippy::cast_possible_truncation,
                reason = "accumulate in f64, store the op's f32 spectrum component"
            )]
            {
                out[base] = p.re as f32;
                out[base + 1] = p.im as f32;
            }
        }
    }
    out
}

/// The inverse 2-D DFT of a spectrum, returning only the *real* part.
///
/// The reconstructed real samples form a `channels`-interleaved spatial plane
/// (length `width·height·channels`).
/// The full `1/(width·height)` normalization is applied here. The imaginary part
/// of an exact inverse of a forward-real spectrum is zero up to rounding and is
/// discarded; [`inverse_complex`] returns it for callers that need it.
#[must_use]
pub fn inverse_real(spectrum: &[f32], width: usize, height: usize, channels: usize) -> Vec<f32> {
    let complex = inverse_complex(spectrum, width, height, channels);
    let pixels = width * height;
    let mut out = vec![0.0_f32; pixels * channels];
    for idx in 0..pixels * channels {
        out[idx] = complex[idx * 2];
    }
    out
}

/// The inverse 2-D DFT of an interleaved complex spectrum buffer.
///
/// Returns the full complex result as an interleaved real/imaginary buffer
/// (length `width·height·channels·2`), with the `1/(width·height)` scale applied.
#[must_use]
pub fn inverse_complex(spectrum: &[f32], width: usize, height: usize, channels: usize) -> Vec<f32> {
    let pixels = width * height;
    let mut out = vec![0.0_f32; pixels * channels * 2];
    if pixels == 0 || channels == 0 {
        return out;
    }
    #[allow(
        clippy::cast_precision_loss,
        reason = "pixels is bounded by the op's max-extent policy, exact in f64"
    )]
    let scale = 1.0 / (pixels as f64);
    let mut plane = vec![Complex::default(); pixels];
    for c in 0..channels {
        for (idx, p) in plane.iter_mut().enumerate() {
            let base = (idx * channels + c) * 2;
            *p = Complex::new(f64::from(spectrum[base]), f64::from(spectrum[base + 1]));
        }
        fft2_plane(&mut plane, width, height, INVERSE);
        for (idx, p) in plane.iter().enumerate() {
            let base = (idx * channels + c) * 2;
            #[allow(
                clippy::cast_possible_truncation,
                reason = "accumulate in f64, store the op's f32 sample component"
            )]
            {
                out[base] = (p.re * scale) as f32;
                out[base + 1] = (p.im * scale) as f32;
            }
        }
    }
    out
}

/// The cycles-per-axis frequency of spectrum bin index `k` on an axis of length
/// `n`.
///
/// Bins `0..n/2` are non-negative frequencies `k`; bins above the Nyquist fold
/// to negative frequencies `k - n` (the standard DFT aliasing layout).
#[must_use]
pub const fn signed_frequency(k: usize, n: usize) -> i64 {
    #[allow(
        clippy::cast_possible_wrap,
        reason = "k and n are small axis indices bounded by the op's max-extent policy"
    )]
    let (k, n) = (k as i64, n as i64);
    if 2 * k <= n { k } else { k - n }
}

/// The *normalized* radial frequency of bin `(kx, ky)` on a `width × height`
/// grid, in cycles-per-pixel.
///
/// Defined as `sqrt((fx/width)² + (fy/height)²)`, where `fx`/`fy` are the signed
/// bin frequencies. DC is `0.0`; the per-axis Nyquist is `0.5`.
#[must_use]
pub fn radial_frequency(kx: usize, ky: usize, width: usize, height: usize) -> f64 {
    #[allow(
        clippy::cast_precision_loss,
        reason = "extents are bounded by the op's max-extent policy, exact in f64"
    )]
    let (wf, hf) = (width as f64, height as f64);
    #[allow(
        clippy::cast_precision_loss,
        reason = "signed frequencies are bounded by the axis length"
    )]
    let fx = if width == 0 {
        0.0
    } else {
        signed_frequency(kx, width) as f64 / wf
    };
    #[allow(
        clippy::cast_precision_loss,
        reason = "signed frequencies are bounded by the axis length"
    )]
    let fy = if height == 0 {
        0.0
    } else {
        signed_frequency(ky, height) as f64 / hf
    };
    fx.hypot(fy)
}

#[cfg(test)]
mod tests;
