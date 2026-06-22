//! Analytic and property tests for the deterministic 2-D DFT primitives.

use super::{forward_real, inverse_complex, inverse_real, radial_frequency, signed_frequency};

/// Narrow an `f64` to the op's `f32` sample type for fixture construction.
#[allow(
    clippy::cast_possible_truncation,
    reason = "test fixture: an f64 fixture value narrowed to the op's f32 sample type"
)]
fn to_f32(v: f64) -> f32 {
    v as f32
}

/// A pseudo-random-but-deterministic ramp sample in `[0, 1)` for index `i`.
fn ramp(i: usize, mul: u32, modulo: u32) -> f32 {
    let n = u32::try_from(i).unwrap_or(0).wrapping_mul(mul) % modulo;
    #[allow(
        clippy::cast_precision_loss,
        reason = "test fixture: n < modulo, a small integer exact in f32"
    )]
    let v = n as f32 / modulo as f32;
    v
}

/// A horizontal cosine plane `cos(2π·fx·x / w)`.
fn cosine(w: usize, h: usize, fx: usize) -> Vec<f32> {
    let mut s = vec![0.0_f32; w * h];
    for y in 0..h {
        for x in 0..w {
            #[allow(
                clippy::cast_precision_loss,
                reason = "test fixture: small indices exact in f64"
            )]
            let arg = 2.0 * std::f64::consts::PI * (fx as f64) * (x as f64) / (w as f64);
            s[y * w + x] = to_f32(arg.cos());
        }
    }
    s
}

/// A forward-then-inverse round trip reconstructs a real plane within tolerance,
/// for both a power-of-two and a non-power-of-two extent.
#[test]
fn forward_inverse_round_trips() {
    for (w, h) in [(8usize, 8usize), (12, 12), (6, 10), (16, 4)] {
        let samples: Vec<f32> = (0..w * h).map(|i| ramp(i, 37, 101)).collect();
        let spec = forward_real(&samples, w, h, 1);
        let recon = inverse_real(&spec, w, h, 1);
        assert_eq!(recon.len(), samples.len());
        for (r, s) in recon.iter().zip(samples.iter()) {
            assert!((r - s).abs() < 1e-4, "{w}x{h}: recon {r} vs {s}");
        }
    }
}

/// A constant plane has all its energy in the DC bin and nowhere else.
#[test]
fn constant_plane_is_pure_dc() {
    let (w, h) = (8usize, 8usize);
    let samples = vec![0.5_f32; w * h];
    let spec = forward_real(&samples, w, h, 1);
    // DC bin (index 0) = sum of samples = 0.5 * 64 = 32.
    assert!((f64::from(spec[0]) - 32.0).abs() < 1e-3, "DC {}", spec[0]);
    assert!(f64::from(spec[1]).abs() < 1e-3, "DC imag {}", spec[1]);
    // Every other bin is ~0.
    for k in 1..w * h {
        assert!(f64::from(spec[k * 2]).abs() < 1e-3, "bin {k} re");
        assert!(f64::from(spec[k * 2 + 1]).abs() < 1e-3, "bin {k} im");
    }
}

/// A horizontal sinusoid `cos(2π·fx·x / W)` produces conjugate-symmetric peaks
/// at the `±fx` bins of row 0 and nowhere else.
#[test]
fn sinusoid_has_expected_spectral_peak() {
    let (w, h) = (16usize, 16usize);
    let fx = 3usize;
    let samples = cosine(w, h, fx);
    let spec = forward_real(&samples, w, h, 1);
    // The energy of bin (kx, ky) is |X|^2; find the dominant bins.
    let mut energy = vec![0.0_f64; w * h];
    for (k, e) in energy.iter_mut().enumerate() {
        let re = f64::from(spec[k * 2]);
        let im = f64::from(spec[k * 2 + 1]);
        *e = re.mul_add(re, im * im);
    }
    // Peaks should be at row 0, columns fx and W-fx (each amplitude W*H/2).
    #[allow(
        clippy::cast_precision_loss,
        reason = "test fixture: w*h is a small extent exact in f64"
    )]
    let peak = (w * h) as f64 / 2.0;
    let e_pos = energy[fx];
    let e_neg = energy[w - fx];
    assert!(
        (e_pos.sqrt() - peak).abs() < 1.0,
        "+fx peak {}",
        e_pos.sqrt()
    );
    assert!(
        (e_neg.sqrt() - peak).abs() < 1.0,
        "-fx peak {}",
        e_neg.sqrt()
    );
    // No energy in ky != 0 rows.
    for ky in 1..h {
        for kx in 0..w {
            assert!(energy[ky * w + kx] < 1e-2, "leak at ({kx},{ky})");
        }
    }
}

/// The transform is bit-identical across reruns (determinism criterion).
#[test]
fn transform_is_deterministic() {
    let (w, h) = (12usize, 8usize);
    let samples: Vec<f32> = (0..w * h).map(|i| ramp(i, 1, 13)).collect();
    let a = forward_real(&samples, w, h, 1);
    let b = forward_real(&samples, w, h, 1);
    assert_eq!(a, b, "forward transform not bit-identical");
    let ra = inverse_complex(&a, w, h, 1);
    let rb = inverse_complex(&b, w, h, 1);
    assert_eq!(ra, rb, "inverse transform not bit-identical");
}

/// Multi-channel planes are transformed independently and interleaved.
#[test]
fn channels_are_independent() {
    let (w, h, ch) = (8usize, 8usize, 3usize);
    let samples: Vec<f32> = (0..w * h * ch).map(|i| ramp(i, 1, 7)).collect();
    let spec = forward_real(&samples, w, h, ch);
    let recon = inverse_real(&spec, w, h, ch);
    for (r, s) in recon.iter().zip(samples.iter()) {
        assert!((r - s).abs() < 1e-4, "multi-channel recon {r} vs {s}");
    }
}

/// Signed frequency folds bins above Nyquist to negative frequencies.
#[test]
fn signed_frequency_folds_at_nyquist() {
    assert_eq!(signed_frequency(0, 8), 0);
    assert_eq!(signed_frequency(3, 8), 3);
    assert_eq!(signed_frequency(4, 8), 4); // Nyquist: 2*4 == 8, stays positive
    assert_eq!(signed_frequency(5, 8), -3);
    assert_eq!(signed_frequency(7, 8), -1);
}

/// Radial frequency: DC is 0, the per-axis Nyquist is 0.5.
#[test]
fn radial_frequency_dc_and_nyquist() {
    assert!(radial_frequency(0, 0, 8, 8).abs() < 1e-12);
    assert!((radial_frequency(4, 0, 8, 8) - 0.5).abs() < 1e-12);
    assert!((radial_frequency(0, 4, 8, 8) - 0.5).abs() < 1e-12);
}
