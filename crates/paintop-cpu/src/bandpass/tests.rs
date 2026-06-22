//! Tests for `frequency.bandpass@1`: the declared-band attenuation, DC
//! preservation, full-band identity, response semantics, and determinism.

use super::{BANDPASS_OP_ID, BandParams, Bandpass, Mode, Window, apply_band};
use paintop_core::executor::{InputValues, OpImplementation, ResourceValue};
use paintop_ir::{
    AlphaRepresentation, ChannelLayout, ColorEncoding, ColorRange, CoordinateConvention, Extent,
    ImageDescriptor, OperationManifest, ResourceDescriptor, ScalarType, SemanticRole,
};

/// Narrow an `f64` to the op's `f32` sample type for fixture construction.
#[allow(
    clippy::cast_possible_truncation,
    reason = "test fixture: an f64 fixture value narrowed to the op's f32 sample type"
)]
fn to_f32(v: f64) -> f32 {
    v as f32
}

fn image_value(extent: Extent, samples: Vec<f32>) -> ResourceValue {
    let d = ResourceDescriptor::Image(ImageDescriptor {
        extent,
        layout: ChannelLayout::Gray,
        scalar: ScalarType::F32,
        color: ColorEncoding::LinearSrgb,
        range: ColorRange::SceneReferred,
        alpha: AlphaRepresentation::Premultiplied,
        coordinates: CoordinateConvention::PixelCenterUpperLeft,
        semantic: SemanticRole::Color,
    });
    ResourceValue::new(d, 1, samples).unwrap()
}

fn run(input: ResourceValue, params: &serde_json::Value) -> ResourceValue {
    let mut inputs = InputValues::new();
    inputs.insert("input".to_owned(), input);
    let out = Bandpass::new().compute(&inputs, params).unwrap();
    out.get("output").unwrap().clone()
}

/// Build a horizontal cosine plane `cos(2π·fx·x / w)`.
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

/// The sum of squared samples (energy) of a plane.
fn energy(s: &[f32]) -> f64 {
    s.iter().map(|&v| f64::from(v) * f64::from(v)).sum()
}

#[test]
fn manifest_validates_and_matches_id() {
    let m: OperationManifest = Bandpass::manifest().unwrap();
    m.validate().unwrap();
    assert_eq!(m.id.to_string(), BANDPASS_OP_ID);
}

#[test]
fn band_stop_attenuates_the_declared_band() {
    // A cosine at fx=4 on a 16-wide plane has frequency 4/16 = 0.25 cyc/px.
    // A band-stop covering [0.2, 0.3] must kill almost all of its energy.
    let samples = cosine(16, 16, 4);
    let before = energy(&samples);
    let out = run(
        image_value(Extent::new(16, 16), samples),
        &serde_json::json!({"low": 0.2, "high": 0.3, "mode": "band-stop"}),
    );
    let after = energy(out.samples());
    assert!(
        after < before * 1e-2,
        "band-stop left {after} of {before} energy"
    );
}

#[test]
fn band_pass_keeps_the_declared_band() {
    // The complementary band-pass over [0.2, 0.3] preserves the fx=4 cosine.
    let samples = cosine(16, 16, 4);
    let before = energy(&samples);
    let out = run(
        image_value(Extent::new(16, 16), samples),
        &serde_json::json!({"low": 0.2, "high": 0.3, "mode": "band-pass"}),
    );
    let after = energy(out.samples());
    assert!(
        (after - before).abs() < before * 1e-2,
        "pass {after} vs {before}"
    );
}

#[test]
fn low_pass_preserves_the_dc_mean() {
    // A constant-plus-high-frequency plane low-passed to [0, 0.05] keeps the
    // mean (DC at f=0 admitted) and removes the high-frequency ripple.
    let mut samples = vec![0.5_f32; 16 * 16];
    let ripple = cosine(16, 16, 6); // 6/16 = 0.375 cyc/px, well above the cutoff
    for (s, r) in samples.iter_mut().zip(ripple.iter()) {
        *s += 0.3 * r;
    }
    let out = run(
        image_value(Extent::new(16, 16), samples),
        &serde_json::json!({"low": 0.0, "high": 0.05, "mode": "band-pass"}),
    );
    // The low-passed result is ~constant 0.5 everywhere.
    for &v in out.samples() {
        assert!((f64::from(v) - 0.5).abs() < 1e-2, "low-pass residual {v}");
    }
}

#[test]
fn full_band_is_the_identity() {
    // A band-pass covering the whole spectrum reconstructs the input (it is
    // fft2 -> *1 -> ifft2). sqrt(2)/2 ~ 0.7071 is the maximum radial frequency.
    let samples: Vec<f32> = (0..64usize)
        .map(|i| {
            let n = u32::try_from(i).unwrap_or(0).wrapping_mul(29) % 53;
            #[allow(
                clippy::cast_precision_loss,
                reason = "test fixture: n < 53, exact in f32"
            )]
            let v = n as f32 / 53.0;
            v
        })
        .collect();
    let out = run(
        image_value(Extent::new(8, 8), samples.clone()),
        &serde_json::json!({"low": 0.0, "high": std::f64::consts::SQRT_2 * 0.5, "mode": "band-pass"}),
    );
    for (o, s) in out.samples().iter().zip(samples.iter()) {
        assert!((o - s).abs() < 1e-4, "identity {o} vs {s}");
    }
}

#[test]
fn ideal_response_is_a_brick_wall() {
    let band = BandParams {
        low: 0.1,
        high: 0.3,
        window: Window::Ideal,
        mode: Mode::Pass,
    };
    assert!(band.response(0.05).abs() < 1e-12);
    assert!((band.response(0.2) - 1.0).abs() < 1e-12);
    assert!(band.response(0.4).abs() < 1e-12);
    // The complement (band-stop) is 1 - response.
    let stop = BandParams {
        mode: Mode::Stop,
        ..band
    };
    assert!(stop.response(0.2).abs() < 1e-12);
    assert!((stop.response(0.05) - 1.0).abs() < 1e-12);
}

#[test]
fn gaussian_response_peaks_at_band_center() {
    let band = BandParams {
        low: 0.1,
        high: 0.3,
        window: Window::Gaussian,
        mode: Mode::Pass,
    };
    let center = band.response(0.2);
    assert!((center - 1.0).abs() < 1e-9, "center {center}");
    // The edge sits at 1 sigma => exp(-0.5) ~ 0.6065.
    let edge = band.response(0.3);
    assert!((edge - (-0.5f64).exp()).abs() < 1e-9, "edge {edge}");
    // Far outside is strongly attenuated.
    assert!(band.response(0.5) < 0.05);
}

#[test]
fn is_deterministic_bit_for_bit() {
    let samples = cosine(16, 16, 5);
    let band = BandParams {
        low: 0.1,
        high: 0.4,
        window: Window::Gaussian,
        mode: Mode::Pass,
    };
    let a = apply_band(&samples, Extent::new(16, 16), 1, band);
    let b = apply_band(&samples, Extent::new(16, 16), 1, band);
    assert_eq!(a, b, "band filter must be bit-identical across runs");
}

#[test]
fn rejects_bad_parameters() {
    let value = image_value(Extent::new(4, 4), vec![0.0; 16]);
    let mut inputs = InputValues::new();
    inputs.insert("input".to_owned(), value);
    // Missing cutoffs.
    assert!(
        Bandpass::new()
            .compute(&inputs, &serde_json::json!({}))
            .is_err()
    );
    // high < low.
    assert!(
        Bandpass::new()
            .compute(&inputs, &serde_json::json!({"low": 0.4, "high": 0.1}))
            .is_err()
    );
    // Unknown window token.
    assert!(
        Bandpass::new()
            .compute(
                &inputs,
                &serde_json::json!({"low": 0.0, "high": 0.1, "window": "box"})
            )
            .is_err()
    );
}
