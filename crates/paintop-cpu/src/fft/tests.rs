//! Tests for `frequency.fft2@1` and `frequency.ifft2@1`: the round-trip
//! reconstruction contract, the pure-DC constant fixture, the known-sinusoid
//! spectral peak, channel independence, and determinism.

use super::{FFT2_OP_ID, Fft2, IFFT2_OP_ID, Ifft2};
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

fn image_value(extent: Extent, layout: ChannelLayout, samples: Vec<f32>) -> ResourceValue {
    let d = ResourceDescriptor::Image(ImageDescriptor {
        extent,
        layout,
        scalar: ScalarType::F32,
        color: ColorEncoding::LinearSrgb,
        range: ColorRange::SceneReferred,
        alpha: AlphaRepresentation::Premultiplied,
        coordinates: CoordinateConvention::PixelCenterUpperLeft,
        semantic: SemanticRole::Color,
    });
    ResourceValue::new(d, layout.channel_count(), samples).unwrap()
}

fn fft2(input: ResourceValue) -> ResourceValue {
    let mut inputs = InputValues::new();
    inputs.insert("input".to_owned(), input);
    let out = Fft2::new()
        .compute(&inputs, &serde_json::Value::Null)
        .unwrap();
    out.get("spectrum").unwrap().clone()
}

fn ifft2(spectrum: ResourceValue) -> ResourceValue {
    let mut inputs = InputValues::new();
    inputs.insert("spectrum".to_owned(), spectrum);
    let out = Ifft2::new()
        .compute(&inputs, &serde_json::Value::Null)
        .unwrap();
    out.get("image").unwrap().clone()
}

#[test]
fn manifests_validate_and_match_ids() {
    let f: OperationManifest = Fft2::manifest().unwrap();
    f.validate().unwrap();
    assert_eq!(f.id.to_string(), FFT2_OP_ID);
    let i: OperationManifest = Ifft2::manifest().unwrap();
    i.validate().unwrap();
    assert_eq!(i.id.to_string(), IFFT2_OP_ID);
}

#[test]
fn fft2_then_ifft2_round_trips() {
    for (w, h) in [(8u32, 8u32), (12, 12), (10, 6)] {
        let pixels = (w * h) as usize;
        let samples: Vec<f32> = (0..pixels).map(|i| ramp(i, 41, 97)).collect();
        let input = image_value(Extent::new(w, h), ChannelLayout::Gray, samples.clone());
        let spectrum = fft2(input);
        assert_eq!(spectrum.channels(), 1);
        assert_eq!(spectrum.extent(), Extent::new(w, h));
        // Spectrum buffer is W*H*channels*2 (interleaved re/im).
        assert_eq!(spectrum.samples().len(), pixels * 2);
        let recon = ifft2(spectrum);
        assert_eq!(recon.samples().len(), samples.len());
        for (r, s) in recon.samples().iter().zip(samples.iter()) {
            assert!((r - s).abs() < 1e-4, "{w}x{h}: {r} vs {s}");
        }
    }
}

#[test]
fn constant_image_is_pure_dc() {
    let input = image_value(Extent::new(8, 8), ChannelLayout::Gray, vec![0.5_f32; 64]);
    let spectrum = fft2(input);
    let s = spectrum.samples();
    // DC = sum = 0.5*64 = 32; everything else ~0.
    assert!((f64::from(s[0]) - 32.0).abs() < 1e-3);
    for k in 1..64 {
        assert!(f64::from(s[k * 2]).abs() < 1e-3, "bin {k} re");
        assert!(f64::from(s[k * 2 + 1]).abs() < 1e-3, "bin {k} im");
    }
}

#[test]
fn sinusoid_produces_expected_peak() {
    let extent = Extent::new(16, 16);
    let (w, h) = (extent.width as usize, extent.height as usize);
    let fx = 4usize;
    let mut samples = vec![0.0_f32; w * h];
    for y in 0..h {
        for x in 0..w {
            #[allow(
                clippy::cast_precision_loss,
                reason = "test fixture: small indices exact in f64"
            )]
            let arg = 2.0 * std::f64::consts::PI * (fx as f64) * (x as f64) / (w as f64);
            samples[y * w + x] = to_f32(arg.cos());
        }
    }
    let input = image_value(extent, ChannelLayout::Gray, samples);
    let spectrum = fft2(input);
    let s = spectrum.samples();
    let energy = |k: usize| -> f64 {
        let re = f64::from(s[k * 2]);
        let im = f64::from(s[k * 2 + 1]);
        re.mul_add(re, im * im)
    };
    #[allow(
        clippy::cast_precision_loss,
        reason = "test fixture: w*h is a small extent exact in f64"
    )]
    let peak = (w * h) as f64 / 2.0;
    assert!((energy(fx).sqrt() - peak).abs() < 1.0, "+fx peak");
    assert!((energy(w - fx).sqrt() - peak).abs() < 1.0, "-fx peak");
}

#[test]
fn round_trip_is_bit_identical_across_reruns() {
    let samples: Vec<f32> = (0..64usize).map(|i| ramp(i, 1, 11)).collect();
    let a = fft2(image_value(
        Extent::new(8, 8),
        ChannelLayout::Gray,
        samples.clone(),
    ));
    let b = fft2(image_value(Extent::new(8, 8), ChannelLayout::Gray, samples));
    assert_eq!(a.samples(), b.samples(), "fft2 must be bit-identical");
}

#[test]
fn rgb_channels_round_trip_independently() {
    let mut samples = Vec::new();
    for i in 0..64usize {
        let f = ramp(i, 1, 5);
        samples.extend_from_slice(&[f, 1.0 - f, 0.5 * f]);
    }
    let input = image_value(Extent::new(8, 8), ChannelLayout::Rgb, samples.clone());
    let spectrum = fft2(input);
    assert_eq!(spectrum.channels(), 3);
    let recon = ifft2(spectrum);
    assert_eq!(recon.channels(), 3);
    for (r, s) in recon.samples().iter().zip(samples.iter()) {
        assert!((r - s).abs() < 1e-4, "{r} vs {s}");
    }
}

#[test]
fn single_channel_inverse_is_a_field() {
    let input = image_value(Extent::new(4, 4), ChannelLayout::Gray, vec![0.3_f32; 16]);
    let spectrum = fft2(input);
    let recon = ifft2(spectrum);
    assert!(matches!(recon.descriptor(), ResourceDescriptor::Field1(_)));
}

#[test]
fn ifft2_rejects_non_spectrum_input() {
    // Feeding an image where a spectrum is required is a type error.
    let img = image_value(Extent::new(4, 4), ChannelLayout::Gray, vec![0.0; 16]);
    let mut inputs = InputValues::new();
    inputs.insert("spectrum".to_owned(), img);
    assert!(
        Ifft2::new()
            .compute(&inputs, &serde_json::Value::Null)
            .is_err()
    );
}
