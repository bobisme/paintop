//! Tests for `analyze.frequency_energy@1`: per-band energy on analytic
//! multi-frequency fixtures, the mask window, and determinism.

use super::{FREQUENCY_ENERGY_OP_ID, FrequencyEnergy};
use paintop_core::executor::{InputValues, OpImplementation, ResourceValue};
use paintop_ir::{
    AlphaRepresentation, ChannelLayout, ColorEncoding, ColorRange, CoordinateConvention, Extent,
    FrequencyEnergyData, ImageDescriptor, MaskDescriptor, MaskMeaning, OperationManifest,
    ResourceDescriptor, ScalarType, SemanticRole, ValidRange,
};

fn gray_image(extent: Extent, samples: Vec<f32>) -> ResourceValue {
    let descriptor = ResourceDescriptor::Image(ImageDescriptor {
        extent,
        layout: ChannelLayout::Gray,
        scalar: ScalarType::F32,
        color: ColorEncoding::LinearSrgb,
        range: ColorRange::SceneReferred,
        alpha: AlphaRepresentation::Premultiplied,
        coordinates: CoordinateConvention::PixelCenterUpperLeft,
        semantic: SemanticRole::Color,
    });
    ResourceValue::new(descriptor, 1, samples).unwrap()
}

fn mask(extent: Extent, samples: Vec<f32>) -> ResourceValue {
    let descriptor = ResourceDescriptor::Mask(MaskDescriptor {
        extent,
        scalar: ScalarType::F32,
        range: ValidRange::Bounded { min: 0.0, max: 1.0 },
        meaning: MaskMeaning::Coverage,
        coordinates: CoordinateConvention::PixelCenterUpperLeft,
    });
    ResourceValue::new(descriptor, 1, samples).unwrap()
}

fn energy(
    resource: ResourceValue,
    mask: Option<ResourceValue>,
    params: &serde_json::Value,
) -> FrequencyEnergyData {
    let mut inputs = InputValues::new();
    inputs.insert("resource".to_owned(), resource);
    if let Some(m) = mask {
        inputs.insert("mask".to_owned(), m);
    }
    let out = FrequencyEnergy::new().compute(&inputs, params).unwrap();
    out.get("report")
        .unwrap()
        .as_report()
        .unwrap()
        .frequency_energy
        .clone()
        .unwrap()
}

/// A high-frequency checkerboard over an 8x8 grid (values 0/1).
fn checkerboard(extent: Extent) -> Vec<f32> {
    let w = extent.width;
    (0..extent.width * extent.height)
        .map(|i| {
            let x = i % w;
            let y = i / w;
            if (x + y).is_multiple_of(2) { 1.0 } else { 0.0 }
        })
        .collect()
}

#[test]
fn manifest_validates_and_matches_id() {
    let m: OperationManifest = FrequencyEnergy::manifest().unwrap();
    m.validate().unwrap();
    assert_eq!(m.id.to_string(), FREQUENCY_ENERGY_OP_ID);
}

#[test]
fn constant_image_concentrates_energy_in_the_low_pass_band() {
    // Laplacian: a constant has ~0 band-pass energy in the fine bands; all its
    // energy is in the coarsest (low-pass) band.
    let base = vec![0.5_f32; 8 * 8];
    let e = energy(
        gray_image(Extent::new(8, 8), base),
        None,
        &serde_json::json!({"bands": 3, "decomposition": "laplacian"}),
    );
    assert_eq!(e.bands, 3);
    // Fine bands (0, 1) have negligible energy.
    assert!(
        e.band_energy[0] < 1e-6,
        "fine band energy {}",
        e.band_energy[0]
    );
    assert!(
        e.band_energy[1] < 1e-6,
        "mid band energy {}",
        e.band_energy[1]
    );
    // The coarsest band carries the low-pass energy.
    assert!(e.band_energy[2] > 0.0);
}

#[test]
fn checkerboard_puts_energy_in_the_finest_band() {
    // A 1-pixel checkerboard is the highest representable frequency: the finest
    // Laplacian band carries the dominant energy.
    let base = checkerboard(Extent::new(8, 8));
    let e = energy(
        gray_image(Extent::new(8, 8), base),
        None,
        &serde_json::json!({"bands": 3}),
    );
    let total: f64 = e.band_energy.iter().sum();
    assert!(total > 0.0);
    // The finest band holds the majority of the energy.
    assert!(
        e.band_energy[0] > total * 0.5,
        "finest band {} of total {total}",
        e.band_energy[0]
    );
}

#[test]
fn total_energy_is_the_sum_of_bands() {
    let base = checkerboard(Extent::new(8, 8));
    let e = energy(
        gray_image(Extent::new(8, 8), base),
        None,
        &serde_json::json!({"bands": 4}),
    );
    let sum: f64 = e.band_energy.iter().sum();
    assert!((e.total_energy - sum).abs() < 1e-9);
    assert_eq!(e.band_pixels.len(), 4);
}

#[test]
fn gaussian_decomposition_reports_per_level_energy() {
    let base = vec![0.5_f32; 8 * 8];
    let e = energy(
        gray_image(Extent::new(8, 8), base),
        None,
        &serde_json::json!({"bands": 3, "decomposition": "gaussian"}),
    );
    assert_eq!(e.decomposition, "gaussian");
    // Every Gaussian level of a constant 0.5 has energy 0.25 * pixel_count.
    assert!((e.band_energy[0] - 16.0).abs() < 1e-4);
    assert!((e.band_energy[1] - 4.0).abs() < 1e-4);
    assert!((e.band_energy[2] - 1.0).abs() < 1e-4);
}

#[test]
fn mask_windows_the_analysis_region() {
    // With a mask covering the left half, only the left half's content
    // contributes; energy is strictly less than the full-image energy when the
    // right half is non-zero.
    let base = vec![1.0_f32; 8 * 8];
    let full = energy(
        gray_image(Extent::new(8, 8), base.clone()),
        None,
        &serde_json::json!({"bands": 2, "decomposition": "gaussian"}),
    );
    // Mask: left 4 columns covered, right 4 uncovered.
    let mask_samples: Vec<f32> = (0..64)
        .map(|i| if (i % 8) < 4 { 1.0 } else { 0.0 })
        .collect();
    let windowed = energy(
        gray_image(Extent::new(8, 8), base),
        Some(mask(Extent::new(8, 8), mask_samples)),
        &serde_json::json!({"bands": 2, "decomposition": "gaussian"}),
    );
    assert!(
        windowed.band_energy[0] < full.band_energy[0],
        "windowed {} should be < full {}",
        windowed.band_energy[0],
        full.band_energy[0]
    );
}

#[test]
fn energy_is_deterministic_bit_for_bit() {
    let base = checkerboard(Extent::new(8, 8));
    let params = serde_json::json!({"bands": 3});
    let a = energy(gray_image(Extent::new(8, 8), base.clone()), None, &params);
    let b = energy(gray_image(Extent::new(8, 8), base), None, &params);
    assert_eq!(a.band_energy, b.band_energy);
    assert_eq!(a.total_energy.to_bits(), b.total_energy.to_bits());
}

#[test]
fn rejects_missing_bands() {
    let value = gray_image(Extent::new(4, 4), vec![0.0; 16]);
    let mut inputs = InputValues::new();
    inputs.insert("resource".to_owned(), value);
    assert!(
        FrequencyEnergy::new()
            .compute(&inputs, &serde_json::json!({}))
            .is_err()
    );
}
