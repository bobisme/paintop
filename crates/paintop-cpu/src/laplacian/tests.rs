//! Tests for `frequency.laplacian_split@1` and `frequency.recombine@1`: the
//! split/recombine reconstruction contract (bounded, odd, single-level
//! fixtures), determinism, and the band-extent convention.

use super::{
    LAPLACIAN_SPLIT_OP_ID, LaplacianSplit, RECOMBINE_OP_ID, Recombine, build_laplacian_samples,
};
use crate::gaussian_pyramid::PyramidParams;
use paintop_core::executor::{InputValues, OpImplementation, ResourceValue};
use paintop_ir::{
    AlphaRepresentation, ChannelLayout, ColorEncoding, ColorRange, CoordinateConvention, Extent,
    ImageDescriptor, OperationManifest, PyramidPhase, ResourceDescriptor, ScalarType, SemanticRole,
};

fn image_descriptor(extent: Extent, layout: ChannelLayout) -> ResourceDescriptor {
    ResourceDescriptor::Image(ImageDescriptor {
        extent,
        layout,
        scalar: ScalarType::F32,
        color: ColorEncoding::LinearSrgb,
        range: ColorRange::SceneReferred,
        alpha: AlphaRepresentation::Premultiplied,
        coordinates: CoordinateConvention::PixelCenterUpperLeft,
        semantic: SemanticRole::Color,
    })
}

fn image_value(extent: Extent, layout: ChannelLayout, samples: Vec<f32>) -> ResourceValue {
    ResourceValue::new(
        image_descriptor(extent, layout),
        layout.channel_count(),
        samples,
    )
    .unwrap()
}

/// Run split then recombine, returning the reconstructed sample buffer.
fn round_trip(
    extent: Extent,
    layout: ChannelLayout,
    base: Vec<f32>,
    params: &serde_json::Value,
) -> Vec<f32> {
    let value = image_value(extent, layout, base);
    let mut split_inputs = InputValues::new();
    split_inputs.insert("input".to_owned(), value);
    let split_out = LaplacianSplit::new()
        .compute(&split_inputs, params)
        .unwrap();
    let pyramid = split_out.get("pyramid").unwrap().clone();

    let mut recomb_inputs = InputValues::new();
    recomb_inputs.insert("pyramid".to_owned(), pyramid);
    let recomb_out = Recombine::new()
        .compute(&recomb_inputs, &serde_json::Value::Null)
        .unwrap();
    recomb_out.get("image").unwrap().samples().to_vec()
}

#[test]
fn manifests_validate_and_match_ids() {
    let split: OperationManifest = LaplacianSplit::manifest().unwrap();
    split.validate().unwrap();
    assert_eq!(split.id.to_string(), LAPLACIAN_SPLIT_OP_ID);
    let recomb: OperationManifest = Recombine::manifest().unwrap();
    recomb.validate().unwrap();
    assert_eq!(recomb.id.to_string(), RECOMBINE_OP_ID);
}

#[test]
fn split_recombine_reconstructs_the_original() {
    let base: Vec<f32> = (0..16u16 * 16)
        .map(|i| f32::from(i.wrapping_mul(53) % 97) / 97.0)
        .collect();
    let recon = round_trip(
        Extent::new(16, 16),
        ChannelLayout::Gray,
        base.clone(),
        &serde_json::json!({"levels": 4}),
    );
    assert_eq!(recon.len(), base.len());
    for (r, b) in recon.iter().zip(base.iter()) {
        assert!((r - b).abs() < 1e-5, "reconstruction drift {r} vs {b}");
    }
}

#[test]
fn single_level_round_trips_exactly() {
    let base: Vec<f32> = (0..12u16).map(|i| f32::from(i) / 12.0).collect();
    let recon = round_trip(
        Extent::new(4, 3),
        ChannelLayout::Gray,
        base.clone(),
        &serde_json::json!({"levels": 1}),
    );
    // A 1-level Laplacian pyramid is just the low-pass = the input, so the
    // reconstruction is bit-exact.
    assert_eq!(recon, base);
}

#[test]
fn odd_extent_round_trips_under_both_phases() {
    for phase in ["floor", "ceil"] {
        let base: Vec<f32> = (0..7u16 * 5).map(|i| f32::from(i % 11) / 11.0).collect();
        let recon = round_trip(
            Extent::new(7, 5),
            ChannelLayout::Gray,
            base.clone(),
            &serde_json::json!({"levels": 3, "phase": phase}),
        );
        for (r, b) in recon.iter().zip(base.iter()) {
            assert!((r - b).abs() < 1e-5, "phase {phase}: {r} vs {b}");
        }
    }
}

#[test]
fn rgba_round_trips_per_channel() {
    let mut base = Vec::new();
    for i in 0..16u16 {
        let f = f32::from(i) / 16.0;
        base.extend_from_slice(&[f, 1.0 - f, 0.5, f * 0.25]);
    }
    let recon = round_trip(
        Extent::new(4, 4),
        ChannelLayout::Rgba,
        base.clone(),
        &serde_json::json!({"levels": 3}),
    );
    assert_eq!(recon.len(), base.len());
    for (r, b) in recon.iter().zip(base.iter()) {
        assert!((r - b).abs() < 1e-5, "{r} vs {b}");
    }
}

#[test]
fn split_is_deterministic_bit_for_bit() {
    let base: Vec<f32> = (0..8u16 * 8).map(|i| f32::from(i % 7) / 7.0).collect();
    let params = PyramidParams {
        levels: 3,
        sigma: 1.0,
        phase: PyramidPhase::Floor,
    };
    let a = build_laplacian_samples(&base, Extent::new(8, 8), 1, params);
    let b = build_laplacian_samples(&base, Extent::new(8, 8), 1, params);
    assert_eq!(a, b, "laplacian split must be bit-identical across runs");
}

#[test]
fn coarsest_band_is_the_low_pass_not_a_residual() {
    // The coarsest Laplacian band equals the coarsest Gaussian level (the
    // low-pass), so a constant image keeps a constant coarsest band.
    let base = vec![0.3_f32; 8 * 8];
    let value = image_value(Extent::new(8, 8), ChannelLayout::Gray, base);
    let mut inputs = InputValues::new();
    inputs.insert("input".to_owned(), value);
    let out = LaplacianSplit::new()
        .compute(&inputs, &serde_json::json!({"levels": 3}))
        .unwrap();
    let pyramid = out.get("pyramid").unwrap();
    // Coarsest level (index 2) is the low-pass of a constant => constant 0.3.
    for &v in pyramid.pyramid_level(2).unwrap() {
        assert!((v - 0.3).abs() < 1e-6, "coarsest band not low-pass: {v}");
    }
    // A finer band of a constant image is ~0 (residual of equal planes).
    for &v in pyramid.pyramid_level(0).unwrap() {
        assert!(
            v.abs() < 1e-5,
            "high-pass band of a constant should be ~0: {v}"
        );
    }
}

#[test]
fn recombine_rejects_a_non_pyramid_input() {
    let img = image_value(Extent::new(4, 4), ChannelLayout::Gray, vec![0.0; 16]);
    let mut inputs = InputValues::new();
    inputs.insert("pyramid".to_owned(), img);
    assert!(
        Recombine::new()
            .compute(&inputs, &serde_json::Value::Null)
            .is_err()
    );
}
