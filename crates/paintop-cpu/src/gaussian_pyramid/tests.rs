//! Tests for `frequency.gaussian_pyramid@1`: the level-extent convention,
//! constant preservation, the impulse/step behaviour, determinism, and the
//! degenerate (1-level, odd-size) fixtures.

use super::{GAUSSIAN_PYRAMID_OP_ID, GaussianPyramid, PyramidParams, build_pyramid_samples};
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

fn run(input: ResourceValue, params: &serde_json::Value) -> ResourceValue {
    let mut inputs = InputValues::new();
    inputs.insert("input".to_owned(), input);
    let out = GaussianPyramid::new().compute(&inputs, params).unwrap();
    out.get("pyramid").unwrap().clone()
}

#[test]
fn manifest_validates_and_matches_id() {
    let m: OperationManifest = GaussianPyramid::manifest().unwrap();
    m.validate().unwrap();
    assert_eq!(m.id.to_string(), GAUSSIAN_PYRAMID_OP_ID);
}

#[test]
fn level_zero_is_the_input_verbatim() {
    let base: Vec<f32> = (0..16u16).map(f32::from).collect();
    let value = image_value(Extent::new(4, 4), ChannelLayout::Gray, base.clone());
    let pyramid = run(value, &serde_json::json!({"levels": 3}));
    assert_eq!(pyramid.pyramid_level(0).unwrap(), base.as_slice());
}

#[test]
fn derived_level_extents_follow_the_floor_phase() {
    let base = vec![0.5_f32; 8 * 8];
    let value = image_value(Extent::new(8, 8), ChannelLayout::Gray, base);
    let pyramid = run(value, &serde_json::json!({"levels": 4, "phase": "floor"}));
    assert_eq!(pyramid.pyramid_level(0).unwrap().len(), 64);
    assert_eq!(pyramid.pyramid_level(1).unwrap().len(), 16);
    assert_eq!(pyramid.pyramid_level(2).unwrap().len(), 4);
    assert_eq!(pyramid.pyramid_level(3).unwrap().len(), 1);
}

#[test]
fn constant_image_yields_constant_levels() {
    let base = vec![0.25_f32; 8 * 8];
    let value = image_value(Extent::new(8, 8), ChannelLayout::Gray, base);
    let pyramid = run(value, &serde_json::json!({"levels": 4}));
    for level in 0..4 {
        for &v in pyramid.pyramid_level(level).unwrap() {
            assert!((v - 0.25).abs() < 1e-6, "level {level}: {v}");
        }
    }
}

#[test]
fn odd_extent_floor_and_ceil_differ() {
    let base = vec![0.0_f32; 5 * 5];
    let value = image_value(Extent::new(5, 5), ChannelLayout::Gray, base.clone());
    let floor = run(value, &serde_json::json!({"levels": 2, "phase": "floor"}));
    assert_eq!(floor.pyramid_level(1).unwrap().len(), 4); // 2x2

    let value = image_value(Extent::new(5, 5), ChannelLayout::Gray, base);
    let ceil = run(value, &serde_json::json!({"levels": 2, "phase": "ceil"}));
    assert_eq!(ceil.pyramid_level(1).unwrap().len(), 9); // 3x3
}

#[test]
fn one_level_pyramid_is_just_the_base() {
    let base: Vec<f32> = (0..15u16).map(|i| f32::from(i) * 0.1).collect();
    let value = image_value(Extent::new(5, 3), ChannelLayout::Gray, base.clone());
    let pyramid = run(value, &serde_json::json!({"levels": 1}));
    assert_eq!(pyramid.pyramid_level(0).unwrap(), base.as_slice());
    assert!(pyramid.pyramid_level(1).is_none());
}

#[test]
fn impulse_energy_spreads_but_is_bounded() {
    // A single bright pixel: each coarser level's peak is <= the base peak
    // (a positive normalized blur cannot amplify), and the total stays positive.
    let mut base = vec![0.0_f32; 8 * 8];
    base[4 * 8 + 4] = 1.0; // center-ish impulse
    let value = image_value(Extent::new(8, 8), ChannelLayout::Gray, base);
    let pyramid = run(value, &serde_json::json!({"levels": 3}));
    let base_peak = pyramid
        .pyramid_level(0)
        .unwrap()
        .iter()
        .copied()
        .fold(0.0_f32, f32::max);
    assert!((base_peak - 1.0).abs() < 1e-6);
    for level in 1..3 {
        let peak = pyramid
            .pyramid_level(level)
            .unwrap()
            .iter()
            .copied()
            .fold(0.0_f32, f32::max);
        assert!(
            peak <= base_peak + 1e-6,
            "level {level} peak {peak} exceeds base"
        );
        assert!(peak >= 0.0);
    }
}

#[test]
fn build_is_deterministic_bit_for_bit() {
    let base: Vec<f32> = (0..16u16 * 16)
        .map(|i| f32::from((i.wrapping_mul(37)) % 101) / 101.0)
        .collect();
    let params = PyramidParams {
        levels: 4,
        sigma: 1.0,
        phase: PyramidPhase::Floor,
    };
    let a = build_pyramid_samples(&base, Extent::new(16, 16), 1, params);
    let b = build_pyramid_samples(&base, Extent::new(16, 16), 1, params);
    assert_eq!(a, b, "pyramid build must be bit-identical across runs");
}

#[test]
fn rgba_channels_are_built_independently() {
    // Each channel a distinct constant; every level preserves the per-channel
    // constants (no cross-channel bleed in the separable blur).
    let mut base = Vec::new();
    for _ in 0..16 {
        base.extend_from_slice(&[0.1, 0.2, 0.3, 0.4]);
    }
    let value = image_value(Extent::new(4, 4), ChannelLayout::Rgba, base);
    let pyramid = run(value, &serde_json::json!({"levels": 3}));
    assert_eq!(pyramid.channels(), 4);
    for level in 0..3 {
        for pixel in pyramid.pyramid_level(level).unwrap().chunks_exact(4) {
            assert!((pixel[0] - 0.1).abs() < 1e-6);
            assert!((pixel[1] - 0.2).abs() < 1e-6);
            assert!((pixel[2] - 0.3).abs() < 1e-6);
            assert!((pixel[3] - 0.4).abs() < 1e-6);
        }
    }
}

#[test]
fn rejects_zero_levels_and_bad_phase() {
    let value = image_value(Extent::new(4, 4), ChannelLayout::Gray, vec![0.0; 16]);
    let mut inputs = InputValues::new();
    inputs.insert("input".to_owned(), value);
    assert!(
        GaussianPyramid::new()
            .compute(&inputs, &serde_json::json!({"levels": 0}))
            .is_err()
    );
    assert!(
        GaussianPyramid::new()
            .compute(&inputs, &serde_json::json!({"levels": 2, "phase": "weird"}))
            .is_err()
    );
}
