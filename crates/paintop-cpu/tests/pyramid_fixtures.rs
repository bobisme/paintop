//! M4 pyramid degenerate-size and reconstruction-tolerance fixtures
//! (`OP_CATALOG` §13, bn-k5rh).
//!
//! These are the consolidated *evidence* fixtures for the pyramid family: they
//! exercise the edge cases the per-op unit tests touch (1-level pyramids, odd
//! extents, a 1x1 degenerate base) and pin the documented reconstruction
//! tolerance for `laplacian_split` -> `recombine`, plus the serde / resource
//! validation of the `Pyramid` descriptor. Running them through the *registered*
//! ops (not internal helpers) proves the wired ops honour the convention.

use paintop_core::executor::{InputValues, OpImplementation, ResourceValue};
use paintop_cpu::gaussian_pyramid::{GAUSSIAN_PYRAMID_OP_ID, GaussianPyramid};
use paintop_cpu::laplacian::{LAPLACIAN_SPLIT_OP_ID, LaplacianSplit, RECOMBINE_OP_ID, Recombine};
use paintop_ir::{
    AlphaRepresentation, ChannelLayout, ColorEncoding, ColorRange, CoordinateConvention,
    DownsampleFactor, Extent, ImageDescriptor, PyramidDescriptor, PyramidPhase, ResourceDescriptor,
    ScalarType, SemanticRole,
};

/// The documented reconstruction tolerance: `laplacian_split` then `recombine`
/// reproduces the input to within this absolute per-sample bound (f32 rounding
/// across the blur / up/downsample telescope).
const RECON_TOLERANCE: f32 = 1.0e-5;

fn image_value(extent: Extent, layout: ChannelLayout, samples: Vec<f32>) -> ResourceValue {
    let descriptor = ResourceDescriptor::Image(ImageDescriptor {
        extent,
        layout,
        scalar: ScalarType::F32,
        color: ColorEncoding::LinearSrgb,
        range: ColorRange::SceneReferred,
        alpha: AlphaRepresentation::Premultiplied,
        coordinates: CoordinateConvention::PixelCenterUpperLeft,
        semantic: SemanticRole::Color,
    });
    ResourceValue::new(descriptor, layout.channel_count(), samples).expect("sized image buffer")
}

/// A deterministic, bounded test pattern of `n` samples in `[0, 1)`.
fn pattern(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let m = u16::try_from(i % 4096).unwrap_or(0);
            f32::from(m.wrapping_mul(151) % 251) / 251.0
        })
        .collect()
}

/// Build a Gaussian pyramid through the registered op.
fn gaussian_pyramid(value: ResourceValue, params: &serde_json::Value) -> ResourceValue {
    let mut inputs = InputValues::new();
    inputs.insert("input".to_owned(), value);
    let out = GaussianPyramid::new()
        .compute(&inputs, params)
        .expect("gaussian_pyramid compute");
    out.get("pyramid").expect("pyramid output").clone()
}

/// Split then recombine through the registered ops; returns the reconstruction.
fn split_recombine(value: ResourceValue, params: &serde_json::Value) -> Vec<f32> {
    let mut split_inputs = InputValues::new();
    split_inputs.insert("input".to_owned(), value);
    let split = LaplacianSplit::new()
        .compute(&split_inputs, params)
        .expect("laplacian_split compute");
    let pyramid = split.get("pyramid").expect("pyramid output").clone();

    let mut recomb_inputs = InputValues::new();
    recomb_inputs.insert("pyramid".to_owned(), pyramid);
    let recomb = Recombine::new()
        .compute(&recomb_inputs, &serde_json::Value::Null)
        .expect("recombine compute");
    recomb
        .get("image")
        .expect("image output")
        .samples()
        .to_vec()
}

#[test]
fn op_ids_are_the_canonical_frequency_ids() {
    assert_eq!(GAUSSIAN_PYRAMID_OP_ID, "frequency.gaussian_pyramid@1");
    assert_eq!(LAPLACIAN_SPLIT_OP_ID, "frequency.laplacian_split@1");
    assert_eq!(RECOMBINE_OP_ID, "frequency.recombine@1");
}

#[test]
fn one_level_pyramid_is_the_base_for_both_families() {
    let base = pattern(5 * 3);
    let value = image_value(Extent::new(5, 3), ChannelLayout::Gray, base.clone());
    let gp = gaussian_pyramid(value, &serde_json::json!({"levels": 1}));
    assert_eq!(gp.pyramid_level(0).unwrap(), base.as_slice());
    assert!(gp.pyramid_level(1).is_none());

    // The Laplacian split of a 1-level pyramid is the low-pass = the base, and
    // recombine returns it bit-exactly.
    let value = image_value(Extent::new(5, 3), ChannelLayout::Gray, base.clone());
    let recon = split_recombine(value, &serde_json::json!({"levels": 1}));
    assert_eq!(recon, base);
}

#[test]
fn degenerate_one_by_one_base_clamps_every_level() {
    // A 1x1 base: every deeper level is also 1x1 (clamped), and reconstruction
    // is exact (no spatial structure to lose).
    let base = vec![0.42_f32];
    let value = image_value(Extent::new(1, 1), ChannelLayout::Gray, base.clone());
    let gp = gaussian_pyramid(value, &serde_json::json!({"levels": 4}));
    for level in 0..4 {
        let plane = gp.pyramid_level(level).unwrap();
        assert_eq!(plane.len(), 1, "level {level} should stay 1x1");
        assert!((plane[0] - 0.42).abs() < RECON_TOLERANCE);
    }
    let value = image_value(Extent::new(1, 1), ChannelLayout::Gray, base);
    let recon = split_recombine(value, &serde_json::json!({"levels": 4}));
    assert_eq!(recon.len(), 1);
    assert!((recon[0] - 0.42).abs() < RECON_TOLERANCE);
}

#[test]
fn odd_extents_reconstruct_within_tolerance_under_both_phases() {
    for phase in ["floor", "ceil"] {
        for &(w, h) in &[(7u32, 5u32), (9, 3), (5, 9), (13, 11)] {
            let base = pattern((w * h) as usize);
            let value = image_value(Extent::new(w, h), ChannelLayout::Gray, base.clone());
            let recon = split_recombine(value, &serde_json::json!({"levels": 3, "phase": phase}));
            assert_eq!(recon.len(), base.len(), "{w}x{h} {phase}");
            let mut max_err = 0.0_f32;
            for (r, b) in recon.iter().zip(base.iter()) {
                max_err = max_err.max((r - b).abs());
            }
            assert!(
                max_err < RECON_TOLERANCE,
                "{w}x{h} phase {phase}: reconstruction max error {max_err} exceeds tolerance"
            );
        }
    }
}

#[test]
fn reconstruction_tolerance_holds_for_deep_pyramids() {
    // A deeper pyramid (more telescoped bands) still reconstructs within the
    // documented tolerance — the tolerance is not silently inflated by depth.
    let base = pattern(32 * 32);
    let value = image_value(Extent::new(32, 32), ChannelLayout::Gray, base.clone());
    let recon = split_recombine(value, &serde_json::json!({"levels": 5}));
    let mut max_err = 0.0_f32;
    for (r, b) in recon.iter().zip(base.iter()) {
        max_err = max_err.max((r - b).abs());
    }
    assert!(
        max_err < RECON_TOLERANCE,
        "deep pyramid max error {max_err}"
    );
}

#[test]
fn pyramid_descriptor_serde_round_trips_and_validates() {
    let d = PyramidDescriptor {
        base_extent: Extent::new(7, 5),
        levels: 3,
        channels: 1,
        scalar: ScalarType::F32,
        factor: DownsampleFactor::Half,
        phase: PyramidPhase::Ceil,
        coordinates: CoordinateConvention::PixelCenterUpperLeft,
    };
    d.validate().expect("valid pyramid");
    let json = serde_json::to_value(d).unwrap();
    let back: PyramidDescriptor = serde_json::from_value(json).unwrap();
    assert_eq!(back, d);

    // A malformed chain (zero levels) fails clearly.
    let bad = PyramidDescriptor { levels: 0, ..d };
    let err = bad.validate().unwrap_err();
    assert_eq!(err.code, "E_PYRAMID_LEVELS");

    // deny_unknown_fields rejects an unexpected field.
    let mut value = serde_json::to_value(d).unwrap();
    value["unexpected"] = serde_json::json!(1);
    assert!(serde_json::from_value::<PyramidDescriptor>(value).is_err());
}
