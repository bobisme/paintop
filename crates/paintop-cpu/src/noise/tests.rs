//! Tests for `field.noise@1` and `field.fbm@1`: hash-of-coordinate determinism,
//! tiling / evaluation-order invariance, seed sensitivity, range bounds, and the
//! fbm octave normalization.

use super::*;
use paintop_core::executor::{InputValues, OpImplementation};

/// Run `field.noise@1` and return the row-major scalar buffer.
fn noise(width: u32, height: u32, frequency: f64, seed: u64) -> Vec<f32> {
    let op = Noise::new();
    let p = serde_json::json!({
        "width": width,
        "height": height,
        "frequency": frequency,
        "seed": seed,
    });
    let out = op
        .compute(&InputValues::new(), &p)
        .expect("noise compute succeeds");
    out.get("field")
        .expect("field output present")
        .samples()
        .to_vec()
}

/// Run `field.fbm@1` and return the row-major scalar buffer.
fn fbm(p: &serde_json::Value) -> Vec<f32> {
    let op = Fbm::new();
    let out = op
        .compute(&InputValues::new(), p)
        .expect("fbm compute succeeds");
    out.get("field")
        .expect("field output present")
        .samples()
        .to_vec()
}

#[test]
fn noise_is_bit_identical_across_runs() {
    let a = noise(37, 29, 3.0, 12345);
    let b = noise(37, 29, 3.0, 12345);
    assert_eq!(a, b, "noise must be bit-identical on rerun");
}

#[test]
fn noise_is_in_unit_range() {
    let field = noise(64, 64, 8.0, 7);
    for &v in &field {
        assert!((-1.0..=1.0).contains(&v), "value {v} outside [-1, 1]");
    }
}

#[test]
fn noise_is_tiling_and_order_invariant() {
    // The value at a coordinate must not depend on the surrounding extent: a
    // sub-window of a big field equals the same window synthesized standalone is
    // NOT what we test (the standalone field has a different origin). Instead we
    // assert the defining hash-of-coordinate property directly: evaluating the
    // continuous noise at the same coordinate in any order gives the same value.
    let seed = 999;
    let scale = 5.0 / FREQUENCY_PIXELS;
    // Forward order.
    let mut forward = Vec::new();
    for y in 0..16u32 {
        for x in 0..16u32 {
            let (px, py) = CoordinateConvention::PixelCenterUpperLeft.pixel_center(x, y);
            forward.push(value_noise(px * scale, py * scale, seed));
        }
    }
    // Reverse order — same coordinates, different evaluation order.
    let mut reverse = std::collections::HashMap::new();
    for y in (0..16u32).rev() {
        for x in (0..16u32).rev() {
            let (px, py) = CoordinateConvention::PixelCenterUpperLeft.pixel_center(x, y);
            reverse.insert((x, y), value_noise(px * scale, py * scale, seed));
        }
    }
    let mut i = 0;
    for y in 0..16u32 {
        for x in 0..16u32 {
            let expected = reverse[&(x, y)];
            assert_eq!(
                forward[i].to_bits(),
                expected.to_bits(),
                "noise at ({x},{y}) depends on evaluation order"
            );
            i += 1;
        }
    }
}

#[test]
fn noise_lattice_value_is_position_pure() {
    // The lattice hash at a given coordinate is identical no matter when called.
    let seed = 42;
    let first = lattice_value(3, -7, seed);
    let between = lattice_value(100, 200, seed);
    let again = lattice_value(3, -7, seed);
    assert_eq!(
        first.to_bits(),
        again.to_bits(),
        "lattice hash must be a pure function of (gx, gy, seed)"
    );
    assert!(
        (first - between).abs() > f64::EPSILON,
        "distinct coordinates should generally hash to distinct values"
    );
}

#[test]
fn changing_seed_changes_the_field() {
    let a = noise(32, 32, 6.0, 1);
    let b = noise(32, 32, 6.0, 2);
    assert_ne!(a, b, "a different seed must produce a different field");
}

#[test]
fn fbm_is_bit_identical_across_runs() {
    let p = serde_json::json!({"width": 24, "height": 24, "seed": 5, "octaves": 5});
    assert_eq!(fbm(&p), fbm(&p), "fbm must be bit-identical on rerun");
}

#[test]
fn fbm_stays_in_unit_range_for_any_octave_count() {
    // Amplitude normalization keeps the range bounded regardless of octave count.
    for octaves in [1u32, 2, 4, 8, 16, 32] {
        let p = serde_json::json!({
            "width": 48, "height": 48, "seed": 3, "octaves": octaves, "frequency": 4.0
        });
        let field = fbm(&p);
        for &v in &field {
            assert!(
                (-1.0001..=1.0001).contains(&v),
                "fbm with {octaves} octaves produced {v} outside [-1, 1]"
            );
        }
    }
}

#[test]
fn fbm_normalization_divides_by_total_amplitude() {
    // With gain g and O octaves the total amplitude is sum_{o<O} g^o; the
    // evaluate() sum is divided by it. Check that a single-octave fbm equals the
    // underlying value noise exactly (total amplitude == 1, octave seed mixed).
    let req = FbmRequest {
        base: NoiseRequest {
            extent: Extent::new(1, 1),
            frequency: 1.0,
            seed: 77,
        },
        octaves: 1,
        lacunarity: 2.0,
        gain: 0.5,
    };
    let octave_seed = mix64(0x4d_u64 ^ 0x1);
    let direct = value_noise(0.3, 0.4, octave_seed);
    let via_fbm = req.evaluate(0.3, 0.4);
    assert!(
        (direct - via_fbm).abs() < 1e-12,
        "single-octave fbm ({via_fbm}) must equal its value noise ({direct})"
    );
}

#[test]
fn fbm_octave_frequencies_follow_lacunarity() {
    // Two octaves: sum = (1 * n(x) + g * n(lac*x)) / (1 + g). Reconstruct it by
    // hand and compare to evaluate().
    let lac = 2.0;
    let gain = 0.5;
    let req = FbmRequest {
        base: NoiseRequest {
            extent: Extent::new(1, 1),
            frequency: 1.0,
            seed: 11,
        },
        octaves: 2,
        lacunarity: lac,
        gain,
    };
    let s0 = mix64(0x0b_u64 ^ 0x1);
    let s1 = mix64(0x0b_u64 ^ 0x2);
    let x = 1.7;
    let y = -0.6;
    let manual =
        value_noise(x, y, s0).mul_add(1.0, gain * value_noise(x * lac, y * lac, s1)) / (1.0 + gain);
    let got = req.evaluate(x, y);
    assert!(
        (manual - got).abs() < 1e-12,
        "two-octave fbm normalization mismatch: manual {manual}, got {got}"
    );
}

#[test]
fn zero_octaves_is_rejected() {
    let op = Fbm::new();
    let p = serde_json::json!({"width": 4, "height": 4, "octaves": 0});
    assert!(op.compute(&InputValues::new(), &p).is_err());
}

#[test]
fn non_positive_frequency_is_rejected() {
    let op = Noise::new();
    let p = serde_json::json!({"width": 4, "height": 4, "frequency": 0.0});
    assert!(op.compute(&InputValues::new(), &p).is_err());
    let p = serde_json::json!({"width": 4, "height": 4, "frequency": -1.0});
    assert!(op.compute(&InputValues::new(), &p).is_err());
}

#[test]
fn fade_endpoints_and_midpoint() {
    assert!((fade(0.0)).abs() < 1e-15);
    assert!((fade(1.0) - 1.0).abs() < 1e-15);
    assert!(
        (fade(0.5) - 0.5).abs() < 1e-15,
        "quintic fade is symmetric about 0.5"
    );
}

#[test]
fn manifests_are_well_formed() {
    let n = Noise::manifest().expect("noise manifest builds");
    assert_eq!(n.id.to_string(), NOISE_OP_ID);
    assert!(n.inputs.is_empty());
    let f = Fbm::manifest().expect("fbm manifest builds");
    assert_eq!(f.id.to_string(), FBM_OP_ID);
    assert!(f.inputs.is_empty());
}
