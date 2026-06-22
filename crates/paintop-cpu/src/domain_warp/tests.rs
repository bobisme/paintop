//! Verification suite for `field.domain_warp@1` (`OP_CATALOG` §11):
//!
//! - **schema/contract**: the manifest validates and matches its contract;
//! - **zero-displacement identity**: a zero displacement reproduces the source
//!   bit-for-bit;
//! - **constant-displacement translation**: an integer constant displacement is
//!   an exact pixel shift (nearest), and matches a clamp-boundary translate;
//! - **fractional translate**: a half-pixel constant displacement is the bilinear
//!   average of the two straddled columns/rows;
//! - **round-trip**: warp by `d` then by `-d` returns the source within the
//!   bilinear tolerance away from edges;
//! - **boundary honoring**: a displacement that points off-edge reads the boundary
//!   value the mode dictates;
//! - **determinism**: a rerun is bit-identical.

use paintop_core::executor::{InputValues, OpImplementation, ResourceValue};
use paintop_ir::{
    AlphaRepresentation, ChannelLayout, ColorEncoding, ColorRange, CoordinateConvention, Extent,
    FieldArity, FieldDescriptor, ImageDescriptor, ResourceDescriptor, ScalarType, SemanticRole,
};

use super::{DOMAIN_WARP_OP_ID, DomainWarp};

/// Wrap a flat row-major scalar buffer as a Field1 source value.
fn field1(width: u32, height: u32, samples: Vec<f32>) -> ResourceValue {
    let descriptor = ResourceDescriptor::Field1(FieldDescriptor {
        arity: FieldArity::Field1,
        extent: Extent::new(width, height),
        scalar: ScalarType::F32,
        semantic: SemanticRole::Material,
        coordinates: CoordinateConvention::PixelCenterUpperLeft,
        space: None,
        normalization: None,
        encoding: None,
    });
    ResourceValue::new(descriptor, 1, samples).expect("field buffer matches descriptor")
}

/// Wrap a flat (dx, dy)-interleaved buffer as a Field2 displacement value.
fn displacement(width: u32, height: u32, samples: Vec<f32>) -> ResourceValue {
    let descriptor = ResourceDescriptor::Field2(FieldDescriptor {
        arity: FieldArity::Field2,
        extent: Extent::new(width, height),
        scalar: ScalarType::F32,
        semantic: SemanticRole::Flow,
        coordinates: CoordinateConvention::PixelCenterUpperLeft,
        space: None,
        normalization: None,
        encoding: None,
    });
    ResourceValue::new(descriptor, 2, samples).expect("displacement buffer matches descriptor")
}

/// A constant (dx, dy) displacement field of the given extent.
fn constant_displacement(width: u32, height: u32, dx: f32, dy: f32) -> ResourceValue {
    let n = (width as usize) * (height as usize);
    let mut samples = Vec::with_capacity(n * 2);
    for _ in 0..n {
        samples.push(dx);
        samples.push(dy);
    }
    displacement(width, height, samples)
}

/// A ramp source where pixel `(x, y)` holds `x + 10*y` (distinct per pixel).
fn ramp(width: u32, height: u32) -> ResourceValue {
    let mut samples = Vec::with_capacity((width * height) as usize);
    for y in 0..height {
        for x in 0..width {
            samples.push(f32::from(u16::try_from(x + 10 * y).unwrap_or(0)));
        }
    }
    field1(width, height, samples)
}

/// Run the warp with the given params and return the warped buffer.
fn warp(source: &ResourceValue, disp: &ResourceValue, params: &serde_json::Value) -> Vec<f32> {
    let mut inputs = InputValues::new();
    inputs.insert("source".to_owned(), source.clone());
    inputs.insert("displacement".to_owned(), disp.clone());
    let out = DomainWarp::new()
        .compute(&inputs, params)
        .expect("warp computes");
    out.get("warped")
        .expect("warped output present")
        .samples()
        .to_vec()
}

#[test]
fn manifest_matches_contract() {
    let manifest = DomainWarp::manifest().expect("manifest builds");
    assert_eq!(manifest.id.to_string(), DOMAIN_WARP_OP_ID);
    assert_eq!(manifest.inputs.len(), 2);
}

#[test]
fn zero_displacement_is_identity_bilinear() {
    let source = ramp(8, 6);
    let disp = constant_displacement(8, 6, 0.0, 0.0);
    let out = warp(&source, &disp, &serde_json::json!({"filter": "bilinear"}));
    for (got, want) in out.iter().zip(source.samples()) {
        assert_eq!(got.to_bits(), want.to_bits(), "zero warp must be identity");
    }
}

#[test]
fn zero_displacement_is_identity_nearest() {
    let source = ramp(8, 6);
    let disp = constant_displacement(8, 6, 0.0, 0.0);
    let out = warp(&source, &disp, &serde_json::json!({"filter": "nearest"}));
    assert_eq!(
        out,
        source.samples(),
        "zero warp (nearest) must be identity"
    );
}

#[test]
fn integer_constant_displacement_is_a_shift() {
    // dx = 1, dy = 0: output (x, y) reads source (x + 1, y). With nearest and
    // clamp, the rightmost column replicates.
    let source = ramp(5, 3);
    let disp = constant_displacement(5, 3, 1.0, 0.0);
    let out = warp(
        &source,
        &disp,
        &serde_json::json!({"filter": "nearest", "boundary": {"mode": "clamp"}}),
    );
    let src = source.samples();
    let w = 5usize;
    for y in 0..3usize {
        for x in 0..w {
            let read_x = (x + 1).min(w - 1);
            let want = src[y * w + read_x];
            assert_eq!(
                out[y * w + x].to_bits(),
                want.to_bits(),
                "shift mismatch at ({x},{y})"
            );
        }
    }
}

#[test]
fn half_pixel_displacement_is_the_bilinear_average() {
    // dx = 0.5: output (x, y) reads source center (x + 1.0); with the pixel-center
    // convention that lands exactly between centers x and x+1, so bilinear gives
    // their average. dy = 0 keeps rows aligned.
    let source = ramp(6, 1);
    let disp = constant_displacement(6, 1, 0.5, 0.0);
    let out = warp(
        &source,
        &disp,
        &serde_json::json!({"filter": "bilinear", "boundary": {"mode": "clamp"}}),
    );
    let src = source.samples();
    let w = 6usize;
    for x in 0..w {
        let a = f64::from(src[x]);
        let b = f64::from(src[(x + 1).min(w - 1)]);
        let want = (a + b) * 0.5;
        assert!(
            (f64::from(out[x]) - want).abs() < 1e-4,
            "half-pixel average mismatch at {x}: got {} want {want}",
            out[x]
        );
    }
}

#[test]
fn warp_inverse_warp_round_trips_in_interior() {
    // A constant fractional shift `d` followed by `-d` is the inverse warp; on a
    // smooth low-gradient source the bilinear round-trip recovers the source to a
    // small, bounded error in the interior (a constant displacement's negation is
    // exactly its inverse, so the only error is the bilinear reconstruction blur).
    let w = 16u32;
    let h = 16u32;
    // A smooth, low-gradient source: a coarse sinusoid so the slope is gentle and
    // a sub-pixel misregistration cannot blow up into a large value error.
    let mut samples = Vec::with_capacity((w * h) as usize);
    for y in 0..h {
        for x in 0..w {
            let v = (f64::from(x) * 0.2).sin() + (f64::from(y) * 0.2).cos();
            #[allow(clippy::cast_possible_truncation, reason = "test source to f32")]
            samples.push(v as f32);
        }
    }
    let source = field1(w, h, samples);
    let fwd = constant_displacement(w, h, 0.37, -0.21);
    let bwd = constant_displacement(w, h, -0.37, 0.21);
    let params = serde_json::json!({"filter": "bilinear", "boundary": {"mode": "clamp"}});
    let once = warp(&source, &fwd, &params);
    let once_field = field1(w, h, once);
    let twice = warp(&once_field, &bwd, &params);
    // Check the interior (away from the 1px border where clamping breaks the
    // inverse).
    let src = source.samples();
    let ww = w as usize;
    for y in 2..(h as usize - 2) {
        for x in 2..(ww - 2) {
            let i = y * ww + x;
            assert!(
                (f64::from(twice[i]) - f64::from(src[i])).abs() < 0.05,
                "round-trip drifted at ({x},{y}): {} vs {}",
                twice[i],
                src[i]
            );
        }
    }
}

#[test]
fn boundary_constant_reads_the_constant_off_edge() {
    // A large negative dx points off the left edge; with constant boundary the
    // value must be the declared constant.
    let source = ramp(4, 1);
    let disp = constant_displacement(4, 1, -100.0, 0.0);
    let out = warp(
        &source,
        &disp,
        &serde_json::json!({"filter": "nearest", "boundary": {"mode": "constant", "value": -7.0}}),
    );
    for &v in &out {
        assert_eq!(
            v.to_bits(),
            (-7.0_f32).to_bits(),
            "off-edge must read the constant"
        );
    }
}

#[test]
fn boundary_wrap_is_periodic() {
    // dx = width points exactly one period right; wrap returns the same row.
    let source = ramp(5, 1);
    let disp = constant_displacement(5, 1, 5.0, 0.0);
    let out = warp(
        &source,
        &disp,
        &serde_json::json!({"filter": "nearest", "boundary": {"mode": "wrap"}}),
    );
    assert_eq!(out, source.samples(), "a full-period wrap is the identity");
}

#[test]
fn warp_is_bit_identical_across_runs() {
    let source = ramp(12, 9);
    let disp = constant_displacement(12, 9, 1.3, -0.7);
    let p = serde_json::json!({"filter": "bilinear"});
    assert_eq!(warp(&source, &disp, &p), warp(&source, &disp, &p));
}

#[test]
fn descriptor_is_preserved_for_an_image_source() {
    let descriptor = ResourceDescriptor::Image(ImageDescriptor {
        extent: Extent::new(3, 2),
        layout: ChannelLayout::Rgb,
        scalar: ScalarType::F32,
        color: ColorEncoding::LinearSrgb,
        range: ColorRange::SceneReferred,
        alpha: AlphaRepresentation::Straight,
        coordinates: CoordinateConvention::PixelCenterUpperLeft,
        semantic: SemanticRole::Color,
    });
    let samples = vec![0.0_f32; 3 * 2 * 3];
    let source = ResourceValue::new(descriptor, 3, samples).expect("image value");
    let disp = constant_displacement(3, 2, 0.0, 0.0);
    let mut inputs = InputValues::new();
    inputs.insert("source".to_owned(), source.clone());
    inputs.insert("displacement".to_owned(), disp);
    let out = DomainWarp::new()
        .compute(&inputs, &serde_json::json!({}))
        .expect("image warp computes");
    let warped = out.get("warped").expect("warped present");
    assert_eq!(
        warped.descriptor(),
        source.descriptor(),
        "the warp must preserve the source descriptor"
    );
}

#[test]
fn extent_mismatch_is_rejected() {
    let source = ramp(4, 4);
    let disp = constant_displacement(3, 4, 0.0, 0.0);
    let mut inputs = InputValues::new();
    inputs.insert("source".to_owned(), source);
    inputs.insert("displacement".to_owned(), disp);
    assert!(
        DomainWarp::new()
            .compute(&inputs, &serde_json::json!({}))
            .is_err()
    );
}
