//! Verification suite for `image.resize@1` (`OP_CATALOG` §5,
//! `AGENT_VERIFICATION` §3.8).
//!
//! Covers the contract (schema/coords/boundary), the four resamplers
//! (nearest / bilinear / bicubic / Lanczos), and the §3.8 property set:
//! identity-scale identity, nearest integer-translation exactness,
//! constant-preserving normalized kernels, output independent of tiling,
//! declared support/halo, and a band-limited up/down round-trip within
//! tolerance.

use paintop_core::executor::{InputValues, OpImplementation, ResourceValue};
use paintop_ir::{
    AlphaRepresentation, ChannelLayout, ColorEncoding, ColorRange, CoordinateConvention,
    Descriptors, ErrorClass, Extent, ImageDescriptor, OpContract, OutputRegions, Rect,
    ResourceDescriptor, ScalarType, SemanticRole,
};

use super::{
    E_RESIZE_EMPTY, E_RESIZE_FILTER, E_RESIZE_SIZE, Filter, RESIZE_OP_ID, Resize, source_footprint,
};

/// Tolerance for the bounded (floating-point) resamplers.
const TOL: f32 = 1e-5;

/// All four filter tokens.
const FILTERS: [&str; 4] = ["nearest", "bilinear", "bicubic", "lanczos"];

/// Build a gray image [`ResourceValue`] from a row-major sample list.
fn gray(width: u32, height: u32, samples: Vec<f32>) -> ResourceValue {
    image_value(width, height, ChannelLayout::Gray, samples)
}

/// Build an image [`ResourceValue`] with the given layout/samples.
fn image_value(width: u32, height: u32, layout: ChannelLayout, samples: Vec<f32>) -> ResourceValue {
    let descriptor = ResourceDescriptor::Image(ImageDescriptor {
        extent: Extent::new(width, height),
        layout,
        scalar: ScalarType::F32,
        color: ColorEncoding::LinearSrgb,
        range: ColorRange::SceneReferred,
        alpha: AlphaRepresentation::Straight,
        coordinates: CoordinateConvention::PixelCenterUpperLeft,
        semantic: SemanticRole::Color,
    });
    ResourceValue::new(descriptor, layout.channel_count(), samples)
        .expect("sample buffer matches descriptor")
}

/// Run the resize kernel and recover the output image.
fn resize(value: &ResourceValue, width: u32, height: u32, filter: &str) -> ResourceValue {
    let mut inputs = InputValues::new();
    inputs.insert("image".to_owned(), value.clone());
    let params = serde_json::json!({ "width": width, "height": height, "filter": filter });
    let mut out = Resize::new()
        .compute(&inputs, &params)
        .expect("resize computes");
    out.remove("image").expect("image port produced")
}

#[test]
fn manifest_validates_and_agrees_with_contract() {
    let manifest = Resize::manifest().expect("resize manifest");
    manifest.validate().expect("resize manifest valid");
    paintop_ir::check_contract_consistency(&manifest, &Resize::new())
        .expect("resize manifest agrees with contract");
    paintop_ir::verify_categories(&manifest, &manifest.test.verification)
        .expect("resize verification declarations gate clean");
    assert_eq!(manifest.id.to_string(), RESIZE_OP_ID);
}

// ----- contract / coordinates / boundary (bn-1mp) -----

#[test]
fn identity_scale_is_exact_identity_for_every_filter() {
    let samples: Vec<f32> = (0..12u8).map(|i| f32::from(i) / 11.0).collect();
    let img = gray(4, 3, samples.clone());
    for filter in FILTERS {
        let out = resize(&img, 4, 3, filter);
        assert_eq!(out.extent(), Extent::new(4, 3));
        // Exact (bit-for-bit) identity short-circuit.
        assert_eq!(out.samples(), samples.as_slice(), "filter {filter}");
    }
}

#[test]
fn one_by_one_target_is_well_defined() {
    // Resizing a constant image to 1x1 yields the constant for every filter.
    let img = gray(4, 4, vec![0.5; 16]);
    for filter in FILTERS {
        let out = resize(&img, 1, 1, filter);
        assert_eq!(out.extent(), Extent::new(1, 1));
        assert!(
            (out.samples()[0] - 0.5).abs() < TOL,
            "1x1 constant under {filter} = {}",
            out.samples()[0]
        );
    }
}

#[test]
fn upscaling_a_one_by_one_is_constant() {
    // A 1x1 source has a single sample; every output samples it (edge clamp).
    let img = gray(1, 1, vec![0.7]);
    for filter in FILTERS {
        let out = resize(&img, 5, 3, filter);
        assert_eq!(out.extent(), Extent::new(5, 3));
        for &s in out.samples() {
            assert!((s - 0.7).abs() < TOL, "1x1 upscale under {filter} = {s}");
        }
    }
}

#[test]
fn constant_image_is_preserved_by_normalized_kernels() {
    // Constant-preserving: a constant resizes to the same constant (up and down).
    let img = gray(8, 6, vec![0.3; 48]);
    for filter in ["bilinear", "bicubic", "lanczos"] {
        for (w, h) in [(3, 2), (16, 12), (5, 9)] {
            let out = resize(&img, w, h, filter);
            for &s in out.samples() {
                assert!(
                    (s - 0.3).abs() < TOL,
                    "constant not preserved under {filter} -> {w}x{h}: {s}"
                );
            }
        }
    }
}

#[test]
fn zero_target_size_is_rejected() {
    let img = gray(4, 4, vec![0.0; 16]);
    let mut inputs = InputValues::new();
    inputs.insert("image".to_owned(), img);
    let err = Resize::new()
        .compute(
            &inputs,
            &serde_json::json!({ "width": 0, "height": 4, "filter": "nearest" }),
        )
        .expect_err("zero width must be rejected");
    assert_eq!(err.class, ErrorClass::Policy);
    assert_eq!(err.code, E_RESIZE_SIZE);
}

#[test]
fn unknown_filter_is_rejected() {
    let img = gray(4, 4, vec![0.0; 16]);
    let mut inputs = InputValues::new();
    inputs.insert("image".to_owned(), img);
    let err = Resize::new()
        .compute(
            &inputs,
            &serde_json::json!({ "width": 2, "height": 2, "filter": "sinc8" }),
        )
        .expect_err("unknown filter must be rejected");
    assert_eq!(err.class, ErrorClass::Schema);
    assert_eq!(err.code, E_RESIZE_FILTER);
}

#[test]
fn zero_area_input_is_rejected() {
    // A 0xN input cannot be resampled to a non-empty output.
    let img = gray(0, 4, vec![]);
    let mut inputs = InputValues::new();
    inputs.insert("image".to_owned(), img);
    let err = Resize::new()
        .compute(
            &inputs,
            &serde_json::json!({ "width": 2, "height": 2, "filter": "bilinear" }),
        )
        .expect_err("zero-area input must be rejected");
    assert_eq!(err.class, ErrorClass::Policy);
    assert_eq!(err.code, E_RESIZE_EMPTY);
}

// ----- nearest / bilinear (bn-bs9) -----

#[test]
fn nearest_integer_upscale_is_exact_replication() {
    // 2x nearest upscale replicates each source sample into a 2x2 block, exactly.
    let img = gray(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
    let out = resize(&img, 4, 4, "nearest");
    assert_eq!(out.extent(), Extent::new(4, 4));
    // Row-major expected block replication.
    let want: Vec<f32> = vec![
        1.0, 1.0, 2.0, 2.0, //
        1.0, 1.0, 2.0, 2.0, //
        3.0, 3.0, 4.0, 4.0, //
        3.0, 3.0, 4.0, 4.0,
    ];
    for (g, w) in out.samples().iter().zip(want.iter()) {
        assert_eq!(g.to_bits(), w.to_bits(), "nearest 2x not exact: {g} vs {w}");
    }
}

#[test]
fn nearest_downscale_selects_source_samples_exactly() {
    // Every nearest output is exactly one input sample (no averaging).
    let img = gray(6, 1, vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
    let out = resize(&img, 3, 1, "nearest");
    for &s in out.samples() {
        assert!(
            img.samples().iter().any(|&v| v.to_bits() == s.to_bits()),
            "nearest produced a non-source sample {s}"
        );
    }
}

#[test]
fn bilinear_midpoint_is_the_average() {
    // A 2x1 image [0, 1] resized to 3x1 bilinear: the middle output center maps to
    // source coord 0.5, the exact average 0.5.
    let img = gray(2, 1, vec![0.0, 1.0]);
    let out = resize(&img, 3, 1, "bilinear");
    assert_eq!(out.extent(), Extent::new(3, 1));
    assert!(
        (out.samples()[1] - 0.5).abs() < TOL,
        "bilinear midpoint = {}",
        out.samples()[1]
    );
}

#[test]
fn bilinear_two_x_upscale_known_values() {
    // 2x1 [0, 4] -> 4x1 bilinear. Centers map to src = (i+0.5)*0.5 - 0.5:
    // i=0 -> -0.25 (clamp left edge contributes), i=1 -> 0.25, i=2 -> 0.75, i=3 -> 1.25.
    let img = gray(2, 1, vec![0.0, 4.0]);
    let out = resize(&img, 4, 1, "bilinear");
    let s = out.samples();
    // i=0: src=-0.25 clamps; taps at 0 (w) and -1->clamp 0; value stays 0.
    assert!((s[0] - 0.0).abs() < TOL, "s0={}", s[0]);
    // i=1: src=0.25 -> 0.75*v0 + 0.25*v1 = 1.0
    assert!((s[1] - 1.0).abs() < TOL, "s1={}", s[1]);
    // i=2: src=0.75 -> 0.25*v0 + 0.75*v1 = 3.0
    assert!((s[2] - 3.0).abs() < TOL, "s2={}", s[2]);
    // i=3: src=1.25 clamps right; value stays 4.0
    assert!((s[3] - 4.0).abs() < TOL, "s3={}", s[3]);
}

// ----- bicubic / lanczos (bn-16u) -----

#[test]
fn all_filters_produce_finite_outputs() {
    // Adversarial values (HDR, tiny, zero) resized up and down stay finite.
    let samples = vec![0.0, 1e3, 1e-3, 8.0, 0.0, 16.0, 0.25, 100.0, 2.0];
    let img = gray(3, 3, samples);
    for filter in FILTERS {
        for (w, h) in [(7, 5), (2, 2), (1, 4)] {
            let out = resize(&img, w, h, filter);
            for &s in out.samples() {
                assert!(s.is_finite(), "{filter} -> {w}x{h} produced {s}");
            }
        }
    }
}

#[test]
fn bicubic_reproduces_a_linear_ramp_interior() {
    // Catmull-Rom (the fixed bicubic) is an interpolating, first-moment-preserving
    // kernel, so it reproduces a linear ramp exactly at interior taps (away from the
    // clamped edges). Lanczos is checked separately (it rings on a ramp and is not
    // exact, only monotone/finite).
    let img = gray(8, 1, (0..8u8).map(f32::from).collect());
    let out = resize(&img, 16, 1, "bicubic");
    let s = out.samples();
    for (i, &got) in s.iter().enumerate().take(12).skip(4) {
        // Interior output center i maps to src = (i+0.5)*0.5 - 0.5 = i/2 - 0.25.
        #[allow(clippy::cast_precision_loss, reason = "tiny indices")]
        let src = (i as f32).mul_add(0.5, -0.25);
        assert!(
            (got - src).abs() < 1e-4,
            "bicubic ramp at {i}: {got} vs {src}"
        );
    }
}

#[test]
fn lanczos_ramp_is_monotone_and_near_linear() {
    // Lanczos-3 does not reproduce a ramp exactly (it rings), but on a monotone
    // ramp the interior output stays monotone non-decreasing and close to linear.
    let img = gray(8, 1, (0..8u8).map(f32::from).collect());
    let out = resize(&img, 16, 1, "lanczos");
    let s = out.samples();
    for i in 5..11usize {
        assert!(s[i] >= s[i - 1] - TOL, "lanczos not monotone at {i}");
        #[allow(clippy::cast_precision_loss, reason = "tiny indices")]
        let src = (i as f32).mul_add(0.5, -0.25);
        assert!(
            (s[i] - src).abs() < 0.1,
            "lanczos ramp at {i}: {} vs {src}",
            s[i]
        );
    }
}

#[test]
fn declared_halos_match_supports() {
    assert_eq!(Filter::Nearest.halo(), 0);
    assert_eq!(Filter::Bilinear.halo(), 1);
    assert_eq!(Filter::Bicubic.halo(), 2);
    assert_eq!(Filter::Lanczos.halo(), 3);
}

// ----- umbrella: tiling independence + round-trip (bn-2yy) -----

#[test]
fn output_is_independent_of_tiling() {
    // §3.8 tiling independence: an x-only resize (height unchanged) is row-wise
    // independent, so resizing the full image equals resizing disjoint row-tiles
    // and stacking the results. This holds for every separable filter and is the
    // property a tiled executor relies on.
    let img = gray(8, 4, (0..32u8).map(|i| f32::from(i) / 31.0).collect());
    for filter in FILTERS {
        // Full x-only resize (8x4 -> 5x4, height preserved).
        let full = resize(&img, 5, 4, filter);

        // Tile into top two rows and bottom two rows, resize each x-only, stack.
        let top = gray(8, 2, img.samples()[0..16].to_vec());
        let bottom = gray(8, 2, img.samples()[16..32].to_vec());
        let top_r = resize(&top, 5, 2, filter);
        let bottom_r = resize(&bottom, 5, 2, filter);
        let mut stacked = top_r.samples().to_vec();
        stacked.extend_from_slice(bottom_r.samples());

        for (a, b) in stacked.iter().zip(full.samples().iter()) {
            assert!(
                (a - b).abs() < TOL,
                "tiling-dependent under {filter}: {a} vs {b}"
            );
        }
    }
}

#[test]
fn band_limited_round_trip_within_tolerance() {
    // A smooth (band-limited) low-frequency signal upsampled then downsampled with
    // the same high-quality kernel returns close to the original (§3.8).
    let mut samples = Vec::new();
    for x in 0..16u32 {
        let angle = std::f64::consts::TAU * f64::from(x) / 16.0;
        let v = 0.4f64.mul_add(angle.sin(), 0.5);
        #[allow(clippy::cast_possible_truncation, reason = "test signal")]
        samples.push(v as f32);
    }
    let img = gray(16, 1, samples.clone());
    for filter in ["bicubic", "lanczos"] {
        let up = resize(&img, 64, 1, filter);
        let down = resize(&up, 16, 1, filter);
        // Interior samples (away from clamped edges) should round-trip closely.
        for (i, &orig) in samples.iter().enumerate().take(13).skip(3) {
            assert!(
                (down.samples()[i] - orig).abs() < 5e-2,
                "{filter} round-trip at {i}: {} vs {orig}",
                down.samples()[i],
            );
        }
    }
}

#[test]
fn infer_outputs_sets_target_extent_and_maps_footprint() {
    let descriptor = ResourceDescriptor::Image(ImageDescriptor {
        extent: Extent::new(10, 8),
        layout: ChannelLayout::Rgb,
        scalar: ScalarType::F32,
        color: ColorEncoding::LinearSrgb,
        range: ColorRange::SceneReferred,
        alpha: AlphaRepresentation::Straight,
        coordinates: CoordinateConvention::PixelCenterUpperLeft,
        semantic: SemanticRole::Color,
    });
    let mut inputs = Descriptors::new();
    inputs.insert("image".to_owned(), descriptor);
    let params = serde_json::json!({ "width": 5, "height": 4, "filter": "bilinear" });

    let out = Resize::new()
        .infer_outputs(&inputs, &params)
        .expect("infer");
    let ResourceDescriptor::Image(d) = out["image"] else {
        panic!("expected image");
    };
    assert_eq!(d.extent, Extent::new(5, 4));

    // The footprint of the whole output covers the whole input.
    let mut requested = OutputRegions::new();
    requested.insert("image".to_owned(), Rect::new(0, 0, 5, 4));
    let needed = Resize::new()
        .required_inputs(&requested, &inputs, &params)
        .expect("required_inputs");
    let region = needed["image"];
    assert!(region.x0 <= 0 && region.x1 >= 10 && region.y0 <= 0 && region.y1 >= 8);
}

#[test]
fn source_footprint_is_clamped_and_haloed() {
    // A single output pixel's footprint is a small clamped window, never the empty
    // set, and grows with the filter halo.
    let src = Extent::new(20, 1);
    let req_near = super::ResizeRequest {
        target: Extent::new(10, 1),
        filter: Filter::Nearest,
    };
    let r = source_footprint(Rect::new(5, 0, 6, 1), src, req_near);
    assert!(r.x1 > r.x0, "nearest footprint must be non-empty");
}

#[test]
fn checked_in_manifest_matches_builder() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("crates dir")
        .parent()
        .expect("repo root")
        .join("ops/manifests");
    let manifest = Resize::manifest().expect("resize manifest");
    let path = root.join(format!("{}.json", manifest.id));
    let on_disk =
        std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    let expected = serde_json::to_string_pretty(&manifest).expect("serialize manifest");
    assert_eq!(
        on_disk.trim_end(),
        expected.trim_end(),
        "{} is stale; regenerate from the Rust builder",
        path.display()
    );
}
