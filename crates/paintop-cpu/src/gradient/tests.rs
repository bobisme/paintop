//! Verification suite for `paint.linear_gradient@1` and
//! `paint.radial_gradient@1` (`OP_CATALOG` §6):
//!
//! - **schema/contract**: each manifest validates, agrees with its contract, and
//!   its declarations gate clean; the checked-in manifests match the builders;
//! - **stop exactness**: a pixel whose parameter equals a stop position
//!   reproduces that stop's color bit-exactly;
//! - **interpolation**: the midpoint between two stops is their per-channel
//!   average, in the declared color space;
//! - **monotonicity**: the parameter (and so a single-component gradient) is
//!   monotone along the gradient axis / radius;
//! - **metamorphic (linear)**: translating both endpoints and the sample by the
//!   same vector leaves the value unchanged (translation covariance);
//! - **degenerate handling**: a zero-length linear axis and a non-positive radial
//!   radius are rejected;
//! - **rejection**: malformed / unsorted / non-spanning stops, and an
//!   out-of-range stop color, are rejected.

use paintop_core::executor::{InputValues, OpImplementation, ResourceValue};
use paintop_ir::{
    AlphaRepresentation, ChannelLayout, ColorEncoding, ColorRange, CoordinateConvention,
    Descriptors, ErrorClass, Extent, ImageDescriptor, OpContract, ResourceDescriptor, ScalarType,
    SemanticRole, check_contract_consistency, verify_categories,
};

use super::{
    E_GRADIENT_PARAM, E_GRADIENT_STOPS, LINEAR_GRADIENT_OP_ID, LinearGradient,
    RADIAL_GRADIENT_OP_ID, RadialGradient,
};

/// An RGBA image descriptor of `width × height` (used only as the `extent_from`
/// source; its samples are never read).
fn rgba_descriptor(width: u32, height: u32) -> ImageDescriptor {
    ImageDescriptor {
        extent: Extent::new(width, height),
        layout: ChannelLayout::Rgba,
        scalar: ScalarType::F32,
        color: ColorEncoding::Srgb,
        range: ColorRange::DisplayReferred,
        alpha: AlphaRepresentation::Straight,
        coordinates: CoordinateConvention::PixelCenterUpperLeft,
        semantic: SemanticRole::Color,
    }
}

/// An `extent_from` source image of `width × height`.
fn extent_source(width: u32, height: u32) -> ResourceValue {
    let len = (width as usize) * (height as usize) * 4;
    ResourceValue::new(
        ResourceDescriptor::Image(rgba_descriptor(width, height)),
        4,
        vec![0.0; len],
    )
    .expect("extent source")
}

/// A black-to-white RGBA stop list spanning [0, 1].
fn black_white_stops() -> serde_json::Value {
    serde_json::json!([
        { "position": 0.0, "color": [0.0, 0.0, 0.0, 1.0] },
        { "position": 1.0, "color": [1.0, 1.0, 1.0, 1.0] },
    ])
}

/// Run a linear gradient over a `width × height` canvas with the given params.
fn run_linear(width: u32, height: u32, params: &serde_json::Value) -> ResourceValue {
    let mut inputs = InputValues::new();
    inputs.insert("extent_from".to_owned(), extent_source(width, height));
    let mut out = LinearGradient::new()
        .compute(&inputs, params)
        .expect("linear gradient computes");
    out.remove("image").expect("image produced")
}

/// Run a radial gradient over a `width × height` canvas with the given params.
fn run_radial(width: u32, height: u32, params: &serde_json::Value) -> ResourceValue {
    let mut inputs = InputValues::new();
    inputs.insert("extent_from".to_owned(), extent_source(width, height));
    let mut out = RadialGradient::new()
        .compute(&inputs, params)
        .expect("radial gradient computes");
    out.remove("image").expect("image produced")
}

/// The RGBA pixel at `(x, y)` of a row-major RGBA `width`-wide buffer.
fn pixel(samples: &[f32], width: u32, x: u32, y: u32) -> [f32; 4] {
    let idx = ((y * width + x) as usize) * 4;
    [
        samples[idx],
        samples[idx + 1],
        samples[idx + 2],
        samples[idx + 3],
    ]
}

/// Assert a pixel equals an expected RGBA color bit-exactly.
fn assert_pixel_bits(got: [f32; 4], expected: [f32; 4], label: &str) {
    for c in 0..4 {
        assert_eq!(
            got[c].to_bits(),
            expected[c].to_bits(),
            "{label}: channel {c} got {} expected {}",
            got[c],
            expected[c],
        );
    }
}

#[test]
fn linear_manifest_validates_and_agrees_with_contract() {
    let manifest = LinearGradient::manifest().expect("linear manifest");
    manifest.validate().expect("manifest valid");
    check_contract_consistency(&manifest, &LinearGradient::new()).expect("contract agrees");
    verify_categories(&manifest, &manifest.test.verification).expect("declarations gate clean");
    assert_eq!(manifest.id.to_string(), LINEAR_GRADIENT_OP_ID);
}

#[test]
fn radial_manifest_validates_and_agrees_with_contract() {
    let manifest = RadialGradient::manifest().expect("radial manifest");
    manifest.validate().expect("manifest valid");
    check_contract_consistency(&manifest, &RadialGradient::new()).expect("contract agrees");
    verify_categories(&manifest, &manifest.test.verification).expect("declarations gate clean");
    assert_eq!(manifest.id.to_string(), RADIAL_GRADIENT_OP_ID);
}

/// Stop-exactness: a horizontal gradient with endpoints at the left/right pixel
/// centers reproduces the end stop colors bit-exactly at the endpoints.
#[test]
fn linear_reproduces_stops_at_endpoints() {
    // Endpoints at pixel centers (0.5, 0.5) and (3.5, 0.5) of a 4×1 canvas.
    let out = run_linear(
        4,
        1,
        &serde_json::json!({
            "start_px": [0.5, 0.5],
            "end_px": [3.5, 0.5],
            "stops": black_white_stops(),
        }),
    );
    let s = out.samples();
    // t = 0 at x = 0 -> black; t = 1 at x = 3 -> white, both exact.
    assert_pixel_bits(pixel(s, 4, 0, 0), [0.0, 0.0, 0.0, 1.0], "start stop");
    assert_pixel_bits(pixel(s, 4, 3, 0), [1.0, 1.0, 1.0, 1.0], "end stop");
}

/// Stop-exactness with an interior stop: a pixel landing exactly on a stop
/// position takes that stop's color verbatim.
#[test]
fn linear_reproduces_interior_stop_exactly() {
    // 3-stop gradient: red at 0, green at 0.5, blue at 1. A 3-wide canvas with
    // endpoints at the first/last pixel centers puts the middle pixel at t = 0.5.
    let out = run_linear(
        3,
        1,
        &serde_json::json!({
            "start_px": [0.5, 0.5],
            "end_px": [2.5, 0.5],
            "stops": [
                { "position": 0.0, "color": [1.0, 0.0, 0.0, 1.0] },
                { "position": 0.5, "color": [0.0, 1.0, 0.0, 1.0] },
                { "position": 1.0, "color": [0.0, 0.0, 1.0, 1.0] },
            ],
        }),
    );
    let s = out.samples();
    assert_pixel_bits(pixel(s, 3, 1, 0), [0.0, 1.0, 0.0, 1.0], "middle stop exact");
}

/// Interpolation: the midpoint between two stops is their per-channel average.
#[test]
fn linear_midpoint_is_average() {
    // 2-wide canvas, endpoints at center of pixel 0 and just past pixel 1 so the
    // two pixel centers sit at t = 0.25 and t = 0.75.
    let out = run_linear(
        2,
        1,
        &serde_json::json!({
            "start_px": [0.0, 0.5],
            "end_px": [2.0, 0.5],
            "stops": black_white_stops(),
        }),
    );
    let s = out.samples();
    // Pixel center x=0.5 -> t=0.25; x=1.5 -> t=0.75.
    let left = pixel(s, 2, 0, 0);
    let right = pixel(s, 2, 1, 0);
    for c in 0..3 {
        assert!((left[c] - 0.25).abs() < 1e-6, "left {left:?}");
        assert!((right[c] - 0.75).abs() < 1e-6, "right {right:?}");
    }
}

/// Monotonicity: a single-component (gray) gradient is non-decreasing left to
/// right along the axis.
#[test]
fn linear_is_monotone_along_axis() {
    let out = run_linear(
        16,
        1,
        &serde_json::json!({
            "start_px": [0.5, 0.5],
            "end_px": [15.5, 0.5],
            "layout": "gray",
            "stops": [
                { "position": 0.0, "color": [0.0] },
                { "position": 1.0, "color": [1.0] },
            ],
        }),
    );
    let s = out.samples();
    for x in 1..16 {
        assert!(s[x] >= s[x - 1], "not monotone at x={x}: {s:?}");
    }
}

/// Translation covariance: shifting both endpoints by a vector and reading the
/// correspondingly-shifted pixel yields the same value.
#[test]
fn linear_translation_covariant() {
    let base = run_linear(
        8,
        8,
        &serde_json::json!({
            "start_px": [0.5, 0.5],
            "end_px": [6.5, 0.5],
            "layout": "gray",
            "stops": [
                { "position": 0.0, "color": [0.0] },
                { "position": 1.0, "color": [1.0] },
            ],
        }),
    );
    let shifted = run_linear(
        8,
        8,
        &serde_json::json!({
            "start_px": [1.5, 2.5],
            "end_px": [7.5, 2.5],
            "layout": "gray",
            "stops": [
                { "position": 0.0, "color": [0.0] },
                { "position": 1.0, "color": [1.0] },
            ],
        }),
    );
    let b = base.samples();
    let sh = shifted.samples();
    // base pixel (x, 0) == shifted pixel (x+1, 2) for x in 0..7.
    for x in 0_usize..7 {
        let bv = b[x];
        let sv = sh[(2 * 8) + (x + 1)];
        assert!((bv - sv).abs() < 1e-6, "x={x}: base {bv} != shifted {sv}");
    }
}

/// Radial: the center pixel is the first stop and the parameter grows monotonically
/// with distance.
#[test]
fn radial_center_is_first_stop_and_grows() {
    let out = run_radial(
        9,
        9,
        &serde_json::json!({
            "center_px": [4.5, 4.5],
            "radius_px": 4.0,
            "layout": "gray",
            "stops": [
                { "position": 0.0, "color": [0.0] },
                { "position": 1.0, "color": [1.0] },
            ],
        }),
    );
    let s = out.samples();
    // Center pixel (4,4) center is exactly (4.5,4.5) -> t = 0 -> first stop.
    assert_eq!(s[4 * 9 + 4].to_bits(), 0.0_f32.to_bits());
    // Moving out along the row is non-decreasing from the center.
    for x in 5_usize..9 {
        assert!(
            s[4 * 9 + x] >= s[4 * 9 + (x - 1)],
            "radial not monotone outward at x={x}"
        );
    }
}

/// A zero-length linear axis (`start == end`) is rejected.
#[test]
fn linear_degenerate_axis_is_rejected() {
    let mut inputs = InputValues::new();
    inputs.insert("extent_from".to_owned(), extent_source(4, 4));
    let err = LinearGradient::new()
        .compute(
            &inputs,
            &serde_json::json!({
                "start_px": [1.0, 1.0],
                "end_px": [1.0, 1.0],
                "stops": black_white_stops(),
            }),
        )
        .expect_err("degenerate axis must be rejected");
    assert_eq!(err.class, ErrorClass::Schema);
    assert_eq!(err.code, E_GRADIENT_PARAM);
}

/// A non-positive radial radius is rejected.
#[test]
fn radial_zero_radius_is_rejected() {
    let mut inputs = InputValues::new();
    inputs.insert("extent_from".to_owned(), extent_source(4, 4));
    let err = RadialGradient::new()
        .compute(
            &inputs,
            &serde_json::json!({
                "center_px": [2.0, 2.0],
                "radius_px": 0.0,
                "stops": black_white_stops(),
            }),
        )
        .expect_err("zero radius must be rejected");
    assert_eq!(err.class, ErrorClass::Schema);
    assert_eq!(err.code, E_GRADIENT_PARAM);
}

/// Unsorted stops are rejected.
#[test]
fn unsorted_stops_are_rejected() {
    let mut inputs = InputValues::new();
    inputs.insert("extent_from".to_owned(), extent_source(4, 4));
    let err = LinearGradient::new()
        .compute(
            &inputs,
            &serde_json::json!({
                "start_px": [0.5, 0.5],
                "end_px": [3.5, 0.5],
                "stops": [
                    { "position": 0.0, "color": [0.0, 0.0, 0.0, 1.0] },
                    { "position": 0.8, "color": [0.5, 0.5, 0.5, 1.0] },
                    { "position": 0.3, "color": [1.0, 1.0, 1.0, 1.0] },
                    { "position": 1.0, "color": [1.0, 1.0, 1.0, 1.0] },
                ],
            }),
        )
        .expect_err("unsorted stops must be rejected");
    assert_eq!(err.class, ErrorClass::Schema);
    assert_eq!(err.code, E_GRADIENT_STOPS);
}

/// Stops that do not span [0, 1] are rejected (no implicit extrapolation).
#[test]
fn non_spanning_stops_are_rejected() {
    let mut inputs = InputValues::new();
    inputs.insert("extent_from".to_owned(), extent_source(4, 4));
    let err = LinearGradient::new()
        .compute(
            &inputs,
            &serde_json::json!({
                "start_px": [0.5, 0.5],
                "end_px": [3.5, 0.5],
                "stops": [
                    { "position": 0.2, "color": [0.0, 0.0, 0.0, 1.0] },
                    { "position": 0.9, "color": [1.0, 1.0, 1.0, 1.0] },
                ],
            }),
        )
        .expect_err("non-spanning stops must be rejected");
    assert_eq!(err.class, ErrorClass::Schema);
    assert_eq!(err.code, E_GRADIENT_STOPS);
}

/// An out-of-range stop color for a display-referred image is rejected.
#[test]
fn out_of_range_stop_color_is_rejected() {
    let mut inputs = InputValues::new();
    inputs.insert("extent_from".to_owned(), extent_source(4, 4));
    let err = LinearGradient::new()
        .compute(
            &inputs,
            &serde_json::json!({
                "start_px": [0.5, 0.5],
                "end_px": [3.5, 0.5],
                "stops": [
                    { "position": 0.0, "color": [0.0, 0.0, 0.0, 1.0] },
                    { "position": 1.0, "color": [1.5, 1.0, 1.0, 1.0] },
                ],
            }),
        )
        .expect_err("out-of-range stop color must be rejected");
    assert_eq!(err.class, ErrorClass::Schema);
    assert_eq!(err.code, E_GRADIENT_STOPS);
}

/// `infer_outputs` produces an RGBA image of the source extent.
#[test]
fn infer_outputs_sizes_from_extent() {
    let mut inputs = Descriptors::new();
    inputs.insert(
        "extent_from".to_owned(),
        ResourceDescriptor::Image(rgba_descriptor(12, 7)),
    );
    let outputs = LinearGradient::new()
        .infer_outputs(
            &inputs,
            &serde_json::json!({
                "start_px": [0.5, 0.5],
                "end_px": [11.5, 0.5],
                "stops": black_white_stops(),
            }),
        )
        .expect("infer");
    let ResourceDescriptor::Image(d) = outputs.get("image").expect("image out") else {
        panic!("expected image");
    };
    assert_eq!(d.extent, Extent::new(12, 7));
    assert_eq!(d.layout, ChannelLayout::Rgba);
}

/// The checked-in `ops/manifests/<id>.json` files must match the Rust builders.
#[test]
fn checked_in_manifests_match_builders() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("crates dir")
        .parent()
        .expect("repo root")
        .join("ops/manifests");
    for manifest in [
        LinearGradient::manifest().expect("linear manifest"),
        RadialGradient::manifest().expect("radial manifest"),
    ] {
        let path = root.join(format!("{}.json", manifest.id));
        let on_disk = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
        let expected = serde_json::to_string_pretty(&manifest).expect("serialize manifest");
        assert_eq!(
            on_disk.trim_end(),
            expected.trim_end(),
            "{} is stale; regenerate from the Rust builder",
            path.display()
        );
    }
}
