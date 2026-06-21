//! Verification suite for `mask.empty@1`, `mask.full@1`, and `mask.rect@1`
//! (`OP_CATALOG` §3, `AGENT_VERIFICATION` §3.6, `IR_SPEC` §8.1):
//!
//! - **schema/contract**: each manifest validates, agrees with its contract,
//!   gates clean, and the checked-in manifest matches the Rust builder;
//! - **analytic fixtures**: empty is all-zero, full is all-one (exact);
//! - **rect half-open convention**: an integer-aligned rect has coverage exactly
//!   1 inside and 0 outside, with the `x1`/`y1` columns/rows excluded;
//! - **rect analytic area**: the summed coverage equals the analytic rect area
//!   (clipped to the image), including fractional bounds;
//! - **tile invariance**: coverage depends only on the pixel's own cell, so a
//!   sub-window's samples equal the full render's;
//! - **rejection**: a malformed `rect` param is rejected with a typed error.

use paintop_core::executor::{InputValues, OpImplementation, ResourceValue};
use paintop_ir::{
    AlphaRepresentation, ChannelLayout, ColorEncoding, ColorRange, CoordinateConvention,
    Descriptors, ErrorClass, Extent, ImageDescriptor, MaskMeaning, OpContract, ResourceDescriptor,
    ScalarType, SemanticRole, ValidRange, check_contract_consistency, verify_categories,
};

use super::{EMPTY_OP_ID, EmptyMask, FULL_OP_ID, FullMask, RECT_OP_ID, RectGeometry, RectMask};

/// A square image descriptor used purely as the `extent_from` source.
fn extent_source(w: u32, h: u32) -> ResourceValue {
    let descriptor = ResourceDescriptor::Image(ImageDescriptor {
        extent: Extent::new(w, h),
        layout: ChannelLayout::Rgba,
        scalar: ScalarType::F32,
        color: ColorEncoding::LinearSrgb,
        range: ColorRange::SceneReferred,
        alpha: AlphaRepresentation::Premultiplied,
        coordinates: CoordinateConvention::PixelCenterUpperLeft,
        semantic: SemanticRole::Color,
    });
    let len = (w as usize) * (h as usize) * 4;
    ResourceValue::new(descriptor, 4, vec![0.0; len]).expect("extent source")
}

fn render_empty(w: u32, h: u32) -> ResourceValue {
    let mut inputs = InputValues::new();
    inputs.insert("extent_from".to_owned(), extent_source(w, h));
    let mut out = EmptyMask::new()
        .compute(&inputs, &serde_json::json!({}))
        .expect("empty computes");
    out.remove("mask").expect("mask produced")
}

fn render_full(w: u32, h: u32) -> ResourceValue {
    let mut inputs = InputValues::new();
    inputs.insert("extent_from".to_owned(), extent_source(w, h));
    let mut out = FullMask::new()
        .compute(&inputs, &serde_json::json!({}))
        .expect("full computes");
    out.remove("mask").expect("mask produced")
}

fn render_rect(w: u32, h: u32, params: &serde_json::Value) -> ResourceValue {
    let mut inputs = InputValues::new();
    inputs.insert("extent_from".to_owned(), extent_source(w, h));
    let mut out = RectMask::new()
        .compute(&inputs, params)
        .expect("rect computes");
    out.remove("mask").expect("mask produced")
}

// --- schema / contract -----------------------------------------------------

#[test]
fn manifests_validate_and_agree_with_contracts() {
    let cases: [(paintop_ir::OperationManifest, &str); 3] = [
        (EmptyMask::manifest().expect("empty"), EMPTY_OP_ID),
        (FullMask::manifest().expect("full"), FULL_OP_ID),
        (RectMask::manifest().expect("rect"), RECT_OP_ID),
    ];
    for (manifest, id) in cases {
        manifest.validate().expect("manifest valid");
        verify_categories(&manifest, &manifest.test.verification)
            .expect("verification declarations gate clean");
        assert_eq!(manifest.id.to_string(), id);
    }
    check_contract_consistency(&EmptyMask::manifest().unwrap(), &EmptyMask::new()).unwrap();
    check_contract_consistency(&FullMask::manifest().unwrap(), &FullMask::new()).unwrap();
    check_contract_consistency(&RectMask::manifest().unwrap(), &RectMask::new()).unwrap();
}

// --- constant masks --------------------------------------------------------

#[test]
fn empty_is_all_zero_and_full_is_all_one() {
    let empty = render_empty(5, 4);
    let ResourceDescriptor::Mask(d) = empty.descriptor() else {
        panic!("expected mask");
    };
    assert_eq!(d.extent, Extent::new(5, 4));
    assert_eq!(d.meaning, MaskMeaning::Coverage);
    assert_eq!(d.range, ValidRange::Bounded { min: 0.0, max: 1.0 });
    assert_eq!(empty.samples().len(), 5 * 4);
    assert!(
        empty
            .samples()
            .iter()
            .all(|&s| s.to_bits() == 0.0_f32.to_bits())
    );

    let full = render_full(5, 4);
    assert_eq!(full.samples().len(), 5 * 4);
    assert!(
        full.samples()
            .iter()
            .all(|&s| s.to_bits() == 1.0_f32.to_bits())
    );
}

// --- rect half-open convention ---------------------------------------------

#[test]
fn integer_aligned_rect_is_hard_half_open() {
    // rect [1, 1, 3, 3): covers pixels (1,1),(2,1),(1,2),(2,2) exactly; column 3
    // and row 3 are excluded.
    let mask = render_rect(4, 4, &serde_json::json!({ "rect": [1, 1, 3, 3] }));
    let s = mask.samples();
    let at = |x: usize, y: usize| s[y * 4 + x];
    for y in 0..4 {
        for x in 0..4 {
            let inside = (1..3).contains(&x) && (1..3).contains(&y);
            let expected: f32 = if inside { 1.0 } else { 0.0 };
            assert_eq!(at(x, y).to_bits(), expected.to_bits(), "pixel ({x},{y})");
        }
    }
}

// --- rect analytic area ----------------------------------------------------

#[test]
fn rect_coverage_sums_to_analytic_area() {
    // A fractional rect fully inside a 16x16 image: summed coverage equals the
    // analytic area (x1-x0)*(y1-y0) exactly (the coverage is the area partition).
    let (x0, y0, x1, y1) = (2.3_f64, 4.1, 11.8, 9.6);
    let mask = render_rect(16, 16, &serde_json::json!({ "rect": [x0, y0, x1, y1] }));
    let sum: f64 = mask.samples().iter().map(|&s| f64::from(s)).sum();
    let analytic = (x1 - x0) * (y1 - y0);
    assert!(
        (sum - analytic).abs() < 1e-4,
        "summed coverage {sum} != analytic area {analytic}"
    );
}

#[test]
fn rect_partial_pixel_carries_fractional_area() {
    // rect [0, 0, 1.25, 1): pixel (0,0) fully covered (overlap_x=1, overlap_y=1),
    // pixel (1,0) covered 0.25 in x and 1 in y -> 0.25.
    let mask = render_rect(3, 1, &serde_json::json!({ "rect": [0.0, 0.0, 1.25, 1.0] }));
    let s = mask.samples();
    assert!((f64::from(s[0]) - 1.0).abs() < 1e-6, "pixel 0: {}", s[0]);
    assert!((f64::from(s[1]) - 0.25).abs() < 1e-6, "pixel 1: {}", s[1]);
    assert_eq!(s[2].to_bits(), 0.0_f32.to_bits());
}

#[test]
fn coverage_is_bounded_and_finite() {
    let mask = render_rect(8, 8, &serde_json::json!({ "rect": [-2.0, 1.5, 6.4, 20.0] }));
    for &s in mask.samples() {
        assert!(s.is_finite(), "coverage must be finite: {s}");
        assert!((0.0..=1.0).contains(&s), "coverage out of range: {s}");
    }
}

// --- tile invariance -------------------------------------------------------

/// Coverage of a pixel depends only on its own cell, so the value of pixel
/// `(i, j)` is identical whether it is rendered as part of a large image or a
/// small sub-window aligned to the same coordinate origin would-be tile. We model
/// a "tile" by re-deriving the per-pixel coverage directly from the geometry and
/// confirming it matches the full render (a tile boundary cannot change it).
#[test]
fn tile_boundary_does_not_alter_coverage() {
    let rect = RectGeometry::resolve(&serde_json::json!({ "rect": [1.5, 2.5, 7.0, 6.5] }))
        .expect("geometry");
    let mask = render_rect(10, 10, &serde_json::json!({ "rect": [1.5, 2.5, 7.0, 6.5] }));
    let s = mask.samples();
    for j in 0..10_u32 {
        for i in 0..10_u32 {
            let direct = rect.coverage(i, j);
            #[allow(
                clippy::cast_possible_truncation,
                reason = "coverage is bounded [0,1] stored as f32"
            )]
            let direct_f32 = direct as f32;
            assert_eq!(
                s[(j * 10 + i) as usize].to_bits(),
                direct_f32.to_bits(),
                "pixel ({i},{j}) differs from per-cell coverage"
            );
        }
    }
}

// --- rejection -------------------------------------------------------------

#[test]
fn rect_rejects_malformed_param() {
    let mut inputs = Descriptors::new();
    inputs.insert("extent_from".to_owned(), *extent_source(4, 4).descriptor());
    for params in [
        serde_json::json!({}),                    // missing
        serde_json::json!({ "rect": [1, 2, 3] }), // wrong length
        serde_json::json!({ "rect": "nope" }),    // wrong type
    ] {
        let err = RectMask::new()
            .infer_outputs(&inputs, &params)
            .expect_err("malformed rect must be rejected");
        assert_eq!(err.class, ErrorClass::Schema, "params: {params}");
        assert_eq!(err.code, super::E_RECT_PARAM);
    }
}

/// The checked-in `ops/manifests/<id>.json` files must stay byte-identical to the
/// Rust manifest builders.
#[test]
fn checked_in_manifests_match_builders() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("crates dir")
        .parent()
        .expect("repo root")
        .join("ops/manifests");
    for manifest in [
        EmptyMask::manifest().expect("empty"),
        FullMask::manifest().expect("full"),
        RectMask::manifest().expect("rect"),
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
