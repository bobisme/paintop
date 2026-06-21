//! Verification suite for `mask.polygon@1` (`OP_CATALOG` §3,
//! `AGENT_VERIFICATION` §3.6, §2.9):
//!
//! - **schema/contract**: the manifest validates, agrees with its contract,
//!   gates clean, and the checked-in manifest matches the Rust builder;
//! - **half-open convention**: an integer-aligned square has coverage exactly 1
//!   inside and 0 outside, with the right/bottom edges excluded;
//! - **analytic area convergence**: a tilted triangle's summed coverage
//!   approaches its analytic area as the raster resolution increases;
//! - **fill rule**: a self-intersecting pentagram fills its centre under
//!   `nonzero` but carves it out under `even-odd`;
//! - **degenerate edges**: repeated / zero-length / collinear vertices produce
//!   finite, bounded coverage (no NaN);
//! - **rotation metamorphic**: a 90° rotation of the polygon equals the 90°
//!   rotation of the rendered mask;
//! - **rejection**: a malformed `points` / `fill_rule` param is a typed error.

use paintop_core::executor::{InputValues, OpImplementation, ResourceValue};
use paintop_ir::{
    AlphaRepresentation, ChannelLayout, ColorEncoding, ColorRange, CoordinateConvention,
    Descriptors, ErrorClass, Extent, ImageDescriptor, OpContract, ResourceDescriptor, ScalarType,
    SemanticRole, check_contract_consistency, verify_categories,
};

use super::{POLYGON_OP_ID, PolygonMask};

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

/// Render a polygon mask and return its samples.
fn render(w: u32, h: u32, params: &serde_json::Value) -> Vec<f32> {
    let mut inputs = InputValues::new();
    inputs.insert("extent_from".to_owned(), extent_source(w, h));
    let mut out = PolygonMask::new()
        .compute(&inputs, params)
        .expect("polygon computes");
    out.remove("mask").expect("mask").into_samples()
}

// --- schema / contract -----------------------------------------------------

#[test]
fn manifest_validates_and_agrees_with_contract() {
    let manifest = PolygonMask::manifest().expect("manifest");
    manifest.validate().expect("manifest valid");
    verify_categories(&manifest, &manifest.test.verification)
        .expect("verification declarations gate clean");
    assert_eq!(manifest.id.to_string(), POLYGON_OP_ID);
    check_contract_consistency(&manifest, &PolygonMask::new()).unwrap();
}

// --- half-open convention --------------------------------------------------

#[test]
fn integer_square_is_hard_half_open() {
    // The square [1,4) x [1,4): pixels (1..3, 1..3) are fully inside; the column
    // x=4 / row y=4 are excluded by the half-open ray cast.
    let params = serde_json::json!({
        "points": [[1, 1], [4, 1], [4, 4], [1, 4]],
        "fill_rule": "nonzero"
    });
    let w = 6;
    let s = render(w, 6, &params);
    let at = |x: usize, y: usize| s[y * (w as usize) + x];
    for y in 0..6 {
        for x in 0..6 {
            let inside = (1..4).contains(&x) && (1..4).contains(&y);
            let expected: f32 = if inside { 1.0 } else { 0.0 };
            assert_eq!(at(x, y).to_bits(), expected.to_bits(), "pixel ({x},{y})");
        }
    }
}

// --- analytic area convergence ---------------------------------------------

/// The summed coverage of a fixed tilted triangle, rendered with the polygon
/// scaled by `scale` into a `scale*base` raster, in *base-unit²* area.
fn triangle_area_at(scale: f64, size: u32) -> f64 {
    // A triangle with analytic area 0.5 * base * height in base units. We scale
    // both the geometry and the raster by `scale`, so the rendered area in raster
    // pixels divided by scale² recovers the base-unit area.
    let p = |x: f64, y: f64| serde_json::json!([x * scale, y * scale]);
    let params = serde_json::json!({
        "points": [p(1.0, 1.0), p(7.0, 2.0), p(3.0, 7.0)],
        "fill_rule": "nonzero"
    });
    let samples = render(size, size, &params);
    let raster_area: f64 = samples.iter().map(|&s| f64::from(s)).sum();
    raster_area / (scale * scale)
}

#[test]
fn area_converges_to_analytic_with_resolution() {
    // Analytic area of triangle (1,1),(7,2),(3,7) via the shoelace formula
    // 0.5 * |x0(y1-y2) + x1(y2-y0) + x2(y0-y1)|
    //     = 0.5 * |1*(2-7) + 7*(7-1) + 3*(1-2)| = 0.5 * |-5 + 42 - 3| = 17.
    let analytic: f64 = 17.0;
    let coarse = (triangle_area_at(1.0, 8) - analytic).abs();
    let fine = (triangle_area_at(8.0, 64) - analytic).abs();
    assert!(
        fine < coarse,
        "area error must shrink with resolution: coarse {coarse}, fine {fine}"
    );
    assert!(fine < 0.5, "fine area error too large: {fine}");
}

// --- fill rule on a self-intersecting polygon ------------------------------

/// A pentagram (5-pointed star) centred at `(c, c)` with circumradius `r`, traced
/// in the star order that makes the edges self-intersect.
fn pentagram(c: f64, r: f64) -> serde_json::Value {
    use std::f64::consts::PI;
    // Vertices every 144° (2 steps of 72°) give the classic self-intersecting
    // star path.
    let mut pts = Vec::new();
    for k in 0..5 {
        let theta = (f64::from(k) * 2.0).mul_add(2.0 * PI / 5.0, -PI / 2.0);
        pts.push(serde_json::json!([
            r.mul_add(theta.cos(), c),
            r.mul_add(theta.sin(), c)
        ]));
    }
    serde_json::Value::Array(pts)
}

#[test]
fn fill_rule_changes_pentagram_centre() {
    let size = 40;
    let centre_index = ((size / 2) * size + (size / 2)) as usize;
    let points = pentagram(20.0, 15.0);

    let nonzero = render(
        size,
        size,
        &serde_json::json!({ "points": points, "fill_rule": "nonzero" }),
    );
    let even_odd = render(
        size,
        size,
        &serde_json::json!({ "points": points, "fill_rule": "even-odd" }),
    );

    // The central pentagon is wound twice: nonzero fills it, even-odd carves it
    // out (a hole).
    assert!(
        nonzero[centre_index] > 0.5,
        "nonzero must fill the star centre: {}",
        nonzero[centre_index]
    );
    assert!(
        even_odd[centre_index] < 0.5,
        "even-odd must carve out the star centre: {}",
        even_odd[centre_index]
    );

    // The total filled area under nonzero strictly exceeds even-odd (it includes
    // the central pentagon).
    let area_nz: f64 = nonzero.iter().map(|&s| f64::from(s)).sum();
    let area_eo: f64 = even_odd.iter().map(|&s| f64::from(s)).sum();
    assert!(
        area_nz > area_eo,
        "nonzero area {area_nz} <= even-odd {area_eo}"
    );
}

// --- degenerate edges (no NaN) ---------------------------------------------

#[test]
fn degenerate_edges_produce_finite_coverage() {
    for points in [
        // Repeated / zero-length edge.
        serde_json::json!([[1, 1], [1, 1], [5, 1], [5, 5], [1, 5]]),
        // Collinear vertices on an edge.
        serde_json::json!([[1, 1], [3, 1], [5, 1], [5, 5], [1, 5]]),
        // A self-touching "bowtie" sharing the centre vertex.
        serde_json::json!([[0, 0], [6, 6], [6, 0], [0, 6]]),
        // A spike that doubles back on itself.
        serde_json::json!([[1, 1], [5, 1], [3, 1], [3, 5]]),
        // Fewer than three vertices: a well-defined empty mask.
        serde_json::json!([[1, 1], [4, 4]]),
    ] {
        for rule in ["nonzero", "even-odd"] {
            let s = render(
                8,
                8,
                &serde_json::json!({ "points": points, "fill_rule": rule }),
            );
            for &v in &s {
                assert!(v.is_finite(), "coverage must be finite ({rule}): {v}");
                assert!(
                    (0.0..=1.0).contains(&v),
                    "coverage out of range ({rule}): {v}"
                );
            }
        }
    }
}

#[test]
fn fewer_than_three_vertices_is_empty() {
    let s = render(6, 6, &serde_json::json!({ "points": [[1, 1], [4, 4]] }));
    assert!(s.iter().all(|&v| v.to_bits() == 0.0_f32.to_bits()));
}

// --- rotation metamorphic --------------------------------------------------

/// Rotate a square mask 90° clockwise: out(x, y) = in(y, W-1-x) for a W×W mask.
fn rotate90_cw(samples: &[f32], n: u32) -> Vec<f32> {
    let n = n as usize;
    let mut out = vec![0.0_f32; n * n];
    for y in 0..n {
        for x in 0..n {
            out[y * n + x] = samples[(n - 1 - x) * n + y];
        }
    }
    out
}

#[test]
fn rotation_covariance() {
    // A polygon and its 90°-clockwise rotation about the square's centre render to
    // 90°-rotated masks. For a W×W image the rotation that maps pixel (x, y) to
    // (W-1-y, x) is realized in continuous coords by (px, py) -> (W - py, px).
    let n = 16;
    let nf = f64::from(n);
    let base_pts: [(f64, f64); 3] = [(3.0, 2.0), (12.0, 4.0), (6.0, 13.0)];
    let to_json = |pts: &[(f64, f64)]| -> serde_json::Value {
        serde_json::Value::Array(
            pts.iter()
                .map(|&(x, y)| serde_json::json!([x, y]))
                .collect(),
        )
    };
    let rot_pts: Vec<(f64, f64)> = base_pts.iter().map(|&(x, y)| (nf - y, x)).collect();

    let base = render(n, n, &serde_json::json!({ "points": to_json(&base_pts) }));
    let rotated = render(n, n, &serde_json::json!({ "points": to_json(&rot_pts) }));
    let base_then_rotated = rotate90_cw(&base, n);

    for (i, (&a, &b)) in rotated.iter().zip(&base_then_rotated).enumerate() {
        assert_eq!(a.to_bits(), b.to_bits(), "pixel {i}: {a} != {b}");
    }
}

// --- rejection -------------------------------------------------------------

#[test]
fn rejects_malformed_params() {
    let mut inputs = Descriptors::new();
    inputs.insert("extent_from".to_owned(), *extent_source(4, 4).descriptor());
    for params in [
        serde_json::json!({}),                        // missing points
        serde_json::json!({ "points": "nope" }),      // wrong type
        serde_json::json!({ "points": [[1, 2, 3]] }), // vertex wrong length
        serde_json::json!({ "points": [[1, 1], [2, 2], [3, 3]], "fill_rule": "bogus" }),
    ] {
        let err = PolygonMask::new()
            .infer_outputs(&inputs, &params)
            .expect_err("malformed param must be rejected");
        assert_eq!(err.class, ErrorClass::Schema, "params: {params}");
        assert_eq!(err.code, super::E_POLYGON_PARAM);
    }
}

/// The checked-in `ops/manifests/mask.polygon@1.json` must stay byte-identical to
/// the Rust manifest builder.
#[test]
fn checked_in_manifest_matches_builder() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("crates dir")
        .parent()
        .expect("repo root")
        .join("ops/manifests");
    let manifest = PolygonMask::manifest().expect("manifest");
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
