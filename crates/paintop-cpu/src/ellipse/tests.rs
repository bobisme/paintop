//! Verification suite for `mask.ellipse@1` (`OP_CATALOG` §3,
//! `AGENT_VERIFICATION` §3.6, `M0_DECISIONS` D1):
//!
//! - **schema/contract**: the manifest validates, agrees with its contract, and
//!   its verification declarations gate clean; the checked-in manifest stays in
//!   lockstep with the Rust builder;
//! - **analytic fixtures**: rendered coverage area converges to the analytic
//!   ellipse area `π·rx·ry` as resolution increases;
//! - **metamorphic**: a 90° rotation of a circle is covariant (a circle is
//!   invariant); translating the center translates the coverage;
//! - **property**: the soft-edge transition width equals `2·half_width_px` in
//!   physical pixels, coverage stays in `[0, 1]`, and no sample is `NaN`;
//! - **fixture (half-open)**: a hard edge covers a pixel center exactly on the
//!   boundary and excludes one just outside;
//! - **rejection**: degenerate (non-positive / non-finite) radii and unsupported
//!   antialias / edge profiles are rejected with `schema` errors.

use paintop_core::executor::{InputValues, OpImplementation, ResourceValue};
use paintop_ir::{
    AlphaRepresentation, AssertionStatus, ChannelLayout, ColorEncoding, ColorRange,
    CoordinateConvention, Descriptors, ErrorClass, Extent, ImageDescriptor, MaskMeaning,
    OpContract, OutputRegions, Rect, ResourceDescriptor, ScalarType, SemanticRole, ValidRange,
    check_contract_consistency, verify_categories,
};

use super::{ELLIPSE_OP_ID, EllipseGeometry, EllipseMask, feather};

/// Coverage tolerance for the `bounded` (sqrt-based) feather.
const TOL: f64 = 1e-6;

/// A square image descriptor of side `n`, used purely as the `extent_from`
/// source (its samples are never read by the mask).
fn extent_source(n: u32) -> ResourceValue {
    let descriptor = ResourceDescriptor::Image(ImageDescriptor {
        extent: Extent::new(n, n),
        layout: ChannelLayout::Rgba,
        scalar: ScalarType::F32,
        color: ColorEncoding::LinearSrgb,
        range: ColorRange::SceneReferred,
        alpha: AlphaRepresentation::Premultiplied,
        coordinates: CoordinateConvention::PixelCenterUpperLeft,
        semantic: SemanticRole::Color,
    });
    let channels = ChannelLayout::Rgba.channel_count();
    let len = (n as usize) * (n as usize) * channels as usize;
    ResourceValue::new(descriptor, channels, vec![0.0; len]).expect("extent source")
}

/// Run the kernel and recover the produced mask value.
fn render(extent: u32, params: &serde_json::Value) -> ResourceValue {
    let mut inputs = InputValues::new();
    inputs.insert("extent_from".to_owned(), extent_source(extent));
    let mut out = EllipseMask::new()
        .compute(&inputs, params)
        .expect("ellipse computes");
    out.remove("mask").expect("mask port produced")
}

#[test]
fn manifest_validates_and_agrees_with_contract() {
    let manifest = EllipseMask::manifest().expect("ellipse manifest");
    manifest.validate().expect("ellipse manifest valid");
    check_contract_consistency(&manifest, &EllipseMask::new())
        .expect("ellipse manifest agrees with contract");
    verify_categories(&manifest, &manifest.test.verification)
        .expect("ellipse verification declarations gate clean");
    assert_eq!(manifest.id.to_string(), ELLIPSE_OP_ID);
}

#[test]
fn produces_a_coverage_mask_descriptor() {
    let params = serde_json::json!({
        "center_px": [8.0, 8.0],
        "radii_px": [4.0, 4.0],
    });
    let mask = render(16, &params);
    let ResourceDescriptor::Mask(d) = mask.descriptor() else {
        panic!("expected mask output");
    };
    assert_eq!(d.extent, Extent::new(16, 16));
    assert_eq!(d.meaning, MaskMeaning::Coverage);
    assert_eq!(d.scalar, ScalarType::F32);
    assert_eq!(d.range, ValidRange::Bounded { min: 0.0, max: 1.0 });
    assert_eq!(mask.channels(), 1);
    assert_eq!(mask.samples().len(), 16 * 16);
}

#[test]
fn coverage_is_bounded_and_finite() {
    // A rotated, feathered ellipse: every sample must be a finite [0, 1] value.
    let params = serde_json::json!({
        "center_px": [33.0, 24.0],
        "radii_px": [18.0, 7.0],
        "angle_rad": 0.6,
        "edge": { "profile": "smoothstep", "half_width_px": 3.0 },
    });
    let mask = render(64, &params);
    for &s in mask.samples() {
        assert!(s.is_finite(), "coverage must be finite, got {s}");
        assert!((0.0..=1.0).contains(&s), "coverage out of range: {s}");
    }
}

/// Analytic area convergence (`AGENT_VERIFICATION` §3.6): the summed hard-edge
/// coverage approaches the analytic area `π·rx·ry` as resolution grows.
#[test]
fn rendered_area_converges_to_analytic_area() {
    let rx = 0.30_f64;
    let ry = 0.18_f64;
    // Measure the coverage fraction of a centered ellipse at increasing
    // resolution; it must approach π·rx·ry (in unit-square fractions).
    let analytic = std::f64::consts::PI * rx * ry;
    let mut prev_err = f64::INFINITY;
    for &n in &[64_u32, 128, 256] {
        #[allow(clippy::cast_precision_loss, reason = "small test extents")]
        let side = f64::from(n);
        let params = serde_json::json!({
            "center_px": [side / 2.0, side / 2.0],
            "radii_px": [rx * side, ry * side],
        });
        let mask = render(n, &params);
        let covered: f64 = mask.samples().iter().map(|&s| f64::from(s)).sum();
        let fraction = covered / (side * side);
        let err = (fraction - analytic).abs();
        assert!(
            err < prev_err + 1e-4,
            "area error did not shrink at n={n}: {err} vs {prev_err}"
        );
        prev_err = err;
    }
    // The finest grid is close to the analytic area.
    assert!(prev_err < 1e-3, "area not converged: err={prev_err}");
}

/// 90° rotation covariance (`AGENT_VERIFICATION` §3.6): a circle is invariant
/// under a 90° rotation of its (equal) axes.
#[test]
fn circle_is_invariant_under_ninety_degree_rotation() {
    let base = serde_json::json!({
        "center_px": [16.0, 16.0],
        "radii_px": [9.0, 9.0],
        "angle_rad": 0.0,
        "edge": { "profile": "smoothstep", "half_width_px": 2.0 },
    });
    let rotated = serde_json::json!({
        "center_px": [16.0, 16.0],
        "radii_px": [9.0, 9.0],
        "angle_rad": std::f64::consts::FRAC_PI_2,
        "edge": { "profile": "smoothstep", "half_width_px": 2.0 },
    });
    let a = render(32, &base);
    let b = render(32, &rotated);
    for (x, y) in a.samples().iter().zip(b.samples().iter()) {
        assert!((f64::from(*x) - f64::from(*y)).abs() < TOL, "{x} vs {y}");
    }
}

/// 90° rotation covariance for an anisotropic ellipse: rotating the ellipse 90°
/// is the same as swapping its radii (`AGENT_VERIFICATION` §3.6).
#[test]
fn quarter_turn_equals_swapping_radii() {
    let rotated = serde_json::json!({
        "center_px": [16.0, 16.0],
        "radii_px": [10.0, 4.0],
        "angle_rad": std::f64::consts::FRAC_PI_2,
    });
    let swapped = serde_json::json!({
        "center_px": [16.0, 16.0],
        "radii_px": [4.0, 10.0],
        "angle_rad": 0.0,
    });
    let a = render(32, &rotated);
    let b = render(32, &swapped);
    for (x, y) in a.samples().iter().zip(b.samples().iter()) {
        assert!((f64::from(*x) - f64::from(*y)).abs() < TOL, "{x} vs {y}");
    }
}

/// Translation metamorphic: shifting the center by an integer pixel offset
/// shifts the coverage by the same offset.
#[test]
fn integer_translation_shifts_coverage() {
    let side = 40_u32;
    let base = serde_json::json!({
        "center_px": [18.0, 18.0],
        "radii_px": [6.0, 9.0],
        "angle_rad": 0.3,
        "edge": { "profile": "smoothstep", "half_width_px": 2.0 },
    });
    let shifted = serde_json::json!({
        "center_px": [22.0, 21.0],
        "radii_px": [6.0, 9.0],
        "angle_rad": 0.3,
        "edge": { "profile": "smoothstep", "half_width_px": 2.0 },
    });
    let dx = 4_usize;
    let dy = 3_usize;
    let unshifted = render(side, &base);
    let moved = render(side, &shifted);
    let width = side as usize;
    let height = side as usize;
    for row in 0..(height - dy) {
        for col in 0..(width - dx) {
            let src = unshifted.samples()[row * width + col];
            let dst = moved.samples()[(row + dy) * width + (col + dx)];
            assert!(
                (f64::from(src) - f64::from(dst)).abs() < TOL,
                "translation mismatch at ({col},{row}): {src} vs {dst}"
            );
        }
    }
}

/// The soft-edge transition width equals `2·half_width_px` in physical pixels
/// (`bn-2z8` acceptance): measure the band where `0 < coverage < 1` along the
/// boundary normal of an axis-aligned circle and confirm it spans `2h`.
#[test]
fn feather_transition_width_equals_two_half_widths() {
    // Sample coverage along the +x ray from a large centered circle, where the
    // boundary normal is exactly +x, so the analytic signed distance equals the
    // pixel distance from the boundary. The feather coverage maps directly to it.
    let radius = 200.0_f64;
    let h = 6.0_f64;
    let geometry = EllipseGeometry::resolve(&serde_json::json!({
        "center_px": [0.0, 0.0],
        "radii_px": [radius, radius],
        "edge": { "profile": "smoothstep", "half_width_px": h },
    }))
    .expect("geometry");

    // Coverage is 1 at distance r - h from center and 0 at r + h: the transition
    // band is [r - h, r + h], width 2h.
    let inside_edge = geometry.coverage(radius - h, 0.0);
    let outside_edge = geometry.coverage(radius + h, 0.0);
    let at_boundary = geometry.coverage(radius, 0.0);
    assert!(
        (inside_edge - 1.0).abs() < 1e-3,
        "coverage at inner band edge should be ~1, got {inside_edge}"
    );
    assert!(
        outside_edge.abs() < 1e-3,
        "coverage at outer band edge should be ~0, got {outside_edge}"
    );
    assert!(
        (at_boundary - 0.5).abs() < 1e-3,
        "coverage at boundary should be ~0.5, got {at_boundary}"
    );

    // Just inside the band the coverage is strictly between 0 and 1; just outside
    // it is saturated. This pins the band width to exactly 2h.
    assert!(geometry.coverage(radius - h + 0.5, 0.0) < 1.0);
    assert!(geometry.coverage(radius + h - 0.5, 0.0) > 0.0);
    assert!((geometry.coverage(radius - h - 0.5, 0.0) - 1.0).abs() < TOL);
    assert!(geometry.coverage(radius + h + 0.5, 0.0).abs() < TOL);
}

/// The `feather` primitive is a monotone non-increasing map from signed distance
/// to coverage, pinned at the band edges.
#[test]
fn feather_is_monotone_and_pinned() {
    let half = 4.0_f64;
    assert!((feather(-half - 1.0, half) - 1.0).abs() < TOL);
    assert!(feather(half + 1.0, half).abs() < TOL);
    assert!((feather(0.0, half) - 0.5).abs() < TOL);
    let mut prev = f64::INFINITY;
    for i in 0..=100 {
        let fraction = f64::from(i) / 100.0;
        let sd = (4.0 * half).mul_add(fraction, -2.0 * half);
        let coverage = feather(sd, half);
        assert!(coverage <= prev + TOL, "feather not monotone at sd={sd}");
        assert!((0.0..=1.0).contains(&coverage));
        prev = coverage;
    }
    // h = 0 is a hard, half-open edge.
    assert!((feather(0.0, 0.0) - 1.0).abs() < TOL);
    assert!(feather(1e-9, 0.0).abs() < TOL);
}

/// Half-open pixel convention (`AGENT_VERIFICATION` §3.6): with a hard edge a
/// pixel center lying exactly on the analytic boundary is covered, and one just
/// outside is not.
#[test]
fn half_open_convention_on_pixel_aligned_boundary() {
    // A circle of radius 4 centered at a pixel center (8.5, 8.5): the pixel whose
    // center sits at distance exactly 4 along +x is at x = 12.5 (center 8.5 + 4),
    // i.e. pixel column 12. A hard edge (no `edge`) must cover it (sd == 0 -> in)
    // and exclude column 13 (sd > 0 -> out).
    let params = serde_json::json!({
        "center_px": [8.5, 8.5],
        "radii_px": [4.0, 4.0],
    });
    let mask = render(17, &params);
    let width = 17_usize;
    let row = 8_usize; // y center 8.5, the circle's center row
    let on_boundary = mask.samples()[row * width + 12];
    let just_outside = mask.samples()[row * width + 13];
    assert!(
        (f64::from(on_boundary) - 1.0).abs() < TOL,
        "pixel center on the boundary is covered, got {on_boundary}"
    );
    assert!(
        f64::from(just_outside).abs() < TOL,
        "pixel just outside is not covered, got {just_outside}"
    );
}

#[test]
fn degenerate_zero_radius_is_rejected() {
    let mut inputs = InputValues::new();
    inputs.insert("extent_from".to_owned(), extent_source(8));
    let params = serde_json::json!({
        "center_px": [4.0, 4.0],
        "radii_px": [0.0, 3.0],
    });
    let err = EllipseMask::new()
        .compute(&inputs, &params)
        .expect_err("zero radius must be rejected");
    assert_eq!(err.class, ErrorClass::Schema);
    assert_eq!(err.code, super::E_ELLIPSE_PARAM);
}

#[test]
fn degenerate_negative_radius_is_rejected() {
    let mut inputs = InputValues::new();
    inputs.insert("extent_from".to_owned(), extent_source(8));
    let params = serde_json::json!({
        "center_px": [4.0, 4.0],
        "radii_px": [3.0, -2.0],
    });
    let err = EllipseMask::new()
        .compute(&inputs, &params)
        .expect_err("negative radius must be rejected");
    assert_eq!(err.class, ErrorClass::Schema);
}

#[test]
fn non_finite_center_is_rejected() {
    // JSON cannot carry NaN/Inf directly; a missing element / wrong shape is the
    // reachable degenerate. A wrong-arity center is rejected.
    let mut inputs = InputValues::new();
    inputs.insert("extent_from".to_owned(), extent_source(8));
    let params = serde_json::json!({
        "center_px": [4.0],
        "radii_px": [3.0, 3.0],
    });
    let err = EllipseMask::new()
        .compute(&inputs, &params)
        .expect_err("malformed center must be rejected");
    assert_eq!(err.class, ErrorClass::Schema);
}

#[test]
fn unsupported_antialias_is_rejected() {
    let mut inputs = InputValues::new();
    inputs.insert("extent_from".to_owned(), extent_source(8));
    let params = serde_json::json!({
        "center_px": [4.0, 4.0],
        "radii_px": [3.0, 3.0],
        "antialias": "supersample",
    });
    let err = EllipseMask::new()
        .compute(&inputs, &params)
        .expect_err("unsupported antialias must be rejected");
    assert_eq!(err.class, ErrorClass::Schema);
}

#[test]
fn unsupported_edge_profile_is_rejected() {
    let mut inputs = InputValues::new();
    inputs.insert("extent_from".to_owned(), extent_source(8));
    let params = serde_json::json!({
        "center_px": [4.0, 4.0],
        "radii_px": [3.0, 3.0],
        "edge": { "profile": "linear", "half_width_px": 2.0 },
    });
    let err = EllipseMask::new()
        .compute(&inputs, &params)
        .expect_err("unsupported edge profile must be rejected");
    assert_eq!(err.class, ErrorClass::Schema);
}

#[test]
fn negative_half_width_is_rejected() {
    let mut inputs = InputValues::new();
    inputs.insert("extent_from".to_owned(), extent_source(8));
    let params = serde_json::json!({
        "center_px": [4.0, 4.0],
        "radii_px": [3.0, 3.0],
        "edge": { "profile": "smoothstep", "half_width_px": -1.0 },
    });
    let err = EllipseMask::new()
        .compute(&inputs, &params)
        .expect_err("negative half width must be rejected");
    assert_eq!(err.class, ErrorClass::Schema);
}

#[test]
fn infer_outputs_records_mask_and_empty_input_roi() {
    let descriptor = ResourceDescriptor::Image(ImageDescriptor {
        extent: Extent::new(20, 12),
        layout: ChannelLayout::Rgba,
        scalar: ScalarType::F32,
        color: ColorEncoding::LinearSrgb,
        range: ColorRange::SceneReferred,
        alpha: AlphaRepresentation::Premultiplied,
        coordinates: CoordinateConvention::PixelCenterUpperLeft,
        semantic: SemanticRole::Color,
    });
    let mut inputs = Descriptors::new();
    inputs.insert("extent_from".to_owned(), descriptor);
    let params = serde_json::json!({
        "center_px": [10.0, 6.0],
        "radii_px": [5.0, 3.0],
    });

    let out = EllipseMask::new()
        .infer_outputs(&inputs, &params)
        .expect("infer");
    let ResourceDescriptor::Mask(d) = out["mask"] else {
        panic!("expected mask");
    };
    assert_eq!(d.extent, Extent::new(20, 12));

    // The mask reads no input samples: the demanded input region is empty.
    let mut requested = OutputRegions::new();
    requested.insert("mask".to_owned(), Rect::new(0, 0, 20, 12));
    let needed = EllipseMask::new()
        .required_inputs(&requested, &inputs, &params)
        .expect("required_inputs");
    assert_eq!(needed["extent_from"], Rect::new(0, 0, 0, 0));

    // Postconditions hold.
    let results = EllipseMask::new()
        .validate_postconditions(&out, &params)
        .expect("postconditions");
    assert!(results.iter().all(|r| r.status == AssertionStatus::Pass));
}

#[test]
fn infer_outputs_rejects_degenerate_geometry() {
    let descriptor = ResourceDescriptor::Image(ImageDescriptor {
        extent: Extent::new(8, 8),
        layout: ChannelLayout::Rgba,
        scalar: ScalarType::F32,
        color: ColorEncoding::LinearSrgb,
        range: ColorRange::SceneReferred,
        alpha: AlphaRepresentation::Premultiplied,
        coordinates: CoordinateConvention::PixelCenterUpperLeft,
        semantic: SemanticRole::Color,
    });
    let mut inputs = Descriptors::new();
    inputs.insert("extent_from".to_owned(), descriptor);
    let params = serde_json::json!({
        "center_px": [4.0, 4.0],
        "radii_px": [0.0, 0.0],
    });
    let err = EllipseMask::new()
        .infer_outputs(&inputs, &params)
        .expect_err("degenerate geometry must be rejected at infer time");
    assert_eq!(err.class, ErrorClass::Schema);
}

/// The checked-in `ops/manifests/<id>.json` file (read by `cargo xtask
/// verify-op`) must stay byte-identical to the Rust manifest builder, the source
/// of truth. Regenerate with `serde_json::to_string_pretty` if this fails.
#[test]
fn checked_in_manifest_matches_builder() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("crates dir")
        .parent()
        .expect("repo root")
        .join("ops/manifests");
    let manifest = EllipseMask::manifest().expect("ellipse manifest");
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
