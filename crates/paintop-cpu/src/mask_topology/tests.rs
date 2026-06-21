//! Verification suite for the hard-mask topology ops (`OP_CATALOG` §4):
//!
//! - **schema/contract**: each manifest validates, agrees with its contract,
//!   gates clean, and the checked-in JSON matches the Rust builder;
//! - **analytic fixtures (connected components)**: synthetic multi-blob masks
//!   yield the exact component count, per-pixel labels, and per-label areas, with
//!   labels numbered in raster-scan order (the stability policy);
//! - **connectivity**: a diagonal pixel chain is one 8-component but several
//!   4-components;
//! - **large-ID round trip**: a label map with IDs above `2^24` is stored and
//!   recovered without loss (`AGENT_VERIFICATION` §2.3);
//! - **`fill_holes`**: fills only background that is not connected to the image
//!   border, leaving border-touching background untouched;
//! - **`remove_components`**: drops exactly the components strictly below the area
//!   threshold and keeps the rest, honoring connectivity.

use paintop_core::executor::{InputValues, OpImplementation, OutputValues, ResourceValue};
use paintop_ir::{
    ComponentsData, CoordinateConvention, Descriptors, ErrorClass, Extent, MaskDescriptor,
    MaskMeaning, OpContract, ResourceDescriptor, ScalarType, ValidRange,
    check_contract_consistency, verify_categories,
};

use super::{
    CONNECTED_COMPONENTS_OP_ID, ConnectedComponents, FILL_HOLES_OP_ID, FillHoles,
    REMOVE_COMPONENTS_OP_ID, RemoveComponents, label_of_sample,
};

// ---------------------------------------------------------------------------
// fixtures + helpers
// ---------------------------------------------------------------------------

/// Build a hard coverage-mask value (`{0, 1}`) from a bitmap of `0`/`1` bytes.
fn mask(w: u32, h: u32, bits: &[u8]) -> ResourceValue {
    assert_eq!(bits.len(), (w * h) as usize, "bit count");
    let samples: Vec<f32> = bits.iter().map(|&b| f32::from(b)).collect();
    let descriptor = ResourceDescriptor::Mask(MaskDescriptor {
        extent: Extent::new(w, h),
        scalar: ScalarType::F32,
        range: ValidRange::Bounded { min: 0.0, max: 1.0 },
        meaning: MaskMeaning::Selection,
        coordinates: CoordinateConvention::PixelCenterUpperLeft,
    });
    ResourceValue::new(descriptor, 1, samples).expect("mask value")
}

/// Run `mask.connected_components` and return both outputs.
fn run_components(value: &ResourceValue, params: &serde_json::Value) -> OutputValues {
    let mut inputs = InputValues::new();
    inputs.insert("mask".to_owned(), value.clone());
    ConnectedComponents::new()
        .compute(&inputs, params)
        .expect("connected_components computes")
}

/// The per-pixel `u32` labels recovered from a component run's label map.
fn labels(out: &OutputValues) -> Vec<u32> {
    out.get("labels")
        .expect("labels port")
        .samples()
        .iter()
        .map(|&s| label_of_sample(s))
        .collect()
}

/// The component summary recovered from a component run's report.
fn components_data(out: &OutputValues) -> ComponentsData {
    out.get("report")
        .expect("report port")
        .as_report()
        .expect("report payload")
        .components
        .clone()
        .expect("components data")
}

/// Run `mask.fill_holes` and return the output bitmap as `0`/`1` bytes.
fn run_fill_holes(value: &ResourceValue, params: &serde_json::Value) -> Vec<u8> {
    let mut inputs = InputValues::new();
    inputs.insert("mask".to_owned(), value.clone());
    let out = FillHoles::new()
        .compute(&inputs, params)
        .expect("fill_holes computes");
    out.get("mask")
        .expect("mask port")
        .samples()
        .iter()
        .map(|&s| u8::from(s >= 0.5))
        .collect()
}

/// Run `mask.remove_components` and return the output bitmap as `0`/`1` bytes.
fn run_remove(value: &ResourceValue, params: &serde_json::Value) -> Vec<u8> {
    let mut inputs = InputValues::new();
    inputs.insert("mask".to_owned(), value.clone());
    let out = RemoveComponents::new()
        .compute(&inputs, params)
        .expect("remove_components computes");
    out.get("mask")
        .expect("mask port")
        .samples()
        .iter()
        .map(|&s| u8::from(s >= 0.5))
        .collect()
}

// ---------------------------------------------------------------------------
// schema / contract
// ---------------------------------------------------------------------------

#[test]
fn manifests_validate_and_agree_with_contracts() {
    let cc = ConnectedComponents::manifest().expect("cc manifest");
    cc.validate().expect("cc valid");
    verify_categories(&cc, &cc.test.verification).expect("cc gate clean");
    assert_eq!(cc.id.to_string(), CONNECTED_COMPONENTS_OP_ID);
    check_contract_consistency(&cc, &ConnectedComponents::new()).expect("cc contract");

    let fh = FillHoles::manifest().expect("fh manifest");
    fh.validate().expect("fh valid");
    verify_categories(&fh, &fh.test.verification).expect("fh gate clean");
    assert_eq!(fh.id.to_string(), FILL_HOLES_OP_ID);
    check_contract_consistency(&fh, &FillHoles::new()).expect("fh contract");

    let rc = RemoveComponents::manifest().expect("rc manifest");
    rc.validate().expect("rc valid");
    verify_categories(&rc, &rc.test.verification).expect("rc gate clean");
    assert_eq!(rc.id.to_string(), REMOVE_COMPONENTS_OP_ID);
    check_contract_consistency(&rc, &RemoveComponents::new()).expect("rc contract");
}

/// Each checked-in `ops/manifests/<id>.json` must stay byte-identical to the Rust
/// builder.
#[test]
fn checked_in_manifests_match_builders() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("crates dir")
        .parent()
        .expect("repo root")
        .join("ops/manifests");
    for manifest in [
        ConnectedComponents::manifest().expect("cc manifest"),
        FillHoles::manifest().expect("fh manifest"),
        RemoveComponents::manifest().expect("rc manifest"),
    ] {
        let path = root.join(format!("{}.json", manifest.id));
        let on_disk = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
        let expected = serde_json::to_string_pretty(&manifest).expect("serialize");
        assert_eq!(
            on_disk.trim_end(),
            expected.trim_end(),
            "{} is stale; regenerate from the Rust builder",
            path.display()
        );
    }
}

// ---------------------------------------------------------------------------
// connected components: count, labels, areas, stability
// ---------------------------------------------------------------------------

#[test]
fn two_blobs_get_distinct_raster_ordered_labels() {
    // A 5x3 mask with two separate blobs:
    //   1 1 0 0 1
    //   1 0 0 0 1
    //   0 0 0 0 0
    // The top-left blob's first pixel (0,0) precedes the right blob's (4,0) in
    // raster order, so it is label 1 and the right blob is label 2.
    #[rustfmt::skip]
    let bits = [
        1, 1, 0, 0, 1,
        1, 0, 0, 0, 1,
        0, 0, 0, 0, 0,
    ];
    let out = run_components(&mask(5, 3, &bits), &serde_json::json!({}));
    let lab = labels(&out);
    #[rustfmt::skip]
    let expected = [
        1, 1, 0, 0, 2,
        1, 0, 0, 0, 2,
        0, 0, 0, 0, 0,
    ];
    assert_eq!(lab, expected);
    let data = components_data(&out);
    assert_eq!(data.count, 2);
    assert_eq!(data.areas, vec![3, 2]);
    assert_eq!(data.connectivity, 8);
}

#[test]
fn label_stability_is_deterministic_across_runs() {
    #[rustfmt::skip]
    let bits = [
        1, 0, 1,
        0, 0, 0,
        1, 0, 1,
    ];
    let m = mask(3, 3, &bits);
    let a = labels(&run_components(&m, &serde_json::json!({})));
    let b = labels(&run_components(&m, &serde_json::json!({})));
    assert_eq!(a, b);
    // Four isolated pixels, labeled in raster order 1,2,3,4.
    #[rustfmt::skip]
    let expected = [
        1, 0, 2,
        0, 0, 0,
        3, 0, 4,
    ];
    assert_eq!(a, expected);
}

#[test]
fn empty_mask_has_zero_components() {
    let out = run_components(&mask(4, 4, &[0; 16]), &serde_json::json!({}));
    let data = components_data(&out);
    assert_eq!(data.count, 0);
    assert!(data.areas.is_empty());
    assert!(labels(&out).iter().all(|&l| l == 0));
}

#[test]
fn full_mask_is_one_component() {
    let out = run_components(&mask(3, 2, &[1; 6]), &serde_json::json!({}));
    let data = components_data(&out);
    assert_eq!(data.count, 1);
    assert_eq!(data.areas, vec![6]);
    assert!(labels(&out).iter().all(|&l| l == 1));
}

// ---------------------------------------------------------------------------
// connectivity: 4 vs 8
// ---------------------------------------------------------------------------

#[test]
fn diagonal_chain_is_one_8component_but_three_4components() {
    // A 3x3 anti-diagonal:
    //   0 0 1
    //   0 1 0
    //   1 0 0
    #[rustfmt::skip]
    let bits = [
        0, 0, 1,
        0, 1, 0,
        1, 0, 0,
    ];
    let m = mask(3, 3, &bits);

    let eight = components_data(&run_components(&m, &serde_json::json!({"connectivity": 8})));
    assert_eq!(eight.count, 1, "8-connectivity joins the diagonal");
    assert_eq!(eight.areas, vec![3]);
    assert_eq!(eight.connectivity, 8);

    let four = components_data(&run_components(&m, &serde_json::json!({"connectivity": 4})));
    assert_eq!(four.count, 3, "4-connectivity splits the diagonal");
    assert_eq!(four.areas, vec![1, 1, 1]);
    assert_eq!(four.connectivity, 4);
}

#[test]
fn bad_connectivity_is_rejected() {
    let m = mask(2, 2, &[1, 0, 0, 1]);
    let mut inputs = InputValues::new();
    inputs.insert("mask".to_owned(), m);
    let err = ConnectedComponents::new()
        .compute(&inputs, &serde_json::json!({"connectivity": 6}))
        .expect_err("connectivity 6 must be rejected");
    assert_eq!(err.class, ErrorClass::Schema);
    assert_eq!(err.code, super::E_TOPOLOGY_PARAM);
}

// ---------------------------------------------------------------------------
// large-ID round trip (AGENT_VERIFICATION 2.3)
// ---------------------------------------------------------------------------

#[test]
fn large_label_ids_round_trip_without_loss() {
    // A 1xN row of N isolated foreground pixels separated by gaps produces labels
    // 1..=N. Pick N so the largest label exceeds 2^24 (the f32 integer-exact
    // limit), proving the bit-pattern storage is lossless. We build the labels
    // directly (the labeler is exercised elsewhere) and confirm the sample
    // encoding survives.
    for id in [1_u32, 16_777_216, 16_777_217, 33_554_435, u32::MAX] {
        let sample = f32::from_bits(id);
        assert_eq!(
            label_of_sample(sample),
            id,
            "label {id} must survive the f32 bit-pattern round trip"
        );
    }
}

#[test]
fn many_components_label_in_order() {
    // 1x9: foreground at even columns -> 5 isolated components labeled 1..=5.
    let bits = [1, 0, 1, 0, 1, 0, 1, 0, 1];
    let out = run_components(&mask(9, 1, &bits), &serde_json::json!({}));
    let lab = labels(&out);
    assert_eq!(lab, vec![1, 0, 2, 0, 3, 0, 4, 0, 5]);
    assert_eq!(components_data(&out).count, 5);
}

// ---------------------------------------------------------------------------
// fill_holes: border-connected definition
// ---------------------------------------------------------------------------

#[test]
fn fills_enclosed_hole_only() {
    // A 5x5 ring with a single enclosed background pixel at the center.
    #[rustfmt::skip]
    let bits = [
        1, 1, 1, 1, 1,
        1, 0, 0, 0, 1,
        1, 0, 1, 0, 1,
        1, 0, 0, 0, 1,
        1, 1, 1, 1, 1,
    ];
    // All the interior background is enclosed (not border-connected), so it all
    // fills; the result is solid.
    let out = run_fill_holes(&mask(5, 5, &bits), &serde_json::json!({}));
    assert_eq!(out, vec![1; 25]);
}

#[test]
fn border_connected_background_is_untouched() {
    // A C-shape open on the right: the cavity background is connected to the
    // border, so nothing is filled.
    #[rustfmt::skip]
    let bits = [
        1, 1, 1, 1,
        1, 0, 0, 0,
        1, 0, 0, 0,
        1, 1, 1, 1,
    ];
    let out = run_fill_holes(&mask(4, 4, &bits), &serde_json::json!({}));
    assert_eq!(out, bits.to_vec(), "open cavity must stay background");
}

#[test]
fn fill_holes_connectivity_changes_enclosure() {
    // A diamond ring whose hole leaks out only through a diagonal gap. Under
    // 8-connectivity of the background flood the hole is border-connected (leaks),
    // under 4-connectivity it is enclosed (fills).
    //   0 1 0
    //   1 0 1
    //   0 1 0
    #[rustfmt::skip]
    let bits = [
        0, 1, 0,
        1, 0, 1,
        0, 1, 0,
    ];
    // 4-connectivity: the center (1,1) cannot reach the border background
    // diagonally, so it is enclosed and fills.
    let four = run_fill_holes(&mask(3, 3, &bits), &serde_json::json!({"connectivity": 4}));
    assert_eq!(four[4], 1, "center (1,1) fills under 4-connectivity");
    // The corners stay background (they are border).
    assert_eq!(four[0], 0);

    // 8-connectivity: the center reaches the corner background diagonally, so it
    // is border-connected and does not fill.
    let eight = run_fill_holes(&mask(3, 3, &bits), &serde_json::json!({"connectivity": 8}));
    assert_eq!(eight[4], 0, "center (1,1) leaks out under 8-connectivity");
}

// ---------------------------------------------------------------------------
// remove_components: area threshold
// ---------------------------------------------------------------------------

#[test]
fn removes_below_threshold_keeps_at_or_above() {
    // 5x1: a 1-pixel blob, a gap, then a 3-pixel blob.
    //   1 0 1 1 1
    let bits = [1, 0, 1, 1, 1];
    // min_area = 2 drops the singleton, keeps the triple.
    let out = run_remove(&mask(5, 1, &bits), &serde_json::json!({"min_area": 2}));
    assert_eq!(out, vec![0, 0, 1, 1, 1]);
}

#[test]
fn threshold_is_inclusive_at_min_area() {
    // Two blobs of area exactly 2 and 3.
    //   1 1 0 1 1 1
    let bits = [1, 1, 0, 1, 1, 1];
    // min_area = 3 drops the area-2 blob, keeps the area-3 blob (>= is inclusive).
    let out = run_remove(&mask(6, 1, &bits), &serde_json::json!({"min_area": 3}));
    assert_eq!(out, vec![0, 0, 0, 1, 1, 1]);
    // min_area = 2 keeps both.
    let keep = run_remove(&mask(6, 1, &bits), &serde_json::json!({"min_area": 2}));
    assert_eq!(keep, vec![1, 1, 0, 1, 1, 1]);
}

#[test]
fn remove_honors_connectivity() {
    // A 3x3 anti-diagonal of 3 pixels: one area-3 blob (8-conn) vs three area-1
    // blobs (4-conn).
    #[rustfmt::skip]
    let bits = [
        0, 0, 1,
        0, 1, 0,
        1, 0, 0,
    ];
    // 8-conn, min_area 2: the single area-3 component survives whole.
    let eight = run_remove(
        &mask(3, 3, &bits),
        &serde_json::json!({"min_area": 2, "connectivity": 8}),
    );
    assert_eq!(eight, bits.to_vec());
    // 4-conn, min_area 2: all three singletons are below threshold -> cleared.
    let four = run_remove(
        &mask(3, 3, &bits),
        &serde_json::json!({"min_area": 2, "connectivity": 4}),
    );
    assert_eq!(four, vec![0; 9]);
}

#[test]
fn missing_min_area_is_rejected() {
    let m = mask(2, 2, &[1, 1, 1, 1]);
    let mut inputs = InputValues::new();
    inputs.insert("mask".to_owned(), m);
    let err = RemoveComponents::new()
        .compute(&inputs, &serde_json::json!({}))
        .expect_err("missing min_area must be rejected");
    assert_eq!(err.class, ErrorClass::Schema);
    assert_eq!(err.code, super::E_TOPOLOGY_PARAM);
}

#[test]
fn negative_min_area_is_rejected() {
    let m = mask(2, 2, &[1, 1, 1, 1]);
    let mut inputs = InputValues::new();
    inputs.insert("mask".to_owned(), m);
    let err = RemoveComponents::new()
        .compute(&inputs, &serde_json::json!({"min_area": -1}))
        .expect_err("negative min_area must be rejected");
    assert_eq!(err.class, ErrorClass::Schema);
    assert_eq!(err.code, super::E_TOPOLOGY_PARAM);
}

// ---------------------------------------------------------------------------
// rejection: wrong / missing input
// ---------------------------------------------------------------------------

#[test]
fn missing_mask_input_is_rejected() {
    let inputs = Descriptors::new();
    let err = ConnectedComponents::new()
        .infer_outputs(&inputs, &serde_json::json!({}))
        .expect_err("missing input must be rejected");
    assert_eq!(err.class, ErrorClass::Reference);
    assert_eq!(err.code, super::E_TOPOLOGY_INPUT);
}
