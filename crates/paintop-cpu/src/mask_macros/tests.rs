//! Verification suite for the morphology macros `mask.grow@1`,
//! `mask.shrink@1`, `mask.feather@1` (`plan.md` §17 macro rule):
//!
//! - **schema/contract**: each manifest validates, agrees with its contract, and
//!   gates clean;
//! - **hash identity** (the headline acceptance): a plan using a macro expands to
//!   the explicit `mask.to_sdf → sdf.offset → sdf.to_mask` subgraph and has the
//!   *identical semantic hash* to the hand-written expansion;
//! - **expansion shape**: the expanded plan exposes the three SDF nodes with the
//!   deterministic ids and param mapping; downstream refs to the macro id stay
//!   valid;
//! - **output identity**: the macro's direct kernel produces the same coverage as
//!   running the expanded subgraph kernel-by-kernel;
//! - **param mapping**: grow offsets `+r`, shrink `−r`, feather feathers; the
//!   round-trip of a hard mask through `feather(0)` is the identity at the
//!   contour;
//! - **idempotent normalize**: re-expanding an already-expanded plan is stable;
//! - **rejection**: a negative size / missing mask is a typed error.

use paintop_core::executor::{InputValues, OpImplementation, ResourceValue};
use paintop_ir::{
    CoordinateConvention, Extent, MaskDescriptor, MaskMeaning, ResourceDescriptor, ScalarType,
    ValidRange, check_contract_consistency, parse_plan, semantic_hash, verify_categories,
};

use crate::mask_to_sdf::MaskToSdf;
use crate::sdf_ops::{SdfOffset, SdfToMask};

use super::{
    FEATHER_OP_ID, GROW_OP_ID, MaskMacro, SHRINK_OP_ID, expand_plan, is_macro_op, offset_node_id,
    to_sdf_node_id,
};

/// Build a coverage-mask value from explicit samples sized `w * h`.
fn mask(w: u32, h: u32, samples: Vec<f32>) -> ResourceValue {
    assert_eq!(samples.len(), (w * h) as usize, "sample count");
    let descriptor = ResourceDescriptor::Mask(MaskDescriptor {
        extent: Extent::new(w, h),
        scalar: ScalarType::F32,
        range: ValidRange::Bounded { min: 0.0, max: 1.0 },
        meaning: MaskMeaning::Coverage,
        coordinates: CoordinateConvention::PixelCenterUpperLeft,
    });
    ResourceValue::new(descriptor, 1, samples).expect("mask value")
}

// --- schema / contract -----------------------------------------------------

#[test]
fn manifests_validate_and_agree_with_contracts() {
    let cases = [
        (
            MaskMacro::grow_manifest().expect("grow"),
            GROW_OP_ID,
            MaskMacro::grow(),
        ),
        (
            MaskMacro::shrink_manifest().expect("shrink"),
            SHRINK_OP_ID,
            MaskMacro::shrink(),
        ),
        (
            MaskMacro::feather_manifest().expect("feather"),
            FEATHER_OP_ID,
            MaskMacro::feather(),
        ),
    ];
    for (manifest, id, op) in cases {
        manifest.validate().expect("manifest valid");
        verify_categories(&manifest, &manifest.test.verification).expect("gates clean");
        assert_eq!(manifest.id.to_string(), id);
        check_contract_consistency(&manifest, &op).expect("contract consistent");
    }
}

#[test]
fn macro_op_ids_are_recognized() {
    assert!(is_macro_op(GROW_OP_ID));
    assert!(is_macro_op(SHRINK_OP_ID));
    assert!(is_macro_op(FEATHER_OP_ID));
    assert!(!is_macro_op("mask.to_sdf@1"));
}

// --- hash identity (headline acceptance) -----------------------------------

/// A plan whose single op is the macro `op` with `params`, wired from one input.
fn macro_plan(op: &str, params: &serde_json::Value) -> String {
    serde_json::json!({
        "paintop": "1.0",
        "inputs": { "src": { "kind": "Mask", "path": "src.png" } },
        "nodes": [{
            "id": "morphed",
            "op": op,
            "in": { "mask": "input:src" },
            "params": params
        }],
        "exports": { "out": "node:morphed/mask" }
    })
    .to_string()
}

/// The hand-written `mask.to_sdf → sdf.offset → sdf.to_mask` plan that a macro
/// must normalize to, parameterized to match `expand_plan`'s output exactly.
fn hand_expanded_plan(threshold: f64, distance_px: f64, half_width_px: f64) -> String {
    serde_json::json!({
        "paintop": "1.0",
        "inputs": { "src": { "kind": "Mask", "path": "src.png" } },
        "nodes": [
            {
                "id": "morphed.to_sdf",
                "op": "mask.to_sdf@1",
                "in": { "mask": "input:src" },
                "params": { "threshold": threshold }
            },
            {
                "id": "morphed.offset",
                "op": "sdf.offset@1",
                "in": { "sdf": "node:morphed.to_sdf/sdf" },
                "params": { "distance_px": distance_px }
            },
            {
                "id": "morphed",
                "op": "sdf.to_mask@1",
                "in": { "sdf": "node:morphed.offset/sdf" },
                "params": { "profile": "smoothstep", "half_width_px": half_width_px }
            }
        ],
        "exports": { "out": "node:morphed/mask" }
    })
    .to_string()
}

#[test]
fn grow_expands_to_identical_semantic_hash() {
    let macro_p = parse_plan(&macro_plan(
        GROW_OP_ID,
        &serde_json::json!({ "radius_px": 3.0 }),
    ))
    .expect("macro plan");
    let expanded = expand_plan(&macro_p).expect("expand");
    let hand = parse_plan(&hand_expanded_plan(0.5, 3.0, 0.0)).expect("hand plan");
    assert_eq!(
        semantic_hash(&expanded).expect("expanded hash"),
        semantic_hash(&hand).expect("hand hash"),
        "grow macro must normalize to the hand-written SDF subgraph hash"
    );
}

#[test]
fn shrink_expands_to_identical_semantic_hash() {
    let macro_p = parse_plan(&macro_plan(
        SHRINK_OP_ID,
        &serde_json::json!({ "radius_px": 2.0 }),
    ))
    .expect("macro plan");
    let expanded = expand_plan(&macro_p).expect("expand");
    let hand = parse_plan(&hand_expanded_plan(0.5, -2.0, 0.0)).expect("hand plan");
    assert_eq!(
        semantic_hash(&expanded).expect("expanded hash"),
        semantic_hash(&hand).expect("hand hash"),
        "shrink macro must normalize to distance_px = -radius_px"
    );
}

#[test]
fn feather_expands_to_identical_semantic_hash() {
    let macro_p = parse_plan(&macro_plan(
        FEATHER_OP_ID,
        &serde_json::json!({ "half_width_px": 1.5 }),
    ))
    .expect("macro plan");
    let expanded = expand_plan(&macro_p).expect("expand");
    let hand = parse_plan(&hand_expanded_plan(0.5, 0.0, 1.5)).expect("hand plan");
    assert_eq!(
        semantic_hash(&expanded).expect("expanded hash"),
        semantic_hash(&hand).expect("hand hash"),
        "feather macro must normalize to offset 0 + smoothstep half_width"
    );
}

#[test]
fn custom_threshold_is_forwarded() {
    let macro_p = parse_plan(&macro_plan(
        GROW_OP_ID,
        &serde_json::json!({ "radius_px": 1.0, "threshold": 0.25 }),
    ))
    .expect("macro plan");
    let expanded = expand_plan(&macro_p).expect("expand");
    let hand = parse_plan(&hand_expanded_plan(0.25, 1.0, 0.0)).expect("hand plan");
    assert_eq!(
        semantic_hash(&expanded).expect("expanded hash"),
        semantic_hash(&hand).expect("hand hash"),
        "a custom threshold flows to mask.to_sdf"
    );
}

// --- expansion shape -------------------------------------------------------

#[test]
fn expansion_exposes_three_sdf_nodes_with_stable_ids() {
    let macro_p = parse_plan(&macro_plan(
        GROW_OP_ID,
        &serde_json::json!({ "radius_px": 2.0 }),
    ))
    .expect("macro plan");
    let expanded = expand_plan(&macro_p).expect("expand");
    assert_eq!(expanded.nodes.len(), 3, "three expanded nodes");
    // The terminal node keeps the macro id so downstream refs stay valid.
    let terminal = expanded
        .nodes
        .iter()
        .find(|n| n.id == "morphed")
        .expect("terminal");
    assert_eq!(terminal.op, "sdf.to_mask@1");
    let to_sdf = expanded
        .nodes
        .iter()
        .find(|n| n.id == to_sdf_node_id("morphed"))
        .expect("to_sdf node");
    assert_eq!(to_sdf.op, "mask.to_sdf@1");
    assert_eq!(to_sdf.inputs["mask"], "input:src");
    let offset = expanded
        .nodes
        .iter()
        .find(|n| n.id == offset_node_id("morphed"))
        .expect("offset node");
    assert_eq!(offset.op, "sdf.offset@1");
    assert_eq!(offset.inputs["sdf"], "node:morphed.to_sdf/sdf");
}

#[test]
fn non_macro_nodes_pass_through_expansion_unchanged() {
    let plan = serde_json::json!({
        "paintop": "1.0",
        "inputs": { "src": { "kind": "Mask", "path": "src.png" } },
        "nodes": [{
            "id": "inv",
            "op": "mask.invert@1",
            "in": { "mask": "input:src" }
        }],
        "exports": { "out": "node:inv/mask" }
    })
    .to_string();
    let parsed = parse_plan(&plan).expect("plan");
    let expanded = expand_plan(&parsed).expect("expand");
    assert_eq!(expanded.nodes.len(), 1);
    assert_eq!(expanded.nodes[0].op, "mask.invert@1");
}

#[test]
fn expansion_is_idempotent() {
    // Re-expanding an already-expanded plan changes nothing (it has no macros).
    let macro_p = parse_plan(&macro_plan(
        FEATHER_OP_ID,
        &serde_json::json!({ "half_width_px": 2.0 }),
    ))
    .expect("macro plan");
    let once = expand_plan(&macro_p).expect("expand once");
    let twice = expand_plan(&once).expect("expand twice");
    assert_eq!(
        semantic_hash(&once).expect("hash once"),
        semantic_hash(&twice).expect("hash twice"),
        "expansion reaches a fixed point"
    );
}

// --- output identity -------------------------------------------------------

/// Run the explicit subgraph kernel-by-kernel, returning the coverage samples.
fn run_subgraph(
    m: &ResourceValue,
    threshold: f64,
    distance_px: f64,
    half_width_px: f64,
) -> Vec<f32> {
    let mut i1 = InputValues::new();
    i1.insert("mask".to_owned(), m.clone());
    let mut s = MaskToSdf::new()
        .compute(&i1, &serde_json::json!({ "threshold": threshold }))
        .expect("to_sdf");
    let sdf = s.remove("sdf").expect("sdf");

    let mut i2 = InputValues::new();
    i2.insert("sdf".to_owned(), sdf);
    let mut o = SdfOffset::new()
        .compute(&i2, &serde_json::json!({ "distance_px": distance_px }))
        .expect("offset");
    let off = o.remove("sdf").expect("sdf");

    let mut i3 = InputValues::new();
    i3.insert("sdf".to_owned(), off);
    let mut mm = SdfToMask::new()
        .compute(
            &i3,
            &serde_json::json!({ "profile": "smoothstep", "half_width_px": half_width_px }),
        )
        .expect("to_mask");
    mm.remove("mask").expect("mask").into_samples()
}

/// Run a macro's direct kernel, returning the coverage samples.
fn run_macro(op: MaskMacro, m: &ResourceValue, params: &serde_json::Value) -> Vec<f32> {
    let mut inputs = InputValues::new();
    inputs.insert("mask".to_owned(), m.clone());
    let mut out = op.compute(&inputs, params).expect("macro computes");
    out.remove("mask").expect("mask").into_samples()
}

#[test]
fn grow_kernel_matches_subgraph() {
    // A small block; grow by 1px and compare the macro kernel to the subgraph.
    let w = 7;
    let h = 7;
    let mut s = vec![0.0_f32; (w * h) as usize];
    for y in 2..5 {
        for x in 2..5 {
            s[(y * w + x) as usize] = 1.0;
        }
    }
    let m = mask(w, h, s);
    let macro_out = run_macro(
        MaskMacro::grow(),
        &m,
        &serde_json::json!({ "radius_px": 1.0 }),
    );
    let subgraph_out = run_subgraph(&m, 0.5, 1.0, 0.0);
    assert_eq!(macro_out, subgraph_out, "grow kernel == subgraph");
}

#[test]
fn feather_kernel_matches_subgraph() {
    let w = 9;
    let h = 1;
    let s = vec![1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let m = mask(w, h, s);
    let macro_out = run_macro(
        MaskMacro::feather(),
        &m,
        &serde_json::json!({ "half_width_px": 2.0 }),
    );
    let subgraph_out = run_subgraph(&m, 0.5, 0.0, 2.0);
    assert_eq!(macro_out, subgraph_out, "feather kernel == subgraph");
}

// --- param mapping ---------------------------------------------------------

#[test]
fn grow_enlarges_and_shrink_reduces_the_inside_set() {
    // A 5x5 filled block in an 11x11 field. Boundary pixels of the region sit at
    // phi = -1 (their nearest outside pixel is 1 away), so shrinking by exactly 1
    // would leave them at phi = 0 — still inside under the hard `phi <= 0`
    // reconstruct. We shrink by 1.5 (> the boundary ring's |phi|) so the ring is
    // genuinely removed, exercising the offset mapping.
    let w = 11;
    let h = 11;
    let mut s = vec![0.0_f32; (w * h) as usize];
    for y in 3..8 {
        for x in 3..8 {
            s[(y * w + x) as usize] = 1.0;
        }
    }
    let m = mask(w, h, s);
    let base_count = m.samples().iter().filter(|&&v| v >= 0.5).count();

    let grown = run_macro(
        MaskMacro::grow(),
        &m,
        &serde_json::json!({ "radius_px": 1.0 }),
    );
    let grown_count = grown.iter().filter(|&&v| v >= 0.5).count();
    assert!(grown_count > base_count, "grow enlarges the region");

    let shrunk = run_macro(
        MaskMacro::shrink(),
        &m,
        &serde_json::json!({ "radius_px": 1.5 }),
    );
    let shrunk_count = shrunk.iter().filter(|&&v| v >= 0.5).count();
    assert!(shrunk_count < base_count, "shrink reduces the region");
}

#[test]
fn feather_zero_is_the_identity_at_the_contour() {
    // feather(0) is a hard reconstruct of the thresholded mask — the original
    // hard mask is recovered exactly.
    let w = 5;
    let h = 1;
    let s = vec![1.0, 1.0, 0.0, 0.0, 0.0];
    let m = mask(w, h, s.clone());
    let out = run_macro(
        MaskMacro::feather(),
        &m,
        &serde_json::json!({ "half_width_px": 0.0 }),
    );
    assert_eq!(out, s, "feather(0) round-trips the hard mask");
}

// --- rejection -------------------------------------------------------------

#[test]
fn rejects_negative_radius() {
    let m = mask(1, 1, vec![1.0]);
    let mut inputs = InputValues::new();
    inputs.insert("mask".to_owned(), m);
    let err = MaskMacro::grow()
        .compute(&inputs, &serde_json::json!({ "radius_px": -1.0 }))
        .expect_err("negative radius rejected");
    assert_eq!(err.class, paintop_ir::ErrorClass::Schema);
}

#[test]
fn rejects_missing_mask_input_in_expansion() {
    let plan = serde_json::json!({
        "paintop": "1.0",
        "inputs": {},
        "nodes": [{ "id": "m", "op": "mask.grow@1", "params": { "radius_px": 1.0 } }],
        "exports": {}
    })
    .to_string();
    let parsed = parse_plan(&plan).expect("plan parses");
    let err = expand_plan(&parsed).expect_err("missing mask input rejected");
    assert_eq!(err.class, paintop_ir::ErrorClass::Reference);
}
