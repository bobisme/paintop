//! Canonicalization **replay** + negative fixtures (`bn-2ui`; `IR_SPEC` §17,
//! §2.4; `AGENT_VERIFICATION` §2.4; `plan.md` §10.3).
//!
//! The replay loop is the project's core reproducibility bet, made executable:
//!
//! ```text
//! text --parse--> Plan --serialize--> Value --canonical bytes--> blake3 hash
//!                          |                                          |
//!                          +--canonical text--> reparse --> hash ----=+  (must match)
//! ```
//!
//! A plan that is parsed, re-emitted in canonical form, and parsed again must
//! produce a byte-identical semantic hash — no drift across a serialize/reparse
//! round trip (`AGENT_VERIFICATION` §2.4: "serializing and reparsing normalized
//! plan preserves semantic hash"). On top of that property this file pins:
//!
//! - **non-semantic edits don't move the hash** — reordering object keys and
//!   reflowing whitespace (`IR_SPEC` §17 r5) leave the semantic hash identical;
//! - **semantic edits do move the hash** — changing a real parameter, an op id,
//!   a reference, or swapping an integer for an integral float flips it;
//! - **negative fixtures** — duplicate object keys (`IR_SPEC` §17 r12) and
//!   unstable / non-round-trippable floats (`IR_SPEC` §10, §17 r11; under- and
//!   over-flowing magnitudes) are rejected at parse, never canonicalized.
//!
//! There is no defaults/seed/description-stripping *normalizer* yet (that lands
//! with the larger canonicalization goal, `IR_SPEC` §17 r1–r4, r10); the replay
//! here exercises the canonical serializer and the semantic-hash API that exist
//! today, which is exactly the surface a later normalizer must keep stable.

use paintop_ir::{ErrorClass, HashDomain, SemanticHash, hash_value, parse_plan};
use proptest::prelude::*;
use serde_json::{Value, json};

const FIXTURES: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/fixtures/plans");

fn load(name: &str) -> String {
    let path = format!("{FIXTURES}/{name}");
    std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read fixture {path}: {e}"))
}

/// Parse `text` into a plan and return its semantic hash: the BLAKE3 of the
/// canonical bytes of the re-serialized [`Plan`]. Two texts that parse to the
/// same logical plan share this id.
fn semantic_hash(text: &str) -> SemanticHash {
    let plan = parse_plan(text).expect("fixture/plan text must parse");
    let value = serde_json::to_value(&plan).expect("a parsed Plan re-serializes to a Value");
    hash_value(HashDomain::Plan, &value).expect("a parsed Plan canonicalizes and hashes")
}

/// The canonical *text* of a parsed plan: parse, re-serialize, canonicalize.
/// This is the readable, stable form that a reparse must agree with.
fn canonical_text(text: &str) -> String {
    let plan = parse_plan(text).expect("fixture/plan text must parse");
    let value = serde_json::to_value(&plan).expect("a parsed Plan re-serializes to a Value");
    paintop_ir::to_canonical_string(&value).expect("a parsed Plan canonicalizes")
}

// ---------------------------------------------------------------------------
// Replay: parse -> canonicalize -> reparse preserves the semantic hash.
// ---------------------------------------------------------------------------

#[test]
fn replay_of_the_touchup_fixture_preserves_the_semantic_hash() {
    let original = load("replay-touchup.json");
    let first = semantic_hash(&original);

    // Emit the canonical text and feed it straight back through the parser.
    let canonical = canonical_text(&original);
    let replayed = semantic_hash(&canonical);
    assert_eq!(
        first, replayed,
        "serialize -> reparse must preserve the semantic hash"
    );

    // The canonical form is a fixed point: canonicalizing it again is a no-op.
    assert_eq!(
        canonical,
        canonical_text(&canonical),
        "canonical text must be a fixed point of the canonicalizer"
    );
}

#[test]
fn replay_of_the_empty_fixture_preserves_the_semantic_hash() {
    let original = load("empty-valid.json");
    assert_eq!(
        semantic_hash(&original),
        semantic_hash(&canonical_text(&original)),
        "even a minimal plan must survive a replay round trip"
    );
}

// ---------------------------------------------------------------------------
// Non-semantic edits must not move the hash.
// ---------------------------------------------------------------------------

#[test]
fn key_reordering_and_whitespace_do_not_change_the_hash() {
    // Same plan, keys in a different order and reflowed onto one line.
    let pretty = r#"{
        "paintop": "1.0",
        "nodes": [ { "op": "filter.gaussian_blur@1", "id": "b", "params": { "sigma": 8.0 } } ],
        "inputs": {},
        "exports": {}
    }"#;
    let reordered_compact = r#"{"exports":{},"inputs":{},"nodes":[{"id":"b","op":"filter.gaussian_blur@1","params":{"sigma":8.0}}],"paintop":"1.0"}"#;
    assert_eq!(
        semantic_hash(pretty),
        semantic_hash(reordered_compact),
        "key order and whitespace are non-semantic"
    );
}

#[test]
fn a_described_plan_replays_without_drift() {
    // The touchup fixture carries a `description`. Stripping non-semantic prose
    // from the semantic hash (`IR_SPEC` §17 r10) is the job of the not-yet-built
    // *normalizer* (the larger canonicalization goal); this bone only owns the
    // *replay* property, which must hold whatever the normalizer later decides.
    // Here we pin that a described plan survives a serialize -> reparse round
    // trip with an identical hash, and that re-describing it still replays
    // identically to its own canonical form.
    let base = load("replay-touchup.json");
    assert_eq!(
        semantic_hash(&base),
        semantic_hash(&canonical_text(&base)),
        "a plan that carries a description must still replay without drift"
    );

    // Swap in different prose; the rewritten plan likewise replays to itself.
    let mut value: Value = serde_json::from_str(&base).expect("fixture is valid JSON");
    value.as_object_mut().expect("plan is an object").insert(
        "description".to_owned(),
        json!("completely different prose"),
    );
    let edited = serde_json::to_string(&value).expect("re-serialize edited plan");
    assert_eq!(
        semantic_hash(&edited),
        semantic_hash(&canonical_text(&edited)),
        "re-describing a plan must not break the replay round trip"
    );
}

// ---------------------------------------------------------------------------
// Semantic edits must move the hash.
// ---------------------------------------------------------------------------

#[test]
fn changing_a_real_parameter_changes_the_hash() {
    let base = r#"{
        "paintop": "1.0",
        "inputs": {},
        "nodes": [{"id": "b", "op": "filter.gaussian_blur@1", "params": {"sigma": 8.0}}],
        "exports": {}
    }"#;
    let changed = base.replace("8.0", "8.5");
    assert_ne!(
        semantic_hash(base),
        semantic_hash(&changed),
        "a changed parameter value is a semantic change"
    );

    // Integer vs. integral float is a semantic distinction the canonical emitter
    // preserves, so the hash must reflect it too.
    let as_int = base.replace("8.0", "8");
    assert_ne!(
        semantic_hash(base),
        semantic_hash(&as_int),
        "float 8.0 vs integer 8 is a semantic distinction"
    );
}

#[test]
fn changing_op_id_or_reference_changes_the_hash() {
    let base = r#"{
        "paintop": "1.0",
        "inputs": {},
        "nodes": [
            {"id": "a", "op": "color.convert@1", "in": {"image": "input:source"}, "params": {"to": "linear-srgb"}}
        ],
        "exports": {}
    }"#;
    let baseline = semantic_hash(base);

    // A different op semantic version is a different operation (`IR_SPEC` §6).
    let bumped_major = base.replace("color.convert@1", "color.convert@2");
    assert_ne!(
        baseline,
        semantic_hash(&bumped_major),
        "the op semantic major version is part of the identity"
    );

    // A rewired input reference changes the graph.
    let rewired = base.replace("input:source", "input:other");
    assert_ne!(
        baseline,
        semantic_hash(&rewired),
        "a changed resource reference is a semantic change"
    );
}

// ---------------------------------------------------------------------------
// Negative fixtures: duplicate keys and unstable floats reject before canon.
// ---------------------------------------------------------------------------

#[test]
fn duplicate_top_level_key_is_rejected_before_canonicalization() {
    let err = parse_plan(&load("duplicate-key.json"))
        .expect_err("a duplicate top-level key must be rejected, never canonicalized");
    assert_eq!(err.class, ErrorClass::Parse);
    assert_eq!(err.code, "E_DUPLICATE_KEY");
}

#[test]
fn duplicate_nested_param_key_is_rejected_before_canonicalization() {
    let err = parse_plan(&load("duplicate-node-param-key.json"))
        .expect_err("a duplicate nested param key must be rejected");
    assert_eq!(err.class, ErrorClass::Parse);
    assert_eq!(err.code, "E_DUPLICATE_KEY");
}

#[test]
fn underflowing_float_is_rejected_before_canonicalization() {
    // `1e-400` silently coerces to `0.0`, so it has no stable round-trippable
    // float form (`IR_SPEC` §10, §17 r11) and must be rejected.
    let err = parse_plan(&load("underflow-number.json"))
        .expect_err("an underflowing magnitude must be rejected");
    assert_eq!(err.class, ErrorClass::Parse);
    assert_eq!(err.code, "E_INVALID_NUMBER");
}

#[test]
fn overflowing_float_is_rejected_before_canonicalization() {
    // `1e400` coerces to infinity, which the canonical serializer cannot emit;
    // the parser rejects it up front.
    let err = parse_plan(&load("overflow-number.json"))
        .expect_err("an overflowing magnitude must be rejected");
    assert_eq!(err.class, ErrorClass::Parse);
    assert_eq!(err.code, "E_INVALID_NUMBER");
}

// ---------------------------------------------------------------------------
// Property: over generated small plans, a serialize -> reparse round trip never
// drifts the semantic hash, and the canonical text is a fixed point.
// ---------------------------------------------------------------------------

/// A small, finite, round-trippable JSON scalar for generated `params`.
fn arb_scalar() -> impl Strategy<Value = Value> {
    prop_oneof![
        Just(Value::Null),
        any::<bool>().prop_map(Value::Bool),
        (-1000i64..1000).prop_map(|n| json!(n)),
        // Bounded finite floats keep every value round-trippable through the
        // canonical single-float form.
        (-1000.0f64..1000.0).prop_map(|f| json!(f)),
        "[a-z][a-z0-9_.-]{0,8}".prop_map(Value::String),
    ]
}

/// A small `params` object with sorted-distinct keys (so generation never emits
/// a duplicate key, which the parser would reject by design).
fn arb_params() -> impl Strategy<Value = serde_json::Map<String, Value>> {
    prop::collection::btree_map("[a-z][a-z_]{0,6}", arb_scalar(), 0..4)
        .prop_map(|m| m.into_iter().collect())
}

/// A small node with a valid id/op and generated params.
fn arb_node() -> impl Strategy<Value = Value> {
    (
        "[a-z][a-z0-9_.-]{0,8}",
        "[a-z]{1,6}\\.[a-z]{1,8}@[1-3]",
        arb_params(),
    )
        .prop_map(|(id, op, params)| json!({"id": id, "op": op, "params": Value::Object(params)}))
}

/// A small plan with the four required fields and 0..4 generated nodes.
fn arb_plan() -> impl Strategy<Value = String> {
    prop::collection::vec(arb_node(), 0..4).prop_map(|nodes| {
        json!({
            "paintop": "1.0",
            "inputs": {},
            "nodes": nodes,
            "exports": {}
        })
        .to_string()
    })
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    /// The core replay property (`AGENT_VERIFICATION` §2.4): for any small plan,
    /// canonicalizing then reparsing yields the same semantic hash, and the
    /// canonical text is idempotent under a second canonicalization.
    #[test]
    fn replay_preserves_semantic_hash(text in arb_plan()) {
        let original = semantic_hash(&text);
        let canonical = canonical_text(&text);
        let replayed = semantic_hash(&canonical);
        prop_assert_eq!(original, replayed);
        prop_assert_eq!(&canonical, &canonical_text(&canonical));
    }
}
