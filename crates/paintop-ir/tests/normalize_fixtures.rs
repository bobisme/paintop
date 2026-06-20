//! Plan-normalization integration tests (`bn-2l1`; `IR_SPEC` §17 r10;
//! `M0_DECISIONS` D4; `AGENT_VERIFICATION` §2.4).
//!
//! These exercise the public [`paintop_ir::semantic_hash`] / [`normalize`] API
//! over real fixtures and assert the bone's contract end to end:
//!
//! - a plan and a copy that differs ONLY in non-semantic prose (`description` /
//!   `name`) — and in whitespace / key order — share an identical `blake3:`
//!   semantic hash (§17 r10);
//! - any change to a real semantic field (a param, an op id, a reference) still
//!   moves the hash;
//! - normalization is idempotent and survives a serialize -> reparse replay with
//!   no hash drift.

use paintop_ir::{normalize, parse_plan, semantic_hash, to_canonical_string};
use serde_json::{Value, json};

const FIXTURES: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/fixtures/plans");

fn load(name: &str) -> String {
    let path = format!("{FIXTURES}/{name}");
    std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read fixture {path}: {e}"))
}

/// Mutate a fixture's JSON via `edit` and return the re-serialized text.
fn edited(name: &str, edit: impl FnOnce(&mut serde_json::Map<String, Value>)) -> String {
    let mut value: Value = serde_json::from_str(&load(name)).expect("fixture is valid JSON");
    edit(value.as_object_mut().expect("plan is a JSON object"));
    value.to_string()
}

#[test]
fn description_only_diff_hashes_identically_over_a_real_fixture() {
    // The touchup fixture already carries a `description`; rewrite ONLY that prose.
    let base = load("replay-touchup.json");
    let rephrased = edited("replay-touchup.json", |obj| {
        obj.insert(
            "description".to_owned(),
            json!("totally different non-semantic prose"),
        );
    });

    let base_plan = parse_plan(&base).expect("base fixture parses");
    let rephrased_plan = parse_plan(&rephrased).expect("rephrased plan parses");

    assert_eq!(
        semantic_hash(&base_plan).expect("base hashes"),
        semantic_hash(&rephrased_plan).expect("rephrased hashes"),
        "a description-only edit must not move the semantic hash (IR_SPEC §17 r10)"
    );
}

#[test]
fn name_only_diff_hashes_identically() {
    let base = load("replay-touchup.json");
    let renamed = edited("replay-touchup.json", |obj| {
        obj.insert("name".to_owned(), json!("a-different-human-name"));
    });
    assert_eq!(
        semantic_hash(&parse_plan(&base).unwrap()).unwrap(),
        semantic_hash(&parse_plan(&renamed).unwrap()).unwrap(),
        "a name-only edit is non-semantic"
    );
}

#[test]
fn dropping_the_description_entirely_hashes_identically() {
    let base = load("replay-touchup.json");
    let stripped = edited("replay-touchup.json", |obj| {
        obj.remove("description");
        obj.remove("name");
    });
    assert_eq!(
        semantic_hash(&parse_plan(&base).unwrap()).unwrap(),
        semantic_hash(&parse_plan(&stripped).unwrap()).unwrap(),
        "a described plan and the same plan with no prose are the same plan"
    );
}

#[test]
fn a_real_semantic_field_change_moves_the_hash() {
    let base = load("replay-touchup.json");
    let baseline = semantic_hash(&parse_plan(&base).unwrap()).unwrap();

    // Change a real param value: the ellipse angle.
    let reparam = base.replace("-0.16", "-0.17");
    assert_ne!(
        baseline,
        semantic_hash(&parse_plan(&reparam).unwrap()).unwrap(),
        "a changed parameter is a semantic change"
    );

    // Rewire a reference.
    let rewired = base.replace("node:base/image", "node:linear/image");
    assert_ne!(
        baseline,
        semantic_hash(&parse_plan(&rewired).unwrap()).unwrap(),
        "a changed reference is a semantic change"
    );

    // Bump an op semantic major version.
    let bumped = base.replace("color.convert@1", "color.convert@2");
    assert_ne!(
        baseline,
        semantic_hash(&parse_plan(&bumped).unwrap()).unwrap(),
        "an op semantic version is part of the identity"
    );
}

#[test]
fn normalize_is_idempotent_over_the_fixture() {
    let plan = parse_plan(&load("replay-touchup.json")).unwrap();
    let once = normalize(&plan);
    let twice = normalize(&once);
    assert_eq!(once, twice, "normalize(normalize(x)) == normalize(x)");
}

#[test]
fn normalized_plan_replays_without_hash_drift() {
    // Normalize -> canonical text -> reparse -> re-normalize must preserve the
    // semantic hash (`AGENT_VERIFICATION` §2.4).
    let plan = parse_plan(&load("replay-touchup.json")).unwrap();
    let first = semantic_hash(&plan).unwrap();

    let normalized = normalize(&plan);
    let value = serde_json::to_value(&normalized).unwrap();
    let canonical = to_canonical_string(&value).unwrap();
    let replayed = parse_plan(&canonical).unwrap();

    assert_eq!(
        first,
        semantic_hash(&replayed).unwrap(),
        "normalize -> serialize -> reparse must not drift the hash"
    );
}
