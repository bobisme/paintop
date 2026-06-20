//! Fixture-driven coverage for the plan parser's structural front door
//! (`IR_SPEC` §2, §5). The positive fixture must round-trip; the negative
//! unknown-field fixtures must be rejected with the `schema` / `E_UNKNOWN_FIELD`
//! contract.

use paintop_ir::{ErrorClass, PlanLimits, check_limits, parse_plan};

const FIXTURES: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/fixtures/plans");

fn load(name: &str) -> String {
    let path = format!("{FIXTURES}/{name}");
    std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read fixture {path}: {e}"))
}

#[test]
fn empty_valid_fixture_parses() {
    let plan = parse_plan(&load("empty-valid.json")).expect("empty-valid plan must parse");
    assert_eq!(plan.paintop, "1.0");
    assert!(plan.nodes.is_empty());
    assert!(plan.inputs.is_empty());
    assert!(plan.exports.is_empty());
}

#[test]
fn unknown_top_level_field_fixture_is_schema_error() {
    let err = parse_plan(&load("unknown-top-level-field.json"))
        .expect_err("unknown top-level field must be rejected");
    assert_eq!(err.class, ErrorClass::Schema);
    assert_eq!(err.code, "E_UNKNOWN_FIELD");
}

#[test]
fn unknown_node_field_fixture_is_schema_error() {
    let err = parse_plan(&load("unknown-node-field.json"))
        .expect_err("unknown node field must be rejected");
    assert_eq!(err.class, ErrorClass::Schema);
    assert_eq!(err.code, "E_UNKNOWN_FIELD");
}

#[test]
fn duplicate_top_level_key_fixture_is_parse_error() {
    // serde_json would silently keep the last `exports`; the scan must reject
    // it (`IR_SPEC` §17 r12) *before* the typed plan is built.
    let err = parse_plan(&load("duplicate-key.json"))
        .expect_err("duplicate top-level key must be rejected");
    assert_eq!(err.class, ErrorClass::Parse);
    assert_eq!(err.code, "E_DUPLICATE_KEY");
}

#[test]
fn duplicate_node_param_key_fixture_is_parse_error() {
    let err = parse_plan(&load("duplicate-node-param-key.json"))
        .expect_err("duplicate nested param key must be rejected");
    assert_eq!(err.class, ErrorClass::Parse);
    assert_eq!(err.code, "E_DUPLICATE_KEY");
}

#[test]
fn nan_number_fixture_is_parse_error() {
    let err = parse_plan(&load("nan-number.json")).expect_err("NaN literal must be rejected");
    assert_eq!(err.class, ErrorClass::Parse);
    assert_eq!(err.code, "E_INVALID_NUMBER");
}

#[test]
fn infinity_number_fixture_is_parse_error() {
    let err =
        parse_plan(&load("infinity-number.json")).expect_err("Infinity literal must be rejected");
    assert_eq!(err.class, ErrorClass::Parse);
    assert_eq!(err.code, "E_INVALID_NUMBER");
}

#[test]
fn underflow_number_fixture_is_parse_error() {
    let err = parse_plan(&load("underflow-number.json"))
        .expect_err("non-round-trippable magnitude must be rejected");
    assert_eq!(err.class, ErrorClass::Parse);
    assert_eq!(err.code, "E_INVALID_NUMBER");
}

#[test]
fn overflow_number_fixture_is_parse_error() {
    // `1e400` coerces to infinity, which has no round-trippable float form.
    let err = parse_plan(&load("overflow-number.json"))
        .expect_err("overflowing magnitude must be rejected");
    assert_eq!(err.class, ErrorClass::Parse);
    assert_eq!(err.code, "E_INVALID_NUMBER");
}

/// A baseline of deliberately-generous ceilings that no committed fixture trips;
/// each limit test tightens exactly one field so only the intended cap fires.
/// (Sharing the default 64-byte/64-level/100k-node ceilings would force
/// multi-megabyte fixtures, so the static fixtures are checked against these.)
const LOOSE: PlanLimits = PlanLimits {
    max_bytes: 64 * 1024,
    max_depth: 64,
    max_nodes: 1000,
    max_inline_len: 1000,
};

#[test]
fn depth_limit_fixture_is_policy_error() {
    let json = load("limit-depth.json");
    // The default ceilings admit the fixture; only the tight depth cap rejects it.
    parse_plan(&json).expect("depth fixture is within the default ceilings");
    let limits = PlanLimits {
        max_depth: 6,
        ..LOOSE
    };
    let err = check_limits(&json, &limits).expect_err("excessive nesting must be rejected");
    assert_eq!(err.class, ErrorClass::Policy);
    assert_eq!(err.code, "E_MAX_DEPTH");
}

#[test]
fn node_count_limit_fixture_is_policy_error() {
    let json = load("limit-node-count.json");
    parse_plan(&json).expect("node-count fixture is within the default ceilings");
    let limits = PlanLimits {
        max_nodes: 4,
        ..LOOSE
    };
    let err = check_limits(&json, &limits).expect_err("too many graph nodes must be rejected");
    assert_eq!(err.class, ErrorClass::Policy);
    assert_eq!(err.code, "E_MAX_NODES");
    assert_eq!(err.context.path.as_deref(), Some("/nodes"));
}

#[test]
fn inline_splat_payload_limit_fixture_is_policy_error() {
    let json = load("limit-inline-splats.json");
    parse_plan(&json).expect("inline-splat fixture is within the default ceilings");
    // 30 is above any op-id/key string in the fixture but below the 40-element
    // `splats` batch, so the batch is what trips the inline cap.
    let limits = PlanLimits {
        max_inline_len: 30,
        ..LOOSE
    };
    let err =
        check_limits(&json, &limits).expect_err("an oversized inline splat batch must be rejected");
    assert_eq!(err.class, ErrorClass::Policy);
    assert_eq!(err.code, "E_MAX_INLINE_PAYLOAD");
    assert_eq!(err.context.path.as_deref(), Some("/nodes/0/params/splats"));
}
