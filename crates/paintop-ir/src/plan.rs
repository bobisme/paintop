//! The **plan document** serde model (`IR_SPEC` §2, §4, §5).
//!
//! A plan is the strict, machine-authored description of an editing graph: a
//! language version, optional human metadata, a `policy` block, typed external
//! `inputs`, the operation `nodes`, postcondition `assertions`, explicit
//! `exports`, requested `evidence`, and namespaced `extensions`.
//!
//! This module owns the *structural* contract — the exact set of fields each
//! object may carry — and enforces it with `#[serde(deny_unknown_fields)]` on
//! every struct so that a typo (`nodez`, `parmas`, `prot`) fails the parse
//! rather than being silently ignored (`M0_DECISIONS` D4, `IR_SPEC` §2:
//! "Unknown top-level or node fields are errors.").
//!
//! Deeper validation that is *not* expressible as struct shape — duplicate JSON
//! keys, document size/nesting limits, the node-id regex, `NaN`/`Infinity`
//! rejection, and reference-syntax checks (`node:<id>/<port>`) — is layered on
//! by the parse-pipeline bone (`plan.md` §10.1) on top of these types. Here,
//! references and ids are carried as opaque strings; `params`/`hints`/`policy`/
//! `extensions` are carried as canonical JSON so this module need not know every
//! op's parameter schema.
//!
//! # Entry point
//!
//! [`parse_plan`] deserializes plan JSON into a [`Plan`] and maps any serde
//! failure onto the central error taxonomy (`IR_SPEC` §19): an unknown field
//! becomes a [`schema`](crate::ErrorClass::Schema) error with code
//! `E_UNKNOWN_FIELD`; a malformed token becomes a
//! [`parse`](crate::ErrorClass::Parse) error.
//!
//! ```
//! use paintop_ir::plan::parse_plan;
//!
//! let plan = parse_plan(r#"{
//!     "paintop": "1.0",
//!     "inputs": {},
//!     "nodes": [],
//!     "exports": {}
//! }"#)
//! .expect("minimal plan parses");
//! assert_eq!(plan.paintop, "1.0");
//! assert!(plan.nodes.is_empty());
//! ```

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::error::{Error, ErrorClass, ErrorContext, Result};

/// A namespaced extensions bag: experimental, non-core metadata keyed by a
/// reverse-domain or repository-qualified namespace (`IR_SPEC` §2.2).
///
/// The values are arbitrary canonical JSON; this crate does not interpret them.
/// A `BTreeMap` keeps the keys ordered so canonicalization is deterministic.
pub type Extensions = BTreeMap<String, serde_json::Value>;

/// The top-level plan document (`IR_SPEC` §2).
///
/// Required fields (`paintop`, `inputs`, `nodes`, `exports`) have no defaults;
/// the optional fields (`name`, `description`, `policy`, `assertions`,
/// `evidence`, `extensions`) default to empty so a minimal plan need only carry
/// the four required keys. Unknown top-level fields are rejected.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Plan {
    /// Plan language major/minor version, e.g. `"1.0"`. Carried as a string so
    /// the loader can reason about compatibility without losing the minor.
    pub paintop: String,
    /// Stable human-readable plan name (non-semantic).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// Free-form description. Excluded from content identity unless policy says
    /// otherwise (`IR_SPEC` §2.2).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// Resource, execution, model, path, and determinism constraints. Carried as
    /// canonical JSON; the policy schema is owned by the policy bone.
    #[serde(default, skip_serializing_if = "serde_json::Map::is_empty")]
    pub policy: serde_json::Map<String, serde_json::Value>,
    /// External resource declarations, keyed by input id (`IR_SPEC` §4).
    pub inputs: BTreeMap<String, InputDecl>,
    /// Operation nodes in any topological-compatible order (`IR_SPEC` §5).
    pub nodes: Vec<Node>,
    /// Postconditions or cross-resource predicates. Carried as canonical JSON;
    /// the assertion schema is owned by the assertions bone.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub assertions: Vec<serde_json::Value>,
    /// Explicit resource sinks, keyed by export id (`IR_SPEC` §2.1).
    pub exports: BTreeMap<String, serde_json::Value>,
    /// Requested debug artifacts and trace detail. Carried as canonical JSON.
    #[serde(default, skip_serializing_if = "serde_json::Map::is_empty")]
    pub evidence: serde_json::Map<String, serde_json::Value>,
    /// Namespaced non-core metadata (`IR_SPEC` §2.2).
    #[serde(default, skip_serializing_if = "Extensions::is_empty")]
    pub extensions: Extensions,
}

/// An external input declaration (`IR_SPEC` §4).
///
/// Inputs are typed by `kind` (`image.file`, `mask.file`, `json.file`,
/// `binary.file`, …) and policy-bound. The decode and limit blocks vary by kind,
/// so they are carried as canonical JSON and validated by the input-loading bone;
/// the structural contract enforced here is only the closed set of top-level
/// keys.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InputDecl {
    /// The input kind, e.g. `image.file`, `mask.file`, `json.file`.
    pub kind: String,
    /// The source path, resolved under the invocation's declared input root.
    /// Plans cannot escape roots (enforced by the loader, `IR_SPEC` §4).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub path: Option<String>,
    /// Kind-specific decode directives (color/alpha/format/channel). Carried as
    /// canonical JSON.
    #[serde(default, skip_serializing_if = "serde_json::Map::is_empty")]
    pub decode: serde_json::Map<String, serde_json::Value>,
    /// Kind-specific resource limits (max width/height/pixels). Carried as
    /// canonical JSON.
    #[serde(default, skip_serializing_if = "serde_json::Map::is_empty")]
    pub limits: serde_json::Map<String, serde_json::Value>,
}

/// A single operation node (`IR_SPEC` §5).
///
/// `id` and `op` are always required; `in` carries the wired input ports under a
/// single object (`M0_DECISIONS` D3 Q1); `params`, `hints`, and `extensions` are
/// optional. Input references are carried as opaque strings — their
/// `node:<id>/<port>` / `input:<id>` syntax (D3 Q2) is validated by the parse
/// pipeline, not by struct shape.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Node {
    /// The unique node id (`IR_SPEC` §3.1). The id regex is enforced by the
    /// parse pipeline.
    pub id: String,
    /// The versioned operation id, e.g. `filter.gaussian_blur@1` (`IR_SPEC` §6).
    pub op: String,
    /// Wired input ports: a single object mapping port name to a reference
    /// string (`M0_DECISIONS` D3 Q1). Absent when the op has no inputs.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty", rename = "in")]
    pub inputs: BTreeMap<String, String>,
    /// Operation parameters. Carried as canonical JSON; validated against the
    /// op's manifest by the type-check bone.
    #[serde(default, skip_serializing_if = "serde_json::Map::is_empty")]
    pub params: serde_json::Map<String, serde_json::Value>,
    /// Schedule hints that may influence execution but not semantics
    /// (`IR_SPEC` §5).
    #[serde(default, skip_serializing_if = "serde_json::Map::is_empty")]
    pub hints: serde_json::Map<String, serde_json::Value>,
    /// Namespaced non-core node metadata.
    #[serde(default, skip_serializing_if = "Extensions::is_empty")]
    pub extensions: Extensions,
}

/// Parse plan JSON into a [`Plan`], mapping serde failures onto the central
/// error taxonomy (`IR_SPEC` §19).
///
/// The strict front door runs in three phases (`plan.md` §10.1 phase 1–2):
/// 0. [`check_limits`](crate::limits::check_limits) enforces the hard structural
///    ceilings (byte size, nesting depth, node count, inline payload) *before*
///    anything is allocated, so an oversized or pathologically deep document
///    fails in O(1)/bounded work rather than driving the builder to exhaust
///    memory (`plan.md` §10.1: "apply strict size/depth limits before allocating
///    large structures").
/// 1. [`scan_json`](crate::scan::scan_json) walks the raw token stream *before*
///    any typed model is allocated, rejecting duplicate object keys (which
///    `serde_json` would otherwise silently keep last-wins, `IR_SPEC` §17 r12)
///    and invalid numeric forms (`NaN`/`Infinity`/non-round-trippable
///    magnitudes, `IR_SPEC` §10).
/// 2. `serde` deserialization enforces the structural contract via
///    `#[serde(deny_unknown_fields)]` on every plan struct: an unknown top-level
///    or node field (a typo) is rejected rather than silently ignored.
///
/// The limits applied are [`PlanLimits::DEFAULT`](crate::limits::PlanLimits::DEFAULT);
/// reference-syntax checks and the per-plan `policy.resources` budgets are
/// layered on by later parse-pipeline bones. Use
/// [`check_limits`](crate::limits::check_limits) directly to supply custom limits.
///
/// # Errors
/// - [`policy`](ErrorClass::Policy) / `E_MAX_PLAN_BYTES`, `E_MAX_DEPTH`,
///   `E_MAX_NODES`, or `E_MAX_INLINE_PAYLOAD` if the document exceeds a hard
///   structural ceiling.
/// - [`parse`](ErrorClass::Parse) / `E_DUPLICATE_KEY` if any object carries the
///   same key twice.
/// - [`parse`](ErrorClass::Parse) / `E_INVALID_NUMBER` for `NaN`, `Infinity`, or
///   a numeric token that does not round-trip through `f64`.
/// - [`schema`](ErrorClass::Schema) / `E_UNKNOWN_FIELD` if any object carries a
///   field outside its declared set.
/// - [`schema`](ErrorClass::Schema) / `E_MISSING_FIELD` if a required field is
///   absent.
/// - [`parse`](ErrorClass::Parse) / `E_INVALID_JSON` for any other
///   deserialization failure (malformed JSON, wrong value type, …).
pub fn parse_plan(json: &str) -> Result<Plan> {
    // Phase 0: enforce the hard structural ceilings (byte size, nesting depth,
    // node count, inline payload) before any walk or allocation, so an oversized
    // or pathologically deep document fails fast (`plan.md` §10.1 phase 1).
    crate::limits::check_limits(json, &crate::limits::PlanLimits::DEFAULT)?;
    // Phase 1: reject duplicate keys and invalid numbers before allocating the
    // typed model. An adversarial document fails here, fast.
    crate::scan::scan_json(json)?;
    // Phase 2: structural deserialization with deny_unknown_fields.
    serde_json::from_str(json).map_err(|err| map_serde_error(&err))
}

/// Map a `serde_json` deserialization error onto the central taxonomy.
///
/// Unknown-field and missing-field failures are *schema* errors (the document
/// parsed as JSON but violated the plan schema); everything else (syntax,
/// type-mismatch, EOF) is a *parse* error. The classification keys off the
/// stable `serde_json` message prefixes, which are part of its public contract.
fn map_serde_error(err: &serde_json::Error) -> Error {
    let message = err.to_string();
    let line = err.line();
    let column = err.column();
    let path = (line > 0).then(|| format!("line {line}, column {column}"));

    let (class, code) = if message.starts_with("unknown field") {
        (ErrorClass::Schema, "E_UNKNOWN_FIELD")
    } else if message.starts_with("missing field") {
        (ErrorClass::Schema, "E_MISSING_FIELD")
    } else {
        (ErrorClass::Parse, "E_INVALID_JSON")
    };

    let mut context = ErrorContext::default();
    if let Some(path) = path {
        context = context.with_path(path);
    }
    Error::new(class, code, message).with_context(context)
}

#[cfg(test)]
mod tests {
    use super::{Node, Plan, parse_plan};
    use crate::error::ErrorClass;
    use serde_json::json;

    const MINIMAL_VALID: &str = r#"{
        "paintop": "1.0",
        "inputs": {},
        "nodes": [],
        "exports": {}
    }"#;

    #[test]
    fn minimal_plan_parses_and_defaults_optionals() {
        let plan = parse_plan(MINIMAL_VALID).unwrap();
        assert_eq!(plan.paintop, "1.0");
        assert!(plan.name.is_none());
        assert!(plan.description.is_none());
        assert!(plan.policy.is_empty());
        assert!(plan.inputs.is_empty());
        assert!(plan.nodes.is_empty());
        assert!(plan.assertions.is_empty());
        assert!(plan.exports.is_empty());
        assert!(plan.evidence.is_empty());
        assert!(plan.extensions.is_empty());
    }

    #[test]
    fn full_plan_round_trips_through_serde() {
        let value = json!({
            "paintop": "1.0",
            "name": "darken-and-repair",
            "description": "Localized repair with bounded changes",
            "policy": {"resources": {"max_nodes": 64}},
            "inputs": {
                "source": {
                    "kind": "image.file",
                    "path": "input.png",
                    "decode": {"desired_format": "rgba8", "alpha": "straight"},
                    "limits": {"max_pixels": 67_108_864}
                }
            },
            "nodes": [{
                "id": "blurred",
                "op": "filter.gaussian_blur@1",
                "in": {"image": "input:source"},
                "params": {"sigma": 8.0},
                "hints": {"materialize": false},
                "extensions": {"com.example.note": "hi"}
            }],
            "assertions": [{"kind": "assert.no_change_outside_mask@1"}],
            "exports": {"result": {"from": "node:blurred/image"}},
            "evidence": {"trace": "summary"},
            "extensions": {"com.example.meta": 1}
        });
        let plan: Plan = serde_json::from_value(value.clone()).unwrap();
        // The model carries the structural fields losslessly.
        assert_eq!(plan.name.as_deref(), Some("darken-and-repair"));
        assert_eq!(plan.nodes.len(), 1);
        assert_eq!(plan.nodes[0].id, "blurred");
        assert_eq!(plan.nodes[0].inputs["image"], "input:source");
        // Re-serializing yields an equal JSON value.
        let back = serde_json::to_value(&plan).unwrap();
        assert_eq!(back, value);
    }

    #[test]
    fn unknown_top_level_field_is_schema_error() {
        let json = r#"{
            "paintop": "1.0",
            "inputs": {},
            "nodes": [],
            "exports": {},
            "nodez": []
        }"#;
        let err = parse_plan(json).unwrap_err();
        assert_eq!(err.class, ErrorClass::Schema);
        assert_eq!(err.code, "E_UNKNOWN_FIELD");
        assert!(err.message.contains("nodez"));
    }

    #[test]
    fn unknown_node_field_is_schema_error() {
        let json = r#"{
            "paintop": "1.0",
            "inputs": {},
            "nodes": [{"id": "n", "op": "filter.gaussian_blur@1", "parmas": {}}],
            "exports": {}
        }"#;
        let err = parse_plan(json).unwrap_err();
        assert_eq!(err.class, ErrorClass::Schema);
        assert_eq!(err.code, "E_UNKNOWN_FIELD");
        assert!(err.message.contains("parmas"));
    }

    #[test]
    fn unknown_input_decl_field_is_schema_error() {
        let json = r#"{
            "paintop": "1.0",
            "inputs": {"source": {"kind": "image.file", "pth": "x.png"}},
            "nodes": [],
            "exports": {}
        }"#;
        let err = parse_plan(json).unwrap_err();
        assert_eq!(err.class, ErrorClass::Schema);
        assert_eq!(err.code, "E_UNKNOWN_FIELD");
    }

    #[test]
    fn missing_required_top_level_field_is_schema_error() {
        // `nodes` omitted.
        let json = r#"{"paintop": "1.0", "inputs": {}, "exports": {}}"#;
        let err = parse_plan(json).unwrap_err();
        assert_eq!(err.class, ErrorClass::Schema);
        assert_eq!(err.code, "E_MISSING_FIELD");
        assert!(err.message.contains("nodes"));
    }

    #[test]
    fn missing_required_node_field_is_schema_error() {
        // node `op` omitted.
        let json = r#"{
            "paintop": "1.0",
            "inputs": {},
            "nodes": [{"id": "n"}],
            "exports": {}
        }"#;
        let err = parse_plan(json).unwrap_err();
        assert_eq!(err.class, ErrorClass::Schema);
        assert_eq!(err.code, "E_MISSING_FIELD");
    }

    #[test]
    fn malformed_json_is_parse_error() {
        let err = parse_plan("{ this is not json").unwrap_err();
        assert_eq!(err.class, ErrorClass::Parse);
        assert_eq!(err.code, "E_INVALID_JSON");
    }

    #[test]
    fn wrong_value_type_is_parse_error() {
        // `nodes` must be an array, not a string.
        let json = r#"{"paintop": "1.0", "inputs": {}, "nodes": "x", "exports": {}}"#;
        let err = parse_plan(json).unwrap_err();
        assert_eq!(err.class, ErrorClass::Parse);
        assert_eq!(err.code, "E_INVALID_JSON");
    }

    #[test]
    fn node_without_inputs_defaults_to_empty_in() {
        let json = r#"{"id": "n", "op": "mask.rect@1"}"#;
        let node: Node = serde_json::from_str(json).unwrap();
        assert!(node.inputs.is_empty());
        assert!(node.params.is_empty());
        // Round-trips without re-emitting the empty `in`.
        let back = serde_json::to_value(&node).unwrap();
        assert_eq!(back, json!({"id": "n", "op": "mask.rect@1"}));
    }

    #[test]
    fn parse_error_carries_location_context() {
        let err = parse_plan("{ \"paintop\": ").unwrap_err();
        assert_eq!(err.class, ErrorClass::Parse);
        assert!(err.context.path.is_some());
    }

    #[test]
    fn duplicate_key_is_rejected_before_deserialization() {
        // `paintop` declared twice: serde would keep the last; the scan rejects.
        let json = r#"{
            "paintop": "1.0",
            "paintop": "9.9",
            "inputs": {},
            "nodes": [],
            "exports": {}
        }"#;
        let err = parse_plan(json).unwrap_err();
        assert_eq!(err.class, ErrorClass::Parse);
        assert_eq!(err.code, "E_DUPLICATE_KEY");
    }

    #[test]
    fn nan_param_is_rejected_at_parse() {
        let json = r#"{
            "paintop": "1.0",
            "inputs": {},
            "nodes": [{"id": "n", "op": "filter.gaussian_blur@1", "params": {"sigma": NaN}}],
            "exports": {}
        }"#;
        let err = parse_plan(json).unwrap_err();
        assert_eq!(err.class, ErrorClass::Parse);
        assert_eq!(err.code, "E_INVALID_NUMBER");
    }

    #[test]
    fn scan_runs_before_schema_check() {
        // A document with BOTH a duplicate key and an unknown field must fail on
        // the duplicate (phase 1) before the schema phase ever runs.
        let json = r#"{
            "paintop": "1.0",
            "paintop": "1.0",
            "inputs": {},
            "nodes": [],
            "exports": {},
            "nodez": []
        }"#;
        let err = parse_plan(json).unwrap_err();
        assert_eq!(err.code, "E_DUPLICATE_KEY");
    }
}
