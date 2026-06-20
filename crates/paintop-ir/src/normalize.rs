//! Plan **normalization** — the semantic projection a plan's semantic hash is
//! computed over (`IR_SPEC` §17; `M0_DECISIONS` D4).
//!
//! Canonicalization ([`to_canonical_bytes`](crate::to_canonical_bytes)) fixes the
//! *byte shape* of a value — sorted keys, stable floats, no whitespace — so two
//! values that are already equal up to formatting hash identically. It does
//! **not**, on its own, decide *which fields carry meaning*. `IR_SPEC` §17 draws
//! that second line: rule **r10** ("Remove non-semantic comments/descriptions
//! from the semantic hash") means two plans that differ only in human prose are
//! the *same* plan and must share a semantic id. Hashing the raw plan violates
//! this — a `description` edit moves the hash.
//!
//! [`normalize`] is that semantic projection. It walks a parsed [`Plan`] and
//! drops the documented non-semantic annotation fields before the plan is
//! canonicalized and hashed, so a description-only (or name-only) edit is
//! invisible to the semantic hash while any real semantic change still moves it.
//! Plan semantic hashing routes through it via [`normalized_value`] /
//! [`semantic_hash`].
//!
//! # What is stripped (M0)
//!
//! The fields the central plan model already documents as carrying no semantics:
//!
//! - the plan `description` (`IR_SPEC` §2.2, §17 r10 — free-form prose);
//! - the plan `name` (`IR_SPEC` §2.2 — a "stable human-readable plan name",
//!   non-semantic identity for humans, not for content);
//! - each node's `hints` block (`IR_SPEC` §5 — "schedule hints that may influence
//!   execution but **not** semantics").
//!
//! Everything else — `paintop`, `policy`, `inputs`, `nodes` (id / op / wired
//! `in` / `params`), `assertions`, `exports`, `evidence`, and `extensions` — is
//! preserved verbatim and carried into the semantic hash, so any change to a real
//! semantic field still flips the id.
//!
//! # What is deferred
//!
//! `IR_SPEC` §17 rules **r1–r4** (resolve defaults, resolve seeds, expand
//! aliases / sequential sugar, emit canonical op ids) and **r13** (namespace-gated
//! extension inclusion) all require the **operation manifests** that only exist in
//! segment 2. In particular **default injection** (filling a param a manifest
//! declares a default for, so an omitted param and an explicitly-default param
//! hash alike) cannot be faked without a manifest's parameter schema. That hook is
//! the private `inject_manifest_defaults` step (a no-op today) and is exercised by
//! a tracked `#[ignore]`d follow-up test rather than stubbed with fake data.
//!
//! # Idempotence
//!
//! [`normalize`] is a pure projection: it only ever *removes* fields, never adds
//! or reorders, so `normalize(normalize(p)) == normalize(p)`. The follow-up
//! manifest-default step must preserve that fixed-point property.

use serde_json::Value;

use crate::error::Result;
use crate::hash::{HashDomain, SemanticHash, hash_value};
use crate::plan::{Node, Plan};

/// Produce the **normalized** semantic form of a plan (`IR_SPEC` §17).
///
/// The returned [`Plan`] is the input with every documented non-semantic
/// annotation removed (`description`, `name`, and each node's `hints`); all
/// semantic fields are preserved unchanged. This is the value the plan's semantic
/// hash is computed over.
///
/// The function is idempotent — it only strips fields — so
/// `normalize(&normalize(&p)) == normalize(&p)`.
#[must_use]
pub fn normalize(plan: &Plan) -> Plan {
    let mut normalized = plan.clone();

    // r10: non-semantic human prose / identity is excluded from semantic identity.
    normalized.name = None;
    normalized.description = None;

    for node in &mut normalized.nodes {
        normalize_node(node);
    }

    // r1–r4 / r13 hook: manifest-driven default injection, seed resolution, alias
    // expansion, and namespace-gated extension inclusion land with real op
    // manifests (segment 2). Deferred, not faked.
    inject_manifest_defaults(&mut normalized);

    normalized
}

/// Strip a node's documented non-semantic fields.
///
/// `hints` are "schedule hints that may influence execution but not semantics"
/// (`IR_SPEC` §5), so they are removed before hashing. `id`, `op`, the wired
/// `in` references, `params`, and `extensions` are semantic and kept verbatim.
fn normalize_node(node: &mut Node) {
    node.hints.clear();
}

/// **Deferred (segment 2):** inject manifest-declared parameter defaults so that
/// an omitted parameter and an explicitly-default parameter normalize — and
/// therefore hash — identically (`IR_SPEC` §17 r2).
///
/// This cannot be implemented faithfully in M0: without an operation's manifest
/// there is no parameter schema to read defaults from, and inventing one would
/// silently corrupt the semantic hash. Until the manifest registry carries real
/// op manifests (segment 2) this is intentionally a no-op; the follow-up is
/// pinned by a tracked `#[ignore]`d test (`defaults_injection_is_manifest_driven`)
/// so the gap is visible, not forgotten.
///
/// When implemented it must remain a pure, idempotent projection so that
/// [`normalize`]'s fixed-point property is preserved.
#[allow(
    clippy::needless_pass_by_ref_mut,
    reason = "the signature is the segment-2 hook: default injection mutates the plan in place; kept mutable so callers and the follow-up test bind the final shape"
)]
#[allow(
    clippy::missing_const_for_fn,
    reason = "the segment-2 implementation will read op manifests and mutate the plan, neither of which is const; keeping it non-const pins the real signature today"
)]
fn inject_manifest_defaults(_plan: &mut Plan) {
    // Intentionally empty until operation manifests exist (segment 2).
}

/// The normalized plan as a canonical-ready [`serde_json::Value`].
///
/// This is [`normalize`] followed by `serde_json::to_value`; it is the exact
/// value [`semantic_hash`] canonicalizes and hashes. Exposed so callers (CLI,
/// evidence emitter) can render the normalized plan without re-deriving the rule
/// set.
///
/// # Errors
/// Returns the serialization error if the normalized plan cannot be turned into a
/// [`serde_json::Value`]; a plan that came through
/// [`parse_plan`](crate::plan::parse_plan) cannot hit this case.
pub fn normalized_value(plan: &Plan) -> Result<Value> {
    let normalized = normalize(plan);
    // A `Plan` is a plain serde struct over string-keyed maps, so this conversion
    // is infallible in practice; surface a typed error rather than `expect` to
    // honor the no-unwrap rule (mirrors `to_canonical_string`'s UTF-8 guard).
    serde_json::to_value(&normalized).map_err(|err| {
        crate::error::Error::new(
            crate::error::ErrorClass::Schema,
            "E_NORMALIZE_SERIALIZE",
            format!("normalized plan could not be serialized to a value: {err}"),
        )
    })
}

/// The plan **semantic hash** (`IR_SPEC` §17): the BLAKE3 of the canonical bytes
/// of the *normalized* plan, in the [`Plan`](HashDomain::Plan) domain.
///
/// This is the single entry point for plan content identity: it routes through
/// [`normalize`] so non-semantic prose can never reach the digest. Two plans that
/// differ only in `description` / `name` / node `hints` share this id; any
/// semantic difference flips it.
///
/// # Errors
/// Returns the canonicalization error if the normalized plan carries a non-finite
/// float (impossible for a plan from [`parse_plan`](crate::plan::parse_plan)), or
/// the serialization error from [`normalized_value`].
pub fn semantic_hash(plan: &Plan) -> Result<SemanticHash> {
    let value = normalized_value(plan)?;
    hash_value(HashDomain::Plan, &value)
}

#[cfg(test)]
mod tests {
    use super::{normalize, normalized_value, semantic_hash};
    use crate::plan::parse_plan;
    use serde_json::json;

    /// A small plan with a description, a name, and a node carrying hints — every
    /// stripped field is present so normalization has something to remove.
    fn described_plan(description: &str, name: &str) -> String {
        json!({
            "paintop": "1.0",
            "name": name,
            "description": description,
            "inputs": {},
            "nodes": [{
                "id": "b",
                "op": "filter.gaussian_blur@1",
                "params": {"sigma": 8.0},
                "hints": {"materialize": false}
            }],
            "exports": {}
        })
        .to_string()
    }

    #[test]
    fn normalize_strips_description_name_and_hints() {
        let plan = parse_plan(&described_plan("some prose", "my-plan")).unwrap();
        let normalized = normalize(&plan);
        assert!(normalized.description.is_none());
        assert!(normalized.name.is_none());
        assert!(normalized.nodes[0].hints.is_empty());
        // Semantic fields survive untouched.
        assert_eq!(normalized.nodes[0].id, "b");
        assert_eq!(normalized.nodes[0].op, "filter.gaussian_blur@1");
        assert_eq!(normalized.nodes[0].params["sigma"], json!(8.0));
    }

    #[test]
    fn description_only_difference_hashes_identically() {
        // The bone's core guarantee (`IR_SPEC` §17 r10): plans that differ ONLY in
        // non-semantic prose share a semantic hash.
        let a = parse_plan(&described_plan("first prose", "name-a")).unwrap();
        let b = parse_plan(&described_plan("completely different prose", "name-b")).unwrap();
        assert_eq!(
            semantic_hash(&a).unwrap(),
            semantic_hash(&b).unwrap(),
            "a description/name-only edit must not move the semantic hash"
        );
    }

    #[test]
    fn node_hints_only_difference_hashes_identically() {
        let with = parse_plan(&described_plan("p", "n")).unwrap();
        let mut value: serde_json::Value = serde_json::from_str(&described_plan("p", "n")).unwrap();
        // Drop the node's hints entirely in the second plan.
        value["nodes"][0].as_object_mut().unwrap().remove("hints");
        let without = parse_plan(&value.to_string()).unwrap();
        assert_eq!(
            semantic_hash(&with).unwrap(),
            semantic_hash(&without).unwrap(),
            "node hints are non-semantic and must not move the hash"
        );
    }

    #[test]
    fn a_real_semantic_change_still_moves_the_hash() {
        let base = parse_plan(&described_plan("prose", "name")).unwrap();
        let changed = parse_plan(&described_plan("prose", "name").replace("8.0", "8.5")).unwrap();
        assert_ne!(
            semantic_hash(&base).unwrap(),
            semantic_hash(&changed).unwrap(),
            "a changed parameter is a semantic change"
        );
    }

    #[test]
    fn normalize_is_idempotent() {
        let plan = parse_plan(&described_plan("prose", "name")).unwrap();
        let once = normalize(&plan);
        let twice = normalize(&once);
        assert_eq!(once, twice, "normalize must be a fixed point");
        // And the hash agrees: normalizing an already-normalized plan is a no-op.
        assert_eq!(semantic_hash(&plan).unwrap(), semantic_hash(&once).unwrap());
    }

    #[test]
    fn normalized_value_matches_normalized_plan() {
        let plan = parse_plan(&described_plan("prose", "name")).unwrap();
        let value = normalized_value(&plan).unwrap();
        // The non-semantic keys are gone from the value form.
        let obj = value.as_object().unwrap();
        assert!(!obj.contains_key("description"));
        assert!(!obj.contains_key("name"));
        // Semantic structure remains.
        assert_eq!(obj["paintop"], json!("1.0"));
        assert!(obj.contains_key("nodes"));
    }

    /// **Deferred (segment 2):** manifest-driven default injection (`IR_SPEC` §17
    /// r2). An omitted parameter and an explicitly-default parameter must
    /// normalize — and therefore hash — identically, which requires reading the
    /// op's manifest for its declared defaults. There are no real op manifests in
    /// M0, so this is tracked and ignored rather than faked.
    #[test]
    #[ignore = "manifest-default injection (IR_SPEC §17 r2) is deferred to segment 2 when real op manifests exist"]
    fn defaults_injection_is_manifest_driven() {
        // When op manifests land, a node that omits a defaulted parameter and a
        // node that spells the default out explicitly must produce the SAME
        // semantic hash. Encoded against a hypothetical
        // `filter.gaussian_blur@1` whose `boundary` defaults to `"clamp"`.
        let omitted = parse_plan(
            &json!({
                "paintop": "1.0",
                "inputs": {},
                "nodes": [{"id": "b", "op": "filter.gaussian_blur@1", "params": {"sigma": 8.0}}],
                "exports": {}
            })
            .to_string(),
        )
        .unwrap();
        let explicit = parse_plan(
            &json!({
                "paintop": "1.0",
                "inputs": {},
                "nodes": [{
                    "id": "b",
                    "op": "filter.gaussian_blur@1",
                    "params": {"sigma": 8.0, "boundary": "clamp"}
                }],
                "exports": {}
            })
            .to_string(),
        )
        .unwrap();
        assert_eq!(
            semantic_hash(&omitted).unwrap(),
            semantic_hash(&explicit).unwrap(),
            "an omitted defaulted param must hash like its explicit default"
        );
    }
}
