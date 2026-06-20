//! Minimal-replay reduction (`AGENT_VERIFICATION` §5.4, `plan.md` §18.2).
//!
//! When an assertion fails, an agent does not want the whole plan — it wants the
//! smallest self-contained reproducer that still fails. This module reduces a
//! parsed [`Plan`] to a **minimal replay**: the transitive producer cone of the
//! failing assertion's target node, the external inputs that cone actually
//! reads, and a [`ReplaySpec`] pinning the reproduction context (fixed
//! backend/implementation, exact seed, failing ROI plus halo, the failing
//! assertion). This is the compiler-testcase-reducer analogue §5.4 calls for —
//! the difference between an agent re-running a 200-node plan and attacking a
//! three-node bug specimen.
//!
//! ## What "minimal" means here
//!
//! M0's reducer is **structural and conservative**: it keeps exactly the nodes
//! the target transitively depends on (via `node:<id>/<port>` input references),
//! drops every unrelated node and export, and prunes the inputs map to the
//! external resources the surviving cone references (`input:<id>`). It does *not*
//! yet crop rasters or delta-debug parameter values — that is a later bone — so
//! the emitted replay is a faithful, smaller plan, never a semantically altered
//! one. The pinned context (`ReplaySpec`) is carried alongside the reduced plan
//! rather than mutated into it, so the replay re-parses as an ordinary plan.
//!
//! ## What this bone owns
//!
//! The reduction algorithm + the replay document schema. The assertion stage
//! (later bones) supplies the failing node and the [`ReplaySpec`]; the
//! [`BundleWriter`](crate::evidence::BundleWriter) writes the canonical replay
//! JSON under `replays/<assertion-id>.json`.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use paintop_ir::{Plan, Rect};

use crate::evidence::trace::ImplRef;

/// The `input:<id>` reference prefix that names an external plan input.
const INPUT_PREFIX: &str = "input:";
/// The `node:<id>/<port>` reference prefix that names another node's output.
const NODE_PREFIX: &str = "node:";

/// The pinned reproduction context of a minimal replay (`AGENT_VERIFICATION`
/// §5.4).
///
/// Carries the knobs that make a failure *reproducible*: which assertion failed,
/// the node it targets, the fixed implementation to dispatch on, the exact seed,
/// and the failing region of interest plus the halo the op reads around it.
/// Every field beyond the assertion/node identity is optional so a partial
/// specimen (e.g. before the ROI is known) still serializes cleanly.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReplaySpec {
    /// The stable id of the failing assertion this replay reproduces.
    pub assertion: String,
    /// The graph node the failing assertion targets (the reduction root).
    pub target: String,
    /// The implementation to pin the dispatch to, so the replay is not
    /// re-selected onto a different backend (`plan.md` §18.2 "fixed
    /// backend/implementation").
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub implementation: Option<ImplRef>,
    /// The exact seed to reproduce any stochastic behavior.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
    /// The failing region of interest (already including sufficient halo).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub roi: Option<Rect>,
}

impl ReplaySpec {
    /// Build a spec from the failing assertion id and its target node, with no
    /// pinned implementation / seed / ROI yet.
    #[must_use]
    pub fn new(assertion: impl Into<String>, target: impl Into<String>) -> Self {
        Self {
            assertion: assertion.into(),
            target: target.into(),
            implementation: None,
            seed: None,
            roi: None,
        }
    }

    /// Pin the implementation the replay must dispatch on.
    #[must_use]
    pub fn with_implementation(mut self, implementation: impl Into<ImplRef>) -> Self {
        self.implementation = Some(implementation.into());
        self
    }

    /// Pin the exact seed.
    #[must_use]
    pub const fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Pin the failing ROI (which the caller has already grown by the op halo).
    #[must_use]
    pub const fn with_roi(mut self, roi: Rect) -> Self {
        self.roi = Some(roi);
        self
    }
}

/// A minimal replay document: a reduced plan plus its pinned reproduction
/// context (`AGENT_VERIFICATION` §5.4).
///
/// The `plan` is the structurally reduced graph (transitive producer cone of the
/// target, pruned inputs, no unrelated nodes/exports); the `spec` pins the
/// implementation, seed, and ROI. The two are kept separate so the embedded plan
/// is an ordinary, re-parseable [`Plan`] — the replay context lives beside it,
/// not folded into the plan's semantic identity.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MinimalReplay {
    /// The reduced plan that still reproduces the failure.
    pub plan: Plan,
    /// The pinned reproduction context.
    pub spec: ReplaySpec,
}

impl MinimalReplay {
    /// The conventional bundle-relative path of a replay for assertion `id`,
    /// under the `replays/` subdirectory.
    #[must_use]
    pub fn path_for(assertion_id: &str) -> String {
        format!("replays/{assertion_id}.json")
    }

    /// Reduce `plan` to a minimal replay for the failure described by `spec`.
    ///
    /// Keeps exactly the transitive producer cone of `spec.target` (the target
    /// plus every node it reaches through `node:<id>/<port>` input references),
    /// prunes `inputs` to the external resources that cone reads, and drops every
    /// `export`. If the target is not present in the plan the cone is just the
    /// inputs it cannot resolve — the reduction stays total and never panics.
    #[must_use]
    pub fn reduce(plan: &Plan, spec: ReplaySpec) -> Self {
        let keep = transitive_producers(plan, &spec.target);

        let nodes: Vec<_> = plan
            .nodes
            .iter()
            .filter(|n| keep.contains(&n.id))
            .cloned()
            .collect();

        // Prune inputs to the external resources the surviving cone references.
        let referenced_inputs = referenced_input_ids(&nodes);
        let inputs: BTreeMap<_, _> = plan
            .inputs
            .iter()
            .filter(|(id, _)| referenced_inputs.contains(id.as_str()))
            .map(|(id, decl)| (id.clone(), decl.clone()))
            .collect();

        let reduced = Plan {
            paintop: plan.paintop.clone(),
            name: plan.name.clone(),
            description: None,
            policy: plan.policy.clone(),
            inputs,
            nodes,
            // The failing assertion is pinned in the spec; the reduced plan keeps
            // no other postconditions so the specimen fails for exactly one
            // reason.
            assertions: Vec::new(),
            // A replay reproduces an internal failure, not an export.
            exports: BTreeMap::new(),
            evidence: serde_json::Map::new(),
            extensions: BTreeMap::new(),
        };

        Self {
            plan: reduced,
            spec,
        }
    }
}

/// The set of node ids the `target` transitively depends on (inclusive of the
/// target), following `node:<id>/<port>` input references.
///
/// A breadth-first walk over the producer edges; a node referenced but absent
/// from the plan is simply not added (the reduction is conservative, not
/// validating — validation already happened upstream).
fn transitive_producers(plan: &Plan, target: &str) -> BTreeSet<String> {
    let by_id: BTreeMap<&str, &paintop_ir::Node> =
        plan.nodes.iter().map(|n| (n.id.as_str(), n)).collect();

    let mut keep = BTreeSet::new();
    let mut frontier = vec![target.to_owned()];
    while let Some(id) = frontier.pop() {
        if !keep.insert(id.clone()) {
            continue;
        }
        if let Some(node) = by_id.get(id.as_str()) {
            for reference in node.inputs.values() {
                if let Some(producer) = node_reference_id(reference) {
                    frontier.push(producer.to_owned());
                }
            }
        }
    }
    keep
}

/// The set of external input ids referenced by any node in `nodes`.
fn referenced_input_ids(nodes: &[paintop_ir::Node]) -> BTreeSet<&str> {
    let mut ids = BTreeSet::new();
    for node in nodes {
        for reference in node.inputs.values() {
            if let Some(input_id) = input_reference_id(reference) {
                ids.insert(input_id);
            }
        }
    }
    ids
}

/// Extract the producer node id from a `node:<id>/<port>` reference, or `None`
/// if the reference is not a node reference.
fn node_reference_id(reference: &str) -> Option<&str> {
    let rest = reference.strip_prefix(NODE_PREFIX)?;
    // `node:<id>/<port>`: the id is everything before the first `/`.
    Some(rest.split_once('/').map_or(rest, |(id, _)| id))
}

/// Extract the input id from an `input:<id>` reference, or `None` if the
/// reference is not an input reference.
fn input_reference_id(reference: &str) -> Option<&str> {
    reference.strip_prefix(INPUT_PREFIX)
}

#[cfg(test)]
mod tests {
    use super::{MinimalReplay, ReplaySpec};
    use paintop_ir::{Rect, parse_plan};

    /// A diamond plan: `out` depends on `left` and `right`, both of which read
    /// `src`; `unrelated` reads `src` but feeds nothing the target needs.
    const DIAMOND: &str = r#"{
        "paintop": "1.0",
        "name": "diamond",
        "inputs": {
            "src": {"kind": "image.file", "path": "src.png"},
            "extra": {"kind": "image.file", "path": "extra.png"}
        },
        "nodes": [
            {"id": "left",  "op": "filter.gaussian_blur@1", "in": {"image": "input:src"}},
            {"id": "right", "op": "filter.gaussian_blur@1", "in": {"image": "input:src"}},
            {"id": "out",   "op": "blend.over@1", "in": {"a": "node:left/image", "b": "node:right/image"}},
            {"id": "unrelated", "op": "filter.gaussian_blur@1", "in": {"image": "input:extra"}}
        ],
        "exports": {"result": {"from": "node:out/image"}}
    }"#;

    #[test]
    fn reduction_keeps_only_the_target_cone() {
        let plan = parse_plan(DIAMOND).expect("parse");
        let replay = MinimalReplay::reduce(&plan, ReplaySpec::new("localized", "out"));
        let kept: Vec<_> = replay.plan.nodes.iter().map(|n| n.id.as_str()).collect();
        assert!(kept.contains(&"out"));
        assert!(kept.contains(&"left"));
        assert!(kept.contains(&"right"));
        // The unrelated node is dropped.
        assert!(!kept.contains(&"unrelated"));
        assert_eq!(kept.len(), 3);
    }

    #[test]
    fn reduction_prunes_unreferenced_inputs_and_exports() {
        let plan = parse_plan(DIAMOND).expect("parse");
        let replay = MinimalReplay::reduce(&plan, ReplaySpec::new("localized", "out"));
        // Only `src` survives; `extra` was read only by the dropped node.
        assert!(replay.plan.inputs.contains_key("src"));
        assert!(!replay.plan.inputs.contains_key("extra"));
        // A replay carries no exports or postconditions.
        assert!(replay.plan.exports.is_empty());
        assert!(replay.plan.assertions.is_empty());
    }

    #[test]
    fn reduced_replay_plan_reparses() {
        let plan = parse_plan(DIAMOND).expect("parse");
        let replay = MinimalReplay::reduce(&plan, ReplaySpec::new("localized", "left"));
        // A single-node cone re-serializes to a valid, re-parseable plan.
        let json = serde_json::to_string(&replay.plan).expect("serialize");
        let back = parse_plan(&json).expect("reduced plan re-parses");
        assert_eq!(back.nodes.len(), 1);
        assert_eq!(back.nodes[0].id, "left");
        assert!(back.inputs.contains_key("src"));
    }

    #[test]
    fn spec_pins_implementation_seed_and_roi() {
        let plan = parse_plan(DIAMOND).expect("parse");
        let spec = ReplaySpec::new("localized", "out")
            .with_implementation("cpu.separable@1")
            .with_seed(42)
            .with_roi(Rect::new(700, 400, 740, 440));
        let replay = MinimalReplay::reduce(&plan, spec);
        assert_eq!(
            replay.spec.implementation.as_deref(),
            Some("cpu.separable@1")
        );
        assert_eq!(replay.spec.seed, Some(42));
        assert_eq!(replay.spec.roi, Some(Rect::new(700, 400, 740, 440)));
        // The document round-trips.
        let v = serde_json::to_value(&replay).expect("serialize");
        let back: MinimalReplay = serde_json::from_value(v).expect("round trip");
        assert_eq!(back, replay);
    }

    #[test]
    fn unknown_target_reduces_to_a_lone_root_without_panicking() {
        let plan = parse_plan(DIAMOND).expect("parse");
        let replay = MinimalReplay::reduce(&plan, ReplaySpec::new("x", "ghost"));
        // The ghost node is not in the plan, so no nodes survive.
        assert!(replay.plan.nodes.is_empty());
        assert!(replay.plan.inputs.is_empty());
    }

    #[test]
    fn path_for_lives_under_replays() {
        assert_eq!(
            MinimalReplay::path_for("localized"),
            "replays/localized.json"
        );
    }
}
