//! Backward demand and dead-node elimination (`plan.md` §10.1 phase 7).
//!
//! Execution is *demand-driven*: a node runs only if something downstream needs
//! its output. The demand roots are the plan's **exports**, its **assertions**,
//! and any **requested debug resources** (`plan.md` §10.1 phase 7: "Start from
//! exports, assertions, and requested debug resources"). Every node reachable
//! backward from a root — through the resolved `node:<id>/<port>` edges — is
//! *live*; every other node is *dead* and is eliminated, never to be executed.
//!
//! This bone does whole-image execution, so demand here is at node granularity:
//! a node is either demanded or not. Region-level backward propagation (the halo
//! arithmetic of `required_inputs`) is an M2 concern and is not computed here.
//!
//! # Why assertions and debug requests are scanned, not typed
//!
//! The plan's `assertions` and `evidence` blocks carry op-/feature-specific
//! schemas owned by later bones, so they arrive as opaque canonical JSON. To find
//! the node outputs they demand *without* coupling to those schemas, this pass
//! recursively scans their JSON for strings that parse as a
//! [`Reference`] (`node:<id>/<port>` / `input:<id>`). A
//! `node:` reference whose node exists in the graph is a demand root; anything
//! else (a plain string, an `input:` reference, a dangling `node:` reference) is
//! ignored — dangling references were already rejected for *exports* during
//! resolution, and assertion/debug schemas validate their own references in their
//! own bones.

use std::collections::BTreeSet;

use paintop_ir::{Plan, Reference, ResolvedGraph};

/// The result of the backward-demand pass over a resolved graph.
///
/// Carries the demanded (live) nodes in the graph's deterministic topological
/// order — the exact order and set the executor runs — alongside the eliminated
/// (dead) nodes, retained so a caller (and the trace) can prove a node was *not*
/// run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DemandTrace {
    demanded: Vec<String>,
    eliminated: Vec<String>,
}

impl DemandTrace {
    /// The demanded (live) node ids, in topological execution order.
    #[must_use]
    pub fn demanded(&self) -> &[String] {
        &self.demanded
    }

    /// The eliminated (dead) node ids, in topological order. A node here is
    /// reachable from no export, assertion, or requested debug resource and is
    /// never executed.
    #[must_use]
    pub fn eliminated(&self) -> &[String] {
        &self.eliminated
    }

    /// Whether `node` is demanded (and will therefore be executed).
    #[must_use]
    pub fn is_demanded(&self, node: &str) -> bool {
        self.demanded.iter().any(|n| n == node)
    }

    /// Whether `node` was eliminated as dead.
    #[must_use]
    pub fn is_eliminated(&self, node: &str) -> bool {
        self.eliminated.iter().any(|n| n == node)
    }
}

/// Compute the demanded node set of `graph`, eliminating dead nodes
/// (`plan.md` §10.1 phase 7).
///
/// Demand roots are the resolved exports plus every `node:<id>/<port>` reference
/// found in the plan's `assertions` and `evidence` (requested debug) blocks.
/// Starting from those roots, the pass walks the resolved input edges backward to
/// mark every transitively-needed node live; the rest are eliminated.
///
/// The returned [`DemandTrace`] lists the live nodes in the graph's topological
/// order, so the executor can iterate it directly.
#[must_use]
pub fn compute_demand(plan: &Plan, graph: &ResolvedGraph) -> DemandTrace {
    // 1. Seed the live set with the demand roots: export targets and any
    //    node-references inside assertions / requested debug resources.
    let mut roots: Vec<&str> = Vec::new();
    for export in graph.exports() {
        if let Reference::Node { node, .. } = &export.resource {
            roots.push(node.as_str());
        }
    }
    let mut scanned: Vec<String> = Vec::new();
    for assertion in &plan.assertions {
        collect_node_references(assertion, graph, &mut scanned);
    }
    collect_node_references(
        &serde_json::Value::Object(plan.evidence.clone()),
        graph,
        &mut scanned,
    );
    for node in &scanned {
        roots.push(node.as_str());
    }

    // 2. Mark every node reachable backward from a root as live.
    let mut live: BTreeSet<String> = BTreeSet::new();
    let mut stack: Vec<String> = roots.into_iter().map(str::to_owned).collect();
    while let Some(node_id) = stack.pop() {
        if !live.insert(node_id.clone()) {
            continue;
        }
        let Some(node) = graph.node(&node_id) else {
            continue;
        };
        for reference in node.inputs.values() {
            if let Reference::Node { node: upstream, .. } = reference
                && !live.contains(upstream)
            {
                stack.push(upstream.clone());
            }
        }
    }

    // 3. Split the topological order into demanded and eliminated, preserving the
    //    deterministic order for both.
    let mut demanded = Vec::new();
    let mut eliminated = Vec::new();
    for node_id in graph.topological_order() {
        if live.contains(node_id) {
            demanded.push(node_id.clone());
        } else {
            eliminated.push(node_id.clone());
        }
    }

    DemandTrace {
        demanded,
        eliminated,
    }
}

/// Recursively collect every `node:<id>/<port>` reference in `value` whose node
/// exists in `graph`, appending the node ids to `out`.
fn collect_node_references(
    value: &serde_json::Value,
    graph: &ResolvedGraph,
    out: &mut Vec<String>,
) {
    match value {
        serde_json::Value::String(s) => {
            if let Ok(Reference::Node { node, .. }) = Reference::parse(s)
                && graph.node(&node).is_some()
            {
                out.push(node);
            }
        }
        serde_json::Value::Array(items) => {
            for item in items {
                collect_node_references(item, graph, out);
            }
        }
        serde_json::Value::Object(map) => {
            for item in map.values() {
                collect_node_references(item, graph, out);
            }
        }
        serde_json::Value::Null | serde_json::Value::Bool(_) | serde_json::Value::Number(_) => {}
    }
}

#[cfg(test)]
mod tests {
    use super::compute_demand;
    use paintop_ir::{
        DeterminismTier, InputSpec, OperationManifest, OperationRegistry, OutputSpec, Plan,
        ResourceKind, RoiCategory, RoiPolicy, TestMetadata, parse_plan, resolve_plan,
    };

    fn op(id: &str, inputs: &[&str], outputs: &[&str]) -> OperationManifest {
        OperationManifest {
            id: id.parse().unwrap(),
            impl_version: 1,
            summary: String::new(),
            determinism: DeterminismTier::Exact,
            roi: RoiPolicy {
                category: RoiCategory::Pointwise,
                halo_px: None,
            },
            inputs: inputs
                .iter()
                .map(|name| InputSpec {
                    name: (*name).to_owned(),
                    kind: ResourceKind::Image,
                    required: true,
                    doc: String::new(),
                })
                .collect(),
            outputs: outputs
                .iter()
                .map(|name| OutputSpec {
                    name: (*name).to_owned(),
                    kind: ResourceKind::Image,
                    doc: String::new(),
                })
                .collect(),
            params: vec![],
            implementations: vec!["cpu.reference@1".parse().unwrap()],
            test: TestMetadata::default(),
        }
    }

    fn registry() -> OperationRegistry {
        OperationRegistry::from_manifests([
            op("source.create@1", &[], &["image"]),
            op("filter.invert@1", &["image"], &["image"]),
        ])
        .unwrap()
    }

    fn resolve(plan: &Plan) -> paintop_ir::ResolvedGraph {
        resolve_plan(plan, &registry()).unwrap()
    }

    #[test]
    fn eliminates_a_node_no_export_demands() {
        // `used` feeds the export; `dead` feeds nothing.
        let plan = parse_plan(
            r#"{
                "paintop": "1.0",
                "inputs": {},
                "nodes": [
                    {"id": "src", "op": "source.create@1"},
                    {"id": "used", "op": "filter.invert@1", "in": {"image": "node:src/image"}},
                    {"id": "dead", "op": "filter.invert@1", "in": {"image": "node:src/image"}}
                ],
                "exports": {"out": {"resource": "node:used/image", "kind": "image", "path": "o.png"}}
            }"#,
        )
        .unwrap();
        let graph = resolve(&plan);
        let demand = compute_demand(&plan, &graph);
        assert_eq!(demand.demanded(), &["src", "used"]);
        assert_eq!(demand.eliminated(), &["dead"]);
        assert!(demand.is_demanded("src"));
        assert!(demand.is_eliminated("dead"));
    }

    #[test]
    fn no_exports_eliminates_everything() {
        let plan = parse_plan(
            r#"{
                "paintop": "1.0",
                "inputs": {},
                "nodes": [{"id": "src", "op": "source.create@1"}],
                "exports": {}
            }"#,
        )
        .unwrap();
        let graph = resolve(&plan);
        let demand = compute_demand(&plan, &graph);
        assert!(demand.demanded().is_empty());
        assert_eq!(demand.eliminated(), &["src"]);
    }

    #[test]
    fn an_assertion_reference_demands_its_node() {
        // No export, but an assertion references `kept` -> it stays live.
        let plan = parse_plan(
            r#"{
                "paintop": "1.0",
                "inputs": {},
                "nodes": [
                    {"id": "src", "op": "source.create@1"},
                    {"id": "kept", "op": "filter.invert@1", "in": {"image": "node:src/image"}},
                    {"id": "dead", "op": "filter.invert@1", "in": {"image": "node:src/image"}}
                ],
                "assertions": [{"kind": "assert.x@1", "subject": "node:kept/image"}],
                "exports": {}
            }"#,
        )
        .unwrap();
        let graph = resolve(&plan);
        let demand = compute_demand(&plan, &graph);
        assert_eq!(demand.demanded(), &["src", "kept"]);
        assert!(demand.is_eliminated("dead"));
    }

    #[test]
    fn a_requested_debug_resource_demands_its_node() {
        let plan = parse_plan(
            r#"{
                "paintop": "1.0",
                "inputs": {},
                "nodes": [
                    {"id": "src", "op": "source.create@1"},
                    {"id": "peek", "op": "filter.invert@1", "in": {"image": "node:src/image"}}
                ],
                "evidence": {"debug_resources": ["node:peek/image"]},
                "exports": {}
            }"#,
        )
        .unwrap();
        let graph = resolve(&plan);
        let demand = compute_demand(&plan, &graph);
        assert!(demand.is_demanded("peek"));
        assert!(demand.is_demanded("src"));
    }

    #[test]
    fn demand_follows_a_transitive_chain() {
        // a -> b -> c, export demands c; all three live, in topo order.
        let plan = parse_plan(
            r#"{
                "paintop": "1.0",
                "inputs": {},
                "nodes": [
                    {"id": "a", "op": "source.create@1"},
                    {"id": "b", "op": "filter.invert@1", "in": {"image": "node:a/image"}},
                    {"id": "c", "op": "filter.invert@1", "in": {"image": "node:b/image"}}
                ],
                "exports": {"out": {"resource": "node:c/image", "kind": "image", "path": "o.png"}}
            }"#,
        )
        .unwrap();
        let graph = resolve(&plan);
        let demand = compute_demand(&plan, &graph);
        assert_eq!(demand.demanded(), &["a", "b", "c"]);
        assert!(demand.eliminated().is_empty());
    }
}
