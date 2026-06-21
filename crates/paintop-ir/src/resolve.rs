//! Reference resolution and DAG validation (`plan.md` §10.1 phase 3,
//! `IR_SPEC` §3).
//!
//! Phases 1–2 (the strict parser, `plan.rs`) yield a structurally valid
//! [`Plan`] whose references are still opaque strings. This module is phase 3:
//! it resolves every `input:<id>` / `node:<id>/<port>` reference against the
//! plan's declared inputs and the operation [`OperationRegistry`], rejects
//! duplicate and malformed node ids, requires that wired ports are *declared*
//! (`M0_DECISIONS` D3 Q1/Q2), builds the normalized dependency graph, and
//! rejects cycles and dangling references with stable
//! [`reference`](crate::ErrorClass::Reference) errors.
//!
//! The output is a [`ResolvedGraph`]: a checked, edge-resolved view over the
//! plan with a deterministic topological order, on which later phases (type /
//! shape / color checking, ROI analysis) build.
//!
//! # Reference grammar (`IR_SPEC` §3.2, `M0_DECISIONS` D3 Q2)
//!
//! ```text
//! input:<input-id>
//! node:<node-id>/<output-port>
//! ```
//!
//! Every reference is fully qualified — even a single-output node is referenced
//! `node:<id>/<port>`; there is no bare-`node:<id>` shorthand in v1.
//!
//! ```
//! use paintop_ir::manifest::{
//!     DeterminismTier, InputSpec, OperationManifest, OutputSpec, ResourceKind, RoiCategory,
//!     RoiPolicy,
//! };
//! use paintop_ir::plan::parse_plan;
//! use paintop_ir::registry::OperationRegistry;
//! use paintop_ir::resolve::resolve_plan;
//!
//! let blur = OperationManifest {
//!     id: "filter.invert@1".parse().unwrap(),
//!     impl_version: 1,
//!     summary: String::new(),
//!     determinism: DeterminismTier::Exact,
//!     roi: RoiPolicy { category: RoiCategory::Pointwise, halo_px: None },
//!     inputs: vec![InputSpec {
//!         name: "image".to_owned(),
//!         kind: ResourceKind::Image,
//!         required: true,
//!         doc: String::new(),
//!     }],
//!     outputs: vec![OutputSpec {
//!         name: "image".to_owned(),
//!         kind: ResourceKind::Image,
//!         doc: String::new(),
//!     }],
//!     params: vec![],
//!     implementations: vec!["cpu.reference@1".parse().unwrap()],
//!     test: Default::default(),
//! };
//! let registry = OperationRegistry::from_manifests([blur]).unwrap();
//!
//! let plan = parse_plan(r#"{
//!     "paintop": "1.0",
//!     "inputs": {"source": {"kind": "image.file", "path": "in.png"}},
//!     "nodes": [{"id": "a", "op": "filter.invert@1", "in": {"image": "input:source"}}],
//!     "exports": {"out": {"resource": "node:a/image", "kind": "image", "path": "o.png"}}
//! }"#)
//! .unwrap();
//!
//! let graph = resolve_plan(&plan, &registry).unwrap();
//! assert_eq!(graph.topological_order(), &["a"]);
//! ```

use std::collections::BTreeMap;

use crate::error::{Error, ErrorClass, ErrorContext, Result};
use crate::manifest::OpId;
use crate::plan::Plan;
use crate::registry::OperationRegistry;

/// A node id was wired or referenced twice, was malformed, or was otherwise
/// not a legal `IR_SPEC` §3.1 identifier.
pub const E_INVALID_NODE_ID: &str = "E_INVALID_NODE_ID";

/// Two nodes in the plan declared the same `id`.
pub const E_DUPLICATE_NODE_ID: &str = "E_DUPLICATE_NODE_ID";

/// A reference string did not match the `input:<id>` / `node:<id>/<port>`
/// grammar (`IR_SPEC` §3.2, `M0_DECISIONS` D3 Q2).
pub const E_INVALID_REFERENCE: &str = "E_INVALID_REFERENCE";

/// A reference named an input or node that the plan does not declare
/// (`plan.md` §10.1: "detect ... missing references").
pub const E_DANGLING_REFERENCE: &str = "E_DANGLING_REFERENCE";

/// A `node:<id>/<port>` reference named a port the producing operation does not
/// declare as an output (`IR_SPEC` §3.3: "A node may not invent ports").
pub const E_UNKNOWN_OUTPUT_PORT: &str = "E_UNKNOWN_OUTPUT_PORT";

/// A node wired a port under `in` that its operation does not declare as an
/// input (`M0_DECISIONS` D3 Q1: "every key under `in` must be a declared input
/// port").
pub const E_UNKNOWN_INPUT_PORT: &str = "E_UNKNOWN_INPUT_PORT";

/// A required input port of an operation was left unwired.
pub const E_MISSING_INPUT_PORT: &str = "E_MISSING_INPUT_PORT";

/// The dependency graph induced by the plan's `node:` references is cyclic; a
/// plan must be a DAG (`plan.md` §10.1: "normalizes it into an immutable
/// directed acyclic graph").
pub const E_GRAPH_CYCLE: &str = "E_GRAPH_CYCLE";

/// A resolved resource reference: either an external `input:` handle or a
/// `node:<id>/<port>` output of another node (`IR_SPEC` §3.2).
///
/// Construct one with [`Reference::parse`]; the variants are the exact two legal
/// forms after the `node:id/port` / `input:id` grammar is validated. The bare
/// (`/port`-less) `node:id` form is *not* accepted (`M0_DECISIONS` D3 Q2).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Reference {
    /// `input:<input-id>` — an external resource declared under `plan.inputs`.
    Input {
        /// The declared input id.
        input: String,
    },
    /// `node:<node-id>/<output-port>` — an output port of another node.
    Node {
        /// The producing node's id.
        node: String,
        /// The producing node's output port name.
        port: String,
    },
}

impl Reference {
    /// Parse a canonical reference string (`IR_SPEC` §3.2).
    ///
    /// The grammar is strict: an `input:` reference is `input:<non-empty-id>`;
    /// a `node:` reference is `node:<non-empty-id>/<non-empty-port>` with
    /// exactly one `/`. Anything else — a missing scheme, an empty segment, a
    /// bare `node:id` without `/port`, or an extra `/` — is rejected.
    ///
    /// # Errors
    /// Returns a [`reference`](ErrorClass::Reference) error with code
    /// [`E_INVALID_REFERENCE`] on any grammar violation.
    pub fn parse(reference: &str) -> Result<Self> {
        let invalid = |msg: String| {
            Error::new(ErrorClass::Reference, E_INVALID_REFERENCE, msg)
                .with_context(ErrorContext::default().with_actual(reference.to_owned()))
        };

        if let Some(input) = reference.strip_prefix("input:") {
            if input.is_empty() {
                return Err(invalid(format!(
                    "reference {reference:?} has an empty input id; expected `input:<id>`"
                )));
            }
            return Ok(Self::Input {
                input: input.to_owned(),
            });
        }

        if let Some(rest) = reference.strip_prefix("node:") {
            let Some((node, port)) = rest.split_once('/') else {
                return Err(invalid(format!(
                    "reference {reference:?} is missing the `/<port>` suffix; every node \
                     reference is `node:<id>/<port>` (no bare-node shorthand)"
                )));
            };
            if node.is_empty() || port.is_empty() {
                return Err(invalid(format!(
                    "reference {reference:?} has an empty node id or port; expected \
                     `node:<id>/<port>`"
                )));
            }
            if port.contains('/') {
                return Err(invalid(format!(
                    "reference {reference:?} has more than one `/`; expected a single \
                     `node:<id>/<port>`"
                )));
            }
            return Ok(Self::Node {
                node: node.to_owned(),
                port: port.to_owned(),
            });
        }

        Err(invalid(format!(
            "reference {reference:?} has no recognized scheme; expected `input:<id>` or \
             `node:<id>/<port>`"
        )))
    }
}

/// One resolved node: its id, its operation id, and its resolved input edges in
/// declaration (port-name) order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedNode {
    /// The node's unique id.
    pub id: String,
    /// The node's resolved, registry-known operation id.
    pub op: OpId,
    /// The resolved input edges, keyed by port name. A [`BTreeMap`] keeps the
    /// ordering deterministic.
    pub inputs: BTreeMap<String, Reference>,
}

/// One resolved export: the export id and the resource reference it reads.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedExport {
    /// The export id (the key under `plan.exports`).
    pub id: String,
    /// The resolved resource the export writes out.
    pub resource: Reference,
}

/// A reference-resolved, cycle-checked view over a [`Plan`] (`plan.md`
/// §10.1 phase 3).
///
/// Every node id is unique and well-formed; every reference resolves to a
/// declared input or to a real output port of a real node; every required input
/// port is wired and no port is invented; and the induced graph is acyclic, so
/// [`topological_order`](ResolvedGraph::topological_order) is a valid execution
/// order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedGraph {
    nodes: BTreeMap<String, ResolvedNode>,
    exports: Vec<ResolvedExport>,
    topo: Vec<String>,
}

impl ResolvedGraph {
    /// The resolved nodes, keyed by id (deterministic iteration order).
    #[must_use]
    pub const fn nodes(&self) -> &BTreeMap<String, ResolvedNode> {
        &self.nodes
    }

    /// The resolved exports, in the plan's (sorted) export-id order.
    #[must_use]
    pub fn exports(&self) -> &[ResolvedExport] {
        &self.exports
    }

    /// A valid topological order of the node ids: every node appears after all
    /// the nodes it depends on. Ties are broken by node id so the order is
    /// deterministic across runs (`plan.md` §1).
    #[must_use]
    pub fn topological_order(&self) -> &[String] {
        &self.topo
    }

    /// Look up a resolved node by id.
    #[must_use]
    pub fn node(&self, id: &str) -> Option<&ResolvedNode> {
        self.nodes.get(id)
    }

    /// Assemble a [`ResolvedGraph`] from already-validated parts.
    ///
    /// This is the constructor graph rewriters ([`crate::simplify`]) use to emit a
    /// transformed graph without re-running [`resolve_plan`]. The caller is
    /// responsible for the invariants `resolve_plan` would otherwise establish:
    /// every reference resolves, no port is invented, and `topo` is a valid
    /// topological order of `nodes`. The simplification pass preserves these by
    /// construction (it only ever *removes* nodes/edges or rewires a consumer onto
    /// an equivalent producer), and re-derives `topo` from the surviving nodes.
    #[must_use]
    pub(crate) const fn from_parts(
        nodes: BTreeMap<String, ResolvedNode>,
        exports: Vec<ResolvedExport>,
        topo: Vec<String>,
    ) -> Self {
        Self {
            nodes,
            exports,
            topo,
        }
    }
}

/// Resolve every reference in `plan` against its declared inputs and `registry`,
/// build the dependency DAG, and reject cycles and dangling references
/// (`plan.md` §10.1 phase 3).
///
/// This runs after the strict parser ([`parse_plan`](crate::plan::parse_plan))
/// has produced a structurally valid [`Plan`]. The checks, in order:
///
/// 1. every node `id` is a valid `IR_SPEC` §3.1 identifier and is unique;
/// 2. every node's `op` resolves in `registry` (delegated to its lookup errors);
/// 3. every key under a node's `in` is a *declared input port* of its op
///    (`M0_DECISIONS` D3 Q1), and every *required* input port is wired;
/// 4. every reference parses as `input:<id>` / `node:<id>/<port>`, names a
///    declared input or a real output port of a real node, and does not invent
///    an output port (`IR_SPEC` §3.3);
/// 5. the induced graph is acyclic; the returned topological order is stable.
///
/// # Errors
/// Returns the first violation as a typed error: an invalid/duplicate node id
/// ([`E_INVALID_NODE_ID`] / [`E_DUPLICATE_NODE_ID`]), an op-lookup failure
/// (from [`OperationRegistry::get`]), a malformed reference
/// ([`E_INVALID_REFERENCE`]), a dangling input/node/export reference
/// ([`E_DANGLING_REFERENCE`]), an invented output port
/// ([`E_UNKNOWN_OUTPUT_PORT`]), an undeclared/unwired input port
/// ([`E_UNKNOWN_INPUT_PORT`] / [`E_MISSING_INPUT_PORT`]), or a cycle
/// ([`E_GRAPH_CYCLE`]). All are [`reference`](ErrorClass::Reference) errors.
pub fn resolve_plan(plan: &Plan, registry: &OperationRegistry) -> Result<ResolvedGraph> {
    // 1. Validate node ids and build the id -> node index, rejecting duplicates.
    let mut nodes: BTreeMap<String, ResolvedNode> = BTreeMap::new();
    for node in &plan.nodes {
        validate_node_id(&node.id)?;
        if nodes.contains_key(&node.id) {
            return Err(ref_error(
                E_DUPLICATE_NODE_ID,
                format!("two nodes declare the id {:?}", node.id),
            )
            .with_context(ErrorContext::default().with_node(node.id.clone())));
        }
        // Resolve the op against the registry up front so a node referencing a
        // missing/unsupported op fails before its edges are examined.
        let op: OpId = node.op.parse()?;
        registry.get(&op)?;

        nodes.insert(
            node.id.clone(),
            ResolvedNode {
                id: node.id.clone(),
                op,
                inputs: BTreeMap::new(),
            },
        );
    }

    // 2. Resolve each node's input edges: ports must be declared, references must
    //    parse and resolve, and required ports must be wired.
    for (index, node) in plan.nodes.iter().enumerate() {
        let manifest = registry.get(&nodes[&node.id].op)?;

        for (port, reference) in &node.inputs {
            // D3 Q1: every key under `in` must be a declared input port.
            let declared = manifest.inputs.iter().any(|i| &i.name == port);
            if !declared {
                return Err(ref_error(
                    E_UNKNOWN_INPUT_PORT,
                    format!(
                        "node {:?} wires input port {port:?} that {} does not declare",
                        node.id, node.op
                    ),
                )
                .with_context(
                    ErrorContext::default()
                        .with_node(node.id.clone())
                        .with_path(format!("/nodes/{index}/in/{port}")),
                ));
            }

            let resolved = Reference::parse(reference).map_err(|err| {
                // Re-attribute the parse error to this node/port location.
                err.with_context(
                    ErrorContext::default()
                        .with_node(node.id.clone())
                        .with_path(format!("/nodes/{index}/in/{port}"))
                        .with_actual(reference.clone()),
                )
            })?;
            resolve_target(&resolved, plan, registry, &nodes, &node.id, index, port)?;

            if let Some(resolved_node) = nodes.get_mut(&node.id) {
                resolved_node.inputs.insert(port.clone(), resolved);
            }
        }

        // Every required input port must be wired.
        for spec in &manifest.inputs {
            if spec.required && !node.inputs.contains_key(&spec.name) {
                return Err(ref_error(
                    E_MISSING_INPUT_PORT,
                    format!(
                        "node {:?} leaves required input port {:?} of {} unwired",
                        node.id, spec.name, node.op
                    ),
                )
                .with_context(
                    ErrorContext::default()
                        .with_node(node.id.clone())
                        .with_path(format!("/nodes/{index}/in")),
                ));
            }
        }
    }

    // 3. Resolve exports against the resolved nodes / declared inputs.
    let exports = resolve_exports(plan, registry, &nodes)?;

    // 4. Build the dependency graph and topologically sort it, rejecting cycles.
    let topo = topological_order(&nodes)?;

    Ok(ResolvedGraph {
        nodes,
        exports,
        topo,
    })
}

/// Build a [`reference`](ErrorClass::Reference) error with no extra context.
fn ref_error(code: &str, message: String) -> Error {
    Error::new(ErrorClass::Reference, code, message)
}

/// Validate a node id against the `IR_SPEC` §3.1 grammar
/// `^[A-Za-z][A-Za-z0-9_.-]{0,127}$`.
fn validate_node_id(id: &str) -> Result<()> {
    let mut chars = id.chars();
    let starts_ok = matches!(chars.next(), Some('A'..='Z' | 'a'..='z'));
    let body_ok = chars
        .clone()
        .all(|c| matches!(c, 'A'..='Z' | 'a'..='z' | '0'..='9' | '_' | '.' | '-'));
    // 1 leading char + up to 127 trailing chars => at most 128 chars total.
    let len_ok = (1..=128).contains(&id.chars().count());
    if starts_ok && body_ok && len_ok {
        return Ok(());
    }
    Err(ref_error(
        E_INVALID_NODE_ID,
        format!(
            "node id {id:?} must match ^[A-Za-z][A-Za-z0-9_.-]{{0,127}}$ (ASCII, starts with a \
             letter, at most 128 chars)"
        ),
    )
    .with_context(ErrorContext::default().with_actual(id.to_owned())))
}

/// Resolve a parsed [`Reference`] against the declared inputs and node index,
/// rejecting dangling references and invented output ports.
fn resolve_target(
    reference: &Reference,
    plan: &Plan,
    registry: &OperationRegistry,
    nodes: &BTreeMap<String, ResolvedNode>,
    consumer: &str,
    index: usize,
    port: &str,
) -> Result<()> {
    let context = || {
        ErrorContext::default()
            .with_node(consumer.to_owned())
            .with_path(format!("/nodes/{index}/in/{port}"))
    };

    match reference {
        Reference::Input { input } => {
            if !plan.inputs.contains_key(input) {
                return Err(ref_error(
                    E_DANGLING_REFERENCE,
                    format!("node {consumer:?} reads undeclared input {input:?}"),
                )
                .with_context(context().with_actual(format!("input:{input}"))));
            }
        }
        Reference::Node {
            node: target,
            port: out_port,
        } => {
            let Some(producer) = nodes.get(target) else {
                return Err(ref_error(
                    E_DANGLING_REFERENCE,
                    format!("node {consumer:?} references unknown node {target:?}"),
                )
                .with_context(context().with_actual(format!("node:{target}/{out_port}"))));
            };
            // The producing op must declare the named output port.
            let manifest = registry.get(&producer.op)?;
            if !manifest.outputs.iter().any(|o| &o.name == out_port) {
                return Err(ref_error(
                    E_UNKNOWN_OUTPUT_PORT,
                    format!(
                        "node {consumer:?} references output port {out_port:?} that node \
                         {target:?} ({}) does not declare",
                        producer.op
                    ),
                )
                .with_context(context().with_actual(format!("node:{target}/{out_port}"))));
            }
        }
    }
    Ok(())
}

/// Resolve every export's `resource` reference against the resolved nodes and
/// declared inputs.
fn resolve_exports(
    plan: &Plan,
    registry: &OperationRegistry,
    nodes: &BTreeMap<String, ResolvedNode>,
) -> Result<Vec<ResolvedExport>> {
    let mut exports = Vec::with_capacity(plan.exports.len());
    for (id, value) in &plan.exports {
        let resource_str = value.get("resource").and_then(serde_json::Value::as_str);
        let Some(resource_str) = resource_str else {
            return Err(ref_error(
                E_INVALID_REFERENCE,
                format!(
                    "export {id:?} is missing a string `resource` reference (\
                     `node:<id>/<port>` or `input:<id>`)"
                ),
            )
            .with_context(ErrorContext::default().with_path(format!("/exports/{id}/resource"))));
        };

        let resource = Reference::parse(resource_str).map_err(|err| {
            err.with_context(
                ErrorContext::default()
                    .with_path(format!("/exports/{id}/resource"))
                    .with_actual(resource_str.to_owned()),
            )
        })?;

        match &resource {
            Reference::Input { input } => {
                if !plan.inputs.contains_key(input) {
                    return Err(ref_error(
                        E_DANGLING_REFERENCE,
                        format!("export {id:?} reads undeclared input {input:?}"),
                    )
                    .with_context(
                        ErrorContext::default().with_path(format!("/exports/{id}/resource")),
                    ));
                }
            }
            Reference::Node { node, port } => {
                let Some(producer) = nodes.get(node) else {
                    return Err(ref_error(
                        E_DANGLING_REFERENCE,
                        format!("export {id:?} references unknown node {node:?}"),
                    )
                    .with_context(
                        ErrorContext::default().with_path(format!("/exports/{id}/resource")),
                    ));
                };
                let manifest = registry.get(&producer.op)?;
                if !manifest.outputs.iter().any(|o| &o.name == port) {
                    return Err(ref_error(
                        E_UNKNOWN_OUTPUT_PORT,
                        format!(
                            "export {id:?} references output port {port:?} that node {node:?} \
                             ({}) does not declare",
                            producer.op
                        ),
                    )
                    .with_context(
                        ErrorContext::default().with_path(format!("/exports/{id}/resource")),
                    ));
                }
            }
        }

        exports.push(ResolvedExport {
            id: id.clone(),
            resource,
        });
    }
    Ok(exports)
}

/// Compute a deterministic topological order of the resolved nodes, rejecting
/// cycles with [`E_GRAPH_CYCLE`].
///
/// Uses Kahn's algorithm over the node-to-node edges only (`input:` edges are
/// external sources, not graph nodes). Ties among ready nodes are broken by id
/// (the `BTreeMap` iteration order) so the order is stable across runs.
fn topological_order(nodes: &BTreeMap<String, ResolvedNode>) -> Result<Vec<String>> {
    // in_degree[n] = number of distinct upstream nodes n depends on.
    let mut in_degree: BTreeMap<&str, usize> = nodes.keys().map(|k| (k.as_str(), 0)).collect();
    // dependents[u] = nodes that consume an output of u.
    let mut dependents: BTreeMap<&str, Vec<&str>> = BTreeMap::new();

    for node in nodes.values() {
        // Distinct upstream node ids this node depends on (a node may wire the
        // same producer on two ports; that is a single edge for ordering).
        let mut upstream: Vec<&str> = node
            .inputs
            .values()
            .filter_map(|r| match r {
                Reference::Node { node, .. } => Some(node.as_str()),
                Reference::Input { .. } => None,
            })
            .collect();
        upstream.sort_unstable();
        upstream.dedup();

        if let Some(deg) = in_degree.get_mut(node.id.as_str()) {
            *deg = upstream.len();
        }
        for up in upstream {
            dependents.entry(up).or_default().push(node.id.as_str());
        }
    }

    // Seed the ready set with all zero-in-degree nodes, in id order.
    let mut ready: Vec<&str> = in_degree
        .iter()
        .filter(|&(_, &deg)| deg == 0)
        .map(|(&id, _)| id)
        .collect();
    // `in_degree` is a BTreeMap so `ready` is already id-sorted; keep it sorted
    // as we push so ties stay deterministic.
    let mut order: Vec<String> = Vec::with_capacity(nodes.len());

    while let Some(current) = pop_min(&mut ready) {
        order.push(current.to_owned());
        if let Some(children) = dependents.get(current) {
            for &child in children {
                if let Some(deg) = in_degree.get_mut(child) {
                    *deg -= 1;
                    if *deg == 0 {
                        ready.push(child);
                    }
                }
            }
        }
    }

    if order.len() != nodes.len() {
        // The nodes that never reached in-degree zero form (and feed) the cycle.
        let mut stuck: Vec<&str> = in_degree
            .iter()
            .filter(|&(_, &deg)| deg > 0)
            .map(|(&id, _)| id)
            .collect();
        stuck.sort_unstable();
        return Err(ref_error(
            E_GRAPH_CYCLE,
            format!(
                "plan node graph contains a cycle involving: {}",
                stuck.join(", ")
            ),
        ));
    }

    Ok(order)
}

/// Remove and return the lexicographically smallest id from `ready`, keeping
/// tie-breaking deterministic.
fn pop_min<'a>(ready: &mut Vec<&'a str>) -> Option<&'a str> {
    let min_index = ready
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| a.cmp(b))
        .map(|(i, _)| i)?;
    Some(ready.swap_remove(min_index))
}

#[cfg(test)]
mod tests {
    use super::{
        E_DANGLING_REFERENCE, E_DUPLICATE_NODE_ID, E_GRAPH_CYCLE, E_INVALID_NODE_ID,
        E_INVALID_REFERENCE, E_MISSING_INPUT_PORT, E_UNKNOWN_INPUT_PORT, E_UNKNOWN_OUTPUT_PORT,
        Reference, resolve_plan,
    };
    use crate::error::ErrorClass;
    use crate::manifest::{
        DeterminismTier, InputSpec, OperationManifest, OutputSpec, ResourceKind, RoiCategory,
        RoiPolicy,
    };
    use crate::plan::parse_plan;
    use crate::registry::OperationRegistry;

    /// A pointwise op with the given input/output ports (all inputs required).
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
            test: crate::manifest::TestMetadata::default(),
        }
    }

    /// A source op with no inputs and one `image` output.
    fn registry() -> OperationRegistry {
        OperationRegistry::from_manifests([
            op("source.create@1", &[], &["image"]),
            op("filter.invert@1", &["image"], &["image"]),
            op("filter.blend@1", &["a", "b"], &["image"]),
            op("frequency.split@1", &["image"], &["lowpass", "residuals"]),
        ])
        .unwrap()
    }

    // ---- Reference parsing -------------------------------------------------

    #[test]
    fn parses_input_and_node_references() {
        assert_eq!(
            Reference::parse("input:source").unwrap(),
            Reference::Input {
                input: "source".to_owned()
            }
        );
        assert_eq!(
            Reference::parse("node:blur.low/lowpass").unwrap(),
            Reference::Node {
                node: "blur.low".to_owned(),
                port: "lowpass".to_owned()
            }
        );
    }

    #[test]
    fn rejects_malformed_references() {
        for bad in [
            "source",        // no scheme
            "node:blur",     // bare node, no /port (D3 Q2)
            "node:blur/",    // empty port
            "node:/lowpass", // empty node id
            "node:a/b/c",    // extra slash
            "input:",        // empty input id
            "output:final",  // unknown scheme
        ] {
            let err = Reference::parse(bad).unwrap_err();
            assert_eq!(err.class, ErrorClass::Reference, "{bad:?}");
            assert_eq!(err.code, E_INVALID_REFERENCE, "{bad:?}");
        }
    }

    // ---- Happy path: multi-node DAG ---------------------------------------

    #[test]
    fn resolves_multi_node_dag_with_stable_topological_order() {
        // source -> a -> blend <- b ; nodes deliberately out of order in JSON.
        let plan = parse_plan(
            r#"{
                "paintop": "1.0",
                "inputs": {"src": {"kind": "image.file", "path": "in.png"}},
                "nodes": [
                    {"id": "out", "op": "filter.blend@1",
                     "in": {"a": "node:a/image", "b": "node:src/image"}},
                    {"id": "a", "op": "filter.invert@1", "in": {"image": "node:src/image"}},
                    {"id": "src", "op": "source.create@1"}
                ],
                "exports": {"final": {"resource": "node:out/image", "kind": "image", "path": "o.png"}}
            }"#,
        )
        .unwrap();

        let graph = resolve_plan(&plan, &registry()).unwrap();
        // Topological order: src before a before out; ties broken by id.
        assert_eq!(graph.topological_order(), &["src", "a", "out"]);
        assert_eq!(graph.nodes().len(), 3);
        assert_eq!(graph.exports().len(), 1);
        assert_eq!(
            graph.node("out").unwrap().inputs["a"],
            Reference::Node {
                node: "a".to_owned(),
                port: "image".to_owned()
            }
        );
    }

    #[test]
    fn topological_order_is_deterministic_across_runs() {
        let plan = parse_plan(
            r#"{
                "paintop": "1.0",
                "inputs": {},
                "nodes": [
                    {"id": "z", "op": "source.create@1"},
                    {"id": "m", "op": "source.create@1"},
                    {"id": "a", "op": "source.create@1"}
                ],
                "exports": {}
            }"#,
        )
        .unwrap();
        let reg = registry();
        let first = resolve_plan(&plan, &reg).unwrap();
        let second = resolve_plan(&plan, &reg).unwrap();
        assert_eq!(first.topological_order(), &["a", "m", "z"]);
        assert_eq!(first.topological_order(), second.topological_order());
    }

    // ---- Cycle detection ---------------------------------------------------

    #[test]
    fn rejects_a_two_node_cycle() {
        let plan = parse_plan(
            r#"{
                "paintop": "1.0",
                "inputs": {},
                "nodes": [
                    {"id": "a", "op": "filter.invert@1", "in": {"image": "node:b/image"}},
                    {"id": "b", "op": "filter.invert@1", "in": {"image": "node:a/image"}}
                ],
                "exports": {}
            }"#,
        )
        .unwrap();
        let err = resolve_plan(&plan, &registry()).unwrap_err();
        assert_eq!(err.class, ErrorClass::Reference);
        assert_eq!(err.code, E_GRAPH_CYCLE);
        assert!(err.message.contains('a') && err.message.contains('b'));
    }

    #[test]
    fn rejects_a_self_loop() {
        let plan = parse_plan(
            r#"{
                "paintop": "1.0",
                "inputs": {},
                "nodes": [
                    {"id": "a", "op": "filter.invert@1", "in": {"image": "node:a/image"}}
                ],
                "exports": {}
            }"#,
        )
        .unwrap();
        assert_eq!(
            resolve_plan(&plan, &registry()).unwrap_err().code,
            E_GRAPH_CYCLE
        );
    }

    // ---- Dangling references ----------------------------------------------

    #[test]
    fn rejects_dangling_input_reference() {
        let plan = parse_plan(
            r#"{
                "paintop": "1.0",
                "inputs": {},
                "nodes": [
                    {"id": "a", "op": "filter.invert@1", "in": {"image": "input:missing"}}
                ],
                "exports": {}
            }"#,
        )
        .unwrap();
        let err = resolve_plan(&plan, &registry()).unwrap_err();
        assert_eq!(err.code, E_DANGLING_REFERENCE);
        assert_eq!(err.context.node.as_deref(), Some("a"));
    }

    #[test]
    fn rejects_dangling_node_reference() {
        let plan = parse_plan(
            r#"{
                "paintop": "1.0",
                "inputs": {},
                "nodes": [
                    {"id": "a", "op": "filter.invert@1", "in": {"image": "node:ghost/image"}}
                ],
                "exports": {}
            }"#,
        )
        .unwrap();
        assert_eq!(
            resolve_plan(&plan, &registry()).unwrap_err().code,
            E_DANGLING_REFERENCE
        );
    }

    #[test]
    fn rejects_dangling_export_reference() {
        let plan = parse_plan(
            r#"{
                "paintop": "1.0",
                "inputs": {},
                "nodes": [{"id": "src", "op": "source.create@1"}],
                "exports": {"final": {"resource": "node:ghost/image", "kind": "image", "path": "o.png"}}
            }"#,
        )
        .unwrap();
        assert_eq!(
            resolve_plan(&plan, &registry()).unwrap_err().code,
            E_DANGLING_REFERENCE
        );
    }

    // ---- Port checks -------------------------------------------------------

    #[test]
    fn rejects_unknown_output_port() {
        // `frequency.split@1` has lowpass/residuals, not `image`.
        let plan = parse_plan(
            r#"{
                "paintop": "1.0",
                "inputs": {"src": {"kind": "image.file", "path": "in.png"}},
                "nodes": [
                    {"id": "s", "op": "frequency.split@1", "in": {"image": "input:src"}},
                    {"id": "a", "op": "filter.invert@1", "in": {"image": "node:s/image"}}
                ],
                "exports": {}
            }"#,
        )
        .unwrap();
        let err = resolve_plan(&plan, &registry()).unwrap_err();
        assert_eq!(err.code, E_UNKNOWN_OUTPUT_PORT);
        assert_eq!(err.context.node.as_deref(), Some("a"));
    }

    #[test]
    fn rejects_unknown_input_port() {
        // `filter.invert@1` declares only the `image` input port.
        let plan = parse_plan(
            r#"{
                "paintop": "1.0",
                "inputs": {"src": {"kind": "image.file", "path": "in.png"}},
                "nodes": [
                    {"id": "a", "op": "filter.invert@1",
                     "in": {"image": "input:src", "mask": "input:src"}}
                ],
                "exports": {}
            }"#,
        )
        .unwrap();
        let err = resolve_plan(&plan, &registry()).unwrap_err();
        assert_eq!(err.code, E_UNKNOWN_INPUT_PORT);
        assert!(err.message.contains("mask"));
    }

    #[test]
    fn rejects_missing_required_input_port() {
        // `filter.blend@1` requires both `a` and `b`; only `a` is wired.
        let plan = parse_plan(
            r#"{
                "paintop": "1.0",
                "inputs": {"src": {"kind": "image.file", "path": "in.png"}},
                "nodes": [
                    {"id": "x", "op": "filter.blend@1", "in": {"a": "input:src"}}
                ],
                "exports": {}
            }"#,
        )
        .unwrap();
        let err = resolve_plan(&plan, &registry()).unwrap_err();
        assert_eq!(err.code, E_MISSING_INPUT_PORT);
        assert!(err.message.contains('b'));
    }

    // ---- Node id checks ----------------------------------------------------

    #[test]
    fn rejects_duplicate_node_ids() {
        let plan = parse_plan(
            r#"{
                "paintop": "1.0",
                "inputs": {},
                "nodes": [
                    {"id": "a", "op": "source.create@1"},
                    {"id": "a", "op": "source.create@1"}
                ],
                "exports": {}
            }"#,
        )
        .unwrap();
        assert_eq!(
            resolve_plan(&plan, &registry()).unwrap_err().code,
            E_DUPLICATE_NODE_ID
        );
    }

    #[test]
    fn rejects_malformed_node_id() {
        let plan = parse_plan(
            r#"{
                "paintop": "1.0",
                "inputs": {},
                "nodes": [{"id": "mask/jacket", "op": "source.create@1"}],
                "exports": {}
            }"#,
        )
        .unwrap();
        assert_eq!(
            resolve_plan(&plan, &registry()).unwrap_err().code,
            E_INVALID_NODE_ID
        );
    }

    #[test]
    fn accepts_dotted_and_numbered_node_ids() {
        let plan = parse_plan(
            r#"{
                "paintop": "1.0",
                "inputs": {},
                "nodes": [{"id": "candidate.inpaint.0", "op": "source.create@1"}],
                "exports": {}
            }"#,
        )
        .unwrap();
        let graph = resolve_plan(&plan, &registry()).unwrap();
        assert_eq!(graph.topological_order(), &["candidate.inpaint.0"]);
    }

    #[test]
    fn unknown_op_surfaces_registry_error() {
        let plan = parse_plan(
            r#"{
                "paintop": "1.0",
                "inputs": {},
                "nodes": [{"id": "a", "op": "no.such_op@1"}],
                "exports": {}
            }"#,
        )
        .unwrap();
        // Delegated to the registry: a reference-class lookup failure.
        let err = resolve_plan(&plan, &registry()).unwrap_err();
        assert_eq!(err.class, ErrorClass::Reference);
    }

    #[test]
    fn a_node_may_reference_one_producer_on_two_ports() {
        // The same producer wired to both blend ports is a single graph edge.
        let plan = parse_plan(
            r#"{
                "paintop": "1.0",
                "inputs": {},
                "nodes": [
                    {"id": "src", "op": "source.create@1"},
                    {"id": "x", "op": "filter.blend@1",
                     "in": {"a": "node:src/image", "b": "node:src/image"}}
                ],
                "exports": {}
            }"#,
        )
        .unwrap();
        let graph = resolve_plan(&plan, &registry()).unwrap();
        assert_eq!(graph.topological_order(), &["src", "x"]);
    }
}
