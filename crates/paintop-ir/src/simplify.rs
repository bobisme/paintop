//! Safe, handwritten canonical graph simplification (`plan.md` §10.1 phase 8;
//! `ALIEN_OPS` §13.2, §13.3).
//!
//! This is the **conservative** rewrite pass: the small set of provably
//! semantic-preserving simplifications that need no e-graph and no cost model.
//! It is deliberately *not* equality saturation (that is M8) and *not* GPU
//! pointwise fusion (that is M3) — those choose between alternatives by cost;
//! here every rewrite is unconditionally an improvement that preserves the
//! semantic value of every graph **output**.
//!
//! # The rewrites (`ALIEN_OPS` §13.2)
//!
//! 1. **Identity-node elimination.** A node whose operation is a declared no-op
//!    (`crate::manifest::OperationManifest`-level `identity` ⇒ here, an op whose
//!    canonical id is in [`IDENTITY_OPS`], e.g. a `mask.intersect` with `full`)
//!    is spliced out: every consumer of its single output is rewired to the
//!    node's single input, and the node is deleted.
//! 2. **Conversion cancellation.** An adjacent inverse pair on a single wire —
//!    `unpremultiply ∘ premultiply`, `premultiply ∘ unpremultiply`, or
//!    `convert(A→B) ∘ convert(B→A)` — collapses to the identity wire: the
//!    consumer is rewired to the producer's input. The premultiply pair carries
//!    the §13.2 side condition `alpha > ε`; we only fire it when the producer's
//!    declared alpha makes the round-trip exact (see `is_inverse_pair`).
//! 3. **Common-subexpression elimination.** Two nodes with the *same operation,
//!    same parameters, and same resolved inputs* compute the same value; the
//!    later one is replaced by the earlier and its consumers rewired, dropping a
//!    duplicate subgraph. CSE is applied to a fixed point so a shared duplicate
//!    cascade collapses fully.
//! 4. **Constant folding (structural).** A node both of whose semantics and
//!    inputs are compile-time constant *and* whose operation is a declared
//!    foldable constant source is left to a later numeric bone; the structural
//!    hook here folds only the trivial case of a duplicate constant source
//!    (which CSE already covers), so constant folding is realized as a *barrier
//!    discipline* plus CSE today and the entry point is reserved.
//!
//! # Barriers (`ALIEN_OPS` §13.3)
//!
//! A rewrite never crosses, and never removes, a node that is:
//!
//! - **requested for debug materialization** — referenced by the plan's
//!   `evidence` block, or produced by a `debug.materialize` op;
//! - **observed by an assertion** — referenced by the plan's `assertions`;
//! - **stochastic** — its op declares
//!   [`DeterminismTier::Stochastic`](crate::DeterminismTier), so reordering or
//!   deduplicating it would change candidate identity;
//! - **a model call or side-effect sink** — encode/io ops that write or call out.
//!
//! A barrier node is *pinned*: it is never eliminated, never CSE-merged, and is
//! never spliced through. Conversion cancellation additionally refuses to fire
//! across a **nonlinear encoding** unless the algebra proves the round-trip exact
//! (`ALIEN_OPS` §13.3: "clamping/nonlinear encodings unless algebra proves
//! safety").
//!
//! # The guarantee
//!
//! Every rewrite preserves the semantic value of every **export / assertion /
//! debug** root. The differential suite (in `paintop-core`) pins this: the
//! simplified graph executed whole-image produces byte-identical output to the
//! unsimplified graph for exact ops. Simplification is **disable-able**
//! ([`SimplifyOptions::DISABLED`]); the disabled pass is the identity transform.

use std::collections::{BTreeMap, BTreeSet};

use crate::manifest::DeterminismTier;
use crate::plan::Plan;
use crate::registry::OperationRegistry;
use crate::resolve::{Reference, ResolvedExport, ResolvedGraph, ResolvedNode};

/// Operation ids whose node *may* be a semantic identity on its single input
/// wire (`ALIEN_OPS` §13.2: "identity elimination").
///
/// Each id here is an op that, with the right parameters, returns its input
/// unchanged. Membership alone is not enough: a parameter-aware check must
/// confirm the params select the no-op before a node is spliced out. The list is
/// intentionally tiny and conservative; richer per-parameter identities (e.g.
/// `image.crop` to the full extent, `color.adjust` with all-zero deltas) are
/// added as their checks are written.
pub const IDENTITY_OPS: &[&str] = &["image.flip@1"];

/// Operation ids that are side-effect sinks or model/IO calls.
///
/// Such a node must never be rewritten away or merged (`ALIEN_OPS` §13.3: "model
/// calls; side-effect sinks"): even if duplicated, deduplicating it could drop a
/// write or a model invocation.
pub const SINK_OPS: &[&str] = &["io.encode_image@1", "io.decode_image@1"];

/// The op id that forces a debug materialization point (`ALIEN_OPS` §13.3:
/// "requested debug materialization").
pub const DEBUG_MATERIALIZE_OP: &str = "debug.materialize@1";

/// Options controlling the simplification pass.
///
/// The pass is **disable-able** (`plan.md` §10.1 phase 8 is an optimization, not
/// a correctness requirement): a disabled pass returns the input graph unchanged
/// with an empty [`SimplificationReport`], so a caller can A/B the optimized and
/// unoptimized graphs (the differential suite relies on exactly this).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SimplifyOptions {
    enabled: bool,
}

impl Default for SimplifyOptions {
    fn default() -> Self {
        Self::ENABLED
    }
}

impl SimplifyOptions {
    /// The default: every safe rewrite is applied.
    pub const ENABLED: Self = Self { enabled: true };
    /// The pass is a no-op identity transform.
    pub const DISABLED: Self = Self { enabled: false };

    /// Whether the pass will apply any rewrites.
    #[must_use]
    pub const fn is_enabled(self) -> bool {
        self.enabled
    }
}

/// One applied rewrite, named for the trace / evidence so a caller can see
/// exactly which transforms fired and why a node count dropped.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Rewrite {
    /// An identity node `node` was spliced out; its consumers were rewired to its
    /// input.
    IdentityElimination {
        /// The eliminated node id.
        node: String,
    },
    /// An adjacent inverse pair `(producer, consumer)` cancelled; the consumer's
    /// consumers were rewired to the producer's input.
    ConversionCancellation {
        /// The downstream (consumer) node that was removed.
        consumer: String,
        /// The upstream (producer) node that was removed.
        producer: String,
    },
    /// The duplicate node `removed` was replaced by the equivalent earlier node
    /// `kept` (common-subexpression elimination).
    CommonSubexpression {
        /// The surviving canonical node.
        kept: String,
        /// The duplicate that was removed.
        removed: String,
    },
}

/// The record of what a simplification pass did.
///
/// Carries the rewrites applied, in application order, and the set of nodes that
/// were **pinned** as barriers (never rewritten). An empty report means the graph
/// was already canonical (or the pass was disabled).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SimplificationReport {
    rewrites: Vec<Rewrite>,
    pinned: BTreeSet<String>,
}

impl SimplificationReport {
    /// The rewrites applied, in order.
    #[must_use]
    pub fn rewrites(&self) -> &[Rewrite] {
        &self.rewrites
    }

    /// The barrier-pinned node ids (never rewritten).
    pub fn pinned(&self) -> impl Iterator<Item = &str> {
        self.pinned.iter().map(String::as_str)
    }

    /// Whether `node` was pinned as a barrier.
    #[must_use]
    pub fn is_pinned(&self, node: &str) -> bool {
        self.pinned.contains(node)
    }

    /// The number of rewrites applied.
    #[must_use]
    pub const fn rewrite_count(&self) -> usize {
        self.rewrites.len()
    }
}

/// Simplify a resolved graph, returning the canonicalized graph and the record of
/// what changed (`plan.md` §10.1 phase 8; `ALIEN_OPS` §13.2, §13.3).
///
/// `plan` supplies the params (for CSE equality and parameter-aware identity),
/// the `assertions` and `evidence` blocks (for barrier discovery), and the op
/// `hints`. `registry` supplies each op's determinism tier (for the stochastic
/// barrier). The returned [`ResolvedGraph`] is *equivalent* to `graph`: every
/// export/assertion/debug root computes the same value.
///
/// When [`options`](SimplifyOptions::DISABLED) is disabled the input graph is
/// returned unchanged with an empty report.
#[must_use]
pub fn simplify(
    plan: &Plan,
    graph: &ResolvedGraph,
    registry: &OperationRegistry,
    options: SimplifyOptions,
) -> (ResolvedGraph, SimplificationReport) {
    let mut report = SimplificationReport::default();
    if !options.is_enabled() {
        return (graph.clone(), report);
    }

    let pinned = compute_barriers(plan, graph, registry);
    report.pinned.clone_from(&pinned);

    // Work on a mutable working set keyed by id; rebuild a ResolvedGraph at the
    // end from the survivors with a fresh topological order.
    let mut nodes: BTreeMap<String, ResolvedNode> = graph.nodes().clone();
    let mut exports: Vec<ResolvedExport> = graph.exports().to_vec();

    // Apply rewrites to a fixed point: one rewrite can expose another (an
    // identity splice can make two formerly-distinct nodes identical, enabling
    // CSE; a CSE merge can expose a fresh inverse pair). We loop until a full
    // sweep makes no change, so the result is order-independent and canonical.
    loop {
        let mut changed = false;

        // 1. Conversion cancellation (adjacent inverse pairs on a single wire).
        if let Some(pair) = find_conversion_cancellation(&nodes, &exports, &pinned, registry) {
            apply_conversion_cancellation(&mut nodes, &mut exports, &pair);
            report.rewrites.push(Rewrite::ConversionCancellation {
                consumer: pair.consumer,
                producer: pair.producer,
            });
            changed = true;
        }
        // 2. Identity-node elimination.
        else if let Some(node) = find_identity_node(plan, &nodes, &pinned) {
            apply_identity_elimination(&mut nodes, &mut exports, &node);
            report.rewrites.push(Rewrite::IdentityElimination { node });
            changed = true;
        }
        // 3. Common-subexpression elimination.
        else if let Some((kept, removed)) = find_common_subexpression(plan, &nodes, &pinned) {
            apply_cse(&mut nodes, &mut exports, &kept, &removed);
            report
                .rewrites
                .push(Rewrite::CommonSubexpression { kept, removed });
            changed = true;
        }

        if !changed {
            break;
        }
    }

    let topo = topological_order(&nodes);
    (ResolvedGraph::from_parts(nodes, exports, topo), report)
}

/// Compute the set of barrier-pinned node ids (`ALIEN_OPS` §13.3).
///
/// A node is pinned when it is observed by an assertion, requested for debug
/// materialization (referenced by `evidence` or produced by a `debug.materialize`
/// op), is stochastic, or is a side-effect/model sink.
fn compute_barriers(
    plan: &Plan,
    graph: &ResolvedGraph,
    registry: &OperationRegistry,
) -> BTreeSet<String> {
    let mut pinned = BTreeSet::new();

    // Op-level barriers: stochastic tier, debug.materialize, sink/IO ops.
    for (id, node) in graph.nodes() {
        let op_str = node.op.to_string();
        let is_sink = SINK_OPS.contains(&op_str.as_str());
        let is_debug = op_str == DEBUG_MATERIALIZE_OP;
        let is_stochastic = registry
            .get(&node.op)
            .is_ok_and(|m| m.determinism == DeterminismTier::Stochastic);
        if is_sink || is_debug || is_stochastic {
            pinned.insert(id.clone());
        }
    }

    // Reference-level barriers: any node observed by an assertion or requested as
    // a debug resource is pinned. Assertion / evidence schemas are op-specific and
    // arrive as opaque JSON, so we scan recursively for `node:<id>/<port>`
    // references — exactly as the demand pass does.
    for assertion in &plan.assertions {
        collect_node_refs(assertion, &mut pinned);
    }
    for value in plan.evidence.values() {
        collect_node_refs(value, &mut pinned);
    }

    // Drop any reference that does not name a real node (a dangling `node:` in an
    // assertion/evidence block; those are validated in their own bones).
    pinned.retain(|id| graph.node(id).is_some());
    pinned
}

/// Recursively collect every `node:<id>/<port>` reference's *node id* from an
/// arbitrary JSON value into `out` (`ALIEN_OPS` §13.3 barrier discovery).
fn collect_node_refs(value: &serde_json::Value, out: &mut BTreeSet<String>) {
    match value {
        serde_json::Value::String(s) => {
            if let Ok(Reference::Node { node, .. }) = Reference::parse(s) {
                out.insert(node);
            }
        }
        serde_json::Value::Array(items) => {
            for item in items {
                collect_node_refs(item, out);
            }
        }
        serde_json::Value::Object(map) => {
            for item in map.values() {
                collect_node_refs(item, out);
            }
        }
        _ => {}
    }
}

/// Find a node whose op + default params make it a semantic identity on its
/// single input wire and which is safe to splice (`ALIEN_OPS` §13.2).
fn find_identity_node(
    plan: &Plan,
    nodes: &BTreeMap<String, ResolvedNode>,
    pinned: &BTreeSet<String>,
) -> Option<String> {
    nodes
        .values()
        .find(|n| !pinned.contains(&n.id) && is_identity_node(plan, n))
        .map(|n| n.id.clone())
}

/// Whether `node` is a parameter-aware identity: its op is in [`IDENTITY_OPS`],
/// it has exactly one input edge, and its params select the no-op behavior.
///
/// Today only `image.flip@1` with `axis: "none"` qualifies; the entry exists so
/// the identity set is parameter-aware (an op in [`IDENTITY_OPS`] is *not*
/// blanket-eliminated — its params must select the no-op).
fn is_identity_node(plan: &Plan, node: &ResolvedNode) -> bool {
    let op_str = node.op.to_string();
    if !IDENTITY_OPS.contains(&op_str.as_str()) || node.inputs.len() != 1 {
        return false;
    }
    let params = plan
        .nodes
        .iter()
        .find(|n| n.id == node.id)
        .map(|n| &n.params);
    match op_str.as_str() {
        "image.flip@1" => params
            .and_then(|p| p.get("axis"))
            .and_then(serde_json::Value::as_str)
            .is_some_and(|axis| axis == "none"),
        _ => false,
    }
}

/// Find an adjacent inverse pair `(producer, consumer)` on a single wire that
/// cancels to the identity (`ALIEN_OPS` §13.2 conversion cancellation).
///
/// The consumer must read exactly the producer's single output, the producer
/// must have exactly one input, and the op ids must be inverse (see
/// [`is_inverse_pair`]). Neither node may be pinned.
fn find_conversion_cancellation(
    nodes: &BTreeMap<String, ResolvedNode>,
    exports: &[ResolvedExport],
    pinned: &BTreeSet<String>,
    registry: &OperationRegistry,
) -> Option<ConversionPair> {
    for consumer in nodes.values() {
        if pinned.contains(&consumer.id) || consumer.inputs.len() != 1 {
            continue;
        }
        let edge = consumer.inputs.values().next()?;
        let Reference::Node {
            node: producer_id, ..
        } = edge
        else {
            continue;
        };
        if pinned.contains(producer_id) {
            continue;
        }
        let Some(producer) = nodes.get(producer_id) else {
            continue;
        };
        if producer.inputs.len() != 1 {
            continue;
        }
        // The producer's output must feed *only* this consumer, and no export may
        // read it directly; otherwise splicing it out would orphan a reader.
        if !is_sole_consumer(nodes, producer_id, &consumer.id) {
            continue;
        }
        if exports
            .iter()
            .any(|e| references_node(&e.resource, producer_id))
        {
            continue;
        }
        if is_inverse_pair(&producer.op.to_string(), &consumer.op.to_string(), registry) {
            return Some(ConversionPair {
                consumer: consumer.id.clone(),
                producer: producer_id.clone(),
            });
        }
    }
    None
}

/// An adjacent inverse pair to cancel.
struct ConversionPair {
    consumer: String,
    producer: String,
}

/// Whether `reference` is a `node:<producer_id>/<port>` reference.
fn references_node(reference: &Reference, producer_id: &str) -> bool {
    matches!(reference, Reference::Node { node, .. } if node == producer_id)
}

/// Whether `producer_id`'s output is read by exactly the one node `consumer_id`
/// (and by no export — checked by the caller, which holds the export list).
fn is_sole_consumer(
    nodes: &BTreeMap<String, ResolvedNode>,
    producer_id: &str,
    consumer_id: &str,
) -> bool {
    for n in nodes.values() {
        if n.id == consumer_id {
            continue;
        }
        if n.inputs.values().any(|e| references_node(e, producer_id)) {
            return false;
        }
    }
    true
}

/// Whether the op pair `(producer, consumer)` is a semantic-preserving inverse
/// pair under its §13.2 side condition.
///
/// - `alpha.premultiply@1` ∘ `alpha.unpremultiply@1` (either order) cancels under
///   the `alpha > ε` condition; we require the producer to declare the
///   determinism tier `Exact` so the round-trip is bit-exact (a `Bounded`/
///   nonlinear variant is *not* cancelled — `ALIEN_OPS` §13.3 "nonlinear
///   encodings unless algebra proves safety").
/// - `color.convert@1` ∘ `color.convert@1` is *not* blanket-cancelled here: a
///   generic convert pair is only an inverse when the two conversions are exact
///   round-trips, which needs the resolved params; that param-aware case is
///   reserved and currently returns `false` so we never fire an unsound rewrite.
fn is_inverse_pair(producer_op: &str, consumer_op: &str, registry: &OperationRegistry) -> bool {
    const PREMUL: &str = "alpha.premultiply@1";
    const UNPREMUL: &str = "alpha.unpremultiply@1";

    let pair = (producer_op, consumer_op);
    let is_alpha_pair = pair == (PREMUL, UNPREMUL) || pair == (UNPREMUL, PREMUL);
    if !is_alpha_pair {
        return false;
    }
    // The §13.3 nonlinear-encoding guard: only cancel when BOTH ops are exact, so
    // the premultiply/unpremultiply round-trip is provably bit-exact.
    [producer_op, consumer_op].iter().all(|op| {
        op.parse()
            .ok()
            .and_then(|id| registry.get(&id).ok())
            .is_some_and(|m| m.determinism == DeterminismTier::Exact)
    })
}

/// Find a common-subexpression pair `(kept, removed)`: two distinct, unpinned
/// nodes with the same op, params, and resolved inputs. `kept` is the
/// id-lexicographically-smaller node (deterministic), `removed` the other.
fn find_common_subexpression(
    plan: &Plan,
    nodes: &BTreeMap<String, ResolvedNode>,
    pinned: &BTreeSet<String>,
) -> Option<(String, String)> {
    // Bucket by a structural key; the first collision is the pair to merge.
    let mut seen: BTreeMap<CseKey, String> = BTreeMap::new();
    for node in nodes.values() {
        if pinned.contains(&node.id) {
            continue;
        }
        let key = cse_key(plan, node);
        if let Some(existing) = seen.get(&key) {
            // BTreeMap iteration is id-sorted, so `existing` is the smaller id and
            // is kept; `node` is the duplicate removed.
            return Some((existing.clone(), node.id.clone()));
        }
        seen.insert(key, node.id.clone());
    }
    None
}

/// The structural identity of a node for CSE: its op, its resolved input edges,
/// and its canonical params. Two nodes with an equal key compute the same value.
type CseKey = (String, BTreeMap<String, RefKey>, String);

/// A reference's structural key (the resolved edge target).
type RefKey = String;

/// Build the [`CseKey`] for a node: op id, input edges as canonical strings, and
/// the canonicalized params (so `{"a":1,"b":2}` and `{"b":2,"a":1}` collide).
fn cse_key(plan: &Plan, node: &ResolvedNode) -> CseKey {
    let inputs: BTreeMap<String, RefKey> = node
        .inputs
        .iter()
        .map(|(port, reference)| (port.clone(), reference_key(reference)))
        .collect();
    let params = plan
        .nodes
        .iter()
        .find(|n| n.id == node.id)
        .map(|n| canonical_params(&n.params))
        .unwrap_or_default();
    (node.op.to_string(), inputs, params)
}

/// Canonical string form of a node's params, key-sorted, so two parameter maps
/// that differ only in key order are equal for CSE.
fn canonical_params(params: &serde_json::Map<String, serde_json::Value>) -> String {
    let value = serde_json::Value::Object(params.clone());
    crate::to_canonical_string(&value).unwrap_or_else(|_| value.to_string())
}

/// The canonical string of a resolved reference (its edge target).
fn reference_key(reference: &Reference) -> RefKey {
    match reference {
        Reference::Input { input } => format!("input:{input}"),
        Reference::Node { node, port } => format!("node:{node}/{port}"),
    }
}

// ---- Rewrite application ---------------------------------------------------

/// Splice out an identity node: rewire every consumer of the node's single
/// output to the node's single input, then delete the node.
fn apply_identity_elimination(
    nodes: &mut BTreeMap<String, ResolvedNode>,
    exports: &mut [ResolvedExport],
    node_id: &str,
) {
    let Some(node) = nodes.get(node_id) else {
        return;
    };
    let Some(replacement) = node.inputs.values().next().cloned() else {
        return;
    };
    let output_port = "image"; // identity ops carry a single `image` output.
    rewire(nodes, exports, node_id, output_port, &replacement);
    nodes.remove(node_id);
}

/// Cancel an inverse pair: rewire the consumer's output consumers to the
/// producer's input, then delete both the consumer and producer.
fn apply_conversion_cancellation(
    nodes: &mut BTreeMap<String, ResolvedNode>,
    exports: &mut [ResolvedExport],
    pair: &ConversionPair,
) {
    let Some(producer) = nodes.get(&pair.producer) else {
        return;
    };
    let Some(replacement) = producer.inputs.values().next().cloned() else {
        return;
    };
    // Everything that read the consumer's output now reads the producer's input.
    rewire(nodes, exports, &pair.consumer, "image", &replacement);
    nodes.remove(&pair.consumer);
    nodes.remove(&pair.producer);
}

/// Merge a duplicate node: rewire every consumer of `removed`'s outputs onto the
/// corresponding `kept` outputs, then delete `removed`.
fn apply_cse(
    nodes: &mut BTreeMap<String, ResolvedNode>,
    exports: &mut [ResolvedExport],
    kept: &str,
    removed: &str,
) {
    // `kept` and `removed` share an op (CSE equality requires it), hence the same
    // output port names: rewire every reader of any of `removed`'s output ports
    // onto the same-named port of `kept`, then drop `removed`.
    rewire_all_ports(nodes, exports, removed, kept);
    nodes.remove(removed);
}

/// Rewire every reference to `from_node`'s output `port` to point at
/// `replacement` instead, across all node edges and exports.
fn rewire(
    nodes: &mut BTreeMap<String, ResolvedNode>,
    exports: &mut [ResolvedExport],
    from_node: &str,
    port: &str,
    replacement: &Reference,
) {
    for node in nodes.values_mut() {
        if node.id == from_node {
            continue;
        }
        for edge in node.inputs.values_mut() {
            if let Reference::Node { node: n, port: p } = edge
                && n == from_node
                && p == port
            {
                *edge = replacement.clone();
            }
        }
    }
    for export in exports.iter_mut() {
        if let Reference::Node { node: n, port: p } = &export.resource
            && n == from_node
            && p == port
        {
            export.resource = replacement.clone();
        }
    }
}

/// Rewire every reference to *any* output port of `from_node` onto the same port
/// of `to_node` (used by CSE, where the two nodes share output ports by op).
fn rewire_all_ports(
    nodes: &mut BTreeMap<String, ResolvedNode>,
    exports: &mut [ResolvedExport],
    from_node: &str,
    to_node: &str,
) {
    for node in nodes.values_mut() {
        if node.id == from_node {
            continue;
        }
        for edge in node.inputs.values_mut() {
            if let Reference::Node { node: n, .. } = edge
                && n == from_node
            {
                to_node.clone_into(n);
            }
        }
    }
    for export in exports.iter_mut() {
        if let Reference::Node { node: n, .. } = &mut export.resource
            && n == from_node
        {
            to_node.clone_into(n);
        }
    }
}

/// A deterministic topological order over the surviving nodes (`resolve.rs`
/// re-uses Kahn's algorithm; we duplicate the small routine here so simplify does
/// not need a public re-resolve). Ties broken by id.
fn topological_order(nodes: &BTreeMap<String, ResolvedNode>) -> Vec<String> {
    let mut in_degree: BTreeMap<&str, usize> = nodes.keys().map(|k| (k.as_str(), 0)).collect();
    let mut dependents: BTreeMap<&str, Vec<&str>> = BTreeMap::new();

    for node in nodes.values() {
        let mut upstream: Vec<&str> = node
            .inputs
            .values()
            .filter_map(|r| match r {
                Reference::Node { node, .. } => Some(node.as_str()),
                Reference::Input { .. } => None,
            })
            // A node may reference a producer that was removed mid-rewrite only
            // transiently; the final graph has no such edges, but guard anyway.
            .filter(|id| nodes.contains_key(*id))
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

    let mut ready: Vec<&str> = in_degree
        .iter()
        .filter(|&(_, &deg)| deg == 0)
        .map(|(&id, _)| id)
        .collect();
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
    order
}

/// Remove and return the lexicographically smallest id from `ready`.
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
    use super::{Rewrite, SimplifyOptions, simplify};
    use crate::manifest::{
        DeterminismTier, InputSpec, OperationManifest, OutputSpec, ResourceKind, RoiCategory,
        RoiPolicy,
    };
    use crate::plan::parse_plan;
    use crate::registry::OperationRegistry;
    use crate::resolve::{Reference, resolve_plan};

    fn op(id: &str, det: DeterminismTier, inputs: &[&str], outputs: &[&str]) -> OperationManifest {
        OperationManifest {
            id: id.parse().unwrap(),
            impl_version: 1,
            summary: String::new(),
            determinism: det,
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

    fn seed_param(id: &str) -> OperationManifest {
        let mut m = op(id, DeterminismTier::Stochastic, &["image"], &["image"]);
        m.params = vec![crate::manifest::ParamSpec {
            name: "seed".to_owned(),
            ty: crate::manifest::ParamType::Seed,
            unit: None,
            required: true,
            default: None,
            choices: vec![],
            doc: String::new(),
        }];
        m
    }

    fn registry() -> OperationRegistry {
        OperationRegistry::from_manifests([
            op("source.create@1", DeterminismTier::Exact, &[], &["image"]),
            op(
                "filter.invert@1",
                DeterminismTier::Exact,
                &["image"],
                &["image"],
            ),
            op(
                "filter.gaussian_blur@1",
                DeterminismTier::Exact,
                &["image"],
                &["image"],
            ),
            op(
                "alpha.premultiply@1",
                DeterminismTier::Exact,
                &["image"],
                &["image"],
            ),
            op(
                "alpha.unpremultiply@1",
                DeterminismTier::Exact,
                &["image"],
                &["image"],
            ),
            op(
                "image.flip@1",
                DeterminismTier::Exact,
                &["image"],
                &["image"],
            ),
            op(
                "io.encode_image@1",
                DeterminismTier::Exact,
                &["image"],
                &["image"],
            ),
            op(
                "debug.materialize@1",
                DeterminismTier::Exact,
                &["image"],
                &["image"],
            ),
            op(
                "assert.range@1",
                DeterminismTier::Exact,
                &["image"],
                &["image"],
            ),
            seed_param("model.inpaint@1"),
        ])
        .unwrap()
    }

    fn resolved(json: &str) -> (crate::plan::Plan, crate::resolve::ResolvedGraph) {
        let plan = parse_plan(json).unwrap();
        let graph = resolve_plan(&plan, &registry()).unwrap();
        (plan, graph)
    }

    #[test]
    fn disabled_pass_is_identity() {
        let (plan, graph) = resolved(
            r#"{
                "paintop": "1.0",
                "inputs": {"src": {"kind": "image.file", "path": "in.png"}},
                "nodes": [
                    {"id": "p", "op": "alpha.premultiply@1", "in": {"image": "input:src"}},
                    {"id": "u", "op": "alpha.unpremultiply@1", "in": {"image": "node:p/image"}}
                ],
                "exports": {"o": {"resource": "node:u/image", "kind": "image", "path": "o.png"}}
            }"#,
        );
        let (out, report) = simplify(&plan, &graph, &registry(), SimplifyOptions::DISABLED);
        assert_eq!(out, graph);
        assert_eq!(report.rewrite_count(), 0);
    }

    #[test]
    fn cancels_premultiply_unpremultiply_pair() {
        // src -> premultiply -> unpremultiply -> export should collapse so the
        // export reads `input:src` directly.
        let (plan, graph) = resolved(
            r#"{
                "paintop": "1.0",
                "inputs": {"src": {"kind": "image.file", "path": "in.png"}},
                "nodes": [
                    {"id": "p", "op": "alpha.premultiply@1", "in": {"image": "input:src"}},
                    {"id": "u", "op": "alpha.unpremultiply@1", "in": {"image": "node:p/image"}}
                ],
                "exports": {"o": {"resource": "node:u/image", "kind": "image", "path": "o.png"}}
            }"#,
        );
        let (out, report) = simplify(&plan, &graph, &registry(), SimplifyOptions::ENABLED);
        assert!(out.nodes().is_empty(), "both ops cancel out");
        assert_eq!(
            out.exports()[0].resource,
            Reference::Input {
                input: "src".to_owned()
            }
        );
        assert!(matches!(
            report.rewrites()[0],
            Rewrite::ConversionCancellation { .. }
        ));
    }

    #[test]
    fn does_not_cancel_when_producer_feeds_two_consumers() {
        // premultiply feeds BOTH an unpremultiply and a blur; splicing it out
        // would orphan the blur, so the pair must NOT cancel.
        let (plan, graph) = resolved(
            r#"{
                "paintop": "1.0",
                "inputs": {"src": {"kind": "image.file", "path": "in.png"}},
                "nodes": [
                    {"id": "p", "op": "alpha.premultiply@1", "in": {"image": "input:src"}},
                    {"id": "u", "op": "alpha.unpremultiply@1", "in": {"image": "node:p/image"}},
                    {"id": "b", "op": "filter.gaussian_blur@1", "in": {"image": "node:p/image"}}
                ],
                "exports": {
                    "a": {"resource": "node:u/image", "kind": "image", "path": "a.png"},
                    "c": {"resource": "node:b/image", "kind": "image", "path": "c.png"}
                }
            }"#,
        );
        let (out, report) = simplify(&plan, &graph, &registry(), SimplifyOptions::ENABLED);
        assert_eq!(report.rewrite_count(), 0);
        assert_eq!(out.nodes().len(), 3);
    }

    #[test]
    fn eliminates_identity_flip() {
        let (plan, graph) = resolved(
            r#"{
                "paintop": "1.0",
                "inputs": {"src": {"kind": "image.file", "path": "in.png"}},
                "nodes": [
                    {"id": "f", "op": "image.flip@1", "in": {"image": "input:src"},
                     "params": {"axis": "none"}}
                ],
                "exports": {"o": {"resource": "node:f/image", "kind": "image", "path": "o.png"}}
            }"#,
        );
        let (out, report) = simplify(&plan, &graph, &registry(), SimplifyOptions::ENABLED);
        assert!(out.nodes().is_empty());
        assert!(matches!(
            report.rewrites()[0],
            Rewrite::IdentityElimination { .. }
        ));
        assert_eq!(
            out.exports()[0].resource,
            Reference::Input {
                input: "src".to_owned()
            }
        );
    }

    #[test]
    fn keeps_non_identity_flip() {
        let (plan, graph) = resolved(
            r#"{
                "paintop": "1.0",
                "inputs": {"src": {"kind": "image.file", "path": "in.png"}},
                "nodes": [
                    {"id": "f", "op": "image.flip@1", "in": {"image": "input:src"},
                     "params": {"axis": "horizontal"}}
                ],
                "exports": {"o": {"resource": "node:f/image", "kind": "image", "path": "o.png"}}
            }"#,
        );
        let (out, report) = simplify(&plan, &graph, &registry(), SimplifyOptions::ENABLED);
        assert_eq!(out.nodes().len(), 1);
        assert_eq!(report.rewrite_count(), 0);
    }

    #[test]
    fn cse_collapses_duplicate_subgraphs() {
        // Two inverts of the same source feed a blend; they are the same value, so
        // CSE collapses to one.
        let (plan, graph) = resolved(
            r#"{
                "paintop": "1.0",
                "inputs": {"src": {"kind": "image.file", "path": "in.png"}},
                "nodes": [
                    {"id": "a", "op": "filter.invert@1", "in": {"image": "input:src"}},
                    {"id": "b", "op": "filter.invert@1", "in": {"image": "input:src"}},
                    {"id": "blur1", "op": "filter.gaussian_blur@1", "in": {"image": "node:a/image"}},
                    {"id": "blur2", "op": "filter.gaussian_blur@1", "in": {"image": "node:b/image"}}
                ],
                "exports": {
                    "x": {"resource": "node:blur1/image", "kind": "image", "path": "x.png"},
                    "y": {"resource": "node:blur2/image", "kind": "image", "path": "y.png"}
                }
            }"#,
        );
        let before = graph.nodes().len();
        let (out, report) = simplify(&plan, &graph, &registry(), SimplifyOptions::ENABLED);
        assert!(out.nodes().len() < before, "node count drops");
        // The two inverts collapse to one and the two blurs collapse to one.
        assert_eq!(out.nodes().len(), 2);
        assert!(report.rewrite_count() >= 2);
    }

    #[test]
    fn assertion_observed_intermediate_is_pinned() {
        // An assertion observing node:p/image pins it, so the premultiply pair
        // must NOT cancel even though it otherwise would.
        let (plan, graph) = resolved(
            r#"{
                "paintop": "1.0",
                "inputs": {"src": {"kind": "image.file", "path": "in.png"}},
                "nodes": [
                    {"id": "p", "op": "alpha.premultiply@1", "in": {"image": "input:src"}},
                    {"id": "u", "op": "alpha.unpremultiply@1", "in": {"image": "node:p/image"}}
                ],
                "assertions": [{"kind": "assert.finite@1", "resource": "node:p/image"}],
                "exports": {"o": {"resource": "node:u/image", "kind": "image", "path": "o.png"}}
            }"#,
        );
        let (out, report) = simplify(&plan, &graph, &registry(), SimplifyOptions::ENABLED);
        assert!(report.is_pinned("p"));
        assert_eq!(report.rewrite_count(), 0);
        assert_eq!(out.nodes().len(), 2);
    }

    #[test]
    fn debug_materialize_request_pins_node() {
        // The evidence block requests node:a/image materialized; CSE must not
        // merge `a` away.
        let (plan, graph) = resolved(
            r#"{
                "paintop": "1.0",
                "inputs": {"src": {"kind": "image.file", "path": "in.png"}},
                "nodes": [
                    {"id": "a", "op": "filter.invert@1", "in": {"image": "input:src"}},
                    {"id": "b", "op": "filter.invert@1", "in": {"image": "input:src"}}
                ],
                "evidence": {"materialize": ["node:a/image"]},
                "exports": {
                    "x": {"resource": "node:a/image", "kind": "image", "path": "x.png"},
                    "y": {"resource": "node:b/image", "kind": "image", "path": "y.png"}
                }
            }"#,
        );
        let (_out, report) = simplify(&plan, &graph, &registry(), SimplifyOptions::ENABLED);
        assert!(report.is_pinned("a"));
    }

    #[test]
    fn stochastic_node_is_pinned() {
        let (plan, graph) = resolved(
            r#"{
                "paintop": "1.0",
                "inputs": {"src": {"kind": "image.file", "path": "in.png"}},
                "nodes": [
                    {"id": "m1", "op": "model.inpaint@1", "in": {"image": "input:src"},
                     "params": {"seed": 7}},
                    {"id": "m2", "op": "model.inpaint@1", "in": {"image": "input:src"},
                     "params": {"seed": 7}}
                ],
                "exports": {
                    "x": {"resource": "node:m1/image", "kind": "image", "path": "x.png"},
                    "y": {"resource": "node:m2/image", "kind": "image", "path": "y.png"}
                }
            }"#,
        );
        let (out, report) = simplify(&plan, &graph, &registry(), SimplifyOptions::ENABLED);
        // Even though m1 and m2 are structurally identical, both are pinned
        // (stochastic) so CSE must not merge them.
        assert!(report.is_pinned("m1") && report.is_pinned("m2"));
        assert_eq!(out.nodes().len(), 2);
    }

    #[test]
    fn does_not_cancel_across_bounded_premultiply() {
        // A `Bounded` premultiply variant is a nonlinear-safe barrier: the pair
        // must not cancel (ALIEN_OPS §13.3).
        let reg = OperationRegistry::from_manifests([
            op(
                "alpha.premultiply@1",
                DeterminismTier::Bounded,
                &["image"],
                &["image"],
            ),
            op(
                "alpha.unpremultiply@1",
                DeterminismTier::Exact,
                &["image"],
                &["image"],
            ),
        ])
        .unwrap();
        let json = r#"{
            "paintop": "1.0",
            "inputs": {"src": {"kind": "image.file", "path": "in.png"}},
            "nodes": [
                {"id": "p", "op": "alpha.premultiply@1", "in": {"image": "input:src"}},
                {"id": "u", "op": "alpha.unpremultiply@1", "in": {"image": "node:p/image"}}
            ],
            "exports": {"o": {"resource": "node:u/image", "kind": "image", "path": "o.png"}}
        }"#;
        let plan = parse_plan(json).unwrap();
        let graph = resolve_plan(&plan, &reg).unwrap();
        let (out, report) = simplify(&plan, &graph, &reg, SimplifyOptions::ENABLED);
        assert_eq!(report.rewrite_count(), 0);
        assert_eq!(out.nodes().len(), 2);
    }

    #[test]
    fn simplification_is_a_fixed_point() {
        // Running simplify twice yields the same graph (idempotent on its own
        // output).
        let (plan, graph) = resolved(
            r#"{
                "paintop": "1.0",
                "inputs": {"src": {"kind": "image.file", "path": "in.png"}},
                "nodes": [
                    {"id": "p", "op": "alpha.premultiply@1", "in": {"image": "input:src"}},
                    {"id": "u", "op": "alpha.unpremultiply@1", "in": {"image": "node:p/image"}}
                ],
                "exports": {"o": {"resource": "node:u/image", "kind": "image", "path": "o.png"}}
            }"#,
        );
        let (once, _) = simplify(&plan, &graph, &registry(), SimplifyOptions::ENABLED);
        let (twice, report) = simplify(&plan, &once, &registry(), SimplifyOptions::ENABLED);
        assert_eq!(once, twice);
        assert_eq!(report.rewrite_count(), 0);
    }
}
