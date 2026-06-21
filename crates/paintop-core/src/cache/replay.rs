//! Cache-aware execution and **zero-recompute replay** (`plan.md` §10.3).
//!
//! [`execute_cached`] runs a demanded graph exactly like the whole-image executor
//! ([`crate::executor::execute`]) but interposes the content-addressed cache at
//! every node: it computes the node's [`CacheKey`] from its op, semantic version,
//! params, and the *content hashes* of its assembled inputs, then
//!
//! 1. **looks up** every declared output port; if all hit, it reuses the cached
//!    values and **does not call the implementation at all** — the node's pure
//!    producer work is skipped (trace `cache = hit`);
//! 2. otherwise **computes** the node, **stores** each output under its key, and
//!    threads the values downstream (trace `cache = miss`).
//!
//! The store is keyed by content, so a second run over an unchanged plan finds
//! every node already cached and executes **zero** producer nodes — the property
//! the M2 zero-recompute gate checks. Conversely, changing any semantic input
//! flips the affected node's key (and, transitively, its consumers' input content
//! hashes), so only the affected subgraph misses and recomputes.
//!
//! Because the cache key encodes the full semantic identity of a computation, a
//! hit's value is *bit-identical* to what recomputation would produce: caching is
//! an optimization that never changes output.

use std::collections::BTreeMap;

use paintop_ir::{
    OperationManifest, OperationRegistry, Plan, Reference, ResolvedGraph, ResolvedNode,
};

use crate::cache::content::content_hash_value;
use crate::cache::error::CacheResult;
use crate::cache::key::{CacheKey, CacheKeyInputs, InputContribution};
use crate::cache::store::{CacheStore, CacheValidation};
use crate::evidence::trace::{
    CacheLookup, CacheOutcome, DispatchCompleted, DispatchStarted, DispatchStatus, TraceEvent,
};
use crate::executor::{
    DemandTrace, ExecError, ExecResult, ImplRegistry, InputValues, OutputValues, ResourceValue,
    compute_demand,
};

/// The product of one cache-aware execution.
///
/// Mirrors [`crate::executor::Execution`] but additionally exposes which nodes hit
/// the cache and which were recomputed, so a replay test can assert zero-recompute
/// directly without parsing the trace.
#[derive(Debug)]
pub struct CachedExecution {
    demand: DemandTrace,
    node_outputs: BTreeMap<String, OutputValues>,
    exports: Vec<(String, ResourceValue)>,
    trace: Vec<TraceEvent>,
    /// Node ids served entirely from cache (no `compute` call), in run order.
    cache_hits: Vec<String>,
    /// Node ids that were recomputed (at least one output missed), in run order.
    recomputed: Vec<String>,
}

impl CachedExecution {
    /// The demand trace: which nodes were demanded and which were dead.
    #[must_use]
    pub const fn demand(&self) -> &DemandTrace {
        &self.demand
    }

    /// The value produced on node `node`'s output port `port`.
    #[must_use]
    pub fn output(&self, node: &str, port: &str) -> Option<&ResourceValue> {
        self.node_outputs.get(node).and_then(|p| p.get(port))
    }

    /// The resolved export values, in export order.
    #[must_use]
    pub fn exports(&self) -> &[(String, ResourceValue)] {
        &self.exports
    }

    /// The structured trace events, in emission order.
    #[must_use]
    pub fn trace(&self) -> &[TraceEvent] {
        &self.trace
    }

    /// The node ids served entirely from cache (their producer work was skipped).
    #[must_use]
    pub fn cache_hits(&self) -> &[String] {
        &self.cache_hits
    }

    /// The node ids that were recomputed.
    #[must_use]
    pub fn recomputed(&self) -> &[String] {
        &self.recomputed
    }
}

/// Run the demanded nodes of `graph` whole-image with the content-addressed cache
/// `cache` interposed (`plan.md` §10.3).
///
/// Identical to [`crate::executor::execute`] except that each node is looked up in
/// `cache` before dispatch and stored after a miss. A node whose every declared
/// output is found (and validates) is reused without calling its implementation.
///
/// # Errors
/// - The executor's dispatch failures ([`ExecError`]).
/// - A cache failure ([`crate::cache::CacheError`], lifted into the central
///   taxonomy) if a key cannot be computed or a stored entry is corrupt /
///   incompatible.
pub fn execute_cached(
    plan: &Plan,
    graph: &ResolvedGraph,
    manifests: &OperationRegistry,
    implementations: &ImplRegistry,
    inputs: &BTreeMap<String, ResourceValue>,
    cache: &mut CacheStore,
) -> ExecResult<CachedExecution> {
    let demand = compute_demand(plan, graph);

    let params_by_node: BTreeMap<&str, &serde_json::Map<String, serde_json::Value>> = plan
        .nodes
        .iter()
        .map(|n| (n.id.as_str(), &n.params))
        .collect();

    let mut node_outputs: BTreeMap<String, OutputValues> = BTreeMap::new();
    let mut trace: Vec<TraceEvent> = Vec::new();
    let mut cache_hits: Vec<String> = Vec::new();
    let mut recomputed: Vec<String> = Vec::new();

    for node_id in demand.demanded() {
        let Some(node) = graph.node(node_id) else {
            continue;
        };
        let params = params_by_node
            .get(node_id.as_str())
            .map_or_else(serde_json::Map::new, |p| (*p).clone());

        let outputs = run_node(
            node_id,
            node,
            manifests,
            implementations,
            inputs,
            &node_outputs,
            &params,
            cache,
            &mut trace,
            &mut cache_hits,
            &mut recomputed,
        )?;
        node_outputs.insert(node_id.clone(), outputs);
    }

    let mut exports = Vec::with_capacity(graph.exports().len());
    for export in graph.exports() {
        let value = resolve_value(&export.resource, inputs, &node_outputs).ok_or_else(|| {
            ExecError::InputNotAvailable {
                node: reference_node(&export.resource),
                port: export.id.clone(),
                detail: describe_reference(&export.resource),
            }
        })?;
        exports.push((export.id.clone(), value.clone()));
    }

    Ok(CachedExecution {
        demand,
        node_outputs,
        exports,
        trace,
        cache_hits,
        recomputed,
    })
}

/// Run one node: assemble inputs, compute its cache key, attempt a full-cache
/// hit, and on a miss compute, store, and trace.
#[expect(
    clippy::too_many_arguments,
    reason = "the cache-aware dispatch threads the executor's registries, the \
              store, and the per-run accounting; bundling them would only move the \
              argument list into a short-lived struct"
)]
fn run_node(
    node_id: &str,
    node: &ResolvedNode,
    manifests: &OperationRegistry,
    implementations: &ImplRegistry,
    inputs: &BTreeMap<String, ResourceValue>,
    node_outputs: &BTreeMap<String, OutputValues>,
    params: &serde_json::Map<String, serde_json::Value>,
    cache: &mut CacheStore,
    trace: &mut Vec<TraceEvent>,
    cache_hits: &mut Vec<String>,
    recomputed: &mut Vec<String>,
) -> ExecResult<OutputValues> {
    let op_str = node.op.to_string();
    let manifest = manifests
        .get(&node.op)
        .map_err(|_| ExecError::ImplementationNotFound {
            node: node_id.to_owned(),
            op: op_str.clone(),
        })?;

    // Assemble this node's input values up front (needed for both the key and a
    // recompute).
    let mut input_values: InputValues = InputValues::new();
    for (port, reference) in &node.inputs {
        let value = resolve_value(reference, inputs, node_outputs).ok_or_else(|| {
            ExecError::InputNotAvailable {
                node: node_id.to_owned(),
                port: port.clone(),
                detail: describe_reference(reference),
            }
        })?;
        input_values.insert(port.clone(), value.clone());
    }

    let key = node_cache_key(manifest, params, &input_values).map_err(ExecError::from_cache)?;
    let validation = node_validation(manifest);

    // Try a full-cache hit: every declared output present and valid.
    if let Some(outputs) =
        try_cache_hit(cache, &key, &validation, manifest).map_err(ExecError::from_cache)?
    {
        trace.push(TraceEvent::CacheLookup(CacheLookup {
            node: node_id.to_owned(),
            op: op_str,
            outcome: CacheOutcome::Hit,
            key: Some(key.to_string()),
        }));
        cache_hits.push(node_id.to_owned());
        return Ok(outputs);
    }

    // Miss: compute, store, trace.
    trace.push(TraceEvent::CacheLookup(CacheLookup {
        node: node_id.to_owned(),
        op: op_str.clone(),
        outcome: CacheOutcome::Miss,
        key: Some(key.to_string()),
    }));
    let implementation =
        implementations
            .get(&node.op)
            .ok_or_else(|| ExecError::ImplementationNotFound {
                node: node_id.to_owned(),
                op: op_str.clone(),
            })?;

    let impl_str = reference_impl_id(manifest);
    trace.push(TraceEvent::DispatchStarted(DispatchStarted {
        node: node_id.to_owned(),
        op: op_str.clone(),
        implementation: impl_str.clone(),
    }));

    let params_value = serde_json::Value::Object(params.clone());
    let produced = implementation
        .compute(&input_values, &params_value)
        .map_err(|source| ExecError::Dispatch {
            node: node_id.to_owned(),
            op: op_str.clone(),
            source: Box::new(source),
        })?;

    let mut outputs: OutputValues = OutputValues::new();
    for spec in &manifest.outputs {
        let value = produced
            .get(&spec.name)
            .ok_or_else(|| ExecError::OutputNotProduced {
                node: node_id.to_owned(),
                op: op_str.clone(),
                port: spec.name.clone(),
            })?;
        cache
            .put(&key.for_output(&spec.name), value, validation.clone())
            .map_err(ExecError::from_cache)?;
        outputs.insert(spec.name.clone(), value.clone());
    }

    trace.push(TraceEvent::DispatchCompleted(DispatchCompleted {
        node: node_id.to_owned(),
        op: op_str,
        implementation: impl_str,
        status: DispatchStatus::Completed,
        elapsed_ms: None,
        alloc_bytes: None,
    }));
    recomputed.push(node_id.to_owned());
    Ok(outputs)
}

/// Attempt to satisfy *every* declared output of a node from the cache.
///
/// Returns `Some(outputs)` only when all declared output ports hit and validate;
/// if any port misses, returns `None` so the caller recomputes the whole node (a
/// partial hit is not reused — the node is computed as a unit).
fn try_cache_hit(
    cache: &CacheStore,
    key: &CacheKey,
    validation: &CacheValidation,
    manifest: &OperationManifest,
) -> CacheResult<Option<OutputValues>> {
    let mut outputs = OutputValues::new();
    for spec in &manifest.outputs {
        let port_key = key.for_output(&spec.name);
        match cache.get(&port_key, validation)? {
            Some(value) => {
                outputs.insert(spec.name.clone(), value);
            }
            None => return Ok(None),
        }
    }
    Ok(Some(outputs))
}

/// Build the node's [`CacheKey`] from its manifest, params, and the *content
/// hashes* of its assembled input values.
fn node_cache_key(
    manifest: &OperationManifest,
    params: &serde_json::Map<String, serde_json::Value>,
    input_values: &InputValues,
) -> CacheResult<CacheKey> {
    let contributions = input_values
        .iter()
        .map(|(port, value)| {
            InputContribution::new(port.clone(), content_hash_value(value), *value.descriptor())
        })
        .collect();
    let key_inputs = CacheKeyInputs::new(
        manifest.id.clone(),
        manifest.impl_version,
        params.clone(),
        contributions,
    );
    CacheKey::compute(&key_inputs)
}

/// The validation metadata a node's entries carry.
fn node_validation(manifest: &OperationManifest) -> CacheValidation {
    CacheValidation {
        op_id: manifest.id.to_string(),
        op_semantic_version: manifest.impl_version,
        backend_semantics_version: crate::cache::key::BACKEND_SEMANTICS_VERSION,
    }
}

/// Resolve a [`Reference`] to its concrete value (external input or upstream
/// output).
fn resolve_value<'a>(
    reference: &Reference,
    inputs: &'a BTreeMap<String, ResourceValue>,
    node_outputs: &'a BTreeMap<String, OutputValues>,
) -> Option<&'a ResourceValue> {
    match reference {
        Reference::Input { input } => inputs.get(input),
        Reference::Node { node, port } => node_outputs.get(node).and_then(|p| p.get(port)),
    }
}

/// The node/input id a reference names, for an error.
fn reference_node(reference: &Reference) -> String {
    match reference {
        Reference::Node { node, .. } => node.clone(),
        Reference::Input { input } => input.clone(),
    }
}

/// A short human-readable rendering of a reference for an error detail.
fn describe_reference(reference: &Reference) -> String {
    match reference {
        Reference::Input { input } => format!("external input `{input}` was not supplied"),
        Reference::Node { node, port } => {
            format!("upstream output `node:{node}/{port}` was not produced")
        }
    }
}

/// The `cpu.reference@<v>` oracle id declared by `manifest`, for the trace.
fn reference_impl_id(manifest: &OperationManifest) -> String {
    manifest
        .implementations
        .iter()
        .find(|i| {
            i.backend() == paintop_ir::CPU_REFERENCE_BACKEND
                && i.name() == paintop_ir::CPU_REFERENCE_NAME
        })
        .map_or_else(
            || {
                format!(
                    "{}.{}@{}",
                    paintop_ir::CPU_REFERENCE_BACKEND,
                    paintop_ir::CPU_REFERENCE_NAME,
                    manifest.impl_version
                )
            },
            ToString::to_string,
        )
}
