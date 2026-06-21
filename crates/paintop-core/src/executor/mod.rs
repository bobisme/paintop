//! The sequential whole-image executor (`plan.md` §10.1 phases 7 & 11).
//!
//! This is the minimal execution spine the MVP ops plug into. Given a
//! type-checked plan, it
//!
//! 1. computes **backward demand** from exports, assertions, and requested debug
//!    resources, eliminating dead nodes ([`demand`]);
//! 2. runs the surviving nodes in the resolved graph's deterministic
//!    **topological order**, *whole image* (no tiling, no cache — those are M2);
//! 3. dispatches each node through its operation's `cpu.reference`
//!    [`OpImplementation`], threading produced [`ResourceValue`]s to downstream
//!    consumers;
//! 4. emits a **structured trace event** per node (implementation selected,
//!    dispatch started, dispatch completed with elapsed time and `cache = bypass`)
//!    so an agent can follow each node's fate ([`crate::evidence::trace`]).
//!
//! A node the demand pass eliminated is never dispatched and never appears in the
//! trace — the proof, asserted by the integration tests, that dead-node
//! elimination bites.
//!
//! # What this bone deliberately does *not* do
//!
//! No tiling, no content-addressed cache, no parallelism, no policy/deadline
//! checks, and no real MVP ops (those land in segment 2). Cache lookups are traced
//! as [`CacheOutcome::Bypass`] because caching is M2; dispatch is single-threaded
//! and whole-image. The executor is exercised here with stub/identity
//! implementations registered in an [`ImplRegistry`].

pub mod demand;
pub mod dispatch;
pub mod error;
pub mod op_impl;
pub mod roi;
pub mod value;

use std::collections::BTreeMap;
use std::time::Instant;

use paintop_ir::{OperationRegistry, Plan, Reference, ResolvedGraph};

use crate::evidence::trace::{
    CacheOutcome, DispatchCompleted, DispatchStarted, DispatchStatus, TraceEvent,
};

pub use demand::{DemandTrace, compute_demand};
pub use dispatch::{
    BackendId, BackendPolicy, BackendSelection, E_BACKEND_UNSUPPORTED, select_backend,
};
pub use error::{
    E_IMPLEMENTATION_NOT_FOUND, E_INPUT_NOT_AVAILABLE, E_OP_DISPATCH_FAILED, E_OUTPUT_NOT_PRODUCED,
    ExecError, ExecResult,
};
pub use op_impl::{
    E_DUPLICATE_IMPLEMENTATION, ImplRegistry, InputValues, OpImplementation, OutputValues,
};
pub use roi::{RoiAnalysis, analyze_roi, analyze_roi_from_seeds};
pub use value::ResourceValue;

/// The product of one whole-image execution: the demand trace, every produced
/// node-output value, the resolved export values, and the structured trace
/// events emitted while running.
#[derive(Debug)]
pub struct Execution {
    demand: DemandTrace,
    node_outputs: BTreeMap<String, OutputValues>,
    exports: Vec<(String, ResourceValue)>,
    trace: Vec<TraceEvent>,
}

impl Execution {
    /// The demand trace: which nodes were demanded (and ran) and which were
    /// eliminated as dead (and did not).
    #[must_use]
    pub const fn demand(&self) -> &DemandTrace {
        &self.demand
    }

    /// The value produced on node `node`'s output port `port`, if `node` ran.
    #[must_use]
    pub fn output(&self, node: &str, port: &str) -> Option<&ResourceValue> {
        self.node_outputs.get(node).and_then(|p| p.get(port))
    }

    /// The resolved export values, in the resolved graph's export order.
    #[must_use]
    pub fn exports(&self) -> &[(String, ResourceValue)] {
        &self.exports
    }

    /// The structured trace events emitted during execution, in emission order.
    #[must_use]
    pub fn trace(&self) -> &[TraceEvent] {
        &self.trace
    }
}

/// Run the demanded nodes of `graph` whole-image, in topological order, through
/// their `cpu.reference` implementations (`plan.md` §10.1 phase 11).
///
/// `manifests` is the registry the graph resolved against; it supplies each
/// node's declared output ports and `cpu.reference` implementation id for the
/// trace. `inputs` supplies the concrete [`ResourceValue`] of each `input:`
/// resource (loaded from the plan's inputs by an earlier stage). `implementations`
/// holds the executable [`OpImplementation`] for every operation the demanded
/// subgraph uses. Demand is computed first, so a node reachable from no export,
/// assertion, or requested debug resource is eliminated and never dispatched.
///
/// Each dispatched node emits an `implementation_selected`, a `dispatch_started`,
/// a `cache_lookup` with [`CacheOutcome::Bypass`] (caching is M2), and a
/// `dispatch_completed` event carrying its elapsed time.
///
/// # Errors
/// - [`E_IMPLEMENTATION_NOT_FOUND`] if a demanded node's operation has no
///   registered implementation.
/// - [`E_INPUT_NOT_AVAILABLE`] if a node's wired input has no value at dispatch.
/// - [`E_OP_DISPATCH_FAILED`] if an implementation raises while computing a node.
/// - [`E_OUTPUT_NOT_PRODUCED`] if an implementation omits a declared output port.
pub fn execute(
    plan: &Plan,
    graph: &ResolvedGraph,
    manifests: &OperationRegistry,
    implementations: &ImplRegistry,
    inputs: &BTreeMap<String, ResourceValue>,
) -> ExecResult<Execution> {
    execute_with_policy(
        plan,
        graph,
        manifests,
        implementations,
        inputs,
        &BackendPolicy::reference(),
    )
}

/// Run the demanded subgraph under an explicit backend [`BackendPolicy`].
///
/// Identical to [`execute`] but the scheduler consults `policy` to choose which
/// backend serves each node (`plan.md` §12.2): a preferred optimized/`wgpu` kernel
/// when the op exposes one and it is registered, else the `cpu.reference` oracle
/// (or an explicit [`E_BACKEND_UNSUPPORTED`] error
/// for a *required* backend that cannot run the op). The selected backend is named
/// in each node's `implementation_selected` trace event, so the evidence records
/// the backend that served the node (`plan.md` §15).
///
/// [`execute`] is exactly `execute_with_policy(.., &BackendPolicy::reference())`,
/// so the default path is byte-identical to the pre-M3 reference executor.
///
/// # Errors
/// The same failures as [`execute`], plus
/// [`E_BACKEND_UNSUPPORTED`] lifted into an
/// [`ExecError::Dispatch`] when a required backend cannot serve a node.
pub fn execute_with_policy(
    plan: &Plan,
    graph: &ResolvedGraph,
    manifests: &OperationRegistry,
    implementations: &ImplRegistry,
    inputs: &BTreeMap<String, ResourceValue>,
    policy: &BackendPolicy,
) -> ExecResult<Execution> {
    let demand = compute_demand(plan, graph);

    // Node params are not retained on the resolved graph (they are not wiring);
    // index them back from the plan so param-dependent compute sees them.
    let params_by_node: BTreeMap<&str, &serde_json::Map<String, serde_json::Value>> = plan
        .nodes
        .iter()
        .map(|n| (n.id.as_str(), &n.params))
        .collect();

    let mut node_outputs: BTreeMap<String, OutputValues> = BTreeMap::new();
    let mut trace: Vec<TraceEvent> = Vec::new();

    // Demanded nodes are already in topological order, so every upstream producer
    // ran before its consumer.
    for node_id in demand.demanded() {
        let Some(node) = graph.node(node_id) else {
            continue;
        };
        let params = params_by_node
            .get(node_id.as_str())
            .map_or_else(serde_json::Value::default, |params| {
                serde_json::Value::Object((*params).clone())
            });
        let ctx = DispatchContext {
            manifests,
            implementations,
            inputs,
            node_outputs: &node_outputs,
            policy,
        };
        let outputs = dispatch_node(node_id, node, &ctx, &params, &mut trace)?;
        node_outputs.insert(node_id.clone(), outputs);
    }

    // Resolve every export's value from the produced outputs / external inputs.
    let mut exports = Vec::with_capacity(graph.exports().len());
    for export in graph.exports() {
        let value = resolve_value(&export.resource, inputs, &node_outputs).ok_or_else(|| {
            let node = match &export.resource {
                Reference::Node { node, .. } => node.clone(),
                Reference::Input { input } => input.clone(),
            };
            ExecError::InputNotAvailable {
                node,
                port: export.id.clone(),
                detail: describe_reference(&export.resource),
            }
        })?;
        exports.push((export.id.clone(), value.clone()));
    }

    Ok(Execution {
        demand,
        node_outputs,
        exports,
        trace,
    })
}

/// The borrowed registries and produced-so-far values one node dispatch needs.
struct DispatchContext<'a> {
    manifests: &'a OperationRegistry,
    implementations: &'a ImplRegistry,
    inputs: &'a BTreeMap<String, ResourceValue>,
    node_outputs: &'a BTreeMap<String, OutputValues>,
    policy: &'a BackendPolicy,
}

/// Dispatch one demanded node whole-image: assemble its inputs, select and run
/// its `cpu.reference` implementation, verify it produced every declared output,
/// and append the node's trace events to `trace`.
fn dispatch_node(
    node_id: &str,
    node: &paintop_ir::ResolvedNode,
    ctx: &DispatchContext<'_>,
    params: &serde_json::Value,
    trace: &mut Vec<TraceEvent>,
) -> ExecResult<OutputValues> {
    let op_str = node.op.to_string();

    // The manifest is the authority on declared output ports. Resolution proved
    // the op is registered.
    let manifest = ctx
        .manifests
        .get(&node.op)
        .map_err(|_| ExecError::ImplementationNotFound {
            node: node_id.to_owned(),
            op: op_str.clone(),
        })?;

    // Select the backend this node is dispatched on from policy (default =
    // reference). A required backend that cannot run the op surfaces as an
    // explicit dispatch failure, never a silent fallback.
    let selection =
        dispatch::select_backend(&node.op, ctx.manifests, ctx.implementations, ctx.policy)
            .map_err(|source| {
                // A selection failure where the op has *some* registered kernel is a
                // genuine backend-unsupported dispatch failure (e.g. a required
                // non-reference backend the op simply lacks). With no registered
                // kernel at all (an empty or partial registry) it is the classic
                // missing-implementation case, surfaced under its established code.
                if ctx.implementations.contains(&node.op) {
                    ExecError::Dispatch {
                        node: node_id.to_owned(),
                        op: op_str.clone(),
                        source: Box::new(source),
                    }
                } else {
                    ExecError::ImplementationNotFound {
                        node: node_id.to_owned(),
                        op: op_str.clone(),
                    }
                }
            })?;
    let impl_str = selection.impl_id().to_string();

    // Bind the executable kernel for the selected backend.
    let implementation = ctx
        .implementations
        .get_backend(&node.op, &selection.backend())
        .ok_or_else(|| ExecError::ImplementationNotFound {
            node: node_id.to_owned(),
            op: op_str.clone(),
        })?;

    // Assemble this node's input values; a miss is a runtime availability failure.
    let mut input_values: InputValues = InputValues::new();
    for (port, reference) in &node.inputs {
        let value = resolve_value(reference, ctx.inputs, ctx.node_outputs).ok_or_else(|| {
            ExecError::InputNotAvailable {
                node: node_id.to_owned(),
                port: port.clone(),
                detail: describe_reference(reference),
            }
        })?;
        input_values.insert(port.clone(), value.clone());
    }

    trace.push(TraceEvent::implementation_selected(
        node_id.to_owned(),
        op_str.clone(),
        impl_str.clone(),
    ));
    trace.push(TraceEvent::DispatchStarted(DispatchStarted {
        node: node_id.to_owned(),
        op: op_str.clone(),
        implementation: impl_str.clone(),
    }));
    // No cache in M0: the lookup is bypassed, recorded so the trace is honest.
    trace.push(TraceEvent::cache_lookup(
        node_id.to_owned(),
        op_str.clone(),
        CacheOutcome::Bypass,
    ));

    let started = Instant::now();
    let produced = implementation
        .compute(&input_values, params)
        .map_err(|source| ExecError::Dispatch {
            node: node_id.to_owned(),
            op: op_str.clone(),
            source: Box::new(source),
        })?;
    let elapsed_ms = started.elapsed().as_secs_f64() * 1_000.0;

    // The implementation must produce a value for every declared output port.
    let mut outputs: OutputValues = OutputValues::new();
    for spec in &manifest.outputs {
        let value = produced
            .get(&spec.name)
            .ok_or_else(|| ExecError::OutputNotProduced {
                node: node_id.to_owned(),
                op: op_str.clone(),
                port: spec.name.clone(),
            })?;
        outputs.insert(spec.name.clone(), value.clone());
    }

    trace.push(TraceEvent::DispatchCompleted(DispatchCompleted {
        node: node_id.to_owned(),
        op: op_str,
        implementation: impl_str,
        status: DispatchStatus::Completed,
        elapsed_ms: Some(elapsed_ms),
        alloc_bytes: None,
    }));

    Ok(outputs)
}

/// Resolve a [`Reference`] to the concrete value it carries: an external input
/// from `inputs`, or an upstream node output from `node_outputs`.
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

/// A short human-readable rendering of a reference for an error detail.
fn describe_reference(reference: &Reference) -> String {
    match reference {
        Reference::Input { input } => format!("external input `{input}` was not supplied"),
        Reference::Node { node, port } => {
            format!("upstream output `node:{node}/{port}` was not produced")
        }
    }
}
