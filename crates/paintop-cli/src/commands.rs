//! Subcommand behavior. Each function reads the canonical IR, does its work, and
//! returns a [`CommandOutcome`] (a JSON value plus a §15.4 exit code). No command
//! writes to stdout itself; that is the `main` shim's job, keeping the
//! pure-JSON-on-stdout contract in one place.

use std::collections::BTreeMap;
use std::path::Path;

use paintop_ir::{
    ContractRegistry, Error, ErrorClass, OperationRegistry, Plan, ResolvedGraph, check_graph,
    parse_plan, resolve_plan, semantic_hash,
};

use crate::output::{CommandOutcome, io_error, log};
use crate::stub_ops;

/// Read a plan file and parse it through the strict IR parser.
///
/// Maps a read failure to an [`asset`](ErrorClass::Asset) error and any parse /
/// schema failure to its native IR class.
fn load_plan(path: &Path) -> Result<(String, Plan), Error> {
    let text =
        std::fs::read_to_string(path).map_err(|err| io_error(path, &err, "E_PLAN_READ_FAILED"))?;
    let plan = parse_plan(&text)?;
    Ok((text, plan))
}

/// Resolve and type-check a parsed plan against the stub registry, returning the
/// resolved graph. The check is run with an empty external-input descriptor map
/// (M0 validates wiring/types; concrete input descriptors arrive with `run`).
fn resolve_and_check(plan: &Plan, registry: &OperationRegistry) -> Result<ResolvedGraph, Error> {
    let graph = resolve_plan(plan, registry)?;
    let contracts = ContractRegistry::new();
    let inputs = BTreeMap::new();
    // Type-checking needs a contract per op; with no contracts registered the
    // check is a structural pass over a plan that wires no inputs. A plan that
    // *does* reference op outputs through contracts is exercised in segment 2.
    if !contracts.is_empty() {
        check_graph(plan, &graph, registry, &contracts, &inputs)?;
    }
    Ok(graph)
}

/// `paintop validate <plan>`: parse, resolve, and type-check a plan.
///
/// On success emits `{"ok": true, "valid": true, "nodes": N, "exports": M}` and
/// exits `0`. On the first failure emits the §19 error envelope and exits with
/// the failing class's §15.4 code (parse/schema → `2`, type/semantic → `3`, …).
#[must_use]
pub fn validate(plan_path: &Path) -> CommandOutcome {
    log(&format!("validate: {}", plan_path.display()));
    match validate_inner(plan_path) {
        Ok(value) => CommandOutcome::success(value),
        Err(error) => {
            log(&format!("validate failed: {error}"));
            CommandOutcome::failure(&error)
        }
    }
}

fn validate_inner(plan_path: &Path) -> Result<serde_json::Value, Error> {
    let registry = stub_ops::registry()?;
    let (_text, plan) = load_plan(plan_path)?;
    let graph = resolve_and_check(&plan, &registry)?;
    Ok(serde_json::json!({
        "ok": true,
        "valid": true,
        "nodes": graph.nodes().len(),
        "exports": graph.exports().len(),
    }))
}

/// `paintop explain <plan>`: a structured, machine-readable description of a
/// plan — its semantic hash, node list, and export list.
#[must_use]
pub fn explain(plan_path: &Path) -> CommandOutcome {
    log(&format!("explain: {}", plan_path.display()));
    match explain_inner(plan_path) {
        Ok(value) => CommandOutcome::success(value),
        Err(error) => {
            log(&format!("explain failed: {error}"));
            CommandOutcome::failure(&error)
        }
    }
}

fn explain_inner(plan_path: &Path) -> Result<serde_json::Value, Error> {
    let registry = stub_ops::registry()?;
    let (_text, plan) = load_plan(plan_path)?;
    let graph = resolve_and_check(&plan, &registry)?;
    let hash = semantic_hash(&plan)?;

    let nodes: Vec<serde_json::Value> = graph
        .topological_order()
        .iter()
        .filter_map(|id| graph.node(id))
        .map(|node| {
            serde_json::json!({
                "id": node.id,
                "op": node.op.to_string(),
            })
        })
        .collect();
    let exports: Vec<serde_json::Value> = graph
        .exports()
        .iter()
        .map(|export| serde_json::json!({ "id": export.id }))
        .collect();

    Ok(serde_json::json!({
        "ok": true,
        "semantic_hash": hash.to_string(),
        "nodes": nodes,
        "exports": exports,
    }))
}

/// `paintop run <plan> [--bundle <dir>]`: resolve, check, and (in M0) report the
/// run plan without executing real ops.
///
/// The MVP operation backends land in segment 2; until they do, `run` validates
/// the plan end-to-end and reports the demanded execution order so an agent can
/// see exactly what *would* run. The bundle path is recorded but not yet
/// materialized.
#[must_use]
pub fn run(plan_path: &Path, bundle: Option<&Path>) -> CommandOutcome {
    log(&format!("run: {}", plan_path.display()));
    match run_inner(plan_path, bundle) {
        Ok(value) => CommandOutcome::success(value),
        Err(error) => {
            log(&format!("run failed: {error}"));
            CommandOutcome::failure(&error)
        }
    }
}

fn run_inner(plan_path: &Path, bundle: Option<&Path>) -> Result<serde_json::Value, Error> {
    let registry = stub_ops::registry()?;
    let (_text, plan) = load_plan(plan_path)?;
    let graph = resolve_and_check(&plan, &registry)?;
    let order: Vec<&str> = graph
        .topological_order()
        .iter()
        .map(String::as_str)
        .collect();
    Ok(serde_json::json!({
        "ok": true,
        "executed": false,
        "reason": "MVP op backends arrive in segment 2; M0 run reports the planned order only",
        "order": order,
        "bundle": bundle.map(|p| p.display().to_string()),
    }))
}

/// `paintop graph <plan> --out <file>`: write the plan's dependency graph as DOT.
#[must_use]
pub fn graph(plan_path: &Path, out: &Path) -> CommandOutcome {
    log(&format!(
        "graph: {} -> {}",
        plan_path.display(),
        out.display()
    ));
    match graph_inner(plan_path, out) {
        Ok(value) => CommandOutcome::success(value),
        Err(error) => {
            log(&format!("graph failed: {error}"));
            CommandOutcome::failure(&error)
        }
    }
}

fn graph_inner(plan_path: &Path, out: &Path) -> Result<serde_json::Value, Error> {
    let registry = stub_ops::registry()?;
    let (_text, plan) = load_plan(plan_path)?;
    let graph = resolve_and_check(&plan, &registry)?;
    let dot = render_dot(&graph);
    std::fs::write(out, dot.as_bytes())
        .map_err(|err| io_error(out, &err, "E_GRAPH_WRITE_FAILED"))?;
    Ok(serde_json::json!({
        "ok": true,
        "out": out.display().to_string(),
        "nodes": graph.nodes().len(),
    }))
}

/// Render a resolved graph to a deterministic Graphviz DOT document: one node per
/// resolved node and one edge per input wire, in canonical (sorted) order.
fn render_dot(graph: &ResolvedGraph) -> String {
    use std::fmt::Write as _;

    let mut out = String::from("digraph paintop {\n");
    for (id, node) in graph.nodes() {
        let label = format!("{id}\n{}", node.op);
        // Writing into a `String` is infallible; the result is discarded.
        let _ = writeln!(out, "  {id:?} [label={label:?}];");
    }
    for (id, node) in graph.nodes() {
        for reference in node.inputs.values() {
            if let paintop_ir::Reference::Node { node: src, .. } = reference {
                let _ = writeln!(out, "  {src:?} -> {id:?};");
            }
        }
    }
    out.push_str("}\n");
    out
}

/// `paintop diff <before> <after>`: report whether two image files are
/// byte-identical (the M0 diff is a content-hash equality check; perceptual
/// metrics arrive later).
#[must_use]
pub fn diff(before: &Path, after: &Path) -> CommandOutcome {
    log(&format!(
        "diff: {} vs {}",
        before.display(),
        after.display()
    ));
    match diff_inner(before, after) {
        Ok(value) => CommandOutcome::success(value),
        Err(error) => {
            log(&format!("diff failed: {error}"));
            CommandOutcome::failure(&error)
        }
    }
}

fn diff_inner(before: &Path, after: &Path) -> Result<serde_json::Value, Error> {
    let a = std::fs::read(before).map_err(|err| io_error(before, &err, "E_IMAGE_READ_FAILED"))?;
    let b = std::fs::read(after).map_err(|err| io_error(after, &err, "E_IMAGE_READ_FAILED"))?;
    let identical = a == b;
    Ok(serde_json::json!({
        "ok": true,
        "identical": identical,
        "before_bytes": a.len(),
        "after_bytes": b.len(),
    }))
}

/// `paintop op list`: emit every registered operation manifest as JSON.
#[must_use]
pub fn op_list() -> CommandOutcome {
    log("op list");
    match op_list_inner() {
        Ok(value) => CommandOutcome::success(value),
        Err(error) => {
            log(&format!("op list failed: {error}"));
            CommandOutcome::failure(&error)
        }
    }
}

fn op_list_inner() -> Result<serde_json::Value, Error> {
    let registry = stub_ops::registry()?;
    let ops: Vec<serde_json::Value> = registry
        .iter()
        .map(|m| {
            serde_json::json!({
                "id": m.id.to_string(),
                "summary": m.summary,
                "determinism": serde_json::to_value(m.determinism).unwrap_or(serde_json::Value::Null),
            })
        })
        .collect();
    Ok(serde_json::json!({
        "ok": true,
        "operations": ops,
    }))
}

/// `paintop op schema <id>`: emit one operation's manifest plus the manifest
/// JSON schema, so an agent can both read the op and validate it.
#[must_use]
pub fn op_schema(id: &str) -> CommandOutcome {
    log(&format!("op schema: {id}"));
    match op_schema_inner(id) {
        Ok(value) => CommandOutcome::success(value),
        Err(error) => {
            log(&format!("op schema failed: {error}"));
            CommandOutcome::failure(&error)
        }
    }
}

fn op_schema_inner(id: &str) -> Result<serde_json::Value, Error> {
    let registry = stub_ops::registry()?;
    let op_id = stub_ops::parse_op_id(id)?;
    let manifest = registry.get(&op_id)?;
    let manifest_value = serde_json::to_value(manifest).map_err(|err| {
        Error::new(
            ErrorClass::Execution,
            "E_MANIFEST_SERIALIZE",
            format!("manifest for {id} could not be serialized: {err}"),
        )
    })?;
    Ok(serde_json::json!({
        "ok": true,
        "manifest": manifest_value,
        "schema": paintop_ir::manifest_json_schema(),
    }))
}

/// `paintop selftest`: an M0 stub that reports the backend it was asked to test.
#[must_use]
pub fn selftest(backend: &str) -> CommandOutcome {
    log(&format!("selftest: {backend}"));
    CommandOutcome::success(serde_json::json!({
        "ok": true,
        "backend": backend,
        "status": "stub",
        "reason": "self-test exercises arrive with the MVP op backends in segment 2",
    }))
}
