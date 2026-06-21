//! Graphviz DOT rendering of the normalized execution graph (`plan.md` §15.4,
//! §15.1).
//!
//! `paintop graph plan.json --out graph.dot` renders the *normalized* DAG — the
//! exact graph the executor runs — as a Graphviz DOT document. The rendering is
//! the evidence-bundle `graph.dot` artifact (`plan.md` §15.1) and is meant to be
//! read by an agent or a human and round-tripped through `dot`/`neato`.
//!
//! # What a node carries
//!
//! Each graph node is one DOT node whose record label shows:
//!
//! - the **node id** and its **op id** (`filter.gaussian_blur@1`);
//! - the op's declared **input ports** and **output ports**, each with its
//!   **resource kind** (`image : Image`, `mask : Mask`, …) drawn from the
//!   operation manifest;
//! - the node's **demand status** — `demanded` (it runs) or `eliminated` (dead,
//!   removed by the backward-demand pass), so the picture matches what executes;
//! - the op's **ROI category** (`pointwise`, `local-halo(24px)`, `full-domain`,
//!   …) as the per-node ROI annotation.
//!
//! # What an edge carries
//!
//! One directed edge per wired input port, from the producer node to the consumer
//! node, labeled `<producer-port> → <consumer-port> : <kind>` so the data flow,
//! the ports it connects, and the resource kind on the wire are all visible. An
//! external `input:<id>` edge is drawn from a distinct source node so inputs are
//! visible without being confused with graph nodes.
//!
//! # Determinism + round-trip
//!
//! Nodes and edges are emitted in the resolved graph's deterministic order
//! (`BTreeMap` / topological), so the DOT bytes are stable across runs. The
//! emitted document is valid DOT (every identifier and label is quoted/escaped)
//! and the node/edge set matches the normalized graph exactly — the properties
//! the structural test and the Graphviz round-trip pin.

use std::collections::BTreeSet;
use std::fmt::Write as _;

use paintop_ir::{
    OperationRegistry, Plan, Reference, ResolvedGraph, ResolvedNode, ResourceKind, RoiCategory,
};

use crate::executor::compute_demand;

/// Render the normalized graph of `plan` / `graph` to a Graphviz DOT document
/// (`plan.md` §15.4).
///
/// `registry` supplies each op's declared ports, their resource kinds, and its
/// ROI category for the per-node annotations; the backward-demand pass supplies
/// each node's demanded/eliminated status. The output is deterministic and valid
/// DOT (every label is escaped) whose node and edge set matches the normalized
/// graph exactly.
#[must_use]
pub fn render_dot(plan: &Plan, graph: &ResolvedGraph, registry: &OperationRegistry) -> String {
    let demand = compute_demand(plan, graph);

    let mut out = String::from("digraph paintop {\n");
    out.push_str("  rankdir=LR;\n");
    out.push_str("  node [shape=record];\n");

    // External input source nodes: one per declared input id, drawn distinctly so
    // an `input:<id>` edge is visible without being mistaken for a graph node.
    let used_inputs = collect_used_inputs(graph);
    for input in &used_inputs {
        let label = format!("input:{input}");
        let _ = writeln!(
            out,
            "  {} [label={}, shape=oval, style=dashed];",
            quote(&input_node_id(input)),
            quote(&label)
        );
    }

    // Graph nodes, in deterministic id order.
    for (id, node) in graph.nodes() {
        let label = node_label(node, registry, demand.is_demanded(id));
        let _ = writeln!(out, "  {} [label={}];", quote(id), quote(&label));
    }

    // Edges, in deterministic (node id, port name) order.
    for (consumer_id, node) in graph.nodes() {
        for (consumer_port, reference) in &node.inputs {
            let (src, edge_label) = edge_for(reference, consumer_port, registry, graph);
            let _ = writeln!(
                out,
                "  {} -> {} [label={}];",
                quote(&src),
                quote(consumer_id),
                quote(&edge_label)
            );
        }
    }

    out.push_str("}\n");
    out
}

/// The set of `input:<id>` resources actually referenced by the graph (by node
/// edges or exports), in deterministic order.
fn collect_used_inputs(graph: &ResolvedGraph) -> Vec<String> {
    let mut inputs = BTreeSet::new();
    for node in graph.nodes().values() {
        for reference in node.inputs.values() {
            if let Reference::Input { input } = reference {
                inputs.insert(input.clone());
            }
        }
    }
    for export in graph.exports() {
        if let Reference::Input { input } = &export.resource {
            inputs.insert(input.clone());
        }
    }
    inputs.into_iter().collect()
}

/// The DOT node id for an external input source (namespaced so it can never
/// collide with a graph node id, which may not contain `:`).
fn input_node_id(input: &str) -> String {
    format!("input:{input}")
}

/// Build a node's record label: id, op, ports-with-kinds, demand status, and ROI.
fn node_label(node: &ResolvedNode, registry: &OperationRegistry, demanded: bool) -> String {
    let mut parts: Vec<String> = Vec::new();
    parts.push(node.id.clone());
    parts.push(node.op.to_string());

    if let Ok(manifest) = registry.get(&node.op) {
        if !manifest.inputs.is_empty() {
            let ins: Vec<String> = manifest
                .inputs
                .iter()
                .map(|i| format!("{} : {}", i.name, kind_str(i.kind)))
                .collect();
            parts.push(format!("in: {}", ins.join(", ")));
        }
        let outs: Vec<String> = manifest
            .outputs
            .iter()
            .map(|o| format!("{} : {}", o.name, kind_str(o.kind)))
            .collect();
        parts.push(format!("out: {}", outs.join(", ")));
        parts.push(format!("roi: {}", roi_str(&manifest.roi)));
    }

    parts.push(if demanded {
        "demanded".to_owned()
    } else {
        "eliminated".to_owned()
    });
    // Record labels separate fields with `|`; we already escape the whole label.
    parts.join(" | ")
}

/// Build the `(source-dot-id, edge-label)` for one consumer input edge.
fn edge_for(
    reference: &Reference,
    consumer_port: &str,
    registry: &OperationRegistry,
    graph: &ResolvedGraph,
) -> (String, String) {
    match reference {
        Reference::Input { input } => {
            let kind = export_input_kind();
            (input_node_id(input), format!("→ {consumer_port}{kind}"))
        }
        Reference::Node {
            node: producer,
            port: producer_port,
        } => {
            let kind = producer_port_kind(graph, registry, producer, producer_port)
                .map(|k| format!(" : {}", kind_str(k)))
                .unwrap_or_default();
            (
                producer.clone(),
                format!("{producer_port} → {consumer_port}{kind}"),
            )
        }
    }
}

/// The resource kind a producer node emits on `port`, looked up via the
/// producer's op manifest. `None` if the producer or port is unknown (a graph
/// that resolved cannot hit that for a node edge).
fn producer_port_kind(
    graph: &ResolvedGraph,
    registry: &OperationRegistry,
    producer: &str,
    port: &str,
) -> Option<ResourceKind> {
    let node = graph.node(producer)?;
    let manifest = registry.get(&node.op).ok()?;
    manifest
        .outputs
        .iter()
        .find(|o| o.name == port)
        .map(|o| o.kind)
}

/// An `input:` edge carries no manifest-declared kind here (the input's kind is
/// declared on the plan's `inputs` block, which the renderer does not resolve to
/// a `ResourceKind`); we leave the kind off rather than guess.
const fn export_input_kind() -> &'static str {
    ""
}

/// A short stable token for a [`ResourceKind`]. `#[non_exhaustive]`, so a future
/// variant falls back to a generic label rather than failing to build.
const fn kind_str(kind: ResourceKind) -> &'static str {
    match kind {
        ResourceKind::Image => "Image",
        ResourceKind::Mask => "Mask",
        ResourceKind::Field1 => "Field1",
        ResourceKind::Field2 => "Field2",
        ResourceKind::Field3 => "Field3",
        ResourceKind::SdfMask => "SdfMask",
        ResourceKind::CandidateSet => "CandidateSet",
        ResourceKind::Report => "Report",
        _ => "Resource",
    }
}

/// A short ROI annotation for a node, e.g. `pointwise`, `local-halo(24px)`,
/// `full-domain`. `#[non_exhaustive]`, so a future category falls back to a
/// generic label.
fn roi_str(roi: &paintop_ir::RoiPolicy) -> String {
    match roi.category {
        RoiCategory::Pointwise => "pointwise".to_owned(),
        RoiCategory::LocalHalo => roi
            .halo_px
            .map_or_else(|| "local-halo".to_owned(), |h| format!("local-halo({h}px)")),
        RoiCategory::Geometric => "geometric".to_owned(),
        RoiCategory::ConnectedComponent => "connected-component".to_owned(),
        RoiCategory::FullDomain => "full-domain".to_owned(),
        _ => "roi".to_owned(),
    }
}

/// Quote and escape a string as a DOT double-quoted ID (`plan.md` §15.4 valid
/// DOT). Backslashes and double-quotes are escaped; newlines become the DOT
/// line-break `\n`. This keeps every label and id a single valid quoted token so
/// the document round-trips through Graphviz.
fn quote(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            other => out.push(other),
        }
    }
    out.push('"');
    out
}

/// The set of node ids and the directed `(producer, consumer)` edges the rendered
/// DOT contains — the *structural* graph, decoupled from formatting.
///
/// This is what the structural test compares against the normalized graph: the
/// rendered node set must equal the graph's node set (plus the external input
/// sources), and the edge set must equal the resolved input wires.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DotStructure {
    /// Graph node ids (excludes the external `input:` source ovals).
    pub nodes: Vec<String>,
    /// External `input:<id>` source ids that appear in the rendering.
    pub input_sources: Vec<String>,
    /// Directed edges `(source-dot-id, consumer-id)` in emission order.
    pub edges: Vec<(String, String)>,
}

/// Extract the structural node/edge set the renderer would emit, so a test can
/// assert it matches the normalized graph without parsing DOT text.
#[must_use]
pub fn dot_structure(graph: &ResolvedGraph) -> DotStructure {
    let input_sources = collect_used_inputs(graph)
        .into_iter()
        .map(|i| input_node_id(&i))
        .collect();
    let nodes: Vec<String> = graph.nodes().keys().cloned().collect();
    let mut edges: Vec<(String, String)> = Vec::new();
    for (consumer_id, node) in graph.nodes() {
        for reference in node.inputs.values() {
            let src = match reference {
                Reference::Input { input } => input_node_id(input),
                Reference::Node { node, .. } => node.clone(),
            };
            edges.push((src, consumer_id.clone()));
        }
    }
    DotStructure {
        nodes,
        input_sources,
        edges,
    }
}

#[cfg(test)]
mod tests {
    use super::{dot_structure, render_dot};
    use paintop_ir::{
        DeterminismTier, InputSpec, OperationManifest, OperationRegistry, OutputSpec, ResourceKind,
        RoiCategory, RoiPolicy, parse_plan, resolve_plan,
    };

    fn op(
        id: &str,
        cat: RoiCategory,
        halo: Option<u32>,
        inputs: &[(&str, ResourceKind)],
        outputs: &[(&str, ResourceKind)],
    ) -> OperationManifest {
        OperationManifest {
            id: id.parse().expect("ok"),
            impl_version: 1,
            summary: String::new(),
            determinism: DeterminismTier::Exact,
            roi: RoiPolicy {
                category: cat,
                halo_px: halo,
            },
            inputs: inputs
                .iter()
                .map(|(name, kind)| InputSpec {
                    name: (*name).to_owned(),
                    kind: *kind,
                    required: true,
                    doc: String::new(),
                })
                .collect(),
            outputs: outputs
                .iter()
                .map(|(name, kind)| OutputSpec {
                    name: (*name).to_owned(),
                    kind: *kind,
                    doc: String::new(),
                })
                .collect(),
            params: vec![],
            implementations: vec!["cpu.reference@1".parse().expect("ok")],
            test: paintop_ir::TestMetadata::default(),
        }
    }

    fn registry() -> OperationRegistry {
        OperationRegistry::from_manifests([
            op(
                "source.create@1",
                RoiCategory::Pointwise,
                None,
                &[],
                &[("image", ResourceKind::Image)],
            ),
            op(
                "filter.gaussian_blur@1",
                RoiCategory::LocalHalo,
                Some(24),
                &[("image", ResourceKind::Image)],
                &[("image", ResourceKind::Image)],
            ),
            op(
                "mask.rect@1",
                RoiCategory::Pointwise,
                None,
                &[],
                &[("mask", ResourceKind::Mask)],
            ),
            op(
                "filter.invert@1",
                RoiCategory::Pointwise,
                None,
                &[("image", ResourceKind::Image)],
                &[("image", ResourceKind::Image)],
            ),
        ])
        .expect("ok")
    }

    fn graph(json: &str) -> (paintop_ir::Plan, paintop_ir::ResolvedGraph) {
        let plan = parse_plan(json).expect("parse");
        let g = resolve_plan(&plan, &registry()).expect("resolve");
        (plan, g)
    }

    const BLUR_PLAN: &str = r#"{
        "paintop": "1.0",
        "inputs": {"src": {"kind": "image.file", "path": "in.png"}},
        "nodes": [
            {"id": "b", "op": "filter.gaussian_blur@1", "in": {"image": "input:src"}}
        ],
        "exports": {"o": {"resource": "node:b/image", "kind": "image", "path": "o.png"}}
    }"#;

    #[test]
    fn dot_is_valid_and_carries_ports_kinds_roi() {
        let (plan, g) = graph(BLUR_PLAN);
        let dot = render_dot(&plan, &g, &registry());
        // Well-formed digraph envelope.
        assert!(dot.starts_with("digraph paintop {\n"));
        assert!(dot.trim_end().ends_with('}'));
        // Node id + op id present.
        assert!(dot.contains("filter.gaussian_blur@1"));
        // Port + resource kind labeled.
        assert!(dot.contains("image : Image"));
        // ROI annotation present (local-halo with halo px).
        assert!(dot.contains("local-halo(24px)"));
        // Demand annotation present.
        assert!(dot.contains("demanded"));
        // External input source drawn.
        assert!(dot.contains("input:src"));
        // Every quote is balanced (a crude validity proxy).
        assert_eq!(dot.matches('"').count() % 2, 0);
    }

    #[test]
    fn structure_matches_normalized_graph_exactly() {
        let (_plan, g) = graph(
            r#"{
                "paintop": "1.0",
                "inputs": {"src": {"kind": "image.file", "path": "in.png"}},
                "nodes": [
                    {"id": "b", "op": "filter.gaussian_blur@1", "in": {"image": "input:src"}},
                    {"id": "i", "op": "filter.invert@1", "in": {"image": "node:b/image"}}
                ],
                "exports": {"o": {"resource": "node:i/image", "kind": "image", "path": "o.png"}}
            }"#,
        );
        let s = dot_structure(&g);
        // Node set equals the graph's node set.
        assert_eq!(s.nodes, vec!["b".to_owned(), "i".to_owned()]);
        // One external input source.
        assert_eq!(s.input_sources, vec!["input:src".to_owned()]);
        // Edges: src -> b, and b -> i.
        assert_eq!(
            s.edges,
            vec![
                ("input:src".to_owned(), "b".to_owned()),
                ("b".to_owned(), "i".to_owned()),
            ]
        );
    }

    #[test]
    fn eliminated_node_is_labeled_eliminated() {
        // `dead` is reachable from no export, so the demand pass eliminates it; the
        // rendering must say so (the picture matches what executes).
        let (plan, g) = graph(
            r#"{
                "paintop": "1.0",
                "inputs": {"src": {"kind": "image.file", "path": "in.png"}},
                "nodes": [
                    {"id": "used", "op": "filter.invert@1", "in": {"image": "input:src"}},
                    {"id": "dead", "op": "filter.invert@1", "in": {"image": "input:src"}}
                ],
                "exports": {"o": {"resource": "node:used/image", "kind": "image", "path": "o.png"}}
            }"#,
        );
        let dot = render_dot(&plan, &g, &registry());
        assert!(dot.contains("eliminated"), "dead node labeled eliminated");
        assert!(dot.contains("demanded"), "used node labeled demanded");
    }

    #[test]
    fn label_escaping_keeps_dot_valid() {
        // A node-record label embeds the op id and `:` separators; ensure they are
        // inside quoted tokens (no stray unescaped quote).
        let (plan, g) = graph(BLUR_PLAN);
        let dot = render_dot(&plan, &g, &registry());
        for line in dot.lines() {
            assert_eq!(
                line.matches('"').count() % 2,
                0,
                "unbalanced quotes: {line}"
            );
        }
    }

    #[test]
    fn rendering_is_deterministic() {
        let (plan, g) = graph(BLUR_PLAN);
        let a = render_dot(&plan, &g, &registry());
        let b = render_dot(&plan, &g, &registry());
        assert_eq!(a, b);
    }
}
