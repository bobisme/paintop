//! The structured trace schema for an evidence bundle's `trace.jsonl`
//! (`plan.md` §15.1, §18.1; `AGENT_VERIFICATION` §5.2).
//!
//! The trace is a stream of [JSON Lines](https://jsonlines.org/): one JSON object
//! per line, each describing one thing the runtime did — a plan parsed, a node
//! normalized, a tile scheduled, an implementation selected, a dispatch finished,
//! an assertion measured, an output written, a failure raised. Because the trace
//! is streamed, every line must **validate independently**: a reader can parse any
//! single line without the others, so each carries its own [schema version](
//! TraceEvent::SCHEMA) and the full set of **stable keys** for its event kind.
//!
//! ## Identity is preserved end to end
//!
//! The trace's job is to let an agent follow one node's fate through compilation
//! and execution. So the keys that establish identity — the graph `node` id, the
//! versioned `op` id, and the selected `implementation` id — are spelled the same
//! way in every event that carries them ([`NodeRef`], [`OpRef`], [`ImplRef`]). A
//! reader can `grep` a node id across the file and reconstruct exactly which op
//! and implementation served it.
//!
//! ## What this bone owns
//!
//! This bone defines the **schema and the event-construction helpers** plus the
//! [`TraceWriter`] that appends canonical JSON lines to `trace.jsonl`. The
//! executor, scheduler, and assertion stages (later bones) emit these events; they
//! do not redefine the wire form. Wire stability is the contract, so the enum is
//! tagged on a stable `"event"` discriminant and every payload struct is
//! `deny_unknown_fields`.
//!
//! ```
//! use paintop_core::evidence::trace::{TraceEvent, CacheOutcome};
//!
//! let ev = TraceEvent::cache_lookup("blur_1", "filter.gaussian_blur@1", CacheOutcome::Miss);
//! let line = ev.to_line().expect("serialize");
//! assert!(!line.contains('\n'));
//! let back = TraceEvent::from_line(&line).expect("parse");
//! assert_eq!(back, ev);
//! ```

use serde::{Deserialize, Serialize};

use paintop_ir::{Rect, to_canonical_string};

use crate::evidence::error::{BundleError, BundleResult, E_BUNDLE_SERIALIZE};
use crate::evidence::layout::{BundleLayout, files};

/// The graph node identity carried by node-scoped trace events.
///
/// Spelled identically (`"node"`) in every event so a node id can be `grep`-ed
/// across the whole trace.
pub type NodeRef = String;

/// A versioned operation identity, e.g. `"filter.gaussian_blur@1"`.
pub type OpRef = String;

/// A selected implementation identity, e.g. `"cpu.separable@1"`.
pub type ImplRef = String;

/// The outcome of a content-addressed cache lookup (`AGENT_VERIFICATION` §5.2
/// "cache lookup"; the wire strings match the `cache` field of `plan.md` §18.1).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
#[non_exhaustive]
pub enum CacheOutcome {
    /// The result was found in the cache and reused.
    Hit,
    /// The result was absent and had to be computed.
    Miss,
    /// The lookup was skipped (caching disabled / not applicable).
    Bypass,
}

/// The terminal status of a dispatched node computation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
#[non_exhaustive]
pub enum DispatchStatus {
    /// The dispatch produced its output successfully.
    Completed,
    /// The dispatch failed.
    Failed,
    /// The dispatch was cancelled before completion.
    Cancelled,
}

/// The status of a measured runtime assertion (`AGENT_VERIFICATION` §5.3).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
#[non_exhaustive]
pub enum AssertionStatus {
    /// The assertion held.
    Passed,
    /// The assertion was violated.
    Failed,
    /// The assertion could not be evaluated and was skipped.
    Skipped,
}

/// The tile accounting for a scheduled node (`plan.md` §18.1 `tiles`).
///
/// Identity tiles (`identity`) are regions the op leaves unchanged and can copy
/// through; `executed` excludes them. All counts are non-negative.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TileCounts {
    /// Tiles the schedule wanted to cover the output region.
    pub requested: u64,
    /// Tiles actually computed (excludes identity tiles).
    pub executed: u64,
    /// Tiles short-circuited as identity (copied through unchanged).
    pub identity: u64,
}

/// A `plan parsed` event: the input plan was parsed and its semantic hash taken
/// (`AGENT_VERIFICATION` §5.2).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PlanParsed {
    /// The plan's `blake3:…` semantic hash.
    pub plan_semantic_hash: String,
    /// The number of nodes in the parsed graph.
    pub node_count: u64,
}

/// A `node normalized` event: a node was canonicalized (defaults resolved, prose
/// stripped) during compilation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NodeNormalized {
    /// The graph node identity.
    pub node: NodeRef,
    /// The versioned op the node invokes.
    pub op: OpRef,
}

/// A `resource demanded` event: a node declared the output region it must produce
/// and the input regions it needs to do so (`plan.md` §18.1
/// `input_regions`/`output_region`/`halo`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResourceDemanded {
    /// The graph node identity.
    pub node: NodeRef,
    /// The versioned op the node invokes.
    pub op: OpRef,
    /// The region the node must produce.
    pub output_region: Rect,
    /// The per-input regions the node reads, keyed by input port name.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub input_regions: Vec<InputRegion>,
    /// The halo (in pixels) the op reads beyond its output region.
    #[serde(default, skip_serializing_if = "is_zero")]
    pub halo: u32,
}

/// One named input region demanded by a node (`plan.md` §18.1 `input_regions`).
///
/// Modeled as an explicit list (not a map) so the trace line is order-stable and
/// canonicalizes deterministically.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InputRegion {
    /// The input port name (e.g. `"image"`, `"mask"`).
    pub port: String,
    /// The region read on that port.
    pub region: Rect,
}

/// A `cache lookup` event: a content-addressed lookup for a node's result.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CacheLookup {
    /// The graph node identity.
    pub node: NodeRef,
    /// The versioned op the node invokes.
    pub op: OpRef,
    /// The lookup outcome.
    pub outcome: CacheOutcome,
    /// The content-addressed cache key (`blake3:…`), if the lookup computed one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub key: Option<String>,
}

/// A `tile scheduled` event: the scheduler decided the tile breakdown for a node.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TileScheduled {
    /// The graph node identity.
    pub node: NodeRef,
    /// The versioned op the node invokes.
    pub op: OpRef,
    /// The tile accounting for the node.
    pub tiles: TileCounts,
}

/// An `implementation selected` event: the policy engine bound a node's op to a
/// concrete backend implementation.
///
/// This is the event that records *which* implementation served a node — the
/// trace's load-bearing identity for differential and conformance debugging
/// (`plan.md` §15.6: "trace and evidence output identify the implementation
/// used").
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ImplementationSelected {
    /// The graph node identity.
    pub node: NodeRef,
    /// The versioned op the node invokes.
    pub op: OpRef,
    /// The selected backend implementation identity.
    pub implementation: ImplRef,
}

/// A `dispatch started` event: a node's computation began on its bound
/// implementation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DispatchStarted {
    /// The graph node identity.
    pub node: NodeRef,
    /// The versioned op the node invokes.
    pub op: OpRef,
    /// The implementation serving the dispatch.
    pub implementation: ImplRef,
}

/// A `dispatch completed` event: a node's computation finished (or failed), with
/// its cost accounting (`plan.md` §18.1 `elapsed_ms`/`alloc_bytes`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DispatchCompleted {
    /// The graph node identity.
    pub node: NodeRef,
    /// The versioned op the node invokes.
    pub op: OpRef,
    /// The implementation that served the dispatch.
    pub implementation: ImplRef,
    /// The terminal status of the dispatch.
    pub status: DispatchStatus,
    /// Wall-clock cost in milliseconds, if measured.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub elapsed_ms: Option<f64>,
    /// Peak bytes allocated for the dispatch, if measured.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub alloc_bytes: Option<u64>,
}

/// An `assertion measured` event: a runtime assertion was evaluated
/// (`AGENT_VERIFICATION` §5.2/§5.3).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AssertionMeasured {
    /// The assertion's stable id (e.g. `"localized"`).
    pub id: String,
    /// The versioned assertion op (e.g. `"assert.no_change_outside_mask@1"`).
    pub op: OpRef,
    /// Whether the assertion held.
    pub status: AssertionStatus,
}

/// An `output written` event: an export artifact was materialized.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OutputWritten {
    /// The plan export name this artifact realizes.
    pub name: String,
    /// The bundle-relative path written.
    pub path: String,
    /// The content hash of the artifact bytes (`blake3:…`), if computed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content_hash: Option<String>,
}

/// A `failure` event: the run was cancelled or a node failed
/// (`AGENT_VERIFICATION` §5.2 "cancellation/failure").
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Failure {
    /// The stable machine error code (e.g. `"E_ASSERTION_VIOLATED"`).
    pub code: String,
    /// A human-readable message.
    pub message: String,
    /// The node the failure is attributed to, if any.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub node: Option<NodeRef>,
}

/// One structured trace event (`AGENT_VERIFICATION` §5.2).
///
/// Serialized as a self-describing JSON object: a stable `"event"` discriminant, a
/// `"schema"` version, and the event-specific stable keys flattened alongside.
/// Because the discriminant and version travel on every line, any single line
/// validates independently of the rest of the stream.
///
/// The enum is `non_exhaustive`: later bones may add event kinds, and a reader
/// built against this schema version must tolerate that.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "event", rename_all = "snake_case")]
#[non_exhaustive]
pub enum TraceEvent {
    /// The input plan was parsed.
    PlanParsed(PlanParsed),
    /// A node was normalized during compilation.
    NodeNormalized(NodeNormalized),
    /// A node declared its resource demand.
    ResourceDemanded(ResourceDemanded),
    /// A content-addressed cache lookup occurred.
    CacheLookup(CacheLookup),
    /// The scheduler decided a node's tile breakdown.
    TileScheduled(TileScheduled),
    /// The policy engine bound a node to an implementation.
    ImplementationSelected(ImplementationSelected),
    /// A node's dispatch began.
    DispatchStarted(DispatchStarted),
    /// A node's dispatch finished.
    DispatchCompleted(DispatchCompleted),
    /// A runtime assertion was evaluated.
    AssertionMeasured(AssertionMeasured),
    /// An export artifact was written.
    OutputWritten(OutputWritten),
    /// The run was cancelled or a node failed.
    Failure(Failure),
}

/// The wire key under which every event carries its schema version.
const SCHEMA_KEY: &str = "schema";

impl TraceEvent {
    /// The trace schema version stamped onto every emitted line.
    ///
    /// Bumped only on an incompatible change to the event wire form; readers can
    /// gate on it.
    pub const SCHEMA: u32 = 1;

    /// Construct a [`TraceEvent::PlanParsed`].
    #[must_use]
    pub fn plan_parsed(plan_semantic_hash: impl Into<String>, node_count: u64) -> Self {
        Self::PlanParsed(PlanParsed {
            plan_semantic_hash: plan_semantic_hash.into(),
            node_count,
        })
    }

    /// Construct a [`TraceEvent::NodeNormalized`].
    #[must_use]
    pub fn node_normalized(node: impl Into<NodeRef>, op: impl Into<OpRef>) -> Self {
        Self::NodeNormalized(NodeNormalized {
            node: node.into(),
            op: op.into(),
        })
    }

    /// Construct a [`TraceEvent::CacheLookup`].
    #[must_use]
    pub fn cache_lookup(
        node: impl Into<NodeRef>,
        op: impl Into<OpRef>,
        outcome: CacheOutcome,
    ) -> Self {
        Self::CacheLookup(CacheLookup {
            node: node.into(),
            op: op.into(),
            outcome,
            key: None,
        })
    }

    /// Construct a [`TraceEvent::ImplementationSelected`].
    #[must_use]
    pub fn implementation_selected(
        node: impl Into<NodeRef>,
        op: impl Into<OpRef>,
        implementation: impl Into<ImplRef>,
    ) -> Self {
        Self::ImplementationSelected(ImplementationSelected {
            node: node.into(),
            op: op.into(),
            implementation: implementation.into(),
        })
    }

    /// The stable `"event"` discriminant string for this event kind.
    #[must_use]
    pub const fn kind(&self) -> &'static str {
        match self {
            Self::PlanParsed(_) => "plan_parsed",
            Self::NodeNormalized(_) => "node_normalized",
            Self::ResourceDemanded(_) => "resource_demanded",
            Self::CacheLookup(_) => "cache_lookup",
            Self::TileScheduled(_) => "tile_scheduled",
            Self::ImplementationSelected(_) => "implementation_selected",
            Self::DispatchStarted(_) => "dispatch_started",
            Self::DispatchCompleted(_) => "dispatch_completed",
            Self::AssertionMeasured(_) => "assertion_measured",
            Self::OutputWritten(_) => "output_written",
            Self::Failure(_) => "failure",
        }
    }

    /// The graph node id this event is attributed to, if any.
    ///
    /// Lets a reader follow one node's identity across the trace regardless of
    /// event kind.
    #[must_use]
    pub fn node(&self) -> Option<&str> {
        match self {
            Self::NodeNormalized(e) => Some(&e.node),
            Self::ResourceDemanded(e) => Some(&e.node),
            Self::CacheLookup(e) => Some(&e.node),
            Self::TileScheduled(e) => Some(&e.node),
            Self::ImplementationSelected(e) => Some(&e.node),
            Self::DispatchStarted(e) => Some(&e.node),
            Self::DispatchCompleted(e) => Some(&e.node),
            Self::Failure(e) => e.node.as_deref(),
            Self::PlanParsed(_) | Self::AssertionMeasured(_) | Self::OutputWritten(_) => None,
        }
    }

    /// The versioned op id this event is attributed to, if any.
    #[must_use]
    pub fn op(&self) -> Option<&str> {
        match self {
            Self::NodeNormalized(e) => Some(&e.op),
            Self::ResourceDemanded(e) => Some(&e.op),
            Self::CacheLookup(e) => Some(&e.op),
            Self::TileScheduled(e) => Some(&e.op),
            Self::ImplementationSelected(e) => Some(&e.op),
            Self::DispatchStarted(e) => Some(&e.op),
            Self::DispatchCompleted(e) => Some(&e.op),
            Self::AssertionMeasured(e) => Some(&e.op),
            Self::PlanParsed(_) | Self::OutputWritten(_) | Self::Failure(_) => None,
        }
    }

    /// The selected implementation id this event is attributed to, if any.
    #[must_use]
    pub fn implementation(&self) -> Option<&str> {
        match self {
            Self::ImplementationSelected(e) => Some(&e.implementation),
            Self::DispatchStarted(e) => Some(&e.implementation),
            Self::DispatchCompleted(e) => Some(&e.implementation),
            _ => None,
        }
    }

    /// Serialize this event to one canonical JSON object value, with the `"event"`
    /// discriminant and `"schema"` version present.
    ///
    /// The returned value has sorted keys (canonical form) so a re-run produces
    /// byte-identical trace lines.
    ///
    /// # Errors
    /// Returns [`BundleError::Serialize`] if the event cannot be serialized.
    pub fn to_value(&self) -> BundleResult<serde_json::Value> {
        let mut value = serde_json::to_value(self).map_err(|e| {
            BundleError::serialize(
                "serializing trace event",
                paintop_ir::Error::new(
                    paintop_ir::ErrorClass::Export,
                    E_BUNDLE_SERIALIZE,
                    e.to_string(),
                ),
            )
        })?;
        // `to_value` of an internally-tagged enum is always a JSON object; stamp
        // the schema version alongside the discriminant the tag added.
        if let Some(obj) = value.as_object_mut() {
            obj.insert(SCHEMA_KEY.to_owned(), serde_json::Value::from(Self::SCHEMA));
        }
        Ok(value)
    }

    /// Serialize this event to a single canonical JSONL line (no trailing
    /// newline).
    ///
    /// The line is one canonical JSON object — sorted keys, single float format —
    /// and contains no embedded newline, so it is a valid JSON Lines record.
    ///
    /// # Errors
    /// Returns [`BundleError::Serialize`] if the event cannot be canonicalized.
    pub fn to_line(&self) -> BundleResult<String> {
        let value = self.to_value()?;
        to_canonical_string(&value)
            .map_err(|e| BundleError::serialize("canonicalizing trace line", e))
    }

    /// Parse one JSONL line back into a [`TraceEvent`], validating it
    /// independently of the rest of the stream.
    ///
    /// Rejects a line whose `"schema"` version does not match [`Self::SCHEMA`] and
    /// any line that does not deserialize into a known event with the exact stable
    /// keys (payload structs are `deny_unknown_fields`).
    ///
    /// # Errors
    /// Returns [`BundleError::Serialize`] if the line is not a JSON object, carries
    /// an unrecognized schema version, or does not match an event shape.
    pub fn from_line(line: &str) -> BundleResult<Self> {
        let mut value: serde_json::Value = serde_json::from_str(line).map_err(|e| {
            BundleError::serialize(
                "parsing trace line as json",
                paintop_ir::Error::new(
                    paintop_ir::ErrorClass::Export,
                    E_BUNDLE_SERIALIZE,
                    e.to_string(),
                ),
            )
        })?;
        let obj = value.as_object_mut().ok_or_else(|| {
            BundleError::serialize(
                "trace line is not a json object",
                paintop_ir::Error::new(
                    paintop_ir::ErrorClass::Export,
                    E_BUNDLE_SERIALIZE,
                    "expected a json object".to_owned(),
                ),
            )
        })?;
        // Validate and strip the schema version before delegating to the tagged
        // enum (whose payloads `deny_unknown_fields` and would reject `schema`).
        match obj.remove(SCHEMA_KEY) {
            Some(serde_json::Value::Number(n)) if n.as_u64() == Some(u64::from(Self::SCHEMA)) => {}
            other => {
                return Err(BundleError::serialize(
                    "trace line carries an unsupported schema version",
                    paintop_ir::Error::new(
                        paintop_ir::ErrorClass::Export,
                        E_BUNDLE_SERIALIZE,
                        format!("expected schema {}, found {other:?}", Self::SCHEMA),
                    ),
                ));
            }
        }
        serde_json::from_value(value).map_err(|e| {
            BundleError::serialize(
                "matching trace line against the event schema",
                paintop_ir::Error::new(
                    paintop_ir::ErrorClass::Export,
                    E_BUNDLE_SERIALIZE,
                    e.to_string(),
                ),
            )
        })
    }
}

/// Appends structured events to a bundle's `trace.jsonl`.
///
/// JSON Lines is an append-only stream, so the writer opens `trace.jsonl` for
/// appending and writes one canonical line (newline-terminated) per event. This
/// is intentionally *not* the temp-then-rename path used for whole canonical
/// artifacts: the trace grows incrementally as a run proceeds, and each line is
/// independently valid, so a crash truncates only the last (partial) line rather
/// than corrupting an otherwise-complete file.
#[derive(Debug)]
pub struct TraceWriter {
    path: std::path::PathBuf,
    file: std::fs::File,
}

impl TraceWriter {
    /// Open (creating or truncating) `trace.jsonl` under `layout`'s bundle root
    /// for writing.
    ///
    /// # Errors
    /// Returns [`BundleError::Io`] if the file cannot be created.
    pub fn create(layout: &BundleLayout) -> BundleResult<Self> {
        let path = layout.join(files::TRACE);
        let file = std::fs::File::create(&path)
            .map_err(|e| BundleError::io_source(&path, "creating trace.jsonl", e))?;
        Ok(Self { path, file })
    }

    /// Open `trace.jsonl` under `layout`'s bundle root for appending, creating it
    /// if absent.
    ///
    /// # Errors
    /// Returns [`BundleError::Io`] if the file cannot be opened.
    pub fn append(layout: &BundleLayout) -> BundleResult<Self> {
        let path = layout.join(files::TRACE);
        let file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .map_err(|e| BundleError::io_source(&path, "opening trace.jsonl for append", e))?;
        Ok(Self { path, file })
    }

    /// Append one event as a canonical, newline-terminated JSONL record.
    ///
    /// # Errors
    /// Returns [`BundleError::Serialize`] if the event cannot be canonicalized or
    /// [`BundleError::Io`] if the write fails.
    pub fn write_event(&mut self, event: &TraceEvent) -> BundleResult<()> {
        use std::io::Write as _;
        let mut line = event.to_line()?;
        line.push('\n');
        self.file
            .write_all(line.as_bytes())
            .map_err(|e| BundleError::io_source(&self.path, "appending trace event", e))
    }

    /// Flush buffered trace bytes to the OS.
    ///
    /// # Errors
    /// Returns [`BundleError::Io`] if the flush fails.
    pub fn flush(&mut self) -> BundleResult<()> {
        use std::io::Write as _;
        self.file
            .flush()
            .map_err(|e| BundleError::io_source(&self.path, "flushing trace.jsonl", e))
    }
}

/// Whether a `u32` is zero (serde `skip_serializing_if` predicate).
#[expect(
    clippy::trivially_copy_pass_by_ref,
    reason = "serde skip_serializing_if requires a by-reference predicate"
)]
const fn is_zero(value: &u32) -> bool {
    *value == 0
}

#[cfg(test)]
mod tests {
    use super::{
        AssertionMeasured, AssertionStatus, CacheLookup, CacheOutcome, DispatchCompleted,
        DispatchStatus, Failure, ImplementationSelected, InputRegion, NodeNormalized,
        OutputWritten, PlanParsed, ResourceDemanded, TileCounts, TileScheduled, TraceEvent,
        TraceWriter,
    };
    use crate::evidence::layout::{BundleLayout, files};
    use paintop_ir::Rect;
    use serde_json::json;
    use std::sync::atomic::{AtomicU64, Ordering};

    fn scratch_dir(tag: &str) -> std::path::PathBuf {
        static SEQ: AtomicU64 = AtomicU64::new(0);
        let seq = SEQ.fetch_add(1, Ordering::Relaxed);
        let mut dir = std::env::temp_dir();
        dir.push(format!("paintop-trace-{}-{tag}-{seq}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("scratch dir");
        dir
    }

    /// Every event variant round-trips line -> event -> line and carries its
    /// schema version and discriminant.
    fn sample_events() -> Vec<TraceEvent> {
        vec![
            TraceEvent::PlanParsed(PlanParsed {
                plan_semantic_hash: "blake3:abc".to_owned(),
                node_count: 3,
            }),
            TraceEvent::NodeNormalized(NodeNormalized {
                node: "blur_1".to_owned(),
                op: "filter.gaussian_blur@1".to_owned(),
            }),
            TraceEvent::ResourceDemanded(ResourceDemanded {
                node: "blur_1".to_owned(),
                op: "filter.gaussian_blur@1".to_owned(),
                output_region: Rect::new(160, 96, 704, 448),
                input_regions: vec![InputRegion {
                    port: "image".to_owned(),
                    region: Rect::new(128, 64, 768, 512),
                }],
                halo: 32,
            }),
            TraceEvent::CacheLookup(CacheLookup {
                node: "blur_1".to_owned(),
                op: "filter.gaussian_blur@1".to_owned(),
                outcome: CacheOutcome::Miss,
                key: Some("blake3:dead".to_owned()),
            }),
            TraceEvent::TileScheduled(TileScheduled {
                node: "blur_1".to_owned(),
                op: "filter.gaussian_blur@1".to_owned(),
                tiles: TileCounts {
                    requested: 12,
                    executed: 8,
                    identity: 4,
                },
            }),
            TraceEvent::ImplementationSelected(ImplementationSelected {
                node: "blur_1".to_owned(),
                op: "filter.gaussian_blur@1".to_owned(),
                implementation: "cpu.separable@1".to_owned(),
            }),
            TraceEvent::DispatchCompleted(DispatchCompleted {
                node: "blur_1".to_owned(),
                op: "filter.gaussian_blur@1".to_owned(),
                implementation: "cpu.separable@1".to_owned(),
                status: DispatchStatus::Completed,
                elapsed_ms: Some(1.82),
                alloc_bytes: Some(4_194_304),
            }),
            TraceEvent::AssertionMeasured(AssertionMeasured {
                id: "localized".to_owned(),
                op: "assert.no_change_outside_mask@1".to_owned(),
                status: AssertionStatus::Failed,
            }),
            TraceEvent::OutputWritten(OutputWritten {
                name: "result".to_owned(),
                path: "outputs/result.png".to_owned(),
                content_hash: Some("blake3:beef".to_owned()),
            }),
            TraceEvent::Failure(Failure {
                code: "E_ASSERTION_VIOLATED".to_owned(),
                message: "outside mask changed".to_owned(),
                node: Some("blur_1".to_owned()),
            }),
        ]
    }

    #[test]
    fn every_event_round_trips_through_a_line() {
        for ev in sample_events() {
            let line = ev.to_line().expect("serialize");
            assert!(!line.contains('\n'), "line must be single-line: {line}");
            let back = TraceEvent::from_line(&line).expect("parse");
            assert_eq!(back, ev);
        }
    }

    #[test]
    fn every_line_carries_schema_and_event_discriminant() {
        for ev in sample_events() {
            let value: serde_json::Value =
                serde_json::from_str(&ev.to_line().expect("line")).expect("json");
            let obj = value.as_object().expect("object");
            assert_eq!(
                obj.get("schema"),
                Some(&json!(TraceEvent::SCHEMA)),
                "missing schema version"
            );
            assert_eq!(
                obj.get("event").and_then(serde_json::Value::as_str),
                Some(ev.kind()),
                "discriminant must match kind()"
            );
        }
    }

    #[test]
    fn identity_keys_are_spelled_stably_across_events() {
        // node/op/implementation appear under the same keys regardless of event.
        let sel = TraceEvent::implementation_selected(
            "blur_1",
            "filter.gaussian_blur@1",
            "cpu.separable@1",
        );
        let value: serde_json::Value =
            serde_json::from_str(&sel.to_line().expect("line")).expect("json");
        let obj = value.as_object().expect("object");
        assert_eq!(obj.get("node"), Some(&json!("blur_1")));
        assert_eq!(obj.get("op"), Some(&json!("filter.gaussian_blur@1")));
        assert_eq!(obj.get("implementation"), Some(&json!("cpu.separable@1")));

        assert_eq!(sel.node(), Some("blur_1"));
        assert_eq!(sel.op(), Some("filter.gaussian_blur@1"));
        assert_eq!(sel.implementation(), Some("cpu.separable@1"));
    }

    #[test]
    fn lines_are_canonical_and_byte_stable() {
        let ev = TraceEvent::cache_lookup("n", "op@1", CacheOutcome::Hit);
        let a = ev.to_line().expect("a");
        let b = ev.to_line().expect("b");
        assert_eq!(a, b, "canonical lines must be byte-identical across runs");
        // Canonical form sorts keys: `event` < `node` < `op` < `outcome` < `schema`.
        assert!(
            a.find("\"event\"").unwrap() < a.find("\"schema\"").unwrap(),
            "keys must be sorted: {a}"
        );
    }

    #[test]
    fn wrong_schema_version_is_rejected() {
        let line = r#"{"event":"node_normalized","node":"n","op":"o@1","schema":999}"#;
        let err = TraceEvent::from_line(line).expect_err("must reject");
        assert_eq!(err.code(), super::E_BUNDLE_SERIALIZE);
    }

    #[test]
    fn missing_schema_version_is_rejected() {
        let line = r#"{"event":"node_normalized","node":"n","op":"o@1"}"#;
        assert!(TraceEvent::from_line(line).is_err());
    }

    #[test]
    fn unknown_key_in_payload_is_rejected() {
        let line =
            r#"{"event":"node_normalized","node":"n","op":"o@1","surprise":true,"schema":1}"#;
        assert!(TraceEvent::from_line(line).is_err());
    }

    #[test]
    fn unknown_event_discriminant_is_rejected() {
        let line = r#"{"event":"teleported","schema":1}"#;
        assert!(TraceEvent::from_line(line).is_err());
    }

    #[test]
    fn optional_fields_are_omitted_when_absent() {
        let ev = TraceEvent::cache_lookup("n", "op@1", CacheOutcome::Bypass);
        let value: serde_json::Value =
            serde_json::from_str(&ev.to_line().expect("line")).expect("json");
        assert!(!value.as_object().expect("obj").contains_key("key"));
    }

    #[test]
    fn writer_appends_one_independently_valid_line_per_event() {
        let dir = scratch_dir("writer");
        let layout = BundleLayout::new(&dir);
        let events = sample_events();
        {
            let mut writer = TraceWriter::create(&layout).expect("create");
            for ev in &events {
                writer.write_event(ev).expect("write");
            }
            writer.flush().expect("flush");
        }
        let text = std::fs::read_to_string(layout.join(files::TRACE)).expect("read");
        let parsed: Vec<TraceEvent> = text
            .lines()
            .map(|l| TraceEvent::from_line(l).expect("each line parses independently"))
            .collect();
        assert_eq!(parsed, events);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn writer_append_mode_extends_an_existing_trace() {
        let dir = scratch_dir("append");
        let layout = BundleLayout::new(&dir);
        {
            let mut w = TraceWriter::create(&layout).expect("create");
            w.write_event(&TraceEvent::plan_parsed("blake3:00", 1))
                .expect("write");
            w.flush().expect("flush");
        }
        {
            let mut w = TraceWriter::append(&layout).expect("append");
            w.write_event(&TraceEvent::node_normalized("n", "o@1"))
                .expect("write");
            w.flush().expect("flush");
        }
        let text = std::fs::read_to_string(layout.join(files::TRACE)).expect("read");
        assert_eq!(text.lines().count(), 2);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
