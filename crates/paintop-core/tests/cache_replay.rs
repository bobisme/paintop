//! Replay / zero-recompute verification for the content-addressed cache (bn-3vr).
//!
//! These exercise the cache-aware executor ([`execute_cached`]) end to end:
//!
//! - a fully-cached replay of an unchanged plan executes **zero** pure producer
//!   nodes — proved both by the per-node compute counters and by the trace
//!   carrying `cache = hit` for every node;
//! - a cache hit returns output **equal** to the uncached run (caching never
//!   changes the result);
//! - changing a semantic param recomputes only the affected node (and its
//!   downstream consumers, whose input content changed), not the unaffected
//!   prefix;
//! - a stale / incompatible store entry is **rejected**, not silently reused.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use paintop_core::cache::{CacheStore, CacheValidation, execute_cached};
use paintop_core::evidence::trace::{CacheOutcome, TraceEvent};
use paintop_core::executor::{
    ImplRegistry, InputValues, OpImplementation, OutputValues, ResourceValue, execute,
};
use paintop_ir::{
    AlphaRepresentation, ChannelLayout, ColorEncoding, ColorRange, CoordinateConvention,
    DeterminismTier, Error, ErrorClass, Extent, ImageDescriptor, InputSpec, OperationManifest,
    OperationRegistry, OutputSpec, Plan, ResourceDescriptor, ResourceKind, RoiCategory, RoiPolicy,
    ScalarType, SemanticRole, TestMetadata, parse_plan, resolve_plan,
};
use serde_json::Value;

const EXTENT: Extent = Extent::new(2, 2);
const CHANNELS: u32 = 4;

const fn linear_premul() -> ResourceDescriptor {
    ResourceDescriptor::Image(ImageDescriptor {
        extent: EXTENT,
        layout: ChannelLayout::Rgba,
        scalar: ScalarType::F32,
        color: ColorEncoding::LinearSrgb,
        range: ColorRange::SceneReferred,
        alpha: AlphaRepresentation::Premultiplied,
        coordinates: CoordinateConvention::PixelCenterUpperLeft,
        semantic: SemanticRole::Color,
    })
}

fn value(fill: f32) -> ResourceValue {
    let len = (EXTENT.width * EXTENT.height * CHANNELS) as usize;
    ResourceValue::new(linear_premul(), CHANNELS, vec![fill; len]).expect("buffer")
}

// ---- Manifests -------------------------------------------------------------

fn op(id: &str, inputs: &[&str], outputs: &[&str]) -> OperationManifest {
    OperationManifest {
        id: id.parse().expect("op id"),
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
        implementations: vec!["cpu.reference@1".parse().expect("impl")],
        test: TestMetadata::default(),
    }
}

fn registry() -> OperationRegistry {
    OperationRegistry::from_manifests([
        op("source.create@1", &[], &["image"]),
        op("filter.bias@1", &["image"], &["image"]),
    ])
    .expect("registry")
}

// ---- Counting implementations ---------------------------------------------

/// A source op that emits a fixed fill and counts how many times it computes.
struct CountingSource {
    fill: f32,
    calls: Arc<AtomicUsize>,
}
impl OpImplementation for CountingSource {
    fn compute(&self, _i: &InputValues, _p: &Value) -> Result<OutputValues, Error> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        let mut out = OutputValues::new();
        out.insert("image".to_owned(), value(self.fill));
        Ok(out)
    }
}

/// A bias op that adds its `bias` param to every sample and counts its computes.
struct CountingBias {
    calls: Arc<AtomicUsize>,
}
impl OpImplementation for CountingBias {
    fn compute(&self, inputs: &InputValues, params: &Value) -> Result<OutputValues, Error> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        let image = inputs.get("image").cloned().ok_or_else(|| {
            Error::new(ErrorClass::Execution, "E_MISSING_INPUT", "bias needs image")
        })?;
        #[expect(
            clippy::cast_possible_truncation,
            reason = "test op: a plan bias param is a small f64 cast to the f32 \
                      sample type"
        )]
        let bias = params
            .get("bias")
            .and_then(Value::as_f64)
            .map_or(0.0, |b| b as f32);
        let descriptor = *image.descriptor();
        let channels = image.channels();
        let biased: Vec<f32> = image.samples().iter().map(|s| s + bias).collect();
        let out_value = ResourceValue::new(descriptor, channels, biased).map_err(|n| {
            Error::new(
                ErrorClass::Execution,
                "E_BAD_BUFFER",
                format!("bias produced {n} samples"),
            )
        })?;
        let mut out = OutputValues::new();
        out.insert("image".to_owned(), out_value);
        Ok(out)
    }
}

/// Build a fresh implementation registry, returning the per-op call counters.
fn implementations() -> (ImplRegistry, Arc<AtomicUsize>, Arc<AtomicUsize>) {
    let source_calls = Arc::new(AtomicUsize::new(0));
    let bias_calls = Arc::new(AtomicUsize::new(0));
    let mut r = ImplRegistry::new();
    r.register(
        "source.create@1".parse().expect("op"),
        Box::new(CountingSource {
            fill: 0.25,
            calls: Arc::clone(&source_calls),
        }),
    )
    .expect("register source");
    r.register(
        "filter.bias@1".parse().expect("op"),
        Box::new(CountingBias {
            calls: Arc::clone(&bias_calls),
        }),
    )
    .expect("register bias");
    (r, source_calls, bias_calls)
}

// ---- Plans -----------------------------------------------------------------

/// src -> b1(bias 0.1) -> b2(bias 0.2) -> export. Two producer stages over a
/// source, so a key-change at b1 must ripple into b2.
fn pipeline(bias1: f64, bias2: f64) -> Plan {
    parse_plan(&format!(
        r#"{{
            "paintop": "1.0",
            "inputs": {{}},
            "nodes": [
                {{"id": "src", "op": "source.create@1"}},
                {{"id": "b1", "op": "filter.bias@1", "in": {{"image": "node:src/image"}}, "params": {{"bias": {bias1}}}}},
                {{"id": "b2", "op": "filter.bias@1", "in": {{"image": "node:b1/image"}}, "params": {{"bias": {bias2}}}}}
            ],
            "exports": {{"out": {{"resource": "node:b2/image", "kind": "image", "path": "o.png"}}}}
        }}"#,
    ))
    .expect("plan")
}

const fn no_inputs() -> BTreeMap<String, ResourceValue> {
    BTreeMap::new()
}

// ---- Tests -----------------------------------------------------------------

#[test]
fn cache_hit_output_equals_uncached_output() {
    let plan = pipeline(0.1, 0.2);
    let reg = registry();
    let graph = resolve_plan(&plan, &reg).expect("resolve");

    // Uncached reference output (the existing whole-image executor).
    let (impls, _, _) = implementations();
    let reference = execute(&plan, &graph, &reg, &impls, &no_inputs()).expect("uncached");
    let reference_export = reference.exports()[0].1.clone();

    // First cached run (all misses) populates the store.
    let (impls, _, _) = implementations();
    let mut cache = CacheStore::in_memory();
    let first =
        execute_cached(&plan, &graph, &reg, &impls, &no_inputs(), &mut cache).expect("first");
    assert_eq!(
        first.exports()[0].1,
        reference_export,
        "cached first run must equal the uncached output"
    );

    // Second cached run (all hits) must return byte-identical output.
    let (impls, _, _) = implementations();
    let second =
        execute_cached(&plan, &graph, &reg, &impls, &no_inputs(), &mut cache).expect("second");
    assert_eq!(
        second.exports()[0].1,
        reference_export,
        "cache hit must equal the uncached output exactly"
    );
}

#[test]
fn full_cache_replay_executes_zero_producer_nodes() {
    let plan = pipeline(0.1, 0.2);
    let reg = registry();
    let graph = resolve_plan(&plan, &reg).expect("resolve");
    let mut cache = CacheStore::in_memory();

    // Warm the cache: first run computes every producer.
    let (impls, src_calls, bias_calls) = implementations();
    let first =
        execute_cached(&plan, &graph, &reg, &impls, &no_inputs(), &mut cache).expect("warm");
    assert_eq!(src_calls.load(Ordering::Relaxed), 1, "source computed once");
    assert_eq!(bias_calls.load(Ordering::Relaxed), 2, "both bias nodes ran");
    assert_eq!(first.recomputed().len(), 3, "three nodes recomputed");
    assert!(first.cache_hits().is_empty(), "cold run hits nothing");

    // Replay over the warm cache: a *fresh* set of counters must stay at zero.
    let (impls, src_calls, bias_calls) = implementations();
    let replay =
        execute_cached(&plan, &graph, &reg, &impls, &no_inputs(), &mut cache).expect("replay");
    assert_eq!(
        src_calls.load(Ordering::Relaxed),
        0,
        "zero source recompute on replay"
    );
    assert_eq!(
        bias_calls.load(Ordering::Relaxed),
        0,
        "zero bias recompute on replay"
    );
    assert_eq!(replay.cache_hits().len(), 3, "every node hit the cache");
    assert!(replay.recomputed().is_empty(), "nothing recomputed");

    // Trace-verified: every cache_lookup is a hit and no dispatch_completed fired.
    let lookups: Vec<&TraceEvent> = replay
        .trace()
        .iter()
        .filter(|e| matches!(e, TraceEvent::CacheLookup(_)))
        .collect();
    assert_eq!(lookups.len(), 3, "one lookup per node");
    for ev in &lookups {
        if let TraceEvent::CacheLookup(lookup) = ev {
            assert_eq!(lookup.outcome, CacheOutcome::Hit, "replay lookups are hits");
            assert!(lookup.key.is_some(), "a hit carries its key");
        }
    }
    assert!(
        !replay
            .trace()
            .iter()
            .any(|e| matches!(e, TraceEvent::DispatchCompleted(_))),
        "no node dispatched on a full-cache replay"
    );
}

#[test]
fn changing_a_param_recomputes_only_the_affected_subgraph() {
    let reg = registry();
    let mut cache = CacheStore::in_memory();

    // Warm the cache with the original pipeline.
    let plan = pipeline(0.1, 0.2);
    let graph = resolve_plan(&plan, &reg).expect("resolve");
    let (impls, _, _) = implementations();
    execute_cached(&plan, &graph, &reg, &impls, &no_inputs(), &mut cache).expect("warm");

    // Change b1's bias param. src is unchanged (hit); b1's key flips (miss); b2's
    // input content changes, so b2 also misses and recomputes.
    let changed = pipeline(0.9, 0.2);
    let changed_graph = resolve_plan(&changed, &reg).expect("resolve changed");
    let (impls, src_calls, bias_calls) = implementations();
    let run = execute_cached(
        &changed,
        &changed_graph,
        &reg,
        &impls,
        &no_inputs(),
        &mut cache,
    )
    .expect("changed run");

    assert_eq!(
        src_calls.load(Ordering::Relaxed),
        0,
        "the unchanged source prefix is reused"
    );
    assert_eq!(
        bias_calls.load(Ordering::Relaxed),
        2,
        "the changed node and its downstream consumer recompute"
    );
    assert_eq!(run.cache_hits(), &["src".to_owned()]);
    assert_eq!(run.recomputed(), &["b1".to_owned(), "b2".to_owned()]);
}

#[test]
fn stale_incompatible_entry_is_rejected_not_reused() {
    let plan = pipeline(0.1, 0.2);
    let reg = registry();
    let graph = resolve_plan(&plan, &reg).expect("resolve");
    let mut cache = CacheStore::in_memory();

    // Warm the cache.
    let (impls, _, _) = implementations();
    let warm = execute_cached(&plan, &graph, &reg, &impls, &no_inputs(), &mut cache).expect("warm");

    // Hand-poison the store: overwrite the source node's cached output with an
    // entry whose validation claims an incompatible op semantic version. The
    // replay must reject it (a cache failure) rather than serve a stale value.
    // We recompute the source's per-output key by re-running the key derivation
    // through a second cold cache and reading back which keys it stored.
    let _ = warm; // warm is only needed to populate the (shared) cache above.

    // Build an incompatible value under one of the stored keys by directly
    // walking the in-memory store: replace every entry's validation with a
    // bumped op semantic version, then confirm the replay errors.
    poison_validation(&mut cache);

    let (impls, _, _) = implementations();
    let err = execute_cached(&plan, &graph, &reg, &impls, &no_inputs(), &mut cache)
        .expect_err("a poisoned (incompatible) entry must be rejected");
    assert_eq!(
        err.code(),
        paintop_core::cache::E_CACHE_CORRUPT,
        "incompatible entry is rejected as corrupt, not reused"
    );
}

/// Rewrite every entry in an in-memory store so its recorded validation claims an
/// incompatible op semantic version, simulating a stale/incompatible store.
fn poison_validation(cache: &mut CacheStore) {
    let CacheStore::Memory(map) = cache else {
        unreachable!("test uses the in-memory store");
    };
    for entry in map.values_mut() {
        entry.validation = CacheValidation {
            op_id: entry.validation.op_id.clone(),
            // Bump the op semantic version so the request validation no longer
            // matches: an incompatible-semantics boundary.
            op_semantic_version: entry.validation.op_semantic_version + 1,
            backend_semantics_version: entry.validation.backend_semantics_version,
        };
    }
}
