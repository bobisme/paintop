//! Compute-pipeline cache keyed by normalized fused expression + format
//! (`plan.md` §12.3: "compile/cache pipelines by normalized fused expression and
//! format"; bn-2vi).
//!
//! Compiling a GPU compute pipeline (shader module + pipeline layout) is expensive
//! and **purely a function of the work's shape**: the normalized chain of fused
//! pointwise stages and the resource format they operate on. Two nodes that fuse to
//! the *same* normalized expression on the *same* format can share one compiled
//! pipeline; two that differ in any semantic way must not.
//!
//! This module owns:
//! * [`FusedExpr`] — the normalized description of a fused pointwise op chain (an
//!   ordered list of `(op, canonical params)` [`FusedStage`]s);
//! * [`ResourceFormat`] — the texel format (channel count + scalar) the chain runs
//!   on, the other half of the key;
//! * [`PipelineKey`] — the content-addressed `blake3:…` key derived from the two,
//!   through the same canonical-bytes + BLAKE3 path the cache layer uses (so it is
//!   deterministic and never hashes raw `serde_json`);
//! * [`PipelineCache`] — a compile-on-miss / reuse-on-hit cache, generic over the
//!   compiled artifact so the **key + reuse logic is unit-testable GPU-less** while
//!   a live run caches real [`wgpu::ComputePipeline`]s.

use std::collections::BTreeMap;
use std::sync::Arc;

use paintop_ir::{HashDomain, OpId, SemanticHash, hash_value};
use serde_json::{Value, json};

/// One stage of a fused pointwise chain: an operation and its canonical params.
///
/// The params are the resolved canonical JSON the plan carried for the node;
/// object-key order is irrelevant because the key derivation canonicalizes before
/// hashing. Two stages are semantically equal iff their op id **and** canonical
/// params match.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FusedStage {
    /// The operation fused at this stage (e.g. `color.adjust@1`).
    pub op: OpId,
    /// The stage's resolved parameters as canonical JSON.
    pub params: Value,
}

impl FusedStage {
    /// A fused stage for `op` with `params`.
    #[must_use]
    pub const fn new(op: OpId, params: Value) -> Self {
        Self { op, params }
    }

    /// The stage's contribution to the key's canonical value: a stable object
    /// `{op, params}`. Object-key order is normalized by canonicalization.
    fn to_key_value(&self) -> Value {
        json!({ "op": self.op.to_string(), "params": self.params })
    }
}

/// The normalized fused pointwise expression a pipeline implements.
///
/// An **ordered** list of [`FusedStage`]s: order is semantic (composition is not
/// commutative), so the key preserves it. Built incrementally with
/// [`push`](Self::push) / [`with`](Self::with) as the fusion pass walks a chain.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct FusedExpr {
    stages: Vec<FusedStage>,
}

impl FusedExpr {
    /// An empty fused expression (identity).
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Append a stage (mutating).
    pub fn push(&mut self, stage: FusedStage) {
        self.stages.push(stage);
    }

    /// Append a stage (builder style).
    #[must_use]
    pub fn with(mut self, stage: FusedStage) -> Self {
        self.stages.push(stage);
        self
    }

    /// The fused stages, in order.
    #[must_use]
    pub fn stages(&self) -> &[FusedStage] {
        &self.stages
    }

    /// Whether the expression has no stages.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.stages.is_empty()
    }

    /// The canonical value the key hashes: the ordered list of stage values.
    fn to_key_value(&self) -> Value {
        Value::Array(self.stages.iter().map(FusedStage::to_key_value).collect())
    }
}

/// The texel format a fused pipeline operates on — the other half of the key.
///
/// A pipeline compiled for `Rgba`/`F32` is not interchangeable with one for
/// `R`/`F32`: the binding layout and shader differ. Captured as a small, stable
/// value so two otherwise-identical expressions on different formats key apart.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResourceFormat {
    /// The number of components per texel (1–4).
    pub channels: u32,
    /// The scalar type of each component (currently always `f32`).
    pub scalar: ScalarFormat,
}

impl ResourceFormat {
    /// An `f32` format with `channels` components.
    #[must_use]
    pub const fn f32(channels: u32) -> Self {
        Self {
            channels,
            scalar: ScalarFormat::F32,
        }
    }

    /// The stable string form mixed into the key (e.g. `"rgba32f"`-style tag).
    fn tag(self) -> String {
        format!("{}x{}", self.channels, self.scalar.tag())
    }
}

/// The scalar component type of a [`ResourceFormat`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ScalarFormat {
    /// 32-bit float.
    F32,
}

impl ScalarFormat {
    const fn tag(self) -> &'static str {
        match self {
            Self::F32 => "f32",
        }
    }
}

/// The content-addressed key a pipeline is cached under.
///
/// Derived from the normalized [`FusedExpr`] **and** the [`ResourceFormat`] through
/// the canonical-bytes + BLAKE3 path (`HashDomain::CacheEntry`), so:
/// * equal `(expr, format)` pairs yield the same key (cache hit / reuse), and
/// * any semantic difference — stage order, op id, a param value, the channel
///   count, the scalar — yields a different key (no collision).
///
/// The serialized form is the underlying `blake3:<hex>` string.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PipelineKey(String);

impl PipelineKey {
    /// Derive the key for a fused expression on a resource format.
    ///
    /// # Errors
    /// Returns the canonicalization [`Error`](paintop_ir::Error) only if a param
    /// value carries a non-finite float; params that came through the plan parser
    /// cannot hit this.
    pub fn derive(expr: &FusedExpr, format: ResourceFormat) -> Result<Self, paintop_ir::Error> {
        // One canonical document binds the expression to its format so the two
        // halves can never be confused (an expr on format A vs the same expr on
        // format B always key apart).
        let value = json!({
            "fused": expr.to_key_value(),
            "format": format.tag(),
        });
        let hash: SemanticHash = hash_value(HashDomain::CacheEntry, &value)?;
        Ok(Self(hash.to_string()))
    }

    /// The `blake3:<hex>` string form of the key.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for PipelineKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// The outcome of a [`PipelineCache::get_or_compile`] lookup.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheOutcome {
    /// The key was already present; the cached pipeline was reused.
    Hit,
    /// The key was absent; the pipeline was compiled and inserted.
    Miss,
}

impl CacheOutcome {
    /// Whether this outcome reused a cached pipeline.
    #[must_use]
    pub const fn is_hit(self) -> bool {
        matches!(self, Self::Hit)
    }
}

/// A compile-on-miss, reuse-on-hit pipeline cache.
///
/// Generic over the compiled artifact `P` so the key + reuse logic is exercised
/// GPU-less in tests (with a cheap dummy `P`), while a live run instantiates
/// `PipelineCache<wgpu::ComputePipeline>`. A hit returns the *same* `Arc<P>` a
/// prior compile produced — never recompiling identical work — which is exactly the
/// reuse property the acceptance test asserts.
#[derive(Debug)]
pub struct PipelineCache<P> {
    entries: BTreeMap<PipelineKey, Arc<P>>,
    hits: u64,
    misses: u64,
}

impl<P> Default for PipelineCache<P> {
    fn default() -> Self {
        Self {
            entries: BTreeMap::new(),
            hits: 0,
            misses: 0,
        }
    }
}

impl<P> PipelineCache<P> {
    /// An empty cache.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Fetch the pipeline for `key`, compiling it with `compile` on a miss.
    ///
    /// On a **hit** the cached `Arc<P>` is cloned and returned with
    /// [`CacheOutcome::Hit`] — `compile` is *not* called. On a **miss** `compile`
    /// runs, its result is inserted, and the inserted `Arc<P>` is returned with
    /// [`CacheOutcome::Miss`]. Determinism: the same key always resolves to the same
    /// stored artifact for the cache's lifetime.
    ///
    /// # Errors
    /// Propagates any error from `compile` (e.g. a shader compile failure); a failed
    /// compile inserts nothing, so a later retry can succeed.
    pub fn get_or_compile<F, E>(
        &mut self,
        key: PipelineKey,
        compile: F,
    ) -> Result<(Arc<P>, CacheOutcome), E>
    where
        F: FnOnce() -> Result<P, E>,
    {
        if let Some(existing) = self.entries.get(&key) {
            self.hits += 1;
            return Ok((Arc::clone(existing), CacheOutcome::Hit));
        }
        let pipeline = Arc::new(compile()?);
        self.entries.insert(key, Arc::clone(&pipeline));
        self.misses += 1;
        Ok((pipeline, CacheOutcome::Miss))
    }

    /// Whether a pipeline is already cached for `key`.
    #[must_use]
    pub fn contains(&self, key: &PipelineKey) -> bool {
        self.entries.contains_key(key)
    }

    /// The number of distinct cached pipelines.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the cache holds no pipelines.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// The number of cache hits (reuses) served so far.
    #[must_use]
    pub const fn hits(&self) -> u64 {
        self.hits
    }

    /// The number of cache misses (compiles) served so far.
    #[must_use]
    pub const fn misses(&self) -> u64 {
        self.misses
    }
}

#[cfg(test)]
mod tests {
    use super::{CacheOutcome, FusedExpr, FusedStage, PipelineCache, PipelineKey, ResourceFormat};
    use serde_json::json;

    fn stage(op: &str, params: serde_json::Value) -> FusedStage {
        FusedStage::new(op.parse().expect("op id"), params)
    }

    fn expr_a() -> FusedExpr {
        FusedExpr::new()
            .with(stage("color.adjust@1", json!({ "gain": 1.5, "bias": 0.0 })))
            .with(stage("alpha.premultiply@1", json!({})))
    }

    const RGBA_F32: ResourceFormat = ResourceFormat::f32(4);

    #[test]
    fn identical_expr_and_format_yield_identical_keys() {
        let k1 = PipelineKey::derive(&expr_a(), RGBA_F32).expect("key");
        let k2 = PipelineKey::derive(&expr_a(), RGBA_F32).expect("key");
        assert_eq!(k1, k2);
        assert!(k1.as_str().starts_with("blake3:"));
    }

    #[test]
    fn param_order_does_not_change_the_key() {
        // Canonicalization collapses object-key order: same semantics, same key.
        let reordered = FusedExpr::new()
            .with(stage("color.adjust@1", json!({ "bias": 0.0, "gain": 1.5 })))
            .with(stage("alpha.premultiply@1", json!({})));
        let k1 = PipelineKey::derive(&expr_a(), RGBA_F32).expect("key");
        let k2 = PipelineKey::derive(&reordered, RGBA_F32).expect("key");
        assert_eq!(k1, k2);
    }

    #[test]
    fn different_format_keys_apart() {
        let k_rgba = PipelineKey::derive(&expr_a(), ResourceFormat::f32(4)).expect("key");
        let k_r = PipelineKey::derive(&expr_a(), ResourceFormat::f32(1)).expect("key");
        assert_ne!(k_rgba, k_r, "channel count is part of the key");
    }

    #[test]
    fn different_param_value_keys_apart() {
        let other = FusedExpr::new()
            .with(stage("color.adjust@1", json!({ "gain": 2.0, "bias": 0.0 })))
            .with(stage("alpha.premultiply@1", json!({})));
        let k1 = PipelineKey::derive(&expr_a(), RGBA_F32).expect("key");
        let k2 = PipelineKey::derive(&other, RGBA_F32).expect("key");
        assert_ne!(k1, k2);
    }

    #[test]
    fn stage_order_keys_apart() {
        // Composition is not commutative: reversing the chain is a different
        // pipeline.
        let reversed = FusedExpr::new()
            .with(stage("alpha.premultiply@1", json!({})))
            .with(stage("color.adjust@1", json!({ "gain": 1.5, "bias": 0.0 })));
        let k1 = PipelineKey::derive(&expr_a(), RGBA_F32).expect("key");
        let k2 = PipelineKey::derive(&reversed, RGBA_F32).expect("key");
        assert_ne!(k1, k2);
    }

    #[test]
    fn different_op_keys_apart() {
        let other = FusedExpr::new()
            .with(stage("color.invert@1", json!({ "gain": 1.5, "bias": 0.0 })))
            .with(stage("alpha.premultiply@1", json!({})));
        let k1 = PipelineKey::derive(&expr_a(), RGBA_F32).expect("key");
        let k2 = PipelineKey::derive(&other, RGBA_F32).expect("key");
        assert_ne!(k1, k2);
    }

    /// A dummy compiled "pipeline" carrying a unique id so a test can prove a hit
    /// returns the *same* artifact a prior compile produced.
    #[derive(Debug, PartialEq, Eq)]
    struct DummyPipeline(u32);

    #[test]
    fn same_key_reuses_the_same_pipeline() {
        let mut cache = PipelineCache::<DummyPipeline>::new();
        let key = PipelineKey::derive(&expr_a(), RGBA_F32).expect("key");

        let mut compiles = 0_u32;
        let mut compile_once = || {
            compiles += 1;
            DummyPipeline(compiles)
        };

        // First lookup: miss -> compiles pipeline #1.
        let (p1, o1) = cache
            .get_or_compile::<_, std::convert::Infallible>(key.clone(), || Ok(compile_once()))
            .expect("compile");
        assert_eq!(o1, CacheOutcome::Miss);
        assert_eq!(p1.0, 1);

        // Second lookup with the SAME key: hit -> reuses #1, no recompile.
        let (p2, o2) = cache
            .get_or_compile::<_, std::convert::Infallible>(key, || {
                panic!("must not recompile on a cache hit")
            })
            .expect("hit");
        assert_eq!(o2, CacheOutcome::Hit);
        assert!(std::sync::Arc::ptr_eq(&p1, &p2), "same cached Arc reused");
        assert_eq!(cache.len(), 1);
        assert_eq!(cache.hits(), 1);
        assert_eq!(cache.misses(), 1);
    }

    #[test]
    fn different_keys_compile_separately() {
        let mut cache = PipelineCache::<DummyPipeline>::new();
        let k_rgba = PipelineKey::derive(&expr_a(), ResourceFormat::f32(4)).expect("key");
        let k_r = PipelineKey::derive(&expr_a(), ResourceFormat::f32(1)).expect("key");

        let (_, o1) = cache
            .get_or_compile::<_, std::convert::Infallible>(k_rgba, || Ok(DummyPipeline(1)))
            .expect("c1");
        let (_, o2) = cache
            .get_or_compile::<_, std::convert::Infallible>(k_r, || Ok(DummyPipeline(2)))
            .expect("c2");
        assert_eq!(o1, CacheOutcome::Miss);
        assert_eq!(o2, CacheOutcome::Miss);
        assert_eq!(cache.len(), 2, "distinct keys do not collide");
        assert_eq!(cache.misses(), 2);
    }

    #[test]
    fn failed_compile_inserts_nothing_and_can_retry() {
        let mut cache = PipelineCache::<DummyPipeline>::new();
        let key = PipelineKey::derive(&expr_a(), RGBA_F32).expect("key");

        let first = cache.get_or_compile::<_, &str>(key.clone(), || Err("shader error"));
        assert!(first.is_err());
        assert_eq!(cache.len(), 0, "a failed compile caches nothing");

        let (p, outcome) = cache
            .get_or_compile::<_, &str>(key, || Ok(DummyPipeline(7)))
            .expect("retry");
        assert_eq!(outcome, CacheOutcome::Miss);
        assert_eq!(p.0, 7);
        assert_eq!(cache.len(), 1);
    }
}
