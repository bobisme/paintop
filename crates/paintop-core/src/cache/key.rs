//! The content-addressed **cache key** model (`plan.md` §10.3).
//!
//! A cache key is the identity of a *computation*, not of a file. Two runs that
//! would compute the same producer node's output — same operation at the same
//! semantic version, same canonical parameters, same input *content*, same
//! resource semantics, same seed, same backend semantics — must produce the same
//! key, and any semantically meaningful change must produce a different one. The
//! key is therefore built from exactly the inputs `plan.md` §10.3 enumerates:
//!
//! ```text
//! hash(
//!   op_id + op_semantic_version +
//!   canonical_parameters +
//!   ordered_input_content_hashes +
//!   resource_semantics +
//!   seed +
//!   backend_semantics_version
//! )
//! ```
//!
//! # What is and is not in the key
//!
//! - **In** (a change *must* invalidate): the op id, the op semantic version, the
//!   canonical params, each input's *content hash* (not its path), the resource
//!   semantics of every input, the seed, and the backend semantics version.
//! - **Out** (a change must *not* invalidate): anything provenance-only — a file
//!   path, a wall-clock time, a compiler/runtime build id, a node label. The key
//!   never sees these because they are never assembled into [`CacheKeyInputs`].
//!
//! Per `plan.md` §10.3 and the lint wall, the key is never the hash of a raw
//! `serde_json` document: every component is assembled into a canonical JSON
//! value and hashed through the M0 canonicalization + BLAKE3 path
//! ([`paintop_ir::hash`]), with the [`HashDomain::CacheEntry`] domain label so a
//! cache key can never collide with a plan, resource, or content hash.

use std::fmt;

use paintop_ir::{
    HashDomain, OpId, ResourceDescriptor, SemanticHash, hash_canonical_bytes, hash_value,
};
use serde::de::Error as _;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_json::{Map, Value, json};

use super::content::ContentHash;
use super::error::{CacheError, CacheResult};

/// The backend-semantics version: a monotonically-bumped integer that names the
/// *observable numerical contract* of the reference backend (`plan.md` §10.3
/// `backend_semantics_version`).
///
/// Bump this when a backend change can alter exact output bytes for an unchanged
/// op/params/inputs — it forces a cache miss across the version boundary so a
/// stale entry from an incompatible backend is never reused. It is **not** a
/// build id or compiler version (those are provenance, never hashed); it is a
/// deliberate, declared semantic boundary.
pub const BACKEND_SEMANTICS_VERSION: u32 = 1;

/// One input port's contribution to a cache key: its port name, the *content
/// hash* of the value flowing in, and that value's resource semantics.
///
/// The content hash — never a path — is the input's identity (`plan.md` §10.3:
/// "Do not hash a path as the input identity; hash content and relevant
/// metadata."). The resource semantics travel alongside so two inputs that share
/// bytes but differ in declared meaning (e.g. linear vs. sRGB) still key apart.
#[derive(Debug, Clone, PartialEq)]
pub struct InputContribution {
    /// The input port name (e.g. `"image"`, `"mask"`).
    pub port: String,
    /// The `blake3:…` content hash of the value on that port.
    pub content_hash: ContentHash,
    /// The resource semantics (descriptor) of the value on that port.
    pub semantics: ResourceDescriptor,
}

impl InputContribution {
    /// Build a contribution for `port` from a content hash and descriptor.
    #[must_use]
    pub fn new(
        port: impl Into<String>,
        content_hash: ContentHash,
        semantics: ResourceDescriptor,
    ) -> Self {
        Self {
            port: port.into(),
            content_hash,
            semantics,
        }
    }
}

/// The full set of inputs that determine a producer node's cache key
/// (`plan.md` §10.3).
///
/// Assemble one of these for a node and hand it to [`CacheKey::compute`]. The
/// struct deliberately contains *only* semantic identity — no path, clock, or
/// build id — so two equal `CacheKeyInputs` always hash to the same key and any
/// semantic change flips it.
#[derive(Debug, Clone, PartialEq)]
pub struct CacheKeyInputs {
    /// The versioned operation id (its major version is part of the id).
    pub op_id: OpId,
    /// The operation's *semantic* version — the manifest's `impl_version`. A bump
    /// here marks an output-affecting change to the op's contract and invalidates
    /// every prior entry for the op.
    pub op_semantic_version: u32,
    /// The node's resolved parameters, as the canonical JSON object the plan
    /// carried (object-key order is irrelevant — canonicalization collapses it).
    pub params: Map<String, Value>,
    /// Each input port's content hash + semantics, *ordered by port name* for a
    /// stable, deterministic key regardless of insertion order.
    pub inputs: Vec<InputContribution>,
    /// The deterministic seed in scope for this node (`0` when the op is not
    /// seeded). Part of the key so two seeds never share a cached result.
    pub seed: u64,
    /// The backend-semantics version in force (see [`BACKEND_SEMANTICS_VERSION`]).
    pub backend_semantics_version: u32,
}

impl CacheKeyInputs {
    /// Assemble cache-key inputs for an op, with the current backend semantics
    /// version and an empty (unseeded) seed.
    ///
    /// `inputs` may be supplied in any order; [`CacheKey::compute`] sorts them by
    /// port name before hashing, so the key is order-independent.
    #[must_use]
    pub const fn new(
        op_id: OpId,
        op_semantic_version: u32,
        params: Map<String, Value>,
        inputs: Vec<InputContribution>,
    ) -> Self {
        Self {
            op_id,
            op_semantic_version,
            params,
            inputs,
            seed: 0,
            backend_semantics_version: BACKEND_SEMANTICS_VERSION,
        }
    }

    /// Set the deterministic seed for this node (builder style).
    #[must_use]
    pub const fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Assemble the canonical JSON value the key is hashed from.
    ///
    /// Every component is named under a stable key so the framing is unambiguous;
    /// inputs are emitted as a list *sorted by port name* so insertion order can
    /// never change the key. Resource semantics are serialized through serde,
    /// which yields the same canonical bytes for equal descriptors.
    ///
    /// # Errors
    /// Returns [`CacheError::Serialize`] if an input's resource descriptor cannot
    /// be serialized to JSON.
    pub(crate) fn to_canonical_value(&self) -> CacheResult<Value> {
        let mut inputs: Vec<&InputContribution> = self.inputs.iter().collect();
        inputs.sort_by(|a, b| a.port.cmp(&b.port));

        let mut input_values = Vec::with_capacity(inputs.len());
        for input in inputs {
            let semantics =
                serde_json::to_value(input.semantics).map_err(|e| CacheError::Serialize {
                    key: format!("input `{}`", input.port),
                    detail: e.to_string(),
                })?;
            input_values.push(json!({
                "port": input.port,
                "content_hash": input.content_hash.to_string(),
                "semantics": semantics,
            }));
        }

        Ok(json!({
            "op_id": self.op_id.to_string(),
            "op_semantic_version": self.op_semantic_version,
            "params": Value::Object(self.params.clone()),
            "inputs": input_values,
            "seed": self.seed,
            "backend_semantics_version": self.backend_semantics_version,
        }))
    }
}

/// A content-addressed cache key: the BLAKE3 identity of a producer computation
/// (`plan.md` §10.3).
///
/// Constructed by [`CacheKey::compute`] over a [`CacheKeyInputs`]; its serialized
/// form is the underlying [`SemanticHash`]'s `blake3:<hex>` string, carrying the
/// [`HashDomain::CacheEntry`] domain separation so it can never alias a plan,
/// resource, or content hash even on identical bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CacheKey(SemanticHash);

impl PartialOrd for CacheKey {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for CacheKey {
    /// Order by the raw digest bytes, giving a total order so a [`CacheKey`] can
    /// key an ordered map deterministically.
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.as_bytes().cmp(other.0.as_bytes())
    }
}

impl Serialize for CacheKey {
    /// Serialize as the underlying `blake3:<hex>` string.
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(&self.0.to_string())
    }
}

impl<'de> Deserialize<'de> for CacheKey {
    /// Deserialize from a `blake3:<hex>` string, rejecting any malformed id.
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let text = String::deserialize(deserializer)?;
        SemanticHash::parse(&text)
            .map(Self)
            .map_err(D::Error::custom)
    }
}

impl CacheKey {
    /// Compute the cache key for a producer node from its [`CacheKeyInputs`].
    ///
    /// The inputs are assembled into one canonical JSON value and hashed through
    /// the M0 canonical-bytes + BLAKE3 path under [`HashDomain::CacheEntry`]; the
    /// result depends only on the semantic inputs, never on a path, clock, or
    /// build id.
    ///
    /// # Errors
    /// Returns [`CacheError::Serialize`] if an input descriptor cannot be
    /// serialized, or if the assembled value carries a non-finite float (a
    /// canonicalization failure surfaced from [`hash_value`]).
    pub fn compute(inputs: &CacheKeyInputs) -> CacheResult<Self> {
        let value = inputs.to_canonical_value()?;
        let hash =
            hash_value(HashDomain::CacheEntry, &value).map_err(|e| CacheError::Serialize {
                key: inputs.op_id.to_string(),
                detail: e.to_string(),
            })?;
        Ok(Self(hash))
    }

    /// Derive a per-output-port key from this node key.
    ///
    /// A node may produce several output ports; each must cache under a distinct
    /// key so two ports never alias. This re-hashes the node key together with the
    /// (length-framed) port name under [`HashDomain::CacheEntry`], so the derived
    /// key inherits the node's full semantic identity and adds only the port.
    #[must_use]
    pub fn for_output(&self, port: &str) -> Self {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(self.0.as_bytes());
        let port_bytes = port.as_bytes();
        bytes.extend_from_slice(&(port_bytes.len() as u64).to_le_bytes());
        bytes.extend_from_slice(port_bytes);
        Self(hash_canonical_bytes(HashDomain::CacheEntry, &bytes))
    }

    /// The underlying [`SemanticHash`].
    #[must_use]
    pub const fn hash(&self) -> SemanticHash {
        self.0
    }

    /// Parse a serialized `blake3:<hex>` cache key.
    ///
    /// # Errors
    /// Returns [`CacheError::Corrupt`] if `text` is not a valid `blake3:<hex>` id.
    pub fn parse(text: &str) -> CacheResult<Self> {
        SemanticHash::parse(text)
            .map(Self)
            .map_err(|e| CacheError::corrupt(text, e.to_string()))
    }
}

impl fmt::Display for CacheKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

#[cfg(test)]
mod tests {
    use super::{BACKEND_SEMANTICS_VERSION, CacheKey, CacheKeyInputs, InputContribution};
    use crate::cache::content::content_hash_descriptor;
    use paintop_ir::{
        AlphaRepresentation, ChannelLayout, ColorEncoding, ColorRange, CoordinateConvention,
        Extent, ImageDescriptor, OpId, ResourceDescriptor, ScalarType, SemanticRole,
    };
    use serde_json::{Map, Value, json};

    fn image(color: ColorEncoding) -> ResourceDescriptor {
        ResourceDescriptor::Image(ImageDescriptor {
            extent: Extent::new(8, 8),
            layout: ChannelLayout::Rgba,
            scalar: ScalarType::F32,
            color,
            range: ColorRange::SceneReferred,
            alpha: AlphaRepresentation::Premultiplied,
            coordinates: CoordinateConvention::PixelCenterUpperLeft,
            semantic: SemanticRole::Color,
        })
    }

    fn op() -> OpId {
        "filter.gaussian_blur@1".parse().expect("op id")
    }

    fn params(sigma: f64) -> Map<String, Value> {
        json!({ "sigma": sigma })
            .as_object()
            .expect("object")
            .clone()
    }

    fn input(port: &str, color: ColorEncoding) -> InputContribution {
        let desc = image(color);
        InputContribution::new(port, content_hash_descriptor(&desc, &[1.0, 2.0, 3.0]), desc)
    }

    fn base() -> CacheKeyInputs {
        CacheKeyInputs::new(
            op(),
            1,
            params(8.0),
            vec![input("image", ColorEncoding::LinearSrgb)],
        )
    }

    #[test]
    fn identical_inputs_yield_identical_keys() {
        let a = CacheKey::compute(&base()).expect("a");
        let b = CacheKey::compute(&base()).expect("b");
        assert_eq!(a, b);
        assert!(a.to_string().starts_with("blake3:"));
    }

    #[test]
    fn input_order_does_not_change_the_key() {
        let mut a = base();
        a.inputs = vec![
            input("image", ColorEncoding::LinearSrgb),
            input("mask", ColorEncoding::LinearSrgb),
        ];
        let mut b = base();
        b.inputs = vec![
            input("mask", ColorEncoding::LinearSrgb),
            input("image", ColorEncoding::LinearSrgb),
        ];
        assert_eq!(
            CacheKey::compute(&a).expect("a"),
            CacheKey::compute(&b).expect("b"),
            "port order must not affect the key"
        );
    }

    #[test]
    fn param_key_order_does_not_change_the_key() {
        let mut a = base();
        a.params = json!({ "a": 1, "b": 2 }).as_object().expect("o").clone();
        let mut b = base();
        b.params = json!({ "b": 2, "a": 1 }).as_object().expect("o").clone();
        assert_eq!(
            CacheKey::compute(&a).expect("a"),
            CacheKey::compute(&b).expect("b"),
        );
    }

    // --- key-sensitivity matrix: every semantic component flips the key ---

    #[test]
    fn changing_op_semantic_version_flips_the_key() {
        let base_key = CacheKey::compute(&base()).expect("base");
        let mut bumped = base();
        bumped.op_semantic_version = 2;
        assert_ne!(base_key, CacheKey::compute(&bumped).expect("bumped"));
    }

    #[test]
    fn changing_a_param_flips_the_key() {
        let base_key = CacheKey::compute(&base()).expect("base");
        let mut changed = base();
        changed.params = params(8.5);
        assert_ne!(base_key, CacheKey::compute(&changed).expect("changed"));
    }

    #[test]
    fn changing_input_content_flips_the_key() {
        let base_key = CacheKey::compute(&base()).expect("base");
        let mut changed = base();
        let desc = image(ColorEncoding::LinearSrgb);
        changed.inputs = vec![InputContribution::new(
            "image",
            content_hash_descriptor(&desc, &[9.0, 9.0, 9.0]),
            desc,
        )];
        assert_ne!(base_key, CacheKey::compute(&changed).expect("changed"));
    }

    #[test]
    fn changing_input_resource_semantics_flips_the_key() {
        let base_key = CacheKey::compute(&base()).expect("base");
        let mut changed = base();
        // Same content bytes, different declared color encoding -> different key.
        changed.inputs = vec![input("image", ColorEncoding::Srgb)];
        assert_ne!(base_key, CacheKey::compute(&changed).expect("changed"));
    }

    #[test]
    fn changing_seed_flips_the_key() {
        let base_key = CacheKey::compute(&base()).expect("base");
        let seeded = base().with_seed(42);
        assert_ne!(base_key, CacheKey::compute(&seeded).expect("seeded"));
    }

    #[test]
    fn changing_backend_semantics_version_flips_the_key() {
        let base_key = CacheKey::compute(&base()).expect("base");
        let mut changed = base();
        changed.backend_semantics_version = BACKEND_SEMANTICS_VERSION + 1;
        assert_ne!(base_key, CacheKey::compute(&changed).expect("changed"));
    }

    #[test]
    fn changing_op_id_flips_the_key() {
        let base_key = CacheKey::compute(&base()).expect("base");
        let mut changed = base();
        changed.op_id = "filter.box_blur@1".parse().expect("op");
        assert_ne!(base_key, CacheKey::compute(&changed).expect("changed"));
    }

    #[test]
    fn key_round_trips_through_parse() {
        let key = CacheKey::compute(&base()).expect("key");
        let text = key.to_string();
        assert_eq!(CacheKey::parse(&text).expect("parse"), key);
    }

    #[test]
    fn key_uses_the_cache_entry_domain() {
        // A cache key over bytes must not equal a plan hash over the same value:
        // the CacheEntry domain label separates them. We prove the key carries the
        // domain by confirming it differs from a Plan-domain hash of the same JSON.
        let inputs = base();
        let key = CacheKey::compute(&inputs).expect("key");
        let value = inputs.to_canonical_value().expect("value");
        let plan_hash =
            paintop_ir::hash_value(paintop_ir::HashDomain::Plan, &value).expect("plan hash");
        assert_ne!(key.hash(), plan_hash);
    }
}
