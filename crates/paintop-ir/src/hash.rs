//! Semantic / content hashing with algorithm-prefixed BLAKE3 ids
//! (`M0_DECISIONS` D "Hashing"; `plan.md` §10.3, §17; `IR_SPEC` §17).
//!
//! Hashing, content-addressed caching, replay, and meaningful diffs all turn the
//! **canonical bytes** of a plan / resource / cache key into a stable identity.
//! This module owns that final step: it takes canonical bytes (the output of
//! [`to_canonical_bytes`]) and produces a stable,
//! self-describing id.
//!
//! # The contract
//!
//! 1. **Algorithm-prefixed ids.** Every serialized hash carries its algorithm as
//!    a prefix so the wire form is self-describing and future-proof: internal
//!    semantic / content / cache hashes render as `blake3:<64 lowercase hex>`
//!    (`M0_DECISIONS` D). The prefix is *part of the identity* — a bare hex
//!    digest is never emitted.
//! 2. **BLAKE3 for everything internal.** All internal semantic-plan, content,
//!    and cache hashing uses BLAKE3 (`plan.md §10.3`). SHA-256 (`sha256:…`) is
//!    reserved for external interop boundaries and is *not* produced here.
//! 3. **Domain separation.** A plan, a resource descriptor, a manifest, and a
//!    cache key that happen to share canonical bytes must still hash to
//!    *different* ids, so a value from one domain can never be confused for, or
//!    collide with, a value from another. Each [`HashDomain`] contributes a
//!    stable, unique label that is mixed into the digest before the payload.
//! 4. **No wall-clock / provenance in semantic identity.** This API hashes only
//!    the canonical bytes it is handed. It never reads the clock, the
//!    environment, a compiler/runtime version, or any other provenance: two runs
//!    over the same semantic bytes always produce the same id (`plan.md §17`:
//!    "A compiler version must not be included in the semantic hash …
//!    Execution provenance separately records compiler/runtime versions.").
//!
//! # Entry points
//!
//! - [`hash_canonical_bytes`] — hash bytes you have *already* canonicalized.
//! - [`hash_value`] — canonicalize a [`serde_json::Value`] (via
//!   [`to_canonical_bytes`]) and then hash it; a
//!   convenience for callers holding a `Value` rather than raw bytes.
//!
//! ```
//! use paintop_ir::hash::{HashDomain, hash_value};
//! use serde_json::json;
//!
//! // The same semantic value always hashes to the same id (no wall-clock).
//! let a = hash_value(HashDomain::Plan, &json!({"b": 1, "a": 2})).unwrap();
//! let b = hash_value(HashDomain::Plan, &json!({"a": 2, "b": 1})).unwrap();
//! assert_eq!(a, b);
//! assert!(a.to_string().starts_with("blake3:"));
//!
//! // Different domains never collide, even on identical bytes.
//! let plan = hash_value(HashDomain::Plan, &json!({"x": 1})).unwrap();
//! let resource = hash_value(HashDomain::Resource, &json!({"x": 1})).unwrap();
//! assert_ne!(plan, resource);
//! ```

use std::fmt;

use serde_json::Value;

use crate::error::{Error, ErrorClass, Result};
use crate::to_canonical_bytes;

/// The algorithm prefix for every internal hash this module produces
/// (`M0_DECISIONS` D). Serialized hashes are `"blake3:" + 64 lowercase hex`.
pub const BLAKE3_PREFIX: &str = "blake3:";

/// Stable code for a hash string that does not match the `blake3:<hex>` shape.
pub const E_INVALID_HASH: &str = "E_INVALID_HASH";

/// The length, in bytes, of a BLAKE3 digest (256 bits).
const DIGEST_LEN: usize = 32;

/// The length, in hex characters, of a serialized BLAKE3 digest.
const DIGEST_HEX_LEN: usize = DIGEST_LEN * 2;

/// A hashing **domain**: the kind of artifact whose canonical bytes are being
/// hashed.
///
/// Domain separation guarantees that two artifacts of *different* kinds never
/// produce the same id even when their canonical bytes are byte-identical: each
/// variant contributes a unique, stable label ([`HashDomain::label`]) that is
/// mixed into the BLAKE3 input ahead of the payload. Adding a domain here only
/// ever introduces a *new* label; the existing labels are frozen, so existing
/// hashes never change.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum HashDomain {
    /// A normalized plan document — the plan **semantic hash** (`plan.md §17`,
    /// `IR_SPEC` §17: `semantic-hash`).
    Plan,
    /// A resource descriptor / decoded-input semantics block (`plan.md §10.3`
    /// `resource_semantics`; `IR_SPEC` §4).
    Resource,
    /// The content of a produced/exported resource — its **content hash**
    /// (`plan.md §10.3` `ordered_input_content_hashes`; `IR_SPEC` §17).
    Content,
    /// An operation manifest's canonical bytes (`IR_SPEC` §15).
    Manifest,
    /// A content-addressed **cache key** built from op id, params, input
    /// content hashes, seed, and backend semantics (`plan.md §10.3`).
    CacheEntry,
}

impl HashDomain {
    /// Every domain, in declaration order. Useful for exhaustive table tests
    /// and for asserting that all labels are pairwise distinct.
    pub const ALL: [Self; 5] = [
        Self::Plan,
        Self::Resource,
        Self::Content,
        Self::Manifest,
        Self::CacheEntry,
    ];

    /// The stable, unique domain-separation label mixed into the digest.
    ///
    /// These strings are part of the on-disk hash identity: **never change an
    /// existing label**, or every hash in that domain silently changes. The
    /// `paintop/` namespace and the trailing `/v1` keep the labels collision-
    /// free against any plausible payload and leave room to revise a single
    /// domain's framing without disturbing the others.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Plan => "paintop/plan/v1",
            Self::Resource => "paintop/resource/v1",
            Self::Content => "paintop/content/v1",
            Self::Manifest => "paintop/manifest/v1",
            Self::CacheEntry => "paintop/cache-entry/v1",
        }
    }
}

/// A self-describing, algorithm-prefixed semantic / content hash id.
///
/// Internally this is a BLAKE3 digest; its [`Display`](fmt::Display) /
/// serialized form is always `blake3:<64 lowercase hex>` so the wire value
/// names its own algorithm (`M0_DECISIONS` D). The type is opaque: construct it
/// with the [`hash_canonical_bytes`] / [`hash_value`] free functions or
/// [`SemanticHash::parse`], and read it back with its `Display`
/// (`blake3:<hex>`) or [`SemanticHash::hex`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SemanticHash {
    /// The raw 32-byte BLAKE3 digest.
    digest: [u8; DIGEST_LEN],
}

impl SemanticHash {
    /// Wrap a raw 32-byte BLAKE3 digest.
    #[must_use]
    const fn from_digest(digest: [u8; DIGEST_LEN]) -> Self {
        Self { digest }
    }

    /// The raw 32-byte BLAKE3 digest behind this hash.
    #[must_use]
    pub const fn as_bytes(&self) -> &[u8; DIGEST_LEN] {
        &self.digest
    }

    /// The bare lowercase-hex digest **without** the `blake3:` prefix.
    ///
    /// Prefer the prefixed `Display` form (`to_string`) for anything that is
    /// serialized, stored, or compared across the wire — the prefix is part of
    /// the identity.
    #[must_use]
    pub fn hex(&self) -> String {
        let mut out = String::with_capacity(DIGEST_HEX_LEN);
        for byte in self.digest {
            // `{:02x}` always yields exactly two lowercase hex digits.
            out.push(char::from(hex_digit(byte >> 4)));
            out.push(char::from(hex_digit(byte & 0x0f)));
        }
        out
    }

    /// Parse a serialized `blake3:<64 lowercase hex>` id back into a
    /// [`SemanticHash`].
    ///
    /// # Errors
    /// Returns a [`parse`](ErrorClass::Parse) / [`E_INVALID_HASH`] error if the
    /// string is missing the `blake3:` prefix, is the wrong length, or contains
    /// a non-hex (or uppercase) digit.
    pub fn parse(text: &str) -> Result<Self> {
        let hex = text.strip_prefix(BLAKE3_PREFIX).ok_or_else(|| {
            Error::new(
                ErrorClass::Parse,
                E_INVALID_HASH,
                format!("hash `{text}` is missing the required `{BLAKE3_PREFIX}` prefix"),
            )
        })?;
        if hex.len() != DIGEST_HEX_LEN {
            return Err(Error::new(
                ErrorClass::Parse,
                E_INVALID_HASH,
                format!(
                    "hash digest must be {DIGEST_HEX_LEN} hex chars, got {}",
                    hex.len()
                ),
            ));
        }
        let mut digest = [0u8; DIGEST_LEN];
        let bytes = hex.as_bytes();
        for (index, slot) in digest.iter_mut().enumerate() {
            // Each output byte consumes two hex chars; the length check above
            // guarantees both indices are in bounds.
            let high = decode_hex_digit(bytes[index * 2])?;
            let low = decode_hex_digit(bytes[index * 2 + 1])?;
            *slot = (high << 4) | low;
        }
        Ok(Self::from_digest(digest))
    }
}

impl fmt::Display for SemanticHash {
    /// Emit the self-describing `blake3:<64 lowercase hex>` form.
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(BLAKE3_PREFIX)?;
        for byte in self.digest {
            write!(formatter, "{byte:02x}")?;
        }
        Ok(())
    }
}

/// Hash already-canonical bytes within a [`HashDomain`], returning a
/// `blake3:<hex>` [`SemanticHash`].
///
/// The domain's [`label`](HashDomain::label) and a length-prefixed framing of
/// it are absorbed *before* the payload so two domains can never alias and a
/// payload can never be confused with a domain label. The result depends only
/// on `domain` and `canonical_bytes` — never on the clock, environment, or any
/// provenance — so it is stable across runs and machines.
///
/// Callers that have a [`serde_json::Value`] rather than canonical bytes should
/// use [`hash_value`], which canonicalizes first.
#[must_use]
pub fn hash_canonical_bytes(domain: HashDomain, canonical_bytes: &[u8]) -> SemanticHash {
    let mut hasher = blake3::Hasher::new();
    // Domain separation: absorb the label with an explicit length prefix so the
    // boundary between label and payload is unambiguous (a label can never run
    // into the payload to forge another domain's input). The length is fixed
    // width and little-endian for a stable framing.
    let label = domain.label().as_bytes();
    let label_len = label.len() as u64;
    hasher.update(&label_len.to_le_bytes());
    hasher.update(label);
    hasher.update(canonical_bytes);
    let digest = *hasher.finalize().as_bytes();
    SemanticHash::from_digest(digest)
}

/// Canonicalize a [`serde_json::Value`] and hash it within `domain`.
///
/// This is [`hash_canonical_bytes`] composed with
/// [`to_canonical_bytes`]: equivalent values (e.g.
/// differing only in object-key order) canonicalize to identical bytes and
/// therefore hash to the same id.
///
/// # Errors
/// Returns the [`parse`](ErrorClass::Parse) error from
/// [`to_canonical_bytes`] if `value` carries a
/// non-finite float; a value that came through
/// [`parse_plan`](crate::plan::parse_plan) cannot hit this case.
pub fn hash_value(domain: HashDomain, value: &Value) -> Result<SemanticHash> {
    let bytes = to_canonical_bytes(value)?;
    Ok(hash_canonical_bytes(domain, &bytes))
}

/// Map a nibble (`0x0..=0xf`) to its lowercase ASCII hex digit.
const fn hex_digit(nibble: u8) -> u8 {
    match nibble {
        0..=9 => b'0' + nibble,
        _ => b'a' + (nibble - 10),
    }
}

/// Decode one lowercase-hex ASCII byte into its `0x0..=0xf` value.
fn decode_hex_digit(byte: u8) -> Result<u8> {
    match byte {
        b'0'..=b'9' => Ok(byte - b'0'),
        b'a'..=b'f' => Ok(byte - b'a' + 10),
        other => Err(Error::new(
            ErrorClass::Parse,
            E_INVALID_HASH,
            format!(
                "invalid hex digit `{}` in hash digest (must be lowercase `0-9a-f`)",
                char::from(other)
            ),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::{
        BLAKE3_PREFIX, DIGEST_HEX_LEN, E_INVALID_HASH, HashDomain, SemanticHash,
        hash_canonical_bytes, hash_value,
    };
    use crate::error::ErrorClass;
    use serde_json::json;

    #[test]
    fn serialized_form_is_algorithm_prefixed() {
        let hash = hash_canonical_bytes(HashDomain::Plan, b"{}");
        let text = hash.to_string();
        assert!(text.starts_with(BLAKE3_PREFIX), "got `{text}`");
        let hex = text.strip_prefix(BLAKE3_PREFIX).unwrap();
        assert_eq!(hex.len(), DIGEST_HEX_LEN);
        assert!(
            hex.bytes()
                .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
        );
        // The bare hex helper agrees with the prefixed form.
        assert_eq!(format!("{BLAKE3_PREFIX}{}", hash.hex()), text);
    }

    #[test]
    fn matches_reference_blake3_with_domain_framing() {
        // Independently reconstruct the exact bytes the hasher absorbs: the
        // little-endian u64 label length, the label, then the payload. This
        // pins the domain-separation framing so it cannot silently change.
        let payload = b"canonical-bytes";
        let label = HashDomain::Plan.label().as_bytes();
        let mut expected = blake3::Hasher::new();
        expected.update(&(label.len() as u64).to_le_bytes());
        expected.update(label);
        expected.update(payload);
        let expected = *expected.finalize().as_bytes();
        let got = hash_canonical_bytes(HashDomain::Plan, payload);
        assert_eq!(got.as_bytes(), &expected);
    }

    #[test]
    fn stable_across_calls_no_wall_clock() {
        // The bone's exit criterion: identical semantic bytes -> identical hash,
        // run to run (nothing time/provenance dependent is mixed in).
        let value = json!({"paintop": "1.0", "nodes": [], "exports": {}});
        let a = hash_value(HashDomain::Plan, &value).unwrap();
        let b = hash_value(HashDomain::Plan, &value).unwrap();
        assert_eq!(a, b);
        assert_eq!(a.to_string(), b.to_string());
    }

    #[test]
    fn key_order_does_not_change_the_hash() {
        // Canonicalization collapses key order, so equivalent plans share an id.
        let a = hash_value(HashDomain::Plan, &json!({"b": 1, "a": 2})).unwrap();
        let b = hash_value(HashDomain::Plan, &json!({"a": 2, "b": 1})).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn changed_semantic_field_changes_the_hash() {
        // The other half of the exit criterion: a real semantic change flips it.
        let base = hash_value(HashDomain::Plan, &json!({"sigma": 8.0})).unwrap();
        let changed = hash_value(HashDomain::Plan, &json!({"sigma": 8.5})).unwrap();
        assert_ne!(base, changed);
        // An integer vs. an integral float is a semantic distinction the
        // canonical emitter preserves, so the hash must distinguish them too.
        let as_int = hash_value(HashDomain::Plan, &json!({"sigma": 8})).unwrap();
        let as_float = hash_value(HashDomain::Plan, &json!({"sigma": 8.0})).unwrap();
        assert_ne!(as_int, as_float);
    }

    #[test]
    fn domains_separate_identical_bytes() {
        // Same payload, different domain -> different id, for every pair.
        let payload = b"{\"x\":1}";
        let hashes: Vec<_> = HashDomain::ALL
            .iter()
            .map(|&domain| hash_canonical_bytes(domain, payload))
            .collect();
        for (i, a) in hashes.iter().enumerate() {
            for (j, b) in hashes.iter().enumerate() {
                if i != j {
                    assert_ne!(a, b, "domains {i} and {j} collided");
                }
            }
        }
    }

    #[test]
    fn domain_labels_are_unique() {
        let mut labels: Vec<&str> = HashDomain::ALL.iter().map(|d| d.label()).collect();
        let count = labels.len();
        labels.sort_unstable();
        labels.dedup();
        assert_eq!(labels.len(), count, "domain labels must be pairwise unique");
    }

    #[test]
    fn label_payload_boundary_is_unambiguous() {
        // Without a length-prefixed label, `label="ab" payload="cd"` and
        // `label="abc" payload="d"` would hash identically. Two real domains
        // never share a prefix, but this proves the framing defends against it:
        // moving a byte across the label/payload boundary changes the digest.
        let domain = HashDomain::Plan;
        let label = domain.label();
        let split = &label[..label.len() - 1];
        // Reconstruct a "shifted" framing by hand and confirm it differs.
        let canonical = hash_canonical_bytes(domain, b"X");
        let mut shifted = blake3::Hasher::new();
        let shifted_label = split.as_bytes();
        shifted.update(&(shifted_label.len() as u64).to_le_bytes());
        shifted.update(shifted_label);
        shifted.update(&[label.as_bytes()[label.len() - 1], b'X']);
        let shifted = *shifted.finalize().as_bytes();
        assert_ne!(canonical.as_bytes(), &shifted);
    }

    #[test]
    fn round_trips_through_parse() {
        let hash = hash_canonical_bytes(HashDomain::CacheEntry, b"payload");
        let text = hash.to_string();
        let parsed = SemanticHash::parse(&text).unwrap();
        assert_eq!(parsed, hash);
        assert_eq!(parsed.to_string(), text);
    }

    #[test]
    fn parse_rejects_missing_prefix() {
        let bare = hash_canonical_bytes(HashDomain::Plan, b"x").hex();
        let err = SemanticHash::parse(&bare).unwrap_err();
        assert_eq!(err.class, ErrorClass::Parse);
        assert_eq!(err.code, E_INVALID_HASH);
    }

    #[test]
    fn parse_rejects_wrong_length() {
        let err = SemanticHash::parse("blake3:abcd").unwrap_err();
        assert_eq!(err.code, E_INVALID_HASH);
    }

    #[test]
    fn parse_rejects_non_hex_and_uppercase() {
        // A `g` is not hex.
        let bad = format!("blake3:{}", "g".repeat(DIGEST_HEX_LEN));
        assert_eq!(SemanticHash::parse(&bad).unwrap_err().code, E_INVALID_HASH);
        // Uppercase hex is not the canonical lowercase form we emit/accept.
        let upper = format!("blake3:{}", "A".repeat(DIGEST_HEX_LEN));
        assert_eq!(
            SemanticHash::parse(&upper).unwrap_err().code,
            E_INVALID_HASH
        );
    }

    #[test]
    fn empty_payload_still_hashes() {
        // An empty payload is fine; the domain label alone seeds the digest, so
        // empty payloads in different domains still differ.
        let a = hash_canonical_bytes(HashDomain::Plan, b"");
        let b = hash_canonical_bytes(HashDomain::Resource, b"");
        assert_ne!(a, b);
        assert!(a.to_string().starts_with(BLAKE3_PREFIX));
    }
}
