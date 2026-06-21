//! The content-addressed result **store**: keyed insert, lookup, validation
//! metadata, and corruption detection (`plan.md` §10.3).
//!
//! A [`CacheStore`] maps a [`CacheKey`] to a producer node's output value. Because
//! the key already encodes the full semantic identity of the computation (op,
//! semantic version, params, input content, seed, backend semantics — see
//! [`super::key`]), a hit *is* the correct result: no recomputation is needed.
//!
//! Each entry carries [`CacheValidation`] metadata — the op id, op semantic
//! version, and backend semantics version that produced it. The key already keys
//! these apart, so a clean store never serves a value across an incompatible
//! version; the metadata is a *belt-and-suspenders* guard for the on-disk store,
//! where a stale or hand-edited file could otherwise be replayed. On lookup the
//! store re-derives the entry's digest and checks the validation block; a
//! mismatch is reported as [`CacheError::Corrupt`] and the entry is **ignored
//! (treated as a miss), never silently reused**.
//!
//! Two backends share one API:
//! - an **in-memory** map for a single run (the executor's working cache);
//! - an **on-disk** directory store for persistence across runs, where each entry
//!   is one canonical-JSON file named by its key and self-verifying via an
//!   embedded body digest.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use paintop_ir::{HashDomain, ResourceDescriptor, hash_canonical_bytes};
use serde::{Deserialize, Serialize};

use super::error::{CacheError, CacheResult};
use super::key::CacheKey;
use crate::executor::ResourceValue;

/// The validation metadata recorded alongside every cached value.
///
/// These fields are *already* folded into the [`CacheKey`], so a clean store can
/// never serve a value across an incompatible version (the keys differ). The
/// block is re-checked on lookup as a defense for the on-disk store: an entry
/// whose recorded version does not match the version being requested is rejected
/// rather than reused, so a stale or hand-edited file cannot replay into an
/// incompatible run.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CacheValidation {
    /// The versioned op id that produced the entry (e.g. `filter.blur@1`).
    pub op_id: String,
    /// The op semantic version (`impl_version`) that produced the entry.
    pub op_semantic_version: u32,
    /// The backend semantics version that produced the entry.
    pub backend_semantics_version: u32,
}

impl CacheValidation {
    /// Whether `self` is compatible with a request validated by `other`.
    ///
    /// Compatibility requires an exact match on op id, op semantic version, and
    /// backend semantics version: any difference is an incompatible-semantics
    /// boundary across which a cached value must never be reused.
    #[must_use]
    pub fn is_compatible_with(&self, other: &Self) -> bool {
        self == other
    }
}

/// The serializable body of a cached value: enough to reconstruct a
/// [`ResourceValue`] exactly.
///
/// A raster value stores its descriptor, channel count, and the row-major `f32`
/// samples; a report value stores the structured [`Report`](paintop_ir::Report).
/// The enum is the on-disk shape, so it is `deny_unknown_fields` and tagged.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
enum StoredBody {
    /// A whole-image raster value.
    Raster {
        /// The value's typed descriptor.
        descriptor: ResourceDescriptor,
        /// Interleaved samples per pixel.
        channels: u32,
        /// The row-major `f32` sample buffer.
        samples: Vec<f32>,
    },
    /// A structured report value (no raster).
    Report {
        /// The structured report (boxed: a [`Report`](paintop_ir::Report) is much
        /// larger than a raster body's handful of fields).
        report: Box<paintop_ir::Report>,
    },
}

impl StoredBody {
    /// Capture a [`ResourceValue`] into its serializable body.
    fn capture(value: &ResourceValue) -> Self {
        if let Some(report) = value.as_report() {
            return Self::Report {
                report: Box::new(report.clone()),
            };
        }
        Self::Raster {
            descriptor: *value.descriptor(),
            channels: value.channels(),
            samples: value.samples().to_vec(),
        }
    }

    /// Reconstruct the [`ResourceValue`] this body captured.
    ///
    /// # Errors
    /// Returns [`CacheError::Corrupt`] (attributed to `key`) if a raster body's
    /// sample buffer does not match its descriptor — the signature of a truncated
    /// or tampered entry.
    fn reconstruct(self, key: &CacheKey) -> CacheResult<ResourceValue> {
        match self {
            Self::Report { report } => Ok(ResourceValue::report(*report)),
            Self::Raster {
                descriptor,
                channels,
                samples,
            } => ResourceValue::new(descriptor, channels, samples).map_err(|actual| {
                CacheError::corrupt(
                    key.to_string(),
                    format!("raster sample buffer length {actual} does not match its descriptor"),
                )
            }),
        }
    }

    /// The canonical bytes of this body, for the integrity digest.
    ///
    /// A raster body's `f32` samples are *not* representable in canonical JSON
    /// (it forbids non-finite floats), so the digest is taken over an explicit
    /// little-endian byte framing rather than a JSON document.
    fn digest_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        match self {
            Self::Raster {
                descriptor,
                channels,
                samples,
            } => {
                bytes.push(0x01);
                push_descriptor(&mut bytes, descriptor);
                bytes.extend_from_slice(&channels.to_le_bytes());
                push_len(&mut bytes, samples.len());
                for sample in samples {
                    bytes.extend_from_slice(&sample.to_le_bytes());
                }
            }
            Self::Report { report } => {
                bytes.push(0x02);
                let report_bytes = serde_json::to_value(report)
                    .ok()
                    .and_then(|v| paintop_ir::to_canonical_bytes(&v).ok())
                    .unwrap_or_default();
                push_len(&mut bytes, report_bytes.len());
                bytes.extend_from_slice(&report_bytes);
            }
        }
        bytes
    }
}

/// One stored cache entry: a captured value, its validation metadata, and a
/// self-verifying integrity digest over the body.
///
/// The `body_digest` is the BLAKE3 (`blake3:…`) of the body's canonical bytes.
/// On lookup the store re-derives it; a mismatch means the bytes were truncated
/// or tampered with, and the entry is rejected as [`CacheError::Corrupt`] rather
/// than reused.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CacheEntry {
    /// The version metadata that produced the value.
    pub validation: CacheValidation,
    /// The captured value body.
    body: StoredBody,
    /// The `blake3:…` integrity digest over the body's canonical bytes.
    body_digest: String,
}

impl CacheEntry {
    /// Capture a value and its validation metadata into a verifiable entry.
    #[must_use]
    pub fn new(value: &ResourceValue, validation: CacheValidation) -> Self {
        let body = StoredBody::capture(value);
        let body_digest = body_digest(&body);
        Self {
            validation,
            body,
            body_digest,
        }
    }

    /// Verify the entry's integrity digest against its body, returning the
    /// captured value if it holds.
    ///
    /// # Errors
    /// Returns [`CacheError::Corrupt`] (attributed to `key`) if the recorded
    /// digest does not match the body's bytes, or if the body cannot be
    /// reconstructed into a well-formed value.
    pub fn verified_value(self, key: &CacheKey) -> CacheResult<ResourceValue> {
        let expected = body_digest(&self.body);
        if expected != self.body_digest {
            return Err(CacheError::corrupt(
                key.to_string(),
                "stored body digest does not match the body bytes",
            ));
        }
        self.body.reconstruct(key)
    }
}

/// The integrity digest of a stored body, rendered as a `blake3:…` string.
fn body_digest(body: &StoredBody) -> String {
    hash_canonical_bytes(HashDomain::Content, &body.digest_bytes()).to_string()
}

/// A content-addressed cache store with an in-memory and an on-disk backend.
///
/// Both backends share the same [`get`](CacheStore::get) / [`put`](CacheStore::put)
/// contract: a [`put`](CacheStore::put) records a value under its key; a
/// [`get`](CacheStore::get) returns the value only if the entry exists, its body
/// digest verifies, and its validation metadata is compatible with the request.
/// A corrupt entry surfaces as [`CacheError::Corrupt`] so the caller can warn and
/// recompute (a miss), never a silent stale reuse.
#[derive(Debug)]
pub enum CacheStore {
    /// An in-memory map for a single run.
    Memory(BTreeMap<CacheKey, CacheEntry>),
    /// An on-disk directory store, one canonical-JSON file per entry.
    Disk(PathBuf),
}

impl CacheStore {
    /// A fresh empty in-memory store.
    #[must_use]
    pub const fn in_memory() -> Self {
        Self::Memory(BTreeMap::new())
    }

    /// A disk-backed store rooted at `dir`, creating the directory if absent.
    ///
    /// # Errors
    /// Returns [`CacheError::Io`] if the directory cannot be created.
    pub fn on_disk(dir: impl Into<PathBuf>) -> CacheResult<Self> {
        let dir = dir.into();
        std::fs::create_dir_all(&dir).map_err(|e| CacheError::Io {
            path: dir.display().to_string(),
            detail: e.to_string(),
        })?;
        Ok(Self::Disk(dir))
    }

    /// Record `value` under `key` with its validation metadata.
    ///
    /// # Errors
    /// Returns [`CacheError::Serialize`] / [`CacheError::Io`] if an on-disk entry
    /// cannot be written. The in-memory backend never fails.
    pub fn put(
        &mut self,
        key: &CacheKey,
        value: &ResourceValue,
        validation: CacheValidation,
    ) -> CacheResult<()> {
        let entry = CacheEntry::new(value, validation);
        match self {
            Self::Memory(map) => {
                map.insert(*key, entry);
                Ok(())
            }
            Self::Disk(dir) => write_entry(dir, key, &entry),
        }
    }

    /// Look up the value cached under `key`, validated against `request`.
    ///
    /// Returns `Ok(None)` for a clean miss (no entry). Returns the value only when
    /// the entry exists, its digest verifies, **and** its validation metadata is
    /// compatible with `request`; an incompatible entry is reported as
    /// [`CacheError::Corrupt`] — rejected, never reused.
    ///
    /// # Errors
    /// Returns [`CacheError::Corrupt`] if the stored entry is unparseable, its
    /// digest does not verify, or its validation metadata is incompatible with
    /// `request`. Returns [`CacheError::Io`] on an on-disk read failure other than
    /// a plain missing file.
    pub fn get(
        &self,
        key: &CacheKey,
        request: &CacheValidation,
    ) -> CacheResult<Option<ResourceValue>> {
        let entry = match self {
            Self::Memory(map) => map.get(key).cloned(),
            Self::Disk(dir) => read_entry(dir, key)?,
        };
        let Some(entry) = entry else {
            return Ok(None);
        };
        // Reject an entry whose recorded semantics are incompatible with the
        // request: a stale/incompatible entry must never be silently reused.
        if !entry.validation.is_compatible_with(request) {
            return Err(CacheError::corrupt(
                key.to_string(),
                format!(
                    "entry validation {:?} is incompatible with the requested {request:?}",
                    entry.validation
                ),
            ));
        }
        let value = entry.verified_value(key)?;
        Ok(Some(value))
    }

    /// Whether an entry is present under `key` (without validating it).
    #[must_use]
    pub fn contains(&self, key: &CacheKey) -> bool {
        match self {
            Self::Memory(map) => map.contains_key(key),
            Self::Disk(dir) => entry_path(dir, key).exists(),
        }
    }
}

/// The on-disk file path for `key` under `dir`.
///
/// The key's `blake3:` prefix is replaced with a `blake3-` filename-safe form so
/// the entry is one flat file per key (no colon in the name on any filesystem).
fn entry_path(dir: &Path, key: &CacheKey) -> PathBuf {
    let name = key.to_string().replace(':', "-");
    dir.join(format!("{name}.json"))
}

/// Write `entry` as a canonical-JSON file named by `key`.
fn write_entry(dir: &Path, key: &CacheKey, entry: &CacheEntry) -> CacheResult<()> {
    let value = serde_json::to_value(entry).map_err(|e| CacheError::Serialize {
        key: key.to_string(),
        detail: e.to_string(),
    })?;
    let bytes = paintop_ir::to_canonical_bytes(&value).map_err(|e| CacheError::Serialize {
        key: key.to_string(),
        detail: e.to_string(),
    })?;
    let path = entry_path(dir, key);
    std::fs::write(&path, &bytes).map_err(|e| CacheError::Io {
        path: path.display().to_string(),
        detail: e.to_string(),
    })
}

/// Read and parse the entry for `key` under `dir`, if the file exists.
///
/// A missing file is a clean miss (`Ok(None)`); unparseable bytes are
/// [`CacheError::Corrupt`].
fn read_entry(dir: &Path, key: &CacheKey) -> CacheResult<Option<CacheEntry>> {
    let path = entry_path(dir, key);
    let bytes = match std::fs::read(&path) {
        Ok(bytes) => bytes,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => {
            return Err(CacheError::Io {
                path: path.display().to_string(),
                detail: e.to_string(),
            });
        }
    };
    let entry: CacheEntry = serde_json::from_slice(&bytes).map_err(|e| {
        CacheError::corrupt(key.to_string(), format!("entry file is not valid: {e}"))
    })?;
    Ok(Some(entry))
}

/// Append the length-prefixed canonical-JSON descriptor to `bytes`.
fn push_descriptor(bytes: &mut Vec<u8>, descriptor: &ResourceDescriptor) {
    let descriptor_bytes = serde_json::to_value(descriptor)
        .ok()
        .and_then(|v| paintop_ir::to_canonical_bytes(&v).ok())
        .unwrap_or_default();
    push_len(bytes, descriptor_bytes.len());
    bytes.extend_from_slice(&descriptor_bytes);
}

/// Append a little-endian `u64` length prefix to `bytes`.
fn push_len(bytes: &mut Vec<u8>, len: usize) {
    bytes.extend_from_slice(&(len as u64).to_le_bytes());
}

#[cfg(test)]
mod tests {
    use super::{CacheEntry, CacheStore, CacheValidation};
    use crate::cache::content::content_hash_descriptor;
    use crate::cache::key::{CacheKey, CacheKeyInputs, InputContribution};
    use crate::executor::ResourceValue;
    use paintop_ir::{
        AlphaRepresentation, ChannelLayout, ColorEncoding, ColorRange, CoordinateConvention,
        Extent, ImageDescriptor, ResourceDescriptor, ScalarType, SemanticRole,
    };
    use serde_json::json;
    use std::sync::atomic::{AtomicU64, Ordering};

    fn image() -> ResourceDescriptor {
        ResourceDescriptor::Image(ImageDescriptor {
            extent: Extent::new(2, 2),
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
        ResourceValue::new(image(), 4, vec![fill; 2 * 2 * 4]).expect("value")
    }

    fn key(seed: u64) -> CacheKey {
        let desc = image();
        let inputs = CacheKeyInputs::new(
            "filter.blur@1".parse().expect("op"),
            1,
            json!({ "sigma": 2.0 }).as_object().expect("o").clone(),
            vec![InputContribution::new(
                "image",
                content_hash_descriptor(&desc, &[1.0]),
                desc,
            )],
        )
        .with_seed(seed);
        CacheKey::compute(&inputs).expect("key")
    }

    fn validation(op_v: u32, backend_v: u32) -> CacheValidation {
        CacheValidation {
            op_id: "filter.blur@1".to_owned(),
            op_semantic_version: op_v,
            backend_semantics_version: backend_v,
        }
    }

    fn scratch_dir(tag: &str) -> std::path::PathBuf {
        static SEQ: AtomicU64 = AtomicU64::new(0);
        let seq = SEQ.fetch_add(1, Ordering::Relaxed);
        let mut dir = std::env::temp_dir();
        dir.push(format!("paintop-cache-{}-{tag}-{seq}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("scratch");
        dir
    }

    #[test]
    fn in_memory_hit_returns_the_stored_value() {
        let mut store = CacheStore::in_memory();
        let k = key(0);
        store.put(&k, &value(0.5), validation(1, 1)).expect("put");
        let got = store.get(&k, &validation(1, 1)).expect("get").expect("hit");
        assert_eq!(got, value(0.5));
    }

    #[test]
    fn miss_returns_none() {
        let store = CacheStore::in_memory();
        assert!(
            store
                .get(&key(7), &validation(1, 1))
                .expect("get")
                .is_none()
        );
    }

    #[test]
    fn on_disk_round_trips_a_value() {
        let dir = scratch_dir("disk");
        let mut store = CacheStore::on_disk(&dir).expect("store");
        let k = key(1);
        store.put(&k, &value(0.25), validation(1, 1)).expect("put");
        // A fresh handle on the same dir sees the entry (persistence).
        let reopened = CacheStore::on_disk(&dir).expect("reopen");
        let got = reopened
            .get(&k, &validation(1, 1))
            .expect("get")
            .expect("hit");
        assert_eq!(got, value(0.25));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn incompatible_op_semantic_version_is_rejected_not_reused() {
        let mut store = CacheStore::in_memory();
        let k = key(0);
        store.put(&k, &value(0.5), validation(1, 1)).expect("put");
        // Requesting with a bumped op semantic version must not reuse the entry.
        let err = store.get(&k, &validation(2, 1)).expect_err("must reject");
        assert_eq!(err.code(), super::super::error::E_CACHE_CORRUPT);
    }

    #[test]
    fn incompatible_backend_version_is_rejected_not_reused() {
        let mut store = CacheStore::in_memory();
        let k = key(0);
        store.put(&k, &value(0.5), validation(1, 1)).expect("put");
        let err = store.get(&k, &validation(1, 2)).expect_err("must reject");
        assert_eq!(err.code(), super::super::error::E_CACHE_CORRUPT);
    }

    #[test]
    fn corrupted_disk_entry_is_ignored_with_an_error() {
        let dir = scratch_dir("corrupt");
        let mut store = CacheStore::on_disk(&dir).expect("store");
        let k = key(2);
        store.put(&k, &value(0.5), validation(1, 1)).expect("put");
        // Overwrite the entry file with garbage.
        let name = k.to_string().replace(':', "-");
        std::fs::write(dir.join(format!("{name}.json")), b"not json").expect("clobber");
        let err = store.get(&k, &validation(1, 1)).expect_err("must reject");
        assert_eq!(err.code(), super::super::error::E_CACHE_CORRUPT);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn tampered_body_digest_is_detected() {
        // Construct an entry, then mutate its body so the digest no longer matches.
        let entry = CacheEntry::new(&value(0.5), validation(1, 1));
        // Re-serialize, flip a sample, and re-parse to simulate disk tampering
        // that leaves the recorded digest stale.
        let mut as_value = serde_json::to_value(&entry).expect("ser");
        as_value["body"]["samples"][0] = json!(99.0);
        let tampered: CacheEntry = serde_json::from_value(as_value).expect("de");
        let err = tampered.verified_value(&key(0)).expect_err("must reject");
        assert_eq!(err.code(), super::super::error::E_CACHE_CORRUPT);
    }

    #[test]
    fn contains_reflects_presence() {
        let mut store = CacheStore::in_memory();
        let k = key(0);
        assert!(!store.contains(&k));
        store.put(&k, &value(0.5), validation(1, 1)).expect("put");
        assert!(store.contains(&k));
    }
}
