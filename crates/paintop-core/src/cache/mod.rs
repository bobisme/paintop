//! The content-addressed result cache (`plan.md` §10.3).
//!
//! A producer node's output is identified by *what it is* — the operation, its
//! semantic version, its canonical parameters, its inputs' content, their
//! resource semantics, the seed, and the backend semantics version — not by
//! where it came from. This module owns that identity and the store it keys:
//!
//! - [`key`] — the [`CacheKey`] model and its semantic-version-aware
//!   invalidation: any output-affecting change flips the key, any provenance-only
//!   change leaves it unchanged.
//! - [`content`] — [`content hashing`](content::content_hash_value) of a resource
//!   value, the input identity a key folds in (never a path).
//! - [`store`] — the content-addressed result store: lookup, insert, validation
//!   metadata, and corruption detection.
//!
//! Every hash flows through the M0 canonical-bytes + BLAKE3 path
//! ([`paintop_ir::hash`]) with a domain label, so a cache key, a content hash, a
//! plan hash, and a resource hash can never alias.

pub mod content;
pub mod error;
pub mod key;
pub mod replay;
pub mod store;

pub use content::{ContentHash, content_hash_descriptor, content_hash_value};
pub use error::{CacheError, CacheResult, E_CACHE_CORRUPT, E_CACHE_IO, E_CACHE_SERIALIZE};
pub use key::{BACKEND_SEMANTICS_VERSION, CacheKey, CacheKeyInputs, InputContribution};
pub use replay::{CachedExecution, execute_cached};
pub use store::{CacheEntry, CacheStore, CacheValidation};
