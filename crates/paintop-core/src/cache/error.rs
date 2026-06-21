//! Typed errors for the content-addressed cache (`plan.md` §10.3).
//!
//! Cache failures are [`ErrorClass::Execution`]-class in the central taxonomy:
//! they arise while a run is executing (computing a key, reading a stored
//! artifact, detecting corruption). This module owns the local `thiserror` enum
//! and the lift into the central [`paintop_ir::Error`].

use paintop_ir::{Error, ErrorClass};

/// Stable machine code: a stored cache entry was corrupt.
///
/// Its bytes could not be parsed, its recorded digest did not match its bytes, or
/// its envelope was structurally malformed. A corrupt entry is *ignored* (treated
/// as a miss), never reused.
pub const E_CACHE_CORRUPT: &str = "E_CACHE_CORRUPT";

/// Stable machine code: a cache value could not be serialized for storage.
pub const E_CACHE_SERIALIZE: &str = "E_CACHE_SERIALIZE";

/// Stable machine code: an I/O failure occurred reading or writing the on-disk
/// cache store.
pub const E_CACHE_IO: &str = "E_CACHE_IO";

/// Convenience result alias for the cache subsystem.
pub type CacheResult<T> = std::result::Result<T, CacheError>;

/// A failure raised while computing a cache key, or reading / writing the store.
#[derive(Debug, thiserror::Error)]
pub enum CacheError {
    /// A stored entry was corrupt: unparseable bytes, a digest mismatch, or a
    /// structurally malformed envelope. The store treats this as a miss and
    /// surfaces the reason so a caller can warn.
    #[error("cache entry for key `{key}` is corrupt and was ignored: {detail}")]
    Corrupt {
        /// The cache key whose stored entry was corrupt.
        key: String,
        /// What was wrong with the stored entry.
        detail: String,
    },
    /// A cache value could not be serialized for storage.
    #[error("failed to serialize a cache value for key `{key}`: {detail}")]
    Serialize {
        /// The cache key being written.
        key: String,
        /// The serialization failure detail.
        detail: String,
    },
    /// An I/O failure occurred reading or writing the on-disk store.
    #[error("cache I/O failed at `{path}`: {detail}")]
    Io {
        /// The path that failed.
        path: String,
        /// The I/O failure detail.
        detail: String,
    },
}

impl CacheError {
    /// The stable machine code for this failure.
    #[must_use]
    pub const fn code(&self) -> &'static str {
        match self {
            Self::Corrupt { .. } => E_CACHE_CORRUPT,
            Self::Serialize { .. } => E_CACHE_SERIALIZE,
            Self::Io { .. } => E_CACHE_IO,
        }
    }

    /// Build a [`CacheError::Corrupt`] for `key` with a reason.
    #[must_use]
    pub fn corrupt(key: impl Into<String>, detail: impl Into<String>) -> Self {
        Self::Corrupt {
            key: key.into(),
            detail: detail.into(),
        }
    }

    /// Lift this cache error into the central [`paintop_ir::Error`] taxonomy as an
    /// [`execution`](ErrorClass::Execution)-class failure.
    #[must_use]
    pub fn into_paintop(self) -> Error {
        let code = self.code();
        Error::new(ErrorClass::Execution, code, self.to_string())
    }
}

impl From<CacheError> for Error {
    fn from(err: CacheError) -> Self {
        err.into_paintop()
    }
}
