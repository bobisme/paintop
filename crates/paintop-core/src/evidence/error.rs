//! Typed errors for evidence-bundle writing.
//!
//! Bundle writing is an export concern: it materializes the run's evidence to
//! disk. Failures here map onto the central [`ErrorClass::Export`] bucket and
//! its stable exit code (`plan.md` §15.4), so a failed bundle write surfaces
//! through the same agent-facing contract as any other export failure. This
//! module owns the local `thiserror` enum and the lift into the central
//! [`paintop_ir::Error`].

use std::io;
use std::path::{Path, PathBuf};

use paintop_ir::{Error, ErrorClass, ErrorContext};

/// Stable machine code: a bundle could not be written to disk.
pub const E_BUNDLE_IO: &str = "E_BUNDLE_IO";
/// Stable machine code: an artifact could not be serialized to canonical bytes.
pub const E_BUNDLE_SERIALIZE: &str = "E_BUNDLE_SERIALIZE";

/// Convenience result alias for the evidence subsystem.
pub type BundleResult<T> = std::result::Result<T, BundleError>;

/// A failure while writing an evidence bundle.
///
/// These are all [`export`](ErrorClass::Export)-class failures in the central
/// taxonomy; [`BundleError::into_paintop`] (and the `From` impl) perform the
/// lift, attaching the offending path as locating context.
#[derive(Debug, thiserror::Error)]
pub enum BundleError {
    /// A filesystem operation (create dir, atomic write, rename) failed.
    #[error("evidence bundle I/O failed at `{path}`: {message}")]
    Io {
        /// The path the operation targeted.
        path: PathBuf,
        /// What was being attempted.
        message: String,
        /// The underlying OS error, if any.
        #[source]
        source: Option<io::Error>,
    },
    /// An artifact value could not be canonicalized / serialized.
    #[error("evidence artifact could not be serialized: {message}")]
    Serialize {
        /// What was being serialized.
        message: String,
        /// The underlying serialization error, lifted from the IR crate.
        #[source]
        source: Box<Error>,
    },
}

impl BundleError {
    /// An I/O failure with no underlying `io::Error` (a precondition we detected
    /// ourselves, e.g. a path with no parent directory).
    pub(crate) fn io_at(path: &Path, message: impl Into<String>) -> Self {
        Self::Io {
            path: path.to_path_buf(),
            message: message.into(),
            source: None,
        }
    }

    /// An I/O failure wrapping an underlying `io::Error`.
    pub(crate) fn io_source(path: &Path, message: impl Into<String>, source: io::Error) -> Self {
        Self::Io {
            path: path.to_path_buf(),
            message: message.into(),
            source: Some(source),
        }
    }

    /// A serialization failure lifting an IR-level [`Error`].
    pub(crate) fn serialize(message: impl Into<String>, source: Error) -> Self {
        Self::Serialize {
            message: message.into(),
            source: Box::new(source),
        }
    }

    /// The stable machine code for this failure.
    #[must_use]
    pub const fn code(&self) -> &'static str {
        match self {
            Self::Io { .. } => E_BUNDLE_IO,
            Self::Serialize { .. } => E_BUNDLE_SERIALIZE,
        }
    }

    /// Lift this bundle error into the central [`paintop_ir::Error`] taxonomy as
    /// an [`export`](ErrorClass::Export)-class failure, attaching the offending
    /// path (when known) as locating context.
    #[must_use]
    pub fn into_paintop(self) -> Error {
        let code = self.code();
        let message = self.to_string();
        let path = match &self {
            Self::Io { path, .. } => Some(path.display().to_string()),
            Self::Serialize { .. } => None,
        };
        let mut err = Error::new(ErrorClass::Export, code, message);
        if let Some(path) = path {
            err = err.with_context(ErrorContext::default().with_path(path));
        }
        err
    }
}

impl From<BundleError> for Error {
    fn from(err: BundleError) -> Self {
        err.into_paintop()
    }
}
