//! Typed errors for the `wgpu` backend (`plan.md` §12.3, §19 M3 criterion 4).
//!
//! Every GPU failure surfaces as a paintop [`Error`] with a stable class + code so
//! the runtime can map it to a deterministic exit class and an honest evidence
//! record — never a panic, never a silent wrong result.

use paintop_ir::{Error, ErrorClass};

/// Stable machine code: no compatible GPU adapter/device is available on this host
/// and the caller required the GPU backend (no reference fallback was allowed).
///
/// Class [`Policy`](ErrorClass::Policy): like
/// [`E_BACKEND_UNSUPPORTED`](paintop_core::executor::dispatch::E_BACKEND_UNSUPPORTED),
/// a *required* backend that cannot run is a policy rejection, not an execution
/// crash. A caller that permits fallback never sees this — it gets the typed
/// [`GpuUnavailable`](crate::gpu::probe::GpuUnavailable) and runs the oracle.
pub const E_GPU_UNAVAILABLE: &str = "E_GPU_UNAVAILABLE";

/// Stable machine code: a GPU dispatch was rejected before submission.
///
/// Raised when its dimensions are invalid — zero-sized, or exceeding the device's
/// per-dimension / per-group workgroup limits, or overflowing the index space.
///
/// Class [`Execution`](ErrorClass::Execution): the work itself could not be
/// scheduled. Caught *before* any GPU submission, so a malformed dispatch can never
/// hang or silently truncate (`plan.md` §12.3: "validate dispatch dimensions and
/// overflow").
pub const E_GPU_DISPATCH_INVALID: &str = "E_GPU_DISPATCH_INVALID";

/// An error raised by the `wgpu` backend.
///
/// A thin typed wrapper that converts into a paintop [`Error`] with the matching
/// stable code, so the GPU backend integrates with the same error envelope and
/// exit-class machinery as every other crate.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum GpuError {
    /// No compatible GPU adapter/device is available and the GPU was required.
    #[error("no compatible GPU adapter is available: {reason}")]
    Unavailable {
        /// Why the GPU is unavailable (e.g. "no adapter matched the request").
        reason: String,
    },
    /// A dispatch was rejected for invalid dimensions before submission.
    #[error("invalid GPU dispatch: {reason}")]
    DispatchInvalid {
        /// What about the dispatch was invalid (zero extent, limit exceeded, …).
        reason: String,
    },
}

impl GpuError {
    /// The stable machine code for this error.
    #[must_use]
    pub const fn code(&self) -> &'static str {
        match self {
            Self::Unavailable { .. } => E_GPU_UNAVAILABLE,
            Self::DispatchInvalid { .. } => E_GPU_DISPATCH_INVALID,
        }
    }

    /// The error class this error maps to.
    #[must_use]
    pub const fn class(&self) -> ErrorClass {
        match self {
            Self::Unavailable { .. } => ErrorClass::Policy,
            Self::DispatchInvalid { .. } => ErrorClass::Execution,
        }
    }
}

impl From<GpuError> for Error {
    fn from(err: GpuError) -> Self {
        let class = err.class();
        let code = err.code();
        Self::new(class, code, err.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::{E_GPU_DISPATCH_INVALID, E_GPU_UNAVAILABLE, GpuError};
    use paintop_ir::{Error, ErrorClass};

    #[test]
    fn unavailable_maps_to_policy_class() {
        let g = GpuError::Unavailable {
            reason: "no adapter".to_owned(),
        };
        assert_eq!(g.code(), E_GPU_UNAVAILABLE);
        assert_eq!(g.class(), ErrorClass::Policy);
        let e: Error = g.into();
        assert_eq!(e.class, ErrorClass::Policy);
        assert_eq!(e.code, E_GPU_UNAVAILABLE);
    }

    #[test]
    fn dispatch_invalid_maps_to_execution_class() {
        let g = GpuError::DispatchInvalid {
            reason: "zero extent".to_owned(),
        };
        assert_eq!(g.code(), E_GPU_DISPATCH_INVALID);
        assert_eq!(g.class(), ErrorClass::Execution);
        let e: Error = g.into();
        assert_eq!(e.class, ErrorClass::Execution);
        assert_eq!(e.code, E_GPU_DISPATCH_INVALID);
    }
}
