//! Typed errors for the sequential whole-image executor.
//!
//! Execution failures map onto the central [`ErrorClass::Execution`] bucket and
//! its stable exit code (`plan.md` §15.4), so a failed dispatch surfaces through
//! the same agent-facing contract as any other runtime failure. This module owns
//! the local `thiserror` enum and the lift into the central [`paintop_ir::Error`].

use paintop_ir::{Error, ErrorClass, ErrorContext};

/// Stable machine code: no executable implementation is registered for an
/// operation the demanded graph uses.
pub const E_IMPLEMENTATION_NOT_FOUND: &str = "E_IMPLEMENTATION_NOT_FOUND";

/// Stable machine code: a required input resource was not available when a node
/// was dispatched (an upstream producer did not run, or an external input value
/// was not supplied).
pub const E_INPUT_NOT_AVAILABLE: &str = "E_INPUT_NOT_AVAILABLE";

/// Stable machine code: an op implementation did not produce a value for a port
/// it (and its manifest) declares as an output.
pub const E_OUTPUT_NOT_PRODUCED: &str = "E_OUTPUT_NOT_PRODUCED";

/// Stable machine code: an op implementation raised a runtime failure while
/// computing a node's output.
pub const E_OP_DISPATCH_FAILED: &str = "E_OP_DISPATCH_FAILED";

/// Convenience result alias for the executor subsystem.
pub type ExecResult<T> = std::result::Result<T, ExecError>;

/// A failure raised while executing a demanded graph.
///
/// These are all [`execution`](ErrorClass::Execution)-class failures in the
/// central taxonomy; [`ExecError::into_paintop`] (and the `From` impl) perform
/// the lift, attaching the offending node as locating context.
#[derive(Debug, thiserror::Error)]
pub enum ExecError {
    /// A demanded node's operation has no registered executable implementation.
    #[error("node `{node}` uses operation `{op}` but no executable implementation is registered")]
    ImplementationNotFound {
        /// The graph node that could not be dispatched.
        node: String,
        /// The versioned operation id.
        op: String,
    },
    /// A node's wired input had no resolved value when it was dispatched.
    #[error("node `{node}` input port `{port}` had no available value: {detail}")]
    InputNotAvailable {
        /// The consuming node.
        node: String,
        /// The unwired-at-runtime input port.
        port: String,
        /// What was missing (an external input or an upstream output).
        detail: String,
    },
    /// An implementation omitted a declared output port's value.
    #[error("node `{node}` (`{op}`) did not produce a value for declared output port `{port}`")]
    OutputNotProduced {
        /// The node whose implementation under-produced.
        node: String,
        /// The versioned operation id.
        op: String,
        /// The output port left unproduced.
        port: String,
    },
    /// An op implementation itself failed while computing a node.
    #[error("node `{node}` (`{op}`) failed during dispatch")]
    Dispatch {
        /// The node that failed.
        node: String,
        /// The versioned operation id.
        op: String,
        /// The underlying error the implementation raised, lifted from the IR
        /// crate.
        #[source]
        source: Box<Error>,
    },
}

impl ExecError {
    /// The stable machine code for this failure.
    #[must_use]
    pub const fn code(&self) -> &'static str {
        match self {
            Self::ImplementationNotFound { .. } => E_IMPLEMENTATION_NOT_FOUND,
            Self::InputNotAvailable { .. } => E_INPUT_NOT_AVAILABLE,
            Self::OutputNotProduced { .. } => E_OUTPUT_NOT_PRODUCED,
            Self::Dispatch { .. } => E_OP_DISPATCH_FAILED,
        }
    }

    /// The graph node this failure is attributed to.
    #[must_use]
    pub fn node(&self) -> &str {
        match self {
            Self::ImplementationNotFound { node, .. }
            | Self::InputNotAvailable { node, .. }
            | Self::OutputNotProduced { node, .. }
            | Self::Dispatch { node, .. } => node,
        }
    }

    /// Lift this executor error into the central [`paintop_ir::Error`] taxonomy
    /// as an [`execution`](ErrorClass::Execution)-class failure, attaching the
    /// offending node as locating context.
    #[must_use]
    pub fn into_paintop(self) -> Error {
        let code = self.code();
        let message = self.to_string();
        let node = self.node().to_owned();
        Error::new(ErrorClass::Execution, code, message)
            .with_context(ErrorContext::default().with_node(node))
    }
}

impl From<ExecError> for Error {
    fn from(err: ExecError) -> Self {
        err.into_paintop()
    }
}
