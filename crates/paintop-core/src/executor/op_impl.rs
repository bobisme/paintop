//! The executable operation implementation (the `cpu.reference` oracle) and its
//! registry (`plan.md` §10.1 phase 11, §6.1).
//!
//! The descriptor-level [`OpContract`](paintop_ir::OpContract) is *cheap*: it
//! infers output descriptors and propagates regions over metadata only, never
//! pixels. Execution needs the complementary half — the deterministic Rust that
//! actually *computes* a node's output samples from its input samples. That is an
//! [`OpImplementation`]: the runtime realization of an operation's
//! `cpu.reference@<v>` backend.
//!
//! The MVP ops (segment 2) supply real implementations; this bone defines the
//! trait, the [`ImplRegistry`], and uses a stub/identity implementation to test
//! the executor end to end.

use std::collections::BTreeMap;

use paintop_ir::{Error, OpId};

use crate::executor::value::ResourceValue;

/// The input resource values flowing into an operation, keyed by input port name.
///
/// A [`BTreeMap`] keeps iteration deterministic, matching the descriptor-level
/// [`Descriptors`](paintop_ir::Descriptors) convention.
pub type InputValues = BTreeMap<String, ResourceValue>;

/// The output resource values produced by an operation, keyed by output port
/// name.
pub type OutputValues = BTreeMap<String, ResourceValue>;

/// The executable, deterministic compute kernel of one operation — its
/// `cpu.reference` oracle (`plan.md` §6.1, §15.6).
///
/// The trait is **object-safe** (it is held behind `dyn OpImplementation` in the
/// [`ImplRegistry`]) and its [`compute`](OpImplementation::compute) must be
/// deterministic: identical inputs and params yield byte-identical outputs on
/// every run and machine (`plan.md` §1). Implementations receive resolved
/// parameters as canonical JSON so the executor need not know each op's concrete
/// param struct, mirroring [`OpContract`](paintop_ir::OpContract).
pub trait OpImplementation {
    /// Compute every declared output value from the input values and params.
    ///
    /// The returned map must cover exactly the operation's declared output ports;
    /// the executor verifies that and raises
    /// [`E_OUTPUT_NOT_PRODUCED`](crate::executor::error::E_OUTPUT_NOT_PRODUCED) if
    /// a declared port is missing.
    ///
    /// # Errors
    /// Returns an [`Error`] if the operation cannot compute its result from the
    /// given inputs (e.g. an input it requires is absent, or a param is out of
    /// range). The executor lifts it into an
    /// [`E_OP_DISPATCH_FAILED`](crate::executor::error::E_OP_DISPATCH_FAILED)
    /// execution failure attributed to the node.
    fn compute(
        &self,
        inputs: &InputValues,
        params: &serde_json::Value,
    ) -> Result<OutputValues, Error>;
}

/// A contract was already registered for an operation when a second was offered.
pub const E_DUPLICATE_IMPLEMENTATION: &str = "E_DUPLICATE_IMPLEMENTATION";

/// An in-memory index from an [`OpId`] to its executable [`OpImplementation`].
///
/// Distinct from both the manifest [`OperationRegistry`](paintop_ir::OperationRegistry)
/// (data) and the [`ContractRegistry`](paintop_ir::ContractRegistry)
/// (descriptor-level code): this holds the *compute* kernels the executor
/// dispatches. Keeping it separate lets the IR crate type-check a plan without
/// linking any pixel code, and lets the executor run with only the
/// implementations a given plan needs.
#[derive(Default)]
pub struct ImplRegistry {
    by_id: BTreeMap<OpId, Box<dyn OpImplementation>>,
}

impl ImplRegistry {
    /// Create an empty implementation registry.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Register the executable implementation for `id`.
    ///
    /// # Errors
    /// Returns an [`execution`](paintop_ir::ErrorClass::Execution) [`Error`] with
    /// code [`E_DUPLICATE_IMPLEMENTATION`] if an implementation is already
    /// registered for `id`.
    pub fn register(
        &mut self,
        id: OpId,
        implementation: Box<dyn OpImplementation>,
    ) -> Result<(), Error> {
        if self.by_id.contains_key(&id) {
            return Err(Error::new(
                paintop_ir::ErrorClass::Execution,
                E_DUPLICATE_IMPLEMENTATION,
                format!("an implementation is already registered for operation {id}"),
            ));
        }
        self.by_id.insert(id, implementation);
        Ok(())
    }

    /// Look up the implementation for `id`.
    #[must_use]
    pub fn get(&self, id: &OpId) -> Option<&dyn OpImplementation> {
        self.by_id.get(id).map(Box::as_ref)
    }

    /// Whether an implementation is registered for `id`.
    #[must_use]
    pub fn contains(&self, id: &OpId) -> bool {
        self.by_id.contains_key(id)
    }

    /// The number of registered implementations.
    #[must_use]
    pub fn len(&self) -> usize {
        self.by_id.len()
    }

    /// Whether the registry holds no implementations.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.by_id.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::{
        E_DUPLICATE_IMPLEMENTATION, ImplRegistry, InputValues, OpImplementation, OutputValues,
    };
    use paintop_ir::Error;

    struct Noop;
    impl OpImplementation for Noop {
        fn compute(
            &self,
            _inputs: &InputValues,
            _params: &serde_json::Value,
        ) -> Result<OutputValues, Error> {
            Ok(OutputValues::new())
        }
    }

    #[test]
    fn register_then_lookup() {
        let mut reg = ImplRegistry::new();
        let id = "source.create@1".parse().unwrap();
        reg.register(id, Box::new(Noop)).unwrap();
        let id2 = "source.create@1".parse().unwrap();
        assert!(reg.contains(&id2));
        assert!(reg.get(&id2).is_some());
        assert_eq!(reg.len(), 1);
        assert!(!reg.is_empty());
    }

    #[test]
    fn duplicate_registration_is_rejected() {
        let mut reg = ImplRegistry::new();
        reg.register("source.create@1".parse().unwrap(), Box::new(Noop))
            .unwrap();
        let err = reg
            .register("source.create@1".parse().unwrap(), Box::new(Noop))
            .unwrap_err();
        assert_eq!(err.code, E_DUPLICATE_IMPLEMENTATION);
        assert_eq!(reg.len(), 1);
    }
}
