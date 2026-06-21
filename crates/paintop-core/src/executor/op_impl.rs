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

use paintop_ir::{Error, ErrorClass, ImplId, OpId};

use crate::executor::dispatch::BackendId;
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
///
/// # Thread safety
///
/// A kernel is [`Send`] + [`Sync`] so the scheduler-owned tile parallelism (M3,
/// `plan.md` §12.2) can dispatch independent output tiles of one node across a
/// bounded Rayon pool from a shared `&dyn OpImplementation`. This costs nothing:
/// every kernel is a stateless, deterministic pure function of `(inputs, params)`
/// — it holds no mutable state, so sharing it across threads is trivially sound,
/// and the bound simply forbids a future kernel from smuggling in interior
/// mutability that would make parallel dispatch non-deterministic.
pub trait OpImplementation: Send + Sync {
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

/// An in-memory index from an [`OpId`] to its executable [`OpImplementation`]s.
///
/// Distinct from both the manifest [`OperationRegistry`](paintop_ir::OperationRegistry)
/// (data) and the [`ContractRegistry`](paintop_ir::ContractRegistry)
/// (descriptor-level code): this holds the *compute* kernels the executor
/// dispatches. Keeping it separate lets the IR crate type-check a plan without
/// linking any pixel code, and lets the executor run with only the
/// implementations a given plan needs.
///
/// Each op carries its `cpu.reference` oracle ([`register`](Self::register)) and,
/// in M3, optionally additional backends ([`register_backend`](Self::register_backend))
/// — an optimized CPU kernel, a `wgpu` kernel — keyed by
/// [`BackendId`]. The backend a node is
/// dispatched on is chosen by
/// [`select_backend`](crate::executor::dispatch::select_backend) from policy, not
/// by the op; this registry just supplies whatever kernel the selection names.
#[derive(Default)]
pub struct ImplRegistry {
    /// The `cpu.reference` oracle for each op, registered via [`Self::register`].
    by_id: BTreeMap<OpId, Box<dyn OpImplementation>>,
    /// Additional (non-reference) backend kernels, keyed by `(op, backend)`.
    by_backend: BTreeMap<(OpId, BackendId), Box<dyn OpImplementation>>,
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

    /// Register an additional, **non-reference** backend kernel for `op` under the
    /// backend named by `impl_id` (e.g. `cpu.optimized@1`, `wgpu.separable@1`).
    ///
    /// The reference oracle is registered via [`register`](Self::register); this is
    /// for the optimized/GPU backends a node may be dispatched on under policy.
    /// Selecting which backend serves a node is
    /// [`select_backend`](crate::executor::dispatch::select_backend)'s job — this
    /// only makes the kernel *available*.
    ///
    /// # Errors
    /// Returns an [`execution`](paintop_ir::ErrorClass::Execution) [`Error`] with
    /// code [`E_DUPLICATE_IMPLEMENTATION`] if a kernel is already registered for
    /// `(op, backend)`, or if `impl_id` names the `cpu.reference` backend (which
    /// must be registered via [`register`](Self::register), not here).
    pub fn register_backend(
        &mut self,
        op: OpId,
        impl_id: &ImplId,
        implementation: Box<dyn OpImplementation>,
    ) -> Result<(), Error> {
        let backend = BackendId::from(impl_id);
        if backend.is_reference() {
            return Err(Error::new(
                ErrorClass::Execution,
                E_DUPLICATE_IMPLEMENTATION,
                format!(
                    "operation {op}: the cpu.reference oracle must be registered via \
                     `register`, not `register_backend`"
                ),
            ));
        }
        let key = (op, backend);
        if self.by_backend.contains_key(&key) {
            return Err(Error::new(
                ErrorClass::Execution,
                E_DUPLICATE_IMPLEMENTATION,
                format!(
                    "a `{}` implementation is already registered for operation {}",
                    key.1, key.0
                ),
            ));
        }
        self.by_backend.insert(key, implementation);
        Ok(())
    }

    /// Look up the `cpu.reference` implementation for `id`.
    #[must_use]
    pub fn get(&self, id: &OpId) -> Option<&dyn OpImplementation> {
        self.by_id.get(id).map(Box::as_ref)
    }

    /// Look up the kernel for `op` on `backend`, whichever backend that names.
    ///
    /// Resolves the `cpu.reference` backend from the oracle slot and any other
    /// backend from the per-backend table, so the executor can fetch exactly the
    /// kernel [`select_backend`](crate::executor::dispatch::select_backend) chose.
    #[must_use]
    pub fn get_backend(&self, op: &OpId, backend: &BackendId) -> Option<&dyn OpImplementation> {
        if backend.is_reference() {
            self.get(op)
        } else {
            self.by_backend
                .get(&(op.clone(), backend.clone()))
                .map(Box::as_ref)
        }
    }

    /// Whether a kernel is registered for `op` on the (non-reference) `backend`.
    ///
    /// For the reference backend, use [`contains`](Self::contains); this is the
    /// availability probe [`select_backend`](crate::executor::dispatch::select_backend)
    /// uses for optimized/GPU backends.
    #[must_use]
    pub fn has_backend(&self, op: &OpId, backend: &BackendId) -> bool {
        if backend.is_reference() {
            self.contains(op)
        } else {
            self.by_backend.contains_key(&(op.clone(), backend.clone()))
        }
    }

    /// Whether a `cpu.reference` implementation is registered for `id`.
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
