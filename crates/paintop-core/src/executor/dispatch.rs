//! Backend selection and dispatch policy (`plan.md` §6.1, §12.2, §12.3, §15).
//!
//! The reference executor ([`super::execute`]) dispatches every node through its
//! single `cpu.reference` oracle. M3 adds *more than one* implementation per op —
//! an optimized CPU kernel, a `wgpu` kernel — and a **policy** that the scheduler
//! consults to pick which backend serves each node. Ops never spawn their own
//! parallelism or pick their own device: they expose implementations, and the
//! scheduler selects one from policy (`plan.md` §12.2).
//!
//! This module owns the *selection substrate*, deliberately separate from the
//! [`ImplRegistry`] (which now indexes the
//! implementations) and the executor (which runs whatever is selected):
//!
//! * [`BackendId`] — the `<backend>.<name>` half of an [`ImplId`] that identifies a
//!   backend across op versions (e.g. `cpu.reference`, `cpu.optimized`,
//!   `wgpu.separable`);
//! * [`BackendPolicy`] — the ordered backend preference the scheduler applies, with
//!   [`BackendPolicy::reference`] as the default so existing plans stay
//!   byte-identical;
//! * [`select_backend`] — resolves a node's op to a concrete [`ImplId`] under the
//!   policy, falling back to the reference oracle when a preferred backend is
//!   unavailable for that op, or returning an explicit
//!   [`E_BACKEND_UNSUPPORTED`] error when a *required* backend cannot run it — never
//!   a silent wrong answer;
//! * [`BackendSelection`] — the resolved `(ImplId, fell_back)` the scheduler records
//!   in the evidence bundle, naming the backend that served the node.

use paintop_ir::{
    CPU_REFERENCE_BACKEND, CPU_REFERENCE_NAME, Error, ErrorClass, ImplId, OpId, OperationRegistry,
};

use crate::executor::op_impl::ImplRegistry;

/// Stable machine code: a backend the policy *required* cannot run an op, and the
/// policy does not allow falling back to the reference oracle.
pub const E_BACKEND_UNSUPPORTED: &str = "E_BACKEND_UNSUPPORTED";

/// The backend-identifying `<backend>.<name>` half of an [`ImplId`].
///
/// An [`ImplId`] is `<backend>.<name>@<version>` (e.g. `cpu.reference@1`). The
/// *version* is provenance/cache identity, not a different backend, so backend
/// selection keys on the `<backend>.<name>` pair alone: a policy that prefers
/// `cpu.optimized` matches `cpu.optimized@1`, `cpu.optimized@2`, … indifferently,
/// and the registry supplies whatever version the op declares.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BackendId {
    backend: String,
    name: String,
}

impl BackendId {
    /// Build a backend id from its `<backend>` and `<name>` segments.
    #[must_use]
    pub fn new(backend: impl Into<String>, name: impl Into<String>) -> Self {
        Self {
            backend: backend.into(),
            name: name.into(),
        }
    }

    /// The conventional `cpu.reference` oracle backend — always the safe fallback
    /// and the default policy.
    #[must_use]
    pub fn reference() -> Self {
        Self::new(CPU_REFERENCE_BACKEND, CPU_REFERENCE_NAME)
    }

    /// The backend segment, e.g. `cpu`.
    #[must_use]
    pub fn backend(&self) -> &str {
        &self.backend
    }

    /// The name segment, e.g. `reference`.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Whether this backend id is the `cpu.reference` oracle.
    #[must_use]
    pub fn is_reference(&self) -> bool {
        self.backend == CPU_REFERENCE_BACKEND && self.name == CPU_REFERENCE_NAME
    }

    /// Whether `impl_id` belongs to this backend (matches `<backend>.<name>`,
    /// ignoring the version).
    #[must_use]
    pub fn matches(&self, impl_id: &ImplId) -> bool {
        impl_id.backend() == self.backend && impl_id.name() == self.name
    }
}

impl std::fmt::Display for BackendId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}.{}", self.backend, self.name)
    }
}

impl From<&ImplId> for BackendId {
    fn from(id: &ImplId) -> Self {
        Self::new(id.backend(), id.name())
    }
}

/// The scheduler's ordered backend preference (`plan.md` §12.2).
///
/// The scheduler tries each backend in `prefer` order for a node's op; the first
/// one the op *declares and has a registered implementation for* serves the node.
/// When none of the preferred backends apply, behaviour depends on
/// [`require`](BackendPolicy::require):
///
/// * `require == false` (the default, and the only mode the reference policy uses):
///   fall back to the `cpu.reference` oracle, which every op declares — so a plan
///   always runs, just on the oracle for ops the optimized/GPU backends do not
///   cover.
/// * `require == true`: a preferred backend that cannot run the op is an explicit
///   [`E_BACKEND_UNSUPPORTED`] error, never a silent fallback. This is the mode the
///   differential harness uses to *force* a specific backend and prove it ran.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BackendPolicy {
    prefer: Vec<BackendId>,
    require: bool,
}

impl Default for BackendPolicy {
    fn default() -> Self {
        Self::reference()
    }
}

impl BackendPolicy {
    /// The default policy: the `cpu.reference` oracle for every op.
    ///
    /// Under this policy a plan executes byte-identically to the pre-M3 reference
    /// executor — the selection layer is transparent.
    #[must_use]
    pub fn reference() -> Self {
        Self {
            prefer: vec![BackendId::reference()],
            require: false,
        }
    }

    /// A policy that *prefers* the given backends in order, falling back to the
    /// `cpu.reference` oracle for any op none of them cover.
    ///
    /// The reference oracle is always appended as the final fallback, so the
    /// policy never strands an op. `require` is `false`.
    #[must_use]
    pub fn prefer(backends: impl IntoIterator<Item = BackendId>) -> Self {
        let mut prefer: Vec<BackendId> = backends.into_iter().collect();
        if !prefer.iter().any(BackendId::is_reference) {
            prefer.push(BackendId::reference());
        }
        Self {
            prefer,
            require: false,
        }
    }

    /// A policy that *requires* a single backend: if an op cannot run on it, a
    /// [`select_backend`] for that op is an explicit [`E_BACKEND_UNSUPPORTED`]
    /// error rather than a silent fallback.
    ///
    /// Used by the differential harness to force a specific backend and fail
    /// loudly if it is missing, so a "passing" differential can never be a
    /// reference-vs-reference comparison in disguise.
    #[must_use]
    pub fn require(backend: BackendId) -> Self {
        Self {
            prefer: vec![backend],
            require: true,
        }
    }

    /// The backends this policy prefers, in order.
    #[must_use]
    pub fn preferred(&self) -> &[BackendId] {
        &self.prefer
    }

    /// Whether this policy requires its preferred backend (no reference fallback).
    #[must_use]
    pub const fn is_required(&self) -> bool {
        self.require
    }
}

/// The resolved backend a node will be dispatched on.
///
/// Carries the concrete [`ImplId`] (with its provenance version) the scheduler
/// selected and whether the selection *fell back* to the reference oracle because
/// no preferred backend covered the op — both recorded in the evidence so the
/// trace honestly names the backend that served the node (`plan.md` §15).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BackendSelection {
    impl_id: ImplId,
    fell_back: bool,
}

impl BackendSelection {
    /// The selected implementation id (e.g. `wgpu.separable@1`).
    #[must_use]
    pub const fn impl_id(&self) -> &ImplId {
        &self.impl_id
    }

    /// The selected backend id (`<backend>.<name>`).
    #[must_use]
    pub fn backend(&self) -> BackendId {
        BackendId::from(&self.impl_id)
    }

    /// Whether the selection fell back to the reference oracle because no
    /// preferred backend covered the op.
    #[must_use]
    pub const fn fell_back(&self) -> bool {
        self.fell_back
    }

    /// Whether the selected implementation is the `cpu.reference` oracle.
    #[must_use]
    pub fn is_reference(&self) -> bool {
        self.backend().is_reference()
    }
}

/// Select the backend to dispatch `op` on under `policy`.
///
/// The op's `manifests` entry *declares* which implementations it exposes; the
/// `implementations` registry holds the ones actually *registered* (compiled in).
/// A backend is selectable for an op only when it is both declared and registered.
/// The function walks `policy.preferred()` in order and returns the first
/// selectable backend's concrete [`ImplId`].
///
/// When no preferred backend is selectable:
/// * if the policy is [`require`](BackendPolicy::require)d, returns an explicit
///   [`E_BACKEND_UNSUPPORTED`] error (the caller must fall back deliberately, never
///   silently);
/// * otherwise falls back to the `cpu.reference` oracle (which every op declares),
///   marking the selection as [`fell_back`](BackendSelection::fell_back).
///
/// # Errors
/// - [`E_BACKEND_UNSUPPORTED`] (class [`policy`](ErrorClass::Policy)) if a required
///   backend cannot run `op`, or if even the reference oracle is somehow absent for
///   `op` (a malformed registry).
pub fn select_backend(
    op: &OpId,
    manifests: &OperationRegistry,
    implementations: &ImplRegistry,
    policy: &BackendPolicy,
) -> Result<BackendSelection, Error> {
    let declared: &[ImplId] = manifests
        .get(op)
        .map_or(&[], |m| m.implementations.as_slice());

    // The first preferred backend is the op's "intended" backend; selecting any
    // later one (or the reference net below) is a recorded fall-back.
    let intended = policy.preferred().first();
    for backend in policy.preferred() {
        if let Some(impl_id) = resolve(op, backend, declared, implementations) {
            let fell_back = intended != Some(backend);
            return Ok(BackendSelection { impl_id, fell_back });
        }
    }

    if policy.is_required() {
        let wanted = policy
            .preferred()
            .first()
            .map_or_else(|| BackendId::reference().to_string(), ToString::to_string);
        return Err(Error::new(
            ErrorClass::Policy,
            E_BACKEND_UNSUPPORTED,
            format!(
                "operation {op} has no registered `{wanted}` implementation and the policy \
                 forbids falling back to the reference oracle"
            ),
        ));
    }

    // Final safety net: the reference oracle, which every op declares.
    let reference = BackendId::reference();
    if let Some(impl_id) = resolve(op, &reference, declared, implementations) {
        return Ok(BackendSelection {
            impl_id,
            fell_back: true,
        });
    }

    Err(Error::new(
        ErrorClass::Policy,
        E_BACKEND_UNSUPPORTED,
        format!("operation {op} has no registered `cpu.reference` oracle to dispatch"),
    ))
}

/// Resolve one backend to the op's concrete [`ImplId`], requiring it to be both
/// declared by the manifest and registered as an executable implementation.
fn resolve(
    op: &OpId,
    backend: &BackendId,
    declared: &[ImplId],
    implementations: &ImplRegistry,
) -> Option<ImplId> {
    // The declared id (carrying the op's provenance version for the backend).
    let declared_id = declared.iter().find(|id| backend.matches(id))?;
    // It must also have a registered compute kernel for that backend.
    if implementations.has_backend(op, backend) {
        Some(declared_id.clone())
    } else if backend.is_reference() && implementations.contains(op) {
        // The reference kernel is registered under the op's reference slot.
        Some(declared_id.clone())
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::{BackendId, BackendPolicy, E_BACKEND_UNSUPPORTED, select_backend};
    use crate::executor::op_impl::{ImplRegistry, InputValues, OpImplementation, OutputValues};
    use paintop_ir::{
        DeterminismTier, Error, ImplId, InputSpec, OperationManifest, OperationRegistry,
        OutputSpec, ResourceKind, RoiCategory, RoiPolicy, TestMetadata,
    };

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

    fn manifest(id: &str, impls: &[&str]) -> OperationManifest {
        OperationManifest {
            id: id.parse().unwrap(),
            impl_version: 1,
            summary: String::new(),
            determinism: DeterminismTier::Exact,
            roi: RoiPolicy {
                category: RoiCategory::Pointwise,
                halo_px: None,
            },
            inputs: vec![InputSpec {
                name: "image".to_owned(),
                kind: ResourceKind::Image,
                required: true,
                doc: String::new(),
            }],
            outputs: vec![OutputSpec {
                name: "image".to_owned(),
                kind: ResourceKind::Image,
                doc: String::new(),
            }],
            params: vec![],
            implementations: impls.iter().map(|s| s.parse().unwrap()).collect(),
            test: TestMetadata::default(),
        }
    }

    fn registry(manifests: Vec<OperationManifest>) -> OperationRegistry {
        OperationRegistry::from_manifests(manifests).unwrap()
    }

    #[test]
    fn default_policy_selects_the_reference_oracle() {
        let op = "filter.scale@1".parse().unwrap();
        let manifests = registry(vec![manifest("filter.scale@1", &["cpu.reference@1"])]);
        let mut impls = ImplRegistry::new();
        impls.register(op, Box::new(Noop)).unwrap();

        let op = "filter.scale@1".parse().unwrap();
        let sel = select_backend(&op, &manifests, &impls, &BackendPolicy::reference()).unwrap();
        assert!(sel.is_reference());
        assert!(
            !sel.fell_back(),
            "reference under the default policy is not a fallback"
        );
        assert_eq!(sel.impl_id().to_string(), "cpu.reference@1");
    }

    #[test]
    fn prefers_optimized_when_registered() {
        let op: ImplId = "cpu.optimized@1".parse().unwrap();
        let _ = op;
        let op_id = "filter.scale@1".parse().unwrap();
        let manifests = registry(vec![manifest(
            "filter.scale@1",
            &["cpu.reference@1", "cpu.optimized@1"],
        )]);
        let mut impls = ImplRegistry::new();
        impls.register(op_id, Box::new(Noop)).unwrap();
        let op_id = "filter.scale@1".parse().unwrap();
        impls
            .register_backend(op_id, &"cpu.optimized@1".parse().unwrap(), Box::new(Noop))
            .unwrap();

        let op_id = "filter.scale@1".parse().unwrap();
        let policy = BackendPolicy::prefer([BackendId::new("cpu", "optimized")]);
        let sel = select_backend(&op_id, &manifests, &impls, &policy).unwrap();
        assert_eq!(sel.impl_id().to_string(), "cpu.optimized@1");
        assert!(!sel.fell_back());
    }

    #[test]
    fn falls_back_to_reference_when_optimized_absent() {
        let op_id = "filter.scale@1".parse().unwrap();
        // Manifest declares only the reference; optimized is not available.
        let manifests = registry(vec![manifest("filter.scale@1", &["cpu.reference@1"])]);
        let mut impls = ImplRegistry::new();
        impls.register(op_id, Box::new(Noop)).unwrap();

        let op_id = "filter.scale@1".parse().unwrap();
        let policy = BackendPolicy::prefer([BackendId::new("cpu", "optimized")]);
        let sel = select_backend(&op_id, &manifests, &impls, &policy).unwrap();
        assert!(sel.is_reference());
        assert!(sel.fell_back(), "fell back to the oracle, recorded as such");
    }

    #[test]
    fn required_backend_absent_is_explicit_error() {
        let op_id = "filter.scale@1".parse().unwrap();
        let manifests = registry(vec![manifest("filter.scale@1", &["cpu.reference@1"])]);
        let mut impls = ImplRegistry::new();
        impls.register(op_id, Box::new(Noop)).unwrap();

        let op_id = "filter.scale@1".parse().unwrap();
        let policy = BackendPolicy::require(BackendId::new("wgpu", "separable"));
        let err = select_backend(&op_id, &manifests, &impls, &policy).unwrap_err();
        assert_eq!(err.code, E_BACKEND_UNSUPPORTED);
        assert_eq!(err.class, paintop_ir::ErrorClass::Policy);
    }

    #[test]
    fn declared_but_unregistered_backend_is_not_selected() {
        // The manifest declares wgpu.separable but no kernel is registered: the
        // backend is not selectable, so a preferring policy falls back.
        let op_id = "filter.scale@1".parse().unwrap();
        let manifests = registry(vec![manifest(
            "filter.scale@1",
            &["cpu.reference@1", "wgpu.separable@1"],
        )]);
        let mut impls = ImplRegistry::new();
        impls.register(op_id, Box::new(Noop)).unwrap();

        let op_id = "filter.scale@1".parse().unwrap();
        let policy = BackendPolicy::prefer([BackendId::new("wgpu", "separable")]);
        let sel = select_backend(&op_id, &manifests, &impls, &policy).unwrap();
        assert!(sel.is_reference());
        assert!(sel.fell_back());
    }

    #[test]
    fn backend_id_matches_ignore_version() {
        let b = BackendId::new("cpu", "optimized");
        assert!(b.matches(&"cpu.optimized@1".parse::<ImplId>().unwrap()));
        assert!(b.matches(&"cpu.optimized@7".parse::<ImplId>().unwrap()));
        assert!(!b.matches(&"cpu.reference@1".parse::<ImplId>().unwrap()));
    }
}
