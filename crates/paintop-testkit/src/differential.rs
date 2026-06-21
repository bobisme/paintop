//! The cross-backend differential harness (`plan.md` §11, §12;
//! `AGENT_VERIFICATION` §3).
//!
//! The `cpu.reference` implementation is the **semantic oracle** (`plan.md` §12.1):
//! every optimized-CPU or `wgpu` backend must reproduce its result within the op's
//! declared tolerance. This module runs an op on each *available* backend and
//! compares the output against the oracle, returning an [`DifferentialReport`] that
//! is `Pass` within tolerance, `Fail` with a saved [`ErrorMap`] otherwise, or
//! `Skipped` when a backend is not available on the host (e.g. a `wgpu` backend
//! with no GPU adapter — the harness skips it cleanly, never fails).
//!
//! Every later optimized/GPU bone drives this harness: it is the single place that
//! turns "the backend ran" into "the backend is *correct*", so a backend can never
//! ship a silently-wrong kernel.
//!
//! # Tolerance comes from the contract, not the test
//!
//! The comparison tolerance is derived from the op's declared
//! [`DeterminismTier`] (`plan.md` §4.9), read from the
//! manifest — never hard-coded per test. An [`Exact`](paintop_ir::DeterminismTier::Exact)
//! op must match the oracle **bit-for-bit**; a
//! [`Bounded`](paintop_ir::DeterminismTier::Bounded) /
//! [`Reproducible`](paintop_ir::DeterminismTier::Reproducible) op must match within
//! a bounded L∞ / L2 envelope. The bounded envelope's numeric magnitude can be
//! tightened per op via [`Tolerance`], but the *tier* — exact vs bounded — is the
//! contract.

use paintop_core::executor::value::ResourceValue;
use paintop_core::executor::{BackendId, ImplRegistry};
use paintop_ir::{DeterminismTier, OpId, OperationRegistry};

/// The numeric envelope a non-reference backend must match the oracle within.
///
/// Built from the op's [`DeterminismTier`]: an exact op tolerates **zero**
/// difference (bit-for-bit), a bounded/reproducible op tolerates a small L∞ (max
/// absolute) and L2 (root-mean-square) difference. The defaults are deliberately
/// tight; a specific op's suite can override them while keeping the tier contract.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Tolerance {
    /// Whether the backend must match the oracle bit-for-bit (the exact tier).
    pub exact: bool,
    /// The maximum tolerated per-sample absolute difference (L∞).
    pub max_abs: f32,
    /// The maximum tolerated root-mean-square difference over all samples (L2).
    pub max_rms: f32,
}

impl Tolerance {
    /// The exact-tier tolerance: bit-for-bit, zero slack.
    pub const EXACT: Self = Self {
        exact: true,
        max_abs: 0.0,
        max_rms: 0.0,
    };

    /// The default bounded-tier envelope: a tight absolute and RMS slack for
    /// floating-point reassociation across backends.
    pub const BOUNDED: Self = Self {
        exact: false,
        max_abs: 1.0e-4,
        max_rms: 1.0e-5,
    };

    /// The tolerance implied by an op's declared determinism tier.
    ///
    /// [`Exact`](DeterminismTier::Exact) → [`EXACT`](Self::EXACT) (bit-for-bit).
    /// [`Bounded`](DeterminismTier::Bounded) /
    /// [`Reproducible`](DeterminismTier::Reproducible) /
    /// [`Stochastic`](DeterminismTier::Stochastic) → [`BOUNDED`](Self::BOUNDED).
    #[must_use]
    pub const fn for_tier(tier: DeterminismTier) -> Self {
        match tier {
            DeterminismTier::Exact => Self::EXACT,
            // Bounded / Reproducible / Stochastic — and any future, non-exhaustive
            // tier — get the conservative bounded envelope.
            _ => Self::BOUNDED,
        }
    }

    /// Override the bounded envelope's magnitudes while keeping the tier's
    /// exact/bounded contract.
    #[must_use]
    pub const fn with_bounds(mut self, max_abs: f32, max_rms: f32) -> Self {
        self.max_abs = max_abs;
        self.max_rms = max_rms;
        self
    }
}

/// A per-sample difference map between a backend's output and the oracle, saved on
/// a differential failure for triage (`AGENT_VERIFICATION` §3.x: "saving an error
/// map on failure").
///
/// Carries the dominant statistics (max absolute and RMS difference, and the
/// flat index of the worst sample) plus the full per-sample absolute-difference
/// buffer so a caller can localize where a kernel diverges.
#[derive(Debug, Clone, PartialEq)]
pub struct ErrorMap {
    /// The flat sample index of the largest absolute difference.
    pub argmax: usize,
    /// The largest per-sample absolute difference (L∞).
    pub max_abs: f32,
    /// The root-mean-square difference over all samples (L2).
    pub rms: f32,
    /// The per-sample absolute differences, in the values' row-major order. Empty
    /// when the two values' shapes differ (a shape mismatch is reported via
    /// [`shape_mismatch`](Self::shape_mismatch) instead).
    pub abs_diff: Vec<f32>,
    /// Set when the backend's output shape (extent / channel count) differs from
    /// the oracle's, which is always a failure regardless of numeric tolerance.
    pub shape_mismatch: bool,
}

impl ErrorMap {
    /// Build the per-sample error map of `candidate` against the `oracle` value.
    ///
    /// A shape mismatch yields a map flagged [`shape_mismatch`](Self::shape_mismatch)
    /// with infinite magnitudes and an empty per-sample buffer, so it can never
    /// pass a finite tolerance.
    #[must_use]
    pub fn compute(oracle: &ResourceValue, candidate: &ResourceValue) -> Self {
        if oracle.extent() != candidate.extent()
            || oracle.channels() != candidate.channels()
            || oracle.samples().len() != candidate.samples().len()
        {
            return Self {
                argmax: 0,
                max_abs: f32::INFINITY,
                rms: f32::INFINITY,
                abs_diff: Vec::new(),
                shape_mismatch: true,
            };
        }

        let mut abs_diff = Vec::with_capacity(oracle.samples().len());
        let mut max_abs = 0.0_f32;
        let mut argmax = 0_usize;
        let mut sum_sq = 0.0_f64;
        for (i, (o, c)) in oracle
            .samples()
            .iter()
            .zip(candidate.samples().iter())
            .enumerate()
        {
            let d = (o - c).abs();
            abs_diff.push(d);
            if d > max_abs {
                max_abs = d;
                argmax = i;
            }
            sum_sq = f64::from(d).mul_add(f64::from(d), sum_sq);
        }
        #[expect(
            clippy::cast_precision_loss,
            reason = "sample counts well below 2^52; the divisor is exact in f64"
        )]
        let n = abs_diff.len().max(1) as f64;
        #[expect(
            clippy::cast_possible_truncation,
            reason = "rms collapses to an f32 summary; full precision is in abs_diff"
        )]
        let rms = (sum_sq / n).sqrt() as f32;

        Self {
            argmax,
            max_abs,
            rms,
            abs_diff,
            shape_mismatch: false,
        }
    }

    /// Whether this error map is within `tolerance`.
    ///
    /// A shape mismatch is never within tolerance. An exact-tier tolerance requires
    /// every per-sample difference to be exactly zero; a bounded-tier tolerance
    /// requires both the L∞ and L2 statistics to be within their envelopes.
    #[must_use]
    pub fn within(&self, tolerance: &Tolerance) -> bool {
        if self.shape_mismatch {
            return false;
        }
        if tolerance.exact {
            // Bit-for-bit: every per-sample abs diff is exactly +0.0 (abs() never
            // yields -0.0, so a zero `to_bits` is an exact-zero L∞).
            return self.max_abs.to_bits() == 0;
        }
        let abs_ok = self.max_abs <= tolerance.max_abs;
        let rms_ok = self.rms <= tolerance.max_rms;
        abs_ok && rms_ok
    }
}

/// The outcome of comparing one backend against the oracle.
#[derive(Debug, Clone, PartialEq)]
pub enum Outcome {
    /// The backend reproduced the oracle within tolerance.
    Pass {
        /// The backend's error map vs the oracle (within tolerance).
        error_map: ErrorMap,
    },
    /// The backend diverged from the oracle beyond tolerance; the error map is
    /// saved for triage.
    Fail {
        /// The saved error map locating the divergence.
        error_map: ErrorMap,
    },
    /// The backend was not available on this host (e.g. a `wgpu` backend with no
    /// GPU adapter), so it was skipped — not failed.
    Skipped {
        /// Why the backend was skipped (e.g. "no GPU adapter").
        reason: String,
    },
}

impl Outcome {
    /// Whether this outcome is a pass.
    #[must_use]
    pub const fn is_pass(&self) -> bool {
        matches!(self, Self::Pass { .. })
    }

    /// Whether this outcome is a failure.
    #[must_use]
    pub const fn is_fail(&self) -> bool {
        matches!(self, Self::Fail { .. })
    }

    /// Whether this outcome is a clean skip.
    #[must_use]
    pub const fn is_skipped(&self) -> bool {
        matches!(self, Self::Skipped { .. })
    }

    /// The saved error map, if this outcome carries one (pass or fail).
    #[must_use]
    pub const fn error_map(&self) -> Option<&ErrorMap> {
        match self {
            Self::Pass { error_map } | Self::Fail { error_map } => Some(error_map),
            Self::Skipped { .. } => None,
        }
    }
}

/// The result of differentially checking one op's backends against the oracle.
#[derive(Debug, Clone, PartialEq)]
pub struct DifferentialReport {
    /// The op that was checked.
    pub op: OpId,
    /// The tolerance applied (derived from the op's tier).
    pub tolerance: Tolerance,
    /// The per-backend outcome, in the order the backends were checked.
    pub backends: Vec<(BackendId, Outcome)>,
}

impl DifferentialReport {
    /// Whether every checked (non-skipped) backend passed.
    ///
    /// A report with only skipped backends passes vacuously (nothing diverged).
    #[must_use]
    pub fn all_pass(&self) -> bool {
        self.backends.iter().all(|(_, o)| !o.is_fail())
    }

    /// The backends that failed, with their saved error maps.
    #[must_use]
    pub fn failures(&self) -> Vec<(&BackendId, &ErrorMap)> {
        self.backends
            .iter()
            .filter_map(|(b, o)| match o {
                Outcome::Fail { error_map } => Some((b, error_map)),
                _ => None,
            })
            .collect()
    }
}

/// Whether a backend should be skipped on this host, e.g. a `wgpu` backend with no
/// GPU adapter (`plan.md` §19 M3: GPU cases skip cleanly when no adapter).
///
/// The harness is GPU-agnostic: it does not link `wgpu` itself. Callers supply an
/// availability predicate — typically "is a GPU adapter present?" — so the same
/// harness runs GPU-less (every `wgpu` backend skipped) on CI and GPU-resident on a
/// host with an adapter.
pub trait BackendAvailability {
    /// Whether `backend` can run on this host. Return `None` if available, or
    /// `Some(reason)` to skip it cleanly.
    fn unavailable(&self, backend: &BackendId) -> Option<String>;
}

/// An availability policy that treats every backend as available — the trivial
/// host where no backend needs probing (used for the pure-CPU optimized path).
#[derive(Debug, Clone, Copy, Default)]
pub struct AllAvailable;

impl BackendAvailability for AllAvailable {
    fn unavailable(&self, _backend: &BackendId) -> Option<String> {
        None
    }
}

/// An availability policy that skips every `wgpu`-backend op when no GPU adapter is
/// present.
///
/// `adapter_present` is the host probe result (a caller probes `wgpu::Instance` for
/// an adapter once and passes the boolean here). When `false`, any backend whose id
/// is in the `wgpu` family is reported unavailable with a clean reason, so the
/// harness skips it instead of failing.
#[derive(Debug, Clone, Copy)]
pub struct GpuAdapter {
    /// Whether a GPU adapter was found on this host.
    pub adapter_present: bool,
}

impl GpuAdapter {
    /// The `wgpu` backend family this policy gates.
    pub const WGPU_BACKEND: &'static str = "wgpu";

    /// Build the policy from a host adapter probe result.
    #[must_use]
    pub const fn new(adapter_present: bool) -> Self {
        Self { adapter_present }
    }
}

impl BackendAvailability for GpuAdapter {
    fn unavailable(&self, backend: &BackendId) -> Option<String> {
        if backend.backend() == Self::WGPU_BACKEND && !self.adapter_present {
            Some("no GPU adapter present; wgpu backend skipped".to_owned())
        } else {
            None
        }
    }
}

/// An error from running an op kernel during a differential check.
#[derive(Debug, thiserror::Error)]
pub enum DifferentialError {
    /// The op has no manifest in the registry, so its tier/backends are unknown.
    #[error("operation `{0}` is not registered, so its tolerance tier is unknown")]
    OpNotRegistered(OpId),
    /// The op has no registered `cpu.reference` oracle kernel to compare against.
    #[error("operation `{0}` has no registered cpu.reference oracle to compare against")]
    NoOracle(OpId),
    /// A backend kernel (oracle or candidate) raised while computing the op.
    #[error("operation `{op}` failed on backend `{backend}`: {source}")]
    Kernel {
        /// The op that failed.
        op: OpId,
        /// The backend whose kernel failed.
        backend: BackendId,
        /// The underlying kernel error (boxed to keep the error type small).
        #[source]
        source: Box<paintop_ir::Error>,
    },
    /// A backend kernel did not produce the requested output port.
    #[error("operation `{op}` on backend `{backend}` produced no `{port}` output")]
    MissingOutput {
        /// The op.
        op: OpId,
        /// The backend.
        backend: BackendId,
        /// The output port that was expected.
        port: String,
    },
}

/// The inputs and params one differential op invocation needs.
pub struct OpInvocation<'a> {
    /// The input values, keyed by input port name.
    pub inputs: &'a paintop_core::executor::InputValues,
    /// The resolved params as canonical JSON.
    pub params: &'a serde_json::Value,
    /// The output port to compare across backends.
    pub output_port: &'a str,
}

/// Run `op` on every declared backend and compare each against the `cpu.reference`
/// oracle within the op's tier tolerance (`plan.md` §11, §12).
///
/// The oracle output is computed once from `implementations`' reference kernel.
/// Each *other* backend the op declares (via its manifest) and has a registered
/// kernel for is then run and compared; a backend `availability` reports
/// unavailable is [`Skipped`](Outcome::Skipped) cleanly. The result is a
/// [`DifferentialReport`] whose failures carry a saved [`ErrorMap`].
///
/// `tolerance_override` lets a specific op's suite tighten the bounded envelope; it
/// does not change the exact/bounded *tier*, which is read from the manifest.
///
/// # Errors
/// - [`DifferentialError::OpNotRegistered`] if `op` has no manifest.
/// - [`DifferentialError::NoOracle`] if no `cpu.reference` kernel is registered.
/// - [`DifferentialError::Kernel`] / [`DifferentialError::MissingOutput`] if a
///   kernel raises or under-produces while computing the op.
#[expect(
    clippy::result_large_err,
    reason = "DifferentialError carries owned op/backend ids for triage; this is a \
              test harness where the error path is cold and ergonomics win over size"
)]
pub fn differential_check<A: BackendAvailability>(
    op: &OpId,
    manifests: &OperationRegistry,
    implementations: &ImplRegistry,
    invocation: &OpInvocation<'_>,
    availability: &A,
    tolerance_override: Option<Tolerance>,
) -> Result<DifferentialReport, DifferentialError> {
    let manifest = manifests
        .get(op)
        .map_err(|_| DifferentialError::OpNotRegistered(op.clone()))?;

    let base = Tolerance::for_tier(manifest.determinism);
    // An override may tighten the bounded envelope but never weaken the tier: an
    // exact op stays bit-for-bit regardless of an override.
    let tolerance = match (base.exact, tolerance_override) {
        (true, _) | (false, None) => base,
        (false, Some(o)) => Tolerance { exact: false, ..o },
    };

    // The oracle output (computed once).
    let reference = BackendId::reference();
    let oracle = run_backend(op, implementations, &reference, invocation)?
        .ok_or_else(|| DifferentialError::NoOracle(op.clone()))?;

    // Each *non-reference* declared backend, compared against the oracle.
    let mut backends = Vec::new();
    for impl_id in &manifest.implementations {
        let backend = BackendId::from(impl_id);
        if backend.is_reference() {
            continue;
        }
        if let Some(reason) = availability.unavailable(&backend) {
            backends.push((backend, Outcome::Skipped { reason }));
            continue;
        }
        let Some(candidate) = run_backend(op, implementations, &backend, invocation)? else {
            // Declared but not registered on this host: skip cleanly rather than
            // fail — the kernel was simply not compiled in.
            backends.push((
                backend,
                Outcome::Skipped {
                    reason: "backend declared but no kernel registered on this host".to_owned(),
                },
            ));
            continue;
        };
        let error_map = ErrorMap::compute(&oracle, &candidate);
        let outcome = if error_map.within(&tolerance) {
            Outcome::Pass { error_map }
        } else {
            Outcome::Fail { error_map }
        };
        backends.push((backend, outcome));
    }

    Ok(DifferentialReport {
        op: op.clone(),
        tolerance,
        backends,
    })
}

/// Run `op` on one backend, returning the requested output value, or `None` when no
/// kernel is registered for that backend.
#[expect(
    clippy::result_large_err,
    reason = "shares DifferentialError with the public entry point; the error path is cold"
)]
fn run_backend(
    op: &OpId,
    implementations: &ImplRegistry,
    backend: &BackendId,
    invocation: &OpInvocation<'_>,
) -> Result<Option<ResourceValue>, DifferentialError> {
    let Some(kernel) = implementations.get_backend(op, backend) else {
        return Ok(None);
    };
    let produced = kernel
        .compute(invocation.inputs, invocation.params)
        .map_err(|source| DifferentialError::Kernel {
            op: op.clone(),
            backend: backend.clone(),
            source: Box::new(source),
        })?;
    let value = produced
        .get(invocation.output_port)
        .cloned()
        .ok_or_else(|| DifferentialError::MissingOutput {
            op: op.clone(),
            backend: backend.clone(),
            port: invocation.output_port.to_owned(),
        })?;
    Ok(Some(value))
}

#[cfg(test)]
mod tests {
    use super::{
        AllAvailable, BackendId, DifferentialError, ErrorMap, GpuAdapter, OpInvocation, Tolerance,
        differential_check,
    };
    use paintop_core::executor::value::ResourceValue;
    use paintop_core::executor::{ImplRegistry, InputValues, OpImplementation, OutputValues};
    use paintop_ir::{
        AlphaRepresentation, ChannelLayout, ColorEncoding, ColorRange, CoordinateConvention,
        DeterminismTier, Error, Extent, ImageDescriptor, InputSpec, OperationManifest,
        OperationRegistry, OutputSpec, ResourceDescriptor, ResourceKind, RoiCategory, RoiPolicy,
        ScalarType, SemanticRole, TestMetadata,
    };

    const EXTENT: Extent = Extent::new(2, 2);

    fn descriptor() -> ResourceDescriptor {
        ResourceDescriptor::Image(ImageDescriptor {
            extent: EXTENT,
            layout: ChannelLayout::Rgba,
            scalar: ScalarType::F32,
            color: ColorEncoding::LinearSrgb,
            range: ColorRange::SceneReferred,
            alpha: AlphaRepresentation::Premultiplied,
            coordinates: CoordinateConvention::PixelCenterUpperLeft,
            semantic: SemanticRole::Color,
        })
    }

    fn image(fill: f32) -> ResourceValue {
        let len = (EXTENT.width * EXTENT.height * 4) as usize;
        ResourceValue::new(descriptor(), 4, vec![fill; len]).expect("sized")
    }

    /// A pointwise scale kernel: out = in * factor + bias. The oracle uses
    /// `(2.0, 0.0)`; candidate kernels perturb it to model an optimized backend.
    struct Scale {
        factor: f32,
        bias: f32,
    }
    impl OpImplementation for Scale {
        fn compute(
            &self,
            inputs: &InputValues,
            _params: &serde_json::Value,
        ) -> Result<OutputValues, Error> {
            let v = inputs.get("image").expect("image input");
            let samples: Vec<f32> = v
                .samples()
                .iter()
                .map(|s| s.mul_add(self.factor, self.bias))
                .collect();
            let mut out = OutputValues::new();
            out.insert(
                "image".to_owned(),
                ResourceValue::new(*v.descriptor(), v.channels(), samples).expect("sized"),
            );
            Ok(out)
        }
    }

    fn manifest(tier: DeterminismTier, impls: &[&str]) -> OperationManifest {
        OperationManifest {
            id: "filter.scale@1".parse().expect("id"),
            impl_version: 1,
            summary: String::new(),
            determinism: tier,
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
            implementations: impls.iter().map(|s| s.parse().expect("impl")).collect(),
            test: TestMetadata::default(),
        }
    }

    fn registry(tier: DeterminismTier, impls: &[&str]) -> OperationRegistry {
        OperationRegistry::from_manifests([manifest(tier, impls)]).expect("registry")
    }

    #[expect(
        clippy::result_large_err,
        reason = "forwards the harness's DifferentialError; the error path is cold"
    )]
    fn run(
        manifests: &OperationRegistry,
        impls: &ImplRegistry,
        availability: &impl super::BackendAvailability,
    ) -> Result<super::DifferentialReport, DifferentialError> {
        let op = "filter.scale@1".parse().expect("op");
        let mut inputs = InputValues::new();
        inputs.insert("image".to_owned(), image(0.5));
        let params = serde_json::Value::Null;
        let invocation = OpInvocation {
            inputs: &inputs,
            params: &params,
            output_port: "image",
        };
        differential_check(&op, manifests, impls, &invocation, availability, None)
    }

    #[test]
    fn exact_tier_optimized_matching_oracle_passes() {
        let manifests = registry(
            DeterminismTier::Exact,
            &["cpu.reference@1", "cpu.optimized@1"],
        );
        let mut impls = ImplRegistry::new();
        let op = "filter.scale@1".parse().expect("op");
        impls
            .register(
                op,
                Box::new(Scale {
                    factor: 2.0,
                    bias: 0.0,
                }),
            )
            .expect("ref");
        let op = "filter.scale@1".parse().expect("op");
        // The optimized kernel computes the identical result.
        impls
            .register_backend(
                op,
                &"cpu.optimized@1".parse().expect("impl"),
                Box::new(Scale {
                    factor: 2.0,
                    bias: 0.0,
                }),
            )
            .expect("opt");

        let report = run(&manifests, &impls, &AllAvailable).expect("ok");
        assert!(report.tolerance.exact, "exact tier => bit-for-bit");
        assert!(report.all_pass());
        let (backend, outcome) = &report.backends[0];
        assert_eq!(backend, &BackendId::new("cpu", "optimized"));
        assert!(outcome.is_pass());
        assert_eq!(outcome.error_map().expect("map").max_abs.to_bits(), 0);
    }

    #[test]
    fn wrong_optimized_kernel_fails_with_an_error_map() {
        let manifests = registry(
            DeterminismTier::Exact,
            &["cpu.reference@1", "cpu.optimized@1"],
        );
        let mut impls = ImplRegistry::new();
        let op = "filter.scale@1".parse().expect("op");
        impls
            .register(
                op,
                Box::new(Scale {
                    factor: 2.0,
                    bias: 0.0,
                }),
            )
            .expect("ref");
        let op = "filter.scale@1".parse().expect("op");
        // An intentionally-wrong optimized kernel (bias off).
        impls
            .register_backend(
                op,
                &"cpu.optimized@1".parse().expect("impl"),
                Box::new(Scale {
                    factor: 2.0,
                    bias: 0.1,
                }),
            )
            .expect("opt");

        let report = run(&manifests, &impls, &AllAvailable).expect("ok");
        assert!(!report.all_pass(), "wrong kernel must fail");
        let failures = report.failures();
        assert_eq!(failures.len(), 1);
        let (backend, error_map) = failures[0];
        assert_eq!(backend, &BackendId::new("cpu", "optimized"));
        // The error map localizes the divergence (bias of 0.1 on every sample).
        assert!(
            (error_map.max_abs - 0.1).abs() < 1.0e-6,
            "{}",
            error_map.max_abs
        );
        assert!(!error_map.abs_diff.is_empty(), "per-sample map is saved");
    }

    #[test]
    fn bounded_tier_tolerates_tiny_optimized_drift() {
        let manifests = registry(
            DeterminismTier::Bounded,
            &["cpu.reference@1", "cpu.optimized@1"],
        );
        let mut impls = ImplRegistry::new();
        let op = "filter.scale@1".parse().expect("op");
        impls
            .register(
                op,
                Box::new(Scale {
                    factor: 2.0,
                    bias: 0.0,
                }),
            )
            .expect("ref");
        let op = "filter.scale@1".parse().expect("op");
        // A drift of 1e-6, well within the bounded envelope but NOT bit-exact.
        impls
            .register_backend(
                op,
                &"cpu.optimized@1".parse().expect("impl"),
                Box::new(Scale {
                    factor: 2.0,
                    bias: 1.0e-6,
                }),
            )
            .expect("opt");

        let report = run(&manifests, &impls, &AllAvailable).expect("ok");
        assert!(!report.tolerance.exact, "bounded tier is not bit-for-bit");
        assert!(
            report.all_pass(),
            "tiny drift is within the bounded envelope"
        );
        assert!(report.backends[0].1.is_pass());
    }

    #[test]
    fn wgpu_backend_skips_cleanly_with_no_adapter() {
        let manifests = registry(
            DeterminismTier::Bounded,
            &["cpu.reference@1", "wgpu.separable@1"],
        );
        let mut impls = ImplRegistry::new();
        let op = "filter.scale@1".parse().expect("op");
        impls
            .register(
                op,
                Box::new(Scale {
                    factor: 2.0,
                    bias: 0.0,
                }),
            )
            .expect("ref");
        // No wgpu kernel registered; with no adapter the backend skips before that
        // even matters.

        let report = run(&manifests, &impls, &GpuAdapter::new(false)).expect("ok");
        assert!(report.all_pass(), "a skipped backend is not a failure");
        let (backend, outcome) = &report.backends[0];
        assert_eq!(backend, &BackendId::new("wgpu", "separable"));
        assert!(outcome.is_skipped());
        assert!(outcome.error_map().is_none());
    }

    #[test]
    fn declared_but_unregistered_backend_skips_not_fails() {
        // The op declares cpu.optimized but no kernel is registered: the harness
        // skips it cleanly (the kernel was simply not compiled in here).
        let manifests = registry(
            DeterminismTier::Exact,
            &["cpu.reference@1", "cpu.optimized@1"],
        );
        let mut impls = ImplRegistry::new();
        let op = "filter.scale@1".parse().expect("op");
        impls
            .register(
                op,
                Box::new(Scale {
                    factor: 2.0,
                    bias: 0.0,
                }),
            )
            .expect("ref");

        let report = run(&manifests, &impls, &AllAvailable).expect("ok");
        assert!(report.all_pass());
        assert!(report.backends[0].1.is_skipped());
    }

    #[test]
    fn no_oracle_registered_is_an_explicit_error() {
        let manifests = registry(
            DeterminismTier::Exact,
            &["cpu.reference@1", "cpu.optimized@1"],
        );
        // Empty impl registry: no oracle kernel at all.
        let impls = ImplRegistry::new();
        let err = run(&manifests, &impls, &AllAvailable).expect_err("no oracle");
        assert!(matches!(err, DifferentialError::NoOracle(_)));
    }

    #[test]
    fn tolerance_override_cannot_weaken_an_exact_tier() {
        // Even with a loose override, an exact-tier op stays bit-for-bit.
        let t = Tolerance::for_tier(DeterminismTier::Exact);
        assert!(t.exact);
        // A shape mismatch never passes any tolerance.
        let oracle = image(0.5);
        let other = {
            let len = (3 * 3 * 4) as usize;
            ResourceValue::new(
                ResourceDescriptor::Image(ImageDescriptor {
                    extent: Extent::new(3, 3),
                    ..match descriptor() {
                        ResourceDescriptor::Image(d) => d,
                        _ => unreachable!(),
                    }
                }),
                4,
                vec![0.5; len],
            )
            .expect("sized")
        };
        let map = ErrorMap::compute(&oracle, &other);
        assert!(map.shape_mismatch);
        assert!(!map.within(&Tolerance::BOUNDED));
    }

    #[test]
    fn gpu_adapter_present_does_not_skip_wgpu() {
        let policy = GpuAdapter::new(true);
        assert!(
            super::BackendAvailability::unavailable(&policy, &BackendId::new("wgpu", "separable"))
                .is_none()
        );
        // CPU backends are never gated by the GPU adapter probe.
        assert!(
            super::BackendAvailability::unavailable(&policy, &BackendId::new("cpu", "optimized"))
                .is_none()
        );
    }
}
