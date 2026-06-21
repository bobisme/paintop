//! The `wgpu` GPU backend module tree (`plan.md` §12.3).
//!
//! Submodules:
//! * [`error`] — typed GPU errors mapping to stable paintop error codes;
//! * [`probe`] — adapter/device acquisition, capability probe, and the forced
//!   no-adapter fallback path (bn-3cg);
//! * [`resource`] — the GPU storage-texture/buffer resource model and
//!   dispatch-dimension + overflow validation (bn-3ov);
//! * [`pipeline`] — the compute-pipeline cache keyed by normalized fused
//!   expression + format (bn-2vi);
//! * [`fusion`] — pointwise fusion eligibility + the normalized fused-expression
//!   key (bn-t2v);
//! * [`pointwise`] — the WGSL fused-pointwise compute pipeline (bn-125).

pub mod error;
pub mod fusion;
pub mod pipeline;
pub mod pointwise;
pub mod probe;
pub mod readback;
pub mod resource;
pub mod separable;
pub mod splat;
pub mod splat_kernel;

use paintop_core::executor::dispatch::BackendId;

/// The `wgpu` backend family name, as it appears in an
/// [`ImplId`](paintop_ir::ImplId) (`wgpu.<name>@<v>`) and in the differential
/// harness's adapter gate (`paintop_testkit::differential::GpuAdapter`).
///
/// Backend *selection* keys on the `<backend>.<name>` pair (e.g. `wgpu.separable`);
/// this is the shared `<backend>` segment every GPU kernel registers under, so the
/// scheduler's policy and the harness's GPU gate agree on which ops are GPU-served.
pub const WGPU_BACKEND: &str = "wgpu";

/// A marker for the `wgpu` backend family.
///
/// The concrete per-op kernels are registered into the executor's
/// [`ImplRegistry`](paintop_core::executor::ImplRegistry) under
/// `wgpu.<name>@<v>` ids; this type centralizes the family identity (the
/// `<backend>` segment) so callers do not re-spell the `"wgpu"` string.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct GpuBackend;

impl GpuBackend {
    /// The backend-family name (`"wgpu"`).
    #[must_use]
    pub const fn family() -> &'static str {
        WGPU_BACKEND
    }

    /// The [`BackendId`] for a named `wgpu` kernel, e.g. `wgpu.separable`.
    ///
    /// `name` is the kernel's `<name>` segment (e.g. `"separable"`, `"pointwise"`);
    /// the returned id selects that kernel under a
    /// [`BackendPolicy`](paintop_core::executor::dispatch::BackendPolicy) regardless
    /// of the op's provenance version.
    #[must_use]
    pub fn backend_id(name: impl Into<String>) -> BackendId {
        BackendId::new(WGPU_BACKEND, name)
    }
}

#[cfg(test)]
mod tests {
    use super::{GpuBackend, WGPU_BACKEND};

    #[test]
    fn family_is_the_wgpu_segment() {
        assert_eq!(GpuBackend::family(), "wgpu");
        assert_eq!(WGPU_BACKEND, "wgpu");
    }

    #[test]
    fn backend_id_carries_the_family_and_name() {
        let id = GpuBackend::backend_id("separable");
        assert_eq!(id.backend(), "wgpu");
        assert_eq!(id.name(), "separable");
        assert_eq!(id.to_string(), "wgpu.separable");
    }
}
