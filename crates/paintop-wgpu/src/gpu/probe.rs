//! Adapter/device probe, capability probe, and the forced-fallback path
//! (`plan.md` §12.3, §19 M3 criterion 4).
//!
//! A GPU is a *runtime* resource: it may be present, absent, or deliberately
//! disabled. This module turns that uncertainty into an explicit, typed outcome —
//! never a panic, never a silent wrong result:
//!
//! * [`probe`] acquires a `wgpu` instance, requests a compatible adapter, and (if
//!   found) a device + queue, returning a live [`GpuContext`]. When no compatible
//!   adapter exists it returns a typed [`GpuUnavailable`].
//! * [`probe_forced`] takes a `force_no_adapter` flag so the **fallback path is
//!   testable even on a host that has a GPU** (`plan.md` §19 M3 criterion 4: GPU
//!   absence is exercised whether or not a GPU is present).
//! * [`GpuUnavailable::into_unsupported`] converts the soft "no GPU" state into the
//!   explicit [`E_GPU_UNAVAILABLE`](super::error::E_GPU_UNAVAILABLE) error a caller
//!   raises when the GPU was *required* (no reference fallback allowed).
//!
//! The adapter's identity ([`AdapterIdentity`]) is captured at probe time so it can
//! be surfaced into the evidence bundle's [`Platform`](paintop_core::evidence::Platform)
//! (`plan.md` §12.3: "expose adapter/device identity in evidence").

use super::error::GpuError;

/// Identity of the GPU adapter a [`GpuContext`] was built on.
///
/// Captured from `wgpu`'s [`AdapterInfo`](wgpu::AdapterInfo) at probe time and
/// surfaced into the evidence bundle so a trace honestly records *which* device
/// served a GPU dispatch (`plan.md` §12.3, §15). All fields are **provenance**: they
/// describe this execution's hardware and are excluded from a plan's semantic
/// identity (`AGENT_VERIFICATION` §5.1).
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct AdapterIdentity {
    /// The adapter's human-readable name (e.g. `"NVIDIA GeForce RTX 3090"`).
    pub name: String,
    /// The graphics backend the adapter uses (e.g. `"vulkan"`, `"metal"`).
    pub backend: String,
    /// The device category (e.g. `"discrete-gpu"`, `"integrated-gpu"`, `"cpu"`).
    pub device_type: String,
    /// The backend-specific vendor id (a PCI vendor id for most backends).
    pub vendor: u32,
    /// The backend-specific device id.
    pub device: u32,
    /// The driver name, if the backend reports one.
    pub driver: String,
    /// Free-form driver version / build info, if the backend reports it.
    pub driver_info: String,
}

impl AdapterIdentity {
    /// Build an identity record from `wgpu`'s adapter info.
    #[must_use]
    pub fn from_info(info: &wgpu::AdapterInfo) -> Self {
        Self {
            name: info.name.clone(),
            backend: info.backend.to_string(),
            device_type: device_type_str(info.device_type).to_owned(),
            vendor: info.vendor,
            device: info.device,
            driver: info.driver.clone(),
            driver_info: info.driver_info.clone(),
        }
    }

    /// The evidence-bundle `gpu` string: a stable, human-readable device
    /// description combining the backend and adapter name (`plan.md` §12.3).
    #[must_use]
    pub fn evidence_gpu(&self) -> String {
        format!("{} ({})", self.name, self.backend)
    }

    /// The evidence-bundle `driver` string, if the backend reported one.
    ///
    /// Returns `None` (not an empty placeholder) when the driver is unknown, so a
    /// clean `Platform` object is produced — matching the
    /// [`Platform`](paintop_core::evidence::Platform) convention.
    #[must_use]
    pub fn evidence_driver(&self) -> Option<String> {
        match (self.driver.is_empty(), self.driver_info.is_empty()) {
            (true, true) => None,
            (false, true) => Some(self.driver.clone()),
            (true, false) => Some(self.driver_info.clone()),
            (false, false) => Some(format!("{} ({})", self.driver, self.driver_info)),
        }
    }

    /// The host [`Platform`](paintop_core::evidence::Platform) with this adapter's
    /// `gpu`/`driver` provenance fields filled in.
    ///
    /// Starts from [`Platform::current`](paintop_core::evidence::Platform::current)
    /// (os/arch from the build target) and records this device's identity, so a GPU
    /// dispatch's evidence honestly names the backend that served it (`plan.md`
    /// §12.3, §15). When no GPU was used, callers leave `Platform::current` as-is and
    /// the device fields stay absent.
    #[must_use]
    pub fn to_platform(&self) -> paintop_core::evidence::Platform {
        paintop_core::evidence::Platform {
            gpu: Some(self.evidence_gpu()),
            driver: self.evidence_driver(),
            ..paintop_core::evidence::Platform::current()
        }
    }
}

/// Map `wgpu`'s [`DeviceType`](wgpu::DeviceType) to a stable kebab-case string.
const fn device_type_str(ty: wgpu::DeviceType) -> &'static str {
    match ty {
        wgpu::DeviceType::Other => "other",
        wgpu::DeviceType::IntegratedGpu => "integrated-gpu",
        wgpu::DeviceType::DiscreteGpu => "discrete-gpu",
        wgpu::DeviceType::VirtualGpu => "virtual-gpu",
        wgpu::DeviceType::Cpu => "cpu",
    }
}

/// A live GPU device + queue paintop can dispatch on, plus the probed identity and
/// limits.
///
/// Holds the `wgpu` [`Device`](wgpu::Device) and [`Queue`](wgpu::Queue) (both cheap
/// `Arc`-backed handles), the captured [`AdapterIdentity`], and the device
/// [`Limits`](wgpu::Limits) the resource model validates dispatches against
/// (bn-3ov). Acquired once per run and shared; the executor dispatches GPU nodes on
/// it under the scheduler's policy.
#[derive(Debug)]
pub struct GpuContext {
    device: wgpu::Device,
    queue: wgpu::Queue,
    identity: AdapterIdentity,
    limits: wgpu::Limits,
}

impl GpuContext {
    /// The GPU device handle.
    #[must_use]
    pub const fn device(&self) -> &wgpu::Device {
        &self.device
    }

    /// The GPU queue handle.
    #[must_use]
    pub const fn queue(&self) -> &wgpu::Queue {
        &self.queue
    }

    /// The identity of the adapter this context was built on.
    #[must_use]
    pub const fn identity(&self) -> &AdapterIdentity {
        &self.identity
    }

    /// The device's reported limits (used by dispatch-dimension validation).
    #[must_use]
    pub const fn limits(&self) -> &wgpu::Limits {
        &self.limits
    }
}

/// The typed "no compatible GPU" outcome a probe returns instead of panicking.
///
/// A *soft* state: a caller that permits fallback runs the `cpu.reference` oracle
/// and never escalates. A caller that *required* the GPU converts this into the
/// explicit [`E_GPU_UNAVAILABLE`](super::error::E_GPU_UNAVAILABLE) error via
/// [`into_unsupported`](Self::into_unsupported).
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{reason}")]
pub struct GpuUnavailable {
    /// Why no compatible GPU is available on this host.
    pub reason: String,
}

impl GpuUnavailable {
    /// Build an unavailable outcome with a human-readable reason.
    #[must_use]
    pub fn new(reason: impl Into<String>) -> Self {
        Self {
            reason: reason.into(),
        }
    }

    /// Escalate this soft outcome to the explicit
    /// [`GpuError::Unavailable`] a caller raises when the GPU was *required* (the
    /// policy forbids reference fallback).
    #[must_use]
    pub fn into_unsupported(self) -> GpuError {
        GpuError::Unavailable {
            reason: self.reason,
        }
    }
}

/// The graphics backends paintop's probe will consider, in `wgpu`'s default
/// priority order. Vulkan/Metal/DX12 are the compute-capable native backends; the
/// no-op backend is intentionally excluded so a probe never "succeeds" onto a
/// non-functional stub device that would silently produce wrong results.
fn compute_backends() -> wgpu::Backends {
    wgpu::Backends::VULKAN | wgpu::Backends::METAL | wgpu::Backends::DX12
}

/// Probe the host for a usable GPU device.
///
/// Creates a `wgpu` instance over the native compute backends, requests a
/// high-performance compatible adapter, and acquires a device + queue with default
/// limits. Returns a live [`GpuContext`] on success.
///
/// # Errors
/// Returns [`GpuUnavailable`] when no compatible adapter is present, or when an
/// adapter is found but a device cannot be acquired — both clean, recoverable
/// states a caller turns into a `cpu.reference` fallback or an explicit
/// unsupported error.
pub fn probe() -> Result<GpuContext, GpuUnavailable> {
    probe_forced(false)
}

/// Probe with an optional forced no-adapter outcome, for exercising the fallback
/// path even on a host that *has* a GPU (`plan.md` §19 M3 criterion 4).
///
/// When `force_no_adapter` is `true`, the probe short-circuits to
/// [`GpuUnavailable`] **before touching `wgpu`**, deterministically simulating a
/// GPU-less host. When `false`, it behaves exactly like [`probe`].
///
/// # Errors
/// Returns [`GpuUnavailable`] when forced, when no compatible adapter is present,
/// or when device acquisition fails.
pub fn probe_forced(force_no_adapter: bool) -> Result<GpuContext, GpuUnavailable> {
    if force_no_adapter {
        return Err(GpuUnavailable::new(
            "GPU disabled by forced no-adapter probe (simulated GPU-less host)",
        ));
    }

    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
        backends: compute_backends(),
        ..Default::default()
    });

    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
    }))
    .map_err(|e| GpuUnavailable::new(format!("no compatible GPU adapter found: {e}")))?;

    let identity = AdapterIdentity::from_info(&adapter.get_info());

    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("paintop-wgpu device"),
        required_features: wgpu::Features::empty(),
        required_limits: wgpu::Limits::default(),
        memory_hints: wgpu::MemoryHints::default(),
        trace: wgpu::Trace::default(),
        experimental_features: wgpu::ExperimentalFeatures::default(),
    }))
    .map_err(|e| {
        GpuUnavailable::new(format!(
            "adapter `{}` found but device acquisition failed: {e}",
            identity.name
        ))
    })?;

    let limits = device.limits();

    Ok(GpuContext {
        device,
        queue,
        identity,
        limits,
    })
}

/// Whether a compatible GPU adapter is present on this host.
///
/// A cheap boolean probe (no device acquisition) the differential harness feeds to
/// `paintop_testkit::differential::GpuAdapter` so GPU-requiring tests skip cleanly
/// when no adapter is present and run when one is.
#[must_use]
pub fn adapter_present() -> bool {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
        backends: compute_backends(),
        ..Default::default()
    });
    pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
    }))
    .is_ok()
}

#[cfg(test)]
mod tests {
    use super::{
        AdapterIdentity, GpuUnavailable, adapter_present, device_type_str, probe, probe_forced,
    };
    use crate::gpu::error::E_GPU_UNAVAILABLE;
    use paintop_ir::{Error, ErrorClass};

    #[test]
    fn forced_no_adapter_is_a_clean_unavailable_not_a_panic() {
        // The fallback path must be exercisable even on a host WITH a GPU.
        let outcome = probe_forced(true);
        let err = outcome.expect_err("forced probe yields unavailable");
        assert!(err.reason.contains("forced"), "{}", err.reason);
    }

    #[test]
    fn unavailable_escalates_to_typed_unsupported_error() {
        let unavailable = GpuUnavailable::new("no adapter on this host");
        let gpu_err = unavailable.into_unsupported();
        let err: Error = gpu_err.into();
        assert_eq!(err.code, E_GPU_UNAVAILABLE);
        assert_eq!(err.class, ErrorClass::Policy);
        assert!(err.message.contains("no adapter on this host"));
    }

    #[test]
    fn device_type_strings_are_stable_kebab_case() {
        assert_eq!(
            device_type_str(wgpu::DeviceType::DiscreteGpu),
            "discrete-gpu"
        );
        assert_eq!(
            device_type_str(wgpu::DeviceType::IntegratedGpu),
            "integrated-gpu"
        );
        assert_eq!(device_type_str(wgpu::DeviceType::Cpu), "cpu");
        assert_eq!(device_type_str(wgpu::DeviceType::Other), "other");
        assert_eq!(device_type_str(wgpu::DeviceType::VirtualGpu), "virtual-gpu");
    }

    #[test]
    fn identity_evidence_strings_omit_unknown_driver() {
        let id = AdapterIdentity {
            name: "Test GPU".to_owned(),
            backend: "vulkan".to_owned(),
            device_type: "discrete-gpu".to_owned(),
            vendor: 0x10de,
            device: 0x2204,
            driver: String::new(),
            driver_info: String::new(),
        };
        assert_eq!(id.evidence_gpu(), "Test GPU (vulkan)");
        assert_eq!(id.evidence_driver(), None);
    }

    #[test]
    fn identity_fills_platform_gpu_and_driver_provenance() {
        let id = AdapterIdentity {
            name: "Test GPU".to_owned(),
            backend: "vulkan".to_owned(),
            device_type: "discrete-gpu".to_owned(),
            vendor: 0x10de,
            device: 0x2204,
            driver: "NVIDIA".to_owned(),
            driver_info: "550.1".to_owned(),
        };
        let platform = id.to_platform();
        assert_eq!(platform.gpu, Some("Test GPU (vulkan)".to_owned()));
        assert_eq!(platform.driver, Some("NVIDIA (550.1)".to_owned()));
        // os/arch come from the build target, not placeholders.
        assert!(!platform.os.is_empty());
        assert!(!platform.arch.is_empty());
    }

    #[test]
    fn identity_evidence_driver_combines_name_and_info() {
        let id = AdapterIdentity {
            name: "Test GPU".to_owned(),
            backend: "vulkan".to_owned(),
            device_type: "discrete-gpu".to_owned(),
            vendor: 0,
            device: 0,
            driver: "NVIDIA".to_owned(),
            driver_info: "550.1".to_owned(),
        };
        assert_eq!(id.evidence_driver(), Some("NVIDIA (550.1)".to_owned()));
    }

    /// When a real adapter is present, a non-forced probe must succeed and capture a
    /// non-empty identity; when absent it must fail cleanly. Either way: no panic,
    /// no wrong result. The forced path is always tested above.
    #[test]
    fn real_probe_matches_adapter_presence() {
        let present = adapter_present();
        match probe() {
            Ok(ctx) => {
                assert!(present, "probe succeeded so an adapter must be present");
                assert!(
                    !ctx.identity().name.is_empty(),
                    "a live context records its adapter name"
                );
                // The identity surfaces clean evidence strings.
                assert!(!ctx.identity().evidence_gpu().is_empty());
                // Limits are real (positive workgroup bound).
                assert!(ctx.limits().max_compute_workgroups_per_dimension > 0);
            }
            Err(unavailable) => {
                assert!(
                    !present,
                    "probe failed so no adapter should be present: {}",
                    unavailable.reason
                );
            }
        }
    }
}
