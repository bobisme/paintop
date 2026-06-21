//! GPU resource model + dispatch-dimension / overflow validation (`plan.md`
//! §12.3, bn-3ov).
//!
//! GPU kernels operate on **storage textures and storage buffers**, dispatched as a
//! grid of compute workgroups. Two classes of malformed work must be rejected
//! *before* any GPU submission, so a kernel can never hang, silently truncate, or
//! produce a wrong result (`plan.md` §12.3: "use storage textures/buffers
//! intentionally; validate dispatch dimensions and overflow"):
//!
//! * a **resource** larger than the device supports (a texture wider than
//!   `max_texture_dimension_2d`, or a storage buffer beyond
//!   `max_storage_buffer_binding_size` / `max_buffer_size`);
//! * a **dispatch** whose workgroup size exceeds the device's per-axis or
//!   per-workgroup invocation limits, or whose workgroup *count* (per axis or as a
//!   `u32` product) exceeds `max_compute_workgroups_per_dimension` or overflows.
//!
//! All checks run against a [`DeviceLimits`] snapshot — a small, `Copy`,
//! `wgpu`-free value — so the whole resource model is unit-testable on a host with
//! **no GPU**, while a real run snapshots the live device's limits via
//! [`DeviceLimits::from_wgpu`].

use paintop_ir::{Extent, ScalarType};

use super::error::GpuError;
use super::probe::GpuContext;

/// A `wgpu`-free snapshot of the device limits the resource model validates
/// against.
///
/// Decoupling from [`wgpu::Limits`] keeps dispatch/resource validation a pure
/// function of plain integers, so it is exhaustively testable GPU-less. A live run
/// snapshots the device's real limits with [`from_wgpu`](Self::from_wgpu); tests
/// construct small synthetic limits directly.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeviceLimits {
    /// Maximum width/height of a 2D texture, in texels.
    pub max_texture_dimension_2d: u32,
    /// Maximum size of a single storage-buffer binding, in bytes.
    pub max_storage_buffer_binding_size: u64,
    /// Maximum size of any single buffer, in bytes.
    pub max_buffer_size: u64,
    /// Maximum workgroup size along X.
    pub max_workgroup_size_x: u32,
    /// Maximum workgroup size along Y.
    pub max_workgroup_size_y: u32,
    /// Maximum workgroup size along Z.
    pub max_workgroup_size_z: u32,
    /// Maximum total invocations (`x*y*z`) within one workgroup.
    pub max_invocations_per_workgroup: u32,
    /// Maximum number of workgroups dispatched along any single dimension.
    pub max_workgroups_per_dimension: u32,
}

impl DeviceLimits {
    /// Snapshot the limits from a live `wgpu` device.
    #[must_use]
    pub fn from_wgpu(limits: &wgpu::Limits) -> Self {
        Self {
            max_texture_dimension_2d: limits.max_texture_dimension_2d,
            max_storage_buffer_binding_size: u64::from(limits.max_storage_buffer_binding_size),
            max_buffer_size: limits.max_buffer_size,
            max_workgroup_size_x: limits.max_compute_workgroup_size_x,
            max_workgroup_size_y: limits.max_compute_workgroup_size_y,
            max_workgroup_size_z: limits.max_compute_workgroup_size_z,
            max_invocations_per_workgroup: limits.max_compute_invocations_per_workgroup,
            max_workgroups_per_dimension: limits.max_compute_workgroups_per_dimension,
        }
    }

    /// Snapshot the limits of an acquired [`GpuContext`].
    #[must_use]
    pub fn of_context(context: &GpuContext) -> Self {
        Self::from_wgpu(context.limits())
    }
}

/// The specification of a GPU **storage texture** resource: a 2D raster of
/// `channels` `f32` texels per pixel.
///
/// paintop's resource buffers are row-major `f32` samples ([`ResourceValue`]); a
/// GPU texture mirrors that as an `extent.width × extent.height` grid of
/// `channels`-component texels. [`validate`](Self::validate) rejects a texture the
/// device cannot allocate *before* any GPU work, returning a typed
/// [`GpuError::DispatchInvalid`].
///
/// [`ResourceValue`]: paintop_core::executor::value::ResourceValue
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StorageTextureSpec {
    /// The texture's pixel extent.
    pub extent: Extent,
    /// The number of `f32` components per texel (1–4).
    pub channels: u32,
}

impl StorageTextureSpec {
    /// A storage texture for a raster of `extent` with `channels` components.
    #[must_use]
    pub const fn new(extent: Extent, channels: u32) -> Self {
        Self { extent, channels }
    }

    /// The backing storage size in bytes (`width * height * channels * 4`),
    /// overflow-checked.
    ///
    /// # Errors
    /// [`GpuError::DispatchInvalid`] if the product overflows `u64`.
    pub fn byte_size(&self) -> Result<u64, GpuError> {
        self.extent
            .byte_count(self.channels, ScalarType::F32)
            .map_err(|e| GpuError::DispatchInvalid {
                reason: format!("storage texture size overflows: {}", e.message),
            })
    }

    /// Validate this texture against `limits`.
    ///
    /// # Errors
    /// [`GpuError::DispatchInvalid`] if the extent is zero in either axis, exceeds
    /// `max_texture_dimension_2d`, has an out-of-range channel count, or whose
    /// backing size exceeds the device's buffer limits / overflows.
    pub fn validate(&self, limits: &DeviceLimits) -> Result<(), GpuError> {
        if self.extent.width == 0 || self.extent.height == 0 {
            return Err(GpuError::DispatchInvalid {
                reason: format!(
                    "zero-sized storage texture {}x{}",
                    self.extent.width, self.extent.height
                ),
            });
        }
        if self.channels == 0 || self.channels > 4 {
            return Err(GpuError::DispatchInvalid {
                reason: format!(
                    "storage texture channel count {} not in 1..=4",
                    self.channels
                ),
            });
        }
        let dim_limit = limits.max_texture_dimension_2d;
        if self.extent.width > dim_limit || self.extent.height > dim_limit {
            return Err(GpuError::DispatchInvalid {
                reason: format!(
                    "storage texture {}x{} exceeds max_texture_dimension_2d {dim_limit}",
                    self.extent.width, self.extent.height
                ),
            });
        }
        let bytes = self.byte_size()?;
        if bytes > limits.max_buffer_size {
            return Err(GpuError::DispatchInvalid {
                reason: format!(
                    "storage texture backing {bytes} bytes exceeds max_buffer_size {}",
                    limits.max_buffer_size
                ),
            });
        }
        Ok(())
    }
}

/// The specification of a GPU **storage buffer**: a flat run of `len` `f32`
/// samples.
///
/// Mirrors a paintop resource buffer for kernels that consume it as a linear array
/// (e.g. reductions, splat batches). [`validate`](Self::validate) rejects a buffer
/// the device cannot bind.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StorageBufferSpec {
    /// The number of `f32` elements.
    pub len: u64,
}

impl StorageBufferSpec {
    /// A storage buffer of `len` `f32` elements.
    #[must_use]
    pub const fn new(len: u64) -> Self {
        Self { len }
    }

    /// The buffer size in bytes (`len * 4`), overflow-checked.
    ///
    /// # Errors
    /// [`GpuError::DispatchInvalid`] if `len * 4` overflows `u64`.
    pub fn byte_size(&self) -> Result<u64, GpuError> {
        self.len
            .checked_mul(u64::from(ScalarType::F32.byte_size()))
            .ok_or_else(|| GpuError::DispatchInvalid {
                reason: format!("storage buffer length {} * 4 overflows u64", self.len),
            })
    }

    /// Validate this buffer against `limits`.
    ///
    /// # Errors
    /// [`GpuError::DispatchInvalid`] if empty, or if its byte size exceeds the
    /// device's storage-binding / buffer limits or overflows.
    pub fn validate(&self, limits: &DeviceLimits) -> Result<(), GpuError> {
        if self.len == 0 {
            return Err(GpuError::DispatchInvalid {
                reason: "zero-length storage buffer".to_owned(),
            });
        }
        let bytes = self.byte_size()?;
        if bytes > limits.max_storage_buffer_binding_size {
            return Err(GpuError::DispatchInvalid {
                reason: format!(
                    "storage buffer {bytes} bytes exceeds max_storage_buffer_binding_size {}",
                    limits.max_storage_buffer_binding_size
                ),
            });
        }
        if bytes > limits.max_buffer_size {
            return Err(GpuError::DispatchInvalid {
                reason: format!(
                    "storage buffer {bytes} bytes exceeds max_buffer_size {}",
                    limits.max_buffer_size
                ),
            });
        }
        Ok(())
    }
}

/// A compute workgroup's local size (`@workgroup_size(x, y, z)`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WorkgroupSize {
    /// Local size along X.
    pub x: u32,
    /// Local size along Y.
    pub y: u32,
    /// Local size along Z.
    pub z: u32,
}

impl WorkgroupSize {
    /// A workgroup local size.
    #[must_use]
    pub const fn new(x: u32, y: u32, z: u32) -> Self {
        Self { x, y, z }
    }

    /// The total invocations per workgroup (`x*y*z`), overflow-checked.
    ///
    /// # Errors
    /// [`GpuError::DispatchInvalid`] if the product overflows `u32`.
    pub fn invocations(&self) -> Result<u32, GpuError> {
        self.x
            .checked_mul(self.y)
            .and_then(|n| n.checked_mul(self.z))
            .ok_or_else(|| GpuError::DispatchInvalid {
                reason: format!(
                    "workgroup size {}x{}x{} invocation product overflows u32",
                    self.x, self.y, self.z
                ),
            })
    }
}

impl Default for WorkgroupSize {
    /// The conventional 8×8×1 tile for 2D image kernels.
    fn default() -> Self {
        Self::new(8, 8, 1)
    }
}

/// A validated compute dispatch: the workgroup local size and the per-axis
/// workgroup *counts* the kernel is dispatched with.
///
/// Built via [`for_extent`](Self::for_extent), which computes the ceil-division
/// group counts to cover a 2D problem and validates every dimension + overflow
/// against the device limits. A `Dispatch` only exists if it is dispatchable, so a
/// caller can hand it straight to `dispatch_workgroups` with no further checks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Dispatch {
    workgroup: WorkgroupSize,
    groups_x: u32,
    groups_y: u32,
    groups_z: u32,
}

impl Dispatch {
    /// Build and validate a dispatch covering a 2D `extent` with `workgroup`-sized
    /// tiles, against `limits`.
    ///
    /// The per-axis group counts are `ceil(extent / workgroup)`; Z is a single
    /// group. Every workgroup-size axis, the per-workgroup invocation total, every
    /// group-count axis, and the `u32` group-count product are validated against the
    /// device limits with overflow checks.
    ///
    /// # Errors
    /// [`GpuError::DispatchInvalid`] if the extent is zero, the workgroup size is
    /// zero or exceeds a per-axis / per-workgroup limit, or a group count exceeds
    /// `max_workgroups_per_dimension` or overflows `u32`.
    pub fn for_extent(
        extent: Extent,
        workgroup: WorkgroupSize,
        limits: &DeviceLimits,
    ) -> Result<Self, GpuError> {
        if extent.width == 0 || extent.height == 0 {
            return Err(GpuError::DispatchInvalid {
                reason: format!(
                    "zero-sized dispatch extent {}x{}",
                    extent.width, extent.height
                ),
            });
        }
        Self::validate_workgroup(workgroup, limits)?;

        let groups_x = ceil_div(extent.width, workgroup.x)?;
        let groups_y = ceil_div(extent.height, workgroup.y)?;
        let groups_z = 1_u32;

        let dispatch = Self {
            workgroup,
            groups_x,
            groups_y,
            groups_z,
        };
        dispatch.validate_group_counts(limits)?;
        Ok(dispatch)
    }

    /// The workgroup local size.
    #[must_use]
    pub const fn workgroup(&self) -> WorkgroupSize {
        self.workgroup
    }

    /// The per-axis workgroup counts `(x, y, z)` to pass to `dispatch_workgroups`.
    #[must_use]
    pub const fn groups(&self) -> (u32, u32, u32) {
        (self.groups_x, self.groups_y, self.groups_z)
    }

    /// The total number of workgroups (`x*y*z`), overflow-checked.
    ///
    /// # Errors
    /// [`GpuError::DispatchInvalid`] if the product overflows `u32`.
    pub fn total_groups(&self) -> Result<u32, GpuError> {
        self.groups_x
            .checked_mul(self.groups_y)
            .and_then(|n| n.checked_mul(self.groups_z))
            .ok_or_else(|| GpuError::DispatchInvalid {
                reason: format!(
                    "total workgroup count {}x{}x{} overflows u32",
                    self.groups_x, self.groups_y, self.groups_z
                ),
            })
    }

    fn validate_workgroup(workgroup: WorkgroupSize, limits: &DeviceLimits) -> Result<(), GpuError> {
        if workgroup.x == 0 || workgroup.y == 0 || workgroup.z == 0 {
            return Err(GpuError::DispatchInvalid {
                reason: format!(
                    "zero workgroup size {}x{}x{}",
                    workgroup.x, workgroup.y, workgroup.z
                ),
            });
        }
        if workgroup.x > limits.max_workgroup_size_x
            || workgroup.y > limits.max_workgroup_size_y
            || workgroup.z > limits.max_workgroup_size_z
        {
            return Err(GpuError::DispatchInvalid {
                reason: format!(
                    "workgroup size {}x{}x{} exceeds device per-axis limits {}x{}x{}",
                    workgroup.x,
                    workgroup.y,
                    workgroup.z,
                    limits.max_workgroup_size_x,
                    limits.max_workgroup_size_y,
                    limits.max_workgroup_size_z,
                ),
            });
        }
        let invocations = workgroup.invocations()?;
        if invocations > limits.max_invocations_per_workgroup {
            return Err(GpuError::DispatchInvalid {
                reason: format!(
                    "workgroup {invocations} invocations exceed max_invocations_per_workgroup {}",
                    limits.max_invocations_per_workgroup
                ),
            });
        }
        Ok(())
    }

    fn validate_group_counts(&self, limits: &DeviceLimits) -> Result<(), GpuError> {
        let per_dim = limits.max_workgroups_per_dimension;
        for (axis, count) in [
            ("x", self.groups_x),
            ("y", self.groups_y),
            ("z", self.groups_z),
        ] {
            if count > per_dim {
                return Err(GpuError::DispatchInvalid {
                    reason: format!(
                        "workgroup count {count} on axis {axis} exceeds \
                         max_workgroups_per_dimension {per_dim}"
                    ),
                });
            }
        }
        // Surfaces a `u32` overflow of the product as an explicit error.
        let _ = self.total_groups()?;
        Ok(())
    }
}

/// `ceil(value / divisor)` for positive `u32`s, overflow-checked.
fn ceil_div(value: u32, divisor: u32) -> Result<u32, GpuError> {
    // `divisor` is validated non-zero by the caller, but stay defensive.
    if divisor == 0 {
        return Err(GpuError::DispatchInvalid {
            reason: "ceil-division by zero workgroup size".to_owned(),
        });
    }
    value
        .checked_add(divisor - 1)
        .map(|n| n / divisor)
        .ok_or_else(|| GpuError::DispatchInvalid {
            reason: format!("ceil-division {value}/{divisor} overflows u32"),
        })
}

#[cfg(test)]
mod tests {
    use super::{
        DeviceLimits, Dispatch, StorageBufferSpec, StorageTextureSpec, WorkgroupSize, ceil_div,
    };
    use paintop_ir::Extent;

    /// Small synthetic limits for GPU-less validation tests.
    const LIMITS: DeviceLimits = DeviceLimits {
        max_texture_dimension_2d: 8192,
        max_storage_buffer_binding_size: 128 << 20,
        max_buffer_size: 256 << 20,
        max_workgroup_size_x: 256,
        max_workgroup_size_y: 256,
        max_workgroup_size_z: 64,
        max_invocations_per_workgroup: 256,
        max_workgroups_per_dimension: 65535,
    };

    #[test]
    fn valid_texture_passes() {
        let spec = StorageTextureSpec::new(Extent::new(1920, 1080), 4);
        assert!(spec.validate(&LIMITS).is_ok());
        assert_eq!(spec.byte_size().unwrap(), 1920 * 1080 * 4 * 4);
    }

    #[test]
    fn zero_sized_texture_is_rejected() {
        let spec = StorageTextureSpec::new(Extent::new(0, 16), 4);
        let err = spec.validate(&LIMITS).unwrap_err();
        assert!(err.to_string().contains("zero-sized"));
    }

    #[test]
    fn texture_over_dimension_limit_is_rejected() {
        let spec = StorageTextureSpec::new(Extent::new(8193, 16), 4);
        let err = spec.validate(&LIMITS).unwrap_err();
        assert!(err.to_string().contains("max_texture_dimension_2d"));
    }

    #[test]
    fn texture_bad_channel_count_is_rejected() {
        assert!(
            StorageTextureSpec::new(Extent::new(8, 8), 0)
                .validate(&LIMITS)
                .is_err()
        );
        assert!(
            StorageTextureSpec::new(Extent::new(8, 8), 5)
                .validate(&LIMITS)
                .is_err()
        );
    }

    #[test]
    fn texture_over_buffer_size_is_rejected() {
        // 8192*8192*4*4 = 1 GiB > max_buffer_size (256 MiB).
        let spec = StorageTextureSpec::new(Extent::new(8192, 8192), 4);
        let err = spec.validate(&LIMITS).unwrap_err();
        assert!(err.to_string().contains("max_buffer_size"));
    }

    #[test]
    fn valid_buffer_passes() {
        let spec = StorageBufferSpec::new(1024);
        assert!(spec.validate(&LIMITS).is_ok());
        assert_eq!(spec.byte_size().unwrap(), 4096);
    }

    #[test]
    fn zero_length_buffer_is_rejected() {
        assert!(StorageBufferSpec::new(0).validate(&LIMITS).is_err());
    }

    #[test]
    fn buffer_over_binding_size_is_rejected() {
        // (128 MiB / 4) + 1 elements -> just over the binding size.
        let spec = StorageBufferSpec::new((128 << 20) / 4 + 1);
        let err = spec.validate(&LIMITS).unwrap_err();
        assert!(err.to_string().contains("max_storage_buffer_binding_size"));
    }

    #[test]
    fn buffer_byte_size_overflow_is_rejected() {
        let spec = StorageBufferSpec::new(u64::MAX);
        assert!(spec.byte_size().is_err());
    }

    #[test]
    fn dispatch_for_extent_computes_ceil_div_groups() {
        // 1920/8 = 240, 1080/8 = 135.
        let d = Dispatch::for_extent(Extent::new(1920, 1080), WorkgroupSize::default(), &LIMITS)
            .unwrap();
        assert_eq!(d.groups(), (240, 135, 1));
        assert_eq!(d.total_groups().unwrap(), 240 * 135);
    }

    #[test]
    fn dispatch_ceil_div_rounds_up_partial_tiles() {
        // 17/8 = 3 groups (covers 24 >= 17), 1/8 = 1 group.
        let d =
            Dispatch::for_extent(Extent::new(17, 1), WorkgroupSize::default(), &LIMITS).unwrap();
        assert_eq!(d.groups(), (3, 1, 1));
    }

    #[test]
    fn dispatch_zero_extent_is_rejected() {
        let err =
            Dispatch::for_extent(Extent::new(0, 8), WorkgroupSize::default(), &LIMITS).unwrap_err();
        assert!(err.to_string().contains("zero-sized dispatch"));
    }

    #[test]
    fn dispatch_workgroup_over_invocation_limit_is_rejected() {
        // 16*16*2 = 512 > max_invocations_per_workgroup (256).
        let err = Dispatch::for_extent(Extent::new(64, 64), WorkgroupSize::new(16, 16, 2), &LIMITS)
            .unwrap_err();
        assert!(err.to_string().contains("max_invocations_per_workgroup"));
    }

    #[test]
    fn dispatch_workgroup_over_axis_limit_is_rejected() {
        let err = Dispatch::for_extent(Extent::new(64, 64), WorkgroupSize::new(257, 1, 1), &LIMITS)
            .unwrap_err();
        assert!(err.to_string().contains("per-axis"));
    }

    #[test]
    fn dispatch_group_count_over_per_dimension_limit_is_rejected() {
        // A tiny per-dim limit forces the group count over the edge.
        let tight = DeviceLimits {
            max_workgroups_per_dimension: 4,
            ..LIMITS
        };
        // 1000/8 = 125 groups on X >> 4.
        let err = Dispatch::for_extent(Extent::new(1000, 8), WorkgroupSize::default(), &tight)
            .unwrap_err();
        assert!(err.to_string().contains("max_workgroups_per_dimension"));
    }

    #[test]
    fn workgroup_invocation_overflow_is_rejected() {
        let wg = WorkgroupSize::new(u32::MAX, 2, 1);
        assert!(wg.invocations().is_err());
    }

    #[test]
    fn ceil_div_overflow_is_rejected() {
        // value + (divisor - 1) overflows u32 when value is near the max and the
        // divisor leaves no headroom.
        assert!(ceil_div(u32::MAX, 2).is_err());
        // divisor == 1 never overflows (adds zero).
        assert_eq!(ceil_div(u32::MAX, 1).unwrap(), u32::MAX);
        assert_eq!(ceil_div(10, 4).unwrap(), 3);
        assert_eq!(ceil_div(8, 8).unwrap(), 1);
    }
}
