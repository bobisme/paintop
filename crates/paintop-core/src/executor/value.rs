//! The whole-image resource value an op implementation reads and writes.
//!
//! The descriptor-level pipeline ([`paintop_ir::check`]) reasons about a
//! resource's *type* — its extent, color encoding, alpha representation — without
//! ever touching pixels. Execution, by contrast, moves bulk data: an op
//! implementation reads concrete input [`ResourceValue`]s and produces concrete
//! output ones. This module owns that runtime value.
//!
//! Per this bone's scope (`plan.md` §10.1 phase 11, `M0_DECISIONS` D2) the executor
//! is *whole-image and sequential*: a value carries the resource's full extent in
//! one contiguous buffer, with no tiling or region windows (those are M2). A value
//! is a typed [`ResourceDescriptor`] paired with a flat, row-major `f32` sample
//! buffer of length `width * height * channels`. Keeping the buffer to a single
//! scalar type is sufficient for the MVP ops (segment 2) and for the stub/identity
//! ops this bone tests; richer storage is a later concern.

use paintop_ir::{Extent, ResourceDescriptor};

/// A concrete whole-image resource value: a typed descriptor and its row-major
/// `f32` sample buffer.
///
/// The buffer holds `extent.width * extent.height * channels` samples in
/// row-major, channel-interleaved order. The descriptor is the authority on the
/// channel count, so [`ResourceValue::new`] validates the buffer length against
/// it and refuses a mismatch.
#[derive(Debug, Clone, PartialEq)]
pub struct ResourceValue {
    descriptor: ResourceDescriptor,
    channels: u32,
    samples: Vec<f32>,
}

impl ResourceValue {
    /// Wrap a row-major `f32` sample buffer with its descriptor.
    ///
    /// `channels` is the number of interleaved samples per pixel; the buffer
    /// length must be exactly `extent.width * extent.height * channels`.
    ///
    /// # Errors
    /// Returns `Err(actual_len)` carrying the buffer's actual length if it does
    /// not match the descriptor's extent and channel count, so a caller can
    /// report the mismatch precisely. The expected length is recoverable from the
    /// descriptor.
    pub fn new(
        descriptor: ResourceDescriptor,
        channels: u32,
        samples: Vec<f32>,
    ) -> Result<Self, usize> {
        let expected = expected_len(descriptor.extent(), channels);
        if samples.len() == expected {
            Ok(Self {
                descriptor,
                channels,
                samples,
            })
        } else {
            Err(samples.len())
        }
    }

    /// The resource's typed descriptor.
    #[must_use]
    pub const fn descriptor(&self) -> &ResourceDescriptor {
        &self.descriptor
    }

    /// The resource's pixel extent.
    #[must_use]
    pub const fn extent(&self) -> Extent {
        self.descriptor.extent()
    }

    /// The number of interleaved samples per pixel.
    #[must_use]
    pub const fn channels(&self) -> u32 {
        self.channels
    }

    /// The row-major, channel-interleaved `f32` sample buffer.
    #[must_use]
    pub fn samples(&self) -> &[f32] {
        &self.samples
    }

    /// Consume the value, returning its sample buffer.
    #[must_use]
    pub fn into_samples(self) -> Vec<f32> {
        self.samples
    }
}

/// The expected sample-buffer length for an extent and channel count, saturating
/// rather than overflowing (an over-long buffer is rejected by the length check
/// regardless).
const fn expected_len(extent: Extent, channels: u32) -> usize {
    (extent.width as usize)
        .saturating_mul(extent.height as usize)
        .saturating_mul(channels as usize)
}

#[cfg(test)]
mod tests {
    use super::ResourceValue;
    use paintop_ir::{
        AlphaRepresentation, ChannelLayout, ColorEncoding, ColorRange, CoordinateConvention,
        Extent, ImageDescriptor, ResourceDescriptor, ScalarType, SemanticRole,
    };

    fn image(extent: Extent) -> ResourceDescriptor {
        ResourceDescriptor::Image(ImageDescriptor {
            extent,
            layout: ChannelLayout::Rgba,
            scalar: ScalarType::F32,
            color: ColorEncoding::LinearSrgb,
            range: ColorRange::SceneReferred,
            alpha: AlphaRepresentation::Premultiplied,
            coordinates: CoordinateConvention::PixelCenterUpperLeft,
            semantic: SemanticRole::Color,
        })
    }

    #[test]
    fn wraps_a_correctly_sized_buffer() {
        let v = ResourceValue::new(image(Extent::new(2, 3)), 4, vec![0.0; 2 * 3 * 4]).unwrap();
        assert_eq!(v.extent(), Extent::new(2, 3));
        assert_eq!(v.channels(), 4);
        assert_eq!(v.samples().len(), 24);
    }

    #[test]
    fn rejects_a_mis_sized_buffer() {
        let err = ResourceValue::new(image(Extent::new(2, 2)), 4, vec![0.0; 8]).unwrap_err();
        assert_eq!(err, 8);
    }

    #[test]
    fn into_samples_returns_the_buffer() {
        let v = ResourceValue::new(image(Extent::new(1, 1)), 1, vec![0.5]).unwrap();
        assert_eq!(v.into_samples(), vec![0.5]);
    }
}
