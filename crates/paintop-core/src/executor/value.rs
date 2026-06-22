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

use paintop_ir::{
    Extent, PATCH_FIELD_CHANNELS, PatchFieldDescriptor, PyramidDescriptor, Report,
    ResourceDescriptor, SpectrumDescriptor,
};

/// A concrete whole-image resource value: a typed descriptor and its row-major
/// `f32` sample buffer.
///
/// The buffer holds `extent.width * extent.height * channels` samples in
/// row-major, channel-interleaved order. The descriptor is the authority on the
/// channel count, so [`ResourceValue::new`] validates the buffer length against
/// it and refuses a mismatch.
///
/// A [`Report`] resource (`OP_CATALOG` §1) carries no raster;
/// it is wrapped with [`ResourceValue::report`], which stores the structured
/// report and leaves the sample buffer empty. [`as_report`](ResourceValue::as_report)
/// recovers it.
#[derive(Debug, Clone, PartialEq)]
pub struct ResourceValue {
    descriptor: ResourceDescriptor,
    channels: u32,
    samples: Vec<f32>,
    report: Option<Report>,
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
                report: None,
            })
        } else {
            Err(samples.len())
        }
    }

    /// Wrap a structured [`Report`] (`OP_CATALOG` §1) as a resource value.
    ///
    /// A report carries no raster, so the descriptor is the report's
    /// [`ReportDescriptor`](paintop_ir::ReportDescriptor), the channel count is
    /// `0`, and the sample buffer is empty. The structured report is recoverable
    /// via [`as_report`](Self::as_report).
    #[must_use]
    pub const fn report(report: Report) -> Self {
        Self {
            descriptor: ResourceDescriptor::Report(report.descriptor()),
            channels: 0,
            samples: Vec::new(),
            report: Some(report),
        }
    }

    /// Wrap a level-major (finest-first) concatenated sample buffer as a
    /// [`Pyramid`](ResourceDescriptor::Pyramid) resource value.
    ///
    /// A pyramid value stores every level's samples in one flat buffer, level
    /// `0` (the base) first, each level row-major and channel-interleaved. The
    /// buffer length must equal the descriptor's
    /// [`total_samples`](PyramidDescriptor::total_samples) — the sum of
    /// `width_l·height_l·channels` over all levels — so a caller cannot wrap a
    /// truncated or padded pyramid. The descriptor's `channels` is the per-pixel
    /// sample count reported by [`channels`](Self::channels), and the individual
    /// level slices are recovered with [`pyramid_level`](Self::pyramid_level).
    ///
    /// # Errors
    /// Returns `Err(actual_len)` carrying the buffer's actual length if it does
    /// not match `descriptor.total_samples()`, or if the descriptor's level
    /// chain overflows (reported as a `0` expected length).
    pub fn pyramid(descriptor: PyramidDescriptor, samples: Vec<f32>) -> Result<Self, usize> {
        let expected = usize::try_from(descriptor.total_samples().map_err(|_| samples.len())?)
            .map_err(|_| samples.len())?;
        if samples.len() == expected {
            Ok(Self {
                channels: descriptor.channels,
                descriptor: ResourceDescriptor::Pyramid(descriptor),
                samples,
                report: None,
            })
        } else {
            Err(samples.len())
        }
    }

    /// The row-major, channel-interleaved sample slice of pyramid level `level`,
    /// or `None` if this value is not a pyramid or the level is out of range.
    ///
    /// Levels are stored finest-first and contiguous, so level `l`'s slice
    /// starts after the summed sample counts of levels `0..l`.
    #[must_use]
    pub fn pyramid_level(&self, level: u32) -> Option<&[f32]> {
        let ResourceDescriptor::Pyramid(d) = self.descriptor else {
            return None;
        };
        let mut offset = 0usize;
        for l in 0..d.levels {
            let extent = d.level_extent(l)?;
            let count = (extent.width as usize)
                .checked_mul(extent.height as usize)?
                .checked_mul(d.channels as usize)?;
            if l == level {
                return self.samples.get(offset..offset.checked_add(count)?);
            }
            offset = offset.checked_add(count)?;
        }
        None
    }

    /// Wrap an interleaved complex sample buffer as a
    /// [`Spectrum`](ResourceDescriptor::Spectrum) resource value.
    ///
    /// The buffer packs each complex bin as two consecutive `f32` (real then
    /// imaginary), in row-major, channel-interleaved order: bin `(x, y)`,
    /// channel `c` occupies indices `2·((y·W + x)·channels + c)` (real) and
    /// `+1` (imaginary). Its length must equal the descriptor's
    /// [`total_samples`](SpectrumDescriptor::total_samples) — `W·H·channels·2` —
    /// so a caller cannot wrap a truncated buffer. The reported
    /// [`channels`](Self::channels) is the *logical* (complex) channel count.
    ///
    /// # Errors
    /// Returns `Err(actual_len)` carrying the buffer's actual length if it does
    /// not match `descriptor.total_samples()` (a descriptor whose count
    /// overflows reports a `0` expected length, so any non-empty buffer is
    /// rejected).
    pub fn spectrum(descriptor: SpectrumDescriptor, samples: Vec<f32>) -> Result<Self, usize> {
        let expected = usize::try_from(descriptor.total_samples().map_err(|_| samples.len())?)
            .map_err(|_| samples.len())?;
        if samples.len() == expected {
            Ok(Self {
                channels: descriptor.channels,
                descriptor: ResourceDescriptor::Spectrum(descriptor),
                samples,
                report: None,
            })
        } else {
            Err(samples.len())
        }
    }

    /// Wrap an interleaved correspondence buffer as a
    /// [`PatchField`](ResourceDescriptor::PatchField) resource value.
    ///
    /// The buffer packs each target pixel's correspondence as
    /// [`PATCH_FIELD_CHANNELS`] consecutive `f32` — `src_x`, `src_y`, then
    /// `cost` — in row-major, channel-interleaved order: target pixel `(x, y)`
    /// occupies indices `3·(y·W + x) + {0,1,2}`. Its length must equal the
    /// descriptor's [`total_samples`](PatchFieldDescriptor::total_samples) —
    /// `target_W·target_H·3` — so a caller cannot wrap a truncated buffer. The
    /// reported [`channels`](Self::channels) is [`PATCH_FIELD_CHANNELS`].
    ///
    /// # Errors
    /// Returns `Err(actual_len)` carrying the buffer's actual length if it does
    /// not match `descriptor.total_samples()` (a descriptor whose count overflows
    /// reports a `0` expected length, so any non-empty buffer is rejected).
    pub fn patch_field(descriptor: PatchFieldDescriptor, samples: Vec<f32>) -> Result<Self, usize> {
        let expected = usize::try_from(descriptor.total_samples().map_err(|_| samples.len())?)
            .map_err(|_| samples.len())?;
        if samples.len() == expected {
            Ok(Self {
                channels: PATCH_FIELD_CHANNELS,
                descriptor: ResourceDescriptor::PatchField(descriptor),
                samples,
                report: None,
            })
        } else {
            Err(samples.len())
        }
    }

    /// The structured [`Report`] this value carries, if it is a report resource.
    #[must_use]
    pub const fn as_report(&self) -> Option<&Report> {
        self.report.as_ref()
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

    #[test]
    fn pyramid_wraps_and_slices_levels() {
        use paintop_ir::{DownsampleFactor, PyramidDescriptor, PyramidPhase};
        let d = PyramidDescriptor {
            base_extent: Extent::new(4, 4),
            levels: 3,
            channels: 1,
            scalar: ScalarType::F32,
            factor: DownsampleFactor::Half,
            phase: PyramidPhase::Floor,
            coordinates: CoordinateConvention::PixelCenterUpperLeft,
        };
        // 16 + 4 + 1 = 21 samples; fill each level with its level index.
        let mut samples = Vec::new();
        samples.extend(std::iter::repeat_n(0.0_f32, 16));
        samples.extend(std::iter::repeat_n(1.0_f32, 4));
        samples.extend(std::iter::repeat_n(2.0_f32, 1));
        let v = ResourceValue::pyramid(d, samples).unwrap();
        assert_eq!(v.channels(), 1);
        assert_eq!(v.extent(), Extent::new(4, 4));
        assert_eq!(v.pyramid_level(0).unwrap().len(), 16);
        assert_eq!(v.pyramid_level(1).unwrap(), &[1.0; 4]);
        assert_eq!(v.pyramid_level(2).unwrap(), &[2.0]);
        assert!(v.pyramid_level(3).is_none());
    }

    #[test]
    fn patch_field_wraps_and_reports_three_channels() {
        use paintop_ir::PatchFieldDescriptor;
        let d = PatchFieldDescriptor {
            target_extent: Extent::new(2, 2),
            source_extent: Extent::new(4, 4),
            radius: 1,
            coordinates: CoordinateConvention::PixelCenterUpperLeft,
        };
        // 2*2 target pixels * 3 channels = 12 samples.
        let v = ResourceValue::patch_field(d, vec![0.0; 12]).unwrap();
        assert_eq!(v.channels(), 3);
        assert_eq!(v.extent(), Extent::new(2, 2));
        assert_eq!(v.samples().len(), 12);
    }

    #[test]
    fn patch_field_rejects_a_mis_sized_buffer() {
        use paintop_ir::PatchFieldDescriptor;
        let d = PatchFieldDescriptor {
            target_extent: Extent::new(2, 2),
            source_extent: Extent::new(4, 4),
            radius: 1,
            coordinates: CoordinateConvention::PixelCenterUpperLeft,
        };
        // Expected 12; supply 8.
        let err = ResourceValue::patch_field(d, vec![0.0; 8]).unwrap_err();
        assert_eq!(err, 8);
    }

    #[test]
    fn pyramid_rejects_a_mis_sized_buffer() {
        use paintop_ir::{DownsampleFactor, PyramidDescriptor, PyramidPhase};
        let d = PyramidDescriptor {
            base_extent: Extent::new(2, 2),
            levels: 2,
            channels: 1,
            scalar: ScalarType::F32,
            factor: DownsampleFactor::Half,
            phase: PyramidPhase::Floor,
            coordinates: CoordinateConvention::PixelCenterUpperLeft,
        };
        // Expected 4 + 1 = 5; supply 4.
        let err = ResourceValue::pyramid(d, vec![0.0; 4]).unwrap_err();
        assert_eq!(err, 4);
    }
}
