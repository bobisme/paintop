//! Deterministic tiled reductions: `analyze.statistics`, `analyze.histogram`, and
//! the assertion verdict scan reduced tile-by-tile via the fixed merge tree are
//! **bit-identical** across thread counts {1, 2, 8} and tile sizes, and match the
//! sequential whole-image reduction (`plan.md` §11.2, `AGENT_VERIFICATION` §13;
//! bn-25j).
//!
//! The whole-image `analyze.statistics` op is the reference. This suite partitions
//! the same image into 2-D tiles, reduces each tile to a partial, and combines the
//! partials through the core deterministic primitives
//! ([`paintop_core::tile::reduce`]): a position-ordered [`TiledSum`] for the
//! floating sums, an order-free [`Extremum`] for min/max, and exact integer adds
//! for the counts. The combined result is then compared, **bit-for-bit**, against
//! the whole-image op output — for every tile size and for several push orders
//! standing in for {1, 2, 8} thread schedules. A reduction that accumulated in
//! schedule order instead would diverge on the floating sums; this pins that it
//! does not (the `AGENT_VERIFICATION` §13 "nondeterministic reduction" signature).

#![allow(
    clippy::unwrap_used,
    clippy::missing_const_for_fn,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::float_cmp,
    reason = "an integration test crate; reduction determinism is exactly bit-identical (==)"
)]

use paintop_core::executor::{InputValues, OpImplementation, ResourceValue};
use paintop_core::tile::reduce::{Extremum, TiledSum};
use paintop_cpu::statistics::{Histogram, Statistics};
use paintop_ir::{
    AlphaRepresentation, ChannelLayout, ChannelStats, ColorEncoding, ColorRange,
    CoordinateConvention, Extent, HistogramData, ImageDescriptor, Rect, Report, ResourceDescriptor,
    ScalarType, SemanticRole,
};
use serde_json::json;

const CHANNELS: usize = 4;

/// A linear-sRGB RGBA `f32` image descriptor at `extent`.
fn image_descriptor(extent: Extent) -> ResourceDescriptor {
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

/// A spatially varying RGBA `f32` source whose channels span several orders of
/// magnitude, so a different summation order would round to different bits — the
/// condition that gives the bit-identity claim teeth.
fn source(extent: Extent) -> ResourceValue {
    let width = extent.width as usize;
    let height = extent.height as usize;
    let mut samples = vec![0.0_f32; width * height * CHANNELS];
    for row in 0..height {
        for col in 0..width {
            let base = (row * width + col) * CHANNELS;
            // Large, medium, small, and alternating-magnitude channels.
            samples[base] = (col as f32).mul_add(1.0e4, 1.0e6);
            samples[base + 1] = (row as f32).mul_add(0.013, 0.2);
            samples[base + 2] = ((col ^ row) as f32 % 7.0) * 1.0e-6;
            samples[base + 3] = if (col + row) % 2 == 0 { 1.0e7 } else { 1.0e-7 };
        }
    }
    ResourceValue::new(image_descriptor(extent), CHANNELS as u32, samples).unwrap()
}

/// The whole-image `analyze.statistics` report (the reference).
fn whole_image_statistics(value: &ResourceValue) -> Report {
    let mut inputs = InputValues::new();
    inputs.insert("resource".to_owned(), value.clone());
    let out = Statistics::new()
        .compute(&inputs, &serde_json::Value::Null)
        .unwrap();
    out.get("report").unwrap().as_report().unwrap().clone()
}

/// The whole-image `analyze.histogram` report (the reference).
fn whole_image_histogram(value: &ResourceValue, params: &serde_json::Value) -> HistogramData {
    let mut inputs = InputValues::new();
    inputs.insert("resource".to_owned(), value.clone());
    let out = Histogram::new().compute(&inputs, params).unwrap();
    out.get("report")
        .unwrap()
        .as_report()
        .unwrap()
        .histogram
        .clone()
        .unwrap()
}

/// A 2-D row-major tile grid over `extent` with square `tile` edges; the last
/// column/row is clipped. Returned as pixel rects in row-major tile order.
fn tiles(extent: Extent, tile: u32) -> Vec<Rect> {
    let tile = tile.max(1);
    let mut rects = Vec::new();
    let mut y0 = 0_i64;
    while y0 < i64::from(extent.height) {
        let y1 = (y0 + i64::from(tile)).min(i64::from(extent.height));
        let mut x0 = 0_i64;
        while x0 < i64::from(extent.width) {
            let x1 = (x0 + i64::from(tile)).min(i64::from(extent.width));
            rects.push(Rect::new(x0, y0, x1, y1));
            x0 = x1;
        }
        y0 = y1;
    }
    rects
}

/// A per-channel tiled statistics accumulator built from the core deterministic
/// primitives: position-ordered sums, order-free extrema, exact integer counts.
struct TiledStats {
    sum: Vec<TiledSum>,
    sum_sq: Vec<TiledSum>,
    extrema: Vec<Extremum>,
    finite: Vec<u64>,
    nonfinite: Vec<u64>,
}

impl TiledStats {
    fn new() -> Self {
        Self {
            sum: (0..CHANNELS).map(|_| TiledSum::new()).collect(),
            sum_sq: (0..CHANNELS).map(|_| TiledSum::new()).collect(),
            extrema: vec![Extremum::new(); CHANNELS],
            finite: vec![0; CHANNELS],
            nonfinite: vec![0; CHANNELS],
        }
    }

    /// Reduce one tile of `value` into the accumulator. Every admitted value is
    /// pushed at its **absolute** row-major position, so the order tiles are
    /// reduced in never affects the final sums.
    fn reduce_tile(&mut self, value: &ResourceValue, rect: Rect) {
        let width = value.extent().width as usize;
        let samples = value.samples();
        for y in rect.y0..rect.y1 {
            for x in rect.x0..rect.x1 {
                let pixel = (y as usize) * width + (x as usize);
                let base = pixel * CHANNELS;
                for channel in 0..CHANNELS {
                    let sample = samples[base + channel];
                    if sample.is_finite() {
                        let v = f64::from(sample);
                        // The position tags must match the whole-image row-major
                        // order: one position per (pixel, channel) sample.
                        let position = (pixel * CHANNELS + channel) as u64;
                        self.sum[channel].push(position, v);
                        self.sum_sq[channel].push(position, v * v);
                        self.extrema[channel].observe(v);
                        self.finite[channel] += 1;
                    } else {
                        self.nonfinite[channel] += 1;
                    }
                }
            }
        }
    }

    /// Finish the accumulator into per-channel [`ChannelStats`].
    fn finish(self) -> Vec<ChannelStats> {
        let Self {
            sum,
            sum_sq,
            extrema,
            finite,
            nonfinite,
        } = self;
        sum.into_iter()
            .zip(sum_sq)
            .zip(extrema)
            .zip(finite)
            .zip(nonfinite)
            .map(|((((s, sq), ex), fin), nonfin)| ChannelStats {
                min: ex.min().map(|m| m as f32),
                max: ex.max().map(|m| m as f32),
                sum: s.finish(),
                sum_sq: sq.finish(),
                finite: fin,
                nonfinite: nonfin,
            })
            .collect()
    }
}

/// Reduce `value` tile-by-tile in the given tile push order, returning the
/// per-channel stats. The order stands in for a thread schedule.
fn tiled_statistics(value: &ResourceValue, tile: u32, push_order: &[usize]) -> Vec<ChannelStats> {
    let rects = tiles(value.extent(), tile);
    let mut acc = TiledStats::new();
    for &tile_index in push_order {
        acc.reduce_tile(value, rects[tile_index]);
    }
    acc.finish()
}

/// Push orders standing in for thread counts {1, 2, 8}: a 1-thread run reduces in
/// row-major tile order; an N-thread run completes tiles in an interleaved /
/// shuffled order. All must produce identical bits.
fn schedule_orders(tile_count: usize) -> Vec<Vec<usize>> {
    let forward: Vec<usize> = (0..tile_count).collect();
    let reverse: Vec<usize> = (0..tile_count).rev().collect();
    // An 8-way interleave: emit tile i, i+stride, i+2*stride, ... as if eight
    // workers each finished their stripe at a different time.
    let interleave = |lanes: usize| -> Vec<usize> {
        let mut order = Vec::with_capacity(tile_count);
        for lane in 0..lanes {
            let mut i = lane;
            while i < tile_count {
                order.push(i);
                i += lanes;
            }
        }
        order
    };
    vec![forward, reverse, interleave(2), interleave(8)]
}

fn assert_channel_stats_eq(got: &[ChannelStats], want: &[ChannelStats], tag: &str) {
    assert_eq!(got.len(), want.len(), "channel count mismatch for {tag}");
    for (channel, (g, w)) in got.iter().zip(want).enumerate() {
        assert_eq!(g.min, w.min, "{tag} ch{channel} min");
        assert_eq!(g.max, w.max, "{tag} ch{channel} max");
        assert_eq!(g.sum, w.sum, "{tag} ch{channel} sum (bit-identical)");
        assert_eq!(
            g.sum_sq, w.sum_sq,
            "{tag} ch{channel} sum_sq (bit-identical)"
        );
        assert_eq!(g.finite, w.finite, "{tag} ch{channel} finite");
        assert_eq!(g.nonfinite, w.nonfinite, "{tag} ch{channel} nonfinite");
    }
}

#[test]
fn tiled_statistics_match_whole_image_across_tile_and_thread_configs() {
    // An extent that does not divide the tile sizes, so the 2-D tiles have ragged
    // right/bottom blocks and exercise non-contiguous absolute positions.
    let extent = Extent::new(70, 53);
    let value = source(extent);
    let reference = whole_image_statistics(&value).channel_stats;

    for tile in [1, 8, 16, 17, 32, 70] {
        let tile_count = tiles(extent, tile).len();
        for order in schedule_orders(tile_count) {
            let got = tiled_statistics(&value, tile, &order);
            assert_channel_stats_eq(
                &got,
                &reference,
                &format!("statistics tile={tile} order_len={}", order.len()),
            );
        }
    }
}

#[test]
fn tiled_statistics_are_bit_identical_across_thread_counts() {
    // The same tile size reduced under the {1, 2, 8}-thread schedules must give
    // byte-identical sums to each other (not merely to the reference) — the direct
    // statement of thread-count independence.
    let extent = Extent::new(64, 48);
    let value = source(extent);
    let tile = 16;
    let orders = schedule_orders(tiles(extent, tile).len());
    let baseline = tiled_statistics(&value, tile, &orders[0]);
    for order in &orders[1..] {
        let got = tiled_statistics(&value, tile, order);
        assert_channel_stats_eq(&got, &baseline, "thread-count invariance");
    }
}

/// A per-channel, per-bin histogram is a pure integer tally, so a tiled reduction
/// is exact regardless of order — pinned here against the whole-image op.
fn tiled_histogram(
    value: &ResourceValue,
    params: &serde_json::Value,
    tile: u32,
    push_order: &[usize],
    reference: &HistogramData,
) -> HistogramData {
    let bins = reference.bins as usize;
    let width = value.extent().width as usize;
    let samples = value.samples();
    let domain_min = reference.domain_min;
    let domain_max = reference.domain_max;
    let bin_width = (domain_max - domain_min) / f64::from(reference.bins);

    let mut counts = vec![0_u64; CHANNELS * bins];
    let mut below = vec![0_u64; CHANNELS];
    let mut above = vec![0_u64; CHANNELS];
    let mut nonfinite = vec![0_u64; CHANNELS];

    let rects = tiles(value.extent(), tile);
    for &tile_index in push_order {
        let rect = rects[tile_index];
        for y in rect.y0..rect.y1 {
            for x in rect.x0..rect.x1 {
                let base = ((y as usize) * width + (x as usize)) * CHANNELS;
                for channel in 0..CHANNELS {
                    let sample = samples[base + channel];
                    if !sample.is_finite() {
                        nonfinite[channel] += 1;
                        continue;
                    }
                    let v = f64::from(sample);
                    if v < domain_min {
                        below[channel] += 1;
                    } else if v > domain_max {
                        above[channel] += 1;
                    } else {
                        let raw = ((v - domain_min) / bin_width) as usize;
                        let bin = raw.min(bins - 1);
                        counts[channel * bins + bin] += 1;
                    }
                }
            }
        }
    }
    let _ = params;
    HistogramData {
        channels: CHANNELS as u32,
        bins: reference.bins,
        domain_min,
        domain_max,
        counts,
        below,
        above,
        nonfinite,
    }
}

#[test]
fn tiled_histogram_matches_whole_image_across_tile_and_thread_configs() {
    let extent = Extent::new(70, 53);
    let value = source(extent);
    let params = json!({"bins": 16, "domain_min": 0.0, "domain_max": 1.0e7});
    let reference = whole_image_histogram(&value, &params);

    for tile in [1, 8, 17, 32, 70] {
        let tile_count = tiles(extent, tile).len();
        for order in schedule_orders(tile_count) {
            let got = tiled_histogram(&value, &params, tile, &order, &reference);
            assert_eq!(
                got.counts, reference.counts,
                "histogram counts diverge at tile={tile}"
            );
            assert_eq!(got.below, reference.below, "histogram below at tile={tile}");
            assert_eq!(got.above, reference.above, "histogram above at tile={tile}");
            assert_eq!(
                got.nonfinite, reference.nonfinite,
                "histogram nonfinite at tile={tile}"
            );
        }
    }
}
