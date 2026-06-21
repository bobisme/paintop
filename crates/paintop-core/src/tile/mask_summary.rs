//! Sparse mask tile occupancy summaries (`plan.md` §11.4).
//!
//! A masked edit is the common touch-up shape: a pointwise (or composite)
//! operation gated by a coverage mask, where the mask is non-zero only over a
//! small region. The executor exploits this by summarizing the mask *per tile*:
//! a tile the mask covers nowhere is an **empty** tile, and the gated branch on
//! that tile is provably the identity — the input passes through untouched
//! without evaluating the expensive op (`plan.md` §11.2: "empty masked tiles
//! become identity without evaluating the expensive branch"). A tile the mask
//! covers everywhere at full coverage is a **full** tile, and a blend that reads
//! the mask can skip the per-pixel mask read where that is legal. Everything else
//! is a **mixed** tile that must be evaluated normally.
//!
//! This module owns that summary. A [`MaskTileSummary`] records, over one tile's
//! pixels, the minimum and maximum coverage, the count of non-zero pixels, and
//! the tight bounding box of the non-zero pixels — exactly the
//! `MaskTileSummary { min_coverage, max_coverage, nonzero_count, bounds_of_nonzero }`
//! the plan calls for. From those it classifies the tile's [`Occupancy`]. A
//! whole-mask [`MaskSummary`] is the per-tile summaries over a [`TileGrid`], plus
//! the union of the per-tile non-zero bounds, which equals the mask's overall
//! non-zero bounds — the equality the bone pins against `mask.bounds`.

use paintop_ir::{Extent, Rect, Region};

use super::grid::{Tile, TileGrid};
use crate::executor::ResourceValue;

/// The occupancy class of a mask over one tile (`plan.md` §11.2, §11.4).
///
/// The classification gates the executor's fast paths: an [`Empty`](Self::Empty)
/// tile makes a masked branch the identity (skip the branch); a [`Full`](Self::Full)
/// tile lets a blend skip the per-pixel mask read; a [`Mixed`](Self::Mixed) tile
/// must be evaluated normally.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Occupancy {
    /// Every pixel of the tile has zero coverage: the masked branch is identity.
    Empty,
    /// Every pixel of the tile has full coverage (`max_coverage == min_coverage
    /// == 1.0` over a non-empty tile): a mask read may be skipped where legal.
    Full,
    /// The tile has some but not uniform-full coverage: evaluate normally.
    Mixed,
}

/// The occupancy summary of a coverage mask over one tile (`plan.md` §11.4).
///
/// Computed by scanning the tile's mask samples once in row-major order, so it is
/// a deterministic function of the mask and the tile. The non-zero bounds are the
/// tight bounding box (in image pixel space) of the pixels with strictly positive
/// coverage, or [`Region::EMPTY`] when the tile is empty.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MaskTileSummary {
    tile: Rect,
    min_coverage: f32,
    max_coverage: f32,
    nonzero_count: u64,
    nonzero_bounds: Region,
}

impl MaskTileSummary {
    /// The tile region this summary covers.
    #[must_use]
    pub const fn tile(&self) -> Rect {
        self.tile
    }

    /// The minimum coverage over the tile's pixels (`1.0` for an empty tile, the
    /// identity of the running min, so callers should gate on
    /// [`nonzero_count`](Self::nonzero_count) or [`occupancy`](Self::occupancy)).
    #[must_use]
    pub const fn min_coverage(&self) -> f32 {
        self.min_coverage
    }

    /// The maximum coverage over the tile's pixels (`0.0` for an empty tile).
    #[must_use]
    pub const fn max_coverage(&self) -> f32 {
        self.max_coverage
    }

    /// The number of pixels in the tile with strictly positive coverage.
    #[must_use]
    pub const fn nonzero_count(&self) -> u64 {
        self.nonzero_count
    }

    /// The tight bounding box (in image pixel space) of the tile's non-zero
    /// pixels, or [`Region::EMPTY`] when no pixel is covered.
    #[must_use]
    pub const fn nonzero_bounds(&self) -> Region {
        self.nonzero_bounds
    }

    /// The tile's occupancy class (`plan.md` §11.2).
    ///
    /// [`Empty`](Occupancy::Empty) when no pixel is covered; [`Full`](Occupancy::Full)
    /// when every pixel is covered at exactly `1.0` (uniform full coverage over a
    /// non-empty tile); [`Mixed`](Occupancy::Mixed) otherwise.
    #[must_use]
    pub fn occupancy(&self) -> Occupancy {
        if self.nonzero_count == 0 {
            Occupancy::Empty
        } else if self.is_uniform_full() {
            Occupancy::Full
        } else {
            Occupancy::Mixed
        }
    }

    /// Whether the masked branch over this tile is provably the identity — i.e.
    /// the tile is [`Empty`](Occupancy::Empty), so a gated op produces its
    /// pass-through input unchanged without evaluating the expensive branch.
    #[must_use]
    pub fn is_identity(&self) -> bool {
        matches!(self.occupancy(), Occupancy::Empty)
    }

    /// Whether every covered pixel is at full coverage over a non-empty tile.
    fn is_uniform_full(&self) -> bool {
        // `width`/`height` are non-negative for a valid clipped tile.
        let covered = u64::try_from(self.tile.width())
            .unwrap_or(0)
            .saturating_mul(u64::try_from(self.tile.height()).unwrap_or(0));
        self.nonzero_count == covered && self.min_coverage >= 1.0 && self.max_coverage <= 1.0
    }
}

/// Summarize the coverage of `mask` over a single output `tile`
/// (`plan.md` §11.4).
///
/// `mask` is read as a single-channel coverage raster (channel 0 if the value
/// carries more than one channel — masks are single-channel, but a tolerant read
/// keeps the summary kind-agnostic). The `tile` rect is intersected with the
/// mask's extent, so a tile partly outside the mask summarizes only the in-bounds
/// pixels. The scan visits the tile's pixels once in row-major order.
#[must_use]
pub fn summarize_tile(mask: &ResourceValue, tile: Rect) -> MaskTileSummary {
    let extent = mask.extent();
    let clipped = tile.clamp_to_extent(extent);
    let channels = mask.channels().max(1) as usize;
    let width = extent.width as usize;
    let samples = mask.samples();

    let mut min_coverage = f32::INFINITY;
    let mut max_coverage = 0.0_f32;
    let mut nonzero_count = 0_u64;
    let mut nonzero = Region::EMPTY;

    if !clipped.is_empty() {
        for y in clipped.y0..clipped.y1 {
            for x in clipped.x0..clipped.x1 {
                // `clipped` is intersected with the extent, so 0 <= x,y < extent,
                // which fits `usize`; the cast is exact.
                #[allow(
                    clippy::cast_possible_truncation,
                    clippy::cast_sign_loss,
                    reason = "x,y are clamped to [0, extent) which fits usize"
                )]
                let pixel = (y as usize) * width + (x as usize);
                let coverage = samples.get(pixel * channels).copied().unwrap_or(0.0);
                min_coverage = min_coverage.min(coverage);
                max_coverage = max_coverage.max(coverage);
                if coverage > 0.0 {
                    nonzero_count += 1;
                    nonzero.push(Rect::new(x, y, x + 1, y + 1));
                }
            }
        }
    }

    // An all-zero (or out-of-bounds) tile reports min == 0.0 rather than +inf.
    if min_coverage.is_infinite() {
        min_coverage = 0.0;
    }

    MaskTileSummary {
        tile: clipped,
        min_coverage,
        max_coverage,
        nonzero_count,
        nonzero_bounds: nonzero,
    }
}

/// The whole-mask occupancy index: a [`MaskTileSummary`] per tile of a
/// [`TileGrid`], plus aggregate non-zero bounds (`plan.md` §11.4).
///
/// The aggregate [`nonzero_bounds`](MaskSummary::nonzero_bounds) is the union of
/// the per-tile non-zero bounds, which (because the tiles partition the mask) is
/// the mask's overall non-zero bounding box — the value a `mask.bounds`
/// assertion compares against.
#[derive(Debug, Clone)]
pub struct MaskSummary {
    grid: TileGrid,
    tiles: Vec<(Tile, MaskTileSummary)>,
    nonzero_bounds: Region,
}

impl MaskSummary {
    /// Summarize `mask` over `grid`, computing a [`MaskTileSummary`] for every
    /// tile in row-major order.
    #[must_use]
    pub fn compute(mask: &ResourceValue, grid: TileGrid) -> Self {
        let mut tiles = Vec::with_capacity(grid.tile_count() as usize);
        let mut nonzero_bounds = Region::EMPTY;
        for tile in grid.tiles() {
            let summary = summarize_tile(mask, tile.rect);
            nonzero_bounds = nonzero_bounds.union(summary.nonzero_bounds());
            tiles.push((tile, summary));
        }
        Self {
            grid,
            tiles,
            nonzero_bounds,
        }
    }

    /// The grid this summary is computed over.
    #[must_use]
    pub const fn grid(&self) -> TileGrid {
        self.grid
    }

    /// The per-tile summaries, in row-major tile order.
    #[must_use]
    pub fn tiles(&self) -> &[(Tile, MaskTileSummary)] {
        &self.tiles
    }

    /// The mask's overall non-zero bounding box — the union of the per-tile
    /// non-zero bounds, equal to `mask.bounds`.
    #[must_use]
    pub const fn nonzero_bounds(&self) -> Region {
        self.nonzero_bounds
    }

    /// The number of [`Empty`](Occupancy::Empty) tiles — the tiles a masked
    /// branch can skip as identity.
    #[must_use]
    pub fn empty_tile_count(&self) -> usize {
        self.tiles
            .iter()
            .filter(|(_, s)| s.occupancy() == Occupancy::Empty)
            .count()
    }

    /// The tiles whose occupancy is `occupancy`, in row-major order.
    pub fn tiles_with(
        &self,
        occupancy: Occupancy,
    ) -> impl Iterator<Item = &(Tile, MaskTileSummary)> {
        self.tiles
            .iter()
            .filter(move |(_, s)| s.occupancy() == occupancy)
    }
}

/// The tight bounding box of every strictly-positive pixel of `mask`, computed
/// directly (without tiling) — the reference the per-tile aggregate is checked
/// against (`plan.md` §11.4).
#[must_use]
pub fn dense_nonzero_bounds(mask: &ResourceValue) -> Region {
    let extent = mask.extent();
    summarize_tile(mask, Rect::from_extent(extent)).nonzero_bounds()
}

/// The full domain [`Rect`] of a mask's extent, for callers building a
/// single-tile summary.
#[must_use]
pub const fn full_tile(extent: Extent) -> Rect {
    Rect::from_extent(extent)
}

#[cfg(test)]
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    clippy::cast_lossless,
    clippy::float_cmp,
    reason = "test fixtures build masks from small exact integer pixel indices"
)]
mod tests {
    use super::{MaskSummary, Occupancy, dense_nonzero_bounds, full_tile, summarize_tile};
    use crate::executor::ResourceValue;
    use crate::tile::grid::TileGrid;
    use paintop_ir::{
        CoordinateConvention, Extent, MaskDescriptor, MaskMeaning, Rect, Region,
        ResourceDescriptor, ScalarType, ValidRange,
    };

    fn mask(extent: Extent, samples: Vec<f32>) -> ResourceValue {
        ResourceValue::new(
            ResourceDescriptor::Mask(MaskDescriptor {
                extent,
                scalar: ScalarType::F32,
                range: ValidRange::Bounded { min: 0.0, max: 1.0 },
                meaning: MaskMeaning::Coverage,
                coordinates: CoordinateConvention::PixelCenterUpperLeft,
            }),
            1,
            samples,
        )
        .unwrap()
    }

    /// A `w x h` mask that is `1.0` inside `rect` and `0.0` elsewhere.
    fn rect_mask(extent: Extent, rect: Rect) -> ResourceValue {
        let w = extent.width as i64;
        let h = extent.height as i64;
        let mut samples = vec![0.0_f32; (w * h) as usize];
        for y in 0..h {
            for x in 0..w {
                if rect.contains(x, y) {
                    samples[(y * w + x) as usize] = 1.0;
                }
            }
        }
        mask(extent, samples)
    }

    #[test]
    fn empty_tile_is_identity_and_has_empty_bounds() {
        let extent = Extent::new(8, 8);
        let m = mask(extent, vec![0.0; 64]);
        let s = summarize_tile(&m, Rect::new(0, 0, 4, 4));
        assert_eq!(s.occupancy(), Occupancy::Empty);
        assert!(s.is_identity());
        assert_eq!(s.nonzero_count(), 0);
        assert_eq!(s.max_coverage(), 0.0);
        assert_eq!(s.min_coverage(), 0.0);
        assert!(s.nonzero_bounds().is_empty());
    }

    #[test]
    fn full_tile_is_full_occupancy() {
        let extent = Extent::new(8, 8);
        let m = mask(extent, vec![1.0; 64]);
        let s = summarize_tile(&m, Rect::new(0, 0, 4, 4));
        assert_eq!(s.occupancy(), Occupancy::Full);
        assert!(!s.is_identity());
        assert_eq!(s.nonzero_count(), 16);
        assert_eq!(s.min_coverage(), 1.0);
        assert_eq!(s.max_coverage(), 1.0);
        assert_eq!(s.nonzero_bounds().bounding_rect(), Rect::new(0, 0, 4, 4));
    }

    #[test]
    fn partially_covered_tile_is_mixed_with_tight_bounds() {
        let extent = Extent::new(8, 8);
        // Cover a 2x2 block at (1,1)..(3,3).
        let m = rect_mask(extent, Rect::new(1, 1, 3, 3));
        let s = summarize_tile(&m, Rect::new(0, 0, 4, 4));
        assert_eq!(s.occupancy(), Occupancy::Mixed);
        assert_eq!(s.nonzero_count(), 4);
        assert_eq!(s.min_coverage(), 0.0); // the surrounding zeros
        assert_eq!(s.max_coverage(), 1.0);
        // Tight non-zero bounds.
        assert_eq!(s.nonzero_bounds().bounding_rect(), Rect::new(1, 1, 3, 3));
    }

    #[test]
    fn fractional_full_coverage_is_mixed_not_full() {
        // Uniform but < 1.0 coverage is Mixed, not Full (a blend must still read).
        let extent = Extent::new(4, 4);
        let m = mask(extent, vec![0.5; 16]);
        let s = summarize_tile(&m, Rect::from_extent(extent));
        assert_eq!(s.occupancy(), Occupancy::Mixed);
        assert_eq!(s.nonzero_count(), 16);
        assert_eq!(s.min_coverage(), 0.5);
        assert_eq!(s.max_coverage(), 0.5);
    }

    #[test]
    fn tile_partly_outside_the_mask_summarizes_in_bounds_only() {
        let extent = Extent::new(6, 6);
        let m = mask(extent, vec![1.0; 36]);
        // Tile extends past the right/bottom edge; only the 6x6 - overlap counts.
        let s = summarize_tile(&m, Rect::new(4, 4, 12, 12));
        assert_eq!(s.tile(), Rect::new(4, 4, 6, 6));
        assert_eq!(s.nonzero_count(), 4);
        assert_eq!(s.occupancy(), Occupancy::Full);
    }

    #[test]
    fn rect_mask_aggregate_bounds_equal_dense_bounds() {
        // Scattered occupancy across multiple tiles: the union of per-tile
        // non-zero bounds equals the mask's overall non-zero bounds.
        let extent = Extent::new(64, 64);
        let m = rect_mask(extent, Rect::new(10, 12, 50, 40));
        let grid = TileGrid::new(extent, 16); // 4x4 tiles

        let summary = MaskSummary::compute(&m, grid);
        let dense = dense_nonzero_bounds(&m);
        assert_eq!(summary.nonzero_bounds(), dense);
        assert_eq!(dense.bounding_rect(), Rect::new(10, 12, 50, 40));
    }

    #[test]
    fn ellipse_mask_bounds_match_dense() {
        // An ellipse (curved boundary) produces scattered per-tile occupancy; the
        // aggregate bounds must still equal the dense bounds.
        let extent = Extent::new(48, 48);
        let cx = 24.0;
        let cy = 24.0;
        let rx = 12.0;
        let ry = 9.0;
        let w = extent.width as i64;
        let mut samples = vec![0.0_f32; (w * w) as usize];
        for y in 0..w {
            for x in 0..w {
                let nx = (x as f64 + 0.5 - cx) / rx;
                let ny = (y as f64 + 0.5 - cy) / ry;
                if nx * nx + ny * ny <= 1.0 {
                    samples[(y * w + x) as usize] = 1.0;
                }
            }
        }
        let m = mask(extent, samples);
        let grid = TileGrid::new(extent, 16);
        let summary = MaskSummary::compute(&m, grid);
        assert_eq!(summary.nonzero_bounds(), dense_nonzero_bounds(&m));
        // Some tiles are empty (corners), some mixed (boundary), some full (center).
        assert!(summary.empty_tile_count() > 0);
        assert!(summary.tiles_with(Occupancy::Mixed).count() > 0);
    }

    #[test]
    fn scattered_mask_classifies_each_tile() {
        // Two disjoint covered blocks; tiles between them are empty.
        let extent = Extent::new(32, 16);
        let w = extent.width as i64;
        let mut samples = vec![0.0_f32; (w * 16) as usize];
        let set = |s: &mut Vec<f32>, x: i64, y: i64| s[(y * w + x) as usize] = 1.0;
        for y in 0..4 {
            for x in 0..4 {
                set(&mut samples, x, y);
                set(&mut samples, 28 + x, 12 + y);
            }
        }
        let m = mask(extent, samples);
        let grid = TileGrid::new(extent, 8); // 4x2 tiles
        let summary = MaskSummary::compute(&m, grid);
        // Top-left and bottom-right tiles are mixed; the rest empty.
        assert_eq!(summary.tiles_with(Occupancy::Mixed).count(), 2);
        assert_eq!(summary.empty_tile_count(), 6);
        // Aggregate bounds span both blocks.
        assert_eq!(
            summary.nonzero_bounds().bounding_rect(),
            Rect::new(0, 0, 32, 16)
        );
    }

    #[test]
    fn empty_tile_identity_matches_dense_passthrough() {
        // Differential: an empty-tile masked blend is identity. Model the gated
        // op as `out = lerp(base, fg, coverage)`; on an empty tile coverage == 0
        // everywhere, so `out == base` — bit-identical to the dense computation
        // restricted to that tile.
        let extent = Extent::new(16, 16);
        let m = rect_mask(extent, Rect::new(0, 0, 4, 4)); // covered only top-left
        let grid = TileGrid::new(extent, 8); // 2x2 tiles
        let base: Vec<f32> = (0..256).map(|i| i as f32).collect();
        let fg: Vec<f32> = (0..256).map(|i| 1000.0 + i as f32).collect();

        // Coverage is binary here, so the blend is an exact pixel select (no
        // floating-point lerp): a covered pixel takes `fg`, an uncovered one
        // `base`. The fast path replaces an empty tile wholesale with `base`.
        let covered = Rect::new(0, 0, 4, 4);
        let summary = MaskSummary::compute(&m, grid);
        for (tile, s) in summary.tiles() {
            let r = tile.rect;
            for y in r.y0..r.y1 {
                for x in r.x0..r.x1 {
                    let p = (y * 16 + x) as usize;
                    let dense_out = if covered.contains(x, y) {
                        fg[p]
                    } else {
                        base[p]
                    };
                    // The fast path: an empty tile yields base unchanged.
                    let fast_out = if s.is_identity() {
                        base[p]
                    } else if covered.contains(x, y) {
                        fg[p]
                    } else {
                        base[p]
                    };
                    assert_eq!(fast_out.to_bits(), dense_out.to_bits(), "seam at ({x},{y})");
                }
            }
        }
    }

    #[test]
    fn full_tile_helper_is_the_extent_rect() {
        assert_eq!(full_tile(Extent::new(7, 9)), Rect::new(0, 0, 7, 9));
    }

    #[test]
    fn empty_grid_summary_has_empty_bounds() {
        let extent = Extent::new(0, 0);
        let m = mask(extent, vec![]);
        let summary = MaskSummary::compute(&m, TileGrid::with_default(extent));
        assert!(summary.nonzero_bounds().is_empty());
        assert_eq!(summary.tiles().len(), 0);
        assert_eq!(dense_nonzero_bounds(&m), Region::EMPTY);
    }
}
