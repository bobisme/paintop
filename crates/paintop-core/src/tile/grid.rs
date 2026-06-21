//! The logical tile model (`plan.md` §11.1–§11.2).
//!
//! Tiling is the foundation M2 execution rests on: a 4K `f32` image is hundreds
//! of megabytes, but most edits touch a small region, so the executor must be
//! able to evaluate *only the tiles a demand actually needs* and to bound its
//! working set. This module owns the geometry half of that — the partition of an
//! image extent into a deterministic grid of output tiles, and the per-op
//! computation of the *input halo* each output tile reads.
//!
//! # The grid
//!
//! A [`TileGrid`] partitions the full domain `[0, width) × [0, height)` of an
//! [`Extent`] into a row-major sequence of axis-aligned [`Rect`] tiles of a
//! configurable base size (initially 128 or 256 px per the plan). Edge tiles are
//! clipped to the extent, so every pixel of the image belongs to exactly one
//! tile and the tiles tile the image with no overlap and no gap. The grid is a
//! pure function of `(extent, tile_size)`, so two runs enumerate the same tiles
//! in the same order — the determinism the tiled-vs-whole differential relies on.
//!
//! # Per-op input halo
//!
//! A tiled op producing output tile `T` does not in general read only `T` of its
//! input: a blur of radius `r` reads `T` dilated by `r`, a geometric warp reads a
//! transformed footprint, a full-domain reduction reads the whole input. The
//! *input region* an output tile needs is exactly what the op's backward ROI
//! contract ([`OpContract::required_inputs`]) already computes, so
//! [`input_tile_region`] drives that contract for a single output tile and clamps
//! the result to the input extent. The tile scheduler (a later bone) uses this to
//! decide which input tiles to evaluate before an output tile, and the bounded
//! memory proof relies on the halo of a pointwise op being exactly the tile.

use paintop_ir::{
    Descriptors, Extent, OpContract, OutputRegions, Rect, Region, ResourceKind, Result,
};

/// The default logical base tile edge in pixels (`plan.md` §11.2: "initially
/// 128×128 or 256×256"). 256 keeps the tile count low on 4K inputs while staying
/// within a bounded working set.
pub const DEFAULT_TILE_SIZE: u32 = 256;

/// The alternative base tile edge the plan calls out, for the smaller-tile
/// configuration.
pub const SMALL_TILE_SIZE: u32 = 128;

/// A deterministic partition of an [`Extent`] into a row-major grid of
/// fixed-size output tiles (`plan.md` §11.2).
///
/// The grid covers exactly the full domain `[0, width) × [0, height)`: tiles do
/// not overlap, leave no gap, and edge tiles are clipped to the extent. The base
/// `tile_size` is the edge of an interior tile; the last column/row may be
/// narrower when the extent is not a multiple of the tile size. A zero-area
/// extent yields an empty grid.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TileGrid {
    extent: Extent,
    tile_size: u32,
}

impl TileGrid {
    /// Build a grid over `extent` with the given square base `tile_size`.
    ///
    /// A `tile_size` of `0` is treated as `1` so the grid is always well-formed
    /// (one tile per pixel) rather than dividing by zero; callers should pass a
    /// realistic size ([`DEFAULT_TILE_SIZE`] or [`SMALL_TILE_SIZE`]).
    #[must_use]
    pub const fn new(extent: Extent, tile_size: u32) -> Self {
        let tile_size = if tile_size == 0 { 1 } else { tile_size };
        Self { extent, tile_size }
    }

    /// Build a grid over `extent` with the [`DEFAULT_TILE_SIZE`] base tile.
    #[must_use]
    pub const fn with_default(extent: Extent) -> Self {
        Self::new(extent, DEFAULT_TILE_SIZE)
    }

    /// The extent this grid partitions.
    #[must_use]
    pub const fn extent(&self) -> Extent {
        self.extent
    }

    /// The base (interior) tile edge in pixels.
    #[must_use]
    pub const fn tile_size(&self) -> u32 {
        self.tile_size
    }

    /// The number of tile columns (the last may be narrower than `tile_size`).
    #[must_use]
    pub const fn cols(&self) -> u32 {
        div_ceil(self.extent.width, self.tile_size)
    }

    /// The number of tile rows (the last may be shorter than `tile_size`).
    #[must_use]
    pub const fn rows(&self) -> u32 {
        div_ceil(self.extent.height, self.tile_size)
    }

    /// The total number of tiles, `cols * rows`.
    #[must_use]
    pub const fn tile_count(&self) -> u32 {
        self.cols() * self.rows()
    }

    /// The tile at grid position `(col, row)`, clipped to the extent, or [`None`]
    /// if the position is outside the grid.
    #[must_use]
    pub fn tile_at(&self, col: u32, row: u32) -> Option<Rect> {
        if col >= self.cols() || row >= self.rows() {
            return None;
        }
        let x0 = i64::from(col) * i64::from(self.tile_size);
        let y0 = i64::from(row) * i64::from(self.tile_size);
        let x1 = (x0 + i64::from(self.tile_size)).min(i64::from(self.extent.width));
        let y1 = (y0 + i64::from(self.tile_size)).min(i64::from(self.extent.height));
        Some(Rect::new(x0, y0, x1, y1))
    }

    /// Every tile, in row-major order (row 0 left-to-right, then row 1, …).
    ///
    /// The order is fixed and total, so an enumeration is deterministic — the
    /// property a fixed merge tree and the tiled-vs-whole differential depend on.
    pub fn tiles(&self) -> impl Iterator<Item = Tile> + '_ {
        (0..self.rows()).flat_map(move |row| {
            (0..self.cols()).filter_map(move |col| {
                self.tile_at(col, row).map(|rect| Tile {
                    col,
                    row,
                    index: row * self.cols() + col,
                    rect,
                })
            })
        })
    }

    /// The tiles whose area intersects `region` — the output tiles a demanded
    /// [`Region`] actually touches (`plan.md` §11.3). An empty region touches no
    /// tile.
    pub fn tiles_in_region(&self, region: Region) -> impl Iterator<Item = Tile> + '_ {
        let bounds = region.bounding_rect();
        self.tiles()
            .filter(move |tile| !tile.rect.intersect(bounds).is_empty())
    }
}

/// One output tile of a [`TileGrid`]: its grid coordinates, row-major index, and
/// clipped pixel [`Rect`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Tile {
    /// The tile column (0-based, left to right).
    pub col: u32,
    /// The tile row (0-based, top to bottom).
    pub row: u32,
    /// The row-major tile index, `row * cols + col`.
    pub index: u32,
    /// The tile's clipped pixel region.
    pub rect: Rect,
}

/// The input region one output `tile` of an op needs, per the op's backward ROI
/// contract (`plan.md` §11.2–§11.3).
///
/// This is the per-output-tile input halo: the op's
/// [`required_inputs`](OpContract::required_inputs) maps the output tile `Rect`
/// to the input region that produces it (the same `Rect` for a pointwise op,
/// `Rect` dilated by the halo for a blur, the whole input for a full-domain
/// reduction), and the result is clamped to the input port's extent so it never
/// names a pixel the producer does not have.
///
/// `input_port` selects which input port's region to return; the op may read
/// several inputs, and the scheduler asks per port. The result is [`Region::EMPTY`]
/// if the op does not read `input_port` for this output tile.
///
/// # Errors
/// Propagates any error the op's
/// [`required_inputs`](OpContract::required_inputs) raises.
pub fn input_tile_region(
    contract: &dyn OpContract,
    output_port: &str,
    input_port: &str,
    tile: Rect,
    inputs: &Descriptors,
    params: &serde_json::Value,
) -> Result<Region> {
    let mut requested = OutputRegions::new();
    requested.insert(output_port.to_owned(), tile);
    let needed = contract.required_inputs(&requested, inputs, params)?;
    let Some(region) = needed.get(input_port).copied() else {
        return Ok(Region::EMPTY);
    };
    // Clamp to the input port's extent so the halo never names a missing pixel.
    let region = Region::from_rect(region);
    Ok(inputs.get(input_port).map_or(region, |descriptor| {
        region.clamp_to_extent(descriptor.extent())
    }))
}

/// The kind-agnostic halo classification of a single output tile against the
/// input region it reads, for a one-input op.
///
/// A *pointwise* tile reads exactly its own footprint; a *haloed* tile reads a
/// strictly larger region (a blur, a geometric footprint); a *whole-domain* tile
/// reads the entire input. The scheduler uses this to size the working set and to
/// decide whether tiling an op is sound without recomputing the contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HaloClass {
    /// The input region equals the output tile (pointwise).
    Pointwise,
    /// The input region is larger than the tile but not the whole input.
    Haloed,
    /// The input region is the whole input extent (full-domain).
    WholeDomain,
}

/// Classify how the `input_region` an op reads for output `tile` relates to the
/// tile, against the input's full `extent`.
#[must_use]
pub fn classify_halo(tile: Rect, input_region: Region, extent: Extent) -> HaloClass {
    let read = input_region.bounding_rect();
    if read == Rect::from_extent(extent) && read != tile {
        HaloClass::WholeDomain
    } else if read.contains_rect(tile) && read == tile {
        HaloClass::Pointwise
    } else {
        HaloClass::Haloed
    }
}

/// Whether `kind` is a raster resource kind tiling applies to.
///
/// An image, mask, field, or SDF carries per-pixel samples; a report carries no
/// raster, so it is never tiled. Kind-agnostic so a future raster kind variant is
/// handled by the catch-all rather than breaking the build.
#[must_use]
pub const fn is_raster_kind(kind: ResourceKind) -> bool {
    !matches!(kind, ResourceKind::Report | ResourceKind::CandidateSet)
}

/// `ceil(a / b)` for positive `b`, computed without overflow for `u32`.
const fn div_ceil(a: u32, b: u32) -> u32 {
    if b == 0 {
        return 0;
    }
    a / b + if a.is_multiple_of(b) { 0 } else { 1 }
}

#[cfg(test)]
mod tests {
    use super::{
        DEFAULT_TILE_SIZE, HaloClass, SMALL_TILE_SIZE, TileGrid, classify_halo, div_ceil,
        input_tile_region, is_raster_kind,
    };
    use paintop_ir::{
        AlphaRepresentation, ChannelLayout, ColorEncoding, ColorRange, CoordinateConvention,
        Descriptors, Extent, ImageDescriptor, InputRegions, OpContract, OutputDescriptors,
        OutputRegions, Rect, Region, ResourceDescriptor, ResourceKind, ScalarType, SemanticRole,
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

    fn descriptors(extent: Extent) -> Descriptors {
        let mut d = Descriptors::new();
        d.insert("image".to_owned(), image(extent));
        d
    }

    // A pointwise op: input region == output region.
    struct Pointwise;
    impl OpContract for Pointwise {
        fn declared_inputs(&self) -> Vec<(String, ResourceKind)> {
            vec![("image".to_owned(), ResourceKind::Image)]
        }
        fn declared_outputs(&self) -> Vec<(String, ResourceKind)> {
            vec![("image".to_owned(), ResourceKind::Image)]
        }
        fn infer_outputs(
            &self,
            i: &Descriptors,
            _p: &serde_json::Value,
        ) -> paintop_ir::Result<OutputDescriptors> {
            let mut o = OutputDescriptors::new();
            o.insert("image".to_owned(), i["image"]);
            Ok(o)
        }
        fn required_inputs(
            &self,
            o: &OutputRegions,
            _i: &Descriptors,
            _p: &serde_json::Value,
        ) -> paintop_ir::Result<InputRegions> {
            let mut r = InputRegions::new();
            if let Some(region) = o.get("image") {
                r.insert("image".to_owned(), *region);
            }
            Ok(r)
        }
        fn validate_postconditions(
            &self,
            _o: &OutputDescriptors,
            _p: &serde_json::Value,
        ) -> paintop_ir::Result<Vec<paintop_ir::AssertionResult>> {
            Ok(vec![])
        }
    }

    // A halo op: input region == output region dilated by a fixed radius, clamped.
    struct Halo(u32);
    impl OpContract for Halo {
        fn declared_inputs(&self) -> Vec<(String, ResourceKind)> {
            vec![("image".to_owned(), ResourceKind::Image)]
        }
        fn declared_outputs(&self) -> Vec<(String, ResourceKind)> {
            vec![("image".to_owned(), ResourceKind::Image)]
        }
        fn infer_outputs(
            &self,
            i: &Descriptors,
            _p: &serde_json::Value,
        ) -> paintop_ir::Result<OutputDescriptors> {
            let mut o = OutputDescriptors::new();
            o.insert("image".to_owned(), i["image"]);
            Ok(o)
        }
        fn required_inputs(
            &self,
            o: &OutputRegions,
            i: &Descriptors,
            _p: &serde_json::Value,
        ) -> paintop_ir::Result<InputRegions> {
            let mut r = InputRegions::new();
            if let Some(region) = o.get("image") {
                let extent = i["image"].extent();
                let grown = Region::from_rect(*region)
                    .dilate(self.0)
                    .clamp_to_extent(extent);
                r.insert("image".to_owned(), grown.bounding_rect());
            }
            Ok(r)
        }
        fn validate_postconditions(
            &self,
            _o: &OutputDescriptors,
            _p: &serde_json::Value,
        ) -> paintop_ir::Result<Vec<paintop_ir::AssertionResult>> {
            Ok(vec![])
        }
    }

    // A full-domain op: any output reads the whole input.
    struct FullDomain;
    impl OpContract for FullDomain {
        fn declared_inputs(&self) -> Vec<(String, ResourceKind)> {
            vec![("image".to_owned(), ResourceKind::Image)]
        }
        fn declared_outputs(&self) -> Vec<(String, ResourceKind)> {
            vec![("image".to_owned(), ResourceKind::Image)]
        }
        fn infer_outputs(
            &self,
            i: &Descriptors,
            _p: &serde_json::Value,
        ) -> paintop_ir::Result<OutputDescriptors> {
            let mut o = OutputDescriptors::new();
            o.insert("image".to_owned(), i["image"]);
            Ok(o)
        }
        fn required_inputs(
            &self,
            _o: &OutputRegions,
            i: &Descriptors,
            _p: &serde_json::Value,
        ) -> paintop_ir::Result<InputRegions> {
            let mut r = InputRegions::new();
            let extent = i["image"].extent();
            r.insert("image".to_owned(), Rect::from_extent(extent));
            Ok(r)
        }
        fn validate_postconditions(
            &self,
            _o: &OutputDescriptors,
            _p: &serde_json::Value,
        ) -> paintop_ir::Result<Vec<paintop_ir::AssertionResult>> {
            Ok(vec![])
        }
    }

    #[test]
    fn grid_tiles_cover_the_extent_without_overlap_or_gap() {
        let extent = Extent::new(300, 200);
        let grid = TileGrid::new(extent, 128);
        assert_eq!(grid.cols(), 3); // 128, 128, 44
        assert_eq!(grid.rows(), 2); // 128, 72
        assert_eq!(grid.tile_count(), 6);

        // Every pixel belongs to exactly one tile: total area equals the extent.
        let total: i64 = grid.tiles().map(|t| t.rect.area()).sum();
        assert_eq!(total, i64::from(extent.width) * i64::from(extent.height));

        // Tiles are pairwise disjoint.
        let rects: Vec<_> = grid.tiles().map(|t| t.rect).collect();
        for i in 0..rects.len() {
            for j in (i + 1)..rects.len() {
                assert!(
                    rects[i].intersect(rects[j]).is_empty(),
                    "tiles {i} and {j} overlap"
                );
            }
        }
    }

    #[test]
    fn edge_tiles_are_clipped_to_the_extent() {
        let grid = TileGrid::new(Extent::new(300, 200), 128);
        // Last column tile is 44 wide (300 - 256); last row tile is 72 tall.
        let corner = grid.tile_at(2, 1).unwrap();
        assert_eq!(corner, Rect::new(256, 128, 300, 200));
        assert!(grid.tile_at(3, 0).is_none());
        assert!(grid.tile_at(0, 2).is_none());
    }

    #[test]
    fn tiles_enumerate_in_row_major_order_with_dense_indices() {
        let grid = TileGrid::new(Extent::new(300, 200), 128);
        let tiles: Vec<_> = grid.tiles().collect();
        assert_eq!(tiles.len(), 6);
        for (expected, tile) in tiles.iter().enumerate() {
            assert_eq!(tile.index as usize, expected);
        }
        // Row-major: (0,0),(1,0),(2,0),(0,1),...
        assert_eq!((tiles[0].col, tiles[0].row), (0, 0));
        assert_eq!((tiles[3].col, tiles[3].row), (0, 1));
    }

    #[test]
    fn exact_multiple_extent_has_uniform_tiles() {
        let grid = TileGrid::new(Extent::new(512, 256), 256);
        assert_eq!(grid.tile_count(), 2);
        for tile in grid.tiles() {
            assert_eq!(tile.rect.width(), 256);
            assert_eq!(tile.rect.height(), 256);
        }
    }

    #[test]
    fn empty_extent_yields_no_tiles() {
        let grid = TileGrid::new(Extent::new(0, 0), 256);
        assert_eq!(grid.tile_count(), 0);
        assert_eq!(grid.tiles().count(), 0);
    }

    #[test]
    fn tiles_in_region_selects_only_touched_tiles() {
        let grid = TileGrid::new(Extent::new(512, 512), 128); // 4x4 = 16 tiles
        // A region inside the top-left tile only.
        let region = Region::from_rect(Rect::new(10, 10, 40, 40));
        let touched: Vec<_> = grid.tiles_in_region(region).collect();
        assert_eq!(touched.len(), 1);
        assert_eq!((touched[0].col, touched[0].row), (0, 0));

        // A region straddling two tiles horizontally.
        let region = Region::from_rect(Rect::new(120, 10, 200, 40));
        assert_eq!(grid.tiles_in_region(region).count(), 2);

        // Empty region touches nothing.
        assert_eq!(grid.tiles_in_region(Region::EMPTY).count(), 0);
    }

    #[test]
    fn pointwise_input_tile_region_equals_the_tile() {
        let extent = Extent::new(512, 512);
        let tile = Rect::new(128, 128, 256, 256);
        let region = input_tile_region(
            &Pointwise,
            "image",
            "image",
            tile,
            &descriptors(extent),
            &serde_json::Value::Null,
        )
        .unwrap();
        assert_eq!(region.bounding_rect(), tile);
        assert_eq!(classify_halo(tile, region, extent), HaloClass::Pointwise);
    }

    #[test]
    fn haloed_input_tile_region_dilates_and_clamps() {
        let extent = Extent::new(512, 512);
        // An interior tile dilates on every side.
        let tile = Rect::new(128, 128, 256, 256);
        let region = input_tile_region(
            &Halo(4),
            "image",
            "image",
            tile,
            &descriptors(extent),
            &serde_json::Value::Null,
        )
        .unwrap();
        assert_eq!(region.bounding_rect(), Rect::new(124, 124, 260, 260));
        assert_eq!(classify_halo(tile, region, extent), HaloClass::Haloed);

        // A corner tile clamps the halo back to the extent edge.
        let corner = Rect::new(0, 0, 128, 128);
        let region = input_tile_region(
            &Halo(4),
            "image",
            "image",
            corner,
            &descriptors(extent),
            &serde_json::Value::Null,
        )
        .unwrap();
        assert_eq!(region.bounding_rect(), Rect::new(0, 0, 132, 132));
    }

    #[test]
    fn full_domain_input_tile_region_is_the_whole_input() {
        let extent = Extent::new(512, 512);
        let tile = Rect::new(0, 0, 128, 128);
        let region = input_tile_region(
            &FullDomain,
            "image",
            "image",
            tile,
            &descriptors(extent),
            &serde_json::Value::Null,
        )
        .unwrap();
        assert_eq!(region.bounding_rect(), Rect::from_extent(extent));
        assert_eq!(classify_halo(tile, region, extent), HaloClass::WholeDomain);
    }

    #[test]
    fn input_tile_region_is_empty_for_an_unread_port() {
        let extent = Extent::new(64, 64);
        let region = input_tile_region(
            &Pointwise,
            "image",
            "mask", // not read by Pointwise
            Rect::new(0, 0, 32, 32),
            &descriptors(extent),
            &serde_json::Value::Null,
        )
        .unwrap();
        assert!(region.is_empty());
    }

    #[test]
    fn tile_size_constants_and_zero_guard() {
        assert_eq!(DEFAULT_TILE_SIZE, 256);
        assert_eq!(SMALL_TILE_SIZE, 128);
        // A zero tile size degrades to one pixel per tile rather than dividing by 0.
        let grid = TileGrid::new(Extent::new(2, 2), 0);
        assert_eq!(grid.tile_size(), 1);
        assert_eq!(grid.tile_count(), 4);
    }

    #[test]
    fn raster_kind_classification() {
        assert!(is_raster_kind(ResourceKind::Image));
        assert!(is_raster_kind(ResourceKind::Mask));
        assert!(is_raster_kind(ResourceKind::Field1));
        assert!(is_raster_kind(ResourceKind::SdfMask));
        assert!(!is_raster_kind(ResourceKind::Report));
        assert!(!is_raster_kind(ResourceKind::CandidateSet));
    }

    #[test]
    fn div_ceil_rounds_up() {
        assert_eq!(div_ceil(0, 256), 0);
        assert_eq!(div_ceil(256, 256), 1);
        assert_eq!(div_ceil(257, 256), 2);
        assert_eq!(div_ceil(300, 128), 3);
    }
}
