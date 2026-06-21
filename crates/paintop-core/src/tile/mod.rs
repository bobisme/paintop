//! The logical tile model and sparse mask occupancy index (`plan.md` §11).
//!
//! M2 execution is tiled: an image is partitioned into a deterministic grid of
//! output tiles, each op declares the input *halo* one output tile reads, and
//! masks carry a per-tile occupancy summary so the executor can skip tiles a
//! masked branch leaves untouched. This module owns the *geometry and occupancy*
//! foundation those rest on:
//!
//! * [`grid`] — the [`TileGrid`] partition of an [`Extent`](paintop_ir::Extent)
//!   into row-major output tiles, the per-op [`input_tile_region`] halo, and the
//!   [`HaloClass`] classification of a tile's input footprint;
//! * [`mask_summary`] — the [`MaskTileSummary`] / [`MaskSummary`] occupancy index
//!   (min/max coverage, non-zero count, non-zero bounds) with empty/full/mixed
//!   [`Occupancy`] classification and the empty-tile identity fast path.
//!
//! The demand-driven *scheduler* and the *pointwise tiled execution path* that
//! consume these are later bones.

pub mod execute;
pub mod grid;
pub mod mask_summary;
pub mod parallel;
pub mod reduce;
pub mod schedule;

pub use execute::{TileStats, TiledExecution, execute_tiled, execute_tiled_with, export_region};
pub use grid::{
    DEFAULT_TILE_SIZE, HaloClass, SMALL_TILE_SIZE, Tile, TileGrid, classify_halo,
    input_tile_region, is_raster_kind,
};
pub use mask_summary::{
    MaskSummary, MaskTileSummary, Occupancy, dense_nonzero_bounds, summarize_tile,
};
pub use parallel::ThreadCap;
pub use reduce::{ArgMax, Extremum, TiledSum, pairwise_sum};
pub use schedule::{
    LivenessPoint, LivenessTrace, TileInput, TileSchedule, TileWorkItem, schedule_tiles,
};
