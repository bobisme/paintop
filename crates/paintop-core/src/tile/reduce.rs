//! Deterministic tiled reductions via a fixed merge tree (`plan.md` §11.2,
//! `AGENT_VERIFICATION` §13 "nondeterministic reduction" signature; bn-25j).
//!
//! A reduction (`analyze.statistics`, `analyze.histogram`, the assertion verdict
//! scans) folds every admitted sample of a resource into a small summary. Under
//! tiled, potentially parallel execution the samples arrive **tile by tile, in an
//! order that depends on the scheduler and the thread count**. A naive running
//! accumulation would then make the result depend on that order — for floating-
//! point sums, *bit-for-bit* — which is the
//! `AGENT_VERIFICATION` §13 "iterative/reduction order" determinism bug.
//!
//! This module removes that dependency. Every reduction here is expressed as the
//! combination of **per-tile partial results** under a **fixed merge tree keyed by
//! the tile's row-major grid index**, never by the order tiles happen to be
//! produced. The merge is therefore a pure function of *which* tiles contributed
//! and *what* each contributed, independent of the schedule and the thread count:
//!
//! * **integer tallies** (histogram bins, finite/non-finite counts, leaking-pixel
//!   counts) combine by exact integer addition — associative and commutative, so
//!   any merge order gives the identical total;
//! * **extrema** (min / max) combine by `min` / `max` — likewise order-free;
//! * **arg-extrema** (the worst leaking pixel) keep the better metric, breaking
//!   ties toward the **lower absolute pixel position**, so the choice is fixed
//!   regardless of which tile is merged first;
//! * **floating-point sums** (`sum`, `sum_sq` for the mean/variance) are the one
//!   genuinely order-sensitive case. They are made deterministic by tagging every
//!   admitted value with its **absolute row-major pixel position**, ordering all
//!   contributions by that position, and reducing the position-ordered sequence
//!   with the same [`pairwise_sum`] tree the whole-image reference uses. Position
//!   order is a total order fixed by the image geometry — independent of how the
//!   image was partitioned into tiles and of which thread produced which tile — so
//!   the result is bit-identical to the sequential whole-image sum over the same
//!   admitted values for **any** tile shape (1-D row strips, 2-D blocks, a single
//!   whole-image "tile"), tile size, and thread count.
//!
//! The accumulation order is thus fixed and documented: **canonical row-major
//! position order**, materialized by sorting contributions on their absolute pixel
//! position before the merge. The differential tests pin that a reduction split
//! across tile sizes and merged in arbitrary (shuffled) order is bit-identical to
//! the single-pass whole-image reduction.

/// The stable pairwise (`f64`) sum of a slice: recursively split the slice in
/// half and add the two halves, with a small left-to-right base case.
///
/// The tree shape is a fixed function of the slice's *order and length*, so the
/// result is bit-identical across runs and machines; the pairwise shape also
/// bounds the accumulated rounding error to `O(log n)` rather than the `O(n)` of a
/// running scalar sum. This is the same primitive `analyze.statistics` reduces
/// with, reproduced here so the tiled merge and the whole-image reference share
/// one canonical tree (any divergence would be a determinism bug).
#[must_use]
pub fn pairwise_sum(values: &[f64]) -> f64 {
    const BLOCK: usize = 8;
    if values.len() <= BLOCK {
        let mut acc = 0.0_f64;
        for &v in values {
            acc += v;
        }
        return acc;
    }
    let mid = values.len() / 2;
    pairwise_sum(&values[..mid]) + pairwise_sum(&values[mid..])
}

/// A deterministic tiled floating-point sum: position-tagged contributions merged
/// in canonical row-major position order and reduced with the [`pairwise_sum`] tree.
///
/// Every admitted value is contributed with its **absolute row-major pixel
/// position** (`y * width + x`, scaled by the channel for an interleaved buffer).
/// [`finish`](Self::finish) sorts all contributions by that position and reduces
/// the position-ordered sequence. Position order is fixed by the image geometry,
/// so the result is **bit-identical to the sequential whole-image
/// [`pairwise_sum`]** over the same admitted values and is independent of the tile
/// shape, the tile size, and the order the contributions arrived in (the scheduler
/// / thread count).
///
/// Contributing a whole tile at once is the common path: [`push_run`](Self::push_run)
/// adds a contiguous run of values starting at a base position, which a 1-D row
/// strip or a single whole-image pass produces directly; a 2-D block tile pushes
/// one run per scan-line.
#[derive(Debug, Default, Clone)]
pub struct TiledSum {
    contributions: Vec<(u64, f64)>,
}

impl TiledSum {
    /// An empty tiled sum.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            contributions: Vec::new(),
        }
    }

    /// Contribute one admitted value at absolute row-major `position`.
    pub fn push(&mut self, position: u64, value: f64) {
        self.contributions.push((position, value));
    }

    /// Contribute a contiguous run of admitted `values` whose first element is at
    /// absolute `base` position and which are stored at unit position stride.
    ///
    /// This is the whole-tile fast path for a 1-D row strip or a whole-image pass;
    /// a 2-D block contributes one run per scan-line.
    pub fn push_run(&mut self, base: u64, values: &[f64]) {
        self.contributions.reserve(values.len());
        for (offset, &value) in values.iter().enumerate() {
            self.contributions.push((base + offset as u64, value));
        }
    }

    /// Reduce every contribution to the canonical sum: sort by absolute position,
    /// then [`pairwise_sum`] the position-ordered values. Bit-identical to the
    /// whole-image sum over the same admitted values regardless of tile shape,
    /// tile size, or push order.
    #[must_use]
    pub fn finish(mut self) -> f64 {
        // A stable sort on the absolute position fixes the accumulation order to
        // canonical row-major order, independent of how tiles were scheduled.
        self.contributions.sort_by_key(|(position, _)| *position);
        let ordered: Vec<f64> = self.contributions.iter().map(|(_, value)| *value).collect();
        pairwise_sum(&ordered)
    }
}

/// A deterministic running extremum that combines order-free (`min` / `max`).
///
/// `None` is the empty extremum; merging is associative and commutative, so a
/// tiled fold gives the same result in any merge order.
#[derive(Debug, Default, Clone, Copy, PartialEq)]
pub struct Extremum {
    min: Option<f64>,
    max: Option<f64>,
}

impl Extremum {
    /// The empty extremum (no samples seen).
    #[must_use]
    pub const fn new() -> Self {
        Self {
            min: None,
            max: None,
        }
    }

    /// Fold one finite sample into the extremum.
    pub fn observe(&mut self, value: f64) {
        self.min = Some(self.min.map_or(value, |m| m.min(value)));
        self.max = Some(self.max.map_or(value, |m| m.max(value)));
    }

    /// Merge another extremum into this one (order-free).
    #[must_use]
    pub fn merge(self, other: Self) -> Self {
        let min = match (self.min, other.min) {
            (Some(a), Some(b)) => Some(a.min(b)),
            (a, b) => a.or(b),
        };
        let max = match (self.max, other.max) {
            (Some(a), Some(b)) => Some(a.max(b)),
            (a, b) => a.or(b),
        };
        Self { min, max }
    }

    /// The minimum observed sample, if any.
    #[must_use]
    pub const fn min(self) -> Option<f64> {
        self.min
    }

    /// The maximum observed sample, if any.
    #[must_use]
    pub const fn max(self) -> Option<f64> {
        self.max
    }
}

/// A deterministic arg-extremum.
///
/// Tracks the largest metric seen, breaking ties toward the **lowest absolute
/// position**, so the winner is fixed regardless of merge order (`plan.md` §11.2;
/// the assertion "worst pixel" reduction).
#[derive(Debug, Default, Clone, Copy, PartialEq)]
pub struct ArgMax {
    best: Option<(f64, u64)>,
}

impl ArgMax {
    /// The empty arg-max.
    #[must_use]
    pub const fn new() -> Self {
        Self { best: None }
    }

    /// Fold one `(metric, position)` candidate in. A strictly larger metric wins;
    /// an equal metric keeps the **lower position** (first-wins by position).
    #[allow(
        clippy::float_cmp,
        reason = "the tie-break needs an exact metric equality; positions then \
                  decide deterministically, so no epsilon is appropriate"
    )]
    pub fn observe(&mut self, metric: f64, position: u64) {
        let better = match self.best {
            None => true,
            Some((m, p)) => metric > m || (metric == m && position < p),
        };
        if better {
            self.best = Some((metric, position));
        }
    }

    /// Merge another arg-max in (order-free: same metric/position tie-break).
    #[must_use]
    pub fn merge(self, other: Self) -> Self {
        let mut merged = self;
        if let Some((metric, position)) = other.best {
            merged.observe(metric, position);
        }
        merged
    }

    /// The winning `(metric, position)`, if any candidate was seen.
    #[must_use]
    pub const fn winner(self) -> Option<(f64, u64)> {
        self.best
    }
}

#[cfg(test)]
#[allow(
    clippy::float_cmp,
    clippy::unwrap_used,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    reason = "the determinism property is exactly bit-identical (==) equality; \
              fixtures build f64 values from small integer indices"
)]
mod tests {
    use super::{ArgMax, Extremum, TiledSum, pairwise_sum};

    /// A value sequence with enough magnitude spread that a different summation
    /// order would round differently (so the bit-identity claim has teeth).
    fn spread_values(n: usize) -> Vec<f64> {
        (0..n)
            .map(|i| {
                let x = i as f64;
                // Interleave large and tiny magnitudes.
                if i % 2 == 0 {
                    1.0e8 + x
                } else {
                    1.0e-8 * (x + 1.0)
                }
            })
            .collect()
    }

    /// Partition `[0, n)` into contiguous runs of `tile_len` and feed them to a
    /// `TiledSum` in the given push order, returning the finished sum. Each run is
    /// pushed at its absolute base position, so the position-ordered merge must
    /// reconstruct the whole sequence regardless of push order.
    fn tiled_sum_in_order(values: &[f64], tile_len: usize, push_order: &[usize]) -> f64 {
        let step = tile_len.max(1);
        let tiles: Vec<&[f64]> = values.chunks(step).collect();
        let mut sum = TiledSum::new();
        for &tile_index in push_order {
            let base = (tile_index * step) as u64;
            sum.push_run(base, tiles[tile_index]);
        }
        sum.finish()
    }

    #[test]
    fn tiled_sum_matches_whole_image_across_tile_sizes() {
        let values = spread_values(1000);
        let whole = pairwise_sum(&values);
        for tile_len in [1, 2, 3, 7, 8, 16, 64, 128, 333, 1000, 4096] {
            let tile_count = values.len().div_ceil(tile_len.max(1));
            let order: Vec<usize> = (0..tile_count).collect();
            let got = tiled_sum_in_order(&values, tile_len, &order);
            assert_eq!(
                got, whole,
                "tiled sum at tile_len {tile_len} differs from whole-image"
            );
        }
    }

    #[test]
    fn tiled_sum_is_independent_of_push_order() {
        let values = spread_values(777);
        let whole = pairwise_sum(&values);
        let tile_len = 16;
        let tile_count = values.len().div_ceil(tile_len);

        // Forward, reverse, and a deterministic shuffle (simulating thread races)
        // must all reduce to the identical bits.
        let forward: Vec<usize> = (0..tile_count).collect();
        let reverse: Vec<usize> = (0..tile_count).rev().collect();
        let shuffled: Vec<usize> = {
            let mut v: Vec<usize> = (0..tile_count).collect();
            // A fixed pseudo-shuffle: swap pairs at a stride.
            for i in (0..tile_count).step_by(3) {
                let j = (i * 7 + 1) % tile_count;
                v.swap(i, j);
            }
            v
        };
        for order in [&forward, &reverse, &shuffled] {
            assert_eq!(tiled_sum_in_order(&values, tile_len, order), whole);
        }
    }

    #[test]
    fn empty_and_single_tile_sums_are_exact() {
        assert_eq!(TiledSum::new().finish(), 0.0);
        let mut one = TiledSum::new();
        one.push_run(0, &[1.0, 2.0, 3.0]);
        assert_eq!(one.finish(), 6.0);
    }

    #[test]
    fn scattered_position_pushes_reorder_to_canonical() {
        // Pushing individual values out of position order must still reduce in
        // position order, matching the whole-image pairwise sum.
        let values = spread_values(64);
        let whole = pairwise_sum(&values);
        let mut sum = TiledSum::new();
        // Push odd positions first, then even — a worst-case scramble.
        for (position, &value) in values.iter().enumerate() {
            if position % 2 == 1 {
                sum.push(position as u64, value);
            }
        }
        for (position, &value) in values.iter().enumerate() {
            if position % 2 == 0 {
                sum.push(position as u64, value);
            }
        }
        assert_eq!(sum.finish(), whole);
    }

    #[test]
    fn extremum_merge_is_order_free() {
        let mut a = Extremum::new();
        for v in [3.0, -1.0, 7.0] {
            a.observe(v);
        }
        let mut b = Extremum::new();
        for v in [2.0, 9.0, -4.0] {
            b.observe(v);
        }
        let ab = a.merge(b);
        let ba = b.merge(a);
        assert_eq!(ab, ba);
        assert_eq!(ab.min(), Some(-4.0));
        assert_eq!(ab.max(), Some(9.0));
    }

    #[test]
    fn argmax_breaks_ties_toward_lowest_position() {
        // Two equal metrics at positions 5 and 9: the lower position must win, no
        // matter which tile (order) is merged first.
        let mut left = ArgMax::new();
        left.observe(1.0, 9);
        let mut right = ArgMax::new();
        right.observe(1.0, 5);
        assert_eq!(left.merge(right).winner(), Some((1.0, 5)));
        assert_eq!(right.merge(left).winner(), Some((1.0, 5)));

        // A strictly larger metric always wins regardless of position.
        let mut hi = ArgMax::new();
        hi.observe(2.0, 100);
        assert_eq!(hi.merge(right).winner(), Some((2.0, 100)));
    }
}
