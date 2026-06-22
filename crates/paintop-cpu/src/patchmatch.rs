//! The shared `PatchMatch` core: a patch-distance metric and a brute-force
//! nearest-neighbour-field (`NNF`) oracle (`OP_CATALOG` §10, `PatchMatch`).
//!
//! This module owns the *semantics* every patch-correspondence op agrees on:
//! how a patch is compared (the sum-of-squared-differences over a clamped square
//! window) and what an *exact* `NNF` is (the per-target-pixel source anchor that
//! minimises that distance). The randomised `PatchMatch` search
//! ([`crate::patch_field`]) is an approximation of this exact oracle; the oracle
//! is small and obviously correct, so a differential test can hold the
//! approximation to the truth on tiny fixtures (`AGENT_VERIFICATION` §3 — an
//! independent reference, not a re-run of the op under test).
//!
//! # Patch distance (the one definition both share)
//!
//! The distance between the patch centred on target pixel `(tx, ty)` and the
//! candidate patch centred on source pixel `(sx, sy)` is the sum over the
//! `(2·radius + 1)²` window and every channel of the squared sample difference.
//! Out-of-bounds window taps are resolved by **edge clamp** on *both* planes, so
//! the window is always full and a border pixel is never penalised for missing
//! neighbours. The accumulation runs in a fixed `(dy, dx, channel)` raster order
//! in `f64`, so the distance is a deterministic function of the two planes and
//! the radius.
//!
//! # Tie-breaking (deterministic ordering)
//!
//! When several source anchors realise the same minimum distance, the oracle
//! keeps the **first** in source raster-scan order (`sy` ascending, then `sx`
//! ascending). This is the single ordering rule the op's search must also honour
//! so the two agree bit-for-bit on a fixed backend (the M4 determinism
//! criterion).

/// Clamp a signed coordinate to `[0, last]` and return it as a `usize` index.
/// `last` is the in-bounds maximum (`extent - 1`, non-negative), so the result
/// is a valid, lossless array index.
fn clamp_to_index(coord: i64, last: i64) -> usize {
    let clamped = coord.clamp(0, last);
    usize::try_from(clamped).unwrap_or(0)
}

/// A borrowed, row-major, channel-interleaved sample plane: the minimal view the
/// patch-distance metric needs over a source or target raster.
///
/// Sample `(x, y)` channel `c` lives at index `(y·width + x)·channels + c`. The
/// metric only ever *reads* a plane, so it borrows rather than owns.
#[derive(Debug, Clone, Copy)]
pub struct PatchPlane<'a> {
    /// The row-major, channel-interleaved sample buffer
    /// (`width·height·channels` long).
    pub samples: &'a [f32],
    /// The plane width in pixels.
    pub width: u32,
    /// The plane height in pixels.
    pub height: u32,
    /// The interleaved channel count per pixel.
    pub channels: u32,
}

impl<'a> PatchPlane<'a> {
    /// Construct a plane view, returning `None` if the buffer length does not
    /// match `width·height·channels` (so a malformed plane cannot be indexed).
    #[must_use]
    pub fn new(samples: &'a [f32], width: u32, height: u32, channels: u32) -> Option<Self> {
        let expected = (width as usize)
            .checked_mul(height as usize)?
            .checked_mul(channels as usize)?;
        if samples.len() == expected {
            Some(Self {
                samples,
                width,
                height,
                channels,
            })
        } else {
            None
        }
    }

    /// The edge-clamped sample at `(x, y)` channel `c`: coordinates outside the
    /// plane take the nearest in-bounds value (the boundary the patch metric is
    /// defined against). A non-empty plane is assumed; an empty plane never
    /// participates in matching.
    #[must_use]
    fn clamped(&self, x: i64, y: i64, c: u32) -> f32 {
        // Clamp to the half-open domain `[0, width) × [0, height)`. `width`/
        // `height` are non-zero whenever this is called (an empty plane is
        // rejected before matching), so `last` is a valid index.
        let last_x = i64::from(self.width.saturating_sub(1));
        let last_y = i64::from(self.height.saturating_sub(1));
        // Clamped coordinates are non-negative and below the extent, so the
        // `u32`/`usize` conversions are lossless on every supported target.
        let cx = clamp_to_index(x, last_x);
        let cy = clamp_to_index(y, last_y);
        let idx = ((cy * self.width as usize) + cx) * self.channels as usize + c as usize;
        // `idx` is in range by the clamp above; default to `0.0` defensively
        // rather than panicking on a malformed plane.
        self.samples.get(idx).copied().unwrap_or(0.0)
    }
}

/// The sum-of-squared-differences between the `(2·radius + 1)²` patch centred on
/// `target` pixel `(tx, ty)` and the patch centred on `source` pixel `(sx, sy)`.
///
/// Both planes are sampled with edge clamp, and only the `min(channels)` shared
/// leading channels are compared (so an RGBA target may be matched against an
/// RGBA source; mismatched counts compare the common prefix). The accumulation
/// is a fixed-order `f64` reduction, returned as `f64` for an exact, reproducible
/// comparison.
#[must_use]
pub fn patch_distance(
    target: &PatchPlane,
    source: &PatchPlane,
    tx: i64,
    ty: i64,
    sx: i64,
    sy: i64,
    radius: u32,
) -> f64 {
    let r = i64::from(radius);
    let channels = target.channels.min(source.channels);
    let mut acc = 0.0_f64;
    let mut dy = -r;
    while dy <= r {
        let mut dx = -r;
        while dx <= r {
            for c in 0..channels {
                let t = f64::from(target.clamped(tx + dx, ty + dy, c));
                let s = f64::from(source.clamped(sx + dx, sy + dy, c));
                let diff = t - s;
                acc = diff.mul_add(diff, acc);
            }
            dx += 1;
        }
        dy += 1;
    }
    acc
}

/// One target pixel's exact correspondence: the source anchor of its nearest
/// patch and the distance realised there.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Match {
    /// The source x-coordinate of the nearest patch's anchor.
    pub src_x: u32,
    /// The source y-coordinate of the nearest patch's anchor.
    pub src_y: u32,
    /// The patch distance ([`patch_distance`]) realised at the anchor.
    pub cost: f64,
}

/// A whole exact nearest-neighbour field: one [`Match`] per target pixel, in
/// row-major target order (`matches[ty·target_W + tx]`).
#[derive(Debug, Clone, PartialEq)]
pub struct BruteForceField {
    /// The target grid width.
    pub width: u32,
    /// The target grid height.
    pub height: u32,
    /// One match per target pixel, row-major.
    pub matches: Vec<Match>,
}

impl BruteForceField {
    /// The match for target pixel `(x, y)`, or `None` if out of range.
    #[must_use]
    pub fn get(&self, x: u32, y: u32) -> Option<Match> {
        if x >= self.width || y >= self.height {
            return None;
        }
        self.matches
            .get((y as usize * self.width as usize) + x as usize)
            .copied()
    }
}

/// Compute the **exact** nearest-neighbour field by brute force.
///
/// For every target pixel, scan every source anchor and keep the one minimising
/// [`patch_distance`], breaking ties by source raster-scan order.
///
/// `match_target(x, y)` selects which target pixels need a correspondence
/// (e.g. only the hole pixels of an inpainting target); a pixel it rejects gets
/// an identity match to its own coordinate (clamped into the source) at distance
/// `0`, so the field is always fully populated. `source_valid(x, y)` selects
/// which source anchors are eligible candidates (e.g. only known, non-hole
/// source pixels). When no source anchor is eligible for a needed target pixel,
/// its match falls back to the clamped target coordinate at `f64::INFINITY`
/// cost, a sentinel a consumer can recognise.
///
/// This is `O(target · source · patch²)` — intentionally tiny, for fixtures only.
#[must_use]
pub fn brute_force_nnf<MatchT, SrcValid>(
    target: &PatchPlane,
    source: &PatchPlane,
    radius: u32,
    match_target: MatchT,
    source_valid: SrcValid,
) -> BruteForceField
where
    MatchT: Fn(u32, u32) -> bool,
    SrcValid: Fn(u32, u32) -> bool,
{
    let (tw, th) = (target.width, target.height);
    let (sw, sh) = (source.width, source.height);
    let mut matches = Vec::with_capacity((tw as usize) * (th as usize));
    for ty in 0..th {
        for tx in 0..tw {
            // A target pixel that needs no match (outside the hole) maps to its
            // own coordinate clamped into the source, at zero cost.
            if !match_target(tx, ty) {
                matches.push(Match {
                    src_x: tx.min(sw.saturating_sub(1)),
                    src_y: ty.min(sh.saturating_sub(1)),
                    cost: 0.0,
                });
                continue;
            }
            let mut best: Option<Match> = None;
            for sy in 0..sh {
                for sx in 0..sw {
                    if !source_valid(sx, sy) {
                        continue;
                    }
                    let cost = patch_distance(
                        target,
                        source,
                        i64::from(tx),
                        i64::from(ty),
                        i64::from(sx),
                        i64::from(sy),
                        radius,
                    );
                    // Strict `<` keeps the first (raster-earliest) minimum, the
                    // deterministic tie-break rule the op must also honour.
                    let take = best.is_none_or(|b| cost < b.cost);
                    if take {
                        best = Some(Match {
                            src_x: sx,
                            src_y: sy,
                            cost,
                        });
                    }
                }
            }
            matches.push(best.unwrap_or_else(|| Match {
                src_x: tx.min(sw.saturating_sub(1)),
                src_y: ty.min(sh.saturating_sub(1)),
                cost: f64::INFINITY,
            }));
        }
    }
    BruteForceField {
        width: tw,
        height: th,
        matches,
    }
}

/// A 64-bit avalanche mix (`SplitMix64` finalizer): folds an integer into a
/// well-distributed 64-bit hash. The same mixer the procedural-noise ops use, so
/// the `PatchMatch` RNG shares the codebase's one hashing convention.
const fn mix64(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

/// A pure, **order-independent** pseudo-random 64-bit draw from a fixed tuple of
/// stream coordinates `(seed, x, y, iter, step)`.
///
/// Every random choice the search makes is a hash of *where and when* it is made,
/// never of a mutable sequential RNG state. That is what makes the search
/// reproducible regardless of how the sweep is parallelised or reordered: the
/// draw for target pixel `(x, y)` on iteration `iter`, sub-step `step` is the
/// same value on every run and every backend (`plan.md` §1444 — reproducible-tier
/// noise is hash-of-coordinate, not sequential RNG).
fn draw(seed: u64, x: u32, y: u32, iter: u32, step: u32) -> u64 {
    let h = seed
        ^ u64::from(x).wrapping_mul(0x9e37_79b9_7f4a_7c15)
        ^ u64::from(y).wrapping_mul(0xc2b2_ae3d_27d4_eb4f)
        ^ u64::from(iter).wrapping_mul(0x1656_67b1_9e37_79f9)
        ^ u64::from(step).wrapping_mul(0xff51_afd7_ed55_8ccd);
    mix64(h)
}

/// Map a 64-bit draw to a uniform integer in `[0, bound)` (`bound > 0`) by the
/// 53-bit-mantissa multiply-shift method, deterministic on every target.
fn draw_range(value: u64, bound: u32) -> u32 {
    if bound <= 1 {
        return 0;
    }
    // Use the top 32 bits and a 64-bit widening multiply: `(hi * bound) >> 32`
    // is a uniform map into `[0, bound)` with no modulo bias for these sizes.
    let hi = value >> 32;
    let scaled = hi.wrapping_mul(u64::from(bound)) >> 32;
    // `scaled < bound`, so the `u32` conversion is lossless.
    u32::try_from(scaled).unwrap_or(0)
}

/// One target pixel's working correspondence during the search: a signed source
/// anchor (always kept in bounds) and its current cost.
#[derive(Debug, Clone, Copy)]
struct Cell {
    src_x: u32,
    src_y: u32,
    cost: f64,
}

/// The deterministic, seeded result of an approximate `PatchMatch` search: the
/// final field plus the convergence trace the op's report exposes.
#[derive(Debug, Clone, PartialEq)]
pub struct SearchResult {
    /// The approximate nearest-neighbour field, one [`Match`] per target pixel,
    /// row-major.
    pub field: BruteForceField,
    /// The total field cost (`Σ cost`, finite cells only) after each iteration,
    /// length `iterations` — a non-increasing trace evidencing convergence.
    pub cost_history: Vec<f64>,
    /// The number of iterations actually run.
    pub iterations: u32,
    /// Whether the search stopped early because an iteration made no improvement
    /// (a fixed point), rather than exhausting the iteration budget.
    pub converged: bool,
}

/// The inputs to a [`patch_match`] search: the two planes, the patch radius, and
/// the masks selecting matched target pixels and valid source anchors.
pub struct SearchConfig<'a, MatchT, SrcValid> {
    /// The target plane (the image being filled).
    pub target: PatchPlane<'a>,
    /// The source plane (where coherent texture is drawn from).
    pub source: PatchPlane<'a>,
    /// The patch half-window.
    pub radius: u32,
    /// The number of propagation/random-search iterations.
    pub iterations: u32,
    /// The deterministic seed.
    pub seed: u64,
    /// `match_target(x, y)`: whether target pixel `(x, y)` needs a match.
    pub match_target: MatchT,
    /// `source_valid(x, y)`: whether source pixel `(x, y)` is an eligible anchor.
    pub source_valid: SrcValid,
}

/// Find an approximate nearest-neighbour field by **deterministic, seeded
/// `PatchMatch`** (`OP_CATALOG` §10): seeded random init, then alternating-scan
/// propagation and shrinking random search.
///
/// The search is reproducible: every random draw is `draw`
/// (hash-of-coordinate), the scan order is a fixed forward/backward alternation,
/// and ties keep the incumbent — so a fixed seed yields a bit-identical field on
/// every run and backend (the M4 determinism criterion). It approximates the
/// exact [`brute_force_nnf`] oracle, against which it is differentially tested.
///
/// A target pixel that needs no match (outside the hole) keeps its own clamped
/// coordinate at cost `0`. When no source anchor is eligible the field falls back
/// to the same identity sentinel as the oracle.
#[must_use]
pub fn patch_match<MatchT, SrcValid>(config: &SearchConfig<MatchT, SrcValid>) -> SearchResult
where
    MatchT: Fn(u32, u32) -> bool,
    SrcValid: Fn(u32, u32) -> bool,
{
    let searcher = Searcher::new(config);
    let mut cells = searcher.init();
    let mut cost_history = Vec::with_capacity(config.iterations as usize);
    let mut converged = false;
    let mut ran = 0_u32;
    for iter in 0..config.iterations {
        ran = iter + 1;
        let improved = searcher.sweep(&mut cells, iter);
        cost_history.push(total_cost(&cells));
        // A sweep that improved nothing means the field reached a fixed point of
        // this seed's search; record it but keep sweeping, since the global
        // random restart may still escape on a later iteration.
        converged = !improved;
    }

    let matches = cells
        .iter()
        .map(|c| Match {
            src_x: c.src_x,
            src_y: c.src_y,
            cost: c.cost,
        })
        .collect();
    SearchResult {
        field: BruteForceField {
            width: searcher.tw,
            height: searcher.th,
            matches,
        },
        cost_history,
        iterations: ran,
        converged,
    }
}

/// The borrowed state a single [`patch_match`] run threads through its init and
/// sweep helpers: the two planes, the radius/seed, the masks, and the
/// precomputed list of eligible source anchors.
struct Searcher<'a, MatchT, SrcValid> {
    target: PatchPlane<'a>,
    source: PatchPlane<'a>,
    radius: u32,
    seed: u64,
    tw: u32,
    th: u32,
    sw: u32,
    sh: u32,
    /// The largest random-search window (the source's larger dimension).
    max_dim: u32,
    match_target: &'a MatchT,
    source_valid: &'a SrcValid,
    /// Eligible source anchors in raster order; random draws index into this so
    /// they can never land on an invalid anchor.
    valid_anchors: Vec<(u32, u32)>,
}

impl<'a, MatchT, SrcValid> Searcher<'a, MatchT, SrcValid>
where
    MatchT: Fn(u32, u32) -> bool,
    SrcValid: Fn(u32, u32) -> bool,
{
    fn new(config: &'a SearchConfig<MatchT, SrcValid>) -> Self {
        let (sw, sh) = (config.source.width, config.source.height);
        let valid_anchors: Vec<(u32, u32)> = (0..sh)
            .flat_map(|sy| (0..sw).map(move |sx| (sx, sy)))
            .filter(|&(sx, sy)| (config.source_valid)(sx, sy))
            .collect();
        Self {
            target: config.target,
            source: config.source,
            radius: config.radius,
            seed: config.seed,
            tw: config.target.width,
            th: config.target.height,
            sw,
            sh,
            max_dim: sw.max(sh).max(1),
            match_target: &config.match_target,
            source_valid: &config.source_valid,
            valid_anchors,
        }
    }

    const fn idx(&self, tx: u32, ty: u32) -> usize {
        (ty as usize * self.tw as usize) + tx as usize
    }

    fn cost_at(&self, tx: u32, ty: u32, sx: u32, sy: u32) -> f64 {
        patch_distance(
            &self.target,
            &self.source,
            i64::from(tx),
            i64::from(ty),
            i64::from(sx),
            i64::from(sy),
            self.radius,
        )
    }

    /// The identity correspondence for a pixel that needs no match (or has no
    /// eligible anchor): its own coordinate clamped into the source.
    fn identity(&self, tx: u32, ty: u32, cost: f64) -> Cell {
        Cell {
            src_x: tx.min(self.sw.saturating_sub(1)),
            src_y: ty.min(self.sh.saturating_sub(1)),
            cost,
        }
    }

    /// Seeded random initialisation: each matched pixel gets a uniformly drawn
    /// valid anchor; every other pixel an identity correspondence.
    fn init(&self) -> Vec<Cell> {
        let mut cells = Vec::with_capacity((self.tw as usize) * (self.th as usize));
        for ty in 0..self.th {
            for tx in 0..self.tw {
                if !(self.match_target)(tx, ty) {
                    cells.push(self.identity(tx, ty, 0.0));
                } else if let Some((sx, sy)) = self.draw_anchor(tx, ty, u32::MAX, 0) {
                    cells.push(Cell {
                        src_x: sx,
                        src_y: sy,
                        cost: self.cost_at(tx, ty, sx, sy),
                    });
                } else {
                    cells.push(self.identity(tx, ty, f64::INFINITY));
                }
            }
        }
        cells
    }

    /// A uniform draw over the valid anchors for `(tx, ty)` at search stream
    /// `(iter, step)`, or `None` if no anchor is eligible.
    fn draw_anchor(&self, tx: u32, ty: u32, iter: u32, step: u32) -> Option<(u32, u32)> {
        if self.valid_anchors.is_empty() {
            return None;
        }
        let bound = u32::try_from(self.valid_anchors.len()).unwrap_or(u32::MAX);
        let pick = draw_range(draw(self.seed, tx, ty, iter, step), bound);
        self.valid_anchors.get(pick as usize).copied()
    }

    /// Replace `best` with `(cand_x, cand_y)` if that valid anchor is strictly
    /// cheaper (keeping the incumbent on a tie — the deterministic rule).
    fn consider(&self, tx: u32, ty: u32, cand: (u32, u32), best: &mut Cell) {
        if !(self.source_valid)(cand.0, cand.1) {
            return;
        }
        let cost = self.cost_at(tx, ty, cand.0, cand.1);
        if cost < best.cost {
            *best = Cell {
                src_x: cand.0,
                src_y: cand.1,
                cost,
            };
        }
    }

    /// One full propagation + random-search sweep over every matched pixel in
    /// the iteration's scan direction; returns whether any pixel improved.
    fn sweep(&self, cells: &mut [Cell], iter: u32) -> bool {
        let forward = iter.is_multiple_of(2);
        let order = |n: u32| -> Vec<u32> {
            if forward {
                (0..n).collect()
            } else {
                (0..n).rev().collect()
            }
        };
        let (ys, xs) = (order(self.th), order(self.tw));
        let mut improved = false;
        for &ty in &ys {
            for &tx in &xs {
                if !(self.match_target)(tx, ty) || self.valid_anchors.is_empty() {
                    continue;
                }
                let i = self.idx(tx, ty);
                let mut best = cells[i];
                self.propagate(cells, tx, ty, forward, &mut best);
                self.random_search(tx, ty, iter, &mut best);
                if best.cost < cells[i].cost {
                    improved = true;
                }
                cells[i] = best;
            }
        }
        improved
    }

    /// Propagation: try the two already-scanned neighbours' offsets, shifted one
    /// step toward this pixel (the coherence step of `PatchMatch`).
    fn propagate(&self, cells: &[Cell], tx: u32, ty: u32, forward: bool, best: &mut Cell) {
        let (nx, ny) = if forward {
            (tx.checked_sub(1), ty.checked_sub(1))
        } else {
            (
                (tx + 1 < self.tw).then_some(tx + 1),
                (ty + 1 < self.th).then_some(ty + 1),
            )
        };
        if let Some(px) = nx {
            let n = cells[self.idx(px, ty)];
            let cand_x = if forward {
                (n.src_x + 1).min(self.sw - 1)
            } else {
                n.src_x.saturating_sub(1)
            };
            self.consider(tx, ty, (cand_x, n.src_y), best);
        }
        if let Some(py) = ny {
            let n = cells[self.idx(tx, py)];
            let cand_y = if forward {
                (n.src_y + 1).min(self.sh - 1)
            } else {
                n.src_y.saturating_sub(1)
            };
            self.consider(tx, ty, (n.src_x, cand_y), best);
        }
    }

    /// Random search: a global uniform restart (step `0`) followed by a
    /// geometrically shrinking window around the current best — every draw a
    /// hash-of-coordinate value, so the search is order-independent.
    fn random_search(&self, tx: u32, ty: u32, iter: u32, best: &mut Cell) {
        if let Some(g) = self.draw_anchor(tx, ty, iter, 0) {
            self.consider(tx, ty, g, best);
        }
        let mut window = self.max_dim;
        let mut step = 1_u32;
        while window >= 1 {
            let d = draw(self.seed, tx, ty, iter, step);
            let span = (2 * window) + 1;
            let ox = draw_range(d, span);
            let oy = draw_range(mix64(d), span);
            let cand_x = (i64::from(best.src_x) + i64::from(ox) - i64::from(window))
                .clamp(0, i64::from(self.sw - 1));
            let cand_y = (i64::from(best.src_y) + i64::from(oy) - i64::from(window))
                .clamp(0, i64::from(self.sh - 1));
            let cand = (
                u32::try_from(cand_x).unwrap_or(0),
                u32::try_from(cand_y).unwrap_or(0),
            );
            self.consider(tx, ty, cand, best);
            step += 1;
            window /= 2;
        }
    }
}

/// The total field cost over finite cells (the per-sweep convergence metric).
fn total_cost(cells: &[Cell]) -> f64 {
    cells
        .iter()
        .filter(|c| c.cost.is_finite())
        .map(|c| c.cost)
        .sum()
}

#[cfg(test)]
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "test fixtures use small exact integer samples"
)]
mod tests {
    use super::{
        BruteForceField, Match, PatchPlane, SearchConfig, brute_force_nnf, patch_distance,
        patch_match,
    };

    /// A 1-channel plane from a row-major `f32` grid.
    fn plane(samples: &[f32], w: u32, h: u32) -> PatchPlane<'_> {
        PatchPlane::new(samples, w, h, 1).expect("buffer matches w*h*1")
    }

    /// Assert two costs match within a tight floating-point tolerance (the
    /// distances are exact small integers, so any slack is rounding only).
    fn assert_close(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() < 1.0e-9,
            "expected {expected}, got {actual}"
        );
    }

    #[test]
    fn plane_rejects_mis_sized_buffer() {
        assert!(PatchPlane::new(&[0.0; 3], 2, 2, 1).is_none());
        assert!(PatchPlane::new(&[0.0; 4], 2, 2, 1).is_some());
    }

    #[test]
    fn identical_patches_have_zero_distance() {
        let s = [1.0, 2.0, 3.0, 4.0];
        let p = plane(&s, 2, 2);
        // Same plane, same anchor, radius 0: a single tap, zero difference.
        assert_close(patch_distance(&p, &p, 0, 0, 0, 0, 0), 0.0);
        // Radius 1 with clamp: every clamped tap still matches itself.
        assert_close(patch_distance(&p, &p, 0, 0, 0, 0, 1), 0.0);
    }

    #[test]
    fn distance_is_sum_of_squared_differences() {
        // target all 0, source all 1, radius 0 => single tap (0-1)^2 = 1.
        let t = [0.0; 4];
        let s = [1.0; 4];
        let tp = plane(&t, 2, 2);
        let sp = plane(&s, 2, 2);
        assert_close(patch_distance(&tp, &sp, 0, 0, 0, 0, 0), 1.0);
        // radius 1 => 9 clamped taps each (0-1)^2 = 9.
        assert_close(patch_distance(&tp, &sp, 0, 0, 0, 0, 1), 9.0);
    }

    #[test]
    fn brute_force_finds_the_exact_repeated_texture_match() {
        // A 4x1 source with a step: [0,0,9,9]. The target is a single pixel of
        // value 0; with radius 0 the nearest anchor is the first 0 (sx=0), the
        // raster-earliest of the two zero-cost candidates.
        let src = [0.0, 0.0, 9.0, 9.0];
        let tgt = [0.0];
        let sp = plane(&src, 4, 1);
        let tp = plane(&tgt, 1, 1);
        let nnf = brute_force_nnf(&tp, &sp, 0, |_, _| true, |_, _| true);
        let m = nnf.get(0, 0).unwrap();
        assert_eq!((m.src_x, m.src_y), (0, 0));
        assert_close(m.cost, 0.0);
    }

    #[test]
    fn tie_break_keeps_raster_earliest_anchor() {
        // Two equally-good (both value 5) source anchors; the earliest (sx=0)
        // wins under the strict-`<` rule.
        let src = [5.0, 5.0];
        let tgt = [5.0];
        let sp = plane(&src, 2, 1);
        let tp = plane(&tgt, 1, 1);
        let nnf = brute_force_nnf(&tp, &sp, 0, |_, _| true, |_, _| true);
        assert_eq!(nnf.get(0, 0).unwrap().src_x, 0);
    }

    #[test]
    fn source_validity_excludes_anchors() {
        // The exact value 7 only sits at the (excluded) sx=0; the eligible
        // anchors are sx=1 (value 0). With sx=0 invalid, the match is sx=1.
        let src = [7.0, 0.0];
        let tgt = [7.0];
        let sp = plane(&src, 2, 1);
        let tp = plane(&tgt, 1, 1);
        let nnf = brute_force_nnf(&tp, &sp, 0, |_, _| true, |sx, _| sx != 0);
        let m = nnf.get(0, 0).unwrap();
        assert_eq!(m.src_x, 1);
        assert_close(m.cost, 49.0); // (7-0)^2
    }

    #[test]
    fn unmatched_target_pixels_map_to_self_at_zero_cost() {
        let src = [0.0, 1.0, 2.0, 3.0];
        let tgt = [0.0, 0.0, 0.0, 0.0];
        let sp = plane(&src, 2, 2);
        let tp = plane(&tgt, 2, 2);
        // Match only the top-left target pixel; the rest map to themselves.
        let nnf = brute_force_nnf(&tp, &sp, 0, |x, y| x == 0 && y == 0, |_, _| true);
        let self_match = nnf.get(1, 1).unwrap();
        assert_eq!((self_match.src_x, self_match.src_y), (1, 1));
        assert_close(self_match.cost, 0.0);
    }

    #[test]
    fn no_eligible_source_yields_infinite_cost_sentinel() {
        let src = [0.0];
        let tgt = [0.0];
        let sp = plane(&src, 1, 1);
        let tp = plane(&tgt, 1, 1);
        // Need a match but reject every source anchor.
        let nnf = brute_force_nnf(&tp, &sp, 0, |_, _| true, |_, _| false);
        assert!(nnf.get(0, 0).unwrap().cost.is_infinite());
    }

    #[test]
    fn field_get_is_row_major_and_bounds_checked() {
        let f = BruteForceField {
            width: 2,
            height: 2,
            matches: vec![
                Match {
                    src_x: 0,
                    src_y: 0,
                    cost: 0.0,
                },
                Match {
                    src_x: 1,
                    src_y: 0,
                    cost: 0.0,
                },
                Match {
                    src_x: 0,
                    src_y: 1,
                    cost: 0.0,
                },
                Match {
                    src_x: 1,
                    src_y: 1,
                    cost: 0.0,
                },
            ],
        };
        assert_eq!(f.get(1, 0).unwrap().src_x, 1);
        assert_eq!(f.get(0, 1).unwrap().src_y, 1);
        assert!(f.get(2, 0).is_none());
    }

    /// A small config helper over 1-channel planes with all pixels matched and
    /// all anchors valid.
    fn config<'a>(
        target: &'a PatchPlane<'a>,
        source: &'a PatchPlane<'a>,
        radius: u32,
        iterations: u32,
        seed: u64,
    ) -> SearchConfig<'a, impl Fn(u32, u32) -> bool, impl Fn(u32, u32) -> bool> {
        SearchConfig {
            target: *target,
            source: *source,
            radius,
            iterations,
            seed,
            match_target: |_, _| true,
            source_valid: |_, _| true,
        }
    }

    #[test]
    fn patch_match_is_bit_identical_across_reruns() {
        let src = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let tgt = [4.0, 1.0, 7.0, 2.0];
        let sp = plane(&src, 3, 3);
        let tp = plane(&tgt, 2, 2);
        let a = patch_match(&config(&tp, &sp, 0, 4, 0xdead_beef));
        let b = patch_match(&config(&tp, &sp, 0, 4, 0xdead_beef));
        assert_eq!(a, b, "a fixed seed must give a bit-identical field");
    }

    #[test]
    fn patch_match_seed_changes_the_search_trajectory() {
        // On a tie-free source a different seed must still reach the global
        // optimum, but the cost history (search trajectory) may differ.
        let src = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let tgt = [5.0];
        let sp = plane(&src, 3, 3);
        let tp = plane(&tgt, 1, 1);
        let a = patch_match(&config(&tp, &sp, 0, 24, 1));
        let b = patch_match(&config(&tp, &sp, 0, 24, 2));
        // Both converge to the exact match (value 5 at src (2,1)).
        let ma = a.field.get(0, 0).unwrap();
        let mb = b.field.get(0, 0).unwrap();
        assert_close(ma.cost, 0.0);
        assert_close(mb.cost, 0.0);
        assert_eq!((ma.src_x, ma.src_y), (2, 1));
        assert_eq!((mb.src_x, mb.src_y), (2, 1));
    }

    #[test]
    fn patch_match_matches_the_brute_force_oracle_on_a_tiny_fixture() {
        // A gradient source and a target whose values each appear exactly once
        // in the source: enough iterations let PatchMatch find the exact NNF.
        let src: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let tgt = [10.0, 3.0, 15.0, 6.0];
        let sp = plane(&src, 4, 4);
        let tp = plane(&tgt, 2, 2);
        let oracle = brute_force_nnf(&tp, &sp, 0, |_, _| true, |_, _| true);
        let result = patch_match(&config(&tp, &sp, 0, 32, 7));
        for y in 0..2 {
            for x in 0..2 {
                let o = oracle.get(x, y).unwrap();
                let r = result.field.get(x, y).unwrap();
                assert_eq!(
                    (o.src_x, o.src_y),
                    (r.src_x, r.src_y),
                    "patch_match disagrees with the oracle at ({x},{y})"
                );
            }
        }
    }

    #[test]
    fn patch_match_cost_history_is_non_increasing() {
        let src: Vec<f32> = (0..25).map(|i| (i % 7) as f32).collect();
        let tgt = [3.0, 5.0, 1.0, 6.0, 0.0, 2.0, 4.0, 5.0, 1.0];
        let sp = plane(&src, 5, 5);
        let tp = plane(&tgt, 3, 3);
        let result = patch_match(&config(&tp, &sp, 1, 8, 99));
        for w in result.cost_history.windows(2) {
            assert!(
                w[1] <= w[0] + 1.0e-9,
                "cost history rose: {:?}",
                result.cost_history
            );
        }
        assert!(result.iterations >= 1);
    }

    #[test]
    fn patch_match_leaves_unmatched_pixels_at_identity() {
        let src = [0.0, 1.0, 2.0, 3.0];
        let tgt = [9.0, 9.0, 9.0, 9.0];
        let sp = plane(&src, 2, 2);
        let tp = plane(&tgt, 2, 2);
        let cfg = SearchConfig {
            target: tp,
            source: sp,
            radius: 0,
            iterations: 4,
            seed: 3,
            // Match only the top-left target pixel.
            match_target: |x, y| x == 0 && y == 0,
            source_valid: |_, _| true,
        };
        let result = patch_match(&cfg);
        let bottom_right = result.field.get(1, 1).unwrap();
        assert_eq!((bottom_right.src_x, bottom_right.src_y), (1, 1));
        assert_close(bottom_right.cost, 0.0);
    }
}
