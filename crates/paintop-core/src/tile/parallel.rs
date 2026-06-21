//! Scheduler-owned tile parallelism over the M2 tile executor (`plan.md` §12.2).
//!
//! M2 lays out a deterministic, demand-driven schedule of `(node, output tile)`
//! work items and runs them whole-image-equivalent, tile by tile, on one thread.
//! M3 keeps that schedule and *runs independent tiles of a node concurrently* on a
//! bounded [Rayon](rayon) pool the **scheduler owns** — no op spawns its own
//! parallelism (`plan.md` §12.2: "scheduler-owned, no nested uncontrolled
//! parallelism"). The thread width is capped by a [`ThreadCap`] taken from policy.
//!
//! # Why the result is bit-identical across thread counts
//!
//! Parallelism is introduced **only** in the per-tile *compute* step, and only
//! across the tiles of one node, which the M2 model already guarantees are
//! independent:
//!
//! * a **pointwise** tile is a pure function of its co-located input crop, and the
//!   tiles partition the output into disjoint regions, so computing them in any
//!   order and on any thread yields the identical per-tile bytes;
//! * a **neighbourhood** tile is a pure function of its haloed input window (also
//!   read-only), with the same disjoint-output-region property;
//! * the **scatter** of each computed tile into the node's full-extent buffer
//!   writes a disjoint region, and we perform every scatter back on the calling
//!   thread in the fixed schedule order, so the final buffer is independent of
//!   which worker produced which tile;
//! * **reductions** (`analyze.statistics`, the assertion scans) are not folded
//!   here at all — they go through the M2 fixed merge trees
//!   ([`crate::tile::reduce`]), which key every contribution by its absolute
//!   row-major position and merge in that canonical order, so they too are
//!   independent of the thread count.
//!
//! Because the only thing the pool changes is *which thread evaluates a pure
//! function over read-only inputs*, the output is a pure function of the schedule —
//! bit-identical to the M2 single-threaded tiled result for thread counts
//! `{1, 2, 8, …}` alike, the property the parallel-determinism suite pins.

/// The cap on the width of the scheduler-owned tile pool (`plan.md` §12.2).
///
/// The scheduler — never an op — decides how many worker threads tile work may
/// fan out across. A cap of `1` runs the schedule sequentially (identical to the
/// M2 path); a larger cap bounds the Rayon pool so a deep graph cannot oversubscribe
/// the machine, and so no nested op-level parallelism can multiply out of control.
///
/// The default ([`ThreadCap::auto`]) defers to the available parallelism of the
/// host; an explicit [`ThreadCap::fixed`] pins a width, which the determinism suite
/// uses to prove the output is identical at `{1, 2, 8}` threads.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ThreadCap {
    /// The maximum worker count, or `None` to defer to host parallelism.
    max_threads: Option<usize>,
}

impl Default for ThreadCap {
    fn default() -> Self {
        Self::auto()
    }
}

impl ThreadCap {
    /// A cap that defers to the host's available parallelism (the default).
    #[must_use]
    pub const fn auto() -> Self {
        Self { max_threads: None }
    }

    /// A cap pinned to exactly `threads` workers (clamped to at least one).
    ///
    /// `fixed(1)` runs sequentially — the M2 single-threaded path — which the
    /// determinism suite uses as the bit-identical baseline.
    #[must_use]
    pub const fn fixed(threads: usize) -> Self {
        Self {
            max_threads: Some(if threads == 0 { 1 } else { threads }),
        }
    }

    /// Whether this cap forces single-threaded execution (`fixed(1)`).
    #[must_use]
    pub const fn is_sequential(self) -> bool {
        matches!(self.max_threads, Some(1))
    }

    /// The resolved worker count for the pool: the explicit cap, or the host's
    /// available parallelism when deferring, never zero.
    #[must_use]
    pub fn resolve(self) -> usize {
        self.max_threads.unwrap_or_else(|| {
            std::thread::available_parallelism().map_or(1, std::num::NonZeroUsize::get)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::ThreadCap;

    #[test]
    fn auto_defers_to_host_parallelism() {
        let cap = ThreadCap::auto();
        assert!(!cap.is_sequential() || cap.resolve() == 1);
        assert!(cap.resolve() >= 1);
    }

    #[test]
    fn fixed_clamps_zero_to_one() {
        assert_eq!(ThreadCap::fixed(0).resolve(), 1);
        assert!(ThreadCap::fixed(0).is_sequential());
    }

    #[test]
    fn fixed_pins_the_width() {
        assert_eq!(ThreadCap::fixed(8).resolve(), 8);
        assert!(!ThreadCap::fixed(8).is_sequential());
        assert!(ThreadCap::fixed(1).is_sequential());
    }

    #[test]
    fn default_is_auto() {
        assert_eq!(ThreadCap::default(), ThreadCap::auto());
    }
}
