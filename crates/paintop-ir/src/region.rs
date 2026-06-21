//! Integer region algebra for backward ROI propagation (`IR_SPEC` §18,
//! `plan.md` §10–§11).
//!
//! Backward demand walks the graph from exports/assertions/debug roots toward
//! sources, asking each operation's
//! [`required_inputs`](crate::contract::OpContract::required_inputs) which input
//! pixels a demanded output region needs. A single consumer asks for one
//! [`Rect`]; but a *producer* may be demanded by several consumers at once, so
//! the regions those consumers ask for must be **accumulated** into one demand
//! per output port before the producer's own `required_inputs` is evaluated.
//!
//! This module owns that accumulation. A [`Region`] is the union of zero or more
//! half-open [`Rect`]s, normalized to a single deterministic **bounding box**:
//! paintop's ROI model is conservative (a demanded region is always *at least*
//! the true contributor set), so collapsing a union of rects to their bounding
//! box is sound — it can only demand *more* pixels, never fewer, and keeps the
//! algebra cheap and order-independent (a fixed, associative merge). The bound is
//! the exact contributor set whenever the union is itself a rectangle, which is
//! the common case (tiles of one image, the target region of a masked edit).
//!
//! The region operations the executable contracts and the demand graph use are:
//!
//! - **union** — accumulate another rect / region into the demand
//!   ([`Region::push`], [`Region::union`]);
//! - **intersect** — clip a demand to a producer's valid extent
//!   ([`Region::intersect_rect`]);
//! - **dilate-by-halo** — grow a demand by a neighbourhood operation's halo
//!   ([`Region::dilate`]);
//! - **transform** — translate a demand under a crop/pad/composite offset
//!   ([`Region::translate`]).
//!
//! Each operation handles the four cases the bone calls out: an **empty** region
//! (no demand), a **full** region (the whole producer), a **clipped** region (a
//! demand partly outside the producer), and a **halo-expanded** region (a demand
//! grown past the producer's edge then clipped back).

use serde::{Deserialize, Serialize};

use crate::resource::{Extent, Rect};

/// A conservative integer region: the union of the rects pushed into it,
/// represented by their deterministic bounding box (`IR_SPEC` §18).
///
/// A region is either *empty* (it demands no pixels) or carries one bounding
/// [`Rect`] that contains every rect unioned into it. Because the ROI model is
/// conservative, the bounding box is a sound over-approximation of the true
/// union: it never drops a contributor. All operations are total and never
/// panic; out-of-range arithmetic saturates rather than wrapping.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Region {
    /// The bounding rect of the union, or [`None`] when the region is empty. An
    /// empty region and a region carrying a zero-area rect are normalized to the
    /// same value (`None`) so equality is canonical.
    bounds: Option<Rect>,
}

impl Region {
    /// The empty region: it demands no pixels. The identity element of
    /// [`union`](Region::union).
    pub const EMPTY: Self = Self { bounds: None };

    /// An empty region. Equivalent to [`Region::EMPTY`].
    #[must_use]
    pub const fn new() -> Self {
        Self::EMPTY
    }

    /// A region covering exactly one [`Rect`]. A zero-area (empty) rect produces
    /// the empty region, so a degenerate demand never inflates a bound.
    #[must_use]
    pub const fn from_rect(rect: Rect) -> Self {
        if rect.is_empty() {
            Self::EMPTY
        } else {
            Self { bounds: Some(rect) }
        }
    }

    /// A region covering the full domain of `extent` (origin `(0, 0)`, size
    /// `extent`). The "whole producer" demand a full-domain op emits.
    #[must_use]
    pub const fn from_extent(extent: Extent) -> Self {
        Self::from_rect(Rect::from_extent(extent))
    }

    /// Whether this region demands no pixels.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.bounds.is_none()
    }

    /// The region's conservative bounding [`Rect`], or [`None`] when empty.
    #[must_use]
    pub const fn bounds(&self) -> Option<Rect> {
        self.bounds
    }

    /// The region's bounding rect, or [`Rect::EMPTY`] when the region is empty —
    /// the convenient single-rect view the executable contracts return.
    #[must_use]
    pub fn bounding_rect(&self) -> Rect {
        self.bounds.unwrap_or(Rect::EMPTY)
    }

    /// Accumulate `rect` into this region in place: the region becomes the
    /// bounding box of its previous contents and `rect`. An empty `rect` is a
    /// no-op, so `push`ing demands in any order yields the same region.
    pub fn push(&mut self, rect: Rect) {
        if rect.is_empty() {
            return;
        }
        self.bounds = Some(self.bounds.map_or(rect, |existing| existing.union(rect)));
    }

    /// The union of two regions: the bounding box containing every pixel of
    /// either. Empty is the identity. Commutative and associative, so demand
    /// accumulation is order-independent.
    #[must_use]
    pub fn union(self, other: Self) -> Self {
        match (self.bounds, other.bounds) {
            (None, None) => Self::EMPTY,
            (Some(a), None) => Self { bounds: Some(a) },
            (None, Some(b)) => Self { bounds: Some(b) },
            (Some(a), Some(b)) => Self {
                bounds: Some(a.union(b)),
            },
        }
    }

    /// This region clipped to `rect`: the part of the demand that falls inside
    /// `rect`. An empty result (the demand lies wholly outside `rect`) normalizes
    /// to [`Region::EMPTY`].
    #[must_use]
    pub fn intersect_rect(self, rect: Rect) -> Self {
        self.bounds
            .map_or(Self::EMPTY, |b| Self::from_rect(b.intersect(rect)))
    }

    /// This region clipped to the full domain of `extent`: the canonical "clamp a
    /// demand to the producer's actual pixels" operation.
    #[must_use]
    pub fn clamp_to_extent(self, extent: Extent) -> Self {
        self.intersect_rect(Rect::from_extent(extent))
    }

    /// This region translated by `(dx, dy)` — the geometric transform a crop/pad
    /// offset applies when mapping an output demand back to its input. Empty in,
    /// empty out.
    #[must_use]
    pub fn translate(self, dx: i64, dy: i64) -> Self {
        Self {
            bounds: self.bounds.map(|b| b.translate(dx, dy)),
        }
    }

    /// This region dilated outward by a uniform `halo` on every side — a
    /// neighbourhood op growing an output demand to its input footprint. Empty in,
    /// empty out.
    #[must_use]
    pub fn dilate(self, halo: u32) -> Self {
        Self {
            bounds: self.bounds.map(|b| b.dilate(halo)),
        }
    }

    /// This region dilated by independent horizontal/vertical halos. Empty in,
    /// empty out.
    #[must_use]
    pub fn dilate_xy(self, dx: u32, dy: u32) -> Self {
        Self {
            bounds: self.bounds.map(|b| b.dilate_xy(dx, dy)),
        }
    }

    /// Whether this region contains every pixel of `rect` — the cover check the
    /// ROI differential suite uses to prove a demanded source region includes all
    /// contributors.
    #[must_use]
    pub fn contains_rect(&self, rect: Rect) -> bool {
        self.bounds
            .map_or_else(|| rect.is_empty(), |b| b.contains_rect(rect))
    }
}

impl Default for Region {
    fn default() -> Self {
        Self::EMPTY
    }
}

impl From<Rect> for Region {
    fn from(rect: Rect) -> Self {
        Self::from_rect(rect)
    }
}

impl FromIterator<Rect> for Region {
    fn from_iter<I: IntoIterator<Item = Rect>>(iter: I) -> Self {
        let mut region = Self::EMPTY;
        for rect in iter {
            region.push(rect);
        }
        region
    }
}

#[cfg(test)]
mod tests {
    use super::Region;
    use crate::resource::{Extent, Rect};

    #[test]
    fn empty_region_demands_nothing() {
        let r = Region::EMPTY;
        assert!(r.is_empty());
        assert_eq!(r.bounds(), None);
        assert_eq!(r.bounding_rect(), Rect::EMPTY);
        // A zero-area rect collapses to empty.
        assert!(Region::from_rect(Rect::new(5, 5, 5, 9)).is_empty());
    }

    #[test]
    fn from_extent_is_the_full_domain() {
        let r = Region::from_extent(Extent::new(64, 48));
        assert_eq!(r.bounding_rect(), Rect::new(0, 0, 64, 48));
        assert!(!r.is_empty());
    }

    #[test]
    fn push_accumulates_to_the_bounding_box_order_independently() {
        let a = Rect::new(0, 0, 10, 10);
        let b = Rect::new(20, 5, 30, 25);

        let mut forward = Region::EMPTY;
        forward.push(a);
        forward.push(b);

        let mut reverse = Region::EMPTY;
        reverse.push(b);
        reverse.push(a);

        assert_eq!(forward, reverse);
        assert_eq!(forward.bounding_rect(), Rect::new(0, 0, 30, 25));
        // The conservative bound contains every contributor.
        assert!(forward.contains_rect(a));
        assert!(forward.contains_rect(b));
    }

    #[test]
    fn push_ignores_empty_rects() {
        let mut r = Region::from_rect(Rect::new(2, 2, 8, 8));
        r.push(Rect::EMPTY);
        r.push(Rect::new(4, 4, 4, 9)); // zero width
        assert_eq!(r.bounding_rect(), Rect::new(2, 2, 8, 8));
    }

    #[test]
    fn union_is_commutative_with_empty_identity() {
        let a = Region::from_rect(Rect::new(0, 0, 4, 4));
        let b = Region::from_rect(Rect::new(8, 8, 12, 12));
        assert_eq!(a.union(b), b.union(a));
        assert_eq!(a.union(Region::EMPTY), a);
        assert_eq!(Region::EMPTY.union(a), a);
        assert_eq!(a.union(b).bounding_rect(), Rect::new(0, 0, 12, 12));
    }

    #[test]
    fn intersect_clips_a_demand_partly_outside_a_producer() {
        // A demand straddling the producer's right edge clips to the overlap.
        let demand = Region::from_rect(Rect::new(50, 10, 80, 40));
        let clipped = demand.clamp_to_extent(Extent::new(64, 48));
        assert_eq!(clipped.bounding_rect(), Rect::new(50, 10, 64, 40));

        // A demand wholly outside the producer clips to empty.
        let outside = Region::from_rect(Rect::new(100, 100, 120, 120));
        assert!(outside.clamp_to_extent(Extent::new(64, 48)).is_empty());
    }

    #[test]
    fn dilate_grows_then_clips_back_to_the_producer() {
        // A halo grows the demand past the producer edge; clamping returns it.
        let demand = Region::from_rect(Rect::new(0, 0, 8, 8));
        let grown = demand.dilate(3);
        assert_eq!(grown.bounding_rect(), Rect::new(-3, -3, 11, 11));
        let clipped = grown.clamp_to_extent(Extent::new(64, 48));
        assert_eq!(clipped.bounding_rect(), Rect::new(0, 0, 11, 11));

        // Dilating an empty region is empty.
        assert!(Region::EMPTY.dilate(5).is_empty());
    }

    #[test]
    fn dilate_xy_uses_independent_axes() {
        let r = Region::from_rect(Rect::new(10, 10, 20, 20)).dilate_xy(2, 5);
        assert_eq!(r.bounding_rect(), Rect::new(8, 5, 22, 25));
    }

    #[test]
    fn translate_shifts_the_demand() {
        let r = Region::from_rect(Rect::new(0, 0, 10, 10)).translate(4, 7);
        assert_eq!(r.bounding_rect(), Rect::new(4, 7, 14, 17));
        assert!(Region::EMPTY.translate(9, 9).is_empty());
    }

    #[test]
    fn contains_rect_is_the_cover_check() {
        let r = Region::from_rect(Rect::new(0, 0, 20, 20));
        assert!(r.contains_rect(Rect::new(2, 2, 8, 8)));
        assert!(!r.contains_rect(Rect::new(15, 15, 25, 25)));
        // An empty rect is contained by anything, including the empty region.
        assert!(r.contains_rect(Rect::EMPTY));
        assert!(Region::EMPTY.contains_rect(Rect::EMPTY));
        assert!(!Region::EMPTY.contains_rect(Rect::new(0, 0, 1, 1)));
    }

    #[test]
    fn from_iter_collects_a_union() {
        let region: Region = [
            Rect::new(0, 0, 4, 4),
            Rect::EMPTY,
            Rect::new(10, 10, 14, 14),
        ]
        .into_iter()
        .collect();
        assert_eq!(region.bounding_rect(), Rect::new(0, 0, 14, 14));
    }

    #[test]
    fn region_serde_round_trips_canonically() {
        let r = Region::from_rect(Rect::new(1, 2, 9, 8));
        let back: Region = serde_json::from_value(serde_json::to_value(r).unwrap()).unwrap();
        assert_eq!(back, r);
        let empty: Region =
            serde_json::from_value(serde_json::to_value(Region::EMPTY).unwrap()).unwrap();
        assert!(empty.is_empty());
    }
}
