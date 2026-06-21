//! Exact Euclidean distance transform (EDT) — a `paintop-cpu` library primitive.
//!
//! This module is the numeric foundation under every signed-distance-field op
//! (`mask.to_sdf`, `sdf.*`): it computes, for each pixel of a binary grid, the
//! **exact squared Euclidean distance** to the nearest *set* (foreground) pixel.
//! It is a primitive plus a brute-force oracle, **not** a public graph op, so it
//! carries no manifest, no registration, and no `verify-op` gate; its only
//! contract is this module's documentation and its differential agreement with
//! the oracle (`RESEARCH` §3.1, `ALIEN_OPS` §2.4/§2.5, `plan.md` M4).
//!
//! # The binary set
//!
//! The transform operates over a [`BinaryGrid`]: a `width × height`, row-major
//! view in which each cell is either **set** (a member of the foreground set
//! `S`, also called a *seed* or *site*) or **clear**. SDF callers obtain this
//! grid by thresholding a coverage [`Mask`](paintop_ir::ResourceKind::Mask) at a
//! caller-chosen level (`coverage >= threshold` ⇒ set); the thresholding policy
//! belongs to the SDF op, not to the EDT, which sees only booleans.
//!
//! # Semantics — what the transform returns
//!
//! For grid coordinate `p = (x, y)` (pixel *centers*, the project's
//! [`PixelCenterUpperLeft`](paintop_ir::CoordinateConvention::PixelCenterUpperLeft)
//! convention) the **squared** distance transform is
//!
//! ```text
//! D²(p) = min over s in S of (x - s.x)² + (y - s.y)²
//! ```
//!
//! the squared Euclidean distance from `p` to the nearest set pixel, measured
//! between pixel centers in pixel units. The squared distance is the natural
//! output because the separable algorithm is exact over the *integer* squared
//! distances (no early square root, hence no accumulated rounding); callers that
//! want the unsigned distance take `sqrt` of each sample (see [`distance`]).
//!
//! ## Inside / outside and the SDF sign convention
//!
//! The EDT itself is unsigned: it answers "how far to the nearest set pixel".
//! It deliberately does **not** decide a sign — that is the SDF op's job. The
//! project's signed-distance sign convention is **negative inside** the set
//! (`IR_SPEC` §7.4): an SDF is built by running the EDT twice, once over `S`
//! (the outside distance, `>= 0`) and once over the complement `\overline S`
//! (the inside distance), and combining them as
//! `sdf(p) = sqrt(D²_{S}(p)) - sqrt(D²_{\overline S}(p))`, which is `0` on the
//! boundary, negative strictly inside, and positive strictly outside. This
//! module supplies the two raw squared transforms; the signed combination is
//! layered above it.
//!
//! ## Degenerate inputs
//!
//! - **Empty set** (no set pixel): every distance is `+∞` — there is no nearest
//!   seed. The squared transform stores [`f32::INFINITY`] / the brute force the
//!   same, so the two agree exactly.
//! - **Full set** (every pixel set): every squared distance is `0`.
//! - **Single set pixel** at `s`: `D²(p) = |p - s|²`, the plain squared distance
//!   to that one site.
//!
//! # Determinism and tie behavior
//!
//! `D²` is a `min` over an unordered set, so its *value* is independent of any
//! traversal order and is bit-identical across platforms: every intermediate is
//! an exact non-negative integer (a sum of squared `i64` index differences) held
//! losslessly, and only the final cast to `f32` rounds — and only for distances
//! beyond `2^24`, far past any realistic image. When several seeds are
//! equidistant the *distance* is unambiguous (ties do not change `D²`); this
//! module returns distances, not the identity of the nearest seed, so tie
//! behavior is fully determined.

use paintop_ir::Extent;

/// A binary `width × height` grid: each cell is *set* (a foreground seed) or
/// *clear*, stored row-major (`index = y * width + x`).
///
/// This is the EDT's only input shape. It borrows the caller's membership slice
/// so thresholding a coverage mask never copies the booleans.
#[derive(Debug, Clone, Copy)]
pub struct BinaryGrid<'a> {
    width: u32,
    height: u32,
    /// Row-major membership, length `width * height`; `true` ⇒ the cell is set.
    set: &'a [bool],
}

/// A `BinaryGrid` could not be formed because the membership slice length did
/// not equal `width * height`.
///
/// This is a programming error in the caller (the grid is internally
/// constructed from a sized resource), surfaced as a typed value rather than a
/// panic so the lint wall's `unwrap_used`/panic-free policy holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error(
    "binary grid membership length {actual} does not match extent {width}x{height} = {expected}"
)]
pub struct GridShapeError {
    /// The grid width the membership slice was supposed to fill.
    pub width: u32,
    /// The grid height the membership slice was supposed to fill.
    pub height: u32,
    /// The expected length `width * height`.
    pub expected: usize,
    /// The actual membership slice length.
    pub actual: usize,
}

impl<'a> BinaryGrid<'a> {
    /// Borrow a `width × height` membership slice as a binary grid.
    ///
    /// # Errors
    /// [`GridShapeError`] if `set.len() != width * height` (computed saturating,
    /// so an overflowing extent simply fails the length check).
    pub const fn new(extent: Extent, set: &'a [bool]) -> Result<Self, GridShapeError> {
        let expected = (extent.width as usize).saturating_mul(extent.height as usize);
        if set.len() == expected {
            Ok(Self {
                width: extent.width,
                height: extent.height,
                set,
            })
        } else {
            Err(GridShapeError {
                width: extent.width,
                height: extent.height,
                expected,
                actual: set.len(),
            })
        }
    }

    /// The grid extent.
    #[must_use]
    pub const fn extent(&self) -> Extent {
        Extent::new(self.width, self.height)
    }

    /// The grid width in pixels.
    #[must_use]
    pub const fn width(&self) -> u32 {
        self.width
    }

    /// The grid height in pixels.
    #[must_use]
    pub const fn height(&self) -> u32 {
        self.height
    }

    /// The number of cells, `width * height`, as a `usize` (always equal to the
    /// membership length by construction).
    #[must_use]
    pub const fn len(&self) -> usize {
        self.set.len()
    }

    /// Whether the grid has no cells (a zero-area extent).
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.set.is_empty()
    }

    /// Whether cell `(x, y)` is set. Out-of-range coordinates are *clear*; the
    /// EDT never indexes outside the grid, so this is only a defensive guard.
    #[must_use]
    pub fn is_set(&self, x: u32, y: u32) -> bool {
        if x >= self.width || y >= self.height {
            return false;
        }
        let index = (y as usize) * (self.width as usize) + (x as usize);
        self.set.get(index).copied().unwrap_or(false)
    }
}

/// The exact squared Euclidean distance from each cell to the nearest set pixel.
///
/// Computed by exhaustive search — the `O(W·H·|S|)` **oracle** the linear-time
/// separable transform is differentially checked against on small fixtures.
///
/// The result is row-major, length `width * height`. A cell over an empty set
/// (no seed anywhere) is [`f32::INFINITY`]; otherwise it is the integer squared
/// distance to the nearest seed, exactly representable in `f32` for any image
/// whose squared diagonal is below `2^24`.
///
/// This is intentionally the simplest correct definition — a direct transcription
/// of `D²(p) = min_s |p − s|²` — so it has no shared logic with the fast path and
/// makes an independent oracle.
#[must_use]
pub fn brute_force_sq(grid: &BinaryGrid<'_>) -> Vec<f32> {
    let width = grid.width as usize;
    let height = grid.height as usize;
    // Collect the seed coordinates once as i64 so the inner loop is pure
    // integer arithmetic with no per-seed bounds checks.
    let mut seeds: Vec<(i64, i64)> = Vec::new();
    for y in 0..grid.height {
        for x in 0..grid.width {
            if grid.is_set(x, y) {
                seeds.push((i64::from(x), i64::from(y)));
            }
        }
    }

    let mut out = vec![f32::INFINITY; width.saturating_mul(height)];
    if seeds.is_empty() {
        return out;
    }
    for y in 0..grid.height {
        for x in 0..grid.width {
            let px = i64::from(x);
            let py = i64::from(y);
            let mut best: i64 = i64::MAX;
            for &(sx, sy) in &seeds {
                let dx = px - sx;
                let dy = py - sy;
                let d2 = dx * dx + dy * dy;
                if d2 < best {
                    best = d2;
                    if best == 0 {
                        break;
                    }
                }
            }
            let index = (y as usize) * width + (x as usize);
            #[allow(
                clippy::cast_precision_loss,
                reason = "integer squared distance; lossless below 2^24, far beyond any image"
            )]
            {
                out[index] = best as f32;
            }
        }
    }
    out
}

/// The exact squared Euclidean distance from each cell to the nearest set pixel,
/// computed by the **Felzenszwalb–Huttenlocher** linear-time separable transform.
///
/// This is the fast path the SDF ops use: `O(W·H)` regardless of how many cells
/// are set, versus the oracle's `O(W·H·|S|)`. The result is identical to
/// [`brute_force_sq`] sample-for-sample (exact, max abs error 0) — the two are
/// differentially checked against each other.
///
/// # Algorithm
///
/// The squared Euclidean distance is separable: `D²(x, y) = min_{x', y'}
/// (x−x')² + (y−y')² + 𝟙[(x',y') not set]·∞`. Felzenszwalb–Huttenlocher computes
/// it as two passes of a 1-D distance transform — first down each **column**,
/// then across each **row** of the column result — where the 1-D transform is the
/// lower envelope of the parabolas `(q − v)² + f(v)` rooted at each sample `v`.
/// Each 1-D transform is linear time via the classic two-stack envelope sweep, so
/// the whole transform is linear in the pixel count.
///
/// # Output policy and determinism
///
/// The seed grid contributes only the value `0` (set) or `+∞` (clear), so every
/// finite intermediate is an exact non-negative integer; the envelope arithmetic
/// runs in `f64`, where those integers and their squared-index offsets are
/// represented losslessly for any realistic image, and the final cast to the
/// `f32` output is exact below `2^24`. The traversal order is fixed, the `min` is
/// order-independent, and there is no platform-variant intrinsic, so the result
/// is bit-identical across runs and platforms. An empty set yields `+∞`
/// everywhere (the envelope of no finite parabola); a full set yields `0`.
#[must_use]
pub fn transform_sq(grid: &BinaryGrid<'_>) -> Vec<f32> {
    let width = grid.width as usize;
    let height = grid.height as usize;
    let cells = width.saturating_mul(height);
    if cells == 0 {
        return Vec::new();
    }

    // Seed field: 0 where set, +inf where clear, row-major in f64.
    let mut field = vec![f64::INFINITY; cells];
    for y in 0..grid.height {
        for x in 0..grid.width {
            if grid.is_set(x, y) {
                field[(y as usize) * width + (x as usize)] = 0.0;
            }
        }
    }

    // Scratch buffers sized to the longer axis, reused across lines so the whole
    // transform allocates O(W + H) beyond the field itself.
    let max_len = width.max(height);
    let mut column = vec![0.0_f64; height];
    let mut line_out = vec![0.0_f64; max_len];
    // `v`: indices of the parabolas forming the lower envelope; `z`: the envelope
    // boundaries between consecutive parabolas (`z` has one extra slot).
    let mut v = vec![0_usize; max_len];
    let mut z = vec![0.0_f64; max_len + 1];

    // Pass 1 — transform down each column (vertical, the y axis).
    for x in 0..width {
        for y in 0..height {
            column[y] = field[y * width + x];
        }
        dt_1d(&column[..height], &mut line_out[..height], &mut v, &mut z);
        for y in 0..height {
            field[y * width + x] = line_out[y];
        }
    }

    // Pass 2 — transform across each row (horizontal, the x axis) in place over
    // the column result; the row is contiguous so we transform it directly.
    let mut row = vec![0.0_f64; width];
    for y in 0..height {
        let base = y * width;
        row.copy_from_slice(&field[base..base + width]);
        dt_1d(&row[..width], &mut line_out[..width], &mut v, &mut z);
        field[base..base + width].copy_from_slice(&line_out[..width]);
    }

    field
        .iter()
        .map(|&d2| {
            #[allow(
                clippy::cast_possible_truncation,
                reason = "exact integer squared distance (or +inf); lossless f64->f32 below 2^24"
            )]
            {
                d2 as f32
            }
        })
        .collect()
}

/// The exact 1-D squared distance transform of `f` into `out`: for each query
/// index `q`, `out[q] = min_v (q − v)² + f[v]`.
///
/// The classic Felzenszwalb–Huttenlocher lower-envelope sweep. `f` is the sampled
/// function (`+∞` where there is no site); `out`, `v`, and `z` are caller-owned
/// scratch (`v`/`z` need at least `f.len()` / `f.len() + 1` slots). A column of
/// all-`+∞` leaves the envelope empty and the output `+∞`, the empty-set case.
#[allow(
    clippy::many_single_char_names,
    reason = "v (parabola index), z (envelope boundary), and q (query) are the \
              standard Felzenszwalb–Huttenlocher symbols; renaming obscures the algorithm"
)]
fn dt_1d(sampled: &[f64], out: &mut [f64], parab: &mut [usize], bound: &mut [f64]) {
    let n = sampled.len();
    debug_assert_eq!(out.len(), n);
    if n == 0 {
        return;
    }

    // Build the lower envelope over the finite parabolas only. `envelope_len` is
    // the number of envelope parabolas; their roots are in `parab`, the crossover
    // abscissae between consecutive ones in `bound`.
    let mut envelope_len: usize = 0;
    for q in 0..n {
        let fq = sampled[q];
        if fq.is_infinite() {
            continue;
        }
        #[allow(
            clippy::cast_precision_loss,
            reason = "indices are small grid coordinates, exact in f64"
        )]
        let qf = q as f64;
        // Pop envelope parabolas that the new one at `q` hides, then place it.
        loop {
            if envelope_len == 0 {
                parab[0] = q;
                bound[0] = f64::NEG_INFINITY;
                bound[1] = f64::INFINITY;
                envelope_len = 1;
                break;
            }
            let p = parab[envelope_len - 1];
            #[allow(
                clippy::cast_precision_loss,
                reason = "indices are small grid coordinates, exact in f64"
            )]
            let pf = p as f64;
            // Intersection abscissa of the parabolas rooted at `p` and `q`:
            //   s = ((f[q] + q²) − (f[p] + p²)) / (2q − 2p).
            // The `q²`/`p²` terms are formed with `mul_add` so the exact integer
            // sums round only once; both operands are finite here.
            let numer = qf.mul_add(qf, fq) - pf.mul_add(pf, sampled[p]);
            let cross = numer / (2.0 * (qf - pf));
            if cross <= bound[envelope_len - 1] {
                // The new parabola hides the current rightmost one: pop and retry.
                envelope_len -= 1;
            } else {
                parab[envelope_len] = q;
                bound[envelope_len] = cross;
                bound[envelope_len + 1] = f64::INFINITY;
                envelope_len += 1;
                break;
            }
        }
    }

    if envelope_len == 0 {
        // No finite site in this line: distance is +inf everywhere.
        for slot in out.iter_mut() {
            *slot = f64::INFINITY;
        }
        return;
    }

    // Sweep queries left to right, advancing through the envelope segments. The
    // crossover boundaries are monotone in `k`, so each segment is entered once;
    // a `loop`/`break` (not a `while`) does the float-bounded advance.
    let mut k = 0usize;
    for (q, slot) in out.iter_mut().enumerate() {
        #[allow(
            clippy::cast_precision_loss,
            reason = "indices are small grid coordinates, exact in f64"
        )]
        let qf = q as f64;
        loop {
            if bound[k + 1] < qf {
                k += 1;
            } else {
                break;
            }
        }
        let root = parab[k];
        #[allow(
            clippy::cast_precision_loss,
            reason = "indices are small grid coordinates, exact in f64"
        )]
        let root_f = root as f64;
        let delta = qf - root_f;
        *slot = delta.mul_add(delta, sampled[root]);
    }
}

/// The unsigned Euclidean distance for each cell: the elementwise square root of
/// a squared-distance buffer.
///
/// `+∞` (the empty-set sentinel) maps to `+∞`; every finite squared distance to
/// its non-negative root. Provided so callers (and tests) share one rounding
/// policy for the `sqrt`.
#[must_use]
pub fn distance(squared: &[f32]) -> Vec<f32> {
    squared.iter().map(|&d2| d2.sqrt()).collect()
}

#[cfg(test)]
mod tests;
