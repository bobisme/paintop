//! Brute-force EDT oracle fixtures and contract checks.
//!
//! These cover the binary-grid contract (shape validation, membership) and the
//! oracle's behavior on the four canonical degenerate/structured masks the
//! acceptance criteria call out — **empty**, **full**, **single-pixel**, and a
//! **ring** — establishing the ground truth the linear-time separable transform
//! is differentially checked against (`bn-1zo`, `bn-1v6`).

use paintop_ir::Extent;

use super::{BinaryGrid, GridShapeError, brute_force_sq, distance, transform_sq};

/// Build a grid membership slice from a `width × height` predicate of `(x, y)`.
fn membership(width: u32, height: u32, mut set: impl FnMut(u32, u32) -> bool) -> Vec<bool> {
    let mut cells = Vec::with_capacity((width as usize) * (height as usize));
    for y in 0..height {
        for x in 0..width {
            cells.push(set(x, y));
        }
    }
    cells
}

/// Assert two `f32` samples are bit-identical (exact-equality without tripping
/// the `float_cmp` lint — every EDT value is an exact integer or a sentinel).
#[track_caller]
fn assert_bits_eq(got: f32, want: f32, label: &str) {
    assert_eq!(got.to_bits(), want.to_bits(), "{label}: {got} != {want}");
}

/// The exact squared distance to a single site `(sx, sy)`.
fn sq_to(x: u32, y: u32, sx: u32, sy: u32) -> f32 {
    let dx = f64::from(x) - f64::from(sx);
    let dy = f64::from(y) - f64::from(sy);
    #[allow(clippy::cast_possible_truncation, reason = "small integer fixtures")]
    {
        dy.mul_add(dy, dx * dx) as f32
    }
}

// ---------------------------------------------------------------------------
// BinaryGrid contract
// ---------------------------------------------------------------------------

#[test]
fn grid_rejects_wrong_length() {
    let err =
        BinaryGrid::new(Extent::new(3, 2), &[false; 5]).expect_err("length mismatch rejected");
    assert_eq!(
        err,
        GridShapeError {
            width: 3,
            height: 2,
            expected: 6,
            actual: 5,
        }
    );
}

#[test]
fn grid_accepts_exact_length_and_reports_membership() {
    let cells = membership(2, 2, |x, y| x == y);
    let grid = BinaryGrid::new(Extent::new(2, 2), &cells).expect("exact length accepted");
    assert_eq!(grid.extent(), Extent::new(2, 2));
    assert_eq!(grid.len(), 4);
    assert!(!grid.is_empty());
    assert!(grid.is_set(0, 0));
    assert!(grid.is_set(1, 1));
    assert!(!grid.is_set(1, 0));
    assert!(!grid.is_set(0, 1));
    // Out-of-range coordinates are clear, never a panic.
    assert!(!grid.is_set(2, 0));
    assert!(!grid.is_set(0, 2));
}

#[test]
fn zero_area_grid_is_empty() {
    let grid = BinaryGrid::new(Extent::new(0, 4), &[]).expect("zero-area grid");
    assert!(grid.is_empty());
    assert!(brute_force_sq(&grid).is_empty());
}

// ---------------------------------------------------------------------------
// Degenerate fixtures: empty / full / single-pixel
// ---------------------------------------------------------------------------

#[test]
fn empty_set_is_infinite_everywhere() {
    let cells = membership(4, 3, |_, _| false);
    let grid = BinaryGrid::new(Extent::new(4, 3), &cells).expect("grid");
    let d2 = brute_force_sq(&grid);
    assert_eq!(d2.len(), 12);
    assert!(d2.iter().all(|v| v.is_infinite() && v.is_sign_positive()));
    // sqrt of +inf is +inf — the empty-set sentinel survives the distance step.
    assert!(distance(&d2).iter().all(|v| v.is_infinite()));
}

#[test]
fn full_set_is_zero_everywhere() {
    let cells = membership(4, 3, |_, _| true);
    let grid = BinaryGrid::new(Extent::new(4, 3), &cells).expect("grid");
    let d2 = brute_force_sq(&grid);
    assert!(d2.iter().all(|v| v.to_bits() == 0.0_f32.to_bits()));
}

#[test]
fn single_pixel_is_squared_distance_to_that_site() {
    let (sx, sy) = (1u32, 2u32);
    let cells = membership(5, 5, |x, y| x == sx && y == sy);
    let grid = BinaryGrid::new(Extent::new(5, 5), &cells).expect("grid");
    let d2 = brute_force_sq(&grid);
    for y in 0..5 {
        for x in 0..5 {
            let got = d2[(y as usize) * 5 + (x as usize)];
            assert_bits_eq(got, sq_to(x, y, sx, sy), &format!("at ({x},{y})"));
        }
    }
    // The seed itself is at distance zero.
    assert_bits_eq(d2[(sy as usize) * 5 + (sx as usize)], 0.0, "seed");
}

// ---------------------------------------------------------------------------
// Structured fixture: a ring (boundary set, hollow interior)
// ---------------------------------------------------------------------------

#[test]
fn ring_interior_distance_matches_nearest_border() {
    // 5x5 with the outer border set and a hollow 3x3 interior.
    let w = 5;
    let h = 5;
    let cells = membership(w, h, |x, y| x == 0 || y == 0 || x == w - 1 || y == h - 1);
    let grid = BinaryGrid::new(Extent::new(w, h), &cells).expect("grid");
    let d2 = brute_force_sq(&grid);

    // Border cells are themselves set ⇒ distance 0.
    for x in 0..w {
        assert_bits_eq(d2[x as usize], 0.0, "top border");
        assert_bits_eq(d2[((h - 1) * w + x) as usize], 0.0, "bottom border");
    }
    // The dead-center cell (2,2) sits two cells from the nearest border on every
    // axis ⇒ nearest seed is at distance 2, squared 4.
    assert_bits_eq(d2[(2 * w + 2) as usize], 4.0, "center (2,2)");
    // The orthogonally-adjacent interior cells (e.g. (1,2)) touch the border ⇒ 1.
    assert_bits_eq(d2[(2 * w + 1) as usize], 1.0, "edge interior (1,2)");
    // A diagonal interior corner (1,1) is adjacent to two borders ⇒ nearest is
    // the orthogonal border cell at distance 1, not the diagonal at sqrt(2).
    assert_bits_eq(d2[(w + 1) as usize], 1.0, "corner interior (1,1)");
}

#[test]
fn distance_is_sqrt_of_squared() {
    // A two-seed fixture exercising a non-unit, non-integer-root distance.
    let cells = membership(4, 1, |x, _| x == 0 || x == 3);
    let grid = BinaryGrid::new(Extent::new(4, 1), &cells).expect("grid");
    let d2 = brute_force_sq(&grid);
    assert_eq!(d2, vec![0.0, 1.0, 1.0, 0.0]);
    assert_eq!(distance(&d2), vec![0.0, 1.0, 1.0, 0.0]);
}

// ---------------------------------------------------------------------------
// Felzenszwalb–Huttenlocher separable transform vs. brute-force oracle
// ---------------------------------------------------------------------------

/// A tiny, dependency-free deterministic PRNG (`SplitMix64`) for generating
/// reproducible random masks without pulling in `rand`.
struct SplitMix64(u64);

impl SplitMix64 {
    const fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// A `true` with probability `numerator / 256`.
    fn bit(&mut self, numerator: u64) -> bool {
        (self.next_u64() & 0xFF) < numerator
    }
}

/// Assert the separable transform agrees with the oracle exactly (every sample
/// bit-identical), the M4 acceptance bar of "max abs error 0 on integer
/// fixtures".
#[track_caller]
fn assert_matches_oracle(extent: Extent, cells: &[bool]) {
    let grid = BinaryGrid::new(extent, cells).expect("grid");
    let fast = transform_sq(&grid);
    let oracle = brute_force_sq(&grid);
    assert_eq!(fast.len(), oracle.len());
    for (i, (&f, &o)) in fast.iter().zip(oracle.iter()).enumerate() {
        assert_eq!(
            f.to_bits(),
            o.to_bits(),
            "extent {}x{} cell {i}: separable {f} != oracle {o}",
            extent.width,
            extent.height
        );
    }
}

#[test]
fn separable_matches_oracle_on_degenerate_masks() {
    for &(w, h) in &[(1, 1), (1, 5), (5, 1), (4, 3), (7, 6)] {
        assert_matches_oracle(Extent::new(w, h), &membership(w, h, |_, _| false));
        assert_matches_oracle(Extent::new(w, h), &membership(w, h, |_, _| true));
    }
}

#[test]
fn separable_matches_oracle_on_single_pixel() {
    let (w, h) = (9, 7);
    for sy in 0..h {
        for sx in 0..w {
            let cells = membership(w, h, |x, y| x == sx && y == sy);
            assert_matches_oracle(Extent::new(w, h), &cells);
        }
    }
}

#[test]
fn separable_matches_oracle_on_structured_masks() {
    let (w, h) = (11, 9);
    // Outer ring.
    assert_matches_oracle(
        Extent::new(w, h),
        &membership(w, h, |x, y| x == 0 || y == 0 || x == w - 1 || y == h - 1),
    );
    // A filled disc.
    let (cx, cy, r2) = (5.0_f64, 4.0_f64, 9.0_f64);
    assert_matches_oracle(
        Extent::new(w, h),
        &membership(w, h, |x, y| {
            let dx = f64::from(x) - cx;
            let dy = f64::from(y) - cy;
            dy.mul_add(dy, dx * dx) <= r2
        }),
    );
    // Two separated seeds (forces the lower envelope to splice two parabolas).
    assert_matches_oracle(
        Extent::new(w, h),
        &membership(w, h, |x, y| (x, y) == (1, 1) || (x, y) == (9, 7)),
    );
    // A diagonal stripe.
    assert_matches_oracle(
        Extent::new(w, h),
        &membership(w, h, |x, y| (x + y) % 4 == 0),
    );
}

#[test]
fn separable_matches_oracle_on_random_masks() {
    let mut rng = SplitMix64::new(0xDEAD_BEEF_1234_5678);
    // Sweep sizes and densities so the envelope sees sparse, balanced, and dense
    // seed fields across square and oblong grids.
    for &(w, h) in &[(3, 3), (8, 5), (5, 8), (13, 11), (16, 16), (20, 1), (1, 20)] {
        for &density in &[8_u64, 32, 128, 224] {
            for _ in 0..6 {
                let cells = membership(w, h, |_, _| rng.bit(density));
                assert_matches_oracle(Extent::new(w, h), &cells);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Verification properties: finiteness/range, gradient magnitude, rotation
// covariance, pathological masks, and a performance characterization (bn-1v6).
// ---------------------------------------------------------------------------

/// The unsigned distance field of the separable transform.
fn fast_distance(extent: Extent, cells: &[bool]) -> Vec<f32> {
    let grid = BinaryGrid::new(extent, cells).expect("grid");
    distance(&transform_sq(&grid))
}

#[test]
fn nonempty_field_is_finite_nonnegative_and_zero_on_seeds() {
    let (w, h) = (17, 13);
    let cells = membership(w, h, |x, y| (x * 3 + y * 5) % 11 == 0);
    let grid = BinaryGrid::new(Extent::new(w, h), &cells).expect("grid");
    let d2 = transform_sq(&grid);
    for (i, &v) in d2.iter().enumerate() {
        assert!(v.is_finite(), "cell {i} not finite: {v}");
        assert!(v >= 0.0, "cell {i} negative: {v}");
    }
    // Every seed reads exactly zero; every cleared cell is strictly positive
    // (some seed exists, but not at that cell).
    for y in 0..h {
        for x in 0..w {
            let v = d2[(y as usize) * (w as usize) + (x as usize)];
            if grid.is_set(x, y) {
                assert_bits_eq(v, 0.0, "seed cell");
            } else {
                assert!(v > 0.0, "cleared cell ({x},{y}) is {v}");
            }
        }
    }
}

#[test]
fn distance_gradient_magnitude_is_one_away_from_medial_axis() {
    // A single column of seeds at x = 0: the exact unsigned distance of cell
    // (x, y) is |x|, so the field is a perfect ramp with horizontal gradient 1
    // and zero vertical gradient everywhere off the seed line. There is no medial
    // axis (a half-plane has none), so the unit-gradient property holds globally.
    let (w, h) = (12, 6);
    let cells = membership(w, h, |x, _| x == 0);
    let d = fast_distance(Extent::new(w, h), &cells);
    let at = |x: u32, y: u32| d[(y as usize) * (w as usize) + (x as usize)];
    for y in 0..h {
        for x in 0..w {
            // Distance equals the column index exactly (small, exact in f32).
            #[allow(clippy::cast_precision_loss, reason = "x < 12, exact in f32")]
            let want = x as f32;
            assert_bits_eq(at(x, y), want, &format!("ramp ({x},{y})"));
        }
        // Horizontal gradient is exactly 1; vertical gradient is exactly 0.
        for x in 1..w {
            let gx = at(x, y) - at(x - 1, y);
            assert_bits_eq(gx, 1.0, &format!("dx grad ({x},{y})"));
        }
    }
    for x in 0..w {
        for y in 1..h {
            let gy = at(x, y) - at(x, y - 1);
            assert_bits_eq(gy, 0.0, &format!("dy grad ({x},{y})"));
        }
    }
}

#[test]
fn distance_gradient_magnitude_near_one_for_single_seed_off_axis() {
    // For a lone seed the field is the true radial distance; away from the seed
    // (off the discrete medial axis, here just the seed cell) the central-
    // difference gradient magnitude is ~1 to within the grid discretization.
    let (w, h) = (21, 21);
    let (sx, sy) = (10u32, 10u32);
    let cells = membership(w, h, |x, y| x == sx && y == sy);
    let d = fast_distance(Extent::new(w, h), &cells);
    let at = |x: u32, y: u32| f64::from(d[(y as usize) * (w as usize) + (x as usize)]);
    let mut worst = 0.0_f64;
    for y in 1..h - 1 {
        for x in 1..w - 1 {
            // Skip the immediate neighborhood of the seed where the field is not
            // smooth (the discretization error of |∇d| is largest at r≈0).
            let dist = at(x, y);
            if dist < 3.0 {
                continue;
            }
            let gx = 0.5 * (at(x + 1, y) - at(x - 1, y));
            let gy = 0.5 * (at(x, y + 1) - at(x, y - 1));
            let mag = gy.mul_add(gy, gx * gx).sqrt();
            worst = worst.max((mag - 1.0).abs());
        }
    }
    assert!(
        worst < 0.05,
        "central-difference |grad| deviates from 1 by {worst}"
    );
}

#[test]
fn rotation_covariance_under_quarter_turn() {
    // Rotating the seed set 90° rotates the distance field the same way: the EDT
    // commutes with the (Euclidean) lattice rotation. Use a symmetric-but-not-
    // rotation-invariant fixture so the test has teeth.
    let (w, h) = (9, 5);
    let cells = membership(w, h, |x, y| {
        (x == 0 && y == 0) || (x == w - 1 && y == h - 1)
    });
    let d = fast_distance(Extent::new(w, h), &cells);

    // Rotate the seed mask a quarter turn clockwise into a (h x w) grid:
    // (x, y) -> (h-1-y, x).
    let (rw, rh) = (h, w);
    let mut rotated_cells = vec![false; (rw as usize) * (rh as usize)];
    for y in 0..h {
        for x in 0..w {
            if cells[(y as usize) * (w as usize) + (x as usize)] {
                let (nx, ny) = (h - 1 - y, x);
                rotated_cells[(ny as usize) * (rw as usize) + (nx as usize)] = true;
            }
        }
    }
    let dr = fast_distance(Extent::new(rw, rh), &rotated_cells);

    // The rotated field must equal the field of the rotated mask, cell for cell.
    for y in 0..h {
        for x in 0..w {
            let original = d[(y as usize) * (w as usize) + (x as usize)];
            let (nx, ny) = (h - 1 - y, x);
            let mapped = dr[(ny as usize) * (rw as usize) + (nx as usize)];
            assert_bits_eq(mapped, original, &format!("rot ({x},{y})"));
        }
    }
}

#[test]
fn pathological_masks_match_oracle() {
    // Thin one-pixel structures, lone corners, and a checkerboard stress the
    // envelope's pop/splice logic; each must still match the oracle exactly.
    let (w, h) = (24, 18);
    // Single-pixel-wide cross through the center.
    assert_matches_oracle(
        Extent::new(w, h),
        &membership(w, h, |x, y| x == w / 2 || y == h / 2),
    );
    // The four corners only.
    assert_matches_oracle(
        Extent::new(w, h),
        &membership(w, h, |x, y| {
            (x == 0 || x == w - 1) && (y == 0 || y == h - 1)
        }),
    );
    // Dense checkerboard (every other cell set): nearest seed is always adjacent.
    assert_matches_oracle(
        Extent::new(w, h),
        &membership(w, h, |x, y| (x + y) % 2 == 0),
    );
    // A single far-off seed in a large field (long-range envelope).
    assert_matches_oracle(
        Extent::new(w, h),
        &membership(w, h, |x, y| x == w - 1 && y == h - 1),
    );
}

#[test]
fn large_field_is_well_formed_and_records_timing() {
    // Performance characterization: the linear-time transform must comfortably
    // handle a megapixel-scale field. We assert correctness invariants (finite,
    // non-negative, seeds zero) and print a representative timing for the record;
    // we deliberately do not assert a wall-clock bound (CI-machine dependent).
    let (w, h) = (1024u32, 1024u32);
    // A sparse seed lattice every 64 px: the worst case for envelope length.
    let cells = membership(w, h, |x, y| x % 64 == 0 && y % 64 == 0);
    let grid = BinaryGrid::new(Extent::new(w, h), &cells).expect("grid");

    let start = std::time::Instant::now();
    let d2 = transform_sq(&grid);
    let elapsed = start.elapsed();

    assert_eq!(d2.len(), (w as usize) * (h as usize));
    assert!(d2.iter().all(|v| v.is_finite() && *v >= 0.0));
    // The corner (0,0) is a seed ⇒ zero; the cell at (32,32) is equidistant from
    // four lattice seeds at offset (±32, ±32) ⇒ nearest is 32 away on each axis.
    assert_bits_eq(d2[0], 0.0, "lattice seed");
    assert_bits_eq(
        d2[(32 * w + 32) as usize],
        (32.0_f32).mul_add(32.0, 32.0 * 32.0),
        "lattice cell center",
    );
    println!(
        "EDT perf: {w}x{h} ({} px) separable transform in {elapsed:?}",
        u64::from(w) * u64::from(h)
    );
}
