//! Metamorphic-relation helpers for op verification (`AGENT_VERIFICATION` §2.5).
//!
//! A *metamorphic relation* asserts how an op's output must change under a
//! transformation of its input, without needing a precomputed golden: e.g. an
//! **involution** (`f(f(x)) == x`), a **periodic identity** (`fⁿ(x) == x`), or a
//! **covariance** (`f(T(x)) == T(f(x))` for some transform `T`). These relations
//! are the backbone of the geometry-op suites (90° rotation covariance, reflection
//! covariance) and the keystone exactness checks (double-flip, four-turn
//! identity).
//!
//! The helpers here are deliberately generic over a "run the op" closure so any
//! op's test module can reuse them: they compare [`ResourceValue`]s by exact
//! sample and extent equality (the geometry ops are
//! [`Exact`](paintop_ir::DeterminismTier::Exact)), or within a tolerance for
//! resampling ops.

use paintop_core::executor::ResourceValue;

/// Whether two resource values have the same extent, channel count, and
/// bit-identical samples (the exactness predicate for integer remap ops).
#[must_use]
pub fn samples_bit_identical(a: &ResourceValue, b: &ResourceValue) -> bool {
    a.extent() == b.extent()
        && a.channels() == b.channels()
        && a.samples().len() == b.samples().len()
        && a.samples()
            .iter()
            .zip(b.samples().iter())
            .all(|(x, y)| x.to_bits() == y.to_bits())
}

/// The maximum absolute per-sample difference between two equally-shaped values,
/// or `f32::INFINITY` if their shapes differ (so a shape mismatch never passes a
/// tolerance check).
#[must_use]
pub fn max_abs_diff(a: &ResourceValue, b: &ResourceValue) -> f32 {
    if a.extent() != b.extent()
        || a.channels() != b.channels()
        || a.samples().len() != b.samples().len()
    {
        return f32::INFINITY;
    }
    a.samples()
        .iter()
        .zip(b.samples().iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f32, f32::max)
}

/// Assert that `op` is an **involution** on `input`: applying it twice reproduces
/// the input bit-for-bit. Panics with a descriptive message otherwise.
///
/// # Panics
/// If `op(op(input))` is not bit-identical to `input`.
pub fn assert_involution<F>(input: &ResourceValue, op: F)
where
    F: Fn(&ResourceValue) -> ResourceValue,
{
    let once = op(input);
    let twice = op(&once);
    assert!(
        samples_bit_identical(&twice, input),
        "metamorphic involution failed: op(op(x)) != x \
         (extent {:?} -> {:?}, max_abs_diff {})",
        input.extent(),
        twice.extent(),
        max_abs_diff(&twice, input),
    );
}

/// Assert that applying `op` exactly `period` times is the **identity** on
/// `input`, bit-for-bit (e.g. four 90° turns). Panics otherwise.
///
/// # Panics
/// If `op` applied `period` times does not reproduce `input` bit-identically, or
/// if `period == 0`.
pub fn assert_periodic_identity<F>(input: &ResourceValue, period: u32, op: F)
where
    F: Fn(&ResourceValue) -> ResourceValue,
{
    assert!(period > 0, "period must be positive");
    let mut current = op(input);
    for _ in 1..period {
        current = op(&current);
    }
    assert!(
        samples_bit_identical(&current, input),
        "metamorphic periodic identity failed: op^{period}(x) != x \
         (extent {:?} -> {:?}, max_abs_diff {})",
        input.extent(),
        current.extent(),
        max_abs_diff(&current, input),
    );
}

/// Assert a **covariance** relation `op(transform(x)) == transform(op(x))`,
/// bit-for-bit (e.g. a pointwise op commutes with a flip). Panics otherwise.
///
/// # Panics
/// If the two composition orders disagree bit-identically.
pub fn assert_covariant<Op, Tr>(input: &ResourceValue, op: Op, transform: Tr)
where
    Op: Fn(&ResourceValue) -> ResourceValue,
    Tr: Fn(&ResourceValue) -> ResourceValue,
{
    let op_then_transform = transform(&op(input));
    let transform_then_op = op(&transform(input));
    assert!(
        samples_bit_identical(&op_then_transform, &transform_then_op),
        "metamorphic covariance failed: op(T(x)) != T(op(x)) \
         (max_abs_diff {})",
        max_abs_diff(&op_then_transform, &transform_then_op),
    );
}
