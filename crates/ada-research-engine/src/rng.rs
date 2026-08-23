//! Deterministic pseudo-random stream (`SplitMix64`).
//!
//! Conceptually reused from the `SciRust` algogen `DeterministicRng`: the same
//! `SplitMix64` update rule with Lemire multiply-shift index reduction, reimplemented
//! locally so the ADA research engine has no dependency on (or absolute path into)
//! the `SciRust` workspace. A given seed yields the identical sequence on every
//! platform; nothing consults the OS, threads or wall-clock time.

/// A deterministic, explicitly seeded random stream.
#[derive(Debug, Clone)]
pub struct SearchRng {
    state: u64,
}

impl SearchRng {
    /// Create a stream from `seed`.
    #[must_use]
    pub const fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// Next raw 64-bit value.
    #[must_use]
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform index in `[0, bound)`; returns 0 for `bound <= 1`.
    #[allow(clippy::cast_possible_truncation)]
    #[must_use]
    pub fn below(&mut self, bound: usize) -> usize {
        if bound <= 1 {
            return 0;
        }
        let product =
            u128::from(self.next_u64()).wrapping_mul(u128::try_from(bound).unwrap_or(u128::MAX));
        (product >> 64) as usize
    }

    /// Uniform `f64` in `[0, 1)`.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn unit(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64) * (1.0 / 9_007_199_254_740_992.0)
    }

    /// Finite `f64` magnitude sample in `[-magnitude, magnitude]`.
    ///
    /// A non-finite or non-positive magnitude degrades to `1.0` so the result
    /// can never be non-finite.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn signed_magnitude(&mut self, magnitude: f64) -> f64 {
        let magnitude = if magnitude.is_finite() && magnitude > 0.0 {
            magnitude
        } else {
            1.0
        };
        (2.0 * self.unit() - 1.0) * magnitude
    }

    /// Choose an element uniformly.
    #[must_use]
    pub fn choose<'t, T>(&mut self, items: &'t [T]) -> Option<&'t T> {
        if items.is_empty() {
            return None;
        }
        Some(&items[self.below(items.len())])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_seed_same_stream() {
        let mut a = SearchRng::new(42);
        let mut b = SearchRng::new(42);
        for _ in 0..64 {
            assert_eq!(a.next_u64(), b.next_u64());
            assert_eq!(a.unit().to_bits(), b.unit().to_bits());
        }
    }

    #[test]
    fn different_seeds_diverge() {
        let mut a = SearchRng::new(1);
        let mut b = SearchRng::new(2);
        let different = (0..16).any(|_| a.next_u64() != b.next_u64());
        assert!(different);
    }

    #[test]
    fn below_stays_in_range_and_handles_degenerate_bounds() {
        let mut rng = SearchRng::new(7);
        for _ in 0..512 {
            let value = rng.below(10);
            assert!(value < 10);
        }
        assert_eq!(rng.below(0), 0);
        assert_eq!(rng.below(1), 0);
    }

    #[test]
    fn signed_magnitude_is_bounded_and_finite() {
        let mut rng = SearchRng::new(9);
        for _ in 0..512 {
            let value = rng.signed_magnitude(3.5);
            assert!(value.is_finite());
            assert!((-3.5..=3.5).contains(&value));
        }
        // A degenerate magnitude degrades to 1.0, so results stay finite and
        // bounded even for NaN input.
        let degraded = SearchRng::new(1).signed_magnitude(f64::NAN);
        assert!(degraded.is_finite() && degraded.abs() <= 1.0);
    }
}
