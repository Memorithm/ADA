#![forbid(unsafe_code)]

mod semantic;

pub use semantic::{
    DiagnosticEvidenceKind, DiagnosticEvidenceRef, FlatGraduationRecord, ImplementationCandidateId,
    MaskContract, QualificationVerdict, SemanticContractError, SemanticDescriptor, SemanticFamily,
    SemanticId, StateContract, WeightContract,
};

/// Logical algorithmic work. These fields are deliberately not hardware
/// instruction or bandwidth counters.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct LogicalMetrics {
    pub qk_pairs_evaluated: usize,
    pub exp_evaluations: usize,
    pub log_evaluations: usize,
    pub value_accumulate_elements: usize,
}

impl LogicalMetrics {
    #[must_use]
    pub const fn total_transcendentals(self) -> usize {
        self.exp_evaluations + self.log_evaluations
    }
}

/// One single-query attention result used by the initial ADA-A1 laboratory.
#[derive(Debug, Clone, PartialEq)]
pub struct AttentionResult {
    pub output: Vec<f32>,
    pub lse: f32,
    pub metrics: LogicalMetrics,
}

/// A deterministic single-query case. `values` is row-major `[seq_len, head_dim]`.
#[derive(Debug, Clone, PartialEq)]
pub struct AttentionCase {
    pub logits: Vec<f32>,
    pub values: Vec<f32>,
    pub head_dim: usize,
}

impl AttentionCase {
    /// Validate the structural and numerical preconditions of an ADA-A1 case.
    ///
    /// # Errors
    ///
    /// Returns an error when the case has no logits, `head_dim` is zero,
    /// `values` does not contain exactly `seq_len * head_dim` elements, or any
    /// input value is non-finite.
    pub fn validate(&self) -> Result<(), &'static str> {
        if self.logits.is_empty() {
            return Err("ADA-A1 requires at least one logit");
        }
        if self.head_dim == 0 {
            return Err("head_dim must be non-zero");
        }
        if self.values.len() != self.logits.len() * self.head_dim {
            return Err("values must have seq_len * head_dim elements");
        }
        if self.logits.iter().any(|x| !x.is_finite()) || self.values.iter().any(|x| !x.is_finite())
        {
            return Err("ADA-A1 E0 cases must be finite");
        }
        Ok(())
    }
}

/// Collision-hardened binding between a prebuilt index and the exact key
/// matrix it was derived from.
///
/// Two independently mixed 64-bit lanes (order-sensitive FNV-1a and a
/// rotate-xor-multiply accumulator) plus an explicit element-count sentinel
/// guard against bit corruption, transposed or reshaped matrices, and the
/// prefix-extension collisions a single-lane digest cannot exclude.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KeyFingerprint {
    primary: u64,
    secondary: u64,
    len_sentinel: u64,
}

impl KeyFingerprint {
    /// Fingerprint a flat `f64` buffer exactly, bit-for-bit.
    #[must_use]
    pub fn of_f64_slice(values: &[f64]) -> Self {
        const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
        const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;
        const MIX_MULT: u64 = 0xff51_afd7_ed55_8ccd;

        let mut primary = FNV_OFFSET;
        let mut secondary = FNV_OFFSET;
        for &value in values {
            let bits = value.to_bits();
            primary ^= bits;
            primary = primary.wrapping_mul(FNV_PRIME);
            secondary ^= bits;
            secondary = secondary.rotate_left(27).wrapping_mul(MIX_MULT);
        }

        // Saturating length sentinel: folds the element count into both lanes
        // so prefix extensions and truncations always invalidate the digest.
        let len_sentinel = u64::try_from(values.len()).unwrap_or(u64::MAX);
        primary ^= len_sentinel;
        primary = primary.wrapping_mul(FNV_PRIME);
        secondary = secondary.rotate_left(31) ^ len_sentinel;

        Self {
            primary,
            secondary,
            len_sentinel,
        }
    }

    /// Element count bound into this fingerprint.
    #[must_use]
    pub const fn len_sentinel(self) -> u64 {
        self.len_sentinel
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fingerprint_is_deterministic_and_sensitive() {
        let base = [1.5_f64, -2.25, 3.0e-9, f64::MIN_POSITIVE];
        let reference = KeyFingerprint::of_f64_slice(&base);

        assert_eq!(reference, KeyFingerprint::of_f64_slice(&base));

        // Single-bit corruption in any lane position must invalidate.
        let mut flipped = base;
        flipped[2] = f64::from_bits(base[2].to_bits() ^ 1);
        assert_ne!(reference, KeyFingerprint::of_f64_slice(&flipped));

        // Order sensitivity (FNV-1a is order-dependent by construction).
        let reordered = [base[1], base[0], base[2], base[3]];
        assert_ne!(reference, KeyFingerprint::of_f64_slice(&reordered));
    }

    #[test]
    fn length_sentinel_blocks_extension_and_truncation() {
        let full = [1.0_f64, 2.0, 3.0];
        let reference = KeyFingerprint::of_f64_slice(&full);
        assert_eq!(reference.len_sentinel(), 3);

        // Prefix extension: [1,2] is a prefix of [1,2,3].
        assert_ne!(
            reference,
            KeyFingerprint::of_f64_slice(&full[..2]),
            "prefix must not collide"
        );
        // Truncation from the tail of a longer buffer sharing the digest
        // input order.
        let extended = [full.as_slice(), &[4.0]].concat();
        assert_ne!(reference, KeyFingerprint::of_f64_slice(&extended));

        // Empty slice has a defined, distinct sentinel.
        assert_eq!(KeyFingerprint::of_f64_slice(&[]).len_sentinel(), 0);
    }

    #[test]
    fn reshaped_matrix_does_not_collide() {
        // Row-major [2,2] versus its transpose: identical multisets, same
        // total element count, different flat layout.
        let row_major = [1.0_f64, 2.0, 3.0, 4.0];
        let transposed = [1.0_f64, 3.0, 2.0, 4.0];
        assert_ne!(
            KeyFingerprint::of_f64_slice(&row_major),
            KeyFingerprint::of_f64_slice(&transposed)
        );
    }
}
