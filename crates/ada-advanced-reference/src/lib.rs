//! Bounded executable references for attention families that sit outside ADA's
//! original dense semantic evaluator.
//!
//! Each module defines an explicit reference rule and fail-closed input
//! contract. These CPU/f64 implementations are research oracles, not production
//! kernels and not evidence of model quality or hardware performance.

#![forbid(unsafe_code)]

use std::fmt::{Display, Formatter};

pub mod latent;
pub mod recurrent;
pub mod ring;
pub mod sparse;

/// Shared advanced-reference failure.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AdvancedReferenceError {
    /// A workload or rule is outside the bounded reference domain.
    Unsupported(&'static str),
    /// A required dimension or vector length is inconsistent.
    ShapeMismatch {
        /// Rejected field.
        field: &'static str,
        /// Expected element count.
        expected: usize,
        /// Actual element count.
        actual: usize,
    },
    /// A semantic or reconstruction identifier does not match its contract.
    IdentityMismatch(&'static str),
    /// A structural field is invalid.
    InvalidField(&'static str),
    /// A non-finite scalar was observed or produced.
    NonFinite(&'static str),
    /// A bounded allocation or operation count would be excessive.
    ExceedsLimit(&'static str),
    /// A sparse/routed row contains no selectable key.
    EmptySelection(usize),
}

impl Display for AdvancedReferenceError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unsupported(reason) => {
                write!(formatter, "unsupported advanced reference: {reason}")
            }
            Self::ShapeMismatch {
                field,
                expected,
                actual,
            } => write!(
                formatter,
                "{field} has {actual} elements; expected {expected}"
            ),
            Self::IdentityMismatch(field) => write!(formatter, "identity mismatch: {field}"),
            Self::InvalidField(field) => {
                write!(formatter, "invalid advanced-reference field: {field}")
            }
            Self::NonFinite(stage) => {
                write!(formatter, "non-finite advanced-reference value at {stage}")
            }
            Self::ExceedsLimit(field) => {
                write!(formatter, "advanced-reference limit exceeded: {field}")
            }
            Self::EmptySelection(row) => {
                write!(formatter, "attention row {row} has no selected key")
            }
        }
    }
}

impl std::error::Error for AdvancedReferenceError {}

pub(crate) const MAX_ELEMENTS: usize = 1 << 24;

pub(crate) fn checked_product(
    left: usize,
    right: usize,
    field: &'static str,
) -> Result<usize, AdvancedReferenceError> {
    let value = left
        .checked_mul(right)
        .ok_or(AdvancedReferenceError::ExceedsLimit(field))?;
    if value > MAX_ELEMENTS {
        return Err(AdvancedReferenceError::ExceedsLimit(field));
    }
    Ok(value)
}

pub(crate) fn check_len(
    field: &'static str,
    expected: usize,
    actual: usize,
) -> Result<(), AdvancedReferenceError> {
    if expected == actual {
        Ok(())
    } else {
        Err(AdvancedReferenceError::ShapeMismatch {
            field,
            expected,
            actual,
        })
    }
}

pub(crate) fn ensure_finite(
    values: &[f64],
    stage: &'static str,
) -> Result<(), AdvancedReferenceError> {
    if values.iter().all(|value| value.is_finite()) {
        Ok(())
    } else {
        Err(AdvancedReferenceError::NonFinite(stage))
    }
}

pub(crate) fn stable_softmax(scores: &[f64]) -> Result<Vec<f64>, AdvancedReferenceError> {
    let maximum = scores
        .iter()
        .copied()
        .max_by(f64::total_cmp)
        .ok_or(AdvancedReferenceError::InvalidField("empty score row"))?;
    if !maximum.is_finite() {
        return Err(AdvancedReferenceError::NonFinite("softmax maximum"));
    }
    let mut weights = Vec::with_capacity(scores.len());
    let mut normalizer = 0.0_f64;
    for &score in scores {
        let weight = (score - maximum).exp();
        if !weight.is_finite() {
            return Err(AdvancedReferenceError::NonFinite("softmax exponential"));
        }
        normalizer += weight;
        weights.push(weight);
    }
    if !normalizer.is_finite() || normalizer <= 0.0 {
        return Err(AdvancedReferenceError::NonFinite("softmax normalizer"));
    }
    for weight in &mut weights {
        *weight /= normalizer;
    }
    Ok(weights)
}
