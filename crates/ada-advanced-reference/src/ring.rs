//! Numerically stable distributed/ring softmax reduction simulator.
//!
//! The simulator models exact shard-local online-softmax states and their
//! associative merge rule. It performs no networking and makes no collective or
//! hardware-performance claim.

use crate::{AdvancedReferenceError, check_len, checked_product, ensure_finite};

/// One logical KV shard in a distributed softmax reduction.
#[derive(Debug, Clone, PartialEq)]
pub struct RingShard {
    /// Local score vector.
    pub scores: Vec<f64>,
    /// Row-major local values `[local_key][value_dimension]`.
    pub values: Vec<f64>,
}

/// Final merged online-softmax state and normalized output.
#[derive(Debug, Clone, PartialEq)]
pub struct RingSoftmaxOutput {
    maximum: f64,
    normalizer: f64,
    numerator: Vec<f64>,
    output: Vec<f64>,
}

impl RingSoftmaxOutput {
    /// Global maximum score.
    #[must_use]
    pub const fn maximum(&self) -> f64 {
        self.maximum
    }

    /// Global shifted softmax normalizer.
    #[must_use]
    pub const fn normalizer(&self) -> f64 {
        self.normalizer
    }

    /// Global shifted unnormalized value numerator.
    #[must_use]
    pub fn numerator(&self) -> &[f64] {
        &self.numerator
    }

    /// Normalized softmax-weighted value output.
    #[must_use]
    pub fn output(&self) -> &[f64] {
        &self.output
    }
}

/// Merge arbitrary non-empty score/value shards with stable online softmax.
///
/// Shard boundaries are execution partitions only. The returned result is the
/// same mathematical softmax reduction over the concatenated logical rows,
/// modulo ordinary IEEE-754 reduction-order rounding.
///
/// # Errors
///
/// Rejects zero value dimension, an empty global key set, shard shape mismatch,
/// excessive bounded tensor size, or non-finite scores/values/intermediates.
pub fn ring_softmax_reduce(
    shards: &[RingShard],
    value_dimension: usize,
) -> Result<RingSoftmaxOutput, AdvancedReferenceError> {
    if value_dimension == 0 {
        return Err(AdvancedReferenceError::InvalidField(
            "ring value dimension",
        ));
    }
    let mut global: Option<OnlineState> = None;
    for shard in shards {
        if shard.scores.is_empty() {
            if !shard.values.is_empty() {
                return Err(AdvancedReferenceError::InvalidField(
                    "empty ring shard has values",
                ));
            }
            continue;
        }
        check_len(
            "ring shard values",
            checked_product(
                shard.scores.len(),
                value_dimension,
                "ring shard values",
            )?,
            shard.values.len(),
        )?;
        ensure_finite(&shard.scores, "ring scores")?;
        ensure_finite(&shard.values, "ring values")?;
        let local = local_state(shard, value_dimension)?;
        global = Some(match global {
            None => local,
            Some(previous) => previous.merge(local)?,
        });
    }
    let global = global.ok_or(AdvancedReferenceError::InvalidField(
        "empty global ring key set",
    ))?;
    if !global.normalizer.is_finite() || global.normalizer <= 0.0 {
        return Err(AdvancedReferenceError::NonFinite(
            "ring global normalizer",
        ));
    }
    let output = global
        .numerator
        .iter()
        .map(|value| value / global.normalizer)
        .collect::<Vec<_>>();
    ensure_finite(&output, "ring output")?;
    Ok(RingSoftmaxOutput {
        maximum: global.maximum,
        normalizer: global.normalizer,
        numerator: global.numerator,
        output,
    })
}

#[derive(Debug, Clone, PartialEq)]
struct OnlineState {
    maximum: f64,
    normalizer: f64,
    numerator: Vec<f64>,
}

impl OnlineState {
    fn merge(self, other: Self) -> Result<Self, AdvancedReferenceError> {
        let maximum = self.maximum.max(other.maximum);
        let left_scale = (self.maximum - maximum).exp();
        let right_scale = (other.maximum - maximum).exp();
        let normalizer = self.normalizer * left_scale + other.normalizer * right_scale;
        let numerator = self
            .numerator
            .into_iter()
            .zip(other.numerator)
            .map(|(left, right)| left * left_scale + right * right_scale)
            .collect::<Vec<_>>();
        if !maximum.is_finite()
            || !normalizer.is_finite()
            || numerator.iter().any(|value| !value.is_finite())
        {
            return Err(AdvancedReferenceError::NonFinite("ring state merge"));
        }
        Ok(Self {
            maximum,
            normalizer,
            numerator,
        })
    }
}

fn local_state(
    shard: &RingShard,
    value_dimension: usize,
) -> Result<OnlineState, AdvancedReferenceError> {
    let maximum = shard
        .scores
        .iter()
        .copied()
        .max_by(f64::total_cmp)
        .ok_or(AdvancedReferenceError::InvalidField("empty local shard"))?;
    let mut normalizer = 0.0;
    let mut numerator = vec![0.0; value_dimension];
    for (key, &score) in shard.scores.iter().enumerate() {
        let weight = (score - maximum).exp();
        normalizer += weight;
        let value_start = key * value_dimension;
        for dimension in 0..value_dimension {
            numerator[dimension] += weight * shard.values[value_start + dimension];
        }
    }
    if !normalizer.is_finite()
        || normalizer <= 0.0
        || numerator.iter().any(|value| !value.is_finite())
    {
        return Err(AdvancedReferenceError::NonFinite("ring local state"));
    }
    Ok(OnlineState {
        maximum,
        normalizer,
        numerator,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ring_partitions_match_single_shard_softmax() {
        let single = ring_softmax_reduce(
            &[RingShard {
                scores: vec![1.0, 2.0, 3.0],
                values: vec![10.0, 20.0, 40.0],
            }],
            1,
        )
        .unwrap();
        let partitioned = ring_softmax_reduce(
            &[
                RingShard {
                    scores: vec![1.0],
                    values: vec![10.0],
                },
                RingShard {
                    scores: vec![2.0, 3.0],
                    values: vec![20.0, 40.0],
                },
            ],
            1,
        )
        .unwrap();
        assert!((single.output()[0] - partitioned.output()[0]).abs() < 1.0e-12);
    }

    #[test]
    fn ring_order_changes_only_rounding_not_mathematical_result() {
        let first = RingShard {
            scores: vec![1000.0, 999.0],
            values: vec![1.0, 2.0],
        };
        let second = RingShard {
            scores: vec![-1000.0, 998.0],
            values: vec![3.0, 4.0],
        };
        let forward = ring_softmax_reduce(&[first.clone(), second.clone()], 1).unwrap();
        let reverse = ring_softmax_reduce(&[second, first], 1).unwrap();
        assert!((forward.output()[0] - reverse.output()[0]).abs() < 1.0e-12);
        assert_eq!(forward.maximum(), 1000.0);
    }

    #[test]
    fn all_empty_shards_fail_closed() {
        assert_eq!(
            ring_softmax_reduce(
                &[RingShard {
                    scores: Vec::new(),
                    values: Vec::new(),
                }],
                1,
            ),
            Err(AdvancedReferenceError::InvalidField(
                "empty global ring key set"
            ))
        );
    }
}
