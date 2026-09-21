//! Strengthened / exact reference evaluation for the semantic IR.
//!
//! Complements the naive left-to-right f64 evaluator with:
//! 1. Exact `i128` scaled-dot-product affinity when Q/K entries are exact
//!    integers and the affinity scale is an exact non-positive power of two.
//! 2. Exact uniform softmax weights `1/n` when every selected score is
//!    bit-identical (dyadic when `n` is a power of two).
//! 3. Neumaier-compensated accumulation for general softmax mass and value
//!    mixing, with fail-closed non-finite checks.
//!
//! These paths are higher-assurance reference helpers for differential checks.
//! They do not claim usefulness, novelty, or FLAT adoption.

use crate::{
    AffinityRule, NormalizationSummary, ReferenceInput, ReferenceOutput, SemanticIrError,
    SemanticProgram, WeightRule, select_keys, transform_inputs,
};

/// Largest magnitude with exact integer representation in binary64.
const EXACT_F64_INT_MAX: i128 = 1_i128 << 53;
const EXACT_F64_INT_MAX_F: f64 = 9_007_199_254_740_992.0;

/// Failures specific to the exact integer/dyadic affinity path.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExactAffinityError {
    /// A vector entry was non-finite or not an exact integer.
    NonExactInteger,
    /// Intermediate `i128` arithmetic overflowed.
    Overflow,
    /// The dyadic result cannot be represented exactly in finite f64.
    NonExactDyadic,
    /// Affinity scale is not an exact non-positive power of two in range.
    UnsupportedScale,
}

impl From<ExactAffinityError> for SemanticIrError {
    fn from(value: ExactAffinityError) -> Self {
        match value {
            ExactAffinityError::NonExactInteger => {
                Self::InvalidField("exact affinity requires exact integer entries")
            }
            ExactAffinityError::Overflow => Self::Overflow("exact affinity"),
            ExactAffinityError::NonExactDyadic => Self::Overflow("exact affinity dyadic"),
            ExactAffinityError::UnsupportedScale => {
                Self::InvalidField("exact affinity scale must be 2^-k")
            }
        }
    }
}

/// Evaluate the semantic program with strengthened / exact reference arithmetic.
pub(crate) fn evaluate_strengthened(
    program: &SemanticProgram,
    input: &ReferenceInput,
) -> Result<ReferenceOutput, SemanticIrError> {
    program.validate_components()?;
    input.validate()?;
    if let crate::MaskRule::External { .. } = program.mask() {
        if input.external_mask().is_none() {
            return Err(SemanticIrError::MissingExternalMask);
        }
    }
    let (queries, keys) = transform_inputs(program.input_transform(), input)?;
    let mut scores = vec![0.0_f64; input.query_count() * input.key_count()];
    for query in 0..input.query_count() {
        for key in 0..input.key_count() {
            let query_row =
                &queries[query * input.q_dimension()..(query + 1) * input.q_dimension()];
            let key_row = &keys[key * input.q_dimension()..(key + 1) * input.q_dimension()];
            let score = match program.affinity() {
                AffinityRule::ScaledDotProduct { scale } => {
                    match scaled_dot_product_exact(query_row, key_row, scale) {
                        Ok(exact) => exact,
                        Err(
                            ExactAffinityError::NonExactInteger
                            | ExactAffinityError::UnsupportedScale,
                        ) => scaled_dot_product_neumaier(query_row, key_row, scale)?,
                        Err(other) => return Err(other.into()),
                    }
                }
            };
            if !score.is_finite() {
                return Err(SemanticIrError::NonFiniteValue("affinity"));
            }
            scores[query * input.key_count() + key] = score;
        }
    }

    let mut output = vec![0.0_f64; input.query_count() * input.value_dimension()];
    let mut weights = vec![0.0_f64; input.query_count() * input.key_count()];
    let mut normalizations = Vec::with_capacity(input.query_count());
    let mut selected_keys = Vec::with_capacity(input.query_count());
    for query in 0..input.query_count() {
        let selected = select_keys(program, input, &scores, query)?;
        let selected_scores = selected
            .iter()
            .map(|&key| scores[query * input.key_count() + key])
            .collect::<Vec<_>>();
        let (selected_weights, normalization) =
            weights_for_strengthened(program.weight(), &selected_scores)?;
        for (&key, &weight) in selected.iter().zip(&selected_weights) {
            weights[query * input.key_count() + key] = weight;
        }
        let output_row =
            &mut output[query * input.value_dimension()..(query + 1) * input.value_dimension()];
        for (lane, slot) in output_row.iter_mut().enumerate() {
            let mut sum = 0.0_f64;
            let mut compensation = 0.0_f64;
            for (&key, &weight) in selected.iter().zip(&selected_weights) {
                let value = input.values()[key * input.value_dimension() + lane];
                let term = weight * value;
                if !term.is_finite() {
                    return Err(SemanticIrError::NonFiniteValue("value mixing"));
                }
                let (next, next_comp) = neumaier_add(sum, compensation, term);
                sum = next;
                compensation = next_comp;
            }
            let mixed = sum + compensation;
            if !mixed.is_finite() {
                return Err(SemanticIrError::NonFiniteValue("value mixing"));
            }
            *slot = mixed;
        }
        normalizations.push(normalization);
        selected_keys.push(selected);
    }
    Ok(ReferenceOutput {
        output,
        weights,
        normalizations,
        selected_keys,
    })
}

fn weights_for_strengthened(
    rule: WeightRule,
    scores: &[f64],
) -> Result<(Vec<f64>, NormalizationSummary), SemanticIrError> {
    match rule {
        WeightRule::Softmax => {
            let (weights, lse) = stable_softmax_strengthened(scores, 1.0)?;
            Ok((weights, NormalizationSummary::Softmax { log_sum_exp: lse }))
        }
        WeightRule::SignedDifference {
            positive_scale,
            negative_scale,
        } => {
            let (positive, positive_lse) = stable_softmax_strengthened(scores, positive_scale)?;
            let (negative, negative_lse) = stable_softmax_strengthened(scores, negative_scale)?;
            let weights = positive
                .into_iter()
                .zip(negative)
                .map(|(positive, negative)| positive - negative)
                .collect::<Vec<_>>();
            Ok((
                weights,
                NormalizationSummary::SignedDifference {
                    positive_log_sum_exp: positive_lse,
                    negative_log_sum_exp: negative_lse,
                },
            ))
        }
    }
}

/// Exact uniform weight `1/n` when representable as a finite f64.
///
/// # Errors
///
/// Fails closed when `count` is zero, the count exceeds the exact binary64
/// integer range, or `1/count` is not finite.
pub fn exact_uniform_weight(count: usize) -> Result<f64, SemanticIrError> {
    if count == 0 {
        return Err(SemanticIrError::InvalidField("empty score selection"));
    }
    let count_bits = u64::try_from(count)
        .map_err(|_| SemanticIrError::Overflow("exact uniform weight count"))?;
    if i128::from(count_bits) > EXACT_F64_INT_MAX {
        return Err(SemanticIrError::Overflow("exact uniform weight count"));
    }
    let as_i64 = i64::try_from(count_bits)
        .map_err(|_| SemanticIrError::Overflow("exact uniform weight count"))?;
    let weight = 1.0_f64 / i64_to_f64_exact(as_i64);
    if !weight.is_finite() {
        return Err(SemanticIrError::NonFiniteValue("exact uniform weight"));
    }
    Ok(weight)
}

fn all_scores_bit_equal(scores: &[f64]) -> bool {
    let Some(first) = scores.first() else {
        return true;
    };
    let bits = first.to_bits();
    scores.iter().all(|score| score.to_bits() == bits)
}

fn stable_softmax_strengthened(
    scores: &[f64],
    scale: f64,
) -> Result<(Vec<f64>, f64), SemanticIrError> {
    if scores.is_empty() {
        return Err(SemanticIrError::InvalidField("empty score selection"));
    }
    if !scale.is_finite() {
        return Err(SemanticIrError::NonFiniteValue("softmax scale"));
    }
    if all_scores_bit_equal(scores) {
        let weight = exact_uniform_weight(scores.len())?;
        let maximum = scores[0];
        if !maximum.is_finite() {
            return Err(SemanticIrError::NonFiniteValue("softmax maximum"));
        }
        let scaled_maximum = maximum * scale;
        if !scaled_maximum.is_finite() {
            return Err(SemanticIrError::NonFiniteValue("softmax scale"));
        }
        // For equal scores, LSE = scale * score + ln(n).
        let count =
            u64::try_from(scores.len()).map_err(|_| SemanticIrError::Overflow("softmax count"))?;
        if i128::from(count) > EXACT_F64_INT_MAX {
            return Err(SemanticIrError::Overflow("softmax count"));
        }
        let count_f64 = i64_to_f64_exact(
            i64::try_from(count).map_err(|_| SemanticIrError::Overflow("softmax count"))?,
        );
        let log_sum_exp = scaled_maximum + count_f64.ln();
        if !log_sum_exp.is_finite() {
            return Err(SemanticIrError::NonFiniteValue("softmax log-sum-exp"));
        }
        return Ok((vec![weight; scores.len()], log_sum_exp));
    }

    let maximum = scores
        .iter()
        .copied()
        .max_by(f64::total_cmp)
        .ok_or(SemanticIrError::InvalidField("empty score selection"))?;
    if !maximum.is_finite() {
        return Err(SemanticIrError::NonFiniteValue("softmax maximum"));
    }
    let scaled_maximum = maximum * scale;
    if !scaled_maximum.is_finite() {
        return Err(SemanticIrError::NonFiniteValue("softmax scale"));
    }
    let mut unnormalized = Vec::with_capacity(scores.len());
    for &score in scores {
        let scaled = score * scale;
        if !scaled.is_finite() {
            return Err(SemanticIrError::NonFiniteValue("softmax scale"));
        }
        let weight = (scaled - scaled_maximum).exp();
        if !weight.is_finite() {
            return Err(SemanticIrError::NonFiniteValue("softmax exponential"));
        }
        unnormalized.push(weight);
    }
    let (sum, compensation) = neumaier_sum(unnormalized.iter().copied());
    let mass = sum + compensation;
    if !mass.is_finite() || mass <= 0.0 {
        return Err(SemanticIrError::NonFiniteValue("softmax normalizer"));
    }
    let log_sum_exp = scaled_maximum + mass.ln();
    if !log_sum_exp.is_finite() {
        return Err(SemanticIrError::NonFiniteValue("softmax log-sum-exp"));
    }
    let mut weights = Vec::with_capacity(unnormalized.len());
    for value in unnormalized {
        let weight = value / mass;
        if !weight.is_finite() {
            return Err(SemanticIrError::NonFiniteValue("softmax weight"));
        }
        weights.push(weight);
    }
    Ok((weights, log_sum_exp))
}

/// Exact scaled dot product for integer Q/K and dyadic scale `2^-shift`.
///
/// # Errors
///
/// Returns [`ExactAffinityError`] when shapes disagree, entries are not exact
/// integers, the scale is not an exact non-positive power of two, intermediate
/// `i128` arithmetic overflows, or the dyadic result is not exact in f64.
pub fn scaled_dot_product_exact(
    query: &[f64],
    key: &[f64],
    scale: f64,
) -> Result<f64, ExactAffinityError> {
    if query.len() != key.len() {
        return Err(ExactAffinityError::Overflow);
    }
    let shift = dyadic_scale_shift(scale)?;
    let mut acc = 0_i128;
    for (&q, &k) in query.iter().zip(key) {
        let q_i = f64_exact_i64(q)?;
        let k_i = f64_exact_i64(k)?;
        let term = i128::from(q_i)
            .checked_mul(i128::from(k_i))
            .ok_or(ExactAffinityError::Overflow)?;
        acc = acc.checked_add(term).ok_or(ExactAffinityError::Overflow)?;
    }
    dyadic_i128_to_f64(acc, shift)
}

fn scaled_dot_product_neumaier(
    query: &[f64],
    key: &[f64],
    scale: f64,
) -> Result<f64, SemanticIrError> {
    if !scale.is_finite() {
        return Err(SemanticIrError::NonFiniteValue("affinity scale"));
    }
    let mut sum = 0.0_f64;
    let mut compensation = 0.0_f64;
    for (&q, &k) in query.iter().zip(key) {
        let term = q * k;
        if !term.is_finite() {
            return Err(SemanticIrError::NonFiniteValue("affinity"));
        }
        let (next, next_comp) = neumaier_add(sum, compensation, term);
        sum = next;
        compensation = next_comp;
    }
    let dot = sum + compensation;
    let score = dot * scale;
    if !score.is_finite() {
        return Err(SemanticIrError::NonFiniteValue("affinity"));
    }
    Ok(score)
}

fn dyadic_scale_shift(scale: f64) -> Result<u32, ExactAffinityError> {
    if !scale.is_finite() || scale <= 0.0 {
        return Err(ExactAffinityError::UnsupportedScale);
    }
    // Accept scale = 2^-k for k in 0..=62 by repeated doubling to 1.
    let mut value = scale;
    for shift in 0..=62 {
        if value.to_bits() == 1.0_f64.to_bits() {
            return Ok(shift);
        }
        value *= 2.0;
        if !value.is_finite() {
            break;
        }
    }
    Err(ExactAffinityError::UnsupportedScale)
}

fn f64_exact_i64(value: f64) -> Result<i64, ExactAffinityError> {
    if !value.is_finite() || value.fract() != 0.0 {
        return Err(ExactAffinityError::NonExactInteger);
    }
    if value.abs() > EXACT_F64_INT_MAX_F {
        return Err(ExactAffinityError::NonExactInteger);
    }
    #[allow(clippy::cast_possible_truncation)]
    let as_i64 = value as i64;
    if i64_to_f64_exact(as_i64).to_bits() != value.to_bits() {
        return Err(ExactAffinityError::NonExactInteger);
    }
    Ok(as_i64)
}

fn dyadic_i128_to_f64(numerator: i128, denominator_shift: u32) -> Result<f64, ExactAffinityError> {
    let mut value = i128_to_exact_f64(numerator)?;
    for _ in 0..denominator_shift {
        value *= 0.5;
        if !value.is_finite() {
            return Err(ExactAffinityError::NonExactDyadic);
        }
    }
    Ok(value)
}

fn i128_to_exact_f64(value: i128) -> Result<f64, ExactAffinityError> {
    if value.abs() > EXACT_F64_INT_MAX {
        return Err(ExactAffinityError::Overflow);
    }
    let as_i64 = i64::try_from(value).map_err(|_| ExactAffinityError::Overflow)?;
    Ok(i64_to_f64_exact(as_i64))
}

fn i64_to_f64_exact(value: i64) -> f64 {
    // Safe: callers restrict to |value| <= 2^53.
    #[allow(clippy::cast_precision_loss)]
    {
        value as f64
    }
}

fn neumaier_sum(values: impl Iterator<Item = f64>) -> (f64, f64) {
    let mut sum = 0.0_f64;
    let mut compensation = 0.0_f64;
    for value in values {
        let (next, next_comp) = neumaier_add(sum, compensation, value);
        sum = next;
        compensation = next_comp;
    }
    (sum, compensation)
}

fn neumaier_add(sum: f64, compensation: f64, value: f64) -> (f64, f64) {
    let corrected = value - compensation;
    let next = sum + corrected;
    let next_compensation = (next - sum) - corrected;
    (next, next_compensation)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{MaskRule, ReferenceInput, ReferenceInputSpec, SelectionRule, SemanticProgram};
    use ada_core::{SemanticFamily, SemanticId};

    fn semantic_id(name: &str) -> SemanticId {
        SemanticId::new(SemanticFamily::StandardSoftmax, name, 1)
            .expect("test semantic identity is valid")
    }

    fn program(scale: f64) -> SemanticProgram {
        SemanticProgram::standard_softmax(
            semantic_id("strengthened-softmax"),
            MaskRule::Unmasked,
            SelectionRule::All,
            scale,
        )
        .expect("test semantic program is valid")
    }

    fn assert_bits(actual: f64, expected: f64) {
        assert_eq!(
            actual.to_bits(),
            expected.to_bits(),
            "{actual} bits != {expected} bits"
        );
    }

    #[test]
    fn exact_affinity_matches_hand_derived_integer_dyadic_fixture() {
        // Q=[3,4], K=[1,2], scale=1/2 → (3*1 + 4*2)/2 = 11/2 = 5.5
        let score = scaled_dot_product_exact(&[3.0, 4.0], &[1.0, 2.0], 0.5).expect("exact");
        assert_bits(score, 5.5);
        // Orthogonal integer pair with scale=1 → 0
        let zero = scaled_dot_product_exact(&[1.0, 0.0], &[0.0, 1.0], 1.0).expect("exact");
        assert_bits(zero, 0.0);
    }

    #[test]
    fn exact_affinity_fails_closed_on_non_integer_and_overflow() {
        assert_eq!(
            scaled_dot_product_exact(&[1.5], &[2.0], 1.0),
            Err(ExactAffinityError::NonExactInteger)
        );
        assert_eq!(
            scaled_dot_product_exact(&[1.0], &[1.0], 0.3),
            Err(ExactAffinityError::UnsupportedScale)
        );
        let huge = i64_to_f64_exact(1_i64 << 40);
        assert_eq!(
            scaled_dot_product_exact(&[huge, huge], &[huge, huge], 1.0),
            Err(ExactAffinityError::Overflow)
        );
    }

    #[test]
    fn equal_score_fixture_has_exact_uniform_weights_and_mean_output() {
        // Four identical keys aligned with Q ⇒ equal scores.
        // V = [2,4,6,8], n=4 ⇒ weight=1/4, output mean=5 exactly.
        let input = ReferenceInput::new(ReferenceInputSpec {
            query_count: 1,
            key_count: 4,
            q_dimension: 2,
            value_dimension: 1,
            queries: vec![1.0, 0.0],
            keys: vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
            values: vec![2.0, 4.0, 6.0, 8.0],
            external_mask: None,
        })
        .expect("fixture");
        let output = program(1.0)
            .evaluate_strengthened(&input)
            .expect("strengthened");
        for &weight in output.weights() {
            assert_bits(weight, 0.25);
        }
        assert_bits(output.output()[0], 5.0);
        match output.normalizations()[0] {
            NormalizationSummary::Softmax { log_sum_exp } => {
                // Scores are all 1.0; LSE = 1 + ln(4). Transcendental
                // evaluation is allowed to differ by one ULP under Miri, while
                // the exact uniform weights and mixed output above remain
                // bit-exact.
                let expected = 1.0_f64 + 4.0_f64.ln();
                let ulp_distance = log_sum_exp.to_bits().abs_diff(expected.to_bits());
                assert!(
                    ulp_distance <= 1,
                    "LSE {log_sum_exp} differs from {expected} by {ulp_distance} ULP"
                );
            }
            NormalizationSummary::SignedDifference { .. } => {
                panic!("unexpected signed-difference normalization")
            }
        }
    }

    #[test]
    fn strengthened_path_parity_with_naive_f64_on_shared_fixture() {
        let input = ReferenceInput::new(ReferenceInputSpec {
            query_count: 2,
            key_count: 3,
            q_dimension: 2,
            value_dimension: 1,
            queries: vec![1.0, 0.0, 0.0, 1.0],
            keys: vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            values: vec![10.0, 20.0, 30.0],
            external_mask: None,
        })
        .expect("fixture");
        let naive = program(1.0).evaluate(&input).expect("naive");
        let strengthened = program(1.0)
            .evaluate_strengthened(&input)
            .expect("strengthened");
        assert_eq!(naive.selected_keys(), strengthened.selected_keys());
        for (&left, &right) in naive.output().iter().zip(strengthened.output()) {
            assert!(
                (left - right).abs() <= 4.0 * f64::EPSILON * left.abs().max(1.0),
                "output parity {left} vs {right}"
            );
        }
        for (&left, &right) in naive.weights().iter().zip(strengthened.weights()) {
            assert!(
                (left - right).abs() <= 4.0 * f64::EPSILON,
                "weight parity {left} vs {right}"
            );
        }
    }

    #[test]
    fn exact_two_key_power_of_two_fixture() {
        // Two equal scores ⇒ weight 1/2; V=[8, 2] ⇒ output 5.
        let input = ReferenceInput::new(ReferenceInputSpec {
            query_count: 1,
            key_count: 2,
            q_dimension: 1,
            value_dimension: 1,
            queries: vec![2.0],
            keys: vec![2.0, 2.0],
            values: vec![8.0, 2.0],
            external_mask: None,
        })
        .expect("fixture");
        let output = program(0.5)
            .evaluate_strengthened(&input)
            .expect("strengthened");
        // Affinity uses exact path: scale=1/2, Q=2, K=2 ⇒ score=2 exactly both.
        assert_bits(output.weights()[0], 0.5);
        assert_bits(output.weights()[1], 0.5);
        assert_bits(output.output()[0], 5.0);
    }

    #[test]
    fn non_finite_affinity_fails_closed() {
        // Finite Q/K entries whose product overflows to infinity.
        let input = ReferenceInput::new(ReferenceInputSpec {
            query_count: 1,
            key_count: 1,
            q_dimension: 1,
            value_dimension: 1,
            queries: vec![1.0e200],
            keys: vec![1.0e200],
            values: vec![1.0],
            external_mask: None,
        })
        .expect("finite fixture");
        let result = program(1.0).evaluate_strengthened(&input);
        assert!(
            matches!(result, Err(SemanticIrError::NonFiniteValue("affinity"))),
            "expected fail-closed affinity overflow, got {result:?}"
        );
    }

    #[test]
    fn exact_uniform_weight_fails_closed_on_empty() {
        assert_eq!(
            exact_uniform_weight(0),
            Err(SemanticIrError::InvalidField("empty score selection"))
        );
    }
}
