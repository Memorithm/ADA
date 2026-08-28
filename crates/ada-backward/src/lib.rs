//! Bounded f64 backward reference and finite-difference verification for ADA.
//!
//! This crate deliberately starts with the smooth dense softmax semantic only:
//! identity Q/K transform, scaled dot-product affinity, unmasked `All`
//! selection, softmax weighting, weighted V sum, and identity output. Top-k,
//! windowed selection, signed weighting, masking, low precision, compressed KV,
//! recurrent state, and backend schedules are rejected rather than assigned an
//! unverified gradient rule.

#![forbid(unsafe_code)]

use ada_semantic::{
    AffinityRule, InputTransform, MaskRule, OutputRule, ReferenceInput, ReferenceInputSpec,
    SelectionRule, SemanticIrError, SemanticProgram, ValueMixRule, WeightRule,
};
use std::fmt::{Display, Formatter};

/// Maximum scalar Q/K/V entries that the finite-difference checker perturbs.
pub const MAX_FINITE_DIFFERENCE_VARIABLES: usize = 16_384;

/// Backward-reference failures.
#[derive(Debug, Clone, PartialEq)]
pub enum BackwardError {
    /// The semantic is outside the differentiable reference subset.
    UnsupportedSemantic(&'static str),
    /// One tensor has the wrong number of elements.
    ShapeMismatch {
        /// Tensor name.
        field: &'static str,
        /// Required element count.
        expected: usize,
        /// Supplied element count.
        actual: usize,
    },
    /// A scalar input, intermediate, or gradient became non-finite.
    NonFinite(&'static str),
    /// A dimension product overflowed.
    ArithmeticOverflow(&'static str),
    /// Finite-difference configuration is invalid.
    InvalidFiniteDifferenceConfig(&'static str),
    /// The requested checker would exceed its explicit perturbation budget.
    FiniteDifferenceBudgetExceeded {
        /// Number of scalar variables that would be perturbed.
        variables: usize,
        /// Inclusive configured maximum.
        maximum: usize,
    },
    /// The existing semantic evaluator rejected the reference input.
    Semantic(String),
}

impl Display for BackwardError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnsupportedSemantic(reason) => {
                write!(formatter, "unsupported backward semantic: {reason}")
            }
            Self::ShapeMismatch {
                field,
                expected,
                actual,
            } => write!(
                formatter,
                "{field} has {actual} elements; expected {expected}"
            ),
            Self::NonFinite(field) => write!(formatter, "non-finite backward value: {field}"),
            Self::ArithmeticOverflow(field) => {
                write!(formatter, "backward shape overflow: {field}")
            }
            Self::InvalidFiniteDifferenceConfig(field) => {
                write!(
                    formatter,
                    "invalid finite-difference configuration: {field}"
                )
            }
            Self::FiniteDifferenceBudgetExceeded { variables, maximum } => write!(
                formatter,
                "finite-difference variable count {variables} exceeds maximum {maximum}"
            ),
            Self::Semantic(reason) => write!(formatter, "semantic reference error: {reason}"),
        }
    }
}

impl std::error::Error for BackwardError {}

impl From<SemanticIrError> for BackwardError {
    fn from(value: SemanticIrError) -> Self {
        Self::Semantic(value.to_string())
    }
}

/// Explicit dense Q/K/V values plus an upstream gradient for one reference run.
#[derive(Debug, Clone, PartialEq)]
pub struct BackwardInput {
    /// Query row count.
    pub query_count: usize,
    /// Key/value row count.
    pub key_count: usize,
    /// Q/K feature dimension.
    pub q_dimension: usize,
    /// V/output feature dimension.
    pub value_dimension: usize,
    /// Row-major Q values.
    pub queries: Vec<f64>,
    /// Row-major K values.
    pub keys: Vec<f64>,
    /// Row-major V values.
    pub values: Vec<f64>,
    /// Row-major gradient of a scalar loss with respect to the attention output.
    pub output_gradient: Vec<f64>,
}

impl BackwardInput {
    fn validate(&self) -> Result<(), BackwardError> {
        let q_elements = checked_product(self.query_count, self.q_dimension, "queries")?;
        let k_elements = checked_product(self.key_count, self.q_dimension, "keys")?;
        let v_elements = checked_product(self.key_count, self.value_dimension, "values")?;
        let output_elements =
            checked_product(self.query_count, self.value_dimension, "output_gradient")?;
        if self.query_count == 0
            || self.key_count == 0
            || self.q_dimension == 0
            || self.value_dimension == 0
        {
            return Err(BackwardError::ShapeMismatch {
                field: "zero dimension",
                expected: 1,
                actual: 0,
            });
        }
        check_len("queries", q_elements, self.queries.len())?;
        check_len("keys", k_elements, self.keys.len())?;
        check_len("values", v_elements, self.values.len())?;
        check_len(
            "output_gradient",
            output_elements,
            self.output_gradient.len(),
        )?;
        if self
            .queries
            .iter()
            .chain(&self.keys)
            .chain(&self.values)
            .chain(&self.output_gradient)
            .any(|value| !value.is_finite())
        {
            return Err(BackwardError::NonFinite("input"));
        }
        Ok(())
    }

    fn reference_input(&self) -> Result<ReferenceInput, BackwardError> {
        Ok(ReferenceInput::new(ReferenceInputSpec {
            query_count: self.query_count,
            key_count: self.key_count,
            q_dimension: self.q_dimension,
            value_dimension: self.value_dimension,
            queries: self.queries.clone(),
            keys: self.keys.clone(),
            values: self.values.clone(),
            external_mask: None,
        })?)
    }

    fn variable_count(&self) -> Result<usize, BackwardError> {
        self.queries
            .len()
            .checked_add(self.keys.len())
            .and_then(|value| value.checked_add(self.values.len()))
            .ok_or(BackwardError::ArithmeticOverflow("variable_count"))
    }
}

/// Analytical gradients with the same row-major shapes as Q/K/V.
#[derive(Debug, Clone, PartialEq)]
pub struct AttentionGradients {
    /// Gradient with respect to Q.
    pub queries: Vec<f64>,
    /// Gradient with respect to K.
    pub keys: Vec<f64>,
    /// Gradient with respect to V.
    pub values: Vec<f64>,
}

/// Configuration for bounded central finite differences.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FiniteDifferenceConfig {
    /// Symmetric perturbation magnitude.
    pub epsilon: f64,
    /// Absolute error allowed before a component is considered mismatched.
    pub absolute_tolerance: f64,
    /// Relative error allowed before a component is considered mismatched.
    pub relative_tolerance: f64,
    /// Maximum number of Q/K/V scalar variables to perturb.
    pub max_variables: usize,
}

impl Default for FiniteDifferenceConfig {
    fn default() -> Self {
        Self {
            epsilon: 1.0e-6,
            absolute_tolerance: 2.0e-7,
            relative_tolerance: 2.0e-6,
            max_variables: MAX_FINITE_DIFFERENCE_VARIABLES,
        }
    }
}

impl FiniteDifferenceConfig {
    fn validate(self) -> Result<(), BackwardError> {
        if !self.epsilon.is_finite() || self.epsilon <= 0.0 {
            return Err(BackwardError::InvalidFiniteDifferenceConfig("epsilon"));
        }
        if !self.absolute_tolerance.is_finite() || self.absolute_tolerance < 0.0 {
            return Err(BackwardError::InvalidFiniteDifferenceConfig(
                "absolute_tolerance",
            ));
        }
        if !self.relative_tolerance.is_finite() || self.relative_tolerance < 0.0 {
            return Err(BackwardError::InvalidFiniteDifferenceConfig(
                "relative_tolerance",
            ));
        }
        if self.max_variables == 0 || self.max_variables > MAX_FINITE_DIFFERENCE_VARIABLES {
            return Err(BackwardError::InvalidFiniteDifferenceConfig(
                "max_variables",
            ));
        }
        Ok(())
    }
}

/// Tensor containing the worst finite-difference discrepancy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GradientTensor {
    /// Q gradient.
    Query,
    /// K gradient.
    Key,
    /// V gradient.
    Value,
}

/// Result of comparing the analytical gradient to central finite differences.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GradientCheckReport {
    /// Number of scalar variables checked.
    pub variables_checked: usize,
    /// Largest absolute discrepancy.
    pub max_absolute_error: f64,
    /// Largest relative discrepancy using `max(|analytic|, |numeric|, 1)`.
    pub max_relative_error: f64,
    /// Tensor containing the largest absolute discrepancy.
    pub worst_tensor: GradientTensor,
    /// Flat index within `worst_tensor`.
    pub worst_index: usize,
    /// Whether every component satisfied absolute or relative tolerance.
    pub passed: bool,
}

/// Compute the analytical dense-softmax gradients for Q, K, and V.
///
/// # Errors
///
/// Rejects malformed/non-finite inputs and any semantic outside the deliberately
/// narrow smooth reference subset documented by this crate.
pub fn backward(
    program: &SemanticProgram,
    input: &BackwardInput,
) -> Result<AttentionGradients, BackwardError> {
    input.validate()?;
    let scale = validate_program(program)?;
    let reference = input.reference_input()?;
    let forward = program.evaluate(&reference)?;
    let weights = forward.weights();
    let expected_weights = checked_product(input.query_count, input.key_count, "weights")?;
    check_len("weights", expected_weights, weights.len())?;

    let mut d_queries = vec![0.0; input.queries.len()];
    let mut d_keys = vec![0.0; input.keys.len()];
    let mut d_values = vec![0.0; input.values.len()];
    let mut d_probabilities = vec![0.0; weights.len()];

    for query in 0..input.query_count {
        let upstream = &input.output_gradient
            [query * input.value_dimension..(query + 1) * input.value_dimension];
        for key in 0..input.key_count {
            let probability = weights[query * input.key_count + key];
            let value_row =
                &input.values[key * input.value_dimension..(key + 1) * input.value_dimension];
            let mut d_probability = 0.0;
            for dimension in 0..input.value_dimension {
                d_probability += upstream[dimension] * value_row[dimension];
                d_values[key * input.value_dimension + dimension] +=
                    probability * upstream[dimension];
            }
            d_probabilities[query * input.key_count + key] = d_probability;
        }
    }

    for query in 0..input.query_count {
        let row = query * input.key_count;
        let mut probability_dot = 0.0;
        for key in 0..input.key_count {
            probability_dot += weights[row + key] * d_probabilities[row + key];
        }
        for key in 0..input.key_count {
            let d_score = weights[row + key] * (d_probabilities[row + key] - probability_dot);
            let query_offset = query * input.q_dimension;
            let key_offset = key * input.q_dimension;
            for dimension in 0..input.q_dimension {
                d_queries[query_offset + dimension] +=
                    scale * d_score * input.keys[key_offset + dimension];
                d_keys[key_offset + dimension] +=
                    scale * d_score * input.queries[query_offset + dimension];
            }
        }
    }

    if d_queries
        .iter()
        .chain(&d_keys)
        .chain(&d_values)
        .any(|value| !value.is_finite())
    {
        return Err(BackwardError::NonFinite("analytical gradient"));
    }
    Ok(AttentionGradients {
        queries: d_queries,
        keys: d_keys,
        values: d_values,
    })
}

/// Compare the analytical backward rule with bounded central finite differences.
///
/// The scalar objective is the dot product between the attention output and the
/// supplied `output_gradient`, so the numerical derivative targets exactly the
/// vector-Jacobian product returned by [`backward`].
///
/// # Errors
///
/// Returns an error for unsupported semantics, malformed inputs, invalid checker
/// configuration, non-finite evaluations, or a perturbation budget violation.
pub fn check_finite_difference(
    program: &SemanticProgram,
    input: &BackwardInput,
    config: FiniteDifferenceConfig,
) -> Result<GradientCheckReport, BackwardError> {
    config.validate()?;
    input.validate()?;
    validate_program(program)?;
    let variable_count = input.variable_count()?;
    if variable_count > config.max_variables {
        return Err(BackwardError::FiniteDifferenceBudgetExceeded {
            variables: variable_count,
            maximum: config.max_variables,
        });
    }
    let analytical = backward(program, input)?;
    let numeric_queries =
        numerical_gradient(program, input, GradientTensor::Query, config.epsilon)?;
    let numeric_keys = numerical_gradient(program, input, GradientTensor::Key, config.epsilon)?;
    let numeric_values = numerical_gradient(program, input, GradientTensor::Value, config.epsilon)?;

    let mut report = GradientCheckReport {
        variables_checked: variable_count,
        max_absolute_error: 0.0,
        max_relative_error: 0.0,
        worst_tensor: GradientTensor::Query,
        worst_index: 0,
        passed: true,
    };
    compare_gradient(
        &mut report,
        GradientTensor::Query,
        &analytical.queries,
        &numeric_queries,
        config,
    );
    compare_gradient(
        &mut report,
        GradientTensor::Key,
        &analytical.keys,
        &numeric_keys,
        config,
    );
    compare_gradient(
        &mut report,
        GradientTensor::Value,
        &analytical.values,
        &numeric_values,
        config,
    );
    Ok(report)
}

fn validate_program(program: &SemanticProgram) -> Result<f64, BackwardError> {
    if program.input_transform() != InputTransform::Identity {
        return Err(BackwardError::UnsupportedSemantic("input transform"));
    }
    if !matches!(program.mask(), MaskRule::Unmasked) {
        return Err(BackwardError::UnsupportedSemantic("masking"));
    }
    if program.selection() != SelectionRule::All {
        return Err(BackwardError::UnsupportedSemantic("selection"));
    }
    if program.weight() != WeightRule::Softmax {
        return Err(BackwardError::UnsupportedSemantic("weighting"));
    }
    if program.value_mix() != ValueMixRule::WeightedSum || program.output() != OutputRule::Identity
    {
        return Err(BackwardError::UnsupportedSemantic("value/output rule"));
    }
    match program.affinity() {
        AffinityRule::ScaledDotProduct { scale } if scale.is_finite() && scale > 0.0 => Ok(scale),
        AffinityRule::ScaledDotProduct { .. } => {
            Err(BackwardError::UnsupportedSemantic("affinity scale"))
        }
    }
}

fn numerical_gradient(
    program: &SemanticProgram,
    input: &BackwardInput,
    tensor: GradientTensor,
    epsilon: f64,
) -> Result<Vec<f64>, BackwardError> {
    let length = match tensor {
        GradientTensor::Query => input.queries.len(),
        GradientTensor::Key => input.keys.len(),
        GradientTensor::Value => input.values.len(),
    };
    let mut gradient = Vec::with_capacity(length);
    for index in 0..length {
        let mut positive = input.clone();
        let mut negative = input.clone();
        tensor_slice_mut(&mut positive, tensor)[index] += epsilon;
        tensor_slice_mut(&mut negative, tensor)[index] -= epsilon;
        let positive_loss = scalar_loss(program, &positive)?;
        let negative_loss = scalar_loss(program, &negative)?;
        let derivative = (positive_loss - negative_loss) / (2.0 * epsilon);
        if !derivative.is_finite() {
            return Err(BackwardError::NonFinite("finite difference"));
        }
        gradient.push(derivative);
    }
    Ok(gradient)
}

fn tensor_slice_mut(input: &mut BackwardInput, tensor: GradientTensor) -> &mut [f64] {
    match tensor {
        GradientTensor::Query => &mut input.queries,
        GradientTensor::Key => &mut input.keys,
        GradientTensor::Value => &mut input.values,
    }
}

fn scalar_loss(program: &SemanticProgram, input: &BackwardInput) -> Result<f64, BackwardError> {
    let output = program.evaluate(&input.reference_input()?)?;
    let values = output.output();
    check_len("output", input.output_gradient.len(), values.len())?;
    let loss = values
        .iter()
        .zip(&input.output_gradient)
        .map(|(&value, &gradient)| value * gradient)
        .sum::<f64>();
    if loss.is_finite() {
        Ok(loss)
    } else {
        Err(BackwardError::NonFinite("finite-difference loss"))
    }
}

fn compare_gradient(
    report: &mut GradientCheckReport,
    tensor: GradientTensor,
    analytical: &[f64],
    numerical: &[f64],
    config: FiniteDifferenceConfig,
) {
    for (index, (&analytic, &numeric)) in analytical.iter().zip(numerical).enumerate() {
        let absolute = (analytic - numeric).abs();
        let relative = absolute / analytic.abs().max(numeric.abs()).max(1.0);
        if absolute > report.max_absolute_error {
            report.max_absolute_error = absolute;
            report.worst_tensor = tensor;
            report.worst_index = index;
        }
        report.max_relative_error = report.max_relative_error.max(relative);
        if absolute > config.absolute_tolerance && relative > config.relative_tolerance {
            report.passed = false;
        }
    }
}

fn checked_product(left: usize, right: usize, field: &'static str) -> Result<usize, BackwardError> {
    left.checked_mul(right)
        .ok_or(BackwardError::ArithmeticOverflow(field))
}

fn check_len(field: &'static str, expected: usize, actual: usize) -> Result<(), BackwardError> {
    if expected == actual {
        Ok(())
    } else {
        Err(BackwardError::ShapeMismatch {
            field,
            expected,
            actual,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ada_core::{SemanticFamily, SemanticId};
    use ada_semantic::SemanticProgram;

    fn program() -> SemanticProgram {
        SemanticProgram::standard_softmax(
            SemanticId::new(SemanticFamily::StandardSoftmax, "backward-reference", 1).unwrap(),
            MaskRule::Unmasked,
            SelectionRule::All,
            0.5,
        )
        .unwrap()
    }

    fn case() -> BackwardInput {
        BackwardInput {
            query_count: 2,
            key_count: 3,
            q_dimension: 2,
            value_dimension: 2,
            queries: vec![0.2, -0.4, 0.7, 0.1],
            keys: vec![0.5, -0.3, -0.2, 0.8, 0.4, 0.6],
            values: vec![1.0, -0.5, 0.25, 0.75, -0.6, 0.4],
            output_gradient: vec![0.3, -0.7, -0.2, 0.9],
        }
    }

    #[test]
    fn analytical_backward_matches_central_finite_difference() {
        let report =
            check_finite_difference(&program(), &case(), FiniteDifferenceConfig::default())
                .unwrap();
        assert!(report.passed, "{report:?}");
        assert_eq!(report.variables_checked, 16);
    }

    #[test]
    fn zero_upstream_gradient_produces_zero_qkv_gradients() {
        let mut input = case();
        input.output_gradient.fill(0.0);
        let gradients = backward(&program(), &input).unwrap();
        assert!(
            gradients
                .queries
                .iter()
                .chain(&gradients.keys)
                .chain(&gradients.values)
                .all(|value| *value == 0.0)
        );
    }

    #[test]
    fn unsupported_selection_fails_closed() {
        let semantic = SemanticProgram::standard_softmax(
            SemanticId::new(SemanticFamily::StandardSoftmax, "topk-backward", 1).unwrap(),
            MaskRule::Unmasked,
            SelectionRule::TopK { k: 1 },
            1.0,
        )
        .unwrap();
        assert_eq!(
            backward(&semantic, &case()),
            Err(BackwardError::UnsupportedSemantic("selection"))
        );
    }

    #[test]
    fn checker_budget_is_enforced_before_perturbation() {
        let result = check_finite_difference(
            &program(),
            &case(),
            FiniteDifferenceConfig {
                max_variables: 4,
                ..FiniteDifferenceConfig::default()
            },
        );
        assert_eq!(
            result,
            Err(BackwardError::FiniteDifferenceBudgetExceeded {
                variables: 16,
                maximum: 4,
            })
        );
    }

    #[test]
    fn non_finite_upstream_gradient_fails_closed() {
        let mut input = case();
        input.output_gradient[0] = f64::NAN;
        assert_eq!(
            backward(&program(), &input),
            Err(BackwardError::NonFinite("input"))
        );
    }
}
