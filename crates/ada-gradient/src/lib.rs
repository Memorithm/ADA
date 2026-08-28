//! Bounded training-backward and finite-difference gradient checking for ADA.
//!
//! This crate differentiates the currently executable single-batch/single-head
//! f64 semantic reference path. Backward is expressed as a vector-Jacobian
//! product (VJP) seeded by an explicit output cotangent. An independent central
//! finite-difference oracle can then check the analytic Q/K/V gradients.
//!
//! The v1 backward domain is intentionally fail-closed. It supports fixed
//! visibility (`All`/`Window` plus unmasked, causal, or external masks),
//! `Softmax` and `SignedDifference`, row centering, scaled dot products, and
//! weighted value mixing. `TopK` is rejected because infinitesimal perturbations
//! can change discrete membership and ties, so a single smooth derivative is
//! not generally valid there.

#![forbid(unsafe_code)]

use ada_semantic::{
    AffinityRule, InputTransform, ReferenceInput, ReferenceInputSpec, SelectionRule,
    SemanticIrError, SemanticProgram, WeightRule,
};
use ada_workload::{WorkloadContract, WorkloadMode};
use std::fmt::{Display, Formatter};

/// Maximum number of Q/K/V scalar parameters checked by one finite-difference run.
pub const MAX_GRADIENT_CHECK_PARAMETERS: usize = 65_536;

/// Training-backward or gradient-check failures.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GradientError {
    /// The underlying semantic reference path rejected the program or input.
    Semantic(SemanticIrError),
    /// The requested workload/semantic feature has no v1 backward rule.
    Unsupported(&'static str),
    /// A configuration scalar or limit is invalid.
    InvalidConfig(&'static str),
    /// A caller-provided vector has the wrong shape.
    ShapeMismatch {
        /// Field whose shape was wrong.
        field: &'static str,
        /// Expected scalar count.
        expected: usize,
        /// Actual scalar count.
        actual: usize,
    },
    /// A backward or finite-difference scalar became non-finite.
    NonFinite(&'static str),
    /// The finite-difference request exceeds its explicit evaluation budget.
    ParameterBudgetExceeded {
        /// Requested Q/K/V scalar count.
        requested: usize,
        /// Configured inclusive maximum.
        maximum: usize,
    },
}

impl Display for GradientError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Semantic(error) => write!(formatter, "semantic gradient error: {error}"),
            Self::Unsupported(feature) => write!(formatter, "gradient domain does not support {feature}"),
            Self::InvalidConfig(field) => write!(formatter, "invalid gradient-check configuration: {field}"),
            Self::ShapeMismatch {
                field,
                expected,
                actual,
            } => write!(formatter, "{field} has {actual} elements; expected {expected}"),
            Self::NonFinite(stage) => write!(formatter, "non-finite value during {stage}"),
            Self::ParameterBudgetExceeded { requested, maximum } => write!(
                formatter,
                "gradient check requests {requested} parameters; maximum is {maximum}"
            ),
        }
    }
}

impl std::error::Error for GradientError {}

impl From<SemanticIrError> for GradientError {
    fn from(value: SemanticIrError) -> Self {
        Self::Semantic(value)
    }
}

/// Analytic VJP with respect to the explicit Q/K/V reference tensors.
#[derive(Debug, Clone, PartialEq)]
pub struct BackwardGradients {
    queries: Vec<f64>,
    keys: Vec<f64>,
    values: Vec<f64>,
}

impl BackwardGradients {
    /// Row-major dL/dQ.
    #[must_use]
    pub fn queries(&self) -> &[f64] {
        &self.queries
    }

    /// Row-major dL/dK.
    #[must_use]
    pub fn keys(&self) -> &[f64] {
        &self.keys
    }

    /// Row-major dL/dV.
    #[must_use]
    pub fn values(&self) -> &[f64] {
        &self.values
    }
}

/// Tensor owning one gradient-check coordinate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GradientTensor {
    /// Query tensor.
    Query,
    /// Key tensor.
    Key,
    /// Value tensor.
    Value,
}

/// Coordinate associated with a reported finite-difference error.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GradientCoordinate {
    tensor: GradientTensor,
    index: usize,
}

impl GradientCoordinate {
    /// Tensor containing the scalar.
    #[must_use]
    pub const fn tensor(self) -> GradientTensor {
        self.tensor
    }

    /// Flat row-major scalar index.
    #[must_use]
    pub const fn index(self) -> usize {
        self.index
    }
}

/// Deterministic central finite-difference checker configuration.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GradientCheckConfig {
    /// Relative perturbation scale. The actual step is `epsilon * max(1, |x|)`.
    pub epsilon: f64,
    /// Absolute acceptance tolerance.
    pub absolute_tolerance: f64,
    /// Relative acceptance tolerance.
    pub relative_tolerance: f64,
    /// Maximum number of Q/K/V scalar parameters checked.
    pub max_parameters: usize,
}

impl Default for GradientCheckConfig {
    fn default() -> Self {
        Self {
            epsilon: 1.0e-6,
            absolute_tolerance: 2.0e-7,
            relative_tolerance: 2.0e-5,
            max_parameters: 4_096,
        }
    }
}

impl GradientCheckConfig {
    fn validate(self) -> Result<(), GradientError> {
        if !self.epsilon.is_finite() || self.epsilon <= 0.0 {
            return Err(GradientError::InvalidConfig("epsilon"));
        }
        if !self.absolute_tolerance.is_finite() || self.absolute_tolerance < 0.0 {
            return Err(GradientError::InvalidConfig("absolute_tolerance"));
        }
        if !self.relative_tolerance.is_finite() || self.relative_tolerance < 0.0 {
            return Err(GradientError::InvalidConfig("relative_tolerance"));
        }
        if self.max_parameters == 0 || self.max_parameters > MAX_GRADIENT_CHECK_PARAMETERS {
            return Err(GradientError::InvalidConfig("max_parameters"));
        }
        Ok(())
    }
}

/// Summary of an exhaustive bounded Q/K/V finite-difference comparison.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GradientCheckReport {
    checked_parameters: usize,
    max_absolute_error: f64,
    max_relative_error: f64,
    worst_coordinate: GradientCoordinate,
    passed: bool,
}

impl GradientCheckReport {
    /// Number of Q/K/V scalar derivatives checked.
    #[must_use]
    pub const fn checked_parameters(self) -> usize {
        self.checked_parameters
    }

    /// Maximum absolute analytic-vs-numeric derivative error.
    #[must_use]
    pub const fn max_absolute_error(self) -> f64 {
        self.max_absolute_error
    }

    /// Maximum relative analytic-vs-numeric derivative error.
    #[must_use]
    pub const fn max_relative_error(self) -> f64 {
        self.max_relative_error
    }

    /// Coordinate with the largest absolute discrepancy.
    #[must_use]
    pub const fn worst_coordinate(self) -> GradientCoordinate {
        self.worst_coordinate
    }

    /// Whether every checked derivative met the configured mixed tolerance.
    #[must_use]
    pub const fn passed(self) -> bool {
        self.passed
    }
}

/// Compute the analytic Q/K/V VJP for a training-forward workload.
///
/// `output_cotangent` is row-major dL/dO and therefore defines the scalar loss
/// whose input gradients are returned.
///
/// # Errors
///
/// Returns [`GradientError`] when the workload is outside the executable f64
/// training-forward domain, the reference input/cotangent shape is inconsistent,
/// `TopK` is requested, or any intermediate becomes non-finite.
pub fn backward(
    program: &SemanticProgram,
    workload: &WorkloadContract,
    input_spec: &ReferenceInputSpec,
    output_cotangent: &[f64],
) -> Result<BackwardGradients, GradientError> {
    validate_domain(program, workload, input_spec, output_cotangent)?;
    let input = ReferenceInput::new(input_spec.clone())?;
    let forward = program.evaluate(&input)?;
    let (transformed_queries, transformed_keys) = transformed_qk(program, input_spec)?;

    let query_count = input_spec.query_count;
    let key_count = input_spec.key_count;
    let query_dimension = input_spec.q_dimension;
    let value_dimension = input_spec.value_dimension;
    let affinity_scale = match program.affinity() {
        AffinityRule::ScaledDotProduct { scale } => scale,
    };

    let mut query_gradients = vec![0.0_f64; input_spec.queries.len()];
    let mut key_gradients = vec![0.0_f64; input_spec.keys.len()];
    let mut value_gradients = vec![0.0_f64; input_spec.values.len()];

    for query_index in 0..query_count {
        let selected = &forward.selected_keys()[query_index];
        let cotangent_row = &output_cotangent
            [query_index * value_dimension..(query_index + 1) * value_dimension];
        let query_row = &transformed_queries
            [query_index * query_dimension..(query_index + 1) * query_dimension];

        let mut value_sensitivities = Vec::with_capacity(selected.len());
        for &key_index in selected {
            let weight = forward.weights()[query_index * key_count + key_index];
            let value_row = &input_spec.values
                [key_index * value_dimension..(key_index + 1) * value_dimension];
            let value_gradient_row = &mut value_gradients
                [key_index * value_dimension..(key_index + 1) * value_dimension];
            let mut sensitivity = 0.0_f64;
            for ((gradient, &cotangent), &value) in value_gradient_row
                .iter_mut()
                .zip(cotangent_row)
                .zip(value_row)
            {
                *gradient += weight * cotangent;
                sensitivity += cotangent * value;
            }
            value_sensitivities.push(sensitivity);
        }

        let score_gradients = score_vjp(
            program,
            input_spec,
            &transformed_queries,
            &transformed_keys,
            query_index,
            selected,
            &value_sensitivities,
            forward.weights(),
        )?;

        for (&key_index, &score_gradient) in selected.iter().zip(&score_gradients) {
            let key_row = &transformed_keys
                [key_index * query_dimension..(key_index + 1) * query_dimension];
            let query_gradient_row = &mut query_gradients
                [query_index * query_dimension..(query_index + 1) * query_dimension];
            let key_gradient_row = &mut key_gradients
                [key_index * query_dimension..(key_index + 1) * query_dimension];
            let scaled_gradient = score_gradient * affinity_scale;
            for dimension in 0..query_dimension {
                query_gradient_row[dimension] += scaled_gradient * key_row[dimension];
                key_gradient_row[dimension] += scaled_gradient * query_row[dimension];
            }
        }
    }

    if matches!(program.input_transform(), InputTransform::CenterRows) {
        center_rows_vjp(&mut query_gradients, query_count, query_dimension)?;
        center_rows_vjp(&mut key_gradients, key_count, query_dimension)?;
    }

    ensure_finite(&query_gradients, "query backward")?;
    ensure_finite(&key_gradients, "key backward")?;
    ensure_finite(&value_gradients, "value backward")?;
    Ok(BackwardGradients {
        queries: query_gradients,
        keys: key_gradients,
        values: value_gradients,
    })
}

/// Compare the analytic VJP against an independent central finite-difference
/// oracle for every Q/K/V scalar admitted by the configured budget.
///
/// # Errors
///
/// Returns [`GradientError`] for unsupported/non-smooth semantics, invalid
/// training/input shapes, invalid checker configuration, excessive parameter
/// count, or non-finite perturbed evaluations. A numerical mismatch itself is
/// reported by [`GradientCheckReport::passed`] rather than converted to an error.
pub fn finite_difference_check(
    program: &SemanticProgram,
    workload: &WorkloadContract,
    input_spec: &ReferenceInputSpec,
    output_cotangent: &[f64],
    config: GradientCheckConfig,
) -> Result<GradientCheckReport, GradientError> {
    config.validate()?;
    let analytic = backward(program, workload, input_spec, output_cotangent)?;
    let parameter_count = input_spec
        .queries
        .len()
        .checked_add(input_spec.keys.len())
        .and_then(|count| count.checked_add(input_spec.values.len()))
        .ok_or(GradientError::ParameterBudgetExceeded {
            requested: usize::MAX,
            maximum: config.max_parameters,
        })?;
    if parameter_count > config.max_parameters {
        return Err(GradientError::ParameterBudgetExceeded {
            requested: parameter_count,
            maximum: config.max_parameters,
        });
    }

    let mut accumulator = CheckAccumulator::new();
    check_tensor(
        program,
        input_spec,
        output_cotangent,
        GradientTensor::Query,
        analytic.queries(),
        config,
        &mut accumulator,
    )?;
    check_tensor(
        program,
        input_spec,
        output_cotangent,
        GradientTensor::Key,
        analytic.keys(),
        config,
        &mut accumulator,
    )?;
    check_tensor(
        program,
        input_spec,
        output_cotangent,
        GradientTensor::Value,
        analytic.values(),
        config,
        &mut accumulator,
    )?;
    Ok(accumulator.finish())
}

fn validate_domain(
    program: &SemanticProgram,
    workload: &WorkloadContract,
    input_spec: &ReferenceInputSpec,
    output_cotangent: &[f64],
) -> Result<(), GradientError> {
    program.validate_for_workload(workload)?;
    if workload.mode() != WorkloadMode::TrainingForward {
        return Err(GradientError::Unsupported(
            "backward requires a TrainingForward workload contract",
        ));
    }
    if matches!(program.selection(), SelectionRule::TopK { .. }) {
        return Err(GradientError::Unsupported(
            "TopK backward because selection membership is discontinuous",
        ));
    }
    validate_input_against_workload(workload, input_spec)?;
    ReferenceInput::new(input_spec.clone())?;
    let expected_output = input_spec
        .query_count
        .checked_mul(input_spec.value_dimension)
        .ok_or(GradientError::ShapeMismatch {
            field: "output_cotangent",
            expected: usize::MAX,
            actual: output_cotangent.len(),
        })?;
    if output_cotangent.len() != expected_output {
        return Err(GradientError::ShapeMismatch {
            field: "output_cotangent",
            expected: expected_output,
            actual: output_cotangent.len(),
        });
    }
    ensure_finite(output_cotangent, "output cotangent")
}

fn validate_input_against_workload(
    workload: &WorkloadContract,
    input_spec: &ReferenceInputSpec,
) -> Result<(), GradientError> {
    let geometry = workload.geometry();
    let lengths = geometry.sequence_lengths();
    let expected_queries = lengths
        .query_length_for(0)
        .ok_or(GradientError::Unsupported("missing query length"))?;
    let expected_keys = lengths
        .kv_length_for(0)
        .ok_or(GradientError::Unsupported("missing KV length"))?;
    let expected_dimension = geometry
        .qk_dimension()
        .ok_or(GradientError::Unsupported("missing Q/K dimension"))?;
    for (field, expected, actual) in [
        ("query_count", expected_queries, input_spec.query_count),
        ("key_count", expected_keys, input_spec.key_count),
        ("q_dimension", expected_dimension, input_spec.q_dimension),
        (
            "value_dimension",
            geometry.value_dimension(),
            input_spec.value_dimension,
        ),
    ] {
        if expected != actual {
            return Err(GradientError::ShapeMismatch {
                field,
                expected,
                actual,
            });
        }
    }
    Ok(())
}

fn transformed_qk(
    program: &SemanticProgram,
    input_spec: &ReferenceInputSpec,
) -> Result<(Vec<f64>, Vec<f64>), GradientError> {
    let mut queries = input_spec.queries.clone();
    let mut keys = input_spec.keys.clone();
    if matches!(program.input_transform(), InputTransform::CenterRows) {
        center_rows(&mut queries, input_spec.query_count, input_spec.q_dimension)?;
        center_rows(&mut keys, input_spec.key_count, input_spec.q_dimension)?;
    }
    Ok((queries, keys))
}

fn center_rows(
    values: &mut [f64],
    row_count: usize,
    dimension: usize,
) -> Result<(), GradientError> {
    let dimension_u32 =
        u32::try_from(dimension).map_err(|_| GradientError::Unsupported("dimension > u32"))?;
    let denominator = f64::from(dimension_u32);
    for row_index in 0..row_count {
        let row = &mut values[row_index * dimension..(row_index + 1) * dimension];
        let mean = row.iter().sum::<f64>() / denominator;
        if !mean.is_finite() {
            return Err(GradientError::NonFinite("row centering"));
        }
        for value in row {
            *value -= mean;
        }
    }
    Ok(())
}

fn center_rows_vjp(
    gradients: &mut [f64],
    row_count: usize,
    dimension: usize,
) -> Result<(), GradientError> {
    let dimension_u32 =
        u32::try_from(dimension).map_err(|_| GradientError::Unsupported("dimension > u32"))?;
    let denominator = f64::from(dimension_u32);
    for row_index in 0..row_count {
        let row = &mut gradients[row_index * dimension..(row_index + 1) * dimension];
        let mean = row.iter().sum::<f64>() / denominator;
        if !mean.is_finite() {
            return Err(GradientError::NonFinite("row-centering VJP"));
        }
        for gradient in row {
            *gradient -= mean;
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn score_vjp(
    program: &SemanticProgram,
    input_spec: &ReferenceInputSpec,
    transformed_queries: &[f64],
    transformed_keys: &[f64],
    query_index: usize,
    selected: &[usize],
    value_sensitivities: &[f64],
    forward_weights: &[f64],
) -> Result<Vec<f64>, GradientError> {
    let key_count = input_spec.key_count;
    match program.weight() {
        WeightRule::Softmax => {
            let mean = selected
                .iter()
                .zip(value_sensitivities)
                .map(|(&key_index, &sensitivity)| {
                    forward_weights[query_index * key_count + key_index] * sensitivity
                })
                .sum::<f64>();
            let gradients = selected
                .iter()
                .zip(value_sensitivities)
                .map(|(&key_index, &sensitivity)| {
                    let weight = forward_weights[query_index * key_count + key_index];
                    weight * (sensitivity - mean)
                })
                .collect::<Vec<_>>();
            ensure_finite(&gradients, "softmax score VJP")?;
            Ok(gradients)
        }
        WeightRule::SignedDifference {
            positive_scale,
            negative_scale,
        } => {
            let scores = selected_scores(
                program,
                input_spec,
                transformed_queries,
                transformed_keys,
                query_index,
                selected,
            )?;
            let positive = stable_softmax(&scores, positive_scale)?;
            let negative = stable_softmax(&scores, negative_scale)?;
            let positive_mean = positive
                .iter()
                .zip(value_sensitivities)
                .map(|(&weight, &sensitivity)| weight * sensitivity)
                .sum::<f64>();
            let negative_mean = negative
                .iter()
                .zip(value_sensitivities)
                .map(|(&weight, &sensitivity)| weight * sensitivity)
                .sum::<f64>();
            let gradients = positive
                .iter()
                .zip(&negative)
                .zip(value_sensitivities)
                .map(|((&positive_weight, &negative_weight), &sensitivity)| {
                    positive_scale * positive_weight * (sensitivity - positive_mean)
                        - negative_scale * negative_weight * (sensitivity - negative_mean)
                })
                .collect::<Vec<_>>();
            ensure_finite(&gradients, "signed-difference score VJP")?;
            Ok(gradients)
        }
    }
}

fn selected_scores(
    program: &SemanticProgram,
    input_spec: &ReferenceInputSpec,
    transformed_queries: &[f64],
    transformed_keys: &[f64],
    query_index: usize,
    selected: &[usize],
) -> Result<Vec<f64>, GradientError> {
    let dimension = input_spec.q_dimension;
    let query_row =
        &transformed_queries[query_index * dimension..(query_index + 1) * dimension];
    let affinity_scale = match program.affinity() {
        AffinityRule::ScaledDotProduct { scale } => scale,
    };
    let mut scores = Vec::with_capacity(selected.len());
    for &key_index in selected {
        let key_row = &transformed_keys[key_index * dimension..(key_index + 1) * dimension];
        let mut dot = 0.0_f64;
        for (&query_value, &key_value) in query_row.iter().zip(key_row) {
            dot += query_value * key_value;
        }
        let score = dot * affinity_scale;
        if !score.is_finite() {
            return Err(GradientError::NonFinite("signed-difference affinity"));
        }
        scores.push(score);
    }
    Ok(scores)
}

fn stable_softmax(scores: &[f64], scale: f64) -> Result<Vec<f64>, GradientError> {
    let maximum = scores
        .iter()
        .copied()
        .max_by(f64::total_cmp)
        .ok_or(GradientError::Unsupported("empty score selection"))?;
    let scaled_maximum = maximum * scale;
    if !scaled_maximum.is_finite() {
        return Err(GradientError::NonFinite("gradient softmax scale"));
    }
    let mut weights = Vec::with_capacity(scores.len());
    let mut sum = 0.0_f64;
    for &score in scores {
        let weight = (score * scale - scaled_maximum).exp();
        if !weight.is_finite() {
            return Err(GradientError::NonFinite("gradient softmax exponential"));
        }
        weights.push(weight);
        sum += weight;
    }
    if !sum.is_finite() || sum <= 0.0 {
        return Err(GradientError::NonFinite("gradient softmax normalizer"));
    }
    for weight in &mut weights {
        *weight /= sum;
    }
    Ok(weights)
}

fn check_tensor(
    program: &SemanticProgram,
    input_spec: &ReferenceInputSpec,
    output_cotangent: &[f64],
    tensor: GradientTensor,
    analytic: &[f64],
    config: GradientCheckConfig,
    accumulator: &mut CheckAccumulator,
) -> Result<(), GradientError> {
    for (index, &analytic_value) in analytic.iter().enumerate() {
        let numeric_value = numeric_derivative(
            program,
            input_spec,
            output_cotangent,
            tensor,
            index,
            config.epsilon,
        )?;
        accumulator.observe(
            GradientCoordinate { tensor, index },
            analytic_value,
            numeric_value,
            config,
        )?;
    }
    Ok(())
}

fn numeric_derivative(
    program: &SemanticProgram,
    input_spec: &ReferenceInputSpec,
    output_cotangent: &[f64],
    tensor: GradientTensor,
    index: usize,
    epsilon: f64,
) -> Result<f64, GradientError> {
    let base = tensor_value(input_spec, tensor, index)?;
    let step = epsilon * base.abs().max(1.0);
    if !step.is_finite() || step <= 0.0 {
        return Err(GradientError::NonFinite("finite-difference step"));
    }
    let plus_value = base + step;
    let minus_value = base - step;
    if !plus_value.is_finite() || !minus_value.is_finite() {
        return Err(GradientError::NonFinite("finite-difference perturbation"));
    }
    let mut plus = input_spec.clone();
    let mut minus = input_spec.clone();
    *tensor_value_mut(&mut plus, tensor, index)? = plus_value;
    *tensor_value_mut(&mut minus, tensor, index)? = minus_value;
    let plus_loss = scalar_loss(program, &plus, output_cotangent)?;
    let minus_loss = scalar_loss(program, &minus, output_cotangent)?;
    let derivative = (plus_loss - minus_loss) / (2.0 * step);
    if derivative.is_finite() {
        Ok(derivative)
    } else {
        Err(GradientError::NonFinite("finite-difference derivative"))
    }
}

fn tensor_value(
    input_spec: &ReferenceInputSpec,
    tensor: GradientTensor,
    index: usize,
) -> Result<f64, GradientError> {
    let values = match tensor {
        GradientTensor::Query => &input_spec.queries,
        GradientTensor::Key => &input_spec.keys,
        GradientTensor::Value => &input_spec.values,
    };
    values
        .get(index)
        .copied()
        .ok_or(GradientError::ShapeMismatch {
            field: "gradient coordinate",
            expected: values.len(),
            actual: index.saturating_add(1),
        })
}

fn tensor_value_mut(
    input_spec: &mut ReferenceInputSpec,
    tensor: GradientTensor,
    index: usize,
) -> Result<&mut f64, GradientError> {
    let values = match tensor {
        GradientTensor::Query => &mut input_spec.queries,
        GradientTensor::Key => &mut input_spec.keys,
        GradientTensor::Value => &mut input_spec.values,
    };
    let length = values.len();
    values.get_mut(index).ok_or(GradientError::ShapeMismatch {
        field: "gradient coordinate",
        expected: length,
        actual: index.saturating_add(1),
    })
}

fn scalar_loss(
    program: &SemanticProgram,
    input_spec: &ReferenceInputSpec,
    output_cotangent: &[f64],
) -> Result<f64, GradientError> {
    let input = ReferenceInput::new(input_spec.clone())?;
    let output = program.evaluate(&input)?;
    if output.output().len() != output_cotangent.len() {
        return Err(GradientError::ShapeMismatch {
            field: "output_cotangent",
            expected: output.output().len(),
            actual: output_cotangent.len(),
        });
    }
    let loss = output
        .output()
        .iter()
        .zip(output_cotangent)
        .map(|(&value, &cotangent)| value * cotangent)
        .sum::<f64>();
    if loss.is_finite() {
        Ok(loss)
    } else {
        Err(GradientError::NonFinite("finite-difference scalar loss"))
    }
}

fn ensure_finite(values: &[f64], stage: &'static str) -> Result<(), GradientError> {
    if values.iter().all(|value| value.is_finite()) {
        Ok(())
    } else {
        Err(GradientError::NonFinite(stage))
    }
}

#[derive(Debug, Clone, Copy)]
struct CheckAccumulator {
    checked_parameters: usize,
    max_absolute_error: f64,
    max_relative_error: f64,
    worst_coordinate: GradientCoordinate,
    passed: bool,
}

impl CheckAccumulator {
    const fn new() -> Self {
        Self {
            checked_parameters: 0,
            max_absolute_error: 0.0,
            max_relative_error: 0.0,
            worst_coordinate: GradientCoordinate {
                tensor: GradientTensor::Query,
                index: 0,
            },
            passed: true,
        }
    }

    fn observe(
        &mut self,
        coordinate: GradientCoordinate,
        analytic: f64,
        numeric: f64,
        config: GradientCheckConfig,
    ) -> Result<(), GradientError> {
        if !analytic.is_finite() || !numeric.is_finite() {
            return Err(GradientError::NonFinite("gradient comparison"));
        }
        let absolute_error = (analytic - numeric).abs();
        let scale = analytic.abs().max(numeric.abs());
        let relative_error = if scale == 0.0 {
            0.0
        } else {
            absolute_error / scale
        };
        if absolute_error > self.max_absolute_error {
            self.max_absolute_error = absolute_error;
            self.worst_coordinate = coordinate;
        }
        self.max_relative_error = self.max_relative_error.max(relative_error);
        let tolerance = config.absolute_tolerance + config.relative_tolerance * scale;
        if absolute_error > tolerance {
            self.passed = false;
        }
        self.checked_parameters += 1;
        Ok(())
    }

    const fn finish(self) -> GradientCheckReport {
        GradientCheckReport {
            checked_parameters: self.checked_parameters,
            max_absolute_error: self.max_absolute_error,
            max_relative_error: self.max_relative_error,
            worst_coordinate: self.worst_coordinate,
            passed: self.passed,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ada_core::{MaskContract, SemanticDescriptor, SemanticFamily, SemanticId, StateContract, WeightContract};
    use ada_semantic::{MaskRule, OutputRule, SemanticProgramSpec, ValueMixRule};
    use ada_workload::{
        AttentionGeometry, AttentionTopology, GeometrySpec, HeadGrouping, MaskKind, MaskSpec,
        PrecisionPolicy, ScalarPrecision, SequenceLengths, WorkloadOptions,
    };

    fn workload(mask: MaskSpec) -> WorkloadContract {
        let geometry = AttentionGeometry::new(GeometrySpec {
            sequence_lengths: SequenceLengths::uniform(1, 2, 3).unwrap(),
            query_heads: 1,
            kv_heads: 1,
            qk_dimension: Some(2),
            value_dimension: 2,
            topology: AttentionTopology::SelfAttention,
            head_grouping: HeadGrouping::MultiHead,
        })
        .unwrap();
        WorkloadContract::new(
            geometry,
            WorkloadOptions {
                mode: WorkloadMode::TrainingForward,
                mask,
                precision: PrecisionPolicy::new(
                    ScalarPrecision::F64,
                    ScalarPrecision::F64,
                    ScalarPrecision::F64,
                    ScalarPrecision::F64,
                ),
                ..WorkloadOptions::default()
            },
        )
        .unwrap()
    }

    fn input(mask: Option<Vec<bool>>) -> ReferenceInputSpec {
        ReferenceInputSpec {
            query_count: 2,
            key_count: 3,
            q_dimension: 2,
            value_dimension: 2,
            queries: vec![0.3, -0.7, 0.8, 0.2],
            keys: vec![0.5, -0.4, -0.1, 0.9, 0.7, 0.6],
            values: vec![1.0, -0.5, 0.2, 0.8, -0.3, 1.2],
            external_mask: mask,
        }
    }

    fn id(name: &str) -> SemanticId {
        SemanticId::new(SemanticFamily::Experimental, name, 1).unwrap()
    }

    fn softmax(selection: SelectionRule) -> SemanticProgram {
        SemanticProgram::standard_softmax(id("gradient-softmax"), MaskRule::Unmasked, selection, 0.7)
            .unwrap()
    }

    #[test]
    fn dense_softmax_backward_matches_central_finite_difference() {
        let report = finite_difference_check(
            &softmax(SelectionRule::All),
            &workload(MaskSpec::none()),
            &input(None),
            &[0.4, -0.2, 0.7, 0.3],
            GradientCheckConfig::default(),
        )
        .unwrap();
        assert_eq!(report.checked_parameters(), 16);
        assert!(report.passed(), "{report:?}");
        assert!(report.max_absolute_error() < 1.0e-6, "{report:?}");
    }

    #[test]
    fn signed_difference_backward_matches_finite_difference() {
        let program = SemanticProgram::signed_difference(
            id("gradient-signed"),
            MaskRule::Unmasked,
            SelectionRule::Window { radius: 2 },
            0.9,
            1.3,
            0.6,
        )
        .unwrap();
        let report = finite_difference_check(
            &program,
            &workload(MaskSpec::none()),
            &input(None),
            &[0.2, 0.5, -0.4, 0.1],
            GradientCheckConfig {
                absolute_tolerance: 5.0e-7,
                relative_tolerance: 5.0e-5,
                ..GradientCheckConfig::default()
            },
        )
        .unwrap();
        assert!(report.passed(), "{report:?}");
    }

    #[test]
    fn centered_external_mask_backward_matches_finite_difference() {
        let mask_identity = "gradient-mask".to_string();
        let descriptor = SemanticDescriptor::new(
            id("gradient-centered"),
            MaskContract::ExternalMask,
            StateContract::Stateless,
            WeightContract::ProbabilitySimplex,
        );
        let program = SemanticProgram::new(SemanticProgramSpec {
            descriptor,
            input_transform: InputTransform::CenterRows,
            affinity: AffinityRule::ScaledDotProduct { scale: 0.8 },
            mask: MaskRule::External {
                identity: mask_identity.clone(),
            },
            selection: SelectionRule::All,
            weight: WeightRule::Softmax,
            value_mix: ValueMixRule::WeightedSum,
            output: OutputRule::Identity,
        })
        .unwrap();
        let mask_values = vec![true, false, true, true, true, false];
        let report = finite_difference_check(
            &program,
            &workload(MaskSpec::new(MaskKind::External {
                identity: mask_identity,
            }).unwrap()),
            &input(Some(mask_values)),
            &[0.4, 0.1, -0.2, 0.6],
            GradientCheckConfig::default(),
        )
        .unwrap();
        assert!(report.passed(), "{report:?}");
    }

    #[test]
    fn topk_backward_fails_closed() {
        let result = backward(
            &softmax(SelectionRule::TopK { k: 2 }),
            &workload(MaskSpec::none()),
            &input(None),
            &[1.0, 0.0, 0.0, 1.0],
        );
        assert_eq!(
            result,
            Err(GradientError::Unsupported(
                "TopK backward because selection membership is discontinuous"
            ))
        );
    }

    #[test]
    fn backward_requires_training_forward_contract() {
        let geometry = AttentionGeometry::new(GeometrySpec {
            sequence_lengths: SequenceLengths::uniform(1, 2, 3).unwrap(),
            query_heads: 1,
            kv_heads: 1,
            qk_dimension: Some(2),
            value_dimension: 2,
            topology: AttentionTopology::SelfAttention,
            head_grouping: HeadGrouping::MultiHead,
        })
        .unwrap();
        let prefill = WorkloadContract::new(
            geometry,
            WorkloadOptions {
                precision: PrecisionPolicy::new(
                    ScalarPrecision::F64,
                    ScalarPrecision::F64,
                    ScalarPrecision::F64,
                    ScalarPrecision::F64,
                ),
                ..WorkloadOptions::default()
            },
        )
        .unwrap();
        let result = backward(
            &softmax(SelectionRule::All),
            &prefill,
            &input(None),
            &[1.0, 0.0, 0.0, 1.0],
        );
        assert_eq!(
            result,
            Err(GradientError::Unsupported(
                "backward requires a TrainingForward workload contract"
            ))
        );
    }
}
