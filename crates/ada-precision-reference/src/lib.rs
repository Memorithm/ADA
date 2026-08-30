//! Executable numeric reference models for ADA precision declarations.
//!
//! `ScalarPrecision` in `ada-workload` is intentionally declarative. This crate
//! adds explicit, bounded numeric models that can be used by offline research
//! fixtures. F8/F4 are never treated as unqualified generic formats: callers
//! must choose an exact lattice. I8 likewise requires an explicit scale.
//!
//! These models are deterministic software references, not claims about any
//! accelerator's instruction-level rounding, denormal, NaN, saturation, or
//! throughput behavior.

#![forbid(unsafe_code)]

use std::fmt::{Display, Formatter};

use ada_workload::{PrecisionPolicy, ScalarPrecision, WorkloadContract};

/// Overflow handling for finite reference lattices.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OverflowPolicy {
    /// Reject values that round outside the largest finite representable value.
    Reject,
    /// Clamp values that round outside the finite lattice to its largest magnitude.
    Saturate,
}

/// Concrete executable precision format.
#[derive(Debug, Clone, PartialEq)]
pub enum ExecutablePrecision {
    /// Native finite binary64 reference values.
    F64,
    /// IEEE-like binary32 finite lattice: 8 exponent bits, 23 fraction bits, bias 127.
    F32,
    /// bfloat16 finite lattice: 8 exponent bits, 7 fraction bits, bias 127.
    Bf16,
    /// IEEE-like binary16 finite lattice: 5 exponent bits, 10 fraction bits, bias 15.
    F16,
    /// Explicit IEEE-like E4M3 finite lattice with the all-ones exponent reserved.
    ///
    /// This is deliberately not named E4M3FN: that common FP8 encoding has a
    /// different top-end finite-value convention.
    F8E4M3IeeeLike,
    /// Explicit IEEE-like E5M2 finite lattice with the all-ones exponent reserved.
    F8E5M2,
    /// Explicit IEEE-like E2M1 finite lattice with the all-ones exponent reserved.
    F4E2M1,
    /// Symmetric signed int8 lattice `q * scale`, with integer q in `[-127, 127]`.
    I8Symmetric {
        /// Positive finite scale for one integer step.
        scale: f64,
    },
}

impl ExecutablePrecision {
    /// Declarative workload precision matched by this executable format.
    #[must_use]
    pub const fn scalar_precision(&self) -> ScalarPrecision {
        match self {
            Self::F64 => ScalarPrecision::F64,
            Self::F32 => ScalarPrecision::F32,
            Self::Bf16 => ScalarPrecision::BF16,
            Self::F16 => ScalarPrecision::F16,
            Self::F8E4M3IeeeLike | Self::F8E5M2 => ScalarPrecision::F8,
            Self::F4E2M1 => ScalarPrecision::F4,
            Self::I8Symmetric { .. } => ScalarPrecision::I8,
        }
    }

    fn validate(&self) -> Result<(), PrecisionReferenceError> {
        if let Self::I8Symmetric { scale } = self {
            if !scale.is_finite() || *scale <= 0.0 {
                return Err(PrecisionReferenceError::InvalidScale);
            }
        }
        Ok(())
    }
}

/// Numeric stage used in precision-policy diagnostics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrecisionStage {
    /// Logical values written to/read from storage.
    Storage,
    /// Arithmetic inputs.
    Input,
    /// Running reduction accumulator.
    Accumulation,
    /// Returned arithmetic output.
    Output,
}

/// Fail-closed executable-precision error.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PrecisionReferenceError {
    /// An executable format does not match the workload's declared scalar class.
    PolicyMismatch {
        /// Mismatched precision-policy stage.
        stage: PrecisionStage,
        /// Workload declaration.
        declared: ScalarPrecision,
        /// Executable model's scalar class.
        executable: ScalarPrecision,
    },
    /// Symmetric I8 scale is zero, negative, or non-finite.
    InvalidScale,
    /// Input is NaN or infinite; V1 references accept finite research fixtures only.
    NonFiniteInput,
    /// A value rounds beyond the largest finite point and overflow policy is Reject.
    Overflow,
    /// Dot-product operands have different lengths.
    ShapeMismatch,
}

impl Display for PrecisionReferenceError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::PolicyMismatch {
                stage,
                declared,
                executable,
            } => write!(
                formatter,
                "precision policy mismatch at {stage:?}: declared {declared:?}, executable {executable:?}"
            ),
            Self::InvalidScale => formatter.write_str("I8 scale must be positive and finite"),
            Self::NonFiniteInput => {
                formatter.write_str("precision reference requires finite input")
            }
            Self::Overflow => formatter.write_str("precision reference finite lattice overflow"),
            Self::ShapeMismatch => formatter.write_str("dot-product operand lengths differ"),
        }
    }
}

impl std::error::Error for PrecisionReferenceError {}

/// Fully bound executable interpretation of one [`PrecisionPolicy`].
#[derive(Debug, Clone, PartialEq)]
pub struct PrecisionExecutionSpec {
    declared: PrecisionPolicy,
    input: ExecutablePrecision,
    accumulation: ExecutablePrecision,
    output: ExecutablePrecision,
    storage: ExecutablePrecision,
    overflow: OverflowPolicy,
}

impl PrecisionExecutionSpec {
    /// Bind concrete numeric models to a declarative precision policy.
    ///
    /// # Errors
    ///
    /// Fails when any executable format belongs to a different scalar class or
    /// when an I8 scale is invalid.
    pub fn new(
        declared: PrecisionPolicy,
        input: ExecutablePrecision,
        accumulation: ExecutablePrecision,
        output: ExecutablePrecision,
        storage: ExecutablePrecision,
        overflow: OverflowPolicy,
    ) -> Result<Self, PrecisionReferenceError> {
        input.validate()?;
        accumulation.validate()?;
        output.validate()?;
        storage.validate()?;
        validate_stage(PrecisionStage::Input, declared.input(), &input)?;
        validate_stage(
            PrecisionStage::Accumulation,
            declared.accumulation(),
            &accumulation,
        )?;
        validate_stage(PrecisionStage::Output, declared.output(), &output)?;
        validate_stage(PrecisionStage::Storage, declared.storage(), &storage)?;
        Ok(Self {
            declared,
            input,
            accumulation,
            output,
            storage,
            overflow,
        })
    }

    /// Bind numeric models directly to a validated workload's precision policy.
    ///
    /// # Errors
    ///
    /// Propagates the same format/policy validation as [`Self::new`].
    pub fn for_workload(
        workload: &WorkloadContract,
        input: ExecutablePrecision,
        accumulation: ExecutablePrecision,
        output: ExecutablePrecision,
        storage: ExecutablePrecision,
        overflow: OverflowPolicy,
    ) -> Result<Self, PrecisionReferenceError> {
        Self::new(
            workload.precision(),
            input,
            accumulation,
            output,
            storage,
            overflow,
        )
    }

    /// Declarative policy bound by this spec.
    #[must_use]
    pub const fn declared(&self) -> PrecisionPolicy {
        self.declared
    }

    /// Quantize one value at a named numeric stage.
    ///
    /// # Errors
    ///
    /// Rejects non-finite input, invalid scale, or finite-lattice overflow when
    /// the spec uses [`OverflowPolicy::Reject`].
    pub fn quantize(
        &self,
        stage: PrecisionStage,
        value: f64,
    ) -> Result<f64, PrecisionReferenceError> {
        let format = match stage {
            PrecisionStage::Storage => &self.storage,
            PrecisionStage::Input => &self.input,
            PrecisionStage::Accumulation => &self.accumulation,
            PrecisionStage::Output => &self.output,
        };
        quantize_scalar(format, value, self.overflow)
    }

    /// Simulate storage -> input conversion, per-product accumulation rounding,
    /// and final output rounding for one dot product.
    ///
    /// Multiplication itself is evaluated in f64 between already-quantized
    /// inputs; each product and each running sum are then rounded through the
    /// declared accumulator lattice. This convention is explicit and does not
    /// claim fused-multiply-add behavior for any physical backend.
    ///
    /// # Errors
    ///
    /// Fails on operand shape mismatch, non-finite data, invalid scale, or
    /// rejected overflow.
    pub fn dot_product(
        &self,
        left: &[f64],
        right: &[f64],
    ) -> Result<DotProductReport, PrecisionReferenceError> {
        if left.len() != right.len() {
            return Err(PrecisionReferenceError::ShapeMismatch);
        }
        let mut accumulation = self.quantize(PrecisionStage::Accumulation, 0.0)?;
        let mut trace = Vec::with_capacity(left.len());
        for (&left_value, &right_value) in left.iter().zip(right) {
            let left_stored = self.quantize(PrecisionStage::Storage, left_value)?;
            let right_stored = self.quantize(PrecisionStage::Storage, right_value)?;
            let left_input = self.quantize(PrecisionStage::Input, left_stored)?;
            let right_input = self.quantize(PrecisionStage::Input, right_stored)?;
            let product = self.quantize(PrecisionStage::Accumulation, left_input * right_input)?;
            accumulation = self.quantize(PrecisionStage::Accumulation, accumulation + product)?;
            trace.push(accumulation);
        }
        let output = self.quantize(PrecisionStage::Output, accumulation)?;
        Ok(DotProductReport {
            accumulation_trace: trace,
            output,
        })
    }
}

/// Observable stages from one executable dot-product reference.
#[derive(Debug, Clone, PartialEq)]
pub struct DotProductReport {
    accumulation_trace: Vec<f64>,
    output: f64,
}

impl DotProductReport {
    /// Rounded running accumulator after each product.
    #[must_use]
    pub fn accumulation_trace(&self) -> &[f64] {
        &self.accumulation_trace
    }

    /// Final value after output-stage rounding.
    #[must_use]
    pub const fn output(&self) -> f64 {
        self.output
    }
}

fn validate_stage(
    stage: PrecisionStage,
    declared: ScalarPrecision,
    executable: &ExecutablePrecision,
) -> Result<(), PrecisionReferenceError> {
    let executable_precision = executable.scalar_precision();
    if declared == executable_precision {
        Ok(())
    } else {
        Err(PrecisionReferenceError::PolicyMismatch {
            stage,
            declared,
            executable: executable_precision,
        })
    }
}

fn quantize_scalar(
    format: &ExecutablePrecision,
    value: f64,
    overflow: OverflowPolicy,
) -> Result<f64, PrecisionReferenceError> {
    if !value.is_finite() {
        return Err(PrecisionReferenceError::NonFiniteInput);
    }
    match format {
        ExecutablePrecision::F64 => Ok(value),
        ExecutablePrecision::F32 => quantize_binary(value, 8, 23, 127, overflow),
        ExecutablePrecision::Bf16 => quantize_binary(value, 8, 7, 127, overflow),
        ExecutablePrecision::F16 => quantize_binary(value, 5, 10, 15, overflow),
        ExecutablePrecision::F8E4M3IeeeLike => quantize_binary(value, 4, 3, 7, overflow),
        ExecutablePrecision::F8E5M2 => quantize_binary(value, 5, 2, 15, overflow),
        ExecutablePrecision::F4E2M1 => quantize_binary(value, 2, 1, 1, overflow),
        ExecutablePrecision::I8Symmetric { scale } => {
            quantize_i8_symmetric(value, *scale, overflow)
        }
    }
}

fn quantize_i8_symmetric(
    value: f64,
    scale: f64,
    overflow: OverflowPolicy,
) -> Result<f64, PrecisionReferenceError> {
    if !scale.is_finite() || scale <= 0.0 {
        return Err(PrecisionReferenceError::InvalidScale);
    }
    let rounded = (value / scale).round_ties_even();
    let bounded = if rounded.abs() <= 127.0 {
        rounded
    } else {
        match overflow {
            OverflowPolicy::Reject => return Err(PrecisionReferenceError::Overflow),
            OverflowPolicy::Saturate => rounded.clamp(-127.0, 127.0),
        }
    };
    Ok(bounded * scale)
}

fn quantize_binary(
    value: f64,
    exponent_bits: u32,
    fraction_bits: i32,
    bias: i32,
    overflow: OverflowPolicy,
) -> Result<f64, PrecisionReferenceError> {
    if value == 0.0 {
        return Ok(value);
    }
    let sign = if value.is_sign_negative() { -1.0 } else { 1.0 };
    let magnitude = value.abs();
    let minimum_exponent = 1 - bias;
    let maximum_exponent = ((1_i32 << exponent_bits) - 2) - bias;
    let minimum_normal = pow2(minimum_exponent);
    let subnormal_step = pow2(minimum_exponent - fraction_bits);
    let maximum_finite = (2.0 - pow2(-fraction_bits)) * pow2(maximum_exponent);

    let quantized = if magnitude < minimum_normal {
        (magnitude / subnormal_step).round_ties_even() * subnormal_step
    } else {
        let exponent = floor_log2(magnitude);
        let step = pow2(exponent - fraction_bits);
        (magnitude / step).round_ties_even() * step
    };

    if quantized > maximum_finite {
        return match overflow {
            OverflowPolicy::Reject => Err(PrecisionReferenceError::Overflow),
            OverflowPolicy::Saturate => Ok(sign * maximum_finite),
        };
    }
    Ok(sign * quantized)
}

fn floor_log2(value: f64) -> i32 {
    let bits = value.to_bits();
    let raw_exponent = ((bits >> 52) & 0x7ff) as i32;
    if raw_exponent != 0 {
        return raw_exponent - 1023;
    }
    let fraction = bits & ((1_u64 << 52) - 1);
    let highest_bit =
        63_i32 - i32::try_from(fraction.leading_zeros()).expect("leading-zero count fits i32");
    highest_bit - 1074
}

fn pow2(exponent: i32) -> f64 {
    debug_assert!((-1022..=1023).contains(&exponent));
    f64::from_bits(u64::try_from(exponent + 1023).expect("biased exponent is non-negative") << 52)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn uniform(
        precision: ScalarPrecision,
        format: ExecutablePrecision,
        overflow: OverflowPolicy,
    ) -> PrecisionExecutionSpec {
        PrecisionExecutionSpec::new(
            PrecisionPolicy::new(precision, precision, precision, precision),
            format.clone(),
            format.clone(),
            format.clone(),
            format,
            overflow,
        )
        .unwrap()
    }

    #[test]
    fn binary16_tie_rounds_to_even() {
        let spec = uniform(
            ScalarPrecision::F16,
            ExecutablePrecision::F16,
            OverflowPolicy::Reject,
        );
        let halfway = 1.0 + pow2(-11);
        assert_eq!(
            spec.quantize(PrecisionStage::Input, halfway)
                .unwrap()
                .to_bits(),
            1.0_f64.to_bits()
        );
    }

    #[test]
    fn bf16_tie_rounds_to_even() {
        let spec = uniform(
            ScalarPrecision::BF16,
            ExecutablePrecision::Bf16,
            OverflowPolicy::Reject,
        );
        let halfway = 1.0 + pow2(-8);
        assert_eq!(
            spec.quantize(PrecisionStage::Input, halfway)
                .unwrap()
                .to_bits(),
            1.0_f64.to_bits()
        );
    }

    #[test]
    fn f4_e2m1_is_explicit_and_executable() {
        let spec = uniform(
            ScalarPrecision::F4,
            ExecutablePrecision::F4E2M1,
            OverflowPolicy::Reject,
        );
        assert_eq!(
            spec.quantize(PrecisionStage::Input, 0.26)
                .unwrap()
                .to_bits(),
            0.5_f64.to_bits()
        );
        assert_eq!(
            spec.quantize(PrecisionStage::Input, 1.25)
                .unwrap()
                .to_bits(),
            1.0_f64.to_bits()
        );
        assert_eq!(
            spec.quantize(PrecisionStage::Input, 2.8).unwrap().to_bits(),
            3.0_f64.to_bits()
        );
    }

    #[test]
    fn f8_choice_is_bound_to_generic_f8_declaration() {
        let policy = PrecisionPolicy::new(
            ScalarPrecision::F8,
            ScalarPrecision::F8,
            ScalarPrecision::F8,
            ScalarPrecision::F8,
        );
        assert!(
            PrecisionExecutionSpec::new(
                policy,
                ExecutablePrecision::F8E4M3IeeeLike,
                ExecutablePrecision::F8E5M2,
                ExecutablePrecision::F8E4M3IeeeLike,
                ExecutablePrecision::F8E5M2,
                OverflowPolicy::Reject,
            )
            .is_ok()
        );
    }

    #[test]
    fn mismatched_concrete_format_fails_closed() {
        let result = PrecisionExecutionSpec::new(
            PrecisionPolicy::new(
                ScalarPrecision::F16,
                ScalarPrecision::F16,
                ScalarPrecision::F16,
                ScalarPrecision::F16,
            ),
            ExecutablePrecision::F32,
            ExecutablePrecision::F16,
            ExecutablePrecision::F16,
            ExecutablePrecision::F16,
            OverflowPolicy::Reject,
        );
        assert_eq!(
            result,
            Err(PrecisionReferenceError::PolicyMismatch {
                stage: PrecisionStage::Input,
                declared: ScalarPrecision::F16,
                executable: ScalarPrecision::F32,
            })
        );
    }

    #[test]
    fn i8_scale_and_overflow_are_explicit() {
        let spec = uniform(
            ScalarPrecision::I8,
            ExecutablePrecision::I8Symmetric { scale: 0.125 },
            OverflowPolicy::Saturate,
        );
        assert_eq!(
            spec.quantize(PrecisionStage::Input, 0.1875)
                .unwrap()
                .to_bits(),
            0.25_f64.to_bits()
        );
        assert_eq!(
            spec.quantize(PrecisionStage::Input, 99.0)
                .unwrap()
                .to_bits(),
            15.875_f64.to_bits()
        );
    }

    #[test]
    fn dot_product_rounds_every_declared_stage() {
        let spec = uniform(
            ScalarPrecision::F4,
            ExecutablePrecision::F4E2M1,
            OverflowPolicy::Saturate,
        );
        let report = spec.dot_product(&[1.25, 0.75], &[1.0, 1.0]).unwrap();
        assert_eq!(report.accumulation_trace(), &[1.0, 2.0]);
        assert_eq!(report.output().to_bits(), 2.0_f64.to_bits());
    }

    #[test]
    fn reject_overflow_does_not_silently_saturate() {
        let spec = uniform(
            ScalarPrecision::F4,
            ExecutablePrecision::F4E2M1,
            OverflowPolicy::Reject,
        );
        assert_eq!(
            spec.quantize(PrecisionStage::Input, 100.0),
            Err(PrecisionReferenceError::Overflow)
        );
    }
}
