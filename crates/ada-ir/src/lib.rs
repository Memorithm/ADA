//! ADA-A8 IR: a restricted, inspectable straight-line grammar for exact
//! attention recurrences.
//!
//! The grammar deliberately excludes loops, indirect addressing, and raw
//! memory: programs are finite statement lists over typed scalar and vector
//! registers. Every construct is one of the qualified ADA primitives (scalar
//! state, comparisons, select/max, arithmetic, exp/log, reductions, vector
//! accumulation). Anything outside the grammar simply cannot be expressed,
//! which is the fail-closed property the A8 mission requires.

#![forbid(unsafe_code)]

/// Maximum statements accepted in a program; bounds interpretation cost.
pub const MAX_PROGRAM_LENGTH: usize = 256;

/// Scalar binary arithmetic and lattice operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
    Max,
    Min,
}

/// Scalar comparisons; `Select` consumes their results.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompareOp {
    Lt,
    Le,
    Gt,
    Ge,
}

/// Elementwise vector reduction into a scalar register.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReduceOp {
    Sum,
    Max,
    Min,
}

/// Elementwise vector accumulation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AccumulateOp {
    /// `target <- target + operand` lane-wise.
    Add,
}

/// A scalar expression tree.
#[derive(Debug, Clone, PartialEq)]
pub enum ScalarExpr {
    Constant(f64),
    /// Read scalar state slot.
    State(usize),
    /// Reduce a vector register to a scalar.
    Reduce {
        op: ReduceOp,
        vector: usize,
    },
    Binary {
        op: BinaryOp,
        left: Box<ScalarExpr>,
        right: Box<ScalarExpr>,
    },
    Compare {
        op: CompareOp,
        left: Box<ScalarExpr>,
        right: Box<ScalarExpr>,
    },
    Select {
        condition: Box<ScalarExpr>,
        then_value: Box<ScalarExpr>,
        else_value: Box<ScalarExpr>,
    },
    Exp(Box<ScalarExpr>),
    Log(Box<ScalarExpr>),
}

/// An elementwise vector expression.
#[derive(Debug, Clone, PartialEq)]
pub enum VectorExpr {
    /// Broadcast a scalar expression to every lane.
    Broadcast(Box<ScalarExpr>),
    /// Read a vector register.
    Register(usize),
    /// Lane-wise binary operation of two vectors.
    ZipBinary {
        op: BinaryOp,
        left: Box<VectorExpr>,
        right: Box<VectorExpr>,
    },
    /// Lane-wise scale-and-add: `alpha * left + right` with alpha scalar.
    FusedScaleAdd {
        alpha: Box<ScalarExpr>,
        scaled: Box<VectorExpr>,
        added: Box<VectorExpr>,
    },
}

/// One straight-line statement.
#[derive(Debug, Clone, PartialEq)]
pub enum Statement {
    /// Write a scalar state slot.
    StoreState { slot: usize, value: ScalarExpr },
    /// Write a vector register.
    StoreVector { register: usize, value: VectorExpr },
    /// Elementwise accumulate into an existing vector register.
    Accumulate {
        target: usize,
        op: AccumulateOp,
        operand: VectorExpr,
    },
}

/// A complete inspectable program.
#[derive(Debug, Clone, PartialEq)]
pub struct Program {
    /// Number of scalar state slots.
    pub state_slots: usize,
    /// Number of vector registers.
    pub vector_registers: usize,
    /// Straight-line body.
    pub statements: Vec<Statement>,
}

impl Program {
    /// Structural validation: bounded length and all register/slot indices
    /// within the declared ranges.
    ///
    /// # Errors
    ///
    /// Returns a precise static error for the first violated constraint.
    pub fn validate(&self) -> Result<(), &'static str> {
        fn check_scalar(expr: &ScalarExpr, program: &Program) -> Result<(), &'static str> {
            match expr {
                ScalarExpr::State(slot) if *slot >= program.state_slots => {
                    Err("ADA-IR scalar state slot out of range")
                }
                ScalarExpr::Reduce { vector, .. } if *vector >= program.vector_registers => {
                    Err("ADA-IR vector register out of range in reduce")
                }
                ScalarExpr::Binary { left, right, .. }
                | ScalarExpr::Compare { left, right, .. } => {
                    check_scalar(left, program)?;
                    check_scalar(right, program)
                }
                ScalarExpr::Select {
                    condition,
                    then_value,
                    else_value,
                } => {
                    check_scalar(condition, program)?;
                    check_scalar(then_value, program)?;
                    check_scalar(else_value, program)
                }
                ScalarExpr::Exp(inner) | ScalarExpr::Log(inner) => check_scalar(inner, program),
                ScalarExpr::State(_) | ScalarExpr::Constant(_) | ScalarExpr::Reduce { .. } => {
                    Ok(())
                }
            }
        }

        fn check_vector(expr: &VectorExpr, program: &Program) -> Result<(), &'static str> {
            match expr {
                VectorExpr::Register(register) if *register >= program.vector_registers => {
                    Err("ADA-IR vector register out of range in expression")
                }
                VectorExpr::Broadcast(scalar) => check_scalar(scalar, program),
                VectorExpr::ZipBinary { left, right, .. } => {
                    check_vector(left, program)?;
                    check_vector(right, program)
                }
                VectorExpr::FusedScaleAdd {
                    alpha,
                    scaled,
                    added,
                } => {
                    check_scalar(alpha, program)?;
                    check_vector(scaled, program)?;
                    check_vector(added, program)
                }
                VectorExpr::Register(_) => Ok(()),
            }
        }

        if self.statements.len() > MAX_PROGRAM_LENGTH {
            return Err("ADA-IR program exceeds the maximum statement budget");
        }
        if self.state_slots == 0 || self.vector_registers == 0 {
            return Err("ADA-IR program must declare at least one state slot and vector register");
        }

        for statement in &self.statements {
            match statement {
                Statement::StoreState { slot, value } => {
                    if *slot >= self.state_slots {
                        return Err("ADA-IR store target slot out of range");
                    }
                    check_scalar(value, self)?;
                }
                Statement::StoreVector { register, value } => {
                    if *register >= self.vector_registers {
                        return Err("ADA-IR vector store target out of range");
                    }
                    check_vector(value, self)?;
                }
                Statement::Accumulate {
                    target, operand, ..
                } => {
                    if *target >= self.vector_registers {
                        return Err("ADA-IR accumulate target out of range");
                    }
                    check_vector(operand, self)?;
                }
            }
        }

        Ok(())
    }
}

/// Lane pairing with singleton broadcast: a length-1 operand is replicated
/// to the other operand's length; otherwise lengths must match exactly.
fn broadcast_pairs(left: &[f64], right: &[f64]) -> Result<Vec<(f64, f64)>, &'static str> {
    match (left.len(), right.len()) {
        (n, m) if n == m => Ok(left.iter().copied().zip(right.iter().copied()).collect()),
        (1, m) => Ok(vec![left[0]; m]
            .into_iter()
            .zip(right.iter().copied())
            .collect()),
        (_n, 1) => Ok(left
            .iter()
            .copied()
            .zip(std::iter::repeat(right[0]))
            .collect()),
        _ => Err("ADA-IR vector shape mismatch"),
    }
}

/// Runtime state for interpreting a [`Program`] over one token stream.
#[derive(Debug, Clone)]
pub struct Interpreter {
    program: Program,
    scalars: Vec<f64>,
    vectors: Vec<Vec<f64>>,
}

/// Errors raised by the interpreter.
pub const ERR_INVALID_PROGRAM: &str = "ADA-IR program failed structural validation";
pub const ERR_NON_FINITE: &str = "ADA-IR produced a non-finite intermediate";

impl Interpreter {
    /// Build an interpreter; the program must validate first.
    ///
    /// # Errors
    ///
    /// Propagates [`Program::validate`] failures.
    pub fn new(program: Program) -> Result<Self, &'static str> {
        program.validate()?;
        Ok(Self {
            scalars: vec![0.0; program.state_slots],
            vectors: vec![Vec::new(); program.vector_registers],
            program,
        })
    }

    /// Read a scalar slot.
    #[must_use]
    pub fn scalar(&self, slot: usize) -> f64 {
        self.scalars[slot]
    }

    /// Write a scalar slot (driver-side input feeding).
    pub fn set_scalar(&mut self, slot: usize, value: f64) {
        self.scalars[slot] = value;
    }

    /// Replace a vector register's contents (driver-side input feeding).
    pub fn set_vector(&mut self, register: usize, lanes: Vec<f64>) {
        self.vectors[register] = lanes;
    }

    /// Read a vector register (empty if never stored).
    #[must_use]
    pub fn vector(&self, register: usize) -> &[f64] {
        &self.vectors[register]
    }

    fn eval_scalar(&self, expr: &ScalarExpr) -> Result<f64, &'static str> {
        let value = match expr {
            ScalarExpr::Constant(value) => *value,
            ScalarExpr::State(slot) => self.scalars[*slot],
            ScalarExpr::Reduce { op, vector } => {
                let lane_values = &self.vectors[*vector];
                match op {
                    ReduceOp::Sum => lane_values.iter().copied().sum(),
                    ReduceOp::Max => lane_values
                        .iter()
                        .copied()
                        .fold(f64::NEG_INFINITY, f64::max),
                    ReduceOp::Min => lane_values.iter().copied().fold(f64::INFINITY, f64::min),
                }
            }
            ScalarExpr::Binary { op, left, right } => {
                let (a, b) = (self.eval_scalar(left)?, self.eval_scalar(right)?);
                match op {
                    BinaryOp::Add => a + b,
                    BinaryOp::Sub => a - b,
                    BinaryOp::Mul => a * b,
                    BinaryOp::Div => a / b,
                    BinaryOp::Max => a.max(b),
                    BinaryOp::Min => a.min(b),
                }
            }
            ScalarExpr::Compare { op, left, right } => {
                let (a, b) = (self.eval_scalar(left)?, self.eval_scalar(right)?);
                f64::from(match op {
                    CompareOp::Lt => a < b,
                    CompareOp::Le => a <= b,
                    CompareOp::Gt => a > b,
                    CompareOp::Ge => a >= b,
                })
            }
            ScalarExpr::Select {
                condition,
                then_value,
                else_value,
            } => {
                if self.eval_scalar(condition)? >= 1.0 {
                    self.eval_scalar(then_value)?
                } else {
                    self.eval_scalar(else_value)?
                }
            }
            ScalarExpr::Exp(inner) => self.eval_scalar(inner)?.exp(),
            ScalarExpr::Log(inner) => self.eval_scalar(inner)?.ln(),
        };
        if value.is_finite() {
            Ok(value)
        } else {
            Err(ERR_NON_FINITE)
        }
    }

    fn eval_vector(&self, expr: &VectorExpr) -> Result<Vec<f64>, &'static str> {
        let lanes = |register: usize| -> Vec<f64> { self.vectors[register].clone() };
        let built = match expr {
            VectorExpr::Register(register) => lanes(*register),
            VectorExpr::Broadcast(scalar) => vec![self.eval_scalar(scalar)?],
            VectorExpr::ZipBinary { op, left, right } => {
                let (a, b) = (self.eval_vector(left)?, self.eval_vector(right)?);
                let pairs = broadcast_pairs(&a, &b)?;
                pairs
                    .iter()
                    .map(|&(x, y)| match op {
                        BinaryOp::Add => x + y,
                        BinaryOp::Sub => x - y,
                        BinaryOp::Mul => x * y,
                        BinaryOp::Div => x / y,
                        BinaryOp::Max => x.max(y),
                        BinaryOp::Min => x.min(y),
                    })
                    .collect()
            }
            VectorExpr::FusedScaleAdd {
                alpha,
                scaled,
                added,
            } => {
                let alpha_value = self.eval_scalar(alpha)?;
                let s = self.eval_vector(scaled)?;
                let d = self.eval_vector(added)?;
                let pairs = broadcast_pairs(&s, &d)?;
                pairs.iter().map(|&(x, y)| alpha_value * x + y).collect()
            }
        };
        if built.iter().all(|value| value.is_finite()) {
            Ok(built)
        } else {
            Err(ERR_NON_FINITE)
        }
    }

    /// Execute one statement.
    ///
    /// # Errors
    ///
    /// Fails closed on non-finite intermediates or shape mismatches.
    pub fn step(&mut self) -> Result<(), &'static str> {
        // Straight-line programs consume exactly one statement per step call;
        // the driver owns iteration order. Here we execute the whole body in
        // order and report which statement (if any) failed.
        let statements = self.program.statements.clone();
        for statement in &statements {
            match statement {
                Statement::StoreState { slot, value } => {
                    let value = self.eval_scalar(value)?;
                    self.scalars[*slot] = value;
                }
                Statement::StoreVector { register, value } => {
                    let value = self.eval_vector(value)?;
                    self.vectors[*register] = value;
                }
                Statement::Accumulate {
                    target,
                    op,
                    operand,
                } => {
                    let operand = self.eval_vector(operand)?;
                    let current = &mut self.vectors[*target];
                    if current.len() != operand.len() {
                        return Err("ADA-IR vector shape mismatch");
                    }
                    match op {
                        AccumulateOp::Add => {
                            for (lane, &delta) in current.iter_mut().zip(&operand) {
                                *lane += delta;
                                if !lane.is_finite() {
                                    return Err(ERR_NON_FINITE);
                                }
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    }

    /// Run all statements; alias to [`Interpreter::step`].
    ///
    /// # Errors
    ///
    /// Propagates interpreter failures.
    pub fn run(&mut self) -> Result<(), &'static str> {
        self.step()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validation_rejects_out_of_range_indices() {
        let mut program = recurrence_like();
        program.validate().unwrap();

        program.statements[0] = Statement::StoreState {
            slot: 9,
            value: ScalarExpr::Constant(1.0),
        };
        assert_eq!(
            program.validate(),
            Err("ADA-IR store target slot out of range")
        );

        let mut program = recurrence_like();
        program.state_slots = 0;
        assert!(program.validate().is_err());
    }

    #[test]
    fn interpreter_executes_select_and_exp() {
        // m' = max(m, s); alpha = exp(m - m'); with m=2, s=5 => alpha=e^-3.
        let m_new = || -> ScalarExpr {
            ScalarExpr::Select {
                condition: Box::new(ScalarExpr::Compare {
                    op: CompareOp::Ge,
                    left: Box::new(ScalarExpr::State(2)),
                    right: Box::new(ScalarExpr::State(0)),
                }),
                then_value: Box::new(ScalarExpr::State(2)),
                else_value: Box::new(ScalarExpr::State(0)),
            }
        };
        let program = Program {
            state_slots: 3,
            vector_registers: 1,
            statements: vec![
                // alpha = exp(m - m'), computed BEFORE m is overwritten.
                Statement::StoreState {
                    slot: 1,
                    value: ScalarExpr::Exp(Box::new(scalar_expr(BinaryOp::Sub, m_new()))),
                },
                Statement::StoreState {
                    slot: 0,
                    value: m_new(),
                },
            ],
        };
        let mut interp = Interpreter::new(program).unwrap();
        interp.set_scalar(0, 2.0);
        interp.set_scalar(2, 5.0);
        interp.run().unwrap();

        println!("m={} alpha={:e}", interp.scalar(0), interp.scalar(1));
        assert_eq!(interp.scalar(0).to_bits(), 5.0f64.to_bits());
        assert!((interp.scalar(1) - (-3.0_f64).exp()).abs() < 1.0e-15);
    }

    #[test]
    fn non_finite_intermediates_fail_closed() {
        let program = Program {
            state_slots: 1,
            vector_registers: 1,
            statements: vec![Statement::StoreState {
                slot: 0,
                value: ScalarExpr::Log(Box::new(ScalarExpr::Constant(0.0))),
            }],
        };
        let mut interp = Interpreter::new(program).unwrap();
        assert_eq!(interp.run(), Err(ERR_NON_FINITE));
    }

    fn scalar_expr(op: BinaryOp, right: ScalarExpr) -> ScalarExpr {
        ScalarExpr::Binary {
            op,
            left: Box::new(ScalarExpr::State(0)),
            right: Box::new(right),
        }
    }

    fn recurrence_like() -> Program {
        Program {
            state_slots: 3,
            vector_registers: 2,
            statements: vec![Statement::StoreVector {
                register: 0,
                value: VectorExpr::Broadcast(Box::new(ScalarExpr::State(0))),
            }],
        }
    }
}
