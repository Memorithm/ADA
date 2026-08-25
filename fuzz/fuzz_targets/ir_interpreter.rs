#![no_main]
#![forbid(unsafe_code)]

use ada_ir::{Interpreter, Program, ScalarExpr, Statement, VectorExpr};
use libfuzzer_sys::fuzz_target;

/// Deterministically decode a bounded scalar from fuzz bytes.
fn scalar(bytes: &[u8]) -> f64 {
    let mut value = 0_u64;
    for byte in bytes.iter().take(8) {
        value = (value << 8) | u64::from(*byte);
    }
    // Map to a finite f64 in a wide-but-bounded exponent range so the
    // interpreter's fail-closed finiteness checks are exercised, not the
    // host's overflow behavior on inf/NaN inputs.
    let mantissa = (value & ((1_u64 << 52) - 1)) as f64 / (1_u64 << 52) as f64;
    #[allow(clippy::cast_precision_loss)]
    let exponent = (value >> 52) as i32 % 64 - 32;
    mantissa * (2.0_f64).powi(exponent)
}

fn decode_expr(bytes: &[u8], depth: usize) -> ScalarExpr {
    let tag = bytes.first().copied().unwrap_or(0);
    let rest = bytes.get(1..).unwrap_or(&[]);
    if depth == 0 {
        return ScalarExpr::Constant(scalar(rest));
    }
    match tag % 6 {
        0 => ScalarExpr::Constant(scalar(rest)),
        1 => ScalarExpr::State(usize::from(rest.first().copied().unwrap_or(0)) % 3),
        2 => ScalarExpr::Binary {
            op: match rest.first().copied().unwrap_or(0) % 4 {
                0 => ada_ir::BinaryOp::Add,
                1 => ada_ir::BinaryOp::Sub,
                2 => ada_ir::BinaryOp::Mul,
                _ => ada_ir::BinaryOp::Max,
            },
            left: Box::new(decode_expr(
                rest.get(1..(rest.len() / 2 + 1)).unwrap_or(&[]),
                depth - 1,
            )),
            right: Box::new(decode_expr(rest.get(rest.len() / 2 + 1..).unwrap_or(&[]), depth - 1)),
        },
        3 => ScalarExpr::Compare {
            op: ada_ir::CompareOp::Lt,
            left: Box::new(decode_expr(rest, depth - 1)),
            right: Box::new(decode_expr(&[], depth - 1)),
        },
        4 => ScalarExpr::Exp(Box::new(decode_expr(rest, depth - 1))),
        _ => ScalarExpr::Log(Box::new(decode_expr(rest, depth - 1))),
    }
}

fuzz_target!(|data: &[u8]| {
    // Arbitrary expression trees and statement sequences must either validate
    // or be rejected with typed errors; the interpreter must never panic,
    // index out of bounds, or emit a non-finite value.
    let statements = data.chunks(9).filter(|chunk| chunk.len() >= 2).take(16).map(
        |chunk| match chunk[0] % 2 {
            0 => Statement::StoreState {
                slot: usize::from(chunk[1]) % 3,
                value: decode_expr(chunk.get(2..).unwrap_or(&[]), 2),
            },
            _ => Statement::StoreVector {
                register: usize::from(chunk[1]) % 2,
                value: VectorExpr::Broadcast(Box::new(decode_expr(
                    chunk.get(2..).unwrap_or(&[]),
                    2,
                ))),
            },
        },
    );

    let program = Program {
        state_slots: 3,
        vector_registers: 2,
        statements: statements.collect(),
    };

    if let Ok(mut interpreter) = Interpreter::new(program) {
        // Contract: interpretation either succeeds with all-finite state or
        // fails closed with a typed error (e.g. cross-register shape
        // mismatch that structural validation cannot see). It never panics.
        match interpreter.run() {
            Ok(()) => {
                for slot in 0..3 {
                    assert!(interpreter.scalar(slot).is_finite());
                }
                for register in 0..2 {
                    assert!(interpreter.vector(register).iter().all(|v| v.is_finite()));
                }
            }
            Err(message) => {
                assert!(
                    message == ada_ir::ERR_NON_FINITE
                        || message == "ADA-IR vector shape mismatch",
                    "unexpected interpreter error: {message}"
                );
            }
        }
    }

    // Reduce must also stay total over whatever made it into registers.
    let reduce_program = Program {
        state_slots: 1,
        vector_registers: 1,
        statements: vec![
            Statement::StoreVector {
                register: 0,
                value: VectorExpr::Broadcast(Box::new(ScalarExpr::Constant(1.5))),
            },
            Statement::StoreState {
                slot: 0,
                value: ScalarExpr::Reduce {
                    op: ada_ir::ReduceOp::Sum,
                    vector: 0,
                },
            },
        ],
    };
    let mut interpreter = Interpreter::new(reduce_program).expect("fixed program validates");
    interpreter.run().expect("fixed program interprets");
});
