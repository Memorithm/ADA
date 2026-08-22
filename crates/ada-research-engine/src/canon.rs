//! Canonical form, canonical identity and content digests for candidates.
//!
//! Two expressions are **canonically equivalent** iff their normalized trees
//! are identical. Normalization applies a deliberately conservative, fully
//! deterministic rewrite set:
//!
//! 1. bottom-up constant folding with the exact arithmetic the interpreter
//!    uses (folding is bit-identical to evaluating the folded subexpression);
//!    a fold that would produce a non-finite value is left symbolic so later
//!    gates still catch it;
//! 2. neutral-element elimination that is bitwise safe for finite operands
//!    (`x - +0 -> x`, `x * 1 -> x`, `max(a, a) -> a`);
//! 3. commutative operand sorting for `Add`, `Mul` and `Max` by canonical
//!    string. IEEE-754 addition/multiplication are commutative bitwise for
//!    finite operands (and our interpreter rejects non-finite values), and
//!    [`crate::Expr::Max`] is implemented symmetrically on ties.
//!
//! Identity is the SHA-256 of a versioned canonical byte stream built from the
//! normalized output tuple (`candidate_id`). Hashes are content-integrity identifiers;
//! equality decisions always compare canonical strings, never hashes.

use std::fmt::Write as _;
use std::ops::{Add, Mul, Sub};

use sha2::{Digest, Sha256};

use crate::candidate::Candidate;
use crate::expr::Expr;

/// Domain-separation prefix for candidate identity streams.
pub const CANDIDATE_DIGEST_MAGIC: &[u8] = b"ADA-RESEARCH-CANDIDATE-v2\0";

/// Canonical s-expression string of an expression (not necessarily normalized).
///
/// Constants are encoded through their exact bit pattern so `-0.0` and `0.0`
/// remain distinguishable and no float formatting ambiguity can arise.
#[must_use]
pub fn canon_string(expr: &Expr) -> String {
    let mut out = String::new();
    write_canon(expr, &mut out);
    out
}

fn write_canon(expr: &Expr, out: &mut String) {
    let mut pending = vec![CanonTask::Expression(expr)];
    while let Some(task) = pending.pop() {
        match task {
            CanonTask::Text(text) => out.push_str(text),
            CanonTask::Expression(expression) => match expression {
                Expr::Var(index) => {
                    out.push_str("(v ");
                    let _ = write!(out, "{index}");
                    out.push(')');
                }
                Expr::Const(value) => {
                    out.push_str("(c ");
                    let _ = write!(out, "{:016x}", value.to_bits());
                    out.push(')');
                }
                Expr::Exp(inner) => {
                    pending.push(CanonTask::Text(")"));
                    pending.push(CanonTask::Expression(inner));
                    pending.push(CanonTask::Text("(exp "));
                }
                Expr::Add(lhs, rhs) => push_binary(&mut pending, "(add ", lhs, rhs),
                Expr::Sub(lhs, rhs) => push_binary(&mut pending, "(sub ", lhs, rhs),
                Expr::Mul(lhs, rhs) => push_binary(&mut pending, "(mul ", lhs, rhs),
                Expr::Max(lhs, rhs) => push_binary(&mut pending, "(max ", lhs, rhs),
            },
        }
    }
}

enum CanonTask<'a> {
    Expression(&'a Expr),
    Text(&'static str),
}

fn push_binary<'a>(
    pending: &mut Vec<CanonTask<'a>>,
    opening: &'static str,
    lhs: &'a Expr,
    rhs: &'a Expr,
) {
    pending.push(CanonTask::Text(")"));
    pending.push(CanonTask::Expression(rhs));
    pending.push(CanonTask::Text(" "));
    pending.push(CanonTask::Expression(lhs));
    pending.push(CanonTask::Text(opening));
}

/// Normalize an expression into canonical form.
#[must_use]
pub fn normalize(expr: &Expr) -> Expr {
    match expr {
        Expr::Var(_) | Expr::Const(_) => expr.clone(),
        Expr::Exp(inner) => {
            let inner = normalize(inner);
            if let Expr::Const(value) = inner {
                let folded = value.exp();
                if folded.is_finite() {
                    return Expr::Const(folded);
                }
            }
            Expr::Exp(Box::new(inner))
        }
        Expr::Add(lhs, rhs) => {
            let lhs = normalize(lhs);
            let rhs = normalize(rhs);
            if let Some(folded) = fold_pair(&lhs, &rhs, f64::add) {
                return Expr::Const(folded);
            }
            sort_commutative(Expr::Add, lhs, rhs)
        }
        Expr::Mul(lhs, rhs) => {
            let lhs = normalize(lhs);
            let rhs = normalize(rhs);
            if let Some(folded) = fold_pair(&lhs, &rhs, f64::mul) {
                return Expr::Const(folded);
            }
            if is_one_const(&lhs) {
                return rhs;
            }
            if is_one_const(&rhs) {
                return lhs;
            }
            sort_commutative(Expr::Mul, lhs, rhs)
        }
        Expr::Max(lhs, rhs) => {
            let lhs = normalize(lhs);
            let rhs = normalize(rhs);
            if lhs == rhs {
                return lhs;
            }
            sort_commutative(Expr::Max, lhs, rhs)
        }
        Expr::Sub(lhs, rhs) => {
            let lhs = normalize(lhs);
            let rhs = normalize(rhs);
            if let Some(folded) = fold_pair(&lhs, &rhs, f64::sub) {
                return Expr::Const(folded);
            }
            if let Expr::Const(zero) = rhs {
                if zero.to_bits() == 0 {
                    return lhs;
                }
            }
            Expr::Sub(Box::new(lhs), Box::new(rhs))
        }
    }
}

/// Fold two constant children with `op`; returns `None` when either side is
/// not a constant or the folded value would be non-finite.
fn fold_pair(lhs: &Expr, rhs: &Expr, op: fn(f64, f64) -> f64) -> Option<f64> {
    if let (Expr::Const(a), Expr::Const(b)) = (lhs, rhs) {
        let folded = op(*a, *b);
        if folded.is_finite() {
            return Some(folded);
        }
    }
    None
}

fn is_one_const(expr: &Expr) -> bool {
    matches!(expr, Expr::Const(value) if value.to_bits() == 1.0_f64.to_bits())
}

fn sort_commutative(build: fn(Box<Expr>, Box<Expr>) -> Expr, lhs: Expr, rhs: Expr) -> Expr {
    let (first, second) = if canon_string(&lhs) <= canon_string(&rhs) {
        (lhs, rhs)
    } else {
        (rhs, lhs)
    };
    build(Box::new(first), Box::new(second))
}

/// Normalize every output without changing output order.
#[must_use]
pub fn normalize_candidate(candidate: &Candidate) -> Candidate {
    Candidate::new(candidate.outputs().iter().map(normalize).collect())
}

/// Canonical representation of an ordered candidate output tuple.
#[must_use]
pub fn candidate_canon_string(candidate: &Candidate) -> String {
    let mut out = String::from("(candidate");
    for expression in candidate.outputs() {
        out.push_str(" (out ");
        write_canon(expression, &mut out);
        out.push(')');
    }
    out.push(')');
    out
}

/// Canonical identity digest (hex SHA-256) over the normalized candidate.
#[must_use]
pub fn candidate_id(candidate: &Candidate) -> String {
    let normalized = normalize_candidate(candidate);
    let mut stream = Vec::new();
    stream.extend_from_slice(CANDIDATE_DIGEST_MAGIC);
    stream.extend_from_slice(candidate_canon_string(&normalized).as_bytes());
    let digest = Sha256::digest(&stream);
    hex(&digest)
}

/// Lowercase hex encoding.
#[must_use]
pub fn hex(bytes: &[u8]) -> String {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        out.push(DIGITS[(byte >> 4) as usize] as char);
        out.push(DIGITS[(byte & 0x0f) as usize] as char);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn v(index: usize) -> Expr {
        Expr::Var(index)
    }

    fn c(expression: Expr) -> Candidate {
        Candidate::scalar(expression)
    }

    #[test]
    fn commutative_variants_deduplicate() {
        let a = Expr::Add(Box::new(v(0)), Box::new(v(1)));
        let b = Expr::Add(Box::new(v(1)), Box::new(v(0)));
        assert_eq!(normalize(&a), normalize(&b));
        assert_eq!(candidate_id(&c(a)), candidate_id(&c(b)));

        let m1 = Expr::Mul(Box::new(v(2)), Box::new(Expr::Exp(Box::new(v(3)))));
        let m2 = Expr::Mul(Box::new(Expr::Exp(Box::new(v(3)))), Box::new(v(2)));
        assert_eq!(candidate_id(&c(m1)), candidate_id(&c(m2)));

        let x1 = Expr::Max(Box::new(v(0)), Box::new(v(2)));
        let x2 = Expr::Max(Box::new(v(2)), Box::new(v(0)));
        assert_eq!(candidate_id(&c(x1)), candidate_id(&c(x2)));
    }

    #[test]
    fn nested_commutative_variants_deduplicate() {
        // (a+b)+c vs c+(b+a): associativity is NOT rewritten, but operand
        // sorting inside each Add makes these two share a canonical form only
        // when their child sets match; here they do after sorting both adds.
        let a = Expr::Add(
            Box::new(Expr::Add(Box::new(v(0)), Box::new(v(1)))),
            Box::new(v(2)),
        );
        let b = Expr::Add(
            Box::new(v(2)),
            Box::new(Expr::Add(Box::new(v(1)), Box::new(v(0)))),
        );
        assert_eq!(candidate_id(&c(a)), candidate_id(&c(b)));
    }

    #[test]
    fn constants_fold_exactly() {
        let e = Expr::Add(Box::new(Expr::Const(1.0)), Box::new(Expr::Const(2.0)));
        assert_eq!(normalize(&e), Expr::Const(3.0));

        // Non-finite folds stay symbolic so gates can reject them.
        let big = Expr::Mul(Box::new(Expr::Const(f64::MAX)), Box::new(Expr::Const(4.0)));
        assert_ne!(normalize(&big), Expr::Const(f64::INFINITY));
    }

    #[test]
    fn only_bitwise_safe_neutral_elements_are_removed() {
        let add_zero = Expr::Add(Box::new(v(1)), Box::new(Expr::Const(0.0)));
        assert_ne!(normalize(&add_zero), v(1));
        let signed_zero = Expr::Add(Box::new(Expr::Const(-0.0)), Box::new(Expr::Const(0.0)));
        assert_eq!(
            signed_zero.eval(&[]).unwrap().to_bits(),
            normalize(&signed_zero).eval(&[]).unwrap().to_bits()
        );

        let sub_zero = Expr::Sub(Box::new(v(1)), Box::new(Expr::Const(0.0)));
        assert_eq!(normalize(&sub_zero), v(1));

        let mul_one = Expr::Mul(Box::new(Expr::Const(1.0)), Box::new(v(1)));
        assert_eq!(normalize(&mul_one), v(1));

        // This tempting rewrite is deliberately forbidden: if the shared
        // subtree overflows, the raw interpreter rejects it while +0 would
        // incorrectly hide the failure.
        let self_sub = Expr::Sub(Box::new(v(1)), Box::new(v(1)));
        assert_eq!(normalize(&self_sub), self_sub);

        let self_max = Expr::Max(Box::new(v(1)), Box::new(v(1)));
        assert_eq!(normalize(&self_max), v(1));
    }

    #[test]
    fn non_equivalent_expressions_keep_distinct_ids() {
        let a = Expr::Add(Box::new(v(0)), Box::new(v(1)));
        let b = Expr::Sub(Box::new(v(0)), Box::new(v(1)));
        assert_ne!(candidate_id(&c(a)), candidate_id(&c(b)));

        let zero_a = Expr::Const(0.0);
        let zero_b = Expr::Const(-0.0);
        assert_ne!(candidate_id(&c(zero_a)), candidate_id(&c(zero_b)));
    }

    #[test]
    fn canon_string_is_stable_and_unambiguous() {
        let e = Expr::Sub(Box::new(v(0)), Box::new(Expr::Const(-0.5)));
        assert_eq!(canon_string(&e), "(sub (v 0) (c bfe0000000000000))");
    }

    #[test]
    fn ids_are_hex_sha256_length() {
        let id = candidate_id(&c(v(0)));
        assert_eq!(id.len(), 64);
        assert!(
            id.chars()
                .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase())
        );
    }

    #[test]
    fn normalization_is_idempotent() {
        let e = Expr::Add(
            Box::new(v(1)),
            Box::new(Expr::Add(Box::new(v(0)), Box::new(Expr::Const(0.0)))),
        );
        let once = normalize(&e);
        let twice = normalize(&once);
        assert_eq!(once, twice);
    }

    #[test]
    fn output_order_is_part_of_candidate_identity() {
        let first = Candidate::new(vec![v(0), v(1)]);
        let second = Candidate::new(vec![v(1), v(0)]);
        assert_ne!(
            candidate_canon_string(&first),
            candidate_canon_string(&second)
        );
        assert_ne!(candidate_id(&first), candidate_id(&second));
    }
}
