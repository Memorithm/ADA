//! Problem corpora and the evaluation protocol boundary.
//!
//! A [`ProblemCorpus`] is a fixed, deterministic set of evaluation cases under
//! one protocol. Corpora are the *only* place where oracle outputs live, and
//! they are owned exclusively by the engine's evaluation phase: the
//! [`crate::CandidateProposer`] API never mentions a corpus type, so a
//! proposer cannot read expected outputs — including the adversarial holdout
//! — through the type system.

use crate::candidate::Candidate;
use crate::canon::hex;
use crate::digest_writer::DigestWriter;
use crate::expr::ExecError;

/// One candidate/oracle comparison produced by a corpus.
#[derive(Debug, Clone, PartialEq)]
pub struct CaseRun {
    /// The candidate's aggregate output for this case (e.g. final state).
    pub candidate_output: Vec<f64>,
    /// The oracle's aggregate output for this case.
    pub oracle_output: Vec<f64>,
    /// Largest absolute error observed under this case's protocol
    /// (for multi-step corpora, over every step).
    pub max_abs_error: f64,
    /// Largest relative error observed under this case's protocol,
    /// using `scale = max(|oracle|, 1e-300)`.
    pub max_rel_error: f64,
}

/// Deterministic corpus failure: the candidate could not be executed on a
/// case (non-finite intermediate, wrong variable count). This is treated as
/// falsification at the gate that surfaced it; it is never coerced into a
/// numeric value.
#[derive(Debug, Clone, PartialEq)]
pub struct CorpusFailure {
    /// Case index that failed.
    pub case_index: usize,
    /// Why execution failed.
    pub reason: ExecError,
}

impl std::fmt::Display for CorpusFailure {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "execution failed on case {}: {}",
            self.case_index, self.reason
        )
    }
}

impl std::error::Error for CorpusFailure {}

/// Relative-error scale floor shared by all corpora so error metrics are
/// always finite.
pub const REL_SCALE_FLOOR: f64 = 1.0e-300;

/// Compute relative error with the standard finite-safe scale.
#[must_use]
pub fn relative_error(candidate: f64, oracle: f64) -> f64 {
    let scale = oracle.abs().max(REL_SCALE_FLOOR);
    (candidate - oracle).abs() / scale
}

/// A deterministic evaluation corpus under one problem protocol.
///
/// Implementations must be pure functions of `(expression, case index)`:
/// no wall-clock time, no global state, no randomness at evaluation time.
pub trait ProblemCorpus {
    /// Role label used in manifests (`"discovery"`, `"probe"`, `"oracle"`,
    /// `"adversarial_holdout"`).
    fn role(&self) -> &'static str;

    /// Number of cases.
    fn len(&self) -> usize;

    /// Whether the corpus is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Execute the candidate expression on case `index` and compare against
    /// the oracle output computed by the corpus itself.
    ///
    /// # Errors
    ///
    /// Returns [`CorpusFailure`] when the candidate cannot be executed on
    /// this case (non-finite intermediate values are failures, never data).
    fn run_case(&self, candidate: &Candidate, index: usize) -> Result<CaseRun, CorpusFailure>;

    /// Labeled input snapshot of a case, used only for counterexample records.
    fn describe_case(&self, index: usize) -> Vec<(String, f64)>;

    /// The oracle's aggregate output for a case, computable without any
    /// candidate. Used to populate counterexamples even when the candidate
    /// fails to execute.
    fn oracle_output(&self, index: usize) -> Vec<f64>;

    /// Stable content digest covering protocol identity, inputs and oracle
    /// outputs of every case.
    #[must_use]
    fn digest(&self) -> String {
        let mut writer = DigestWriter::new(b"ADA-CORPUS-v1");
        // Ignore encoding-length errors: usize fits u64 on supported targets.
        let _ = writer.str(self.role());
        writer.tag(0x01);
        self.digest_cases(&mut writer);
        hex(&writer.finish())
    }

    /// Protocol-specific digest payload (case data). Implementations append
    /// every input and oracle value in fixed order.
    fn digest_cases(&self, writer: &mut DigestWriter);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn relative_error_uses_scale_floor() {
        assert_eq!(relative_error(3.0, 3.0).to_bits(), 0.0f64.to_bits());
        let tiny = relative_error(2.0, 0.0);
        assert!(tiny.is_finite() && tiny > 1.0e299);
        let normal = relative_error(1.5, 1.0);
        assert!((normal - 0.5).abs() < 1e-15);
    }
}
