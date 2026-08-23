//! Fixed human-supplied candidate list.
//!
//! Used by tests and by humans who want to submit a specific expression into
//! the pipeline. A manual submission is just another proposal source: it is
//! normalized, validated and falsified identically to automated candidates,
//! with no privileged status.

use crate::candidate::Candidate;
use crate::canon::candidate_canon_string;
use crate::expr::Expr;
use crate::proposer::{
    CandidateProposer, ProposalContext, ProposalDescriptor, ProposalSource, ProposalSourceKind,
    RawProposal,
};

/// A labeled manual candidate.
#[derive(Debug, Clone, PartialEq)]
pub struct ManualCandidate {
    pub label: String,
    pub candidate: Candidate,
}

impl ManualCandidate {
    /// Convenience constructor.
    #[must_use]
    pub fn new(label: impl Into<String>, expression: Expr) -> Self {
        Self::recurrence(label, Candidate::scalar(expression))
    }

    /// Construct a fixed multi-output recurrence submission.
    #[must_use]
    pub fn recurrence(label: impl Into<String>, candidate: Candidate) -> Self {
        Self {
            label: label.into(),
            candidate,
        }
    }
}

/// Emits its fixed list in order, then reports exhaustion.
#[derive(Debug)]
pub struct ManualProposer {
    candidates: Vec<ManualCandidate>,
    next: usize,
}

impl ManualProposer {
    /// Build from an ordered list.
    #[must_use]
    pub fn new(candidates: Vec<ManualCandidate>) -> Self {
        Self {
            candidates,
            next: 0,
        }
    }
}

impl CandidateProposer for ManualProposer {
    fn descriptor(&self) -> ProposalDescriptor {
        let mut configuration = String::from("manual-list-v1");
        for item in &self.candidates {
            configuration.push('|');
            configuration.push_str(&item.label.len().to_string());
            configuration.push(':');
            configuration.push_str(&item.label);
            configuration.push(':');
            configuration.push_str(&candidate_canon_string(&item.candidate));
        }
        ProposalDescriptor::new(ProposalSourceKind::ManualList, configuration)
    }

    fn propose(&mut self, _context: &ProposalContext<'_>) -> Option<RawProposal> {
        let index = self.next;
        if index >= self.candidates.len() {
            return None;
        }
        self.next += 1;
        let candidate = &self.candidates[index];
        Some(RawProposal {
            candidate: candidate.candidate.clone(),
            source: ProposalSource::Manual {
                label: candidate.label.clone(),
            },
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grammar::{GrammarSpec, OperatorSet};

    #[test]
    fn emits_in_order_then_exhausts() {
        let mut proposer = ManualProposer::new(vec![
            ManualCandidate::new("first", Expr::Var(0)),
            ManualCandidate::new("second", Expr::Var(1)),
        ]);
        let grammar = GrammarSpec {
            inputs: vec!["a".into(), "b".into()],
            outputs: vec!["value".into()],
            constants: vec![],
            operators: OperatorSet::all(),
            max_nodes: 8,
            max_depth: 6,
            version: 1,
        };
        let context = ProposalContext {
            grammar: &grammar,
            budget: &crate::SearchBudget::tiny(),
            feedback: &[],
        };
        let first = proposer.propose(&context).unwrap();
        assert!(matches!(
            first.source,
            ProposalSource::Manual { ref label } if label == "first"
        ));
        let second = proposer.propose(&context).unwrap();
        assert_eq!(second.candidate, Candidate::scalar(Expr::Var(1)));
        assert!(proposer.propose(&context).is_none());
        assert!(proposer.propose(&context).is_none());
    }
}
