//! Exhaustive, canonical-deduplicated grammar enumeration.
//!
//! Generates every expression in the grammar up to a small node budget,
//! ordered by increasing node count and, within a count, by construction
//! order (fixed operator order, fixed leaf order). Generated expressions are
//! normalized before emission so canonically equivalent variants are emitted
//! exactly once. The proposer is fully deterministic and ignores feedback.
//!
//! The node budget must stay small: enumeration is exponential in nodes.
//! Its role is baseline coverage of shallow structure plus a source of
//! building blocks; deeper discovery belongs to the evolutionary proposer.

use std::collections::BTreeSet;

use crate::candidate::Candidate;
use crate::canon::{candidate_canon_string, canon_string, normalize};
use crate::expr::Expr;
use crate::grammar::GrammarSpec;
use crate::proposer::{
    CandidateProposer, ProposalContext, ProposalDescriptor, ProposalSource, ProposalSourceKind,
    RawProposal,
};

type BinaryBuild = fn(Box<Expr>, Box<Expr>) -> Expr;

/// Configuration of the enumerative proposer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EnumerativeConfig {
    /// Maximum node count to enumerate (inclusive). Keep this small.
    pub max_nodes: usize,
    /// Hard cap on emissions; the proposer reports exhaustion afterwards.
    pub max_emissions: usize,
}

impl Default for EnumerativeConfig {
    fn default() -> Self {
        Self {
            max_nodes: 5,
            max_emissions: 20_000,
        }
    }
}

/// Deterministic exhaustive enumerator over a grammar.
#[derive(Debug)]
pub struct EnumerativeProposer {
    config: EnumerativeConfig,
    outbox: Vec<Candidate>,
    next: usize,
    started: bool,
}

impl EnumerativeProposer {
    /// Build the proposer; generation happens lazily on first pull.
    #[must_use]
    pub fn new(config: EnumerativeConfig) -> Self {
        Self {
            config,
            outbox: Vec::new(),
            next: 0,
            started: false,
        }
    }

    fn generate(&mut self, grammar: &GrammarSpec) {
        let effective_max = self.config.max_nodes.min(grammar.max_nodes);
        let mut seen: BTreeSet<String> = BTreeSet::new();
        let mut levels: Vec<Vec<Expr>> = Vec::new();

        // Level 1: leaves (variables first, then constants).
        let mut leaves: Vec<Expr> = Vec::new();
        for index in 0..grammar.leaf_count() {
            if let Some(leaf) = grammar.leaf(index) {
                push_unique(&mut leaves, &mut seen, grammar, &leaf);
            }
        }
        levels.push(leaves);

        let mut builds: Vec<BinaryBuild> = Vec::new();
        if grammar.operators.add {
            builds.push(Expr::Add);
        }
        if grammar.operators.sub {
            builds.push(Expr::Sub);
        }
        if grammar.operators.mul {
            builds.push(Expr::Mul);
        }
        if grammar.operators.max {
            builds.push(Expr::Max);
        }
        let wrap_exp = grammar.operators.exp;

        for size in 2..=effective_max.max(1) {
            let mut current: Vec<Expr> = Vec::new();

            if wrap_exp {
                for inner in &levels[size - 2] {
                    let candidate = Expr::Exp(Box::new(inner.clone()));
                    push_unique(&mut current, &mut seen, grammar, &candidate);
                }
            }

            for left_size in 1..size.saturating_sub(1) {
                let right_size = size - 1 - left_size;
                if right_size == 0 || right_size > levels.len() || left_size > levels.len() {
                    continue;
                }
                for lhs in &levels[left_size - 1] {
                    for rhs in &levels[right_size - 1] {
                        for build in &builds {
                            let candidate = build(Box::new(lhs.clone()), Box::new(rhs.clone()));
                            push_unique(&mut current, &mut seen, grammar, &candidate);
                        }
                    }
                }
            }

            levels.push(current);
            let total: usize = levels.iter().map(Vec::len).sum();
            if total > self.config.max_emissions.saturating_mul(8) {
                break;
            }
        }

        let mut scalar_forms: Vec<Expr> = levels.into_iter().flatten().collect();
        // Candidate tuple enumeration is deliberately bounded. The scalar
        // pool is general grammar output, not target-specific seed data.
        scalar_forms.truncate(256);
        self.outbox = compose_candidates(
            &scalar_forms,
            grammar.output_count(),
            grammar.max_nodes,
            self.config.max_emissions,
        );
        self.outbox.truncate(self.config.max_emissions);
        self.next = 0;
        self.started = true;
    }
}

fn compose_candidates(
    scalar_forms: &[Expr],
    output_count: usize,
    max_nodes: usize,
    max_emissions: usize,
) -> Vec<Candidate> {
    fn visit(
        scalar_forms: &[Expr],
        output_count: usize,
        max_nodes: usize,
        maximum_combinations: usize,
        explored: &mut usize,
        current: &mut Vec<Expr>,
        candidates: &mut Vec<Candidate>,
    ) {
        if *explored >= maximum_combinations {
            return;
        }
        if current.len() == output_count {
            *explored += 1;
            let candidate = Candidate::new(current.clone());
            if candidate.node_count() <= max_nodes {
                candidates.push(candidate);
            }
            return;
        }
        for expression in scalar_forms {
            current.push(expression.clone());
            visit(
                scalar_forms,
                output_count,
                max_nodes,
                maximum_combinations,
                explored,
                current,
                candidates,
            );
            current.pop();
            if *explored >= maximum_combinations {
                break;
            }
        }
    }

    if scalar_forms.is_empty() || output_count == 0 || max_emissions == 0 {
        return Vec::new();
    }
    let maximum_combinations = max_emissions.saturating_mul(64).max(max_emissions);
    let mut explored = 0usize;
    let mut current = Vec::with_capacity(output_count);
    let mut candidates = Vec::new();

    visit(
        scalar_forms,
        output_count,
        max_nodes,
        maximum_combinations,
        &mut explored,
        &mut current,
        &mut candidates,
    );
    candidates.sort_by(|left, right| {
        left.node_count()
            .cmp(&right.node_count())
            .then_with(|| candidate_canon_string(left).cmp(&candidate_canon_string(right)))
    });
    let mut seen = BTreeSet::new();
    candidates.retain(|candidate| seen.insert(candidate_canon_string(candidate)));
    candidates.truncate(max_emissions);
    candidates
}

fn push_unique(
    bucket: &mut Vec<Expr>,
    seen: &mut BTreeSet<String>,
    grammar: &GrammarSpec,
    raw: &Expr,
) {
    let candidate = normalize(raw);
    if candidate.depth() > grammar.max_depth {
        return;
    }
    if seen.insert(canon_string(&candidate)) {
        bucket.push(candidate);
    }
}

impl CandidateProposer for EnumerativeProposer {
    fn descriptor(&self) -> ProposalDescriptor {
        ProposalDescriptor::new(
            ProposalSourceKind::DeterministicEnumeration,
            format!(
                "enumerative-v2;max_nodes={};max_emissions={}",
                self.config.max_nodes, self.config.max_emissions
            ),
        )
    }

    fn propose(&mut self, context: &ProposalContext<'_>) -> Option<RawProposal> {
        if !self.started {
            self.generate(context.grammar);
        }
        if self.next >= self.outbox.len() {
            return None;
        }
        let emission_index = self.next;
        let candidate = self.outbox[emission_index].clone();
        self.next += 1;
        Some(RawProposal {
            candidate,
            source: ProposalSource::Enumerative { emission_index },
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grammar::OperatorSet;

    fn grammar() -> GrammarSpec {
        GrammarSpec {
            inputs: vec!["a".into(), "b".into()],
            outputs: vec!["value".into()],
            constants: vec![],
            operators: OperatorSet {
                add: true,
                sub: true,
                mul: true,
                max: false,
                exp: false,
            },
            max_nodes: 8,
            max_depth: 6,
            version: 1,
        }
    }

    fn collect(proposer: &mut EnumerativeProposer, grammar: &GrammarSpec) -> Vec<String> {
        let mut out = Vec::new();
        while let Some(raw) = proposer.propose(&ProposalContext {
            grammar,
            budget: &crate::SearchBudget::tiny(),
            feedback: &[],
        }) {
            out.push(candidate_canon_string(&raw.candidate));
        }
        out
    }

    #[test]
    fn enumerates_without_duplicates_and_exhausts_cleanly() {
        let mut proposer = EnumerativeProposer::new(EnumerativeConfig {
            max_nodes: 3,
            max_emissions: 10_000,
        });
        let g = grammar();
        let emissions = collect(&mut proposer, &g);
        assert!(!emissions.is_empty());
        let unique: BTreeSet<&String> = emissions.iter().collect();
        assert_eq!(unique.len(), emissions.len(), "no duplicates emitted");
        assert!(
            proposer
                .propose(&ProposalContext {
                    grammar: &g,
                    budget: &crate::SearchBudget::tiny(),
                    feedback: &[],
                })
                .is_none()
        );
    }

    #[test]
    fn respects_node_budget() {
        let mut proposer = EnumerativeProposer::new(EnumerativeConfig {
            max_nodes: 2,
            max_emissions: 10_000,
        });
        let grammar = grammar();
        while let Some(raw) = proposer.propose(&ProposalContext {
            grammar: &grammar,
            budget: &crate::SearchBudget::tiny(),
            feedback: &[],
        }) {
            assert!(raw.candidate.node_count() <= 2);
            raw.candidate.validate(&grammar).unwrap();
        }
    }

    #[test]
    fn commutative_variants_are_collapsed() {
        // With add only over two variables, size 3 yields exactly three
        // distinct canonical adds: v0+v0, v0+v1, v1+v1.
        let mut proposer = EnumerativeProposer::new(EnumerativeConfig {
            max_nodes: 3,
            max_emissions: 10_000,
        });
        let emissions = collect(&mut proposer, &grammar());
        let adds: Vec<&String> = emissions
            .iter()
            .filter(|e| e.starts_with("(candidate (out (add"))
            .collect();
        assert_eq!(adds.len(), 3);
    }

    #[test]
    fn exp_wrapping_respects_operator_availability() {
        let mut no_exp_grammar = grammar();
        no_exp_grammar.operators.exp = true;
        no_exp_grammar.max_nodes = 2;
        let mut proposer = EnumerativeProposer::new(EnumerativeConfig {
            max_nodes: 2,
            max_emissions: 10_000,
        });
        let emissions = collect(&mut proposer, &no_exp_grammar);
        assert!(
            emissions
                .iter()
                .any(|e| e.starts_with("(candidate (out (exp"))
        );
    }
}
