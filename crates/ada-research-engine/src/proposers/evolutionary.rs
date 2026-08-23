//! Deterministic genetic programming over the restricted recurrence grammar.
//!
//! The proposer sees only grammar metadata and loss feedback for candidates it
//! already emitted. It has no corpus, oracle, holdout, or gate handle.

use std::collections::{BTreeSet, VecDeque};

use crate::candidate::Candidate;
use crate::canon::{candidate_canon_string, normalize_candidate};
use crate::expr::Expr;
use crate::grammar::{GrammarSpec, OperatorSet};
use crate::proposer::{
    CandidateProposer, ProposalContext, ProposalDescriptor, ProposalSource, ProposalSourceKind,
    RawProposal, ScoredCandidate,
};
use crate::rng::SearchRng;

/// Replay-relevant configuration of the evolutionary proposer.
#[derive(Debug, Clone, PartialEq)]
pub struct EvolutionaryConfig {
    pub seed: u64,
    pub population_size: usize,
    pub tournament_size: usize,
    pub elitism: usize,
    pub crossover_probability: f64,
    pub mutation_probability: f64,
    pub init_min_nodes: usize,
    pub init_max_nodes: usize,
    pub subtree_max_depth: usize,
    /// Total generations, including generation zero.
    pub max_generations: usize,
    /// Maximum mutation retries per mutation event.
    pub max_mutation_attempts: usize,
}

impl Default for EvolutionaryConfig {
    fn default() -> Self {
        Self {
            seed: 0,
            population_size: 512,
            tournament_size: 2,
            elitism: 8,
            crossover_probability: 0.70,
            mutation_probability: 0.45,
            init_min_nodes: 4,
            init_max_nodes: 18,
            subtree_max_depth: 5,
            max_generations: 64,
            max_mutation_attempts: 8,
        }
    }
}

impl EvolutionaryConfig {
    fn sanitized(self) -> Self {
        let mut result = self;
        result.population_size = result.population_size.max(2);
        result.tournament_size = result.tournament_size.max(1);
        result.elitism = result.elitism.min(result.population_size - 1);
        result.init_min_nodes = result.init_min_nodes.max(1);
        result.init_max_nodes = result.init_max_nodes.max(result.init_min_nodes);
        result.subtree_max_depth = result.subtree_max_depth.max(1);
        result.max_generations = result.max_generations.max(1);
        result.max_mutation_attempts = result.max_mutation_attempts.max(1);
        result
    }

    fn canonical_text(&self) -> String {
        format!(
            concat!(
                "evolutionary-v2;seed={};population={};tournament={};elitism={};",
                "crossover_bits={:016x};mutation_bits={:016x};init_nodes={}..={};",
                "subtree_depth={};generations={};mutation_attempts={}"
            ),
            self.seed,
            self.population_size,
            self.tournament_size,
            self.elitism,
            self.crossover_probability.to_bits(),
            self.mutation_probability.to_bits(),
            self.init_min_nodes,
            self.init_max_nodes,
            self.subtree_max_depth,
            self.max_generations,
            self.max_mutation_attempts,
        )
    }
}

#[derive(Debug, Clone)]
struct Individual {
    candidate: Candidate,
    loss: Option<f64>,
}

/// Seeded, bounded evolutionary candidate source.
#[derive(Debug)]
pub struct EvolutionaryProposer {
    config: EvolutionaryConfig,
    rng: SearchRng,
    outbox: VecDeque<(Candidate, ProposalSource)>,
    population: Vec<Individual>,
    feedback_watermark: usize,
    generation_index: usize,
    started: bool,
    exhausted: bool,
}

#[derive(Debug, Clone, Copy)]
enum OpKind {
    Add,
    Sub,
    Mul,
    Max,
    Exp,
}

fn available_kinds(operators: OperatorSet) -> Vec<OpKind> {
    let mut kinds = Vec::new();
    if operators.add {
        kinds.push(OpKind::Add);
    }
    if operators.sub {
        kinds.push(OpKind::Sub);
    }
    if operators.mul {
        kinds.push(OpKind::Mul);
    }
    if operators.max {
        kinds.push(OpKind::Max);
    }
    if operators.exp {
        kinds.push(OpKind::Exp);
    }
    kinds
}

impl EvolutionaryProposer {
    #[must_use]
    pub fn new(config: EvolutionaryConfig) -> Self {
        let config = config.sanitized();
        Self {
            rng: SearchRng::new(config.seed),
            config,
            outbox: VecDeque::new(),
            population: Vec::new(),
            feedback_watermark: 0,
            generation_index: 0,
            started: false,
            exhausted: false,
        }
    }

    fn random_candidate(&mut self, grammar: &GrammarSpec) -> Candidate {
        let minimum = self
            .config
            .init_min_nodes
            .max(grammar.output_count())
            .min(grammar.max_nodes);
        let maximum = self
            .config
            .init_max_nodes
            .max(minimum)
            .min(grammar.max_nodes);
        let target = minimum + self.rng.below(maximum - minimum + 1);
        grow_candidate(
            grammar,
            &mut self.rng,
            target,
            self.config.subtree_max_depth.min(grammar.max_depth),
        )
    }

    fn start(&mut self, grammar: &GrammarSpec) {
        let mut seen = BTreeSet::new();
        let mut attempts = 0usize;
        let attempt_limit = self.config.population_size.saturating_mul(16);
        while self.population.len() < self.config.population_size && attempts < attempt_limit {
            attempts += 1;
            let candidate = self.random_candidate(grammar);
            let form = candidate_canon_string(&normalize_candidate(&candidate));
            if seen.insert(form) {
                self.population.push(Individual {
                    candidate,
                    loss: None,
                });
            }
        }
        // A tiny grammar can have fewer unique forms than the requested
        // population. Fill deterministically; engine-side canonical dedup is
        // still authoritative and generated-candidate limits prevent loops.
        while self.population.len() < self.config.population_size {
            let candidate = self.random_candidate(grammar);
            self.population.push(Individual {
                candidate,
                loss: None,
            });
        }
        self.emit_population(0);
        self.started = true;
    }

    fn emit_population(&mut self, generation: usize) {
        for (individual, entry) in self.population.iter().enumerate() {
            self.outbox.push_back((
                entry.candidate.clone(),
                ProposalSource::Evolutionary {
                    seed: self.config.seed,
                    generation,
                    individual,
                },
            ));
        }
    }

    fn harvest(&mut self, feedback: &[ScoredCandidate]) {
        let available = feedback.len().saturating_sub(self.feedback_watermark);
        let count = available.min(self.population.len());
        for position in 0..count {
            self.population[position].loss =
                Some(feedback[self.feedback_watermark + position].train_loss);
        }
        self.feedback_watermark += count;
    }

    fn evolve(&mut self, grammar: &GrammarSpec) {
        let mut ranking: Vec<usize> = (0..self.population.len()).collect();
        ranking.sort_by(|&left, &right| {
            let left_entry = &self.population[left];
            let right_entry = &self.population[right];
            left_entry
                .loss
                .unwrap_or(f64::MAX)
                .total_cmp(&right_entry.loss.unwrap_or(f64::MAX))
                .then_with(|| {
                    candidate_canon_string(&normalize_candidate(&left_entry.candidate)).cmp(
                        &candidate_canon_string(&normalize_candidate(&right_entry.candidate)),
                    )
                })
                .then_with(|| left.cmp(&right))
        });

        let mut next = Vec::with_capacity(self.config.population_size);
        let mut seen = BTreeSet::new();
        for &index in ranking.iter().take(self.config.elitism) {
            let candidate = self.population[index].candidate.clone();
            seen.insert(candidate_canon_string(&normalize_candidate(&candidate)));
            next.push(Individual {
                candidate,
                loss: None,
            });
        }

        while next.len() < self.config.population_size {
            let parent_a = self.tournament(&ranking);
            let mut child = self.population[parent_a].candidate.clone();
            if self.rng.unit() < self.config.crossover_probability {
                let parent_b = self.tournament(&ranking);
                let donor = self.population[parent_b].candidate.clone();
                child = self.crossover(&child, &donor, grammar);
            }
            if self.rng.unit() < self.config.mutation_probability {
                child = self.mutate(&child, grammar);
            }
            let mut form = candidate_canon_string(&normalize_candidate(&child));
            for _ in 0..self.config.max_mutation_attempts {
                if seen.insert(form) {
                    break;
                }
                child = self.mutate(&child, grammar);
                form = candidate_canon_string(&normalize_candidate(&child));
            }
            next.push(Individual {
                candidate: child,
                loss: None,
            });
        }

        self.population = next;
        self.generation_index += 1;
        self.emit_population(self.generation_index);
    }

    fn tournament(&mut self, ranking: &[usize]) -> usize {
        let mut best_rank = usize::MAX;
        for _ in 0..self.config.tournament_size {
            best_rank = best_rank.min(self.rng.below(ranking.len()));
        }
        ranking[best_rank.min(ranking.len() - 1)]
    }

    fn crossover(
        &mut self,
        host: &Candidate,
        donor: &Candidate,
        grammar: &GrammarSpec,
    ) -> Candidate {
        let output = self.rng.below(host.output_arity());
        let donor_output = self.rng.below(donor.output_arity());
        let host_expression = &host.outputs()[output];
        let donor_expression = &donor.outputs()[donor_output];
        let host_spot = self.rng.below(host_expression.node_count());
        let donor_spot = self.rng.below(donor_expression.node_count());
        let Some(graft) = subtree_clone(donor_expression, donor_spot) else {
            return host.clone();
        };
        let mut child = host.clone();
        if splice(&mut child.outputs_mut()[output], host_spot, graft).is_err()
            || child.node_count() > grammar.max_nodes
            || child.depth() > grammar.max_depth
        {
            host.clone()
        } else {
            child
        }
    }

    fn mutate(&mut self, candidate: &Candidate, grammar: &GrammarSpec) -> Candidate {
        for _ in 0..self.config.max_mutation_attempts {
            let output = self.rng.below(candidate.output_arity());
            let other_nodes = candidate
                .node_count()
                .saturating_sub(candidate.outputs()[output].node_count());
            let expression_budget = grammar.max_nodes.saturating_sub(other_nodes).max(1);
            let mutated = mutate_expression(
                &candidate.outputs()[output],
                grammar,
                &mut self.rng,
                expression_budget,
                self.config.subtree_max_depth,
            );
            let mut result = candidate.clone();
            result.outputs_mut()[output] = mutated;
            if result.node_count() <= grammar.max_nodes && result.depth() <= grammar.max_depth {
                return result;
            }
        }
        candidate.clone()
    }
}

impl CandidateProposer for EvolutionaryProposer {
    fn descriptor(&self) -> ProposalDescriptor {
        ProposalDescriptor::new(
            ProposalSourceKind::EvolutionarySearch,
            self.config.canonical_text(),
        )
    }

    fn propose(&mut self, context: &ProposalContext<'_>) -> Option<RawProposal> {
        if self.exhausted {
            return None;
        }
        if !self.started {
            self.config.max_generations = self
                .config
                .max_generations
                .min(context.budget.max_generations);
            self.config.max_mutation_attempts = self
                .config
                .max_mutation_attempts
                .min(context.budget.max_mutation_attempts);
            self.start(context.grammar);
        }
        if self.outbox.is_empty() {
            self.harvest(context.feedback);
            if self.generation_index + 1 >= self.config.max_generations {
                self.exhausted = true;
                return None;
            }
            self.evolve(context.grammar);
        }
        let (candidate, source) = self.outbox.pop_front()?;
        Some(RawProposal { candidate, source })
    }
}

fn grow_candidate(
    grammar: &GrammarSpec,
    rng: &mut SearchRng,
    node_budget: usize,
    depth_budget: usize,
) -> Candidate {
    let output_count = grammar.output_count();
    let total = node_budget.max(output_count).min(grammar.max_nodes);
    let mut remaining = total;
    let mut outputs = Vec::with_capacity(output_count);
    for output in 0..output_count {
        let outputs_left = output_count - output;
        let maximum = remaining.saturating_sub(outputs_left - 1).max(1);
        let allocation = if outputs_left == 1 {
            remaining
        } else {
            1 + rng.below(maximum)
        };
        outputs.push(grow_tree(grammar, rng, allocation, depth_budget));
        remaining = remaining.saturating_sub(allocation);
    }
    Candidate::new(outputs)
}

fn grow_tree(
    grammar: &GrammarSpec,
    rng: &mut SearchRng,
    node_budget: usize,
    depth_left: usize,
) -> Expr {
    let kinds = available_kinds(grammar.operators);
    if node_budget <= 1 || depth_left <= 1 || kinds.is_empty() {
        return grammar
            .leaf(rng.below(grammar.leaf_count()))
            .unwrap_or(Expr::Var(0));
    }
    match kinds[rng.below(kinds.len())] {
        OpKind::Exp => Expr::Exp(Box::new(grow_tree(
            grammar,
            rng,
            node_budget - 1,
            depth_left - 1,
        ))),
        kind => {
            if node_budget < 3 {
                return grammar
                    .leaf(rng.below(grammar.leaf_count()))
                    .unwrap_or(Expr::Var(0));
            }
            let child_nodes = node_budget - 1;
            let left_budget = 1 + rng.below(child_nodes - 1);
            let right_budget = child_nodes - left_budget;
            let left = Box::new(grow_tree(grammar, rng, left_budget, depth_left - 1));
            let right = Box::new(grow_tree(grammar, rng, right_budget, depth_left - 1));
            match kind {
                OpKind::Add => Expr::Add(left, right),
                OpKind::Sub => Expr::Sub(left, right),
                OpKind::Mul => Expr::Mul(left, right),
                OpKind::Max => Expr::Max(left, right),
                OpKind::Exp => unreachable!("handled above"),
            }
        }
    }
}

fn mutate_expression(
    expression: &Expr,
    grammar: &GrammarSpec,
    rng: &mut SearchRng,
    node_budget: usize,
    subtree_depth: usize,
) -> Expr {
    let spot = rng.below(expression.node_count());
    match rng.below(4) {
        0 => {
            let Some(victim) = subtree_clone(expression, spot) else {
                return expression.clone();
            };
            let remaining = node_budget
                .saturating_sub(expression.node_count())
                .saturating_add(victim.node_count())
                .max(1);
            let fresh = grow_tree(grammar, rng, remaining, subtree_depth);
            let mut result = expression.clone();
            if splice(&mut result, spot, fresh).is_err() {
                expression.clone()
            } else {
                result
            }
        }
        1 => {
            let mut result = expression.clone();
            if replace_leaf(
                &mut result,
                spot,
                grammar
                    .leaf(rng.below(grammar.leaf_count()))
                    .unwrap_or(Expr::Var(0)),
            ) {
                result
            } else {
                expression.clone()
            }
        }
        2 => {
            let mut result = expression.clone();
            if swap_operator(&mut result, spot, grammar.operators, rng) {
                result
            } else {
                expression.clone()
            }
        }
        _ if grammar.operators.exp && expression.node_count() < node_budget => {
            Expr::Exp(Box::new(expression.clone()))
        }
        _ => expression.clone(),
    }
}

fn subtree_at_mut(expression: &mut Expr, index: usize) -> Option<&mut Expr> {
    let mut remaining = index;
    let mut cursor = expression;
    loop {
        if remaining == 0 {
            return Some(cursor);
        }
        remaining -= 1;
        match cursor {
            Expr::Exp(inner) => cursor = inner,
            Expr::Add(left, right)
            | Expr::Sub(left, right)
            | Expr::Mul(left, right)
            | Expr::Max(left, right) => {
                let left_nodes = left.node_count();
                if remaining < left_nodes {
                    cursor = left;
                } else {
                    remaining -= left_nodes;
                    cursor = right;
                }
            }
            Expr::Var(_) | Expr::Const(_) => return None,
        }
    }
}

fn subtree_clone(expression: &Expr, index: usize) -> Option<Expr> {
    let mut copy = expression.clone();
    subtree_at_mut(&mut copy, index).cloned()
}

fn splice(host: &mut Expr, index: usize, graft: Expr) -> Result<(), ()> {
    let slot = subtree_at_mut(host, index).ok_or(())?;
    *slot = graft;
    Ok(())
}

fn replace_leaf(expression: &mut Expr, index: usize, leaf: Expr) -> bool {
    let Some(slot) = subtree_at_mut(expression, index) else {
        return false;
    };
    if matches!(slot, Expr::Var(_) | Expr::Const(_)) {
        *slot = leaf;
        true
    } else {
        false
    }
}

fn swap_operator(
    expression: &mut Expr,
    index: usize,
    operators: OperatorSet,
    rng: &mut SearchRng,
) -> bool {
    let Some(slot) = subtree_at_mut(expression, index) else {
        return false;
    };
    let binary: Vec<OpKind> = available_kinds(operators)
        .into_iter()
        .filter(|kind| !matches!(kind, OpKind::Exp))
        .collect();
    if binary.is_empty() {
        return false;
    }
    let old = std::mem::replace(slot, Expr::Var(0));
    let (left, right) = match old {
        Expr::Add(left, right)
        | Expr::Sub(left, right)
        | Expr::Mul(left, right)
        | Expr::Max(left, right) => (left, right),
        other => {
            *slot = other;
            return false;
        }
    };
    *slot = match binary[rng.below(binary.len())] {
        OpKind::Add => Expr::Add(left, right),
        OpKind::Sub => Expr::Sub(left, right),
        OpKind::Mul => Expr::Mul(left, right),
        OpKind::Max => Expr::Max(left, right),
        OpKind::Exp => unreachable!("filtered above"),
    };
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::canon::candidate_canon_string;

    fn grammar() -> GrammarSpec {
        GrammarSpec {
            inputs: vec!["a".into(), "b".into(), "c".into()],
            outputs: vec!["x".into(), "y".into()],
            constants: vec![],
            operators: OperatorSet::all(),
            max_nodes: 24,
            max_depth: 10,
            version: 1,
        }
    }

    fn context<'a>(
        grammar: &'a GrammarSpec,
        feedback: &'a [ScoredCandidate],
    ) -> ProposalContext<'a> {
        static BUDGET: crate::SearchBudget = crate::SearchBudget::tiny();
        ProposalContext {
            grammar,
            budget: &BUDGET,
            feedback,
        }
    }

    #[test]
    fn same_seed_emits_same_bounded_order() {
        let grammar = grammar();
        let config = EvolutionaryConfig {
            seed: 11,
            population_size: 32,
            max_generations: 2,
            ..EvolutionaryConfig::default()
        };
        let mut left = EvolutionaryProposer::new(config.clone());
        let mut right = EvolutionaryProposer::new(config);
        for _ in 0..32 {
            let left_raw = left.propose(&context(&grammar, &[])).unwrap();
            let right_raw = right.propose(&context(&grammar, &[])).unwrap();
            assert_eq!(left_raw.source, right_raw.source);
            assert_eq!(
                candidate_canon_string(&left_raw.candidate),
                candidate_canon_string(&right_raw.candidate)
            );
            left_raw.candidate.validate(&grammar).unwrap();
        }
    }

    #[test]
    fn generation_budget_exhausts_stream() {
        let grammar = grammar();
        let mut proposer = EvolutionaryProposer::new(EvolutionaryConfig {
            seed: 3,
            population_size: 8,
            max_generations: 1,
            ..EvolutionaryConfig::default()
        });
        for _ in 0..8 {
            assert!(proposer.propose(&context(&grammar, &[])).is_some());
        }
        assert!(proposer.propose(&context(&grammar, &[])).is_none());
    }

    #[test]
    fn different_seeds_change_population() {
        let grammar = grammar();
        let config = |seed| EvolutionaryConfig {
            seed,
            population_size: 16,
            ..EvolutionaryConfig::default()
        };
        let mut left = EvolutionaryProposer::new(config(1));
        let mut right = EvolutionaryProposer::new(config(2));
        let left_forms: Vec<_> = (0..16)
            .map(|_| {
                candidate_canon_string(&left.propose(&context(&grammar, &[])).unwrap().candidate)
            })
            .collect();
        let right_forms: Vec<_> = (0..16)
            .map(|_| {
                candidate_canon_string(&right.propose(&context(&grammar, &[])).unwrap().candidate)
            })
            .collect();
        assert_ne!(left_forms, right_forms);
    }

    #[test]
    fn feedback_drives_a_deterministic_second_generation() {
        let grammar = grammar();
        let mut proposer = EvolutionaryProposer::new(EvolutionaryConfig {
            seed: 5,
            population_size: 12,
            elitism: 3,
            max_generations: 2,
            ..EvolutionaryConfig::default()
        });
        let first: Vec<Candidate> = (0..12)
            .map(|_| proposer.propose(&context(&grammar, &[])).unwrap().candidate)
            .collect();
        let feedback: Vec<_> = first
            .iter()
            .enumerate()
            .map(|(index, candidate)| ScoredCandidate {
                normalized_candidate: normalize_candidate(candidate),
                train_loss: u32::try_from(index).map_or(f64::from(u32::MAX), f64::from),
                rejected_at: None,
            })
            .collect();
        let second: Vec<Candidate> = (0..12)
            .map(|_| {
                proposer
                    .propose(&context(&grammar, &feedback))
                    .unwrap()
                    .candidate
            })
            .collect();
        for elite in &first[..3] {
            assert!(second.iter().any(|candidate| {
                candidate_canon_string(candidate) == candidate_canon_string(elite)
            }));
        }
    }

    #[test]
    fn subtree_helpers_use_preorder_indices() {
        let expression = Expr::Add(
            Box::new(Expr::Var(0)),
            Box::new(Expr::Exp(Box::new(Expr::Var(1)))),
        );
        assert_eq!(subtree_clone(&expression, 2).unwrap().node_count(), 2);
        let mut changed = expression.clone();
        splice(&mut changed, 2, Expr::Var(2)).unwrap();
        assert_eq!(changed.node_count(), 3);
        assert!(subtree_clone(&expression, 99).is_none());
    }
}
