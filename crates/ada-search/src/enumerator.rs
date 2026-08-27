//! Bounded deterministic candidate generation for semantic attention research.
//!
//! This module is deliberately separate from the historical A8 recurrence
//! fixtures in `lib.rs`. It enumerates a small executable semantic grammar,
//! orders candidates by a declared structural cost, and exposes every
//! generated candidate as a reconstructible semantic program. Evaluation,
//! falsification, and evidence recording are later stages; this layer never
//! calls an oracle or labels a candidate novel.

use ada_core::{
    MaskContract, SemanticDescriptor, SemanticFamily, SemanticId, StateContract, WeightContract,
};
use ada_semantic::{
    AffinityRule, InputTransform, MaskRule, OutputRule, SelectionRule, SemanticIrError,
    SemanticProgram, SemanticProgramSpec, ValueMixRule, WeightRule,
};
use std::collections::BTreeSet;
use std::fmt::{Display, Formatter};

/// Version of the bounded semantic search-space contract.
pub const SEARCH_SPACE_VERSION: u16 = 1;
/// Version of the resumable search checkpoint contract.
pub const SEARCH_CHECKPOINT_VERSION: u16 = 1;
/// Maximum alternatives in any one grammar dimension.
pub const MAX_SEARCH_ALTERNATIVES: usize = 32;
/// Maximum raw combinations in one search space.
pub const MAX_SEARCH_SPACE_CARDINALITY: u64 = 1 << 16;
/// Maximum raw candidate expansions permitted by one search budget.
pub const MAX_SEARCH_EXPANSIONS: u64 = 1 << 16;
/// Maximum structural cost accepted by the semantic search layer.
pub const MAX_PROGRAM_COST: u32 = 64;
/// Maximum encoded candidate text retained in a checkpoint.
pub const MAX_CHECKPOINT_CANDIDATE_BYTES: usize = 1 << 20;
/// Maximum candidates retained in a checkpoint's deduplication set.
pub const MAX_CHECKPOINT_SEEN: u64 = 1 << 16;
/// Maximum canonical checkpoint size.
pub const MAX_CHECKPOINT_TEXT_BYTES: usize = 16 << 20;

/// Search-generation and checkpoint failures.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SearchError {
    /// A search-space or budget declaration is invalid.
    InvalidConfiguration(&'static str),
    /// A bounded search value exceeded its explicit limit.
    ExceedsLimit {
        /// Field whose value was rejected.
        field: &'static str,
        /// Rejected value.
        value: u64,
        /// Inclusive maximum.
        maximum: u64,
    },
    /// A generated semantic program failed its own validation contract.
    CandidateConstruction(SemanticIrError),
    /// A checkpoint is malformed or internally inconsistent.
    InvalidCheckpoint(String),
    /// A checkpoint belongs to a different canonical search space.
    CheckpointSpaceMismatch,
}

impl Display for SearchError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidConfiguration(field) => write!(formatter, "invalid search field: {field}"),
            Self::ExceedsLimit {
                field,
                value,
                maximum,
            } => write!(formatter, "{field}={value} exceeds maximum {maximum}"),
            Self::CandidateConstruction(error) => {
                write!(
                    formatter,
                    "generated candidate failed semantic validation: {error}"
                )
            }
            Self::InvalidCheckpoint(reason) => {
                write!(formatter, "invalid search checkpoint: {reason}")
            }
            Self::CheckpointSpaceMismatch => {
                write!(
                    formatter,
                    "search checkpoint does not match the search space"
                )
            }
        }
    }
}

impl std::error::Error for SearchError {}

impl From<SemanticIrError> for SearchError {
    fn from(error: SemanticIrError) -> Self {
        Self::CandidateConstruction(error)
    }
}

/// Limits governing raw expansions, returned candidates, and structural cost.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SearchBudget {
    expansions: u64,
    candidates: u64,
    program_cost: u32,
}

impl SearchBudget {
    /// Construct a bounded search budget.
    ///
    /// Zero is allowed for an intentional no-op search. The upper bounds keep
    /// accidental unbounded enumeration from becoming an opaque workload.
    ///
    /// # Errors
    ///
    /// Returns an error when a budget field exceeds its explicit bound.
    pub fn new(
        max_expansions: u64,
        max_candidates: u64,
        max_program_cost: u32,
    ) -> Result<Self, SearchError> {
        if max_expansions > MAX_SEARCH_EXPANSIONS {
            return Err(SearchError::ExceedsLimit {
                field: "max_expansions",
                value: max_expansions,
                maximum: MAX_SEARCH_EXPANSIONS,
            });
        }
        if max_candidates > MAX_CHECKPOINT_SEEN {
            return Err(SearchError::ExceedsLimit {
                field: "max_candidates",
                value: max_candidates,
                maximum: MAX_CHECKPOINT_SEEN,
            });
        }
        if max_program_cost > MAX_PROGRAM_COST {
            return Err(SearchError::ExceedsLimit {
                field: "max_program_cost",
                value: u64::from(max_program_cost),
                maximum: u64::from(MAX_PROGRAM_COST),
            });
        }
        Ok(Self {
            expansions: max_expansions,
            candidates: max_candidates,
            program_cost: max_program_cost,
        })
    }

    /// Maximum raw candidate ordinals visited.
    #[must_use]
    pub const fn max_expansions(self) -> u64 {
        self.expansions
    }

    /// Maximum statically valid, deduplicated candidates returned.
    #[must_use]
    pub const fn max_candidates(self) -> u64 {
        self.candidates
    }

    /// Maximum structural cost allowed through static pruning.
    #[must_use]
    pub const fn max_program_cost(self) -> u32 {
        self.program_cost
    }
}

/// Search-stage counters retained for evidence and future CEGIS stages.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct SearchStats {
    generated: u64,
    statically_rejected: u64,
    duplicate: u64,
    oracle_falsified: u64,
    adversarial_falsified: u64,
    cost_dominated: u64,
    surviving: u64,
}

impl SearchStats {
    /// Raw grammar expansions attempted.
    #[must_use]
    pub const fn generated(self) -> u64 {
        self.generated
    }

    /// Candidates rejected before execution by static cost/domain checks.
    #[must_use]
    pub const fn statically_rejected(self) -> u64 {
        self.statically_rejected
    }

    /// Candidates removed by canonical deduplication.
    #[must_use]
    pub const fn duplicate(self) -> u64 {
        self.duplicate
    }

    /// Candidates later falsified by an oracle stage.
    #[must_use]
    pub const fn oracle_falsified(self) -> u64 {
        self.oracle_falsified
    }

    /// Candidates later falsified by adversarial counterexamples.
    #[must_use]
    pub const fn adversarial_falsified(self) -> u64 {
        self.adversarial_falsified
    }

    /// Candidates removed by a later multi-objective dominance stage.
    #[must_use]
    pub const fn cost_dominated(self) -> u64 {
        self.cost_dominated
    }

    /// Candidates surviving this generation/static/deduplication stage.
    #[must_use]
    pub const fn surviving(self) -> u64 {
        self.surviving
    }

    fn generation_fields(self) -> [u64; 4] {
        [
            self.generated,
            self.statically_rejected,
            self.duplicate,
            self.surviving,
        ]
    }

    pub(crate) fn with_values(values: [u64; 7]) -> Self {
        Self {
            generated: values[0],
            statically_rejected: values[1],
            duplicate: values[2],
            oracle_falsified: values[3],
            adversarial_falsified: values[4],
            cost_dominated: values[5],
            surviving: values[6],
        }
    }
}

/// Stable dual-lane fingerprint used for spaces, candidates, and checkpoints.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SearchFingerprint {
    primary: u64,
    secondary: u64,
    length: u64,
}

impl SearchFingerprint {
    pub(crate) fn of_bytes(bytes: &[u8]) -> Self {
        const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
        const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;
        const MIX_MULT: u64 = 0xff51_afd7_ed55_8ccd;
        let mut primary = FNV_OFFSET;
        let mut secondary = FNV_OFFSET;
        for &byte in bytes {
            primary ^= u64::from(byte);
            primary = primary.wrapping_mul(FNV_PRIME);
            secondary ^= u64::from(byte);
            secondary = secondary.rotate_left(27).wrapping_mul(MIX_MULT);
        }
        let length = u64::try_from(bytes.len()).unwrap_or(u64::MAX);
        primary = primary.wrapping_mul(FNV_PRIME) ^ length;
        secondary = secondary.rotate_left(31) ^ length;
        Self {
            primary,
            secondary,
            length,
        }
    }

    /// Compute a stable fingerprint for a canonical text representation.
    ///
    /// This constructor lets downstream search-space implementations use the
    /// same identity contract without depending on the internal hashing
    /// implementation.
    #[must_use]
    pub fn of_canonical_text(text: &str) -> Self {
        Self::of_bytes(text.as_bytes())
    }

    pub(crate) const fn from_parts(primary: u64, secondary: u64, length: u64) -> Self {
        Self {
            primary,
            secondary,
            length,
        }
    }

    /// First fingerprint lane.
    #[must_use]
    pub const fn primary(self) -> u64 {
        self.primary
    }

    /// Second fingerprint lane.
    #[must_use]
    pub const fn secondary(self) -> u64 {
        self.secondary
    }

    /// Canonical byte length included in the fingerprint.
    #[must_use]
    pub const fn length(self) -> u64 {
        self.length
    }
}

impl Display for SearchFingerprint {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "{:016x}-{:016x}-{:016x}",
            self.primary, self.secondary, self.length
        )
    }
}

/// A generated candidate with all information required for inspection.
#[derive(Debug, Clone, PartialEq)]
pub struct SearchCandidate<T> {
    candidate: T,
    ordinal: u64,
    canonical_text: String,
    fingerprint: SearchFingerprint,
    cost: u32,
}

impl<T> SearchCandidate<T> {
    /// Candidate value, normally a semantic or implementation IR object.
    #[must_use]
    pub const fn candidate(&self) -> &T {
        &self.candidate
    }

    /// Consume the wrapper and return the candidate value.
    #[must_use]
    pub fn into_candidate(self) -> T {
        self.candidate
    }

    /// Stable search ordinal after cost ordering.
    #[must_use]
    pub const fn ordinal(&self) -> u64 {
        self.ordinal
    }

    /// Canonical candidate text retained for evidence.
    #[must_use]
    pub fn canonical_text(&self) -> &str {
        &self.canonical_text
    }

    /// Stable candidate fingerprint.
    #[must_use]
    pub const fn fingerprint(&self) -> SearchFingerprint {
        self.fingerprint
    }

    /// Declared structural cost used by static pruning and ordering.
    #[must_use]
    pub const fn cost(&self) -> u32 {
        self.cost
    }
}

/// Minimal interface for future search spaces beyond semantic programs.
pub trait SearchSpace: Clone {
    /// Reconstructible candidate type.
    type Candidate: Clone;

    /// Number of ordered raw ordinals in the bounded space.
    fn cardinality(&self) -> u64;
    /// Fingerprint of the complete canonical space declaration.
    fn fingerprint(&self) -> SearchFingerprint;
    /// Reconstruct one candidate by ordered ordinal.
    ///
    /// # Errors
    ///
    /// Returns an error when the ordinal cannot be reconstructed.
    fn candidate_at(&self, ordinal: u64) -> Result<Self::Candidate, SearchError>;
    /// Canonical candidate representation used for deduplication/evidence.
    fn candidate_canonical_text(&self, candidate: &Self::Candidate) -> String;
    /// Declared static structural cost.
    fn candidate_cost(&self, candidate: &Self::Candidate) -> u32;
}

/// Deterministic bounded search driver shared by semantic and future IR spaces.
#[derive(Debug, Clone)]
pub struct SearchEngine<S: SearchSpace> {
    space: S,
    budget: SearchBudget,
    next_ordinal: u64,
    stats: SearchStats,
    seen: BTreeSet<String>,
}

impl<S: SearchSpace> SearchEngine<S> {
    /// Start a search from ordinal zero.
    ///
    /// # Errors
    ///
    /// Returns an error when the supplied space exceeds the search bounds.
    pub fn new(space: S, budget: SearchBudget) -> Result<Self, SearchError> {
        if space.cardinality() > MAX_SEARCH_SPACE_CARDINALITY {
            return Err(SearchError::ExceedsLimit {
                field: "search_space.cardinality",
                value: space.cardinality(),
                maximum: MAX_SEARCH_SPACE_CARDINALITY,
            });
        }
        Ok(Self {
            space,
            budget,
            next_ordinal: 0,
            stats: SearchStats::default(),
            seen: BTreeSet::new(),
        })
    }

    /// Search space declaration.
    #[must_use]
    pub const fn space(&self) -> &S {
        &self.space
    }

    /// Active budget.
    #[must_use]
    pub const fn budget(&self) -> SearchBudget {
        self.budget
    }

    /// Current statistics, including only stages actually recorded so far.
    #[must_use]
    pub const fn stats(&self) -> SearchStats {
        self.stats
    }

    /// Next ordered raw ordinal, exposed for checkpoint diagnostics.
    #[must_use]
    pub const fn next_ordinal(&self) -> u64 {
        self.next_ordinal
    }

    /// Generate the next statically admissible, deduplicated candidate.
    ///
    /// The method performs no reference execution and makes no novelty or
    /// correctness claim. A generation error stops the search rather than
    /// silently skipping an uninspectable candidate.
    ///
    /// # Errors
    ///
    /// Returns an error when candidate reconstruction or retained canonical
    /// text violates the bounded contract.
    pub fn next_candidate(&mut self) -> Result<Option<SearchCandidate<S::Candidate>>, SearchError> {
        if self.stats.surviving >= self.budget.candidates
            || self.next_ordinal >= self.space.cardinality()
            || self.next_ordinal >= self.budget.expansions
        {
            return Ok(None);
        }
        while self.next_ordinal < self.space.cardinality()
            && self.next_ordinal < self.budget.expansions
        {
            let ordinal = self.next_ordinal;
            self.next_ordinal += 1;
            self.stats.generated += 1;
            let candidate = self.space.candidate_at(ordinal)?;
            let cost = self.space.candidate_cost(&candidate);
            if cost > self.budget.program_cost {
                self.stats.statically_rejected += 1;
                continue;
            }
            let canonical_text = self.space.candidate_canonical_text(&candidate);
            if canonical_text.len() > MAX_CHECKPOINT_CANDIDATE_BYTES {
                return Err(SearchError::ExceedsLimit {
                    field: "candidate.canonical_text_bytes",
                    value: u64::try_from(canonical_text.len()).unwrap_or(u64::MAX),
                    maximum: u64::try_from(MAX_CHECKPOINT_CANDIDATE_BYTES).unwrap_or(u64::MAX),
                });
            }
            if !self.seen.insert(canonical_text.clone()) {
                self.stats.duplicate += 1;
                continue;
            }
            self.stats.surviving += 1;
            let fingerprint = SearchFingerprint::of_bytes(canonical_text.as_bytes());
            return Ok(Some(SearchCandidate {
                candidate,
                ordinal,
                canonical_text,
                fingerprint,
                cost,
            }));
        }
        Ok(None)
    }

    /// Exhaust the configured raw/budgeted prefix and return generated items.
    ///
    /// # Errors
    ///
    /// Propagates candidate-generation or canonical-text errors.
    pub fn run_to_end(&mut self) -> Result<Vec<SearchCandidate<S::Candidate>>, SearchError> {
        let mut candidates = Vec::new();
        while let Some(candidate) = self.next_candidate()? {
            candidates.push(candidate);
        }
        Ok(candidates)
    }

    /// Record a later oracle differential failure without changing candidate
    /// identity. CEGIS/evidence layers own the reason and reproducer.
    pub fn record_oracle_falsified(&mut self) {
        self.stats.oracle_falsified += 1;
    }

    /// Record a later adversarial counterexample failure.
    pub fn record_adversarial_falsified(&mut self) {
        self.stats.adversarial_falsified += 1;
    }

    /// Record a later multi-objective cost-dominance decision.
    pub fn record_cost_dominated(&mut self) {
        self.stats.cost_dominated += 1;
    }

    /// Snapshot all state needed to resume this exact deterministic prefix.
    #[must_use]
    pub fn checkpoint(&self) -> SearchCheckpoint {
        SearchCheckpoint {
            space_fingerprint: self.space.fingerprint(),
            budget: self.budget,
            next_ordinal: self.next_ordinal,
            stats: self.stats,
            seen: self.seen.clone(),
        }
    }

    /// Resume from a validated checkpoint and verify its generated prefix.
    ///
    /// # Errors
    ///
    /// Returns an error when the checkpoint is malformed, belongs to another
    /// search space, or does not reproduce the declared prefix.
    pub fn from_checkpoint(space: S, checkpoint: SearchCheckpoint) -> Result<Self, SearchError> {
        checkpoint.validate_basic()?;
        if checkpoint.space_fingerprint != space.fingerprint() {
            return Err(SearchError::CheckpointSpaceMismatch);
        }
        if checkpoint.next_ordinal > space.cardinality()
            || checkpoint.next_ordinal > checkpoint.budget.expansions
        {
            return Err(SearchError::InvalidCheckpoint(
                "next ordinal is outside the space or budget".into(),
            ));
        }
        if checkpoint.stats.surviving > checkpoint.budget.candidates {
            return Err(SearchError::InvalidCheckpoint(
                "surviving count exceeds the candidate budget".into(),
            ));
        }

        let mut expected_seen = BTreeSet::new();
        let mut expected = SearchStats::default();
        for ordinal in 0..checkpoint.next_ordinal {
            let candidate = space.candidate_at(ordinal)?;
            expected.generated += 1;
            if space.candidate_cost(&candidate) > checkpoint.budget.program_cost {
                expected.statically_rejected += 1;
                continue;
            }
            let canonical_text = space.candidate_canonical_text(&candidate);
            if expected_seen.insert(canonical_text) {
                expected.surviving += 1;
            } else {
                expected.duplicate += 1;
            }
        }
        if expected.generation_fields() != checkpoint.stats.generation_fields()
            || expected_seen != checkpoint.seen
        {
            return Err(SearchError::InvalidCheckpoint(
                "checkpoint prefix does not match deterministic regeneration".into(),
            ));
        }
        Ok(Self {
            space,
            budget: checkpoint.budget,
            next_ordinal: checkpoint.next_ordinal,
            stats: checkpoint.stats,
            seen: checkpoint.seen,
        })
    }
}

/// Serializable state for exact search resume.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SearchCheckpoint {
    pub(crate) space_fingerprint: SearchFingerprint,
    pub(crate) budget: SearchBudget,
    pub(crate) next_ordinal: u64,
    pub(crate) stats: SearchStats,
    pub(crate) seen: BTreeSet<String>,
}

impl SearchCheckpoint {
    /// Search-space fingerprint bound into this checkpoint.
    #[must_use]
    pub const fn space_fingerprint(&self) -> SearchFingerprint {
        self.space_fingerprint
    }

    /// Budget bound into this checkpoint.
    #[must_use]
    pub const fn budget(&self) -> SearchBudget {
        self.budget
    }

    /// Next ordered raw ordinal to process on resume.
    #[must_use]
    pub const fn next_ordinal(&self) -> u64 {
        self.next_ordinal
    }

    /// Search statistics at snapshot time.
    #[must_use]
    pub const fn stats(&self) -> SearchStats {
        self.stats
    }

    /// Number of canonical candidate keys retained for deterministic dedup.
    #[must_use]
    pub fn seen_count(&self) -> usize {
        self.seen.len()
    }

    pub(crate) fn validate_basic(&self) -> Result<(), SearchError> {
        if self.seen.len() > usize::try_from(MAX_CHECKPOINT_SEEN).unwrap_or(usize::MAX) {
            return Err(SearchError::ExceedsLimit {
                field: "checkpoint.seen",
                value: u64::try_from(self.seen.len()).unwrap_or(u64::MAX),
                maximum: MAX_CHECKPOINT_SEEN,
            });
        }
        if self.stats.surviving != u64::try_from(self.seen.len()).unwrap_or(u64::MAX) {
            return Err(SearchError::InvalidCheckpoint(
                "surviving count differs from deduplication set".into(),
            ));
        }
        if self.stats.generated != self.next_ordinal {
            return Err(SearchError::InvalidCheckpoint(
                "generated count differs from next ordinal".into(),
            ));
        }
        if self.stats.statically_rejected > self.stats.generated
            || self.stats.duplicate > self.stats.generated
            || self.stats.surviving > self.stats.generated
        {
            return Err(SearchError::InvalidCheckpoint(
                "generation counters exceed generated candidates".into(),
            ));
        }
        let classified = self
            .stats
            .statically_rejected
            .checked_add(self.stats.duplicate)
            .and_then(|count| count.checked_add(self.stats.surviving))
            .ok_or_else(|| SearchError::InvalidCheckpoint("generation counters overflow".into()))?;
        if classified != self.stats.generated {
            return Err(SearchError::InvalidCheckpoint(
                "generation counters do not partition generated candidates".into(),
            ));
        }
        for text in &self.seen {
            if text.is_empty() || text.len() > MAX_CHECKPOINT_CANDIDATE_BYTES {
                return Err(SearchError::InvalidCheckpoint(
                    "deduplication key is empty or oversized".into(),
                ));
            }
        }
        Ok(())
    }
}

/// Configuration for the bounded semantic grammar enumerator.
#[derive(Debug, Clone, PartialEq)]
pub struct SemanticSearchConfig {
    /// Seed recorded in the space identity. The v1 order is deterministic and
    /// does not consume randomness; future randomized spaces must bind this.
    pub seed: u64,
    /// Input transformations to enumerate.
    pub input_transforms: Vec<InputTransform>,
    /// Positive scaled-dot-product affinities.
    pub affinity_scales: Vec<f64>,
    /// Visibility rules.
    pub masks: Vec<MaskRule>,
    /// Post-mask key-selection rules.
    pub selections: Vec<SelectionRule>,
    /// Normalization/weighting rules.
    pub weights: Vec<WeightRule>,
}

impl Default for SemanticSearchConfig {
    fn default() -> Self {
        Self {
            seed: 0,
            input_transforms: vec![InputTransform::Identity, InputTransform::CenterRows],
            affinity_scales: vec![1.0],
            masks: vec![MaskRule::Unmasked, MaskRule::Causal],
            selections: vec![
                SelectionRule::All,
                SelectionRule::Window { radius: 1 },
                SelectionRule::TopK { k: 1 },
            ],
            weights: vec![
                WeightRule::Softmax,
                WeightRule::SignedDifference {
                    positive_scale: 1.0,
                    negative_scale: 0.5,
                },
            ],
        }
    }
}

/// Canonicalized bounded semantic search space.
#[derive(Debug, Clone)]
pub struct SemanticSearchSpace {
    seed: u64,
    input_transforms: Vec<InputTransform>,
    affinity_scales: Vec<f64>,
    masks: Vec<MaskRule>,
    selections: Vec<SelectionRule>,
    weights: Vec<WeightRule>,
    ordered_raw_indices: Vec<u64>,
    canonical_text: String,
    fingerprint: SearchFingerprint,
}

impl SemanticSearchSpace {
    /// Validate and canonicalize a bounded semantic grammar declaration.
    ///
    /// Every raw combination is constructed and validated before the space is
    /// returned. An invalid domain cannot become a later opaque search failure.
    ///
    /// # Errors
    ///
    /// Returns an error for empty or oversized dimensions, cardinality overflow,
    /// invalid scales, or any combination rejected by the semantic IR.
    pub fn new(mut config: SemanticSearchConfig) -> Result<Self, SearchError> {
        validate_alternatives("input_transforms", config.input_transforms.len())?;
        validate_alternatives("affinity_scales", config.affinity_scales.len())?;
        validate_alternatives("masks", config.masks.len())?;
        validate_alternatives("selections", config.selections.len())?;
        validate_alternatives("weights", config.weights.len())?;
        config.affinity_scales.iter().try_for_each(|&scale| {
            if scale.is_finite() && scale > 0.0 {
                Ok(())
            } else {
                Err(SearchError::InvalidConfiguration("affinity_scales"))
            }
        })?;

        config
            .input_transforms
            .sort_by_key(|transform| input_transform_key(*transform));
        config.affinity_scales.sort_by_key(|scale| scale.to_bits());
        config.masks.sort_by_key(mask_key);
        config.selections.sort_by_key(selection_key);
        config.weights.sort_by_key(weight_key);

        let cardinality = [
            config.input_transforms.len(),
            config.affinity_scales.len(),
            config.masks.len(),
            config.selections.len(),
            config.weights.len(),
        ]
        .into_iter()
        .try_fold(1_u64, |product, count| {
            product
                .checked_mul(u64::try_from(count).unwrap_or(u64::MAX))
                .ok_or(SearchError::ExceedsLimit {
                    field: "search_space.cardinality",
                    value: u64::MAX,
                    maximum: MAX_SEARCH_SPACE_CARDINALITY,
                })
        })?;
        if cardinality > MAX_SEARCH_SPACE_CARDINALITY {
            return Err(SearchError::ExceedsLimit {
                field: "search_space.cardinality",
                value: cardinality,
                maximum: MAX_SEARCH_SPACE_CARDINALITY,
            });
        }

        let mut ordered_raw_indices = Vec::with_capacity(usize::try_from(cardinality).unwrap_or(0));
        for raw_index in 0..cardinality {
            let program = program_for_raw_index(
                raw_index,
                &config.input_transforms,
                &config.affinity_scales,
                &config.masks,
                &config.selections,
                &config.weights,
            )?;
            ordered_raw_indices.push((semantic_program_cost(&program), raw_index));
        }
        ordered_raw_indices.sort_unstable();
        let ordered_raw_indices = ordered_raw_indices
            .into_iter()
            .map(|(_, raw_index)| raw_index)
            .collect::<Vec<_>>();

        let canonical_text = canonical_space_text(&config);
        let fingerprint = SearchFingerprint::of_bytes(canonical_text.as_bytes());
        Ok(Self {
            seed: config.seed,
            input_transforms: config.input_transforms,
            affinity_scales: config.affinity_scales,
            masks: config.masks,
            selections: config.selections,
            weights: config.weights,
            ordered_raw_indices,
            canonical_text,
            fingerprint,
        })
    }

    /// Seed recorded in the canonical space identity.
    #[must_use]
    pub const fn seed(&self) -> u64 {
        self.seed
    }

    /// Number of raw combinations before search-budget pruning/deduplication.
    #[must_use]
    pub fn cardinality(&self) -> u64 {
        u64::try_from(self.ordered_raw_indices.len()).unwrap_or(u64::MAX)
    }

    /// Canonical search-space declaration.
    #[must_use]
    pub fn to_canonical_text(&self) -> &str {
        &self.canonical_text
    }

    /// Stable search-space fingerprint.
    #[must_use]
    pub const fn fingerprint(&self) -> SearchFingerprint {
        self.fingerprint
    }

    /// Reconstruct an ordered semantic candidate.
    ///
    /// # Errors
    ///
    /// Returns an error when the ordinal is outside the bounded space or the
    /// generated semantic program fails validation.
    pub fn candidate_at(&self, ordinal: u64) -> Result<SemanticProgram, SearchError> {
        let ordinal = usize::try_from(ordinal)
            .map_err(|_| SearchError::InvalidConfiguration("candidate ordinal"))?;
        let raw_index = *self
            .ordered_raw_indices
            .get(ordinal)
            .ok_or(SearchError::InvalidConfiguration("candidate ordinal"))?;
        program_for_raw_index(
            raw_index,
            &self.input_transforms,
            &self.affinity_scales,
            &self.masks,
            &self.selections,
            &self.weights,
        )
    }
}

impl SearchSpace for SemanticSearchSpace {
    type Candidate = SemanticProgram;

    fn cardinality(&self) -> u64 {
        self.cardinality()
    }

    fn fingerprint(&self) -> SearchFingerprint {
        self.fingerprint
    }

    fn candidate_at(&self, ordinal: u64) -> Result<Self::Candidate, SearchError> {
        self.candidate_at(ordinal)
    }

    fn candidate_canonical_text(&self, candidate: &Self::Candidate) -> String {
        candidate.to_canonical_text()
    }

    fn candidate_cost(&self, candidate: &Self::Candidate) -> u32 {
        semantic_program_cost(candidate)
    }
}

fn validate_alternatives(field: &'static str, count: usize) -> Result<(), SearchError> {
    if count == 0 {
        return Err(SearchError::InvalidConfiguration(field));
    }
    if count > MAX_SEARCH_ALTERNATIVES {
        return Err(SearchError::ExceedsLimit {
            field,
            value: u64::try_from(count).unwrap_or(u64::MAX),
            maximum: u64::try_from(MAX_SEARCH_ALTERNATIVES).unwrap_or(u64::MAX),
        });
    }
    Ok(())
}

fn raw_indices(
    mut raw_index: u64,
    input_transforms: &[InputTransform],
    affinity_scales: &[f64],
    masks: &[MaskRule],
    selections: &[SelectionRule],
    weights: &[WeightRule],
) -> (usize, usize, usize, usize, usize) {
    let weight =
        usize::try_from(raw_index % u64::try_from(weights.len()).unwrap_or(1)).unwrap_or(0);
    raw_index /= u64::try_from(weights.len()).unwrap_or(1);
    let selection =
        usize::try_from(raw_index % u64::try_from(selections.len()).unwrap_or(1)).unwrap_or(0);
    raw_index /= u64::try_from(selections.len()).unwrap_or(1);
    let mask = usize::try_from(raw_index % u64::try_from(masks.len()).unwrap_or(1)).unwrap_or(0);
    raw_index /= u64::try_from(masks.len()).unwrap_or(1);
    let affinity =
        usize::try_from(raw_index % u64::try_from(affinity_scales.len()).unwrap_or(1)).unwrap_or(0);
    raw_index /= u64::try_from(affinity_scales.len()).unwrap_or(1);
    let input_transform =
        usize::try_from(raw_index % u64::try_from(input_transforms.len()).unwrap_or(1))
            .unwrap_or(0);
    (input_transform, affinity, mask, selection, weight)
}

fn program_for_raw_index(
    raw_index: u64,
    input_transforms: &[InputTransform],
    affinity_scales: &[f64],
    masks: &[MaskRule],
    selections: &[SelectionRule],
    weights: &[WeightRule],
) -> Result<SemanticProgram, SearchError> {
    let (input_index, affinity_index, mask_index, selection_index, weight_index) = raw_indices(
        raw_index,
        input_transforms,
        affinity_scales,
        masks,
        selections,
        weights,
    );
    let input_transform = input_transforms[input_index];
    let affinity_scale = affinity_scales[affinity_index];
    let mask = masks[mask_index].clone();
    let selection = selections[selection_index];
    let weight = weights[weight_index];
    let (family, descriptor_weights, weight_tag) = match weight {
        WeightRule::Softmax => (
            SemanticFamily::StandardSoftmax,
            WeightContract::ProbabilitySimplex,
            "softmax".to_owned(),
        ),
        WeightRule::SignedDifference {
            positive_scale,
            negative_scale,
        } => (
            SemanticFamily::DifferentialSigned,
            WeightContract::Signed,
            format!(
                "signed-{:016x}-{:016x}",
                positive_scale.to_bits(),
                negative_scale.to_bits()
            ),
        ),
    };
    let name = format!(
        "search-i{}-a{:016x}-m{}-s{}-w{}",
        input_transform_key(input_transform),
        affinity_scale.to_bits(),
        mask_key(&mask),
        selection_key(&selection),
        weight_tag
    );
    let id = SemanticId::new(family, name, 1)
        .map_err(|_| SearchError::InvalidConfiguration("generated semantic identity"))?;
    let descriptor = SemanticDescriptor::new(
        id,
        mask_descriptor_contract(&mask),
        StateContract::Stateless,
        descriptor_weights,
    );
    SemanticProgram::new(SemanticProgramSpec {
        descriptor,
        input_transform,
        affinity: AffinityRule::ScaledDotProduct {
            scale: affinity_scale,
        },
        mask,
        selection,
        weight,
        value_mix: ValueMixRule::WeightedSum,
        output: OutputRule::Identity,
    })
    .map_err(SearchError::CandidateConstruction)
}

fn semantic_program_cost(program: &SemanticProgram) -> u32 {
    let transform_cost = match program.input_transform() {
        InputTransform::Identity => 0,
        InputTransform::CenterRows => 1,
    };
    let mask_cost = match program.mask() {
        MaskRule::Unmasked => 0,
        MaskRule::Causal => 1,
        MaskRule::External { .. } => 2,
    };
    let selection_cost = match program.selection() {
        SelectionRule::All => 0,
        SelectionRule::Window { .. } => 1,
        SelectionRule::TopK { .. } => 2,
    };
    let weight_cost = match program.weight() {
        WeightRule::Softmax => 2,
        WeightRule::SignedDifference { .. } => 4,
    };
    1 + transform_cost + 2 + mask_cost + selection_cost + weight_cost + 2
}

fn mask_descriptor_contract(mask: &MaskRule) -> MaskContract {
    match mask {
        MaskRule::Unmasked => MaskContract::Bidirectional,
        MaskRule::Causal => MaskContract::Causal,
        MaskRule::External { .. } => MaskContract::ExternalMask,
    }
}

fn input_transform_key(transform: InputTransform) -> u8 {
    match transform {
        InputTransform::Identity => 0,
        InputTransform::CenterRows => 1,
    }
}

fn mask_key(mask: &MaskRule) -> String {
    match mask {
        MaskRule::Unmasked => "unmasked".into(),
        MaskRule::Causal => "causal".into(),
        MaskRule::External { identity } => format!("external-{}", hex_encode(identity)),
    }
}

fn selection_key(selection: &SelectionRule) -> String {
    match selection {
        SelectionRule::All => "all".into(),
        SelectionRule::Window { radius } => format!("window-{radius}"),
        SelectionRule::TopK { k } => format!("top-k-{k}"),
    }
}

fn weight_key(weight: &WeightRule) -> String {
    match weight {
        WeightRule::Softmax => "softmax".into(),
        WeightRule::SignedDifference {
            positive_scale,
            negative_scale,
        } => format!(
            "signed-{:016x}-{:016x}",
            positive_scale.to_bits(),
            negative_scale.to_bits()
        ),
    }
}

fn canonical_space_text(config: &SemanticSearchConfig) -> String {
    let mut text = format!(
        "ADA-SEARCH-SPACE-V{SEARCH_SPACE_VERSION}\nseed={}\n",
        config.seed
    );
    text.push_str("input_transforms=");
    text.push_str(
        &config
            .input_transforms
            .iter()
            .map(|transform| input_transform_key(*transform).to_string())
            .collect::<Vec<_>>()
            .join(","),
    );
    text.push('\n');
    text.push_str("affinity_scales=");
    text.push_str(
        &config
            .affinity_scales
            .iter()
            .map(|scale| format!("0x{:016x}", scale.to_bits()))
            .collect::<Vec<_>>()
            .join(","),
    );
    text.push('\n');
    text.push_str("masks=");
    text.push_str(
        &config
            .masks
            .iter()
            .map(mask_key)
            .collect::<Vec<_>>()
            .join(","),
    );
    text.push('\n');
    text.push_str("selections=");
    text.push_str(
        &config
            .selections
            .iter()
            .map(selection_key)
            .collect::<Vec<_>>()
            .join(","),
    );
    text.push('\n');
    text.push_str("weights=");
    text.push_str(
        &config
            .weights
            .iter()
            .map(weight_key)
            .collect::<Vec<_>>()
            .join(","),
    );
    text.push('\n');
    text
}

fn hex_encode(value: &str) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut encoded = String::with_capacity(value.len() * 2);
    for byte in value.as_bytes() {
        encoded.push(char::from(HEX[usize::from(byte >> 4)]));
        encoded.push(char::from(HEX[usize::from(byte & 0x0f)]));
    }
    encoded
}

#[cfg(test)]
mod tests {
    use super::*;

    fn budget(max_expansions: u64, max_candidates: u64, max_cost: u32) -> SearchBudget {
        SearchBudget::new(max_expansions, max_candidates, max_cost).unwrap()
    }

    #[test]
    fn default_space_is_cost_ordered_and_candidates_are_inspectable() {
        let space = SemanticSearchSpace::new(SemanticSearchConfig::default()).unwrap();
        let mut engine =
            SearchEngine::new(space.clone(), budget(64, 64, MAX_PROGRAM_COST)).unwrap();
        let candidates = engine.run_to_end().unwrap();
        assert_eq!(
            candidates.len(),
            usize::try_from(space.cardinality()).unwrap()
        );
        assert!(candidates.windows(2).all(|pair| {
            (pair[0].cost(), pair[0].ordinal()) <= (pair[1].cost(), pair[1].ordinal())
        }));
        assert!(candidates.iter().all(|candidate| {
            !candidate.canonical_text().is_empty()
                && candidate.fingerprint()
                    == SearchFingerprint::of_bytes(candidate.canonical_text().as_bytes())
        }));
        assert!(engine.stats().generated() >= engine.stats().surviving());
    }

    #[test]
    fn identical_seed_and_space_produce_identical_candidate_order() {
        let config = SemanticSearchConfig {
            seed: 42,
            ..SemanticSearchConfig::default()
        };
        let left = SemanticSearchSpace::new(config.clone()).unwrap();
        let right = SemanticSearchSpace::new(config).unwrap();
        assert_eq!(left.to_canonical_text(), right.to_canonical_text());
        assert_eq!(left.fingerprint(), right.fingerprint());
        let mut left_engine = SearchEngine::new(left, budget(64, 64, MAX_PROGRAM_COST)).unwrap();
        let mut right_engine = SearchEngine::new(right, budget(64, 64, MAX_PROGRAM_COST)).unwrap();
        let left_text = left_engine
            .run_to_end()
            .unwrap()
            .into_iter()
            .map(|candidate| candidate.canonical_text().to_owned())
            .collect::<Vec<_>>();
        let right_text = right_engine
            .run_to_end()
            .unwrap()
            .into_iter()
            .map(|candidate| candidate.canonical_text().to_owned())
            .collect::<Vec<_>>();
        assert_eq!(left_text, right_text);
    }

    #[test]
    fn duplicate_structural_alternatives_are_removed_deterministically() {
        let config = SemanticSearchConfig {
            input_transforms: vec![InputTransform::Identity],
            affinity_scales: vec![1.0, 1.0],
            masks: vec![MaskRule::Unmasked],
            selections: vec![SelectionRule::All],
            weights: vec![WeightRule::Softmax],
            ..SemanticSearchConfig::default()
        };
        let space = SemanticSearchSpace::new(config).unwrap();
        let mut engine = SearchEngine::new(space, budget(4, 4, MAX_PROGRAM_COST)).unwrap();
        let candidates = engine.run_to_end().unwrap();
        assert_eq!(candidates.len(), 1);
        assert_eq!(engine.stats().generated(), 2);
        assert_eq!(engine.stats().duplicate(), 1);
        assert_eq!(engine.stats().surviving(), 1);
    }

    #[test]
    fn static_cost_and_expansion_budgets_terminate_without_execution() {
        let space = SemanticSearchSpace::new(SemanticSearchConfig::default()).unwrap();
        let mut cost_limited = SearchEngine::new(space.clone(), budget(4, 4, 0)).unwrap();
        assert!(cost_limited.run_to_end().unwrap().is_empty());
        assert_eq!(cost_limited.stats().generated(), 4);
        assert_eq!(cost_limited.stats().statically_rejected(), 4);

        let mut expansion_limited =
            SearchEngine::new(space, budget(2, 64, MAX_PROGRAM_COST)).unwrap();
        let candidates = expansion_limited.run_to_end().unwrap();
        assert_eq!(candidates.len(), 2);
        assert_eq!(expansion_limited.stats().generated(), 2);
    }

    #[test]
    fn checkpoint_round_trip_resumes_to_the_same_final_sequence() {
        let space = SemanticSearchSpace::new(SemanticSearchConfig::default()).unwrap();
        let budget = budget(64, 64, MAX_PROGRAM_COST);
        let mut uninterrupted = SearchEngine::new(space.clone(), budget).unwrap();
        let expected = uninterrupted
            .run_to_end()
            .unwrap()
            .into_iter()
            .map(|candidate| candidate.canonical_text().to_owned())
            .collect::<Vec<_>>();

        let mut partial = SearchEngine::new(space.clone(), budget).unwrap();
        for _ in 0..3 {
            partial.next_candidate().unwrap();
        }
        let checkpoint = partial.checkpoint();
        let text = checkpoint.to_canonical_text();
        let decoded = SearchCheckpoint::from_canonical_text(&text).unwrap();
        assert_eq!(decoded, checkpoint);
        let mut resumed = SearchEngine::from_checkpoint(space, decoded).unwrap();
        let mut actual = partial
            .run_to_end()
            .unwrap()
            .into_iter()
            .map(|candidate| candidate.canonical_text().to_owned())
            .collect::<Vec<_>>();
        let resumed_tail = resumed
            .run_to_end()
            .unwrap()
            .into_iter()
            .map(|candidate| candidate.canonical_text().to_owned())
            .collect::<Vec<_>>();
        actual.splice(0..0, expected[..3].iter().cloned());
        assert_eq!(actual, expected);
        assert_eq!(resumed_tail, expected[3..]);
    }

    #[test]
    fn checkpoint_and_configuration_fail_closed() {
        assert!(
            SemanticSearchSpace::new(SemanticSearchConfig {
                affinity_scales: vec![f64::NAN],
                ..SemanticSearchConfig::default()
            })
            .is_err()
        );
        assert!(SearchBudget::new(MAX_SEARCH_EXPANSIONS + 1, 1, 1).is_err());

        let space = SemanticSearchSpace::new(SemanticSearchConfig::default()).unwrap();
        let mut engine = SearchEngine::new(space.clone(), budget(4, 4, MAX_PROGRAM_COST)).unwrap();
        engine.next_candidate().unwrap();
        let mut checkpoint = engine.checkpoint();
        checkpoint.next_ordinal += 1;
        assert!(SearchEngine::from_checkpoint(space, checkpoint).is_err());

        let space = SemanticSearchSpace::new(SemanticSearchConfig::default()).unwrap();
        let mut engine = SearchEngine::new(space, budget(4, 4, MAX_PROGRAM_COST)).unwrap();
        engine.next_candidate().unwrap();
        let text = engine.checkpoint().to_canonical_text();
        assert!(
            SearchCheckpoint::from_canonical_text(&(text.clone() + "unknown=field\n")).is_err()
        );
        assert!(SearchCheckpoint::from_canonical_text(&text[..text.len() - 1]).is_err());
        assert!(
            SearchCheckpoint::from_canonical_text(
                &text.replace("ADA-SEARCH-CHECKPOINT-V1", "ADA-SEARCH-CHECKPOINT-V2")
            )
            .is_err()
        );
    }

    #[test]
    fn later_evidence_counters_do_not_change_candidate_identity() {
        let space = SemanticSearchSpace::new(SemanticSearchConfig::default()).unwrap();
        let mut engine = SearchEngine::new(space, budget(1, 1, MAX_PROGRAM_COST)).unwrap();
        let candidate = engine.next_candidate().unwrap().unwrap();
        let fingerprint = candidate.fingerprint();
        engine.record_oracle_falsified();
        engine.record_adversarial_falsified();
        engine.record_cost_dominated();
        assert_eq!(candidate.fingerprint(), fingerprint);
        assert_eq!(engine.stats().oracle_falsified(), 1);
        assert_eq!(engine.stats().adversarial_falsified(), 1);
        assert_eq!(engine.stats().cost_dominated(), 1);
    }
}
