//! Typed multi-objective records and deterministic Pareto archiving.
//!
//! This crate keeps correctness, numerical error, logical cost, estimated
//! cost, measured cost, and task quality as separate dimensions. It never
//! invents weights to collapse them into one score. Missing optional
//! dimensions are incomparable rather than silently imputed.

#![forbid(unsafe_code)]

use std::cmp::Ordering;
use std::collections::BTreeSet;
use std::fmt::{Display, Formatter};

mod codec;

pub use codec::{OBJECTIVE_TEXT_HEADER, OBJECTIVE_VECTOR_VERSION};

/// Maximum number of task-quality dimensions in one objective vector.
pub const MAX_QUALITY_METRICS: u64 = 64;
/// Maximum candidate identity text retained by an archive entry.
pub const MAX_CANDIDATE_KEY_BYTES: usize = 1 << 20;
/// Maximum quality metric name length in bytes.
pub const MAX_METRIC_NAME_BYTES: usize = 256;
/// Maximum objective-vector canonical text length.
pub const MAX_OBJECTIVE_TEXT_BYTES: usize = 1 << 20;
/// Maximum Pareto entries retained in one archive.
pub const MAX_ARCHIVE_ENTRIES: u64 = 1 << 16;
/// Maximum insertion decisions retained for audit/reconstruction.
pub const MAX_ARCHIVE_DECISIONS: u64 = 1 << 16;

/// Fail-closed objective-vector or Pareto-archive errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ObjectiveError {
    /// A required field or identity was empty or structurally invalid.
    InvalidField(&'static str),
    /// A quality metric name violated the canonical name contract.
    InvalidMetricName(String),
    /// Two quality metrics used the same canonical name.
    DuplicateMetric(String),
    /// A numerical metric was NaN or infinite.
    NonFiniteMetric(String),
    /// An error metric was negative.
    NegativeMetric(String),
    /// A bounded value exceeded an explicit limit.
    ExceedsLimit {
        /// Field whose value was rejected.
        field: &'static str,
        /// Rejected value.
        value: u64,
        /// Inclusive maximum.
        maximum: u64,
    },
    /// Two vectors cannot be compared because their named-quality schemas
    /// differ.
    SchemaMismatch,
    /// A candidate identity already exists in the archive.
    DuplicateCandidate(String),
    /// The archive has no capacity for an additional non-dominated entry.
    ArchiveFull,
    /// A canonical objective artifact is malformed or non-canonical.
    MalformedCanonical(String),
}

impl Display for ObjectiveError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidField(field) => write!(formatter, "invalid objective field: {field}"),
            Self::InvalidMetricName(name) => {
                write!(formatter, "invalid quality metric name: {name}")
            }
            Self::DuplicateMetric(name) => write!(formatter, "duplicate quality metric: {name}"),
            Self::NonFiniteMetric(name) => write!(formatter, "non-finite metric: {name}"),
            Self::NegativeMetric(name) => write!(formatter, "negative error metric: {name}"),
            Self::ExceedsLimit {
                field,
                value,
                maximum,
            } => write!(formatter, "{field}={value} exceeds maximum {maximum}"),
            Self::SchemaMismatch => write!(formatter, "objective quality schemas differ"),
            Self::DuplicateCandidate(key) => {
                write!(formatter, "duplicate candidate identity: {key}")
            }
            Self::ArchiveFull => write!(formatter, "Pareto archive capacity is exhausted"),
            Self::MalformedCanonical(reason) => {
                write!(formatter, "malformed objective canonical text: {reason}")
            }
        }
    }
}

impl std::error::Error for ObjectiveError {}

/// Direction of one scalar objective dimension.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ObjectiveDirection {
    /// Lower values are preferred.
    Minimize,
    /// Higher values are preferred.
    Maximize,
}

impl Display for ObjectiveDirection {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(match self {
            Self::Minimize => "min",
            Self::Maximize => "max",
        })
    }
}

/// Correctness state is a first-class objective, not a filtering side effect.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CorrectnessStatus {
    /// No correctness qualification has been performed.
    Unknown,
    /// A retained counterexample falsified the candidate.
    Falsified,
    /// The candidate passed a bounded but non-exhaustive qualification.
    Provisional,
    /// The declared correctness protocol qualified the candidate.
    Qualified,
}

impl CorrectnessStatus {
    fn rank(self) -> u8 {
        match self {
            Self::Falsified => 0,
            Self::Unknown => 1,
            Self::Provisional => 2,
            Self::Qualified => 3,
        }
    }
}

impl Display for CorrectnessStatus {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(match self {
            Self::Unknown => "unknown",
            Self::Falsified => "falsified",
            Self::Provisional => "provisional",
            Self::Qualified => "qualified",
        })
    }
}

/// Numerical error objectives. `None` means the dimension was not measured.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct NumericalObjectives {
    /// Maximum absolute output error.
    pub max_abs_error: Option<f64>,
    /// Maximum ULP error where an ULP comparison is meaningful.
    pub max_ulp_error: Option<u64>,
    /// Absolute normalization or log-sum-exp error.
    pub normalization_error: Option<f64>,
}

/// Logical operation-count objectives.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct LogicalCost {
    /// Logical floating-point operations.
    pub flops: Option<u64>,
    /// Query/key affinity evaluations.
    pub qk_evaluations: Option<u64>,
    /// Transcendental operations.
    pub transcendental_operations: Option<u64>,
    /// Value mixing operations.
    pub value_operations: Option<u64>,
}

/// Estimated, non-measured cost dimensions.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct EstimatedCost {
    /// Estimated bytes moved across the modeled memory hierarchy.
    pub bytes_moved: Option<u64>,
    /// Estimated temporary/workspace bytes.
    pub workspace_bytes: Option<u64>,
    /// Estimated KV-cache footprint bytes.
    pub kv_cache_bytes: Option<u64>,
    /// Estimated index-construction operations.
    pub index_construction: Option<u64>,
    /// Estimated communication bytes.
    pub communication_bytes: Option<u64>,
    /// Estimated reduction operations.
    pub reduction_operations: Option<u64>,
}

/// Measured physical cost dimensions. A value is evidence only when supplied
/// by a named measurement protocol.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MeasuredCost {
    /// Measured latency in nanoseconds.
    pub latency_ns: Option<u64>,
    /// Measured energy in nanojoules.
    pub energy_nj: Option<u64>,
}

/// One optional task/model-quality objective with an explicit direction.
#[derive(Debug, Clone, PartialEq)]
pub struct QualityMetric {
    name: String,
    value: Option<f64>,
    direction: ObjectiveDirection,
}

impl QualityMetric {
    /// Construct a named quality metric.
    ///
    /// `None` records that the dimension is part of the schema but was not
    /// measured for this candidate.
    ///
    /// # Errors
    ///
    /// Returns an error for an invalid name or a non-finite value.
    pub fn new(
        name: impl Into<String>,
        value: Option<f64>,
        direction: ObjectiveDirection,
    ) -> Result<Self, ObjectiveError> {
        let name = name.into();
        validate_metric_name(&name)?;
        validate_optional_finite(&name, value)?;
        Ok(Self {
            name,
            value,
            direction,
        })
    }

    /// Stable metric name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Optional measured metric value.
    #[must_use]
    pub const fn value(&self) -> Option<f64> {
        self.value
    }

    /// Optimization direction declared by the experiment.
    #[must_use]
    pub const fn direction(&self) -> ObjectiveDirection {
        self.direction
    }
}

/// A validated vector of independent correctness, error, cost, and quality
/// dimensions.
#[derive(Debug, Clone, PartialEq)]
pub struct ObjectiveVector {
    correctness: CorrectnessStatus,
    numerical: NumericalObjectives,
    logical: LogicalCost,
    estimated: EstimatedCost,
    measured: MeasuredCost,
    quality: Vec<QualityMetric>,
}

impl ObjectiveVector {
    /// Construct a vector with no optional measurements.
    #[must_use]
    pub fn new(correctness: CorrectnessStatus) -> Self {
        Self {
            correctness,
            numerical: NumericalObjectives::default(),
            logical: LogicalCost::default(),
            estimated: EstimatedCost::default(),
            measured: MeasuredCost::default(),
            quality: Vec::new(),
        }
    }

    /// Construct and validate all objective sections at once.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid numerical values, duplicate quality names,
    /// or a quality limit violation. Quality names are canonicalized into
    /// lexical order before the vector is stored.
    pub fn from_parts(
        correctness: CorrectnessStatus,
        numerical: NumericalObjectives,
        logical: LogicalCost,
        estimated: EstimatedCost,
        measured: MeasuredCost,
        mut quality: Vec<QualityMetric>,
    ) -> Result<Self, ObjectiveError> {
        let quality_count = u64::try_from(quality.len()).unwrap_or(u64::MAX);
        if quality_count > MAX_QUALITY_METRICS {
            return Err(ObjectiveError::ExceedsLimit {
                field: "quality_metrics",
                value: quality_count,
                maximum: MAX_QUALITY_METRICS,
            });
        }
        quality.sort_by(|left, right| left.name.cmp(&right.name));
        let vector = Self {
            correctness,
            numerical,
            logical,
            estimated,
            measured,
            quality,
        };
        vector.validate()?;
        Ok(vector)
    }

    /// Replace numerical objectives while preserving validation.
    ///
    /// # Errors
    ///
    /// Returns an error when the resulting vector is invalid.
    pub fn with_numerical(
        mut self,
        numerical: NumericalObjectives,
    ) -> Result<Self, ObjectiveError> {
        self.numerical = numerical;
        self.validate()?;
        Ok(self)
    }

    /// Replace logical operation objectives while preserving validation.
    ///
    /// # Errors
    ///
    /// Returns an error when the resulting vector is invalid.
    pub fn with_logical(mut self, logical: LogicalCost) -> Result<Self, ObjectiveError> {
        self.logical = logical;
        self.validate()?;
        Ok(self)
    }

    /// Replace estimated cost objectives while preserving validation.
    ///
    /// # Errors
    ///
    /// Returns an error when the resulting vector is invalid.
    pub fn with_estimated(mut self, estimated: EstimatedCost) -> Result<Self, ObjectiveError> {
        self.estimated = estimated;
        self.validate()?;
        Ok(self)
    }

    /// Replace measured cost objectives while preserving validation.
    ///
    /// # Errors
    ///
    /// Returns an error when the resulting vector is invalid.
    pub fn with_measured(mut self, measured: MeasuredCost) -> Result<Self, ObjectiveError> {
        self.measured = measured;
        self.validate()?;
        Ok(self)
    }

    /// Replace named quality objectives and canonicalize their order.
    ///
    /// # Errors
    ///
    /// Returns an error when a quality metric is invalid or duplicated.
    pub fn with_quality(mut self, mut quality: Vec<QualityMetric>) -> Result<Self, ObjectiveError> {
        let quality_count = u64::try_from(quality.len()).unwrap_or(u64::MAX);
        if quality_count > MAX_QUALITY_METRICS {
            return Err(ObjectiveError::ExceedsLimit {
                field: "quality_metrics",
                value: quality_count,
                maximum: MAX_QUALITY_METRICS,
            });
        }
        quality.sort_by(|left, right| left.name.cmp(&right.name));
        self.quality = quality;
        self.validate()?;
        Ok(self)
    }

    /// Correctness state.
    #[must_use]
    pub const fn correctness(&self) -> CorrectnessStatus {
        self.correctness
    }

    /// Numerical error dimensions.
    #[must_use]
    pub const fn numerical(&self) -> NumericalObjectives {
        self.numerical
    }

    /// Logical operation dimensions.
    #[must_use]
    pub const fn logical(&self) -> LogicalCost {
        self.logical
    }

    /// Estimated cost dimensions.
    #[must_use]
    pub const fn estimated(&self) -> EstimatedCost {
        self.estimated
    }

    /// Measured cost dimensions.
    #[must_use]
    pub const fn measured(&self) -> MeasuredCost {
        self.measured
    }

    /// Named task/model-quality dimensions in canonical name order.
    #[must_use]
    pub fn quality(&self) -> &[QualityMetric] {
        &self.quality
    }

    /// Validate all finite, bounded, and schema-local objective constraints.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid error metrics, duplicate names, or limits.
    pub fn validate(&self) -> Result<(), ObjectiveError> {
        validate_optional_error("max_abs_error", self.numerical.max_abs_error)?;
        validate_optional_error("normalization_error", self.numerical.normalization_error)?;
        for metric in &self.quality {
            validate_metric_name(&metric.name)?;
            validate_optional_finite(&metric.name, metric.value)?;
        }
        let mut names = BTreeSet::new();
        for metric in &self.quality {
            if !names.insert(metric.name.as_str()) {
                return Err(ObjectiveError::DuplicateMetric(metric.name.clone()));
            }
        }
        let quality_count = u64::try_from(self.quality.len()).unwrap_or(u64::MAX);
        if quality_count > MAX_QUALITY_METRICS {
            return Err(ObjectiveError::ExceedsLimit {
                field: "quality_metrics",
                value: quality_count,
                maximum: MAX_QUALITY_METRICS,
            });
        }
        Ok(())
    }

    /// Determine strict Pareto dominance without applying scalar weights.
    ///
    /// A vector dominates another only when it is no worse on every comparable
    /// dimension and strictly better on at least one. An optional dimension is
    /// comparable only when both vectors provide it.
    ///
    /// # Errors
    ///
    /// Returns [`ObjectiveError::SchemaMismatch`] when named quality schemas
    /// differ.
    pub fn dominates(&self, other: &Self) -> Result<bool, ObjectiveError> {
        self.validate()?;
        other.validate()?;
        if self.quality_schema() != other.quality_schema() {
            return Err(ObjectiveError::SchemaMismatch);
        }
        let mut not_worse = true;
        let mut strictly_better = false;
        compare_rank(
            self.correctness.rank(),
            other.correctness.rank(),
            &mut not_worse,
            &mut strictly_better,
        );
        compare_optional(
            self.numerical.max_abs_error,
            other.numerical.max_abs_error,
            ObjectiveDirection::Minimize,
            &mut not_worse,
            &mut strictly_better,
        );
        compare_optional(
            self.numerical.max_ulp_error,
            other.numerical.max_ulp_error,
            ObjectiveDirection::Minimize,
            &mut not_worse,
            &mut strictly_better,
        );
        compare_optional(
            self.numerical.normalization_error,
            other.numerical.normalization_error,
            ObjectiveDirection::Minimize,
            &mut not_worse,
            &mut strictly_better,
        );
        compare_logical(
            self.logical,
            other.logical,
            &mut not_worse,
            &mut strictly_better,
        );
        compare_estimated(
            self.estimated,
            other.estimated,
            &mut not_worse,
            &mut strictly_better,
        );
        compare_optional(
            self.measured.latency_ns,
            other.measured.latency_ns,
            ObjectiveDirection::Minimize,
            &mut not_worse,
            &mut strictly_better,
        );
        compare_optional(
            self.measured.energy_nj,
            other.measured.energy_nj,
            ObjectiveDirection::Minimize,
            &mut not_worse,
            &mut strictly_better,
        );
        for (left, right) in self.quality.iter().zip(&other.quality) {
            compare_optional(
                left.value,
                right.value,
                left.direction,
                &mut not_worse,
                &mut strictly_better,
            );
        }
        Ok(not_worse && strictly_better)
    }

    /// Encode the objective vector as canonical deterministic text.
    #[must_use]
    pub fn to_canonical_text(&self) -> String {
        self.encode_canonical()
    }

    /// Decode and validate canonical objective text.
    ///
    /// # Errors
    ///
    /// Returns an error for malformed, non-canonical, unsupported, duplicate,
    /// or oversized fields.
    pub fn from_canonical_text(text: &str) -> Result<Self, ObjectiveError> {
        Self::decode_canonical(text)
    }

    fn quality_schema(&self) -> Vec<(String, ObjectiveDirection)> {
        self.quality
            .iter()
            .map(|metric| (metric.name.clone(), metric.direction))
            .collect()
    }
}

/// Stable candidate identity supplied by the caller.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CandidateKey {
    canonical_text: String,
    fingerprint: CandidateFingerprint,
}

impl CandidateKey {
    /// Construct a key from a semantic/implementation/workload-bound canonical
    /// identity. The text is opaque to this crate and is never treated as a
    /// novelty claim.
    ///
    /// # Errors
    ///
    /// Returns an error for empty, oversized, or CR-containing text.
    pub fn new(canonical_text: impl Into<String>) -> Result<Self, ObjectiveError> {
        let canonical_text = canonical_text.into();
        if canonical_text.is_empty() || canonical_text.len() > MAX_CANDIDATE_KEY_BYTES {
            return Err(ObjectiveError::InvalidField("candidate_key.canonical_text"));
        }
        if canonical_text.contains('\r') {
            return Err(ObjectiveError::InvalidField("candidate_key.canonical_text"));
        }
        Ok(Self {
            fingerprint: CandidateFingerprint::of_bytes(canonical_text.as_bytes()),
            canonical_text,
        })
    }

    /// Candidate canonical identity text.
    #[must_use]
    pub fn canonical_text(&self) -> &str {
        &self.canonical_text
    }

    /// Stable dual-lane fingerprint of the candidate identity text.
    #[must_use]
    pub const fn fingerprint(&self) -> CandidateFingerprint {
        self.fingerprint
    }
}

/// Stable dual-lane candidate fingerprint.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CandidateFingerprint {
    primary: u64,
    secondary: u64,
    length: u64,
}

impl CandidateFingerprint {
    fn of_bytes(bytes: &[u8]) -> Self {
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

    /// Canonical identity byte length.
    #[must_use]
    pub const fn length(self) -> u64 {
        self.length
    }
}

impl Display for CandidateFingerprint {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "{:016x}-{:016x}-{:016x}",
            self.primary, self.secondary, self.length
        )
    }
}

/// One candidate and its independently supplied objective vector.
#[derive(Debug, Clone)]
pub struct ParetoEntry<T> {
    key: CandidateKey,
    objectives: ObjectiveVector,
    payload: T,
}

impl<T> ParetoEntry<T> {
    /// Construct a validated archive entry.
    ///
    /// # Errors
    ///
    /// Returns an error when the objective vector is invalid.
    pub fn new(
        key: CandidateKey,
        objectives: ObjectiveVector,
        payload: T,
    ) -> Result<Self, ObjectiveError> {
        objectives.validate()?;
        Ok(Self {
            key,
            objectives,
            payload,
        })
    }

    /// Candidate identity.
    #[must_use]
    pub const fn key(&self) -> &CandidateKey {
        &self.key
    }

    /// Objective vector used for dominance decisions.
    #[must_use]
    pub const fn objectives(&self) -> &ObjectiveVector {
        &self.objectives
    }

    /// Borrow the candidate payload.
    #[must_use]
    pub const fn payload(&self) -> &T {
        &self.payload
    }

    /// Consume the entry and return the payload.
    #[must_use]
    pub fn into_payload(self) -> T {
        self.payload
    }
}

/// Result class for one archive insertion.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ParetoDisposition {
    /// Candidate entered the frontier.
    Inserted,
    /// Candidate was dominated by an existing frontier entry.
    Dominated,
    /// Candidate identity was already present and was not overwritten.
    Duplicate,
}

/// Explain one archive decision in a deterministic, queryable form.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParetoDecision {
    sequence: u64,
    candidate_key: String,
    disposition: ParetoDisposition,
    dominator: Option<String>,
    removed: Vec<String>,
}

impl ParetoDecision {
    /// Monotone insertion-decision sequence number.
    #[must_use]
    pub const fn sequence(&self) -> u64 {
        self.sequence
    }

    /// Candidate identity considered by the decision.
    #[must_use]
    pub fn candidate_key(&self) -> &str {
        &self.candidate_key
    }

    /// Decision class.
    #[must_use]
    pub const fn disposition(&self) -> ParetoDisposition {
        self.disposition
    }

    /// Existing entry that dominated the candidate, if any.
    #[must_use]
    pub fn dominator(&self) -> Option<&str> {
        self.dominator.as_deref()
    }

    /// Frontier entries removed because the candidate dominated them.
    #[must_use]
    pub fn removed(&self) -> &[String] {
        &self.removed
    }
}

/// Details returned from one insertion, also retained in the archive log.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParetoInsertOutcome {
    candidate_key: String,
    disposition: ParetoDisposition,
    dominator: Option<String>,
    removed: Vec<String>,
}

impl ParetoInsertOutcome {
    /// Candidate identity considered by the insertion.
    #[must_use]
    pub fn candidate_key(&self) -> &str {
        &self.candidate_key
    }

    /// Insertion result.
    #[must_use]
    pub const fn disposition(&self) -> ParetoDisposition {
        self.disposition
    }

    /// Existing entry that dominated the candidate, if any.
    #[must_use]
    pub fn dominator(&self) -> Option<&str> {
        self.dominator.as_deref()
    }

    /// Frontier entries removed by a newly dominant candidate.
    #[must_use]
    pub fn removed(&self) -> &[String] {
        &self.removed
    }
}

/// Deterministic bounded Pareto frontier with a decision log.
pub struct ParetoArchive<T> {
    entries: Vec<ParetoEntry<T>>,
    decisions: Vec<ParetoDecision>,
    quality_schema: Option<Vec<(String, ObjectiveDirection)>>,
}

impl<T> Default for ParetoArchive<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> ParetoArchive<T> {
    /// Create an empty bounded archive.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            entries: Vec::new(),
            decisions: Vec::new(),
            quality_schema: None,
        }
    }

    /// Current non-dominated entries in canonical candidate-key order.
    #[must_use]
    pub fn entries(&self) -> &[ParetoEntry<T>] {
        &self.entries
    }

    /// All insertion decisions in insertion order.
    #[must_use]
    pub fn decisions(&self) -> &[ParetoDecision] {
        &self.decisions
    }

    /// Number of current frontier entries.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the frontier is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Insert a candidate and update the strict Pareto frontier.
    ///
    /// Equal objective vectors with different candidate identities are both
    /// retained. A duplicate identity is never overwritten, so new evidence
    /// cannot silently mutate a prior archive entry.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid objectives, quality-schema mismatch,
    /// decision-log exhaustion, or frontier capacity exhaustion. Errors do not
    /// mutate the archive.
    pub fn insert(&mut self, entry: ParetoEntry<T>) -> Result<ParetoInsertOutcome, ObjectiveError> {
        let key = entry.key.canonical_text.clone();
        entry.objectives.validate()?;
        if u64::try_from(self.decisions.len()).unwrap_or(u64::MAX) >= MAX_ARCHIVE_DECISIONS {
            return Err(ObjectiveError::ExceedsLimit {
                field: "archive.decisions",
                value: u64::try_from(self.decisions.len()).unwrap_or(u64::MAX) + 1,
                maximum: MAX_ARCHIVE_DECISIONS,
            });
        }
        if let Some(schema) = &self.quality_schema {
            if schema != &entry.objectives.quality_schema() {
                return Err(ObjectiveError::SchemaMismatch);
            }
        }
        if self
            .entries
            .iter()
            .any(|existing| existing.key == entry.key)
        {
            return Ok(self.record_decision(ParetoInsertOutcome {
                candidate_key: key,
                disposition: ParetoDisposition::Duplicate,
                dominator: None,
                removed: Vec::new(),
            }));
        }

        let mut dominator = None;
        for existing in &self.entries {
            if existing.objectives.dominates(&entry.objectives)? {
                dominator = Some(existing.key.canonical_text.clone());
                break;
            }
        }
        if let Some(dominator) = dominator {
            return Ok(self.record_decision(ParetoInsertOutcome {
                candidate_key: key,
                disposition: ParetoDisposition::Dominated,
                dominator: Some(dominator),
                removed: Vec::new(),
            }));
        }

        let mut removed = Vec::new();
        for existing in &self.entries {
            if entry.objectives.dominates(&existing.objectives)? {
                removed.push(existing.key.canonical_text.clone());
            }
        }
        let final_count = self
            .entries
            .len()
            .saturating_sub(removed.len())
            .saturating_add(1);
        if u64::try_from(final_count).unwrap_or(u64::MAX) > MAX_ARCHIVE_ENTRIES {
            return Err(ObjectiveError::ArchiveFull);
        }
        let removed_keys = removed.iter().collect::<BTreeSet<_>>();
        let old_entries = std::mem::take(&mut self.entries);
        self.entries = old_entries
            .into_iter()
            .filter(|existing| !removed_keys.contains(&existing.key.canonical_text))
            .collect();
        self.entries.push(entry);
        self.entries
            .sort_by_key(|entry| entry.key.canonical_text.clone());
        self.quality_schema = Some(self.entries[0].objectives.quality_schema());
        Ok(self.record_decision(ParetoInsertOutcome {
            candidate_key: key,
            disposition: ParetoDisposition::Inserted,
            dominator: None,
            removed,
        }))
    }

    fn record_decision(&mut self, outcome: ParetoInsertOutcome) -> ParetoInsertOutcome {
        let sequence = u64::try_from(self.decisions.len()).unwrap_or(u64::MAX);
        self.decisions.push(ParetoDecision {
            sequence,
            candidate_key: outcome.candidate_key.clone(),
            disposition: outcome.disposition,
            dominator: outcome.dominator.clone(),
            removed: outcome.removed.clone(),
        });
        outcome
    }
}

fn validate_metric_name(name: &str) -> Result<(), ObjectiveError> {
    if name.is_empty() || name.len() > MAX_METRIC_NAME_BYTES {
        return Err(ObjectiveError::InvalidMetricName(name.to_owned()));
    }
    if !name.bytes().all(|byte| {
        byte.is_ascii_lowercase() || byte.is_ascii_digit() || matches!(byte, b'-' | b'_' | b'.')
    }) {
        return Err(ObjectiveError::InvalidMetricName(name.to_owned()));
    }
    Ok(())
}

fn validate_optional_finite(name: &str, value: Option<f64>) -> Result<(), ObjectiveError> {
    if value.is_some_and(|value| !value.is_finite()) {
        return Err(ObjectiveError::NonFiniteMetric(name.to_owned()));
    }
    Ok(())
}

fn validate_optional_error(name: &'static str, value: Option<f64>) -> Result<(), ObjectiveError> {
    validate_optional_finite(name, value)?;
    if value.is_some_and(|value| value < 0.0) {
        return Err(ObjectiveError::NegativeMetric(name.to_owned()));
    }
    Ok(())
}

fn compare_rank(left: u8, right: u8, not_worse: &mut bool, strictly_better: &mut bool) {
    match left.cmp(&right) {
        Ordering::Less => *not_worse = false,
        Ordering::Equal => {}
        Ordering::Greater => *strictly_better = true,
    }
}

fn compare_optional<T: PartialOrd + PartialEq>(
    left: Option<T>,
    right: Option<T>,
    direction: ObjectiveDirection,
    not_worse: &mut bool,
    strictly_better: &mut bool,
) {
    let (Some(left), Some(right)) = (left, right) else {
        return;
    };
    let ordering = left.partial_cmp(&right).unwrap_or(Ordering::Equal);
    match direction {
        ObjectiveDirection::Minimize => match ordering {
            Ordering::Less => *strictly_better = true,
            Ordering::Equal => {}
            Ordering::Greater => *not_worse = false,
        },
        ObjectiveDirection::Maximize => match ordering {
            Ordering::Less => *not_worse = false,
            Ordering::Equal => {}
            Ordering::Greater => *strictly_better = true,
        },
    }
}

fn compare_logical(
    left: LogicalCost,
    right: LogicalCost,
    not_worse: &mut bool,
    strictly_better: &mut bool,
) {
    compare_optional(
        left.flops,
        right.flops,
        ObjectiveDirection::Minimize,
        not_worse,
        strictly_better,
    );
    compare_optional(
        left.qk_evaluations,
        right.qk_evaluations,
        ObjectiveDirection::Minimize,
        not_worse,
        strictly_better,
    );
    compare_optional(
        left.transcendental_operations,
        right.transcendental_operations,
        ObjectiveDirection::Minimize,
        not_worse,
        strictly_better,
    );
    compare_optional(
        left.value_operations,
        right.value_operations,
        ObjectiveDirection::Minimize,
        not_worse,
        strictly_better,
    );
}

fn compare_estimated(
    left: EstimatedCost,
    right: EstimatedCost,
    not_worse: &mut bool,
    strictly_better: &mut bool,
) {
    compare_optional(
        left.bytes_moved,
        right.bytes_moved,
        ObjectiveDirection::Minimize,
        not_worse,
        strictly_better,
    );
    compare_optional(
        left.workspace_bytes,
        right.workspace_bytes,
        ObjectiveDirection::Minimize,
        not_worse,
        strictly_better,
    );
    compare_optional(
        left.kv_cache_bytes,
        right.kv_cache_bytes,
        ObjectiveDirection::Minimize,
        not_worse,
        strictly_better,
    );
    compare_optional(
        left.index_construction,
        right.index_construction,
        ObjectiveDirection::Minimize,
        not_worse,
        strictly_better,
    );
    compare_optional(
        left.communication_bytes,
        right.communication_bytes,
        ObjectiveDirection::Minimize,
        not_worse,
        strictly_better,
    );
    compare_optional(
        left.reduction_operations,
        right.reduction_operations,
        ObjectiveDirection::Minimize,
        not_worse,
        strictly_better,
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn quality(name: &str, value: f64, direction: ObjectiveDirection) -> QualityMetric {
        QualityMetric::new(name, Some(value), direction).unwrap()
    }

    fn vector(correctness: CorrectnessStatus, error: f64, flops: u64) -> ObjectiveVector {
        ObjectiveVector::from_parts(
            correctness,
            NumericalObjectives {
                max_abs_error: Some(error),
                max_ulp_error: Some(4),
                normalization_error: Some(error),
            },
            LogicalCost {
                flops: Some(flops),
                qk_evaluations: Some(flops / 2),
                transcendental_operations: Some(2),
                value_operations: Some(flops / 2),
            },
            EstimatedCost {
                bytes_moved: Some(flops * 8),
                workspace_bytes: Some(1024),
                kv_cache_bytes: Some(2048),
                index_construction: Some(3),
                communication_bytes: Some(4),
                reduction_operations: Some(5),
            },
            MeasuredCost {
                latency_ns: Some(flops),
                energy_nj: Some(flops * 2),
            },
            vec![quality("task_accuracy", 0.9, ObjectiveDirection::Maximize)],
        )
        .unwrap()
    }

    fn entry<T>(key: &str, objectives: ObjectiveVector, payload: T) -> ParetoEntry<T> {
        ParetoEntry::new(CandidateKey::new(key).unwrap(), objectives, payload).unwrap()
    }

    #[test]
    fn objective_vector_codec_round_trips_exact_float_bits() {
        let vector = vector(CorrectnessStatus::Qualified, 0.001, 100);
        let text = vector.to_canonical_text();
        let decoded = ObjectiveVector::from_canonical_text(&text).unwrap();
        assert_eq!(decoded, vector);
        assert_eq!(
            decoded.numerical().max_abs_error.unwrap().to_bits(),
            0x3f50_624d_d2f1_a9fc_u64
        );
        assert!(ObjectiveVector::from_canonical_text(&(text.clone() + "unknown=x\n")).is_err());
        assert!(ObjectiveVector::from_canonical_text(&text[..text.len() - 1]).is_err());
        assert!(
            ObjectiveVector::from_canonical_text(
                &text.replace(OBJECTIVE_TEXT_HEADER, "ADA-OBJECTIVE-V2")
            )
            .is_err()
        );
    }

    #[test]
    fn invalid_objectives_and_quality_schema_fail_closed() {
        assert!(QualityMetric::new("Bad Name", Some(1.0), ObjectiveDirection::Maximize).is_err());
        assert!(QualityMetric::new("loss", Some(f64::NAN), ObjectiveDirection::Minimize).is_err());
        assert!(
            ObjectiveVector::from_parts(
                CorrectnessStatus::Unknown,
                NumericalObjectives {
                    max_abs_error: Some(-1.0),
                    ..NumericalObjectives::default()
                },
                LogicalCost::default(),
                EstimatedCost::default(),
                MeasuredCost::default(),
                Vec::new(),
            )
            .is_err()
        );
        let duplicate =
            QualityMetric::new("loss", Some(1.0), ObjectiveDirection::Minimize).unwrap();
        assert!(
            ObjectiveVector::from_parts(
                CorrectnessStatus::Unknown,
                NumericalObjectives::default(),
                LogicalCost::default(),
                EstimatedCost::default(),
                MeasuredCost::default(),
                vec![duplicate.clone(), duplicate],
            )
            .is_err()
        );
        let left = ObjectiveVector::new(CorrectnessStatus::Provisional)
            .with_quality(vec![quality("accuracy", 0.9, ObjectiveDirection::Maximize)])
            .unwrap();
        let right = ObjectiveVector::new(CorrectnessStatus::Provisional)
            .with_quality(vec![quality("loss", 0.1, ObjectiveDirection::Minimize)])
            .unwrap();
        assert_eq!(left.dominates(&right), Err(ObjectiveError::SchemaMismatch));
    }

    #[test]
    fn dominance_is_strict_weight_free_and_missing_dimensions_are_incomparable() {
        let better = vector(CorrectnessStatus::Provisional, 0.1, 10);
        let worse = vector(CorrectnessStatus::Provisional, 0.2, 20);
        assert!(better.dominates(&worse).unwrap());
        assert!(!worse.dominates(&better).unwrap());

        let tradeoff = vector(CorrectnessStatus::Provisional, 0.05, 30);
        assert!(!better.dominates(&tradeoff).unwrap());
        assert!(!tradeoff.dominates(&better).unwrap());

        let no_cost = ObjectiveVector::new(CorrectnessStatus::Provisional);
        let measured_cost = no_cost
            .clone()
            .with_logical(LogicalCost {
                flops: Some(1),
                ..LogicalCost::default()
            })
            .unwrap();
        assert!(!no_cost.dominates(&measured_cost).unwrap());
        assert!(!measured_cost.dominates(&no_cost).unwrap());
    }

    #[test]
    fn archive_records_dominance_and_keeps_equal_identity_alternatives() {
        let mut archive = ParetoArchive::new();
        let first = archive
            .insert(entry(
                "semantic=s;implementation=a",
                vector(CorrectnessStatus::Provisional, 0.1, 10),
                "a",
            ))
            .unwrap();
        assert_eq!(first.disposition(), ParetoDisposition::Inserted);
        let dominated = archive
            .insert(entry(
                "semantic=s;implementation=b",
                vector(CorrectnessStatus::Provisional, 0.2, 20),
                "b",
            ))
            .unwrap();
        assert_eq!(dominated.disposition(), ParetoDisposition::Dominated);
        assert_eq!(dominated.dominator(), Some("semantic=s;implementation=a"));
        let equal = archive
            .insert(entry(
                "semantic=s;implementation=c",
                vector(CorrectnessStatus::Provisional, 0.1, 10),
                "c",
            ))
            .unwrap();
        assert_eq!(equal.disposition(), ParetoDisposition::Inserted);
        assert_eq!(archive.len(), 2);
        assert_eq!(archive.decisions().len(), 3);

        let duplicate = archive
            .insert(entry(
                "semantic=s;implementation=a",
                vector(CorrectnessStatus::Qualified, 0.0, 1),
                "replacement",
            ))
            .unwrap();
        assert_eq!(duplicate.disposition(), ParetoDisposition::Duplicate);
        assert_eq!(*archive.entries()[0].payload(), "a");
    }

    #[test]
    fn archive_final_frontier_is_deterministic_across_insertion_order() {
        let items = [
            (
                "candidate-c",
                vector(CorrectnessStatus::Provisional, 0.2, 20),
            ),
            (
                "candidate-a",
                vector(CorrectnessStatus::Provisional, 0.1, 10),
            ),
            (
                "candidate-b",
                vector(CorrectnessStatus::Provisional, 0.05, 30),
            ),
        ];
        let build = |order: &[usize]| {
            let mut archive = ParetoArchive::new();
            for &index in order {
                archive
                    .insert(entry(
                        items[index].0,
                        items[index].1.clone(),
                        items[index].0,
                    ))
                    .unwrap();
            }
            archive
                .entries()
                .iter()
                .map(|entry| entry.key().canonical_text().to_owned())
                .collect::<Vec<_>>()
        };
        assert_eq!(build(&[0, 1, 2]), build(&[2, 1, 0]));
    }

    #[test]
    fn candidate_identity_and_objective_identity_stay_separate() {
        let objectives = vector(CorrectnessStatus::Qualified, 0.0, 10);
        let left = CandidateKey::new("semantic=one;implementation=tile-a;workload=w").unwrap();
        let right = CandidateKey::new("semantic=one;implementation=tile-b;workload=w").unwrap();
        assert_ne!(left, right);
        assert_ne!(left.fingerprint(), right.fingerprint());
        let mut archive = ParetoArchive::new();
        archive
            .insert(ParetoEntry::new(left, objectives.clone(), "tile-a").unwrap())
            .unwrap();
        archive
            .insert(ParetoEntry::new(right, objectives, "tile-b").unwrap())
            .unwrap();
        assert_eq!(archive.len(), 2);
    }
}
