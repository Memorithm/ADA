//! Task-quality contracts and lane-separated evidence fills.
//!
//! This module makes the ADA2 separation explicit:
//! - algorithmic / structural error
//! - numerical error
//! - logical / estimated / measured cost
//! - task-quality metric slots
//!
//! Non-claims (fail-closed):
//! - survival / CEGIS disposition is not task quality;
//! - ITD / TDI diagnostics are not task quality;
//! - logical / estimated / measured cost fields cannot silently fill task quality;
//! - a filled task-quality slot is not adoption, novelty, or FLAT promotion;
//! - mechanistic fixtures here are tiny deterministic oracles, not LM benchmarks.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::{Display, Formatter, Write as _};

use crate::{
    CorrectnessStatus, EstimatedCost, LogicalCost, MAX_METRIC_NAME_BYTES, MAX_QUALITY_METRICS,
    MeasuredCost, NumericalObjectives, ObjectiveDirection, ObjectiveError, ObjectiveVector,
    QualityMetric,
};

/// Version of the task-quality contract codec/schema.
pub const TASK_QUALITY_CONTRACT_VERSION: u16 = 1;
/// Canonical header for task-quality contract text.
pub const TASK_QUALITY_CONTRACT_HEADER: &str = "ADA-TASK-QUALITY-CONTRACT-V1";
/// Maximum declared task-quality slots in one contract.
pub const MAX_TASK_QUALITY_SLOTS: u64 = MAX_QUALITY_METRICS;

/// Fail-closed task-quality / evidence-lane errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TaskQualityError {
    /// Nested objective validation failed.
    Objective(ObjectiveError),
    /// A required identity or field was empty/malformed.
    InvalidField(&'static str),
    /// Unsupported contract version.
    UnsupportedVersion(u16),
    /// Canonical text was malformed.
    MalformedCanonical(String),
    /// A fill targeted an undeclared slot.
    UnknownSlot(String),
    /// A declared slot was left unfilled.
    MissingSlot(String),
    /// A fill used a source that cannot populate task quality.
    ForbiddenSource {
        /// Slot name the caller attempted to fill.
        slot: String,
        /// Rejected source label.
        source: String,
    },
    /// Two fills targeted the same slot.
    DuplicateFill(String),
    /// A candidate evidence packet mixed lanes illegally.
    LaneContamination(&'static str),
}

impl Display for TaskQualityError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Objective(error) => write!(formatter, "{error}"),
            Self::InvalidField(field) => write!(formatter, "invalid task-quality field: {field}"),
            Self::UnsupportedVersion(version) => {
                write!(
                    formatter,
                    "unsupported task-quality contract version {version}"
                )
            }
            Self::MalformedCanonical(reason) => {
                write!(formatter, "malformed task-quality contract: {reason}")
            }
            Self::UnknownSlot(name) => write!(formatter, "unknown task-quality slot: {name}"),
            Self::MissingSlot(name) => write!(formatter, "missing task-quality slot fill: {name}"),
            Self::ForbiddenSource { slot, source } => write!(
                formatter,
                "source {source} cannot fill task-quality slot {slot}"
            ),
            Self::DuplicateFill(name) => write!(formatter, "duplicate task-quality fill: {name}"),
            Self::LaneContamination(reason) => {
                write!(formatter, "evidence lane contamination: {reason}")
            }
        }
    }
}

impl std::error::Error for TaskQualityError {}

impl From<ObjectiveError> for TaskQualityError {
    fn from(value: ObjectiveError) -> Self {
        Self::Objective(value)
    }
}

/// Provenance of a candidate scalar offered as a task-quality fill.
///
/// Only task-oracle and mechanistic-fixture evaluations may populate task
/// quality. Diagnostics, cost, numerical error, and survival/CEGIS outcomes are
/// retained as separate lanes and are rejected here.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum QualityValueSource {
    /// Value produced by evaluating against an explicit task oracle.
    TaskOracleEvaluation,
    /// Value produced by a bounded mechanistic fixture oracle.
    MechanisticFixtureEvaluation,
    /// ITD structural diagnostic. Forbidden for task quality.
    ItdDiagnostic,
    /// TDI recovery/intervention diagnostic. Forbidden for task quality.
    TdiDiagnostic,
    /// Logical operation-count field. Forbidden for task quality.
    LogicalCostField,
    /// Estimated cost field. Forbidden for task quality.
    EstimatedCostField,
    /// Measured cost field. Forbidden for task quality.
    MeasuredCostField,
    /// Numerical-error diagnostic. Forbidden for task quality.
    NumericalDiagnostic,
    /// CEGIS survival / rejection disposition. Forbidden for task quality.
    SurvivalOrCegisDisposition,
}

impl QualityValueSource {
    /// Whether this source may populate a task-quality slot.
    #[must_use]
    pub const fn may_fill_task_quality(self) -> bool {
        matches!(
            self,
            Self::TaskOracleEvaluation | Self::MechanisticFixtureEvaluation
        )
    }

    fn as_text(self) -> &'static str {
        match self {
            Self::TaskOracleEvaluation => "task-oracle",
            Self::MechanisticFixtureEvaluation => "mechanistic-fixture",
            Self::ItdDiagnostic => "itd-diagnostic",
            Self::TdiDiagnostic => "tdi-diagnostic",
            Self::LogicalCostField => "logical-cost",
            Self::EstimatedCostField => "estimated-cost",
            Self::MeasuredCostField => "measured-cost",
            Self::NumericalDiagnostic => "numerical-diagnostic",
            Self::SurvivalOrCegisDisposition => "survival-or-cegis",
        }
    }

    /// Parse a canonical source label.
    ///
    /// # Errors
    ///
    /// Returns an error for an unknown label.
    pub fn from_text(value: &str) -> Result<Self, TaskQualityError> {
        match value {
            "task-oracle" => Ok(Self::TaskOracleEvaluation),
            "mechanistic-fixture" => Ok(Self::MechanisticFixtureEvaluation),
            "itd-diagnostic" => Ok(Self::ItdDiagnostic),
            "tdi-diagnostic" => Ok(Self::TdiDiagnostic),
            "logical-cost" => Ok(Self::LogicalCostField),
            "estimated-cost" => Ok(Self::EstimatedCostField),
            "measured-cost" => Ok(Self::MeasuredCostField),
            "numerical-diagnostic" => Ok(Self::NumericalDiagnostic),
            "survival-or-cegis" => Ok(Self::SurvivalOrCegisDisposition),
            _ => Err(TaskQualityError::MalformedCanonical(
                "unknown quality value source".into(),
            )),
        }
    }
}

impl Display for QualityValueSource {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.as_text())
    }
}

/// Algorithmic / structural error lane, separate from floating-point numerical
/// error and from task quality.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct AlgorithmicError {
    /// Discrete support / index-set mismatches versus an oracle.
    pub support_mismatches: Option<u64>,
    /// Discrete selection / argmax failures versus an oracle.
    pub selection_failures: Option<u64>,
    /// Other structural invariant violations.
    pub invariant_violations: Option<u64>,
}

impl AlgorithmicError {
    /// Construct an empty algorithmic-error lane.
    #[must_use]
    pub const fn empty() -> Self {
        Self {
            support_mismatches: None,
            selection_failures: None,
            invariant_violations: None,
        }
    }
}

/// One declared task-quality metric slot.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TaskQualitySlot {
    name: String,
    direction: ObjectiveDirection,
}

impl TaskQualitySlot {
    /// Declare a named task-quality slot.
    ///
    /// # Errors
    ///
    /// Returns an error for an invalid metric name.
    pub fn new(
        name: impl Into<String>,
        direction: ObjectiveDirection,
    ) -> Result<Self, TaskQualityError> {
        let name = name.into();
        validate_slot_name(&name)?;
        Ok(Self { name, direction })
    }

    /// Slot name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Declared optimization direction.
    #[must_use]
    pub const fn direction(&self) -> ObjectiveDirection {
        self.direction
    }
}

/// Versioned declaration of which task-quality metrics an experiment requires.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TaskQualityContract {
    version: u16,
    slots: Vec<TaskQualitySlot>,
}

impl TaskQualityContract {
    /// Construct a validated contract.
    ///
    /// # Errors
    ///
    /// Returns an error for duplicate names, invalid names, or slot limits.
    pub fn new(slots: Vec<TaskQualitySlot>) -> Result<Self, TaskQualityError> {
        let contract = Self {
            version: TASK_QUALITY_CONTRACT_VERSION,
            slots,
        };
        contract.validate()?;
        Ok(contract)
    }

    /// Validate names, uniqueness, and limits.
    ///
    /// # Errors
    ///
    /// Returns an error when the contract is invalid.
    pub fn validate(&self) -> Result<(), TaskQualityError> {
        if self.version != TASK_QUALITY_CONTRACT_VERSION {
            return Err(TaskQualityError::UnsupportedVersion(self.version));
        }
        let count = u64::try_from(self.slots.len()).unwrap_or(u64::MAX);
        if count > MAX_TASK_QUALITY_SLOTS {
            return Err(TaskQualityError::Objective(ObjectiveError::ExceedsLimit {
                field: "task_quality_slots",
                value: count,
                maximum: MAX_TASK_QUALITY_SLOTS,
            }));
        }
        let mut names = BTreeSet::new();
        for slot in &self.slots {
            validate_slot_name(&slot.name)?;
            if !names.insert(slot.name.as_str()) {
                return Err(TaskQualityError::Objective(
                    ObjectiveError::DuplicateMetric(slot.name.clone()),
                ));
            }
        }
        Ok(())
    }

    /// Contract version.
    #[must_use]
    pub const fn version(&self) -> u16 {
        self.version
    }

    /// Declared slots in construction order.
    #[must_use]
    pub fn slots(&self) -> &[TaskQualitySlot] {
        &self.slots
    }

    /// Apply fills fail-closed: every slot must be filled exactly once from an
    /// allowed source, and forbidden sources are rejected even if the numeric
    /// value looks plausible.
    ///
    /// # Errors
    ///
    /// Returns an error for unknown/missing/duplicate slots or forbidden
    /// sources.
    pub fn materialize_quality(
        &self,
        fills: &[TaskQualityFill],
    ) -> Result<Vec<QualityMetric>, TaskQualityError> {
        self.validate()?;
        let mut by_name = BTreeMap::new();
        for fill in fills {
            if !self.slots.iter().any(|slot| slot.name == fill.name) {
                return Err(TaskQualityError::UnknownSlot(fill.name.clone()));
            }
            if !fill.source.may_fill_task_quality() {
                return Err(TaskQualityError::ForbiddenSource {
                    slot: fill.name.clone(),
                    source: fill.source.as_text().to_owned(),
                });
            }
            if !fill.value.is_finite() {
                return Err(TaskQualityError::Objective(
                    ObjectiveError::NonFiniteMetric(fill.name.clone()),
                ));
            }
            if by_name.insert(fill.name.clone(), fill).is_some() {
                return Err(TaskQualityError::DuplicateFill(fill.name.clone()));
            }
        }
        let mut metrics = Vec::with_capacity(self.slots.len());
        for slot in &self.slots {
            let Some(fill) = by_name.remove(slot.name.as_str()) else {
                return Err(TaskQualityError::MissingSlot(slot.name.clone()));
            };
            metrics.push(QualityMetric::new(
                slot.name.clone(),
                Some(fill.value),
                slot.direction,
            )?);
        }
        if let Some(name) = by_name.into_keys().next() {
            return Err(TaskQualityError::UnknownSlot(name));
        }
        Ok(metrics)
    }

    /// Canonical deterministic text.
    #[must_use]
    pub fn to_canonical_text(&self) -> String {
        let mut text = format!("{TASK_QUALITY_CONTRACT_HEADER}\n");
        let _ = writeln!(text, "version={}", self.version);
        let _ = writeln!(text, "slot_count={}", self.slots.len());
        for (index, slot) in self.slots.iter().enumerate() {
            let _ = writeln!(
                text,
                "slot_{index}_name={}",
                hex_encode(slot.name.as_bytes())
            );
            let _ = writeln!(text, "slot_{index}_direction={}", slot.direction);
        }
        text
    }

    /// Decode a canonical contract.
    ///
    /// # Errors
    ///
    /// Returns an error for malformed or unsupported text.
    pub fn from_canonical_text(text: &str) -> Result<Self, TaskQualityError> {
        if text.contains('\r') {
            return Err(TaskQualityError::MalformedCanonical(
                "CR characters are rejected".into(),
            ));
        }
        let mut lines = text.lines();
        let header = lines
            .next()
            .ok_or_else(|| TaskQualityError::MalformedCanonical("missing header".into()))?;
        if header != TASK_QUALITY_CONTRACT_HEADER {
            return Err(TaskQualityError::MalformedCanonical(
                "unknown header".into(),
            ));
        }
        let mut fields = BTreeMap::new();
        for line in lines {
            let Some((key, value)) = line.split_once('=') else {
                return Err(TaskQualityError::MalformedCanonical(
                    "line is not key=value".into(),
                ));
            };
            if fields.insert(key.to_owned(), value.to_owned()).is_some() {
                return Err(TaskQualityError::MalformedCanonical(format!(
                    "duplicate field {key}"
                )));
            }
        }
        let version = fields
            .get("version")
            .ok_or_else(|| TaskQualityError::MalformedCanonical("missing version".into()))?
            .parse::<u16>()
            .map_err(|_| TaskQualityError::MalformedCanonical("bad version".into()))?;
        if version != TASK_QUALITY_CONTRACT_VERSION {
            return Err(TaskQualityError::UnsupportedVersion(version));
        }
        let slot_count = fields
            .get("slot_count")
            .ok_or_else(|| TaskQualityError::MalformedCanonical("missing slot_count".into()))?
            .parse::<usize>()
            .map_err(|_| TaskQualityError::MalformedCanonical("bad slot_count".into()))?;
        let mut slots = Vec::with_capacity(slot_count);
        for index in 0..slot_count {
            let name = hex_decode(
                fields
                    .remove(&format!("slot_{index}_name"))
                    .ok_or_else(|| {
                        TaskQualityError::MalformedCanonical(format!("missing slot_{index}_name"))
                    })?
                    .as_str(),
            )?;
            let direction = match fields
                .remove(&format!("slot_{index}_direction"))
                .ok_or_else(|| {
                    TaskQualityError::MalformedCanonical(format!("missing slot_{index}_direction"))
                })?
                .as_str()
            {
                "min" => ObjectiveDirection::Minimize,
                "max" => ObjectiveDirection::Maximize,
                _ => {
                    return Err(TaskQualityError::MalformedCanonical(
                        "unknown slot direction".into(),
                    ));
                }
            };
            slots.push(TaskQualitySlot::new(name, direction)?);
        }
        fields.remove("version");
        fields.remove("slot_count");
        if !fields.is_empty() {
            return Err(TaskQualityError::MalformedCanonical(
                "unexpected extra fields".into(),
            ));
        }
        Self::new(slots)
    }
}

/// One proposed fill for a declared task-quality slot.
#[derive(Debug, Clone, PartialEq)]
pub struct TaskQualityFill {
    name: String,
    value: f64,
    source: QualityValueSource,
}

impl TaskQualityFill {
    /// Construct a fill proposal. Validation of source permission happens when
    /// the contract materializes quality metrics.
    ///
    /// # Errors
    ///
    /// Returns an error for an invalid name or non-finite value.
    pub fn new(
        name: impl Into<String>,
        value: f64,
        source: QualityValueSource,
    ) -> Result<Self, TaskQualityError> {
        let name = name.into();
        validate_slot_name(&name)?;
        if !value.is_finite() {
            return Err(TaskQualityError::Objective(
                ObjectiveError::NonFiniteMetric(name),
            ));
        }
        Ok(Self {
            name,
            value,
            source,
        })
    }

    /// Slot name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Proposed metric value.
    #[must_use]
    pub const fn value(&self) -> f64 {
        self.value
    }

    /// Declared provenance.
    #[must_use]
    pub const fn source(&self) -> QualityValueSource {
        self.source
    }
}

/// Explicit inputs for [`LaneSeparatedEvidence::bind`].
#[derive(Debug, Clone, PartialEq)]
pub struct LaneSeparatedEvidenceSpec {
    /// Correctness lane.
    pub correctness: CorrectnessStatus,
    /// Algorithmic / structural error lane.
    pub algorithmic: AlgorithmicError,
    /// Numerical error lane.
    pub numerical: NumericalObjectives,
    /// Logical cost lane.
    pub logical: LogicalCost,
    /// Estimated cost lane.
    pub estimated: EstimatedCost,
    /// Measured cost lane.
    pub measured: MeasuredCost,
    /// Proposed task-quality fills (must satisfy the contract).
    pub fills: Vec<TaskQualityFill>,
}

/// Explicit multi-lane evidence packet for one candidate on one task.
///
/// Lanes are stored separately. Constructing the packet never copies cost or
/// diagnostic values into task quality.
#[derive(Debug, Clone, PartialEq)]
pub struct LaneSeparatedEvidence {
    correctness: CorrectnessStatus,
    algorithmic: AlgorithmicError,
    numerical: NumericalObjectives,
    logical: LogicalCost,
    estimated: EstimatedCost,
    measured: MeasuredCost,
    quality: Vec<QualityMetric>,
}

impl LaneSeparatedEvidence {
    /// Bind lane values after materializing task quality through a contract.
    ///
    /// # Errors
    ///
    /// Returns an error when the contract rejects fills or the resulting
    /// objective vector is invalid.
    pub fn bind(
        contract: &TaskQualityContract,
        spec: &LaneSeparatedEvidenceSpec,
    ) -> Result<Self, TaskQualityError> {
        let quality = contract.materialize_quality(&spec.fills)?;
        // Validate the projected objective vector without folding algorithmic
        // error into numerical or quality lanes.
        ObjectiveVector::from_parts(
            spec.correctness,
            spec.numerical,
            spec.logical,
            spec.estimated,
            spec.measured,
            quality.clone(),
        )?;
        Ok(Self {
            correctness: spec.correctness,
            algorithmic: spec.algorithmic,
            numerical: spec.numerical,
            logical: spec.logical,
            estimated: spec.estimated,
            measured: spec.measured,
            quality,
        })
    }

    /// Correctness lane.
    #[must_use]
    pub const fn correctness(&self) -> CorrectnessStatus {
        self.correctness
    }

    /// Algorithmic / structural error lane.
    #[must_use]
    pub const fn algorithmic(&self) -> AlgorithmicError {
        self.algorithmic
    }

    /// Numerical error lane.
    #[must_use]
    pub const fn numerical(&self) -> NumericalObjectives {
        self.numerical
    }

    /// Logical cost lane.
    #[must_use]
    pub const fn logical(&self) -> LogicalCost {
        self.logical
    }

    /// Estimated cost lane.
    #[must_use]
    pub const fn estimated(&self) -> EstimatedCost {
        self.estimated
    }

    /// Measured cost lane.
    #[must_use]
    pub const fn measured(&self) -> MeasuredCost {
        self.measured
    }

    /// Task-quality metrics materialized through the contract.
    #[must_use]
    pub fn quality(&self) -> &[QualityMetric] {
        &self.quality
    }

    /// Project into an [`ObjectiveVector`]. Algorithmic error remains a
    /// separate lane on this packet and is not folded into numerical error or
    /// quality.
    ///
    /// # Errors
    ///
    /// Returns an error when the projected vector is invalid.
    pub fn to_objective_vector(&self) -> Result<ObjectiveVector, TaskQualityError> {
        Ok(ObjectiveVector::from_parts(
            self.correctness,
            self.numerical,
            self.logical,
            self.estimated,
            self.measured,
            self.quality.clone(),
        )?)
    }
}

/// Tiny deterministic causal-argmax retrieval fixture.
///
/// Given one query score row and scalar values, the oracle selects the highest
/// score among causally visible keys (index `j <= query_index`) and returns
/// that value. This is a mechanistic unit task, not a language-model benchmark,
/// not CEGIS survival, and not an adoption claim.
#[derive(Debug, Clone, PartialEq)]
pub struct CausalArgmaxRetrievalTask {
    scores: Vec<f64>,
    values: Vec<f64>,
    query_index: usize,
}

/// Oracle expectation for [`CausalArgmaxRetrievalTask`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CausalArgmaxOracle {
    expected_index: usize,
    expected_value: f64,
}

/// Candidate answer offered against the mechanistic fixture.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CausalArgmaxCandidate {
    /// Selected key index.
    pub selected_index: usize,
    /// Returned value.
    pub selected_value: f64,
}

impl CausalArgmaxRetrievalTask {
    /// Construct a bounded fixture.
    ///
    /// # Errors
    ///
    /// Returns an error when lengths mismatch, bounds are empty, values are
    /// non-finite, or the query index is out of range.
    pub fn new(
        scores: Vec<f64>,
        values: Vec<f64>,
        query_index: usize,
    ) -> Result<Self, TaskQualityError> {
        if scores.is_empty() || values.is_empty() {
            return Err(TaskQualityError::InvalidField("mechanistic_task.empty"));
        }
        if scores.len() != values.len() {
            return Err(TaskQualityError::InvalidField(
                "mechanistic_task.score_value_length",
            ));
        }
        if query_index >= scores.len() {
            return Err(TaskQualityError::InvalidField(
                "mechanistic_task.query_index",
            ));
        }
        if scores.iter().any(|value| !value.is_finite())
            || values.iter().any(|value| !value.is_finite())
        {
            return Err(TaskQualityError::InvalidField(
                "mechanistic_task.non_finite",
            ));
        }
        Ok(Self {
            scores,
            values,
            query_index,
        })
    }

    /// Score row.
    #[must_use]
    pub fn scores(&self) -> &[f64] {
        &self.scores
    }

    /// Value row.
    #[must_use]
    pub fn values(&self) -> &[f64] {
        &self.values
    }

    /// Query position used for the causal visibility cut.
    #[must_use]
    pub const fn query_index(&self) -> usize {
        self.query_index
    }

    /// Compute the deterministic oracle expectation.
    #[must_use]
    pub fn oracle(&self) -> CausalArgmaxOracle {
        let mut best_index = 0usize;
        let mut best_score = f64::NEG_INFINITY;
        for (index, &score) in self.scores.iter().enumerate() {
            if index > self.query_index {
                break;
            }
            // Strictly prefer the first index on ties for determinism.
            if score > best_score {
                best_score = score;
                best_index = index;
            }
        }
        CausalArgmaxOracle {
            expected_index: best_index,
            expected_value: self.values[best_index],
        }
    }

    /// Default task-quality contract for this fixture: maximize exact retrieval.
    ///
    /// # Errors
    ///
    /// Returns an error if the contract cannot be constructed.
    pub fn quality_contract() -> Result<TaskQualityContract, TaskQualityError> {
        TaskQualityContract::new(vec![TaskQualitySlot::new(
            "exact_retrieval",
            ObjectiveDirection::Maximize,
        )?])
    }

    /// Grade a candidate into separated lanes.
    ///
    /// # Errors
    ///
    /// Returns an error when quality materialization fails.
    pub fn grade(
        &self,
        candidate: CausalArgmaxCandidate,
        logical: LogicalCost,
    ) -> Result<LaneSeparatedEvidence, TaskQualityError> {
        let oracle = self.oracle();
        let selection_failures = u64::from(
            candidate.selected_index != oracle.expected_index
                || candidate.selected_value.to_bits() != oracle.expected_value.to_bits(),
        );
        let exact = f64::from(u8::from(selection_failures == 0));
        let algorithmic = AlgorithmicError {
            support_mismatches: Some(0),
            selection_failures: Some(selection_failures),
            invariant_violations: Some(0),
        };
        // Numerical lane remains unset unless a separate numeric comparator runs.
        let numerical = NumericalObjectives::default();
        let fills = [TaskQualityFill::new(
            "exact_retrieval",
            exact,
            QualityValueSource::MechanisticFixtureEvaluation,
        )?];
        LaneSeparatedEvidence::bind(
            &Self::quality_contract()?,
            &LaneSeparatedEvidenceSpec {
                correctness: if selection_failures == 0 {
                    CorrectnessStatus::Provisional
                } else {
                    CorrectnessStatus::Falsified
                },
                algorithmic,
                numerical,
                logical,
                estimated: EstimatedCost::default(),
                measured: MeasuredCost::default(),
                fills: fills.to_vec(),
            },
        )
    }
}

/// Tiny deterministic masked-position retrieval fixture.
///
/// Among positions with `mask[j] == true`, the oracle selects the highest score
/// (first index on ties) and returns that value. Mask-false positions are
/// invisible even if they have larger scores. Mechanistic unit task only.
#[derive(Debug, Clone, PartialEq)]
pub struct MaskedPositionRetrievalTask {
    scores: Vec<f64>,
    values: Vec<f64>,
    mask: Vec<bool>,
}

/// Oracle expectation for [`MaskedPositionRetrievalTask`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MaskedPositionOracle {
    expected_index: usize,
    expected_value: f64,
}

/// Candidate answer offered against the masked-position fixture.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MaskedPositionCandidate {
    /// Selected key index.
    pub selected_index: usize,
    /// Returned value.
    pub selected_value: f64,
}

impl MaskedPositionRetrievalTask {
    /// Construct a bounded fixture.
    ///
    /// # Errors
    ///
    /// Returns an error when lengths mismatch, no position is visible, or values
    /// are non-finite.
    pub fn new(
        scores: Vec<f64>,
        values: Vec<f64>,
        mask: Vec<bool>,
    ) -> Result<Self, TaskQualityError> {
        if scores.is_empty() || values.is_empty() || mask.is_empty() {
            return Err(TaskQualityError::InvalidField("masked_task.empty"));
        }
        if scores.len() != values.len() || scores.len() != mask.len() {
            return Err(TaskQualityError::InvalidField(
                "masked_task.score_value_mask_length",
            ));
        }
        if !mask.iter().any(|&visible| visible) {
            return Err(TaskQualityError::InvalidField("masked_task.no_visible"));
        }
        if scores.iter().any(|value| !value.is_finite())
            || values.iter().any(|value| !value.is_finite())
        {
            return Err(TaskQualityError::InvalidField("masked_task.non_finite"));
        }
        Ok(Self {
            scores,
            values,
            mask,
        })
    }

    /// Score row.
    #[must_use]
    pub fn scores(&self) -> &[f64] {
        &self.scores
    }

    /// Value row.
    #[must_use]
    pub fn values(&self) -> &[f64] {
        &self.values
    }

    /// Visibility mask (`true` = visible).
    #[must_use]
    pub fn mask(&self) -> &[bool] {
        &self.mask
    }

    /// Compute the deterministic oracle expectation.
    #[must_use]
    pub fn oracle(&self) -> MaskedPositionOracle {
        let mut best_index = 0usize;
        let mut best_score = f64::NEG_INFINITY;
        let mut found = false;
        for (index, (&score, &visible)) in self.scores.iter().zip(&self.mask).enumerate() {
            if !visible {
                continue;
            }
            if !found || score > best_score {
                best_score = score;
                best_index = index;
                found = true;
            }
        }
        debug_assert!(found, "constructor requires at least one visible position");
        MaskedPositionOracle {
            expected_index: best_index,
            expected_value: self.values[best_index],
        }
    }

    /// Default task-quality contract: maximize exact retrieval.
    ///
    /// # Errors
    ///
    /// Returns an error if the contract cannot be constructed.
    pub fn quality_contract() -> Result<TaskQualityContract, TaskQualityError> {
        TaskQualityContract::new(vec![TaskQualitySlot::new(
            "exact_retrieval",
            ObjectiveDirection::Maximize,
        )?])
    }

    /// Grade a candidate into separated lanes.
    ///
    /// # Errors
    ///
    /// Returns an error when quality materialization fails.
    pub fn grade(
        &self,
        candidate: MaskedPositionCandidate,
        logical: LogicalCost,
    ) -> Result<LaneSeparatedEvidence, TaskQualityError> {
        let oracle = self.oracle();
        let masked_violation = u64::from(
            candidate.selected_index >= self.mask.len() || !self.mask[candidate.selected_index],
        );
        let selection_failures = u64::from(
            candidate.selected_index != oracle.expected_index
                || candidate.selected_value.to_bits() != oracle.expected_value.to_bits(),
        );
        let exact = f64::from(u8::from(selection_failures == 0 && masked_violation == 0));
        let algorithmic = AlgorithmicError {
            support_mismatches: Some(masked_violation),
            selection_failures: Some(selection_failures),
            invariant_violations: Some(0),
        };
        let fills = [TaskQualityFill::new(
            "exact_retrieval",
            exact,
            QualityValueSource::MechanisticFixtureEvaluation,
        )?];
        LaneSeparatedEvidence::bind(
            &Self::quality_contract()?,
            &LaneSeparatedEvidenceSpec {
                correctness: if selection_failures == 0 && masked_violation == 0 {
                    CorrectnessStatus::Provisional
                } else {
                    CorrectnessStatus::Falsified
                },
                algorithmic,
                numerical: NumericalObjectives::default(),
                logical,
                estimated: EstimatedCost::default(),
                measured: MeasuredCost::default(),
                fills: fills.to_vec(),
            },
        )
    }
}

impl MaskedPositionOracle {
    /// Expected selected index.
    #[must_use]
    pub const fn expected_index(self) -> usize {
        self.expected_index
    }

    /// Expected selected value.
    #[must_use]
    pub const fn expected_value(self) -> f64 {
        self.expected_value
    }
}

/// Tiny deterministic relative-offset selection fixture.
///
/// The oracle selects `values[query_index + relative_offset]` when the target
/// index is in range; construction fails closed when the offset would leave the
/// sequence. Mechanistic unit task only (not an LM benchmark).
#[derive(Debug, Clone, PartialEq)]
pub struct RelativeOffsetSelectionTask {
    values: Vec<f64>,
    query_index: usize,
    relative_offset: i32,
    target_index: usize,
}

/// Oracle expectation for [`RelativeOffsetSelectionTask`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RelativeOffsetOracle {
    expected_index: usize,
    expected_value: f64,
}

/// Candidate answer offered against the relative-offset fixture.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RelativeOffsetCandidate {
    /// Selected index.
    pub selected_index: usize,
    /// Returned value.
    pub selected_value: f64,
}

impl RelativeOffsetSelectionTask {
    /// Construct a bounded fixture.
    ///
    /// # Errors
    ///
    /// Returns an error when the value row is empty, values are non-finite, the
    /// query index is out of range, or `query_index + relative_offset` is out of
    /// range.
    pub fn new(
        values: Vec<f64>,
        query_index: usize,
        relative_offset: i32,
    ) -> Result<Self, TaskQualityError> {
        if values.is_empty() {
            return Err(TaskQualityError::InvalidField("offset_task.empty"));
        }
        if query_index >= values.len() {
            return Err(TaskQualityError::InvalidField("offset_task.query_index"));
        }
        if values.iter().any(|value| !value.is_finite()) {
            return Err(TaskQualityError::InvalidField("offset_task.non_finite"));
        }
        let target = i64::try_from(query_index)
            .ok()
            .and_then(|index| index.checked_add(i64::from(relative_offset)))
            .and_then(|index| usize::try_from(index).ok())
            .filter(|&index| index < values.len())
            .ok_or(TaskQualityError::InvalidField("offset_task.target_oob"))?;
        Ok(Self {
            values,
            query_index,
            relative_offset,
            target_index: target,
        })
    }

    /// Value row.
    #[must_use]
    pub fn values(&self) -> &[f64] {
        &self.values
    }

    /// Query position.
    #[must_use]
    pub const fn query_index(&self) -> usize {
        self.query_index
    }

    /// Relative offset applied to the query index.
    #[must_use]
    pub const fn relative_offset(&self) -> i32 {
        self.relative_offset
    }

    /// Resolved target index.
    #[must_use]
    pub const fn target_index(&self) -> usize {
        self.target_index
    }

    /// Compute the deterministic oracle expectation.
    #[must_use]
    pub fn oracle(&self) -> RelativeOffsetOracle {
        RelativeOffsetOracle {
            expected_index: self.target_index,
            expected_value: self.values[self.target_index],
        }
    }

    /// Default task-quality contract: maximize exact selection.
    ///
    /// # Errors
    ///
    /// Returns an error if the contract cannot be constructed.
    pub fn quality_contract() -> Result<TaskQualityContract, TaskQualityError> {
        TaskQualityContract::new(vec![TaskQualitySlot::new(
            "exact_selection",
            ObjectiveDirection::Maximize,
        )?])
    }

    /// Grade a candidate into separated lanes.
    ///
    /// # Errors
    ///
    /// Returns an error when quality materialization fails.
    pub fn grade(
        &self,
        candidate: RelativeOffsetCandidate,
        logical: LogicalCost,
    ) -> Result<LaneSeparatedEvidence, TaskQualityError> {
        let oracle = self.oracle();
        let selection_failures = u64::from(
            candidate.selected_index != oracle.expected_index
                || candidate.selected_value.to_bits() != oracle.expected_value.to_bits(),
        );
        let exact = f64::from(u8::from(selection_failures == 0));
        let algorithmic = AlgorithmicError {
            support_mismatches: Some(0),
            selection_failures: Some(selection_failures),
            invariant_violations: Some(0),
        };
        let fills = [TaskQualityFill::new(
            "exact_selection",
            exact,
            QualityValueSource::MechanisticFixtureEvaluation,
        )?];
        LaneSeparatedEvidence::bind(
            &Self::quality_contract()?,
            &LaneSeparatedEvidenceSpec {
                correctness: if selection_failures == 0 {
                    CorrectnessStatus::Provisional
                } else {
                    CorrectnessStatus::Falsified
                },
                algorithmic,
                numerical: NumericalObjectives::default(),
                logical,
                estimated: EstimatedCost::default(),
                measured: MeasuredCost::default(),
                fills: fills.to_vec(),
            },
        )
    }
}

impl RelativeOffsetOracle {
    /// Expected selected index.
    #[must_use]
    pub const fn expected_index(self) -> usize {
        self.expected_index
    }

    /// Expected selected value.
    #[must_use]
    pub const fn expected_value(self) -> f64 {
        self.expected_value
    }
}

/// Tiny deterministic copy-token fixture through attention weights.
///
/// The oracle takes `argmax(weights)` (first on ties) as the source token and
/// expects that token's value. This models hard attention copy, not soft mixing
/// benchmarks or LM tasks.
#[derive(Debug, Clone, PartialEq)]
pub struct CopyTokenAttentionTask {
    weights: Vec<f64>,
    values: Vec<f64>,
}

/// Oracle expectation for [`CopyTokenAttentionTask`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CopyTokenOracle {
    expected_index: usize,
    expected_value: f64,
}

/// Candidate answer offered against the copy-token fixture.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CopyTokenCandidate {
    /// Claimed source token index.
    pub selected_index: usize,
    /// Returned (copied) value.
    pub selected_value: f64,
}

impl CopyTokenAttentionTask {
    /// Construct a bounded fixture.
    ///
    /// # Errors
    ///
    /// Returns an error when lengths mismatch, rows are empty, or values are
    /// non-finite. Weights may be any finite scores; soft distributions are
    /// allowed, but the oracle still uses hard argmax for the copy target.
    pub fn new(weights: Vec<f64>, values: Vec<f64>) -> Result<Self, TaskQualityError> {
        if weights.is_empty() || values.is_empty() {
            return Err(TaskQualityError::InvalidField("copy_token.empty"));
        }
        if weights.len() != values.len() {
            return Err(TaskQualityError::InvalidField(
                "copy_token.weight_value_length",
            ));
        }
        if weights.iter().any(|value| !value.is_finite())
            || values.iter().any(|value| !value.is_finite())
        {
            return Err(TaskQualityError::InvalidField("copy_token.non_finite"));
        }
        Ok(Self { weights, values })
    }

    /// Attention weights / affinities used for hard copy.
    #[must_use]
    pub fn weights(&self) -> &[f64] {
        &self.weights
    }

    /// Token values.
    #[must_use]
    pub fn values(&self) -> &[f64] {
        &self.values
    }

    /// Compute the deterministic oracle expectation.
    #[must_use]
    pub fn oracle(&self) -> CopyTokenOracle {
        let mut best_index = 0usize;
        let mut best_weight = f64::NEG_INFINITY;
        for (index, &weight) in self.weights.iter().enumerate() {
            if weight > best_weight {
                best_weight = weight;
                best_index = index;
            }
        }
        CopyTokenOracle {
            expected_index: best_index,
            expected_value: self.values[best_index],
        }
    }

    /// Default task-quality contract: maximize exact copy.
    ///
    /// # Errors
    ///
    /// Returns an error if the contract cannot be constructed.
    pub fn quality_contract() -> Result<TaskQualityContract, TaskQualityError> {
        TaskQualityContract::new(vec![TaskQualitySlot::new(
            "exact_copy",
            ObjectiveDirection::Maximize,
        )?])
    }

    /// Grade a candidate into separated lanes.
    ///
    /// # Errors
    ///
    /// Returns an error when quality materialization fails.
    pub fn grade(
        &self,
        candidate: CopyTokenCandidate,
        logical: LogicalCost,
    ) -> Result<LaneSeparatedEvidence, TaskQualityError> {
        let oracle = self.oracle();
        let selection_failures = u64::from(
            candidate.selected_index != oracle.expected_index
                || candidate.selected_value.to_bits() != oracle.expected_value.to_bits(),
        );
        let exact = f64::from(u8::from(selection_failures == 0));
        let algorithmic = AlgorithmicError {
            support_mismatches: Some(0),
            selection_failures: Some(selection_failures),
            invariant_violations: Some(0),
        };
        let fills = [TaskQualityFill::new(
            "exact_copy",
            exact,
            QualityValueSource::MechanisticFixtureEvaluation,
        )?];
        LaneSeparatedEvidence::bind(
            &Self::quality_contract()?,
            &LaneSeparatedEvidenceSpec {
                correctness: if selection_failures == 0 {
                    CorrectnessStatus::Provisional
                } else {
                    CorrectnessStatus::Falsified
                },
                algorithmic,
                numerical: NumericalObjectives::default(),
                logical,
                estimated: EstimatedCost::default(),
                measured: MeasuredCost::default(),
                fills: fills.to_vec(),
            },
        )
    }
}

impl CopyTokenOracle {
    /// Expected source index.
    #[must_use]
    pub const fn expected_index(self) -> usize {
        self.expected_index
    }

    /// Expected copied value.
    #[must_use]
    pub const fn expected_value(self) -> f64 {
        self.expected_value
    }
}

impl CausalArgmaxOracle {
    /// Expected selected index.
    #[must_use]
    pub const fn expected_index(self) -> usize {
        self.expected_index
    }

    /// Expected selected value.
    #[must_use]
    pub const fn expected_value(self) -> f64 {
        self.expected_value
    }
}

fn validate_slot_name(name: &str) -> Result<(), TaskQualityError> {
    if name.is_empty() || name.len() > MAX_METRIC_NAME_BYTES {
        return Err(TaskQualityError::Objective(
            ObjectiveError::InvalidMetricName(name.to_owned()),
        ));
    }
    if !name.bytes().all(|byte| {
        byte.is_ascii_lowercase() || byte.is_ascii_digit() || matches!(byte, b'-' | b'_' | b'.')
    }) {
        return Err(TaskQualityError::Objective(
            ObjectiveError::InvalidMetricName(name.to_owned()),
        ));
    }
    Ok(())
}

fn hex_encode(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut encoded = String::with_capacity(bytes.len() * 2);
    for &byte in bytes {
        encoded.push(char::from(HEX[usize::from(byte >> 4)]));
        encoded.push(char::from(HEX[usize::from(byte & 0x0f)]));
    }
    encoded
}

fn hex_decode(value: &str) -> Result<String, TaskQualityError> {
    if value.len() % 2 != 0 {
        return Err(TaskQualityError::MalformedCanonical(
            "odd-length hex".into(),
        ));
    }
    let mut bytes = Vec::with_capacity(value.len() / 2);
    let mut chars = value.bytes();
    while let (Some(high), Some(low)) = (chars.next(), chars.next()) {
        let high = hex_digit(high)
            .ok_or_else(|| TaskQualityError::MalformedCanonical("non-hex digit".into()))?;
        let low = hex_digit(low)
            .ok_or_else(|| TaskQualityError::MalformedCanonical("non-hex digit".into()))?;
        bytes.push((high << 4) | low);
    }
    String::from_utf8(bytes)
        .map_err(|_| TaskQualityError::MalformedCanonical("hex payload is not UTF-8".into()))
}

fn hex_digit(value: u8) -> Option<u8> {
    match value {
        b'0'..=b'9' => Some(value - b'0'),
        b'a'..=b'f' => Some(value - b'a' + 10),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn contract_rejects_diagnostic_and_cost_fills_for_task_quality() {
        let contract = TaskQualityContract::new(vec![
            TaskQualitySlot::new("exact_retrieval", ObjectiveDirection::Maximize).unwrap(),
        ])
        .unwrap();
        for source in [
            QualityValueSource::ItdDiagnostic,
            QualityValueSource::TdiDiagnostic,
            QualityValueSource::LogicalCostField,
            QualityValueSource::EstimatedCostField,
            QualityValueSource::MeasuredCostField,
            QualityValueSource::NumericalDiagnostic,
            QualityValueSource::SurvivalOrCegisDisposition,
        ] {
            let fill = TaskQualityFill::new("exact_retrieval", 1.0, source).unwrap();
            let error = contract.materialize_quality(&[fill]).unwrap_err();
            assert!(matches!(error, TaskQualityError::ForbiddenSource { .. }));
        }
    }

    #[test]
    fn cost_only_candidate_cannot_mark_task_quality() {
        let contract = CausalArgmaxRetrievalTask::quality_contract().unwrap();
        // A candidate that only knows a low flop count still has no task-quality fill.
        let err = LaneSeparatedEvidence::bind(
            &contract,
            &LaneSeparatedEvidenceSpec {
                correctness: CorrectnessStatus::Unknown,
                algorithmic: AlgorithmicError::empty(),
                numerical: NumericalObjectives::default(),
                logical: LogicalCost {
                    flops: Some(1),
                    ..LogicalCost::default()
                },
                estimated: EstimatedCost::default(),
                measured: MeasuredCost::default(),
                fills: Vec::new(),
            },
        )
        .unwrap_err();
        assert!(matches!(err, TaskQualityError::MissingSlot(_)));

        // Attempting to copy flops into the quality slot via a cost source fails.
        let fill =
            TaskQualityFill::new("exact_retrieval", 1.0, QualityValueSource::LogicalCostField)
                .unwrap();
        assert!(matches!(
            contract.materialize_quality(&[fill]),
            Err(TaskQualityError::ForbiddenSource { .. })
        ));
    }

    #[test]
    fn mechanistic_fixture_separates_algorithmic_numerical_cost_and_quality() {
        let task = CausalArgmaxRetrievalTask::new(
            vec![0.1, 2.0, 1.5, 9.0],
            vec![10.0, 20.0, 30.0, 40.0],
            2,
        )
        .unwrap();
        let oracle = task.oracle();
        assert_eq!(oracle.expected_index(), 1);
        assert_eq!(oracle.expected_value().to_bits(), 20.0_f64.to_bits());

        let good = task
            .grade(
                CausalArgmaxCandidate {
                    selected_index: 1,
                    selected_value: 20.0,
                },
                LogicalCost {
                    flops: Some(12),
                    qk_evaluations: Some(3),
                    transcendental_operations: Some(0),
                    value_operations: Some(1),
                },
            )
            .unwrap();
        assert_eq!(good.algorithmic().selection_failures, Some(0));
        assert!(good.numerical().max_abs_error.is_none());
        assert_eq!(good.logical().flops, Some(12));
        assert_eq!(good.quality()[0].value(), Some(1.0));

        let bad = task
            .grade(
                CausalArgmaxCandidate {
                    selected_index: 3, // not causally visible
                    selected_value: 40.0,
                },
                LogicalCost {
                    flops: Some(1), // cheap but wrong
                    ..LogicalCost::default()
                },
            )
            .unwrap();
        assert_eq!(bad.algorithmic().selection_failures, Some(1));
        assert_eq!(bad.quality()[0].value(), Some(0.0));
        assert_eq!(bad.correctness(), CorrectnessStatus::Falsified);
        // Low cost did not become task quality.
        assert_ne!(bad.quality()[0].value(), Some(1.0));
    }

    #[test]
    fn task_quality_contract_codec_round_trips() {
        let contract = TaskQualityContract::new(vec![
            TaskQualitySlot::new("exact_retrieval", ObjectiveDirection::Maximize).unwrap(),
            TaskQualitySlot::new("support_jaccard", ObjectiveDirection::Maximize).unwrap(),
        ])
        .unwrap();
        let text = contract.to_canonical_text();
        assert_eq!(
            TaskQualityContract::from_canonical_text(&text).unwrap(),
            contract
        );
        assert!(TaskQualityContract::from_canonical_text(&format!("{text}extra=1\n")).is_err());
    }

    #[test]
    fn source_labels_round_trip() {
        for source in [
            QualityValueSource::TaskOracleEvaluation,
            QualityValueSource::MechanisticFixtureEvaluation,
            QualityValueSource::ItdDiagnostic,
            QualityValueSource::TdiDiagnostic,
            QualityValueSource::LogicalCostField,
            QualityValueSource::EstimatedCostField,
            QualityValueSource::MeasuredCostField,
            QualityValueSource::NumericalDiagnostic,
            QualityValueSource::SurvivalOrCegisDisposition,
        ] {
            assert_eq!(
                QualityValueSource::from_text(source.as_text()).unwrap(),
                source
            );
        }
    }

    #[test]
    fn masked_position_fixture_ignores_invisible_high_scores() {
        let task = MaskedPositionRetrievalTask::new(
            vec![0.1, 9.0, 1.5],
            vec![10.0, 20.0, 30.0],
            vec![true, false, true],
        )
        .unwrap();
        let oracle = task.oracle();
        assert_eq!(oracle.expected_index(), 2);
        assert_eq!(oracle.expected_value().to_bits(), 30.0_f64.to_bits());
        let good = task
            .grade(
                MaskedPositionCandidate {
                    selected_index: 2,
                    selected_value: 30.0,
                },
                LogicalCost::default(),
            )
            .unwrap();
        assert_eq!(good.quality()[0].value(), Some(1.0));
        let bad = task
            .grade(
                MaskedPositionCandidate {
                    selected_index: 1,
                    selected_value: 20.0,
                },
                LogicalCost {
                    flops: Some(1),
                    ..LogicalCost::default()
                },
            )
            .unwrap();
        assert_eq!(bad.algorithmic().support_mismatches, Some(1));
        assert_eq!(bad.quality()[0].value(), Some(0.0));
        assert!(MaskedPositionRetrievalTask::new(vec![1.0], vec![1.0], vec![false]).is_err());
    }

    #[test]
    fn relative_offset_fixture_is_deterministic_and_fail_closed_on_oob() {
        let task = RelativeOffsetSelectionTask::new(vec![1.0, 2.0, 3.0, 4.0], 2, -1).unwrap();
        assert_eq!(task.target_index(), 1);
        let oracle = task.oracle();
        assert_eq!(oracle.expected_index(), 1);
        assert_eq!(oracle.expected_value().to_bits(), 2.0_f64.to_bits());
        let good = task
            .grade(
                RelativeOffsetCandidate {
                    selected_index: 1,
                    selected_value: 2.0,
                },
                LogicalCost::default(),
            )
            .unwrap();
        assert_eq!(good.quality()[0].name(), "exact_selection");
        assert_eq!(good.quality()[0].value(), Some(1.0));
        assert!(RelativeOffsetSelectionTask::new(vec![1.0, 2.0], 0, -1).is_err());
        assert!(RelativeOffsetSelectionTask::new(vec![1.0, 2.0], 1, 1).is_err());
    }

    #[test]
    fn copy_token_fixture_uses_hard_argmax_of_weights() {
        let task = CopyTokenAttentionTask::new(vec![0.1, 0.8, 0.1], vec![5.0, 7.0, 9.0]).unwrap();
        let oracle = task.oracle();
        assert_eq!(oracle.expected_index(), 1);
        assert_eq!(oracle.expected_value().to_bits(), 7.0_f64.to_bits());
        let good = task
            .grade(
                CopyTokenCandidate {
                    selected_index: 1,
                    selected_value: 7.0,
                },
                LogicalCost {
                    flops: Some(3),
                    ..LogicalCost::default()
                },
            )
            .unwrap();
        assert_eq!(good.quality()[0].name(), "exact_copy");
        assert_eq!(good.quality()[0].value(), Some(1.0));
        assert_eq!(good.logical().flops, Some(3));
        let bad = task
            .grade(
                CopyTokenCandidate {
                    selected_index: 2,
                    selected_value: 9.0,
                },
                LogicalCost::default(),
            )
            .unwrap();
        assert_eq!(bad.quality()[0].value(), Some(0.0));
        assert_eq!(bad.correctness(), CorrectnessStatus::Falsified);
    }
}
