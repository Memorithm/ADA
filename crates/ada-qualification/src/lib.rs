//! Workload-bound qualification bridge for ADA semantic research.
//!
//! `ada-cegis` already owns deterministic candidate falsification and retained
//! counterexamples.  This crate does not implement a second CEGIS engine.  It
//! binds that existing machinery to an explicit [`WorkloadContract`] and then
//! permits versioned A10/E2 evidence to be attached only to a survivor for the
//! same semantic identity and workload fingerprint.

#![forbid(unsafe_code)]

use std::collections::BTreeSet;
use std::convert::Infallible;
use std::fmt::{Display, Formatter, Write as _};

use ada_a10_evidence_schema::{EvidenceWorkloadFingerprint, SemanticEvidenceRecord};
use ada_cegis::{
    AdversarialGenerator, CegisResult, DifferentialOracle, Fixture, FixtureFingerprint,
    OracleOutcome, MAX_FIXTURE_TEXT_BYTES,
};
use ada_core::DiagnosticEvidenceRef;
use ada_search::{SearchCandidate, SearchFingerprint};
use ada_semantic::{ReferenceInput, SemanticIrError, SemanticProgram};
use ada_workload::WorkloadContract;

/// Version of the workload-bound qualification-case contract.
pub const QUALIFICATION_CASE_VERSION: u16 = 1;
/// Maximum caller-supplied canonical input artifact retained in one fixture.
pub const MAX_INPUT_CANONICAL_TEXT_BYTES: usize = 256 << 10;
/// Maximum number of expected output values retained by one qualification case.
pub const MAX_EXPECTED_OUTPUT_VALUES: usize = 1 << 15;

/// Fail-closed errors from workload/oracle/evidence binding.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QualificationError {
    /// The workload itself failed its canonical validation contract.
    InvalidWorkload(String),
    /// Input/output fixture metadata is inconsistent or outside the bounded
    /// qualification domain.
    InvalidCase(&'static str),
    /// A workload is outside the executable semantic-reference domain.
    UnsupportedWorkload(String),
    /// The CEGIS fixture wrapper rejected the canonical case artifact.
    InvalidFixture(String),
    /// The requested candidate fingerprint was not present in the completed
    /// CEGIS result.
    CandidateNotFound,
    /// The requested candidate was explicitly rejected by a retained
    /// counterexample.
    CandidateFalsified,
    /// The requested workload was never evaluated in the CEGIS active corpus.
    WorkloadNotEvaluated,
    /// A survivor no longer validates against the workload it supposedly
    /// survived.  This is treated as an integration failure, not as evidence.
    SurvivorWorkloadMismatch(String),
    /// At least one E2 artifact refers to another semantic identity.
    EvidenceSemanticMismatch,
    /// At least one E2 artifact refers to another workload fingerprint.
    EvidenceWorkloadMismatch,
    /// Two byte-identical E2 artifacts were supplied to one qualification.
    DuplicateEvidence,
    /// Evidence binding was requested with an empty evidence set.
    MissingEvidence,
    /// An E2 record could not be projected to the core evidence-reference
    /// contract.
    InvalidEvidenceReference(String),
}

impl Display for QualificationError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidWorkload(reason) => write!(formatter, "invalid workload: {reason}"),
            Self::InvalidCase(field) => write!(formatter, "invalid qualification case: {field}"),
            Self::UnsupportedWorkload(reason) => {
                write!(formatter, "unsupported qualification workload: {reason}")
            }
            Self::InvalidFixture(reason) => write!(formatter, "invalid qualification fixture: {reason}"),
            Self::CandidateNotFound => formatter.write_str("candidate is absent from the CEGIS result"),
            Self::CandidateFalsified => {
                formatter.write_str("candidate has a retained CEGIS counterexample")
            }
            Self::WorkloadNotEvaluated => {
                formatter.write_str("workload was not evaluated in the CEGIS active corpus")
            }
            Self::SurvivorWorkloadMismatch(reason) => {
                write!(formatter, "survivor/workload mismatch: {reason}")
            }
            Self::EvidenceSemanticMismatch => {
                formatter.write_str("evidence semantic identity does not match the survivor")
            }
            Self::EvidenceWorkloadMismatch => {
                formatter.write_str("evidence workload fingerprint does not match qualification")
            }
            Self::DuplicateEvidence => formatter.write_str("duplicate semantic evidence artifact"),
            Self::MissingEvidence => formatter.write_str("evidence set is empty"),
            Self::InvalidEvidenceReference(reason) => {
                write!(formatter, "invalid diagnostic evidence reference: {reason}")
            }
        }
    }
}

impl std::error::Error for QualificationError {}

/// One deterministic reference case bound to an explicit workload contract.
///
/// The caller supplies canonical input text separately from the typed
/// [`ReferenceInput`].  This follows the CEGIS rule that opaque typed inputs are
/// never serialized implicitly.
#[derive(Debug, Clone, PartialEq)]
pub struct SemanticWorkloadCase {
    workload: WorkloadContract,
    workload_fingerprint: EvidenceWorkloadFingerprint,
    input: ReferenceInput,
    input_canonical_text: String,
    expected_output: Vec<f64>,
    max_abs_tolerance: f64,
}

impl SemanticWorkloadCase {
    /// Construct one bounded workload/oracle case.
    ///
    /// The expected output is caller-owned oracle truth.  The candidate
    /// evaluator never generates it.
    ///
    /// # Errors
    ///
    /// Returns an error when the workload is invalid, typed input geometry does
    /// not match the workload, canonical input text is missing/oversized, the
    /// expected output has the wrong shape or non-finite values, or the
    /// tolerance is invalid.
    pub fn new(
        workload: WorkloadContract,
        input: ReferenceInput,
        input_canonical_text: impl Into<String>,
        expected_output: Vec<f64>,
        max_abs_tolerance: f64,
    ) -> Result<Self, QualificationError> {
        workload
            .validate()
            .map_err(|error| QualificationError::InvalidWorkload(error.to_string()))?;
        let input_canonical_text = input_canonical_text.into();
        if input_canonical_text.is_empty()
            || input_canonical_text.len() > MAX_INPUT_CANONICAL_TEXT_BYTES
            || input_canonical_text.contains('\r')
        {
            return Err(QualificationError::InvalidCase("input_canonical_text"));
        }
        if !max_abs_tolerance.is_finite() || max_abs_tolerance < 0.0 {
            return Err(QualificationError::InvalidCase("max_abs_tolerance"));
        }
        if expected_output.len() > MAX_EXPECTED_OUTPUT_VALUES {
            return Err(QualificationError::InvalidCase("expected_output_limit"));
        }
        if expected_output.iter().any(|value| !value.is_finite()) {
            return Err(QualificationError::InvalidCase("expected_output_finite"));
        }

        let geometry = workload.geometry();
        let sequence = geometry.sequence_lengths();
        if sequence.batch_count() != 1 || geometry.query_heads() != 1 || geometry.kv_heads() != 1 {
            return Err(QualificationError::InvalidCase(
                "qualification reference is single-batch and single-head",
            ));
        }
        if input.query_count() != sequence.query_length_for(0).unwrap_or(0)
            || input.key_count() != sequence.kv_length_for(0).unwrap_or(0)
        {
            return Err(QualificationError::InvalidCase("sequence_geometry"));
        }
        if Some(input.q_dimension()) != geometry.qk_dimension() {
            return Err(QualificationError::InvalidCase("qk_dimension"));
        }
        if input.value_dimension() != geometry.value_dimension() {
            return Err(QualificationError::InvalidCase("value_dimension"));
        }
        let expected_len = input
            .query_count()
            .checked_mul(input.value_dimension())
            .ok_or(QualificationError::InvalidCase("expected_output_shape"))?;
        if expected_output.len() != expected_len {
            return Err(QualificationError::InvalidCase("expected_output_shape"));
        }

        let case = Self {
            workload_fingerprint: EvidenceWorkloadFingerprint::from_workload(&workload),
            workload,
            input,
            input_canonical_text,
            expected_output,
            max_abs_tolerance,
        };
        if case.to_canonical_text().len() > MAX_FIXTURE_TEXT_BYTES {
            return Err(QualificationError::InvalidCase("canonical_case_size"));
        }
        Ok(case)
    }

    /// Workload under which the candidate must be evaluated.
    #[must_use]
    pub const fn workload(&self) -> &WorkloadContract {
        &self.workload
    }

    /// Stable workload fingerprint shared with A10/E2.
    #[must_use]
    pub const fn workload_fingerprint(&self) -> EvidenceWorkloadFingerprint {
        self.workload_fingerprint
    }

    /// Typed f64 reference input.
    #[must_use]
    pub const fn input(&self) -> &ReferenceInput {
        &self.input
    }

    /// Caller-owned canonical input representation retained by CEGIS.
    #[must_use]
    pub fn input_canonical_text(&self) -> &str {
        &self.input_canonical_text
    }

    /// Independent expected output vector.
    #[must_use]
    pub fn expected_output(&self) -> &[f64] {
        &self.expected_output
    }

    /// Maximum absolute output error permitted by this case.
    #[must_use]
    pub const fn max_abs_tolerance(&self) -> f64 {
        self.max_abs_tolerance
    }

    /// Canonical text used as the CEGIS fixture identity.
    #[must_use]
    pub fn to_canonical_text(&self) -> String {
        let mut text = format!("ADA-QUALIFICATION-CASE-V{QUALIFICATION_CASE_VERSION}\n");
        let _ = writeln!(
            text,
            "workload={:016x}-{:016x}-{:016x}",
            self.workload_fingerprint.primary(),
            self.workload_fingerprint.secondary(),
            self.workload_fingerprint.length()
        );
        let _ = writeln!(text, "input={}", hex_encode(&self.input_canonical_text));
        let _ = writeln!(
            text,
            "shape={}:{}:{}:{}",
            self.input.query_count(),
            self.input.key_count(),
            self.input.q_dimension(),
            self.input.value_dimension()
        );
        let expected = self
            .expected_output
            .iter()
            .map(|value| format!("{:016x}", value.to_bits()))
            .collect::<Vec<_>>()
            .join(",");
        let _ = writeln!(text, "expected_bits={expected}");
        let _ = writeln!(
            text,
            "max_abs_tolerance_bits={:016x}",
            self.max_abs_tolerance.to_bits()
        );
        text
    }

    /// Convert this case into the generic CEGIS fixture contract.
    ///
    /// # Errors
    ///
    /// Propagates CEGIS fixture-identity validation failures.
    pub fn into_fixture(self, id: impl Into<String>) -> Result<Fixture<Self>, QualificationError> {
        let canonical_text = self.to_canonical_text();
        Fixture::new(id, canonical_text, self)
            .map_err(|error| QualificationError::InvalidFixture(error.to_string()))
    }
}

/// Differential oracle adapter that makes workload validation part of each
/// CEGIS decision.
#[derive(Debug, Clone, Copy, Default)]
pub struct SemanticWorkloadOracle;

impl DifferentialOracle<SemanticProgram, SemanticWorkloadCase> for SemanticWorkloadOracle {
    type Error = QualificationError;

    fn compare(
        &mut self,
        candidate: &SemanticProgram,
        fixture: &Fixture<SemanticWorkloadCase>,
    ) -> Result<OracleOutcome, Self::Error> {
        let case = fixture.input();
        if let Err(error) = candidate.validate_for_workload(case.workload()) {
            return match error {
                SemanticIrError::InvalidField(_) => Ok(OracleOutcome::Falsified {
                    reason: format!("workload contract mismatch: {error}"),
                }),
                other => Err(QualificationError::UnsupportedWorkload(other.to_string())),
            };
        }

        let actual = match candidate.evaluate(case.input()) {
            Ok(output) => output,
            Err(error) => {
                return Ok(OracleOutcome::Falsified {
                    reason: format!("candidate evaluation rejected: {error}"),
                });
            }
        };
        if actual.output().len() != case.expected_output().len() {
            return Ok(OracleOutcome::Falsified {
                reason: "candidate output shape differs from oracle output".into(),
            });
        }
        let max_abs_error = actual
            .output()
            .iter()
            .zip(case.expected_output())
            .map(|(actual, expected)| (actual - expected).abs())
            .fold(0.0_f64, f64::max);
        if max_abs_error <= case.max_abs_tolerance() {
            Ok(OracleOutcome::Pass)
        } else {
            Ok(OracleOutcome::Falsified {
                reason: format!(
                    "max_abs_output_error_bits={:016x};tolerance_bits={:016x}",
                    max_abs_error.to_bits(),
                    case.max_abs_tolerance().to_bits()
                ),
            })
        }
    }
}

/// Explicit no-op adversarial stage for qualification protocols that only use
/// a frozen deterministic corpus.
#[derive(Debug, Clone, Copy, Default)]
pub struct NoAdversarialGenerator;

impl<C, I> AdversarialGenerator<C, I> for NoAdversarialGenerator {
    type Error = Infallible;

    fn generate(
        &mut self,
        _seed: u64,
        _candidate: &C,
        _active: &[Fixture<I>],
    ) -> Result<Vec<Fixture<I>>, Self::Error> {
        Ok(Vec::new())
    }
}

/// A CEGIS survivor proven to have been evaluated on one explicit workload.
///
/// "Qualified" here is bounded by the completed CEGIS corpus.  It is not a
/// proof of semantic usefulness, novelty, model quality, or FLAT readiness.
#[derive(Debug, Clone)]
pub struct BoundedOracleQualification {
    candidate: SearchCandidate<SemanticProgram>,
    workload: WorkloadContract,
    workload_fingerprint: EvidenceWorkloadFingerprint,
    fixture_fingerprints: Vec<FixtureFingerprint>,
}

impl BoundedOracleQualification {
    /// Bind one completed CEGIS survivor to a workload actually present in the
    /// active corpus.
    ///
    /// # Errors
    ///
    /// Returns `CandidateFalsified` for a retained rejection, `CandidateNotFound`
    /// for an unrelated fingerprint, `WorkloadNotEvaluated` when the requested
    /// workload never appeared in the corpus, or an integration error when the
    /// survivor no longer validates against that workload.
    pub fn from_cegis_result(
        result: &CegisResult<SemanticProgram, SemanticWorkloadCase>,
        candidate_fingerprint: SearchFingerprint,
        workload: &WorkloadContract,
    ) -> Result<Self, QualificationError> {
        workload
            .validate()
            .map_err(|error| QualificationError::InvalidWorkload(error.to_string()))?;
        if result
            .rejected()
            .iter()
            .any(|rejected| rejected.candidate().fingerprint() == candidate_fingerprint)
        {
            return Err(QualificationError::CandidateFalsified);
        }
        let candidate = result
            .survivors()
            .iter()
            .find(|candidate| candidate.fingerprint() == candidate_fingerprint)
            .cloned()
            .ok_or(QualificationError::CandidateNotFound)?;
        let workload_fingerprint = EvidenceWorkloadFingerprint::from_workload(workload);
        let mut fixture_fingerprints = result
            .active_fixtures()
            .iter()
            .filter(|fixture| fixture.input().workload_fingerprint() == workload_fingerprint)
            .map(Fixture::fingerprint)
            .collect::<Vec<_>>();
        if fixture_fingerprints.is_empty() {
            return Err(QualificationError::WorkloadNotEvaluated);
        }
        candidate
            .candidate()
            .validate_for_workload(workload)
            .map_err(|error| QualificationError::SurvivorWorkloadMismatch(error.to_string()))?;
        fixture_fingerprints.sort_by_key(|fingerprint| {
            (
                fingerprint.primary(),
                fingerprint.secondary(),
                fingerprint.length(),
            )
        });
        Ok(Self {
            candidate,
            workload: workload.clone(),
            workload_fingerprint,
            fixture_fingerprints,
        })
    }

    /// Reconstructible semantic candidate that survived CEGIS.
    #[must_use]
    pub const fn candidate(&self) -> &SearchCandidate<SemanticProgram> {
        &self.candidate
    }

    /// Workload explicitly covered by this qualification.
    #[must_use]
    pub const fn workload(&self) -> &WorkloadContract {
        &self.workload
    }

    /// Workload fingerprint used by E2 evidence binding.
    #[must_use]
    pub const fn workload_fingerprint(&self) -> EvidenceWorkloadFingerprint {
        self.workload_fingerprint
    }

    /// Deterministic fixture fingerprints from the active corpus for this
    /// workload.
    #[must_use]
    pub fn fixture_fingerprints(&self) -> &[FixtureFingerprint] {
        &self.fixture_fingerprints
    }

    /// Attach versioned E2 records after verifying exact semantic and workload
    /// identity.
    ///
    /// # Errors
    ///
    /// Returns an error for an empty evidence set, semantic/workload mismatch,
    /// duplicate E2 artifacts, or a record that cannot be projected to the core
    /// diagnostic-reference contract.
    pub fn attach_evidence(
        self,
        mut evidence: Vec<SemanticEvidenceRecord>,
    ) -> Result<EvidenceBoundQualification, QualificationError> {
        if evidence.is_empty() {
            return Err(QualificationError::MissingEvidence);
        }
        let semantic = self.candidate.candidate().descriptor().id();
        let mut canonical_records = BTreeSet::new();
        let mut diagnostic_references = Vec::with_capacity(evidence.len());
        for record in &evidence {
            if record.semantic() != semantic {
                return Err(QualificationError::EvidenceSemanticMismatch);
            }
            if record.workload() != self.workload_fingerprint {
                return Err(QualificationError::EvidenceWorkloadMismatch);
            }
            if !canonical_records.insert(record.to_canonical_text()) {
                return Err(QualificationError::DuplicateEvidence);
            }
            diagnostic_references.push(record.diagnostic_reference().map_err(|error| {
                QualificationError::InvalidEvidenceReference(error.to_string())
            })?);
        }
        evidence.sort_by_key(SemanticEvidenceRecord::to_canonical_text);
        diagnostic_references.sort_by(|left, right| {
            evidence_reference_key(left).cmp(&evidence_reference_key(right))
        });
        Ok(EvidenceBoundQualification {
            oracle: self,
            evidence,
            diagnostic_references,
        })
    }
}

/// A bounded oracle survivor with explicit, identity-matched E2 evidence.
#[derive(Debug, Clone)]
pub struct EvidenceBoundQualification {
    oracle: BoundedOracleQualification,
    evidence: Vec<SemanticEvidenceRecord>,
    diagnostic_references: Vec<DiagnosticEvidenceRef>,
}

impl EvidenceBoundQualification {
    /// Oracle qualification that evidence was attached to.
    #[must_use]
    pub const fn oracle(&self) -> &BoundedOracleQualification {
        &self.oracle
    }

    /// Full versioned E2 records retained for audit and later decisions.
    #[must_use]
    pub fn evidence(&self) -> &[SemanticEvidenceRecord] {
        &self.evidence
    }

    /// Lightweight core evidence references suitable for later graduation
    /// records.  These references do not alter semantic identity.
    #[must_use]
    pub fn diagnostic_references(&self) -> &[DiagnosticEvidenceRef] {
        &self.diagnostic_references
    }
}

fn evidence_reference_key(reference: &DiagnosticEvidenceRef) -> String {
    format!(
        "{:?}|{}|{}|{}",
        reference.kind(),
        reference.repository(),
        reference.artifact(),
        reference.revision_binding()
    )
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
    use ada_a10_evidence_schema::SemanticEvidenceSpec;
    use ada_cegis::{CegisConfig, CegisEngine};
    use ada_core::{DiagnosticEvidenceKind, SemanticFamily, SemanticId};
    use ada_search::{
        SearchBudget, SearchEngine, SemanticSearchConfig, SemanticSearchSpace, MAX_PROGRAM_COST,
    };
    use ada_semantic::{
        InputTransform, MaskRule, ReferenceInputSpec, SelectionRule, WeightRule,
    };
    use ada_workload::{
        AttentionGeometry, AttentionTopology, GeometrySpec, HeadGrouping, MaskKind, MaskSpec,
        PrecisionPolicy, ScalarPrecision, SequenceLengths, WorkloadOptions,
    };

    fn workload(mask: MaskKind) -> WorkloadContract {
        let geometry = AttentionGeometry::new(GeometrySpec {
            sequence_lengths: SequenceLengths::uniform(1, 1, 2).unwrap(),
            query_heads: 1,
            kv_heads: 1,
            qk_dimension: Some(1),
            value_dimension: 1,
            topology: AttentionTopology::CrossAttention,
            head_grouping: HeadGrouping::MultiHead,
        })
        .unwrap();
        WorkloadContract::new(
            geometry,
            WorkloadOptions {
                mask: MaskSpec::new(mask).unwrap(),
                precision: PrecisionPolicy::new(
                    ScalarPrecision::F64,
                    ScalarPrecision::F64,
                    ScalarPrecision::F64,
                    ScalarPrecision::F64,
                ),
                ..WorkloadOptions::default()
            },
        )
        .unwrap()
    }

    fn reference_input() -> ReferenceInput {
        ReferenceInput::new(ReferenceInputSpec {
            query_count: 1,
            key_count: 2,
            q_dimension: 1,
            value_dimension: 1,
            queries: vec![0.0],
            keys: vec![1.0, -1.0],
            values: vec![2.0, 4.0],
            external_mask: None,
        })
        .unwrap()
    }

    fn semantic_search() -> SearchEngine<SemanticSearchSpace> {
        let space = SemanticSearchSpace::new(SemanticSearchConfig {
            seed: 11,
            input_transforms: vec![InputTransform::Identity],
            affinity_scales: vec![1.0],
            masks: vec![MaskRule::Unmasked],
            selections: vec![SelectionRule::All],
            weights: vec![
                WeightRule::Softmax,
                WeightRule::SignedDifference {
                    positive_scale: 1.0,
                    negative_scale: 0.5,
                },
            ],
        })
        .unwrap();
        SearchEngine::new(space, SearchBudget::new(8, 8, MAX_PROGRAM_COST).unwrap()).unwrap()
    }

    fn run() -> CegisResult<SemanticProgram, SemanticWorkloadCase> {
        let case = SemanticWorkloadCase::new(
            workload(MaskKind::None),
            reference_input(),
            "q=[0];k=[1,-1];v=[2,4]",
            vec![3.0],
            0.0,
        )
        .unwrap();
        CegisEngine::new(
            semantic_search(),
            SemanticWorkloadOracle,
            NoAdversarialGenerator,
            CegisConfig::default(),
            vec![case.into_fixture("equal-score-control").unwrap()],
        )
        .unwrap()
        .run_to_end()
        .unwrap()
    }

    fn survivor_qualification(
        result: &CegisResult<SemanticProgram, SemanticWorkloadCase>,
    ) -> BoundedOracleQualification {
        let survivor = result
            .survivors()
            .iter()
            .find(|candidate| candidate.candidate().weight() == WeightRule::Softmax)
            .unwrap();
        BoundedOracleQualification::from_cegis_result(
            result,
            survivor.fingerprint(),
            &workload(MaskKind::None),
        )
        .unwrap()
    }

    fn evidence_for(
        qualification: &BoundedOracleQualification,
        semantic: SemanticId,
        workload_fingerprint: EvidenceWorkloadFingerprint,
    ) -> SemanticEvidenceRecord {
        let _ = qualification;
        SemanticEvidenceRecord::new(SemanticEvidenceSpec {
            semantic,
            workload: workload_fingerprint,
            kind: DiagnosticEvidenceKind::ItdStructural,
            producer_repository: "Memorithm/itd-simulator".into(),
            producer_revision: "a".repeat(40),
            artifact_identity: "attention-structure-v1".into(),
            intervention_identity: None,
            observation_horizon: None,
            metric_identity: "localization-v1".into(),
            sha256_evidence: "b".repeat(64),
            metrics: vec![("localization".into(), 0.25)],
        })
        .unwrap()
    }

    #[test]
    fn equal_score_oracle_keeps_softmax_and_falsifies_signed_difference() {
        let result = run();
        assert_eq!(result.survivors().len(), 1);
        assert_eq!(result.rejected().len(), 1);
        assert_eq!(result.stats().oracle_falsified(), 1);
        assert_eq!(result.search_stats().oracle_falsified(), 1);
        assert_eq!(result.survivors()[0].candidate().weight(), WeightRule::Softmax);
        assert!(matches!(
            result.rejected()[0].candidate().candidate().weight(),
            WeightRule::SignedDifference { .. }
        ));
    }

    #[test]
    fn rejected_candidate_cannot_be_reintroduced_as_qualified() {
        let result = run();
        let rejected = result.rejected()[0].candidate().fingerprint();
        assert_eq!(
            BoundedOracleQualification::from_cegis_result(
                &result,
                rejected,
                &workload(MaskKind::None)
            )
            .unwrap_err(),
            QualificationError::CandidateFalsified
        );
    }

    #[test]
    fn qualification_requires_the_exact_workload_to_have_been_evaluated() {
        let result = run();
        let survivor = result.survivors()[0].fingerprint();
        assert_eq!(
            BoundedOracleQualification::from_cegis_result(
                &result,
                survivor,
                &workload(MaskKind::Causal)
            )
            .unwrap_err(),
            QualificationError::WorkloadNotEvaluated
        );
    }

    #[test]
    fn e2_evidence_binds_only_to_the_surviving_semantic_and_workload() {
        let result = run();
        let qualification = survivor_qualification(&result);
        let semantic = qualification
            .candidate()
            .candidate()
            .descriptor()
            .id()
            .clone();
        let evidence = evidence_for(
            &qualification,
            semantic,
            qualification.workload_fingerprint(),
        );
        let bound = qualification.clone().attach_evidence(vec![evidence]).unwrap();
        assert_eq!(bound.evidence().len(), 1);
        assert_eq!(bound.diagnostic_references().len(), 1);
        assert_eq!(
            bound.oracle().candidate().candidate().descriptor().id(),
            qualification.candidate().candidate().descriptor().id()
        );

        let wrong_semantic = SemanticId::new(SemanticFamily::Experimental, "wrong-semantic", 1)
            .unwrap();
        let wrong_semantic_evidence = evidence_for(
            &qualification,
            wrong_semantic,
            qualification.workload_fingerprint(),
        );
        assert_eq!(
            qualification
                .clone()
                .attach_evidence(vec![wrong_semantic_evidence])
                .unwrap_err(),
            QualificationError::EvidenceSemanticMismatch
        );

        let wrong_workload_evidence = evidence_for(
            &qualification,
            qualification
                .candidate()
                .candidate()
                .descriptor()
                .id()
                .clone(),
            EvidenceWorkloadFingerprint::from_workload(&workload(MaskKind::Causal)),
        );
        assert_eq!(
            qualification
                .attach_evidence(vec![wrong_workload_evidence])
                .unwrap_err(),
            QualificationError::EvidenceWorkloadMismatch
        );
    }

    #[test]
    fn qualification_case_identity_includes_workload_input_truth_and_tolerance() {
        let base = SemanticWorkloadCase::new(
            workload(MaskKind::None),
            reference_input(),
            "fixture-a",
            vec![3.0],
            0.0,
        )
        .unwrap();
        let changed_input = SemanticWorkloadCase::new(
            workload(MaskKind::None),
            reference_input(),
            "fixture-b",
            vec![3.0],
            0.0,
        )
        .unwrap();
        let changed_workload = SemanticWorkloadCase::new(
            workload(MaskKind::Causal),
            reference_input(),
            "fixture-a",
            vec![3.0],
            0.0,
        )
        .unwrap();
        assert_ne!(base.to_canonical_text(), changed_input.to_canonical_text());
        assert_ne!(base.to_canonical_text(), changed_workload.to_canonical_text());
        assert!(base.to_canonical_text().contains("expected_bits=4008000000000000"));
    }
}
