//! Fail-closed interchange contract for structured mathematical operators.
//!
//! This module implements the ADA side of the RB5 structured-operator export
//! boundary from `Memorithm/riemann_ndim_bench`. Importing a record preserves
//! mathematical provenance and evidence class; it does not qualify the operator
//! as an attention/sequence mechanism and does not transfer Riemann/zeta
//! interpretation, AI utility, model-quality, complexity, or hardware claims.

/// Version of the ADA structured-operator import schema.
pub const STRUCTURED_OPERATOR_IMPORT_VERSION: u16 = 1;

/// Maximum UTF-8 byte length accepted for one identifier-like field.
pub const MAX_OPERATOR_IDENTIFIER_BYTES: usize = 128;

/// Mathematical evidence class carried from the producing repository.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OperatorEvidenceClass {
    /// Complete finite identity/derivation under stated assumptions.
    ExactIdentity,
    /// Reproduction of a source-locked formula or benchmark.
    SourceLockedReproduction,
    /// Finite-dimensional numerical validation only.
    NumericalValidation,
    /// Formal asymptotic derivation with an explicitly open proof gap.
    FormalAsymptotic,
    /// Exploratory mathematical hypothesis.
    Heuristic,
}

/// Bounded downstream destination for an imported operator hypothesis.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PreferredDownstreamRoute {
    /// ADA may build an independent semantic candidate and oracle ladder.
    AdaSemanticCandidate,
    /// TDI may define a separately preregistered experiment.
    TdiExperiment,
    /// ITD may study the operator in its isolated AI research namespace.
    ItdResearch,
    /// A genuinely general primitive may be reviewed independently for SciRust.
    SciRustPrimitiveReview,
}

/// One deterministic fixture bound to exact artifact bytes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OperatorFixtureRef {
    /// Repository-relative fixture path or immutable artifact identifier.
    pub artifact: String,
    /// Lowercase SHA-256 of the referenced fixture bytes.
    pub sha256: String,
}

/// Source identity for the mathematical record being imported.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OperatorSourceRef {
    /// Producing repository in `owner/name` form.
    pub repository: String,
    /// Exact 40-hex Git commit containing the defining source/derivation.
    pub git_commit: String,
    /// Repository-relative defining document, formula record, or artifact.
    pub artifact: String,
    /// Lowercase SHA-256 of the defining artifact bytes.
    pub artifact_sha256: String,
}

/// ADA-side representation of one RB5 structured-operator export record.
///
/// Proved properties, numerically validated properties, and open gaps are
/// intentionally separate fields so evidence strength cannot be upgraded by
/// serialization or downstream convenience.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StructuredOperatorImportV1 {
    /// Must equal [`STRUCTURED_OPERATOR_IMPORT_VERSION`].
    pub schema_version: u16,
    /// Stable RB5 identifier such as `RB5-OP-TOEPLITZ`.
    pub operator_id: String,
    /// Positive producer-controlled revision of this operator record.
    pub operator_version: u32,
    /// Weakest applicable mathematical evidence class.
    pub evidence_class: OperatorEvidenceClass,
    /// Exact provenance of the mathematical definition.
    pub source: OperatorSourceRef,
    /// Finite mathematical definition, with conventions explicit enough to
    /// reconstruct a bounded reference implementation.
    pub mathematical_definition: String,
    /// Finite domain, dimensions, indexing and boundary conventions.
    pub finite_domain_and_dimensions: String,
    /// Explicit allowed parameter domain.
    pub parameter_domain: String,
    /// Properties supported by proof/identity at the imported scope.
    pub proved_properties: Vec<String>,
    /// Properties supported only by numerical validation.
    pub numerically_validated_properties: Vec<String>,
    /// Known mathematical gaps relevant to the imported statement.
    pub open_gaps: Vec<String>,
    /// Optional deterministic fixtures with immutable byte identities.
    pub reference_fixtures: Vec<OperatorFixtureRef>,
    /// Interpretations that must not transfer into ADA or other AI benches.
    pub non_transferable_interpretations: Vec<String>,
    /// Preferred bounded research destination. This is not promotion approval.
    pub preferred_downstream_route: PreferredDownstreamRoute,
}

/// Fail-closed validation errors for [`StructuredOperatorImportV1`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StructuredOperatorImportError {
    /// Unsupported interchange schema.
    UnsupportedSchemaVersion(u16),
    /// Producer record revision is zero.
    ZeroOperatorVersion,
    /// Operator identifier is malformed.
    InvalidOperatorId,
    /// Repository is not in `owner/name` form.
    InvalidRepository,
    /// Source Git commit is not exactly 40 lowercase hexadecimal characters.
    InvalidGitCommit,
    /// A SHA-256 field is malformed.
    InvalidSha256(&'static str),
    /// A required text field is empty or has surrounding whitespace.
    InvalidText(&'static str),
    /// A list item is empty or has surrounding whitespace.
    InvalidListItem(&'static str),
    /// A list contains a duplicate item.
    DuplicateListItem(&'static str),
    /// `formal_asymptotic` evidence omitted the proof gap it is required to
    /// preserve.
    FormalAsymptoticWithoutOpenGap,
    /// `exact_identity` was declared without any proved property.
    ExactIdentityWithoutProvedProperty,
    /// No non-transferable interpretation was stated, which would leave the
    /// cross-domain claim boundary ambiguous.
    MissingNonTransferableInterpretation,
}

impl std::fmt::Display for StructuredOperatorImportError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl std::error::Error for StructuredOperatorImportError {}

impl StructuredOperatorImportV1 {
    /// Validate provenance, evidence-class, and claim-boundary invariants.
    ///
    /// # Errors
    ///
    /// Returns a fail-closed error on the first violated interchange rule.
    pub fn validate(&self) -> Result<(), StructuredOperatorImportError> {
        if self.schema_version != STRUCTURED_OPERATOR_IMPORT_VERSION {
            return Err(StructuredOperatorImportError::UnsupportedSchemaVersion(
                self.schema_version,
            ));
        }
        if self.operator_version == 0 {
            return Err(StructuredOperatorImportError::ZeroOperatorVersion);
        }
        validate_operator_id(&self.operator_id)?;
        validate_repository(&self.source.repository)?;
        validate_git_commit(&self.source.git_commit)?;
        validate_text(&self.source.artifact, "source.artifact")?;
        validate_sha256(&self.source.artifact_sha256, "source.artifact_sha256")?;
        validate_text(
            &self.mathematical_definition,
            "mathematical_definition",
        )?;
        validate_text(
            &self.finite_domain_and_dimensions,
            "finite_domain_and_dimensions",
        )?;
        validate_text(&self.parameter_domain, "parameter_domain")?;
        validate_unique_text_list(&self.proved_properties, "proved_properties")?;
        validate_unique_text_list(
            &self.numerically_validated_properties,
            "numerically_validated_properties",
        )?;
        validate_unique_text_list(&self.open_gaps, "open_gaps")?;
        validate_unique_text_list(
            &self.non_transferable_interpretations,
            "non_transferable_interpretations",
        )?;

        if self.non_transferable_interpretations.is_empty() {
            return Err(
                StructuredOperatorImportError::MissingNonTransferableInterpretation,
            );
        }
        if self.evidence_class == OperatorEvidenceClass::FormalAsymptotic
            && self.open_gaps.is_empty()
        {
            return Err(StructuredOperatorImportError::FormalAsymptoticWithoutOpenGap);
        }
        if self.evidence_class == OperatorEvidenceClass::ExactIdentity
            && self.proved_properties.is_empty()
        {
            return Err(StructuredOperatorImportError::ExactIdentityWithoutProvedProperty);
        }

        for fixture in &self.reference_fixtures {
            validate_text(&fixture.artifact, "reference_fixtures.artifact")?;
            validate_sha256(&fixture.sha256, "reference_fixtures.sha256")?;
        }
        ensure_unique_fixture_artifacts(&self.reference_fixtures)?;
        Ok(())
    }
}

fn validate_operator_id(value: &str) -> Result<(), StructuredOperatorImportError> {
    let valid = value.starts_with("RB5-OP-")
        && value.len() <= MAX_OPERATOR_IDENTIFIER_BYTES
        && value.bytes().all(|byte| {
            byte.is_ascii_uppercase() || byte.is_ascii_digit() || byte == b'-'
        });
    if valid {
        Ok(())
    } else {
        Err(StructuredOperatorImportError::InvalidOperatorId)
    }
}

fn validate_repository(value: &str) -> Result<(), StructuredOperatorImportError> {
    let mut parts = value.split('/');
    let owner = parts.next().unwrap_or_default();
    let repository = parts.next().unwrap_or_default();
    let valid_part = |part: &str| {
        !part.is_empty()
            && part
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-'))
    };
    if valid_part(owner) && valid_part(repository) && parts.next().is_none() {
        Ok(())
    } else {
        Err(StructuredOperatorImportError::InvalidRepository)
    }
}

fn validate_git_commit(value: &str) -> Result<(), StructuredOperatorImportError> {
    if value.len() == 40 && is_lower_hex(value) {
        Ok(())
    } else {
        Err(StructuredOperatorImportError::InvalidGitCommit)
    }
}

fn validate_sha256(
    value: &str,
    field: &'static str,
) -> Result<(), StructuredOperatorImportError> {
    if value.len() == 64 && is_lower_hex(value) {
        Ok(())
    } else {
        Err(StructuredOperatorImportError::InvalidSha256(field))
    }
}

fn is_lower_hex(value: &str) -> bool {
    value
        .bytes()
        .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn validate_text(
    value: &str,
    field: &'static str,
) -> Result<(), StructuredOperatorImportError> {
    if !value.is_empty() && value.trim() == value {
        Ok(())
    } else {
        Err(StructuredOperatorImportError::InvalidText(field))
    }
}

fn validate_unique_text_list(
    values: &[String],
    field: &'static str,
) -> Result<(), StructuredOperatorImportError> {
    let mut seen = std::collections::BTreeSet::new();
    for value in values {
        if value.is_empty() || value.trim() != value {
            return Err(StructuredOperatorImportError::InvalidListItem(field));
        }
        if !seen.insert(value.as_str()) {
            return Err(StructuredOperatorImportError::DuplicateListItem(field));
        }
    }
    Ok(())
}

fn ensure_unique_fixture_artifacts(
    fixtures: &[OperatorFixtureRef],
) -> Result<(), StructuredOperatorImportError> {
    let mut seen = std::collections::BTreeSet::new();
    for fixture in fixtures {
        if !seen.insert(fixture.artifact.as_str()) {
            return Err(StructuredOperatorImportError::DuplicateListItem(
                "reference_fixtures.artifact",
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hex(fill: char, count: usize) -> String {
        std::iter::repeat(fill).take(count).collect()
    }

    fn valid_record() -> StructuredOperatorImportV1 {
        StructuredOperatorImportV1 {
            schema_version: STRUCTURED_OPERATOR_IMPORT_VERSION,
            operator_id: "RB5-OP-TOEPLITZ".into(),
            operator_version: 1,
            evidence_class: OperatorEvidenceClass::ExactIdentity,
            source: OperatorSourceRef {
                repository: "Memorithm/riemann_ndim_bench".into(),
                git_commit: hex('a', 40),
                artifact: "docs/example_operator.md".into(),
                artifact_sha256: hex('b', 64),
            },
            mathematical_definition: "Finite symmetric Toeplitz matrix T[i,j]=c[|i-j|].".into(),
            finite_domain_and_dimensions: "n in 1..=64; i,j in 0..n; finite real coefficients.".into(),
            parameter_domain: "c[k] finite binary64 reference values for 0<=k<n.".into(),
            proved_properties: vec!["Toeplitz structure under the stated indexing.".into()],
            numerically_validated_properties: vec!["Reference eigensolver fixture agrees within declared tolerance.".into()],
            open_gaps: Vec::new(),
            reference_fixtures: vec![OperatorFixtureRef {
                artifact: "fixtures/toeplitz_v1.txt".into(),
                sha256: hex('c', 64),
            }],
            non_transferable_interpretations: vec![
                "Riemann/zeta interpretation does not transfer.".into(),
                "No attention quality or performance claim transfers.".into(),
            ],
            preferred_downstream_route: PreferredDownstreamRoute::AdaSemanticCandidate,
        }
    }

    #[test]
    fn conservative_rb5_record_validates() {
        assert_eq!(valid_record().validate(), Ok(()));
    }

    #[test]
    fn formal_asymptotic_record_must_preserve_open_gap() {
        let mut record = valid_record();
        record.evidence_class = OperatorEvidenceClass::FormalAsymptotic;
        record.open_gaps.clear();
        assert_eq!(
            record.validate(),
            Err(StructuredOperatorImportError::FormalAsymptoticWithoutOpenGap)
        );
    }

    #[test]
    fn exact_identity_requires_a_proved_property() {
        let mut record = valid_record();
        record.proved_properties.clear();
        assert_eq!(
            record.validate(),
            Err(StructuredOperatorImportError::ExactIdentityWithoutProvedProperty)
        );
    }

    #[test]
    fn claim_boundary_cannot_be_omitted() {
        let mut record = valid_record();
        record.non_transferable_interpretations.clear();
        assert_eq!(
            record.validate(),
            Err(StructuredOperatorImportError::MissingNonTransferableInterpretation)
        );
    }

    #[test]
    fn provenance_hashes_fail_closed() {
        let mut record = valid_record();
        record.source.git_commit = "ABC".into();
        assert_eq!(
            record.validate(),
            Err(StructuredOperatorImportError::InvalidGitCommit)
        );

        let mut record = valid_record();
        record.reference_fixtures[0].sha256 = "deadbeef".into();
        assert_eq!(
            record.validate(),
            Err(StructuredOperatorImportError::InvalidSha256(
                "reference_fixtures.sha256"
            ))
        );
    }

    #[test]
    fn operator_identity_is_rb5_scoped() {
        let mut record = valid_record();
        record.operator_id = "ADA-OP-TOEPLITZ".into();
        assert_eq!(
            record.validate(),
            Err(StructuredOperatorImportError::InvalidOperatorId)
        );
    }

    #[test]
    fn duplicate_fixtures_are_rejected() {
        let mut record = valid_record();
        record.reference_fixtures.push(record.reference_fixtures[0].clone());
        assert_eq!(
            record.validate(),
            Err(StructuredOperatorImportError::DuplicateListItem(
                "reference_fixtures.artifact"
            ))
        );
    }
}
