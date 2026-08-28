//! Replayable, bit-exact fixture interchange for ADA graduation artifacts.
//!
//! Historical qualification cases deliberately accepted caller-owned opaque
//! input text. That is sufficient for evidence identity, but not for a
//! downstream implementation such as FLAT to reconstruct the exact Q/K/V data
//! that CEGIS evaluated. This crate defines a strict canonical reference-input
//! artifact and helpers that embed it inside the existing qualification V1
//! `input` field. The existing qualification and graduation formats therefore
//! remain unchanged while new fixtures can be proven replayable.
//!
//! Replay remains finite-corpus evidence. Passing these fixtures is not a proof
//! of general correctness, model quality, novelty, or hardware performance.

#![forbid(unsafe_code)]

use std::fmt::{Display, Formatter, Write as _};

use ada_cegis::Fixture;
use ada_graduation::{FlatGraduationBundle, OracleFixtureArtifact};
use ada_qualification::{
    MAX_INPUT_CANONICAL_TEXT_BYTES, QUALIFICATION_CASE_VERSION, SemanticWorkloadCase,
};
use ada_semantic::{ReferenceInput, ReferenceInputSpec, SemanticIrError};
use ada_workload::WorkloadContract;

/// Canonical exact-reference-input schema version.
pub const REFERENCE_INPUT_VERSION: u16 = 1;
/// Canonical reference-input artifact header.
pub const REFERENCE_INPUT_HEADER: &str = "ADA-REFERENCE-INPUT-V1";

/// Fail-closed replay construction, decoding, or verification failures.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReplayError {
    /// The exact reference input could not be constructed or decoded.
    ReferenceInput(String),
    /// Qualification-case construction failed.
    Qualification(String),
    /// CEGIS fixture construction failed.
    Fixture(String),
    /// Canonical replay text is malformed or non-canonical.
    MalformedCanonical(String),
    /// The qualification case belongs to another workload.
    WorkloadMismatch,
    /// The redundant qualification shape disagrees with the exact input.
    ShapeMismatch,
    /// Rebuilding the fixture did not reproduce its retained CEGIS identity.
    FixtureFingerprintMismatch,
    /// A legacy opaque qualification input is not replayable by this contract.
    NonReplayableFixture,
    /// The semantic failed one retained replay fixture.
    OracleReplayMismatch {
        /// Failing fixture identifier.
        fixture_id: String,
        /// IEEE-754 bits of the measured maximum absolute error.
        max_abs_error_bits: u64,
        /// IEEE-754 bits of the allowed tolerance.
        tolerance_bits: u64,
    },
}

impl Display for ReplayError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ReferenceInput(reason) => write!(formatter, "reference-input error: {reason}"),
            Self::Qualification(reason) => write!(formatter, "qualification error: {reason}"),
            Self::Fixture(reason) => write!(formatter, "fixture error: {reason}"),
            Self::MalformedCanonical(reason) => {
                write!(formatter, "malformed replay artifact: {reason}")
            }
            Self::WorkloadMismatch => formatter.write_str("replay fixture workload mismatch"),
            Self::ShapeMismatch => formatter.write_str("replay fixture shape mismatch"),
            Self::FixtureFingerprintMismatch => {
                formatter.write_str("replay fixture does not reproduce retained CEGIS identity")
            }
            Self::NonReplayableFixture => formatter
                .write_str("qualification input is opaque and not ADA-REFERENCE-INPUT-V1"),
            Self::OracleReplayMismatch {
                fixture_id,
                max_abs_error_bits,
                tolerance_bits,
            } => write!(
                formatter,
                "fixture {fixture_id} replay failed: max_abs_error_bits={max_abs_error_bits:016x};tolerance_bits={tolerance_bits:016x}"
            ),
        }
    }
}

impl std::error::Error for ReplayError {}

impl From<SemanticIrError> for ReplayError {
    fn from(value: SemanticIrError) -> Self {
        Self::ReferenceInput(value.to_string())
    }
}

/// Exact row-major f64 Q/K/V data and optional visibility mask.
///
/// Floating-point values are serialized by their exact IEEE-754 bit patterns.
/// Construction delegates all shape, finite-value, and bounded-allocation rules
/// to [`ReferenceInput::new`].
#[derive(Debug, Clone)]
pub struct ReplayReferenceInput {
    query_count: usize,
    key_count: usize,
    q_dimension: usize,
    value_dimension: usize,
    queries: Vec<f64>,
    keys: Vec<f64>,
    values: Vec<f64>,
    external_mask: Option<Vec<bool>>,
}

impl ReplayReferenceInput {
    /// Construct a replayable input from the public ADA reference-input spec.
    ///
    /// # Errors
    ///
    /// Returns an error when ADA rejects the reference input or when its exact
    /// canonical representation would exceed the qualification input bound.
    pub fn new(spec: ReferenceInputSpec) -> Result<Self, ReplayError> {
        ReferenceInput::new(spec.clone())?;
        let input = Self {
            query_count: spec.query_count,
            key_count: spec.key_count,
            q_dimension: spec.q_dimension,
            value_dimension: spec.value_dimension,
            queries: spec.queries,
            keys: spec.keys,
            values: spec.values,
            external_mask: spec.external_mask,
        };
        input.ensure_canonical_bound()?;
        Ok(input)
    }

    /// Decode strict canonical exact-input text.
    ///
    /// # Errors
    ///
    /// Rejects unsupported versions, wrong field order, malformed float bits,
    /// invalid masks, shape violations, non-finite values, trailing fields, and
    /// non-canonical representations.
    pub fn from_canonical_text(text: &str) -> Result<Self, ReplayError> {
        if text.is_empty()
            || text.len() > MAX_INPUT_CANONICAL_TEXT_BYTES
            || text.contains('\r')
            || !text.ends_with('\n')
        {
            return Err(ReplayError::MalformedCanonical(
                "reference input is empty, oversized, contains CR, or lacks final newline".into(),
            ));
        }
        let mut lines = text.lines();
        if lines.next() != Some(REFERENCE_INPUT_HEADER) {
            return Err(ReplayError::NonReplayableFixture);
        }
        let version = parse_u16(next_value(&mut lines, "version")?, "version")?;
        if version != REFERENCE_INPUT_VERSION {
            return Err(ReplayError::MalformedCanonical(
                "unsupported reference-input version".into(),
            ));
        }
        let query_count = parse_usize(next_value(&mut lines, "query_count")?, "query_count")?;
        let key_count = parse_usize(next_value(&mut lines, "key_count")?, "key_count")?;
        let q_dimension = parse_usize(next_value(&mut lines, "q_dimension")?, "q_dimension")?;
        let value_dimension =
            parse_usize(next_value(&mut lines, "value_dimension")?, "value_dimension")?;
        let query_len = checked_product(query_count, q_dimension, "queries")?;
        let key_len = checked_product(key_count, q_dimension, "keys")?;
        let value_len = checked_product(key_count, value_dimension, "values")?;
        let mask_len = checked_product(query_count, key_count, "external_mask")?;
        let queries = parse_f64_bits_list(next_value(&mut lines, "queries_bits")?, query_len)?;
        let keys = parse_f64_bits_list(next_value(&mut lines, "keys_bits")?, key_len)?;
        let values = parse_f64_bits_list(next_value(&mut lines, "values_bits")?, value_len)?;
        let external_mask = parse_mask(next_value(&mut lines, "external_mask")?, mask_len)?;
        if lines.next().is_some() {
            return Err(ReplayError::MalformedCanonical(
                "unexpected trailing reference-input field".into(),
            ));
        }
        let decoded = Self::new(ReferenceInputSpec {
            query_count,
            key_count,
            q_dimension,
            value_dimension,
            queries,
            keys,
            values,
            external_mask,
        })?;
        if decoded.to_canonical_text() != text {
            return Err(ReplayError::MalformedCanonical(
                "reference input is not canonical".into(),
            ));
        }
        Ok(decoded)
    }

    /// Encode exact deterministic text suitable for the historical
    /// qualification case's caller-owned `input` field.
    #[must_use]
    pub fn to_canonical_text(&self) -> String {
        let mut text = String::from(REFERENCE_INPUT_HEADER);
        text.push('\n');
        append_field(&mut text, "version", REFERENCE_INPUT_VERSION);
        append_field(&mut text, "query_count", self.query_count);
        append_field(&mut text, "key_count", self.key_count);
        append_field(&mut text, "q_dimension", self.q_dimension);
        append_field(&mut text, "value_dimension", self.value_dimension);
        append_field(&mut text, "queries_bits", encode_f64_bits(&self.queries));
        append_field(&mut text, "keys_bits", encode_f64_bits(&self.keys));
        append_field(&mut text, "values_bits", encode_f64_bits(&self.values));
        append_field(
            &mut text,
            "external_mask",
            self.external_mask
                .as_deref()
                .map_or_else(|| "none".into(), encode_mask),
        );
        text
    }

    /// Reconstruct the exact ADA semantic reference input.
    ///
    /// # Errors
    ///
    /// Revalidates all ADA reference-input invariants before returning.
    pub fn to_reference_input(&self) -> Result<ReferenceInput, ReplayError> {
        ReferenceInput::new(ReferenceInputSpec {
            query_count: self.query_count,
            key_count: self.key_count,
            q_dimension: self.q_dimension,
            value_dimension: self.value_dimension,
            queries: self.queries.clone(),
            keys: self.keys.clone(),
            values: self.values.clone(),
            external_mask: self.external_mask.clone(),
        })
        .map_err(ReplayError::from)
    }

    /// Query row count.
    #[must_use]
    pub const fn query_count(&self) -> usize {
        self.query_count
    }

    /// KV row count.
    #[must_use]
    pub const fn key_count(&self) -> usize {
        self.key_count
    }

    /// Q/K feature dimension.
    #[must_use]
    pub const fn q_dimension(&self) -> usize {
        self.q_dimension
    }

    /// V/output feature dimension.
    #[must_use]
    pub const fn value_dimension(&self) -> usize {
        self.value_dimension
    }

    /// Exact row-major query values.
    #[must_use]
    pub fn queries(&self) -> &[f64] {
        &self.queries
    }

    /// Exact row-major key values.
    #[must_use]
    pub fn keys(&self) -> &[f64] {
        &self.keys
    }

    /// Exact row-major value values.
    #[must_use]
    pub fn values(&self) -> &[f64] {
        &self.values
    }

    /// Optional row-major visibility mask.
    #[must_use]
    pub fn external_mask(&self) -> Option<&[bool]> {
        self.external_mask.as_deref()
    }

    fn ensure_canonical_bound(&self) -> Result<(), ReplayError> {
        let length = self.to_canonical_text().len();
        if length > MAX_INPUT_CANONICAL_TEXT_BYTES {
            return Err(ReplayError::MalformedCanonical(format!(
                "reference input canonical bytes {length} exceed {MAX_INPUT_CANONICAL_TEXT_BYTES}"
            )));
        }
        Ok(())
    }
}

/// Construction input for a CEGIS case whose identity binds exact Q/K/V bytes.
#[derive(Debug, Clone)]
pub struct ReplayCaseSpec {
    /// Workload under qualification.
    pub workload: WorkloadContract,
    /// Exact replayable reference input.
    pub input: ReplayReferenceInput,
    /// Independent oracle output.
    pub expected_output: Vec<f64>,
    /// Maximum allowed absolute output error.
    pub max_abs_tolerance: f64,
}

impl ReplayCaseSpec {
    /// Convert this replayable case into the existing ADA qualification fixture.
    ///
    /// The historical V1 qualification format is retained. Its opaque `input`
    /// field now contains `ADA-REFERENCE-INPUT-V1`, so the existing CEGIS
    /// fingerprint binds the exact tensor and mask bits without a schema fork.
    ///
    /// # Errors
    ///
    /// Propagates ADA reference-input, qualification, or fixture validation
    /// failures.
    pub fn into_fixture(
        self,
        id: impl Into<String>,
    ) -> Result<Fixture<SemanticWorkloadCase>, ReplayError> {
        let canonical_input = self.input.to_canonical_text();
        let case = SemanticWorkloadCase::new(
            self.workload,
            self.input.to_reference_input()?,
            canonical_input,
            self.expected_output,
            self.max_abs_tolerance,
        )
        .map_err(|error| ReplayError::Qualification(error.to_string()))?;
        case.into_fixture(id)
            .map_err(|error| ReplayError::Fixture(error.to_string()))
    }
}

/// One exact, reconstructed qualification fixture from a graduation bundle.
#[derive(Debug, Clone)]
pub struct ReplayFixture {
    id: String,
    input: ReplayReferenceInput,
    expected_output: Vec<f64>,
    max_abs_tolerance: f64,
}

impl ReplayFixture {
    /// Original retained CEGIS fixture identifier.
    #[must_use]
    pub fn id(&self) -> &str {
        &self.id
    }

    /// Exact Q/K/V/mask input bound by the fixture fingerprint.
    #[must_use]
    pub const fn input(&self) -> &ReplayReferenceInput {
        &self.input
    }

    /// Independent expected output retained by qualification.
    #[must_use]
    pub fn expected_output(&self) -> &[f64] {
        &self.expected_output
    }

    /// Maximum absolute output tolerance retained by qualification.
    #[must_use]
    pub const fn max_abs_tolerance(&self) -> f64 {
        self.max_abs_tolerance
    }
}

/// Summary of a successful finite-corpus ADA reference replay.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ReplayVerificationReport {
    fixture_count: usize,
    worst_max_abs_error: f64,
}

impl ReplayVerificationReport {
    /// Number of retained fixtures replayed successfully.
    #[must_use]
    pub const fn fixture_count(self) -> usize {
        self.fixture_count
    }

    /// Worst maximum absolute output error across retained fixtures.
    #[must_use]
    pub const fn worst_max_abs_error(self) -> f64 {
        self.worst_max_abs_error
    }
}

/// Decode every retained qualification fixture in a graduation bundle and
/// prove that its exact replay representation reconstructs the same CEGIS
/// fixture fingerprint.
///
/// # Errors
///
/// Rejects legacy opaque inputs, workload/shape mismatches, malformed exact
/// tensors, altered expected outputs/tolerance, or fingerprint mismatches.
pub fn decode_graduation_fixtures(
    bundle: &FlatGraduationBundle,
) -> Result<Vec<ReplayFixture>, ReplayError> {
    bundle
        .oracle_fixtures()
        .iter()
        .map(|artifact| decode_fixture(bundle.workload(), artifact))
        .collect()
}

/// Replay the graduated semantic against every exact retained fixture using
/// ADA's independent f64 semantic evaluator.
///
/// This is a finite-corpus integration check only. It does not promote
/// `CorrectnessStatus`, alter the graduation verdict, or make a FLAT/backend
/// claim.
///
/// # Errors
///
/// Returns an error when a fixture cannot be reconstructed or when the semantic
/// exceeds the retained maximum absolute tolerance.
pub fn verify_ada_reference_replay(
    bundle: &FlatGraduationBundle,
) -> Result<ReplayVerificationReport, ReplayError> {
    let fixtures = decode_graduation_fixtures(bundle)?;
    let mut worst_max_abs_error = 0.0_f64;
    for fixture in &fixtures {
        let input = fixture.input.to_reference_input()?;
        let output = bundle
            .semantic()
            .evaluate(&input)
            .map_err(|error| ReplayError::ReferenceInput(error.to_string()))?;
        let max_abs_error = output
            .output()
            .iter()
            .zip(&fixture.expected_output)
            .map(|(actual, expected)| (actual - expected).abs())
            .fold(0.0_f64, f64::max);
        if max_abs_error > fixture.max_abs_tolerance {
            return Err(ReplayError::OracleReplayMismatch {
                fixture_id: fixture.id.clone(),
                max_abs_error_bits: max_abs_error.to_bits(),
                tolerance_bits: fixture.max_abs_tolerance.to_bits(),
            });
        }
        worst_max_abs_error = worst_max_abs_error.max(max_abs_error);
    }
    Ok(ReplayVerificationReport {
        fixture_count: fixtures.len(),
        worst_max_abs_error,
    })
}

fn decode_fixture(
    workload: &WorkloadContract,
    artifact: &OracleFixtureArtifact,
) -> Result<ReplayFixture, ReplayError> {
    let text = artifact.canonical_text();
    if text.contains('\r') || !text.ends_with('\n') {
        return Err(ReplayError::MalformedCanonical(
            "qualification case contains CR or lacks final newline".into(),
        ));
    }
    let mut lines = text.lines();
    let header = format!("ADA-QUALIFICATION-CASE-V{QUALIFICATION_CASE_VERSION}");
    if lines.next() != Some(header.as_str()) {
        return Err(ReplayError::MalformedCanonical(
            "unsupported qualification-case header".into(),
        ));
    }
    let expected_workload = workload_fingerprint_text(workload);
    if next_value(&mut lines, "workload")? != expected_workload {
        return Err(ReplayError::WorkloadMismatch);
    }
    let input_text = hex_decode(next_value(&mut lines, "input")?)?;
    let input = ReplayReferenceInput::from_canonical_text(&input_text)?;
    let shape = parse_shape(next_value(&mut lines, "shape")?)?;
    if shape
        != (
            input.query_count,
            input.key_count,
            input.q_dimension,
            input.value_dimension,
        )
    {
        return Err(ReplayError::ShapeMismatch);
    }
    let expected_len = checked_product(input.query_count, input.value_dimension, "expected")?;
    let expected_output =
        parse_f64_bits_list(next_value(&mut lines, "expected_bits")?, expected_len)?;
    let tolerance = parse_hex_f64(next_value(&mut lines, "max_abs_tolerance_bits")?)?;
    if !tolerance.is_finite() || tolerance < 0.0 {
        return Err(ReplayError::MalformedCanonical(
            "invalid replay tolerance".into(),
        ));
    }
    if lines.next().is_some() {
        return Err(ReplayError::MalformedCanonical(
            "unexpected trailing qualification-case field".into(),
        ));
    }

    let rebuilt_case = SemanticWorkloadCase::new(
        workload.clone(),
        input.to_reference_input()?,
        input.to_canonical_text(),
        expected_output.clone(),
        tolerance,
    )
    .map_err(|error| ReplayError::Qualification(error.to_string()))?;
    let rebuilt = rebuilt_case
        .into_fixture(artifact.id().to_owned())
        .map_err(|error| ReplayError::Fixture(error.to_string()))?;
    let rebuilt_fingerprint = rebuilt.fingerprint();
    let retained_fingerprint = artifact.fingerprint();
    if rebuilt.canonical_text() != artifact.canonical_text()
        || rebuilt_fingerprint.primary() != retained_fingerprint.primary()
        || rebuilt_fingerprint.secondary() != retained_fingerprint.secondary()
        || rebuilt_fingerprint.length() != retained_fingerprint.length()
    {
        return Err(ReplayError::FixtureFingerprintMismatch);
    }
    Ok(ReplayFixture {
        id: artifact.id().to_owned(),
        input,
        expected_output,
        max_abs_tolerance: tolerance,
    })
}

fn workload_fingerprint_text(workload: &WorkloadContract) -> String {
    let fingerprint = workload.fingerprint();
    format!(
        "{:016x}-{:016x}-{:016x}",
        fingerprint.primary(),
        fingerprint.secondary(),
        fingerprint.length()
    )
}

fn encode_f64_bits(values: &[f64]) -> String {
    values
        .iter()
        .map(|value| format!("{:016x}", value.to_bits()))
        .collect::<Vec<_>>()
        .join(",")
}

fn encode_mask(mask: &[bool]) -> String {
    mask.iter()
        .map(|&visible| if visible { '1' } else { '0' })
        .collect()
}

fn parse_f64_bits_list(value: &str, expected: usize) -> Result<Vec<f64>, ReplayError> {
    if expected == 0 {
        return Err(ReplayError::MalformedCanonical(
            "zero-length float vector is invalid".into(),
        ));
    }
    let parts = value.split(',').collect::<Vec<_>>();
    if parts.len() != expected {
        return Err(ReplayError::MalformedCanonical(format!(
            "float vector has {} elements; expected {expected}",
            parts.len()
        )));
    }
    parts.into_iter().map(parse_hex_f64).collect()
}

fn parse_hex_f64(value: &str) -> Result<f64, ReplayError> {
    if value.len() != 16 || !is_lower_hex(value) {
        return Err(ReplayError::MalformedCanonical(
            "f64 bits must be exactly 16 lowercase hex digits".into(),
        ));
    }
    let bits = u64::from_str_radix(value, 16)
        .map_err(|_| ReplayError::MalformedCanonical("invalid f64 bits".into()))?;
    Ok(f64::from_bits(bits))
}

fn parse_mask(value: &str, expected: usize) -> Result<Option<Vec<bool>>, ReplayError> {
    if value == "none" {
        return Ok(None);
    }
    if value.len() != expected || !value.bytes().all(|byte| matches!(byte, b'0' | b'1')) {
        return Err(ReplayError::MalformedCanonical(
            "external mask must be `none` or one 0/1 byte per score".into(),
        ));
    }
    Ok(Some(value.bytes().map(|byte| byte == b'1').collect()))
}

fn parse_shape(value: &str) -> Result<(usize, usize, usize, usize), ReplayError> {
    let mut parts = value.split(':');
    let q = parse_usize(
        parts
            .next()
            .ok_or_else(|| ReplayError::MalformedCanonical("missing query shape".into()))?,
        "shape.query",
    )?;
    let k = parse_usize(
        parts
            .next()
            .ok_or_else(|| ReplayError::MalformedCanonical("missing key shape".into()))?,
        "shape.key",
    )?;
    let qd = parse_usize(
        parts
            .next()
            .ok_or_else(|| ReplayError::MalformedCanonical("missing Q/K dimension".into()))?,
        "shape.q_dimension",
    )?;
    let vd = parse_usize(
        parts
            .next()
            .ok_or_else(|| ReplayError::MalformedCanonical("missing V dimension".into()))?,
        "shape.value_dimension",
    )?;
    if parts.next().is_some() {
        return Err(ReplayError::MalformedCanonical(
            "qualification shape has extra components".into(),
        ));
    }
    Ok((q, k, qd, vd))
}

fn checked_product(left: usize, right: usize, field: &str) -> Result<usize, ReplayError> {
    left.checked_mul(right).ok_or_else(|| {
        ReplayError::MalformedCanonical(format!("overflow while computing {field} length"))
    })
}

fn append_field(text: &mut String, key: &str, value: impl Display) {
    let _ = writeln!(text, "{key}={value}");
}

fn next_value<'a>(
    lines: &mut std::str::Lines<'a>,
    expected_key: &str,
) -> Result<&'a str, ReplayError> {
    let line = lines.next().ok_or_else(|| {
        ReplayError::MalformedCanonical(format!("missing field {expected_key}"))
    })?;
    let (key, value) = line.split_once('=').ok_or_else(|| {
        ReplayError::MalformedCanonical(format!("field {expected_key} lacks '='"))
    })?;
    if key != expected_key || value.is_empty() || value.contains('=') {
        return Err(ReplayError::MalformedCanonical(format!(
            "expected canonical field {expected_key}"
        )));
    }
    Ok(value)
}

fn parse_u16(value: &str, field: &str) -> Result<u16, ReplayError> {
    value.parse::<u16>().map_err(|_| {
        ReplayError::MalformedCanonical(format!("invalid integer field {field}"))
    })
}

fn parse_usize(value: &str, field: &str) -> Result<usize, ReplayError> {
    value.parse::<usize>().map_err(|_| {
        ReplayError::MalformedCanonical(format!("invalid integer field {field}"))
    })
}

fn hex_decode(value: &str) -> Result<String, ReplayError> {
    if value.len() % 2 != 0 || !is_lower_hex(value) {
        return Err(ReplayError::MalformedCanonical(
            "invalid lowercase hex text".into(),
        ));
    }
    let mut bytes = Vec::with_capacity(value.len() / 2);
    for pair in value.as_bytes().chunks_exact(2) {
        bytes.push((hex_nibble(pair[0])? << 4) | hex_nibble(pair[1])?);
    }
    String::from_utf8(bytes)
        .map_err(|_| ReplayError::MalformedCanonical("hex text is not UTF-8".into()))
}

fn hex_nibble(byte: u8) -> Result<u8, ReplayError> {
    match byte {
        b'0'..=b'9' => Ok(byte - b'0'),
        b'a'..=b'f' => Ok(byte - b'a' + 10),
        _ => Err(ReplayError::MalformedCanonical(
            "invalid lowercase hex digit".into(),
        )),
    }
}

fn is_lower_hex(value: &str) -> bool {
    !value.is_empty()
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

#[cfg(test)]
mod tests;
