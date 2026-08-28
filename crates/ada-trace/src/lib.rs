//! Canonical execution/research trace interchange for ADA.
//!
//! Trace V2 binds every event stream to an exact semantic, workload, and
//! implementation fingerprint. A trace records observations; it is not by
//! itself correctness, novelty, or hardware-performance evidence.

#![forbid(unsafe_code)]

use ada_implementation::ImplementationPlan;
use ada_semantic::SemanticProgram;
use ada_workload::WorkloadContract;
use std::fmt::{Display, Formatter};

/// Trace schema version.
pub const TRACE_VERSION: u16 = 2;
/// Canonical trace header.
pub const TRACE_HEADER: &str = "ADA-TRACE-V2";
/// Maximum number of events in one trace.
pub const MAX_TRACE_EVENTS: usize = 1 << 20;
/// Maximum canonical trace byte length.
pub const MAX_TRACE_BYTES: usize = 64 << 20;
/// Maximum event label byte length.
pub const MAX_TRACE_LABEL_BYTES: usize = 256;

/// Trace construction/decoding errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TraceError {
    /// Invalid label or field.
    InvalidField(&'static str),
    /// Non-finite scalar observation.
    NonFiniteScalar,
    /// Event capacity exceeded.
    TooManyEvents,
    /// Event sequence was not contiguous and zero-based.
    InvalidSequence,
    /// Unsupported trace version.
    UnsupportedVersion(u16),
    /// Canonical text was malformed.
    MalformedCanonical(String),
}

impl Display for TraceError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidField(field) => write!(formatter, "invalid trace field: {field}"),
            Self::NonFiniteScalar => formatter.write_str("trace scalar must be finite"),
            Self::TooManyEvents => formatter.write_str("trace event capacity exceeded"),
            Self::InvalidSequence => formatter.write_str("trace event sequence is not contiguous"),
            Self::UnsupportedVersion(version) => write!(formatter, "unsupported trace version {version}"),
            Self::MalformedCanonical(reason) => write!(formatter, "malformed trace: {reason}"),
        }
    }
}

impl std::error::Error for TraceError {}

/// Stable three-lane binding copied from one canonical ADA artifact.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TraceFingerprint {
    primary: u64,
    secondary: u64,
    length: u64,
}

impl TraceFingerprint {
    /// Construct a fingerprint triplet from its exact lanes.
    #[must_use]
    pub const fn new(primary: u64, secondary: u64, length: u64) -> Self {
        Self {
            primary,
            secondary,
            length,
        }
    }

    /// First lane.
    #[must_use]
    pub const fn primary(self) -> u64 {
        self.primary
    }

    /// Second lane.
    #[must_use]
    pub const fn secondary(self) -> u64 {
        self.secondary
    }

    /// Canonical byte length lane.
    #[must_use]
    pub const fn length(self) -> u64 {
        self.length
    }
}

/// Exact ADA identity bindings for a trace stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TraceBinding {
    /// Semantic-program fingerprint.
    pub semantic: TraceFingerprint,
    /// Workload-contract fingerprint.
    pub workload: TraceFingerprint,
    /// Implementation-plan fingerprint.
    pub implementation: TraceFingerprint,
}

impl TraceBinding {
    /// Bind directly to the three canonical ADA artifacts.
    #[must_use]
    pub fn from_artifacts(
        semantic: &SemanticProgram,
        workload: &WorkloadContract,
        implementation: &ImplementationPlan,
    ) -> Self {
        let semantic = semantic.fingerprint();
        let workload = workload.fingerprint();
        let implementation = implementation.fingerprint();
        Self {
            semantic: TraceFingerprint::new(
                semantic.primary(),
                semantic.secondary(),
                semantic.length(),
            ),
            workload: TraceFingerprint::new(
                workload.primary(),
                workload.secondary(),
                workload.length(),
            ),
            implementation: TraceFingerprint::new(
                implementation.primary(),
                implementation.secondary(),
                implementation.length(),
            ),
        }
    }
}

/// Typed trace event payload.
#[derive(Debug, Clone, PartialEq)]
pub enum TraceEventKind {
    /// Named stage boundary.
    StageStart,
    /// Named stage completion.
    StageEnd,
    /// Exact unsigned logical counter.
    Counter(u64),
    /// Finite scalar observation stored by exact f64 bit pattern.
    Scalar(f64),
    /// A bounded logical selection count.
    SelectionCount(u64),
}

impl TraceEventKind {
    fn validate(&self) -> Result<(), TraceError> {
        if let Self::Scalar(value) = self {
            if !value.is_finite() {
                return Err(TraceError::NonFiniteScalar);
            }
        }
        Ok(())
    }
}

/// One deterministically sequenced trace event.
#[derive(Debug, Clone, PartialEq)]
pub struct TraceEvent {
    /// Zero-based event sequence.
    pub sequence: u64,
    /// Stable event/stage label.
    pub label: String,
    /// Typed event payload.
    pub kind: TraceEventKind,
}

impl TraceEvent {
    fn validate(&self) -> Result<(), TraceError> {
        validate_label(&self.label)?;
        self.kind.validate()
    }
}

/// Complete versioned trace record.
#[derive(Debug, Clone, PartialEq)]
pub struct TraceRecord {
    binding: TraceBinding,
    events: Vec<TraceEvent>,
}

impl TraceRecord {
    /// Construct a bound, validated trace.
    ///
    /// # Errors
    ///
    /// Rejects invalid labels/scalars, excessive event count, and non-contiguous
    /// event sequence numbers.
    pub fn new(binding: TraceBinding, events: Vec<TraceEvent>) -> Result<Self, TraceError> {
        if events.len() > MAX_TRACE_EVENTS {
            return Err(TraceError::TooManyEvents);
        }
        for (index, event) in events.iter().enumerate() {
            event.validate()?;
            let expected = u64::try_from(index).map_err(|_| TraceError::TooManyEvents)?;
            if event.sequence != expected {
                return Err(TraceError::InvalidSequence);
            }
        }
        Ok(Self { binding, events })
    }

    /// Artifact bindings.
    #[must_use]
    pub const fn binding(&self) -> TraceBinding {
        self.binding
    }

    /// Events in canonical sequence order.
    #[must_use]
    pub fn events(&self) -> &[TraceEvent] {
        &self.events
    }

    /// Deterministic canonical text.
    #[must_use]
    pub fn to_canonical_text(&self) -> String {
        let mut text = String::from(TRACE_HEADER);
        text.push('\n');
        append(&mut text, "version", &TRACE_VERSION.to_string());
        append_fingerprint(&mut text, "semantic", self.binding.semantic);
        append_fingerprint(&mut text, "workload", self.binding.workload);
        append_fingerprint(&mut text, "implementation", self.binding.implementation);
        append(&mut text, "event_count", &self.events.len().to_string());
        for event in &self.events {
            let prefix = format!("event_{}", event.sequence);
            append(&mut text, &format!("{prefix}_label"), &hex_encode(&event.label));
            match event.kind {
                TraceEventKind::StageStart => append(&mut text, &format!("{prefix}_kind"), "stage-start"),
                TraceEventKind::StageEnd => append(&mut text, &format!("{prefix}_kind"), "stage-end"),
                TraceEventKind::Counter(value) => {
                    append(&mut text, &format!("{prefix}_kind"), "counter");
                    append(&mut text, &format!("{prefix}_value"), &value.to_string());
                }
                TraceEventKind::Scalar(value) => {
                    append(&mut text, &format!("{prefix}_kind"), "scalar-f64-bits");
                    append(
                        &mut text,
                        &format!("{prefix}_value"),
                        &format!("{:016x}", value.to_bits()),
                    );
                }
                TraceEventKind::SelectionCount(value) => {
                    append(&mut text, &format!("{prefix}_kind"), "selection-count");
                    append(&mut text, &format!("{prefix}_value"), &value.to_string());
                }
            }
        }
        text
    }

    /// Decode strict canonical Trace V2 text.
    ///
    /// # Errors
    ///
    /// Rejects oversized, malformed, non-canonical, unknown, non-finite, or
    /// non-contiguously sequenced traces.
    pub fn from_canonical_text(text: &str) -> Result<Self, TraceError> {
        if text.len() > MAX_TRACE_BYTES || text.contains('\r') || !text.ends_with('\n') {
            return malformed("trace exceeds limit, contains CR, or lacks final newline");
        }
        let mut lines = text.lines();
        if lines.next() != Some(TRACE_HEADER) {
            return malformed("invalid trace header");
        }
        let version = parse_u16(next_value(&mut lines, "version")?)?;
        if version != TRACE_VERSION {
            return Err(TraceError::UnsupportedVersion(version));
        }
        let binding = TraceBinding {
            semantic: parse_fingerprint(&mut lines, "semantic")?,
            workload: parse_fingerprint(&mut lines, "workload")?,
            implementation: parse_fingerprint(&mut lines, "implementation")?,
        };
        let count = parse_usize(next_value(&mut lines, "event_count")?)?;
        if count > MAX_TRACE_EVENTS {
            return Err(TraceError::TooManyEvents);
        }
        let mut events = Vec::with_capacity(count);
        for index in 0..count {
            let sequence = u64::try_from(index).map_err(|_| TraceError::TooManyEvents)?;
            let prefix = format!("event_{sequence}");
            let label = hex_decode(next_value(&mut lines, &format!("{prefix}_label"))?)?;
            let kind_text = next_value(&mut lines, &format!("{prefix}_kind"))?;
            let kind = match kind_text {
                "stage-start" => TraceEventKind::StageStart,
                "stage-end" => TraceEventKind::StageEnd,
                "counter" => TraceEventKind::Counter(parse_u64(next_value(
                    &mut lines,
                    &format!("{prefix}_value"),
                )?)?),
                "selection-count" => TraceEventKind::SelectionCount(parse_u64(next_value(
                    &mut lines,
                    &format!("{prefix}_value"),
                )?)?),
                "scalar-f64-bits" => {
                    let raw = next_value(&mut lines, &format!("{prefix}_value"))?;
                    let bits = parse_hex_u64(raw)?;
                    let value = f64::from_bits(bits);
                    if !value.is_finite() {
                        return Err(TraceError::NonFiniteScalar);
                    }
                    TraceEventKind::Scalar(value)
                }
                _ => return malformed("unknown event kind"),
            };
            events.push(TraceEvent {
                sequence,
                label,
                kind,
            });
        }
        if lines.next().is_some() {
            return malformed("unexpected trailing trace field");
        }
        let record = Self::new(binding, events)?;
        if record.to_canonical_text() != text {
            return malformed("trace is not canonical");
        }
        Ok(record)
    }
}

fn validate_label(label: &str) -> Result<(), TraceError> {
    if label.is_empty()
        || label.len() > MAX_TRACE_LABEL_BYTES
        || label.chars().any(char::is_control)
        || label.chars().any(char::is_whitespace)
    {
        return Err(TraceError::InvalidField("event.label"));
    }
    Ok(())
}

fn append_fingerprint(text: &mut String, prefix: &str, fingerprint: TraceFingerprint) {
    append(text, &format!("{prefix}_primary"), &format!("{:016x}", fingerprint.primary()));
    append(text, &format!("{prefix}_secondary"), &format!("{:016x}", fingerprint.secondary()));
    append(text, &format!("{prefix}_length"), &format!("{:016x}", fingerprint.length()));
}

fn parse_fingerprint(
    lines: &mut std::str::Lines<'_>,
    prefix: &str,
) -> Result<TraceFingerprint, TraceError> {
    Ok(TraceFingerprint::new(
        parse_hex_u64(next_value(lines, &format!("{prefix}_primary"))?)?,
        parse_hex_u64(next_value(lines, &format!("{prefix}_secondary"))?)?,
        parse_hex_u64(next_value(lines, &format!("{prefix}_length"))?)?,
    ))
}

fn append(text: &mut String, key: &str, value: &str) {
    text.push_str(key);
    text.push('=');
    text.push_str(value);
    text.push('\n');
}

fn next_value<'a>(
    lines: &mut std::str::Lines<'a>,
    expected: &str,
) -> Result<&'a str, TraceError> {
    let line = lines.next().ok_or_else(|| malformed_error("missing field"))?;
    let (key, value) = line
        .split_once('=')
        .ok_or_else(|| malformed_error("field lacks '='"))?;
    if key != expected || value.is_empty() || value.contains('=') {
        return malformed("unexpected or ambiguous field");
    }
    Ok(value)
}

fn parse_u16(value: &str) -> Result<u16, TraceError> {
    value.parse().map_err(|_| malformed_error("invalid u16"))
}

fn parse_u64(value: &str) -> Result<u64, TraceError> {
    value.parse().map_err(|_| malformed_error("invalid u64"))
}

fn parse_usize(value: &str) -> Result<usize, TraceError> {
    value.parse().map_err(|_| malformed_error("invalid usize"))
}

fn parse_hex_u64(value: &str) -> Result<u64, TraceError> {
    if value.len() != 16 || !value.bytes().all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase()) {
        return malformed("invalid canonical u64 hex");
    }
    u64::from_str_radix(value, 16).map_err(|_| malformed_error("invalid u64 hex"))
}

fn hex_encode(value: &str) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut result = String::with_capacity(value.len() * 2);
    for byte in value.bytes() {
        result.push(char::from(HEX[usize::from(byte >> 4)]));
        result.push(char::from(HEX[usize::from(byte & 0x0f)]));
    }
    result
}

fn hex_decode(value: &str) -> Result<String, TraceError> {
    if value.len() % 2 != 0 {
        return malformed("odd-length label hex");
    }
    let mut bytes = Vec::with_capacity(value.len() / 2);
    let raw = value.as_bytes();
    for pair in raw.chunks_exact(2) {
        let high = hex_digit(pair[0]).ok_or_else(|| malformed_error("invalid label hex"))?;
        let low = hex_digit(pair[1]).ok_or_else(|| malformed_error("invalid label hex"))?;
        bytes.push((high << 4) | low);
    }
    String::from_utf8(bytes).map_err(|_| malformed_error("label is not utf-8"))
}

fn hex_digit(value: u8) -> Option<u8> {
    match value {
        b'0'..=b'9' => Some(value - b'0'),
        b'a'..=b'f' => Some(value - b'a' + 10),
        _ => None,
    }
}

fn malformed_error(reason: &str) -> TraceError {
    TraceError::MalformedCanonical(reason.into())
}

fn malformed<T>(reason: &str) -> Result<T, TraceError> {
    Err(malformed_error(reason))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ada_core::{ImplementationCandidateId, SemanticFamily, SemanticId};
    use ada_implementation::{
        AlgorithmPlan, Buffering, ExpStrategy, MemoryLevel, MemoryPlan, ReductionTopology,
        SchedulePlan, TileShape, WorkPartition,
    };
    use ada_semantic::{MaskRule, SelectionRule};
    use ada_workload::{
        AttentionGeometry, AttentionTopology, GeometrySpec, HeadGrouping, SequenceLengths,
        WorkloadOptions,
    };

    fn artifacts() -> (SemanticProgram, WorkloadContract, ImplementationPlan) {
        let semantic_id = SemanticId::new(SemanticFamily::StandardSoftmax, "trace-test", 1).unwrap();
        let semantic = SemanticProgram::standard_softmax(
            semantic_id.clone(),
            MaskRule::Unmasked,
            SelectionRule::All,
            0.5,
        )
        .unwrap();
        let workload = WorkloadContract::new(
            AttentionGeometry::new(GeometrySpec {
                sequence_lengths: SequenceLengths::uniform(1, 2, 3).unwrap(),
                query_heads: 1,
                kv_heads: 1,
                qk_dimension: Some(4),
                value_dimension: 4,
                topology: AttentionTopology::SelfAttention,
                head_grouping: HeadGrouping::MultiHead,
            })
            .unwrap(),
            WorkloadOptions::default(),
        )
        .unwrap();
        let implementation = ImplementationPlan::new(
            ImplementationCandidateId::new(semantic_id, "trace-impl", 1).unwrap(),
            AlgorithmPlan::DenseBlocked,
            SchedulePlan {
                tile: TileShape {
                    queries: 2,
                    keys: 4,
                    values: 4,
                },
                partition: WorkPartition::QueryTiles,
                reduction: ReductionTopology::Tree,
                exp_strategy: ExpStrategy::Standard,
                pipeline_stages: 1,
                vector_width: 1,
                buffering: Buffering::Single,
            },
            MemoryPlan {
                query: MemoryLevel::Global,
                key: MemoryLevel::Global,
                value: MemoryLevel::Global,
                output: MemoryLevel::Global,
                accumulator: MemoryLevel::Register,
                workspace_bytes: 0,
                alignment_bytes: 1,
                kv_page_rows: None,
            },
        )
        .unwrap();
        (semantic, workload, implementation)
    }

    #[test]
    fn canonical_trace_round_trips_and_binds_all_three_artifacts() {
        let (semantic, workload, implementation) = artifacts();
        let binding = TraceBinding::from_artifacts(&semantic, &workload, &implementation);
        let trace = TraceRecord::new(
            binding,
            vec![
                TraceEvent {
                    sequence: 0,
                    label: "score".into(),
                    kind: TraceEventKind::StageStart,
                },
                TraceEvent {
                    sequence: 1,
                    label: "qk-evals".into(),
                    kind: TraceEventKind::Counter(6),
                },
                TraceEvent {
                    sequence: 2,
                    label: "max-error".into(),
                    kind: TraceEventKind::Scalar(-0.0),
                },
                TraceEvent {
                    sequence: 3,
                    label: "score".into(),
                    kind: TraceEventKind::StageEnd,
                },
            ],
        )
        .unwrap();
        let text = trace.to_canonical_text();
        assert_eq!(TraceRecord::from_canonical_text(&text).unwrap(), trace);
        assert_ne!(binding.semantic, binding.workload);
        assert_ne!(binding.workload, binding.implementation);
    }

    #[test]
    fn non_contiguous_sequence_and_nonfinite_scalars_fail_closed() {
        let binding = TraceBinding {
            semantic: TraceFingerprint::new(1, 2, 3),
            workload: TraceFingerprint::new(4, 5, 6),
            implementation: TraceFingerprint::new(7, 8, 9),
        };
        assert_eq!(
            TraceRecord::new(
                binding,
                vec![TraceEvent {
                    sequence: 1,
                    label: "bad-sequence".into(),
                    kind: TraceEventKind::Counter(1),
                }],
            ),
            Err(TraceError::InvalidSequence)
        );
        assert_eq!(
            TraceRecord::new(
                binding,
                vec![TraceEvent {
                    sequence: 0,
                    label: "nan".into(),
                    kind: TraceEventKind::Scalar(f64::NAN),
                }],
            ),
            Err(TraceError::NonFiniteScalar)
        );
    }

    #[test]
    fn canonical_decoder_rejects_trailing_and_uppercase_hex() {
        let binding = TraceBinding {
            semantic: TraceFingerprint::new(1, 2, 3),
            workload: TraceFingerprint::new(4, 5, 6),
            implementation: TraceFingerprint::new(7, 8, 9),
        };
        let trace = TraceRecord::new(binding, Vec::new()).unwrap();
        let mut trailing = trace.to_canonical_text();
        trailing.push_str("extra=1\n");
        assert!(TraceRecord::from_canonical_text(&trailing).is_err());

        let uppercase = trace
            .to_canonical_text()
            .replacen("semantic_primary=0000000000000001", "semantic_primary=000000000000000A", 1);
        assert!(TraceRecord::from_canonical_text(&uppercase).is_err());
    }
}
