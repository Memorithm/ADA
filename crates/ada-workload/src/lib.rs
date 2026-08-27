//! Versioned, validated research workload contracts for ADA.
//!
//! This crate describes attention workload geometry and experimental mode
//! without changing the historical A1-A10 structures. It is a contract layer:
//! it does not execute tensors, infer a semantic, or claim that a declared
//! precision is available on any physical device.
//!
//! Historical A1 fixtures can be adapted explicitly with
//! `WorkloadContract::from_a1_case`. The adapter records that the fixture
//! contains precomputed scalar logits rather than explicit Q/K vectors.

#![forbid(unsafe_code)]

use std::collections::BTreeMap;
use std::fmt::{Display, Formatter};

/// Version of the canonical workload contract implemented by this crate.
pub const WORKLOAD_CONTRACT_VERSION: u16 = 1;
/// Maximum number of examples in one workload.
pub const MAX_BATCH_COUNT: usize = 1 << 20;
/// Maximum number of query or KV heads.
pub const MAX_HEAD_COUNT: usize = 1 << 16;
/// Maximum query or KV sequence length.
pub const MAX_SEQUENCE_LENGTH: usize = 1 << 24;
/// Maximum Q/K or V dimension in one head.
pub const MAX_HEAD_DIMENSION: usize = 1 << 16;
/// Maximum row/column count of recurrent-state metadata.
pub const MAX_STATE_DIMENSION: usize = 1 << 16;
/// Maximum byte length of an external identifier.
pub const MAX_IDENTIFIER_BYTES: usize = 256;
/// Maximum canonical text artifact accepted by the decoder.
pub const MAX_CANONICAL_TEXT_BYTES: usize = 64 << 20;

/// Contract construction and decoding failures.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WorkloadContractError {
    /// Unsupported contract version.
    UnsupportedVersion(u16),
    /// A required count or dimension was zero.
    ZeroField(&'static str),
    /// A count or dimension exceeded a structural limit.
    ExceedsLimit {
        /// Field whose value was rejected.
        field: &'static str,
        /// Rejected value.
        value: usize,
        /// Inclusive maximum accepted value.
        maximum: usize,
    },
    /// Parallel query/KV length vectors had different sizes.
    LengthMismatch {
        /// Number of query lengths.
        query_lengths: usize,
        /// Number of KV lengths.
        kv_lengths: usize,
    },
    /// A cross-field invariant was violated.
    InvalidField(&'static str),
    /// An external identifier was empty, malformed, or too large.
    InvalidIdentifier(&'static str),
    /// A historical A1 case could not be adapted.
    InvalidHistoricalCase(&'static str),
    /// Canonical text was malformed or incomplete.
    MalformedCanonicalText(String),
}

impl Display for WorkloadContractError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnsupportedVersion(version) => {
                write!(formatter, "unsupported workload contract version {version}")
            }
            Self::ZeroField(field) => write!(formatter, "{field} must be non-zero"),
            Self::ExceedsLimit {
                field,
                value,
                maximum,
            } => write!(formatter, "{field}={value} exceeds maximum {maximum}"),
            Self::LengthMismatch {
                query_lengths,
                kv_lengths,
            } => write!(
                formatter,
                "query/KV length vectors differ ({query_lengths} != {kv_lengths})"
            ),
            Self::InvalidField(field) => write!(formatter, "invalid workload field: {field}"),
            Self::InvalidIdentifier(field) => {
                write!(formatter, "invalid or empty identifier in {field}")
            }
            Self::InvalidHistoricalCase(reason) => {
                write!(formatter, "invalid historical A1 case: {reason}")
            }
            Self::MalformedCanonicalText(reason) => {
                write!(formatter, "malformed canonical workload text: {reason}")
            }
        }
    }
}

impl std::error::Error for WorkloadContractError {}

fn validate_count(
    field: &'static str,
    value: usize,
    maximum: usize,
) -> Result<(), WorkloadContractError> {
    if value == 0 {
        return Err(WorkloadContractError::ZeroField(field));
    }
    if value > maximum {
        return Err(WorkloadContractError::ExceedsLimit {
            field,
            value,
            maximum,
        });
    }
    Ok(())
}

fn validate_identifier(field: &'static str, identifier: &str) -> Result<(), WorkloadContractError> {
    if identifier.is_empty()
        || identifier.len() > MAX_IDENTIFIER_BYTES
        || identifier.chars().any(char::is_control)
        || identifier.chars().any(char::is_whitespace)
    {
        return Err(WorkloadContractError::InvalidIdentifier(field));
    }
    Ok(())
}

fn parse_usize(field: &str, value: &str) -> Result<usize, WorkloadContractError> {
    value.parse::<usize>().map_err(|_| {
        WorkloadContractError::MalformedCanonicalText(format!(
            "{field} is not an unsigned decimal integer"
        ))
    })
}

fn parse_u16(field: &str, value: &str) -> Result<u16, WorkloadContractError> {
    value.parse::<u16>().map_err(|_| {
        WorkloadContractError::MalformedCanonicalText(format!(
            "{field} is not an unsigned 16-bit integer"
        ))
    })
}

fn parse_usize_list(field: &str, value: &str) -> Result<Vec<usize>, WorkloadContractError> {
    if value.is_empty() {
        return Err(WorkloadContractError::MalformedCanonicalText(format!(
            "{field} cannot be empty"
        )));
    }
    value
        .split(',')
        .map(|part| parse_usize(field, part))
        .collect()
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

fn hex_decode(field: &'static str, value: &str) -> Result<String, WorkloadContractError> {
    if value.len() % 2 != 0 {
        return Err(WorkloadContractError::MalformedCanonicalText(format!(
            "{field} has an odd-length hex value"
        )));
    }
    let mut bytes = Vec::with_capacity(value.len() / 2);
    let mut chars = value.bytes();
    while let (Some(high), Some(low)) = (chars.next(), chars.next()) {
        let high = hex_digit(high).ok_or_else(|| {
            WorkloadContractError::MalformedCanonicalText(format!(
                "{field} contains a non-hex digit"
            ))
        })?;
        let low = hex_digit(low).ok_or_else(|| {
            WorkloadContractError::MalformedCanonicalText(format!(
                "{field} contains a non-hex digit"
            ))
        })?;
        bytes.push((high << 4) | low);
    }
    String::from_utf8(bytes).map_err(|_| {
        WorkloadContractError::MalformedCanonicalText(format!(
            "{field} is not UTF-8 after hex decoding"
        ))
    })
}

fn hex_digit(value: u8) -> Option<u8> {
    match value {
        b'0'..=b'9' => Some(value - b'0'),
        b'a'..=b'f' => Some(value - b'a' + 10),
        _ => None,
    }
}

/// Attention topology declared by the workload.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AttentionTopology {
    /// Query and KV sequences belong to the same attention stream.
    SelfAttention,
    /// Query and KV sequences come from distinct streams.
    CrossAttention,
    /// Historical A1 has precomputed scalar logits and no explicit Q/K input.
    HistoricalA1,
}

impl AttentionTopology {
    fn as_text(self) -> &'static str {
        match self {
            Self::SelfAttention => "self",
            Self::CrossAttention => "cross",
            Self::HistoricalA1 => "historical-a1",
        }
    }

    fn from_text(value: &str) -> Result<Self, WorkloadContractError> {
        match value {
            "self" => Ok(Self::SelfAttention),
            "cross" => Ok(Self::CrossAttention),
            "historical-a1" => Ok(Self::HistoricalA1),
            _ => Err(WorkloadContractError::MalformedCanonicalText(
                "unknown topology".into(),
            )),
        }
    }
}

/// Explicit query-head to KV-head sharing policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum HeadGrouping {
    /// One KV head per query head.
    MultiHead,
    /// All query heads share one KV head.
    MultiQuery,
    /// A fixed number of query heads share each KV head.
    GroupedQuery {
        /// Number of query heads mapped to one KV head.
        queries_per_kv: usize,
    },
}

impl HeadGrouping {
    /// Derive MHA/MQA/GQA classification from head counts.
    ///
    /// # Errors
    ///
    /// Returns an error if query heads are not an integer grouping of KV heads.
    pub fn from_head_counts(
        query_heads: usize,
        kv_heads: usize,
    ) -> Result<Self, WorkloadContractError> {
        validate_count("query_heads", query_heads, MAX_HEAD_COUNT)?;
        validate_count("kv_heads", kv_heads, MAX_HEAD_COUNT)?;
        if query_heads == kv_heads {
            return Ok(Self::MultiHead);
        }
        if kv_heads == 1 {
            return Ok(Self::MultiQuery);
        }
        if query_heads % kv_heads == 0 {
            return Ok(Self::GroupedQuery {
                queries_per_kv: query_heads / kv_heads,
            });
        }
        Err(WorkloadContractError::InvalidField(
            "query_heads must be divisible by kv_heads",
        ))
    }

    fn validate(self, query_heads: usize, kv_heads: usize) -> Result<(), WorkloadContractError> {
        match self {
            Self::MultiHead if query_heads == kv_heads => Ok(()),
            Self::MultiQuery if kv_heads == 1 && query_heads > 1 => Ok(()),
            Self::GroupedQuery { queries_per_kv }
                if queries_per_kv >= 2
                    && query_heads % kv_heads == 0
                    && query_heads / kv_heads == queries_per_kv =>
            {
                Ok(())
            }
            _ => Err(WorkloadContractError::InvalidField("head_grouping")),
        }
    }

    fn as_text(self) -> String {
        match self {
            Self::MultiHead => "mha".into(),
            Self::MultiQuery => "mqa".into(),
            Self::GroupedQuery { queries_per_kv } => format!("gqa:{queries_per_kv}"),
        }
    }

    fn from_text(value: &str) -> Result<Self, WorkloadContractError> {
        match value {
            "mha" => Ok(Self::MultiHead),
            "mqa" => Ok(Self::MultiQuery),
            value => {
                let Some(raw) = value.strip_prefix("gqa:") else {
                    return Err(WorkloadContractError::MalformedCanonicalText(
                        "unknown head_grouping".into(),
                    ));
                };
                Ok(Self::GroupedQuery {
                    queries_per_kv: parse_usize("queries_per_kv", raw)?,
                })
            }
        }
    }
}

/// Per-example query and KV lengths.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SequenceLengths {
    query_lengths: Vec<usize>,
    kv_lengths: Vec<usize>,
}

impl SequenceLengths {
    /// Construct a uniform batch with explicit entries for every example.
    ///
    /// # Errors
    ///
    /// Returns an error if a count or length is zero or exceeds its
    /// structural limit.
    pub fn uniform(
        batch_count: usize,
        query_length: usize,
        kv_length: usize,
    ) -> Result<Self, WorkloadContractError> {
        validate_count("batch_count", batch_count, MAX_BATCH_COUNT)?;
        validate_count("query_length", query_length, MAX_SEQUENCE_LENGTH)?;
        validate_count("kv_length", kv_length, MAX_SEQUENCE_LENGTH)?;
        Ok(Self {
            query_lengths: vec![query_length; batch_count],
            kv_lengths: vec![kv_length; batch_count],
        })
    }

    /// Construct a ragged batch with explicit lengths for every example.
    ///
    /// # Errors
    ///
    /// Returns an error if the vectors differ in length or a count/length is
    /// zero or exceeds its structural limit.
    pub fn ragged(
        query_lengths: Vec<usize>,
        kv_lengths: Vec<usize>,
    ) -> Result<Self, WorkloadContractError> {
        if query_lengths.len() != kv_lengths.len() {
            return Err(WorkloadContractError::LengthMismatch {
                query_lengths: query_lengths.len(),
                kv_lengths: kv_lengths.len(),
            });
        }
        validate_count("batch_count", query_lengths.len(), MAX_BATCH_COUNT)?;
        for &length in &query_lengths {
            validate_count("query_length", length, MAX_SEQUENCE_LENGTH)?;
        }
        for &length in &kv_lengths {
            validate_count("kv_length", length, MAX_SEQUENCE_LENGTH)?;
        }
        Ok(Self {
            query_lengths,
            kv_lengths,
        })
    }

    /// Number of examples.
    #[must_use]
    pub fn batch_count(&self) -> usize {
        self.query_lengths.len()
    }

    /// Maximum query length.
    #[must_use]
    pub fn query_length(&self) -> usize {
        self.query_lengths.iter().copied().max().unwrap_or(0)
    }

    /// Maximum KV length.
    #[must_use]
    pub fn kv_length(&self) -> usize {
        self.kv_lengths.iter().copied().max().unwrap_or(0)
    }

    /// Query length for one example.
    #[must_use]
    pub fn query_length_for(&self, batch_index: usize) -> Option<usize> {
        self.query_lengths.get(batch_index).copied()
    }

    /// KV length for one example.
    #[must_use]
    pub fn kv_length_for(&self, batch_index: usize) -> Option<usize> {
        self.kv_lengths.get(batch_index).copied()
    }

    /// Whether the batch has varying query or KV lengths.
    #[must_use]
    pub fn is_ragged(&self) -> bool {
        self.query_lengths.windows(2).any(|pair| pair[0] != pair[1])
            || self.kv_lengths.windows(2).any(|pair| pair[0] != pair[1])
    }

    fn query_lengths(&self) -> &[usize] {
        &self.query_lengths
    }

    fn kv_lengths(&self) -> &[usize] {
        &self.kv_lengths
    }
}

/// Geometry needed to describe a Q/K/V interaction before execution.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct AttentionGeometry {
    sequence_lengths: SequenceLengths,
    query_heads: usize,
    kv_heads: usize,
    qk_dimension: Option<usize>,
    value_dimension: usize,
    topology: AttentionTopology,
    head_grouping: HeadGrouping,
}

/// Constructor input for `AttentionGeometry`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct GeometrySpec {
    /// Per-example query and KV lengths.
    pub sequence_lengths: SequenceLengths,
    /// Number of query heads.
    pub query_heads: usize,
    /// Number of KV heads.
    pub kv_heads: usize,
    /// Q/K dimension, or None for precomputed-score fixtures.
    pub qk_dimension: Option<usize>,
    /// V/output head dimension.
    pub value_dimension: usize,
    /// Self, cross, or explicitly historical topology.
    pub topology: AttentionTopology,
    /// MHA, MQA, or GQA mapping.
    pub head_grouping: HeadGrouping,
}

impl AttentionGeometry {
    /// Construct validated geometry.
    ///
    /// # Errors
    ///
    /// Returns an error if a dimension is invalid or the explicit head
    /// grouping does not match the query/KV head counts.
    pub fn new(spec: GeometrySpec) -> Result<Self, WorkloadContractError> {
        validate_count("query_heads", spec.query_heads, MAX_HEAD_COUNT)?;
        validate_count("kv_heads", spec.kv_heads, MAX_HEAD_COUNT)?;
        validate_count("value_dimension", spec.value_dimension, MAX_HEAD_DIMENSION)?;
        if let Some(dimension) = spec.qk_dimension {
            validate_count("qk_dimension", dimension, MAX_HEAD_DIMENSION)?;
        }
        spec.head_grouping
            .validate(spec.query_heads, spec.kv_heads)?;
        Ok(Self {
            sequence_lengths: spec.sequence_lengths,
            query_heads: spec.query_heads,
            kv_heads: spec.kv_heads,
            qk_dimension: spec.qk_dimension,
            value_dimension: spec.value_dimension,
            topology: spec.topology,
            head_grouping: spec.head_grouping,
        })
    }

    /// Per-example sequence lengths.
    #[must_use]
    pub const fn sequence_lengths(&self) -> &SequenceLengths {
        &self.sequence_lengths
    }

    /// Number of query heads.
    #[must_use]
    pub const fn query_heads(&self) -> usize {
        self.query_heads
    }

    /// Number of KV heads.
    #[must_use]
    pub const fn kv_heads(&self) -> usize {
        self.kv_heads
    }

    /// Q/K dimension when explicit Q/K vectors are available.
    #[must_use]
    pub const fn qk_dimension(&self) -> Option<usize> {
        self.qk_dimension
    }

    /// V/output head dimension.
    #[must_use]
    pub const fn value_dimension(&self) -> usize {
        self.value_dimension
    }

    /// Self/cross/historical topology.
    #[must_use]
    pub const fn topology(&self) -> AttentionTopology {
        self.topology
    }

    /// Explicit MHA/MQA/GQA grouping.
    #[must_use]
    pub const fn head_grouping(&self) -> HeadGrouping {
        self.head_grouping
    }
}

/// Visibility rule for score rows.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum MaskKind {
    /// No mask is applied.
    None,
    /// Every query can see every declared KV position.
    Bidirectional,
    /// A causal visibility rule is part of the contract.
    Causal,
    /// Mask data is supplied by a named external artifact.
    External {
        /// External mask identity.
        identity: String,
    },
}

/// Validated mask metadata.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct MaskSpec {
    kind: MaskKind,
}

impl MaskSpec {
    /// Construct and validate a mask descriptor.
    ///
    /// # Errors
    ///
    /// Returns an error when an external mask identity is empty, contains
    /// whitespace/control characters, or exceeds its size limit.
    pub fn new(kind: MaskKind) -> Result<Self, WorkloadContractError> {
        if let MaskKind::External { identity } = &kind {
            validate_identifier("mask", identity)?;
        }
        Ok(Self { kind })
    }

    /// Unmasked workload.
    #[must_use]
    pub const fn none() -> Self {
        Self {
            kind: MaskKind::None,
        }
    }

    /// Mask kind.
    #[must_use]
    pub const fn kind(&self) -> &MaskKind {
        &self.kind
    }
}

/// Position information used by a semantic/reference implementation.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum PositionInfo {
    /// No position information is declared.
    None,
    /// Positions come from a named absolute-position artifact.
    Absolute {
        /// Position source identity.
        identity: String,
    },
    /// Rotary position information with an explicit dimension.
    Rotary {
        /// Number of dimensions carrying rotary position information.
        dimension: usize,
    },
    /// Relative position information comes from a named artifact.
    Relative {
        /// Position source identity.
        identity: String,
    },
}

impl PositionInfo {
    fn validate(&self) -> Result<(), WorkloadContractError> {
        match self {
            Self::None => Ok(()),
            Self::Absolute { identity } | Self::Relative { identity } => {
                validate_identifier("position", identity)
            }
            Self::Rotary { dimension } => {
                validate_count("position_dimension", *dimension, MAX_HEAD_DIMENSION)
            }
        }
    }
}

/// Optional score-bias source independent of position encoding.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ScoreBiasSpec {
    /// No score bias is declared.
    None,
    /// Bias comes from a named artifact or reference rule.
    Named {
        /// Score-bias identity.
        identity: String,
    },
}

impl ScoreBiasSpec {
    fn validate(&self) -> Result<(), WorkloadContractError> {
        if let Self::Named { identity } = self {
            validate_identifier("score_bias", identity)?;
        }
        Ok(())
    }
}

/// Scalar storage/input/output/accumulation precision declarations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ScalarPrecision {
    /// IEEE binary32 declaration.
    F32,
    /// IEEE binary64 declaration.
    F64,
    /// Brain floating point 16 declaration.
    BF16,
    /// IEEE binary16 declaration.
    F16,
    /// Eight-bit floating point declaration.
    F8,
    /// Four-bit floating point declaration.
    F4,
    /// Eight-bit integer declaration.
    I8,
}

impl ScalarPrecision {
    fn as_text(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::F64 => "f64",
            Self::BF16 => "bf16",
            Self::F16 => "f16",
            Self::F8 => "f8",
            Self::F4 => "f4",
            Self::I8 => "i8",
        }
    }

    fn from_text(value: &str) -> Result<Self, WorkloadContractError> {
        match value {
            "f32" => Ok(Self::F32),
            "f64" => Ok(Self::F64),
            "bf16" => Ok(Self::BF16),
            "f16" => Ok(Self::F16),
            "f8" => Ok(Self::F8),
            "f4" => Ok(Self::F4),
            "i8" => Ok(Self::I8),
            _ => Err(WorkloadContractError::MalformedCanonicalText(
                "unknown scalar precision".into(),
            )),
        }
    }
}

/// Precision policy with accumulation separated from storage and I/O.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PrecisionPolicy {
    input: ScalarPrecision,
    accumulation: ScalarPrecision,
    output: ScalarPrecision,
    storage: ScalarPrecision,
}

impl PrecisionPolicy {
    /// Construct a precision policy. This is a declaration, not a datatype
    /// simulator or physical-hardware claim.
    #[must_use]
    pub const fn new(
        input: ScalarPrecision,
        accumulation: ScalarPrecision,
        output: ScalarPrecision,
        storage: ScalarPrecision,
    ) -> Self {
        Self {
            input,
            accumulation,
            output,
            storage,
        }
    }

    /// Input precision.
    #[must_use]
    pub const fn input(self) -> ScalarPrecision {
        self.input
    }

    /// Accumulator precision.
    #[must_use]
    pub const fn accumulation(self) -> ScalarPrecision {
        self.accumulation
    }

    /// Output precision.
    #[must_use]
    pub const fn output(self) -> ScalarPrecision {
        self.output
    }

    /// KV/storage precision.
    #[must_use]
    pub const fn storage(self) -> ScalarPrecision {
        self.storage
    }
}

/// Matrix memory layout declaration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MatrixLayout {
    /// Consecutive elements advance along columns.
    RowMajor,
    /// Consecutive elements advance along rows.
    ColumnMajor,
    /// Explicit element strides for a strided view.
    Strided {
        /// Stride between adjacent rows.
        row_stride: usize,
        /// Stride between adjacent columns.
        column_stride: usize,
    },
    /// Logical tile shape for a packed layout.
    Tiled {
        /// Tile row extent.
        tile_rows: usize,
        /// Tile column extent.
        tile_columns: usize,
    },
}

impl MatrixLayout {
    fn validate(self, field: &'static str) -> Result<(), WorkloadContractError> {
        match self {
            Self::RowMajor | Self::ColumnMajor => Ok(()),
            Self::Strided {
                row_stride,
                column_stride,
            } => {
                validate_count("row_stride", row_stride, 1 << 30)?;
                validate_count("column_stride", column_stride, 1 << 30)
            }
            Self::Tiled {
                tile_rows,
                tile_columns,
            } => {
                validate_count(field, tile_rows, MAX_SEQUENCE_LENGTH)?;
                validate_count(field, tile_columns, MAX_HEAD_DIMENSION)
            }
        }
    }

    fn as_text(self) -> String {
        match self {
            Self::RowMajor => "row".into(),
            Self::ColumnMajor => "column".into(),
            Self::Strided {
                row_stride,
                column_stride,
            } => format!("strided:{row_stride}:{column_stride}"),
            Self::Tiled {
                tile_rows,
                tile_columns,
            } => format!("tiled:{tile_rows}:{tile_columns}"),
        }
    }

    fn from_text(value: &str) -> Result<Self, WorkloadContractError> {
        match value {
            "row" => Ok(Self::RowMajor),
            "column" => Ok(Self::ColumnMajor),
            value if value.starts_with("strided:") => {
                let parts: Vec<&str> = value[8..].split(':').collect();
                if parts.len() != 2 {
                    return Err(WorkloadContractError::MalformedCanonicalText(
                        "strided layout must have two strides".into(),
                    ));
                }
                Ok(Self::Strided {
                    row_stride: parse_usize("row_stride", parts[0])?,
                    column_stride: parse_usize("column_stride", parts[1])?,
                })
            }
            value if value.starts_with("tiled:") => {
                let parts: Vec<&str> = value[6..].split(':').collect();
                if parts.len() != 2 {
                    return Err(WorkloadContractError::MalformedCanonicalText(
                        "tiled layout must have two tile dimensions".into(),
                    ));
                }
                Ok(Self::Tiled {
                    tile_rows: parse_usize("tile_rows", parts[0])?,
                    tile_columns: parse_usize("tile_columns", parts[1])?,
                })
            }
            _ => Err(WorkloadContractError::MalformedCanonicalText(
                "unknown matrix layout".into(),
            )),
        }
    }
}

/// Per-tensor logical layout declarations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TensorLayout {
    query: MatrixLayout,
    key: MatrixLayout,
    value: MatrixLayout,
    output: MatrixLayout,
}

impl TensorLayout {
    /// Construct a layout declaration.
    #[must_use]
    pub const fn new(
        query: MatrixLayout,
        key: MatrixLayout,
        value: MatrixLayout,
        output: MatrixLayout,
    ) -> Self {
        Self {
            query,
            key,
            value,
            output,
        }
    }

    /// Conventional row-major Q/K/V/output layout.
    #[must_use]
    pub const fn row_major() -> Self {
        Self::new(
            MatrixLayout::RowMajor,
            MatrixLayout::RowMajor,
            MatrixLayout::RowMajor,
            MatrixLayout::RowMajor,
        )
    }

    /// Query layout.
    #[must_use]
    pub const fn query(self) -> MatrixLayout {
        self.query
    }

    /// Key layout.
    #[must_use]
    pub const fn key(self) -> MatrixLayout {
        self.key
    }

    /// Value layout.
    #[must_use]
    pub const fn value(self) -> MatrixLayout {
        self.value
    }

    /// Output layout.
    #[must_use]
    pub const fn output(self) -> MatrixLayout {
        self.output
    }

    fn validate(self) -> Result<(), WorkloadContractError> {
        self.query.validate("query_layout")?;
        self.key.validate("key_layout")?;
        self.value.validate("value_layout")?;
        self.output.validate("output_layout")
    }
}

/// Position treatment attached to a compressed KV representation.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum LatentPositionHandling {
    /// Position information is applied before compression.
    BeforeCompression,
    /// Position information is applied after latent reconstruction.
    AfterCompression,
    /// Position information is kept in a separate named representation.
    Separate {
        /// Position representation identity.
        identity: String,
    },
}

impl LatentPositionHandling {
    fn validate(&self) -> Result<(), WorkloadContractError> {
        if let Self::Separate { identity } = self {
            validate_identifier("latent_position", identity)?;
        }
        Ok(())
    }
}

/// Explicit metadata for latent/compressed KV storage.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct LatentKvSpec {
    latent_dimension: usize,
    key_reconstruction: String,
    value_reconstruction: String,
    position_handling: LatentPositionHandling,
}

impl LatentKvSpec {
    /// Construct a compressed-KV declaration. Reconstruction is named rather
    /// than executed by this crate, so a later semantic/reference layer can
    /// define and test the exact mathematical operation.
    ///
    /// # Errors
    ///
    /// Returns an error when the latent dimension or reconstruction/position
    /// identity is invalid.
    pub fn new(
        latent_dimension: usize,
        key_reconstruction: impl Into<String>,
        value_reconstruction: impl Into<String>,
        position_handling: LatentPositionHandling,
    ) -> Result<Self, WorkloadContractError> {
        let key_reconstruction = key_reconstruction.into();
        let value_reconstruction = value_reconstruction.into();
        validate_count("latent_dimension", latent_dimension, MAX_HEAD_DIMENSION)?;
        validate_identifier("key_reconstruction", &key_reconstruction)?;
        validate_identifier("value_reconstruction", &value_reconstruction)?;
        position_handling.validate()?;
        Ok(Self {
            latent_dimension,
            key_reconstruction,
            value_reconstruction,
            position_handling,
        })
    }

    /// Latent dimension.
    #[must_use]
    pub const fn latent_dimension(&self) -> usize {
        self.latent_dimension
    }

    /// Key reconstruction identity.
    #[must_use]
    pub fn key_reconstruction(&self) -> &str {
        &self.key_reconstruction
    }

    /// Value reconstruction identity.
    #[must_use]
    pub fn value_reconstruction(&self) -> &str {
        &self.value_reconstruction
    }

    /// Position treatment.
    #[must_use]
    pub fn position_handling(&self) -> &LatentPositionHandling {
        &self.position_handling
    }
}

/// Whether KV storage is explicit or compressed.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum KvRepresentation {
    /// Full K/V rows are the declared representation.
    Full,
    /// A latent representation is stored and reconstructed explicitly.
    LatentCompressed(LatentKvSpec),
}

/// KV cache geometry metadata.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum KvCacheSpec {
    /// No cache is part of this workload contract.
    None,
    /// A contiguous cache is used.
    Contiguous,
    /// A paged cache with explicit page-table identity.
    Paged {
        /// Number of logical KV rows in one physical page.
        page_size: usize,
        /// Number of physical pages available to the workload.
        physical_pages: usize,
        /// Block-table identity used to map logical to physical pages.
        block_table_identity: String,
    },
}

impl KvCacheSpec {
    fn validate(&self) -> Result<(), WorkloadContractError> {
        if let Self::Paged {
            page_size,
            physical_pages,
            block_table_identity,
        } = self
        {
            validate_count("page_size", *page_size, MAX_SEQUENCE_LENGTH)?;
            validate_count("physical_pages", *physical_pages, MAX_SEQUENCE_LENGTH)?;
            validate_identifier("block_table", block_table_identity)?;
        }
        Ok(())
    }
}

/// Mapping between logical KV positions and physical storage positions.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum KvIndexing {
    /// Logical and physical positions are identical.
    Identity,
    /// A named mapping artifact defines the physical traversal.
    LogicalToPhysical {
        /// Mapping identity.
        identity: String,
    },
}

impl KvIndexing {
    fn validate(&self) -> Result<(), WorkloadContractError> {
        if let Self::LogicalToPhysical { identity } = self {
            validate_identifier("kv_indexing", identity)?;
        }
        Ok(())
    }
}

/// Input form declared by the workload.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum InputRepresentation {
    /// Explicit Q/K/V tensors are available to a reference evaluator.
    ExplicitQkv,
    /// Scores are precomputed by an external, named fixture.
    PrecomputedScores {
        /// Fixture identity.
        identity: String,
    },
    /// Historical A1 scalar-logit/value fixture; not an explicit Q/K tensor.
    HistoricalA1ScalarFixture,
}

impl InputRepresentation {
    fn validate(&self) -> Result<(), WorkloadContractError> {
        if let Self::PrecomputedScores { identity } = self {
            validate_identifier("precomputed_scores", identity)?;
        }
        Ok(())
    }
}

/// Persistent state metadata for stateful/recurrent workloads.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum StateSpec {
    /// No state persists between declared execution steps.
    Stateless,
    /// Tiny bounded matrix/vector state with explicit external identity.
    Recurrent {
        /// State matrix row count.
        rows: usize,
        /// State matrix column count.
        columns: usize,
        /// State identity.
        identity: String,
    },
}

impl StateSpec {
    fn validate(&self) -> Result<(), WorkloadContractError> {
        if let Self::Recurrent {
            rows,
            columns,
            identity,
        } = self
        {
            validate_count("state_rows", *rows, MAX_STATE_DIMENSION)?;
            validate_count("state_columns", *columns, MAX_STATE_DIMENSION)?;
            validate_identifier("state", identity)?;
        }
        Ok(())
    }
}

/// Experimental mode in which a workload is evaluated.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum WorkloadMode {
    /// Inference-forward prefill over multiple query tokens.
    Prefill,
    /// Inference-forward single-token decode.
    Decode,
    /// Inference-forward multi-token or chunked decode.
    ChunkedDecode,
    /// Training forward pass.
    TrainingForward,
    /// Training backward pass.
    TrainingBackward,
}

impl WorkloadMode {
    fn as_text(self) -> &'static str {
        match self {
            Self::Prefill => "prefill",
            Self::Decode => "decode",
            Self::ChunkedDecode => "chunked-decode",
            Self::TrainingForward => "training-forward",
            Self::TrainingBackward => "training-backward",
        }
    }

    fn from_text(value: &str) -> Result<Self, WorkloadContractError> {
        match value {
            "prefill" => Ok(Self::Prefill),
            "decode" => Ok(Self::Decode),
            "chunked-decode" => Ok(Self::ChunkedDecode),
            "training-forward" => Ok(Self::TrainingForward),
            "training-backward" => Ok(Self::TrainingBackward),
            _ => Err(WorkloadContractError::MalformedCanonicalText(
                "unknown workload mode".into(),
            )),
        }
    }
}

/// Constructor options for `WorkloadContract`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct WorkloadOptions {
    /// Prefill/decode/training mode.
    pub mode: WorkloadMode,
    /// Visibility rule.
    pub mask: MaskSpec,
    /// Position information.
    pub positions: PositionInfo,
    /// Optional score bias.
    pub score_bias: ScoreBiasSpec,
    /// Input/accumulator/output/storage precision.
    pub precision: PrecisionPolicy,
    /// Per-tensor layout.
    pub layout: TensorLayout,
    /// Full or latent KV representation.
    pub kv_representation: KvRepresentation,
    /// Contiguous/paged/no KV cache.
    pub kv_cache: KvCacheSpec,
    /// Logical-to-physical KV mapping.
    pub kv_indexing: KvIndexing,
    /// Explicit Q/K/V, precomputed scores, or historical fixture.
    pub inputs: InputRepresentation,
    /// Persistent recurrent-state metadata.
    pub state: StateSpec,
}

impl Default for WorkloadOptions {
    fn default() -> Self {
        Self {
            mode: WorkloadMode::Prefill,
            mask: MaskSpec::none(),
            positions: PositionInfo::None,
            score_bias: ScoreBiasSpec::None,
            precision: PrecisionPolicy::new(
                ScalarPrecision::F32,
                ScalarPrecision::F32,
                ScalarPrecision::F32,
                ScalarPrecision::F32,
            ),
            layout: TensorLayout::row_major(),
            kv_representation: KvRepresentation::Full,
            kv_cache: KvCacheSpec::None,
            kv_indexing: KvIndexing::Identity,
            inputs: InputRepresentation::ExplicitQkv,
            state: StateSpec::Stateless,
        }
    }
}

/// Versioned, validated description of one research workload.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct WorkloadContract {
    version: u16,
    geometry: AttentionGeometry,
    options: WorkloadOptions,
}

/// Alias emphasizing the common research workload layer.
pub type ResearchWorkload = WorkloadContract;

impl MaskSpec {
    fn validate(&self) -> Result<(), WorkloadContractError> {
        if let MaskKind::External { identity } = &self.kind {
            validate_identifier("mask", identity)?;
        }
        Ok(())
    }
}

impl WorkloadContract {
    /// Construct a version-1 workload contract and validate all cross-field
    /// invariants before returning it.
    ///
    /// # Errors
    ///
    /// Returns an error if any nested contract or cross-field invariant is
    /// invalid.
    pub fn new(
        geometry: AttentionGeometry,
        options: WorkloadOptions,
    ) -> Result<Self, WorkloadContractError> {
        let contract = Self {
            version: WORKLOAD_CONTRACT_VERSION,
            geometry,
            options,
        };
        contract.validate()?;
        Ok(contract)
    }

    /// Explicit adapter for the historical A1 scalar-logit/value fixture.
    ///
    /// The resulting contract uses no Q/K dimension and an explicit historical
    /// input tag. Downstream code can therefore distinguish it from a real
    /// Q/K/V workload without changing the historical case itself.
    ///
    /// # Errors
    ///
    /// Returns an error when the historical A1 case fails its original
    /// validation contract.
    pub fn from_a1_case(case: &ada_core::AttentionCase) -> Result<Self, WorkloadContractError> {
        case.validate()
            .map_err(WorkloadContractError::InvalidHistoricalCase)?;
        let sequence_lengths = SequenceLengths::uniform(1, 1, case.logits.len())?;
        let geometry = AttentionGeometry::new(GeometrySpec {
            sequence_lengths,
            query_heads: 1,
            kv_heads: 1,
            qk_dimension: None,
            value_dimension: case.head_dim,
            topology: AttentionTopology::HistoricalA1,
            head_grouping: HeadGrouping::MultiHead,
        })?;
        Self::new(
            geometry,
            WorkloadOptions {
                inputs: InputRepresentation::HistoricalA1ScalarFixture,
                ..WorkloadOptions::default()
            },
        )
    }

    /// Validate an already-constructed contract.
    ///
    /// # Errors
    ///
    /// Returns an error if the version, nested metadata, or cross-field
    /// invariant is invalid.
    pub fn validate(&self) -> Result<(), WorkloadContractError> {
        if self.version != WORKLOAD_CONTRACT_VERSION {
            return Err(WorkloadContractError::UnsupportedVersion(self.version));
        }
        self.options.mask.validate()?;
        self.options.positions.validate()?;
        self.options.score_bias.validate()?;
        self.options.layout.validate()?;
        self.options.kv_cache.validate()?;
        self.options.kv_indexing.validate()?;
        self.options.inputs.validate()?;
        self.options.state.validate()?;

        match self.options.mode {
            WorkloadMode::Decode => {
                if self
                    .geometry
                    .sequence_lengths
                    .query_lengths()
                    .iter()
                    .any(|&length| length != 1)
                {
                    return Err(WorkloadContractError::InvalidField(
                        "decode requires one query token per example",
                    ));
                }
                if matches!(self.options.kv_cache, KvCacheSpec::None) {
                    return Err(WorkloadContractError::InvalidField(
                        "decode requires explicit KV cache geometry",
                    ));
                }
            }
            WorkloadMode::ChunkedDecode => {
                if matches!(self.options.kv_cache, KvCacheSpec::None) {
                    return Err(WorkloadContractError::InvalidField(
                        "chunked decode requires explicit KV cache geometry",
                    ));
                }
            }
            WorkloadMode::Prefill
            | WorkloadMode::TrainingForward
            | WorkloadMode::TrainingBackward => {}
        }

        if matches!(self.options.kv_cache, KvCacheSpec::None)
            && !matches!(self.options.kv_indexing, KvIndexing::Identity)
        {
            return Err(WorkloadContractError::InvalidField(
                "KV indexing requires KV cache geometry",
            ));
        }
        if matches!(self.options.kv_cache, KvCacheSpec::Paged { .. })
            && !matches!(
                self.options.kv_indexing,
                KvIndexing::LogicalToPhysical { .. }
            )
        {
            return Err(WorkloadContractError::InvalidField(
                "paged KV cache requires logical-to-physical indexing",
            ));
        }

        match self.options.inputs {
            InputRepresentation::ExplicitQkv => {
                if self.geometry.qk_dimension.is_none() {
                    return Err(WorkloadContractError::InvalidField(
                        "explicit QKV inputs require qk_dimension",
                    ));
                }
                if matches!(self.geometry.topology, AttentionTopology::HistoricalA1) {
                    return Err(WorkloadContractError::InvalidField(
                        "historical A1 topology requires the explicit adapter input",
                    ));
                }
            }
            InputRepresentation::PrecomputedScores { .. } => {
                if self.geometry.qk_dimension.is_some()
                    || matches!(self.geometry.topology, AttentionTopology::HistoricalA1)
                {
                    return Err(WorkloadContractError::InvalidField(
                        "precomputed scores must not masquerade as explicit Q/K",
                    ));
                }
            }
            InputRepresentation::HistoricalA1ScalarFixture => {
                if !matches!(self.geometry.topology, AttentionTopology::HistoricalA1)
                    || self.geometry.qk_dimension.is_some()
                    || self.geometry.query_heads != 1
                    || self.geometry.kv_heads != 1
                    || self.geometry.sequence_lengths.batch_count() != 1
                    || self.geometry.sequence_lengths.query_length() != 1
                    || !matches!(self.options.kv_representation, KvRepresentation::Full)
                    || !matches!(self.options.kv_cache, KvCacheSpec::None)
                    || !matches!(self.options.kv_indexing, KvIndexing::Identity)
                    || !matches!(self.options.state, StateSpec::Stateless)
                {
                    return Err(WorkloadContractError::InvalidField(
                        "historical A1 fixture geometry is not explicit",
                    ));
                }
            }
        }
        Ok(())
    }

    /// Contract version.
    #[must_use]
    pub const fn version(&self) -> u16 {
        self.version
    }

    /// Attention geometry.
    #[must_use]
    pub const fn geometry(&self) -> &AttentionGeometry {
        &self.geometry
    }

    /// Workload mode.
    #[must_use]
    pub const fn mode(&self) -> WorkloadMode {
        self.options.mode
    }

    /// Mask contract.
    #[must_use]
    pub const fn mask(&self) -> &MaskSpec {
        &self.options.mask
    }

    /// Position information.
    #[must_use]
    pub const fn positions(&self) -> &PositionInfo {
        &self.options.positions
    }

    /// Optional score bias.
    #[must_use]
    pub const fn score_bias(&self) -> &ScoreBiasSpec {
        &self.options.score_bias
    }

    /// Precision policy.
    #[must_use]
    pub const fn precision(&self) -> PrecisionPolicy {
        self.options.precision
    }

    /// Tensor layouts.
    #[must_use]
    pub const fn layout(&self) -> TensorLayout {
        self.options.layout
    }

    /// Full or latent KV representation.
    #[must_use]
    pub const fn kv_representation(&self) -> &KvRepresentation {
        &self.options.kv_representation
    }

    /// KV cache geometry.
    #[must_use]
    pub const fn kv_cache(&self) -> &KvCacheSpec {
        &self.options.kv_cache
    }

    /// Logical/physical KV indexing declaration.
    #[must_use]
    pub const fn kv_indexing(&self) -> &KvIndexing {
        &self.options.kv_indexing
    }

    /// Input representation.
    #[must_use]
    pub const fn inputs(&self) -> &InputRepresentation {
        &self.options.inputs
    }

    /// Recurrent-state metadata.
    #[must_use]
    pub const fn state(&self) -> &StateSpec {
        &self.options.state
    }

    /// Canonical, deterministic text suitable for evidence artifacts and code
    /// review. Identifiers are hex encoded to keep parsing unambiguous.
    #[must_use]
    #[allow(clippy::too_many_lines)]
    pub fn to_canonical_text(&self) -> String {
        let geometry = &self.geometry;
        let options = &self.options;
        let mut text = String::from("ADA-WORKLOAD-V1\n");
        let qk_dimension = geometry
            .qk_dimension
            .map_or_else(|| "none".into(), |value| value.to_string());
        let (mask_kind, mask_value) = match options.mask.kind() {
            MaskKind::None => ("none", "-".into()),
            MaskKind::Bidirectional => ("bidirectional", "-".into()),
            MaskKind::Causal => ("causal", "-".into()),
            MaskKind::External { identity } => ("external", hex_encode(identity)),
        };
        let (position_kind, position_value) = match &options.positions {
            PositionInfo::None => ("none", "-".into()),
            PositionInfo::Absolute { identity } => ("absolute", hex_encode(identity)),
            PositionInfo::Rotary { dimension } => ("rotary", dimension.to_string()),
            PositionInfo::Relative { identity } => ("relative", hex_encode(identity)),
        };
        let (bias_kind, bias_value) = match &options.score_bias {
            ScoreBiasSpec::None => ("none", "-".into()),
            ScoreBiasSpec::Named { identity } => ("named", hex_encode(identity)),
        };
        let (kv_kind, latent_dimension, key_reconstruction, value_reconstruction, latent_position) =
            match &options.kv_representation {
                KvRepresentation::Full => ("full", "-".into(), "-".into(), "-".into(), "-".into()),
                KvRepresentation::LatentCompressed(spec) => (
                    "latent",
                    spec.latent_dimension.to_string(),
                    hex_encode(&spec.key_reconstruction),
                    hex_encode(&spec.value_reconstruction),
                    latent_position_text(&spec.position_handling),
                ),
            };
        let (cache_kind, page_size, physical_pages, block_table) = match &options.kv_cache {
            KvCacheSpec::None => ("none", "-".into(), "-".into(), "-".into()),
            KvCacheSpec::Contiguous => ("contiguous", "-".into(), "-".into(), "-".into()),
            KvCacheSpec::Paged {
                page_size,
                physical_pages,
                block_table_identity,
            } => (
                "paged",
                page_size.to_string(),
                physical_pages.to_string(),
                hex_encode(block_table_identity),
            ),
        };
        let (index_kind, index_value) = match &options.kv_indexing {
            KvIndexing::Identity => ("identity", "-".into()),
            KvIndexing::LogicalToPhysical { identity } => ("mapping", hex_encode(identity)),
        };
        let (input_kind, input_value) = match &options.inputs {
            InputRepresentation::ExplicitQkv => ("explicit-qkv", "-".into()),
            InputRepresentation::PrecomputedScores { identity } => {
                ("precomputed-scores", hex_encode(identity))
            }
            InputRepresentation::HistoricalA1ScalarFixture => ("historical-a1", "-".into()),
        };
        let (state_kind, state_rows, state_columns, state_value) = match &options.state {
            StateSpec::Stateless => ("stateless", "-".into(), "-".into(), "-".into()),
            StateSpec::Recurrent {
                rows,
                columns,
                identity,
            } => (
                "recurrent",
                rows.to_string(),
                columns.to_string(),
                hex_encode(identity),
            ),
        };
        let precision = format!(
            "{}:{}:{}:{}",
            options.precision.input.as_text(),
            options.precision.accumulation.as_text(),
            options.precision.output.as_text(),
            options.precision.storage.as_text()
        );

        append_field(&mut text, "version", self.version.to_string());
        append_field(
            &mut text,
            "batch_count",
            geometry.sequence_lengths.batch_count().to_string(),
        );
        append_field(
            &mut text,
            "query_lengths",
            join_usizes(geometry.sequence_lengths.query_lengths()),
        );
        append_field(
            &mut text,
            "kv_lengths",
            join_usizes(geometry.sequence_lengths.kv_lengths()),
        );
        append_field(&mut text, "query_heads", geometry.query_heads.to_string());
        append_field(&mut text, "kv_heads", geometry.kv_heads.to_string());
        append_field(&mut text, "qk_dimension", qk_dimension);
        append_field(
            &mut text,
            "value_dimension",
            geometry.value_dimension.to_string(),
        );
        append_field(&mut text, "topology", geometry.topology.as_text());
        append_field(&mut text, "head_grouping", geometry.head_grouping.as_text());
        append_field(&mut text, "mode", options.mode.as_text());
        append_field(&mut text, "mask_kind", mask_kind);
        append_field(&mut text, "mask_value", mask_value);
        append_field(&mut text, "position_kind", position_kind);
        append_field(&mut text, "position_value", position_value);
        append_field(&mut text, "score_bias_kind", bias_kind);
        append_field(&mut text, "score_bias_value", bias_value);
        append_field(&mut text, "precision", precision);
        append_field(&mut text, "layout_query", options.layout.query.as_text());
        append_field(&mut text, "layout_key", options.layout.key.as_text());
        append_field(&mut text, "layout_value", options.layout.value.as_text());
        append_field(&mut text, "layout_output", options.layout.output.as_text());
        append_field(&mut text, "kv_kind", kv_kind);
        append_field(&mut text, "latent_dimension", latent_dimension);
        append_field(&mut text, "key_reconstruction", key_reconstruction);
        append_field(&mut text, "value_reconstruction", value_reconstruction);
        append_field(&mut text, "latent_position", latent_position);
        append_field(&mut text, "cache_kind", cache_kind);
        append_field(&mut text, "page_size", page_size);
        append_field(&mut text, "physical_pages", physical_pages);
        append_field(&mut text, "block_table", block_table);
        append_field(&mut text, "index_kind", index_kind);
        append_field(&mut text, "index_value", index_value);
        append_field(&mut text, "input_kind", input_kind);
        append_field(&mut text, "input_value", input_value);
        append_field(&mut text, "state_kind", state_kind);
        append_field(&mut text, "state_rows", state_rows);
        append_field(&mut text, "state_columns", state_columns);
        append_field(&mut text, "state_value", state_value);
        text
    }

    /// Decode canonical text after validating its exact schema and all
    /// cross-field invariants.
    ///
    /// # Errors
    ///
    /// Returns an error for an unknown version, malformed fields, duplicate or
    /// unknown keys, invalid nested metadata, or a cross-field mismatch.
    #[allow(clippy::too_many_lines)]
    pub fn from_canonical_text(text: &str) -> Result<Self, WorkloadContractError> {
        if text.len() > MAX_CANONICAL_TEXT_BYTES || text.contains('\r') {
            return Err(WorkloadContractError::MalformedCanonicalText(
                "canonical text exceeds its limit or contains CR".into(),
            ));
        }
        if !text.ends_with('\n') {
            return Err(WorkloadContractError::MalformedCanonicalText(
                "canonical text must end with a newline".into(),
            ));
        }
        let mut lines = text.lines();
        if lines.next() != Some("ADA-WORKLOAD-V1") {
            return Err(WorkloadContractError::MalformedCanonicalText(
                "missing ADA-WORKLOAD-V1 header".into(),
            ));
        }
        let mut fields = BTreeMap::new();
        for line in lines {
            let Some((key, value)) = line.split_once('=') else {
                return Err(WorkloadContractError::MalformedCanonicalText(
                    "field is missing '='".into(),
                ));
            };
            if key.is_empty() || value.contains('=') || fields.insert(key, value).is_some() {
                return Err(WorkloadContractError::MalformedCanonicalText(
                    "empty, duplicate, or ambiguous field".into(),
                ));
            }
        }
        let required = [
            "version",
            "batch_count",
            "query_lengths",
            "kv_lengths",
            "query_heads",
            "kv_heads",
            "qk_dimension",
            "value_dimension",
            "topology",
            "head_grouping",
            "mode",
            "mask_kind",
            "mask_value",
            "position_kind",
            "position_value",
            "score_bias_kind",
            "score_bias_value",
            "precision",
            "layout_query",
            "layout_key",
            "layout_value",
            "layout_output",
            "kv_kind",
            "latent_dimension",
            "key_reconstruction",
            "value_reconstruction",
            "latent_position",
            "cache_kind",
            "page_size",
            "physical_pages",
            "block_table",
            "index_kind",
            "index_value",
            "input_kind",
            "input_value",
            "state_kind",
            "state_rows",
            "state_columns",
            "state_value",
        ];
        if fields.len() != required.len() || required.iter().any(|key| !fields.contains_key(key)) {
            return Err(WorkloadContractError::MalformedCanonicalText(
                "canonical field set is incomplete or has unknown keys".into(),
            ));
        }
        let field = |key: &str| {
            fields.get(key).copied().ok_or_else(|| {
                WorkloadContractError::MalformedCanonicalText(format!("missing field {key}"))
            })
        };
        let version = parse_u16("version", field("version")?)?;
        if version != WORKLOAD_CONTRACT_VERSION {
            return Err(WorkloadContractError::UnsupportedVersion(version));
        }
        let batch_count = parse_usize("batch_count", field("batch_count")?)?;
        let query_lengths = parse_usize_list("query_lengths", field("query_lengths")?)?;
        let kv_lengths = parse_usize_list("kv_lengths", field("kv_lengths")?)?;
        if query_lengths.len() != batch_count || kv_lengths.len() != batch_count {
            return Err(WorkloadContractError::LengthMismatch {
                query_lengths: query_lengths.len(),
                kv_lengths: kv_lengths.len(),
            });
        }
        let sequence_lengths = SequenceLengths::ragged(query_lengths, kv_lengths)?;
        let geometry = AttentionGeometry::new(GeometrySpec {
            sequence_lengths,
            query_heads: parse_usize("query_heads", field("query_heads")?)?,
            kv_heads: parse_usize("kv_heads", field("kv_heads")?)?,
            qk_dimension: parse_optional_usize("qk_dimension", field("qk_dimension")?)?,
            value_dimension: parse_usize("value_dimension", field("value_dimension")?)?,
            topology: AttentionTopology::from_text(field("topology")?)?,
            head_grouping: HeadGrouping::from_text(field("head_grouping")?)?,
        })?;
        let options = WorkloadOptions {
            mode: WorkloadMode::from_text(field("mode")?)?,
            mask: parse_mask(field("mask_kind")?, field("mask_value")?)?,
            positions: parse_positions(field("position_kind")?, field("position_value")?)?,
            score_bias: parse_score_bias(field("score_bias_kind")?, field("score_bias_value")?)?,
            precision: parse_precision(field("precision")?)?,
            layout: TensorLayout::new(
                MatrixLayout::from_text(field("layout_query")?)?,
                MatrixLayout::from_text(field("layout_key")?)?,
                MatrixLayout::from_text(field("layout_value")?)?,
                MatrixLayout::from_text(field("layout_output")?)?,
            ),
            kv_representation: parse_kv_representation(
                field("kv_kind")?,
                field("latent_dimension")?,
                field("key_reconstruction")?,
                field("value_reconstruction")?,
                field("latent_position")?,
            )?,
            kv_cache: parse_cache(
                field("cache_kind")?,
                field("page_size")?,
                field("physical_pages")?,
                field("block_table")?,
            )?,
            kv_indexing: parse_indexing(field("index_kind")?, field("index_value")?)?,
            inputs: parse_inputs(field("input_kind")?, field("input_value")?)?,
            state: parse_state(
                field("state_kind")?,
                field("state_rows")?,
                field("state_columns")?,
                field("state_value")?,
            )?,
        };
        let contract = Self {
            version,
            geometry,
            options,
        };
        contract.validate()?;
        Ok(contract)
    }

    /// Stable dual-lane fingerprint of the canonical text representation.
    #[must_use]
    pub fn fingerprint(&self) -> WorkloadFingerprint {
        WorkloadFingerprint::of_bytes(self.to_canonical_text().as_bytes())
    }
}

fn append_field(text: &mut String, key: &str, value: impl Display) {
    text.push_str(key);
    text.push('=');
    text.push_str(&value.to_string());
    text.push('\n');
}

fn join_usizes(values: &[usize]) -> String {
    values
        .iter()
        .map(usize::to_string)
        .collect::<Vec<_>>()
        .join(",")
}

fn parse_optional_usize(field: &str, value: &str) -> Result<Option<usize>, WorkloadContractError> {
    if value == "none" {
        Ok(None)
    } else {
        Ok(Some(parse_usize(field, value)?))
    }
}

fn latent_position_text(position: &LatentPositionHandling) -> String {
    match position {
        LatentPositionHandling::BeforeCompression => "before".into(),
        LatentPositionHandling::AfterCompression => "after".into(),
        LatentPositionHandling::Separate { identity } => {
            format!("separate:{}", hex_encode(identity))
        }
    }
}

fn parse_mask(kind: &str, value: &str) -> Result<MaskSpec, WorkloadContractError> {
    let kind = match kind {
        "none" => MaskKind::None,
        "bidirectional" => MaskKind::Bidirectional,
        "causal" => MaskKind::Causal,
        "external" => MaskKind::External {
            identity: hex_decode("mask_value", value)?,
        },
        _ => {
            return Err(WorkloadContractError::MalformedCanonicalText(
                "unknown mask kind".into(),
            ));
        }
    };
    if !matches!(&kind, MaskKind::External { .. }) && value != "-" {
        return Err(WorkloadContractError::MalformedCanonicalText(
            "non-external mask must use '-' value".into(),
        ));
    }
    MaskSpec::new(kind)
}

fn parse_positions(kind: &str, value: &str) -> Result<PositionInfo, WorkloadContractError> {
    let position = match kind {
        "none" => PositionInfo::None,
        "absolute" => PositionInfo::Absolute {
            identity: hex_decode("position_value", value)?,
        },
        "rotary" => PositionInfo::Rotary {
            dimension: parse_usize("position_dimension", value)?,
        },
        "relative" => PositionInfo::Relative {
            identity: hex_decode("position_value", value)?,
        },
        _ => {
            return Err(WorkloadContractError::MalformedCanonicalText(
                "unknown position kind".into(),
            ));
        }
    };
    if matches!(&position, PositionInfo::None) && value != "-" {
        return Err(WorkloadContractError::MalformedCanonicalText(
            "none position must use '-' value".into(),
        ));
    }
    position.validate()?;
    Ok(position)
}

fn parse_score_bias(kind: &str, value: &str) -> Result<ScoreBiasSpec, WorkloadContractError> {
    let bias = match kind {
        "none" => ScoreBiasSpec::None,
        "named" => ScoreBiasSpec::Named {
            identity: hex_decode("score_bias_value", value)?,
        },
        _ => {
            return Err(WorkloadContractError::MalformedCanonicalText(
                "unknown score-bias kind".into(),
            ));
        }
    };
    if matches!(&bias, ScoreBiasSpec::None) && value != "-" {
        return Err(WorkloadContractError::MalformedCanonicalText(
            "none score bias must use '-' value".into(),
        ));
    }
    bias.validate()?;
    Ok(bias)
}

fn parse_precision(value: &str) -> Result<PrecisionPolicy, WorkloadContractError> {
    let parts: Vec<&str> = value.split(':').collect();
    if parts.len() != 4 {
        return Err(WorkloadContractError::MalformedCanonicalText(
            "precision must contain four components".into(),
        ));
    }
    Ok(PrecisionPolicy::new(
        ScalarPrecision::from_text(parts[0])?,
        ScalarPrecision::from_text(parts[1])?,
        ScalarPrecision::from_text(parts[2])?,
        ScalarPrecision::from_text(parts[3])?,
    ))
}

fn parse_latent_position(value: &str) -> Result<LatentPositionHandling, WorkloadContractError> {
    let position = match value {
        "before" => LatentPositionHandling::BeforeCompression,
        "after" => LatentPositionHandling::AfterCompression,
        value if value.starts_with("separate:") => LatentPositionHandling::Separate {
            identity: hex_decode("latent_position", &value[9..])?,
        },
        _ => {
            return Err(WorkloadContractError::MalformedCanonicalText(
                "unknown latent position handling".into(),
            ));
        }
    };
    position.validate()?;
    Ok(position)
}

fn parse_kv_representation(
    kind: &str,
    dimension: &str,
    key_reconstruction: &str,
    value_reconstruction: &str,
    position: &str,
) -> Result<KvRepresentation, WorkloadContractError> {
    match kind {
        "full" => {
            if dimension != "-"
                || key_reconstruction != "-"
                || value_reconstruction != "-"
                || position != "-"
            {
                return Err(WorkloadContractError::MalformedCanonicalText(
                    "full KV representation must use '-' latent fields".into(),
                ));
            }
            Ok(KvRepresentation::Full)
        }
        "latent" => Ok(KvRepresentation::LatentCompressed(LatentKvSpec::new(
            parse_usize("latent_dimension", dimension)?,
            hex_decode("key_reconstruction", key_reconstruction)?,
            hex_decode("value_reconstruction", value_reconstruction)?,
            parse_latent_position(position)?,
        )?)),
        _ => Err(WorkloadContractError::MalformedCanonicalText(
            "unknown KV representation".into(),
        )),
    }
}

fn parse_cache(
    kind: &str,
    page_size: &str,
    physical_pages: &str,
    block_table: &str,
) -> Result<KvCacheSpec, WorkloadContractError> {
    let cache = match kind {
        "none" => KvCacheSpec::None,
        "contiguous" => KvCacheSpec::Contiguous,
        "paged" => KvCacheSpec::Paged {
            page_size: parse_usize("page_size", page_size)?,
            physical_pages: parse_usize("physical_pages", physical_pages)?,
            block_table_identity: hex_decode("block_table", block_table)?,
        },
        _ => {
            return Err(WorkloadContractError::MalformedCanonicalText(
                "unknown KV cache kind".into(),
            ));
        }
    };
    if matches!(&cache, KvCacheSpec::None | KvCacheSpec::Contiguous)
        && (page_size != "-" || physical_pages != "-" || block_table != "-")
    {
        return Err(WorkloadContractError::MalformedCanonicalText(
            "non-paged KV cache must use '-' page fields".into(),
        ));
    }
    cache.validate()?;
    Ok(cache)
}

fn parse_indexing(kind: &str, value: &str) -> Result<KvIndexing, WorkloadContractError> {
    let indexing = match kind {
        "identity" => KvIndexing::Identity,
        "mapping" => KvIndexing::LogicalToPhysical {
            identity: hex_decode("index_value", value)?,
        },
        _ => {
            return Err(WorkloadContractError::MalformedCanonicalText(
                "unknown KV indexing kind".into(),
            ));
        }
    };
    if matches!(&indexing, KvIndexing::Identity) && value != "-" {
        return Err(WorkloadContractError::MalformedCanonicalText(
            "identity KV indexing must use '-' value".into(),
        ));
    }
    indexing.validate()?;
    Ok(indexing)
}

fn parse_inputs(kind: &str, value: &str) -> Result<InputRepresentation, WorkloadContractError> {
    let inputs = match kind {
        "explicit-qkv" => InputRepresentation::ExplicitQkv,
        "precomputed-scores" => InputRepresentation::PrecomputedScores {
            identity: hex_decode("input_value", value)?,
        },
        "historical-a1" => InputRepresentation::HistoricalA1ScalarFixture,
        _ => {
            return Err(WorkloadContractError::MalformedCanonicalText(
                "unknown input representation".into(),
            ));
        }
    };
    if matches!(
        &inputs,
        InputRepresentation::ExplicitQkv | InputRepresentation::HistoricalA1ScalarFixture
    ) && value != "-"
    {
        return Err(WorkloadContractError::MalformedCanonicalText(
            "non-precomputed input must use '-' value".into(),
        ));
    }
    inputs.validate()?;
    Ok(inputs)
}

fn parse_state(
    kind: &str,
    rows: &str,
    columns: &str,
    value: &str,
) -> Result<StateSpec, WorkloadContractError> {
    let state = match kind {
        "stateless" => StateSpec::Stateless,
        "recurrent" => StateSpec::Recurrent {
            rows: parse_usize("state_rows", rows)?,
            columns: parse_usize("state_columns", columns)?,
            identity: hex_decode("state_value", value)?,
        },
        _ => {
            return Err(WorkloadContractError::MalformedCanonicalText(
                "unknown state kind".into(),
            ));
        }
    };
    if matches!(&state, StateSpec::Stateless) && (rows != "-" || columns != "-" || value != "-") {
        return Err(WorkloadContractError::MalformedCanonicalText(
            "stateless state must use '-' fields".into(),
        ));
    }
    state.validate()?;
    Ok(state)
}

/// Stable dual-lane fingerprint for a workload contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct WorkloadFingerprint {
    primary: u64,
    secondary: u64,
    length: u64,
}

impl WorkloadFingerprint {
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
        primary ^= length;
        primary = primary.wrapping_mul(FNV_PRIME);
        secondary = secondary.rotate_left(31) ^ length;
        Self {
            primary,
            secondary,
            length,
        }
    }

    /// Primary fingerprint lane.
    #[must_use]
    pub const fn primary(self) -> u64 {
        self.primary
    }

    /// Secondary fingerprint lane.
    #[must_use]
    pub const fn secondary(self) -> u64 {
        self.secondary
    }

    /// Canonical text byte length included in the fingerprint.
    #[must_use]
    pub const fn length(self) -> u64 {
        self.length
    }
}

impl Display for WorkloadFingerprint {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "{:016x}-{:016x}-{:016x}",
            self.primary, self.secondary, self.length
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn explicit_workload() -> WorkloadContract {
        let geometry = AttentionGeometry::new(GeometrySpec {
            sequence_lengths: SequenceLengths::ragged(vec![4, 2], vec![8, 3]).unwrap(),
            query_heads: 8,
            kv_heads: 2,
            qk_dimension: Some(64),
            value_dimension: 96,
            topology: AttentionTopology::SelfAttention,
            head_grouping: HeadGrouping::GroupedQuery { queries_per_kv: 4 },
        })
        .unwrap();
        WorkloadContract::new(
            geometry,
            WorkloadOptions {
                mode: WorkloadMode::ChunkedDecode,
                mask: MaskSpec::new(MaskKind::External {
                    identity: "mask-v1".into(),
                })
                .unwrap(),
                positions: PositionInfo::Rotary { dimension: 64 },
                score_bias: ScoreBiasSpec::Named {
                    identity: "alibi-v2".into(),
                },
                precision: PrecisionPolicy::new(
                    ScalarPrecision::BF16,
                    ScalarPrecision::F32,
                    ScalarPrecision::BF16,
                    ScalarPrecision::BF16,
                ),
                layout: TensorLayout::new(
                    MatrixLayout::RowMajor,
                    MatrixLayout::Strided {
                        row_stride: 128,
                        column_stride: 1,
                    },
                    MatrixLayout::Tiled {
                        tile_rows: 16,
                        tile_columns: 32,
                    },
                    MatrixLayout::RowMajor,
                ),
                kv_representation: KvRepresentation::LatentCompressed(
                    LatentKvSpec::new(
                        32,
                        "key-decode-v1",
                        "value-decode-v1",
                        LatentPositionHandling::Separate {
                            identity: "rope-cache-v1".into(),
                        },
                    )
                    .unwrap(),
                ),
                kv_cache: KvCacheSpec::Paged {
                    page_size: 16,
                    physical_pages: 128,
                    block_table_identity: "block-table-v3".into(),
                },
                kv_indexing: KvIndexing::LogicalToPhysical {
                    identity: "page-map-v3".into(),
                },
                inputs: InputRepresentation::ExplicitQkv,
                state: StateSpec::Stateless,
            },
        )
        .unwrap()
    }

    #[test]
    fn general_contract_covers_geometry_modes_and_representation_without_execution() {
        let workload = explicit_workload();
        assert_eq!(workload.version(), WORKLOAD_CONTRACT_VERSION);
        assert_eq!(workload.geometry().sequence_lengths().batch_count(), 2);
        assert!(workload.geometry().sequence_lengths().is_ragged());
        assert_eq!(
            workload.geometry().head_grouping(),
            HeadGrouping::GroupedQuery { queries_per_kv: 4 }
        );
        assert_eq!(
            HeadGrouping::from_head_counts(16, 1).unwrap(),
            HeadGrouping::MultiQuery
        );
        assert!(matches!(workload.kv_cache(), KvCacheSpec::Paged { .. }));
        assert!(matches!(
            workload.kv_representation(),
            KvRepresentation::LatentCompressed(_)
        ));
    }

    #[test]
    fn canonical_text_round_trip_and_fingerprint_are_deterministic() {
        let workload = explicit_workload();
        let text = workload.to_canonical_text();
        assert_eq!(text, workload.to_canonical_text());
        let decoded = WorkloadContract::from_canonical_text(&text).unwrap();
        assert_eq!(decoded, workload);
        assert_eq!(decoded.to_canonical_text(), text);
        assert_eq!(decoded.fingerprint(), workload.fingerprint());
        assert_eq!(format!("{}", workload.fingerprint()).len(), 16 * 3 + 2);
    }

    #[test]
    fn fingerprint_separates_experiment_mode_from_geometry() {
        let workload = explicit_workload();
        let mut changed_mode = workload.clone();
        changed_mode.options.mode = WorkloadMode::Prefill;
        changed_mode.validate().unwrap();
        assert_ne!(workload.fingerprint(), changed_mode.fingerprint());
    }

    #[test]
    fn historical_a1_adapter_is_explicit_and_does_not_infer_qk_dimension() {
        let case = ada_core::AttentionCase {
            logits: vec![1.0, -2.0, 3.0],
            values: vec![0.0, 1.0, 2.0],
            head_dim: 1,
        };
        let workload = WorkloadContract::from_a1_case(&case).unwrap();
        assert_eq!(
            workload.geometry().topology(),
            AttentionTopology::HistoricalA1
        );
        assert_eq!(workload.geometry().qk_dimension(), None);
        assert_eq!(workload.geometry().sequence_lengths().query_length(), 1);
        assert_eq!(workload.geometry().sequence_lengths().kv_length(), 3);
        assert!(matches!(
            workload.inputs(),
            InputRepresentation::HistoricalA1ScalarFixture
        ));
        assert_eq!(case.validate(), Ok(()));
    }

    #[test]
    fn invalid_contracts_fail_closed() {
        assert!(SequenceLengths::ragged(vec![2], vec![2, 3]).is_err());
        assert!(SequenceLengths::uniform(1, 0, 4).is_err());
        assert!(HeadGrouping::from_head_counts(5, 2).is_err());
        assert!(HeadGrouping::MultiQuery.validate(1, 1).is_err());
        assert!(
            MaskSpec::new(MaskKind::External {
                identity: "contains whitespace".into(),
            })
            .is_err()
        );

        let geometry = AttentionGeometry::new(GeometrySpec {
            sequence_lengths: SequenceLengths::uniform(1, 1, 4).unwrap(),
            query_heads: 2,
            kv_heads: 1,
            qk_dimension: Some(8),
            value_dimension: 8,
            topology: AttentionTopology::SelfAttention,
            head_grouping: HeadGrouping::MultiQuery,
        })
        .unwrap();
        assert!(
            WorkloadContract::new(
                geometry,
                WorkloadOptions {
                    mode: WorkloadMode::Decode,
                    ..WorkloadOptions::default()
                },
            )
            .is_err()
        );
    }

    #[test]
    fn malformed_and_adversarial_text_is_rejected_without_partial_contract() {
        let workload = explicit_workload();
        let mut text = workload.to_canonical_text();
        text.push_str("unknown=field\n");
        assert!(WorkloadContract::from_canonical_text(&text).is_err());

        let text = workload
            .to_canonical_text()
            .replace("query_heads=8", "query_heads=not-a-number");
        assert!(WorkloadContract::from_canonical_text(&text).is_err());

        let text = workload
            .to_canonical_text()
            .replace("ADA-WORKLOAD-V1", "ADA-WORKLOAD-V999");
        assert!(WorkloadContract::from_canonical_text(&text).is_err());
    }

    #[test]
    fn decode_requires_cache_and_single_query_token() {
        let geometry = AttentionGeometry::new(GeometrySpec {
            sequence_lengths: SequenceLengths::uniform(1, 1, 8).unwrap(),
            query_heads: 1,
            kv_heads: 1,
            qk_dimension: Some(16),
            value_dimension: 16,
            topology: AttentionTopology::SelfAttention,
            head_grouping: HeadGrouping::MultiHead,
        })
        .unwrap();
        assert!(
            WorkloadContract::new(
                geometry,
                WorkloadOptions {
                    mode: WorkloadMode::Decode,
                    kv_cache: KvCacheSpec::Contiguous,
                    ..WorkloadOptions::default()
                },
            )
            .is_ok()
        );
    }
}
