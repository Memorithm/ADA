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

mod canonical;

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
mod tests;
