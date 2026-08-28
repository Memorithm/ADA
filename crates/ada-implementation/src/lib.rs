//! Backend-neutral implementation, schedule, and memory IR for ADA research.
//!
//! This crate describes **how** a qualified attention semantic may be realized
//! without changing **what** the semantic computes.  It deliberately contains
//! no device identity, benchmark result, task metric, or hardware claim.
//!
//! The first version is intentionally small.  It can distinguish algorithmic
//! realization, tiling/partition/reduction choices, and memory placement while
//! remaining deterministic, inspectable, bounded, and fail-closed.

#![forbid(unsafe_code)]

use ada_core::{ImplementationCandidateId, SemanticFamily, SemanticId};
use std::collections::BTreeMap;
use std::fmt::{Display, Formatter, Write as _};

/// Version of the backend-neutral implementation IR.
pub const IMPLEMENTATION_IR_VERSION: u16 = 1;
/// Maximum canonical artifact size accepted by the decoder.
pub const MAX_CANONICAL_TEXT_BYTES: usize = 1 << 20;
/// Maximum tile extent accepted by the research representation.
pub const MAX_TILE_EXTENT: u32 = 65_536;
/// Maximum split-KV partition count.
pub const MAX_SPLIT_KV_PARTITIONS: u16 = 1_024;
/// Maximum software-pipeline stage count.
pub const MAX_PIPELINE_STAGES: u8 = 32;
/// Maximum declared vector width.
pub const MAX_VECTOR_WIDTH: u16 = 256;
/// Maximum declared memory alignment.
pub const MAX_ALIGNMENT_BYTES: u32 = 65_536;
/// Maximum optional KV page size in rows.
pub const MAX_KV_PAGE_ROWS: u32 = 1 << 20;

/// Construction or decoding failure for an implementation plan.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ImplementationError {
    /// A field violates the v1 structural contract.
    InvalidField(&'static str),
    /// A bounded value exceeded the representation limit.
    ExceedsLimit {
        /// Field whose value was rejected.
        field: &'static str,
        /// Rejected value.
        value: u64,
        /// Inclusive maximum.
        maximum: u64,
    },
    /// Canonical text was malformed, incomplete, duplicated, or non-canonical.
    MalformedCanonical(String),
    /// The artifact uses a version unsupported by this decoder.
    UnsupportedVersion(u16),
}

impl Display for ImplementationError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidField(field) => write!(formatter, "invalid implementation field: {field}"),
            Self::ExceedsLimit {
                field,
                value,
                maximum,
            } => write!(formatter, "{field}={value} exceeds maximum {maximum}"),
            Self::MalformedCanonical(reason) => {
                write!(
                    formatter,
                    "malformed implementation canonical text: {reason}"
                )
            }
            Self::UnsupportedVersion(version) => {
                write!(formatter, "unsupported implementation IR version {version}")
            }
        }
    }
}

impl std::error::Error for ImplementationError {}

/// Semantic-preserving algorithmic realization family.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AlgorithmPlan {
    /// Dense attention evaluated in bounded blocks.
    DenseBlocked,
    /// Online/streaming recurrence over the key/value sequence.
    OnlineStreaming,
    /// Explicit first pass followed by a second accumulation pass.
    TwoPass,
    /// Blocked traversal intended for paged or indirect KV storage.
    PagedBlocked,
    /// Stateful/chunked realization for a semantic that defines carried state.
    RecurrentChunked,
}

impl AlgorithmPlan {
    const fn as_text(self) -> &'static str {
        match self {
            Self::DenseBlocked => "dense-blocked",
            Self::OnlineStreaming => "online-streaming",
            Self::TwoPass => "two-pass",
            Self::PagedBlocked => "paged-blocked",
            Self::RecurrentChunked => "recurrent-chunked",
        }
    }

    fn from_text(value: &str) -> Result<Self, ImplementationError> {
        match value {
            "dense-blocked" => Ok(Self::DenseBlocked),
            "online-streaming" => Ok(Self::OnlineStreaming),
            "two-pass" => Ok(Self::TwoPass),
            "paged-blocked" => Ok(Self::PagedBlocked),
            "recurrent-chunked" => Ok(Self::RecurrentChunked),
            _ => Err(ImplementationError::MalformedCanonical(
                "unknown algorithm plan".into(),
            )),
        }
    }
}

/// Query/KV work partitioning strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum WorkPartition {
    /// One logical worker stream owns the complete operation.
    Serial,
    /// Parallel work is partitioned by query tiles.
    QueryTiles,
    /// KV work is split and later reduced across the declared partition count.
    SplitKv {
        /// Number of logical split-KV partitions.
        partitions: u16,
    },
}

impl WorkPartition {
    fn validate(self) -> Result<(), ImplementationError> {
        if let Self::SplitKv { partitions } = self {
            if partitions == 0 {
                return Err(ImplementationError::InvalidField(
                    "schedule.partition.partitions",
                ));
            }
            if partitions > MAX_SPLIT_KV_PARTITIONS {
                return Err(ImplementationError::ExceedsLimit {
                    field: "schedule.partition.partitions",
                    value: u64::from(partitions),
                    maximum: u64::from(MAX_SPLIT_KV_PARTITIONS),
                });
            }
        }
        Ok(())
    }

    fn to_text(self) -> String {
        match self {
            Self::Serial => "serial".into(),
            Self::QueryTiles => "query-tiles".into(),
            Self::SplitKv { partitions } => format!("split-kv:{partitions}"),
        }
    }

    fn from_text(value: &str) -> Result<Self, ImplementationError> {
        match value {
            "serial" => Ok(Self::Serial),
            "query-tiles" => Ok(Self::QueryTiles),
            _ => {
                let Some(raw) = value.strip_prefix("split-kv:") else {
                    return Err(ImplementationError::MalformedCanonical(
                        "unknown work partition".into(),
                    ));
                };
                Ok(Self::SplitKv {
                    partitions: parse_u16(raw, "schedule.partition.partitions")?,
                })
            }
        }
    }
}

/// Reduction topology used to combine partial results.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ReductionTopology {
    /// Sequential accumulation.
    Serial,
    /// Balanced tree reduction.
    Tree,
    /// Store partials and finalize in a second reduction pass.
    TwoPass,
}

impl ReductionTopology {
    const fn as_text(self) -> &'static str {
        match self {
            Self::Serial => "serial",
            Self::Tree => "tree",
            Self::TwoPass => "two-pass",
        }
    }

    fn from_text(value: &str) -> Result<Self, ImplementationError> {
        match value {
            "serial" => Ok(Self::Serial),
            "tree" => Ok(Self::Tree),
            "two-pass" => Ok(Self::TwoPass),
            _ => Err(ImplementationError::MalformedCanonical(
                "unknown reduction topology".into(),
            )),
        }
    }
}

/// Exponential/rescaling realization strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ExpStrategy {
    /// Ordinary library/backend exponential evaluation.
    Standard,
    /// Rescale only when the running reference changes.
    ConditionalRescale,
    /// Defer exponentiation/rescaling to a later implementation phase.
    Deferred,
}

impl ExpStrategy {
    const fn as_text(self) -> &'static str {
        match self {
            Self::Standard => "standard",
            Self::ConditionalRescale => "conditional-rescale",
            Self::Deferred => "deferred",
        }
    }

    fn from_text(value: &str) -> Result<Self, ImplementationError> {
        match value {
            "standard" => Ok(Self::Standard),
            "conditional-rescale" => Ok(Self::ConditionalRescale),
            "deferred" => Ok(Self::Deferred),
            _ => Err(ImplementationError::MalformedCanonical(
                "unknown exp strategy".into(),
            )),
        }
    }
}

/// Logical buffering policy.  This is schedule metadata, not a hardware claim.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Buffering {
    /// One logical buffer per staged value.
    Single,
    /// Two logical buffers permit overlap in a later backend lowering.
    Double,
}

impl Buffering {
    const fn as_text(self) -> &'static str {
        match self {
            Self::Single => "single",
            Self::Double => "double",
        }
    }

    fn from_text(value: &str) -> Result<Self, ImplementationError> {
        match value {
            "single" => Ok(Self::Single),
            "double" => Ok(Self::Double),
            _ => Err(ImplementationError::MalformedCanonical(
                "unknown buffering policy".into(),
            )),
        }
    }
}

/// Backend-neutral tile shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TileShape {
    /// Query rows per logical tile.
    pub queries: u32,
    /// KV rows per logical tile.
    pub keys: u32,
    /// Value/output lanes per logical tile.
    pub values: u32,
}

impl TileShape {
    fn validate(self) -> Result<(), ImplementationError> {
        validate_nonzero_bounded("schedule.tile.queries", self.queries, MAX_TILE_EXTENT)?;
        validate_nonzero_bounded("schedule.tile.keys", self.keys, MAX_TILE_EXTENT)?;
        validate_nonzero_bounded("schedule.tile.values", self.values, MAX_TILE_EXTENT)
    }
}

/// Backend-neutral schedule description.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SchedulePlan {
    /// Logical tiling geometry.
    pub tile: TileShape,
    /// Work partitioning policy.
    pub partition: WorkPartition,
    /// Partial-result reduction topology.
    pub reduction: ReductionTopology,
    /// Exponential/rescaling implementation strategy.
    pub exp_strategy: ExpStrategy,
    /// Number of software-pipeline stages.
    pub pipeline_stages: u8,
    /// Logical vector width used by a later backend lowering.
    pub vector_width: u16,
    /// Logical buffering policy.
    pub buffering: Buffering,
}

impl SchedulePlan {
    /// Validate all bounded schedule fields.
    ///
    /// # Errors
    ///
    /// Returns an error for zero/oversized tiles, invalid split counts,
    /// unsupported vector widths, or pipeline stages outside the v1 bounds.
    pub fn validate(self) -> Result<(), ImplementationError> {
        self.tile.validate()?;
        self.partition.validate()?;
        if self.pipeline_stages == 0 {
            return Err(ImplementationError::InvalidField(
                "schedule.pipeline_stages",
            ));
        }
        if self.pipeline_stages > MAX_PIPELINE_STAGES {
            return Err(ImplementationError::ExceedsLimit {
                field: "schedule.pipeline_stages",
                value: u64::from(self.pipeline_stages),
                maximum: u64::from(MAX_PIPELINE_STAGES),
            });
        }
        if self.vector_width == 0 || !self.vector_width.is_power_of_two() {
            return Err(ImplementationError::InvalidField("schedule.vector_width"));
        }
        if self.vector_width > MAX_VECTOR_WIDTH {
            return Err(ImplementationError::ExceedsLimit {
                field: "schedule.vector_width",
                value: u64::from(self.vector_width),
                maximum: u64::from(MAX_VECTOR_WIDTH),
            });
        }
        Ok(())
    }
}

/// Abstract memory/storage level for an implementation plan.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MemoryLevel {
    /// Main accelerator/device-visible memory or ordinary process memory.
    Global,
    /// Workgroup/CTA-shared scratch storage.
    Shared,
    /// Worker/thread-local storage that is not promised to be a register.
    Local,
    /// Register-like scalar/vector storage requested from a later lowering.
    Register,
}

impl MemoryLevel {
    const fn as_text(self) -> &'static str {
        match self {
            Self::Global => "global",
            Self::Shared => "shared",
            Self::Local => "local",
            Self::Register => "register",
        }
    }

    fn from_text(value: &str) -> Result<Self, ImplementationError> {
        match value {
            "global" => Ok(Self::Global),
            "shared" => Ok(Self::Shared),
            "local" => Ok(Self::Local),
            "register" => Ok(Self::Register),
            _ => Err(ImplementationError::MalformedCanonical(
                "unknown memory level".into(),
            )),
        }
    }
}

/// Storage and traversal choices separated from semantic identity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MemoryPlan {
    /// Query storage level while consumed by the implementation.
    pub query: MemoryLevel,
    /// Key storage level while consumed by the implementation.
    pub key: MemoryLevel,
    /// Value storage level while consumed by the implementation.
    pub value: MemoryLevel,
    /// Output storage level while produced by the implementation.
    pub output: MemoryLevel,
    /// Accumulator storage level.
    pub accumulator: MemoryLevel,
    /// Declared temporary workspace, excluding semantic inputs/outputs.
    pub workspace_bytes: u64,
    /// Requested alignment for implementation-owned buffers.
    pub alignment_bytes: u32,
    /// Optional KV page size in rows.  Presence does not imply measured paging
    /// performance and is independent of semantic identity.
    pub kv_page_rows: Option<u32>,
}

impl MemoryPlan {
    /// Validate bounded memory-plan metadata.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid alignment or page geometry.
    pub fn validate(self) -> Result<(), ImplementationError> {
        if self.alignment_bytes == 0 || !self.alignment_bytes.is_power_of_two() {
            return Err(ImplementationError::InvalidField("memory.alignment_bytes"));
        }
        if self.alignment_bytes > MAX_ALIGNMENT_BYTES {
            return Err(ImplementationError::ExceedsLimit {
                field: "memory.alignment_bytes",
                value: u64::from(self.alignment_bytes),
                maximum: u64::from(MAX_ALIGNMENT_BYTES),
            });
        }
        if let Some(rows) = self.kv_page_rows {
            validate_nonzero_bounded("memory.kv_page_rows", rows, MAX_KV_PAGE_ROWS)?;
        }
        Ok(())
    }
}

/// Stable three-lane fingerprint of canonical implementation text.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ImplementationFingerprint {
    primary: u64,
    secondary: u64,
    length: u64,
}

impl ImplementationFingerprint {
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

    /// Canonical byte length bound into this fingerprint.
    #[must_use]
    pub const fn length(self) -> u64 {
        self.length
    }
}

impl Display for ImplementationFingerprint {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "{:016x}-{:016x}-{:016x}",
            self.primary, self.secondary, self.length
        )
    }
}

/// Complete inspectable implementation candidate for one existing semantic.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ImplementationPlan {
    id: ImplementationCandidateId,
    algorithm: AlgorithmPlan,
    schedule: SchedulePlan,
    memory: MemoryPlan,
}

impl ImplementationPlan {
    /// Construct a validated backend-neutral implementation plan.
    ///
    /// # Errors
    ///
    /// Returns an error when schedule or memory metadata violates the bounded
    /// implementation contract.
    pub fn new(
        id: ImplementationCandidateId,
        algorithm: AlgorithmPlan,
        schedule: SchedulePlan,
        memory: MemoryPlan,
    ) -> Result<Self, ImplementationError> {
        schedule.validate()?;
        memory.validate()?;
        Ok(Self {
            id,
            algorithm,
            schedule,
            memory,
        })
    }

    /// Implementation identity bound to an existing semantic identity.
    #[must_use]
    pub const fn id(&self) -> &ImplementationCandidateId {
        &self.id
    }

    /// Algorithmic realization choice.
    #[must_use]
    pub const fn algorithm(&self) -> AlgorithmPlan {
        self.algorithm
    }

    /// Backend-neutral schedule metadata.
    #[must_use]
    pub const fn schedule(&self) -> SchedulePlan {
        self.schedule
    }

    /// Memory/storage metadata.
    #[must_use]
    pub const fn memory(&self) -> MemoryPlan {
        self.memory
    }

    /// Canonical implementation artifact suitable for review and evidence.
    #[must_use]
    pub fn to_canonical_text(&self) -> String {
        let semantic = self.id.semantic();
        let mut text = String::new();
        writeln!(text, "ADA-IMPLEMENTATION-V{IMPLEMENTATION_IR_VERSION}")
            .expect("writing to String cannot fail");
        writeln!(
            text,
            "semantic_family={}",
            semantic_family_text(semantic.family())
        )
        .expect("writing to String cannot fail");
        writeln!(text, "semantic_name={}", semantic.name()).expect("writing to String cannot fail");
        writeln!(text, "semantic_revision={}", semantic.revision())
            .expect("writing to String cannot fail");
        writeln!(text, "implementation_name={}", self.id.name())
            .expect("writing to String cannot fail");
        writeln!(text, "implementation_revision={}", self.id.revision())
            .expect("writing to String cannot fail");
        writeln!(text, "algorithm={}", self.algorithm.as_text())
            .expect("writing to String cannot fail");
        writeln!(text, "tile_queries={}", self.schedule.tile.queries)
            .expect("writing to String cannot fail");
        writeln!(text, "tile_keys={}", self.schedule.tile.keys)
            .expect("writing to String cannot fail");
        writeln!(text, "tile_values={}", self.schedule.tile.values)
            .expect("writing to String cannot fail");
        writeln!(text, "partition={}", self.schedule.partition.to_text())
            .expect("writing to String cannot fail");
        writeln!(text, "reduction={}", self.schedule.reduction.as_text())
            .expect("writing to String cannot fail");
        writeln!(
            text,
            "exp_strategy={}",
            self.schedule.exp_strategy.as_text()
        )
        .expect("writing to String cannot fail");
        writeln!(text, "pipeline_stages={}", self.schedule.pipeline_stages)
            .expect("writing to String cannot fail");
        writeln!(text, "vector_width={}", self.schedule.vector_width)
            .expect("writing to String cannot fail");
        writeln!(text, "buffering={}", self.schedule.buffering.as_text())
            .expect("writing to String cannot fail");
        writeln!(text, "query_memory={}", self.memory.query.as_text())
            .expect("writing to String cannot fail");
        writeln!(text, "key_memory={}", self.memory.key.as_text())
            .expect("writing to String cannot fail");
        writeln!(text, "value_memory={}", self.memory.value.as_text())
            .expect("writing to String cannot fail");
        writeln!(text, "output_memory={}", self.memory.output.as_text())
            .expect("writing to String cannot fail");
        writeln!(
            text,
            "accumulator_memory={}",
            self.memory.accumulator.as_text()
        )
        .expect("writing to String cannot fail");
        writeln!(text, "workspace_bytes={}", self.memory.workspace_bytes)
            .expect("writing to String cannot fail");
        writeln!(text, "alignment_bytes={}", self.memory.alignment_bytes)
            .expect("writing to String cannot fail");
        writeln!(
            text,
            "kv_page_rows={}",
            self.memory
                .kv_page_rows
                .map_or_else(|| "none".into(), |rows| rows.to_string())
        )
        .expect("writing to String cannot fail");
        text
    }

    /// Stable fingerprint over the complete canonical implementation plan.
    #[must_use]
    pub fn fingerprint(&self) -> ImplementationFingerprint {
        ImplementationFingerprint::of_bytes(self.to_canonical_text().as_bytes())
    }

    /// Decode strict canonical text.
    ///
    /// The decoder rejects unknown, duplicate, missing, reordered, or
    /// non-canonical fields by reconstructing the plan and requiring a byte
    /// identical canonical re-encoding.
    ///
    /// # Errors
    ///
    /// Returns an error for malformed text, invalid identities, or invalid
    /// schedule/memory metadata.
    pub fn from_canonical_text(text: &str) -> Result<Self, ImplementationError> {
        if text.len() > MAX_CANONICAL_TEXT_BYTES {
            return Err(ImplementationError::ExceedsLimit {
                field: "canonical_text_bytes",
                value: u64::try_from(text.len()).unwrap_or(u64::MAX),
                maximum: u64::try_from(MAX_CANONICAL_TEXT_BYTES).unwrap_or(u64::MAX),
            });
        }
        let mut lines = text.lines();
        let header = lines
            .next()
            .ok_or_else(|| ImplementationError::MalformedCanonical("missing header".into()))?;
        let Some(raw_version) = header.strip_prefix("ADA-IMPLEMENTATION-V") else {
            return Err(ImplementationError::MalformedCanonical(
                "invalid header".into(),
            ));
        };
        let version = raw_version
            .parse::<u16>()
            .map_err(|_| ImplementationError::MalformedCanonical("invalid version".into()))?;
        if version != IMPLEMENTATION_IR_VERSION {
            return Err(ImplementationError::UnsupportedVersion(version));
        }

        let mut fields = BTreeMap::new();
        for line in lines {
            let (key, value) = line.split_once('=').ok_or_else(|| {
                ImplementationError::MalformedCanonical("invalid field line".into())
            })?;
            if key.is_empty() || value.is_empty() || fields.insert(key, value).is_some() {
                return Err(ImplementationError::MalformedCanonical(
                    "empty or duplicate field".into(),
                ));
            }
        }

        const FIELD_NAMES: [&str; 23] = [
            "semantic_family",
            "semantic_name",
            "semantic_revision",
            "implementation_name",
            "implementation_revision",
            "algorithm",
            "tile_queries",
            "tile_keys",
            "tile_values",
            "partition",
            "reduction",
            "exp_strategy",
            "pipeline_stages",
            "vector_width",
            "buffering",
            "query_memory",
            "key_memory",
            "value_memory",
            "output_memory",
            "accumulator_memory",
            "workspace_bytes",
            "alignment_bytes",
            "kv_page_rows",
        ];
        if fields.len() != FIELD_NAMES.len() || fields.keys().any(|key| !FIELD_NAMES.contains(key))
        {
            return Err(ImplementationError::MalformedCanonical(
                "unknown or missing field".into(),
            ));
        }

        let get = |name: &'static str| -> Result<&str, ImplementationError> {
            fields
                .get(name)
                .copied()
                .ok_or_else(|| ImplementationError::MalformedCanonical(format!("missing {name}")))
        };

        let semantic = SemanticId::new(
            semantic_family_from_text(get("semantic_family")?)?,
            get("semantic_name")?,
            parse_u32(get("semantic_revision")?, "semantic_revision")?,
        )
        .map_err(|_| ImplementationError::MalformedCanonical("invalid semantic identity".into()))?;
        let id = ImplementationCandidateId::new(
            semantic,
            get("implementation_name")?,
            parse_u32(get("implementation_revision")?, "implementation_revision")?,
        )
        .map_err(|_| {
            ImplementationError::MalformedCanonical("invalid implementation identity".into())
        })?;

        let kv_page_rows = match get("kv_page_rows")? {
            "none" => None,
            raw => Some(parse_u32(raw, "memory.kv_page_rows")?),
        };
        let plan = Self::new(
            id,
            AlgorithmPlan::from_text(get("algorithm")?)?,
            SchedulePlan {
                tile: TileShape {
                    queries: parse_u32(get("tile_queries")?, "schedule.tile.queries")?,
                    keys: parse_u32(get("tile_keys")?, "schedule.tile.keys")?,
                    values: parse_u32(get("tile_values")?, "schedule.tile.values")?,
                },
                partition: WorkPartition::from_text(get("partition")?)?,
                reduction: ReductionTopology::from_text(get("reduction")?)?,
                exp_strategy: ExpStrategy::from_text(get("exp_strategy")?)?,
                pipeline_stages: parse_u8(get("pipeline_stages")?, "schedule.pipeline_stages")?,
                vector_width: parse_u16(get("vector_width")?, "schedule.vector_width")?,
                buffering: Buffering::from_text(get("buffering")?)?,
            },
            MemoryPlan {
                query: MemoryLevel::from_text(get("query_memory")?)?,
                key: MemoryLevel::from_text(get("key_memory")?)?,
                value: MemoryLevel::from_text(get("value_memory")?)?,
                output: MemoryLevel::from_text(get("output_memory")?)?,
                accumulator: MemoryLevel::from_text(get("accumulator_memory")?)?,
                workspace_bytes: parse_u64(get("workspace_bytes")?, "memory.workspace_bytes")?,
                alignment_bytes: parse_u32(get("alignment_bytes")?, "memory.alignment_bytes")?,
                kv_page_rows,
            },
        )?;
        if plan.to_canonical_text() != text {
            return Err(ImplementationError::MalformedCanonical(
                "text is valid but non-canonical".into(),
            ));
        }
        Ok(plan)
    }
}

fn validate_nonzero_bounded(
    field: &'static str,
    value: u32,
    maximum: u32,
) -> Result<(), ImplementationError> {
    if value == 0 {
        return Err(ImplementationError::InvalidField(field));
    }
    if value > maximum {
        return Err(ImplementationError::ExceedsLimit {
            field,
            value: u64::from(value),
            maximum: u64::from(maximum),
        });
    }
    Ok(())
}

fn parse_u8(value: &str, field: &'static str) -> Result<u8, ImplementationError> {
    value
        .parse::<u8>()
        .map_err(|_| ImplementationError::MalformedCanonical(format!("invalid {field}")))
}

fn parse_u16(value: &str, field: &'static str) -> Result<u16, ImplementationError> {
    value
        .parse::<u16>()
        .map_err(|_| ImplementationError::MalformedCanonical(format!("invalid {field}")))
}

fn parse_u32(value: &str, field: &'static str) -> Result<u32, ImplementationError> {
    value
        .parse::<u32>()
        .map_err(|_| ImplementationError::MalformedCanonical(format!("invalid {field}")))
}

fn parse_u64(value: &str, field: &'static str) -> Result<u64, ImplementationError> {
    value
        .parse::<u64>()
        .map_err(|_| ImplementationError::MalformedCanonical(format!("invalid {field}")))
}

const fn semantic_family_text(family: SemanticFamily) -> &'static str {
    match family {
        SemanticFamily::StandardSoftmax => "standard-softmax",
        SemanticFamily::DifferentialSigned => "differential-signed",
        SemanticFamily::ToeplitzStructured => "toeplitz-structured",
        SemanticFamily::ProlateConcentration => "prolate-concentration",
        SemanticFamily::GroundStateGreen => "ground-state-green",
        SemanticFamily::SpectralFlow => "spectral-flow",
        SemanticFamily::RecurrentMemory => "recurrent-memory",
        SemanticFamily::Hybrid => "hybrid",
        SemanticFamily::Experimental => "experimental",
    }
}

fn semantic_family_from_text(value: &str) -> Result<SemanticFamily, ImplementationError> {
    match value {
        "standard-softmax" => Ok(SemanticFamily::StandardSoftmax),
        "differential-signed" => Ok(SemanticFamily::DifferentialSigned),
        "toeplitz-structured" => Ok(SemanticFamily::ToeplitzStructured),
        "prolate-concentration" => Ok(SemanticFamily::ProlateConcentration),
        "ground-state-green" => Ok(SemanticFamily::GroundStateGreen),
        "spectral-flow" => Ok(SemanticFamily::SpectralFlow),
        "recurrent-memory" => Ok(SemanticFamily::RecurrentMemory),
        "hybrid" => Ok(SemanticFamily::Hybrid),
        "experimental" => Ok(SemanticFamily::Experimental),
        _ => Err(ImplementationError::MalformedCanonical(
            "unknown semantic family".into(),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn semantic() -> SemanticId {
        SemanticId::new(SemanticFamily::StandardSoftmax, "standard-softmax", 1).unwrap()
    }

    fn base_schedule() -> SchedulePlan {
        SchedulePlan {
            tile: TileShape {
                queries: 64,
                keys: 128,
                values: 64,
            },
            partition: WorkPartition::QueryTiles,
            reduction: ReductionTopology::Tree,
            exp_strategy: ExpStrategy::ConditionalRescale,
            pipeline_stages: 3,
            vector_width: 8,
            buffering: Buffering::Double,
        }
    }

    fn base_memory() -> MemoryPlan {
        MemoryPlan {
            query: MemoryLevel::Shared,
            key: MemoryLevel::Shared,
            value: MemoryLevel::Shared,
            output: MemoryLevel::Global,
            accumulator: MemoryLevel::Register,
            workspace_bytes: 16_384,
            alignment_bytes: 128,
            kv_page_rows: None,
        }
    }

    #[test]
    fn two_schedules_keep_exactly_one_semantic_identity() {
        let first_id = ImplementationCandidateId::new(semantic(), "blocked-a", 1).unwrap();
        let second_id = ImplementationCandidateId::new(semantic(), "blocked-b", 1).unwrap();
        let first = ImplementationPlan::new(
            first_id,
            AlgorithmPlan::DenseBlocked,
            base_schedule(),
            base_memory(),
        )
        .unwrap();

        let mut second_schedule = base_schedule();
        second_schedule.tile.keys = 256;
        second_schedule.partition = WorkPartition::SplitKv { partitions: 4 };
        second_schedule.reduction = ReductionTopology::TwoPass;
        let second = ImplementationPlan::new(
            second_id,
            AlgorithmPlan::DenseBlocked,
            second_schedule,
            base_memory(),
        )
        .unwrap();

        assert_eq!(first.id().semantic(), second.id().semantic());
        assert_ne!(first.id(), second.id());
        assert_ne!(first.fingerprint(), second.fingerprint());
    }

    #[test]
    fn canonical_round_trip_is_byte_exact_and_fingerprint_stable() {
        let id = ImplementationCandidateId::new(semantic(), "portable-reference", 3).unwrap();
        let mut memory = base_memory();
        memory.kv_page_rows = Some(128);
        let plan =
            ImplementationPlan::new(id, AlgorithmPlan::PagedBlocked, base_schedule(), memory)
                .unwrap();
        let text = plan.to_canonical_text();
        let decoded = ImplementationPlan::from_canonical_text(&text).unwrap();
        assert_eq!(decoded, plan);
        assert_eq!(decoded.to_canonical_text(), text);
        assert_eq!(decoded.fingerprint(), plan.fingerprint());
    }

    #[test]
    fn invalid_schedule_and_memory_fail_closed() {
        let mut invalid = base_schedule();
        invalid.vector_width = 3;
        assert_eq!(
            invalid.validate(),
            Err(ImplementationError::InvalidField("schedule.vector_width"))
        );

        let mut invalid = base_schedule();
        invalid.tile.queries = 0;
        assert_eq!(
            invalid.validate(),
            Err(ImplementationError::InvalidField("schedule.tile.queries"))
        );

        let mut invalid_memory = base_memory();
        invalid_memory.alignment_bytes = 24;
        assert_eq!(
            invalid_memory.validate(),
            Err(ImplementationError::InvalidField("memory.alignment_bytes"))
        );
    }

    #[test]
    fn malformed_or_noncanonical_text_is_rejected() {
        let id = ImplementationCandidateId::new(semantic(), "canonical", 1).unwrap();
        let plan =
            ImplementationPlan::new(id, AlgorithmPlan::TwoPass, base_schedule(), base_memory())
                .unwrap();
        let canonical = plan.to_canonical_text();
        let reordered = canonical.replace(
            "tile_queries=64\ntile_keys=128",
            "tile_keys=128\ntile_queries=64",
        );
        assert!(ImplementationPlan::from_canonical_text(&reordered).is_err());

        let duplicate = canonical.replace(
            "workspace_bytes=16384\n",
            "workspace_bytes=16384\nworkspace_bytes=16384\n",
        );
        assert!(ImplementationPlan::from_canonical_text(&duplicate).is_err());
    }

    #[test]
    fn representation_contains_no_device_or_evidence_identity() {
        let id = ImplementationCandidateId::new(semantic(), "backend-neutral", 1).unwrap();
        let plan = ImplementationPlan::new(
            id,
            AlgorithmPlan::OnlineStreaming,
            base_schedule(),
            base_memory(),
        )
        .unwrap();
        let text = plan.to_canonical_text();
        assert!(!text.contains("device"));
        assert!(!text.contains("latency"));
        assert!(!text.contains("benchmark"));
        assert!(!text.contains("evidence"));
    }
}
