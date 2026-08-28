//! Backend-neutral implementation, schedule, and memory IR for ADA research.
//!
//! This crate represents **how** one attention semantic may be realized without
//! changing **what** that semantic computes. Device identity, benchmark values,
//! and evidence deliberately do not belong to this representation.

#![forbid(unsafe_code)]

use ada_core::{ImplementationCandidateId, SemanticFamily, SemanticId};
use std::fmt::{Display, Formatter};

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
/// Maximum declared logical vector width.
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
                write!(formatter, "malformed implementation artifact: {reason}")
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
    /// Stateful/chunked realization for a recurrent semantic.
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

    fn parse(value: &str) -> Result<Self, ImplementationError> {
        match value {
            "dense-blocked" => Ok(Self::DenseBlocked),
            "online-streaming" => Ok(Self::OnlineStreaming),
            "two-pass" => Ok(Self::TwoPass),
            "paged-blocked" => Ok(Self::PagedBlocked),
            "recurrent-chunked" => Ok(Self::RecurrentChunked),
            _ => malformed("unknown algorithm plan"),
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
    /// KV work is split and later reduced.
    SplitKv {
        /// Number of logical split-KV partitions.
        partitions: u16,
    },
}

impl WorkPartition {
    fn validate(self) -> Result<(), ImplementationError> {
        if let Self::SplitKv { partitions } = self {
            validate_nonzero_bounded(
                "schedule.partition.partitions",
                u64::from(partitions),
                u64::from(MAX_SPLIT_KV_PARTITIONS),
            )?;
        }
        Ok(())
    }

    fn as_text(self) -> String {
        match self {
            Self::Serial => "serial".into(),
            Self::QueryTiles => "query-tiles".into(),
            Self::SplitKv { partitions } => format!("split-kv:{partitions}"),
        }
    }

    fn parse(value: &str) -> Result<Self, ImplementationError> {
        match value {
            "serial" => Ok(Self::Serial),
            "query-tiles" => Ok(Self::QueryTiles),
            _ => {
                let raw = value
                    .strip_prefix("split-kv:")
                    .ok_or_else(|| malformed_error("unknown work partition"))?;
                Ok(Self::SplitKv {
                    partitions: parse_number(raw, "schedule.partition.partitions")?,
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
    /// Store partials and finalize in a second pass.
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

    fn parse(value: &str) -> Result<Self, ImplementationError> {
        match value {
            "serial" => Ok(Self::Serial),
            "tree" => Ok(Self::Tree),
            "two-pass" => Ok(Self::TwoPass),
            _ => malformed("unknown reduction topology"),
        }
    }
}

/// Exponential/rescaling realization strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ExpStrategy {
    /// Ordinary backend exponential evaluation.
    Standard,
    /// Rescale only when the running reference changes.
    ConditionalRescale,
    /// Defer exponential/rescale work to a later implementation phase.
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

    fn parse(value: &str) -> Result<Self, ImplementationError> {
        match value {
            "standard" => Ok(Self::Standard),
            "conditional-rescale" => Ok(Self::ConditionalRescale),
            "deferred" => Ok(Self::Deferred),
            _ => malformed("unknown exp strategy"),
        }
    }
}

/// Logical buffering policy, not a physical hardware claim.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Buffering {
    /// One logical staging buffer.
    Single,
    /// Two logical staging buffers.
    Double,
}

impl Buffering {
    const fn as_text(self) -> &'static str {
        match self {
            Self::Single => "single",
            Self::Double => "double",
        }
    }

    fn parse(value: &str) -> Result<Self, ImplementationError> {
        match value {
            "single" => Ok(Self::Single),
            "double" => Ok(Self::Double),
            _ => malformed("unknown buffering policy"),
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
        validate_nonzero_bounded(
            "schedule.tile.queries",
            u64::from(self.queries),
            u64::from(MAX_TILE_EXTENT),
        )?;
        validate_nonzero_bounded(
            "schedule.tile.keys",
            u64::from(self.keys),
            u64::from(MAX_TILE_EXTENT),
        )?;
        validate_nonzero_bounded(
            "schedule.tile.values",
            u64::from(self.values),
            u64::from(MAX_TILE_EXTENT),
        )
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
    /// Exponential/rescaling strategy.
    pub exp_strategy: ExpStrategy,
    /// Number of logical software-pipeline stages.
    pub pipeline_stages: u8,
    /// Logical vector width for a later backend lowering.
    pub vector_width: u16,
    /// Logical buffering policy.
    pub buffering: Buffering,
}

impl SchedulePlan {
    /// Validate all bounded schedule fields.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid tiles, split counts, pipeline stages, or
    /// vector widths.
    pub fn validate(self) -> Result<(), ImplementationError> {
        self.tile.validate()?;
        self.partition.validate()?;
        validate_nonzero_bounded(
            "schedule.pipeline_stages",
            u64::from(self.pipeline_stages),
            u64::from(MAX_PIPELINE_STAGES),
        )?;
        validate_nonzero_bounded(
            "schedule.vector_width",
            u64::from(self.vector_width),
            u64::from(MAX_VECTOR_WIDTH),
        )?;
        if !self.vector_width.is_power_of_two() {
            return Err(ImplementationError::InvalidField("schedule.vector_width"));
        }
        Ok(())
    }
}

/// Abstract memory/storage level for an implementation plan.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MemoryLevel {
    /// Main device/process-visible memory.
    Global,
    /// Workgroup/CTA-style shared scratch storage.
    Shared,
    /// Worker/thread-local storage.
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

    fn parse(value: &str) -> Result<Self, ImplementationError> {
        match value {
            "global" => Ok(Self::Global),
            "shared" => Ok(Self::Shared),
            "local" => Ok(Self::Local),
            "register" => Ok(Self::Register),
            _ => malformed("unknown memory level"),
        }
    }
}

/// Storage and traversal choices separated from semantic identity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MemoryPlan {
    /// Query storage level while consumed.
    pub query: MemoryLevel,
    /// Key storage level while consumed.
    pub key: MemoryLevel,
    /// Value storage level while consumed.
    pub value: MemoryLevel,
    /// Output storage level while produced.
    pub output: MemoryLevel,
    /// Accumulator storage level.
    pub accumulator: MemoryLevel,
    /// Declared implementation workspace bytes.
    pub workspace_bytes: u64,
    /// Requested alignment for implementation-owned buffers.
    pub alignment_bytes: u32,
    /// Optional KV page size in rows.
    pub kv_page_rows: Option<u32>,
}

impl MemoryPlan {
    /// Validate bounded memory-plan metadata.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid alignment or page geometry.
    pub fn validate(self) -> Result<(), ImplementationError> {
        validate_nonzero_bounded(
            "memory.alignment_bytes",
            u64::from(self.alignment_bytes),
            u64::from(MAX_ALIGNMENT_BYTES),
        )?;
        if !self.alignment_bytes.is_power_of_two() {
            return Err(ImplementationError::InvalidField("memory.alignment_bytes"));
        }
        if let Some(rows) = self.kv_page_rows {
            validate_nonzero_bounded(
                "memory.kv_page_rows",
                u64::from(rows),
                u64::from(MAX_KV_PAGE_ROWS),
            )?;
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
        const FNV_OFFSET: u64 = 0xcbf_29ce4_8422_2325;
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
    /// Construct a validated implementation plan.
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
        format!(
            "ADA-IMPLEMENTATION-V{IMPLEMENTATION_IR_VERSION}\n\
semantic_family={}\n\
semantic_name={}\n\
semantic_revision={}\n\
implementation_name={}\n\
implementation_revision={}\n\
algorithm={}\n\
tile_queries={}\n\
tile_keys={}\n\
tile_values={}\n\
partition={}\n\
reduction={}\n\
exp_strategy={}\n\
pipeline_stages={}\n\
vector_width={}\n\
buffering={}\n\
query_memory={}\n\
key_memory={}\n\
value_memory={}\n\
output_memory={}\n\
accumulator_memory={}\n\
workspace_bytes={}\n\
alignment_bytes={}\n\
kv_page_rows={}\n",
            semantic_family_text(semantic.family()),
            semantic.name(),
            semantic.revision(),
            self.id.name(),
            self.id.revision(),
            self.algorithm.as_text(),
            self.schedule.tile.queries,
            self.schedule.tile.keys,
            self.schedule.tile.values,
            self.schedule.partition.as_text(),
            self.schedule.reduction.as_text(),
            self.schedule.exp_strategy.as_text(),
            self.schedule.pipeline_stages,
            self.schedule.vector_width,
            self.schedule.buffering.as_text(),
            self.memory.query.as_text(),
            self.memory.key.as_text(),
            self.memory.value.as_text(),
            self.memory.output.as_text(),
            self.memory.accumulator.as_text(),
            self.memory.workspace_bytes,
            self.memory.alignment_bytes,
            self.memory
                .kv_page_rows
                .map_or_else(|| "none".into(), |rows| rows.to_string()),
        )
    }

    /// Stable fingerprint over the complete canonical implementation plan.
    #[must_use]
    pub fn fingerprint(&self) -> ImplementationFingerprint {
        ImplementationFingerprint::of_bytes(self.to_canonical_text().as_bytes())
    }

    /// Decode strict canonical text.
    ///
    /// # Errors
    ///
    /// Returns an error for malformed text, invalid identities, unsupported
    /// versions, or invalid schedule/memory metadata.
    pub fn from_canonical_text(text: &str) -> Result<Self, ImplementationError> {
        if text.len() > MAX_CANONICAL_TEXT_BYTES {
            return Err(ImplementationError::ExceedsLimit {
                field: "canonical_text_bytes",
                value: u64::try_from(text.len()).unwrap_or(u64::MAX),
                maximum: u64::try_from(MAX_CANONICAL_TEXT_BYTES).unwrap_or(u64::MAX),
            });
        }
        let mut lines = text.lines();
        parse_header(&mut lines)?;
        let semantic = parse_semantic_identity(&mut lines)?;
        let id = parse_implementation_identity(semantic, &mut lines)?;
        let algorithm = AlgorithmPlan::parse(next_value(&mut lines, "algorithm")?)?;
        let schedule = parse_schedule(&mut lines)?;
        let memory = parse_memory(&mut lines)?;
        if lines.next().is_some() {
            return malformed("unexpected trailing field");
        }
        let plan = Self::new(id, algorithm, schedule, memory)?;
        if plan.to_canonical_text() != text {
            return malformed("artifact is not canonical");
        }
        Ok(plan)
    }
}

fn parse_header(lines: &mut std::str::Lines<'_>) -> Result<(), ImplementationError> {
    let header = lines
        .next()
        .ok_or_else(|| malformed_error("missing header"))?;
    let raw = header
        .strip_prefix("ADA-IMPLEMENTATION-V")
        .ok_or_else(|| malformed_error("invalid header"))?;
    let version = parse_number::<u16>(raw, "version")?;
    if version != IMPLEMENTATION_IR_VERSION {
        return Err(ImplementationError::UnsupportedVersion(version));
    }
    Ok(())
}

fn parse_semantic_identity(
    lines: &mut std::str::Lines<'_>,
) -> Result<SemanticId, ImplementationError> {
    let family = semantic_family_parse(next_value(lines, "semantic_family")?)?;
    let name = next_value(lines, "semantic_name")?;
    let revision = parse_number(next_value(lines, "semantic_revision")?, "semantic_revision")?;
    SemanticId::new(family, name, revision)
        .map_err(|_| malformed_error("invalid semantic identity"))
}

fn parse_implementation_identity(
    semantic: SemanticId,
    lines: &mut std::str::Lines<'_>,
) -> Result<ImplementationCandidateId, ImplementationError> {
    let name = next_value(lines, "implementation_name")?;
    let revision = parse_number(
        next_value(lines, "implementation_revision")?,
        "implementation_revision",
    )?;
    ImplementationCandidateId::new(semantic, name, revision)
        .map_err(|_| malformed_error("invalid implementation identity"))
}

fn parse_schedule(lines: &mut std::str::Lines<'_>) -> Result<SchedulePlan, ImplementationError> {
    let schedule = SchedulePlan {
        tile: TileShape {
            queries: parse_number(next_value(lines, "tile_queries")?, "tile_queries")?,
            keys: parse_number(next_value(lines, "tile_keys")?, "tile_keys")?,
            values: parse_number(next_value(lines, "tile_values")?, "tile_values")?,
        },
        partition: WorkPartition::parse(next_value(lines, "partition")?)?,
        reduction: ReductionTopology::parse(next_value(lines, "reduction")?)?,
        exp_strategy: ExpStrategy::parse(next_value(lines, "exp_strategy")?)?,
        pipeline_stages: parse_number(
            next_value(lines, "pipeline_stages")?,
            "pipeline_stages",
        )?,
        vector_width: parse_number(next_value(lines, "vector_width")?, "vector_width")?,
        buffering: Buffering::parse(next_value(lines, "buffering")?)?,
    };
    schedule.validate()?;
    Ok(schedule)
}

fn parse_memory(lines: &mut std::str::Lines<'_>) -> Result<MemoryPlan, ImplementationError> {
    let page_text = next_value(lines, "kv_page_rows")?;
    let memory = MemoryPlan {
        query: MemoryLevel::parse(next_value(lines, "query_memory")?)?,
        key: MemoryLevel::parse(next_value(lines, "key_memory")?)?,
        value: MemoryLevel::parse(next_value(lines, "value_memory")?)?,
        output: MemoryLevel::parse(next_value(lines, "output_memory")?)?,
        accumulator: MemoryLevel::parse(next_value(lines, "accumulator_memory")?)?,
        workspace_bytes: parse_number(next_value(lines, "workspace_bytes")?, "workspace_bytes")?,
        alignment_bytes: parse_number(next_value(lines, "alignment_bytes")?, "alignment_bytes")?,
        kv_page_rows: if page_text == "none" {
            None
        } else {
            Some(parse_number(page_text, "kv_page_rows")?)
        },
    };
    memory.validate()?;
    Ok(memory)
}

fn next_value<'a>(
    lines: &mut std::str::Lines<'a>,
    expected: &'static str,
) -> Result<&'a str, ImplementationError> {
    let line = lines
        .next()
        .ok_or_else(|| malformed_error(&format!("missing {expected}")))?;
    let (key, value) = line
        .split_once('=')
        .ok_or_else(|| malformed_error("invalid field line"))?;
    if key != expected || value.is_empty() {
        return malformed(&format!("expected {expected}"));
    }
    Ok(value)
}

fn validate_nonzero_bounded(
    field: &'static str,
    value: u64,
    maximum: u64,
) -> Result<(), ImplementationError> {
    if value == 0 {
        return Err(ImplementationError::InvalidField(field));
    }
    if value > maximum {
        return Err(ImplementationError::ExceedsLimit {
            field,
            value,
            maximum,
        });
    }
    Ok(())
}

fn parse_number<T>(value: &str, field: &'static str) -> Result<T, ImplementationError>
where
    T: std::str::FromStr,
{
    value
        .parse::<T>()
        .map_err(|_| malformed_error(&format!("invalid {field}")))
}

fn malformed<T>(reason: &str) -> Result<T, ImplementationError> {
    Err(malformed_error(reason))
}

fn malformed_error(reason: &str) -> ImplementationError {
    ImplementationError::MalformedCanonical(reason.into())
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

fn semantic_family_parse(value: &str) -> Result<SemanticFamily, ImplementationError> {
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
        _ => malformed("unknown semantic family"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn semantic() -> SemanticId {
        SemanticId::new(SemanticFamily::StandardSoftmax, "standard-softmax", 1).unwrap()
    }

    fn schedule() -> SchedulePlan {
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

    fn memory() -> MemoryPlan {
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
    fn schedules_do_not_redefine_semantics() {
        let first_id = ImplementationCandidateId::new(semantic(), "blocked-a", 1).unwrap();
        let second_id = ImplementationCandidateId::new(semantic(), "blocked-b", 1).unwrap();
        let first =
            ImplementationPlan::new(first_id, AlgorithmPlan::DenseBlocked, schedule(), memory())
                .unwrap();
        let mut second_schedule = schedule();
        second_schedule.tile.keys = 256;
        second_schedule.partition = WorkPartition::SplitKv { partitions: 4 };
        second_schedule.reduction = ReductionTopology::TwoPass;
        let second = ImplementationPlan::new(
            second_id,
            AlgorithmPlan::DenseBlocked,
            second_schedule,
            memory(),
        )
        .unwrap();
        assert_eq!(first.id().semantic(), second.id().semantic());
        assert_ne!(first.id(), second.id());
        assert_ne!(first.fingerprint(), second.fingerprint());
    }

    #[test]
    fn canonical_round_trip_is_exact() {
        let id = ImplementationCandidateId::new(semantic(), "portable-reference", 3).unwrap();
        let mut paged_memory = memory();
        paged_memory.kv_page_rows = Some(128);
        let plan = ImplementationPlan::new(
            id,
            AlgorithmPlan::PagedBlocked,
            schedule(),
            paged_memory,
        )
        .unwrap();
        let text = plan.to_canonical_text();
        let decoded = ImplementationPlan::from_canonical_text(&text).unwrap();
        assert_eq!(decoded, plan);
        assert_eq!(decoded.to_canonical_text(), text);
        assert_eq!(decoded.fingerprint(), plan.fingerprint());
    }

    #[test]
    fn invalid_metadata_fails_closed() {
        let mut bad_schedule = schedule();
        bad_schedule.vector_width = 3;
        assert!(bad_schedule.validate().is_err());
        bad_schedule = schedule();
        bad_schedule.tile.queries = 0;
        assert!(bad_schedule.validate().is_err());
        let mut bad_memory = memory();
        bad_memory.alignment_bytes = 24;
        assert!(bad_memory.validate().is_err());
    }

    #[test]
    fn noncanonical_artifacts_are_rejected() {
        let id = ImplementationCandidateId::new(semantic(), "canonical", 1).unwrap();
        let plan = ImplementationPlan::new(id, AlgorithmPlan::TwoPass, schedule(), memory()).unwrap();
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
    fn representation_contains_no_measurement_identity() {
        let id = ImplementationCandidateId::new(semantic(), "backend-neutral", 1).unwrap();
        let plan = ImplementationPlan::new(
            id,
            AlgorithmPlan::OnlineStreaming,
            schedule(),
            memory(),
        )
        .unwrap();
        let text = plan.to_canonical_text();
        assert!(!text.contains("device"));
        assert!(!text.contains("latency"));
        assert!(!text.contains("benchmark"));
        assert!(!text.contains("evidence"));
    }
}
