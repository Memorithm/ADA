use crate::codec::{append_field, hex_decode, hex_encode, next_value, parse_u16, parse_usize};
use crate::model::{ExperimentError, ExperimentFingerprint, ExperimentRecord, malformed};
use ada_core::{ImplementationCandidateId, SemanticId};
use ada_workload::WorkloadFingerprint;
use std::collections::BTreeMap;

pub const EXPERIMENT_INDEX_VERSION: u16 = 1;
pub const EXPERIMENT_INDEX_HEADER: &str = "ADA-EXPERIMENT-INDEX-V1";
pub const MAX_INDEX_BYTES: usize = 64 << 20;
pub const MAX_INDEX_ENTRIES: usize = 65_536;

#[derive(Debug, Clone, Default)]
pub struct ExperimentIndex {
    records: BTreeMap<ExperimentFingerprint, ExperimentRecord>,
}

impl ExperimentIndex {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert a complete experiment under its deterministic fingerprint.
    ///
    /// # Errors
    ///
    /// Returns [`ExperimentError::IndexFull`] at the configured capacity,
    /// [`ExperimentError::DuplicateExperiment`] for an identical existing
    /// record, or [`ExperimentError::FingerprintCollision`] if different
    /// canonical records map to the same fingerprint.
    pub fn insert(
        &mut self,
        record: ExperimentRecord,
    ) -> Result<ExperimentFingerprint, ExperimentError> {
        if self.records.len() >= MAX_INDEX_ENTRIES {
            return Err(ExperimentError::IndexFull);
        }
        let fingerprint = record.fingerprint();
        if let Some(existing) = self.records.get(&fingerprint) {
            if existing.to_canonical_text() == record.to_canonical_text() {
                return Err(ExperimentError::DuplicateExperiment);
            }
            return Err(ExperimentError::FingerprintCollision);
        }
        self.records.insert(fingerprint, record);
        Ok(fingerprint)
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.records.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    #[must_use]
    pub fn get(&self, fingerprint: ExperimentFingerprint) -> Option<&ExperimentRecord> {
        self.records.get(&fingerprint)
    }

    #[must_use]
    pub fn records_for_semantic(&self, semantic: &SemanticId) -> Vec<&ExperimentRecord> {
        self.records
            .values()
            .filter(|record| record.semantic().descriptor().id() == semantic)
            .collect()
    }

    #[must_use]
    pub fn records_for_workload(&self, workload: WorkloadFingerprint) -> Vec<&ExperimentRecord> {
        self.records
            .values()
            .filter(|record| record.workload().fingerprint() == workload)
            .collect()
    }

    #[must_use]
    pub fn records_for_implementation(
        &self,
        implementation: &ImplementationCandidateId,
    ) -> Vec<&ExperimentRecord> {
        self.records
            .values()
            .filter(|record| record.implementation().id() == implementation)
            .collect()
    }

    #[must_use]
    pub fn records_with_measured_cost(&self) -> Vec<&ExperimentRecord> {
        self.records
            .values()
            .filter(|record| {
                let measured = record.objective().measured();
                measured.latency_ns.is_some() || measured.energy_nj.is_some()
            })
            .collect()
    }

    #[must_use]
    pub fn records_with_estimated_cost(&self) -> Vec<&ExperimentRecord> {
        self.records
            .values()
            .filter(|record| {
                let estimated = record.objective().estimated();
                estimated.bytes_moved.is_some()
                    || estimated.workspace_bytes.is_some()
                    || estimated.kv_cache_bytes.is_some()
                    || estimated.index_construction.is_some()
                    || estimated.communication_bytes.is_some()
                    || estimated.reduction_operations.is_some()
            })
            .collect()
    }

    #[must_use]
    pub fn to_canonical_text(&self) -> String {
        let mut text = String::from(EXPERIMENT_INDEX_HEADER);
        text.push('\n');
        append_field(&mut text, "version", EXPERIMENT_INDEX_VERSION);
        append_field(&mut text, "count", self.records.len());
        for (position, record) in self.records.values().enumerate() {
            append_field(
                &mut text,
                &format!("entry_{position}"),
                hex_encode(&record.to_canonical_text()),
            );
        }
        text
    }

    /// Decode and fully reconstruct a canonical experiment index.
    ///
    /// # Errors
    ///
    /// Returns [`ExperimentError`] for malformed/non-canonical text, unsupported
    /// versions, excessive entry counts, invalid nested experiment records,
    /// duplicate records, or fingerprint collisions.
    pub fn from_canonical_text(text: &str) -> Result<Self, ExperimentError> {
        if text.len() > MAX_INDEX_BYTES || text.contains('\r') || !text.ends_with('\n') {
            return malformed("index exceeds limit, contains CR, or lacks final newline");
        }
        let mut lines = text.lines();
        if lines.next() != Some(EXPERIMENT_INDEX_HEADER) {
            return malformed("invalid index header");
        }
        let version = parse_u16(next_value(&mut lines, "version")?)?;
        if version != EXPERIMENT_INDEX_VERSION {
            return Err(ExperimentError::UnsupportedVersion(version));
        }
        let count = parse_usize(next_value(&mut lines, "count")?)?;
        if count > MAX_INDEX_ENTRIES {
            return Err(ExperimentError::IndexFull);
        }
        let mut index = Self::new();
        for position in 0..count {
            let encoded = next_value(&mut lines, &format!("entry_{position}"))?;
            let record = ExperimentRecord::from_canonical_text(&hex_decode(encoded)?)?;
            index.insert(record)?;
        }
        if lines.next().is_some() {
            return malformed("unexpected trailing index field");
        }
        if index.to_canonical_text() != text {
            return malformed("index is not canonical");
        }
        Ok(index)
    }
}
