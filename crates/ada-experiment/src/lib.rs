//! Versioned, reproducible experiment records and deterministic indexing for ADA.
//!
//! `ADA-EXPERIMENT-V2` binds an exact semantic program, workload, implementation
//! plan, objective vector, producer provenance, and explicit evidence references.
//! `ADA-EXPERIMENT-INDEX-V1` provides a bounded deterministic interchange index.

#![forbid(unsafe_code)]

mod codec;
mod index;
mod model;

pub use index::{EXPERIMENT_INDEX_HEADER, EXPERIMENT_INDEX_VERSION, ExperimentIndex};
pub use model::{
    EXPERIMENT_HEADER, EXPERIMENT_VERSION, EvidenceBinding, ExperimentError,
    ExperimentFingerprint, ExperimentRecord, ExperimentSpec, ProducerProvenance,
};

#[cfg(test)]
mod tests;
