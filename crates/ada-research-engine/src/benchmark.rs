//! Reserved boundary for physical benchmark evidence.
//!
//! Algorithm discovery and hardware qualification are different activities.
//! This module declares — but does not implement — the future adapter through
//! which externally produced, reproducible benchmark artifacts could be
//! attached to a candidate. The discovery engine itself:
//!
//! * never executes shell commands, timers or external processes;
//! * never uses wall-clock measurements as search fitness;
//! * cannot construct a
//!   [`SurvivalClass::BenchmarkEvidenceAvailable`](crate::gates::SurvivalClass::BenchmarkEvidenceAvailable)
//!   value from its own evaluation paths.
//!
//! A future `BenchmarkProvider` implementation would live in a separate
//! reproducible ADA experiment crate and would hand over evidence artifacts
//! by digest only.

use serde::{Deserialize, Serialize};

/// A request for externally produced benchmark evidence.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BenchmarkRequest {
    /// Candidate the evidence would attach to.
    pub candidate_id: String,
    /// Named physical target identifier (e.g. a machine + protocol label).
    pub target: String,
    /// Identifier of the reproducible benchmark protocol.
    pub protocol_id: String,
}

/// Externally produced evidence artifact. The engine treats this as an
/// opaque, pre-qualified artifact: it never generates one and never derives
/// search fitness from one.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BenchmarkEvidence {
    pub artifact_digest: String,
    pub protocol_id: String,
    pub target: String,
}

/// Future boundary trait. Intentionally without implementations in this
/// crate: physical qualification is a separate reproducible ADA experiment.
pub trait BenchmarkProvider {
    /// Fetch previously produced evidence for a candidate, if any exists.
    fn fetch(&self, request: &BenchmarkRequest) -> Option<BenchmarkEvidence>;
}

#[cfg(test)]
mod tests {
    use super::*;

    struct NoProvider;

    impl BenchmarkProvider for NoProvider {
        fn fetch(&self, _request: &BenchmarkRequest) -> Option<BenchmarkEvidence> {
            None
        }
    }

    #[test]
    fn provider_boundary_exists_but_yields_nothing_in_e0() {
        let provider = NoProvider;
        let request = BenchmarkRequest {
            candidate_id: "deadbeef".into(),
            target: "thor".into(),
            protocol_id: "a2-e2-v-late".into(),
        };
        assert!(provider.fetch(&request).is_none());
    }
}
