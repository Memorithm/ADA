//! Bundled candidate proposers.
//!
//! All proposers are deterministic given construction parameters and the
//! engine's feedback sequence. None of them can bypass validation gates:
//! they emit [`crate::proposer::RawProposal`] values that the engine
//! normalizes, validates and falsifies like any other input.

pub mod enumerative;
pub mod evolutionary;
pub mod manual;

pub use enumerative::{EnumerativeConfig, EnumerativeProposer};
pub use evolutionary::{EvolutionaryConfig, EvolutionaryProposer};
pub use manual::{ManualCandidate, ManualProposer};
