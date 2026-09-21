//! Bounded attention-specific adversarial fixture helpers.
//!
//! These helpers grow a caller-owned CEGIS fixture corpus with deterministic,
//! hard-capped attention probes (extreme finite scores, causal/mask edges,
//! all-equal rows, single-spike keys, and near-ties). They never mutate
//! qualification state, never claim usefulness or novelty, and fail closed on
//! non-finite or oversized probes.

use crate::{
    AdversarialGenerator, CegisError, Fixture, MAX_ADVERSARIAL_OUTPUTS, MAX_FIXTURE_ID_BYTES,
    MAX_FIXTURE_TEXT_BYTES,
};
use ada_semantic::{ReferenceInput, ReferenceInputSpec};
use std::collections::BTreeSet;
use std::fmt::{Display, Formatter, Write as _};

/// Schema version for attention-adversarial probe canonical text.
pub const ATTENTION_ADVERSARIAL_PROBE_VERSION: u16 = 1;
/// Hard cap on fixtures emitted by one attention-adversarial generation call.
pub const MAX_ATTENTION_ADVERSARIAL_FIXTURES: u64 = 32;
/// Hard cap on query rows inside one attention-adversarial probe.
pub const MAX_ATTENTION_ADVERSARIAL_QUERIES: usize = 8;
/// Hard cap on key rows inside one attention-adversarial probe.
pub const MAX_ATTENTION_ADVERSARIAL_KEYS: usize = 8;
/// Hard cap on Q/K or V dimensions inside one attention-adversarial probe.
pub const MAX_ATTENTION_ADVERSARIAL_DIM: usize = 4;
/// Finite magnitude used near the upper end of practical score scales.
pub const EXTREME_FINITE_SCORE_MAGNITUDE: f64 = 1.0e200;
/// Relative gap used by near-tie probes (`1 + eps` vs `1`).
pub const NEAR_TIE_RELATIVE_EPS: f64 = 1.0e-12;

/// Named families of bounded attention adversarial probes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum AttentionProbeKind {
    /// Finite Q/K magnitudes near extreme score scales.
    ExtremeFiniteScores,
    /// Causal-edge visibility via an explicit external mask.
    CausalMaskBoundary,
    /// Identical Q/K rows producing all-equal affinities.
    AllEqualScores,
    /// One dominating key against near-zero companions.
    SingleSpike,
    /// Near-tied affinities separated by a tiny relative epsilon.
    NearTies,
}

impl AttentionProbeKind {
    /// Stable lowercase token used in fixture identifiers and canonical text.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::ExtremeFiniteScores => "extreme_finite_scores",
            Self::CausalMaskBoundary => "causal_mask_boundary",
            Self::AllEqualScores => "all_equal_scores",
            Self::SingleSpike => "single_spike",
            Self::NearTies => "near_ties",
        }
    }

    /// Deterministic enumeration order for corpus growth.
    #[must_use]
    pub const fn all() -> [Self; 5] {
        [
            Self::ExtremeFiniteScores,
            Self::CausalMaskBoundary,
            Self::AllEqualScores,
            Self::SingleSpike,
            Self::NearTies,
        ]
    }
}

impl Display for AttentionProbeKind {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// Caller-owned bounds for attention-adversarial generation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AttentionAdversarialConfig {
    /// Maximum fixtures returned by one [`AttentionAdversarialGenerator::generate`] call.
    pub max_outputs: u64,
    /// Maximum query rows per probe.
    pub max_queries: usize,
    /// Maximum key rows per probe.
    pub max_keys: usize,
    /// Maximum Q/K or V dimension per probe.
    pub max_dim: usize,
}

impl AttentionAdversarialConfig {
    /// Construct and validate bounded attention-adversarial configuration.
    ///
    /// # Errors
    ///
    /// Returns an error when any bound exceeds the crate or attention helper limit.
    pub fn new(
        max_outputs: u64,
        max_queries: usize,
        max_keys: usize,
        max_dim: usize,
    ) -> Result<Self, CegisError> {
        if max_outputs == 0 || max_outputs > MAX_ATTENTION_ADVERSARIAL_FIXTURES {
            return Err(CegisError::ExceedsLimit {
                field: "attention_adversarial.max_outputs",
                value: max_outputs,
                maximum: MAX_ATTENTION_ADVERSARIAL_FIXTURES,
            });
        }
        if max_outputs > MAX_ADVERSARIAL_OUTPUTS {
            return Err(CegisError::ExceedsLimit {
                field: "attention_adversarial.max_outputs",
                value: max_outputs,
                maximum: MAX_ADVERSARIAL_OUTPUTS,
            });
        }
        if max_queries == 0 || max_queries > MAX_ATTENTION_ADVERSARIAL_QUERIES {
            return Err(CegisError::ExceedsLimit {
                field: "attention_adversarial.max_queries",
                value: u64::try_from(max_queries).unwrap_or(u64::MAX),
                maximum: u64::try_from(MAX_ATTENTION_ADVERSARIAL_QUERIES).unwrap_or(u64::MAX),
            });
        }
        if max_keys == 0 || max_keys > MAX_ATTENTION_ADVERSARIAL_KEYS {
            return Err(CegisError::ExceedsLimit {
                field: "attention_adversarial.max_keys",
                value: u64::try_from(max_keys).unwrap_or(u64::MAX),
                maximum: u64::try_from(MAX_ATTENTION_ADVERSARIAL_KEYS).unwrap_or(u64::MAX),
            });
        }
        if max_dim == 0 || max_dim > MAX_ATTENTION_ADVERSARIAL_DIM {
            return Err(CegisError::ExceedsLimit {
                field: "attention_adversarial.max_dim",
                value: u64::try_from(max_dim).unwrap_or(u64::MAX),
                maximum: u64::try_from(MAX_ATTENTION_ADVERSARIAL_DIM).unwrap_or(u64::MAX),
            });
        }
        Ok(Self {
            max_outputs,
            max_queries,
            max_keys,
            max_dim,
        })
    }
}

impl Default for AttentionAdversarialConfig {
    fn default() -> Self {
        Self {
            max_outputs: MAX_ATTENTION_ADVERSARIAL_FIXTURES,
            max_queries: MAX_ATTENTION_ADVERSARIAL_QUERIES,
            max_keys: MAX_ATTENTION_ADVERSARIAL_KEYS,
            max_dim: MAX_ATTENTION_ADVERSARIAL_DIM,
        }
    }
}

/// Deterministic, candidate-agnostic attention adversarial generator.
///
/// Probes are derived only from the seed and configuration. The candidate and
/// active corpus arguments are accepted to satisfy [`AdversarialGenerator`] but
/// are not used to invent unbounded or non-deterministic fixtures.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AttentionAdversarialGenerator {
    config: AttentionAdversarialConfig,
}

impl AttentionAdversarialGenerator {
    /// Construct a generator with validated configuration.
    ///
    /// # Errors
    ///
    /// Returns an error when configuration bounds are invalid.
    pub fn new(config: AttentionAdversarialConfig) -> Result<Self, CegisError> {
        let config = AttentionAdversarialConfig::new(
            config.max_outputs,
            config.max_queries,
            config.max_keys,
            config.max_dim,
        )?;
        Ok(Self { config })
    }

    /// Borrow the validated configuration.
    #[must_use]
    pub const fn config(&self) -> AttentionAdversarialConfig {
        self.config
    }

    /// Emit a bounded, deterministic probe suite for `seed`.
    ///
    /// # Errors
    ///
    /// Returns an error when a probe cannot be constructed under the configured
    /// bounds (fail closed; never emits non-finite fixtures).
    pub fn generate_probes(&self, seed: u64) -> Result<Vec<Fixture<ReferenceInput>>, CegisError> {
        let mut fixtures = Vec::new();
        let mut kinds = AttentionProbeKind::all().to_vec();
        // Stable shuffle driven only by seed so generation remains deterministic
        // without depending on candidate identity or ambient corpus state.
        seeded_shuffle(&mut kinds, seed);
        for (index, kind) in kinds.into_iter().enumerate() {
            if u64::try_from(fixtures.len()).unwrap_or(u64::MAX) >= self.config.max_outputs {
                break;
            }
            let geometry = geometry_for(seed, index, &self.config);
            fixtures.push(build_probe_fixture(kind, seed, index as u64, geometry)?);
        }
        Ok(fixtures)
    }
}

impl Default for AttentionAdversarialGenerator {
    fn default() -> Self {
        Self::new(AttentionAdversarialConfig::default()).expect("default attention config is valid")
    }
}

impl<C> AdversarialGenerator<C, ReferenceInput> for AttentionAdversarialGenerator {
    type Error = CegisError;

    fn generate(
        &mut self,
        seed: u64,
        _candidate: &C,
        _active: &[Fixture<ReferenceInput>],
    ) -> Result<Vec<Fixture<ReferenceInput>>, Self::Error> {
        self.generate_probes(seed)
    }
}

/// Attempt to build a non-finite probe and always fail closed.
///
/// Callers that need to prove refusal paths should use this helper rather than
/// patching NaNs into an already-constructed [`ReferenceInput`].
///
/// # Errors
///
/// Always returns [`CegisError::InvalidFixture`] because non-finite Q/K/V values
/// are rejected before a fixture is emitted.
pub fn refuse_nonfinite_attention_probe(
    kind: AttentionProbeKind,
    seed: u64,
    nonfinite: f64,
) -> Result<Fixture<ReferenceInput>, CegisError> {
    let _ = seed;
    if nonfinite.is_finite() {
        return Err(CegisError::InvalidFixture(
            "refuse_nonfinite_attention_probe requires a non-finite sentinel".into(),
        ));
    }
    let spec = ReferenceInputSpec {
        query_count: 1,
        key_count: 1,
        q_dimension: 1,
        value_dimension: 1,
        queries: vec![nonfinite],
        keys: vec![1.0],
        values: vec![1.0],
        external_mask: None,
    };
    match ReferenceInput::new(spec) {
        Ok(_) => Err(CegisError::InvalidFixture(format!(
            "non-finite {kind} probe was unexpectedly accepted"
        ))),
        Err(error) => Err(CegisError::InvalidFixture(format!(
            "non-finite {kind} probe refused: {error}"
        ))),
    }
}

/// Deterministically merge generated fixtures into a caller-owned active corpus.
///
/// Duplicates (same id + canonical text) are skipped. Insertion order of the
/// prior corpus is preserved; new fixtures are appended in sorted identity
/// order. This never mutates global qualification state.
///
/// # Errors
///
/// Returns an error when the merged corpus would exceed `max_active_fixtures`.
pub fn grow_attention_adversarial_corpus(
    active: &[Fixture<ReferenceInput>],
    generated: &[Fixture<ReferenceInput>],
    max_active_fixtures: u64,
) -> Result<Vec<Fixture<ReferenceInput>>, CegisError> {
    let mut merged = active.to_vec();
    let mut keys = merged
        .iter()
        .map(fixture_identity_key)
        .collect::<BTreeSet<_>>();
    let mut newcomers = generated
        .iter()
        .filter(|fixture| !keys.contains(&fixture_identity_key(fixture)))
        .cloned()
        .collect::<Vec<_>>();
    newcomers.sort_by_key(fixture_identity_key);
    for fixture in newcomers {
        let key = fixture_identity_key(&fixture);
        if !keys.insert(key) {
            continue;
        }
        let next = u64::try_from(merged.len()).unwrap_or(u64::MAX) + 1;
        if next > max_active_fixtures {
            return Err(CegisError::ExceedsLimit {
                field: "attention_adversarial.active_fixtures",
                value: next,
                maximum: max_active_fixtures,
            });
        }
        merged.push(fixture);
    }
    Ok(merged)
}

fn small_index_as_f64(index: usize) -> f64 {
    // Attention adversarial geometries are hard-capped well below the f64
    // mantissa width, so the conversion is exact for every legal probe index.
    debug_assert!(
        index
            <= MAX_ATTENTION_ADVERSARIAL_KEYS
                .max(MAX_ATTENTION_ADVERSARIAL_QUERIES)
                .max(MAX_ATTENTION_ADVERSARIAL_DIM)
                .saturating_mul(MAX_ATTENTION_ADVERSARIAL_DIM)
    );
    u16::try_from(index).map_or(0.0, f64::from)
}

#[derive(Debug, Clone, Copy)]
struct ProbeGeometry {
    query_count: usize,
    key_count: usize,
    q_dimension: usize,
    value_dimension: usize,
}

fn geometry_for(seed: u64, index: usize, config: &AttentionAdversarialConfig) -> ProbeGeometry {
    let stream = seed
        .wrapping_mul(0x9e37_79b9_7f4a_7c15)
        .wrapping_add(u64::try_from(index).unwrap_or(0).wrapping_mul(0x85eb_ca6b));
    let query_count =
        1 + usize::try_from(stream % u64::try_from(config.max_queries).unwrap_or(1)).unwrap_or(0);
    let key_count =
        1 + usize::try_from((stream >> 8) % u64::try_from(config.max_keys).unwrap_or(1))
            .unwrap_or(0);
    let q_dimension =
        1 + usize::try_from((stream >> 16) % u64::try_from(config.max_dim).unwrap_or(1))
            .unwrap_or(0);
    let value_dimension =
        1 + usize::try_from((stream >> 24) % u64::try_from(config.max_dim).unwrap_or(1))
            .unwrap_or(0);
    ProbeGeometry {
        query_count: query_count.min(config.max_queries).max(1),
        key_count: key_count.min(config.max_keys).max(1),
        q_dimension: q_dimension.min(config.max_dim).max(1),
        value_dimension: value_dimension.min(config.max_dim).max(1),
    }
}

fn build_probe_fixture(
    kind: AttentionProbeKind,
    seed: u64,
    index: u64,
    geometry: ProbeGeometry,
) -> Result<Fixture<ReferenceInput>, CegisError> {
    let (queries, keys, values, external_mask) = match kind {
        AttentionProbeKind::ExtremeFiniteScores => extreme_finite_scores(geometry),
        AttentionProbeKind::CausalMaskBoundary => causal_mask_boundary(geometry),
        AttentionProbeKind::AllEqualScores => all_equal_scores(geometry),
        AttentionProbeKind::SingleSpike => single_spike(geometry, seed, index),
        AttentionProbeKind::NearTies => near_ties(geometry),
    };
    // Keep every Q/K/V component finite before fixture construction. Extreme
    // probes stay finite here; affinity overflow is the oracle's fail-closed job.
    if queries.iter().any(|value| !value.is_finite())
        || keys.iter().any(|value| !value.is_finite())
        || values.iter().any(|value| !value.is_finite())
    {
        return Err(CegisError::InvalidFixture(format!(
            "attention probe {kind} produced a non-finite component"
        )));
    }
    let spec = ReferenceInputSpec {
        query_count: geometry.query_count,
        key_count: geometry.key_count,
        q_dimension: geometry.q_dimension,
        value_dimension: geometry.value_dimension,
        queries: queries.clone(),
        keys: keys.clone(),
        values: values.clone(),
        external_mask: external_mask.clone(),
    };
    let input = ReferenceInput::new(spec)
        .map_err(|error| CegisError::InvalidFixture(format!("attention probe {kind}: {error}")))?;
    let id = format!("attn-adv-{kind}-{seed:016x}-{index:04}");
    if id.len() > MAX_FIXTURE_ID_BYTES {
        return Err(CegisError::InvalidFixture(
            "attention probe identifier exceeds bound".into(),
        ));
    }
    let canonical_text = probe_canonical_text(
        kind,
        seed,
        index,
        geometry,
        &queries,
        &keys,
        &values,
        external_mask.as_deref(),
    )?;
    Fixture::new(id, canonical_text, input)
}

fn extreme_finite_scores(
    geometry: ProbeGeometry,
) -> (Vec<f64>, Vec<f64>, Vec<f64>, Option<Vec<bool>>) {
    let mut queries = vec![0.0; geometry.query_count * geometry.q_dimension];
    let mut keys = vec![0.0; geometry.key_count * geometry.q_dimension];
    let mut values = vec![0.0; geometry.key_count * geometry.value_dimension];
    for (offset, slot) in queries.iter_mut().enumerate() {
        *slot = if offset % 2 == 0 {
            EXTREME_FINITE_SCORE_MAGNITUDE
        } else {
            -EXTREME_FINITE_SCORE_MAGNITUDE
        };
    }
    for (offset, slot) in keys.iter_mut().enumerate() {
        *slot = if offset % 2 == 0 { 1.0 } else { -1.0 };
    }
    for (offset, slot) in values.iter_mut().enumerate() {
        *slot = small_index_as_f64(offset) + 1.0;
    }
    (queries, keys, values, None)
}

fn causal_mask_boundary(
    geometry: ProbeGeometry,
) -> (Vec<f64>, Vec<f64>, Vec<f64>, Option<Vec<bool>>) {
    let mut queries = vec![0.0; geometry.query_count * geometry.q_dimension];
    let mut keys = vec![0.0; geometry.key_count * geometry.q_dimension];
    let mut values = vec![0.0; geometry.key_count * geometry.value_dimension];
    for query in 0..geometry.query_count {
        queries[query * geometry.q_dimension] = small_index_as_f64(query) + 1.0;
    }
    for key in 0..geometry.key_count {
        keys[key * geometry.q_dimension] = small_index_as_f64(key) + 1.0;
        values[key * geometry.value_dimension] = small_index_as_f64(key) + 0.5;
    }
    let mut mask = vec![false; geometry.query_count * geometry.key_count];
    for query in 0..geometry.query_count {
        for key in 0..geometry.key_count {
            // Causal edge: key visible iff key_index <= query_index.
            mask[query * geometry.key_count + key] = key <= query;
        }
    }
    (queries, keys, values, Some(mask))
}

fn all_equal_scores(geometry: ProbeGeometry) -> (Vec<f64>, Vec<f64>, Vec<f64>, Option<Vec<bool>>) {
    let queries = vec![1.0; geometry.query_count * geometry.q_dimension];
    let keys = vec![1.0; geometry.key_count * geometry.q_dimension];
    let values = vec![2.0; geometry.key_count * geometry.value_dimension];
    (queries, keys, values, None)
}

fn single_spike(
    geometry: ProbeGeometry,
    seed: u64,
    index: u64,
) -> (Vec<f64>, Vec<f64>, Vec<f64>, Option<Vec<bool>>) {
    let mut queries = vec![0.0; geometry.query_count * geometry.q_dimension];
    let mut keys = vec![0.0; geometry.key_count * geometry.q_dimension];
    let mut values = vec![0.0; geometry.key_count * geometry.value_dimension];
    let spike =
        usize::try_from(seed.wrapping_add(index) % u64::try_from(geometry.key_count).unwrap_or(1))
            .unwrap_or(0)
            % geometry.key_count;
    for query in 0..geometry.query_count {
        queries[query * geometry.q_dimension] = 1.0;
    }
    for key in 0..geometry.key_count {
        keys[key * geometry.q_dimension] = if key == spike { 8.0 } else { 0.0 };
        values[key * geometry.value_dimension] = if key == spike { 3.0 } else { 0.25 };
    }
    (queries, keys, values, None)
}

fn near_ties(geometry: ProbeGeometry) -> (Vec<f64>, Vec<f64>, Vec<f64>, Option<Vec<bool>>) {
    let mut queries = vec![0.0; geometry.query_count * geometry.q_dimension];
    let mut keys = vec![0.0; geometry.key_count * geometry.q_dimension];
    let mut values = vec![0.0; geometry.key_count * geometry.value_dimension];
    for query in 0..geometry.query_count {
        queries[query * geometry.q_dimension] = 1.0;
    }
    for key in 0..geometry.key_count {
        let scale = if key == 0 {
            1.0 + NEAR_TIE_RELATIVE_EPS
        } else {
            1.0
        };
        keys[key * geometry.q_dimension] = scale;
        values[key * geometry.value_dimension] = small_index_as_f64(key) + 1.0;
    }
    (queries, keys, values, None)
}

#[allow(clippy::too_many_arguments)]
fn probe_canonical_text(
    kind: AttentionProbeKind,
    seed: u64,
    index: u64,
    geometry: ProbeGeometry,
    queries: &[f64],
    keys: &[f64],
    values: &[f64],
    external_mask: Option<&[bool]>,
) -> Result<String, CegisError> {
    let mut text = format!("ADA-ATTENTION-ADV-PROBE-V{ATTENTION_ADVERSARIAL_PROBE_VERSION}\n");
    let _ = writeln!(text, "kind={}", kind.as_str());
    let _ = writeln!(text, "seed={seed:016x}");
    let _ = writeln!(text, "index={index:04}");
    let _ = writeln!(
        text,
        "shape={}:{}:{}:{}",
        geometry.query_count, geometry.key_count, geometry.q_dimension, geometry.value_dimension
    );
    let _ = writeln!(text, "queries_bits={}", bits_list(queries));
    let _ = writeln!(text, "keys_bits={}", bits_list(keys));
    let _ = writeln!(text, "values_bits={}", bits_list(values));
    let _ = writeln!(text, "external_mask={}", mask_list(external_mask));
    if text.is_empty() || text.len() > MAX_FIXTURE_TEXT_BYTES || text.contains('\r') {
        return Err(CegisError::InvalidFixture(
            "attention probe canonical text is empty, oversized, or contains CR".into(),
        ));
    }
    Ok(text)
}

fn bits_list(values: &[f64]) -> String {
    values
        .iter()
        .map(|value| format!("{:016x}", value.to_bits()))
        .collect::<Vec<_>>()
        .join(",")
}

fn mask_list(mask: Option<&[bool]>) -> String {
    match mask {
        None | Some([]) => "none".into(),
        Some(values) => values
            .iter()
            .map(|visible| if *visible { '1' } else { '0' })
            .collect(),
    }
}

fn fixture_identity_key(fixture: &Fixture<ReferenceInput>) -> String {
    let mut key = String::with_capacity(fixture.id().len() + 1 + fixture.canonical_text().len());
    key.push_str(fixture.id());
    key.push('\n');
    key.push_str(fixture.canonical_text());
    key
}

fn seeded_shuffle<T>(values: &mut [T], seed: u64) {
    if values.len() < 2 {
        return;
    }
    let mut state = seed ^ 0xa076_1d64_78bd_642f;
    for end in (1..values.len()).rev() {
        state = state
            .wrapping_mul(0x9e37_79b9_7f4a_7c15)
            .wrapping_add(0x6c62_272e_e266_da77);
        let index = usize::try_from(state % u64::try_from(end + 1).unwrap_or(1)).unwrap_or(0);
        values.swap(index, end);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        CegisConfig, CegisEngine, CounterexampleSource, DifferentialOracle, OracleOutcome,
    };
    use ada_core::{SemanticFamily, SemanticId};
    use ada_search::{SearchBudget, SearchEngine, SearchError, SearchFingerprint, SearchSpace};
    use ada_semantic::{MaskRule, SelectionRule, SemanticProgram};

    #[test]
    fn generation_respects_hard_caps_and_stays_finite() {
        let generator = AttentionAdversarialGenerator::new(
            AttentionAdversarialConfig::new(3, 2, 2, 1).unwrap(),
        )
        .unwrap();
        let fixtures = generator.generate_probes(0xADA1_A771).unwrap();
        assert_eq!(fixtures.len(), 3);
        for fixture in &fixtures {
            assert!(fixture.id().starts_with("attn-adv-"));
            assert!(
                fixture
                    .canonical_text()
                    .starts_with("ADA-ATTENTION-ADV-PROBE-V1")
            );
            assert!(fixture.input().query_count() <= 2);
            assert!(fixture.input().key_count() <= 2);
            assert!(fixture.input().q_dimension() <= 1);
            assert!(fixture.input().value_dimension() <= 1);
        }
        assert!(
            AttentionAdversarialConfig::new(MAX_ATTENTION_ADVERSARIAL_FIXTURES + 1, 1, 1, 1)
                .is_err()
        );
    }

    #[test]
    fn nonfinite_probe_path_fails_closed() {
        let refused =
            refuse_nonfinite_attention_probe(AttentionProbeKind::ExtremeFiniteScores, 7, f64::NAN);
        assert!(refused.is_err());
        let refused_inf =
            refuse_nonfinite_attention_probe(AttentionProbeKind::NearTies, 7, f64::INFINITY);
        assert!(refused_inf.is_err());
        assert!(
            refuse_nonfinite_attention_probe(AttentionProbeKind::AllEqualScores, 7, 1.0).is_err()
        );
    }

    #[test]
    fn corpus_growth_is_deterministic_across_runs() {
        let generator = AttentionAdversarialGenerator::default();
        let seed = 42_u64;
        let left_generated = generator.generate_probes(seed).unwrap();
        let right_generated = generator.generate_probes(seed).unwrap();
        assert_eq!(
            left_generated
                .iter()
                .map(|fixture| (fixture.id(), fixture.canonical_text()))
                .collect::<Vec<_>>(),
            right_generated
                .iter()
                .map(|fixture| (fixture.id(), fixture.canonical_text()))
                .collect::<Vec<_>>()
        );

        let seed_fixture = Fixture::new(
            "seed",
            "seed-corpus",
            ReferenceInput::new(ReferenceInputSpec {
                query_count: 1,
                key_count: 1,
                q_dimension: 1,
                value_dimension: 1,
                queries: vec![0.0],
                keys: vec![0.0],
                values: vec![1.0],
                external_mask: None,
            })
            .unwrap(),
        )
        .unwrap();
        let left = grow_attention_adversarial_corpus(
            std::slice::from_ref(&seed_fixture),
            &left_generated,
            64,
        )
        .unwrap();
        let right = grow_attention_adversarial_corpus(
            std::slice::from_ref(&seed_fixture),
            &right_generated,
            64,
        )
        .unwrap();
        assert_eq!(
            left.iter()
                .map(|fixture| (fixture.id(), fixture.fingerprint()))
                .collect::<Vec<_>>(),
            right
                .iter()
                .map(|fixture| (fixture.id(), fixture.fingerprint()))
                .collect::<Vec<_>>()
        );
        assert!(grow_attention_adversarial_corpus(&[seed_fixture], &left_generated, 1).is_err());
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum ToyAttentionCandidate {
        Softmax,
        AlwaysZero,
    }

    #[derive(Debug, Clone)]
    struct ToyAttentionSpace {
        values: Vec<ToyAttentionCandidate>,
    }

    impl SearchSpace for ToyAttentionSpace {
        type Candidate = ToyAttentionCandidate;

        fn cardinality(&self) -> u64 {
            u64::try_from(self.values.len()).unwrap()
        }

        fn fingerprint(&self) -> SearchFingerprint {
            SearchFingerprint::of_canonical_text("toy-attention-space")
        }

        fn candidate_at(&self, ordinal: u64) -> Result<Self::Candidate, SearchError> {
            self.values
                .get(usize::try_from(ordinal).unwrap())
                .copied()
                .ok_or(SearchError::InvalidConfiguration("toy ordinal"))
        }

        fn candidate_canonical_text(&self, candidate: &Self::Candidate) -> String {
            match candidate {
                ToyAttentionCandidate::Softmax => "toy=softmax".into(),
                ToyAttentionCandidate::AlwaysZero => "toy=always-zero".into(),
            }
        }

        fn candidate_cost(&self, _candidate: &Self::Candidate) -> u32 {
            1
        }
    }

    struct SoftmaxOracle {
        reference: SemanticProgram,
    }

    impl SoftmaxOracle {
        fn new() -> Self {
            let id = SemanticId::new(SemanticFamily::StandardSoftmax, "attn-adv-ref", 1).unwrap();
            Self {
                reference: SemanticProgram::standard_softmax(
                    id,
                    MaskRule::Unmasked,
                    SelectionRule::All,
                    1.0,
                )
                .unwrap(),
            }
        }
    }

    impl DifferentialOracle<ToyAttentionCandidate, ReferenceInput> for SoftmaxOracle {
        type Error = String;

        fn compare(
            &mut self,
            candidate: &ToyAttentionCandidate,
            fixture: &Fixture<ReferenceInput>,
        ) -> Result<OracleOutcome, Self::Error> {
            let expected = match self.reference.evaluate(fixture.input()) {
                Ok(output) => output,
                Err(error) => {
                    // Extreme probes may overflow affinity; fail closed rather
                    // than silently accepting a non-finite reference path.
                    return Ok(OracleOutcome::Falsified {
                        reason: format!("reference rejected fixture: {error}"),
                    });
                }
            };
            match candidate {
                ToyAttentionCandidate::Softmax => Ok(OracleOutcome::Pass),
                ToyAttentionCandidate::AlwaysZero => {
                    let any_nonzero = expected.output().iter().any(|value| *value != 0.0);
                    if any_nonzero {
                        Ok(OracleOutcome::Falsified {
                            reason: "always-zero candidate disagrees with nonzero reference".into(),
                        })
                    } else {
                        Ok(OracleOutcome::Pass)
                    }
                }
            }
        }
    }

    #[test]
    fn deliberately_wrong_candidate_is_falsified_by_attention_probes() {
        // Prefer mild geometries so the reference path stays finite while still
        // exercising spike / near-tie / all-equal / causal probes.
        let generator = AttentionAdversarialGenerator::new(
            AttentionAdversarialConfig::new(5, 2, 2, 1).unwrap(),
        )
        .unwrap();
        let search = SearchEngine::new(
            ToyAttentionSpace {
                values: vec![
                    ToyAttentionCandidate::AlwaysZero,
                    ToyAttentionCandidate::Softmax,
                ],
            },
            SearchBudget::new(8, 8, 1).unwrap(),
        )
        .unwrap();
        let seed_input = ReferenceInput::new(ReferenceInputSpec {
            query_count: 1,
            key_count: 1,
            q_dimension: 1,
            value_dimension: 1,
            queries: vec![0.0],
            keys: vec![0.0],
            values: vec![0.0],
            external_mask: None,
        })
        .unwrap();
        let seed = Fixture::new("seed-zero", "seed-zero-v1", seed_input).unwrap();
        let runner = CegisEngine::new(
            search,
            SoftmaxOracle::new(),
            generator,
            CegisConfig::new(11, 16, 16, 8).unwrap(),
            vec![seed],
        )
        .unwrap();
        let result = runner.run_to_end().unwrap();
        assert!(
            result
                .rejected()
                .iter()
                .any(|rejected| *rejected.candidate().candidate()
                    == ToyAttentionCandidate::AlwaysZero)
        );
        assert!(result.counterexamples().iter().any(|counterexample| {
            matches!(
                counterexample.source(),
                CounterexampleSource::Adversarial | CounterexampleSource::InitialCorpus
            )
        }));
        assert!(
            result
                .survivors()
                .iter()
                .any(|survivor| *survivor.candidate() == ToyAttentionCandidate::Softmax)
                || result.stats().adversarial_falsified() > 0
        );
    }
}
