//! Small executable semantic IR for bounded attention reference experiments.
//!
//! A semantic program describes the interaction rule being computed. It has no
//! tile size, register allocation, launch geometry, backend, or hardware
//! evidence. The current grammar is intentionally small and supports a
//! deterministic f64 reference path for scaled dot products, masking,
//! bounded selection, softmax or signed-difference weighting, and weighted
//! value mixing.
//!
//! Compressed KV, recurrent updates, backward execution, and implementation
//! schedules remain later layers. Their absence here is explicit: an enum
//! variant or a metadata field is not treated as executable support.

#![forbid(unsafe_code)]

mod codec;

use ada_core::{
    MaskContract, SemanticDescriptor, SemanticFamily, SemanticId, StateContract, WeightContract,
};
use ada_workload::{
    AttentionTopology, InputRepresentation, KvCacheSpec, KvIndexing, KvRepresentation, MaskKind,
    MatrixLayout, PositionInfo, ScalarPrecision, StateSpec, WorkloadContract, WorkloadMode,
};
use std::fmt::{Display, Formatter};

/// Version of the executable semantic IR and canonical codec.
pub const SEMANTIC_IR_VERSION: u16 = 1;
/// Maximum query rows accepted by the reference evaluator.
pub const MAX_REFERENCE_QUERIES: usize = 4_096;
/// Maximum KV rows accepted by the reference evaluator.
pub const MAX_REFERENCE_KEYS: usize = 65_536;
/// Maximum dimension accepted by the reference evaluator.
pub const MAX_REFERENCE_DIMENSION: usize = 4_096;
/// Maximum number of elements in one reference tensor allocation.
pub const MAX_REFERENCE_ELEMENTS: usize = 1 << 24;
/// Maximum number of query/key score or mask entries in one reference run.
pub const MAX_REFERENCE_SCORE_ELEMENTS: usize = 1 << 22;
/// Maximum canonical semantic artifact size.
pub const MAX_CANONICAL_TEXT_BYTES: usize = 4 << 20;
/// Maximum external identity size used by executable semantic rules.
pub const MAX_IDENTITY_BYTES: usize = 256;

/// Construction, validation, decoding, and reference-execution failures.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SemanticIrError {
    /// A semantic IR field violates its domain contract.
    InvalidField(&'static str),
    /// A field exceeded a bounded reference or structural limit.
    ExceedsLimit {
        /// Field whose value was rejected.
        field: &'static str,
        /// Rejected value.
        value: usize,
        /// Inclusive maximum.
        maximum: usize,
    },
    /// Input vectors do not have the declared shape.
    ShapeMismatch {
        /// Field whose length was wrong.
        field: &'static str,
        /// Expected element count.
        expected: usize,
        /// Actual element count.
        actual: usize,
    },
    /// A required external mask was not supplied.
    MissingExternalMask,
    /// A visible selection is empty for a query row.
    EmptyVisibleSelection(usize),
    /// A non-finite input or intermediate was encountered.
    NonFiniteValue(&'static str),
    /// A contract cannot be represented by the current executable domain.
    UnsupportedWorkload(&'static str),
    /// An unsupported IR or codec version was requested.
    UnsupportedVersion(u16),
    /// Canonical text was malformed or incomplete.
    MalformedCanonicalText(String),
}

impl Display for SemanticIrError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidField(field) => write!(formatter, "invalid semantic field: {field}"),
            Self::ExceedsLimit {
                field,
                value,
                maximum,
            } => write!(formatter, "{field}={value} exceeds maximum {maximum}"),
            Self::ShapeMismatch {
                field,
                expected,
                actual,
            } => write!(
                formatter,
                "{field} has {actual} elements; expected {expected}"
            ),
            Self::MissingExternalMask => write!(formatter, "external mask data is missing"),
            Self::EmptyVisibleSelection(query) => {
                write!(formatter, "query {query} has no visible selected keys")
            }
            Self::NonFiniteValue(stage) => write!(formatter, "non-finite value at {stage}"),
            Self::UnsupportedWorkload(reason) => {
                write!(
                    formatter,
                    "workload is outside semantic reference domain: {reason}"
                )
            }
            Self::UnsupportedVersion(version) => {
                write!(formatter, "unsupported semantic IR version {version}")
            }
            Self::MalformedCanonicalText(reason) => {
                write!(formatter, "malformed semantic canonical text: {reason}")
            }
        }
    }
}

impl std::error::Error for SemanticIrError {}

fn validate_count(
    field: &'static str,
    value: usize,
    maximum: usize,
) -> Result<(), SemanticIrError> {
    if value == 0 {
        return Err(SemanticIrError::InvalidField(field));
    }
    if value > maximum {
        return Err(SemanticIrError::ExceedsLimit {
            field,
            value,
            maximum,
        });
    }
    Ok(())
}

fn validate_identity(field: &'static str, identity: &str) -> Result<(), SemanticIrError> {
    if identity.is_empty()
        || identity.len() > MAX_IDENTITY_BYTES
        || identity.chars().any(char::is_control)
        || identity.chars().any(char::is_whitespace)
    {
        return Err(SemanticIrError::InvalidField(field));
    }
    Ok(())
}

/// Transformation applied independently to each query and key row.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum InputTransform {
    /// Leave Q and K rows unchanged.
    Identity,
    /// Subtract each row's arithmetic mean from Q and K.
    CenterRows,
}

impl InputTransform {
    fn as_text(self) -> &'static str {
        match self {
            Self::Identity => "identity",
            Self::CenterRows => "center-rows",
        }
    }

    fn from_text(value: &str) -> Result<Self, SemanticIrError> {
        match value {
            "identity" => Ok(Self::Identity),
            "center-rows" => Ok(Self::CenterRows),
            _ => Err(SemanticIrError::MalformedCanonicalText(
                "unknown input transform".into(),
            )),
        }
    }
}

/// Affinity rule producing one scalar score per query/key pair.
#[derive(Debug, Clone, Copy)]
pub enum AffinityRule {
    /// Compute scale times the ordinary Q dot K product.
    ScaledDotProduct {
        /// Positive score scale.
        scale: f64,
    },
}

impl PartialEq for AffinityRule {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::ScaledDotProduct { scale: left }, Self::ScaledDotProduct { scale: right }) => {
                left.to_bits() == right.to_bits()
            }
        }
    }
}

impl AffinityRule {
    fn validate(self) -> Result<(), SemanticIrError> {
        match self {
            Self::ScaledDotProduct { scale } if scale.is_finite() && scale > 0.0 => Ok(()),
            Self::ScaledDotProduct { .. } => Err(SemanticIrError::InvalidField("affinity.scale")),
        }
    }
}

impl Eq for AffinityRule {}

impl std::hash::Hash for AffinityRule {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        match self {
            Self::ScaledDotProduct { scale } => {
                0_u8.hash(state);
                scale.to_bits().hash(state);
            }
        }
    }
}

/// Visibility rule applied before selection.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum MaskRule {
    /// Every key row is visible.
    Unmasked,
    /// Key index must not exceed query index.
    Causal,
    /// Visibility comes from an explicitly named boolean mask artifact.
    External {
        /// External mask identity.
        identity: String,
    },
}

impl MaskRule {
    fn validate(&self) -> Result<(), SemanticIrError> {
        if let Self::External { identity } = self {
            validate_identity("mask.identity", identity)?;
        }
        Ok(())
    }

    fn descriptor_contract(&self) -> MaskContract {
        match self {
            Self::Unmasked => MaskContract::Bidirectional,
            Self::Causal => MaskContract::Causal,
            Self::External { .. } => MaskContract::ExternalMask,
        }
    }
}

/// Bounded deterministic key selection after masking.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SelectionRule {
    /// Keep every visible key.
    All,
    /// Keep visible keys whose indices are within this radius of the query.
    Window {
        /// Inclusive index radius.
        radius: usize,
    },
    /// Keep the k highest visible scores; ties prefer lower key indices.
    TopK {
        /// Number of selected keys.
        k: usize,
    },
}

impl SelectionRule {
    fn validate(self) -> Result<(), SemanticIrError> {
        match self {
            Self::All => Ok(()),
            Self::Window { radius } => {
                if radius > MAX_REFERENCE_KEYS {
                    Err(SemanticIrError::ExceedsLimit {
                        field: "selection.radius",
                        value: radius,
                        maximum: MAX_REFERENCE_KEYS,
                    })
                } else {
                    Ok(())
                }
            }
            Self::TopK { k } => validate_count("selection.k", k, MAX_REFERENCE_KEYS),
        }
    }
}

/// Normalization/weighting rule applied to selected scores.
#[derive(Debug, Clone, Copy)]
pub enum WeightRule {
    /// Stable probability-simplex softmax.
    Softmax,
    /// Difference of two independently normalized softmax distributions.
    SignedDifference {
        /// Positive branch scale.
        positive_scale: f64,
        /// Negative branch scale.
        negative_scale: f64,
    },
}

impl PartialEq for WeightRule {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Softmax, Self::Softmax) => true,
            (
                Self::SignedDifference {
                    positive_scale: left_positive,
                    negative_scale: left_negative,
                },
                Self::SignedDifference {
                    positive_scale: right_positive,
                    negative_scale: right_negative,
                },
            ) => {
                left_positive.to_bits() == right_positive.to_bits()
                    && left_negative.to_bits() == right_negative.to_bits()
            }
            _ => false,
        }
    }
}

impl WeightRule {
    fn validate(self) -> Result<(), SemanticIrError> {
        match self {
            Self::Softmax => Ok(()),
            Self::SignedDifference {
                positive_scale,
                negative_scale,
            } if positive_scale.is_finite()
                && positive_scale > 0.0
                && negative_scale.is_finite()
                && negative_scale > 0.0 =>
            {
                Ok(())
            }
            Self::SignedDifference { .. } => Err(SemanticIrError::InvalidField("weight.scale")),
        }
    }

    fn descriptor_contract(self) -> WeightContract {
        match self {
            Self::Softmax => WeightContract::ProbabilitySimplex,
            Self::SignedDifference { .. } => WeightContract::Signed,
        }
    }
}

impl Eq for WeightRule {}

impl std::hash::Hash for WeightRule {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        match self {
            Self::Softmax => 0_u8.hash(state),
            Self::SignedDifference {
                positive_scale,
                negative_scale,
            } => {
                1_u8.hash(state);
                positive_scale.to_bits().hash(state);
                negative_scale.to_bits().hash(state);
            }
        }
    }
}

/// Value mixing rule in the initial semantic grammar.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ValueMixRule {
    /// Multiply selected V rows by weights and sum them.
    WeightedSum,
}

/// Output rule in the initial semantic grammar.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OutputRule {
    /// Return the mixed V row without a projection.
    Identity,
}

/// Bounded executable semantic program.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SemanticProgram {
    descriptor: SemanticDescriptor,
    input_transform: InputTransform,
    affinity: AffinityRule,
    mask: MaskRule,
    selection: SelectionRule,
    weight: WeightRule,
    value_mix: ValueMixRule,
    output: OutputRule,
}

/// Components supplied when constructing a semantic program.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SemanticProgramSpec {
    /// Semantic identity and descriptor contracts.
    pub descriptor: SemanticDescriptor,
    /// Query/key input transformation.
    pub input_transform: InputTransform,
    /// Affinity rule.
    pub affinity: AffinityRule,
    /// Visibility rule.
    pub mask: MaskRule,
    /// Post-mask key selection rule.
    pub selection: SelectionRule,
    /// Score weighting/normalization rule.
    pub weight: WeightRule,
    /// Value mixing rule.
    pub value_mix: ValueMixRule,
    /// Output rule.
    pub output: OutputRule,
}

/// Alias emphasizing that this is an IR rather than an implementation.
pub type SemanticIr = SemanticProgram;

impl SemanticProgram {
    /// Construct a semantically bound, validated program.
    ///
    /// # Errors
    ///
    /// Returns an error when a component is outside the bounded grammar or its
    /// descriptor does not match the executable mask/weight rule.
    pub fn new(spec: SemanticProgramSpec) -> Result<Self, SemanticIrError> {
        let SemanticProgramSpec {
            descriptor,
            input_transform,
            affinity,
            mask,
            selection,
            weight,
            value_mix,
            output,
        } = spec;
        if descriptor.state() != StateContract::Stateless {
            return Err(SemanticIrError::UnsupportedWorkload(
                "recurrent semantic state is not executable in IR v1",
            ));
        }
        if descriptor.mask() != mask.descriptor_contract() {
            return Err(SemanticIrError::InvalidField(
                "descriptor.mask does not match mask rule",
            ));
        }
        if descriptor.weights() != weight.descriptor_contract() {
            return Err(SemanticIrError::InvalidField(
                "descriptor.weights does not match weight rule",
            ));
        }
        affinity.validate()?;
        mask.validate()?;
        selection.validate()?;
        weight.validate()?;
        Ok(Self {
            descriptor,
            input_transform,
            affinity,
            mask,
            selection,
            weight,
            value_mix,
            output,
        })
    }

    /// Construct a standard softmax semantic with the supplied identity.
    ///
    /// # Errors
    ///
    /// Returns an error if the identity, scale, mask, or selection rule is
    /// invalid.
    pub fn standard_softmax(
        id: SemanticId,
        mask: MaskRule,
        selection: SelectionRule,
        scale: f64,
    ) -> Result<Self, SemanticIrError> {
        let descriptor = SemanticDescriptor::new(
            id,
            mask.descriptor_contract(),
            StateContract::Stateless,
            WeightContract::ProbabilitySimplex,
        );
        Self::new(SemanticProgramSpec {
            descriptor,
            input_transform: InputTransform::Identity,
            affinity: AffinityRule::ScaledDotProduct { scale },
            mask,
            selection,
            weight: WeightRule::Softmax,
            value_mix: ValueMixRule::WeightedSum,
            output: OutputRule::Identity,
        })
    }

    /// Construct a signed-difference semantic with the supplied identity.
    ///
    /// # Errors
    ///
    /// Returns an error if the identity, scales, mask, or selection rule is
    /// invalid.
    pub fn signed_difference(
        id: SemanticId,
        mask: MaskRule,
        selection: SelectionRule,
        score_scale: f64,
        positive_scale: f64,
        negative_scale: f64,
    ) -> Result<Self, SemanticIrError> {
        let descriptor = SemanticDescriptor::new(
            id,
            mask.descriptor_contract(),
            StateContract::Stateless,
            WeightContract::Signed,
        );
        Self::new(SemanticProgramSpec {
            descriptor,
            input_transform: InputTransform::Identity,
            affinity: AffinityRule::ScaledDotProduct { scale: score_scale },
            mask,
            selection,
            weight: WeightRule::SignedDifference {
                positive_scale,
                negative_scale,
            },
            value_mix: ValueMixRule::WeightedSum,
            output: OutputRule::Identity,
        })
    }

    /// Semantic descriptor and stable semantic identity.
    #[must_use]
    pub const fn descriptor(&self) -> &SemanticDescriptor {
        &self.descriptor
    }

    /// Input transformation.
    #[must_use]
    pub const fn input_transform(&self) -> InputTransform {
        self.input_transform
    }

    /// Affinity rule.
    #[must_use]
    pub const fn affinity(&self) -> AffinityRule {
        self.affinity
    }

    /// Mask rule.
    #[must_use]
    pub const fn mask(&self) -> &MaskRule {
        &self.mask
    }

    /// Selection rule.
    #[must_use]
    pub const fn selection(&self) -> SelectionRule {
        self.selection
    }

    /// Weighting rule.
    #[must_use]
    pub const fn weight(&self) -> WeightRule {
        self.weight
    }

    /// Value mixing rule.
    #[must_use]
    pub const fn value_mix(&self) -> ValueMixRule {
        self.value_mix
    }

    /// Output rule.
    #[must_use]
    pub const fn output(&self) -> OutputRule {
        self.output
    }

    /// Validate this program against the currently executable workload
    /// reference domain.
    ///
    /// # Errors
    ///
    /// Returns an error for batch/head/cache/precision/layout/state features
    /// not implemented by the v1 single-head f64 reference evaluator.
    pub fn validate_for_workload(
        &self,
        workload: &WorkloadContract,
    ) -> Result<(), SemanticIrError> {
        workload
            .validate()
            .map_err(|_| SemanticIrError::UnsupportedWorkload("invalid workload contract"))?;
        validate_reference_geometry(workload)?;
        validate_reference_state_and_mode(workload)?;
        validate_reference_precision_and_layout(workload)?;
        self.validate_reference_selection_and_mask(workload)?;
        Ok(())
    }

    /// Evaluate the semantic program against the independent f64 reference
    /// input path.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid shapes, missing masks, empty selections,
    /// or non-finite inputs/intermediates. No unsupported component is
    /// silently dropped.
    pub fn evaluate(&self, input: &ReferenceInput) -> Result<ReferenceOutput, SemanticIrError> {
        self.validate_components()?;
        input.validate()?;
        if let MaskRule::External { .. } = &self.mask {
            if input.external_mask.is_none() {
                return Err(SemanticIrError::MissingExternalMask);
            }
        }
        let (queries, keys) = transform_inputs(self.input_transform, input)?;
        let mut scores = vec![0.0_f64; input.query_count * input.key_count];
        for query in 0..input.query_count {
            for key in 0..input.key_count {
                let query_row =
                    &queries[query * input.q_dimension..(query + 1) * input.q_dimension];
                let key_row = &keys[key * input.q_dimension..(key + 1) * input.q_dimension];
                let mut dot = 0.0_f64;
                for (&query_value, &key_value) in query_row.iter().zip(key_row) {
                    dot += query_value * key_value;
                }
                let score = match self.affinity {
                    AffinityRule::ScaledDotProduct { scale } => dot * scale,
                };
                if !score.is_finite() {
                    return Err(SemanticIrError::NonFiniteValue("affinity"));
                }
                scores[query * input.key_count + key] = score;
            }
        }

        let mut output = vec![0.0_f64; input.query_count * input.value_dimension];
        let mut weights = vec![0.0_f64; input.query_count * input.key_count];
        let mut normalizations = Vec::with_capacity(input.query_count);
        let mut selected_keys = Vec::with_capacity(input.query_count);
        for query in 0..input.query_count {
            let selected = select_keys(self, input, &scores, query)?;
            let selected_scores = selected
                .iter()
                .map(|&key| scores[query * input.key_count + key])
                .collect::<Vec<_>>();
            let (selected_weights, normalization) = weights_for(self.weight, &selected_scores)?;
            let output_row =
                &mut output[query * input.value_dimension..(query + 1) * input.value_dimension];
            for (&key, &weight) in selected.iter().zip(&selected_weights) {
                weights[query * input.key_count + key] = weight;
                let value_row =
                    &input.values[key * input.value_dimension..(key + 1) * input.value_dimension];
                for (output_value, &value) in output_row.iter_mut().zip(value_row) {
                    *output_value += weight * value;
                }
            }
            if output_row.iter().any(|value| !value.is_finite()) {
                return Err(SemanticIrError::NonFiniteValue("value mixing"));
            }
            normalizations.push(normalization);
            selected_keys.push(selected);
        }
        Ok(ReferenceOutput {
            output,
            weights,
            normalizations,
            selected_keys,
        })
    }

    fn validate_components(&self) -> Result<(), SemanticIrError> {
        self.affinity.validate()?;
        self.mask.validate()?;
        self.selection.validate()?;
        self.weight.validate()?;
        Ok(())
    }

    /// Canonical deterministic text representation.
    #[must_use]
    pub fn to_canonical_text(&self) -> String {
        codec::encode(self)
    }

    /// Decode and validate a canonical semantic program.
    ///
    /// # Errors
    ///
    /// Returns an error for malformed, duplicate, unknown, or incomplete
    /// fields, unsupported versions, and invalid semantic components.
    pub fn from_canonical_text(text: &str) -> Result<Self, SemanticIrError> {
        codec::decode(text)
    }

    /// Stable dual-lane fingerprint of the canonical semantic program.
    #[must_use]
    pub fn fingerprint(&self) -> SemanticFingerprint {
        SemanticFingerprint::of_bytes(self.to_canonical_text().as_bytes())
    }
}

fn validate_reference_geometry(workload: &WorkloadContract) -> Result<(), SemanticIrError> {
    let geometry = workload.geometry();
    if geometry.sequence_lengths().batch_count() != 1
        || geometry.query_heads() != 1
        || geometry.kv_heads() != 1
    {
        return Err(SemanticIrError::UnsupportedWorkload(
            "reference evaluator is single-batch and single-head",
        ));
    }
    if !matches!(
        geometry.topology(),
        AttentionTopology::SelfAttention | AttentionTopology::CrossAttention
    ) {
        return Err(SemanticIrError::UnsupportedWorkload(
            "historical topology is not an executable semantic input",
        ));
    }
    if !matches!(workload.inputs(), InputRepresentation::ExplicitQkv) {
        return Err(SemanticIrError::UnsupportedWorkload(
            "reference evaluator requires explicit Q/K/V inputs",
        ));
    }
    if geometry.qk_dimension().is_none() {
        return Err(SemanticIrError::UnsupportedWorkload(
            "explicit Q/K dimension is missing",
        ));
    }
    Ok(())
}

fn validate_reference_state_and_mode(workload: &WorkloadContract) -> Result<(), SemanticIrError> {
    if !matches!(workload.kv_representation(), KvRepresentation::Full)
        || !matches!(workload.kv_cache(), KvCacheSpec::None)
        || !matches!(workload.kv_indexing(), KvIndexing::Identity)
        || !matches!(workload.state(), StateSpec::Stateless)
    {
        return Err(SemanticIrError::UnsupportedWorkload(
            "compressed, cached, indexed, or recurrent state is not executable in IR v1",
        ));
    }
    if !matches!(
        workload.mode(),
        WorkloadMode::Prefill | WorkloadMode::TrainingForward
    ) {
        return Err(SemanticIrError::UnsupportedWorkload(
            "decode and backward modes are not executable in IR v1",
        ));
    }
    if !matches!(workload.positions(), PositionInfo::None)
        || !matches!(workload.score_bias(), ada_workload::ScoreBiasSpec::None)
    {
        return Err(SemanticIrError::UnsupportedWorkload(
            "position and score-bias reference rules are not executable in IR v1",
        ));
    }
    Ok(())
}

fn validate_reference_precision_and_layout(
    workload: &WorkloadContract,
) -> Result<(), SemanticIrError> {
    let precision = workload.precision();
    if [
        precision.input(),
        precision.accumulation(),
        precision.output(),
        precision.storage(),
    ]
    .into_iter()
    .any(|value| value != ScalarPrecision::F64)
    {
        return Err(SemanticIrError::UnsupportedWorkload(
            "the reference evaluator is explicitly f64",
        ));
    }
    let layout = workload.layout();
    if [
        layout.query(),
        layout.key(),
        layout.value(),
        layout.output(),
    ]
    .into_iter()
    .any(|value| value != MatrixLayout::RowMajor)
    {
        return Err(SemanticIrError::UnsupportedWorkload(
            "the reference evaluator requires row-major tensors",
        ));
    }
    Ok(())
}

impl SemanticProgram {
    fn validate_reference_selection_and_mask(
        &self,
        workload: &WorkloadContract,
    ) -> Result<(), SemanticIrError> {
        if let SelectionRule::TopK { k } = self.selection {
            if workload
                .geometry()
                .sequence_lengths()
                .kv_length_for(0)
                .is_some_and(|length| k > length)
            {
                return Err(SemanticIrError::InvalidField(
                    "selection.k exceeds the workload KV length",
                ));
            }
        }
        let mask_matches = match (workload.mask().kind(), &self.mask) {
            (MaskKind::None | MaskKind::Bidirectional, MaskRule::Unmasked)
            | (MaskKind::Causal, MaskRule::Causal) => true,
            (
                MaskKind::External {
                    identity: workload_id,
                },
                MaskRule::External {
                    identity: program_id,
                },
            ) => workload_id == program_id,
            _ => false,
        };
        if !mask_matches {
            return Err(SemanticIrError::InvalidField(
                "workload mask does not match semantic mask",
            ));
        }
        Ok(())
    }
}

/// Independent bounded f64 Q/K/V input for semantic reference execution.
#[derive(Debug, Clone, PartialEq)]
pub struct ReferenceInput {
    query_count: usize,
    key_count: usize,
    q_dimension: usize,
    value_dimension: usize,
    queries: Vec<f64>,
    keys: Vec<f64>,
    values: Vec<f64>,
    external_mask: Option<Vec<bool>>,
}

/// Constructor input for a bounded reference attention workload.
#[derive(Debug, Clone, PartialEq)]
pub struct ReferenceInputSpec {
    /// Query row count.
    pub query_count: usize,
    /// KV row count.
    pub key_count: usize,
    /// Q/K dimension.
    pub q_dimension: usize,
    /// V/output dimension.
    pub value_dimension: usize,
    /// Row-major Q values.
    pub queries: Vec<f64>,
    /// Row-major K values.
    pub keys: Vec<f64>,
    /// Row-major V values.
    pub values: Vec<f64>,
    /// Optional row-major boolean mask; `true` means the key is visible.
    pub external_mask: Option<Vec<bool>>,
}

impl ReferenceInput {
    /// Construct a shape-checked, finite reference input.
    ///
    /// # Errors
    ///
    /// Returns an error for zero/oversized dimensions, vector shape
    /// mismatches, non-finite values, or an incorrectly shaped external mask.
    pub fn new(spec: ReferenceInputSpec) -> Result<Self, SemanticIrError> {
        let ReferenceInputSpec {
            query_count,
            key_count,
            q_dimension,
            value_dimension,
            queries,
            keys,
            values,
            external_mask,
        } = spec;
        validate_count("query_count", query_count, MAX_REFERENCE_QUERIES)?;
        validate_count("key_count", key_count, MAX_REFERENCE_KEYS)?;
        validate_count("q_dimension", q_dimension, MAX_REFERENCE_DIMENSION)?;
        validate_count("value_dimension", value_dimension, MAX_REFERENCE_DIMENSION)?;
        let query_elements =
            checked_bounded_product(query_count, q_dimension, "queries", MAX_REFERENCE_ELEMENTS)?;
        let key_elements =
            checked_bounded_product(key_count, q_dimension, "keys", MAX_REFERENCE_ELEMENTS)?;
        let value_elements =
            checked_bounded_product(key_count, value_dimension, "values", MAX_REFERENCE_ELEMENTS)?;
        checked_bounded_product(
            query_count,
            value_dimension,
            "output",
            MAX_REFERENCE_ELEMENTS,
        )?;
        checked_bounded_product(
            query_count,
            key_count,
            "score matrix",
            MAX_REFERENCE_SCORE_ELEMENTS,
        )?;
        check_len("queries", query_elements, queries.len())?;
        check_len("keys", key_elements, keys.len())?;
        check_len("values", value_elements, values.len())?;
        if let Some(mask) = &external_mask {
            let mask_elements = checked_bounded_product(
                query_count,
                key_count,
                "external_mask",
                MAX_REFERENCE_SCORE_ELEMENTS,
            )?;
            check_len("external_mask", mask_elements, mask.len())?;
        }
        if queries.iter().any(|value| !value.is_finite())
            || keys.iter().any(|value| !value.is_finite())
            || values.iter().any(|value| !value.is_finite())
        {
            return Err(SemanticIrError::NonFiniteValue("reference input"));
        }
        Ok(Self {
            query_count,
            key_count,
            q_dimension,
            value_dimension,
            queries,
            keys,
            values,
            external_mask,
        })
    }

    /// Query row count.
    #[must_use]
    pub const fn query_count(&self) -> usize {
        self.query_count
    }

    /// KV row count.
    #[must_use]
    pub const fn key_count(&self) -> usize {
        self.key_count
    }

    /// Q/K dimension.
    #[must_use]
    pub const fn q_dimension(&self) -> usize {
        self.q_dimension
    }

    /// V/output dimension.
    #[must_use]
    pub const fn value_dimension(&self) -> usize {
        self.value_dimension
    }

    fn validate(&self) -> Result<(), SemanticIrError> {
        validate_count("query_count", self.query_count, MAX_REFERENCE_QUERIES)?;
        validate_count("key_count", self.key_count, MAX_REFERENCE_KEYS)?;
        validate_count("q_dimension", self.q_dimension, MAX_REFERENCE_DIMENSION)?;
        validate_count(
            "value_dimension",
            self.value_dimension,
            MAX_REFERENCE_DIMENSION,
        )?;
        let query_elements = checked_bounded_product(
            self.query_count,
            self.q_dimension,
            "queries",
            MAX_REFERENCE_ELEMENTS,
        )?;
        let key_elements = checked_bounded_product(
            self.key_count,
            self.q_dimension,
            "keys",
            MAX_REFERENCE_ELEMENTS,
        )?;
        let value_elements = checked_bounded_product(
            self.key_count,
            self.value_dimension,
            "values",
            MAX_REFERENCE_ELEMENTS,
        )?;
        checked_bounded_product(
            self.query_count,
            self.value_dimension,
            "output",
            MAX_REFERENCE_ELEMENTS,
        )?;
        checked_bounded_product(
            self.query_count,
            self.key_count,
            "score matrix",
            MAX_REFERENCE_SCORE_ELEMENTS,
        )?;
        check_len("queries", query_elements, self.queries.len())?;
        check_len("keys", key_elements, self.keys.len())?;
        check_len("values", value_elements, self.values.len())?;
        if self.queries.iter().any(|value| !value.is_finite())
            || self.keys.iter().any(|value| !value.is_finite())
            || self.values.iter().any(|value| !value.is_finite())
        {
            return Err(SemanticIrError::NonFiniteValue("reference input"));
        }
        if let Some(mask) = &self.external_mask {
            let mask_elements = checked_bounded_product(
                self.query_count,
                self.key_count,
                "external_mask",
                MAX_REFERENCE_SCORE_ELEMENTS,
            )?;
            check_len("external_mask", mask_elements, mask.len())?;
        }
        Ok(())
    }
}

fn checked_product(
    left: usize,
    right: usize,
    field: &'static str,
) -> Result<usize, SemanticIrError> {
    left.checked_mul(right)
        .ok_or(SemanticIrError::InvalidField(field))
}

fn checked_bounded_product(
    left: usize,
    right: usize,
    field: &'static str,
    maximum: usize,
) -> Result<usize, SemanticIrError> {
    let value = checked_product(left, right, field)?;
    if value > maximum {
        return Err(SemanticIrError::ExceedsLimit {
            field,
            value,
            maximum,
        });
    }
    Ok(value)
}

fn check_len(field: &'static str, expected: usize, actual: usize) -> Result<(), SemanticIrError> {
    if expected == actual {
        Ok(())
    } else {
        Err(SemanticIrError::ShapeMismatch {
            field,
            expected,
            actual,
        })
    }
}

fn transform_inputs(
    transform: InputTransform,
    input: &ReferenceInput,
) -> Result<(Vec<f64>, Vec<f64>), SemanticIrError> {
    let mut queries = input.queries.clone();
    let mut keys = input.keys.clone();
    if matches!(transform, InputTransform::CenterRows) {
        center_rows(&mut queries, input.query_count, input.q_dimension)?;
        center_rows(&mut keys, input.key_count, input.q_dimension)?;
    }
    Ok((queries, keys))
}

fn center_rows(
    values: &mut [f64],
    row_count: usize,
    dimension: usize,
) -> Result<(), SemanticIrError> {
    for row in 0..row_count {
        let row_values = &mut values[row * dimension..(row + 1) * dimension];
        let dimension_as_u32 = u32::try_from(dimension)
            .map_err(|_| SemanticIrError::InvalidField("input dimension"))?;
        let mean = row_values.iter().sum::<f64>() / f64::from(dimension_as_u32);
        if !mean.is_finite() {
            return Err(SemanticIrError::NonFiniteValue("input transform"));
        }
        for value in row_values {
            *value -= mean;
        }
    }
    Ok(())
}

fn select_keys(
    program: &SemanticProgram,
    input: &ReferenceInput,
    scores: &[f64],
    query: usize,
) -> Result<Vec<usize>, SemanticIrError> {
    let mut visible = Vec::new();
    for key in 0..input.key_count {
        let masked = match (&program.mask, &input.external_mask) {
            (MaskRule::Unmasked, _) => false,
            (MaskRule::Causal, _) => key > query,
            (MaskRule::External { .. }, Some(mask)) => !mask[query * input.key_count + key],
            (MaskRule::External { .. }, None) => return Err(SemanticIrError::MissingExternalMask),
        };
        let selected_by_window = match program.selection {
            SelectionRule::Window { radius } => query.abs_diff(key) <= radius,
            SelectionRule::All | SelectionRule::TopK { .. } => true,
        };
        if !masked && selected_by_window {
            visible.push(key);
        }
    }
    if visible.is_empty() {
        return Err(SemanticIrError::EmptyVisibleSelection(query));
    }
    if let SelectionRule::TopK { k } = program.selection {
        if k > visible.len() {
            return Err(SemanticIrError::InvalidField(
                "selection.k exceeds visible keys",
            ));
        }
        visible.sort_by(|&left, &right| {
            scores[query * input.key_count + right]
                .total_cmp(&scores[query * input.key_count + left])
                .then_with(|| left.cmp(&right))
        });
        visible.truncate(k.min(visible.len()));
        visible.sort_unstable();
    }
    Ok(visible)
}

/// Normalization diagnostics emitted by the independent reference path.
#[derive(Debug, Clone, Copy)]
pub enum NormalizationSummary {
    /// One probability-simplex log-sum-exp value.
    Softmax {
        /// Log-sum-exp of selected scores.
        log_sum_exp: f64,
    },
    /// Two normalizers used by signed-difference weighting.
    SignedDifference {
        /// Positive branch log-sum-exp.
        positive_log_sum_exp: f64,
        /// Negative branch log-sum-exp.
        negative_log_sum_exp: f64,
    },
}

impl PartialEq for NormalizationSummary {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Softmax { log_sum_exp: left }, Self::Softmax { log_sum_exp: right }) => {
                left.to_bits() == right.to_bits()
            }
            (
                Self::SignedDifference {
                    positive_log_sum_exp: left_positive,
                    negative_log_sum_exp: left_negative,
                },
                Self::SignedDifference {
                    positive_log_sum_exp: right_positive,
                    negative_log_sum_exp: right_negative,
                },
            ) => {
                left_positive.to_bits() == right_positive.to_bits()
                    && left_negative.to_bits() == right_negative.to_bits()
            }
            _ => false,
        }
    }
}

impl Eq for NormalizationSummary {}

impl std::hash::Hash for NormalizationSummary {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        match self {
            Self::Softmax { log_sum_exp } => {
                0_u8.hash(state);
                log_sum_exp.to_bits().hash(state);
            }
            Self::SignedDifference {
                positive_log_sum_exp,
                negative_log_sum_exp,
            } => {
                1_u8.hash(state);
                positive_log_sum_exp.to_bits().hash(state);
                negative_log_sum_exp.to_bits().hash(state);
            }
        }
    }
}

fn weights_for(
    rule: WeightRule,
    scores: &[f64],
) -> Result<(Vec<f64>, NormalizationSummary), SemanticIrError> {
    match rule {
        WeightRule::Softmax => {
            let (weights, lse) = stable_softmax(scores, 1.0)?;
            Ok((weights, NormalizationSummary::Softmax { log_sum_exp: lse }))
        }
        WeightRule::SignedDifference {
            positive_scale,
            negative_scale,
        } => {
            let (positive, positive_lse) = stable_softmax(scores, positive_scale)?;
            let (negative, negative_lse) = stable_softmax(scores, negative_scale)?;
            let weights = positive
                .into_iter()
                .zip(negative)
                .map(|(positive, negative)| positive - negative)
                .collect::<Vec<_>>();
            Ok((
                weights,
                NormalizationSummary::SignedDifference {
                    positive_log_sum_exp: positive_lse,
                    negative_log_sum_exp: negative_lse,
                },
            ))
        }
    }
}

fn stable_softmax(scores: &[f64], scale: f64) -> Result<(Vec<f64>, f64), SemanticIrError> {
    let maximum = scores
        .iter()
        .copied()
        .max_by(f64::total_cmp)
        .ok_or(SemanticIrError::InvalidField("empty score selection"))?;
    if !maximum.is_finite() {
        return Err(SemanticIrError::NonFiniteValue("softmax maximum"));
    }
    let mut weights = Vec::with_capacity(scores.len());
    let mut sum = 0.0_f64;
    for &score in scores {
        let scaled = score * scale;
        let scaled_maximum = maximum * scale;
        if !scaled.is_finite() || !scaled_maximum.is_finite() {
            return Err(SemanticIrError::NonFiniteValue("softmax scale"));
        }
        let weight = (scaled - scaled_maximum).exp();
        if !weight.is_finite() {
            return Err(SemanticIrError::NonFiniteValue("softmax exponential"));
        }
        weights.push(weight);
        sum += weight;
    }
    if !sum.is_finite() || sum <= 0.0 {
        return Err(SemanticIrError::NonFiniteValue("softmax normalizer"));
    }
    let log_sum_exp = scaled_maximum_for_lse(maximum, scale, sum)?;
    for weight in &mut weights {
        *weight /= sum;
    }
    Ok((weights, log_sum_exp))
}

fn scaled_maximum_for_lse(maximum: f64, scale: f64, sum: f64) -> Result<f64, SemanticIrError> {
    let result = maximum * scale + sum.ln();
    if result.is_finite() {
        Ok(result)
    } else {
        Err(SemanticIrError::NonFiniteValue("softmax log-sum-exp"))
    }
}

/// Reference output with weights and normalizer diagnostics retained.
#[derive(Debug, Clone, PartialEq)]
pub struct ReferenceOutput {
    output: Vec<f64>,
    weights: Vec<f64>,
    normalizations: Vec<NormalizationSummary>,
    selected_keys: Vec<Vec<usize>>,
}

impl ReferenceOutput {
    /// Row-major mixed output.
    #[must_use]
    pub fn output(&self) -> &[f64] {
        &self.output
    }

    /// Row-major query/key weights, with unselected entries equal to zero.
    #[must_use]
    pub fn weights(&self) -> &[f64] {
        &self.weights
    }

    /// Per-query normalization diagnostics.
    #[must_use]
    pub fn normalizations(&self) -> &[NormalizationSummary] {
        &self.normalizations
    }

    /// Deterministically ordered selected key indices per query.
    #[must_use]
    pub fn selected_keys(&self) -> &[Vec<usize>] {
        &self.selected_keys
    }
}

/// Stable dual-lane fingerprint of a semantic program.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SemanticFingerprint {
    primary: u64,
    secondary: u64,
    length: u64,
}

impl SemanticFingerprint {
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

impl Display for SemanticFingerprint {
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
