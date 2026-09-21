//! Versioned attention surface identity: geometry + mask + mode.
//!
//! This identity is narrower than [`crate::WorkloadContract`]. It deliberately
//! excludes precision, layout, KV representation, cache, indexing, inputs, and
//! recurrent state so two realizations can share a surface without implying
//! full workload, semantic, survival, or adoption equivalence.
//!
//! Non-claims:
//! - surface identity is not an oracle, semantic program, or evidence lane;
//! - fingerprint equality is not novelty, usefulness, or FLAT-adoption proof;
//! - declaring a geometry/mask/mode does not make that mode executable.

use std::fmt::{Display, Formatter};

use super::{
    AttentionGeometry, AttentionTopology, GeometrySpec, HeadGrouping, MAX_CANONICAL_TEXT_BYTES,
    MaskKind, MaskSpec, SequenceLengths, WORKLOAD_CONTRACT_VERSION, WorkloadContract,
    WorkloadContractError, WorkloadFingerprint, WorkloadMode, hex_decode, hex_encode, parse_u16,
    parse_usize, parse_usize_list,
};

/// Canonical header for attention surface identity artifacts.
pub const ATTENTION_SURFACE_HEADER: &str = "ADA-ATTENTION-SURFACE-V1";
/// Version of the attention surface identity codec.
pub const ATTENTION_SURFACE_VERSION: u16 = 1;

/// Fail-closed attention-surface identity errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AttentionSurfaceError {
    /// Nested workload/geometry validation failed.
    Workload(WorkloadContractError),
    /// Canonical text was malformed or incomplete.
    MalformedCanonicalText(String),
    /// Unsupported surface identity version.
    UnsupportedVersion(u16),
}

impl Display for AttentionSurfaceError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Workload(error) => write!(formatter, "{error}"),
            Self::MalformedCanonicalText(reason) => {
                write!(formatter, "malformed attention surface text: {reason}")
            }
            Self::UnsupportedVersion(version) => {
                write!(formatter, "unsupported attention surface version {version}")
            }
        }
    }
}

impl std::error::Error for AttentionSurfaceError {}

impl From<WorkloadContractError> for AttentionSurfaceError {
    fn from(value: WorkloadContractError) -> Self {
        Self::Workload(value)
    }
}

/// Versioned identity of attention geometry, mask, and experiment mode.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct AttentionSurfaceIdentity {
    version: u16,
    geometry: AttentionGeometry,
    mask: MaskSpec,
    mode: WorkloadMode,
}

impl AttentionSurfaceIdentity {
    /// Construct a validated surface identity.
    ///
    /// # Errors
    ///
    /// Returns an error when geometry or mask validation fails.
    pub fn new(
        geometry: AttentionGeometry,
        mask: MaskSpec,
        mode: WorkloadMode,
    ) -> Result<Self, AttentionSurfaceError> {
        mask.validate()?;
        let identity = Self {
            version: ATTENTION_SURFACE_VERSION,
            geometry,
            mask,
            mode,
        };
        identity.validate()?;
        Ok(identity)
    }

    /// Extract the surface identity slice from a full workload contract.
    ///
    /// # Errors
    ///
    /// Returns an error when the workload fails validation.
    pub fn from_workload(workload: &WorkloadContract) -> Result<Self, AttentionSurfaceError> {
        workload.validate()?;
        Self::new(
            workload.geometry().clone(),
            workload.mask().clone(),
            workload.mode(),
        )
    }

    /// Validate version and nested geometry/mask constraints that are
    /// independent of the excluded workload fields.
    ///
    /// # Errors
    ///
    /// Returns an error for unsupported versions or invalid nested fields.
    pub fn validate(&self) -> Result<(), AttentionSurfaceError> {
        if self.version != ATTENTION_SURFACE_VERSION {
            return Err(AttentionSurfaceError::UnsupportedVersion(self.version));
        }
        self.mask.validate()?;
        // Decode-mode query-length/cache cross-checks live on WorkloadContract;
        // surface identity only carries geometry/mask/mode labels.
        let _ = WORKLOAD_CONTRACT_VERSION;
        Ok(())
    }

    /// Surface identity version.
    #[must_use]
    pub const fn version(&self) -> u16 {
        self.version
    }

    /// Attention geometry.
    #[must_use]
    pub const fn geometry(&self) -> &AttentionGeometry {
        &self.geometry
    }

    /// Mask contract.
    #[must_use]
    pub const fn mask(&self) -> &MaskSpec {
        &self.mask
    }

    /// Experiment mode.
    #[must_use]
    pub const fn mode(&self) -> WorkloadMode {
        self.mode
    }

    /// Canonical deterministic text for evidence and review.
    #[must_use]
    pub fn to_canonical_text(&self) -> String {
        let geometry = &self.geometry;
        let (mask_kind, mask_value) = match self.mask.kind() {
            MaskKind::None => ("none", "-".into()),
            MaskKind::Bidirectional => ("bidirectional", "-".into()),
            MaskKind::Causal => ("causal", "-".into()),
            MaskKind::External { identity } => ("external", hex_encode(identity)),
        };
        let qk_dimension = geometry
            .qk_dimension()
            .map_or_else(|| "none".into(), |value| value.to_string());
        let mut text = format!("{ATTENTION_SURFACE_HEADER}\n");
        append_field(&mut text, "version", self.version.to_string());
        append_field(&mut text, "mode", self.mode.as_text());
        append_field(&mut text, "topology", geometry.topology().as_text());
        append_field(
            &mut text,
            "head_grouping",
            geometry.head_grouping().as_text(),
        );
        append_field(&mut text, "query_heads", geometry.query_heads().to_string());
        append_field(&mut text, "kv_heads", geometry.kv_heads().to_string());
        append_field(&mut text, "qk_dimension", qk_dimension);
        append_field(
            &mut text,
            "value_dimension",
            geometry.value_dimension().to_string(),
        );
        append_field(
            &mut text,
            "batch_count",
            geometry.sequence_lengths().batch_count().to_string(),
        );
        append_field(
            &mut text,
            "query_lengths",
            join_usizes(geometry.sequence_lengths().query_lengths()),
        );
        append_field(
            &mut text,
            "kv_lengths",
            join_usizes(geometry.sequence_lengths().kv_lengths()),
        );
        append_field(&mut text, "mask_kind", mask_kind);
        append_field(&mut text, "mask_value", mask_value);
        text
    }

    /// Decode and validate canonical surface identity text.
    ///
    /// # Errors
    ///
    /// Returns an error for malformed, incomplete, oversized, or unsupported
    /// text.
    #[allow(clippy::too_many_lines)]
    pub fn from_canonical_text(text: &str) -> Result<Self, AttentionSurfaceError> {
        if text.len() > MAX_CANONICAL_TEXT_BYTES {
            return Err(AttentionSurfaceError::MalformedCanonicalText(
                "canonical text exceeds size limit".into(),
            ));
        }
        if text.contains('\r') {
            return Err(AttentionSurfaceError::MalformedCanonicalText(
                "CR characters are rejected".into(),
            ));
        }
        let mut lines = text.lines();
        let header = lines.next().ok_or_else(|| {
            AttentionSurfaceError::MalformedCanonicalText("missing header".into())
        })?;
        if header != ATTENTION_SURFACE_HEADER {
            return Err(AttentionSurfaceError::MalformedCanonicalText(
                "unknown header".into(),
            ));
        }
        let mut fields = std::collections::BTreeMap::new();
        for line in lines {
            if line.is_empty() {
                return Err(AttentionSurfaceError::MalformedCanonicalText(
                    "empty lines are rejected".into(),
                ));
            }
            let Some((key, value)) = line.split_once('=') else {
                return Err(AttentionSurfaceError::MalformedCanonicalText(
                    "line is not key=value".into(),
                ));
            };
            if fields.insert(key.to_owned(), value.to_owned()).is_some() {
                return Err(AttentionSurfaceError::MalformedCanonicalText(format!(
                    "duplicate field {key}"
                )));
            }
        }
        let required = [
            "version",
            "mode",
            "topology",
            "head_grouping",
            "query_heads",
            "kv_heads",
            "qk_dimension",
            "value_dimension",
            "batch_count",
            "query_lengths",
            "kv_lengths",
            "mask_kind",
            "mask_value",
        ];
        for key in required {
            if !fields.contains_key(key) {
                return Err(AttentionSurfaceError::MalformedCanonicalText(format!(
                    "missing field {key}"
                )));
            }
        }
        if fields.len() != required.len() {
            return Err(AttentionSurfaceError::MalformedCanonicalText(
                "unexpected extra fields".into(),
            ));
        }

        let version = parse_u16("version", require_field(&fields, "version")?)
            .map_err(AttentionSurfaceError::from)?;
        if version != ATTENTION_SURFACE_VERSION {
            return Err(AttentionSurfaceError::UnsupportedVersion(version));
        }
        let mode = WorkloadMode::from_text(require_field(&fields, "mode")?)
            .map_err(AttentionSurfaceError::from)?;
        let topology = AttentionTopology::from_text(require_field(&fields, "topology")?)
            .map_err(AttentionSurfaceError::from)?;
        let head_grouping = HeadGrouping::from_text(require_field(&fields, "head_grouping")?)
            .map_err(AttentionSurfaceError::from)?;
        let query_heads = parse_usize("query_heads", require_field(&fields, "query_heads")?)
            .map_err(AttentionSurfaceError::from)?;
        let kv_heads = parse_usize("kv_heads", require_field(&fields, "kv_heads")?)
            .map_err(AttentionSurfaceError::from)?;
        let qk_raw = require_field(&fields, "qk_dimension")?;
        let qk_dimension = if qk_raw == "none" {
            None
        } else {
            Some(parse_usize("qk_dimension", qk_raw).map_err(AttentionSurfaceError::from)?)
        };
        let value_dimension = parse_usize(
            "value_dimension",
            require_field(&fields, "value_dimension")?,
        )
        .map_err(AttentionSurfaceError::from)?;
        let batch_count = parse_usize("batch_count", require_field(&fields, "batch_count")?)
            .map_err(AttentionSurfaceError::from)?;
        let query_lengths =
            parse_usize_list("query_lengths", require_field(&fields, "query_lengths")?)
                .map_err(AttentionSurfaceError::from)?;
        let kv_lengths = parse_usize_list("kv_lengths", require_field(&fields, "kv_lengths")?)
            .map_err(AttentionSurfaceError::from)?;
        if query_lengths.len() != batch_count || kv_lengths.len() != batch_count {
            return Err(AttentionSurfaceError::MalformedCanonicalText(
                "batch_count does not match length vectors".into(),
            ));
        }
        let mask = parse_mask(
            require_field(&fields, "mask_kind")?,
            require_field(&fields, "mask_value")?,
        )?;
        let geometry = AttentionGeometry::new(GeometrySpec {
            sequence_lengths: SequenceLengths::ragged(query_lengths, kv_lengths)?,
            query_heads,
            kv_heads,
            qk_dimension,
            value_dimension,
            topology,
            head_grouping,
        })?;
        Self::new(geometry, mask, mode)
    }

    /// Stable dual-lane fingerprint over canonical surface text.
    #[must_use]
    pub fn fingerprint(&self) -> WorkloadFingerprint {
        WorkloadFingerprint::of_bytes(self.to_canonical_text().as_bytes())
    }
}

fn require_field<'a>(
    fields: &'a std::collections::BTreeMap<String, String>,
    key: &str,
) -> Result<&'a str, AttentionSurfaceError> {
    fields.get(key).map(String::as_str).ok_or_else(|| {
        AttentionSurfaceError::MalformedCanonicalText(format!("missing field {key}"))
    })
}

fn append_field(text: &mut String, key: &str, value: impl AsRef<str>) {
    text.push_str(key);
    text.push('=');
    text.push_str(value.as_ref());
    text.push('\n');
}

fn join_usizes(values: &[usize]) -> String {
    values
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join(",")
}

fn parse_mask(kind: &str, value: &str) -> Result<MaskSpec, AttentionSurfaceError> {
    let kind = match kind {
        "none" => MaskKind::None,
        "bidirectional" => MaskKind::Bidirectional,
        "causal" => MaskKind::Causal,
        "external" => MaskKind::External {
            identity: hex_decode("mask_value", value)?,
        },
        _ => {
            return Err(AttentionSurfaceError::MalformedCanonicalText(
                "unknown mask kind".into(),
            ));
        }
    };
    if !matches!(kind, MaskKind::External { .. }) && value != "-" {
        return Err(AttentionSurfaceError::MalformedCanonicalText(
            "non-external mask must use '-' value".into(),
        ));
    }
    Ok(MaskSpec::new(kind)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        InputRepresentation, KvCacheSpec, PrecisionPolicy, ScalarPrecision, TensorLayout,
        WorkloadOptions,
    };

    fn sample_geometry() -> AttentionGeometry {
        AttentionGeometry::new(GeometrySpec {
            sequence_lengths: SequenceLengths::uniform(2, 4, 8).unwrap(),
            query_heads: 8,
            kv_heads: 2,
            qk_dimension: Some(16),
            value_dimension: 16,
            topology: AttentionTopology::SelfAttention,
            head_grouping: HeadGrouping::GroupedQuery { queries_per_kv: 4 },
        })
        .unwrap()
    }

    #[test]
    fn surface_codec_round_trips_and_fingerprints_geometry_mask_mode() {
        let identity = AttentionSurfaceIdentity::new(
            sample_geometry(),
            MaskSpec::new(MaskKind::Causal).unwrap(),
            WorkloadMode::Prefill,
        )
        .unwrap();
        let text = identity.to_canonical_text();
        let decoded = AttentionSurfaceIdentity::from_canonical_text(&text).unwrap();
        assert_eq!(decoded, identity);
        assert_eq!(decoded.fingerprint(), identity.fingerprint());

        let changed_mode = AttentionSurfaceIdentity::new(
            identity.geometry().clone(),
            identity.mask().clone(),
            WorkloadMode::TrainingForward,
        )
        .unwrap();
        assert_ne!(identity.fingerprint(), changed_mode.fingerprint());
    }

    #[test]
    fn surface_identity_ignores_non_surface_workload_fields() {
        let geometry = sample_geometry();
        let left = WorkloadContract::new(
            geometry.clone(),
            WorkloadOptions {
                mode: WorkloadMode::Prefill,
                mask: MaskSpec::new(MaskKind::Causal).unwrap(),
                precision: PrecisionPolicy::new(
                    ScalarPrecision::F32,
                    ScalarPrecision::F32,
                    ScalarPrecision::F32,
                    ScalarPrecision::F32,
                ),
                ..WorkloadOptions::default()
            },
        )
        .unwrap();
        let right = WorkloadContract::new(
            geometry,
            WorkloadOptions {
                mode: WorkloadMode::Prefill,
                mask: MaskSpec::new(MaskKind::Causal).unwrap(),
                precision: PrecisionPolicy::new(
                    ScalarPrecision::BF16,
                    ScalarPrecision::F32,
                    ScalarPrecision::BF16,
                    ScalarPrecision::BF16,
                ),
                layout: TensorLayout::row_major(),
                inputs: InputRepresentation::ExplicitQkv,
                kv_cache: KvCacheSpec::None,
                ..WorkloadOptions::default()
            },
        )
        .unwrap();
        assert_ne!(left.fingerprint(), right.fingerprint());
        assert_eq!(
            AttentionSurfaceIdentity::from_workload(&left)
                .unwrap()
                .fingerprint(),
            AttentionSurfaceIdentity::from_workload(&right)
                .unwrap()
                .fingerprint()
        );
    }

    #[test]
    fn malformed_surface_text_fails_closed() {
        let identity = AttentionSurfaceIdentity::new(
            sample_geometry(),
            MaskSpec::none(),
            WorkloadMode::Prefill,
        )
        .unwrap();
        let mut text = identity.to_canonical_text();
        text.push_str("extra=1\n");
        assert!(AttentionSurfaceIdentity::from_canonical_text(&text).is_err());
        assert!(
            AttentionSurfaceIdentity::from_canonical_text(
                &identity
                    .to_canonical_text()
                    .replace(ATTENTION_SURFACE_HEADER, "ADA-ATTENTION-SURFACE-V9")
            )
            .is_err()
        );
    }
}
