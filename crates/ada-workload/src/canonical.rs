use std::collections::BTreeMap;

use super::{
    AttentionGeometry, AttentionTopology, GeometrySpec, HeadGrouping, InputRepresentation,
    KvCacheSpec, KvIndexing, KvRepresentation, LatentKvSpec, LatentPositionHandling,
    MAX_CANONICAL_TEXT_BYTES, MaskKind, MaskSpec, MatrixLayout, PositionInfo, PrecisionPolicy,
    ScalarPrecision, ScoreBiasSpec, SequenceLengths, StateSpec, TensorLayout,
    WORKLOAD_CONTRACT_VERSION, WorkloadContract, WorkloadContractError, WorkloadFingerprint,
    WorkloadMode, WorkloadOptions, hex_decode, hex_encode, parse_u16, parse_usize,
    parse_usize_list,
};

impl WorkloadContract {
    /// Canonical, deterministic text suitable for evidence artifacts and code
    /// review. Identifiers are hex encoded to keep parsing unambiguous.
    #[must_use]
    // Exhaustive canonical field order is intentionally kept together for auditability.
    #[allow(clippy::too_many_lines)]
    pub fn to_canonical_text(&self) -> String {
        let geometry = &self.geometry;
        let options = &self.options;
        let mut text = String::from("ADA-WORKLOAD-V1\n");
        let qk_dimension = geometry
            .qk_dimension
            .map_or_else(|| "none".into(), |value| value.to_string());
        let (mask_kind, mask_value) = match options.mask.kind() {
            MaskKind::None => ("none", "-".into()),
            MaskKind::Bidirectional => ("bidirectional", "-".into()),
            MaskKind::Causal => ("causal", "-".into()),
            MaskKind::External { identity } => ("external", hex_encode(identity)),
        };
        let (position_kind, position_value) = match &options.positions {
            PositionInfo::None => ("none", "-".into()),
            PositionInfo::Absolute { identity } => ("absolute", hex_encode(identity)),
            PositionInfo::Rotary { dimension } => ("rotary", dimension.to_string()),
            PositionInfo::Relative { identity } => ("relative", hex_encode(identity)),
        };
        let (bias_kind, bias_value) = match &options.score_bias {
            ScoreBiasSpec::None => ("none", "-".into()),
            ScoreBiasSpec::Named { identity } => ("named", hex_encode(identity)),
        };
        let (kv_kind, latent_dimension, key_reconstruction, value_reconstruction, latent_position) =
            match &options.kv_representation {
                KvRepresentation::Full => ("full", "-".into(), "-".into(), "-".into(), "-".into()),
                KvRepresentation::LatentCompressed(spec) => (
                    "latent",
                    spec.latent_dimension.to_string(),
                    hex_encode(&spec.key_reconstruction),
                    hex_encode(&spec.value_reconstruction),
                    latent_position_text(&spec.position_handling),
                ),
            };
        let (cache_kind, page_size, physical_pages, block_table) = match &options.kv_cache {
            KvCacheSpec::None => ("none", "-".into(), "-".into(), "-".into()),
            KvCacheSpec::Contiguous => ("contiguous", "-".into(), "-".into(), "-".into()),
            KvCacheSpec::Paged {
                page_size,
                physical_pages,
                block_table_identity,
            } => (
                "paged",
                page_size.to_string(),
                physical_pages.to_string(),
                hex_encode(block_table_identity),
            ),
        };
        let (index_kind, index_value) = match &options.kv_indexing {
            KvIndexing::Identity => ("identity", "-".into()),
            KvIndexing::LogicalToPhysical { identity } => ("mapping", hex_encode(identity)),
        };
        let (input_kind, input_value) = match &options.inputs {
            InputRepresentation::ExplicitQkv => ("explicit-qkv", "-".into()),
            InputRepresentation::PrecomputedScores { identity } => {
                ("precomputed-scores", hex_encode(identity))
            }
            InputRepresentation::HistoricalA1ScalarFixture => ("historical-a1", "-".into()),
        };
        let (state_kind, state_rows, state_columns, state_value) = match &options.state {
            StateSpec::Stateless => ("stateless", "-".into(), "-".into(), "-".into()),
            StateSpec::Recurrent {
                rows,
                columns,
                identity,
            } => (
                "recurrent",
                rows.to_string(),
                columns.to_string(),
                hex_encode(identity),
            ),
        };
        let precision = format!(
            "{}:{}:{}:{}",
            options.precision.input.as_text(),
            options.precision.accumulation.as_text(),
            options.precision.output.as_text(),
            options.precision.storage.as_text()
        );

        append_field(&mut text, "version", self.version.to_string());
        append_field(
            &mut text,
            "batch_count",
            geometry.sequence_lengths.batch_count().to_string(),
        );
        append_field(
            &mut text,
            "query_lengths",
            join_usizes(geometry.sequence_lengths.query_lengths()),
        );
        append_field(
            &mut text,
            "kv_lengths",
            join_usizes(geometry.sequence_lengths.kv_lengths()),
        );
        append_field(&mut text, "query_heads", geometry.query_heads.to_string());
        append_field(&mut text, "kv_heads", geometry.kv_heads.to_string());
        append_field(&mut text, "qk_dimension", qk_dimension);
        append_field(
            &mut text,
            "value_dimension",
            geometry.value_dimension.to_string(),
        );
        append_field(&mut text, "topology", geometry.topology.as_text());
        append_field(&mut text, "head_grouping", geometry.head_grouping.as_text());
        append_field(&mut text, "mode", options.mode.as_text());
        append_field(&mut text, "mask_kind", mask_kind);
        append_field(&mut text, "mask_value", mask_value);
        append_field(&mut text, "position_kind", position_kind);
        append_field(&mut text, "position_value", position_value);
        append_field(&mut text, "score_bias_kind", bias_kind);
        append_field(&mut text, "score_bias_value", bias_value);
        append_field(&mut text, "precision", precision);
        append_field(&mut text, "layout_query", options.layout.query.as_text());
        append_field(&mut text, "layout_key", options.layout.key.as_text());
        append_field(&mut text, "layout_value", options.layout.value.as_text());
        append_field(&mut text, "layout_output", options.layout.output.as_text());
        append_field(&mut text, "kv_kind", kv_kind);
        append_field(&mut text, "latent_dimension", latent_dimension);
        append_field(&mut text, "key_reconstruction", key_reconstruction);
        append_field(&mut text, "value_reconstruction", value_reconstruction);
        append_field(&mut text, "latent_position", latent_position);
        append_field(&mut text, "cache_kind", cache_kind);
        append_field(&mut text, "page_size", page_size);
        append_field(&mut text, "physical_pages", physical_pages);
        append_field(&mut text, "block_table", block_table);
        append_field(&mut text, "index_kind", index_kind);
        append_field(&mut text, "index_value", index_value);
        append_field(&mut text, "input_kind", input_kind);
        append_field(&mut text, "input_value", input_value);
        append_field(&mut text, "state_kind", state_kind);
        append_field(&mut text, "state_rows", state_rows);
        append_field(&mut text, "state_columns", state_columns);
        append_field(&mut text, "state_value", state_value);
        text
    }

    /// Decode canonical text after validating its exact schema and all
    /// cross-field invariants.
    ///
    /// # Errors
    ///
    /// Returns an error for an unknown version, malformed fields, duplicate or
    /// unknown keys, invalid nested metadata, or a cross-field mismatch.
    // Exact canonical schema validation is intentionally kept together for auditability.
    #[allow(clippy::too_many_lines)]
    pub fn from_canonical_text(text: &str) -> Result<Self, WorkloadContractError> {
        if text.len() > MAX_CANONICAL_TEXT_BYTES || text.contains('\r') {
            return Err(WorkloadContractError::MalformedCanonicalText(
                "canonical text exceeds its limit or contains CR".into(),
            ));
        }
        if !text.ends_with('\n') {
            return Err(WorkloadContractError::MalformedCanonicalText(
                "canonical text must end with a newline".into(),
            ));
        }
        let mut lines = text.lines();
        if lines.next() != Some("ADA-WORKLOAD-V1") {
            return Err(WorkloadContractError::MalformedCanonicalText(
                "missing ADA-WORKLOAD-V1 header".into(),
            ));
        }
        let mut fields = BTreeMap::new();
        for line in lines {
            let Some((key, value)) = line.split_once('=') else {
                return Err(WorkloadContractError::MalformedCanonicalText(
                    "field is missing '='".into(),
                ));
            };
            if key.is_empty() || value.contains('=') || fields.insert(key, value).is_some() {
                return Err(WorkloadContractError::MalformedCanonicalText(
                    "empty, duplicate, or ambiguous field".into(),
                ));
            }
        }
        let required = [
            "version",
            "batch_count",
            "query_lengths",
            "kv_lengths",
            "query_heads",
            "kv_heads",
            "qk_dimension",
            "value_dimension",
            "topology",
            "head_grouping",
            "mode",
            "mask_kind",
            "mask_value",
            "position_kind",
            "position_value",
            "score_bias_kind",
            "score_bias_value",
            "precision",
            "layout_query",
            "layout_key",
            "layout_value",
            "layout_output",
            "kv_kind",
            "latent_dimension",
            "key_reconstruction",
            "value_reconstruction",
            "latent_position",
            "cache_kind",
            "page_size",
            "physical_pages",
            "block_table",
            "index_kind",
            "index_value",
            "input_kind",
            "input_value",
            "state_kind",
            "state_rows",
            "state_columns",
            "state_value",
        ];
        if fields.len() != required.len() || required.iter().any(|key| !fields.contains_key(key)) {
            return Err(WorkloadContractError::MalformedCanonicalText(
                "canonical field set is incomplete or has unknown keys".into(),
            ));
        }
        let field = |key: &str| {
            fields.get(key).copied().ok_or_else(|| {
                WorkloadContractError::MalformedCanonicalText(format!("missing field {key}"))
            })
        };
        let version = parse_u16("version", field("version")?)?;
        if version != WORKLOAD_CONTRACT_VERSION {
            return Err(WorkloadContractError::UnsupportedVersion(version));
        }
        let batch_count = parse_usize("batch_count", field("batch_count")?)?;
        let query_lengths = parse_usize_list("query_lengths", field("query_lengths")?)?;
        let kv_lengths = parse_usize_list("kv_lengths", field("kv_lengths")?)?;
        if query_lengths.len() != batch_count || kv_lengths.len() != batch_count {
            return Err(WorkloadContractError::LengthMismatch {
                query_lengths: query_lengths.len(),
                kv_lengths: kv_lengths.len(),
            });
        }
        let sequence_lengths = SequenceLengths::ragged(query_lengths, kv_lengths)?;
        let geometry = AttentionGeometry::new(GeometrySpec {
            sequence_lengths,
            query_heads: parse_usize("query_heads", field("query_heads")?)?,
            kv_heads: parse_usize("kv_heads", field("kv_heads")?)?,
            qk_dimension: parse_optional_usize("qk_dimension", field("qk_dimension")?)?,
            value_dimension: parse_usize("value_dimension", field("value_dimension")?)?,
            topology: AttentionTopology::from_text(field("topology")?)?,
            head_grouping: HeadGrouping::from_text(field("head_grouping")?)?,
        })?;
        let options = WorkloadOptions {
            mode: WorkloadMode::from_text(field("mode")?)?,
            mask: parse_mask(field("mask_kind")?, field("mask_value")?)?,
            positions: parse_positions(field("position_kind")?, field("position_value")?)?,
            score_bias: parse_score_bias(field("score_bias_kind")?, field("score_bias_value")?)?,
            precision: parse_precision(field("precision")?)?,
            layout: TensorLayout::new(
                MatrixLayout::from_text(field("layout_query")?)?,
                MatrixLayout::from_text(field("layout_key")?)?,
                MatrixLayout::from_text(field("layout_value")?)?,
                MatrixLayout::from_text(field("layout_output")?)?,
            ),
            kv_representation: parse_kv_representation(
                field("kv_kind")?,
                field("latent_dimension")?,
                field("key_reconstruction")?,
                field("value_reconstruction")?,
                field("latent_position")?,
            )?,
            kv_cache: parse_cache(
                field("cache_kind")?,
                field("page_size")?,
                field("physical_pages")?,
                field("block_table")?,
            )?,
            kv_indexing: parse_indexing(field("index_kind")?, field("index_value")?)?,
            inputs: parse_inputs(field("input_kind")?, field("input_value")?)?,
            state: parse_state(
                field("state_kind")?,
                field("state_rows")?,
                field("state_columns")?,
                field("state_value")?,
            )?,
        };
        let contract = Self {
            version,
            geometry,
            options,
        };
        contract.validate()?;
        Ok(contract)
    }

    /// Stable dual-lane fingerprint of the canonical text representation.
    #[must_use]
    pub fn fingerprint(&self) -> WorkloadFingerprint {
        WorkloadFingerprint::of_bytes(self.to_canonical_text().as_bytes())
    }
}

fn append_field(text: &mut String, key: &str, value: impl std::fmt::Display) {
    text.push_str(key);
    text.push('=');
    text.push_str(&value.to_string());
    text.push('\n');
}

fn join_usizes(values: &[usize]) -> String {
    values
        .iter()
        .map(usize::to_string)
        .collect::<Vec<_>>()
        .join(",")
}

fn parse_optional_usize(field: &str, value: &str) -> Result<Option<usize>, WorkloadContractError> {
    if value == "none" {
        Ok(None)
    } else {
        Ok(Some(parse_usize(field, value)?))
    }
}

fn latent_position_text(position: &LatentPositionHandling) -> String {
    match position {
        LatentPositionHandling::BeforeCompression => "before".into(),
        LatentPositionHandling::AfterCompression => "after".into(),
        LatentPositionHandling::Separate { identity } => {
            format!("separate:{}", hex_encode(identity))
        }
    }
}

fn parse_mask(kind: &str, value: &str) -> Result<MaskSpec, WorkloadContractError> {
    let kind = match kind {
        "none" => MaskKind::None,
        "bidirectional" => MaskKind::Bidirectional,
        "causal" => MaskKind::Causal,
        "external" => MaskKind::External {
            identity: hex_decode("mask_value", value)?,
        },
        _ => {
            return Err(WorkloadContractError::MalformedCanonicalText(
                "unknown mask kind".into(),
            ));
        }
    };
    if !matches!(&kind, MaskKind::External { .. }) && value != "-" {
        return Err(WorkloadContractError::MalformedCanonicalText(
            "non-external mask must use '-' value".into(),
        ));
    }
    MaskSpec::new(kind)
}

fn parse_positions(kind: &str, value: &str) -> Result<PositionInfo, WorkloadContractError> {
    let position = match kind {
        "none" => PositionInfo::None,
        "absolute" => PositionInfo::Absolute {
            identity: hex_decode("position_value", value)?,
        },
        "rotary" => PositionInfo::Rotary {
            dimension: parse_usize("position_dimension", value)?,
        },
        "relative" => PositionInfo::Relative {
            identity: hex_decode("position_value", value)?,
        },
        _ => {
            return Err(WorkloadContractError::MalformedCanonicalText(
                "unknown position kind".into(),
            ));
        }
    };
    if matches!(&position, PositionInfo::None) && value != "-" {
        return Err(WorkloadContractError::MalformedCanonicalText(
            "none position must use '-' value".into(),
        ));
    }
    position.validate()?;
    Ok(position)
}

fn parse_score_bias(kind: &str, value: &str) -> Result<ScoreBiasSpec, WorkloadContractError> {
    let bias = match kind {
        "none" => ScoreBiasSpec::None,
        "named" => ScoreBiasSpec::Named {
            identity: hex_decode("score_bias_value", value)?,
        },
        _ => {
            return Err(WorkloadContractError::MalformedCanonicalText(
                "unknown score-bias kind".into(),
            ));
        }
    };
    if matches!(&bias, ScoreBiasSpec::None) && value != "-" {
        return Err(WorkloadContractError::MalformedCanonicalText(
            "none score bias must use '-' value".into(),
        ));
    }
    bias.validate()?;
    Ok(bias)
}

fn parse_precision(value: &str) -> Result<PrecisionPolicy, WorkloadContractError> {
    let parts: Vec<&str> = value.split(':').collect();
    if parts.len() != 4 {
        return Err(WorkloadContractError::MalformedCanonicalText(
            "precision must contain four components".into(),
        ));
    }
    Ok(PrecisionPolicy::new(
        ScalarPrecision::from_text(parts[0])?,
        ScalarPrecision::from_text(parts[1])?,
        ScalarPrecision::from_text(parts[2])?,
        ScalarPrecision::from_text(parts[3])?,
    ))
}

fn parse_latent_position(value: &str) -> Result<LatentPositionHandling, WorkloadContractError> {
    let position = match value {
        "before" => LatentPositionHandling::BeforeCompression,
        "after" => LatentPositionHandling::AfterCompression,
        value if value.starts_with("separate:") => LatentPositionHandling::Separate {
            identity: hex_decode("latent_position", &value[9..])?,
        },
        _ => {
            return Err(WorkloadContractError::MalformedCanonicalText(
                "unknown latent position handling".into(),
            ));
        }
    };
    position.validate()?;
    Ok(position)
}

fn parse_kv_representation(
    kind: &str,
    dimension: &str,
    key_reconstruction: &str,
    value_reconstruction: &str,
    position: &str,
) -> Result<KvRepresentation, WorkloadContractError> {
    match kind {
        "full" => {
            if dimension != "-"
                || key_reconstruction != "-"
                || value_reconstruction != "-"
                || position != "-"
            {
                return Err(WorkloadContractError::MalformedCanonicalText(
                    "full KV representation must use '-' latent fields".into(),
                ));
            }
            Ok(KvRepresentation::Full)
        }
        "latent" => Ok(KvRepresentation::LatentCompressed(LatentKvSpec::new(
            parse_usize("latent_dimension", dimension)?,
            hex_decode("key_reconstruction", key_reconstruction)?,
            hex_decode("value_reconstruction", value_reconstruction)?,
            parse_latent_position(position)?,
        )?)),
        _ => Err(WorkloadContractError::MalformedCanonicalText(
            "unknown KV representation".into(),
        )),
    }
}

fn parse_cache(
    kind: &str,
    page_size: &str,
    physical_pages: &str,
    block_table: &str,
) -> Result<KvCacheSpec, WorkloadContractError> {
    let cache = match kind {
        "none" => KvCacheSpec::None,
        "contiguous" => KvCacheSpec::Contiguous,
        "paged" => KvCacheSpec::Paged {
            page_size: parse_usize("page_size", page_size)?,
            physical_pages: parse_usize("physical_pages", physical_pages)?,
            block_table_identity: hex_decode("block_table", block_table)?,
        },
        _ => {
            return Err(WorkloadContractError::MalformedCanonicalText(
                "unknown KV cache kind".into(),
            ));
        }
    };
    if matches!(&cache, KvCacheSpec::None | KvCacheSpec::Contiguous)
        && (page_size != "-" || physical_pages != "-" || block_table != "-")
    {
        return Err(WorkloadContractError::MalformedCanonicalText(
            "non-paged KV cache must use '-' page fields".into(),
        ));
    }
    cache.validate()?;
    Ok(cache)
}

fn parse_indexing(kind: &str, value: &str) -> Result<KvIndexing, WorkloadContractError> {
    let indexing = match kind {
        "identity" => KvIndexing::Identity,
        "mapping" => KvIndexing::LogicalToPhysical {
            identity: hex_decode("index_value", value)?,
        },
        _ => {
            return Err(WorkloadContractError::MalformedCanonicalText(
                "unknown KV indexing kind".into(),
            ));
        }
    };
    if matches!(&indexing, KvIndexing::Identity) && value != "-" {
        return Err(WorkloadContractError::MalformedCanonicalText(
            "identity KV indexing must use '-' value".into(),
        ));
    }
    indexing.validate()?;
    Ok(indexing)
}

fn parse_inputs(kind: &str, value: &str) -> Result<InputRepresentation, WorkloadContractError> {
    let inputs = match kind {
        "explicit-qkv" => InputRepresentation::ExplicitQkv,
        "precomputed-scores" => InputRepresentation::PrecomputedScores {
            identity: hex_decode("input_value", value)?,
        },
        "historical-a1" => InputRepresentation::HistoricalA1ScalarFixture,
        _ => {
            return Err(WorkloadContractError::MalformedCanonicalText(
                "unknown input representation".into(),
            ));
        }
    };
    if matches!(
        &inputs,
        InputRepresentation::ExplicitQkv | InputRepresentation::HistoricalA1ScalarFixture
    ) && value != "-"
    {
        return Err(WorkloadContractError::MalformedCanonicalText(
            "non-precomputed input must use '-' value".into(),
        ));
    }
    inputs.validate()?;
    Ok(inputs)
}

fn parse_state(
    kind: &str,
    rows: &str,
    columns: &str,
    value: &str,
) -> Result<StateSpec, WorkloadContractError> {
    let state = match kind {
        "stateless" => StateSpec::Stateless,
        "recurrent" => StateSpec::Recurrent {
            rows: parse_usize("state_rows", rows)?,
            columns: parse_usize("state_columns", columns)?,
            identity: hex_decode("state_value", value)?,
        },
        _ => {
            return Err(WorkloadContractError::MalformedCanonicalText(
                "unknown state kind".into(),
            ));
        }
    };
    if matches!(&state, StateSpec::Stateless) && (rows != "-" || columns != "-" || value != "-") {
        return Err(WorkloadContractError::MalformedCanonicalText(
            "stateless state must use '-' fields".into(),
        ));
    }
    state.validate()?;
    Ok(state)
}
