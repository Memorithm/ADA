use super::*;

fn explicit_workload() -> WorkloadContract {
    let geometry = AttentionGeometry::new(GeometrySpec {
        sequence_lengths: SequenceLengths::ragged(vec![4, 2], vec![8, 3]).unwrap(),
        query_heads: 8,
        kv_heads: 2,
        qk_dimension: Some(64),
        value_dimension: 96,
        topology: AttentionTopology::SelfAttention,
        head_grouping: HeadGrouping::GroupedQuery { queries_per_kv: 4 },
    })
    .unwrap();
    WorkloadContract::new(
        geometry,
        WorkloadOptions {
            mode: WorkloadMode::ChunkedDecode,
            mask: MaskSpec::new(MaskKind::External {
                identity: "mask-v1".into(),
            })
            .unwrap(),
            positions: PositionInfo::Rotary { dimension: 64 },
            score_bias: ScoreBiasSpec::Named {
                identity: "alibi-v2".into(),
            },
            precision: PrecisionPolicy::new(
                ScalarPrecision::BF16,
                ScalarPrecision::F32,
                ScalarPrecision::BF16,
                ScalarPrecision::BF16,
            ),
            layout: TensorLayout::new(
                MatrixLayout::RowMajor,
                MatrixLayout::Strided {
                    row_stride: 128,
                    column_stride: 1,
                },
                MatrixLayout::Tiled {
                    tile_rows: 16,
                    tile_columns: 32,
                },
                MatrixLayout::RowMajor,
            ),
            kv_representation: KvRepresentation::LatentCompressed(
                LatentKvSpec::new(
                    32,
                    "key-decode-v1",
                    "value-decode-v1",
                    LatentPositionHandling::Separate {
                        identity: "rope-cache-v1".into(),
                    },
                )
                .unwrap(),
            ),
            kv_cache: KvCacheSpec::Paged {
                page_size: 16,
                physical_pages: 128,
                block_table_identity: "block-table-v3".into(),
            },
            kv_indexing: KvIndexing::LogicalToPhysical {
                identity: "page-map-v3".into(),
            },
            inputs: InputRepresentation::ExplicitQkv,
            state: StateSpec::Stateless,
        },
    )
    .unwrap()
}

#[test]
fn general_contract_covers_geometry_modes_and_representation_without_execution() {
    let workload = explicit_workload();
    assert_eq!(workload.version(), WORKLOAD_CONTRACT_VERSION);
    assert_eq!(workload.geometry().sequence_lengths().batch_count(), 2);
    assert!(workload.geometry().sequence_lengths().is_ragged());
    assert_eq!(
        workload.geometry().head_grouping(),
        HeadGrouping::GroupedQuery { queries_per_kv: 4 }
    );
    assert_eq!(
        HeadGrouping::from_head_counts(16, 1).unwrap(),
        HeadGrouping::MultiQuery
    );
    assert!(matches!(workload.kv_cache(), KvCacheSpec::Paged { .. }));
    assert!(matches!(
        workload.kv_representation(),
        KvRepresentation::LatentCompressed(_)
    ));
}

#[test]
fn canonical_text_round_trip_and_fingerprint_are_deterministic() {
    let workload = explicit_workload();
    let text = workload.to_canonical_text();
    assert_eq!(text, workload.to_canonical_text());
    let decoded = WorkloadContract::from_canonical_text(&text).unwrap();
    assert_eq!(decoded, workload);
    assert_eq!(decoded.to_canonical_text(), text);
    assert_eq!(decoded.fingerprint(), workload.fingerprint());
    assert_eq!(format!("{}", workload.fingerprint()).len(), 16 * 3 + 2);
}

#[test]
fn fingerprint_separates_experiment_mode_from_geometry() {
    let workload = explicit_workload();
    let mut changed_mode = workload.clone();
    changed_mode.options.mode = WorkloadMode::Prefill;
    changed_mode.validate().unwrap();
    assert_ne!(workload.fingerprint(), changed_mode.fingerprint());
}

#[test]
fn historical_a1_adapter_is_explicit_and_does_not_infer_qk_dimension() {
    let case = ada_core::AttentionCase {
        logits: vec![1.0, -2.0, 3.0],
        values: vec![0.0, 1.0, 2.0],
        head_dim: 1,
    };
    let workload = WorkloadContract::from_a1_case(&case).unwrap();
    assert_eq!(
        workload.geometry().topology(),
        AttentionTopology::HistoricalA1
    );
    assert_eq!(workload.geometry().qk_dimension(), None);
    assert_eq!(workload.geometry().sequence_lengths().query_length(), 1);
    assert_eq!(workload.geometry().sequence_lengths().kv_length(), 3);
    assert!(matches!(
        workload.inputs(),
        InputRepresentation::HistoricalA1ScalarFixture
    ));
    assert_eq!(case.validate(), Ok(()));
}

#[test]
fn invalid_contracts_fail_closed() {
    assert!(SequenceLengths::ragged(vec![2], vec![2, 3]).is_err());
    assert!(SequenceLengths::uniform(1, 0, 4).is_err());
    assert!(HeadGrouping::from_head_counts(5, 2).is_err());
    assert!(HeadGrouping::MultiQuery.validate(1, 1).is_err());
    assert!(
        MaskSpec::new(MaskKind::External {
            identity: "contains whitespace".into(),
        })
        .is_err()
    );

    let geometry = AttentionGeometry::new(GeometrySpec {
        sequence_lengths: SequenceLengths::uniform(1, 1, 4).unwrap(),
        query_heads: 2,
        kv_heads: 1,
        qk_dimension: Some(8),
        value_dimension: 8,
        topology: AttentionTopology::SelfAttention,
        head_grouping: HeadGrouping::MultiQuery,
    })
    .unwrap();
    assert!(
        WorkloadContract::new(
            geometry,
            WorkloadOptions {
                mode: WorkloadMode::Decode,
                ..WorkloadOptions::default()
            },
        )
        .is_err()
    );
}

#[test]
fn malformed_and_adversarial_text_is_rejected_without_partial_contract() {
    let workload = explicit_workload();
    let mut text = workload.to_canonical_text();
    text.push_str("unknown=field\n");
    assert!(WorkloadContract::from_canonical_text(&text).is_err());

    let text = workload
        .to_canonical_text()
        .replace("query_heads=8", "query_heads=not-a-number");
    assert!(WorkloadContract::from_canonical_text(&text).is_err());

    let text = workload
        .to_canonical_text()
        .replace("ADA-WORKLOAD-V1", "ADA-WORKLOAD-V999");
    assert!(WorkloadContract::from_canonical_text(&text).is_err());
}

#[test]
fn decode_requires_cache_and_single_query_token() {
    let geometry = AttentionGeometry::new(GeometrySpec {
        sequence_lengths: SequenceLengths::uniform(1, 1, 8).unwrap(),
        query_heads: 1,
        kv_heads: 1,
        qk_dimension: Some(16),
        value_dimension: 16,
        topology: AttentionTopology::SelfAttention,
        head_grouping: HeadGrouping::MultiHead,
    })
    .unwrap();
    assert!(
        WorkloadContract::new(
            geometry,
            WorkloadOptions {
                mode: WorkloadMode::Decode,
                kv_cache: KvCacheSpec::Contiguous,
                ..WorkloadOptions::default()
            },
        )
        .is_ok()
    );
}
