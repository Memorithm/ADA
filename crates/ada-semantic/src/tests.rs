use super::*;

fn semantic_id(family: SemanticFamily, name: &str) -> SemanticId {
    SemanticId::new(family, name, 1).expect("test semantic identity is valid")
}

fn input(external_mask: Option<Vec<bool>>) -> ReferenceInput {
    ReferenceInput::new(ReferenceInputSpec {
        query_count: 2,
        key_count: 3,
        q_dimension: 2,
        value_dimension: 1,
        queries: vec![1.0, 0.0, 0.0, 1.0],
        keys: vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        values: vec![10.0, 20.0, 30.0],
        external_mask,
    })
    .expect("test reference input is valid")
}

fn standard() -> SemanticProgram {
    SemanticProgram::standard_softmax(
        semantic_id(SemanticFamily::StandardSoftmax, "dense-softmax"),
        MaskRule::Unmasked,
        SelectionRule::All,
        1.0,
    )
    .expect("test semantic program is valid")
}

fn f64_workload() -> WorkloadContract {
    let geometry = ada_workload::AttentionGeometry::new(ada_workload::GeometrySpec {
        sequence_lengths: ada_workload::SequenceLengths::uniform(1, 2, 3).unwrap(),
        query_heads: 1,
        kv_heads: 1,
        qk_dimension: Some(2),
        value_dimension: 1,
        topology: AttentionTopology::SelfAttention,
        head_grouping: ada_workload::HeadGrouping::MultiHead,
    })
    .unwrap();
    WorkloadContract::new(
        geometry,
        ada_workload::WorkloadOptions {
            precision: ada_workload::PrecisionPolicy::new(
                ScalarPrecision::F64,
                ScalarPrecision::F64,
                ScalarPrecision::F64,
                ScalarPrecision::F64,
            ),
            inputs: InputRepresentation::ExplicitQkv,
            ..ada_workload::WorkloadOptions::default()
        },
    )
    .unwrap()
}

#[test]
fn standard_reference_matches_direct_f64_softmax() {
    let output = standard().evaluate(&input(None)).unwrap();
    let denominator = 2.0 * 1.0_f64.exp() + 1.0;
    let expected_first = (10.0 * 1.0_f64.exp() + 20.0 + 30.0 * 1.0_f64.exp()) / denominator;
    let expected_second = (10.0 + 20.0 * 1.0_f64.exp() + 30.0 * 1.0_f64.exp()) / denominator;
    // The reference uses max-shifted exponentials; the direct expression
    // uses an unshifted denominator. Both are f64 paths, but Miri's
    // soft-float exp implementation can accumulate more rounding error
    // than native libm while remaining well within f64 reference accuracy.
    let direct_softmax_tolerance = 1.0e-13;
    assert!((output.output()[0] - expected_first).abs() < direct_softmax_tolerance);
    assert!((output.output()[1] - expected_second).abs() < direct_softmax_tolerance);
    for row in output.weights().chunks_exact(3) {
        assert!((row.iter().sum::<f64>() - 1.0).abs() < 1.0e-15);
    }
    assert!(matches!(
        output.normalizations()[0],
        NormalizationSummary::Softmax { .. }
    ));
}

#[test]
fn two_composable_weighting_semantics_are_distinct() {
    let signed = SemanticProgram::signed_difference(
        semantic_id(SemanticFamily::DifferentialSigned, "signed-difference"),
        MaskRule::Unmasked,
        SelectionRule::All,
        1.0,
        1.0,
        0.5,
    )
    .unwrap();
    let softmax = standard().evaluate(&input(None)).unwrap();
    let signed_output = signed.evaluate(&input(None)).unwrap();
    assert_ne!(softmax.output(), signed_output.output());
    assert!(
        signed_output
            .weights()
            .chunks_exact(3)
            .all(|row| row.iter().sum::<f64>().abs() < 1.0e-15)
    );
    assert!(matches!(
        signed_output.normalizations()[0],
        NormalizationSummary::SignedDifference { .. }
    ));
}

#[test]
fn mask_window_and_top_k_selection_are_explicit_and_deterministic() {
    let causal = SemanticProgram::standard_softmax(
        semantic_id(SemanticFamily::StandardSoftmax, "causal-softmax"),
        MaskRule::Causal,
        SelectionRule::All,
        1.0,
    )
    .unwrap();
    let causal_output = causal.evaluate(&input(None)).unwrap();
    assert_eq!(causal_output.selected_keys(), &[vec![0], vec![0, 1]]);

    let top_k = SemanticProgram::standard_softmax(
        semantic_id(SemanticFamily::StandardSoftmax, "top-k-softmax"),
        MaskRule::Unmasked,
        SelectionRule::TopK { k: 2 },
        1.0,
    )
    .unwrap();
    let top_k_output = top_k.evaluate(&input(None)).unwrap();
    assert_eq!(top_k_output.selected_keys(), &[vec![0, 2], vec![1, 2]]);

    let window = SemanticProgram::standard_softmax(
        semantic_id(SemanticFamily::StandardSoftmax, "window-softmax"),
        MaskRule::Unmasked,
        SelectionRule::Window { radius: 0 },
        1.0,
    )
    .unwrap();
    assert_eq!(
        window.evaluate(&input(None)).unwrap().selected_keys(),
        &[vec![0], vec![1]]
    );
}

#[test]
fn input_transform_and_external_mask_have_reference_paths() {
    let descriptor = SemanticDescriptor::new(
        semantic_id(SemanticFamily::Experimental, "centered-softmax"),
        MaskContract::Bidirectional,
        StateContract::Stateless,
        WeightContract::ProbabilitySimplex,
    );
    let centered = SemanticProgram::new(SemanticProgramSpec {
        descriptor,
        input_transform: InputTransform::CenterRows,
        affinity: AffinityRule::ScaledDotProduct { scale: 1.0 },
        mask: MaskRule::Unmasked,
        selection: SelectionRule::All,
        weight: WeightRule::Softmax,
        value_mix: ValueMixRule::WeightedSum,
        output: OutputRule::Identity,
    })
    .unwrap();
    assert!(centered.evaluate(&input(None)).is_ok());

    let external = SemanticProgram::standard_softmax(
        semantic_id(SemanticFamily::StandardSoftmax, "external-softmax"),
        MaskRule::External {
            identity: "mask-v1".into(),
        },
        SelectionRule::All,
        1.0,
    )
    .unwrap();
    assert_eq!(
        external.evaluate(&input(None)),
        Err(SemanticIrError::MissingExternalMask)
    );
    assert_eq!(
        external.evaluate(&input(Some(vec![false; 6]))),
        Err(SemanticIrError::EmptyVisibleSelection(0))
    );
    let masked = external.evaluate(&input(Some(vec![true, true, false, false, true, true])));
    assert_eq!(masked.unwrap().selected_keys(), &[vec![0, 1], vec![1, 2]]);
}

#[test]
fn canonical_codec_round_trip_and_fingerprint_are_stable() {
    let program = SemanticProgram::signed_difference(
        semantic_id(SemanticFamily::DifferentialSigned, "signed.codec"),
        MaskRule::External {
            identity: "external-mask-v1".into(),
        },
        SelectionRule::TopK { k: 2 },
        0.125,
        1.0,
        0.5,
    )
    .unwrap();
    let text = program.to_canonical_text();
    assert_eq!(text, program.to_canonical_text());
    let decoded = SemanticProgram::from_canonical_text(&text).unwrap();
    assert_eq!(decoded, program);
    assert_eq!(decoded.to_canonical_text(), text);
    assert_eq!(decoded.fingerprint(), program.fingerprint());
    assert_eq!(format!("{}", program.fingerprint()).len(), 16 * 3 + 2);

    let mut malformed = text.clone();
    malformed.push_str("unknown=field\n");
    assert!(SemanticProgram::from_canonical_text(&malformed).is_err());
    let duplicate = format!("{text}output=identity\n");
    assert!(SemanticProgram::from_canonical_text(&duplicate).is_err());
    let future_version = text.replacen("ADA-SEMANTIC-V1", "ADA-SEMANTIC-V2", 1);
    assert_eq!(
        SemanticProgram::from_canonical_text(&future_version),
        Err(SemanticIrError::UnsupportedVersion(2))
    );
}

#[test]
fn invalid_programs_and_inputs_fail_closed() {
    let descriptor = SemanticDescriptor::new(
        semantic_id(SemanticFamily::StandardSoftmax, "bad-descriptor"),
        MaskContract::Bidirectional,
        StateContract::Stateless,
        WeightContract::Signed,
    );
    assert!(
        SemanticProgram::new(SemanticProgramSpec {
            descriptor,
            input_transform: InputTransform::Identity,
            affinity: AffinityRule::ScaledDotProduct { scale: 1.0 },
            mask: MaskRule::Unmasked,
            selection: SelectionRule::All,
            weight: WeightRule::Softmax,
            value_mix: ValueMixRule::WeightedSum,
            output: OutputRule::Identity,
        })
        .is_err()
    );
    assert!(
        SemanticProgram::standard_softmax(
            semantic_id(SemanticFamily::StandardSoftmax, "bad-scale"),
            MaskRule::Unmasked,
            SelectionRule::All,
            0.0,
        )
        .is_err()
    );
    assert!(
        ReferenceInput::new(ReferenceInputSpec {
            query_count: 1,
            key_count: 1,
            q_dimension: 2,
            value_dimension: 1,
            queries: vec![0.0],
            keys: vec![0.0, 0.0],
            values: vec![0.0],
            external_mask: None,
        })
        .is_err()
    );
    assert!(
        ReferenceInput::new(ReferenceInputSpec {
            query_count: 1,
            key_count: 1,
            q_dimension: 2,
            value_dimension: 1,
            queries: vec![f64::NAN, 0.0],
            keys: vec![0.0, 0.0],
            values: vec![0.0],
            external_mask: None,
        })
        .is_err()
    );
}

#[test]
fn floating_point_identity_uses_bits_for_hashable_contracts() {
    let nan = AffinityRule::ScaledDotProduct { scale: f64::NAN };
    assert_eq!(nan, nan);
    assert_ne!(
        AffinityRule::ScaledDotProduct { scale: 0.0 },
        AffinityRule::ScaledDotProduct { scale: -0.0 }
    );
    let signed = WeightRule::SignedDifference {
        positive_scale: 1.0,
        negative_scale: f64::NAN,
    };
    assert_eq!(signed, signed);
}

#[test]
fn extreme_affinity_overflow_is_not_silently_accepted() {
    let input = ReferenceInput::new(ReferenceInputSpec {
        query_count: 1,
        key_count: 1,
        q_dimension: 2,
        value_dimension: 1,
        queries: vec![1.0e308, 0.0],
        keys: vec![1.0e308, 0.0],
        values: vec![1.0],
        external_mask: None,
    })
    .unwrap();
    assert_eq!(
        standard().evaluate(&input),
        Err(SemanticIrError::NonFiniteValue("affinity"))
    );
}

#[test]
fn reference_score_budget_rejects_explosive_shapes_before_allocation() {
    let result = ReferenceInput::new(ReferenceInputSpec {
        query_count: MAX_REFERENCE_QUERIES,
        key_count: MAX_REFERENCE_QUERIES,
        q_dimension: 1,
        value_dimension: 1,
        queries: Vec::new(),
        keys: Vec::new(),
        values: Vec::new(),
        external_mask: None,
    });
    assert_eq!(
        result,
        Err(SemanticIrError::ExceedsLimit {
            field: "score matrix",
            value: MAX_REFERENCE_QUERIES * MAX_REFERENCE_QUERIES,
            maximum: MAX_REFERENCE_SCORE_ELEMENTS,
        })
    );
}

#[test]
fn top_k_cannot_silently_relax_after_masking() {
    let program = SemanticProgram::standard_softmax(
        semantic_id(SemanticFamily::StandardSoftmax, "masked-top-k"),
        MaskRule::External {
            identity: "mask-v1".into(),
        },
        SelectionRule::TopK { k: 2 },
        1.0,
    )
    .unwrap();
    assert_eq!(
        program.evaluate(&input(Some(vec![true, false, false, false, true, false]))),
        Err(SemanticIrError::InvalidField(
            "selection.k exceeds visible keys",
        ))
    );
}

#[test]
fn semantic_identity_is_separate_from_implementation_details_and_workload_limits_are_explicit() {
    let softmax = standard();
    let signed = SemanticProgram::signed_difference(
        semantic_id(SemanticFamily::DifferentialSigned, "signed-difference"),
        MaskRule::Unmasked,
        SelectionRule::All,
        1.0,
        1.0,
        0.5,
    )
    .unwrap();
    assert_ne!(softmax.descriptor().id(), signed.descriptor().id());
    assert!(softmax.validate_for_workload(&f64_workload()).is_ok());

    let f32_workload = WorkloadContract::new(
        f64_workload().geometry().clone(),
        ada_workload::WorkloadOptions {
            precision: ada_workload::PrecisionPolicy::new(
                ScalarPrecision::F32,
                ScalarPrecision::F32,
                ScalarPrecision::F32,
                ScalarPrecision::F32,
            ),
            ..ada_workload::WorkloadOptions::default()
        },
    )
    .unwrap();
    assert!(matches!(
        softmax.validate_for_workload(&f32_workload),
        Err(SemanticIrError::UnsupportedWorkload(_))
    ));
}
