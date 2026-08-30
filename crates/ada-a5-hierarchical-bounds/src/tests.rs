use super::*;
use ada_a4_qk_box::{branch_and_bound_entmax_qk_box, qk_box_entmax_case};

fn assert_close(left: f64, right: f64, tolerance: f64) {
    let scale = left.abs().max(right.abs()).max(1.0);
    assert!(
        (left - right).abs() <= tolerance * scale,
        "{left} != {right}"
    );
}

fn assert_hierarchical_matches_dense(
    case: &QueryKeyPagedCase,
    leaf_size: usize,
) -> HierarchicalResult {
    let index =
        build_hierarchical_key_index(&case.keys, case.head_dim, case.page_size, leaf_size).unwrap();
    let paged = qk_box_entmax_case(case).unwrap();
    let dense = dense_entmax(&paged.scores, case.alpha).unwrap();
    let candidate = branch_and_bound_entmax_hierarchical(case, &index).unwrap();
    assert_close(dense.tau, candidate.distribution.tau, 2.0e-12);
    for (&expected, &actual) in dense
        .probabilities
        .iter()
        .zip(candidate.distribution.probabilities.iter())
    {
        assert_close(expected, actual, 4.0e-12);
    }
    for (token, &probability) in dense.probabilities.iter().enumerate() {
        if probability > 1.0e-12 {
            assert!(candidate.loaded_tokens[token]);
        }
    }
    candidate
}

/// Eager and lazy controllers are bit-identical on native builds; under
/// Miri's software-float powf the two solver runs may differ in the last
/// ulp, so distributions compare within tolerance there.
fn assert_distributions_match(left: &EntmaxDistribution, right: &EntmaxDistribution) {
    #[cfg(not(miri))]
    assert_eq!(left, right);
    #[cfg(miri)]
    {
        assert_close(left.tau, right.tau, 2.0e-12);
        for (&a, &b) in left.probabilities.iter().zip(right.probabilities.iter()) {
            assert_close(a, b, 4.0e-12);
        }
    }
}

fn assert_lazy_matches_eager(
    case: &QueryKeyPagedCase,
    leaf_size: usize,
) -> (HierarchicalResult, LazyHierarchicalResult) {
    let index =
        build_hierarchical_key_index(&case.keys, case.head_dim, case.page_size, leaf_size).unwrap();

    let eager = branch_and_bound_entmax_hierarchical(case, &index).unwrap();

    let lazy = branch_and_bound_entmax_hierarchical_lazy(case, &index).unwrap();

    #[cfg(not(miri))]
    assert_eq!(eager.loaded_tokens, lazy.loaded_tokens);
    #[cfg(miri)]
    for (token, &probability) in eager.distribution.probabilities.iter().enumerate() {
        if probability > 1.0e-12 {
            assert!(
                lazy.loaded_tokens[token],
                "support token {token} missing from the lazy load set"
            );
        }
    }
    assert_distributions_match(&eager.distribution, &lazy.distribution);

    #[cfg(miri)]
    {
        // Soft-float ulp wobble can flip a borderline pruning decision,
        // so the loaded sets and counters may legitimately differ under
        // Miri. Exactness is still guaranteed by the distribution check
        // above plus the support coverage below.
        for (token, &probability) in eager.distribution.probabilities.iter().enumerate() {
            if probability > 1.0e-12 {
                assert!(
                    lazy.loaded_tokens[token],
                    "support token {token} not loaded"
                );
            }
        }
    }

    #[cfg(not(miri))]
    {
        assert_eq!(eager.metrics.nodes_expanded, lazy.metrics.nodes_expanded);
        assert_eq!(eager.metrics.subtrees_pruned, lazy.metrics.subtrees_pruned);
        assert_eq!(eager.metrics.leaves_loaded, lazy.metrics.leaves_loaded);
        assert_eq!(eager.metrics.tokens_loaded, lazy.metrics.tokens_loaded);
        assert_eq!(eager.metrics.tokens_pruned, lazy.metrics.tokens_pruned);
        assert_eq!(
            eager.metrics.threshold_solves,
            lazy.metrics.threshold_solves
        );

        assert_eq!(
            lazy.metrics.nodes_never_evaluated,
            lazy.metrics.nodes_total - lazy.metrics.bound_evaluations
        );

        assert!(lazy.metrics.bound_evaluations <= eager.metrics.bound_evaluations);
    }
    #[cfg(miri)]
    assert!(
        lazy.metrics.bound_evaluations <= eager.metrics.bound_evaluations,
        "lazy evaluation must never exceed the eager bound count"
    );

    (eager, lazy)
}

fn assert_priority_matches_dense(
    case: &QueryKeyPagedCase,
    leaf_size: usize,
) -> PriorityLazyHierarchicalResult {
    let index =
        build_hierarchical_key_index(&case.keys, case.head_dim, case.page_size, leaf_size).unwrap();

    let paged = qk_box_entmax_case(case).unwrap();
    let dense = dense_entmax(&paged.scores, case.alpha).unwrap();

    let candidate = branch_and_bound_entmax_hierarchical_priority_lazy(case, &index).unwrap();

    assert_close(dense.tau, candidate.distribution.tau, 2.0e-12);

    for (&expected, &actual) in dense
        .probabilities
        .iter()
        .zip(candidate.distribution.probabilities.iter())
    {
        assert_close(expected, actual, 4.0e-12);
    }

    for (token, &probability) in dense.probabilities.iter().enumerate() {
        if probability > 1.0e-12 {
            assert!(candidate.loaded_tokens[token]);
        }
    }

    assert!(candidate.metrics.bound_evaluations <= candidate.metrics.nodes_total);

    assert_eq!(
        candidate.metrics.nodes_never_evaluated,
        candidate.metrics.nodes_total - candidate.metrics.bound_evaluations
    );

    candidate
}

#[test]
fn priority_lazy_matches_historical_lazy_on_representative_case() {
    let keys = vec![
        10.0, 0.0, 9.5, 0.0, -10.0, 4.0, -11.0, -4.0, -20.0, 7.0, -21.0, -7.0, -30.0, 9.0, -31.0,
        -9.0,
    ];

    for alpha in [1.5, 2.0] {
        let case = QueryKeyPagedCase {
            query: vec![1.0, 0.0],
            keys: keys.clone(),
            head_dim: 2,
            page_size: 8,
            alpha,
            score_scale: 1.0,
        };

        let index =
            build_hierarchical_key_index(&case.keys, case.head_dim, case.page_size, 2).unwrap();

        let historical = branch_and_bound_entmax_hierarchical_lazy(&case, &index).unwrap();

        let priority = branch_and_bound_entmax_hierarchical_priority_lazy(&case, &index).unwrap();

        assert_eq!(historical.loaded_tokens, priority.loaded_tokens);
        assert_distributions_match(&historical.distribution, &priority.distribution);

        assert_eq!(
            historical.metrics.bound_evaluations,
            priority.metrics.bound_evaluations
        );

        assert_eq!(
            historical.metrics.tokens_loaded,
            priority.metrics.tokens_loaded
        );

        assert_eq!(
            historical.metrics.tokens_pruned,
            priority.metrics.tokens_pruned
        );

        assert!(
            priority.metrics.frontier_logical_operations()
                < historical.metrics.bound_evaluations + historical.metrics.bound_cache_hits
        );
    }
}

#[test]
fn priority_lazy_dense_fallback_remains_exact() {
    let case = QueryKeyPagedCase {
        query: vec![1.0, 0.0],
        keys: vec![
            1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0,
        ],
        head_dim: 2,
        page_size: 8,
        alpha: 2.0,
        score_scale: 1.0,
    };

    let priority = assert_priority_matches_dense(&case, 2);

    assert_eq!(priority.metrics.tokens_loaded, 8);
    assert_eq!(priority.metrics.tokens_pruned, 0);
    assert_eq!(
        priority.metrics.leaves_loaded,
        priority.metrics.leaves_total
    );
    assert_eq!(
        priority.metrics.bound_evaluations,
        priority.metrics.nodes_total
    );
    assert_eq!(priority.metrics.nodes_never_evaluated, 0);
}

#[test]
fn exhaustive_small_hierarchies_priority_match_dense_oracle() {
    const ROWS: [[f64; 2]; 3] = [[-1.0, -1.0], [0.0, 0.0], [1.0, 1.0]];
    const QUERIES: [[f64; 2]; 3] = [[1.0, -1.0], [-1.0, 1.0], [1.0, 1.0]];

    let state_count = ROWS.len().pow(4);

    for state in 0..state_count {
        let mut code = state;
        let mut keys = Vec::with_capacity(8);

        for _ in 0..4 {
            keys.extend_from_slice(&ROWS[code % ROWS.len()]);
            code /= ROWS.len();
        }

        for query in QUERIES {
            for alpha in [1.5, 2.0] {
                let case = QueryKeyPagedCase {
                    query: query.to_vec(),
                    keys: keys.clone(),
                    head_dim: 2,
                    page_size: 4,
                    alpha,
                    score_scale: 2.0_f64.sqrt().recip(),
                };

                assert_priority_matches_dense(&case, 1);
                assert_priority_matches_dense(&case, 2);
            }
        }
    }
}

#[test]
fn hierarchy_bounds_dominate_dense_scores_and_tighten_downward() {
    let case = QueryKeyPagedCase {
        query: vec![2.0, -3.0, 0.5],
        keys: vec![
            1.0, 4.0, -2.0, 3.0, -1.0, 5.0, -4.0, 2.0, 1.0, 0.5, -3.0, 7.0, 2.0, 2.0, 2.0, -1.0,
            -4.0, 3.0, 5.0, 0.0, -2.0, 1.0, 3.0, -5.0,
        ],
        head_dim: 3,
        page_size: 4,
        alpha: 1.5,
        score_scale: 3.0_f64.sqrt().recip(),
    };
    let index = build_hierarchical_key_index(&case.keys, 3, 4, 1).unwrap();
    let bounds = query_hierarchical_upper_bounds(&case, &index).unwrap();
    let scores = dense_qk_scores(&case).unwrap();
    validate_bounds_against_dense_scores(&index, &bounds, &scores).unwrap();

    for (node_index, node) in index.nodes.iter().enumerate() {
        if let Some((left, right)) = node.children() {
            assert!(bounds[left] <= bounds[node_index] + 1.0e-12);
            assert!(bounds[right] <= bounds[node_index] + 1.0e-12);
        }
    }
}

#[test]
fn hierarchical_candidate_matches_dense_for_entmax15_and_sparsemax() {
    let keys = vec![
        9.0, 0.0, 8.5, 0.2, -8.0, 1.0, -9.0, -1.0, 0.0, 7.0, 0.2, 6.5, -4.0, -4.0, -5.0, -3.0,
    ];
    for alpha in [1.5, 2.0] {
        let case = QueryKeyPagedCase {
            query: vec![1.0, 0.25],
            keys: keys.clone(),
            head_dim: 2,
            page_size: 8,
            alpha,
            score_scale: 2.0_f64.sqrt().recip(),
        };
        assert_hierarchical_matches_dense(&case, 2);
    }
}

#[test]
fn hierarchy_can_avoid_tokens_that_flat_page_box_must_load() {
    let case = QueryKeyPagedCase {
        query: vec![1.0, 0.0],
        keys: vec![
            10.0, 0.0, 9.5, 0.0, -10.0, 4.0, -11.0, -4.0, -20.0, 7.0, -21.0, -7.0, -30.0, 9.0,
            -31.0, -9.0,
        ],
        head_dim: 2,
        page_size: 8,
        alpha: 2.0,
        score_scale: 1.0,
    };
    let flat = branch_and_bound_entmax_qk_box(&case).unwrap();
    let hierarchical = assert_hierarchical_matches_dense(&case, 2);
    assert_eq!(flat.metrics.scores_loaded, 8);
    assert_eq!(hierarchical.metrics.tokens_loaded, 2);
    assert!(hierarchical.metrics.tokens_pruned >= 6);
}

#[test]
fn hierarchy_degrades_safely_to_loading_every_leaf() {
    let case = QueryKeyPagedCase {
        query: vec![1.0, 0.0],
        keys: vec![
            1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0,
        ],
        head_dim: 2,
        page_size: 8,
        alpha: 2.0,
        score_scale: 1.0,
    };
    let result = assert_hierarchical_matches_dense(&case, 2);
    assert_eq!(result.metrics.tokens_loaded, 8);
    assert_eq!(result.metrics.tokens_pruned, 0);
    assert_eq!(result.metrics.leaves_loaded, result.metrics.leaves_total);
}

#[test]
fn lazy_candidate_matches_eager_for_entmax15_and_sparsemax() {
    let keys = vec![
        10.0, 0.0, 9.5, 0.0, -10.0, 4.0, -11.0, -4.0, -20.0, 7.0, -21.0, -7.0, -30.0, 9.0, -31.0,
        -9.0,
    ];

    for alpha in [1.5, 2.0] {
        let case = QueryKeyPagedCase {
            query: vec![1.0, 0.0],
            keys: keys.clone(),
            head_dim: 2,
            page_size: 8,
            alpha,
            score_scale: 1.0,
        };

        let (_, lazy) = assert_lazy_matches_eager(&case, 2);

        assert!(lazy.metrics.bound_cache_hits > 0);
    }
}

#[test]
fn lazy_pruned_subtree_leaves_descendants_unevaluated() {
    let case = QueryKeyPagedCase {
        query: vec![1.0, 0.0],
        keys: vec![
            10.0, 0.0, 9.5, 0.0, -10.0, 4.0, -11.0, -4.0, -20.0, 7.0, -21.0, -7.0, -30.0, 9.0,
            -31.0, -9.0,
        ],
        head_dim: 2,
        page_size: 8,
        alpha: 2.0,
        score_scale: 1.0,
    };

    let (_, lazy) = assert_lazy_matches_eager(&case, 2);

    assert!(lazy.metrics.tokens_pruned > 0);
    assert!(lazy.metrics.nodes_never_evaluated > 0);
    assert!(lazy.metrics.bound_evaluations < lazy.metrics.nodes_total);
}

#[test]
fn lazy_dense_fallback_remains_exact() {
    let case = QueryKeyPagedCase {
        query: vec![1.0, 0.0],
        keys: vec![
            1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0,
        ],
        head_dim: 2,
        page_size: 8,
        alpha: 2.0,
        score_scale: 1.0,
    };

    let (_, lazy) = assert_lazy_matches_eager(&case, 2);

    assert_eq!(lazy.metrics.tokens_loaded, 8);
    assert_eq!(lazy.metrics.tokens_pruned, 0);
    assert_eq!(lazy.metrics.leaves_loaded, lazy.metrics.leaves_total);
    assert_eq!(lazy.metrics.bound_evaluations, lazy.metrics.nodes_total);
    assert_eq!(lazy.metrics.nodes_never_evaluated, 0);
}

#[test]
fn index_rejects_wrong_key_matrix_and_invalid_leaf_size() {
    let keys = vec![1.0, 0.0, 2.0, 0.0];
    assert!(build_hierarchical_key_index(&keys, 2, 2, 3).is_err());
    let index = build_hierarchical_key_index(&keys, 2, 2, 1).unwrap();
    let case = QueryKeyPagedCase {
        query: vec![1.0, 0.0],
        keys: vec![1.0, 0.0, 3.0, 0.0],
        head_dim: 2,
        page_size: 2,
        alpha: 2.0,
        score_scale: 1.0,
    };
    assert!(branch_and_bound_entmax_hierarchical(&case, &index).is_err());
}

#[test]
fn exhaustive_small_hierarchies_lazy_match_eager_exactly() {
    const ROWS: [[f64; 2]; 3] = [[-1.0, -1.0], [0.0, 0.0], [1.0, 1.0]];

    const QUERIES: [[f64; 2]; 3] = [[1.0, -1.0], [-1.0, 1.0], [1.0, 1.0]];

    let state_count = ROWS.len().pow(4);

    for state in 0..state_count {
        let mut code = state;
        let mut keys = Vec::with_capacity(8);

        for _ in 0..4 {
            keys.extend_from_slice(&ROWS[code % ROWS.len()]);
            code /= ROWS.len();
        }

        for query in QUERIES {
            for alpha in [1.5, 2.0] {
                let case = QueryKeyPagedCase {
                    query: query.to_vec(),
                    keys: keys.clone(),
                    head_dim: 2,
                    page_size: 4,
                    alpha,
                    score_scale: 2.0_f64.sqrt().recip(),
                };

                for leaf_size in [1, 2] {
                    let (eager, lazy) = assert_lazy_matches_eager(&case, leaf_size);

                    #[cfg(not(miri))]
                    assert_eq!(eager.loaded_tokens, lazy.loaded_tokens);
                    #[cfg(miri)]
                    for (token, &probability) in eager.distribution.probabilities.iter().enumerate()
                    {
                        if probability > 1.0e-12 {
                            assert!(lazy.loaded_tokens[token]);
                        }
                    }

                    assert_distributions_match(&eager.distribution, &lazy.distribution);

                    assert!(lazy.metrics.bound_evaluations <= lazy.metrics.nodes_total);

                    assert_eq!(
                        lazy.metrics.nodes_never_evaluated,
                        lazy.metrics.nodes_total - lazy.metrics.bound_evaluations
                    );
                }
            }
        }
    }
}

#[test]
fn exhaustive_small_hierarchies_match_dense_oracle() {
    const ROWS: [[f64; 2]; 3] = [[-1.0, -1.0], [0.0, 0.0], [1.0, 1.0]];
    const QUERIES: [[f64; 2]; 3] = [[1.0, -1.0], [-1.0, 1.0], [1.0, 1.0]];
    let state_count = ROWS.len().pow(4);
    for state in 0..state_count {
        let mut code = state;
        let mut keys = Vec::with_capacity(8);
        for _ in 0..4 {
            keys.extend_from_slice(&ROWS[code % ROWS.len()]);
            code /= ROWS.len();
        }
        for query in QUERIES {
            for alpha in [1.5, 2.0] {
                let case = QueryKeyPagedCase {
                    query: query.to_vec(),
                    keys: keys.clone(),
                    head_dim: 2,
                    page_size: 4,
                    alpha,
                    score_scale: 2.0_f64.sqrt().recip(),
                };
                assert_hierarchical_matches_dense(&case, 1);
                assert_hierarchical_matches_dense(&case, 2);
            }
        }
    }
}
#[test]
fn support_certificate_accepts_sound_prunes_and_rejects_violations() {
    // Sound: bound exactly at the pruning frontier passes bit-exactly.
    assert_eq!(
        certify_support_exactness(&[1.0, 0.5, -2.0], 1.0, 1.0),
        Ok(())
    );
    // Violation: a pruned bound above the terminating endpoint must fail
    // closed instead of publishing a zero that is not certified.
    assert_eq!(
        certify_support_exactness(&[0.5, 1.25], 1.0, 1.0),
        Err("ADA-A5 support exactness certificate failed")
    );
}

#[test]
fn hierarchical_runners_publish_certified_support() {
    let case = QueryKeyPagedCase {
        query: vec![0.3, -0.4],
        keys: vec![
            1.0, 0.0, -1.0, 0.5, 0.5, 0.7, -0.2, 0.9, -0.6, 0.9, -0.8, 0.3,
        ],
        head_dim: 2,
        page_size: 2,
        alpha: 1.5,
        score_scale: 1.0,
    };
    let index = build_hierarchical_key_index(&case.keys, case.head_dim, case.page_size, 1).unwrap();

    let eager = branch_and_bound_entmax_hierarchical(&case, &index).unwrap();
    let lazy = branch_and_bound_entmax_hierarchical_lazy(&case, &index).unwrap();
    let priority = branch_and_bound_entmax_hierarchical_priority_lazy(&case, &index).unwrap();

    for result in [
        eager.distribution.probabilities,
        lazy.distribution.probabilities,
        priority.distribution.probabilities,
    ] {
        let mass: f64 = result.iter().copied().sum();
        assert_close(mass, 1.0, 4.0e-12);
    }
}
