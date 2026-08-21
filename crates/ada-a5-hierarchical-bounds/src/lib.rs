#![forbid(unsafe_code)]

use ada_a4_entmax_bnb::{EntmaxDistribution, dense_entmax, entmax_threshold_bracket};
use ada_a4_qk_box::{PageKeyBox, QueryKeyPagedCase, dense_qk_scores};

#[derive(Debug, Clone, PartialEq)]
pub struct HierarchyNode {
    start_token: usize,
    end_token: usize,
    key_box: PageKeyBox,
    left: Option<usize>,
    right: Option<usize>,
}

impl HierarchyNode {
    #[must_use]
    pub const fn start_token(&self) -> usize {
        self.start_token
    }

    #[must_use]
    pub const fn end_token(&self) -> usize {
        self.end_token
    }

    #[must_use]
    pub const fn token_count(&self) -> usize {
        self.end_token - self.start_token
    }

    #[must_use]
    pub const fn is_leaf(&self) -> bool {
        self.left.is_none()
    }

    #[must_use]
    pub const fn children(&self) -> Option<(usize, usize)> {
        match (self.left, self.right) {
            (Some(left), Some(right)) => Some((left, right)),
            _ => None,
        }
    }

    #[must_use]
    pub const fn key_box(&self) -> &PageKeyBox {
        &self.key_box
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct HierarchicalKeyIndex {
    head_dim: usize,
    key_count: usize,
    page_size: usize,
    leaf_size: usize,
    key_fingerprint: u64,
    nodes: Vec<HierarchyNode>,
    roots: Vec<usize>,
    leaves: Vec<usize>,
}

impl HierarchicalKeyIndex {
    #[must_use]
    pub const fn head_dim(&self) -> usize {
        self.head_dim
    }

    #[must_use]
    pub const fn key_count(&self) -> usize {
        self.key_count
    }

    #[must_use]
    pub const fn page_size(&self) -> usize {
        self.page_size
    }

    #[must_use]
    pub const fn leaf_size(&self) -> usize {
        self.leaf_size
    }

    #[must_use]
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    #[must_use]
    pub fn leaf_count(&self) -> usize {
        self.leaves.len()
    }

    #[must_use]
    pub fn roots(&self) -> &[usize] {
        &self.roots
    }

    #[must_use]
    pub fn node(&self, index: usize) -> Option<&HierarchyNode> {
        self.nodes.get(index)
    }
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct HierarchicalMetrics {
    pub nodes_total: usize,
    pub nodes_expanded: usize,
    pub subtrees_pruned: usize,
    pub bound_evaluations: usize,
    pub leaves_total: usize,
    pub leaves_loaded: usize,
    pub tokens_loaded: usize,
    pub tokens_pruned: usize,
    pub threshold_solves: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct HierarchicalResult {
    pub distribution: EntmaxDistribution,
    pub loaded_tokens: Vec<bool>,
    pub metrics: HierarchicalMetrics,
}

fn fingerprint_keys(keys: &[f64]) -> u64 {
    let mut fingerprint = 0xcbf2_9ce4_8422_2325_u64;
    for &value in keys {
        fingerprint ^= value.to_bits();
        fingerprint = fingerprint.wrapping_mul(0x0000_0100_0000_01b3);
    }
    fingerprint
}

fn validate_key_matrix(
    keys: &[f64],
    head_dim: usize,
    page_size: usize,
    leaf_size: usize,
) -> Result<(), &'static str> {
    if head_dim == 0 {
        return Err("ADA-A5 head_dim must be non-zero");
    }
    if page_size == 0 {
        return Err("ADA-A5 page_size must be non-zero");
    }
    if leaf_size == 0 {
        return Err("ADA-A5 leaf_size must be non-zero");
    }
    if leaf_size > page_size {
        return Err("ADA-A5 leaf_size must not exceed page_size");
    }
    if keys.is_empty() {
        return Err("ADA-A5 requires at least one key");
    }
    if !keys.chunks_exact(head_dim).remainder().is_empty() {
        return Err("ADA-A5 keys must be row-major [key_count, head_dim]");
    }
    if keys.iter().any(|value| !value.is_finite()) {
        return Err("ADA-A5 keys must be finite");
    }
    Ok(())
}

fn box_for_range(
    keys: &[f64],
    head_dim: usize,
    start_token: usize,
    end_token: usize,
) -> PageKeyBox {
    debug_assert!(start_token < end_token);
    let first_start = start_token * head_dim;
    let first = &keys[first_start..first_start + head_dim];
    let mut minimum = first.to_vec();
    let mut maximum = first.to_vec();
    let values = &keys[first_start..end_token * head_dim];

    for row in values.chunks_exact(head_dim).skip(1) {
        for ((min_value, max_value), &value) in
            minimum.iter_mut().zip(maximum.iter_mut()).zip(row.iter())
        {
            *min_value = min_value.min(value);
            *max_value = max_value.max(value);
        }
    }

    PageKeyBox {
        minimum,
        maximum,
        token_count: end_token - start_token,
    }
}

fn merge_boxes(left: &PageKeyBox, right: &PageKeyBox) -> PageKeyBox {
    debug_assert_eq!(left.minimum.len(), right.minimum.len());
    let minimum = left
        .minimum
        .iter()
        .zip(right.minimum.iter())
        .map(|(&left_value, &right_value)| left_value.min(right_value))
        .collect();
    let maximum = left
        .maximum
        .iter()
        .zip(right.maximum.iter())
        .map(|(&left_value, &right_value)| left_value.max(right_value))
        .collect();
    PageKeyBox {
        minimum,
        maximum,
        token_count: left.token_count + right.token_count,
    }
}

fn build_subtree(
    keys: &[f64],
    head_dim: usize,
    leaf_size: usize,
    start_token: usize,
    end_token: usize,
    nodes: &mut Vec<HierarchyNode>,
    leaves: &mut Vec<usize>,
) -> usize {
    let token_count = end_token - start_token;
    if token_count <= leaf_size {
        let node_index = nodes.len();
        nodes.push(HierarchyNode {
            start_token,
            end_token,
            key_box: box_for_range(keys, head_dim, start_token, end_token),
            left: None,
            right: None,
        });
        leaves.push(node_index);
        return node_index;
    }

    let midpoint = start_token + token_count / 2;
    let left = build_subtree(
        keys,
        head_dim,
        leaf_size,
        start_token,
        midpoint,
        nodes,
        leaves,
    );
    let right = build_subtree(
        keys, head_dim, leaf_size, midpoint, end_token, nodes, leaves,
    );
    let key_box = merge_boxes(&nodes[left].key_box, &nodes[right].key_box);
    let node_index = nodes.len();
    nodes.push(HierarchyNode {
        start_token,
        end_token,
        key_box,
        left: Some(left),
        right: Some(right),
    });
    node_index
}

/// Build nested coordinate-wise min/max metadata within every outer KV page.
///
/// The index is a prefill/cache-construction artifact. Parent boxes are merged
/// from child boxes, while leaves contain at most `leaf_size` contiguous keys.
///
/// # Errors
///
/// Returns an error for malformed/non-finite keys or invalid dimensions.
#[must_use = "the hierarchical metadata is required for A5 query-time pruning"]
pub fn build_hierarchical_key_index(
    keys: &[f64],
    head_dim: usize,
    page_size: usize,
    leaf_size: usize,
) -> Result<HierarchicalKeyIndex, &'static str> {
    validate_key_matrix(keys, head_dim, page_size, leaf_size)?;
    let key_count = keys.len() / head_dim;
    let mut nodes = Vec::new();
    let mut roots = Vec::with_capacity(key_count.div_ceil(page_size));
    let mut leaves = Vec::new();

    for page_start in (0..key_count).step_by(page_size) {
        let page_end = (page_start + page_size).min(key_count);
        roots.push(build_subtree(
            keys,
            head_dim,
            leaf_size,
            page_start,
            page_end,
            &mut nodes,
            &mut leaves,
        ));
    }

    Ok(HierarchicalKeyIndex {
        head_dim,
        key_count,
        page_size,
        leaf_size,
        key_fingerprint: fingerprint_keys(keys),
        nodes,
        roots,
        leaves,
    })
}

fn validate_query_index(
    case: &QueryKeyPagedCase,
    index: &HierarchicalKeyIndex,
) -> Result<(), &'static str> {
    case.validate()?;
    if case.head_dim != index.head_dim {
        return Err("ADA-A5 index head_dim does not match the Q/K case");
    }
    if case.page_size != index.page_size {
        return Err("ADA-A5 index page_size does not match the Q/K case");
    }
    if case.key_count() != index.key_count {
        return Err("ADA-A5 index key_count does not match the Q/K case");
    }
    if fingerprint_keys(&case.keys) != index.key_fingerprint {
        return Err("ADA-A5 index does not belong to the supplied key matrix");
    }
    Ok(())
}

fn query_box_bound(
    query: &[f64],
    key_box: &PageKeyBox,
    score_scale: f64,
) -> Result<f64, &'static str> {
    if key_box.minimum.len() != query.len() || key_box.maximum.len() != query.len() {
        return Err("ADA-A5 node-box dimension mismatch");
    }
    let mut sum = 0.0_f64;
    for ((&query_value, &minimum), &maximum) in query
        .iter()
        .zip(key_box.minimum.iter())
        .zip(key_box.maximum.iter())
    {
        if !minimum.is_finite() || !maximum.is_finite() || minimum > maximum {
            return Err("ADA-A5 node-box coordinates must be finite and ordered");
        }
        sum += (query_value * minimum).max(query_value * maximum);
    }
    let bound = sum * score_scale;
    if !bound.is_finite() {
        return Err("ADA-A5 node upper bound overflowed");
    }
    Ok(bound)
}

/// Evaluate all hierarchy-node score bounds for an already-built index.
///
/// # Errors
///
/// Returns an error for an invalid query/index pairing or malformed metadata.
#[must_use = "hierarchical bounds should be validated or consumed by a controller"]
pub fn query_hierarchical_upper_bounds(
    case: &QueryKeyPagedCase,
    index: &HierarchicalKeyIndex,
) -> Result<Vec<f64>, &'static str> {
    validate_query_index(case, index)?;
    index
        .nodes
        .iter()
        .map(|node| query_box_bound(&case.query, &node.key_box, case.score_scale))
        .collect()
}

fn validate_bounds_against_dense_scores(
    index: &HierarchicalKeyIndex,
    bounds: &[f64],
    dense_scores: &[f64],
) -> Result<(), &'static str> {
    for (node, &bound) in index.nodes.iter().zip(bounds.iter()) {
        let actual_maximum = dense_scores[node.start_token..node.end_token]
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        if bound < actual_maximum {
            return Err("ADA-A5 f64 hierarchy bound is not conservative against the dense oracle");
        }
    }
    Ok(())
}

fn load_leaf(
    node: &HierarchyNode,
    loaded_tokens: &mut [bool],
    loaded_indices: &mut Vec<usize>,
    metrics: &mut HierarchicalMetrics,
) {
    debug_assert!(node.is_leaf());
    for (offset, loaded) in loaded_tokens[node.start_token..node.end_token]
        .iter_mut()
        .enumerate()
    {
        let token = node.start_token + offset;
        debug_assert!(!*loaded);
        *loaded = true;
        loaded_indices.push(token);
        metrics.tokens_loaded += 1;
    }
    metrics.leaves_loaded += 1;
}

fn highest_bound_position(frontier: &[usize], bounds: &[f64]) -> usize {
    let mut best_position = 0;
    for position in 1..frontier.len() {
        if bounds[frontier[position]] > bounds[frontier[best_position]] {
            best_position = position;
        }
    }
    best_position
}

fn seed_highest_bound_leaf(
    index: &HierarchicalKeyIndex,
    bounds: &[f64],
    frontier: &mut Vec<usize>,
    loaded_tokens: &mut [bool],
    loaded_indices: &mut Vec<usize>,
    metrics: &mut HierarchicalMetrics,
) {
    let root_position = highest_bound_position(&index.roots, bounds);
    let mut current = index.roots[root_position];
    frontier.extend(
        index
            .roots
            .iter()
            .enumerate()
            .filter_map(|(position, &root)| (position != root_position).then_some(root)),
    );

    loop {
        let node = &index.nodes[current];
        if let Some((left, right)) = node.children() {
            metrics.nodes_expanded += 1;
            if bounds[left] >= bounds[right] {
                frontier.push(right);
                current = left;
            } else {
                frontier.push(left);
                current = right;
            }
        } else {
            load_leaf(node, loaded_tokens, loaded_indices, metrics);
            return;
        }
    }
}

fn subset_scores(dense_scores: &[f64], loaded_indices: &[usize]) -> Vec<f64> {
    loaded_indices
        .iter()
        .map(|&index| dense_scores[index])
        .collect()
}

fn finalize_distribution(
    case: &QueryKeyPagedCase,
    dense_scores: &[f64],
    loaded_indices: &[usize],
) -> Result<EntmaxDistribution, &'static str> {
    let loaded_scores = subset_scores(dense_scores, loaded_indices);
    let subset_distribution = dense_entmax(&loaded_scores, case.alpha)?;
    let mut probabilities = vec![0.0; dense_scores.len()];
    for (&index, &probability) in loaded_indices
        .iter()
        .zip(subset_distribution.probabilities.iter())
    {
        probabilities[index] = probability;
    }
    Ok(EntmaxDistribution {
        probabilities,
        tau: subset_distribution.tau,
    })
}

/// Run exact subset-threshold branch-and-bound over hierarchical Q/K boxes.
///
/// A node is pruned only when `(alpha - 1) * upper_bound <= tau_lower`, where
/// `tau_lower` is the conservative lower endpoint of the entmax threshold
/// bracket on the currently loaded token subset. An unresolved internal node is
/// replaced by its children; only unresolved leaves cause token-score loading.
///
/// A5-E0 deliberately computes dense QK scores to validate every f64 hierarchy
/// bound and to provide the independent oracle. A production realization must
/// replace that laboratory validation with a numerically certified bound path.
///
/// # Errors
///
/// Returns an error for invalid Q/K/index inputs, a non-conservative f64 bound,
/// or an entmax threshold-solver failure.
#[must_use = "the exact result and hierarchical work metrics should be checked"]
pub fn branch_and_bound_entmax_hierarchical(
    case: &QueryKeyPagedCase,
    index: &HierarchicalKeyIndex,
) -> Result<HierarchicalResult, &'static str> {
    validate_query_index(case, index)?;
    let dense_scores = dense_qk_scores(case)?;
    let bounds = query_hierarchical_upper_bounds(case, index)?;
    validate_bounds_against_dense_scores(index, &bounds, &dense_scores)?;

    let mut metrics = HierarchicalMetrics {
        nodes_total: index.node_count(),
        bound_evaluations: bounds.len(),
        leaves_total: index.leaf_count(),
        ..HierarchicalMetrics::default()
    };
    let mut loaded_tokens = vec![false; index.key_count];
    let mut loaded_indices = Vec::with_capacity(index.key_count);
    let mut frontier = Vec::new();

    seed_highest_bound_leaf(
        index,
        &bounds,
        &mut frontier,
        &mut loaded_tokens,
        &mut loaded_indices,
        &mut metrics,
    );

    loop {
        let loaded_scores = subset_scores(&dense_scores, &loaded_indices);
        let threshold = entmax_threshold_bracket(&loaded_scores, case.alpha)?;
        metrics.threshold_solves += 1;
        let tau_lower = threshold.lower;
        let entmax_scale = case.alpha - 1.0;

        loop {
            let mut unresolved = Vec::with_capacity(frontier.len());
            for node_index in frontier.drain(..) {
                if entmax_scale * bounds[node_index] <= tau_lower {
                    metrics.subtrees_pruned += 1;
                    metrics.tokens_pruned += index.nodes[node_index].token_count();
                } else {
                    unresolved.push(node_index);
                }
            }
            frontier = unresolved;

            if frontier.is_empty() {
                debug_assert_eq!(
                    metrics.tokens_loaded + metrics.tokens_pruned,
                    index.key_count
                );
                return Ok(HierarchicalResult {
                    distribution: finalize_distribution(case, &dense_scores, &loaded_indices)?,
                    loaded_tokens,
                    metrics,
                });
            }

            let best_position = highest_bound_position(&frontier, &bounds);
            let node_index = frontier.swap_remove(best_position);
            let node = &index.nodes[node_index];
            if let Some((left, right)) = node.children() {
                metrics.nodes_expanded += 1;
                frontier.push(left);
                frontier.push(right);
                continue;
            }

            load_leaf(node, &mut loaded_tokens, &mut loaded_indices, &mut metrics);
            break;
        }
    }
}

#[cfg(test)]
mod tests {
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
            build_hierarchical_key_index(&case.keys, case.head_dim, case.page_size, leaf_size)
                .unwrap();
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

    #[test]
    fn hierarchy_bounds_dominate_dense_scores_and_tighten_downward() {
        let case = QueryKeyPagedCase {
            query: vec![2.0, -3.0, 0.5],
            keys: vec![
                1.0, 4.0, -2.0, 3.0, -1.0, 5.0, -4.0, 2.0, 1.0, 0.5, -3.0, 7.0, 2.0, 2.0, 2.0,
                -1.0, -4.0, 3.0, 5.0, 0.0, -2.0, 1.0, 3.0, -5.0,
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
}
