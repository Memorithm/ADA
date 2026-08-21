#![forbid(unsafe_code)]

use ada_a4_entmax_bnb::{EntmaxDistribution, dense_entmax, entmax_threshold_bracket};
use ada_a4_qk_box::{PageKeyBox, QueryKeyPagedCase, dense_qk_scores};

#[derive(Debug, Clone, PartialEq)]
pub struct ContentAwareNode {
    permutation_start: usize,
    permutation_end: usize,
    key_box: PageKeyBox,
    ball_center: Vec<f64>,
    ball_radius: f64,
    left: Option<usize>,
    right: Option<usize>,
}

impl ContentAwareNode {
    #[must_use]
    pub const fn token_count(&self) -> usize {
        self.permutation_end - self.permutation_start
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

    #[must_use]
    pub fn ball_center(&self) -> &[f64] {
        &self.ball_center
    }

    #[must_use]
    pub const fn ball_radius(&self) -> f64 {
        self.ball_radius
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ContentAwareKeyIndex {
    head_dim: usize,
    key_count: usize,
    page_size: usize,
    leaf_size: usize,
    key_fingerprint: u64,
    permutation: Vec<usize>,
    nodes: Vec<ContentAwareNode>,
    roots: Vec<usize>,
    leaves: Vec<usize>,
}

impl ContentAwareKeyIndex {
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
    pub fn permutation(&self) -> &[usize] {
        &self.permutation
    }

    #[must_use]
    pub fn roots(&self) -> &[usize] {
        &self.roots
    }

    #[must_use]
    pub fn node(&self, index: usize) -> Option<&ContentAwareNode> {
        self.nodes.get(index)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NodeUpperBounds {
    pub box_upper: f64,
    pub ball_upper: f64,
    pub hybrid_upper: f64,
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct ContentAwareMetrics {
    pub nodes_total: usize,
    pub nodes_expanded: usize,
    pub subtrees_pruned: usize,
    pub hybrid_bound_evaluations: usize,
    pub ball_bound_wins: usize,
    pub box_bound_wins: usize,
    pub leaves_total: usize,
    pub leaves_loaded: usize,
    pub tokens_loaded: usize,
    pub tokens_pruned: usize,
    pub threshold_solves: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ContentAwareResult {
    pub distribution: EntmaxDistribution,
    pub loaded_tokens: Vec<bool>,
    pub metrics: ContentAwareMetrics,
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
        return Err("ADA-A5 E2 head_dim must be non-zero");
    }
    if page_size == 0 {
        return Err("ADA-A5 E2 page_size must be non-zero");
    }
    if leaf_size == 0 || leaf_size > page_size {
        return Err("ADA-A5 E2 leaf_size must be in 1..=page_size");
    }
    if keys.is_empty() {
        return Err("ADA-A5 E2 requires at least one key");
    }
    if !keys.chunks_exact(head_dim).remainder().is_empty() {
        return Err("ADA-A5 E2 keys must be row-major [key_count, head_dim]");
    }
    if keys.iter().any(|value| !value.is_finite()) {
        return Err("ADA-A5 E2 keys must be finite");
    }
    Ok(())
}

fn key_row(keys: &[f64], head_dim: usize, token: usize) -> &[f64] {
    let start = token * head_dim;
    &keys[start..start + head_dim]
}

fn squared_distance(left: &[f64], right: &[f64]) -> f64 {
    left.iter()
        .zip(right.iter())
        .map(|(&a, &b)| {
            let delta = a - b;
            delta * delta
        })
        .sum()
}

fn center_for_tokens(keys: &[f64], head_dim: usize, tokens: &[usize]) -> Vec<f64> {
    let mut center = vec![0.0_f64; head_dim];
    for &token in tokens {
        for (accumulator, &value) in center.iter_mut().zip(key_row(keys, head_dim, token)) {
            *accumulator += value;
        }
    }
    let denominator = f64::from(u32::try_from(tokens.len()).expect("node size fits in u32"));
    for value in &mut center {
        *value /= denominator;
    }
    center
}

fn box_for_tokens(keys: &[f64], head_dim: usize, tokens: &[usize]) -> PageKeyBox {
    let first = key_row(keys, head_dim, tokens[0]);
    let mut minimum = first.to_vec();
    let mut maximum = first.to_vec();
    for &token in &tokens[1..] {
        for ((min_value, max_value), &value) in minimum
            .iter_mut()
            .zip(maximum.iter_mut())
            .zip(key_row(keys, head_dim, token))
        {
            *min_value = min_value.min(value);
            *max_value = max_value.max(value);
        }
    }
    PageKeyBox {
        minimum,
        maximum,
        token_count: tokens.len(),
    }
}

fn ball_for_tokens(keys: &[f64], head_dim: usize, tokens: &[usize]) -> (Vec<f64>, f64) {
    let center = center_for_tokens(keys, head_dim, tokens);
    let maximum_squared_distance = tokens
        .iter()
        .map(|&token| squared_distance(key_row(keys, head_dim, token), &center))
        .fold(0.0_f64, f64::max);
    (center, maximum_squared_distance.sqrt())
}

fn dot(left: &[f64], right: &[f64]) -> f64 {
    left.iter().zip(right.iter()).map(|(&a, &b)| a * b).sum()
}

fn projection(keys: &[f64], head_dim: usize, token: usize, direction: &[f64]) -> f64 {
    dot(key_row(keys, head_dim, token), direction)
}

fn farthest_token_from_point(
    keys: &[f64],
    head_dim: usize,
    tokens: &[usize],
    point: &[f64],
) -> usize {
    let mut best_token = tokens[0];
    let mut best_distance = squared_distance(key_row(keys, head_dim, best_token), point);
    for &token in &tokens[1..] {
        let distance = squared_distance(key_row(keys, head_dim, token), point);
        if distance > best_distance
            || (distance.to_bits() == best_distance.to_bits() && token < best_token)
        {
            best_token = token;
            best_distance = distance;
        }
    }
    best_token
}

fn content_partition(keys: &[f64], head_dim: usize, tokens: &mut [usize]) {
    if tokens.len() <= 1 {
        return;
    }

    let center = center_for_tokens(keys, head_dim, tokens);

    let anchor_token = farthest_token_from_point(keys, head_dim, tokens, &center);
    let anchor_row = key_row(keys, head_dim, anchor_token);

    let opposite_token = farthest_token_from_point(keys, head_dim, tokens, anchor_row);
    let opposite_row = key_row(keys, head_dim, opposite_token);

    let direction: Vec<f64> = anchor_row
        .iter()
        .zip(opposite_row.iter())
        .map(|(&anchor, &opposite)| opposite - anchor)
        .collect();

    let direction_norm_squared = dot(&direction, &direction);

    if direction_norm_squared > 0.0 {
        tokens.sort_unstable_by(|left, right| {
            projection(keys, head_dim, *left, &direction)
                .total_cmp(&projection(keys, head_dim, *right, &direction))
                .then_with(|| left.cmp(right))
        });
    } else {
        tokens.sort_unstable();
    }
}

struct TreeBuildConfig<'a> {
    keys: &'a [f64],
    head_dim: usize,
    leaf_size: usize,
}

fn build_subtree(
    config: &TreeBuildConfig<'_>,
    permutation: &mut [usize],
    start: usize,
    end: usize,
    nodes: &mut Vec<ContentAwareNode>,
    leaves: &mut Vec<usize>,
) -> usize {
    let token_count = end - start;
    let tokens = &permutation[start..end];

    let key_box = box_for_tokens(config.keys, config.head_dim, tokens);
    let (ball_center, ball_radius) = ball_for_tokens(config.keys, config.head_dim, tokens);

    if token_count <= config.leaf_size {
        let node_index = nodes.len();

        nodes.push(ContentAwareNode {
            permutation_start: start,
            permutation_end: end,
            key_box,
            ball_center,
            ball_radius,
            left: None,
            right: None,
        });

        leaves.push(node_index);
        return node_index;
    }

    content_partition(config.keys, config.head_dim, &mut permutation[start..end]);

    let midpoint = start + token_count / 2;

    let left = build_subtree(config, permutation, start, midpoint, nodes, leaves);

    let right = build_subtree(config, permutation, midpoint, end, nodes, leaves);

    let node_index = nodes.len();

    nodes.push(ContentAwareNode {
        permutation_start: start,
        permutation_end: end,
        key_box,
        ball_center,
        ball_radius,
        left: Some(left),
        right: Some(right),
    });

    node_index
}

/// Build a deterministic content-aware hierarchy independently inside each outer KV page.
///
/// Internal splits use an approximate-diameter direction: a point farthest from
/// the node mean is chosen as the first pivot, then a point farthest from that
/// pivot is chosen as the second. Tokens are sorted by projection onto the pivot
/// direction and split at the median. Original token ids are retained in a
/// permutation, so no semantic reordering of the dense oracle occurs.
///
/// # Errors
///
/// Returns an error for malformed/non-finite keys or invalid dimensions.
#[must_use = "the content-aware metadata is required for A5 E2 query-time pruning"]
pub fn build_content_aware_key_index(
    keys: &[f64],
    head_dim: usize,
    page_size: usize,
    leaf_size: usize,
) -> Result<ContentAwareKeyIndex, &'static str> {
    validate_key_matrix(keys, head_dim, page_size, leaf_size)?;
    let key_count = keys.len() / head_dim;
    let mut permutation: Vec<usize> = (0..key_count).collect();
    let mut nodes = Vec::new();
    let mut roots = Vec::with_capacity(key_count.div_ceil(page_size));
    let mut leaves = Vec::new();

    let build_config = TreeBuildConfig {
        keys,
        head_dim,
        leaf_size,
    };

    for page_start in (0..key_count).step_by(page_size) {
        let page_end = (page_start + page_size).min(key_count);
        roots.push(build_subtree(
            &build_config,
            &mut permutation,
            page_start,
            page_end,
            &mut nodes,
            &mut leaves,
        ));
    }

    Ok(ContentAwareKeyIndex {
        head_dim,
        key_count,
        page_size,
        leaf_size,
        key_fingerprint: fingerprint_keys(keys),
        permutation,
        nodes,
        roots,
        leaves,
    })
}

fn validate_query_index(
    case: &QueryKeyPagedCase,
    index: &ContentAwareKeyIndex,
) -> Result<(), &'static str> {
    case.validate()?;
    if case.head_dim != index.head_dim
        || case.page_size != index.page_size
        || case.key_count() != index.key_count
    {
        return Err("ADA-A5 E2 content-aware index shape does not match Q/K case");
    }
    if fingerprint_keys(&case.keys) != index.key_fingerprint {
        return Err("ADA-A5 E2 content-aware index does not belong to supplied keys");
    }
    Ok(())
}

fn query_box_upper(
    query: &[f64],
    key_box: &PageKeyBox,
    score_scale: f64,
) -> Result<f64, &'static str> {
    if key_box.minimum.len() != query.len() || key_box.maximum.len() != query.len() {
        return Err("ADA-A5 E2 node box dimension mismatch");
    }
    let mut sum = 0.0_f64;
    for ((&query_value, &minimum), &maximum) in query
        .iter()
        .zip(key_box.minimum.iter())
        .zip(key_box.maximum.iter())
    {
        if !minimum.is_finite() || !maximum.is_finite() || minimum > maximum {
            return Err("ADA-A5 E2 malformed node box");
        }
        sum += (query_value * minimum).max(query_value * maximum);
    }
    let bound = sum * score_scale;
    if !bound.is_finite() {
        return Err("ADA-A5 E2 box bound overflowed");
    }
    Ok(bound)
}

fn query_ball_upper(
    query: &[f64],
    center: &[f64],
    radius: f64,
    score_scale: f64,
) -> Result<f64, &'static str> {
    if center.len() != query.len() || !radius.is_finite() || radius < 0.0 {
        return Err("ADA-A5 E2 malformed node ball");
    }
    let query_norm = dot(query, query).sqrt();
    let center_dot = dot(query, center);
    let radius_term = query_norm * radius;
    let raw = center_dot + radius_term;
    let rounding_guard = 64.0 * f64::EPSILON * (1.0 + center_dot.abs() + radius_term.abs());
    let bound = (raw + rounding_guard) * score_scale;
    if !bound.is_finite() {
        return Err("ADA-A5 E2 ball bound overflowed");
    }
    Ok(bound)
}

/// Evaluate coordinate-box, enclosing-ball, and hybrid upper bounds for every node.
///
/// `hybrid_upper = min(box_upper, ball_upper)`. The minimum remains a valid
/// real-arithmetic upper bound when both components are valid upper bounds.
/// A5-E2 still validates every f64 hybrid bound against the dense oracle before
/// allowing pruning; the rounding guard is not claimed as a production proof.
///
/// # Errors
///
/// Returns an error for invalid query/index pairing or malformed metadata.
#[must_use = "content-aware bounds must be validated or consumed by a controller"]
pub fn query_content_aware_upper_bounds(
    case: &QueryKeyPagedCase,
    index: &ContentAwareKeyIndex,
) -> Result<Vec<NodeUpperBounds>, &'static str> {
    validate_query_index(case, index)?;
    index
        .nodes
        .iter()
        .map(|node| {
            let box_upper = query_box_upper(&case.query, &node.key_box, case.score_scale)?;
            let ball_upper = query_ball_upper(
                &case.query,
                &node.ball_center,
                node.ball_radius,
                case.score_scale,
            )?;
            Ok(NodeUpperBounds {
                box_upper,
                ball_upper,
                hybrid_upper: box_upper.min(ball_upper),
            })
        })
        .collect()
}

fn actual_node_maximum(
    dense_scores: &[f64],
    permutation: &[usize],
    node: &ContentAwareNode,
) -> f64 {
    permutation[node.permutation_start..node.permutation_end]
        .iter()
        .map(|&token| dense_scores[token])
        .fold(f64::NEG_INFINITY, f64::max)
}

fn validate_bounds_against_dense_scores(
    index: &ContentAwareKeyIndex,
    bounds: &[NodeUpperBounds],
    dense_scores: &[f64],
) -> Result<(), &'static str> {
    for (node, node_bounds) in index.nodes.iter().zip(bounds.iter()) {
        let actual = actual_node_maximum(dense_scores, &index.permutation, node);
        if node_bounds.box_upper < actual {
            return Err("ADA-A5 E2 f64 box bound is not conservative against dense oracle");
        }
        if node_bounds.ball_upper < actual {
            return Err("ADA-A5 E2 f64 ball bound is not conservative against dense oracle");
        }
        if node_bounds.hybrid_upper < actual {
            return Err("ADA-A5 E2 f64 hybrid bound is not conservative against dense oracle");
        }
    }
    Ok(())
}

fn highest_bound_position(frontier: &[usize], bounds: &[NodeUpperBounds]) -> usize {
    let mut best_position = 0;
    for position in 1..frontier.len() {
        if bounds[frontier[position]].hybrid_upper > bounds[frontier[best_position]].hybrid_upper {
            best_position = position;
        }
    }
    best_position
}

fn load_leaf(
    node: &ContentAwareNode,
    permutation: &[usize],
    loaded_tokens: &mut [bool],
    loaded_indices: &mut Vec<usize>,
    metrics: &mut ContentAwareMetrics,
) {
    debug_assert!(node.is_leaf());
    for &token in &permutation[node.permutation_start..node.permutation_end] {
        debug_assert!(!loaded_tokens[token]);
        loaded_tokens[token] = true;
        loaded_indices.push(token);
        metrics.tokens_loaded += 1;
    }
    metrics.leaves_loaded += 1;
}

fn seed_highest_bound_leaf(
    index: &ContentAwareKeyIndex,
    bounds: &[NodeUpperBounds],
    frontier: &mut Vec<usize>,
    loaded_tokens: &mut [bool],
    loaded_indices: &mut Vec<usize>,
    metrics: &mut ContentAwareMetrics,
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
            if bounds[left].hybrid_upper >= bounds[right].hybrid_upper {
                frontier.push(right);
                current = left;
            } else {
                frontier.push(left);
                current = right;
            }
        } else {
            load_leaf(
                node,
                &index.permutation,
                loaded_tokens,
                loaded_indices,
                metrics,
            );
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

/// Run exact subset-threshold Entmax branch-and-bound over a content-aware hierarchy.
///
/// The tree is content-aware inside each original KV page. Every node uses the
/// minimum of independently safe coordinate-box and enclosing-ball MIPS upper
/// bounds. The Entmax pruning certificate remains unchanged:
/// `(alpha - 1) * upper_bound <= tau_lower`.
///
/// E2 computes dense QK scores only for laboratory validation/oracle purposes.
/// No wall-clock or production memory-traffic claim follows from this function.
///
/// # Errors
///
/// Returns an error for invalid inputs, a non-conservative f64 bound, or an
/// Entmax threshold-solver failure.
#[must_use = "the exact result and content-aware work metrics should be checked"]
pub fn branch_and_bound_entmax_content_aware(
    case: &QueryKeyPagedCase,
    index: &ContentAwareKeyIndex,
) -> Result<ContentAwareResult, &'static str> {
    validate_query_index(case, index)?;
    let dense_scores = dense_qk_scores(case)?;
    let bounds = query_content_aware_upper_bounds(case, index)?;
    validate_bounds_against_dense_scores(index, &bounds, &dense_scores)?;

    let mut metrics = ContentAwareMetrics {
        nodes_total: index.node_count(),
        hybrid_bound_evaluations: bounds.len(),
        leaves_total: index.leaf_count(),
        ..ContentAwareMetrics::default()
    };
    for node_bounds in &bounds {
        if node_bounds.ball_upper < node_bounds.box_upper {
            metrics.ball_bound_wins += 1;
        } else {
            metrics.box_bound_wins += 1;
        }
    }

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
                if entmax_scale * bounds[node_index].hybrid_upper <= tau_lower {
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
                return Ok(ContentAwareResult {
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

            load_leaf(
                node,
                &index.permutation,
                &mut loaded_tokens,
                &mut loaded_indices,
                &mut metrics,
            );
            break;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ada_a5_hierarchical_bounds::{
        branch_and_bound_entmax_hierarchical, build_hierarchical_key_index,
    };

    fn assert_close(left: f64, right: f64, tolerance: f64) {
        let scale = left.abs().max(right.abs()).max(1.0);
        assert!(
            (left - right).abs() <= tolerance * scale,
            "{left} != {right}"
        );
    }

    fn assert_candidate_matches_dense(
        case: &QueryKeyPagedCase,
        leaf_size: usize,
    ) -> ContentAwareResult {
        let index =
            build_content_aware_key_index(&case.keys, case.head_dim, case.page_size, leaf_size)
                .unwrap();
        let dense_scores = dense_qk_scores(case).unwrap();
        let dense = dense_entmax(&dense_scores, case.alpha).unwrap();
        let candidate = branch_and_bound_entmax_content_aware(case, &index).unwrap();
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
    fn content_aware_index_is_deterministic_and_complete() {
        let keys = vec![
            3.0, 0.0, -3.0, 0.0, 2.0, 1.0, -2.0, -1.0, 1.0, 2.0, -1.0, -2.0,
        ];
        let left = build_content_aware_key_index(&keys, 2, 6, 2).unwrap();
        let right = build_content_aware_key_index(&keys, 2, 6, 2).unwrap();
        assert_eq!(left, right);
        let mut permutation = left.permutation.clone();
        permutation.sort_unstable();
        assert_eq!(permutation, (0..6).collect::<Vec<_>>());
    }

    #[test]
    fn hybrid_bounds_dominate_dense_scores_and_do_not_exceed_components() {
        let case = QueryKeyPagedCase {
            query: vec![1.0, -0.5, 2.0],
            keys: vec![
                1.0, 0.0, 2.0, -1.0, 3.0, 0.5, 2.0, -2.0, 1.0, 0.0, 1.0, -3.0,
            ],
            head_dim: 3,
            page_size: 4,
            alpha: 1.5,
            score_scale: 3.0_f64.sqrt().recip(),
        };
        let index = build_content_aware_key_index(&case.keys, 3, 4, 1).unwrap();
        let bounds = query_content_aware_upper_bounds(&case, &index).unwrap();
        let scores = dense_qk_scores(&case).unwrap();
        validate_bounds_against_dense_scores(&index, &bounds, &scores).unwrap();
        for node_bounds in bounds {
            assert!(node_bounds.hybrid_upper <= node_bounds.box_upper);
            assert!(node_bounds.hybrid_upper <= node_bounds.ball_upper);
        }
    }

    #[test]
    fn ball_component_can_tighten_cross_coordinate_box() {
        let case = QueryKeyPagedCase {
            query: vec![1.0, 1.0],
            keys: vec![1.0, 0.0, -1.0, 0.0, 0.0, 1.0, 0.0, -1.0],
            head_dim: 2,
            page_size: 4,
            alpha: 2.0,
            score_scale: 1.0,
        };
        let index = build_content_aware_key_index(&case.keys, 2, 4, 4).unwrap();
        let bounds = query_content_aware_upper_bounds(&case, &index).unwrap();
        let root = bounds[index.roots[0]];
        assert!(root.ball_upper < root.box_upper);
        assert_eq!(root.hybrid_upper.to_bits(), root.ball_upper.to_bits());
    }

    #[test]
    fn content_partition_beats_contiguous_tree_on_interleaved_clusters() {
        let case = QueryKeyPagedCase {
            query: vec![1.0, 0.0],
            keys: vec![
                10.0, 0.0, -20.0, 7.0, 9.5, 0.0, -21.0, -7.0, 9.0, 0.0, -30.0, 9.0, 8.5, 0.0,
                -31.0, -9.0,
            ],
            head_dim: 2,
            page_size: 8,
            alpha: 2.0,
            score_scale: 1.0,
        };
        let contiguous_index = build_hierarchical_key_index(&case.keys, 2, 8, 2).unwrap();
        let contiguous = branch_and_bound_entmax_hierarchical(&case, &contiguous_index).unwrap();
        let content = assert_candidate_matches_dense(&case, 2);
        assert!(content.metrics.tokens_loaded < contiguous.metrics.tokens_loaded);
        assert_eq!(content.metrics.tokens_loaded, 2);
    }

    #[test]
    fn content_aware_candidate_matches_dense_for_entmax15_and_sparsemax() {
        let keys = vec![
            9.0, 0.0, -8.0, 1.0, 8.5, 0.2, -9.0, -1.0, 0.0, 7.0, -4.0, -4.0, 0.2, 6.5, -5.0, -3.0,
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
            assert_candidate_matches_dense(&case, 2);
        }
    }

    #[test]
    fn content_aware_hierarchy_degrades_safely_to_all_tokens() {
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
        let result = assert_candidate_matches_dense(&case, 2);
        assert_eq!(result.metrics.tokens_loaded, 8);
        assert_eq!(result.metrics.tokens_pruned, 0);
    }

    #[test]
    fn exhaustive_small_content_aware_trees_match_dense_oracle() {
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
                    assert_candidate_matches_dense(&case, 1);
                    assert_candidate_matches_dense(&case, 2);
                }
            }
        }
    }
}
