#![forbid(unsafe_code)]

mod index;

pub use index::build_hierarchical_key_index;

use std::cmp::Ordering;
use std::collections::BTreeSet;

use ada_a4_entmax_bnb::{EntmaxDistribution, dense_entmax, entmax_threshold_bracket};
use ada_a4_qk_box::{PageKeyBox, QueryKeyPagedCase, dense_qk_scores};
use ada_core::KeyFingerprint;

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
        self.children().is_none()
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
    key_fingerprint: KeyFingerprint,
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

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct LazyHierarchicalMetrics {
    pub nodes_total: usize,
    pub nodes_expanded: usize,
    pub subtrees_pruned: usize,
    pub bound_evaluations: usize,
    pub bound_cache_hits: usize,
    pub nodes_never_evaluated: usize,
    pub leaves_total: usize,
    pub leaves_loaded: usize,
    pub tokens_loaded: usize,
    pub tokens_pruned: usize,
    pub threshold_solves: usize,
}

// E5 derived ratios are diagnostics only. They never participate in bound
// construction, traversal ordering, threshold solving, or pruning decisions.
#[allow(clippy::cast_precision_loss)]
fn usize_ratio(numerator: usize, denominator: usize) -> f64 {
    debug_assert!(denominator != 0);
    if denominator == 0 {
        return 0.0;
    }
    numerator as f64 / denominator as f64
}

impl LazyHierarchicalMetrics {
    #[must_use]
    pub fn bound_evaluation_fraction(self) -> f64 {
        if self.nodes_total == 0 {
            0.0
        } else {
            usize_ratio(self.bound_evaluations, self.nodes_total)
        }
    }

    #[must_use]
    pub fn bound_avoidance(self) -> f64 {
        1.0 - self.bound_evaluation_fraction()
    }

    #[must_use]
    pub fn score_load_fraction(self) -> f64 {
        let tokens_total = self.tokens_loaded + self.tokens_pruned;
        if tokens_total == 0 {
            0.0
        } else {
            usize_ratio(self.tokens_loaded, tokens_total)
        }
    }

    #[must_use]
    pub fn score_avoidance(self) -> f64 {
        1.0 - self.score_load_fraction()
    }

    #[must_use]
    pub fn bound_evaluations_per_loaded_token(self) -> f64 {
        if self.tokens_loaded == 0 {
            0.0
        } else {
            usize_ratio(self.bound_evaluations, self.tokens_loaded)
        }
    }

    #[must_use]
    pub fn bound_evaluations_per_pruned_token(self) -> f64 {
        if self.tokens_pruned == 0 {
            0.0
        } else {
            usize_ratio(self.bound_evaluations, self.tokens_pruned)
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct LazyHierarchicalResult {
    pub distribution: EntmaxDistribution,
    pub loaded_tokens: Vec<bool>,
    pub metrics: LazyHierarchicalMetrics,
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct PriorityLazyHierarchicalMetrics {
    pub nodes_total: usize,
    pub nodes_expanded: usize,
    pub subtrees_pruned: usize,
    pub bound_evaluations: usize,
    pub nodes_never_evaluated: usize,
    pub frontier_insertions: usize,
    pub frontier_min_checks: usize,
    pub frontier_max_pops: usize,
    pub leaves_total: usize,
    pub leaves_loaded: usize,
    pub tokens_loaded: usize,
    pub tokens_pruned: usize,
    pub threshold_solves: usize,
}

impl PriorityLazyHierarchicalMetrics {
    #[must_use]
    pub fn bound_evaluation_fraction(self) -> f64 {
        if self.nodes_total == 0 {
            0.0
        } else {
            usize_ratio(self.bound_evaluations, self.nodes_total)
        }
    }

    #[must_use]
    pub fn bound_avoidance(self) -> f64 {
        1.0 - self.bound_evaluation_fraction()
    }

    #[must_use]
    pub fn score_load_fraction(self) -> f64 {
        let tokens_total = self.tokens_loaded + self.tokens_pruned;
        if tokens_total == 0 {
            0.0
        } else {
            usize_ratio(self.tokens_loaded, tokens_total)
        }
    }

    #[must_use]
    pub fn score_avoidance(self) -> f64 {
        1.0 - self.score_load_fraction()
    }

    #[must_use]
    pub const fn frontier_logical_operations(self) -> usize {
        self.frontier_insertions + self.frontier_min_checks + self.frontier_max_pops
    }

    #[must_use]
    pub fn frontier_logical_operations_per_pruned_token(self) -> f64 {
        if self.tokens_pruned == 0 {
            0.0
        } else {
            usize_ratio(self.frontier_logical_operations(), self.tokens_pruned)
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct PriorityLazyHierarchicalResult {
    pub distribution: EntmaxDistribution,
    pub loaded_tokens: Vec<bool>,
    pub metrics: PriorityLazyHierarchicalMetrics,
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
    if KeyFingerprint::of_f64_slice(&case.keys) != index.key_fingerprint {
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

struct LazyBoundContext<'a> {
    case: &'a QueryKeyPagedCase,
    index: &'a HierarchicalKeyIndex,
    dense_scores: &'a [f64],
}

fn lazy_bound(
    context: &LazyBoundContext<'_>,
    cache: &mut [Option<f64>],
    node_index: usize,
    metrics: &mut LazyHierarchicalMetrics,
) -> Result<f64, &'static str> {
    if let Some(bound) = cache[node_index] {
        metrics.bound_cache_hits += 1;
        return Ok(bound);
    }

    let node = &context.index.nodes[node_index];
    let bound = query_box_bound(&context.case.query, &node.key_box, context.case.score_scale)?;

    let actual_maximum = context.dense_scores[node.start_token..node.end_token]
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);

    if bound < actual_maximum {
        return Err("ADA-A5 lazy f64 hierarchy bound is not conservative against the dense oracle");
    }

    cache[node_index] = Some(bound);
    metrics.bound_evaluations += 1;
    Ok(bound)
}

fn highest_lazy_bound_position(
    context: &LazyBoundContext<'_>,
    cache: &mut [Option<f64>],
    frontier: &[usize],
    metrics: &mut LazyHierarchicalMetrics,
) -> Result<usize, &'static str> {
    if frontier.is_empty() {
        return Err("ADA-A5 lazy frontier must not be empty");
    }

    let mut best_position = 0;
    let mut best_bound = lazy_bound(context, cache, frontier[0], metrics)?;

    for (position, &node_index) in frontier.iter().enumerate().skip(1) {
        let bound = lazy_bound(context, cache, node_index, metrics)?;

        if bound > best_bound {
            best_position = position;
            best_bound = bound;
        }
    }

    Ok(best_position)
}

fn load_lazy_leaf(
    node: &HierarchyNode,
    loaded_tokens: &mut [bool],
    loaded_indices: &mut Vec<usize>,
    metrics: &mut LazyHierarchicalMetrics,
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

fn seed_highest_lazy_bound_leaf(
    context: &LazyBoundContext<'_>,
    cache: &mut [Option<f64>],
    frontier: &mut Vec<usize>,
    loaded_tokens: &mut [bool],
    loaded_indices: &mut Vec<usize>,
    metrics: &mut LazyHierarchicalMetrics,
) -> Result<(), &'static str> {
    let root_position = highest_lazy_bound_position(context, cache, &context.index.roots, metrics)?;

    let mut current = context.index.roots[root_position];

    frontier.extend(
        context
            .index
            .roots
            .iter()
            .enumerate()
            .filter_map(|(position, &root)| (position != root_position).then_some(root)),
    );

    loop {
        let node = &context.index.nodes[current];

        if let Some((left, right)) = node.children() {
            metrics.nodes_expanded += 1;

            let left_bound = lazy_bound(context, cache, left, metrics)?;

            let right_bound = lazy_bound(context, cache, right, metrics)?;

            if left_bound >= right_bound {
                frontier.push(right);
                current = left;
            } else {
                frontier.push(left);
                current = right;
            }
        } else {
            load_lazy_leaf(node, loaded_tokens, loaded_indices, metrics);
            return Ok(());
        }
    }
}

fn finalize_lazy_metrics(mut metrics: LazyHierarchicalMetrics) -> LazyHierarchicalMetrics {
    debug_assert!(metrics.bound_evaluations <= metrics.nodes_total);
    metrics.nodes_never_evaluated = metrics
        .nodes_total
        .saturating_sub(metrics.bound_evaluations);
    metrics
}

#[derive(Debug, Clone, Copy)]
struct PriorityFrontierEntry {
    bound: f64,
    node_index: usize,
}

impl PartialEq for PriorityFrontierEntry {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for PriorityFrontierEntry {}

impl PartialOrd for PriorityFrontierEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for PriorityFrontierEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        self.bound
            .total_cmp(&other.bound)
            // For equal bounds, a lower node index has higher expansion
            // priority. This keeps the tie-break deterministic and matches
            // left/root order for the common static hierarchy cases.
            .then_with(|| other.node_index.cmp(&self.node_index))
    }
}

fn evaluate_priority_bound(
    context: &LazyBoundContext<'_>,
    evaluated: &mut [bool],
    node_index: usize,
    metrics: &mut PriorityLazyHierarchicalMetrics,
) -> Result<f64, &'static str> {
    if evaluated[node_index] {
        return Err("ADA-A5 E5b priority frontier requested a node bound more than once");
    }

    let node = &context.index.nodes[node_index];
    let bound = query_box_bound(&context.case.query, &node.key_box, context.case.score_scale)?;

    let actual_maximum = context.dense_scores[node.start_token..node.end_token]
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);

    if bound < actual_maximum {
        return Err("ADA-A5 E5b f64 hierarchy bound is not conservative against the dense oracle");
    }

    evaluated[node_index] = true;
    metrics.bound_evaluations += 1;

    Ok(bound)
}

fn insert_priority_entry(
    frontier: &mut BTreeSet<PriorityFrontierEntry>,
    node_index: usize,
    bound: f64,
    metrics: &mut PriorityLazyHierarchicalMetrics,
) -> Result<(), &'static str> {
    let inserted = frontier.insert(PriorityFrontierEntry { bound, node_index });

    if !inserted {
        return Err("ADA-A5 E5b priority frontier contains a duplicate node");
    }

    metrics.frontier_insertions += 1;
    Ok(())
}

fn load_priority_leaf(
    node: &HierarchyNode,
    loaded_tokens: &mut [bool],
    loaded_indices: &mut Vec<usize>,
    metrics: &mut PriorityLazyHierarchicalMetrics,
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

fn seed_priority_leaf(
    context: &LazyBoundContext<'_>,
    evaluated: &mut [bool],
    frontier: &mut BTreeSet<PriorityFrontierEntry>,
    loaded_tokens: &mut [bool],
    loaded_indices: &mut Vec<usize>,
    metrics: &mut PriorityLazyHierarchicalMetrics,
) -> Result<(), &'static str> {
    for &root in &context.index.roots {
        let bound = evaluate_priority_bound(context, evaluated, root, metrics)?;
        insert_priority_entry(frontier, root, bound, metrics)?;
    }

    let seed = frontier
        .pop_last()
        .ok_or("ADA-A5 E5b priority frontier must contain at least one root")?;

    metrics.frontier_max_pops += 1;

    let mut current = seed.node_index;

    loop {
        let node = &context.index.nodes[current];

        if let Some((left, right)) = node.children() {
            metrics.nodes_expanded += 1;

            let left_bound = evaluate_priority_bound(context, evaluated, left, metrics)?;
            let right_bound = evaluate_priority_bound(context, evaluated, right, metrics)?;

            if left_bound >= right_bound {
                insert_priority_entry(frontier, right, right_bound, metrics)?;
                current = left;
            } else {
                insert_priority_entry(frontier, left, left_bound, metrics)?;
                current = right;
            }
        } else {
            load_priority_leaf(node, loaded_tokens, loaded_indices, metrics);
            return Ok(());
        }
    }
}

fn finalize_priority_metrics(
    mut metrics: PriorityLazyHierarchicalMetrics,
) -> PriorityLazyHierarchicalMetrics {
    debug_assert!(metrics.bound_evaluations <= metrics.nodes_total);
    metrics.nodes_never_evaluated = metrics
        .nodes_total
        .saturating_sub(metrics.bound_evaluations);
    metrics
}

fn subset_scores(dense_scores: &[f64], loaded_indices: &[usize]) -> Vec<f64> {
    loaded_indices
        .iter()
        .map(|&index| dense_scores[index])
        .collect()
}

/// Post-hoc support exactness certificate.
///
/// Subtrees are pruned in early rounds against that round's `tau_lower`, which
/// is smaller than or equal to the terminating bracket's endpoint only under
/// the subset-monotonicity assumption. This certificate re-checks every pruned
/// subtree against the TERMINATING threshold endpoint and fails closed when a
/// single one violates it, turning monotonicity from an assumption into a
/// verified fact for the published distribution.
fn certify_support_exactness(
    pruned_bounds: &[f64],
    entmax_scale: f64,
    tau_lower: f64,
) -> Result<(), &'static str> {
    for &bound in pruned_bounds {
        if entmax_scale * bound > tau_lower {
            return Err("ADA-A5 support exactness certificate failed");
        }
    }
    Ok(())
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
    let mut pruned_bounds = Vec::new();

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
                    pruned_bounds.push(bounds[node_index]);
                } else {
                    unresolved.push(node_index);
                }
            }
            frontier = unresolved;

            if frontier.is_empty() {
                certify_support_exactness(&pruned_bounds, entmax_scale, tau_lower)?;
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

/// Run the E5 exact hierarchical controller with lazy node-bound evaluation.
///
/// This function preserves the historical A5 traversal and pruning certificate,
/// but each hierarchy-node coordinate-box bound is evaluated only on first
/// demand and is cached thereafter.
///
/// Dense Q/K scores are still constructed as an independent research oracle.
/// They are not counted as candidate work. A node is checked against the dense
/// oracle only when its lazy bound is first evaluated.
///
/// # Errors
///
/// Returns an error for invalid Q/K/index inputs, a non-conservative evaluated
/// f64 bound, or an Entmax threshold-solver failure.
#[must_use = "the exact result and lazy-bound work metrics should be checked"]
pub fn branch_and_bound_entmax_hierarchical_lazy(
    case: &QueryKeyPagedCase,
    index: &HierarchicalKeyIndex,
) -> Result<LazyHierarchicalResult, &'static str> {
    validate_query_index(case, index)?;
    let dense_scores = dense_qk_scores(case)?;

    let mut metrics = LazyHierarchicalMetrics {
        nodes_total: index.node_count(),
        leaves_total: index.leaf_count(),
        ..LazyHierarchicalMetrics::default()
    };

    let context = LazyBoundContext {
        case,
        index,
        dense_scores: &dense_scores,
    };

    let mut bound_cache = vec![None; index.node_count()];
    let mut loaded_tokens = vec![false; index.key_count];
    let mut loaded_indices = Vec::with_capacity(index.key_count);
    let mut frontier = Vec::new();
    let mut pruned_bounds = Vec::new();

    seed_highest_lazy_bound_leaf(
        &context,
        &mut bound_cache,
        &mut frontier,
        &mut loaded_tokens,
        &mut loaded_indices,
        &mut metrics,
    )?;

    loop {
        let loaded_scores = subset_scores(&dense_scores, &loaded_indices);

        let threshold = entmax_threshold_bracket(&loaded_scores, case.alpha)?;

        metrics.threshold_solves += 1;

        let tau_lower = threshold.lower;
        let entmax_scale = case.alpha - 1.0;

        loop {
            let mut unresolved = Vec::with_capacity(frontier.len());

            for node_index in frontier.drain(..) {
                let bound = lazy_bound(&context, &mut bound_cache, node_index, &mut metrics)?;

                if entmax_scale * bound <= tau_lower {
                    metrics.subtrees_pruned += 1;
                    metrics.tokens_pruned += index.nodes[node_index].token_count();
                    pruned_bounds.push(bound);
                } else {
                    unresolved.push(node_index);
                }
            }

            frontier = unresolved;

            if frontier.is_empty() {
                certify_support_exactness(&pruned_bounds, entmax_scale, tau_lower)?;
                debug_assert_eq!(
                    metrics.tokens_loaded + metrics.tokens_pruned,
                    index.key_count
                );

                let metrics = finalize_lazy_metrics(metrics);

                return Ok(LazyHierarchicalResult {
                    distribution: finalize_distribution(case, &dense_scores, &loaded_indices)?,
                    loaded_tokens,
                    metrics,
                });
            }

            let best_position =
                highest_lazy_bound_position(&context, &mut bound_cache, &frontier, &mut metrics)?;

            let node_index = frontier.swap_remove(best_position);

            let node = &index.nodes[node_index];

            if let Some((left, right)) = node.children() {
                metrics.nodes_expanded += 1;
                frontier.push(left);
                frontier.push(right);
                continue;
            }

            load_lazy_leaf(node, &mut loaded_tokens, &mut loaded_indices, &mut metrics);

            break;
        }
    }
}

/// Run the E5b exact hierarchical controller with an ordered priority frontier.
///
/// Unlike the historical E5 lazy controller, every evaluated node bound is
/// carried inside the frontier entry and may be evaluated at most once. The
/// minimum frontier bound drives threshold pruning and the maximum drives
/// expansion, eliminating repeated full-frontier bound requests.
///
/// Exact equal-bound ties use a deterministic node-index tie-break and may
/// therefore choose a different valid traversal than the historical Vec-based
/// controller. Dense Entmax parity and support preservation remain the
/// correctness contract.
///
/// Dense Q/K scores are still constructed solely as the independent research
/// oracle and for evaluated-bound validation.
///
/// # Errors
///
/// Returns an error for invalid Q/K/index inputs, a repeated node-bound
/// evaluation, a duplicate frontier node, a non-conservative evaluated f64
/// bound, or an Entmax threshold-solver failure.
#[must_use = "the exact result and priority-frontier work metrics should be checked"]
pub fn branch_and_bound_entmax_hierarchical_priority_lazy(
    case: &QueryKeyPagedCase,
    index: &HierarchicalKeyIndex,
) -> Result<PriorityLazyHierarchicalResult, &'static str> {
    validate_query_index(case, index)?;
    let dense_scores = dense_qk_scores(case)?;

    let mut metrics = PriorityLazyHierarchicalMetrics {
        nodes_total: index.node_count(),
        leaves_total: index.leaf_count(),
        ..PriorityLazyHierarchicalMetrics::default()
    };

    let context = LazyBoundContext {
        case,
        index,
        dense_scores: &dense_scores,
    };

    let mut evaluated = vec![false; index.node_count()];
    let mut loaded_tokens = vec![false; index.key_count];
    let mut loaded_indices = Vec::with_capacity(index.key_count);
    let mut frontier = BTreeSet::new();
    let mut pruned_bounds = Vec::new();

    seed_priority_leaf(
        &context,
        &mut evaluated,
        &mut frontier,
        &mut loaded_tokens,
        &mut loaded_indices,
        &mut metrics,
    )?;

    loop {
        let loaded_scores = subset_scores(&dense_scores, &loaded_indices);
        let threshold = entmax_threshold_bracket(&loaded_scores, case.alpha)?;

        metrics.threshold_solves += 1;

        let tau_lower = threshold.lower;
        let entmax_scale = case.alpha - 1.0;

        loop {
            while let Some(entry) = frontier.first().copied() {
                metrics.frontier_min_checks += 1;

                if entmax_scale * entry.bound > tau_lower {
                    break;
                }

                let removed = frontier.pop_first();

                debug_assert_eq!(removed, Some(entry));

                metrics.subtrees_pruned += 1;
                metrics.tokens_pruned += index.nodes[entry.node_index].token_count();
                pruned_bounds.push(entry.bound);
            }

            if frontier.is_empty() {
                certify_support_exactness(&pruned_bounds, entmax_scale, tau_lower)?;
                debug_assert_eq!(
                    metrics.tokens_loaded + metrics.tokens_pruned,
                    index.key_count
                );

                let metrics = finalize_priority_metrics(metrics);

                return Ok(PriorityLazyHierarchicalResult {
                    distribution: finalize_distribution(case, &dense_scores, &loaded_indices)?,
                    loaded_tokens,
                    metrics,
                });
            }

            let entry = frontier
                .pop_last()
                .ok_or("ADA-A5 E5b priority frontier unexpectedly became empty")?;

            metrics.frontier_max_pops += 1;

            let node = &index.nodes[entry.node_index];

            if let Some((left, right)) = node.children() {
                metrics.nodes_expanded += 1;

                let left_bound =
                    evaluate_priority_bound(&context, &mut evaluated, left, &mut metrics)?;

                let right_bound =
                    evaluate_priority_bound(&context, &mut evaluated, right, &mut metrics)?;

                insert_priority_entry(&mut frontier, left, left_bound, &mut metrics)?;
                insert_priority_entry(&mut frontier, right, right_bound, &mut metrics)?;

                continue;
            }

            load_priority_leaf(node, &mut loaded_tokens, &mut loaded_indices, &mut metrics);

            break;
        }
    }
}

#[cfg(test)]
mod tests;
