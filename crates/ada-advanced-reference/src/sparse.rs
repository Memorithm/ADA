//! Static, dynamic top-k, and routed sparse-attention references.

use ada_core::SemanticId;

use crate::{AdvancedReferenceError, check_len, checked_product, ensure_finite, stable_softmax};

/// Sparse key-selection semantics for one reference candidate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SparseSelection {
    /// Exact sorted logical key indices for every query row.
    StaticRows(Vec<Vec<usize>>),
    /// Select the exact top-k affinity scores per query with key-index tie break.
    DynamicTopK { k: usize },
    /// Query and key route ids must match exactly.
    Routed {
        /// Route id for every query row.
        query_routes: Vec<u32>,
        /// Route id for every key row.
        key_routes: Vec<u32>,
    },
}

/// Explicit semantic rule for one sparse reference candidate.
#[derive(Debug, Clone, PartialEq)]
pub struct SparseAttentionSpec {
    semantic: SemanticId,
    scale: f64,
    selection: SparseSelection,
}

impl SparseAttentionSpec {
    /// Construct an explicit sparse-attention semantic rule.
    ///
    /// # Errors
    ///
    /// Rejects a non-finite/non-positive affinity scale or an impossible zero-k
    /// dynamic selector. Row-specific bounds are validated during execution.
    pub fn new(
        semantic: SemanticId,
        scale: f64,
        selection: SparseSelection,
    ) -> Result<Self, AdvancedReferenceError> {
        if !scale.is_finite() || scale <= 0.0 {
            return Err(AdvancedReferenceError::InvalidField("scale"));
        }
        if matches!(selection, SparseSelection::DynamicTopK { k: 0 }) {
            return Err(AdvancedReferenceError::InvalidField("dynamic top-k k"));
        }
        Ok(Self {
            semantic,
            scale,
            selection,
        })
    }

    /// Semantic identity of this sparse rule.
    #[must_use]
    pub const fn semantic(&self) -> &SemanticId {
        &self.semantic
    }

    /// Affinity scale.
    #[must_use]
    pub const fn scale(&self) -> f64 {
        self.scale
    }

    /// Selection semantics.
    #[must_use]
    pub const fn selection(&self) -> &SparseSelection {
        &self.selection
    }
}

/// Explicit single-head row-major Q/K/V sparse-reference input.
#[derive(Debug, Clone, PartialEq)]
pub struct SparseInput {
    /// Query row count.
    pub query_count: usize,
    /// Key/value row count.
    pub key_count: usize,
    /// Query/key dimension.
    pub qk_dimension: usize,
    /// Value/output dimension.
    pub value_dimension: usize,
    /// Row-major queries.
    pub queries: Vec<f64>,
    /// Row-major keys.
    pub keys: Vec<f64>,
    /// Row-major values.
    pub values: Vec<f64>,
}

/// Sparse-reference output plus exact selected-key witnesses.
#[derive(Debug, Clone, PartialEq)]
pub struct SparseOutput {
    output: Vec<f64>,
    selected_keys: Vec<Vec<usize>>,
}

impl SparseOutput {
    /// Row-major sparse attention output.
    #[must_use]
    pub fn output(&self) -> &[f64] {
        &self.output
    }

    /// Selected logical key indices for each query row.
    #[must_use]
    pub fn selected_keys(&self) -> &[Vec<usize>] {
        &self.selected_keys
    }
}

/// Evaluate static, dynamic-top-k, or routed sparse softmax attention.
///
/// Selection is part of this reference semantic, not an implementation hint.
/// Dynamic ties are deterministic: larger score first, then smaller key index.
///
/// # Errors
///
/// Rejects malformed shapes, non-finite input, duplicate/out-of-range static
/// indices, mismatched routes, empty routed/static rows, or top-k larger than
/// the available key set.
pub fn evaluate_sparse(
    spec: &SparseAttentionSpec,
    input: &SparseInput,
) -> Result<SparseOutput, AdvancedReferenceError> {
    validate_input(spec, input)?;
    let mut output =
        vec![0.0; checked_product(input.query_count, input.value_dimension, "sparse output")?];
    let mut selected_keys = Vec::with_capacity(input.query_count);
    for query in 0..input.query_count {
        let scores = score_row(spec, input, query)?;
        let selected = selected_row(spec, input, query, &scores)?;
        let selected_scores = selected.iter().map(|&key| scores[key]).collect::<Vec<_>>();
        let weights = stable_softmax(&selected_scores)?;
        let out_start = query * input.value_dimension;
        for (&key, &weight) in selected.iter().zip(&weights) {
            let value_start = key * input.value_dimension;
            for dimension in 0..input.value_dimension {
                output[out_start + dimension] += weight * input.values[value_start + dimension];
            }
        }
        selected_keys.push(selected);
    }
    ensure_finite(&output, "sparse value mixing")?;
    Ok(SparseOutput {
        output,
        selected_keys,
    })
}

fn validate_input(
    spec: &SparseAttentionSpec,
    input: &SparseInput,
) -> Result<(), AdvancedReferenceError> {
    if input.query_count == 0
        || input.key_count == 0
        || input.qk_dimension == 0
        || input.value_dimension == 0
    {
        return Err(AdvancedReferenceError::InvalidField(
            "zero sparse dimension",
        ));
    }
    check_len(
        "queries",
        checked_product(input.query_count, input.qk_dimension, "queries")?,
        input.queries.len(),
    )?;
    check_len(
        "keys",
        checked_product(input.key_count, input.qk_dimension, "keys")?,
        input.keys.len(),
    )?;
    check_len(
        "values",
        checked_product(input.key_count, input.value_dimension, "values")?,
        input.values.len(),
    )?;
    ensure_finite(&input.queries, "sparse queries")?;
    ensure_finite(&input.keys, "sparse keys")?;
    ensure_finite(&input.values, "sparse values")?;
    match spec.selection() {
        SparseSelection::StaticRows(rows) => {
            check_len("static rows", input.query_count, rows.len())
        }
        SparseSelection::DynamicTopK { k } => {
            if *k > input.key_count {
                Err(AdvancedReferenceError::InvalidField(
                    "dynamic top-k exceeds key count",
                ))
            } else {
                Ok(())
            }
        }
        SparseSelection::Routed {
            query_routes,
            key_routes,
        } => {
            check_len("query routes", input.query_count, query_routes.len())?;
            check_len("key routes", input.key_count, key_routes.len())
        }
    }
}

fn score_row(
    spec: &SparseAttentionSpec,
    input: &SparseInput,
    query: usize,
) -> Result<Vec<f64>, AdvancedReferenceError> {
    let query_start = query * input.qk_dimension;
    let query_row = &input.queries[query_start..query_start + input.qk_dimension];
    let mut scores = Vec::with_capacity(input.key_count);
    for key in 0..input.key_count {
        let key_start = key * input.qk_dimension;
        let key_row = &input.keys[key_start..key_start + input.qk_dimension];
        let dot = query_row
            .iter()
            .zip(key_row)
            .map(|(&left, &right)| left * right)
            .sum::<f64>();
        let score = dot * spec.scale();
        if !score.is_finite() {
            return Err(AdvancedReferenceError::NonFinite("sparse affinity"));
        }
        scores.push(score);
    }
    Ok(scores)
}

fn selected_row(
    spec: &SparseAttentionSpec,
    input: &SparseInput,
    query: usize,
    scores: &[f64],
) -> Result<Vec<usize>, AdvancedReferenceError> {
    let mut selected = match spec.selection() {
        SparseSelection::StaticRows(rows) => validate_static_row(&rows[query], input.key_count)?,
        SparseSelection::DynamicTopK { k } => {
            let mut candidates = (0..input.key_count).collect::<Vec<_>>();
            candidates.sort_by(|&left, &right| {
                scores[right]
                    .total_cmp(&scores[left])
                    .then_with(|| left.cmp(&right))
            });
            candidates.truncate(*k);
            candidates
        }
        SparseSelection::Routed {
            query_routes,
            key_routes,
        } => key_routes
            .iter()
            .enumerate()
            .filter_map(|(key, route)| (*route == query_routes[query]).then_some(key))
            .collect(),
    };
    if selected.is_empty() {
        return Err(AdvancedReferenceError::EmptySelection(query));
    }
    selected.sort_unstable();
    Ok(selected)
}

fn validate_static_row(
    row: &[usize],
    key_count: usize,
) -> Result<Vec<usize>, AdvancedReferenceError> {
    if row.is_empty() {
        return Ok(Vec::new());
    }
    let mut previous = None;
    for &key in row {
        if key >= key_count {
            return Err(AdvancedReferenceError::InvalidField("static key range"));
        }
        if previous.is_some_and(|value| value >= key) {
            return Err(AdvancedReferenceError::InvalidField(
                "static rows must be strictly sorted and unique",
            ));
        }
        previous = Some(key);
    }
    Ok(row.to_vec())
}

#[cfg(test)]
mod tests {
    use ada_core::{SemanticFamily, SemanticId};

    use super::*;

    fn id(name: &str) -> SemanticId {
        SemanticId::new(SemanticFamily::Experimental, name, 1).unwrap()
    }

    fn input() -> SparseInput {
        SparseInput {
            query_count: 2,
            key_count: 3,
            qk_dimension: 1,
            value_dimension: 1,
            queries: vec![1.0, 0.0],
            keys: vec![1.0, 2.0, 2.0],
            values: vec![10.0, 20.0, 30.0],
        }
    }

    #[test]
    fn static_all_rows_matches_dense_reference() {
        let spec = SparseAttentionSpec::new(
            id("static"),
            1.0,
            SparseSelection::StaticRows(vec![vec![0, 1, 2], vec![0, 1, 2]]),
        )
        .unwrap();
        let result = evaluate_sparse(&spec, &input()).unwrap();
        assert_eq!(result.selected_keys(), &[vec![0, 1, 2], vec![0, 1, 2]]);
        assert!((result.output()[1] - 20.0).abs() < 1.0e-12);
    }

    #[test]
    fn dynamic_topk_uses_deterministic_key_tie_break() {
        let spec =
            SparseAttentionSpec::new(id("dynamic"), 1.0, SparseSelection::DynamicTopK { k: 1 })
                .unwrap();
        let result = evaluate_sparse(&spec, &input()).unwrap();
        assert_eq!(result.selected_keys()[0], vec![1]);
        assert_eq!(result.output()[0], 20.0);
    }

    #[test]
    fn routed_rows_only_mix_matching_route() {
        let spec = SparseAttentionSpec::new(
            id("routed"),
            1.0,
            SparseSelection::Routed {
                query_routes: vec![7, 9],
                key_routes: vec![9, 7, 7],
            },
        )
        .unwrap();
        let result = evaluate_sparse(&spec, &input()).unwrap();
        assert_eq!(result.selected_keys(), &[vec![1, 2], vec![0]]);
        assert_eq!(result.output()[1], 10.0);
    }

    #[test]
    fn duplicate_static_indices_fail_closed() {
        let spec = SparseAttentionSpec::new(
            id("duplicates"),
            1.0,
            SparseSelection::StaticRows(vec![vec![0, 0], vec![0]]),
        )
        .unwrap();
        assert_eq!(
            evaluate_sparse(&spec, &input()),
            Err(AdvancedReferenceError::InvalidField(
                "static rows must be strictly sorted and unique"
            ))
        );
    }
}
