use ada_a4_qk_box::PageKeyBox;
use ada_core::KeyFingerprint;

use super::{HierarchicalKeyIndex, HierarchyNode};

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
        key_fingerprint: KeyFingerprint::of_f64_slice(keys),
        nodes,
        roots,
        leaves,
    })
}
