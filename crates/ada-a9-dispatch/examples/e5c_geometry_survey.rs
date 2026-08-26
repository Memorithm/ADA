//! E5c-style geometry ablation survey: historical pivot-diameter split with
//! mean-centered balls versus the v2 PCA-cut / shrunk-ball variant.
//!
//! Usage:
//!   cargo run --release -p ada-a9-dispatch --example `e5c_geometry_survey`
//!   cargo run --release -p ada-a9-dispatch --example `e5c_geometry_survey` -- `<trace.adaqk>` [alpha] [page-size] [leaf-size]
//!
//! Without arguments the survey runs a deterministic synthetic grid; with a
//! frozen natural-trace path it replays the real slice instead. In both modes
//! every configuration must stay exact against the dense oracle; the report
//! compares subtree/token pruning fractions so the geometry question ("does
//! the shrunk ball ever win?") gets measured instead of assumed.

#![forbid(unsafe_code)]

use ada_a4_entmax_bnb::dense_entmax;
use ada_a4_qk_box::{QueryKeyPagedCase, dense_qk_scores};
use ada_a5_content_aware_bounds::{
    ContentAwareGeometry, branch_and_bound_entmax_content_aware,
    build_content_aware_key_index_with_geometry,
};
use std::env;

struct SurveyRow {
    label: String,
    legacy_tokens_pruned: usize,
    v2_tokens_pruned: usize,
    key_count: usize,
    legacy_exact: bool,
    v2_exact: bool,
}

fn run_geometry(
    case: &QueryKeyPagedCase,
    leaf_size: usize,
    geometry: ContentAwareGeometry,
    dense: &ada_a4_entmax_bnb::EntmaxDistribution,
) -> Option<(usize, bool)> {
    let index = build_content_aware_key_index_with_geometry(
        &case.keys,
        case.head_dim,
        case.page_size,
        leaf_size,
        geometry,
    )
    .ok()?;
    let result = branch_and_bound_entmax_content_aware(case, &index).ok()?;
    // Exactness gate: the dense oracle support must be loaded.
    let exact = dense
        .probabilities
        .iter()
        .enumerate()
        .all(|(token, probability)| *probability <= 1.0e-12 || result.loaded_tokens[token]);
    Some((result.metrics.tokens_pruned, exact))
}

fn survey_case(case: &QueryKeyPagedCase, leaf_size: usize, label: String) -> Option<SurveyRow> {
    let dense_scores = dense_qk_scores(case).ok()?;
    let dense = dense_entmax(&dense_scores, case.alpha).ok()?;

    let (legacy_tokens_pruned, legacy_exact) = run_geometry(
        case,
        leaf_size,
        ContentAwareGeometry::DiameterMeanBall,
        &dense,
    )?;
    let (v2_tokens_pruned, v2_exact) =
        run_geometry(case, leaf_size, ContentAwareGeometry::PcaShrunkBall, &dense)?;

    Some(SurveyRow {
        label,
        legacy_tokens_pruned,
        v2_tokens_pruned,
        key_count: case.key_count(),
        legacy_exact,
        v2_exact,
    })
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().skip(1).collect();
    println!("survey=ada_a5_e5c_geometry_ablation");
    println!(
        "label,key_count,tokens_pruned_legacy,tokens_pruned_v2,total,legacy_exact,v2_exact,v2_gain"
    );

    let rows: Vec<SurveyRow> = if let Some(trace_path) = args.first() {
        let alpha: f64 = args.get(1).map_or(2.0, |v| v.parse().unwrap_or(2.0));
        let page_size: usize = args.get(2).map_or(16, |v| v.parse().unwrap_or(16));
        let leaf_size: usize = args.get(3).map_or(8, |v| v.parse().unwrap_or(8));

        let corpus = ada_a5_real_qk_trace::read_trace_file(trace_path)?;
        corpus
            .records()
            .iter()
            .enumerate()
            .filter_map(|(index, record)| {
                let case = record.to_query_key_case(page_size, alpha).ok()?;
                survey_case(
                    &case,
                    leaf_size,
                    format!("natural-{index}-a{alpha}-p{page_size}-l{leaf_size}"),
                )
            })
            .collect()
    } else {
        synthetic_grid()
    };

    let mut total_legacy = 0_usize;
    let mut total_v2 = 0_usize;
    let mut total_keys = 0_usize;
    for row in &rows {
        total_legacy += row.legacy_tokens_pruned;
        total_v2 += row.v2_tokens_pruned;
        total_keys += row.key_count;
        assert!(
            row.legacy_exact,
            "legacy geometry lost support: {}",
            row.label
        );
        assert!(row.v2_exact, "v2 geometry lost support: {}", row.label);
        println!(
            "{},{},{},{},{},{},{},{}",
            row.label,
            row.key_count,
            row.legacy_tokens_pruned,
            row.v2_tokens_pruned,
            row.key_count,
            row.legacy_exact,
            row.v2_exact,
            i64::try_from(row.v2_tokens_pruned).unwrap_or(i64::MAX)
                - i64::try_from(row.legacy_tokens_pruned).unwrap_or(i64::MAX)
        );
    }

    println!(
        "=== TOTAL legacy={} v2={} keys={} delta={} ===",
        total_legacy,
        total_v2,
        total_keys,
        i64::try_from(total_v2).unwrap_or(i64::MAX)
            - i64::try_from(total_legacy).unwrap_or(i64::MAX)
    );
    Ok(())
}

fn synthetic_grid() -> Vec<SurveyRow> {
    let mut rows = Vec::new();
    for &key_count in &[256_usize, 512] {
        for &alpha in &[1.5, 2.0] {
            for &page_size in &[32_usize, 128] {
                let mut keys = Vec::with_capacity(key_count * 2);
                for i in 0..key_count {
                    #[allow(clippy::cast_precision_loss)]
                    let magnitude = i as f64 * 0.031_25;
                    let sign = if i % 2 == 0 { 1.0 } else { -1.0 };
                    #[allow(clippy::cast_precision_loss)]
                    let residue = (i % 11) as f64;
                    keys.extend_from_slice(&[sign * magnitude, residue * 0.5 - 2.5]);
                }
                let case = QueryKeyPagedCase {
                    query: vec![0.4, -0.9],
                    keys,
                    head_dim: 2,
                    page_size,
                    alpha,
                    score_scale: 2.0_f64.sqrt().recip(),
                };
                if let Some(row) = survey_case(
                    &case,
                    8,
                    format!("synthetic-n{key_count}-a{alpha}-p{page_size}"),
                ) {
                    rows.push(row);
                }
            }
        }
    }
    rows
}
