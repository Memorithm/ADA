//! Replay a frozen E4 natural Q/K trace through the ADA-A9 dispatcher.
//!
//! Usage: `cargo run --release -p ada-a9-dispatch --example e4_dispatch_replay -- <trace.adaqk> [alpha] [page-size] [leaf-size]`
//!
//! For every record the example lets the selector choose the plan, executes
//! it, and checks the dispatched distribution against the dense oracle within
//! the documented tolerance. Any mismatch or controller failure aborts with a
//! non-zero exit, so the run doubles as an end-to-end exactness gate wherever
//! the frozen trace file is available.

#![forbid(unsafe_code)]

use ada_a4_qk_box::dense_qk_scores;
use ada_a5_real_qk_trace::read_trace_file;
use ada_a9_dispatch::{DispatchOutcome, execute_selected_plan};
use std::collections::BTreeMap;
use std::env;
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = env::args().skip(1).collect();
    let [trace_path, alpha, page_size, leaf_size] = args.as_slice() else {
        return Err(
            "usage: e4_dispatch_replay <trace.adaqk> [alpha=2.0] [page-size=16] [leaf-size=8]"
                .to_string()
                .into(),
        );
    };

    let alpha: f64 = alpha.parse().unwrap_or(2.0);
    let page_size: usize = page_size.parse().unwrap_or(16);
    let leaf_size: usize = leaf_size.parse().unwrap_or(8);

    let corpus = read_trace_file(trace_path)?;
    println!(
        "dispatch_replay records={} alpha={alpha} page_size={page_size} leaf_size={leaf_size}",
        corpus.len()
    );
    println!("record,plan,tokens,max_abs_O,max_abs_LSE");

    let mut plan_counts: BTreeMap<&'static str, usize> = BTreeMap::new();
    let mut worst_output = 0.0_f64;
    let mut worst_lse = 0.0_f64;

    for (index, record) in corpus.records().iter().enumerate() {
        let case = record.to_query_key_case(page_size, alpha)?;
        case.validate()?;

        let outcome: DispatchOutcome = execute_selected_plan(&case, leaf_size)?;
        *plan_counts.entry(outcome.plan_name()).or_insert(0) += 1;

        // Dense reference for this record (independent path).
        let dense_scores = dense_qk_scores(&case)?;
        let dense = ada_a4_entmax_bnb::dense_entmax(&dense_scores, alpha)?;

        for (&selected, &reference) in outcome
            .distribution
            .probabilities
            .iter()
            .zip(dense.probabilities.iter())
        {
            worst_output = worst_output.max((selected - reference).abs());
        }
        worst_lse = worst_lse.max((outcome.distribution.tau - dense.tau).abs());

        if index % 128 == 0 || index + 1 == corpus.len() {
            println!(
                "{index},{},{},{worst_output:.3e},{worst_lse:.3e}",
                outcome.plan_name(),
                case.key_count()
            );
        }
    }

    println!("=== PLAN DISTRIBUTION ===");
    for (name, count) in &plan_counts {
        println!("{name}={count}");
    }
    println!("worst_abs_O={worst_output:.3e} worst_abs_tau={worst_lse:.3e}");

    if worst_output > 4.0e-12 || worst_lse > 2.0e-12 {
        return Err("dispatch parity violated on the natural trace".into());
    }

    Ok(())
}

impl DispatchOutcomeLike for DispatchOutcome {
    fn plan_name(&self) -> &'static str {
        match self.plan {
            ada_a9_plan_selector::ExecutionPlan::Dense => "dense",
            ada_a9_plan_selector::ExecutionPlan::PagedBranchAndBound => "paged-bnb",
            ada_a9_plan_selector::ExecutionPlan::Hierarchical => "hierarchical",
            ada_a9_plan_selector::ExecutionPlan::ContentAware => "content-aware",
        }
    }
}

trait DispatchOutcomeLike {
    fn plan_name(&self) -> &'static str;
}
