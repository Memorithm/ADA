//! Runnable ARE-E0 online-softmax discovery experiment.

use std::path::PathBuf;

use ada_research_engine::online_softmax::build_e0_problem;
use ada_research_engine::{
    CandidateProposer, EngineOptions, EnumerativeConfig, EnumerativeProposer, EvolutionaryConfig,
    EvolutionaryProposer, ExperimentArchive, ExperimentOutcome, compare_archives, run_experiment,
};

#[derive(Debug)]
struct Args {
    seed: u64,
    candidate_budget: u64,
    generated_budget: u64,
    population: usize,
    generations: usize,
    source_revision: String,
    archive: PathBuf,
    replay: Option<PathBuf>,
}

impl Default for Args {
    fn default() -> Self {
        let seed = 20_260_822;
        Self {
            seed,
            candidate_budget: 50_000,
            generated_budget: 100_000,
            population: 256,
            generations: 128,
            source_revision: std::env::var("ADA_SOURCE_REVISION")
                .unwrap_or_else(|_| "uncommitted-or-unspecified".into()),
            archive: PathBuf::from(format!("are_e0_archive_seed_{seed}.json")),
            replay: None,
        }
    }
}

fn parse_args() -> Result<Args, String> {
    let mut result = Args::default();
    let mut arguments = std::env::args().skip(1);
    while let Some(flag) = arguments.next() {
        let value = arguments
            .next()
            .ok_or_else(|| format!("missing value after {flag}"))?;
        match flag.as_str() {
            "--seed" => result.seed = parse(&flag, &value)?,
            "--budget" => result.candidate_budget = parse(&flag, &value)?,
            "--generated-budget" => result.generated_budget = parse(&flag, &value)?,
            "--population" | "--pop" => result.population = parse(&flag, &value)?,
            "--generations" => result.generations = parse(&flag, &value)?,
            "--source-revision" => result.source_revision = value,
            "--archive" => result.archive = PathBuf::from(value),
            "--replay" => result.replay = Some(PathBuf::from(value)),
            _ => return Err(format!("unknown argument {flag}")),
        }
    }
    if result.candidate_budget == 0
        || result.generated_budget == 0
        || result.population == 0
        || result.generations == 0
    {
        return Err("budgets, population, and generations must be non-zero".into());
    }
    Ok(result)
}

fn parse<T: std::str::FromStr>(flag: &str, value: &str) -> Result<T, String> {
    value
        .parse()
        .map_err(|_| format!("invalid value '{value}' for {flag}"))
}

fn fail<T>(message: impl std::fmt::Display) -> T {
    eprintln!("error={message}");
    std::process::exit(2)
}

fn main() {
    let args = parse_args().unwrap_or_else(fail);
    let mut problem = build_e0_problem(args.seed);
    problem.budget.max_candidate_evaluations = args.candidate_budget;
    problem.budget.max_generated_candidates = args.generated_budget;
    problem.budget.max_generations = args.generations;

    let proposers: Vec<Box<dyn CandidateProposer>> = vec![
        Box::new(EnumerativeProposer::new(EnumerativeConfig {
            max_nodes: 5,
            max_emissions: 4_000,
        })),
        Box::new(EvolutionaryProposer::new(EvolutionaryConfig {
            seed: args.seed ^ 0xE0DD_15C0,
            population_size: args.population,
            max_generations: args.generations,
            ..EvolutionaryConfig::default()
        })),
    ];
    let options = EngineOptions {
        proposers,
        source_revision: args.source_revision,
        ..EngineOptions::default()
    };

    let started = std::time::Instant::now();
    let archive = run_experiment(&problem, options).unwrap_or_else(fail);
    archive.verify().unwrap_or_else(fail);
    let elapsed = started.elapsed();

    let json = archive.to_json().unwrap_or_else(fail);
    std::fs::write(&args.archive, format!("{json}\n")).unwrap_or_else(fail);

    println!("experiment_id={}", archive.manifest.experiment_id);
    println!("seed={}", archive.manifest.seed);
    println!("generated={}", archive.stats.generated);
    println!("canonical_unique={}", archive.stats.canonical_unique);
    println!("rejected_static={}", archive.stats.rejected_static);
    println!("falsified={}", archive.stats.falsified);
    println!("survived_oracle={}", archive.stats.survived_oracle);
    println!(
        "survived_adversarial={}",
        archive.stats.survived_adversarial
    );
    println!("pareto_count={}", archive.pareto_front.len());
    println!(
        "best_candidate_digest={}",
        archive
            .best
            .as_ref()
            .map_or("none", |candidate| candidate.candidate_id.as_str())
    );
    println!("status={}", archive.outcome);
    println!("termination={:?}", archive.termination);
    println!("archive_digest={}", archive.archive_digest);

    if matches!(
        archive.outcome,
        ExperimentOutcome::SurvivedDeclaredGatesWithinTolerance
            | ExperimentOutcome::SurvivedDeclaredGatesExactly
    ) {
        if let Some(best) = &archive.best {
            println!("best_canonical_candidate={}", best.canonical_candidate);
        }
    }

    if let Some(replay_path) = &args.replay {
        let expected_json = std::fs::read_to_string(replay_path).unwrap_or_else(fail);
        let expected = ExperimentArchive::from_json(&expected_json).unwrap_or_else(fail);
        let report = compare_archives(&expected, &archive);
        println!("replay_identical={}", report.identical);
        if !report.identical {
            eprintln!("replay_mismatches={}", report.mismatches.join(","));
            std::process::exit(3);
        }
    }

    eprintln!("archive_written={}", args.archive.display());
    eprintln!("elapsed_seconds={:.3}", elapsed.as_secs_f64());
}
