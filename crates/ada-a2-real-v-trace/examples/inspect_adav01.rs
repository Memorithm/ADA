use std::collections::BTreeSet;
use std::env;
use std::io;

use ada_a2_real_v_trace::{ATTENTION_VALUE_INPUT_PRE_REPEAT_KV_STAGE, read_value_trace_file};

const EXPECTED_MODEL: &str = "Qwen/Qwen3-0.6B";

const EXPECTED_REVISION: &str = "c1899de289a04d12100db370d81485cdf75e47ca";

const EXPECTED_CAPTURE: &str = "qwen3-0.6b-a2-e3b-v-wikitext2raw-val16";

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = env::args_os().nth(1).ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            "usage: inspect_adav01 <trace.adav>",
        )
    })?;

    let corpus = read_value_trace_file(path)?;

    let metadata = corpus.metadata();

    let mut samples = BTreeSet::new();

    let mut identities = BTreeSet::new();

    let mut layers = BTreeSet::new();

    let mut kv_heads = BTreeSet::new();

    let mut counts = BTreeSet::new();

    let mut dims = BTreeSet::new();

    let mut starts = BTreeSet::new();

    for record in corpus.records() {
        samples.insert(record.sample_id.clone());

        let inserted = identities.insert((
            record.sample_id.clone(),
            record.layer_index,
            record.kv_head_index,
        ));

        if !inserted {
            return Err(io::Error::other("duplicate natural V identity").into());
        }

        layers.insert(record.layer_index);

        kv_heads.insert(record.kv_head_index);

        counts.insert(record.value_count);

        dims.insert(record.head_dim);

        starts.insert(record.value_start_position);

        if record.value_end_position()? != 512 {
            return Err(io::Error::other("unexpected V interval end").into());
        }
    }

    println!("format=ADAV01");

    println!("model_id={}", metadata.model_id);

    println!("model_revision={}", metadata.model_revision);

    println!("capture_id={}", metadata.capture_id);

    println!("source_dtype={}", metadata.source_dtype);

    println!("tensor_stage={}", metadata.tensor_stage);

    println!("record_count={}", corpus.len());

    println!("unique_identity_count={}", identities.len());

    println!("sample_count={}", samples.len());

    println!("layers={layers:?}");

    println!("kv_heads={kv_heads:?}");

    println!("value_start_positions={starts:?}");

    println!("value_counts={counts:?}");

    println!("head_dims={dims:?}");

    let first = corpus
        .records()
        .first()
        .ok_or_else(|| io::Error::other("ADAV01 corpus is empty"))?;

    println!(
        "first_sample_fingerprint={:016x}",
        first.sample_fingerprint()
    );

    println!("first_value_end_position={}", first.value_end_position()?);

    println!("first_row_scalars={}", first.row(0)?.len());

    println!("first_prefix64_scalars={}", first.prefix_values(64)?.len());

    let contract_ok = metadata.model_id == EXPECTED_MODEL
        && metadata.model_revision == EXPECTED_REVISION
        && metadata.capture_id == EXPECTED_CAPTURE
        && metadata.source_dtype == "bfloat16"
        && metadata.tensor_stage == ATTENTION_VALUE_INPUT_PRE_REPEAT_KV_STAGE
        && corpus.len() == 384
        && identities.len() == 384
        && samples.len() == 16
        && layers == BTreeSet::from([0_u32, 13_u32, 27_u32])
        && kv_heads == (0_u32..8_u32).collect::<BTreeSet<_>>()
        && starts == BTreeSet::from([0_u64])
        && counts == BTreeSet::from([512_usize])
        && dims == BTreeSet::from([128_usize]);

    println!("real_trace_contract_ok={contract_ok}");

    if !contract_ok {
        return Err(io::Error::other("real ADAV01 trace contract failed").into());
    }

    Ok(())
}
