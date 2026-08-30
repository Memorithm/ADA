from pathlib import Path


def replace_once(text: str, old: str, new: str, label: str) -> str:
    if old not in text:
        raise SystemExit(f"expected pattern not found: {label}")
    return text.replace(old, new, 1)


recurrent = Path("crates/ada-advanced-reference/src/recurrent.rs")
text = recurrent.read_text()
text = replace_once(
    text,
    """        for value_index in 0..geometry.value_dimension {
            for key_index in 0..geometry.qk_dimension {
                let index = value_index * geometry.qk_dimension + key_index;
                state[index] = spec.decay * state[index]
                    + spec.learning_rate * value[value_index] * key[key_index];
            }
        }
""",
    """        for (value_index, &value_component) in value.iter().enumerate() {
            for (key_index, &key_component) in key.iter().enumerate() {
                let index = value_index * geometry.qk_dimension + key_index;
                state[index] = spec.decay * state[index]
                    + spec.learning_rate * value_component * key_component;
            }
        }
""",
    "delta update loops",
)
text = replace_once(
    text,
    """        for dimension in 0..geometry.qk_dimension {
            let state_start = dimension * columns;
            for value_index in 0..geometry.value_dimension {
                state[state_start + value_index] += key_features[dimension] * value[value_index];
            }
            state[state_start + geometry.value_dimension] += key_features[dimension];
        }
""",
    """        for (dimension, &key_feature) in key_features.iter().enumerate() {
            let state_start = dimension * columns;
            for (value_index, &value_component) in value.iter().enumerate() {
                state[state_start + value_index] += key_feature * value_component;
            }
            state[state_start + geometry.value_dimension] += key_feature;
        }
""",
    "linear state update loop",
)
recurrent.write_text(text)

ring = Path("crates/ada-advanced-reference/src/ring.rs")
text = ring.read_text()
text = replace_once(
    text,
    """        for dimension in 0..value_dimension {
            numerator[dimension] += weight * shard.values[value_start + dimension];
        }
""",
    """        for (dimension, numerator_value) in numerator.iter_mut().enumerate() {
            *numerator_value += weight * shard.values[value_start + dimension];
        }
""",
    "ring numerator loop",
)
text = replace_once(
    text,
    "        assert_eq!(forward.maximum(), 1000.0);\n",
    "        assert!((forward.maximum() - 1000.0).abs() < 1.0e-12);\n",
    "ring float assertion",
)
ring.write_text(text)

sparse = Path("crates/ada-advanced-reference/src/sparse.rs")
text = sparse.read_text()
text = replace_once(
    text,
    "        assert_eq!(result.output()[0], 20.0);\n",
    "        assert!((result.output()[0] - 20.0).abs() < 1.0e-12);\n",
    "dynamic sparse float assertion",
)
text = replace_once(
    text,
    "        assert_eq!(result.output()[1], 10.0);\n",
    "        assert!((result.output()[1] - 10.0).abs() < 1.0e-12);\n",
    "routed sparse float assertion",
)
sparse.write_text(text)
