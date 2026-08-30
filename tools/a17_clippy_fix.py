from pathlib import Path


def replace_once(path: str, old: str, new: str, label: str) -> None:
    file_path = Path(path)
    text = file_path.read_text()
    if old not in text:
        raise SystemExit(f"expected pattern not found: {label}")
    file_path.write_text(text.replace(old, new, 1))


replace_once(
    "crates/ada-advanced-reference/src/recurrent.rs",
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

replace_once(
    "crates/ada-advanced-reference/src/recurrent.rs",
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

replace_once(
    "crates/ada-advanced-reference/src/ring.rs",
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

replace_once(
    "crates/ada-advanced-reference/src/ring.rs",
    "        assert_eq!(forward.maximum(), 1000.0);\n",
    "        assert!((forward.maximum() - 1000.0).abs() < 1.0e-12);\n",
    "ring float assertion",
)

replace_once(
    "crates/ada-advanced-reference/src/sparse.rs",
    "        assert_eq!(result.output()[0], 20.0);\n",
    "        assert!((result.output()[0] - 20.0).abs() < 1.0e-12);\n",
    "dynamic sparse float assertion",
)

replace_once(
    "crates/ada-advanced-reference/src/sparse.rs",
    "        assert_eq!(result.output()[1], 10.0);\n",
    "        assert!((result.output()[1] - 10.0).abs() < 1.0e-12);\n",
    "routed sparse float assertion",
)
