//! Exact-bit JSON transport helpers for archive-relevant floating-point data.
//!
//! Decimal JSON parsers are not universally correctly rounded. Schema v2
//! therefore persists each `f64` as a fixed-width hexadecimal bit string while
//! keeping ordinary `f64` values in the public in-memory API.

fn encode(value: f64) -> String {
    format!("{:016x}", value.to_bits())
}

fn decode<E: serde::de::Error>(encoded: &str) -> Result<f64, E> {
    if encoded.len() != 16 {
        return Err(E::custom(
            "f64 bit string must contain exactly 16 hex digits",
        ));
    }
    u64::from_str_radix(encoded, 16)
        .map(f64::from_bits)
        .map_err(|_| E::custom("invalid f64 bit string"))
}

pub(crate) mod scalar {
    use super::{decode, encode};
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    // Serde's `with` module contract passes fields by reference.
    #[allow(clippy::trivially_copy_pass_by_ref)]
    pub fn serialize<S: Serializer>(value: &f64, serializer: S) -> Result<S::Ok, S::Error> {
        encode(*value).serialize(serializer)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(deserializer: D) -> Result<f64, D::Error> {
        decode(&String::deserialize(deserializer)?)
    }
}

pub(crate) mod optional {
    use super::{decode, encode};
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    // Serde's `with` module contract passes `Option` fields by reference.
    #[allow(clippy::ref_option)]
    pub fn serialize<S: Serializer>(value: &Option<f64>, serializer: S) -> Result<S::Ok, S::Error> {
        value.map(encode).serialize(serializer)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(
        deserializer: D,
    ) -> Result<Option<f64>, D::Error> {
        Option::<String>::deserialize(deserializer)?
            .map(|encoded| decode(&encoded))
            .transpose()
    }
}

pub(crate) mod vector {
    use super::{decode, encode};
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    pub fn serialize<S: Serializer>(value: &[f64], serializer: S) -> Result<S::Ok, S::Error> {
        value
            .iter()
            .copied()
            .map(encode)
            .collect::<Vec<_>>()
            .serialize(serializer)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(deserializer: D) -> Result<Vec<f64>, D::Error> {
        Vec::<String>::deserialize(deserializer)?
            .iter()
            .map(|encoded| decode(encoded))
            .collect()
    }
}

pub(crate) mod optional_vector {
    use super::{decode, encode};
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    // Serde's `with` module contract passes `Option` fields by reference.
    #[allow(clippy::ref_option)]
    pub fn serialize<S: Serializer>(
        value: &Option<Vec<f64>>,
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        value
            .as_ref()
            .map(|values| values.iter().copied().map(encode).collect::<Vec<_>>())
            .serialize(serializer)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(
        deserializer: D,
    ) -> Result<Option<Vec<f64>>, D::Error> {
        Option::<Vec<String>>::deserialize(deserializer)?
            .map(|values| values.iter().map(|encoded| decode(encoded)).collect())
            .transpose()
    }
}

pub(crate) mod named_vector {
    use super::{decode, encode};
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    pub fn serialize<S: Serializer>(
        value: &[(String, f64)],
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        value
            .iter()
            .map(|(name, number)| (name, encode(*number)))
            .collect::<Vec<_>>()
            .serialize(serializer)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(
        deserializer: D,
    ) -> Result<Vec<(String, f64)>, D::Error> {
        Vec::<(String, String)>::deserialize(deserializer)?
            .into_iter()
            .map(|(name, encoded)| decode(&encoded).map(|number| (name, number)))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use serde::{Deserialize, Serialize};

    #[derive(Debug, PartialEq, Serialize, Deserialize)]
    struct Fixture {
        #[serde(with = "super::scalar")]
        value: f64,
    }

    #[test]
    fn troublesome_decimal_round_trips_by_bits() {
        let fixture = Fixture {
            value: 0.990_049_833_749_168_3,
        };
        let json = serde_json::to_string(&fixture).unwrap();
        assert!(json.contains(&format!("{:016x}", fixture.value.to_bits())));
        let parsed: Fixture = serde_json::from_str(&json).unwrap();
        assert_eq!(fixture.value.to_bits(), parsed.value.to_bits());
    }
}
