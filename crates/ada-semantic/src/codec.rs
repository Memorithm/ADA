use super::{
    AffinityRule, InputTransform, MAX_CANONICAL_TEXT_BYTES, MaskContract, MaskRule, OutputRule,
    SEMANTIC_IR_VERSION, SelectionRule, SemanticDescriptor, SemanticFamily, SemanticId,
    SemanticIrError, SemanticProgram, SemanticProgramSpec, StateContract, ValueMixRule,
    WeightContract, WeightRule,
};
use std::collections::BTreeMap;
use std::fmt::Display;

fn append_field(text: &mut String, key: &str, value: impl Display) {
    text.push_str(key);
    text.push('=');
    text.push_str(&value.to_string());
    text.push('\n');
}

pub(super) fn encode(program: &SemanticProgram) -> String {
    let mut text = format!("ADA-SEMANTIC-V{SEMANTIC_IR_VERSION}\n");
    append_field(
        &mut text,
        "semantic_family",
        family_text(program.descriptor.id().family()),
    );
    append_field(
        &mut text,
        "semantic_name",
        hex_encode(program.descriptor.id().name()),
    );
    append_field(
        &mut text,
        "semantic_revision",
        program.descriptor.id().revision(),
    );
    append_field(
        &mut text,
        "descriptor_mask",
        descriptor_mask_text(program.descriptor.mask()),
    );
    append_field(&mut text, "descriptor_state", "stateless");
    append_field(
        &mut text,
        "descriptor_weights",
        descriptor_weight_text(program.descriptor.weights()),
    );
    append_field(
        &mut text,
        "input_transform",
        program.input_transform.as_text(),
    );
    match program.affinity {
        AffinityRule::ScaledDotProduct { scale } => {
            append_field(&mut text, "affinity_kind", "scaled-dot-product");
            append_field(&mut text, "affinity_value", float_text(scale));
        }
    }
    match &program.mask {
        MaskRule::Unmasked => {
            append_field(&mut text, "mask_kind", "unmasked");
            append_field(&mut text, "mask_value", "-");
        }
        MaskRule::Causal => {
            append_field(&mut text, "mask_kind", "causal");
            append_field(&mut text, "mask_value", "-");
        }
        MaskRule::External { identity } => {
            append_field(&mut text, "mask_kind", "external");
            append_field(&mut text, "mask_value", hex_encode(identity));
        }
    }
    match program.selection {
        SelectionRule::All => {
            append_field(&mut text, "selection_kind", "all");
            append_field(&mut text, "selection_value", "-");
        }
        SelectionRule::Window { radius } => {
            append_field(&mut text, "selection_kind", "window");
            append_field(&mut text, "selection_value", radius);
        }
        SelectionRule::TopK { k } => {
            append_field(&mut text, "selection_kind", "top-k");
            append_field(&mut text, "selection_value", k);
        }
    }
    match program.weight {
        WeightRule::Softmax => {
            append_field(&mut text, "weight_kind", "softmax");
            append_field(&mut text, "weight_positive_scale", "-");
            append_field(&mut text, "weight_negative_scale", "-");
        }
        WeightRule::SignedDifference {
            positive_scale,
            negative_scale,
        } => {
            append_field(&mut text, "weight_kind", "signed-difference");
            append_field(
                &mut text,
                "weight_positive_scale",
                float_text(positive_scale),
            );
            append_field(
                &mut text,
                "weight_negative_scale",
                float_text(negative_scale),
            );
        }
    }
    append_field(&mut text, "value_mix", "weighted-sum");
    append_field(&mut text, "output", "identity");
    text
}

pub(super) fn decode(text: &str) -> Result<SemanticProgram, SemanticIrError> {
    let fields = parse_fields(text)?;
    let field = |key: &str| {
        fields
            .get(key)
            .copied()
            .ok_or_else(|| SemanticIrError::MalformedCanonicalText(format!("missing field {key}")))
    };
    let family = family_from_text(field("semantic_family")?)?;
    let name = hex_decode("semantic_name", field("semantic_name")?)?;
    let revision = parse_u32("semantic_revision", field("semantic_revision")?)?;
    let id = SemanticId::new(family, name, revision)
        .map_err(|_| SemanticIrError::InvalidField("semantic identity"))?;
    let descriptor_mask = descriptor_mask_from_text(field("descriptor_mask")?)?;
    if field("descriptor_state")? != "stateless" {
        return Err(SemanticIrError::UnsupportedWorkload(
            "semantic IR v1 only supports stateless programs",
        ));
    }
    let descriptor_weights = descriptor_weight_from_text(field("descriptor_weights")?)?;
    let input_transform = InputTransform::from_text(field("input_transform")?)?;
    let affinity = match field("affinity_kind")? {
        "scaled-dot-product" => AffinityRule::ScaledDotProduct {
            scale: parse_float("affinity_value", field("affinity_value")?)?,
        },
        _ => {
            return Err(SemanticIrError::MalformedCanonicalText(
                "unknown affinity kind".into(),
            ));
        }
    };
    let mask = parse_mask(field("mask_kind")?, field("mask_value")?)?;
    let selection = parse_selection(field("selection_kind")?, field("selection_value")?)?;
    let weight = parse_weight(
        field("weight_kind")?,
        field("weight_positive_scale")?,
        field("weight_negative_scale")?,
    )?;
    if field("value_mix")? != "weighted-sum" || field("output")? != "identity" {
        return Err(SemanticIrError::MalformedCanonicalText(
            "unknown value mix or output rule".into(),
        ));
    }
    let descriptor = SemanticDescriptor::new(
        id,
        descriptor_mask,
        StateContract::Stateless,
        descriptor_weights,
    );
    SemanticProgram::new(SemanticProgramSpec {
        descriptor,
        input_transform,
        affinity,
        mask,
        selection,
        weight,
        value_mix: ValueMixRule::WeightedSum,
        output: OutputRule::Identity,
    })
}

fn parse_fields(text: &str) -> Result<BTreeMap<&str, &str>, SemanticIrError> {
    if text.len() > MAX_CANONICAL_TEXT_BYTES || text.contains('\r') {
        return Err(SemanticIrError::MalformedCanonicalText(
            "canonical text exceeds its limit or contains CR".into(),
        ));
    }
    if !text.ends_with('\n') {
        return Err(SemanticIrError::MalformedCanonicalText(
            "canonical text must end with a newline".into(),
        ));
    }
    let mut lines = text.lines();
    let Some(header) = lines.next() else {
        return Err(SemanticIrError::MalformedCanonicalText(
            "missing ADA-SEMANTIC header".into(),
        ));
    };
    let Some(version_text) = header.strip_prefix("ADA-SEMANTIC-V") else {
        return Err(SemanticIrError::MalformedCanonicalText(
            "missing ADA-SEMANTIC version header".into(),
        ));
    };
    let version = version_text.parse::<u16>().map_err(|_| {
        SemanticIrError::MalformedCanonicalText("invalid ADA-SEMANTIC version".into())
    })?;
    if version != SEMANTIC_IR_VERSION {
        return Err(SemanticIrError::UnsupportedVersion(version));
    }
    let mut fields = BTreeMap::new();
    for line in lines {
        let Some((key, value)) = line.split_once('=') else {
            return Err(SemanticIrError::MalformedCanonicalText(
                "field is missing '='".into(),
            ));
        };
        if key.is_empty() || value.contains('=') || fields.insert(key, value).is_some() {
            return Err(SemanticIrError::MalformedCanonicalText(
                "empty, duplicate, or ambiguous field".into(),
            ));
        }
    }
    let required = [
        "semantic_family",
        "semantic_name",
        "semantic_revision",
        "descriptor_mask",
        "descriptor_state",
        "descriptor_weights",
        "input_transform",
        "affinity_kind",
        "affinity_value",
        "mask_kind",
        "mask_value",
        "selection_kind",
        "selection_value",
        "weight_kind",
        "weight_positive_scale",
        "weight_negative_scale",
        "value_mix",
        "output",
    ];
    if fields.len() != required.len() || required.iter().any(|key| !fields.contains_key(key)) {
        return Err(SemanticIrError::MalformedCanonicalText(
            "canonical field set is incomplete or has unknown keys".into(),
        ));
    }
    Ok(fields)
}

fn family_text(family: SemanticFamily) -> &'static str {
    match family {
        SemanticFamily::StandardSoftmax => "standard-softmax",
        SemanticFamily::DifferentialSigned => "differential-signed",
        SemanticFamily::ToeplitzStructured => "toeplitz-structured",
        SemanticFamily::ProlateConcentration => "prolate-concentration",
        SemanticFamily::GroundStateGreen => "ground-state-green",
        SemanticFamily::SpectralFlow => "spectral-flow",
        SemanticFamily::RecurrentMemory => "recurrent-memory",
        SemanticFamily::Hybrid => "hybrid",
        SemanticFamily::Experimental => "experimental",
    }
}

fn family_from_text(value: &str) -> Result<SemanticFamily, SemanticIrError> {
    match value {
        "standard-softmax" => Ok(SemanticFamily::StandardSoftmax),
        "differential-signed" => Ok(SemanticFamily::DifferentialSigned),
        "toeplitz-structured" => Ok(SemanticFamily::ToeplitzStructured),
        "prolate-concentration" => Ok(SemanticFamily::ProlateConcentration),
        "ground-state-green" => Ok(SemanticFamily::GroundStateGreen),
        "spectral-flow" => Ok(SemanticFamily::SpectralFlow),
        "recurrent-memory" => Ok(SemanticFamily::RecurrentMemory),
        "hybrid" => Ok(SemanticFamily::Hybrid),
        "experimental" => Ok(SemanticFamily::Experimental),
        _ => Err(SemanticIrError::MalformedCanonicalText(
            "unknown semantic family".into(),
        )),
    }
}

fn descriptor_mask_text(mask: MaskContract) -> &'static str {
    match mask {
        MaskContract::Bidirectional => "bidirectional",
        MaskContract::Causal => "causal",
        MaskContract::ExternalMask => "external",
    }
}

fn descriptor_mask_from_text(value: &str) -> Result<MaskContract, SemanticIrError> {
    match value {
        "bidirectional" => Ok(MaskContract::Bidirectional),
        "causal" => Ok(MaskContract::Causal),
        "external" => Ok(MaskContract::ExternalMask),
        _ => Err(SemanticIrError::MalformedCanonicalText(
            "unknown descriptor mask".into(),
        )),
    }
}

fn descriptor_weight_text(weight: WeightContract) -> &'static str {
    match weight {
        WeightContract::ProbabilitySimplex => "probability-simplex",
        WeightContract::Signed => "signed",
        WeightContract::StructuredLinear => "structured-linear",
        WeightContract::StateDependent => "state-dependent",
    }
}

fn descriptor_weight_from_text(value: &str) -> Result<WeightContract, SemanticIrError> {
    match value {
        "probability-simplex" => Ok(WeightContract::ProbabilitySimplex),
        "signed" => Ok(WeightContract::Signed),
        "structured-linear" => Ok(WeightContract::StructuredLinear),
        "state-dependent" => Ok(WeightContract::StateDependent),
        _ => Err(SemanticIrError::MalformedCanonicalText(
            "unknown descriptor weight".into(),
        )),
    }
}

fn parse_mask(kind: &str, value: &str) -> Result<MaskRule, SemanticIrError> {
    let mask = match kind {
        "unmasked" => MaskRule::Unmasked,
        "causal" => MaskRule::Causal,
        "external" => MaskRule::External {
            identity: hex_decode("mask_value", value)?,
        },
        _ => {
            return Err(SemanticIrError::MalformedCanonicalText(
                "unknown mask kind".into(),
            ));
        }
    };
    if !matches!(&mask, MaskRule::External { .. }) && value != "-" {
        return Err(SemanticIrError::MalformedCanonicalText(
            "non-external mask must use '-' value".into(),
        ));
    }
    mask.validate()?;
    Ok(mask)
}

fn parse_selection(kind: &str, value: &str) -> Result<SelectionRule, SemanticIrError> {
    let selection = match kind {
        "all" => SelectionRule::All,
        "window" => SelectionRule::Window {
            radius: parse_usize("selection_value", value)?,
        },
        "top-k" => SelectionRule::TopK {
            k: parse_usize("selection_value", value)?,
        },
        _ => {
            return Err(SemanticIrError::MalformedCanonicalText(
                "unknown selection kind".into(),
            ));
        }
    };
    if matches!(&selection, SelectionRule::All) && value != "-" {
        return Err(SemanticIrError::MalformedCanonicalText(
            "all selection must use '-' value".into(),
        ));
    }
    selection.validate()?;
    Ok(selection)
}

fn parse_weight(kind: &str, positive: &str, negative: &str) -> Result<WeightRule, SemanticIrError> {
    let weight = match kind {
        "softmax" => WeightRule::Softmax,
        "signed-difference" => WeightRule::SignedDifference {
            positive_scale: parse_float("weight_positive_scale", positive)?,
            negative_scale: parse_float("weight_negative_scale", negative)?,
        },
        _ => {
            return Err(SemanticIrError::MalformedCanonicalText(
                "unknown weight kind".into(),
            ));
        }
    };
    if matches!(&weight, WeightRule::Softmax) && (positive != "-" || negative != "-") {
        return Err(SemanticIrError::MalformedCanonicalText(
            "softmax must use '-' branch scales".into(),
        ));
    }
    weight.validate()?;
    Ok(weight)
}

fn parse_usize(field: &str, value: &str) -> Result<usize, SemanticIrError> {
    value.parse::<usize>().map_err(|_| {
        SemanticIrError::MalformedCanonicalText(format!(
            "{field} is not an unsigned decimal integer"
        ))
    })
}

fn parse_u32(field: &str, value: &str) -> Result<u32, SemanticIrError> {
    value.parse::<u32>().map_err(|_| {
        SemanticIrError::MalformedCanonicalText(format!(
            "{field} is not an unsigned 32-bit integer"
        ))
    })
}

fn float_text(value: f64) -> String {
    format!("0x{:016x}", value.to_bits())
}

fn parse_float(field: &str, value: &str) -> Result<f64, SemanticIrError> {
    if value.len() != 18
        || !value.starts_with("0x")
        || !value[2..]
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(SemanticIrError::MalformedCanonicalText(format!(
            "{field} is not a canonical 64-bit float"
        )));
    }
    let bits = u64::from_str_radix(&value[2..], 16).map_err(|_| {
        SemanticIrError::MalformedCanonicalText(format!("{field} has invalid float bits"))
    })?;
    Ok(f64::from_bits(bits))
}

fn hex_encode(value: &str) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut encoded = String::with_capacity(value.len() * 2);
    for byte in value.as_bytes() {
        encoded.push(char::from(HEX[usize::from(byte >> 4)]));
        encoded.push(char::from(HEX[usize::from(byte & 0x0f)]));
    }
    encoded
}

fn hex_decode(field: &'static str, value: &str) -> Result<String, SemanticIrError> {
    if value.len() % 2 != 0 {
        return Err(SemanticIrError::MalformedCanonicalText(format!(
            "{field} has an odd-length hex value"
        )));
    }
    let mut bytes = Vec::with_capacity(value.len() / 2);
    let mut chars = value.bytes();
    while let (Some(high), Some(low)) = (chars.next(), chars.next()) {
        let high = hex_digit(high).ok_or_else(|| {
            SemanticIrError::MalformedCanonicalText(format!("{field} contains a non-hex digit"))
        })?;
        let low = hex_digit(low).ok_or_else(|| {
            SemanticIrError::MalformedCanonicalText(format!("{field} contains a non-hex digit"))
        })?;
        bytes.push((high << 4) | low);
    }
    String::from_utf8(bytes).map_err(|_| {
        SemanticIrError::MalformedCanonicalText(format!("{field} is not UTF-8 after hex decoding"))
    })
}

fn hex_digit(value: u8) -> Option<u8> {
    match value {
        b'0'..=b'9' => Some(value - b'0'),
        b'a'..=b'f' => Some(value - b'a' + 10),
        _ => None,
    }
}
