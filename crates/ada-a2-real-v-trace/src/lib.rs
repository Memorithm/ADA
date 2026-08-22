#![forbid(unsafe_code)]

use std::fmt;
use std::fs;
use std::io;
use std::path::Path;

pub const VALUE_TRACE_MAGIC: [u8; 7] = *b"ADAV01\0";
pub const VALUE_TRACE_VERSION: u32 = 1;

pub const ATTENTION_VALUE_INPUT_PRE_REPEAT_KV_STAGE: &str = "attention_value_input_pre_repeat_kv";

const MAX_STRING_BYTES: usize = 1_048_576;
const MAX_RECORDS: usize = 1_000_000;
const MAX_HEAD_DIM: usize = 65_536;
const MAX_VALUE_ROWS: usize = 1_048_576;
const MAX_VALUES_PER_RECORD: usize = 67_108_864;

#[derive(Debug)]
pub enum ValueTraceError {
    Io(io::Error),
    Invalid(&'static str),
}

impl fmt::Display for ValueTraceError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(error) => {
                write!(formatter, "ADAV01 trace I/O error: {error}")
            }
            Self::Invalid(message) => formatter.write_str(message),
        }
    }
}

impl std::error::Error for ValueTraceError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(error) => Some(error),
            Self::Invalid(_) => None,
        }
    }
}

impl From<io::Error> for ValueTraceError {
    fn from(error: io::Error) -> Self {
        Self::Io(error)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValueTraceMetadata {
    pub model_id: String,
    pub model_revision: String,
    pub tokenizer_id: String,
    pub tokenizer_revision: String,
    pub capture_id: String,
    pub source_dtype: String,
    pub tensor_stage: String,
    pub record_count: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ValueTraceRecord {
    pub sample_id: String,
    pub layer_index: u32,
    pub kv_head_index: u32,
    pub value_start_position: u64,
    pub value_count: usize,
    pub head_dim: usize,
    pub values: Vec<f64>,
}

impl ValueTraceRecord {
    #[must_use]
    pub fn sample_fingerprint(&self) -> u64 {
        fingerprint_bytes(self.sample_id.as_bytes())
    }

    /// Return the exclusive end position of the captured V interval.
    ///
    /// # Errors
    ///
    /// Returns an error if adding the captured row count to the start
    /// position would overflow `u64`.
    pub fn value_end_position(&self) -> Result<u64, ValueTraceError> {
        let value_count = u64::try_from(self.value_count)
            .map_err(|_| ValueTraceError::Invalid("ADAV01 value_count does not fit u64"))?;

        self.value_start_position
            .checked_add(value_count)
            .ok_or(ValueTraceError::Invalid("ADAV01 V interval overflow"))
    }

    /// Return one captured V row by absolute sequence position.
    ///
    /// # Errors
    ///
    /// Returns an error if `position` is outside this record's
    /// captured interval.
    pub fn row(&self, position: u64) -> Result<&[f64], ValueTraceError> {
        if position < self.value_start_position || position >= self.value_end_position()? {
            return Err(ValueTraceError::Invalid(
                "ADAV01 requested V position lies outside captured interval",
            ));
        }

        let local = usize::try_from(position - self.value_start_position)
            .map_err(|_| ValueTraceError::Invalid("ADAV01 V row index conversion failed"))?;

        let start = local
            .checked_mul(self.head_dim)
            .ok_or(ValueTraceError::Invalid("ADAV01 V row offset overflow"))?;

        let end = start
            .checked_add(self.head_dim)
            .ok_or(ValueTraceError::Invalid("ADAV01 V row end overflow"))?;

        self.values
            .get(start..end)
            .ok_or(ValueTraceError::Invalid("ADAV01 V row slice is invalid"))
    }

    /// Return a prefix of V rows, flattened row-major.
    ///
    /// The prefix is relative to this record's captured start
    /// position. No tensor data are copied.
    ///
    /// # Errors
    ///
    /// Returns an error if `row_count` exceeds the captured rows.
    pub fn prefix_values(&self, row_count: usize) -> Result<&[f64], ValueTraceError> {
        if row_count > self.value_count {
            return Err(ValueTraceError::Invalid(
                "ADAV01 requested V prefix exceeds captured rows",
            ));
        }

        let scalar_count = row_count
            .checked_mul(self.head_dim)
            .ok_or(ValueTraceError::Invalid(
                "ADAV01 V prefix scalar count overflow",
            ))?;

        self.values
            .get(..scalar_count)
            .ok_or(ValueTraceError::Invalid("ADAV01 V prefix slice is invalid"))
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ValueTraceCorpus {
    metadata: ValueTraceMetadata,
    records: Vec<ValueTraceRecord>,
}

impl ValueTraceCorpus {
    #[must_use]
    pub const fn metadata(&self) -> &ValueTraceMetadata {
        &self.metadata
    }

    #[must_use]
    pub fn records(&self) -> &[ValueTraceRecord] {
        &self.records
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.records.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    #[must_use]
    pub fn find_record(
        &self,
        sample_id: &str,
        layer_index: u32,
        kv_head_index: u32,
    ) -> Option<&ValueTraceRecord> {
        self.records.iter().find(|record| {
            record.sample_id == sample_id
                && record.layer_index == layer_index
                && record.kv_head_index == kv_head_index
        })
    }
}

#[derive(Debug)]
struct ByteReader<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> ByteReader<'a> {
    const fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, offset: 0 }
    }

    fn take(&mut self, length: usize) -> Result<&'a [u8], ValueTraceError> {
        let end = self
            .offset
            .checked_add(length)
            .ok_or(ValueTraceError::Invalid("ADAV01 trace offset overflow"))?;

        let slice = self
            .bytes
            .get(self.offset..end)
            .ok_or(ValueTraceError::Invalid("ADAV01 trace is truncated"))?;

        self.offset = end;

        Ok(slice)
    }

    fn read_array<const N: usize>(&mut self) -> Result<[u8; N], ValueTraceError> {
        let bytes = self.take(N)?;

        <[u8; N]>::try_from(bytes)
            .map_err(|_| ValueTraceError::Invalid("ADAV01 fixed-width decode failed"))
    }

    fn read_u32(&mut self) -> Result<u32, ValueTraceError> {
        Ok(u32::from_le_bytes(self.read_array()?))
    }

    fn read_u64(&mut self) -> Result<u64, ValueTraceError> {
        Ok(u64::from_le_bytes(self.read_array()?))
    }

    fn read_string(&mut self) -> Result<String, ValueTraceError> {
        let length = usize::try_from(self.read_u32()?)
            .map_err(|_| ValueTraceError::Invalid("ADAV01 string length conversion failed"))?;

        if length > MAX_STRING_BYTES {
            return Err(ValueTraceError::Invalid(
                "ADAV01 metadata string exceeds size limit",
            ));
        }

        let bytes = self.take(length)?;

        let text = std::str::from_utf8(bytes)
            .map_err(|_| ValueTraceError::Invalid("ADAV01 metadata is not valid UTF-8"))?;

        Ok(text.to_owned())
    }

    fn read_f32_values(&mut self, count: usize) -> Result<Vec<f64>, ValueTraceError> {
        if count > MAX_VALUES_PER_RECORD {
            return Err(ValueTraceError::Invalid(
                "ADAV01 numeric tensor exceeds per-record value limit",
            ));
        }

        let byte_count =
            count
                .checked_mul(std::mem::size_of::<f32>())
                .ok_or(ValueTraceError::Invalid(
                    "ADAV01 tensor byte length overflow",
                ))?;

        let bytes = self.take(byte_count)?;

        let mut values = Vec::with_capacity(count);

        for chunk in bytes.chunks_exact(std::mem::size_of::<f32>()) {
            let encoded = <[u8; 4]>::try_from(chunk)
                .map_err(|_| ValueTraceError::Invalid("ADAV01 f32 decode failed"))?;

            let value = f32::from_le_bytes(encoded);

            if !value.is_finite() {
                return Err(ValueTraceError::Invalid(
                    "ADAV01 trace contains non-finite V value",
                ));
            }

            values.push(f64::from(value));
        }

        Ok(values)
    }

    const fn remaining(&self) -> usize {
        self.bytes.len() - self.offset
    }
}

fn fingerprint_bytes(bytes: &[u8]) -> u64 {
    let mut fingerprint = 0xcbf2_9ce4_8422_2325_u64;

    for &byte in bytes {
        fingerprint ^= u64::from(byte);

        fingerprint = fingerprint.wrapping_mul(0x0000_0100_0000_01b3);
    }

    fingerprint
}

fn validate_non_empty(value: &str, message: &'static str) -> Result<(), ValueTraceError> {
    if value.is_empty() {
        Err(ValueTraceError::Invalid(message))
    } else {
        Ok(())
    }
}

fn read_metadata(reader: &mut ByteReader<'_>) -> Result<ValueTraceMetadata, ValueTraceError> {
    let model_id = reader.read_string()?;
    let model_revision = reader.read_string()?;
    let tokenizer_id = reader.read_string()?;
    let tokenizer_revision = reader.read_string()?;
    let capture_id = reader.read_string()?;
    let source_dtype = reader.read_string()?;
    let tensor_stage = reader.read_string()?;

    let record_count = usize::try_from(reader.read_u32()?)
        .map_err(|_| ValueTraceError::Invalid("ADAV01 record count conversion failed"))?;

    validate_non_empty(&model_id, "ADAV01 model_id must be non-empty")?;

    validate_non_empty(&model_revision, "ADAV01 model_revision must be non-empty")?;

    validate_non_empty(&tokenizer_id, "ADAV01 tokenizer_id must be non-empty")?;

    validate_non_empty(
        &tokenizer_revision,
        "ADAV01 tokenizer_revision must be non-empty",
    )?;

    validate_non_empty(&capture_id, "ADAV01 capture_id must be non-empty")?;

    validate_non_empty(&source_dtype, "ADAV01 source_dtype must be non-empty")?;

    if tensor_stage != ATTENTION_VALUE_INPUT_PRE_REPEAT_KV_STAGE {
        return Err(ValueTraceError::Invalid(
            "ADAV01 tensor_stage must be attention_value_input_pre_repeat_kv",
        ));
    }

    if record_count == 0 || record_count > MAX_RECORDS {
        return Err(ValueTraceError::Invalid(
            "ADAV01 record_count is outside the supported range",
        ));
    }

    Ok(ValueTraceMetadata {
        model_id,
        model_revision,
        tokenizer_id,
        tokenizer_revision,
        capture_id,
        source_dtype,
        tensor_stage,
        record_count,
    })
}

fn read_record(reader: &mut ByteReader<'_>) -> Result<ValueTraceRecord, ValueTraceError> {
    let sample_id = reader.read_string()?;

    validate_non_empty(&sample_id, "ADAV01 sample_id must be non-empty")?;

    let layer_index = reader.read_u32()?;
    let kv_head_index = reader.read_u32()?;
    let value_start_position = reader.read_u64()?;

    let value_count = usize::try_from(reader.read_u32()?)
        .map_err(|_| ValueTraceError::Invalid("ADAV01 value_count conversion failed"))?;

    let head_dim = usize::try_from(reader.read_u32()?)
        .map_err(|_| ValueTraceError::Invalid("ADAV01 head_dim conversion failed"))?;

    if value_count == 0 || value_count > MAX_VALUE_ROWS {
        return Err(ValueTraceError::Invalid(
            "ADAV01 value_count is outside the supported range",
        ));
    }

    if head_dim == 0 || head_dim > MAX_HEAD_DIM {
        return Err(ValueTraceError::Invalid(
            "ADAV01 head_dim is outside the supported range",
        ));
    }

    let value_count_u64 = u64::try_from(value_count)
        .map_err(|_| ValueTraceError::Invalid("ADAV01 value_count does not fit u64"))?;

    value_start_position
        .checked_add(value_count_u64)
        .ok_or(ValueTraceError::Invalid("ADAV01 V interval overflow"))?;

    let scalar_count = value_count
        .checked_mul(head_dim)
        .ok_or(ValueTraceError::Invalid("ADAV01 V tensor length overflow"))?;

    if scalar_count > MAX_VALUES_PER_RECORD {
        return Err(ValueTraceError::Invalid(
            "ADAV01 V tensor exceeds per-record value limit",
        ));
    }

    let values = reader.read_f32_values(scalar_count)?;

    Ok(ValueTraceRecord {
        sample_id,
        layer_index,
        kv_head_index,
        value_start_position,
        value_count,
        head_dim,
        values,
    })
}

/// Parse an in-memory `ADAV01` value-trace corpus.
///
/// # Errors
///
/// Returns an error for malformed metadata, unsupported
/// versions/stages, invalid dimensions or floating-point values,
/// truncation, trailing bytes, or other violations of the
/// `ADAV01` v1 contract.
#[must_use = "ADAV01 parsing and provenance errors must be checked"]
pub fn parse_value_trace_bytes(bytes: &[u8]) -> Result<ValueTraceCorpus, ValueTraceError> {
    let mut reader = ByteReader::new(bytes);

    if reader.read_array::<7>()? != VALUE_TRACE_MAGIC {
        return Err(ValueTraceError::Invalid("ADAV01 trace magic mismatch"));
    }

    if reader.read_u32()? != VALUE_TRACE_VERSION {
        return Err(ValueTraceError::Invalid(
            "ADAV01 trace version is unsupported",
        ));
    }

    let metadata = read_metadata(&mut reader)?;

    let mut records = Vec::with_capacity(metadata.record_count);

    for _ in 0..metadata.record_count {
        records.push(read_record(&mut reader)?);
    }

    if reader.remaining() != 0 {
        return Err(ValueTraceError::Invalid(
            "ADAV01 trace contains trailing bytes",
        ));
    }

    Ok(ValueTraceCorpus { metadata, records })
}

/// Read and parse an `ADAV01` trace file.
///
/// # Errors
///
/// Returns file-system I/O errors or any `ADAV01` contract
/// violation reported by [`parse_value_trace_bytes`].
#[must_use = "ADAV01 I/O and validation errors must be checked"]
pub fn read_value_trace_file(path: impl AsRef<Path>) -> Result<ValueTraceCorpus, ValueTraceError> {
    let bytes = fs::read(path)?;

    parse_value_trace_bytes(&bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn push_u32(bytes: &mut Vec<u8>, value: u32) {
        bytes.extend_from_slice(&value.to_le_bytes());
    }

    fn push_u64(bytes: &mut Vec<u8>, value: u64) {
        bytes.extend_from_slice(&value.to_le_bytes());
    }

    fn push_string(bytes: &mut Vec<u8>, value: &str) {
        push_u32(bytes, u32::try_from(value.len()).unwrap());

        bytes.extend_from_slice(value.as_bytes());
    }

    fn push_values(bytes: &mut Vec<u8>, values: &[f32]) {
        for value in values {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
    }

    fn valid_trace() -> Vec<u8> {
        let mut bytes = Vec::new();

        bytes.extend_from_slice(&VALUE_TRACE_MAGIC);

        push_u32(&mut bytes, VALUE_TRACE_VERSION);

        push_string(&mut bytes, "model/example");

        push_string(&mut bytes, "0123456789abcdef");

        push_string(&mut bytes, "tokenizer/example");

        push_string(&mut bytes, "fedcba9876543210");

        push_string(&mut bytes, "capture-v-001");

        push_string(&mut bytes, "bfloat16");

        push_string(&mut bytes, ATTENTION_VALUE_INPUT_PRE_REPEAT_KV_STAGE);

        push_u32(&mut bytes, 1);

        push_string(&mut bytes, "sample-7");

        push_u32(&mut bytes, 3);

        push_u32(&mut bytes, 1);

        push_u64(&mut bytes, 0);

        push_u32(&mut bytes, 3);

        push_u32(&mut bytes, 2);

        push_values(&mut bytes, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

        bytes
    }

    #[test]
    fn parses_valid_trace_and_preserves_provenance() {
        let corpus = parse_value_trace_bytes(&valid_trace()).unwrap();

        assert_eq!(corpus.len(), 1,);

        assert_eq!(corpus.metadata().model_id, "model/example",);

        assert_eq!(corpus.metadata().record_count, 1,);

        let record = &corpus.records()[0];

        assert_eq!(record.layer_index, 3,);

        assert_eq!(record.kv_head_index, 1,);

        assert_eq!(record.value_count, 3,);

        assert_eq!(record.head_dim, 2,);

        assert_eq!(record.values.len(), 6,);

        assert_eq!(record.row(1).unwrap(), &[3.0, 4.0],);

        assert_eq!(record.prefix_values(2).unwrap(), &[1.0, 2.0, 3.0, 4.0],);
    }

    #[test]
    fn finds_records_by_natural_gqa_identity() {
        let corpus = parse_value_trace_bytes(&valid_trace()).unwrap();

        assert!(corpus.find_record("sample-7", 3, 1,).is_some());

        assert!(corpus.find_record("sample-7", 3, 2,).is_none());
    }

    #[test]
    fn rejects_bad_magic() {
        let mut bytes = valid_trace();

        bytes[0] ^= 1;

        let error = parse_value_trace_bytes(&bytes).unwrap_err();

        assert!(error.to_string().contains("magic"));
    }

    #[test]
    fn rejects_trailing_bytes() {
        let mut bytes = valid_trace();

        bytes.push(0);

        let error = parse_value_trace_bytes(&bytes).unwrap_err();

        assert!(error.to_string().contains("trailing"));
    }

    #[test]
    fn rejects_wrong_tensor_stage() {
        let mut bytes = valid_trace();

        let needle = ATTENTION_VALUE_INPUT_PRE_REPEAT_KV_STAGE.as_bytes();

        let position = bytes
            .windows(needle.len())
            .position(|window| window == needle)
            .unwrap();

        bytes[position] = b'x';

        let error = parse_value_trace_bytes(&bytes).unwrap_err();

        assert!(error.to_string().contains("tensor_stage"));
    }

    #[test]
    fn rejects_truncated_tensor() {
        let mut bytes = valid_trace();

        bytes.truncate(bytes.len() - 2);

        let error = parse_value_trace_bytes(&bytes).unwrap_err();

        assert!(error.to_string().contains("truncated"));
    }

    #[test]
    fn rejects_non_finite_values() {
        let mut bytes = valid_trace();

        let replacement = f32::NAN.to_le_bytes();

        let offset = bytes.len() - 4;

        bytes[offset..offset + 4].copy_from_slice(&replacement);

        let error = parse_value_trace_bytes(&bytes).unwrap_err();

        assert!(error.to_string().contains("non-finite"));
    }

    #[test]
    fn value_end_position_rejects_overflow() {
        let record = ValueTraceRecord {
            sample_id: "overflow".to_owned(),
            layer_index: 0,
            kv_head_index: 0,
            value_start_position: u64::MAX,
            value_count: 1,
            head_dim: 1,
            values: vec![0.0],
        };

        let error = record.value_end_position().unwrap_err();

        assert!(error.to_string().contains("interval overflow"));
    }

    #[test]
    fn rejects_row_outside_capture() {
        let corpus = parse_value_trace_bytes(&valid_trace()).unwrap();

        let record = &corpus.records()[0];

        assert!(record.row(3).is_err());
    }

    #[test]
    fn rejects_prefix_outside_capture() {
        let corpus = parse_value_trace_bytes(&valid_trace()).unwrap();

        let record = &corpus.records()[0];

        assert!(record.prefix_values(4).is_err());
    }
}
