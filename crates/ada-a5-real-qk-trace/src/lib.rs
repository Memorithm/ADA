#![forbid(unsafe_code)]

use std::fmt;
use std::fs;
use std::io;
use std::path::Path;

use ada_a4_qk_box::QueryKeyPagedCase;

pub const TRACE_MAGIC: [u8; 8] = *b"ADAQK01\0";
pub const TRACE_VERSION: u32 = 1;
pub const ATTENTION_SCORE_INPUT_STAGE: &str = "attention_score_input";

const MAX_STRING_BYTES: usize = 1_048_576;
const MAX_RECORDS: usize = 1_000_000;
const MAX_HEAD_DIM: usize = 65_536;
const MAX_VALUES_PER_RECORD: usize = 67_108_864;

#[derive(Debug)]
pub enum TraceError {
    Io(io::Error),
    Invalid(&'static str),
}

impl fmt::Display for TraceError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(error) => write!(formatter, "trace I/O error: {error}"),
            Self::Invalid(message) => formatter.write_str(message),
        }
    }
}

impl std::error::Error for TraceError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(error) => Some(error),
            Self::Invalid(_) => None,
        }
    }
}

impl From<io::Error> for TraceError {
    fn from(error: io::Error) -> Self {
        Self::Io(error)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TraceMetadata {
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
pub struct TraceRecord {
    pub sample_id: String,
    pub layer_index: u32,
    pub query_head_index: u32,
    pub kv_head_index: u32,
    pub query_position: u64,
    pub key_start_position: u64,
    pub head_dim: usize,
    pub key_count: usize,
    pub score_scale: f64,
    pub query: Vec<f64>,
    pub keys: Vec<f64>,
}

impl TraceRecord {
    #[must_use]
    pub fn sample_fingerprint(&self) -> u64 {
        fingerprint_bytes(self.sample_id.as_bytes())
    }

    /// Convert this captured record into the existing A4/A5 scalar Q/K case.
    ///
    /// The trace already stores the exact visible key interval for this query,
    /// so replay operates on the captured key order directly.
    ///
    /// # Errors
    ///
    /// Returns an error when `page_size` is zero or `alpha` is outside the
    /// current Entmax candidate domain `(1, 2]`.
    pub fn to_query_key_case(
        &self,
        page_size: usize,
        alpha: f64,
    ) -> Result<QueryKeyPagedCase, TraceError> {
        if page_size == 0 {
            return Err(TraceError::Invalid("ADA-A5 E4 page_size must be non-zero"));
        }
        if !alpha.is_finite() || alpha <= 1.0 || alpha > 2.0 {
            return Err(TraceError::Invalid(
                "ADA-A5 E4 alpha must be finite and in (1, 2]",
            ));
        }
        Ok(QueryKeyPagedCase {
            query: self.query.clone(),
            keys: self.keys.clone(),
            head_dim: self.head_dim,
            page_size,
            alpha,
            score_scale: self.score_scale,
        })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct TraceCorpus {
    metadata: TraceMetadata,
    records: Vec<TraceRecord>,
}

impl TraceCorpus {
    #[must_use]
    pub const fn metadata(&self) -> &TraceMetadata {
        &self.metadata
    }

    #[must_use]
    pub fn records(&self) -> &[TraceRecord] {
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

    fn take(&mut self, length: usize) -> Result<&'a [u8], TraceError> {
        let end = self
            .offset
            .checked_add(length)
            .ok_or(TraceError::Invalid("ADA-A5 E4 trace offset overflow"))?;
        let slice = self
            .bytes
            .get(self.offset..end)
            .ok_or(TraceError::Invalid("ADA-A5 E4 trace is truncated"))?;
        self.offset = end;
        Ok(slice)
    }

    fn read_array<const N: usize>(&mut self) -> Result<[u8; N], TraceError> {
        let bytes = self.take(N)?;
        <[u8; N]>::try_from(bytes)
            .map_err(|_| TraceError::Invalid("ADA-A5 E4 fixed-width decode failed"))
    }

    fn read_u32(&mut self) -> Result<u32, TraceError> {
        Ok(u32::from_le_bytes(self.read_array()?))
    }

    fn read_u64(&mut self) -> Result<u64, TraceError> {
        Ok(u64::from_le_bytes(self.read_array()?))
    }

    fn read_f64(&mut self) -> Result<f64, TraceError> {
        Ok(f64::from_le_bytes(self.read_array()?))
    }

    fn read_string(&mut self) -> Result<String, TraceError> {
        let length = usize::try_from(self.read_u32()?)
            .map_err(|_| TraceError::Invalid("ADA-A5 E4 string length conversion failed"))?;
        if length > MAX_STRING_BYTES {
            return Err(TraceError::Invalid(
                "ADA-A5 E4 metadata string exceeds size limit",
            ));
        }
        let bytes = self.take(length)?;
        let text = std::str::from_utf8(bytes)
            .map_err(|_| TraceError::Invalid("ADA-A5 E4 metadata is not valid UTF-8"))?;
        Ok(text.to_owned())
    }

    fn read_f32_values(&mut self, count: usize) -> Result<Vec<f64>, TraceError> {
        if count > MAX_VALUES_PER_RECORD {
            return Err(TraceError::Invalid(
                "ADA-A5 E4 numeric tensor exceeds per-record value limit",
            ));
        }
        let byte_count = count
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or(TraceError::Invalid("ADA-A5 E4 tensor byte length overflow"))?;
        let bytes = self.take(byte_count)?;
        let mut values = Vec::with_capacity(count);
        for chunk in bytes.chunks_exact(std::mem::size_of::<f32>()) {
            let encoded = <[u8; 4]>::try_from(chunk)
                .map_err(|_| TraceError::Invalid("ADA-A5 E4 f32 decode failed"))?;
            let value = f32::from_le_bytes(encoded);
            if !value.is_finite() {
                return Err(TraceError::Invalid(
                    "ADA-A5 E4 trace contains non-finite Q/K value",
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

fn validate_non_empty(value: &str, message: &'static str) -> Result<(), TraceError> {
    if value.is_empty() {
        Err(TraceError::Invalid(message))
    } else {
        Ok(())
    }
}

fn read_metadata(reader: &mut ByteReader<'_>) -> Result<TraceMetadata, TraceError> {
    let model_id = reader.read_string()?;
    let model_revision = reader.read_string()?;
    let tokenizer_id = reader.read_string()?;
    let tokenizer_revision = reader.read_string()?;
    let capture_id = reader.read_string()?;
    let source_dtype = reader.read_string()?;
    let tensor_stage = reader.read_string()?;
    let record_count = usize::try_from(reader.read_u32()?)
        .map_err(|_| TraceError::Invalid("ADA-A5 E4 record count conversion failed"))?;

    validate_non_empty(&model_id, "ADA-A5 E4 model_id must be non-empty")?;
    validate_non_empty(
        &model_revision,
        "ADA-A5 E4 model_revision must be non-empty",
    )?;
    validate_non_empty(&tokenizer_id, "ADA-A5 E4 tokenizer_id must be non-empty")?;
    validate_non_empty(
        &tokenizer_revision,
        "ADA-A5 E4 tokenizer_revision must be non-empty",
    )?;
    validate_non_empty(&capture_id, "ADA-A5 E4 capture_id must be non-empty")?;
    validate_non_empty(&source_dtype, "ADA-A5 E4 source_dtype must be non-empty")?;
    if tensor_stage != ATTENTION_SCORE_INPUT_STAGE {
        return Err(TraceError::Invalid(
            "ADA-A5 E4 tensor_stage must be attention_score_input",
        ));
    }
    if record_count == 0 || record_count > MAX_RECORDS {
        return Err(TraceError::Invalid(
            "ADA-A5 E4 record_count is outside the supported range",
        ));
    }

    Ok(TraceMetadata {
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

fn read_record(reader: &mut ByteReader<'_>) -> Result<TraceRecord, TraceError> {
    let sample_id = reader.read_string()?;
    validate_non_empty(&sample_id, "ADA-A5 E4 sample_id must be non-empty")?;

    let layer_index = reader.read_u32()?;
    let query_head_index = reader.read_u32()?;
    let kv_head_index = reader.read_u32()?;
    let query_position = reader.read_u64()?;
    let key_start_position = reader.read_u64()?;
    let head_dim = usize::try_from(reader.read_u32()?)
        .map_err(|_| TraceError::Invalid("ADA-A5 E4 head_dim conversion failed"))?;
    let key_count = usize::try_from(reader.read_u32()?)
        .map_err(|_| TraceError::Invalid("ADA-A5 E4 key_count conversion failed"))?;
    let score_scale = reader.read_f64()?;

    if head_dim == 0 || head_dim > MAX_HEAD_DIM {
        return Err(TraceError::Invalid(
            "ADA-A5 E4 head_dim is outside the supported range",
        ));
    }
    if key_count == 0 {
        return Err(TraceError::Invalid("ADA-A5 E4 key_count must be non-zero"));
    }
    if !score_scale.is_finite() || score_scale <= 0.0 {
        return Err(TraceError::Invalid(
            "ADA-A5 E4 score_scale must be finite and positive",
        ));
    }

    let key_count_u64 = u64::try_from(key_count)
        .map_err(|_| TraceError::Invalid("ADA-A5 E4 key_count does not fit in u64"))?;
    let visible_end = key_start_position
        .checked_add(key_count_u64)
        .ok_or(TraceError::Invalid(
            "ADA-A5 E4 visible key interval overflow",
        ))?;
    if query_position < key_start_position || query_position >= visible_end {
        return Err(TraceError::Invalid(
            "ADA-A5 E4 query_position must lie inside the visible key interval",
        ));
    }

    let key_value_count = key_count
        .checked_mul(head_dim)
        .ok_or(TraceError::Invalid("ADA-A5 E4 K tensor length overflow"))?;
    if key_value_count > MAX_VALUES_PER_RECORD {
        return Err(TraceError::Invalid(
            "ADA-A5 E4 K tensor exceeds per-record value limit",
        ));
    }

    let query = reader.read_f32_values(head_dim)?;
    let keys = reader.read_f32_values(key_value_count)?;

    Ok(TraceRecord {
        sample_id,
        layer_index,
        query_head_index,
        kv_head_index,
        query_position,
        key_start_position,
        head_dim,
        key_count,
        score_scale,
        query,
        keys,
    })
}

/// Parse an in-memory `ADAQK01` E4 trace corpus.
///
/// # Errors
///
/// Returns an error for malformed metadata, unsupported versions/stages,
/// invalid dimensions or floating-point values, truncation, trailing bytes, or
/// other violations of the E4 v1 trace contract.
#[must_use = "trace parsing errors and provenance must be checked before replay"]
pub fn parse_trace_bytes(bytes: &[u8]) -> Result<TraceCorpus, TraceError> {
    let mut reader = ByteReader::new(bytes);
    if reader.read_array::<8>()? != TRACE_MAGIC {
        return Err(TraceError::Invalid("ADA-A5 E4 trace magic mismatch"));
    }
    if reader.read_u32()? != TRACE_VERSION {
        return Err(TraceError::Invalid(
            "ADA-A5 E4 trace version is unsupported",
        ));
    }

    let metadata = read_metadata(&mut reader)?;
    let mut records = Vec::with_capacity(metadata.record_count);
    for _ in 0..metadata.record_count {
        records.push(read_record(&mut reader)?);
    }
    if reader.remaining() != 0 {
        return Err(TraceError::Invalid(
            "ADA-A5 E4 trace contains trailing bytes",
        ));
    }

    Ok(TraceCorpus { metadata, records })
}

/// Read and parse an `ADAQK01` trace file from disk.
///
/// # Errors
///
/// Returns file-system I/O errors or any E4 v1 trace-contract error reported by
/// [`parse_trace_bytes`].
#[must_use = "trace I/O and validation errors must be checked before replay"]
pub fn read_trace_file(path: impl AsRef<Path>) -> Result<TraceCorpus, TraceError> {
    let bytes = fs::read(path)?;
    parse_trace_bytes(&bytes)
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

    fn push_f64(bytes: &mut Vec<u8>, value: f64) {
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
        bytes.extend_from_slice(&TRACE_MAGIC);
        push_u32(&mut bytes, TRACE_VERSION);
        push_string(&mut bytes, "model/example");
        push_string(&mut bytes, "0123456789abcdef");
        push_string(&mut bytes, "tokenizer/example");
        push_string(&mut bytes, "fedcba9876543210");
        push_string(&mut bytes, "capture-001");
        push_string(&mut bytes, "bfloat16");
        push_string(&mut bytes, ATTENTION_SCORE_INPUT_STAGE);
        push_u32(&mut bytes, 1);

        push_string(&mut bytes, "sample-7");
        push_u32(&mut bytes, 3);
        push_u32(&mut bytes, 5);
        push_u32(&mut bytes, 1);
        push_u64(&mut bytes, 2);
        push_u64(&mut bytes, 0);
        push_u32(&mut bytes, 2);
        push_u32(&mut bytes, 3);
        push_f64(&mut bytes, 0.5);
        push_values(&mut bytes, &[1.0, -2.0]);
        push_values(&mut bytes, &[1.0, 0.0, 0.0, 1.0, -1.0, 2.0]);
        bytes
    }

    #[test]
    fn parses_valid_trace_and_preserves_provenance() {
        let corpus = parse_trace_bytes(&valid_trace()).unwrap();
        assert_eq!(corpus.len(), 1);
        assert_eq!(corpus.metadata().model_id, "model/example");
        assert_eq!(corpus.metadata().record_count, 1);
        let record = &corpus.records()[0];
        assert_eq!(record.layer_index, 3);
        assert_eq!(record.query_head_index, 5);
        assert_eq!(record.kv_head_index, 1);
        assert_eq!(record.query_position, 2);
        assert_eq!(record.key_count, 3);
        assert_eq!(record.query.len(), 2);
        assert_eq!(record.query[0].to_bits(), 1.0_f64.to_bits());
        assert_eq!(record.query[1].to_bits(), (-2.0_f64).to_bits());
        assert_eq!(record.keys.len(), 6);
        let case = record.to_query_key_case(2, 1.5).unwrap();
        assert_eq!(case.page_size, 2);
        assert_eq!(case.alpha.to_bits(), 1.5_f64.to_bits());
        assert_eq!(case.score_scale.to_bits(), 0.5_f64.to_bits());
    }

    #[test]
    fn rejects_bad_magic() {
        let mut bytes = valid_trace();
        bytes[0] ^= 1;
        let error = parse_trace_bytes(&bytes).unwrap_err();
        assert!(error.to_string().contains("magic"));
    }

    #[test]
    fn rejects_trailing_bytes() {
        let mut bytes = valid_trace();
        bytes.push(0);
        let error = parse_trace_bytes(&bytes).unwrap_err();
        assert!(error.to_string().contains("trailing"));
    }

    #[test]
    fn rejects_wrong_tensor_stage() {
        let mut bytes = valid_trace();
        let needle = ATTENTION_SCORE_INPUT_STAGE.as_bytes();
        let position = bytes
            .windows(needle.len())
            .position(|window| window == needle)
            .unwrap();
        bytes[position] = b'x';
        let error = parse_trace_bytes(&bytes).unwrap_err();
        assert!(error.to_string().contains("tensor_stage"));
    }

    #[test]
    fn rejects_truncated_tensor() {
        let mut bytes = valid_trace();
        bytes.truncate(bytes.len() - 2);
        let error = parse_trace_bytes(&bytes).unwrap_err();
        assert!(error.to_string().contains("truncated"));
    }

    #[test]
    fn rejects_query_outside_visible_interval() {
        let mut bytes = valid_trace();
        let corpus = parse_trace_bytes(&bytes).unwrap();
        assert_eq!(corpus.records()[0].query_position, 2);

        let query_position_offset = 8
            + 4
            + (4 + "model/example".len())
            + (4 + "0123456789abcdef".len())
            + (4 + "tokenizer/example".len())
            + (4 + "fedcba9876543210".len())
            + (4 + "capture-001".len())
            + (4 + "bfloat16".len())
            + (4 + ATTENTION_SCORE_INPUT_STAGE.len())
            + 4
            + (4 + "sample-7".len())
            + 4
            + 4
            + 4;
        bytes[query_position_offset..query_position_offset + 8]
            .copy_from_slice(&9_u64.to_le_bytes());
        let error = parse_trace_bytes(&bytes).unwrap_err();
        assert!(error.to_string().contains("visible key interval"));
    }
}
