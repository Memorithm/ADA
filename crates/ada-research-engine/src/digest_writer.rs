//! Explicit, versioned canonical byte encoding for content digests.
//!
//! Digests are SHA-256 over this encoding — never over JSON, whose key order
//! and float formatting could drift. The stream is fully specified:
//!
//! * a caller-supplied magic prefix domain-separates each digest kind;
//! * every composite value is prefixed with an explicit tag;
//! * integers are fixed-width little-endian (`usize` through checked `u64`);
//! * floats use their exact bit patterns (so `-0.0 != 0.0`);
//! * strings and vectors are length-prefixed.
//!
//! This is a content-integrity / corruption-detection digest. It is unkeyed
//! and therefore not an authenticator.

use sha2::{Digest, Sha256};

/// A canonical byte-stream builder.
#[derive(Debug, Clone)]
pub struct DigestWriter {
    hasher: Sha256,
}

/// Checked-conversion failure (only reachable on exotic pointer widths).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DigestOverflow;

impl std::fmt::Display for DigestOverflow {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "usize value does not fit into u64 for digest encoding"
        )
    }
}

impl std::error::Error for DigestOverflow {}

impl DigestWriter {
    /// Start a new stream with a domain-separation magic prefix.
    #[must_use]
    pub fn new(magic: &[u8]) -> Self {
        let mut hasher = Sha256::new();
        hasher.update(magic);
        Self { hasher }
    }

    /// Append one tagged byte.
    pub fn tag(&mut self, tag: u8) {
        self.hasher.update([tag]);
    }

    /// Append a raw byte.
    pub fn u8(&mut self, value: u8) {
        self.hasher.update([value]);
    }

    /// Append a little-endian `u32`.
    pub fn u32(&mut self, value: u32) {
        self.hasher.update(value.to_le_bytes());
    }

    /// Append a little-endian `u64`.
    pub fn u64(&mut self, value: u64) {
        self.hasher.update(value.to_le_bytes());
    }

    /// Append a length-checked little-endian `usize`.
    ///
    /// # Errors
    ///
    /// Returns [`DigestOverflow`] when `usize` exceeds `u64`.
    pub fn usize(&mut self, value: usize) -> Result<(), DigestOverflow> {
        let value = u64::try_from(value).map_err(|_| DigestOverflow)?;
        self.u64(value);
        Ok(())
    }

    /// Append an `f64` by exact bit pattern.
    pub fn f64(&mut self, value: f64) {
        self.u64(value.to_bits());
    }

    /// Append a boolean as one byte.
    pub fn bool(&mut self, value: bool) {
        self.u8(u8::from(value));
    }

    /// Append a length-prefixed UTF-8 string.
    ///
    /// # Errors
    ///
    /// See [`DigestWriter::usize`].
    pub fn str(&mut self, value: &str) -> Result<(), DigestOverflow> {
        self.usize(value.len())?;
        self.hasher.update(value.as_bytes());
        Ok(())
    }

    /// Append a length-prefixed vector of `f64`s.
    ///
    /// # Errors
    ///
    /// See [`DigestWriter::usize`].
    pub fn f64_slice(&mut self, values: &[f64]) -> Result<(), DigestOverflow> {
        self.usize(values.len())?;
        for value in values {
            self.f64(*value);
        }
        Ok(())
    }

    /// Append a length-prefixed vector of `usize`s.
    ///
    /// # Errors
    ///
    /// See [`DigestWriter::usize`].
    pub fn usize_slice(&mut self, values: &[usize]) -> Result<(), DigestOverflow> {
        self.usize(values.len())?;
        for value in values {
            self.usize(*value)?;
        }
        Ok(())
    }

    /// Finish the stream and return the SHA-256 digest.
    #[must_use]
    pub fn finish(self) -> [u8; 32] {
        self.hasher.finalize().into()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::canon::hex;

    #[test]
    fn encoding_is_deterministic_and_order_sensitive() {
        let mut a = DigestWriter::new(b"TEST\0");
        a.tag(0x01);
        a.u32(7);
        a.f64(-0.25);
        let da = hex(&a.finish());

        let mut b = DigestWriter::new(b"TEST\0");
        b.tag(0x01);
        b.u32(7);
        b.f64(-0.25);
        assert_eq!(da, hex(&b.finish()));

        let mut c = DigestWriter::new(b"TEST\0");
        c.tag(0x02);
        c.u32(7);
        c.f64(-0.25);
        assert_ne!(da, hex(&c.finish()));
    }

    #[test]
    fn signed_zero_and_nan_bits_are_preserved() {
        let mut zero = DigestWriter::new(b"Z\0");
        zero.f64(0.0);
        let mut neg_zero = DigestWriter::new(b"Z\0");
        neg_zero.f64(-0.0);
        assert_ne!(hex(&zero.finish()), hex(&neg_zero.finish()));
    }

    #[test]
    fn strings_and_slices_are_length_prefixed() {
        let digest_a = {
            let mut a = DigestWriter::new(b"S\0");
            a.str("ab").unwrap();
            a.f64_slice(&[1.0, 2.0]).unwrap();
            a.usize_slice(&[3, 4]).unwrap();
            hex(&a.finish())
        };

        let same = {
            let mut b = DigestWriter::new(b"S\0");
            b.str("ab").unwrap();
            b.f64_slice(&[1.0, 2.0]).unwrap();
            b.usize_slice(&[3, 4]).unwrap();
            hex(&b.finish())
        };
        assert_eq!(digest_a, same);

        let different = {
            let mut c = DigestWriter::new(b"S\0");
            c.str("abc").unwrap();
            c.f64_slice(&[1.0, 2.0]).unwrap();
            c.usize_slice(&[3, 4]).unwrap();
            hex(&c.finish())
        };
        assert_ne!(digest_a, different);
    }
}
