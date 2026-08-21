#![forbid(unsafe_code)]

/// Logical algorithmic work. These fields are deliberately not hardware
/// instruction or bandwidth counters.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct LogicalMetrics {
    pub qk_pairs_evaluated: usize,
    pub exp_evaluations: usize,
    pub log_evaluations: usize,
    pub value_accumulate_elements: usize,
}

impl LogicalMetrics {
    #[must_use]
    pub const fn total_transcendentals(self) -> usize {
        self.exp_evaluations + self.log_evaluations
    }
}

/// One single-query attention result used by the initial ADA-A1 laboratory.
#[derive(Debug, Clone, PartialEq)]
pub struct AttentionResult {
    pub output: Vec<f32>,
    pub lse: f32,
    pub metrics: LogicalMetrics,
}

/// A deterministic single-query case. `values` is row-major `[seq_len, head_dim]`.
#[derive(Debug, Clone, PartialEq)]
pub struct AttentionCase {
    pub logits: Vec<f32>,
    pub values: Vec<f32>,
    pub head_dim: usize,
}

impl AttentionCase {
    pub fn validate(&self) -> Result<(), &'static str> {
        if self.logits.is_empty() {
            return Err("ADA-A1 requires at least one logit");
        }
        if self.head_dim == 0 {
            return Err("head_dim must be non-zero");
        }
        if self.values.len() != self.logits.len() * self.head_dim {
            return Err("values must have seq_len * head_dim elements");
        }
        if self.logits.iter().any(|x| !x.is_finite())
            || self.values.iter().any(|x| !x.is_finite())
        {
            return Err("ADA-A1 E0 cases must be finite");
        }
        Ok(())
    }
}
