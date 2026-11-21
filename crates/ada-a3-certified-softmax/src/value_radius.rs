//! Value-aware output-error certificate on top of norm-ordered early
//! termination (ADA-A3, research reference).
//!
//! Earlier A3 modules certify the L1 distance of the *probabilities*. A
//! consumer of attention cares about the *output* `o = Σ p_i v_i`. Let the
//! scan have read mass `S` and let the unread fraction be `t = T/(S+T)`
//! (`T ≤ T_hi` from the Cauchy–Schwarz norm bound). The truncated output `ô`
//! (renormalized over read keys) satisfies `o − ô = t (õ − ô)` where `õ` is the
//! mean of the unread values under the true weights, a convex combination of
//! unread values. With a prefill centre `c` (mean of all values) and
//! `R_j = max_{unread} ‖v_i − c‖`, we have `‖õ − ô‖ ≤ R_j + ‖ô − c‖` because
//! `ô` is known exactly at run time. Hence
//! `‖o − ô‖₂ ≤ t_hi (R_j + ‖ô − c‖)`.
//! Three certificate modes are provided; all are valid, they differ in
//! tightness: `Uncentered` (`2·max‖v‖`, the L1-route bound), `Centered`
//! (global radius about `c`) and `SuffixCentered` (radius of the still-unread
//! suffix of the norm order). The scan stops once the bound is ≤ `epsilon_out`.

/// Relative inflation applied to every bound for floating-point rounding.
const BOUND_INFLATION: f64 = 1.0e-9;

/// Which output-error certificate is used to stop the scan.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ValueRadiusMode {
    /// `t_hi · 2 · max_i ‖v_i‖` (what the probability-L1 certificate implies).
    Uncentered,
    /// `t_hi · (R + ‖ô − c‖)` with global radius `R = max ‖v_i − c‖`.
    Centered,
    /// `t_hi · (R_j + ‖ô − c‖)` with the radius of the unread suffix.
    SuffixCentered,
}

/// Logical (not hardware) cost counters for one query.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ValueRadiusCost {
    /// Prefill: key norms, value distances to the centre, one sort.
    pub prefill_row_passes: usize,
    /// Key rows dotted with the query.
    pub key_rows_read: usize,
    /// Value rows accumulated into the output.
    pub value_rows_read: usize,
    /// Rows not read (key and value).
    pub rows_skipped: usize,
    /// Certificate evaluations.
    pub bound_checks: usize,
}

/// Result of the value-radius certified attention output.
#[derive(Clone, Debug, PartialEq)]
pub struct ValueRadiusOutput {
    /// Truncated attention output `ô` (length `value_dim`).
    pub output: Vec<f64>,
    /// Certified upper bound on `‖o − ô‖₂` (0 when every row was read).
    pub certified_l2_bound: f64,
    /// Caller budget on the output L2 error.
    pub epsilon_out: f64,
    /// Whether each key/value row was read, in input order.
    pub read: Vec<bool>,
}

fn norm(row: &[f64]) -> f64 {
    row.iter().map(|v| v * v).sum::<f64>().sqrt()
}

fn dist(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f64>()
        .sqrt()
}

/// Inputs describing one attention call.
#[derive(Clone, Copy, Debug)]
pub struct ValueRadiusInput<'a> {
    /// Query, length `head_dim`.
    pub query: &'a [f64],
    /// Keys, row-major `[n, head_dim]`.
    pub keys: &'a [f64],
    /// Values, row-major `[n, value_dim]`.
    pub values: &'a [f64],
    /// Key/query dimension.
    pub head_dim: usize,
    /// Value dimension.
    pub value_dim: usize,
    /// Positive scale applied to every score.
    pub score_scale: f64,
    /// Budget on the output L2 error.
    pub epsilon_out: f64,
}

fn validate(i: &ValueRadiusInput<'_>) -> Result<usize, &'static str> {
    if i.head_dim == 0 || i.value_dim == 0 {
        return Err("ADA-A3 value-radius dims must be positive");
    }
    if i.query.len() != i.head_dim {
        return Err("ADA-A3 value-radius query must contain head_dim elements");
    }
    if i.keys.is_empty() || i.keys.len() % i.head_dim != 0 {
        return Err("ADA-A3 value-radius keys must be non-empty row-major [n, head_dim]");
    }
    let n = i.keys.len() / i.head_dim;
    if i.values.len() != n * i.value_dim {
        return Err("ADA-A3 value-radius values must be row-major [n, value_dim]");
    }
    if i.query
        .iter()
        .chain(i.keys)
        .chain(i.values)
        .any(|v| !v.is_finite())
    {
        return Err("ADA-A3 value-radius inputs must be finite");
    }
    if !i.score_scale.is_finite() || i.score_scale <= 0.0 {
        return Err("ADA-A3 value-radius score_scale must be finite and positive");
    }
    if !i.epsilon_out.is_finite() || i.epsilon_out < 0.0 {
        return Err("ADA-A3 value-radius epsilon_out must be finite and non-negative");
    }
    Ok(n)
}

/// Norm-ordered scan with an output-L2 certificate.
///
/// # Errors
///
/// Returns an error for inconsistent layout, non-finite input, bad scale or
/// budget, or a non-finite score/mass/output.
#[allow(
    clippy::too_many_lines,
    clippy::cast_precision_loss,
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::needless_for_each
)]
#[must_use = "the certified output should be checked"]
pub fn certified_value_radius_attention_accounted(
    input: &ValueRadiusInput<'_>,
    mode: ValueRadiusMode,
) -> Result<(ValueRadiusOutput, ValueRadiusCost), &'static str> {
    let n = validate(input)?;
    let (d, dv) = (input.head_dim, input.value_dim);
    let q_norm = norm(input.query);
    let knorm: Vec<f64> = input.keys.chunks_exact(d).map(norm).collect();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| knorm[b].total_cmp(&knorm[a]).then(a.cmp(&b)));

    let mut centre = vec![0.0_f64; dv];
    for row in input.values.chunks_exact(dv) {
        for (c, v) in centre.iter_mut().zip(row) {
            *c += v / n as f64;
        }
    }
    let row_v = |i: usize| &input.values[i * dv..(i + 1) * dv];
    let rad: Vec<f64> = (0..n).map(|i| dist(row_v(i), &centre)).collect();
    let vmax = (0..n).map(|i| norm(row_v(i))).fold(0.0_f64, f64::max);
    let r_global = rad.iter().copied().fold(0.0_f64, f64::max);
    // suffix_rad[j] = max radius over order[j..]
    let mut suffix_rad = vec![0.0_f64; n + 1];
    for j in (0..n).rev() {
        suffix_rad[j] = suffix_rad[j + 1].max(rad[order[j]]);
    }
    // Log-domain suffix sum of loose per-key bounds h_i (ln Σ exp(h_i)).
    let h = |i: usize| input.score_scale * q_norm * knorm[i] * (1.0 + BOUND_INFLATION);
    let mut ln_suffix = vec![f64::NEG_INFINITY; n + 1];
    for j in (0..n).rev() {
        let (a, b) = (ln_suffix[j + 1], h(order[j]));
        ln_suffix[j] = if a == f64::NEG_INFINITY {
            b
        } else {
            let m = a.max(b);
            m + ((a - m).exp() + (b - m).exp()).ln()
        };
    }

    let mut cost = ValueRadiusCost {
        prefill_row_passes: 2 * n,
        ..ValueRadiusCost::default()
    };
    let mut acc = vec![0.0_f64; dv]; // Σ exp(z − m) v over read rows
    let mut mass = 0.0_f64;
    let mut m = f64::NEG_INFINITY;
    let mut read = vec![false; n];
    let mut certificate = 0.0_f64;
    let mut read_count = n;
    for (rank, &idx) in order.iter().enumerate() {
        let z = input.score_scale
            * input
                .query
                .iter()
                .zip(&input.keys[idx * d..(idx + 1) * d])
                .map(|(a, b)| a * b)
                .sum::<f64>();
        if !z.is_finite() {
            return Err("ADA-A3 value-radius scores must be finite");
        }
        cost.key_rows_read += 1;
        cost.value_rows_read += 1;
        if z > m {
            let f = if m == f64::NEG_INFINITY {
                0.0
            } else {
                (m - z).exp()
            };
            mass *= f;
            acc.iter_mut().for_each(|a| *a *= f);
            m = z;
        }
        let w = (z - m).exp();
        mass += w;
        for (a, v) in acc.iter_mut().zip(row_v(idx)) {
            *a += w * v;
        }
        read[idx] = true;
        let j = rank + 1;
        if j == n {
            certificate = 0.0;
            break;
        }
        // ln T_hi (skipping rows j..n): ln Σ exp(h_i) − m.
        let ln_t = ln_suffix[j] - m;
        let t_hi = 1.0 / (1.0 + (mass.ln() - ln_t).exp()); // T/(S+T)
        let o_hat: Vec<f64> = acc.iter().map(|a| a / mass).collect();
        let radius = match mode {
            ValueRadiusMode::Uncentered => 2.0 * vmax,
            ValueRadiusMode::Centered => r_global + dist(&o_hat, &centre),
            ValueRadiusMode::SuffixCentered => suffix_rad[j] + dist(&o_hat, &centre),
        };
        cost.bound_checks += 1;
        let cert = t_hi * radius * (1.0 + BOUND_INFLATION);
        if cert <= input.epsilon_out {
            certificate = cert;
            read_count = j;
            break;
        }
    }
    cost.rows_skipped = n - read_count;
    if !mass.is_finite() || mass <= 0.0 {
        return Err("ADA-A3 value-radius mass must be positive and finite");
    }
    let output: Vec<f64> = acc.iter().map(|a| a / mass).collect();
    if output.iter().any(|v| !v.is_finite()) {
        return Err("ADA-A3 value-radius output must be finite");
    }
    Ok((
        ValueRadiusOutput {
            output,
            certified_l2_bound: certificate,
            epsilon_out: input.epsilon_out,
            read,
        },
        cost,
    ))
}

/// Same as [`certified_value_radius_attention_accounted`] without costs.
///
/// # Errors
///
/// Same failure conditions as the accounted variant.
#[must_use = "the certified output should be checked"]
pub fn certified_value_radius_attention(
    input: &ValueRadiusInput<'_>,
    mode: ValueRadiusMode,
) -> Result<ValueRadiusOutput, &'static str> {
    certified_value_radius_attention_accounted(input, mode).map(|(r, _)| r)
}

#[cfg(test)]
#[allow(
    clippy::many_single_char_names,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::needless_range_loop,
    clippy::needless_for_each,
    clippy::too_many_lines,
    clippy::float_cmp
)]
mod tests {
    use super::*;

    /// Independent dense oracle: stable softmax-weighted value mean.
    fn oracle(i: &ValueRadiusInput<'_>) -> Vec<f64> {
        let n = i.keys.len() / i.head_dim;
        let z: Vec<f64> = (0..n)
            .map(|r| {
                i.score_scale
                    * (0..i.head_dim)
                        .map(|c| i.query[c] * i.keys[r * i.head_dim + c])
                        .sum::<f64>()
            })
            .collect();
        let mx = z.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let w: Vec<f64> = z.iter().map(|s| (s - mx).exp()).collect();
        let tot: f64 = w.iter().sum();
        (0..i.value_dim)
            .map(|c| {
                (0..n)
                    .map(|r| w[r] / tot * i.values[r * i.value_dim + c])
                    .sum()
            })
            .collect()
    }

    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> f64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            (self.0 >> 11) as f64 / (1_u64 << 53) as f64
        }
        fn gauss(&mut self) -> f64 {
            let u = self.next().max(1e-300);
            let v = self.next();
            (-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * v).cos()
        }
    }

    const MODES: [ValueRadiusMode; 3] = [
        ValueRadiusMode::Uncentered,
        ValueRadiusMode::Centered,
        ValueRadiusMode::SuffixCentered,
    ];

    fn check(i: &ValueRadiusInput<'_>) -> [usize; 3] {
        let exact = oracle(i);
        let mut reads = [0; 3];
        for (k, mode) in MODES.iter().enumerate() {
            let (r, c) = certified_value_radius_attention_accounted(i, *mode).unwrap();
            let err = dist(&r.output, &exact);
            assert!(
                err <= r.certified_l2_bound + 1e-9 * (1.0 + norm(&exact)),
                "viol {mode:?} err {err} cert {}",
                r.certified_l2_bound
            );
            assert!(r.certified_l2_bound <= i.epsilon_out || r.certified_l2_bound == 0.0);
            assert_eq!(c.key_rows_read + c.rows_skipped, i.keys.len() / i.head_dim);
            reads[k] = c.key_rows_read;
        }
        reads
    }

    #[test]
    fn rejects_invalid_input() {
        let ok = ValueRadiusInput {
            query: &[1.0, 0.0],
            keys: &[1.0, 0.0, 0.0, 1.0],
            values: &[1.0, 2.0],
            head_dim: 2,
            value_dim: 1,
            score_scale: 1.0,
            epsilon_out: 0.1,
        };
        assert!(certified_value_radius_attention(&ok, ValueRadiusMode::Centered).is_ok());
        let bad = [
            ValueRadiusInput { head_dim: 0, ..ok },
            ValueRadiusInput { value_dim: 0, ..ok },
            ValueRadiusInput {
                query: &[1.0],
                ..ok
            },
            ValueRadiusInput { keys: &[], ..ok },
            ValueRadiusInput {
                keys: &[1.0, 0.0, 0.0],
                ..ok
            },
            ValueRadiusInput {
                values: &[1.0],
                ..ok
            },
            ValueRadiusInput {
                values: &[f64::NAN, 1.0],
                ..ok
            },
            ValueRadiusInput {
                query: &[f64::INFINITY, 0.0],
                ..ok
            },
            ValueRadiusInput {
                score_scale: 0.0,
                ..ok
            },
            ValueRadiusInput {
                score_scale: f64::NAN,
                ..ok
            },
            ValueRadiusInput {
                epsilon_out: -1.0,
                ..ok
            },
            ValueRadiusInput {
                epsilon_out: f64::NAN,
                ..ok
            },
        ];
        for b in &bad {
            assert!(certified_value_radius_attention(b, ValueRadiusMode::Centered).is_err());
        }
    }

    #[test]
    fn exhaustive_grid_has_no_violation() {
        let vals = [-2.0, 0.0, 1.5];
        let mut skipped = 0_usize;
        let (max_n, stride) = if cfg!(miri) {
            (2_usize, 5_usize)
        } else {
            (3, 1)
        };
        for n in 1..=max_n {
            for code in (0..3_usize.pow(u32::try_from(2 * n).unwrap())).step_by(stride) {
                let keys: Vec<f64> = (0..2 * n)
                    .map(|p| vals[(code / 3_usize.pow(p as u32)) % 3])
                    .collect();
                for vcode in 0..3_usize.pow(n as u32) {
                    let values: Vec<f64> = (0..n)
                        .map(|p| vals[(vcode / 3_usize.pow(p as u32)) % 3] * 3.0)
                        .collect();
                    for q in [[1.0, 0.5], [-2.0, 1.0], [0.0, 0.0]] {
                        for eps in [0.0, 0.05, 1.0, 100.0] {
                            let i = ValueRadiusInput {
                                query: &q,
                                keys: &keys,
                                values: &values,
                                head_dim: 2,
                                value_dim: 1,
                                score_scale: 1.0,
                                epsilon_out: eps,
                            };
                            let r = check(&i);
                            skipped += usize::from(r.iter().any(|&x| x < n));
                        }
                    }
                }
            }
        }
        assert!(cfg!(miri) || skipped > 0);
    }

    #[test]
    fn adversarial_sweep_has_no_violation() {
        let mut rng = Rng(0xA11CE);
        for t in 0..if cfg!(miri) { 6 } else { 300 } {
            let n = 2 + (rng.next() * 40.0) as usize;
            let (d, dv) = (
                1 + (rng.next() * 6.0) as usize,
                1 + (rng.next() * 5.0) as usize,
            );
            let kscale = [1.0, 30.0, 1000.0][t % 3];
            let mut keys: Vec<f64> = (0..n * d).map(|_| rng.gauss() * kscale).collect();
            let mut values: Vec<f64> = (0..n * dv).map(|_| rng.gauss()).collect();
            if t % 4 == 0 && n > 3 {
                // duplicates and a huge-value outlier on the smallest-norm key
                let (a, b) = keys.split_at_mut(d);
                b[..d].copy_from_slice(a);
                values[(n - 1) * dv] = 1.0e6;
            }
            if t % 5 == 0 {
                values.iter_mut().for_each(|v| *v = 7.0 + 1e-3 * *v); // clustered values, far from 0
            }
            let q: Vec<f64> = (0..d)
                .map(|_| rng.gauss() * if t % 7 == 0 { 100.0 } else { 1.0 })
                .collect();
            for eps in [0.0, 1e-6, 1e-3, 0.1, 10.0] {
                let i = ValueRadiusInput {
                    query: &q,
                    keys: &keys,
                    values: &values,
                    head_dim: d,
                    value_dim: dv,
                    score_scale: 0.3,
                    epsilon_out: eps,
                };
                check(&i);
            }
        }
    }

    #[test]
    fn witness_clustered_values_far_from_origin() {
        // Values ≈ 100 ± 0.01: uncentered bound (2·max‖v‖≈200) is useless,
        // centred bound (R≈0.01) lets the scan stop early.
        let mut rng = Rng(5);
        let (n, d) = (128, 8);
        let keys: Vec<f64> = (0..n * d).map(|_| rng.gauss()).collect();
        let values: Vec<f64> = (0..n).map(|_| 100.0 + 0.01 * rng.gauss()).collect();
        let q: Vec<f64> = (0..d).map(|_| rng.gauss()).collect();
        let i = ValueRadiusInput {
            query: &q,
            keys: &keys,
            values: &values,
            head_dim: d,
            value_dim: 1,
            score_scale: 0.8,
            epsilon_out: 1e-2,
        };
        let r = check(&i);
        assert!(r[1] < r[0], "centred {} not < uncentred {}", r[1], r[0]);
        assert!(r[2] <= r[1]);
    }

    /// Cost table (run with `--nocapture --ignored`): mean fraction of rows read.
    #[test]
    #[ignore = "research table printer"]
    fn print_cost_table() {
        for (sigma_v, spread, label) in [
            (1.0, 0.0, "zero-mean iid values"),
            (0.1, 5.0, "clustered values offset 5"),
            (0.01, 50.0, "tight cluster offset 50"),
        ] {
            for (norm_sigma, eps_rel) in [(0.0, 1e-2), (1.0, 1e-2), (1.0, 1e-1), (1.5, 1e-2)] {
                let mut tot = [0.0_f64; 3];
                let draws = 40;
                for s in 0..draws {
                    let mut rng = Rng(1000 + s);
                    let (n, d) = (256, 16);
                    let mut keys = vec![0.0; n * d];
                    for r in 0..n {
                        let sc = (norm_sigma * rng.gauss()).exp();
                        for c in 0..d {
                            keys[r * d + c] = rng.gauss() * sc;
                        }
                    }
                    let values: Vec<f64> =
                        (0..n * 4).map(|_| spread + sigma_v * rng.gauss()).collect();
                    let q: Vec<f64> = (0..d).map(|_| rng.gauss()).collect();
                    let i = ValueRadiusInput {
                        query: &q,
                        keys: &keys,
                        values: &values,
                        head_dim: d,
                        value_dim: 4,
                        score_scale: 0.5 / (d as f64).sqrt(),
                        epsilon_out: eps_rel * sigma_v,
                    };
                    let r = check(&i);
                    for k in 0..3 {
                        tot[k] += r[k] as f64 / n as f64 / draws as f64;
                    }
                }
                println!(
                    "{label} | norm_sigma {norm_sigma} | eps {eps_rel}*sigma_v | unc {:.3} cen {:.3} suf {:.3}",
                    tot[0], tot[1], tot[2]
                );
            }
        }
    }
}
