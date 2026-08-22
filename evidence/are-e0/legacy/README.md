# Inherited Ox Alpha ARE-E0 archives

These two schema-v1 artifacts are preserved as negative historical evidence;
they were not discarded or rewritten during takeover.

| File | Canonical budget | Recorded result | File SHA-256 |
| --- | ---: | --- | --- |
| `ox-alpha-mode-a-budget-200000.json` | 200,000 | `AllFalsified` | `aac1653e9af26422f72f2aa2383700f7e5c3268fdb2e3b3e20d958dc13f4cf5b` |
| `ox-alpha-mode-a-budget-240000.json` | 240,000 | `AllFalsified` | `68d86a570189138b9f4a89c25d405ade82a7eebb8ddb1f3c9a1c3c38c4296561` |

They used the inherited “Mode A” interface, which supplied oracle-derived
`m_new` as a proposal input and searched only the normalizer output. That is a
target-specific leakage risk and does not satisfy the final ARE-E0 challenge.
Their summary-only archives also predate the schema-v2 per-proposal ledger and
exact-bit JSON transport. They must not be cited as the final discovery run or
compared as replay-equivalent to schema v2.

The artifacts remain useful for documenting the abandoned search formulation,
its bounded negative outcomes, and why takeover tightened the interface to
`[m_old, l_old, score] -> [m_new, l_new]`.

