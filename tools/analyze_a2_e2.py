#!/usr/bin/env python3
"""Analyzer for ADA-A2 E2 three-level physical V-access logs.

Standard library only. Parses one or more raw E2 logs produced by
crates/ada-a2-k-first-v-late/examples/e2_three_level_v_access.rs, validates
them, and emits human plus machine-readable summaries of

    G_A5           = T_FullDense / T_KLoaded     (full_to_k_speedup_ppm / 1e6)
    G_A2_after_A5  = T_KLoaded   / T_Support      (k_to_support_speedup_ppm / 1e6)
    G_total        = T_FullDense / T_Support      (full_to_support_speedup_ppm / 1e6)

The isolated A2 physical criterion is G_A2_after_A5 > 1. G_total must never
be attributed to A2 alone.

Exit codes:
    0  all checks passed
    2  usage error
    3  malformed result record(s)
    4  survey completion missing
    5  correctness tolerance violated
"""

from __future__ import annotations

import argparse
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

REQUIRED_FIELDS = (
    "mode",
    "tokens",
    "value_dim",
    "value_bytes",
    "cache_region",
    "pattern",
    "requested_k_density_ppm",
    "realized_k_density_ppm",
    "k_count",
    "requested_support_density_ppm",
    "realized_support_density_ppm",
    "support_count",
    "full_iterations",
    "k_iterations",
    "support_iterations",
    "full_median_ns",
    "k_median_ns",
    "support_median_ns",
    "full_p95_ns",
    "k_p95_ns",
    "support_p95_ns",
    "full_mad_ns",
    "k_mad_ns",
    "support_mad_ns",
    "full_to_k_speedup_ppm",
    "k_to_support_speedup_ppm",
    "full_to_support_speedup_ppm",
    "max_abs_full_k_difference",
    "max_abs_k_support_difference",
)

FLOAT_FIELDS = {
    "max_abs_full_k_difference",
    "max_abs_k_support_difference",
}

INT_FIELDS = (
    set(REQUIRED_FIELDS)
    - FLOAT_FIELDS
    - {"mode", "cache_region", "pattern"}
)

VALID_MODES = {"warm", "evicted"}
VALID_PATTERNS = {"prefix", "spread"}
VALID_REGIONS = {"l2_capacity", "l3_capacity", "beyond_l3"}

ALPHA2_LIKE = (225_564, 8_344)
ALPHA15_LIKE = (291_461, 15_370)

DEFAULT_TOLERANCE = 2.0e-5

CONTROL_MEDIAN_SUSPECT = 1.05
CONTROL_MAX_SUSPECT = 1.10


@dataclass
class Record:
    source: str
    line_no: int
    fields: dict


@dataclass
class ParseOutcome:
    records: list
    malformed: list
    complete_files: set


def parse_log(text: str, source: str):
    records = []
    malformed = []
    complete = False

    for line_no, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()

        if not line:
            continue

        if line == "survey_status=complete":
            complete = True
            continue

        if not line.startswith("result,"):
            continue

        parts = line.split(",")
        fields = {}
        bad = False

        for part in parts[1:]:
            if "=" not in part:
                bad = True
                break
            key, _, value = part.partition("=")
            if not key or not value or key in fields:
                bad = True
                break
            fields[key] = value

        missing = [name for name in REQUIRED_FIELDS if name not in fields]

        if bad or missing:
            reason = "missing=" + ",".join(missing) if missing else "unparseable"
            malformed.append((source, line_no, reason))
            continue

        coerced = True

        for name in INT_FIELDS:
            try:
                fields[name] = int(fields[name])
            except ValueError:
                malformed.append((source, line_no, f"non-integer {name}"))
                coerced = False
                break

        if not coerced:
            continue

        for name in FLOAT_FIELDS:
            try:
                fields[name] = float(fields[name])
            except ValueError:
                malformed.append((source, line_no, f"non-float {name}"))
                coerced = False
                break

        if not coerced:
            continue

        if fields["mode"] not in VALID_MODES:
            malformed.append((source, line_no, f"unknown mode {fields['mode']}"))
            continue

        if fields["pattern"] not in VALID_PATTERNS:
            malformed.append((source, line_no, f"unknown pattern {fields['pattern']}"))
            continue

        if fields["cache_region"] not in VALID_REGIONS:
            malformed.append(
                (source, line_no, f"unknown cache_region {fields['cache_region']}")
            )
            continue

        records.append(Record(source, line_no, fields))

    return records, malformed, complete


def ratio(records, field):
    values = [rec.fields[field] / 1_000_000.0 for rec in records]

    if not values:
        return None

    return {
        "n": len(values),
        "min": min(values),
        "median": statistics.median(values),
        "max": max(values),
    }


def ident(rec):
    f = rec.fields

    return (
        f"{Path(rec.source).name}:{rec.line_no}"
        f" mode={f['mode']} tokens={f['tokens']} pattern={f['pattern']}"
        f" k={f['requested_k_density_ppm']} s={f['requested_support_density_ppm']}"
        f" k_count={f['k_count']} s_count={f['support_count']}"
    )


def fmt_ratio(stats):
    if stats is None:
        return "n/a"

    return (
        f"n={stats['n']} min={stats['min']:.4f}"
        f" median={stats['median']:.4f} max={stats['max']:.4f}"
    )


def subset(records, predicate):
    return [rec for rec in records if predicate(rec.fields)]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])

    parser.add_argument("logs", nargs="+", help="raw E2 log files ('-' for stdin)")

    parser.add_argument(
        "--tolerance",
        type=float,
        default=DEFAULT_TOLERANCE,
        help=f"max abs f32 output difference tolerance (default {DEFAULT_TOLERANCE:g})",
    )

    args = parser.parse_args(argv)

    records = []
    malformed = []
    complete_files = set()

    for path in args.logs:
        if path == "-":
            text = sys.stdin.read()
            source = "<stdin>"
        else:
            text = Path(path).read_text(encoding="utf-8", errors="replace")
            source = path

        recs, bad, complete = parse_log(text, source)

        records.extend(recs)
        malformed.extend(bad)

        if complete:
            complete_files.add(source)

    print(f"input_files={len(args.logs)}")
    print(f"complete_surveys={len(complete_files)}")

    for path in args.logs:
        source = "<stdin>" if path == "-" else path

        if source not in complete_files:
            print(f"survey_status_missing={source}")

    print(f"records_parsed={len(records)}")

    for entry in malformed:
        print(f"MALFORMED source={entry[0]} line={entry[1]} reason={entry[2]}")

    tolerance_violations = [
        rec
        for rec in records
        if rec.fields["max_abs_full_k_difference"] > args.tolerance
        or rec.fields["max_abs_k_support_difference"] > args.tolerance
    ]

    for rec in tolerance_violations:
        print(
            f"CORRECTNESS_VIOLATION {ident(rec)}"
            f" full_k={rec.fields['max_abs_full_k_difference']:.3e}"
            f" k_support={rec.fields['max_abs_k_support_difference']:.3e}"
        )

    print(f"correctness_violations={len(tolerance_violations)} tolerance={args.tolerance:g}")

    if not records:
        print("error: no valid result records parsed")
        return 3 if malformed else 4

    scopes = {"all": records}

    for mode in ("warm", "evicted"):
        selected = subset(records, lambda f, m=mode: f["mode"] == m)

        if selected:
            scopes[f"mode={mode}"] = selected

    for pattern in ("prefix", "spread"):
        selected = subset(records, lambda f, p=pattern: f["pattern"] == p)

        if selected:
            scopes[f"pattern={pattern}"] = selected

    anchors = {}

    for name, pair in (("alpha2_like", ALPHA2_LIKE), ("alpha1_5_like", ALPHA15_LIKE)):
        selected = subset(
            records,
            lambda f, p=pair: f["requested_k_density_ppm"] == p[0]
            and f["requested_support_density_ppm"] == p[1],
        )

        if selected:
            scopes[name] = selected
            anchors[name] = selected

    natural = [
        rec
        for rec in records
        if (rec.fields["requested_k_density_ppm"], rec.fields["requested_support_density_ppm"])
        in (ALPHA2_LIKE, ALPHA15_LIKE)
    ]

    if natural:
        scopes["natural_anchors"] = natural

    for region in ("l2_capacity", "l3_capacity", "beyond_l3"):
        selected = subset(records, lambda f, r=region: f["cache_region"] == r)

        if selected:
            scopes[f"region={region}"] = selected

    controls = subset(
        records,
        lambda f: f["requested_support_density_ppm"] == f["requested_k_density_ppm"],
    )

    if controls:
        scopes["control_support_eq_kloaded"] = controls

    k_full_controls = subset(
        controls,
        lambda f: f["requested_k_density_ppm"] == 1_000_000,
    )

    print()
    print("=== RATIO SUMMARIES ===")

    for scope, selected in scopes.items():
        print(f"[{scope}]")

        for label, field in (
            ("G_A5", "full_to_k_speedup_ppm"),
            ("G_A2_after_A5", "k_to_support_speedup_ppm"),
            ("G_total", "full_to_support_speedup_ppm"),
        ):
            stats = ratio(selected, field)

            print(f"summary,scope={scope},metric={label},{fmt_ratio(stats)}")

    non_wins = [
        rec for rec in records if rec.fields["k_to_support_speedup_ppm"] <= 1_000_000
    ]

    natural_non_wins = [
        rec for rec in natural if rec.fields["k_to_support_speedup_ppm"] <= 1_000_000
    ]

    print()
    print("=== A2-AFTER-A5 NON-WINS ===")
    print(f"a2_non_wins_total={len(non_wins)}")
    print(f"a2_non_wins_natural_anchors={len(natural_non_wins)}")

    for rec in non_wins:
        print(f"NON_WIN {ident(rec)} g_a2={rec.fields['k_to_support_speedup_ppm'] / 1e6:.4f}")

    worst_natural = None

    if natural:
        worst_natural = min(natural, key=lambda rec: rec.fields["k_to_support_speedup_ppm"])

        print()
        print("=== WORST NATURAL-ANCHOR G_A2_AFTER_A5 ===")
        print(f"WORST_NATURAL_ANCHOR {ident(worst_natural)}")
        print(
            "worst_natural_anchor_g_a2_after_a5="
            f"{worst_natural.fields['k_to_support_speedup_ppm'] / 1e6:.6f}"
        )

    print()
    print("=== SUPPORT=KLOADED CONTROLS ===")

    control_status = "no_data"

    if controls:
        control_g2 = [rec.fields["k_to_support_speedup_ppm"] / 1e6 for rec in controls]
        control_median = statistics.median(control_g2)
        control_max = max(control_g2)
        control_min = min(control_g2)

        print(
            f"summary,scope=control_support_eq_kloaded,g_a2_min={control_min:.6f},"
            f"g_a2_median={control_median:.6f},g_a2_max={control_max:.6f}"
        )
        print(f"control_records={len(controls)}")

        suspect = (
            control_median > CONTROL_MEDIAN_SUSPECT or control_max > CONTROL_MAX_SUSPECT
        )

        control_status = "suspect" if suspect else "ok"
        print(f"control_status={control_status}"
              f" (median threshold {CONTROL_MEDIAN_SUSPECT}, max threshold {CONTROL_MAX_SUSPECT})")

        for rec in controls:
            g2 = rec.fields["k_to_support_speedup_ppm"] / 1e6

            if g2 > CONTROL_MAX_SUSPECT:
                print(f"CONTROL_OUTLIER {ident(rec)} g_a2={g2:.4f}")

        if k_full_controls:
            g5_values = [rec.fields["full_to_k_speedup_ppm"] / 1e6 for rec in k_full_controls]

            print(
                "summary,scope=control_k100_support_eq_k,"
                f"g_a5_min={min(g5_values):.6f},"
                f"g_a5_median={statistics.median(g5_values):.6f},"
                f"g_a5_max={max(g5_values):.6f}"
            )

    print()
    print("=== VERDICT INPUTS ===")

    if natural:
        assert worst_natural is not None

        verdict = (
            "pass"
            if worst_natural.fields["k_to_support_speedup_ppm"] > 1_000_000
            else "fail"
        )

        print("natural_anchor_criterion=G_A2_after_A5>1 robustly")
        print(f"natural_anchor_verdict={verdict}")
    else:
        print("natural_anchor_verdict=no_data")

    print(f"control_verdict={control_status}")

    if malformed:
        return 3

    if len(complete_files) != len(args.logs):
        return 4

    if tolerance_violations:
        return 5

    return 0


if __name__ == "__main__":
    sys.exit(main())
