#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import math
import statistics
from collections import Counter
from pathlib import Path


EXPECTED_GROUP_LINES = 3072
EXPECTED_GROUPS_PER_ALPHA = 1536
EXPECTED_AGGREGATE_LINES = 32

PROBABILITY_TOLERANCE = 2.0e-10
TAU_TOLERANCE = 1.0e-10


def parse_fields(line: str) -> dict[str, str]:
    fields: dict[str, str] = {}

    for field in line.split(",")[1:]:
        key, separator, value = field.partition("=")

        if not separator or not key or not value:
            raise ValueError(
                f"malformed field in line: {field!r}"
            )

        if key in fields:
            raise ValueError(
                f"duplicate field {key!r}"
            )

        fields[key] = value

    return fields


def quantile(
    values: list[float],
    probability: float,
) -> float:
    ordered = sorted(values)

    if not ordered:
        return math.nan

    if len(ordered) == 1:
        return ordered[0]

    position = probability * (len(ordered) - 1)

    lower = math.floor(position)
    upper = math.ceil(position)

    if lower == upper:
        return ordered[lower]

    fraction = position - lower

    return (
        ordered[lower] * (1.0 - fraction)
        + ordered[upper] * fraction
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze ADA-A2 E3a natural GQA "
            "unique-row accounting replay."
        )
    )

    parser.add_argument(
        "log",
        type=Path,
    )

    args = parser.parse_args()

    raw = args.log.read_bytes()
    text = raw.decode("utf-8")

    groups: list[dict[str, str]] = []
    aggregates: list[dict[str, str]] = []
    complete_count = 0

    for line_number, line in enumerate(
        text.splitlines(),
        1,
    ):
        if line.startswith("group,"):
            row = parse_fields(line)
            row["_line"] = str(line_number)
            groups.append(row)

        elif line.startswith("aggregate,"):
            row = parse_fields(line)
            row["_line"] = str(line_number)
            aggregates.append(row)

        elif line == "survey_status=complete":
            complete_count += 1

    print("analyzer=ada_a2_e3a_gqa_union_v1")
    print(
        "log_sha256="
        f"{hashlib.sha256(raw).hexdigest()}"
    )
    print(f"group_count={len(groups)}")
    print(f"aggregate_count={len(aggregates)}")
    print(f"complete_count={complete_count}")

    structural_ok = (
        len(groups) == EXPECTED_GROUP_LINES
        and len(aggregates) == EXPECTED_AGGREGATE_LINES
        and complete_count == 1
    )

    print(f"structural_contract_ok={structural_ok}")

    invariant_failures = 0
    numerical_failures = 0

    for row in groups:
        q_k_sum = int(row["q_k_sum"])
        q_support_sum = int(row["q_support_sum"])

        k_union = int(row["k_union"])
        support_union = int(row["support_union"])

        k_intersection = int(
            row["k_intersection"]
        )

        support_intersection = int(
            row["support_intersection"]
        )

        if support_union > k_union:
            invariant_failures += 1

        if (
            q_k_sum
            != k_union + k_intersection
        ):
            invariant_failures += 1

        if (
            q_support_sum
            != support_union
            + support_intersection
        ):
            invariant_failures += 1

        probability_difference = float(
            row["probability_difference"]
        )

        tau_difference = float(
            row["tau_difference"]
        )

        if (
            probability_difference
            > PROBABILITY_TOLERANCE
            or tau_difference
            > TAU_TOLERANCE
        ):
            numerical_failures += 1

    print(
        "cardinality_invariant_failures="
        f"{invariant_failures}"
    )

    print(
        "numerical_parity_failures="
        f"{numerical_failures}"
    )

    for alpha in ("1.5", "2.0"):
        rows = [
            row
            for row in groups
            if row["alpha"] == alpha
        ]

        print()
        print(f"alpha={alpha}")
        print(f"cases={len(rows)}")

        visible_rows = sum(
            int(row["key_count"])
            for row in rows
        )

        q_k_sum = sum(
            int(row["q_k_sum"])
            for row in rows
        )

        q_support_sum = sum(
            int(row["q_support_sum"])
            for row in rows
        )

        k_union = sum(
            int(row["k_union"])
            for row in rows
        )

        support_union = sum(
            int(row["support_union"])
            for row in rows
        )

        k_intersection = sum(
            int(row["k_intersection"])
            for row in rows
        )

        support_intersection = sum(
            int(row["support_intersection"])
            for row in rows
        )

        naive_after_k = (
            1.0
            - q_support_sum / q_k_sum
        )

        gqa_after_k = (
            1.0
            - support_union / k_union
        )

        gqa_effect_pp = (
            gqa_after_k - naive_after_k
        ) * 100.0

        k_dedup_saving = (
            1.0 - k_union / q_k_sum
        )

        support_dedup_saving = (
            1.0
            - support_union / q_support_sum
        )

        total_v_avoidance = (
            1.0
            - support_union / visible_rows
        )

        print(f"visible_rows={visible_rows}")
        print(f"q_k_sum={q_k_sum}")
        print(
            f"q_support_sum={q_support_sum}"
        )
        print(f"k_union={k_union}")
        print(
            f"support_union={support_union}"
        )
        print(
            f"k_intersection={k_intersection}"
        )
        print(
            "support_intersection="
            f"{support_intersection}"
        )

        print(
            "naive_per_q_a2_after_k="
            f"{naive_after_k:.9f}"
        )

        print(
            "gqa_unique_a2_after_k="
            f"{gqa_after_k:.9f}"
        )

        print(
            "gqa_effect_percentage_points="
            f"{gqa_effect_pp:+.6f}"
        )

        print(
            "total_unique_v_avoidance="
            f"{total_v_avoidance:.9f}"
        )

        print(
            "k_unique_row_dedup_saving="
            f"{k_dedup_saving:.9f}"
        )

        print(
            "support_unique_row_dedup_saving="
            f"{support_dedup_saving:.9f}"
        )

        values = [
            float(
                row[
                    "a2_v_avoidance_after_k"
                ]
            )
            for row in rows
        ]

        print(
            "a2_avoidance_min="
            f"{min(values):.6f}"
        )

        print(
            "a2_avoidance_median="
            f"{statistics.median(values):.6f}"
        )

        print(
            "a2_avoidance_q01="
            f"{quantile(values, 0.01):.6f}"
        )

        print(
            "a2_avoidance_q05="
            f"{quantile(values, 0.05):.6f}"
        )

        print(
            "a2_avoidance_q95="
            f"{quantile(values, 0.95):.6f}"
        )

        print(
            "a2_avoidance_q99="
            f"{quantile(values, 0.99):.6f}"
        )

        no_residual = [
            row
            for row in rows
            if int(row["k_union"])
            == int(row["support_union"])
        ]

        print(
            "groups_without_residual_a2="
            f"{len(no_residual)}"
        )

        print(
            "groups_without_residual_fraction="
            f"{len(no_residual) / len(rows):.9f}"
        )

        by_layer = Counter(
            int(row["layer"])
            for row in no_residual
        )

        by_position = Counter(
            int(row["query_position"])
            for row in no_residual
        )

        by_kv_head = Counter(
            int(row["kv_head"])
            for row in no_residual
        )

        by_k_union = Counter(
            int(row["k_union"])
            for row in no_residual
        )

        print(
            "exceptions_by_layer="
            f"{dict(sorted(by_layer.items()))}"
        )

        print(
            "exceptions_by_position="
            f"{dict(sorted(by_position.items()))}"
        )

        print(
            "exceptions_by_kv_head="
            f"{dict(sorted(by_kv_head.items()))}"
        )

        print(
            "exceptions_by_k_union="
            f"{dict(sorted(by_k_union.items()))}"
        )

        for index, row in enumerate(
            no_residual,
            1,
        ):
            print(
                "no_residual,"
                f"index={index},"
                f"sample={row['sample_fingerprint']},"
                f"layer={row['layer']},"
                f"kv_head={row['kv_head']},"
                f"q0={row['q0']},"
                f"q1={row['q1']},"
                f"position={row['query_position']},"
                f"key_count={row['key_count']},"
                f"k_union={row['k_union']},"
                f"support_union={row['support_union']}"
            )

        if len(rows) != EXPECTED_GROUPS_PER_ALPHA:
            structural_ok = False

    qualification_contract_ok = (
        structural_ok
        and invariant_failures == 0
        and numerical_failures == 0
    )

    print()
    print(
        "qualification_contract_ok="
        f"{qualification_contract_ok}"
    )

    print(
        "classification="
        "A2-E3A-NATURAL-GQA-UNIQUE-V-ROW-"
        "ACCOUNTING-QUALIFIED"
    )

    print("analysis_status=complete")

    return 0 if qualification_contract_ok else 7


if __name__ == "__main__":
    raise SystemExit(main())
