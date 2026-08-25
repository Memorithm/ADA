#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
from collections import Counter
from pathlib import Path


EXPECTED_GROUP_LINES = 3072
EXPECTED_GROUPS_PER_ALPHA = 1536
EXPECTED_HEAD_ALPHA_CASES = 6144

PROBABILITY_TOLERANCE = 2.0e-10
TAU_TOLERANCE = 1.0e-10
OUTPUT_TOLERANCE = 2.0e-10

EXPECTED = {
    "1.5": {
        "k_union": 137_098,
        "support_union": 8_555,
        "no_residual": 6,
    },
    "2.0": {
        "k_union": 110_536,
        "support_union": 4_466,
        "no_residual": 13,
    },
}


def parse_fields(
    line: str,
) -> dict[str, str]:
    fields: dict[str, str] = {}

    for item in line.split(",")[1:]:
        key, separator, value = item.partition("=")

        if not separator or not key:
            raise ValueError(
                f"malformed field: {item!r}"
            )

        if key in fields:
            raise ValueError(
                f"duplicate field: {key}"
            )

        fields[key] = value

    return fields


def main() -> int:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "log",
        type=Path,
    )

    args = parser.parse_args()

    raw = args.log.read_bytes()
    text = raw.decode("utf-8")

    groups: list[
        dict[str, str]
    ] = []

    aggregates: list[
        dict[str, str]
    ] = []

    scalars: dict[
        str,
        str,
    ] = {}

    for line in text.splitlines():
        if line.startswith("group,"):
            groups.append(
                parse_fields(line)
            )

        elif line.startswith(
            "aggregate,"
        ):
            aggregates.append(
                parse_fields(line)
            )

        elif "=" in line:
            key, value = line.split(
                "=",
                1,
            )

            if key in {
                "group_alpha_case_count",
                "head_alpha_case_count",
                "e3a_accounting_reproduced",
                "numerical_output_parity_ok",
                "join_contract_ok",
                "survey_status",
            }:
                scalars[key] = value

    print(
        "analyzer="
        "ada_a2_e3b_natural_v_gqa_v1"
    )

    print(
        "log_sha256="
        f"{hashlib.sha256(raw).hexdigest()}"
    )

    print(
        f"group_count={len(groups)}"
    )

    print(
        f"aggregate_count={len(aggregates)}"
    )

    structural_ok = (
        len(groups)
        == EXPECTED_GROUP_LINES
        and len(aggregates) == 2
        and scalars.get(
            "group_alpha_case_count"
        ) == "3072"
        and scalars.get(
            "head_alpha_case_count"
        ) == str(
            EXPECTED_HEAD_ALPHA_CASES
        )
        and scalars.get(
            "survey_status"
        ) == "complete"
    )

    print(
        f"structural_contract_ok={structural_ok}"
    )

    invariant_failures = 0
    numerical_failures = 0
    exact_a2_v_failures = 0

    for row in groups:
        k_union = int(
            row["k_union"]
        )

        support_union = int(
            row["support_union"]
        )

        if support_union > k_union:
            invariant_failures += 1

        probability_difference = float(
            row[
                "max_probability_difference"
            ]
        )

        tau_difference = float(
            row[
                "max_tau_difference"
            ]
        )

        full_k = float(
            row[
                "max_full_vs_k_linf"
            ]
        )

        full_support = float(
            row[
                "max_full_vs_support_linf"
            ]
        )

        k_support = float(
            row[
                "max_k_vs_support_linf"
            ]
        )

        if (
            probability_difference
            > PROBABILITY_TOLERANCE
            or tau_difference
            > TAU_TOLERANCE
            or full_k
            > OUTPUT_TOLERANCE
            or full_support
            > OUTPUT_TOLERANCE
            or k_support
            > OUTPUT_TOLERANCE
        ):
            numerical_failures += 1

        # This is the direct E3b A2 property:
        # once the same exact sparse distribution is used,
        # omitting every A5-loaded row outside final support
        # must not change O = sum_i p_i V_i at all.
        if k_support != 0.0:
            exact_a2_v_failures += 1

    print(
        "support_union_outside_k_union="
        f"{invariant_failures}"
    )

    print(
        "numerical_tolerance_failures="
        f"{numerical_failures}"
    )

    print(
        "exact_kloaded_vs_support_v_failures="
        f"{exact_a2_v_failures}"
    )

    aggregate_by_alpha = {
        row["alpha"]: row
        for row in aggregates
    }

    accounting_ok = True
    aggregate_output_ok = True

    for alpha in (
        "1.5",
        "2.0",
    ):
        rows = [
            row
            for row in groups
            if row["alpha"] == alpha
        ]

        expected = EXPECTED[alpha]

        k_union = sum(
            int(row["k_union"])
            for row in rows
        )

        support_union = sum(
            int(row["support_union"])
            for row in rows
        )

        no_residual = sum(
            int(row["k_union"])
            == int(
                row["support_union"]
            )
            for row in rows
        )

        max_probability = max(
            float(
                row[
                    "max_probability_difference"
                ]
            )
            for row in rows
        )

        max_tau = max(
            float(
                row[
                    "max_tau_difference"
                ]
            )
            for row in rows
        )

        max_full_k = max(
            float(
                row[
                    "max_full_vs_k_linf"
                ]
            )
            for row in rows
        )

        max_full_support = max(
            float(
                row[
                    "max_full_vs_support_linf"
                ]
            )
            for row in rows
        )

        max_k_support = max(
            float(
                row[
                    "max_k_vs_support_linf"
                ]
            )
            for row in rows
        )

        layer_counts = Counter(
            int(row["layer"])
            for row in rows
        )

        accounting_match = (
            len(rows)
            == EXPECTED_GROUPS_PER_ALPHA
            and k_union
            == expected["k_union"]
            and support_union
            == expected["support_union"]
            and no_residual
            == expected["no_residual"]
        )

        accounting_ok &= (
            accounting_match
        )

        output_ok = (
            max_probability
            <= PROBABILITY_TOLERANCE
            and max_tau
            <= TAU_TOLERANCE
            and max_full_k
            <= OUTPUT_TOLERANCE
            and max_full_support
            <= OUTPUT_TOLERANCE
            and max_k_support == 0.0
        )

        aggregate_output_ok &= (
            output_ok
        )

        residual = (
            1.0
            - support_union / k_union
        )

        print()
        print(f"alpha={alpha}")
        print(
            f"group_cases={len(rows)}"
        )
        print(
            f"k_union={k_union}"
        )
        print(
            f"support_union={support_union}"
        )
        print(
            "weighted_a2_v_avoidance_after_k="
            f"{residual:.9f}"
        )
        print(
            "groups_without_residual_a2="
            f"{no_residual}"
        )
        print(
            "layer_case_counts="
            f"{dict(sorted(layer_counts.items()))}"
        )
        print(
            "max_probability_difference="
            f"{max_probability:.17e}"
        )
        print(
            "max_tau_difference="
            f"{max_tau:.17e}"
        )
        print(
            "max_full_vs_k_linf="
            f"{max_full_k:.17e}"
        )
        print(
            "max_full_vs_support_linf="
            f"{max_full_support:.17e}"
        )
        print(
            "max_k_vs_support_linf="
            f"{max_k_support:.17e}"
        )
        print(
            "e3a_accounting_match="
            f"{accounting_match}"
        )
        print(
            "output_contract_ok="
            f"{output_ok}"
        )

        aggregate = (
            aggregate_by_alpha
            .get(alpha)
        )

        if aggregate is None:
            aggregate_output_ok = False
            accounting_ok = False
            print(
                "aggregate_line_present=False"
            )
        else:
            aggregate_line_ok = (
                int(
                    aggregate[
                        "k_union"
                    ]
                )
                == k_union
                and int(
                    aggregate[
                        "support_union"
                    ]
                )
                == support_union
                and aggregate[
                    "e3a_accounting_match"
                ]
                == "true"
                and aggregate[
                    "output_parity_ok"
                ]
                == "true"
            )

            print(
                "aggregate_line_present=True"
            )
            print(
                "aggregate_line_consistent="
                f"{aggregate_line_ok}"
            )

            accounting_ok &= (
                aggregate_line_ok
            )

    runner_contract_ok = (
        scalars.get(
            "e3a_accounting_reproduced"
        )
        == "true"
        and scalars.get(
            "numerical_output_parity_ok"
        )
        == "true"
        and scalars.get(
            "join_contract_ok"
        )
        == "true"
    )

    qualification_ok = (
        structural_ok
        and invariant_failures == 0
        and numerical_failures == 0
        and exact_a2_v_failures == 0
        and accounting_ok
        and aggregate_output_ok
        and runner_contract_ok
    )

    print()
    print(
        "runner_contract_ok="
        f"{runner_contract_ok}"
    )

    print(
        "qualification_contract_ok="
        f"{qualification_ok}"
    )

    print(
        "classification="
        "A2-E3B-NATURAL-GQA-V-OUTPUT-"
        "CORRECTNESS-QUALIFIED"
    )

    print(
        "physical_v_traffic_measured=false"
    )

    print(
        "analysis_status=complete"
    )

    return (
        0
        if qualification_ok
        else 7
    )


if __name__ == "__main__":
    raise SystemExit(main())
