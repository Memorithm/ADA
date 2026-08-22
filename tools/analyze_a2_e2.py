#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

CORRECTNESS_TOLERANCE = 2.0e-5

INTEGER_FIELDS = {
    "tokens",
    "value_dim",
    "value_bytes",
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
}

FLOAT_FIELDS = {
    "max_abs_full_k_difference",
    "max_abs_k_support_difference",
}

REQUIRED_FIELDS = INTEGER_FIELDS | FLOAT_FIELDS | {
    "mode",
    "cache_region",
    "pattern",
}


@dataclass(frozen=True)
class InputLog:
    path: Path
    sha256: str
    records: list[dict[str, object]]
    complete_count: int


def parse_result_line(
    line: str,
    *,
    path: Path,
    line_number: int,
) -> dict[str, object]:
    fields = line.split(",")

    if not fields or fields[0] != "result":
        raise ValueError(
            f"{path}:{line_number}: malformed result prefix"
        )

    row: dict[str, object] = {}

    for field in fields[1:]:
        if "=" not in field:
            raise ValueError(
                f"{path}:{line_number}: "
                f"malformed field {field!r}"
            )

        key, value = field.split("=", 1)

        if not key:
            raise ValueError(
                f"{path}:{line_number}: empty key"
            )

        if key in row:
            raise ValueError(
                f"{path}:{line_number}: "
                f"duplicate key {key!r}"
            )

        row[key] = value

    missing = REQUIRED_FIELDS - row.keys()

    if missing:
        raise ValueError(
            f"{path}:{line_number}: "
            f"missing fields {sorted(missing)!r}"
        )

    for key in INTEGER_FIELDS:
        try:
            row[key] = int(str(row[key]))
        except ValueError as error:
            raise ValueError(
                f"{path}:{line_number}: "
                f"{key} is not an integer"
            ) from error

    for key in FLOAT_FIELDS:
        try:
            row[key] = float(str(row[key]))
        except ValueError as error:
            raise ValueError(
                f"{path}:{line_number}: "
                f"{key} is not a float"
            ) from error

    return row


def parse_log(path: Path) -> InputLog:
    raw = path.read_bytes()
    text = raw.decode("utf-8")

    records: list[dict[str, object]] = []
    complete_count = 0

    for line_number, line in enumerate(
        text.splitlines(),
        1,
    ):
        if line.startswith("result,"):
            records.append(
                parse_result_line(
                    line,
                    path=path,
                    line_number=line_number,
                )
            )

        elif line == "survey_status=complete":
            complete_count += 1

    if not records:
        raise ValueError(
            f"{path}: no result records"
        )

    if complete_count != 1:
        raise ValueError(
            f"{path}: expected exactly one "
            f"survey_status=complete, "
            f"found {complete_count}"
        )

    return InputLog(
        path=path,
        sha256=hashlib.sha256(raw).hexdigest(),
        records=records,
        complete_count=complete_count,
    )


def ppm_ratio(
    row: dict[str, object],
    field: str,
) -> float:
    return int(row[field]) / 1_000_000.0


def summarize(
    name: str,
    rows: list[dict[str, object]],
) -> None:
    print(f"summary={name}")

    if not rows:
        print("cases=0")
        return

    a5 = [
        ppm_ratio(
            row,
            "full_to_k_speedup_ppm",
        )
        for row in rows
    ]

    a2 = [
        ppm_ratio(
            row,
            "k_to_support_speedup_ppm",
        )
        for row in rows
    ]

    total = [
        ppm_ratio(
            row,
            "full_to_support_speedup_ppm",
        )
        for row in rows
    ]

    nonwins = sum(
        value <= 1.0
        for value in a2
    )

    worst = min(
        rows,
        key=lambda row:
            int(row["k_to_support_speedup_ppm"]),
    )

    print(f"cases={len(rows)}")

    print(
        "a5_min="
        f"{min(a5):.6f}"
    )

    print(
        "a5_median="
        f"{statistics.median(a5):.6f}"
    )

    print(
        "a2_after_a5_min="
        f"{min(a2):.6f}"
    )

    print(
        "a2_after_a5_median="
        f"{statistics.median(a2):.6f}"
    )

    print(
        "a2_after_a5_max="
        f"{max(a2):.6f}"
    )

    print(
        "total_median="
        f"{statistics.median(total):.6f}"
    )

    print(f"a2_nonwins={nonwins}")

    print(
        "worst="
        f"mode:{worst['mode']};"
        f"tokens:{worst['tokens']};"
        f"region:{worst['cache_region']};"
        f"pattern:{worst['pattern']};"
        f"k_ppm:{worst['requested_k_density_ppm']};"
        f"support_ppm:"
        f"{worst['requested_support_density_ppm']};"
        f"k_ns:{worst['k_median_ns']};"
        f"support_ns:{worst['support_median_ns']};"
        f"a2_after_a5:"
        f"{ppm_ratio(worst, 'k_to_support_speedup_ppm'):.6f}"
    )


def subset(
    rows: Iterable[dict[str, object]],
    *,
    k_ppm: int | None = None,
    support_ppm: int | None = None,
    mode: str | None = None,
    pattern: str | None = None,
    region: str | None = None,
) -> list[dict[str, object]]:
    result = []

    for row in rows:
        if (
            k_ppm is not None
            and row["requested_k_density_ppm"] != k_ppm
        ):
            continue

        if (
            support_ppm is not None
            and row["requested_support_density_ppm"]
            != support_ppm
        ):
            continue

        if (
            mode is not None
            and row["mode"] != mode
        ):
            continue

        if (
            pattern is not None
            and row["pattern"] != pattern
        ):
            continue

        if (
            region is not None
            and row["cache_region"] != region
        ):
            continue

        result.append(row)

    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze ADA-A2 E2 three-level "
            "physical V-access logs."
        )
    )

    parser.add_argument(
        "logs",
        nargs="+",
        type=Path,
    )

    args = parser.parse_args()

    parsed = [
        parse_log(path)
        for path in args.logs
    ]

    records = [
        row
        for log in parsed
        for row in log.records
    ]

    print(
        "analyzer="
        "ada_a2_e2_three_level_v1"
    )

    print(f"log_count={len(parsed)}")

    for index, log in enumerate(parsed):
        print(
            f"log_{index}_path={log.path}"
        )
        print(
            f"log_{index}_sha256={log.sha256}"
        )
        print(
            f"log_{index}_records="
            f"{len(log.records)}"
        )

    print(
        f"record_count={len(records)}"
    )

    max_full_k = max(
        float(
            row[
                "max_abs_full_k_difference"
            ]
        )
        for row in records
    )

    max_k_support = max(
        float(
            row[
                "max_abs_k_support_difference"
            ]
        )
        for row in records
    )

    print(
        "max_abs_full_k_difference="
        f"{max_full_k:.9e}"
    )

    print(
        "max_abs_k_support_difference="
        f"{max_k_support:.9e}"
    )

    correctness_ok = (
        max_full_k <= CORRECTNESS_TOLERANCE
        and max_k_support
        <= CORRECTNESS_TOLERANCE
    )

    print(
        f"correctness_ok={correctness_ok}"
    )

    alpha2 = subset(
        records,
        k_ppm=225_564,
        support_ppm=8_344,
    )

    alpha15 = subset(
        records,
        k_ppm=291_461,
        support_ppm=15_370,
    )

    summarize(
        "alpha2_like",
        alpha2,
    )

    summarize(
        "alpha15_like",
        alpha15,
    )

    for label, anchor in (
        ("alpha2", alpha2),
        ("alpha15", alpha15),
    ):
        for mode in ("warm", "evicted"):
            for pattern in (
                "prefix",
                "spread",
            ):
                summarize(
                    f"{label}_{mode}_{pattern}",
                    subset(
                        anchor,
                        mode=mode,
                        pattern=pattern,
                    ),
                )

    for region in (
        "l2_capacity",
        "l3_capacity",
        "beyond_l3",
    ):
        summarize(
            f"natural_{region}",
            subset(
                alpha2 + alpha15,
                region=region,
            ),
        )

    controls = [
        row
        for row in records
        if row["requested_k_density_ppm"]
        == row[
            "requested_support_density_ppm"
        ]
    ]

    summarize(
        "support_equals_k",
        controls,
    )

    natural = alpha2 + alpha15

    natural_nonwins = sum(
        int(
            row[
                "k_to_support_speedup_ppm"
            ]
        )
        <= 1_000_000
        for row in natural
    )

    worst_natural = min(
        natural,
        key=lambda row:
            int(
                row[
                    "k_to_support_speedup_ppm"
                ]
            ),
    )

    print(
        "natural_anchor_a2_nonwins="
        f"{natural_nonwins}"
    )

    print(
        "worst_natural_anchor_a2_after_a5="
        f"{ppm_ratio(worst_natural, 'k_to_support_speedup_ppm'):.6f}"
    )

    print(
        "worst_natural_anchor="
        f"mode:{worst_natural['mode']};"
        f"tokens:{worst_natural['tokens']};"
        f"region:{worst_natural['cache_region']};"
        f"pattern:{worst_natural['pattern']};"
        f"k_ppm:"
        f"{worst_natural['requested_k_density_ppm']};"
        f"support_ppm:"
        f"{worst_natural['requested_support_density_ppm']}"
    )

    qualification_candidate = (
        correctness_ok
        and natural_nonwins == 0
    )

    print(
        "qualification_candidate="
        f"{qualification_candidate}"
    )


if __name__ == "__main__":
    main()
