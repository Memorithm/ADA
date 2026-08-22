#!/usr/bin/env bash
set -u
set -o pipefail

main() {
    ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
    cd "${ROOT}" || return 2

    CORE="${ADA_CPU_CORE:-13}"
    REPEATS="${ADA_REPEATS:-5}"
    ROUNDS="${ADA_E2_ROUNDS:-21}"
    TARGET_SCALARS="${ADA_E2_TARGET_SCALARS:-4000000}"

    RUNNER="target/release/examples/e2_three_level_v_access"
    ANALYZER="tools/analyze_a2_e2.py"

    if ! [[ "${CORE}" =~ ^[0-9]+$ ]]; then
        echo "error: ADA_CPU_CORE must be a non-negative integer" >&2
        return 2
    fi

    if ! [[ "${REPEATS}" =~ ^[1-9][0-9]*$ ]]; then
        echo "error: ADA_REPEATS must be a positive integer" >&2
        return 2
    fi

    if ! [[ "${ROUNDS}" =~ ^[1-9][0-9]*$ ]]; then
        echo "error: ADA_E2_ROUNDS must be a positive integer" >&2
        return 2
    fi

    if ! [[ "${TARGET_SCALARS}" =~ ^[1-9][0-9]*$ ]]; then
        echo "error: ADA_E2_TARGET_SCALARS must be positive" >&2
        return 2
    fi

    for command in \
        git cargo rustc taskset nvpmodel jetson_clocks \
        lscpu sha256sum seq ps head python3 date uname
    do
        if ! command -v "${command}" >/dev/null 2>&1; then
            echo "error: required command missing: ${command}" >&2
            return 2
        fi
    done

    if [[ -n "$(git status --porcelain -uall)" ]]; then
        echo "error: A2-E2 qualification requires a clean Git tree" >&2
        git status --short -uall >&2
        return 3
    fi

    POWER_MODE="$(nvpmodel -q 2>&1 || true)"

    if ! grep -q 'NV Power Mode: MAXN' <<<"${POWER_MODE}"; then
        echo "error: A2-E2 requires MAXN" >&2
        printf '%s\n' "${POWER_MODE}" >&2
        return 3
    fi

    CPU_DIR="/sys/devices/system/cpu/cpu${CORE}/cpufreq"

    if [[ ! -d "${CPU_DIR}" ]]; then
        echo "error: no cpufreq directory for CPU ${CORE}" >&2
        return 2
    fi

    MIN_KHZ="$(cat "${CPU_DIR}/scaling_min_freq")"
    MAX_KHZ="$(cat "${CPU_DIR}/scaling_max_freq")"
    CUR_KHZ="$(cat "${CPU_DIR}/scaling_cur_freq")"
    GOVERNOR="$(cat "${CPU_DIR}/scaling_governor")"

    if [[ "${MIN_KHZ}" != "${MAX_KHZ}" ]]; then
        echo "error: A2-E2 requires fixed CPU frequency" >&2
        echo "core=${CORE} governor=${GOVERNOR} min=${MIN_KHZ} current=${CUR_KHZ} max=${MAX_KHZ}" >&2
        return 3
    fi

    if ! taskset -c "${CORE}" true >/dev/null 2>&1; then
        echo "error: cannot pin CPU ${CORE}" >&2
        return 3
    fi

    echo "=== A2-E2 QUALIFICATION GATES ==="

    cargo fmt --all -- --check || return 4

    cargo clippy \
        --workspace \
        --all-targets \
        -- -D warnings \
        || return 4

    cargo test --workspace || return 4

    cargo build \
        --release \
        -p ada-a2-k-first-v-late \
        --example e2_three_level_v_access \
        || return 4

    python3 -m py_compile "${ANALYZER}" || return 4

    bash -n "$0" || return 4

    SHA="$(git rev-parse HEAD)"
    STAMP="$(date -u +%Y%m%dT%H%M%SZ)"

    OUT_DIR="evidence/a2-k-first-v-late/e2-thor-three-level-${SHA:0:12}-${STAMP}"

    mkdir -p "${OUT_DIR}" || return 2

    ENV_FILE="${OUT_DIR}/environment.txt"
    ANALYSIS_FILE="${OUT_DIR}/analysis.txt"
    HASH_FILE="${OUT_DIR}/SHA256SUMS.txt"

    thermal_snapshot() {
        for zone in /sys/class/thermal/thermal_zone*
        do
            [[ -r "${zone}/type" && -r "${zone}/temp" ]] || continue

            printf '%s=%s\n' \
                "$(cat "${zone}/type")" \
                "$(cat "${zone}/temp")"
        done
    }

    {
        echo "experiment=ADA-A2-E2-three-level-physical-v-access"
        echo "ada_sha=${SHA}"
        echo "utc=${STAMP}"
        echo "cpu_core=${CORE}"
        echo "process_repeats=${REPEATS}"
        echo "benchmark_rounds=${ROUNDS}"
        echo "target_scalars=${TARGET_SCALARS}"
        echo "pmu_counters=false"
        echo "working_tree_before=clean"
        echo

        echo "=== SYSTEM ==="
        uname -a
        echo
        cat /etc/os-release
        echo
        rustc --version
        cargo --version
        echo

        echo "=== CPU ==="
        lscpu
        echo
        echo "pinned_core=${CORE}"
        echo "governor=${GOVERNOR}"
        echo "min_khz=${MIN_KHZ}"
        echo "current_khz_before=${CUR_KHZ}"
        echo "max_khz=${MAX_KHZ}"
        echo "documented_l2_per_core_bytes=1048576"
        echo "documented_l3_shared_bytes=16777216"
        echo

        echo "=== NVPModel ==="
        printf '%s\n' "${POWER_MODE}"
        echo

        echo "=== JETSON CLOCKS ==="
        jetson_clocks --show 2>&1 || true
        echo

        echo "=== THERMALS BEFORE ==="
        thermal_snapshot
        echo

        echo "=== PROCESS SNAPSHOT BEFORE ==="
        ps -eo pid,psr,pcpu,pmem,comm \
            --sort=-pcpu \
            | head -n 20 \
            || true
    } > "${ENV_FILE}"

    RUN_LOGS=()

    echo "=== A2-E2 INDEPENDENT PROCESS RUNS ==="

    for run in $(seq 1 "${REPEATS}")
    do
        RUN_FILE="$(
            printf '%s/run-%02d.txt' \
                "${OUT_DIR}" \
                "${run}"
        )"

        echo "process_run=${run}/${REPEATS}"
        echo "run_file=${RUN_FILE}"

        taskset -c "${CORE}" \
            env \
            ADA_E2_ROUNDS="${ROUNDS}" \
            ADA_E2_TARGET_SCALARS="${TARGET_SCALARS}" \
            "${RUNNER}" \
            > "${RUN_FILE}" 2>&1

        RUN_RC=$?

        if [[ "${RUN_RC}" -ne 0 ]]; then
            echo "error: benchmark run ${run} failed rc=${RUN_RC}" >&2
            return 5
        fi

        RESULT_COUNT="$(
            grep -c '^result,' "${RUN_FILE}" || true
        )"

        COMPLETE_COUNT="$(
            grep -c '^survey_status=complete$' "${RUN_FILE}" || true
        )"

        echo "result_count=${RESULT_COUNT}"
        echo "complete_count=${COMPLETE_COUNT}"

        if [[ "${RESULT_COUNT}" -ne 306 ]]; then
            echo "error: run ${run} expected 306 results, found ${RESULT_COUNT}" >&2
            return 5
        fi

        if [[ "${COMPLETE_COUNT}" -ne 1 ]]; then
            echo "error: run ${run} is incomplete" >&2
            return 5
        fi

        RUN_LOGS+=("${RUN_FILE}")

        sleep 1
    done

    echo
    echo "=== AGGREGATE ANALYSIS ==="

    python3 "${ANALYZER}" "${RUN_LOGS[@]}" \
        | tee "${ANALYSIS_FILE}"

    ANALYZE_RC="${PIPESTATUS[0]}"

    if [[ "${ANALYZE_RC}" -ne 0 ]]; then
        echo "error: analyzer failed rc=${ANALYZE_RC}" >&2
        return 6
    fi

    {
        echo
        echo "=== PINNED CORE AFTER ==="
        echo "current_khz_after=$(cat "${CPU_DIR}/scaling_cur_freq")"
        echo

        echo "=== THERMALS AFTER ==="
        thermal_snapshot
    } >> "${ENV_FILE}"

    echo
    echo "=== EVIDENCE HASHES ==="

    (
        cd "${OUT_DIR}" || return 2

        sha256sum \
            environment.txt \
            run-*.txt \
            analysis.txt \
            > SHA256SUMS.txt

        cat SHA256SUMS.txt

        echo
        echo -n "manifest_sha256="
        sha256sum SHA256SUMS.txt | awk '{print $1}'
    )

    echo
    echo "EVIDENCE_DIR=${OUT_DIR}"

    return 0
}

main "$@"
