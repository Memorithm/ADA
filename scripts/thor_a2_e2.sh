#!/usr/bin/env bash
set -u
set -o pipefail

main() {
    ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
    cd "${ROOT}" || return 2

    CORE="${ADA_CPU_CORE:-13}"
    REPEATS="${ADA_REPEATS:-5}"
    ROUNDS="${ADA_E2_ROUNDS:-21}"
    EVICTED_ROUNDS="${ADA_E2_EVICTED_ROUNDS:-31}"
    TARGET_SCALARS="${ADA_E2_TARGET_SCALARS:-4000000}"

    if ! [[ "${CORE}" =~ ^[0-9]+$ ]]; then
        echo "error: ADA_CPU_CORE must be non-negative" >&2
        return 2
    fi

    if ! [[ "${REPEATS}" =~ ^[1-9][0-9]*$ ]]; then
        echo "error: ADA_REPEATS must be positive" >&2
        return 2
    fi

    for command in \
        git cargo rustc taskset nvpmodel \
        jetson_clocks lscpu sha256sum seq ps head
    do
        if ! command -v "${command}" >/dev/null 2>&1; then
            echo "error: required command missing: ${command}" >&2
            return 2
        fi
    done

    if [[ -n "$(git status --porcelain -uall)" ]]; then
        echo "error: A2-E2 qualification requires clean Git tree" >&2
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

    echo "=== CORRECTNESS GATES ==="

    cargo fmt --all -- --check || return 4

    cargo clippy \
        -p ada-a2-k-first-v-late \
        --example e2_three_level_v_access \
        -- -D warnings \
        || return 4

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

    SHA="$(git rev-parse HEAD)"
    STAMP="$(date -u +%Y%m%dT%H%M%SZ)"

    OUT_DIR="evidence/a2-k-first-v-late"
    OUT="${OUT_DIR}/thor-e2-three-level-v-access-${SHA:0:12}-${STAMP}.txt"

    mkdir -p "${OUT_DIR}" || return 2

    thermal_snapshot() {
        for zone in /sys/class/thermal/thermal_zone*
        do
            [[ -r "${zone}/type" && -r "${zone}/temp" ]] \
                || continue

            printf '%s=%s\n' \
                "$(cat "${zone}/type")" \
                "$(cat "${zone}/temp")"
        done
    }

    {
        echo "ADA experiment: ADA-A2 E2 Three-Level Physical V Access"
        echo "evidence_level=physical_cpu_microbench"
        echo "ada_sha=${SHA}"
        echo "utc=${STAMP}"
        echo "working_tree_before=clean"
        echo "cpu_core=${CORE}"
        echo "process_repeats=${REPEATS}"
        echo "benchmark_rounds=${ROUNDS}"
        echo "evicted_rounds=${EVICTED_ROUNDS}"
        echo "target_scalars=${TARGET_SCALARS}"
        echo "runner=e2_three_level_v_access"
        echo "decomposition=full_dense_vs_k_loaded_vs_support"
        echo "primary_metric=G_A2_after_A5=k_to_support_speedup_ppm/1e6"
        echo "pmu_counters=false"
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
        echo

        echo "=== A2-E2 PINNED RELEASE BENCH ==="

        for run in $(seq 1 "${REPEATS}")
        do
            echo "--- process_run=${run}/${REPEATS} ---"

            taskset -c "${CORE}" \
                env \
                ADA_E2_ROUNDS="${ROUNDS}" \
                ADA_E2_EVICTED_ROUNDS="${EVICTED_ROUNDS}" \
                ADA_E2_TARGET_SCALARS="${TARGET_SCALARS}" \
                target/release/examples/e2_three_level_v_access \
                || return 5

            echo
            sleep 1
        done

        echo "=== PINNED CORE AFTER ==="
        echo "current_khz_after=$(cat "${CPU_DIR}/scaling_cur_freq")"
        echo

        echo "=== THERMALS AFTER ==="
        thermal_snapshot

    } | tee "${OUT}"

    PIPE_RC="${PIPESTATUS[0]}"

    if [[ "${PIPE_RC}" -ne 0 ]]; then
        echo "error: evidence pipeline failed rc=${PIPE_RC}" >&2
        return "${PIPE_RC}"
    fi

    echo
    sha256sum "${OUT}"
    echo "EVIDENCE=${OUT}"

    return 0
}

main "$@"
