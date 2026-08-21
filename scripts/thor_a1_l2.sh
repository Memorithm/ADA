#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

CORE="${ADA_CPU_CORE:-13}"
REPEATS="${ADA_REPEATS:-5}"

if ! [[ "${CORE}" =~ ^[0-9]+$ ]]; then
    echo "error: ADA_CPU_CORE must be a non-negative integer" >&2
    exit 2
fi
if ! [[ "${REPEATS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "error: ADA_REPEATS must be a positive integer" >&2
    exit 2
fi

for command in git cargo rustc taskset nvpmodel jetson_clocks lscpu; do
    if ! command -v "${command}" >/dev/null 2>&1; then
        echo "error: required command not found: ${command}" >&2
        exit 2
    fi
done

CPU_DIR="/sys/devices/system/cpu/cpu${CORE}/cpufreq"
if [[ ! -d "${CPU_DIR}" ]]; then
    echo "error: CPU core ${CORE} has no cpufreq directory" >&2
    exit 2
fi

POWER_MODE="$(nvpmodel -q 2>&1 || true)"
if ! grep -q 'NV Power Mode: MAXN' <<<"${POWER_MODE}"; then
    echo "error: ADA L2 requires Jetson MAXN power mode" >&2
    echo "${POWER_MODE}" >&2
    exit 3
fi

MIN_KHZ="$(cat "${CPU_DIR}/scaling_min_freq")"
MAX_KHZ="$(cat "${CPU_DIR}/scaling_max_freq")"
CUR_KHZ="$(cat "${CPU_DIR}/scaling_cur_freq")"
GOVERNOR="$(cat "${CPU_DIR}/scaling_governor")"

if [[ "${MIN_KHZ}" != "${MAX_KHZ}" ]]; then
    cat >&2 <<EOF
error: ADA L2 requires a fixed CPU frequency on the pinned core.
core=${CORE} governor=${GOVERNOR} min_khz=${MIN_KHZ} current_khz=${CUR_KHZ} max_khz=${MAX_KHZ}
Run 'jetson_clocks', verify with 'jetson_clocks --show', then rerun this script.
EOF
    exit 3
fi

if ! taskset -c "${CORE}" true >/dev/null 2>&1; then
    echo "error: cannot pin work to CPU core ${CORE}" >&2
    exit 3
fi

# Correctness gates are part of the evidence protocol and must pass before timing.
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
cargo build --release -p ada-runner

SHA="$(git rev-parse HEAD)"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT_DIR="evidence/a1-online-softmax"
OUT="${OUT_DIR}/thor-l2-${SHA:0:12}-${STAMP}.txt"
mkdir -p "${OUT_DIR}"

thermal_snapshot() {
    for zone in /sys/class/thermal/thermal_zone*; do
        [[ -r "${zone}/type" && -r "${zone}/temp" ]] || continue
        printf '%s=%s\n' "$(cat "${zone}/type")" "$(cat "${zone}/temp")"
    done
}

{
    echo "ADA experiment: ADA-A1 One-Exp Online Softmax"
    echo "evidence_level=L2"
    echo "ada_sha=${SHA}"
    echo "utc=${STAMP}"
    echo "cpu_core=${CORE}"
    echo "process_repeats=${REPEATS}"
    echo

    echo "=== GIT ==="
    git status --short
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
    echo

    echo "=== NVPModel ==="
    printf '%s\n' "${POWER_MODE}"
    echo

    echo "=== JETSON CLOCKS ==="
    jetson_clocks --show 2>&1 || true
    echo

    echo "=== NVIDIA ==="
    nvidia-smi 2>/dev/null || true
    echo

    echo "=== THERMALS BEFORE ==="
    thermal_snapshot
    echo

    echo "=== PROCESS SNAPSHOT BEFORE ==="
    ps -eo pid,psr,pcpu,pmem,comm --sort=-pcpu | head -n 20 || true
    echo

    echo "=== ADA-A1 PINNED RELEASE BENCH ==="
    for run in $(seq 1 "${REPEATS}"); do
        echo "--- process_run=${run}/${REPEATS} ---"
        taskset -c "${CORE}" target/release/ada-runner
        echo
        sleep 1
    done

    echo "=== PINNED CORE AFTER ==="
    echo "current_khz_after=$(cat "${CPU_DIR}/scaling_cur_freq")"
    echo

    echo "=== THERMALS AFTER ==="
    thermal_snapshot
} | tee "${OUT}"

HASH="$(sha256sum "${OUT}")"
echo
echo "${HASH}"
echo "EVIDENCE=${OUT}"
