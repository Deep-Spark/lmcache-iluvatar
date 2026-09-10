#!/bin/bash
# Copyright (c) 2026, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

# =============================================================================
# Qwen3-8B TP2 1P1D — KV Offload TTFT Benchmark
#
# Methodology follows Mooncake FAST25 benchmark:
#   https://github.com/kvcache-ai/Mooncake/tree/main/FAST25-release/traces
#
# Prerequisites: proxy, decoder, and prefiller running (see README.md).
# Defaults come from env.sh (sourced below).
#
# Usage:
#   bash run-mooncake-benchmark.sh
#   bash run-mooncake-benchmark.sh --quick
#   bash run-mooncake-benchmark.sh --workload synthetic
#   bash run-mooncake-benchmark.sh /other/model/path --quick
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "${SCRIPT_DIR}/env.sh"
COMMON_DIR="$(cd "${SCRIPT_DIR}/../../common" && pwd)"
# shellcheck source=../../common/bench_lib.sh
source "${COMMON_DIR}/bench_lib.sh"

export BENCH_LOG_PREFIX="[bench]"

BENCH_SCRIPT="${BENCH_SCRIPT:-${BENCH_SERVING}}"

# Optional bench.conf overrides (applied after env.sh).
CONF_FILE="${CONF_FILE:-${SCRIPT_DIR}/bench.conf}"
if [ -f "${CONF_FILE}" ]; then
    # shellcheck source=/dev/null
    set -a; source "${CONF_FILE}"; set +a
fi

if [[ $# -gt 0 && "$1" != --* ]]; then
    MODEL_PATH=$1
    shift
fi

TOKENIZER_PATH="${TOKENIZER_PATH:-${MODEL_PATH}}"
PREFILL_HOST="${PREFILL_HOST:-127.0.0.1}"
PREFILL_PORT="${PREFILL_PORT:-${PREFILLER_PORT}}"
DECODE_HOST="${DECODE_HOST:-127.0.0.1}"
DECODE_PORT="${DECODE_PORT:-${DECODER_PORT}}"
PROXY_HOST="${PROXY_HOST:-127.0.0.1}"
PROXY_PORT="${PROXY_PORT:-${PUBLIC_PORT}}"

NUM_WARMUP="${NUM_WARMUP:-3}"
NUM_PROMPTS="${NUM_PROMPTS:-50}"
WORKLOADS="${WORKLOADS:-synthetic}"
CONCURRENCIES="${CONCURRENCIES:-1 4 8}"

INPUT_LENS=(1024 4096 8192)
OUTPUT_LENS=(6 256)

RESULTS_DIR="${RESULTS_DIR:-${SCRIPT_DIR}/results/$(date +%Y%m%d-%H%M%S)}"
LOGS_DIR="${LOGS_DIR:-${SCRIPT_DIR}/logs/$(date +%Y%m%d-%H%M%S)}"

QUICK=0
WORKLOAD_OVERRIDE=""
RANDOM_ONLY=0
RUN_SUITE_B="${RUN_SUITE_B:-0}"
for arg in "$@"; do
    case "$arg" in
        --quick)       QUICK=1 ;;
        --workload=*)  WORKLOAD_OVERRIDE="${arg#*=}" ;;
        --workload)    shift; WORKLOAD_OVERRIDE="$1" ;;
    esac
done

if [ -n "${WORKLOAD_OVERRIDE}" ]; then
    WORKLOADS="${WORKLOAD_OVERRIDE}"
fi
if [ "${QUICK}" -eq 1 ]; then
    NUM_WARMUP=1
    NUM_PROMPTS=20
    CONCURRENCIES="1 4"
    WORKLOADS="synthetic"
fi

mkdir -p "${RESULTS_DIR}" "${LOGS_DIR}"
export RESULTS_DIR

run_bench() {
    local tag=$1; shift
    local extra_args=("$@")
    local log_file="${LOGS_DIR}/${tag}.txt"
    local result_file="${RESULTS_DIR}/${tag}.jsonl"

    bench_log "--- ${tag} ---"
    python3 "${BENCH_SCRIPT}" \
        --backend vllm \
        --model "${SERVED_MODEL_NAME}" \
        --tokenizer "${TOKENIZER_PATH}" \
        --host "${PROXY_HOST}" \
        --port "${PROXY_PORT}" \
        --num-prompts "${NUM_PROMPTS}" \
        --warmup-requests "${NUM_WARMUP}" \
        --output-details \
        --output-file "${result_file}" \
        --tokenize-prompt \
        --disable-decode-concurrency-limit \
        --pd-disagg-ttft \
        "${extra_args[@]}" \
        2>&1 | tee "${log_file}"
}

bench_wait_for_proxy "${PROXY_HOST}" "${PROXY_PORT}" "proxy"
bench_wait_for_health "${PREFILL_HOST}" "${PREFILL_PORT}" "prefill"
bench_wait_for_health "${DECODE_HOST}" "${DECODE_PORT}" "decode"
bench_log "Benchmark requests will be sent to proxy at ${PROXY_HOST}:${PROXY_PORT} (model=${SERVED_MODEL_NAME})."

if [ "${RANDOM_ONLY}" -eq 0 ]; then
    bench_log "=== Suite A: Mooncake trace workloads ==="
    for workload in ${WORKLOADS}; do
        if ! TRACE_PATH="$(bench_ensure_mooncake_trace "${workload}")"; then
            continue
        fi

        for concurrency in ${CONCURRENCIES}; do
            tag="mooncake-${workload}-c${concurrency}"
            run_bench "${tag}" \
                --dataset-name mooncake \
                --mooncake-workload "${workload}" \
                --dataset-path "${TRACE_PATH}" \
                --max-concurrency "${concurrency}"
        done
    done
fi

if [ "${RUN_SUITE_B}" -eq 0 ]; then
    bench_log "=== Suite B: Random-IDs length sweep (skipped, use RUN_SUITE_B=1 to enable) ==="
else
    bench_log "=== Suite B: Random-IDs length sweep ==="
    for input_len in "${INPUT_LENS[@]}"; do
        for output_len in "${OUTPUT_LENS[@]}"; do
            for concurrency in ${CONCURRENCIES}; do
                tag="random-ids-i${input_len}-o${output_len}-c${concurrency}"
                run_bench "${tag}" \
                    --dataset-name random-ids \
                    --random-input-len  "${input_len}" \
                    --random-output     "${output_len}" \
                    --random-range-ratio 1 \
                    --max-concurrency   "${concurrency}"
            done
        done
    done
fi

bench_log ""
bench_log "=== TTFT Summary ==="
bench_print_ttft_summary 55

bench_log ""
bench_log "Results saved to: ${RESULTS_DIR}"
bench_log "Done."
