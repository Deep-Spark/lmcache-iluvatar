#!/usr/bin/env bash
# Copyright (c) 2026, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

# Mooncake FAST25 trace benchmark for the 2P2D P-cache/NIXL-transfer POC.
#
# Prerequisites: lmcache server, proxy, D1/D2, and P1/P2 are running.
#
# Usage:
#   bash run_mooncake_trace_benchmark.sh --quick
#   bash run_mooncake_trace_benchmark.sh --workload synthetic
#   CONCURRENCIES="1 2 4" NUM_PROMPTS=50 bash run_mooncake_trace_benchmark.sh
#   bash run_mooncake_trace_benchmark.sh --compare-decode-limit --quick
#   DISABLE_DECODE_CONCURRENCY_LIMIT=1 bash run_mooncake_trace_benchmark.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "${SCRIPT_DIR}/env.sh"
COMMON_DIR="$(cd "${SCRIPT_DIR}/../../common" && pwd)"
# shellcheck source=../../common/bench_lib.sh
source "${COMMON_DIR}/bench_lib.sh"

export BENCH_LOG_PREFIX="[mooncake-bench-2p2d]"

BENCH_SCRIPT="${BENCH_SCRIPT:-${BENCH_SERVING}}"
TOKENIZER_PATH="${TOKENIZER_PATH:-${MODEL_PATH}}"
PROXY_HOST="${PROXY_HOST:-127.0.0.1}"
SERVER_PORT="${SERVER_PORT:-${PROXY_PORT}}"

NUM_WARMUP="${NUM_WARMUP:-1}"
NUM_PROMPTS="${NUM_PROMPTS:-50}"
WORKLOADS="${WORKLOADS:-synthetic}"
CONCURRENCIES="${CONCURRENCIES:-1 4}"
TAG_PREFIX="${TAG_PREFIX:-p-cache-nixl-2p2d}"
DISABLE_DECODE_CONCURRENCY_LIMIT="${DISABLE_DECODE_CONCURRENCY_LIMIT:-0}"
DECODE_LIMIT_MODE="${DECODE_LIMIT_MODE:-decode-limited}"

RESULTS_DIR="${RESULTS_DIR:-${SCRIPT_DIR}/results/mooncake-$(date +%Y%m%d-%H%M%S)}"
LOGS_DIR="${LOGS_DIR:-${SCRIPT_DIR}/logs/mooncake-$(date +%Y%m%d-%H%M%S)}"

QUICK=0
COMPARE_DECODE_LIMIT=0
WORKLOAD_OVERRIDE=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --quick) QUICK=1 ;;
    --compare-decode-limit) COMPARE_DECODE_LIMIT=1 ;;
    --workload=*) WORKLOAD_OVERRIDE="${1#*=}" ;;
    --workload)
      shift
      WORKLOAD_OVERRIDE="${1:?--workload requires a value}"
      ;;
    --tag-prefix=*) TAG_PREFIX="${1#*=}" ;;
    --tag-prefix)
      shift
      TAG_PREFIX="${1:?--tag-prefix requires a value}"
      ;;
    *)
      echo "Unknown argument: ${1}" >&2
      exit 2
      ;;
  esac
  shift
done

if [[ -n "${WORKLOAD_OVERRIDE}" ]]; then
  WORKLOADS="${WORKLOAD_OVERRIDE}"
fi

if [[ "${QUICK}" -eq 1 ]]; then
  NUM_WARMUP=1
  NUM_PROMPTS=20
  CONCURRENCIES="1 4"
  WORKLOADS="synthetic"
fi

mkdir -p "${RESULTS_DIR}" "${LOGS_DIR}"
export RESULTS_DIR

run_bench() {
  local tag="$1"
  local decode_limit_mode="$2"
  shift 2
  local log_file="${LOGS_DIR}/${tag}.txt"
  local result_file="${RESULTS_DIR}/${tag}.jsonl"
  local -a bench_args=(
    --backend vllm-chat
    --model "${SERVED_MODEL_NAME}"
    --tokenizer "${TOKENIZER_PATH}"
    --host "${PROXY_HOST}"
    --port "${SERVER_PORT}"
    --num-prompts "${NUM_PROMPTS}"
    --warmup-requests "${NUM_WARMUP}"
    --output-details
    --output-file "${result_file}"
    --tokenize-prompt
    --pd-disagg-ttft
  )

  if [[ "${decode_limit_mode}" == "decode-unlimited" ]]; then
    bench_args+=(--disable-decode-concurrency-limit)
  fi

  bench_log "--- ${tag} (decode_limit=${decode_limit_mode}) ---"
  python3 "${BENCH_SCRIPT}" \
    "${bench_args[@]}" \
    "$@" \
    2>&1 | tee "${log_file}"
}

decode_limit_modes_to_run() {
  if [[ "${COMPARE_DECODE_LIMIT}" -eq 1 || "${DECODE_LIMIT_MODE}" == "both" ]]; then
    echo "decode-limited decode-unlimited"
  elif [[ "${DECODE_LIMIT_MODE}" == "decode-unlimited" || "${DISABLE_DECODE_CONCURRENCY_LIMIT}" == "1" ]]; then
    echo "decode-unlimited"
  else
    echo "decode-limited"
  fi
}

bench_wait_for_2p2d_stack "${PROXY_HOST}" "${SERVER_PORT}"
bench_log "Benchmark requests -> http://${PROXY_HOST}:${SERVER_PORT} (model=${SERVED_MODEL_NAME})."
bench_log "workloads=${WORKLOADS} concurrencies=${CONCURRENCIES} num_prompts=${NUM_PROMPTS} warmup=${NUM_WARMUP}"
bench_log "decode_limit_modes=$(decode_limit_modes_to_run | tr '\n' ' ')"
bench_log "results=${RESULTS_DIR}"

for workload in ${WORKLOADS}; do
  if ! trace_path="$(bench_ensure_mooncake_trace "${workload}")"; then
    continue
  fi
  for decode_limit_mode in $(decode_limit_modes_to_run); do
    for concurrency in ${CONCURRENCIES}; do
      tag="${TAG_PREFIX}-${decode_limit_mode}-mooncake-${workload}-c${concurrency}"
      run_bench "${tag}" "${decode_limit_mode}" \
        --dataset-name mooncake \
        --mooncake-workload "${workload}" \
        --dataset-path "${trace_path}" \
        --max-concurrency "${concurrency}"
    done
  done
done

bench_log ""
bench_log "=== TTFT Summary ==="
bench_print_ttft_summary 60

bench_log ""
bench_log "Results saved to: ${RESULTS_DIR}"
bench_log "Logs saved to: ${LOGS_DIR}"
bench_log "Done."
