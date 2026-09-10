#!/usr/bin/env bash
# Copyright (c) 2026, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

# Long-doc QA benchmark for the 1P1D P-cache/NIXL-transfer POC.
#
# Prerequisites: lmcache server, proxy, decoder, and prefiller are running.
#
# Usage:
#   bash run-longQA-benchmark.sh
#   bash run-longQA-benchmark.sh --quick
#   SKIP_PD_WARMUP=1 bash run-longQA-benchmark.sh
#   DOCUMENT_LENGTH=8000 NUM_DOCUMENTS=20 bash run-longQA-benchmark.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "${SCRIPT_DIR}/env.sh"
COMMON_DIR="$(cd "${SCRIPT_DIR}/../../common" && pwd)"
# shellcheck source=../../common/bench_lib.sh
source "${COMMON_DIR}/bench_lib.sh"

export BENCH_LOG_PREFIX="[longQA]"

PROXY_HOST="${PROXY_HOST:-127.0.0.1}"
SERVER_PORT="${SERVER_PORT:-${PROXY_PORT}}"
BASE_URL="${BASE_URL:-http://${PROXY_HOST}:${SERVER_PORT}/v1}"

DOCUMENT_LENGTH="${DOCUMENT_LENGTH:-10000}"
# 1P1D default 20 (2P2D examples use 40).
NUM_DOCUMENTS="${NUM_DOCUMENTS:-20}"
OUTPUT_LEN="${OUTPUT_LEN:-2}"
REPEAT_COUNT="${REPEAT_COUNT:-1}"
REPEAT_MODE="${REPEAT_MODE:-tile}"
MAX_INFLIGHT="${MAX_INFLIGHT:-1}"
SHUFFLE_SEED="${SHUFFLE_SEED:-0}"
SKIP_PD_WARMUP="${SKIP_PD_WARMUP:-0}"

RESULTS_DIR="${RESULTS_DIR:-${SCRIPT_DIR}/results/longqa-$(date +%Y%m%d-%H%M%S)}"
LOGS_DIR="${LOGS_DIR:-${SCRIPT_DIR}/logs/longqa-$(date +%Y%m%d-%H%M%S)}"

QUICK=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --quick) QUICK=1 ;;
    *)
      echo "Unknown argument: ${1}" >&2
      exit 2
      ;;
  esac
  shift
done

if [[ "${QUICK}" -eq 1 ]]; then
  DOCUMENT_LENGTH=8000
  NUM_DOCUMENTS=4
  OUTPUT_LEN=2
  REPEAT_COUNT=1
  MAX_INFLIGHT=1
fi

mkdir -p "${RESULTS_DIR}" "${LOGS_DIR}"

bench_wait_for_1p1d_stack "${PROXY_HOST}" "${SERVER_PORT}"

if [[ "${SKIP_PD_WARMUP}" -eq 0 ]]; then
  bench_log "Warming 1P1D PD path (smoke.sh, NUM_RUNS=1) ..."
  NUM_RUNS=1 MAX_TOKENS=8 PROMPT_REPEAT=40 SKIP_STACK_WAIT=1 \
    bash "${SCRIPT_DIR}/smoke.sh" \
    2>&1 | tee "${LOGS_DIR}/pd-warmup.txt"
else
  bench_log "SKIP_PD_WARMUP=1: skipping PD warm-up."
fi

bench_log "Starting long_doc_qa.py (base-url=${BASE_URL}, model=${SERVED_MODEL_NAME})"
bench_log "document_length=${DOCUMENT_LENGTH} num_documents=${NUM_DOCUMENTS} output_len=${OUTPUT_LEN}"
bench_log "results=${RESULTS_DIR}"

python3 "${LONG_DOC_QA}" \
  --base-url "${BASE_URL}" \
  --model "${SERVED_MODEL_NAME}" \
  --document-length "${DOCUMENT_LENGTH}" \
  --num-documents "${NUM_DOCUMENTS}" \
  --output-len "${OUTPUT_LEN}" \
  --repeat-count "${REPEAT_COUNT}" \
  --repeat-mode "${REPEAT_MODE}" \
  --max-inflight-requests "${MAX_INFLIGHT}" \
  --shuffle-seed "${SHUFFLE_SEED}" \
  --output-dir "${RESULTS_DIR}" \
  --pd-disagg-ttft \
  --json-output \
  --completions \
  2>&1 | tee "${LOGS_DIR}/longqa-bench.txt"

bench_log "CSV results: ${RESULTS_DIR}/warmup_round.csv ${RESULTS_DIR}/query_round.csv"
bench_log "Done."
