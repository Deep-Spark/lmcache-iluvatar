#!/usr/bin/env bash
# Copyright (c) 2026, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

# Long-doc QA benchmark for the 2P2D P-cache/NIXL-transfer POC.
#
# Prerequisites: lmcache server, proxy, D1/D2, and P1/P2 are running.
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

export BENCH_LOG_PREFIX="[longQA-2p2d]"

PROXY_HOST="${PROXY_HOST:-127.0.0.1}"
SERVER_PORT="${SERVER_PORT:-${PROXY_PORT}}"
BASE_URL="${BASE_URL:-http://${PROXY_HOST}:${SERVER_PORT}/v1}"

DOCUMENT_LENGTH="${DOCUMENT_LENGTH:-10000}"
# 2P2D default 40 (1P1D examples use 20); keep aligned with offload-2p2d.
NUM_DOCUMENTS="${NUM_DOCUMENTS:-40}"
OUTPUT_LEN="${OUTPUT_LEN:-2}"
REPEAT_COUNT="${REPEAT_COUNT:-1}"
# Align with offload-2p2d (fair TTFT compare). MP still hits across P under random.
REPEAT_MODE="${REPEAT_MODE:-random}"
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

bench_wait_for_2p2d_stack "${PROXY_HOST}" "${SERVER_PORT}"

if [[ "${SKIP_PD_WARMUP}" -eq 0 ]]; then
  bench_log "Warming 2P2D PD path (smoke.sh, NUM_RUNS=4) ..."
  NUM_RUNS=4 MAX_TOKENS=8 PROMPT_REPEAT=40 SKIP_STACK_WAIT=1 \
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
