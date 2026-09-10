#!/usr/bin/env bash
# Copyright (c) 2026, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

# Long-doc QA: prefix offload hit-rate for aggregated DeepSeek-V4 + LMCache MP.
#
# Cold stores DOCUMENT_LENGTH * TARGET_HIT_RATE; query sends full DOCUMENT_LENGTH
# with the same prefix → expected offload hit ≈ TARGET_HIT_RATE.
#
# Prerequisites: lmcache server + vLLM already running.
#
# Usage:
#   bash run-longQA-benchmark.sh
#   bash run-longQA-benchmark.sh --quick
#   WARMUP_QUERY_WAIT_SECONDS=0 bash run-longQA-benchmark.sh  # reproduce HOL
#   DOCUMENT_LENGTH=1000000 TARGET_HIT_RATE=0.8 bash run-longQA-benchmark.sh
#   DOCUMENT_LENGTH=262144 TARGET_HIT_RATE=0.8 NUM_DOCUMENTS=4 bash run-longQA-benchmark.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "${SCRIPT_DIR}/env.sh"
COMMON_DIR="$(cd "${SCRIPT_DIR}/../../common" && pwd)"
# shellcheck source=../../common/bench_lib.sh
source "${COMMON_DIR}/bench_lib.sh"

export BENCH_LOG_PREFIX="[longQA-dpsk-agg]"

BASE_URL="${BASE_URL:-http://${SERVER_HOST}:${SERVER_PORT}/v1}"
LONG_DOC_QA="${SCRIPT_DIR}/long_doc_qa.py"

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
  DOCUMENT_LENGTH="${DOCUMENT_LENGTH:-8000}"
  NUM_DOCUMENTS="${NUM_DOCUMENTS:-4}"
else
  DOCUMENT_LENGTH="${DOCUMENT_LENGTH:-1000000}"
  NUM_DOCUMENTS="${NUM_DOCUMENTS:-1}"
fi
TARGET_HIT_RATE="${TARGET_HIT_RATE:-0.8}"
OUTPUT_LEN="${OUTPUT_LEN:-8}"
REPEAT_COUNT="${REPEAT_COUNT:-1}"
REPEAT_MODE="${REPEAT_MODE:-tile}"
MAX_INFLIGHT="${MAX_INFLIGHT:-1}"
SHUFFLE_SEED="${SHUFFLE_SEED:-0}"
CHUNK_SIZE="${LMCACHE_CHUNK_SIZE:-2048}"
HEALTH_TIMEOUT_SEC="${HEALTH_TIMEOUT_SEC:-1800}"

RUN_TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
RESULTS_DIR="${RESULTS_DIR:-${SCRIPT_DIR}/results/longqa-${RUN_TIMESTAMP}}"
LOGS_DIR="${LOGS_DIR:-${SCRIPT_DIR}/logs/longqa-${RUN_TIMESTAMP}}"

mkdir -p "${RESULTS_DIR}" "${LOGS_DIR}"

bench_wait_for_health "${SERVER_HOST}" "${SERVER_PORT}" "vllm-agg-dpsk" "${HEALTH_TIMEOUT_SEC}"

bench_log "Starting long_doc_qa.py (base-url=${BASE_URL}, model=${SERVED_MODEL_NAME})"
bench_log "document_length=${DOCUMENT_LENGTH} target_hit_rate=${TARGET_HIT_RATE} num_documents=${NUM_DOCUMENTS} output_len=${OUTPUT_LEN} chunk_size=${CHUNK_SIZE}"
bench_log "warmup_query_wait_seconds=${WARMUP_QUERY_WAIT_SECONDS}"
bench_log "results=${RESULTS_DIR}"

python3 "${LONG_DOC_QA}" \
  --base-url "${BASE_URL}" \
  --model "${SERVED_MODEL_NAME}" \
  --document-length "${DOCUMENT_LENGTH}" \
  --target-hit-rate "${TARGET_HIT_RATE}" \
  --chunk-size "${CHUNK_SIZE}" \
  --num-documents "${NUM_DOCUMENTS}" \
  --output-len "${OUTPUT_LEN}" \
  --repeat-count "${REPEAT_COUNT}" \
  --repeat-mode "${REPEAT_MODE}" \
  --max-inflight-requests "${MAX_INFLIGHT}" \
  --shuffle-seed "${SHUFFLE_SEED}" \
  --sleep-time-after-warmup "${WARMUP_QUERY_WAIT_SECONDS}" \
  --output-dir "${RESULTS_DIR}" \
  --json-output \
  --completions \
  2>&1 | tee "${LOGS_DIR}/longqa-bench.txt"

bench_log "CSV results: ${RESULTS_DIR}/warmup_round.csv ${RESULTS_DIR}/query_round.csv"
bench_log "Done."
