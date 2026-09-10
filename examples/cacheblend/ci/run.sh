#!/usr/bin/env bash
# Copyright (c) 2026, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

# CI entry: CacheBlend shuffle-doc QA smoke.
#
# Lifecycle: vLLM server -> run_benchmark.sh -> benchmark/log assertions.

set -euo pipefail

EXAMPLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=../../ci/lib.sh
source "${EXAMPLE_DIR}/../ci/lib.sh"

trap 'ci_cleanup' EXIT

ci_require_var LMCACHE_CI_LOG_DIR
ci_require_var LMCACHE_CI_MODEL_PATH

export MODEL_PATH="${LMCACHE_CI_MODEL_PATH}"
export TOKENIZER_PATH="${LMCACHE_CI_MODEL_PATH}"
if [[ -n "${LMCACHE_CI_SERVED_MODEL_NAME:-}" ]]; then
  export SERVED_MODEL_NAME="${LMCACHE_CI_SERVED_MODEL_NAME}"
else
  export SERVED_MODEL_NAME="$(basename "${MODEL_PATH%/}")"
fi

if [[ -n "${LMCACHE_CI_GPUS:-}" ]]; then
  IFS=',' read -r GPU_DEVICE _ <<<"${LMCACHE_CI_GPUS},"
  export GPU_DEVICE
fi

export BENCH_PORT="${BENCH_PORT:-18020}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
export GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.70}"
export OUT_DIR="${LMCACHE_CI_LOG_DIR}/client"
export NUM_DOCUMENTS="${NUM_DOCUMENTS:-3}"
export DOCUMENT_LENGTH="${DOCUMENT_LENGTH:-128}"
export NUM_REQUESTS="${NUM_REQUESTS:-2}"
export OUTPUT_LEN="${OUTPUT_LEN:-1}"
export MAX_INFLIGHT_REQUESTS="${MAX_INFLIGHT_REQUESTS:-1}"
export SLEEP_AFTER_WARMUP="${SLEEP_AFTER_WARMUP:-2}"
export PYTHONHASHSEED="${PYTHONHASHSEED:-0}"

mkdir -p "${LMCACHE_CI_LOG_DIR}" "${OUT_DIR}"

ci_register_ports "${BENCH_PORT}"

ci_log "cacheblend CI: model=${MODEL_PATH} gpu=${GPU_DEVICE:-0} port=${BENCH_PORT}"

ci_start_bg cacheblend_server bash "${EXAMPLE_DIR}/start-server.sh"
server_log="$(ci_log_path cacheblend_server)"
ci_wait_for_http "http://127.0.0.1:${BENCH_PORT}/health"

ci_log "running run_benchmark.sh (out=${OUT_DIR})"
bash "${EXAMPLE_DIR}/run_benchmark.sh"

client_log="${OUT_DIR}/client.log"
stats_log="${OUT_DIR}/lmcache_stats.txt"
responses_log="${OUT_DIR}/responses.log"

ci_assert_log_keywords "${responses_log}" "CACHEBLEND SHUFFLE BENCHMARK RESULTS" "Query round prompt count"
ci_assert_log_keywords "${stats_log}" "derangement" "TTFT:"
ci_assert_log_keywords "${responses_log}" "query round"
ci_assert_log_keywords "${server_log}" "LMCacheIluvatarConnectorV1Dynamic"

ci_log "cacheblend CI passed"
