#!/usr/bin/env bash
# Copyright (c) 2026, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

# CI entry for shared-storage MP 2P2D smoke (Qwen3-8B, TP=1, 4 GPU).
#
# Lifecycle: start_lmcache_mp.sh -> proxy -> decoders -> prefillers -> smoke_long_prompt_cache.py

set -euo pipefail

PROFILE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MP_ROOT="$(cd "${PROFILE_DIR}/.." && pwd)"

# shellcheck source=../env.sh
source "${PROFILE_DIR}/env.sh"
# shellcheck source=../../ci/lib.sh
source "${MP_ROOT}/../ci/lib.sh"
# shellcheck source=ci_start_stack.sh
source "${PROFILE_DIR}/ci/ci_start_stack.sh"

trap 'ci_cleanup' EXIT

ci_require_var LMCACHE_CI_LOG_DIR
ci_require_var LMCACHE_CI_MODEL_PATH

export MODEL_PATH="${LMCACHE_CI_MODEL_PATH}"
if [[ -n "${LMCACHE_CI_SERVED_MODEL_NAME:-}" ]]; then
  export SERVED_MODEL_NAME="${LMCACHE_CI_SERVED_MODEL_NAME}"
  export MODEL_NAME="${SERVED_MODEL_NAME}"
fi

if [[ -n "${LMCACHE_CI_GPUS:-}" ]]; then
  ci_xpyd_assign_gpus_from_ci_list
fi

export VLLM_USE_V1="${VLLM_USE_V1:-1}"
export LMCACHE_URL="${LMCACHE_URL:-http://localhost:${LMCACHE_HTTP_PORT}}"

mkdir -p "${LMCACHE_CI_LOG_DIR}"

ci_register_ports \
  "${LMCACHE_MP_PORT}" "${LMCACHE_HTTP_PORT}" \
  "${PROXY_PORT}" "${TELEMETRY_PORT}" \
  "${PREFILLER1_PORT}" "${PREFILLER2_PORT}" \
  "${DECODER1_PORT}" "${DECODER2_PORT}"

ci_log "disagg_prefill_mp_smoke CI: model=${MODEL_PATH}"
ci_log "  gpus: P1=${PREFILLER1_GPU} P2=${PREFILLER2_GPU} D1=${DECODER1_GPU} D2=${DECODER2_GPU}"
ci_log "  lmcache metrics: ${LMCACHE_URL}/metrics"

server_log="$(ci_log_path lmcache_server)"
mp_ci_start_lmcache_server
mp_ci_start_proxy_and_workers

ci_log "running smoke_long_prompt_cache.py"
PROMPT_REPEAT="${PROMPT_REPEAT:-220}" \
NUM_RUNS="${NUM_RUNS:-2}" \
MAX_TOKENS="${MAX_TOKENS:-32}" \
TIMEOUT="${TIMEOUT:-900}" \
LMCACHE_URL="${LMCACHE_URL}" \
PROXY_PORT="${PROXY_PORT}" \
MODEL_NAME="${MODEL_NAME:-${SERVED_MODEL_NAME}}" \
bash "${PROFILE_DIR}/smoke.sh"

sleep 3

ci_assert_log_keywords "${server_log}" "Stored"
ci_assert_metrics_counter_gt \
  "${LMCACHE_URL}/metrics" \
  "lmcache_mp_lookup_hit_tokens_total" \
  0

ci_log "disagg_prefill_mp_smoke CI passed"
