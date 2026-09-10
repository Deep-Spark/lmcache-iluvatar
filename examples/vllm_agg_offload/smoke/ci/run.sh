#!/usr/bin/env bash
# Copyright (c) 2026, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

# CI entry: aggregated vLLM + LMCache MP Server external-hit smoke.
#
# Owner baseline id: vllm_agg_mp_external_hit_smoke
# Lifecycle: start_lmcache_mp.sh -> start_server.sh -> test_external_hit.sh
#            -> MP Stored + lookup-hit metrics.
#
# Env from examples/run_all_lmcache_iluvatar_tests.py:
#   LMCACHE_CI_MODEL_PATH, LMCACHE_CI_GPUS, LMCACHE_CI_LOG_DIR, ...

set -euo pipefail

EXAMPLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=../env.sh
source "${EXAMPLE_DIR}/env.sh"
# shellcheck source=../../../ci/lib.sh
source "${EXAMPLE_DIR}/../../ci/lib.sh"

trap 'ci_cleanup' EXIT

ci_require_var LMCACHE_CI_LOG_DIR
ci_require_var LMCACHE_CI_MODEL_PATH

export MODEL_PATH="${LMCACHE_CI_MODEL_PATH}"
if [[ -n "${LMCACHE_CI_SERVED_MODEL_NAME:-}" ]]; then
  export SERVED_MODEL_NAME="${LMCACHE_CI_SERVED_MODEL_NAME}"
fi

if [[ -n "${LMCACHE_CI_GPUS:-}" ]]; then
  CI_CUDA_VISIBLE_DEVICES="${LMCACHE_CI_GPUS}"
else
  CI_CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"
fi

mkdir -p "${LMCACHE_CI_LOG_DIR}"

ci_register_ports "${LMCACHE_MP_PORT}" "${LMCACHE_HTTP_PORT}" "${SERVER_PORT}"

ci_log "vllm_agg_offload CI: model=${MODEL_PATH}"
ci_log "  CUDA_VISIBLE_DEVICES=${CI_CUDA_VISIBLE_DEVICES:-<unset>}"
ci_log "  lmcache metrics: ${LMCACHE_URL}/metrics"

ci_start_bg lmcache_server bash "${EXAMPLE_DIR}/start_lmcache_mp.sh"
server_log="$(ci_log_path lmcache_server)"
ci_wait_for_port "127.0.0.1" "${LMCACHE_MP_PORT}"
ci_wait_for_http "${LMCACHE_URL}/metrics"

CUDA_VISIBLE_DEVICES="${CI_CUDA_VISIBLE_DEVICES}" \
  ci_start_bg vllm bash "${EXAMPLE_DIR}/start_server.sh"
vllm_log="$(ci_log_path vllm)"
ci_wait_for_http "http://${SERVER_HOST}:${SERVER_PORT}/health"

export LMCACHE_SERVER_LOG="${server_log}"
export LMCACHE_URL
export SERVER_LOG="${vllm_log}"
ci_log "running test_external_hit.sh (mp_log=${LMCACHE_SERVER_LOG} vllm_log=${SERVER_LOG})"
bash "${EXAMPLE_DIR}/test_external_hit.sh"

sleep 2
ci_assert_log_keywords "${server_log}" "Stored"
ci_assert_metrics_counter_gt \
  "${LMCACHE_URL}/metrics" \
  "lmcache_mp_lookup_hit_tokens_total" \
  0

ci_log "vllm_agg_offload CI passed"
