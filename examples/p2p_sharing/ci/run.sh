#!/usr/bin/env bash
# Copyright (c) 2026, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

# CI entry for P2P KV sharing example.
#
# Lifecycle: controller -> vLLM x2 -> test_cache_reuse.sh -> log keyword checks.
# Env (set by examples/run_all_lmcache_iluvatar_tests.py):
#   LMCACHE_CI_MODEL_PATH, LMCACHE_CI_GPUS, LMCACHE_CI_LOG_DIR, ...

set -euo pipefail

EXAMPLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=../env.sh
source "${EXAMPLE_DIR}/env.sh"
# shellcheck source=../../ci/lib.sh
source "${EXAMPLE_DIR}/../ci/lib.sh"

trap 'ci_cleanup' EXIT

ci_require_var LMCACHE_CI_LOG_DIR
ci_require_var LMCACHE_CI_MODEL_PATH

export MODEL_PATH="${LMCACHE_CI_MODEL_PATH}"
if [[ -n "${LMCACHE_CI_SERVED_MODEL_NAME:-}" ]]; then
  export SERVED_MODEL_NAME="${LMCACHE_CI_SERVED_MODEL_NAME}"
fi

if [[ -n "${LMCACHE_CI_GPUS:-}" ]]; then
  IFS=',' read -r CI_INSTANCE1_CUDA_VISIBLE_DEVICES CI_INSTANCE2_CUDA_VISIBLE_DEVICES _ <<<"${LMCACHE_CI_GPUS},,,"
else
  CI_INSTANCE1_CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"
  CI_INSTANCE2_CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"
fi

export VLLM_USE_V1="${VLLM_USE_V1:-1}"
export UCX_TLS="${UCX_TLS:-cuda_ipc,cuda_copy,tcp}"
export UCX_MEM_MMAP_HOOK_MODE="${UCX_MEM_MMAP_HOOK_MODE:-none}"

mkdir -p "${LMCACHE_CI_LOG_DIR}"

ci_register_ports \
  "${CONTROLLER_API_PORT}" "${CONTROLLER_PULL_PORT}" "${CONTROLLER_REPLY_PORT}" \
  "${INSTANCE1_PORT}" "${INSTANCE2_PORT}" \
  8200 8201 8202 8203 8500 8501

ci_log "p2p_sharing CI: model=${MODEL_PATH}"
ci_log "  instance1 CUDA_VISIBLE_DEVICES=${CI_INSTANCE1_CUDA_VISIBLE_DEVICES:-<unset>}"
ci_log "  instance2 CUDA_VISIBLE_DEVICES=${CI_INSTANCE2_CUDA_VISIBLE_DEVICES:-<unset>}"

ci_start_bg controller bash "${EXAMPLE_DIR}/start_controller.sh"
controller_log="$(ci_log_path controller)"
ci_wait_for_port "127.0.0.1" "${CONTROLLER_API_PORT}"

CUDA_VISIBLE_DEVICES="${CI_INSTANCE1_CUDA_VISIBLE_DEVICES}" \
  ci_start_bg instance1 bash "${EXAMPLE_DIR}/start_instance1.sh"
instance1_log="$(ci_log_path instance1)"
ci_wait_for_http "http://127.0.0.1:${INSTANCE1_PORT}/health"

CUDA_VISIBLE_DEVICES="${CI_INSTANCE2_CUDA_VISIBLE_DEVICES}" \
  ci_start_bg instance2 bash "${EXAMPLE_DIR}/start_instance2.sh"
instance2_log="$(ci_log_path instance2)"
ci_wait_for_http "http://127.0.0.1:${INSTANCE2_PORT}/health"

ci_log "running test_cache_reuse.sh"
bash "${EXAMPLE_DIR}/test_cache_reuse.sh"

sleep 3
ci_assert_log_keywords "${instance1_log}" "Stored"
ci_assert_log_keywords "${instance2_log}" "Retrieved"

ci_log "p2p_sharing CI passed"
