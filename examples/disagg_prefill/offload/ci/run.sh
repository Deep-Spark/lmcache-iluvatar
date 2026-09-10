#!/usr/bin/env bash
# Copyright (c) 2026, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

# CI entry for 1P1D local-tiered LMCache offload (no Mooncake).
#
# Lifecycle (validated smoke order):
#   proxy -> decoder -> prefiller -> ci/smoke_test.sh
#
# Requires 4 GPUs (TP=2 prefiller + TP=2 decoder, Qwen3-8B). Uses configs under
# configs/lmcache-local-tiered/ only.

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
  IFS=',' read -r g0 g1 g2 g3 _ <<<"${LMCACHE_CI_GPUS},,,,"
  CI_PREFILLER_CUDA_VISIBLE_DEVICES="${g0},${g1}"
  CI_DECODER_CUDA_VISIBLE_DEVICES="${g2},${g3}"
else
  CI_PREFILLER_CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"
  CI_DECODER_CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"
fi

export VLLM_USE_V1="${VLLM_USE_V1:-1}"
export VLLM_ENABLE_V1_MULTIPROCESSING="${VLLM_ENABLE_V1_MULTIPROCESSING:-1}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export PYTHONHASHSEED="${PYTHONHASHSEED:-0}"
export UCX_TLS="${UCX_TLS:-cuda_ipc,cuda_copy,tcp}"

mkdir -p "${LMCACHE_CI_LOG_DIR}" /data/tmp/.cache

# Match DECODER_*_PORTS used by start-proxy.sh (comma-separated env vars).
read -r -a _decoder_init_ports <<<"${DECODER_INIT_PORTS//,/ }"
read -r -a _decoder_alloc_ports <<<"${DECODER_ALLOC_PORTS//,/ }"
ci_register_ports \
  "${PREFILLER_PORT}" "${DECODER_PORT}" "${PUBLIC_PORT}" "${PROXY_ZMQ_PORT}" \
  "${_decoder_init_ports[@]}" "${_decoder_alloc_ports[@]}"

ci_log "disagg_prefill_offload_local_tiered CI: model=${MODEL_PATH}"
ci_log "  prefiller CUDA_VISIBLE_DEVICES=${CI_PREFILLER_CUDA_VISIBLE_DEVICES:-<unset>}"
ci_log "  decoder CUDA_VISIBLE_DEVICES=${CI_DECODER_CUDA_VISIBLE_DEVICES:-<unset>}"
ci_log "  config=${PREFILLER_CONFIG} (no Mooncake)"

# 1. Disagg proxy (must be up before prefiller/decoder PD handshake)
ci_start_bg proxy bash "${EXAMPLE_DIR}/start-proxy.sh" \
  "${PUBLIC_PORT}" "${PREFILLER_PORT}" "${DECODER_PORT}" \
  "${DECODER_INIT_PORTS}" "${DECODER_ALLOC_PORTS}" "${PROXY_ZMQ_PORT}"
ci_wait_for_port "127.0.0.1" "${PUBLIC_PORT}"

# 2. Decoder
CUDA_VISIBLE_DEVICES="${CI_DECODER_CUDA_VISIBLE_DEVICES}" \
  ci_start_bg decoder bash "${EXAMPLE_DIR}/start-decoder.sh" \
  "${DECODER_CONFIG}" "${MODEL_PATH}" "${DECODER_PORT}" \
  "${DECODER_RPC_PORT}" "${SKIP_LAST_N_TOKENS}"
decoder_log="$(ci_log_path decoder)"
ci_wait_for_http "http://127.0.0.1:${DECODER_PORT}/health"

# 3. Prefiller
CUDA_VISIBLE_DEVICES="${CI_PREFILLER_CUDA_VISIBLE_DEVICES}" \
  ci_start_bg prefiller bash "${EXAMPLE_DIR}/start-prefiller.sh" \
  "${PREFILLER_CONFIG}" "${MODEL_PATH}" "${PREFILLER_PORT}" \
  "${PREFILLER_RPC_PORT}"
prefiller_log="$(ci_log_path prefiller)"
ci_wait_for_http "http://127.0.0.1:${PREFILLER_PORT}/health"

ci_log "running ci/smoke_test.sh"
bash "${EXAMPLE_DIR}/ci/smoke_test.sh"

sleep 3
# PDBackend: confirms Iluvatar PD path is active on both roles.
# Do not assert proxy "KV ready" — that message is logger.debug and absent at default INFO.
ci_assert_log_keywords "${prefiller_log}" "PDBackend"
ci_assert_log_keywords "${decoder_log}" "PDBackend"

ci_log "disagg_prefill_offload_local_tiered CI passed"
