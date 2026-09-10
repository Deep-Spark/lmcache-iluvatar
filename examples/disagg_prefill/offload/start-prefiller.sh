#!/bin/bash
# Copyright (c) 2026, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

# =============================================================================
# Qwen3-8B TP2 1P1D — Start Prefiller
#
# Usage:
#   bash start-prefiller.sh
#   bash start-prefiller.sh [lmcache-config] [model-path] [port] [rpc-port]
#
# Set CUDA_VISIBLE_DEVICES before invoking (same as disagg_prefill_mp profiles).
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "${SCRIPT_DIR}/env.sh"

LMCACHE_CONFIG_FILE_ARG=$(realpath "${1:-${PREFILLER_CONFIG}}")
MODEL_PATH="${2:-${MODEL_PATH}}"
PREFILLER_PORT="${3:-${PREFILLER_PORT}}"
LMCACHE_RPC_PORT="${4:-${PREFILLER_RPC_PORT}}"
LOCAL_DISK_CACHE_DIR=/tmp/.cache

cd "$REPO_ROOT"
mkdir -p "$LOCAL_DISK_CACHE_DIR"
shopt -s dotglob nullglob
rm -rf -- "${LOCAL_DISK_CACHE_DIR:?}/"*
shopt -u dotglob nullglob

export UCX_TLS="${UCX_TLS:-cuda_ipc,cuda_copy,tcp}"
export UCX_MEM_MMAP_HOOK_MODE="${UCX_MEM_MMAP_HOOK_MODE:-none}"
export LMCACHE_CONFIG_FILE="$LMCACHE_CONFIG_FILE_ARG"
export VLLM_ENABLE_V1_MULTIPROCESSING=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_KV_CACHE_LAYOUT="${VLLM_KV_CACHE_LAYOUT:-HND}"
export PYTHONHASHSEED="${PYTHONHASHSEED:-0}"

echo "Starting prefiller"
echo "  CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "  Port: ${PREFILLER_PORT}"
echo "  Model: ${MODEL_PATH}"
echo "  Served name: ${SERVED_MODEL_NAME}"
echo "  LMCache config: ${LMCACHE_CONFIG_FILE_ARG}"
echo "  VLLM_KV_CACHE_LAYOUT: ${VLLM_KV_CACHE_LAYOUT}"

exec vllm serve "$MODEL_PATH" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --tensor-parallel-size 2 \
  --pipeline-parallel-size 1 \
  --optimization-level 0 \
  --enforce-eager \
  --max-model-len "${MAX_MODEL_LEN}" \
  --gpu-memory-utilization "${GPU_MEM_UTIL}" \
  --enable-prefix-caching \
  --port "$PREFILLER_PORT" \
  --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1Dynamic","kv_role":"kv_producer","kv_connector_extra_config":{"discard_partial_chunks":false,"lmcache_rpc_port":"'"$LMCACHE_RPC_PORT"'"},"kv_connector_module_path":"lmcache_iluvatar.integration.vllm.lmcache_iluvatar_connector_v1"}'
