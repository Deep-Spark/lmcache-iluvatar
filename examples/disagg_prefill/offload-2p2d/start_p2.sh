#!/usr/bin/env bash
# Copyright (c) 2026, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

# =============================================================================
# 2P2D PDBackend local-tiered — Prefiller 2 (sender)
#
# Clears only this instance's local_disk dir (PREFILLER2_CACHE_DIR).
# =============================================================================
# Set CUDA_VISIBLE_DEVICES before invoking.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "${SCRIPT_DIR}/env.sh"

: "${MODEL_PATH:?MODEL_PATH is not set}"

LMCACHE_CONFIG_FILE_ARG=$(realpath "${PREFILLER2_CONFIG}")
LOCAL_DISK_CACHE_DIR="${PREFILLER2_CACHE_DIR}"
LMCACHE_RPC_PORT="${PREFILLER2_RPC_PORT}"
PREFILLER_PORT="${PREFILLER2_PORT}"

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
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${PREFILLER2_GPU}}"

echo "Starting prefiller 2 (PDBackend sender, local-tiered)"
echo "  CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "  Port: ${PREFILLER_PORT}"
echo "  Model: ${MODEL_PATH}"
echo "  Served name: ${SERVED_MODEL_NAME}"
echo "  LMCache config: ${LMCACHE_CONFIG_FILE_ARG}"
echo "  Cache dir: ${LOCAL_DISK_CACHE_DIR}"
echo "  VLLM_KV_CACHE_LAYOUT: ${VLLM_KV_CACHE_LAYOUT}"

exec vllm serve "$MODEL_PATH" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --tensor-parallel-size "${TENSOR_PARALLEL}" \
  --pipeline-parallel-size 1 \
  --optimization-level 0 \
  --enforce-eager \
  --max-model-len "${MAX_MODEL_LEN}" \
  --gpu-memory-utilization "${GPU_MEM_UTIL}" \
  --enable-prefix-caching \
  --port "$PREFILLER_PORT" \
  --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1Dynamic","kv_role":"kv_producer","kv_connector_extra_config":{"discard_partial_chunks":false,"lmcache_rpc_port":"'"$LMCACHE_RPC_PORT"'"},"kv_connector_module_path":"lmcache_iluvatar.integration.vllm.lmcache_iluvatar_connector_v1"}'
