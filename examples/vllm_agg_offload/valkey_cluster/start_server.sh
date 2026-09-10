#!/usr/bin/env bash
# Copyright (c) 2026, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

# Start one aggregated vLLM server connected to the LMCache MP daemon.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "${SCRIPT_DIR}/env.sh"

echo "Starting vLLM for Valkey Cluster L2 validation"
echo "  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "  model=${MODEL_PATH} served_model=${SERVED_MODEL_NAME} TP=${TENSOR_PARALLEL}"
echo "  HTTP=${SERVER_HOST}:${SERVER_PORT} MP=${LMCACHE_MP_SERVER_URL}"
echo "  VLLM_KV_CACHE_LAYOUT=${VLLM_KV_CACHE_LAYOUT}"
echo "  VLLM_KV_DISABLE_CROSS_GROUP_SHARE=${VLLM_KV_DISABLE_CROSS_GROUP_SHARE}"

exec python3 -m vllm.entrypoints.openai.api_server \
  --model "${MODEL_PATH}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --host "${SERVER_HOST}" \
  --port "${SERVER_PORT}" \
  --tensor-parallel-size "${TENSOR_PARALLEL}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
  --gpu-memory-utilization "${GPU_MEM_UTIL}" \
  --enable-prefix-caching \
  --mamba-cache-mode align \
  --enforce-eager \
  --trust-remote-code \
  --reasoning-parser qwen3 \
  --kv-transfer-config '{"kv_connector":"LMCacheMPConnector","kv_role":"kv_both","kv_connector_extra_config":{"lmcache.mp.port":'"${LMCACHE_MP_PORT}"',"lmcache.mp.server_urls":"'"${LMCACHE_MP_SERVER_URL}"'"}}'
