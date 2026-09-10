#!/usr/bin/env bash
# Copyright (c) 2026, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

# Start one aggregated vLLM instance connected to its local LMCache MP server.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "${SCRIPT_DIR}/env.sh"

echo "Starting aggregated vLLM with LMCacheMPConnector"
echo "  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "  model=${MODEL_PATH} served_model=${SERVED_MODEL_NAME} TP=${TENSOR_PARALLEL}"
echo "  HTTP=${SERVER_HOST}:${SERVER_PORT} MP=${LMCACHE_MP_SERVER_URL}"
echo "  PYTHONHASHSEED=${PYTHONHASHSEED} layout=${VLLM_KV_CACHE_LAYOUT}"

exec python3 -m vllm.entrypoints.openai.api_server \
  --model "${MODEL_PATH}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --host "${SERVER_HOST}" \
  --port "${SERVER_PORT}" \
  --tensor-parallel-size "${TENSOR_PARALLEL}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --gpu-memory-utilization "${GPU_MEM_UTIL}" \
  --enable-prefix-caching \
  --mamba-cache-mode align \
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
  --trust-remote-code \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --no-enable-log-requests \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","mode":0}' \
  --kv-transfer-config '{"kv_connector":"LMCacheMPConnector","kv_role":"kv_both","kv_connector_extra_config":{"lmcache.mp.port":'"${LMCACHE_MP_PORT}"',"lmcache.mp.server_urls":"'"${LMCACHE_MP_SERVER_URL}"'"}}'
