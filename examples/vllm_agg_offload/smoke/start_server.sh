#!/usr/bin/env bash
# Copyright (c) 2026, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

# Start one vLLM OpenAI server: aggregated kv_both + LMCacheMPConnector.
#
# Prerequisites: lmcache server is already running (bash start_lmcache_mp.sh).
# Set CUDA_VISIBLE_DEVICES before invoking.
# Usage: bash start_server.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "${SCRIPT_DIR}/env.sh"

echo "Starting vLLM aggregated server (LMCacheMPConnector)"
echo "  CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "  Port: ${SERVER_HOST}:${SERVER_PORT}"
echo "  Model: ${MODEL_PATH}"
echo "  TP: ${TENSOR_PARALLEL}"
echo "  MP server: ${LMCACHE_MP_SERVER_URL} (port ${LMCACHE_MP_PORT})"
echo "  PYTHONHASHSEED: ${PYTHONHASHSEED}"
echo "  VLLM_KV_CACHE_LAYOUT: ${VLLM_KV_CACHE_LAYOUT}"

exec python3 -m vllm.entrypoints.openai.api_server \
  --model "${MODEL_PATH}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --host "${SERVER_HOST}" \
  --port "${SERVER_PORT}" \
  --tensor-parallel-size "${TENSOR_PARALLEL}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --gpu-memory-utilization "${GPU_MEM_UTIL}" \
  --enforce-eager \
  --trust-remote-code \
  --no-enable-prefix-caching \
  --kv-transfer-config '{"kv_connector":"LMCacheMPConnector","kv_role":"kv_both","kv_connector_extra_config":{"lmcache.mp.port":'"${LMCACHE_MP_PORT}"',"lmcache.mp.server_urls":"'"${LMCACHE_MP_SERVER_URL}"'"}}'
