#!/usr/bin/env bash
# Copyright (c) 2026, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

# Start one vLLM OpenAI server: aggregated kv_both + LMCacheMPConnector.
#
# Prerequisites: lmcache server already running
#   (start_lmcache_mp_cpu.sh OR start_lmcache_mp_ssd.sh).
# Set CUDA_VISIBLE_DEVICES before invoking (default GPUs 0-15 for PP=16).
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 bash start_server.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "${SCRIPT_DIR}/env.sh"

: "${MODEL_PATH:?MODEL_PATH is not set}"

KV_TRANSFER_CONFIG="$(
  python3 - <<'PY'
import json
import os

config = {
    "kv_connector": "LMCacheMPConnector",
    "kv_role": "kv_both",
    "kv_connector_extra_config": {
        "lmcache.mp.port": int(os.environ["LMCACHE_MP_PORT"]),
        "lmcache.mp.server_urls": os.environ["LMCACHE_MP_SERVER_URL"],
        "lmcache.mp.heartbeat_interval": float(
            os.environ["LMCACHE_MP_HEARTBEAT_INTERVAL"]
        ),
    },
}
print(json.dumps(config, separators=(",", ":")))
PY
)"

echo "Starting vLLM aggregated DeepSeek-V4 (LMCacheMPConnector)"
echo "  CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "  Port: ${SERVER_HOST}:${SERVER_PORT}"
echo "  Model: ${MODEL_PATH}"
echo "  served-model-name: ${SERVED_MODEL_NAME}"
echo "  TP: ${TENSOR_PARALLEL}  PP: ${PIPELINE_PARALLEL}"
echo "  MAX_MODEL_LEN=${MAX_MODEL_LEN} chunked_prefill=${MAX_NUM_BATCHED_TOKENS}"
echo "  MP server: ${LMCACHE_MP_SERVER_URL} (port ${LMCACHE_MP_PORT})"
echo "  MP heartbeat_interval=${LMCACHE_MP_HEARTBEAT_INTERVAL}s"
echo "  PYTHONHASHSEED: ${PYTHONHASHSEED}"
echo "  VLLM_KV_CACHE_LAYOUT: ${VLLM_KV_CACHE_LAYOUT}"
# echo "  IXFORMER_DS4_FLASH_MLA_WITH_KVCACHE_VERSION: ${IXFORMER_DS4_FLASH_MLA_WITH_KVCACHE_VERSION}"

# Hybrid KV cache manager LEFT ENABLED (required for DS4 MoE w4a8 on BI).
exec vllm serve "${MODEL_PATH}" \
  --trust-remote-code \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --host "${SERVER_HOST}" \
  --port "${SERVER_PORT}" \
  --tensor-parallel-size "${TENSOR_PARALLEL}" \
  --pipeline-parallel-size "${PIPELINE_PARALLEL}" \
  --enforce-eager \
  --max-model-len "${MAX_MODEL_LEN}" \
  --enable-chunked-prefill \
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
  --gpu-memory-utilization "${GPU_MEM_UTIL}" \
  --no-enable-prefix-caching \
  --no-async-scheduling \
  --kv-transfer-config "${KV_TRANSFER_CONFIG}"
