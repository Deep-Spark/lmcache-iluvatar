#!/usr/bin/env bash
# Copyright (c) 2026, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

# Prefiller 1: LMCache MP prefix reuse + NixlPush to selected decoder.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "${SCRIPT_DIR}/env.sh"

: "${MODEL_PATH:?MODEL_PATH is not set}"

export UCX_TLS="${UCX_TLS:-cuda_ipc,cuda_copy,tcp}"
export LMCACHE_REQUEST_TELEMETRY_TYPE=fastapi
export LMCACHE_REQUEST_TELEMETRY_ENDPOINT="${LMCACHE_REQUEST_TELEMETRY_ENDPOINT:-http://localhost:${TELEMETRY_PORT}/api/v1/telemetry}"
export VLLM_NIXL_SIDE_CHANNEL_HOST="${NIXL_SIDE_CHANNEL_HOST}"
export VLLM_NIXL_SIDE_CHANNEL_PORT="${NIXL_PREFILL1_SIDE_CHANNEL_PORT}"
export NIXL_PREFILL_ENGINE_ID="${NIXL_PREFILL1_ENGINE_ID}"
export NIXL_KV_PORT="${NIXL_PREFILL1_KV_PORT}"
export PREFILLER_PORT="${PREFILLER1_PORT}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${PREFILLER1_GPU}}"

echo "Starting P-cache+NIXL prefiller 1"
echo "  NIXL_PREFILL_ENGINE_ID=${NIXL_PREFILL_ENGINE_ID}"
echo "  port=${PREFILLER_PORT} side-channel=${VLLM_NIXL_SIDE_CHANNEL_PORT} kv_port=${NIXL_KV_PORT}"
echo "  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} connector=MultiConnector(MP+${NIXL_CONNECTOR})"

KV_TRANSFER_CONFIG="$(
  python3 - <<'PY'
import json
import os

nixl_buffer_size = float(os.environ["NIXL_KV_BUFFER_SIZE"])
nixl_port = int(os.environ["NIXL_KV_PORT"])
lmcache_port = int(os.environ["LMCACHE_MP_PORT"])
engine_id = os.environ["NIXL_PREFILL_ENGINE_ID"]

config = {
    "kv_connector": "MultiConnector",
    "kv_role": "kv_producer",
    "engine_id": engine_id,
    "kv_ip": os.environ["NIXL_KV_IP"],
    "kv_port": nixl_port,
    "kv_buffer_device": os.environ["NIXL_KV_BUFFER_DEVICE"],
    "kv_buffer_size": nixl_buffer_size,
    "kv_connector_extra_config": {
        "connectors": [
            {
                "kv_connector": "LMCacheMPConnector",
                "kv_role": "kv_both",
                "kv_connector_extra_config": {
                    "lmcache.mp.port": lmcache_port,
                    "lmcache.mp.server_urls": os.environ["LMCACHE_MP_SERVER_URL"],
                },
            },
            {
                "kv_connector": os.environ["NIXL_CONNECTOR"],
                "kv_role": "kv_producer",
                "kv_rank": 0,
                "kv_parallel_size": 2,
                "kv_ip": os.environ["NIXL_KV_IP"],
                "kv_port": nixl_port,
                "kv_buffer_device": os.environ["NIXL_KV_BUFFER_DEVICE"],
                "kv_buffer_size": nixl_buffer_size,
            },
        ],
    },
}

print(json.dumps(config, separators=(",", ":")))
PY
)"

exec vllm serve "${MODEL_PATH}" \
  --trust-remote-code \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --tensor-parallel-size "${TENSOR_PARALLEL}" \
  --disable-hybrid-kv-cache-manager \
  --enforce-eager \
  --max-model-len "${MAX_MODEL_LEN}" \
  --gpu-memory-utilization "${GPU_MEM_UTIL}" \
  --port "${PREFILLER_PORT}" \
  --kv-transfer-config "${KV_TRANSFER_CONFIG}"
