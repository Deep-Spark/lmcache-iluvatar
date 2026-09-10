#!/usr/bin/env bash
# Copyright (c) 2026, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

# Prefiller: P→D KV via vLLM NixlPushConnector only (no LMCache MP).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "${SCRIPT_DIR}/env.sh"

: "${MODEL_PATH:?MODEL_PATH is not set}"

export UCX_TLS="${UCX_TLS:-cuda_ipc,cuda_copy,tcp}"
export VLLM_NIXL_SIDE_CHANNEL_HOST="${NIXL_SIDE_CHANNEL_HOST}"
export VLLM_NIXL_SIDE_CHANNEL_PORT="${NIXL_PREFILL_SIDE_CHANNEL_PORT}"

echo "Starting NixlPush prefiller"
echo "  NIXL_PREFILL_ENGINE_ID=${NIXL_PREFILL_ENGINE_ID}"
echo "  port=${PREFILLER1_PORT} connector=${NIXL_CONNECTOR}"

KV_TRANSFER_CONFIG="$(
  python3 - <<'PY'
import json
import os

config = {
    "kv_connector": os.environ["NIXL_CONNECTOR"],
    "kv_role": "kv_producer",
    "engine_id": os.environ["NIXL_PREFILL_ENGINE_ID"],
    "kv_rank": 0,
    "kv_parallel_size": 2,
    "kv_ip": os.environ["NIXL_KV_IP"],
    "kv_port": int(os.environ["NIXL_KV_PORT"]),
    "kv_buffer_device": os.environ["NIXL_KV_BUFFER_DEVICE"],
    "kv_buffer_size": float(os.environ["NIXL_KV_BUFFER_SIZE"]),
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
  --enable-prefix-caching \
  --port "${PREFILLER1_PORT}" \
  --kv-transfer-config "${KV_TRANSFER_CONFIG}"
