#!/usr/bin/env bash
# Copyright (c) 2026, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

# Decoder 2: consume P/D KV through vLLM NIXL only (no MP).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "${SCRIPT_DIR}/env.sh"

: "${MODEL_PATH:?MODEL_PATH is not set}"

export UCX_TLS="${UCX_TLS:-cuda_ipc,cuda_copy,tcp}"
export VLLM_NIXL_SIDE_CHANNEL_HOST="${NIXL_SIDE_CHANNEL_HOST}"
export VLLM_NIXL_SIDE_CHANNEL_PORT="${NIXL_DECODE2_SIDE_CHANNEL_PORT}"
export NIXL_DECODE_ENGINE_ID="${NIXL_DECODE2_ENGINE_ID}"
export NIXL_KV_PORT="${NIXL_DECODE2_KV_PORT}"
export DECODER_PORT="${DECODER2_PORT}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${DECODER2_GPU}}"

echo "Starting P-cache+NIXL decoder 2 (NIXL only, no MP)"
echo "  NIXL_DECODE_ENGINE_ID=${NIXL_DECODE_ENGINE_ID}"
echo "  port=${DECODER_PORT} side-channel=${VLLM_NIXL_SIDE_CHANNEL_PORT} kv_port=${NIXL_KV_PORT}"
echo "  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} connector=${NIXL_CONNECTOR}"

KV_TRANSFER_CONFIG="$(
  python3 - <<'PY'
import json
import os

config = {
    "kv_connector": os.environ["NIXL_CONNECTOR"],
    "kv_role": "kv_consumer",
    "engine_id": os.environ["NIXL_DECODE_ENGINE_ID"],
    "kv_rank": 1,
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
  --port "${DECODER_PORT}" \
  --kv-transfer-config "${KV_TRANSFER_CONFIG}"
