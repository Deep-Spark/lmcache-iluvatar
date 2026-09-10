#!/usr/bin/env bash
# Copyright (c) 2026, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

# Prefiller 1 — pair-1 → LMCache Server A (ZMQ 6555).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "${SCRIPT_DIR}/env.sh"

: "${MODEL_PATH:?MODEL_PATH is not set}"
: "${PREFILLER1_GPU:?PREFILLER1_GPU is not set}"

export LMCACHE_REQUEST_TELEMETRY_TYPE=fastapi
# Endpoint defaults from env.sh → http://${IP_A}:${TELEMETRY_PORT}/...
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${PREFILLER1_GPU}}"

echo "Starting P1 → Server A (mp.port=${LMCACHE_MP_PORT_A}) on IP_A=${IP_A}"
echo "  port=${PREFILLER1_PORT} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "  telemetry=${LMCACHE_REQUEST_TELEMETRY_ENDPOINT}"

exec vllm serve "${MODEL_PATH}" \
  --trust-remote-code \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --tensor-parallel-size "${TENSOR_PARALLEL}" \
  --disable-hybrid-kv-cache-manager \
  --enforce-eager \
  --block-size "${VLLM_BLOCK_SIZE}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --gpu-memory-utilization "${GPU_MEM_UTIL}" \
  --no-enable-prefix-caching \
  --port "${PREFILLER1_PORT}" \
  --kv-transfer-config "{\"kv_connector\":\"LMCacheMPConnector\",\"kv_role\":\"kv_both\",\"kv_load_failure_policy\":\"recompute\",\"kv_connector_extra_config\":{\"lmcache.mp.port\":${LMCACHE_MP_PORT_A}}}"
