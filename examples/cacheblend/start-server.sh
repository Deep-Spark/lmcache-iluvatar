#!/bin/bash
# Copyright (c) 2026, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

# Start the CacheBlend shuffle-doc QA vLLM server.
#
# Run inside an environment that already contains vLLM, LMCache, and
# lmcache-iluvatar. The caller is responsible for exposing GPU devices if this
# script runs inside Docker.
#
# Usage:
#   bash start-server.sh
#   MODEL_PATH=/data/models/Qwen3-8B GPU_DEVICE=0 bash start-server.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

: "${MODEL_PATH:=/data/models/Qwen3-8B}"
: "${SERVED_MODEL_NAME:=$(basename "${MODEL_PATH%/}")}"
: "${BENCH_PORT:=8000}"
: "${GPU_DEVICE:=0}"
: "${MAX_MODEL_LEN:=8192}"
: "${GPU_MEM_UTIL:=0.70}"
: "${LMCACHE_CONFIG_FILE:=$SCRIPT_DIR/lmcache_blend.yaml}"

if [[ ! -f "$LMCACHE_CONFIG_FILE" ]]; then
  echo "ERROR: LMCACHE_CONFIG_FILE does not exist: $LMCACHE_CONFIG_FILE" >&2
  exit 2
fi

export LMCACHE_CONFIG_FILE
export LMCACHE_CHUNK_SIZE="${LMCACHE_CHUNK_SIZE:-256}"
export PYTHONHASHSEED="${PYTHONHASHSEED:-0}"
unset LMCACHE_USE_GPU_CONNECTOR_V3

KV_TRANSFER_CONFIG='{"kv_connector":"LMCacheConnectorV1Dynamic","kv_role":"kv_both","kv_connector_module_path":"lmcache_iluvatar.integration.vllm.lmcache_iluvatar_connector_v1"}'

echo "[cacheblend] model=$MODEL_PATH"
echo "[cacheblend] served_model_name=$SERVED_MODEL_NAME"
echo "[cacheblend] lmcache_config=$LMCACHE_CONFIG_FILE"
echo "[cacheblend] port=$BENCH_PORT"

CUDA_VISIBLE_DEVICES="$GPU_DEVICE" exec vllm serve "$MODEL_PATH" \
  --trust-remote-code \
  --served-model-name "$SERVED_MODEL_NAME" \
  --gpu-memory-utilization "$GPU_MEM_UTIL" \
  --max-model-len "$MAX_MODEL_LEN" \
  --port "$BENCH_PORT" \
  --kv-transfer-config "$KV_TRANSFER_CONFIG" \
  --no-enable-prefix-caching \
  --enforce-eager
