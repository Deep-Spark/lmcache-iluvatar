#!/usr/bin/env bash
# Copyright (c) 2026, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
# vLLM instance 1 for P2P KV cache sharing — Iluvatar edition.
#
# kv_role=kv_both: this instance both produces and consumes KV cache.
# Set CUDA_VISIBLE_DEVICES before invoking (same as disagg_prefill_mp profiles).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "${SCRIPT_DIR}/env.sh"

echo "Starting vLLM instance 1"
echo "  CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "  Port: ${INSTANCE1_PORT}"
echo "  Model: ${MODEL_PATH}"
echo "  LMCache config: ${INSTANCE1_CONFIG_FILE}"
echo "  PYTHONHASHSEED: ${PYTHONHASHSEED}"

LMCACHE_CONFIG_FILE="${INSTANCE1_CONFIG_FILE}" exec vllm serve "${MODEL_PATH}" \
  --trust-remote-code \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --gpu-memory-utilization "${GPU_MEM_UTIL}" \
  --enforce-eager \
  --no-enable-prefix-caching \
  --port "${INSTANCE1_PORT}" \
  --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}'
