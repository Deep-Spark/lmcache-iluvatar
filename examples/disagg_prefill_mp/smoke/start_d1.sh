#!/usr/bin/env bash
# Copyright (c) 2026, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

# Decoder 1 — 2P2D MP smoke.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "${SCRIPT_DIR}/env.sh"

: "${DECODER1_GPU:?DECODER1_GPU is not set}"

CUDA_VISIBLE_DEVICES="${DECODER1_GPU}" exec vllm serve "${MODEL_PATH}" \
  --trust-remote-code \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --tensor-parallel-size "${TENSOR_PARALLEL}" \
  --disable-hybrid-kv-cache-manager \
  --enforce-eager \
  --block-size "${VLLM_BLOCK_SIZE}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --gpu-memory-utilization "${GPU_MEM_UTIL}" \
  --no-enable-prefix-caching \
  --port "${DECODER1_PORT}" \
  --kv-transfer-config "{\"kv_connector\":\"LMCacheMPConnector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"lmcache.mp.port\":${LMCACHE_MP_PORT}}}"
