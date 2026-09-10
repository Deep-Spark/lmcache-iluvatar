#!/bin/bash
# Copyright (c) 2026, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

# =============================================================================
# Qwen3-8B TP2 1P1D — Start Disaggregated Prefill Proxy
#
# Usage:
#   bash start-proxy.sh
#   bash start-proxy.sh [public-port] [prefiller-port] [decoder-port] [decoder-init-ports] [decoder-alloc-ports] [proxy-zmq-port]
#
# Defaults come from env.sh (sourced below). Positional args override env vars.
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "${SCRIPT_DIR}/env.sh"

PUBLIC_PORT="${1:-${PUBLIC_PORT}}"
PREFILLER_PORT="${2:-${PREFILLER_PORT}}"
DECODER_PORT="${3:-${DECODER_PORT}}"
DECODER_INIT_PORTS="${4:-${DECODER_INIT_PORTS}}"
DECODER_ALLOC_PORTS="${5:-${DECODER_ALLOC_PORTS}}"
PROXY_ZMQ_PORT="${6:-${PROXY_ZMQ_PORT}}"

cd "$REPO_ROOT"

echo "Starting disagg proxy"
echo "  Public API: 127.0.0.1:${PUBLIC_PORT}"
echo "  Prefiller: 127.0.0.1:${PREFILLER_PORT}"
echo "  Decoder: 127.0.0.1:${DECODER_PORT}"
echo "  ZMQ metadata: 127.0.0.1:${PROXY_ZMQ_PORT}"

exec python3 examples/disagg_prefill/disagg_proxy_server.py \
  --host 127.0.0.1 \
  --port "$PUBLIC_PORT" \
  --prefiller-host 127.0.0.1 \
  --prefiller-port "$PREFILLER_PORT" \
  --num-prefillers 1 \
  --decoder-host 127.0.0.1 \
  --decoder-port "$DECODER_PORT" \
  --decoder-init-port "$DECODER_INIT_PORTS" \
  --decoder-alloc-port "$DECODER_ALLOC_PORTS" \
  --proxy-host 127.0.0.1 \
  --proxy-port "$PROXY_ZMQ_PORT" \
  --num-decoders 1 \
  --model "${MODEL_PATH}" \
  --chunk-size "${LMCACHE_CHUNK_SIZE}" \
  --pd-buffer-size "${PD_BUFFER_SIZE}"
