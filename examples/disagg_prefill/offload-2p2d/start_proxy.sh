#!/usr/bin/env bash
# Copyright (c) 2026, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

# =============================================================================
# 2P2D PDBackend local-tiered — Start Disaggregated Prefill Proxy
#
# Synchronized RR (P_i↔D_i). Decoder HTTP uses incremental_mode
# (base DECODER1_PORT + num_decoders); init/alloc ports stride by list length
# so TP=2 bases do not overlap on D2.
#
# Usage:
#   bash start_proxy.sh
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "${SCRIPT_DIR}/env.sh"

cd "$REPO_ROOT"

echo "Starting disagg proxy (2P2D PDBackend, sync RR)"
echo "  Public API: 127.0.0.1:${PUBLIC_PORT}"
echo "  Prefillers: 127.0.0.1:${PREFILLER1_PORT},${PREFILLER2_PORT}"
echo "  Decoders:   127.0.0.1:${DECODER1_PORT} (+num_decoders=2 incremental)"
echo "  ZMQ metadata: 127.0.0.1:${PROXY_ZMQ_PORT}"
echo "  decoder-init base: ${DECODER_INIT_PORTS}  (D2 = base + stride)"
echo "  decoder-alloc base: ${DECODER_ALLOC_PORTS}"

exec python3 examples/disagg_prefill/disagg_proxy_server.py \
  --host 127.0.0.1 \
  --port "$PUBLIC_PORT" \
  --prefiller-host 127.0.0.1 \
  --prefiller-port "${PREFILLER1_PORT},${PREFILLER2_PORT}" \
  --num-prefillers 2 \
  --decoder-host 127.0.0.1 \
  --decoder-port "$DECODER1_PORT" \
  --num-decoders 2 \
  --decoder-init-port "$DECODER_INIT_PORTS" \
  --decoder-alloc-port "$DECODER_ALLOC_PORTS" \
  --proxy-host 127.0.0.1 \
  --proxy-port "$PROXY_ZMQ_PORT" \
  --model "${MODEL_PATH}" \
  --chunk-size "${LMCACHE_CHUNK_SIZE}" \
  --pd-buffer-size "${PD_BUFFER_SIZE}"
