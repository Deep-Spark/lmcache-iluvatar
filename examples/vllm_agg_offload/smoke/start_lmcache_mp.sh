#!/usr/bin/env bash
# Copyright (c) 2026, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

# Start the shared LMCache MP server for aggregated vLLM (kv_both).
#
# Start this before start_server.sh. Usage:
#   bash start_lmcache_mp.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "${SCRIPT_DIR}/env.sh"

echo "Starting LMCache MP server"
echo "  ZMQ port: ${LMCACHE_MP_PORT}"
echo "  HTTP metrics: ${LMCACHE_URL}/metrics"
echo "  chunk_size=${LMCACHE_CHUNK_SIZE}  l1_size_gb=${LMCACHE_L1_SIZE_GB}"
echo "  max_workers=${LMCACHE_MAX_WORKERS}"

exec lmcache server \
  --port "${LMCACHE_MP_PORT}" \
  --http-port "${LMCACHE_HTTP_PORT}" \
  --chunk-size "${LMCACHE_CHUNK_SIZE}" \
  --l1-size-gb "${LMCACHE_L1_SIZE_GB}" \
  --max-workers "${LMCACHE_MAX_WORKERS}" \
  --eviction-policy LRU
