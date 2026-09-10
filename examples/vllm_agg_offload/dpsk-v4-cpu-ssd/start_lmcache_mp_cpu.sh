#!/usr/bin/env bash
# Copyright (c) 2026, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

# LMCache MP server — CPU L1 only (no L2 / SSD).
#
# Start this before start_server.sh. Do not run concurrently with
# start_lmcache_mp_ssd.sh (same ZMQ/HTTP ports by default).
#
# Usage:
#   bash start_lmcache_mp_cpu.sh
#   LMCACHE_CPU_L1_SIZE_GB=300 bash start_lmcache_mp_cpu.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "${SCRIPT_DIR}/env.sh"

export LMCACHE_L1_SIZE_GB="${LMCACHE_CPU_L1_SIZE_GB}"

echo "Starting LMCache MP server (CPU L1 only)"
echo "  ZMQ port: ${LMCACHE_MP_PORT}"
echo "  HTTP metrics: ${LMCACHE_URL}/metrics"
echo "  chunk_size=${LMCACHE_CHUNK_SIZE}  l1_size_gb=${LMCACHE_L1_SIZE_GB}"
echo "  max_workers=${LMCACHE_MAX_WORKERS}"
echo "  worker_reap_timeout=${LMCACHE_WORKER_REAP_TIMEOUT_SECONDS}s"
echo "  L2: disabled"

exec lmcache server \
  --port "${LMCACHE_MP_PORT}" \
  --http-port "${LMCACHE_HTTP_PORT}" \
  --chunk-size "${LMCACHE_CHUNK_SIZE}" \
  --l1-size-gb "${LMCACHE_L1_SIZE_GB}" \
  --max-workers "${LMCACHE_MAX_WORKERS}" \
  --worker-reap-timeout-seconds "${LMCACHE_WORKER_REAP_TIMEOUT_SECONDS}" \
  --eviction-policy LRU
