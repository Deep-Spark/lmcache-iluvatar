#!/usr/bin/env bash
# Copyright (c) 2026, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

# Start LMCache MP server with GDS L1 (--gds-l1-path).
#
# Prerequisites: GDS_L1_PATH set to a GDS-capable filesystem directory.
# Start this before start_server.sh. Usage:
#   GDS_L1_PATH=/mnt/nvme/lmcache-gds-l1 bash start_lmcache_mp.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "${SCRIPT_DIR}/env.sh"

mkdir -p "${GDS_L1_PATH}"

DIRECT_IO_ARGS=()
case "${GDS_L1_USE_DIRECT_IO,,}" in
  0|false|no|off)
    DIRECT_IO_ARGS+=(--no-gds-l1-use-direct-io)
    ;;
  *)
    DIRECT_IO_ARGS+=(--gds-l1-use-direct-io)
    ;;
esac

echo "Starting LMCache MP server (GDS L1)"
echo "  ZMQ port: ${LMCACHE_MP_PORT}"
echo "  HTTP metrics: ${LMCACHE_URL}/metrics"
echo "  chunk_size=${LMCACHE_CHUNK_SIZE}  l1_size_gb=${LMCACHE_L1_SIZE_GB} (slab)"
echo "  gds_l1_path=${GDS_L1_PATH}"
echo "  gds_l1_use_direct_io=${GDS_L1_USE_DIRECT_IO}"
echo "  max_workers=${LMCACHE_MAX_WORKERS}"

exec lmcache server \
  --port "${LMCACHE_MP_PORT}" \
  --http-port "${LMCACHE_HTTP_PORT}" \
  --chunk-size "${LMCACHE_CHUNK_SIZE}" \
  --l1-size-gb "${LMCACHE_L1_SIZE_GB}" \
  --max-workers "${LMCACHE_MAX_WORKERS}" \
  --eviction-policy LRU \
  --gds-l1-path "${GDS_L1_PATH}" \
  "${DIRECT_IO_ARGS[@]}"
