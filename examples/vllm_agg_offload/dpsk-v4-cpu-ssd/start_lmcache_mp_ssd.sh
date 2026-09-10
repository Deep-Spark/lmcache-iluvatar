#!/usr/bin/env bash
# Copyright (c) 2026, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

# LMCache MP server — small L1 + filesystem L2 on SSD.
#
# Warm hits are expected from L2 (disk). L1 is a small staging buffer for the
# GPU→CPU→Disk path. With --l2-store-policy skip_l1, keys are dropped from L1
# after a successful L2 store so warm retrieve must come from SSD.
#
# Start this before start_server.sh. Do not run concurrently with
# start_lmcache_mp_cpu.sh (same ZMQ/HTTP ports by default).
#
# Usage:
#   bash start_lmcache_mp_ssd.sh
#   LMCACHE_SSD_L1_SIZE_GB=8 LMCACHE_L2_BASE_PATH=/mnt/nvme0/lmcache \
#     bash start_lmcache_mp_ssd.sh
#
# Clear previous L2 contents before a clean cold start (optional):
#   CLEAR_L2=1 bash start_lmcache_mp_ssd.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "${SCRIPT_DIR}/env.sh"

export LMCACHE_L1_SIZE_GB="${LMCACHE_SSD_L1_SIZE_GB}"

case "${LMCACHE_L2_STORE_POLICY}" in
  default|skip_l1) ;;
  *)
    echo "Invalid LMCACHE_L2_STORE_POLICY=${LMCACHE_L2_STORE_POLICY}: expected default or skip_l1" >&2
    exit 1
    ;;
esac

mkdir -p "${LMCACHE_L2_BASE_PATH}"
if [[ "${CLEAR_L2:-0}" == "1" ]]; then
  echo "CLEAR_L2=1: removing contents under ${LMCACHE_L2_BASE_PATH}"
  find "${LMCACHE_L2_BASE_PATH}" -mindepth 1 -maxdepth 1 -exec rm -rf {} +
fi

echo "Starting LMCache MP server (SSD L2)"
echo "  ZMQ port: ${LMCACHE_MP_PORT}"
echo "  HTTP metrics: ${LMCACHE_URL}/metrics"
echo "  chunk_size=${LMCACHE_CHUNK_SIZE}  l1_size_gb=${LMCACHE_L1_SIZE_GB} (staging)"
echo "  max_workers=${LMCACHE_MAX_WORKERS}"
echo "  worker_reap_timeout=${LMCACHE_WORKER_REAP_TIMEOUT_SECONDS}s"
echo "  l2_store_policy=${LMCACHE_L2_STORE_POLICY}"
echo "  L2: fs_native base_path=${LMCACHE_L2_BASE_PATH} num_workers=${LMCACHE_L2_NUM_WORKERS} max_capacity_gb=${LMCACHE_L2_MAX_CAPACITY_GB}"

exec lmcache server \
  --port "${LMCACHE_MP_PORT}" \
  --http-port "${LMCACHE_HTTP_PORT}" \
  --chunk-size "${LMCACHE_CHUNK_SIZE}" \
  --l1-size-gb "${LMCACHE_L1_SIZE_GB}" \
  --max-workers "${LMCACHE_MAX_WORKERS}" \
  --worker-reap-timeout-seconds "${LMCACHE_WORKER_REAP_TIMEOUT_SECONDS}" \
  --eviction-policy LRU \
  --l2-store-policy "${LMCACHE_L2_STORE_POLICY}" \
  --l2-adapter '{"type":"fs_native","base_path":"'"${LMCACHE_L2_BASE_PATH}"'","num_workers":'"${LMCACHE_L2_NUM_WORKERS}"',"max_capacity_gb":'"${LMCACHE_L2_MAX_CAPACITY_GB}"'}'
