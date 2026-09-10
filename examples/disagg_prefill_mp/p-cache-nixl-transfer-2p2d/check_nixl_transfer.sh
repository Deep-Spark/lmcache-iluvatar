#!/usr/bin/env bash
# Copyright (c) 2026, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

# Quick health check: did NixlPush transfer KV on the 2P2D stack?
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "${SCRIPT_DIR}/env.sh"

P_PORTS=("${PREFILLER1_PORT}" "${PREFILLER2_PORT}")
D_PORTS=("${DECODER1_PORT}" "${DECODER2_PORT}")

echo_metrics() {
  local label="$1"
  local port="$2"
  echo "=== ${label} (:${port}) nixl ==="
  curl -s "http://127.0.0.1:${port}/metrics" \
    | grep -E 'nixl_bytes_transferred_(sum|count)|nixl_xfer_time_seconds_(sum|count)|nixl_num_failed_transfers_total|nixl_num_kv_expired|request_prefill_time_seconds_sum|request_prefill_time_seconds_count|prefix_cache_hits_total' \
    | grep -v '#' || true
  echo
}

i=1
for port in "${P_PORTS[@]}"; do
  echo_metrics "P${i}" "${port}"
  i=$((i + 1))
done

i=1
for port in "${D_PORTS[@]}"; do
  echo_metrics "D${i}" "${port}"
  i=$((i + 1))
done

echo "Expect after successful transfers: vllm:nixl_bytes_transferred_sum > 0 (or _count > 0)"
echo "on at least one P. If both stay 0 while D prefill_time grows, D is recomputing —"
echo "check per-instance engine_id / side-channel CSV on the proxy."
