#!/usr/bin/env bash
# Copyright (c) 2026, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

# Quick health check: did NixlPush actually transfer KV?
set -euo pipefail

P_METRICS="${P_METRICS:-http://127.0.0.1:8100/metrics}"
D_METRICS="${D_METRICS:-http://127.0.0.1:8200/metrics}"

echo "=== P nixl ==="
curl -s "${P_METRICS}" | grep -E 'nixl_bytes_transferred_count|nixl_xfer_time_seconds_count|nixl_num_failed_transfers_total|nixl_num_kv_expired' | grep -v '#' || true

echo "=== D nixl / prefill ==="
curl -s "${D_METRICS}" | grep -E 'nixl_bytes_transferred_count|nixl_xfer_time_seconds_count|request_prefill_time_seconds_sum|request_prefill_time_seconds_count|prefix_cache_hits_total' | grep -v '#' || true

echo
echo "Expect after successful transfers: nixl_bytes_transferred_count > 0 on P (and usually D)."
echo "If count stays 0 while D prefill_time grows ~P prefill, D is recomputing (check NIXL_ENGINE_ID match)."
