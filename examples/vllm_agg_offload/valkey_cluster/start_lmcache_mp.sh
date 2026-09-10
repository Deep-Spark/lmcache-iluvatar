#!/usr/bin/env bash
# Copyright (c) 2026, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

# Start LMCache MP with a Valkey Cluster L2 adapter.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "${SCRIPT_DIR}/env.sh"

if [[ -z "${VALKEY_STARTUP_NODES:-}" ]]; then
  echo "VALKEY_STARTUP_NODES is required (host:port[,host:port...])" >&2
  exit 2
fi

L2_ADAPTER_JSON="$({
  VALKEY_STARTUP_NODES="${VALKEY_STARTUP_NODES}" \
  VALKEY_CLUSTER_MODE="${VALKEY_CLUSTER_MODE}" \
  VALKEY_NUM_WORKERS="${VALKEY_NUM_WORKERS}" \
  VALKEY_REQUEST_TIMEOUT="${VALKEY_REQUEST_TIMEOUT}" \
  VALKEY_CONNECTION_TIMEOUT="${VALKEY_CONNECTION_TIMEOUT}" \
  VALKEY_KEY_PREFIX="${VALKEY_KEY_PREFIX}" \
  VALKEY_MAX_CAPACITY_GB="${VALKEY_MAX_CAPACITY_GB}" \
  python3 - <<'PY'
import json
import os

cluster_mode = os.environ["VALKEY_CLUSTER_MODE"].strip().lower()
if cluster_mode not in {"true", "false"}:
    raise SystemExit("VALKEY_CLUSTER_MODE must be true or false")

print(json.dumps({
    "type": "valkey",
    "startup_nodes": os.environ["VALKEY_STARTUP_NODES"],
    "cluster_mode": cluster_mode == "true",
    "num_workers": int(os.environ["VALKEY_NUM_WORKERS"]),
    "key_prefix": os.environ["VALKEY_KEY_PREFIX"],
    "request_timeout": float(os.environ["VALKEY_REQUEST_TIMEOUT"]),
    "connection_timeout": float(os.environ["VALKEY_CONNECTION_TIMEOUT"]),
    "max_capacity_gb": float(os.environ["VALKEY_MAX_CAPACITY_GB"]),
}, separators=(",", ":")))
PY
})"

echo "Starting LMCache MP with Valkey L2"
echo "  ZMQ: ${LMCACHE_MP_SERVER_URL}"
echo "  metrics: ${LMCACHE_URL}/metrics"
echo "  chunk_size=${LMCACHE_CHUNK_SIZE} l1_size_gb=${LMCACHE_L1_SIZE_GB} init=${LMCACHE_L1_INIT_SIZE_GB}"
echo "  l2_store_policy=${LMCACHE_L2_STORE_POLICY}"
echo "  cluster_mode=${VALKEY_CLUSTER_MODE} workers=${VALKEY_NUM_WORKERS}"
echo "  startup_nodes=${VALKEY_STARTUP_NODES}"
echo "  key_prefix=${VALKEY_KEY_PREFIX}"

exec lmcache server \
  --port "${LMCACHE_MP_PORT}" \
  --http-port "${LMCACHE_HTTP_PORT}" \
  --chunk-size "${LMCACHE_CHUNK_SIZE}" \
  --l1-size-gb "${LMCACHE_L1_SIZE_GB}" \
  --l1-init-size-gb "${LMCACHE_L1_INIT_SIZE_GB}" \
  --max-workers "${LMCACHE_MAX_WORKERS}" \
  --eviction-policy LRU \
  --l2-store-policy "${LMCACHE_L2_STORE_POLICY}" \
  --l2-prefetch-policy default \
  --l2-prefetch-max-in-flight "${LMCACHE_L2_PREFETCH_MAX_IN_FLIGHT}" \
  --l2-adapter "${L2_ADAPTER_JSON}"
