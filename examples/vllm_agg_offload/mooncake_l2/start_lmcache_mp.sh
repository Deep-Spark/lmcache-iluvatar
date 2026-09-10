#!/usr/bin/env bash
# Copyright (c) 2026, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

# Start LMCache MP with Mooncake Store as its external L2 backend.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "${SCRIPT_DIR}/env.sh"

if [[ -z "${MOONCAKE_REQUESTER_HOST}" ]]; then
  echo "MOONCAKE_REQUESTER_HOST is required and must be reachable by Mooncake peers" >&2
  exit 2
fi
case "${MOONCAKE_PROTOCOL}" in
  tcp) ;;
  rdma)
    if [[ -z "${MOONCAKE_RDMA_DEVICES}" ]]; then
      echo "MOONCAKE_RDMA_DEVICES is required when MOONCAKE_PROTOCOL=rdma" >&2
      exit 2
    fi
    ;;
  *)
    echo "MOONCAKE_PROTOCOL must be tcp or rdma, got: ${MOONCAKE_PROTOCOL}" >&2
    exit 2
    ;;
esac

L2_ADAPTER_JSON="$({
  MOONCAKE_REQUESTER_HOST="${MOONCAKE_REQUESTER_HOST}" \
  MOONCAKE_MASTER_ADDR="${MOONCAKE_MASTER_ADDR}" \
  MOONCAKE_METADATA_SERVER="${MOONCAKE_METADATA_SERVER}" \
  MOONCAKE_PROTOCOL="${MOONCAKE_PROTOCOL}" \
  MOONCAKE_RDMA_DEVICES="${MOONCAKE_RDMA_DEVICES}" \
  MOONCAKE_TENANT_ID="${MOONCAKE_TENANT_ID}" \
  LMCACHE_LOCAL_BUFFER_SIZE="${LMCACHE_LOCAL_BUFFER_SIZE}" \
  LMCACHE_L2_NUM_WORKERS="${LMCACHE_L2_NUM_WORKERS}" \
  python3 - <<'PY'
import json
import os

print(json.dumps({
    "type": "mooncake_store",
    "config": {
        "local_hostname": os.environ["MOONCAKE_REQUESTER_HOST"],
        "metadata_server": os.environ["MOONCAKE_METADATA_SERVER"],
        "master_server_addr": os.environ["MOONCAKE_MASTER_ADDR"],
        "protocol": os.environ["MOONCAKE_PROTOCOL"],
        "rdma_devices": os.environ["MOONCAKE_RDMA_DEVICES"],
        "tenant_id": os.environ["MOONCAKE_TENANT_ID"],
        "global_segment_size": 0,
        "local_buffer_size": int(os.environ["LMCACHE_LOCAL_BUFFER_SIZE"]),
        "num_workers": int(os.environ["LMCACHE_L2_NUM_WORKERS"]),
    },
}, separators=(",", ":")))
PY
})"

if [[ "${PRINT_L2_ADAPTER_JSON:-0}" == "1" ]]; then
  printf '%s\n' "${L2_ADAPTER_JSON}"
  exit 0
fi

echo "Starting LMCache MP with Mooncake L2"
echo "  ZMQ: ${LMCACHE_MP_SERVER_URL} metrics: ${LMCACHE_URL}/metrics"
echo "  master: ${MOONCAKE_MASTER_ADDR} tenant: ${MOONCAKE_TENANT_ID}"
echo "  requester: ${MOONCAKE_REQUESTER_HOST} protocol: ${MOONCAKE_PROTOCOL}"
echo "  chunk_size=${LMCACHE_CHUNK_SIZE}"

exec lmcache server \
  --port "${LMCACHE_MP_PORT}" \
  --http-port "${LMCACHE_HTTP_PORT}" \
  --chunk-size "${LMCACHE_CHUNK_SIZE}" \
  --l1-size-gb "${LMCACHE_L1_SIZE_GB}" \
  --max-workers "${LMCACHE_MAX_WORKERS}" \
  --eviction-policy LRU \
  --l2-store-policy default \
  --l2-prefetch-policy default \
  --l2-prefetch-max-in-flight "${LMCACHE_L2_PREFETCH_MAX_IN_FLIGHT}" \
  --l2-adapter "${L2_ADAPTER_JSON}"
