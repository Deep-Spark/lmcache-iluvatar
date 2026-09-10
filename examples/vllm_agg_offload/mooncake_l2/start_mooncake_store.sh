#!/usr/bin/env bash
# Copyright (c) 2026, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

# Start one standalone Mooncake Store process that contributes a DRAM segment.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "${SCRIPT_DIR}/env.sh"

if [[ -z "${MOONCAKE_STORE_HOST}" ]]; then
  echo "MOONCAKE_STORE_HOST is required and must be reachable by all requesters" >&2
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

command -v mooncake_client >/dev/null || {
  echo "mooncake_client is not installed or not in PATH" >&2
  exit 2
}

args=(
  "--host=${MOONCAKE_STORE_HOST}"
  "--port=${MOONCAKE_STORE_PORT}"
  "--global_segment_size=${MOONCAKE_STORE_SIZE}"
  "--master_server_address=${MOONCAKE_MASTER_ADDR}"
  "--metadata_server=${MOONCAKE_METADATA_SERVER}"
  "--protocol=${MOONCAKE_PROTOCOL}"
  "--threads=${MOONCAKE_STORE_THREADS}"
  "--tenant_id=${MOONCAKE_TENANT_ID}"
  "--enable_http_server=true"
  "--http_port=${MOONCAKE_STORE_HTTP_PORT}"
)
if [[ "${MOONCAKE_PROTOCOL}" == "rdma" ]]; then
  args+=("--device_names=${MOONCAKE_RDMA_DEVICES}")
fi

echo "Starting Mooncake Store"
echo "  endpoint: ${MOONCAKE_STORE_HOST}:${MOONCAKE_STORE_PORT}"
echo "  master: ${MOONCAKE_MASTER_ADDR} tenant: ${MOONCAKE_TENANT_ID}"
echo "  segment: ${MOONCAKE_STORE_SIZE} protocol: ${MOONCAKE_PROTOCOL}"
echo "  RDMA devices: ${MOONCAKE_RDMA_DEVICES:-<none>}"

exec mooncake_client "${args[@]}"
