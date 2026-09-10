#!/usr/bin/env bash
# Copyright (c) 2026, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

# LMCache Server B — pair-2 (P2+D2) on IP_B. P2P enabled via --p2p-advertise-url.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "${SCRIPT_DIR}/env.sh"

echo "Starting LMCache Server B (instance=${LMCACHE_INSTANCE_ID_B})"
echo "  ZMQ=${LMCACHE_ZMQ_HOST}:${LMCACHE_MP_PORT_B} HTTP=${LMCACHE_HTTP_HOST_B}:${LMCACHE_HTTP_PORT_B}"
echo "  P2P listen=${LMCACHE_P2P_LISTEN_URL_B} advertise=${LMCACHE_P2P_ADVERTISE_URL_B}"
echo "  coordinator=${LMCACHE_COORDINATOR_URL} advertise_ip=${LMCACHE_COORDINATOR_ADVERTISE_IP_B}"
echo "  L1=${LMCACHE_L1_SIZE_GB}GB align=${LMCACHE_L1_ALIGN_BYTES}"

exec lmcache server \
  --host "${LMCACHE_ZMQ_HOST}" \
  --port "${LMCACHE_MP_PORT_B}" \
  --http-host "${LMCACHE_HTTP_HOST_B}" \
  --http-port "${LMCACHE_HTTP_PORT_B}" \
  --chunk-size "${LMCACHE_CHUNK_SIZE}" \
  --l1-size-gb "${LMCACHE_L1_SIZE_GB}" \
  --l1-align-bytes "${LMCACHE_L1_ALIGN_BYTES}" \
  --max-workers "${LMCACHE_MAX_WORKERS}" \
  --eviction-policy LRU \
  --instance-id "${LMCACHE_INSTANCE_ID_B}" \
  --coordinator-url "${LMCACHE_COORDINATOR_URL}" \
  --coordinator-advertise-ip "${LMCACHE_COORDINATOR_ADVERTISE_IP_B}" \
  --p2p-advertise-url "${LMCACHE_P2P_ADVERTISE_URL_B}" \
  --p2p-listen-url "${LMCACHE_P2P_LISTEN_URL_B}"
