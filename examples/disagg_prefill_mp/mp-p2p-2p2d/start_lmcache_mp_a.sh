#!/usr/bin/env bash
# Copyright (c) 2026, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

# LMCache Server A — pair-1 (P1+D1) on IP_A. P2P enabled via --p2p-advertise-url.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "${SCRIPT_DIR}/env.sh"

echo "Starting LMCache Server A (instance=${LMCACHE_INSTANCE_ID_A})"
echo "  ZMQ=${LMCACHE_ZMQ_HOST}:${LMCACHE_MP_PORT_A} HTTP=${LMCACHE_HTTP_HOST_A}:${LMCACHE_HTTP_PORT_A}"
echo "  P2P listen=${LMCACHE_P2P_LISTEN_URL_A} advertise=${LMCACHE_P2P_ADVERTISE_URL_A}"
echo "  coordinator=${LMCACHE_COORDINATOR_URL} advertise_ip=${LMCACHE_COORDINATOR_ADVERTISE_IP_A}"
echo "  L1=${LMCACHE_L1_SIZE_GB}GB align=${LMCACHE_L1_ALIGN_BYTES}"

exec lmcache server \
  --host "${LMCACHE_ZMQ_HOST}" \
  --port "${LMCACHE_MP_PORT_A}" \
  --http-host "${LMCACHE_HTTP_HOST_A}" \
  --http-port "${LMCACHE_HTTP_PORT_A}" \
  --chunk-size "${LMCACHE_CHUNK_SIZE}" \
  --l1-size-gb "${LMCACHE_L1_SIZE_GB}" \
  --l1-align-bytes "${LMCACHE_L1_ALIGN_BYTES}" \
  --max-workers "${LMCACHE_MAX_WORKERS}" \
  --eviction-policy LRU \
  --instance-id "${LMCACHE_INSTANCE_ID_A}" \
  --coordinator-url "${LMCACHE_COORDINATOR_URL}" \
  --coordinator-advertise-ip "${LMCACHE_COORDINATOR_ADVERTISE_IP_A}" \
  --p2p-advertise-url "${LMCACHE_P2P_ADVERTISE_URL_A}" \
  --p2p-listen-url "${LMCACHE_P2P_LISTEN_URL_A}"
