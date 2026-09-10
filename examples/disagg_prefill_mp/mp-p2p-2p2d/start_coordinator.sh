#!/usr/bin/env bash
# Copyright (c) 2026, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

# Start the MP coordinator (membership only — no KV data path).
# Runs on IP_A (primary).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "${SCRIPT_DIR}/env.sh"

echo "Starting lmcache coordinator on ${LMCACHE_COORDINATOR_HOST}:${LMCACHE_COORDINATOR_PORT}"
echo "  peers will use ${LMCACHE_COORDINATOR_URL} (IP_A=${IP_A})"

exec lmcache coordinator \
  --host "${LMCACHE_COORDINATOR_HOST}" \
  --port "${LMCACHE_COORDINATOR_PORT}"
