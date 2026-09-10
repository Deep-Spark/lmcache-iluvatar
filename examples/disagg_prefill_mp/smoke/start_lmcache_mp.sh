#!/usr/bin/env bash
# Copyright (c) 2026, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

# Start the shared LMCache MP server for 2P2D smoke.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "${SCRIPT_DIR}/env.sh"

exec lmcache server \
  --port "${LMCACHE_MP_PORT}" \
  --chunk-size "${LMCACHE_CHUNK_SIZE}" \
  --l1-size-gb "${LMCACHE_L1_SIZE_GB}" \
  --max-gpu-workers "${LMCACHE_MAX_GPU_WORKERS}" \
  --eviction-policy LRU
