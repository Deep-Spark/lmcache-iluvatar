#!/usr/bin/env bash
# Copyright (c) 2026, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

# Start the 2P2D disagg proxy (pure MP — no NIXL). Runs on IP_A.
#
# disagg_proxy_server.py uses the same round-robin idx for P and D, so
# request i → (P1,D1) or (P2,D2) and stays on one lmcache server.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "${SCRIPT_DIR}/env.sh"

echo "Starting MP P2P proxy on :${PROXY_PORT} (same-idx RR: P1↔D1 / P2↔D2)"
echo "  prefiller-hosts=${PREFILLER_HOSTS} decoder-hosts=${DECODER_HOSTS}"

python3 "${MP_ROOT}/disagg_proxy_server.py" \
  --host "${BIND_HOST}" \
  --port "${PROXY_PORT}" \
  --prefiller-host "${PREFILLER_HOSTS}" \
  --prefiller-port "${PREFILLER1_PORT},${PREFILLER2_PORT}" \
  --num-prefillers 2 \
  --decoder-host "${DECODER_HOSTS}" \
  --decoder-port "${DECODER1_PORT},${DECODER2_PORT}" \
  --num-decoders 2 \
  --telemetry-port "${TELEMETRY_PORT}"
