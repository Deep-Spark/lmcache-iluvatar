#!/usr/bin/env bash
# Copyright (c) 2026, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

# Start disagg proxy + telemetry for 2P2D MP smoke.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "${SCRIPT_DIR}/env.sh"

python3 "${MP_ROOT}/disagg_proxy_server.py" \
  --port "${PROXY_PORT}" \
  --prefiller-host localhost --prefiller-port "${PREFILLER1_PORT},${PREFILLER2_PORT}" \
  --decoder-host localhost --decoder-port "${DECODER1_PORT},${DECODER2_PORT}" \
  --telemetry-port "${TELEMETRY_PORT}"
