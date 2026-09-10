#!/usr/bin/env bash
# Copyright (c) 2026, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

# Start the Mooncake Store metadata master.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "${SCRIPT_DIR}/env.sh"

command -v mooncake_master >/dev/null || {
  echo "mooncake_master is not installed or not in PATH" >&2
  exit 2
}

echo "Starting Mooncake Master"
echo "  RPC port: ${MOONCAKE_MASTER_PORT}"
echo "  metrics port: ${MOONCAKE_MASTER_METRICS_PORT}"

exec mooncake_master \
  -v=1 \
  --rpc_port="${MOONCAKE_MASTER_PORT}" \
  --metrics_port="${MOONCAKE_MASTER_METRICS_PORT}"
