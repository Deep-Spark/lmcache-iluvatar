#!/usr/bin/env bash
# Copyright (c) 2026, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
# Start LMCache controller for P2P KV cache sharing — Iluvatar edition.
#
# The controller tracks which peer has which chunk hash.
# All processes (controller + vLLM instances) MUST share the same PYTHONHASHSEED.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "${SCRIPT_DIR}/env.sh"

echo "Starting LMCache controller..."
echo "  PYTHONHASHSEED: ${PYTHONHASHSEED}"
echo "  API:  ${CONTROLLER_HOST}:${CONTROLLER_API_PORT}"
echo "  Pull: ${CONTROLLER_HOST}:${CONTROLLER_PULL_PORT}"
echo "  Reply: ${CONTROLLER_HOST}:${CONTROLLER_REPLY_PORT}"

exec lmcache_controller \
  --host "${CONTROLLER_HOST}" \
  --port "${CONTROLLER_API_PORT}" \
  --monitor-ports "{\"pull\": ${CONTROLLER_PULL_PORT}, \"reply\": ${CONTROLLER_REPLY_PORT}}"
