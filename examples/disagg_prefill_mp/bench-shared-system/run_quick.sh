#!/bin/bash
# Copyright (c) 2026, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

# Quick shared-system multi-user benchmark (CI / smoke scale).
#
# Prerequisites: MP 2P2D stack running (see README.md).
#
# Usage:
#   bash run_quick.sh
#   NUM_USERS=8 SHARED_SYSTEM_TOKENS=1200 bash run_quick.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "${SCRIPT_DIR}/env.sh"

export NUM_USERS="${NUM_USERS:-8}"
export WARMUP_USERS="${WARMUP_USERS:-1}"
export BENCH_ROUNDS="${BENCH_ROUNDS:-1}"
export SHARED_SYSTEM_TOKENS="${SHARED_SYSTEM_TOKENS:-1200}"
export USER_UNIQUE_TOKENS="${USER_UNIQUE_TOKENS:-24}"
export MAX_TOKENS="${MAX_TOKENS:-32}"
export MAX_INFLIGHT="${MAX_INFLIGHT:-8}"

RESULTS_DIR="${RESULTS_DIR:-${SCRIPT_DIR}/results/quick-$(date +%Y%m%d-%H%M%S)}"
mkdir -p "${RESULTS_DIR}"

echo "=== MP shared-system concurrent (quick) ==="
echo "  model=${MODEL_NAME}  proxy=${PROXY_BASE_URL}"
echo "  users=${NUM_USERS}  shared_tokens=${SHARED_SYSTEM_TOKENS}"
echo "  results=${RESULTS_DIR}"

if ! curl -sf --noproxy '*' "http://127.0.0.1:${PROXY_PORT}/v1/models" >/dev/null; then
  echo "Proxy is not reachable on port ${PROXY_PORT}. Start the stack first (see README.md)." >&2
  exit 1
fi

exec python3 "${SCRIPT_DIR}/bench_shared_system.py" \
  --base-url "${PROXY_BASE_URL}" \
  --model "${MODEL_NAME}" \
  --model-path "${MODEL_PATH}" \
  --lmcache-url "${LMCACHE_URL}" \
  --shared-system-tokens "${SHARED_SYSTEM_TOKENS}" \
  --user-unique-tokens "${USER_UNIQUE_TOKENS}" \
  --user-query "${USER_QUERY}" \
  --max-tokens "${MAX_TOKENS}" \
  --num-users "${NUM_USERS}" \
  --warmup-users "${WARMUP_USERS}" \
  --bench-rounds "${BENCH_ROUNDS}" \
  --max-inflight "${MAX_INFLIGHT}" \
  --request-timeout "${REQUEST_TIMEOUT}" \
  --output-dir "${RESULTS_DIR}" \
  --assert-hit
