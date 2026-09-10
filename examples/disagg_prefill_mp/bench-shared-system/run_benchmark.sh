#!/bin/bash
# Copyright (c) 2026, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

# Full shared-system multi-user benchmark (manual).
#
# Prerequisites: MP 2P2D stack running (see README.md).
#
# Usage:
#   bash run_benchmark.sh
#   NUM_USERS=16 BENCH_ROUNDS=2 bash run_benchmark.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "${SCRIPT_DIR}/env.sh"

RESULTS_DIR="${RESULTS_DIR:-${SCRIPT_DIR}/results/bench-$(date +%Y%m%d-%H%M%S)}"
mkdir -p "${RESULTS_DIR}"

echo "=== MP shared-system concurrent (full) ==="
echo "  model=${MODEL_PATH}  served=${MODEL_NAME}"
echo "  proxy=${PROXY_BASE_URL}  lmcache=${LMCACHE_URL}"
echo "  users=${NUM_USERS}  warmup=${WARMUP_USERS}  rounds=${BENCH_ROUNDS}"
echo "  shared_tokens=${SHARED_SYSTEM_TOKENS}  results=${RESULTS_DIR}"

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
  --output-dir "${RESULTS_DIR}"
