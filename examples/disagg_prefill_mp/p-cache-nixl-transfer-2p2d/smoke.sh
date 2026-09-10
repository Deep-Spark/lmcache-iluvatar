#!/usr/bin/env bash
# Copyright (c) 2026, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

# 2P2D smoke: long-prefix MP reuse + NIXL transfer + RR coverage.
#
# Usage:
#   bash smoke.sh
#   PROMPT_REPEAT=80 NUM_RUNS=4 bash smoke.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMMON_DIR="$(cd "${SCRIPT_DIR}/../../common" && pwd)"

# shellcheck source=env.sh
source "${SCRIPT_DIR}/env.sh"
# shellcheck source=../../common/bench_lib.sh
source "${COMMON_DIR}/bench_lib.sh"

export BENCH_LOG_PREFIX="${BENCH_LOG_PREFIX:-[smoke-2p2d]}"
export SMOKE_TITLE="${SMOKE_TITLE:-=== 2P2D P-side MP cache reuse + NIXL P/D transfer test ===}"

PROXY_HOST="${PROXY_HOST:-127.0.0.1}"
SERVER_PORT="${SERVER_PORT:-${PROXY_PORT}}"
# Need ≥4 runs so independent RR hits both P and both D at least once.
export NUM_RUNS="${NUM_RUNS:-4}"

_metric_sum() {
  local url="$1"
  local name="$2"
  local text
  text="$(curl -sf "${url}" 2>/dev/null || true)"
  if [[ -z "${text}" ]]; then
    echo ""
    return 0
  fi
  # Match bare name or vLLM-prefixed form (e.g. vllm:nixl_bytes_transferred_sum).
  printf '%s\n' "${text}" | python3 -c '
import sys
name = sys.argv[1]
total = 0.0
found = False
for line in sys.stdin:
    line = line.strip()
    if not line or line.startswith("#"):
        continue
    metric = line.split("{", 1)[0]
    if metric == name or metric.endswith(":" + name):
        found = True
        total += float(line.rsplit(None, 1)[-1])
print(f"{total}" if found else "")
' "${name}"
}

_instance_saw_traffic() {
  local port="$1"
  local label="$2"
  local url="http://127.0.0.1:${port}/metrics"
  local val
  for name in \
    "request_success_total" \
    "prompt_tokens_total" \
    "http_requests_total"
  do
    val="$(_metric_sum "${url}" "${name}")"
    if [[ -n "${val}" ]] && python3 -c "import sys; sys.exit(0 if float('${val}') > 0 else 1)"; then
      bench_log "RR check: ${label} (:${port}) ${name}=${val} (>0)"
      return 0
    fi
  done
  bench_log "RR check FAIL: ${label} (:${port}) showed no request traffic in /metrics"
  return 1
}

_nixl_bytes_on_any_prefiller() {
  # This vLLM exposes a Histogram: prefer _sum (total bytes), fall back to
  # _count (num xfers). Bare "nixl_bytes_transferred_count" misses the
  # "vllm:" prefix and falsely fails while transfers already happened.
  local port bytes
  for port in "${PREFILLER1_PORT}" "${PREFILLER2_PORT}"; do
    for name in \
      "nixl_bytes_transferred_sum" \
      "nixl_bytes_transferred_count"
    do
      bytes="$(_metric_sum "http://127.0.0.1:${port}/metrics" "${name}")"
      if [[ -n "${bytes}" ]] && python3 -c "import sys; sys.exit(0 if float('${bytes}') > 0 else 1)"; then
        bench_log "NIXL check: prefiller :${port} ${name}=${bytes} (>0)"
        return 0
      fi
    done
  done
  bench_log "NIXL check FAIL: nixl_bytes_transferred_{sum,count}==0 on both prefillers"
  return 1
}

if [[ "${SKIP_STACK_WAIT:-0}" != "1" ]]; then
  bench_wait_for_2p2d_stack "${PROXY_HOST}" "${SERVER_PORT}"
fi

python3 "${COMMON_DIR}/smoke_long_prompt_cache.py" "$@"

bench_log "Verifying NIXL transfer and RR coverage (NUM_RUNS=${NUM_RUNS}) ..."
_nixl_bytes_on_any_prefiller
_instance_saw_traffic "${PREFILLER1_PORT}" "P1"
_instance_saw_traffic "${PREFILLER2_PORT}" "P2"
_instance_saw_traffic "${DECODER1_PORT}" "D1"
_instance_saw_traffic "${DECODER2_PORT}" "D2"
bench_log "Smoke post-checks passed (HTTP already verified by smoke_long_prompt_cache.py)."
