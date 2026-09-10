#!/usr/bin/env bash
# Copyright (c) 2026, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

# 2P2D MP P2P smoke: HTTP OK + RR both pairs + P2P registered.
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

export BENCH_LOG_PREFIX="${BENCH_LOG_PREFIX:-[smoke-mp-p2p]}"
export SMOKE_TITLE="${SMOKE_TITLE:-=== 2P2D MP P2P (dual server) smoke ===}"

PROXY_HOST="${PROXY_HOST:-${IP_A}}"
SERVER_PORT="${SERVER_PORT:-${PROXY_PORT}}"
# Need ≥4 runs so same-idx RR hits both pairs at least twice.
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
  local host="$1"
  local port="$2"
  local label="$3"
  local url="http://${host}:${port}/metrics"
  local val
  for name in \
    "request_success_total" \
    "prompt_tokens_total" \
    "http_requests_total"
  do
    val="$(_metric_sum "${url}" "${name}")"
    if [[ -n "${val}" ]] && python3 -c "import sys; sys.exit(0 if float('${val}') > 0 else 1)"; then
      bench_log "RR check: ${label} (${host}:${port}) ${name}=${val} (>0)"
      return 0
    fi
  done
  bench_log "RR check FAIL: ${label} (${host}:${port}) showed no request traffic in /metrics"
  return 1
}

_check_p2p_registered_once() {
  local url="$1"
  local label="$2"
  local json
  json="$(curl -sf "${url}/status" 2>/dev/null || true)"
  if [[ -z "${json}" ]]; then
    bench_log "P2P check: ${label} ${url}/status unreachable"
    return 1
  fi
  python3 -c '
import json, sys
data = json.loads(sys.argv[1])
state = data.get("p2p_state", "")
count = data.get("p2p_peer_count", 0)
label = sys.argv[2]
print(f"P2P check: {label} p2p_state={state} p2p_peer_count={count}", file=sys.stderr)
sys.exit(0 if state == "registered" and int(count) >= 1 else 1)
' "${json}" "${label}"
}

_check_coordinator_instances_once() {
  local url="${LMCACHE_COORDINATOR_URL}/instances"
  local json
  json="$(curl -sf "${url}" 2>/dev/null || true)"
  if [[ -z "${json}" ]]; then
    bench_log "Coordinator check: ${url} unreachable"
    return 1
  fi
  python3 -c '
import json, sys
data = json.loads(sys.argv[1])
# Accept list or {"instances": [...]} shapes.
if isinstance(data, dict):
    items = data.get("instances") or data.get("items") or list(data.values())
    if items and isinstance(items[0], dict):
        pass
    elif all(isinstance(v, dict) for v in data.values()):
        items = list(data.values())
    else:
        items = []
elif isinstance(data, list):
    items = data
else:
    items = []
n = len(items)
advertised = sum(1 for i in items if isinstance(i, dict) and i.get("p2p_advertised_url"))
print(f"Coordinator: {n} instance(s), {advertised} with p2p_advertised_url", file=sys.stderr)
sys.exit(0 if n >= 2 and advertised >= 2 else 1)
' "${json}"
}

# Peer discovery is async (coordinator heartbeat / poll). Retry briefly so
# smoke does not flake right after Server A/B come up.
_wait_for_p2p_ready() {
  local timeout_sec="${P2P_READY_TIMEOUT_SEC:-30}"
  local deadline=$((SECONDS + timeout_sec))
  while (( SECONDS < deadline )); do
    if _check_coordinator_instances_once \
      && _check_p2p_registered_once "${LMCACHE_URL_A}" "Server A" \
      && _check_p2p_registered_once "${LMCACHE_URL_B}" "Server B"; then
      return 0
    fi
    sleep 2
  done
  bench_log "P2P check FAIL: coordinator/peers not ready within ${timeout_sec}s"
  return 1
}

if [[ "${SKIP_STACK_WAIT:-0}" != "1" ]]; then
  bench_wait_for_2p2d_stack "${PROXY_HOST}" "${SERVER_PORT}" "${IP_A}" "${IP_B}"
fi

bench_log "Checking coordinator membership + P2P registration ..."
_wait_for_p2p_ready

python3 "${COMMON_DIR}/smoke_long_prompt_cache.py" "$@"

bench_log "Verifying RR coverage (NUM_RUNS=${NUM_RUNS}) IP_A=${IP_A} IP_B=${IP_B} ..."
_instance_saw_traffic "${IP_A}" "${PREFILLER1_PORT}" "P1"
_instance_saw_traffic "${IP_B}" "${PREFILLER2_PORT}" "P2"
_instance_saw_traffic "${IP_A}" "${DECODER1_PORT}" "D1"
_instance_saw_traffic "${IP_B}" "${DECODER2_PORT}" "D2"
bench_log "Smoke post-checks passed (HTTP already verified by smoke_long_prompt_cache.py)."
