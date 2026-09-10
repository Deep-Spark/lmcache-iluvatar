#!/usr/bin/env bash
# Copyright (c) 2026, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

# Manual integration test: aggregated vLLM + LMCache MP + Valkey Cluster L2.
#
# This script is intentionally not registered in examples/ci/manifest.json.
# It only stops process groups that it starts itself. If a required port is
# already occupied, the test fails instead of killing the existing listener.

set -euo pipefail

EXAMPLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VALKEY_KEY_PREFIX_WAS_SET="${VALKEY_KEY_PREFIX+x}"
# shellcheck source=env.sh
source "${EXAMPLE_DIR}/env.sh"
# Reuse process-start and readiness helpers, but deliberately avoid the shared
# global cleanup because it may kill unrelated vLLM processes/port owners.
# shellcheck source=../../ci/lib.sh
source "${EXAMPLE_DIR}/../../ci/lib.sh"

# Keep the shared helpers' messages accurate for a manually invoked test.
ci_log() {
  echo "[valkey-l2] $*" >&2
}

RUN_TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
TEST_LOG_DIR="${TEST_LOG_DIR:-${EXAMPLE_DIR}/logs/valkey-cluster-${RUN_TIMESTAMP}}"
STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-1800}"
REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-600}"
CLEANUP_TIMEOUT="${CLEANUP_TIMEOUT:-120}"

export LMCACHE_CI_LOG_DIR="${TEST_LOG_DIR}"
export LMCACHE_CI_STARTUP_TIMEOUT="${STARTUP_TIMEOUT}"
export LMCACHE_CI_CLEANUP_TIMEOUT="${CLEANUP_TIMEOUT}"

if [[ -z "${VALKEY_STARTUP_NODES:-}" ]]; then
  echo "VALKEY_STARTUP_NODES is required (host:port[,host:port...])" >&2
  exit 2
fi
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo "CUDA_VISIBLE_DEVICES is required; explicitly select the ${TENSOR_PARALLEL} test GPU(s)" >&2
  exit 2
fi
if [[ -z "${VALKEY_KEY_PREFIX_WAS_SET}" ]]; then
  export VALKEY_KEY_PREFIX="${VALKEY_KEY_PREFIX}-${RUN_TIMESTAMP}-$$"
fi

mkdir -p "${TEST_LOG_DIR}"

python3 - <<'PY'
from importlib.metadata import PackageNotFoundError, version

try:
    installed = version("valkey-glide-sync")
except PackageNotFoundError as exc:
    raise SystemExit("valkey-glide-sync>=2.3 is required") from exc

try:
    from packaging.version import Version
except ImportError as exc:
    raise SystemExit("packaging is required to validate valkey-glide-sync") from exc

if Version(installed) < Version("2.3"):
    raise SystemExit(f"valkey-glide-sync>=2.3 is required, found {installed}")

try:
    import glide_sync  # noqa: F401
except ImportError as exc:
    raise SystemExit(
        f"valkey-glide-sync {installed} is installed but glide_sync cannot be imported"
    ) from exc

print(f"valkey-glide-sync {installed}: OK")
PY

required_ports=("${LMCACHE_MP_PORT}" "${LMCACHE_HTTP_PORT}" "${SERVER_PORT}")

require_ports_free() {
  local port
  for port in "${required_ports[@]}"; do
    if ci_port_is_open "127.0.0.1" "${port}"; then
      echo "run_test.sh: required port 127.0.0.1:${port} is already occupied; existing service was not changed" >&2
      return 1
    fi
  done
}

stop_owned_processes() {
  local pid
  local end
  local still_running

  if ((${#CI_PIDS[@]} == 0)); then
    return 0
  fi

  ci_log "stopping ${#CI_PIDS[@]} process group(s) started by this test"
  for pid in "${CI_PIDS[@]}"; do
    kill -TERM -- "-${pid}" 2>/dev/null || kill -TERM "${pid}" 2>/dev/null || true
  done

  end=$((SECONDS + CLEANUP_TIMEOUT))
  while ((SECONDS < end)); do
    still_running=0
    for pid in "${CI_PIDS[@]}"; do
      if kill -0 "${pid}" 2>/dev/null; then
        still_running=1
        break
      fi
    done
    if ((still_running == 0)); then
      break
    fi
    sleep 2
  done

  for pid in "${CI_PIDS[@]}"; do
    if kill -0 "${pid}" 2>/dev/null; then
      ci_log "process group ${pid} did not stop in ${CLEANUP_TIMEOUT}s; sending SIGKILL to that owned group"
      kill -KILL -- "-${pid}" 2>/dev/null || kill -KILL "${pid}" 2>/dev/null || true
    fi
  done
  for pid in "${CI_PIDS[@]}"; do
    wait "${pid}" 2>/dev/null || true
  done
  CI_PIDS=()

  if ! ci_wait_ports_closed "127.0.0.1" "${required_ports[@]}"; then
    ci_log "WARNING: an expected port is still open; no unowned listener was killed"
    return 1
  fi
}

trap 'stop_owned_processes || true' EXIT

wait_for_log_regex() {
  local log_file="$1"
  local regex="$2"
  local description="$3"
  local timeout="${4:-${STARTUP_TIMEOUT}}"
  local end=$((SECONDS + timeout))

  ci_log "waiting for ${description} in ${log_file} (timeout=${timeout}s)"
  while ((SECONDS < end)); do
    if [[ -f "${log_file}" ]] && grep -Eq "${regex}" "${log_file}"; then
      ci_log "log evidence found: ${description}"
      return 0
    fi
    sleep 2
  done

  echo "run_test.sh: timed out waiting for ${description} in ${log_file}" >&2
  [[ -f "${log_file}" ]] && tail -n 80 "${log_file}" >&2
  return 1
}

wait_for_active_sessions() {
  local expected="$1"
  local timeout="${2:-${STARTUP_TIMEOUT}}"
  local end=$((SECONDS + timeout))
  local status_json
  local active_sessions

  ci_log "waiting for ${expected} LMCache MP session(s) (timeout=${timeout}s)"
  while ((SECONDS < end)); do
    active_sessions=""
    if status_json="$(ci_curl_local "${LMCACHE_URL}/status" 2>/dev/null)"; then
      active_sessions="$(
        python3 -c \
          'import json,sys; print(int(json.load(sys.stdin).get("active_sessions", 0)))' \
          <<<"${status_json}" 2>/dev/null
      )" || active_sessions=""
    fi
    if [[ "${active_sessions}" =~ ^[0-9]+$ ]] && ((active_sessions >= expected)); then
      ci_log "LMCache MP sessions ready: ${active_sessions}/${expected}"
      return 0
    fi
    sleep 2
  done

  echo "run_test.sh: timed out waiting for ${expected} LMCache MP session(s); last observed=${active_sessions:-unknown}" >&2
  return 1
}

run_request() {
  local label="$1"
  local output_path="${TEST_LOG_DIR}/${label}.json"

  ci_log "sending ${label^^} semantic request"
  python3 "${EXAMPLE_DIR}/semantic_l2_request.py" \
    --base-url "http://${SERVER_HOST}:${SERVER_PORT}" \
    --model "${SERVED_MODEL_NAME}" \
    --label "${label}" \
    --output "${output_path}" \
    --expected "${EXPECTED_ANSWER}" \
    --prompt-repeat "${PROMPT_REPEAT}" \
    --min-prompt-tokens "${MIN_PROMPT_TOKENS}" \
    --max-tokens "${REQUEST_MAX_TOKENS}" \
    --timeout "${REQUEST_TIMEOUT}"
}

start_stack() {
  local phase="$1"
  local lmcache_name="lmcache_${phase}"
  local vllm_name="vllm_${phase}"
  local lmcache_log

  require_ports_free

  ci_start_bg "${lmcache_name}" bash "${EXAMPLE_DIR}/start_lmcache_mp.sh"
  lmcache_log="$(ci_log_path "${lmcache_name}")"
  ci_wait_for_port "127.0.0.1" "${LMCACHE_MP_PORT}" "${STARTUP_TIMEOUT}"
  ci_wait_for_http "${LMCACHE_URL}/metrics" "${STARTUP_TIMEOUT}"
  wait_for_log_regex \
    "${lmcache_log}" \
    'ValkeyL2Adapter ready:.*cluster_mode=True' \
    "Valkey Cluster adapter readiness (${phase})"

  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
    ci_start_bg "${vllm_name}" bash "${EXAMPLE_DIR}/start_server.sh"
  ci_wait_for_http \
    "http://${SERVER_HOST}:${SERVER_PORT}/health" \
    "${STARTUP_TIMEOUT}"
  wait_for_active_sessions "${TENSOR_PARALLEL}" "${STARTUP_TIMEOUT}"
}

ci_log "manual Valkey Cluster L2 test"
ci_log "  model=${MODEL_PATH} served_model=${SERVED_MODEL_NAME}"
ci_log "  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} TP=${TENSOR_PARALLEL}"
ci_log "  startup_nodes=${VALKEY_STARTUP_NODES}"
ci_log "  key_prefix=${VALKEY_KEY_PREFIX}"
ci_log "  logs=${TEST_LOG_DIR}"

start_stack cold
run_request cold
cold_lmcache_log="$(ci_log_path lmcache_cold)"
wait_for_log_regex "${cold_lmcache_log}" 'Stored' "COLD L2 store"

ci_log "restarting both vLLM and LMCache to clear GPU/L1 caches"
stop_owned_processes
require_ports_free

start_stack warm
run_request warm
warm_lmcache_log="$(ci_log_path lmcache_warm)"
warm_vllm_log="$(ci_log_path vllm_warm)"
wait_for_log_regex \
  "${warm_lmcache_log}" \
  'Prefetch request completed \(L1\+L2\).*\([0-9]+ L1, [1-9][0-9]* L2\)' \
  "positive WARM L2 prefetch"

python3 - \
  "${TEST_LOG_DIR}/cold.json" \
  "${TEST_LOG_DIR}/warm.json" \
  "${warm_vllm_log}" \
  "${TENSOR_PARALLEL}" <<'PY'
import json
import re
import sys
from pathlib import Path

cold_path, warm_path, vllm_log_path = map(Path, sys.argv[1:4])
tensor_parallel = int(sys.argv[4])
cold = json.loads(cold_path.read_text(encoding="utf-8"))
warm = json.loads(warm_path.read_text(encoding="utf-8"))

log = vllm_log_path.read_text(encoding="utf-8", errors="replace")
retrieved_by_tp = {}
for tp, count in re.findall(r"Worker_TP(\d+).*Retrieved ([1-9][0-9]*)", log):
    retrieved_by_tp[int(tp)] = max(retrieved_by_tp.get(int(tp), 0), int(count))

missing = [rank for rank in range(tensor_parallel) if retrieved_by_tp.get(rank, 0) <= 0]
if missing:
    raise SystemExit(
        f"WARM request has no positive Retrieved evidence for TP rank(s) {missing}; "
        f"observed={retrieved_by_tp}"
    )

print(f"COLD answer: {cold['text']!r}; WARM answer: {warm['text']!r}")
print(f"WARM retrieved tokens by TP rank: {retrieved_by_tp}")
PY

ci_log "manual Valkey Cluster L2 test passed"
ci_log "evidence saved under ${TEST_LOG_DIR}"
