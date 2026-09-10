#!/usr/bin/env bash
# Copyright (c) 2026, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

# Shared helpers for per-example ci/run.sh scripts.
#
# Expected env from the orchestrator (run_all_lmcache_iluvatar_tests.py):
#   LMCACHE_CI_LOG_DIR      - directory for service logs
#   LMCACHE_CI_STARTUP_TIMEOUT - seconds to wait for ports/health
# Optional:
#   LMCACHE_CI_GPUS           - comma-separated GPU ids

set -euo pipefail

CI_PIDS=()
CI_PORTS=()

ci_require_var() {
  local name="$1"
  if [[ -z "${!name:-}" ]]; then
    echo "ci/lib.sh: required env ${name} is not set" >&2
    exit 1
  fi
}

ci_log() {
  echo "[ci] $*" >&2
}

# curl to loopback must bypass HTTP(S)_PROXY — corporate proxies hijack 127.0.0.1.
ci_curl_local() {
  curl -sf --noproxy '*' --max-time "${CI_CURL_MAX_TIME:-5}" "$@"
}

ci_wait_for_port() {
  local host="$1" port="$2" timeout="${3:-${LMCACHE_CI_STARTUP_TIMEOUT:-600}}"
  local end=$((SECONDS + timeout))
  ci_log "waiting for ${host}:${port} (timeout=${timeout}s)"
  while ((SECONDS < end)); do
    if python3 - "${host}" "${port}" <<'PY'
import socket, sys
host, port = sys.argv[1], int(sys.argv[2])
s = socket.socket()
s.settimeout(1)
try:
    s.connect((host, port))
except OSError:
    raise SystemExit(1)
finally:
    s.close()
PY
    then
      ci_log "port ${host}:${port} is open"
      return 0
    fi
    sleep 2
  done
  echo "ci/lib.sh: timed out waiting for ${host}:${port}" >&2
  return 1
}

ci_wait_for_http() {
  local url="$1" timeout="${2:-${LMCACHE_CI_STARTUP_TIMEOUT:-600}}"
  local end=$((SECONDS + timeout))
  ci_log "waiting for HTTP ${url} (timeout=${timeout}s)"
  while ((SECONDS < end)); do
    if ci_curl_local "${url}" >/dev/null 2>&1; then
      ci_log "HTTP ready: ${url}"
      return 0
    fi
    sleep 2
  done
  echo "ci/lib.sh: timed out waiting for HTTP ${url}" >&2
  return 1
}

ci_register_ports() {
  local port
  for port in "$@"; do
    CI_PORTS+=("${port}")
  done
}

ci_port_is_open() {
  local host="$1" port="$2"
  python3 - "${host}" "${port}" <<'PY'
import socket, sys
host, port = sys.argv[1], int(sys.argv[2])
s = socket.socket()
s.settimeout(1)
try:
    s.connect((host, port))
except OSError:
    raise SystemExit(1)
finally:
    s.close()
PY
}

ci_wait_ports_closed() {
  local host="${1:-127.0.0.1}"
  shift
  local ports=("$@")
  local timeout="${LMCACHE_CI_CLEANUP_TIMEOUT:-120}"
  local end=$((SECONDS + timeout))
  local still_open=0

  if ((${#ports[@]} == 0)); then
    return 0
  fi

  ci_log "waiting for ports to close on ${host}: ${ports[*]} (timeout=${timeout}s)"
  while ((SECONDS < end)); do
    still_open=0
    for port in "${ports[@]}"; do
      if ci_port_is_open "${host}" "${port}"; then
        still_open=1
        break
      fi
    done
    if ((still_open == 0)); then
      ci_log "all registered ports closed on ${host}"
      return 0
    fi
    sleep 2
  done

  for port in "${ports[@]}"; do
    if ci_port_is_open "${host}" "${port}"; then
      echo "ci/lib.sh: port ${host}:${port} still open after cleanup" >&2
    fi
  done
  return 1
}

ci_log_path() {
  local name="$1"
  ci_require_var LMCACHE_CI_LOG_DIR
  echo "${LMCACHE_CI_LOG_DIR}/${name}.log"
}

ci_start_bg() {
  local name="$1"
  shift
  local log_path
  log_path="$(ci_log_path "${name}")"
  ci_log "starting ${name} -> ${log_path}"
  # setsid: one session per service so ci_cleanup can SIGTERM the whole tree.
  # Do not echo log_path here — command substitution runs in a subshell and
  # would drop updates to CI_PIDS. Callers use ci_log_path "${name}" instead.
  #
  # Truncate once, then append: TP/multiprocess workers (spawn) may reopen the
  # same path; shell '>' per process would leave only the last writer's output.
  : >"${log_path}"
  setsid stdbuf -oL -eL "$@" >>"${log_path}" 2>&1 &
  CI_PIDS+=("$!")
}

ci_assert_log_keywords() {
  local log_file="$1"
  shift
  if [[ ! -f "${log_file}" ]]; then
    echo "ci/lib.sh: log file missing: ${log_file}" >&2
    return 1
  fi
  local pattern
  for pattern in "$@"; do
    if ! grep -q "${pattern}" "${log_file}"; then
      echo "ci/lib.sh: log ${log_file} missing keyword: ${pattern}" >&2
      return 1
    fi
    ci_log "log ok (${log_file}): found '${pattern}'"
  done
}

ci_assert_metrics_counter_gt() {
  local metrics_url="$1"
  local metric_name="$2"
  local min_value="$3"
  local value

  value="$(
    ci_curl_local "${metrics_url}" \
      | awk -v name="${metric_name}" '
          $1 == name || index($1, name "{") == 1 { sum += $2; found = 1 }
          END { if (found) print sum }
        '
  )" || {
    echo "ci/lib.sh: failed to fetch metrics from ${metrics_url}" >&2
    return 1
  }

  if [[ -z "${value}" ]]; then
    echo "ci/lib.sh: metric ${metric_name} not found at ${metrics_url}" >&2
    return 1
  fi

  if ! python3 - "${value}" "${min_value}" <<'PY'
import sys
value = float(sys.argv[1])
minimum = float(sys.argv[2])
raise SystemExit(0 if value > minimum else 1)
PY
  then
    echo "ci/lib.sh: ${metric_name}=${value} is not > ${min_value}" >&2
    return 1
  fi

  ci_log "metrics ok (${metrics_url}): ${metric_name}=${value} > ${min_value}"
}

# 2P2D xPyD: each of 4 instances needs TENSOR_PARALLEL GPUs (comma-separated per role).
ci_xpyd_required_gpu_count() {
  local tp="${1:-${TENSOR_PARALLEL:-1}}"
  echo $((4 * tp))
}

ci_xpyd_assign_gpus_from_ci_list() {
  local tp="${TENSOR_PARALLEL:-1}"
  if [[ -z "${LMCACHE_CI_GPUS:-}" ]]; then
    return 0
  fi

  local -a ids=()
  IFS=',' read -ra ids <<< "${LMCACHE_CI_GPUS}"
  local need
  need="$(ci_xpyd_required_gpu_count "${tp}")"

  if ((${#ids[@]} < need)); then
    echo "ci/lib.sh: 2P2D with TP=${tp} needs ${need} GPU id(s), got ${#ids[@]} in LMCACHE_CI_GPUS" >&2
    return 1
  fi

  local offset=0
  local count="${tp}"
  local joined=""
  local i

  _ci_xpyd_join_slice() {
    offset="$1"
    count="$2"
    joined=""
    for ((i = 0; i < count; i++)); do
      if [[ -n "${joined}" ]]; then
        joined+=","
      fi
      joined+="${ids[offset + i]}"
    done
    echo "${joined}"
  }

  export PREFILLER1_GPU="$(_ci_xpyd_join_slice "${offset}" "${count}")"
  offset=$((offset + count))
  export PREFILLER2_GPU="$(_ci_xpyd_join_slice "${offset}" "${count}")"
  offset=$((offset + count))
  export DECODER1_GPU="$(_ci_xpyd_join_slice "${offset}" "${count}")"
  offset=$((offset + count))
  export DECODER2_GPU="$(_ci_xpyd_join_slice "${offset}" "${count}")"
}

ci_force_kill_port_listeners() {
  local host="${1:-127.0.0.1}"
  shift
  local ports=("$@")
  local port

  for port in "${ports[@]}"; do
    if ! ci_port_is_open "${host}" "${port}"; then
      continue
    fi
    ci_log "force killing listeners on ${host}:${port}"
    if command -v fuser >/dev/null 2>&1; then
      fuser -k "${port}/tcp" 2>/dev/null || true
      continue
    fi
    python3 - "${host}" "${port}" <<'PY'
import os, signal, socket, sys

host, port = sys.argv[1], int(sys.argv[2])
s = socket.socket()
s.settimeout(1)
try:
    s.connect((host, port))
except OSError:
    raise SystemExit(0)
finally:
    s.close()

try:
    import psutil
except ImportError:
    raise SystemExit(0)

for conn in psutil.net_connections(kind="tcp"):
    if conn.laddr and conn.laddr.port == port and conn.status == "LISTEN" and conn.pid:
        try:
            os.kill(conn.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
PY
  done
  sleep 2
}

ci_cleanup() {
  local pid
  if ((${#CI_PIDS[@]} > 0)); then
    ci_log "stopping ${#CI_PIDS[@]} background service(s)"
    for pid in "${CI_PIDS[@]}"; do
      kill -TERM -- "-${pid}" 2>/dev/null || kill -TERM "${pid}" 2>/dev/null || true
    done
    sleep 5
    for pid in "${CI_PIDS[@]}"; do
      if kill -0 "${pid}" 2>/dev/null; then
        kill -KILL -- "-${pid}" 2>/dev/null || kill -KILL "${pid}" 2>/dev/null || true
      fi
    done
    sleep 2
  fi

  if ((${#CI_PORTS[@]} > 0)); then
    ci_force_kill_port_listeners "127.0.0.1" "${CI_PORTS[@]}"
    if ! ci_wait_ports_closed "127.0.0.1" "${CI_PORTS[@]}"; then
      ci_log "WARNING: cleanup finished but some ports are still open; next case may fail"
    fi
  fi

  # vLLM may leave detached EngineCore/APIServer processes after startup
  # failures; they do not necessarily own the HTTP ports but still hold GPUs.
  pkill -KILL -f "VLLM::EngineCore" 2>/dev/null || true
  pkill -KILL -f "VLLM::APIServer" 2>/dev/null || true
  pkill -KILL -f "VLLM::Worker" 2>/dev/null || true
  sleep 2

  CI_PIDS=()
  CI_PORTS=()
}
