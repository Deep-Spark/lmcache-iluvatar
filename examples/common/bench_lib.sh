#!/usr/bin/env bash
# Copyright (c) 2026, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

# Shared helpers for example benchmark wrapper scripts.
#
# Usage (from an example directory):
#   COMMON_DIR="$(cd "${SCRIPT_DIR}/../../common" && pwd)"
#   # shellcheck source=../../common/bench_lib.sh
#   source "${COMMON_DIR}/bench_lib.sh"

_bench_lib_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export BENCH_COMMON_DIR="${_bench_lib_dir}"
export BENCH_SERVING="${BENCH_SERVING:-${BENCH_COMMON_DIR}/bench_serving.py}"
export LONG_DOC_QA="${LONG_DOC_QA:-${BENCH_COMMON_DIR}/long_doc_qa.py}"

bench_log() {
  echo "${BENCH_LOG_PREFIX:-[bench]} $*" >&2
}

bench_wait_for_health() {
  local host=$1
  local port=$2
  local label=$3
  local timeout_sec=${4:-300}
  bench_log "Waiting for ${label} at ${host}:${port} ..."
  timeout "${timeout_sec}" bash -c "
    until curl -sf --noproxy '*' http://${host}:${port}/health >/dev/null 2>&1; do
      sleep 2
    done
  " && bench_log "${label} is ready." || {
    bench_log "ERROR: ${label} did not start within ${timeout_sec}s."
    return 1
  }
}

bench_wait_for_proxy() {
  local host=$1
  local port=$2
  local label=${3:-proxy}
  local timeout_sec=${4:-300}
  bench_log "Waiting for ${label} at ${host}:${port} (/v1/models) ..."
  timeout "${timeout_sec}" bash -c "
    until curl -sf --noproxy '*' http://${host}:${port}/v1/models >/dev/null 2>&1; do
      sleep 2
    done
  " && bench_log "${label} is ready." || {
    bench_log "ERROR: ${label} did not become ready within ${timeout_sec}s."
    return 1
  }
}

# Wait for 1P1D disagg stack: prefiller, decoder, then proxy /v1/models.
bench_wait_for_1p1d_stack() {
  local proxy_host=${1:-127.0.0.1}
  local proxy_port=${2:-${PROXY_PORT}}
  local pd_host=${3:-127.0.0.1}
  local timeout_sec=${4:-300}
  bench_wait_for_health "${pd_host}" "${PREFILLER1_PORT}" "prefiller" "${timeout_sec}"
  bench_wait_for_health "${pd_host}" "${DECODER1_PORT}" "decoder" "${timeout_sec}"
  bench_wait_for_proxy "${proxy_host}" "${proxy_port}" "proxy" "${timeout_sec}"
}

# Wait for 2P2D disagg stack: both prefillers/decoders, then proxy /v1/models.
# Args: proxy_host proxy_port [pair1_host] [pair2_host] [timeout_sec]
# pair2_host defaults to pair1_host (single-node). Dual-node: pass IP_A IP_B.
bench_wait_for_2p2d_stack() {
  local proxy_host=${1:-127.0.0.1}
  local proxy_port=${2:-${PROXY_PORT}}
  local pair1_host=${3:-127.0.0.1}
  local pair2_host=${4:-${pair1_host}}
  local timeout_sec=${5:-300}
  bench_wait_for_health "${pair1_host}" "${PREFILLER1_PORT}" "prefiller1" "${timeout_sec}"
  bench_wait_for_health "${pair2_host}" "${PREFILLER2_PORT}" "prefiller2" "${timeout_sec}"
  bench_wait_for_health "${pair1_host}" "${DECODER1_PORT}" "decoder1" "${timeout_sec}"
  bench_wait_for_health "${pair2_host}" "${DECODER2_PORT}" "decoder2" "${timeout_sec}"
  bench_wait_for_proxy "${proxy_host}" "${proxy_port}" "proxy" "${timeout_sec}"
}

bench_mooncake_trace_url() {
  case "$1" in
    synthetic)
      echo "https://raw.githubusercontent.com/kvcache-ai/Mooncake/main/FAST25-release/traces/synthetic_trace.jsonl"
      ;;
    conversation)
      echo "https://raw.githubusercontent.com/kvcache-ai/Mooncake/main/FAST25-release/traces/conversation_trace.jsonl"
      ;;
    toolagent)
      echo "https://raw.githubusercontent.com/kvcache-ai/Mooncake/main/FAST25-release/traces/toolagent_trace.jsonl"
      ;;
    mooncake)
      echo "https://raw.githubusercontent.com/kvcache-ai/Mooncake/main/FAST25-release/arxiv-trace/mooncake_trace.jsonl"
      ;;
    *)
      return 1
      ;;
  esac
}

# Ensure a Mooncake FAST25 trace exists locally; print the trace path on success.
bench_ensure_mooncake_trace() {
  local workload=$1
  local trace_dir=${2:-/tmp}
  local trace_path="${trace_dir}/${workload}_trace.jsonl"

  if [[ -f "${trace_path}" ]]; then
    echo "${trace_path}"
    return 0
  fi

  local trace_url
  if ! trace_url="$(bench_mooncake_trace_url "${workload}")"; then
    bench_log "WARNING: unknown Mooncake workload '${workload}', skipping."
    return 1
  fi

  bench_log "Downloading ${workload} trace to ${trace_path} ..."
  wget -q -O "${trace_path}" "${trace_url}" || \
    curl -sL -o "${trace_path}" "${trace_url}" || {
      bench_log "WARNING: failed to download ${workload} trace."
      rm -f "${trace_path}"
      return 1
    }
  echo "${trace_path}"
}

bench_print_ttft_summary() {
  local width=${1:-60}
  python3 "${BENCH_COMMON_DIR}/bench_summarize_ttft.py" --width "${width}"
}
