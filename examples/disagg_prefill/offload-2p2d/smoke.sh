#!/usr/bin/env bash
# Copyright (c) 2026, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

# 2P2D PDBackend local-tiered smoke: ≥4 sequential /v1/completions via proxy.
#
# Sync RR (P_i↔D_i) should hit both pairs across 4 requests. Local-tiered
# caches are per-instance — cross-P miss under RR is expected.
#
# Prerequisites: proxy, D1/D2, P1/P2 running (see README.md).
#
# Usage:
#   bash smoke.sh
#   NUM_RUNS=4 MAX_TOKENS=4 bash smoke.sh
#   SKIP_STACK_WAIT=1 bash smoke.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMMON_DIR="$(cd "${SCRIPT_DIR}/../../common" && pwd)"

# shellcheck source=env.sh
source "${SCRIPT_DIR}/env.sh"
# shellcheck source=../../common/bench_lib.sh
source "${COMMON_DIR}/bench_lib.sh"

export BENCH_LOG_PREFIX="${BENCH_LOG_PREFIX:-[smoke-pd-2p2d]}"

PROXY_HOST="${PROXY_HOST:-127.0.0.1}"
SERVER_PORT="${SERVER_PORT:-${PROXY_PORT}}"
export MODEL_NAME="${MODEL_NAME:-${SERVED_MODEL_NAME}}"
export MAX_TOKENS="${MAX_TOKENS:-4}"
# Need ≥4 runs so sync RR hits both P↔D pairs at least twice.
export NUM_RUNS="${NUM_RUNS:-4}"

_metric_sum() {
  local url="$1"
  local name="$2"
  local text
  text="$(curl -sf --noproxy '*' "${url}" 2>/dev/null || true)"
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

_send_one() {
  local run_idx="$1"
  local request_file response_file http_code
  request_file="$(mktemp)"
  response_file="$(mktemp)"

  python3 - "${request_file}" "${run_idx}" <<'PY'
import json
import os
import sys

run_idx = int(sys.argv[2])
prompt = (
    "LMCache PDBackend 2P2D local-tiered smoke test run "
    f"{run_idx}. Briefly confirm KV offload works in one short sentence."
)
payload = {
    "model": os.environ["MODEL_NAME"],
    "prompt": prompt,
    "max_tokens": int(os.environ.get("MAX_TOKENS", "4")),
    "temperature": 0.0,
    "stream": False,
}
with open(sys.argv[1], "w", encoding="utf-8") as f:
    json.dump(payload, f)
print(f"run={run_idx} prompt_chars={len(prompt)}")
PY

  http_code="$(
    curl -sS -N -o "${response_file}" -w '%{http_code}' \
      --noproxy '*' \
      --max-time "${SMOKE_TIMEOUT}" \
      -H 'Content-Type: application/json' \
      -H 'Authorization: Bearer EMPTY' \
      -d @"${request_file}" \
      "http://${PROXY_HOST}:${SERVER_PORT}/v1/completions"
  )"

  if [[ "${http_code}" != "200" ]]; then
    echo "Smoke run ${run_idx} failed: HTTP ${http_code}" >&2
    head -c 500 "${response_file}" >&2 || true
    rm -f "${request_file}" "${response_file}"
    return 1
  fi

  if ! python3 - "${response_file}" <<'PY'
import json
import sys


def load_response(path: str) -> str:
    with open(path, encoding="utf-8") as f:
        body = f.read().strip()

    if not body:
        raise SystemExit("empty response body")

    if body.startswith("{"):
        data = json.loads(body)
        choices = data.get("choices") or []
        if not choices:
            raise SystemExit("missing choices in JSON response")
        text = choices[0].get("text", "")
        if not str(text).strip():
            raise SystemExit("empty completion text in JSON response")
        return str(text).strip()

    parts: list[str] = []
    for line in body.splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        payload = line[len("data:") :].strip()
        if not payload or payload == "[DONE]":
            continue
        chunk = json.loads(payload)
        for choice in chunk.get("choices") or []:
            if choice.get("text"):
                parts.append(str(choice["text"]))
            delta = choice.get("delta") or {}
            if delta.get("content"):
                parts.append(str(delta["content"]))

    text = "".join(parts).strip()
    if not text:
        raise SystemExit("empty completion text in SSE response")
    return text


print(load_response(sys.argv[1])[:200])
PY
  then
    echo "Smoke run ${run_idx} failed: could not parse proxy response" >&2
    head -c 500 "${response_file}" >&2 || true
    rm -f "${request_file}" "${response_file}"
    return 1
  fi

  rm -f "${request_file}" "${response_file}"
  bench_log "run ${run_idx}/${NUM_RUNS}: HTTP 200 OK"
  return 0
}

if [[ "${SKIP_STACK_WAIT:-0}" != "1" ]]; then
  bench_wait_for_2p2d_stack "${PROXY_HOST}" "${SERVER_PORT}"
fi

echo "=== PDBackend 2P2D local-tiered smoke ==="
echo "proxy=http://${PROXY_HOST}:${SERVER_PORT}  model=${MODEL_NAME}  NUM_RUNS=${NUM_RUNS}"

for ((i = 1; i <= NUM_RUNS; i++)); do
  _send_one "${i}"
done

bench_log "Verifying sync RR coverage (NUM_RUNS=${NUM_RUNS}) ..."
_instance_saw_traffic "${PREFILLER1_PORT}" "P1"
_instance_saw_traffic "${PREFILLER2_PORT}" "P2"
_instance_saw_traffic "${DECODER1_PORT}" "D1"
_instance_saw_traffic "${DECODER2_PORT}" "D2"

echo ""
echo "[PASS] PDBackend 2P2D local-tiered smoke succeeded."
