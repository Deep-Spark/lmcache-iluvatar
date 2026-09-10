#!/usr/bin/env bash
# Copyright (c) 2026, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

# Minimal 1P1D local-tiered smoke: one proxy /v1/completions request.
#
# Prerequisites: prefiller, decoder, and disagg proxy are running.
#
# Note: disagg_proxy_server always responds with SSE (data: {...}) even when
# stream=false in the request body.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../env.sh
source "${SCRIPT_DIR}/../env.sh"

export MODEL_NAME="${MODEL_NAME:-${SERVED_MODEL_NAME}}"
export MAX_TOKENS="${MAX_TOKENS:-4}"

echo "=== local-tiered 1P1D smoke ==="
echo "proxy=http://127.0.0.1:${PUBLIC_PORT}  model=${MODEL_NAME}"

request_file="$(mktemp)"
response_file="$(mktemp)"
trap 'rm -f "${request_file}" "${response_file}"' EXIT

python3 - "${request_file}" <<PY
import json
import os
import sys

prompt = (
    "LMCache local-tiered disaggregated prefill smoke test. "
    "Briefly confirm KV offload works in one short sentence."
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
print(f"prompt_chars={len(prompt)}")
PY

http_code="$(
  curl -sS -N -o "${response_file}" -w '%{http_code}' \
    --max-time "${SMOKE_TIMEOUT}" \
    -H 'Content-Type: application/json' \
    -H 'Authorization: Bearer EMPTY' \
    -d @"${request_file}" \
    "http://127.0.0.1:${PUBLIC_PORT}/v1/completions"
)"

if [[ "${http_code}" != "200" ]]; then
  echo "Smoke test failed: HTTP ${http_code}" >&2
  head -c 500 "${response_file}" >&2 || true
  exit 1
fi

if ! python3 - "${response_file}" <<'PY'
import json
import sys


def load_response(path: str) -> str:
    with open(path, encoding="utf-8") as f:
        body = f.read().strip()

    if not body:
        raise SystemExit("empty response body")

    # Non-streaming JSON (future/alternate proxy behaviour)
    if body.startswith("{"):
        data = json.loads(body)
        choices = data.get("choices") or []
        if not choices:
            raise SystemExit("missing choices in JSON response")
        text = choices[0].get("text", "")
        if not str(text).strip():
            raise SystemExit("empty completion text in JSON response")
        return str(text).strip()

    # disagg_proxy_server SSE: data: {...}\n\n
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
  echo "Smoke test failed: could not parse proxy response" >&2
  head -c 500 "${response_file}" >&2 || true
  exit 1
fi

echo ""
echo "[PASS] local-tiered 1P1D smoke test succeeded."
