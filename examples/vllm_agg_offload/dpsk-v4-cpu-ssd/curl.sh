#!/usr/bin/env bash
# Copyright (c) 2026, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

# Send one or two completion requests to aggregated DeepSeek-V4 + LMCache MP.
#
# Usage:
#   bash curl.sh
#   WARM=1 bash curl.sh
#   PROMPT_REPEAT=40 MAX_TOKENS=16 WARM=1 bash curl.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "${SCRIPT_DIR}/env.sh"

MODEL_NAME="${MODEL_NAME:-${SERVED_MODEL_NAME}}"
MAX_TOKENS="${MAX_TOKENS:-32}"
TIMEOUT="${TIMEOUT:-${LMCACHE_CI_REQUEST_TIMEOUT:-600}}"
PROMPT="${PROMPT:-Explain the significance of KV cache in language models. }"
PROMPT_REPEAT="${PROMPT_REPEAT:-100}"
WARM="${WARM:-0}"

request_file="$(mktemp)"
response_file="$(mktemp)"
trap 'rm -f "${request_file}" "${response_file}"' EXIT

PROMPT="${PROMPT}" \
PROMPT_REPEAT="${PROMPT_REPEAT}" \
MODEL_NAME="${MODEL_NAME}" \
MAX_TOKENS="${MAX_TOKENS}" \
python3 - "${request_file}" <<'PY'
import json
import os
import sys

prompt = os.environ["PROMPT"] * int(os.environ["PROMPT_REPEAT"])
payload = {
    "model": os.environ["MODEL_NAME"],
    "prompt": prompt,
    "max_tokens": int(os.environ["MAX_TOKENS"]),
    "temperature": 0.0,
}
with open(sys.argv[1], "w", encoding="utf-8") as f:
    json.dump(payload, f)
print(f"prompt_chars={len(prompt)}")
PY

echo "=== curl.sh (dpsk-v4-cpu-ssd) ==="
echo "server=http://${SERVER_HOST}:${SERVER_PORT}  model=${MODEL_NAME}"
echo "max_tokens=${MAX_TOKENS}  warm=${WARM}"
echo ""

send_request() {
  local label="$1"
  local http_code

  http_code="$(curl -sS -o "${response_file}" -w '%{http_code}' \
    --max-time "${TIMEOUT}" \
    -X POST "http://${SERVER_HOST}:${SERVER_PORT}/v1/completions" \
    -H "Content-Type: application/json" \
    -d @"${request_file}")"

  if [[ "${http_code}" != "200" ]]; then
    echo "[${label}] HTTP ${http_code}"
    head -c 800 "${response_file}" || true
    echo ""
    return 1
  fi

  python3 - "${response_file}" "${label}" <<'PY'
import json
import sys

path, label = sys.argv[1], sys.argv[2]
with open(path, encoding="utf-8") as f:
    data = json.load(f)
usage = data.get("usage", {})
text = data["choices"][0].get("text", "")
print(f"[{label}] prompt_tokens={usage.get('prompt_tokens', '?')} "
      f"completion_tokens={usage.get('completion_tokens', '?')}")
print(f"[{label}] text:")
print(text)
print()
PY
}

curl -sf "http://${SERVER_HOST}:${SERVER_PORT}/health" >/dev/null
echo "health: ok"
echo ""

send_request "REQ1"
if [[ "${WARM}" == "1" ]]; then
  send_request "REQ2"
  echo "Compare REQ1 vs REQ2 — check MP metrics / vLLM hit tokens for cache reuse."
fi
