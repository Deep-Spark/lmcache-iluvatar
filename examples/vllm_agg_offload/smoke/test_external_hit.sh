#!/usr/bin/env bash
# Copyright (c) 2026, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

# External LMCache prefix hit on aggregated vLLM + MP Server.
#
# Sends the same long prompt twice. Pass requires MP store/lookup-hit evidence
# (HTTP 200 alone is not enough). Prefer LMCACHE_SERVER_LOG / LMCACHE_URL;
# SERVER_LOG (vLLM) hit-token lines remain an optional fallback.
#
# Usage:
#   bash test_external_hit.sh
#   LMCACHE_SERVER_LOG=/path/to/lmcache.log LMCACHE_URL=http://localhost:8080 \
#     bash test_external_hit.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "${SCRIPT_DIR}/env.sh"

MODEL_NAME="${MODEL_NAME:-${SERVED_MODEL_NAME}}"
MAX_TOKENS="${MAX_TOKENS:-16}"
TIMEOUT="${TIMEOUT:-${LMCACHE_CI_REQUEST_TIMEOUT:-600}}"
PROMPT_SENTENCE="${PROMPT_SENTENCE:-Explain the significance of KV cache in language models. }"
PROMPT_REPEAT="${PROMPT_REPEAT:-100}"
SERVER_LOG="${SERVER_LOG:-}"
LMCACHE_SERVER_LOG="${LMCACHE_SERVER_LOG:-}"

echo "=== vLLM agg + MP external-hit test ==="
echo "server=http://${SERVER_HOST}:${SERVER_PORT}  model=${MODEL_NAME}"
echo "mp=${LMCACHE_MP_SERVER_URL}  metrics=${LMCACHE_URL}/metrics"
echo "max_tokens=${MAX_TOKENS}  prompt_repeat=${PROMPT_REPEAT}"
echo ""

request_file="$(mktemp)"
response_file="$(mktemp)"
trap 'rm -f "${request_file}" "${response_file}"' EXIT

PROMPT_SENTENCE="${PROMPT_SENTENCE}" \
PROMPT_REPEAT="${PROMPT_REPEAT}" \
MODEL_NAME="${MODEL_NAME}" \
MAX_TOKENS="${MAX_TOKENS}" \
python3 - "${request_file}" <<'PY'
import json
import os
import sys

prompt = os.environ["PROMPT_SENTENCE"] * int(os.environ["PROMPT_REPEAT"])
payload = {
    "model": os.environ["MODEL_NAME"],
    "prompt": prompt,
    "max_tokens": int(os.environ["MAX_TOKENS"]),
    "temperature": 0.0,
}
with open(sys.argv[1], "w", encoding="utf-8") as f:
    json.dump(payload, f)
print(f"  prompt_chars={len(prompt)}  prompt_tokens_est={len(prompt) // 4}")
PY

run_request() {
  local label="$1"
  local start end http_code

  echo ""
  echo "--- ${label} ---"
  start="$(date +%s.%N)"
  http_code="$(curl -sS -o "${response_file}" -w '%{http_code}' \
    --max-time "${TIMEOUT}" \
    -X POST "http://${SERVER_HOST}:${SERVER_PORT}/v1/completions" \
    -H "Content-Type: application/json" \
    -d @"${request_file}")"
  end="$(date +%s.%N)"

  local elapsed
  elapsed="$(python3 -c "print(f'{float(${end}) - float(${start}):.3f}')")"

  if [[ "${http_code}" != "200" ]]; then
    echo "  ${label} FAILED: HTTP ${http_code}  elapsed=${elapsed}s"
    head -c 500 "${response_file}" || true
    echo ""
    return 1
  fi

  python3 - "${response_file}" "${label}" "${elapsed}" <<'PY'
import json
import sys

path, label, elapsed = sys.argv[1], sys.argv[2], sys.argv[3]
with open(path, encoding="utf-8") as f:
    data = json.load(f)
usage = data.get("usage", {})
text = data["choices"][0].get("text", "").strip()
print(
    f"  {label}: elapsed={elapsed}s  "
    f"prompt_tokens={usage.get('prompt_tokens', '?')}  "
    f"completion_tokens={usage.get('completion_tokens', '?')}"
)
if len(text) > 120:
    print(f"         text: {text[:120]}...")
else:
    print(f"         text: {text}")
PY
}

echo "Step 1: COLD — first request (populate MP L1)"
run_request "COLD"

echo ""
echo "Step 2: WARM — same prompt (expect MP lookup hit)"
run_request "WARM"

evidence=0

if [[ -n "${LMCACHE_SERVER_LOG}" && -f "${LMCACHE_SERVER_LOG}" ]]; then
  echo ""
  echo "Checking MP server log for store evidence: ${LMCACHE_SERVER_LOG}"
  if ! grep -q "Stored" "${LMCACHE_SERVER_LOG}"; then
    echo "FAIL: MP server log missing 'Stored' after COLD/WARM" >&2
    exit 1
  fi
  echo "PASS: MP server log contains 'Stored'"
  evidence=1
fi

if [[ -n "${LMCACHE_URL:-}" ]]; then
  echo ""
  echo "Checking MP metrics for lookup hit: ${LMCACHE_URL}/metrics"
  # Prefer curl --noproxy so corporate HTTP(S)_PROXY cannot intercept localhost.
  if ! metrics_text="$(curl -sf --noproxy '*' --max-time 30 "${LMCACHE_URL}/metrics")"; then
    echo "FAIL: cannot fetch metrics from ${LMCACHE_URL}/metrics" >&2
    exit 1
  fi
  printf '%s\n' "${metrics_text}" | python3 -c '
import re
import sys

text = sys.stdin.read()
total = 0.0
found = False
for line in text.splitlines():
    if line.startswith("#"):
        continue
    m = re.match(r"(lmcache_mp_lookup_hit_tokens_total(?:\{[^}]*\})?)\s+(\S+)", line)
    if not m:
        continue
    found = True
    total += float(m.group(2))
if not found:
    raise SystemExit(
        "FAIL: metric lmcache_mp_lookup_hit_tokens_total not found in metrics body"
    )
if total <= 0:
    raise SystemExit(
        f"FAIL: lmcache_mp_lookup_hit_tokens_total={total} (expected > 0 on WARM). "
        "Check PYTHONHASHSEED=0, VLLM_KV_CACHE_LAYOUT=HND, MP server running, "
        "and prompt length (chunk_size=256)."
    )
print(f"PASS: lmcache_mp_lookup_hit_tokens_total={total}")
'
  evidence=1
fi

if [[ "${evidence}" -eq 0 && -n "${SERVER_LOG}" && -f "${SERVER_LOG}" ]]; then
  echo ""
  echo "Checking vLLM log for hit-token fallback: ${SERVER_LOG}"
  python3 - "${SERVER_LOG}" <<'PY'
import re
import sys

path = sys.argv[1]
text = open(path, encoding="utf-8", errors="replace").read()
hits = [int(m.group(1)) for m in re.finditer(r"LMCache hit tokens:\s*(\d+)", text)]
if len(hits) < 1:
    raise SystemExit(
        "FAIL: no 'LMCache hit tokens:' lines in vLLM log "
        "(prefer LMCACHE_SERVER_LOG / LMCACHE_URL for MP evidence)"
    )
warm_hit = hits[-1]
if warm_hit <= 0:
    raise SystemExit(
        f"FAIL: last LMCache hit tokens={warm_hit} (expected > 0 on WARM request). "
        "Check PYTHONHASHSEED=0, VLLM_KV_CACHE_LAYOUT=HND, and prompt length."
    )
print(f"PASS: last LMCache hit tokens={warm_hit} (vLLM log fallback)")
PY
  evidence=1
fi

if [[ "${evidence}" -eq 0 ]]; then
  echo ""
  echo "No LMCACHE_SERVER_LOG / LMCACHE_URL / SERVER_LOG — skip automated hit check."
  echo "Inspect MP server logs for 'Stored' / 'Retrieved', or:"
  echo "  curl -s --noproxy '*' \${LMCACHE_URL:-http://localhost:8080}/metrics | grep lmcache_mp_lookup_hit_tokens_total"
fi

echo ""
echo "[DONE] External-hit test finished (HTTP path). Cache-hit proof is in MP logs/metrics."
