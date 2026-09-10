#!/usr/bin/env bash
# Copyright (c) 2026, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

# External LMCache prefix hit on aggregated vLLM + MP GDS L1.
#
# Pass requires:
#   1) MP store + lookup-hit evidence (same as smoke/)
#   2) GDS L1 path evidence in LMCACHE_SERVER_LOG
#      (GDS L1 tier enabled / GDSContext slab created)
#
# Usage:
#   LMCACHE_SERVER_LOG=/path/to/lmcache.log \
#     GDS_L1_PATH=/mnt/nvme/lmcache-gds-l1 bash test_external_hit.sh

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

if [[ -z "${LMCACHE_SERVER_LOG}" || ! -f "${LMCACHE_SERVER_LOG}" ]]; then
  echo "LMCACHE_SERVER_LOG is required and must exist (GDS L1 path proof)." >&2
  echo "Redirect lmcache server stdout/stderr to a file, then:" >&2
  echo "  LMCACHE_SERVER_LOG=/path/to/lmcache.log bash test_external_hit.sh" >&2
  exit 2
fi

echo "=== vLLM agg + MP GDS L1 external-hit test ==="
echo "server=http://${SERVER_HOST}:${SERVER_PORT}  model=${MODEL_NAME}"
echo "mp=${LMCACHE_MP_SERVER_URL}  metrics=${LMCACHE_URL}/metrics"
echo "gds_l1_path=${GDS_L1_PATH}"
echo "lmcache_log=${LMCACHE_SERVER_LOG}"
echo "max_tokens=${MAX_TOKENS}  prompt_repeat=${PROMPT_REPEAT}"
echo ""

echo "Checking GDS L1 path evidence in MP server log (before traffic)..."
if ! grep -q "GDS L1 tier enabled" "${LMCACHE_SERVER_LOG}"; then
  echo "FAIL: MP log missing 'GDS L1 tier enabled' (CPU pinned-DRAM L1 should be disabled)" >&2
  exit 1
fi
echo "PASS: log contains 'GDS L1 tier enabled'"

if ! grep -q "GDSContext: slab created" "${LMCACHE_SERVER_LOG}"; then
  echo "FAIL: MP log missing 'GDSContext: slab created'" >&2
  exit 1
fi
echo "PASS: log contains 'GDSContext: slab created'"

SLAB_FILE="${GDS_L1_PATH}/lmcache_gds_slab.bin"
if [[ ! -f "${SLAB_FILE}" ]]; then
  echo "FAIL: expected slab file missing: ${SLAB_FILE}" >&2
  exit 1
fi
echo "PASS: slab file exists: ${SLAB_FILE}"

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

echo "Step 1: COLD — first request (populate GDS L1 slab)"
run_request "COLD"

echo ""
echo "Step 2: WARM — same prompt (expect MP lookup hit)"
run_request "WARM"

echo ""
echo "Checking MP server log for store evidence: ${LMCACHE_SERVER_LOG}"
if ! grep -q "Stored" "${LMCACHE_SERVER_LOG}"; then
  echo "FAIL: MP server log missing 'Stored' after COLD/WARM" >&2
  exit 1
fi
echo "PASS: MP server log contains 'Stored'"

echo ""
echo "Checking MP metrics for lookup hit: ${LMCACHE_URL}/metrics"
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

if [[ -n "${SERVER_LOG}" && -f "${SERVER_LOG}" ]]; then
  echo ""
  echo "Optional: checking vLLM log for hit-token lines: ${SERVER_LOG}"
  python3 - "${SERVER_LOG}" <<'PY' || true
import re
import sys

path = sys.argv[1]
text = open(path, encoding="utf-8", errors="replace").read()
hits = [int(m.group(1)) for m in re.finditer(r"LMCache hit tokens:\s*(\d+)", text)]
if hits:
    print(f"INFO: last LMCache hit tokens={hits[-1]} (vLLM log)")
else:
    print("INFO: no 'LMCache hit tokens:' lines in vLLM log (optional)")
PY
fi

echo ""
echo "[DONE] GDS L1 external-hit test passed (path evidence + store + lookup hit)."
