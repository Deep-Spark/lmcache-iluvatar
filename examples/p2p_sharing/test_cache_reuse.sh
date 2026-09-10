#!/bin/bash
# Copyright (c) 2026, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

# P2P KV cache reuse test — Iluvatar edition.
#
# Sends the same long prompt to instance 1 (cold — computes KV), then to
# instance 2 (warm — should retrieve KV from instance 1 via P2P).
# Instance 2's log should show "Retrieved ... tokens" confirming the hit.
#
# Usage:
#   bash test_cache_reuse.sh
#   PROMPT_REPEAT=120 RUN_WARM_REPEAT=0 bash test_cache_reuse.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "${SCRIPT_DIR}/env.sh"

MODEL_NAME="${MODEL_NAME:-${SERVED_MODEL_NAME:-Qwen3-8B}}"
MAX_TOKENS="${MAX_TOKENS:-10}"
TIMEOUT="${TIMEOUT:-600}"
PROMPT_SENTENCE="${PROMPT_SENTENCE:-Explain the significance of KV cache in language models. }"
PROMPT_REPEAT="${PROMPT_REPEAT:-100}"
RUN_WARM_REPEAT="${RUN_WARM_REPEAT:-1}"

echo "=== P2P KV cache reuse test ==="
echo "instance1=http://localhost:${INSTANCE1_PORT}"
echo "instance2=http://localhost:${INSTANCE2_PORT}"
echo "model=${MODEL_NAME}  max_tokens=${MAX_TOKENS}  repeat=${PROMPT_REPEAT}"
echo ""

# ---- build the repeated-prefix prompt ----------------------------------------
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
prompt_tokens = len(prompt) // 4
print(f'  prompt_chars={len(prompt)}  prompt_tokens_est={prompt_tokens}')
PY

# ---- helper: send request and measure ----------------------------------------
run_test() {
  local label="$1" port="$2" start end http_code

  echo ""
  echo "--- ${label} (port ${port}) ---"

  start="$(date +%s.%N)"
  http_code="$(curl -sS -o "${response_file}" -w '%{http_code}' \
    --max-time "${TIMEOUT}" \
    -X POST "http://localhost:${port}/v1/completions" \
    -H "Content-Type: application/json" \
    -d @"${request_file}")"
  end="$(date +%s.%N)"

  local elapsed
  elapsed="$(python3 -c "print(f'{float(${end}) - float(${start}):.3f}')")"

  if [[ "${http_code}" != "200" ]]; then
    echo "  ${label} FAILED: HTTP ${http_code}  elapsed=${elapsed}s"
    python3 - "${response_file}" <<'PY'
import sys
from pathlib import Path

print(Path(sys.argv[1]).read_text(errors="replace")[:500])
PY
    exit 1
  fi

  python3 -c "
import json, sys
path, label, elapsed = sys.argv[1], sys.argv[2], sys.argv[3]
with open(path) as f: d = json.load(f)
u   = d.get('usage', {})
pt  = u.get('prompt_tokens', '?')
ct  = u.get('completion_tokens', '?')
txt = d['choices'][0].get('text', '').strip()
print(f'  {label}: elapsed={elapsed}s  prompt_tokens={pt}  completion_tokens={ct}')
print(f'         text: {txt[:120]}...' if len(txt)>120 else f'         text: {txt}')
" "${response_file}" "${label}" "${elapsed}"
}

# ---- main --------------------------------------------------------------------
echo "Step 1: Cold request to instance 1 (computes KV, stores in LocalCPUBackend)"
run_test "COLD-instance1" "${INSTANCE1_PORT}"

echo ""
echo "  Waiting 3s for P2P metadata to propagate..."
sleep 3

echo ""
echo "Step 2: Same prompt to instance 2 (should retrieve KV from instance 1 via P2P)"
echo "  Check instance 2 logs for: 'Retrieved ... tokens' from P2PBackend"
run_test "WARM-instance2" "${INSTANCE2_PORT}"

echo ""
if [[ "${RUN_WARM_REPEAT}" == "1" ]]; then
  echo "Step 3: Repeat to instance 1 (optional warm repeat, should hit local cache)"
  run_test "WARM-instance1" "${INSTANCE1_PORT}"
else
  echo "Step 3: Skipped warm repeat (RUN_WARM_REPEAT=${RUN_WARM_REPEAT})"
fi

echo ""
echo "[DONE] P2P KV cache reuse test complete."
echo "Cache-hit proof is in the service logs, not just HTTP 200."
echo "Verify instance 2 logs contain:"
echo "  LMCache hit tokens: 512"
echo "  need to load: 512"
echo "  Retrieved 512 out of 512 required tokens"
