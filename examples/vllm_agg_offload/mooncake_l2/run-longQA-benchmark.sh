#!/usr/bin/env bash
# Copyright (c) 2026, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

# Run LongQA on the writer first, then on the reader against the shared L2.
# Services are intentionally started manually because they live on two hosts.

set -euo pipefail

EXAMPLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMMON_DIR="$(cd "${EXAMPLE_DIR}/../../common" && pwd)"
LONG_DOC_QA="${LONG_DOC_QA:-${COMMON_DIR}/long_doc_qa.py}"
# shellcheck source=env.sh
source "${EXAMPLE_DIR}/env.sh"

: "${WRITER_BASE_URL:?WRITER_BASE_URL is required, for example http://writer-host:18040}"
: "${READER_BASE_URL:?READER_BASE_URL is required, for example http://reader-host:18040}"

RUN_ID="${RUN_ID:-mooncake-${MOONCAKE_PROTOCOL}-$(date +%Y%m%d-%H%M%S)-$$}"
RESULT_DIR="${RESULT_DIR:-${EXAMPLE_DIR}/results/${RUN_ID}}"
STORE_SETTLE_SECONDS="${STORE_SETTLE_SECONDS:-15}"
mkdir -p "${RESULT_DIR}"

run_longqa() {
  local label="$1" base_url="$2"
  local output_dir="${RESULT_DIR}/${label}"
  base_url="${base_url%/}"
  if [[ "${base_url}" != */v1 ]]; then
    base_url="${base_url}/v1"
  fi
  mkdir -p "${output_dir}"

  python3 "${LONG_DOC_QA}" \
    --base-url "${base_url}" \
    --model "${SERVED_MODEL_NAME}" \
    --document-length "${LONGQA_DOCUMENT_LENGTH}" \
    --num-documents "${LONGQA_NUM_DOCUMENTS}" \
    --output-len "${LONGQA_OUTPUT_LEN}" \
    --repeat-count "${LONGQA_REPEAT_COUNT}" \
    --repeat-mode "${LONGQA_REPEAT_MODE}" \
    --max-inflight-requests "${LONGQA_MAX_INFLIGHT}" \
    --shuffle-seed "${LONGQA_SHUFFLE_SEED}" \
    --sleep-time-after-warmup "${LONGQA_SLEEP_AFTER_WARMUP}" \
    --output "${output_dir}/responses.txt" \
    --output-dir "${output_dir}" \
    --json-output \
    --completions \
    2>&1 | tee "${output_dir}/longqa.log"
}

echo "Mooncake cross-instance L2 test"
echo "  protocol=${MOONCAKE_PROTOCOL} tenant=${MOONCAKE_TENANT_ID}"
echo "  writer=${WRITER_BASE_URL} reader=${READER_BASE_URL}"
echo "  documents=${LONGQA_NUM_DOCUMENTS} length=${LONGQA_DOCUMENT_LENGTH} results=${RESULT_DIR}"

echo "Step 1: writer LongQA (warmup is cold and populates shared L2)"
run_longqa writer "${WRITER_BASE_URL}"

echo "Waiting ${STORE_SETTLE_SECONDS}s for asynchronous L2 store"
sleep "${STORE_SETTLE_SECONDS}"

echo "Step 2: reader LongQA (warmup expects empty L1 and positive L2 hits)"
run_longqa reader "${READER_BASE_URL}"

python3 - \
  "${RESULT_DIR}/writer/warmup_round.csv" \
  "${RESULT_DIR}/writer/query_round.csv" \
  "${RESULT_DIR}/reader/warmup_round.csv" \
  "${RESULT_DIR}/reader/query_round.csv" \
  "${LONGQA_NUM_DOCUMENTS}" \
  "${LONGQA_REPEAT_COUNT}" <<'PY'
import csv
import statistics
import sys
from pathlib import Path

paths = list(map(Path, sys.argv[1:5]))
num_documents = int(sys.argv[5])
repeat_count = int(sys.argv[6])
expected_rows = [num_documents, num_documents * repeat_count] * 2
rounds = []

for path, expected in zip(paths, expected_rows):
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    successful = [row for row in rows if row["successful"].lower() == "true"]
    if len(rows) != expected or len(successful) != expected:
        raise SystemExit(
            f"{path}: expected {expected}/{expected} successful requests, "
            f"got {len(successful)}/{len(rows)}"
        )
    rounds.append(statistics.mean(float(row["ttft"]) for row in successful))

writer_warmup, _, reader_warmup, _ = rounds
gain = writer_warmup / reader_warmup if reader_warmup > 0 else float("inf")
print(
    f"LongQA checks passed: writer cold warmup={writer_warmup:.3f}s, "
    f"reader cross-instance warmup={reader_warmup:.3f}s, gain={gain:.2f}x"
)
PY

echo "LongQA request phase passed. Run verify_reader_logs.sh on the reader host."
echo "The test is complete only after reader logs prove '(0 L1, N L2)' with N > 0."
