#!/bin/bash
# Copyright (c) 2026, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

# CacheBlend shuffle-doc QA client benchmark.
#
# Start the vLLM server separately with start-server.sh, then run this script to
# send deranged multi-document requests.
#
# Usage:
#   bash start-server.sh
#   bash run_benchmark.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

: "${SERVED_MODEL_NAME:=Qwen3-8B}"
: "${TOKENIZER_PATH:=$SERVED_MODEL_NAME}"
: "${BENCH_PORT:=8000}"
: "${NUM_DOCUMENTS:=3}"
: "${DOCUMENT_LENGTH:=512}"
: "${NUM_REQUESTS:=2}"
: "${OUTPUT_LEN:=1}"
: "${MAX_INFLIGHT_REQUESTS:=1}"
: "${SLEEP_AFTER_WARMUP:=2}"
: "${RANDOM_SEED:=0}"
: "${OUT_DIR:=/tmp/cacheblend_shuffle_doc_qa}"

mkdir -p "$OUT_DIR"
export PYTHONHASHSEED="${PYTHONHASHSEED:-0}"

echo "[cacheblend] endpoint=http://localhost:$BENCH_PORT/v1"
echo "[cacheblend] model=$SERVED_MODEL_NAME"
echo "[cacheblend] tokenizer=$TOKENIZER_PATH"
echo "[cacheblend] out_dir=$OUT_DIR"

python3 "$SCRIPT_DIR/cacheblend_shuffle_doc_qa.py" \
  --num-documents "$NUM_DOCUMENTS" \
  --document-length "$DOCUMENT_LENGTH" \
  --num-requests "$NUM_REQUESTS" \
  --output-len "$OUTPUT_LEN" \
  --model "$SERVED_MODEL_NAME" \
  --tokenizer "$TOKENIZER_PATH" \
  --port "$BENCH_PORT" \
  --max-inflight-requests "$MAX_INFLIGHT_REQUESTS" \
  --sleep-time-after-warmup "$SLEEP_AFTER_WARMUP" \
  --random-seed "$RANDOM_SEED" \
  --output "$OUT_DIR/responses.log" \
  2>&1 | tee "$OUT_DIR/client.log"

{
  echo "=== derangement requests ==="
  grep -E "derangement|TTFT:" "$OUT_DIR/responses.log" || true
} > "$OUT_DIR/lmcache_stats.txt"

echo "[cacheblend] done"
echo "[cacheblend] logs: $OUT_DIR"
