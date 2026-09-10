#!/bin/bash
# Copyright (c) 2026, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

# =============================================================================
# Qwen3-8B TP2 1P1D — Long-Doc QA Benchmark
#
# Prerequisites: proxy, decoder, and prefiller running (see README.md).
# Defaults come from env.sh (sourced below).
#
# Usage:
#   bash run-longQA-benchmark.sh
#   bash run-longQA-benchmark.sh http://127.0.0.1:19100/v1
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "${SCRIPT_DIR}/env.sh"
COMMON_DIR="$(cd "${SCRIPT_DIR}/../../common" && pwd)"
# shellcheck source=../../common/bench_lib.sh
source "${COMMON_DIR}/bench_lib.sh"

export BENCH_LOG_PREFIX="[longQA]"

DEFAULT_BASE_URL="http://127.0.0.1:${PUBLIC_PORT}/v1"
BASE_URL="${1:-${BASE_URL:-${DEFAULT_BASE_URL}}}"

python3 "${LONG_DOC_QA}" \
  --base-url "$BASE_URL" \
  --model "$SERVED_MODEL_NAME" \
  --document-length 10000 \
  --num-documents 20 \
  --output-len 2 \
  --repeat-count 1 \
  --repeat-mode tile \
  --max-inflight-requests 1 \
  --shuffle-seed 0 \
  --pd-disagg-ttft \
  --json-output \
  --completions
