#!/usr/bin/env bash
# Copyright (c) 2026, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

# Perf profile: Qwen3-32B-W8A8, TP=2, 8 GPUs (manual benchmark only, not CI).

PROFILE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MP_ROOT="$(cd "${PROFILE_DIR}/.." && pwd)"

# shellcheck source=../env.defaults.sh
source "${MP_ROOT}/env.defaults.sh"

export PROFILE_ENV="${PROFILE_DIR}/env.sh"
export PROFILE_DIR
export MP_ROOT

export MODEL_PATH="/data/nlp/Qwen3-32B-W8A8"
export SERVED_MODEL_NAME="Qwen3-32B-W8A8"
export MODEL_NAME="${SERVED_MODEL_NAME}"
export GPU_MEM_UTIL="0.85"
export MAX_MODEL_LEN="8192"
export TENSOR_PARALLEL="2"
export LMCACHE_MAX_GPU_WORKERS="$((4 * TENSOR_PARALLEL))"
export LMCACHE_L1_SIZE_GB="${LMCACHE_L1_SIZE_GB:-100}"

export PREFILLER1_GPU="0,1"
export PREFILLER2_GPU="2,3"
export DECODER1_GPU="4,5"
export DECODER2_GPU="6,7"

# Benchmark workload defaults
export SHARED_SYSTEM_TOKENS="1500"
export USER_UNIQUE_TOKENS="32"
export USER_QUERY="Summarize the system context in one sentence."
export MAX_TOKENS="32"
export NUM_USERS="16"
export WARMUP_USERS="1"
export BENCH_ROUNDS="1"
export MAX_INFLIGHT="8"
export REQUEST_TIMEOUT="900"
export PROXY_BASE_URL="http://127.0.0.1:${PROXY_PORT}/v1"
export LMCACHE_URL="http://127.0.0.1:${LMCACHE_HTTP_PORT}"
