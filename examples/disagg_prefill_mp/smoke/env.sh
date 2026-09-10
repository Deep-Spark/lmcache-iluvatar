#!/usr/bin/env bash
# Copyright (c) 2026, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

# Smoke profile: Qwen3-8B, TP=1, 4 GPUs (CI + local smoke).

PROFILE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MP_ROOT="$(cd "${PROFILE_DIR}/.." && pwd)"

# shellcheck source=../env.defaults.sh
source "${MP_ROOT}/env.defaults.sh"

export PROFILE_ENV="${PROFILE_DIR}/env.sh"
export PROFILE_DIR
export MP_ROOT

export MODEL_PATH="/data/nlp/Qwen3-8B/"
export SERVED_MODEL_NAME="Qwen3-8B"
export MODEL_NAME="${SERVED_MODEL_NAME}"
export GPU_MEM_UTIL="0.8"
export MAX_MODEL_LEN="8192"
export TENSOR_PARALLEL="1"
export LMCACHE_MAX_GPU_WORKERS="$((4 * TENSOR_PARALLEL))"
export LMCACHE_L1_SIZE_GB="${LMCACHE_L1_SIZE_GB:-100}"

export PREFILLER1_GPU="0"
export PREFILLER2_GPU="1"
export DECODER1_GPU="2"
export DECODER2_GPU="3"
