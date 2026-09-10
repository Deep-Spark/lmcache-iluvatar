#!/usr/bin/env bash
# Copyright (c) 2026, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

# 1P1D profile: pure vLLM NixlPush P/D transfer (no LMCache MP).
# Used to isolate NixlPushConnector latency vs PDBackend / MP+NIXL.

SCENARIO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MP_ROOT="$(cd "${SCENARIO_DIR}/.." && pwd)"

# shellcheck source=../env.defaults.sh
source "${MP_ROOT}/env.defaults.sh"

export PROFILE_ENV="${SCENARIO_DIR}/env.sh"
export SCENARIO_DIR
export MP_ROOT

export MODEL_PATH="${MODEL_PATH:-/data/nlp/Qwen3-32B-W8A8/}"
export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen3-32B-W8A8}"
export MODEL_NAME="${MODEL_NAME:-${SERVED_MODEL_NAME}}"
export GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.8}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-40960}"
export TENSOR_PARALLEL="${TENSOR_PARALLEL:-2}"

export PYTHONHASHSEED="${PYTHONHASHSEED:-0}"
export VLLM_KV_CACHE_LAYOUT="${VLLM_KV_CACHE_LAYOUT:-HND}"
export UCX_RCACHE_MAX_UNRELEASED="${UCX_RCACHE_MAX_UNRELEASED:-4096}"
export UCX_TLS="${UCX_TLS:-cuda_ipc,cuda_copy,tcp}"

# Prefer explicit push connector (NixlConnector may alias to pull in some builds).
export NIXL_CONNECTOR="${NIXL_CONNECTOR:-NixlPushConnector}"
# P and D MUST use different engine_ids (NixlPush peers are keyed by engine_id).
export NIXL_PREFILL_ENGINE_ID="${NIXL_PREFILL_ENGINE_ID:-nixl-push-p1}"
export NIXL_DECODE_ENGINE_ID="${NIXL_DECODE_ENGINE_ID:-nixl-push-d1}"
# Back-compat alias used by older notes; prefer the role-specific vars above.
export NIXL_ENGINE_ID="${NIXL_ENGINE_ID:-${NIXL_PREFILL_ENGINE_ID}}"
export NIXL_KV_IP="${NIXL_KV_IP:-127.0.0.1}"
export NIXL_KV_PORT="${NIXL_KV_PORT:-14579}"
export NIXL_KV_BUFFER_DEVICE="${NIXL_KV_BUFFER_DEVICE:-cuda}"
export NIXL_KV_BUFFER_SIZE="${NIXL_KV_BUFFER_SIZE:-5368709120}"
export NIXL_SIDE_CHANNEL_HOST="${NIXL_SIDE_CHANNEL_HOST:-localhost}"
export NIXL_PREFILL_SIDE_CHANNEL_PORT="${NIXL_PREFILL_SIDE_CHANNEL_PORT:-5600}"
export NIXL_DECODE_SIDE_CHANNEL_PORT="${NIXL_DECODE_SIDE_CHANNEL_PORT:-5601}"
