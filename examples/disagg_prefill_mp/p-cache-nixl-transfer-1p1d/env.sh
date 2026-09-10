#!/usr/bin/env bash
# Copyright (c) 2026, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

# 1P1D profile: P-side LMCache MP reuse + vLLM NIXL P/D transfer.

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

# This POC is intentionally 1P1D.
export LMCACHE_MAX_WORKERS="${LMCACHE_MAX_WORKERS:-$((2 * TENSOR_PARALLEL))}"
export LMCACHE_L1_SIZE_GB="${LMCACHE_L1_SIZE_GB:-100}"

# Keep hashes stable for cache-key repeatability in cache-reuse experiments.
export PYTHONHASHSEED="${PYTHONHASHSEED:-0}"
export VLLM_KV_CACHE_LAYOUT="${VLLM_KV_CACHE_LAYOUT:-HND}"
export UCX_RCACHE_MAX_UNRELEASED="${UCX_RCACHE_MAX_UNRELEASED:-4096}"

# vLLM NIXL connector knobs. In this vLLM build, NixlConnector is an alias for
# NixlPullConnector.  This example wants P to push KV to D, so default to the
# explicit push connector.
export NIXL_CONNECTOR="${NIXL_CONNECTOR:-NixlPushConnector}"
# P and D MUST use different engine_ids (NixlPush peers are keyed by engine_id).
export NIXL_PREFILL_ENGINE_ID="${NIXL_PREFILL_ENGINE_ID:-p-cache-nixl-p1}"
export NIXL_DECODE_ENGINE_ID="${NIXL_DECODE_ENGINE_ID:-p-cache-nixl-d1}"
# Back-compat alias; prefer the role-specific vars above.
export NIXL_ENGINE_ID="${NIXL_ENGINE_ID:-${NIXL_PREFILL_ENGINE_ID}}"
export NIXL_KV_IP="${NIXL_KV_IP:-127.0.0.1}"
export NIXL_KV_PORT="${NIXL_KV_PORT:-14579}"
export NIXL_KV_BUFFER_DEVICE="${NIXL_KV_BUFFER_DEVICE:-cuda}"
export NIXL_KV_BUFFER_SIZE="${NIXL_KV_BUFFER_SIZE:-5368709120}"
export NIXL_SIDE_CHANNEL_HOST="${NIXL_SIDE_CHANNEL_HOST:-localhost}"
export NIXL_PREFILL_SIDE_CHANNEL_PORT="${NIXL_PREFILL_SIDE_CHANNEL_PORT:-5600}"
export NIXL_DECODE_SIDE_CHANNEL_PORT="${NIXL_DECODE_SIDE_CHANNEL_PORT:-5601}"
export UCX_TLS="${UCX_TLS:-cuda_ipc,cuda_copy,tcp}"

# LMCache MP connector accepts either port or server_urls depending on the
# installed vLLM/LMCache integration version. Supplying both keeps the example
# tolerant across the versions we have seen.
export LMCACHE_MP_SERVER_URL="${LMCACHE_MP_SERVER_URL:-tcp://127.0.0.1:${LMCACHE_MP_PORT}}"
