#!/usr/bin/env bash
# Copyright (c) 2026, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

# 2P2D profile: P-side LMCache MP reuse + vLLM NIXL P/D transfer.
# Independent round-robin across P1/P2 and D1/D2.

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

# Match 1P1D headroom (2×TP per prefiller) scaled to 2P → 4×TP.
export LMCACHE_MAX_WORKERS="${LMCACHE_MAX_WORKERS:-$((4 * TENSOR_PARALLEL))}"
# 40-doc longQA (Qwen3-32B TP2) overflows L1=100 (watermark 0.8) → eviction → cold query.
# Default 200GB so MP cross-P hits stay warm under NUM_DOCUMENTS=40.
export LMCACHE_L1_SIZE_GB="${LMCACHE_L1_SIZE_GB:-200}"

export PYTHONHASHSEED="${PYTHONHASHSEED:-0}"
export VLLM_KV_CACHE_LAYOUT="${VLLM_KV_CACHE_LAYOUT:-HND}"
export UCX_RCACHE_MAX_UNRELEASED="${UCX_RCACHE_MAX_UNRELEASED:-4096}"

# GPU layout (8 cards). start_*.sh set CUDA_VISIBLE_DEVICES from these.
export PREFILLER1_GPU="${PREFILLER1_GPU:-0,1}"
export PREFILLER2_GPU="${PREFILLER2_GPU:-2,3}"
export DECODER1_GPU="${DECODER1_GPU:-4,5}"
export DECODER2_GPU="${DECODER2_GPU:-6,7}"

# vLLM NIXL connector knobs.
export NIXL_CONNECTOR="${NIXL_CONNECTOR:-NixlPushConnector}"
# Each instance MUST use a unique engine_id (NixlPush peers are keyed by it).
export NIXL_PREFILL1_ENGINE_ID="${NIXL_PREFILL1_ENGINE_ID:-p-cache-nixl-p1}"
export NIXL_PREFILL2_ENGINE_ID="${NIXL_PREFILL2_ENGINE_ID:-p-cache-nixl-p2}"
export NIXL_DECODE1_ENGINE_ID="${NIXL_DECODE1_ENGINE_ID:-p-cache-nixl-d1}"
export NIXL_DECODE2_ENGINE_ID="${NIXL_DECODE2_ENGINE_ID:-p-cache-nixl-d2}"
# CSV forms for the proxy (aligned with num_prefillers / num_decoders).
export NIXL_PREFILL_ENGINE_IDS="${NIXL_PREFILL_ENGINE_IDS:-${NIXL_PREFILL1_ENGINE_ID},${NIXL_PREFILL2_ENGINE_ID}}"
export NIXL_DECODE_ENGINE_IDS="${NIXL_DECODE_ENGINE_IDS:-${NIXL_DECODE1_ENGINE_ID},${NIXL_DECODE2_ENGINE_ID}}"

export NIXL_KV_IP="${NIXL_KV_IP:-127.0.0.1}"
# Unique kv_port per instance to avoid same-host bind conflicts.
export NIXL_PREFILL1_KV_PORT="${NIXL_PREFILL1_KV_PORT:-14579}"
export NIXL_PREFILL2_KV_PORT="${NIXL_PREFILL2_KV_PORT:-14581}"
export NIXL_DECODE1_KV_PORT="${NIXL_DECODE1_KV_PORT:-14580}"
export NIXL_DECODE2_KV_PORT="${NIXL_DECODE2_KV_PORT:-14582}"
export NIXL_KV_BUFFER_DEVICE="${NIXL_KV_BUFFER_DEVICE:-cuda}"
export NIXL_KV_BUFFER_SIZE="${NIXL_KV_BUFFER_SIZE:-5368709120}"

export NIXL_SIDE_CHANNEL_HOST="${NIXL_SIDE_CHANNEL_HOST:-localhost}"
export NIXL_PREFILL1_SIDE_CHANNEL_PORT="${NIXL_PREFILL1_SIDE_CHANNEL_PORT:-5600}"
export NIXL_DECODE1_SIDE_CHANNEL_PORT="${NIXL_DECODE1_SIDE_CHANNEL_PORT:-5601}"
export NIXL_PREFILL2_SIDE_CHANNEL_PORT="${NIXL_PREFILL2_SIDE_CHANNEL_PORT:-5602}"
export NIXL_DECODE2_SIDE_CHANNEL_PORT="${NIXL_DECODE2_SIDE_CHANNEL_PORT:-5603}"
export NIXL_PREFILL_SIDE_CHANNEL_PORTS="${NIXL_PREFILL_SIDE_CHANNEL_PORTS:-${NIXL_PREFILL1_SIDE_CHANNEL_PORT},${NIXL_PREFILL2_SIDE_CHANNEL_PORT}}"
export UCX_TLS="${UCX_TLS:-cuda_ipc,cuda_copy,tcp}"

export LMCACHE_MP_SERVER_URL="${LMCACHE_MP_SERVER_URL:-tcp://127.0.0.1:${LMCACHE_MP_PORT}}"
