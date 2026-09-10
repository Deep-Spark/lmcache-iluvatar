#!/usr/bin/env bash
# Copyright (c) 2026, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

# Shared defaults for aggregated (single vLLM kv_both) + LMCache MP Server.
#
# Topology: lmcache server (ZMQ/IPC) + one vLLM process (TP=2). No P/D proxy.
# Override inline before invoking a script, e.g.:
#   MODEL_PATH=/data/nlp/Qwen3-8B CUDA_VISIBLE_DEVICES=0,1 bash start_server.sh

# Loopback curl must bypass HTTP(S)_PROXY (e.g. corporate Squid).
export NO_PROXY="127.0.0.1,localhost,::1${NO_PROXY:+,${NO_PROXY}}"
export no_proxy="127.0.0.1,localhost,::1${no_proxy:+,${no_proxy}}"

# Required for stable cache keys and correct H2D layout on Iluvatar (see
# .trellis/spec/backend/lmcache-vllm-runtime-env.md).
export PYTHONHASHSEED="${PYTHONHASHSEED:-0}"
export VLLM_KV_CACHE_LAYOUT="${VLLM_KV_CACHE_LAYOUT:-HND}"

export MODEL_PATH="${MODEL_PATH:-/data/nlp/Qwen3-8B}"
export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen3-8B}"
export GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-40960}"
export TENSOR_PARALLEL="${TENSOR_PARALLEL:-2}"
export SERVER_HOST="${SERVER_HOST:-127.0.0.1}"
export SERVER_PORT="${SERVER_PORT:-18010}"

# LMCache MP server (control plane ZMQ + HTTP metrics).
export LMCACHE_MP_PORT="${LMCACHE_MP_PORT:-6555}"
export LMCACHE_HTTP_PORT="${LMCACHE_HTTP_PORT:-8080}"
export LMCACHE_URL="${LMCACHE_URL:-http://localhost:${LMCACHE_HTTP_PORT}}"
export LMCACHE_CHUNK_SIZE="${LMCACHE_CHUNK_SIZE:-256}"
export LMCACHE_L1_SIZE_GB="${LMCACHE_L1_SIZE_GB:-100}"
export LMCACHE_MAX_WORKERS="${LMCACHE_MAX_WORKERS:-$((2 * TENSOR_PARALLEL))}"
# Version-tolerant MP connector URL (port alone also works on newer vLLM).
export LMCACHE_MP_SERVER_URL="${LMCACHE_MP_SERVER_URL:-tcp://127.0.0.1:${LMCACHE_MP_PORT}}"
