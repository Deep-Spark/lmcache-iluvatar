#!/usr/bin/env bash
# Copyright (c) 2026, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

# Shared defaults for the aggregated vLLM + LMCache MP + Valkey Cluster test.

# Loopback HTTP requests must bypass corporate proxies.
export NO_PROXY="127.0.0.1,localhost,::1${NO_PROXY:+,${NO_PROXY}}"
export no_proxy="127.0.0.1,localhost,::1${no_proxy:+,${no_proxy}}"

export PYTHONHASHSEED="${PYTHONHASHSEED:-42}"
export VLLM_KV_CACHE_LAYOUT="${VLLM_KV_CACHE_LAYOUT:-HND}"
# Qwen3.5 hybrid groups have different KV shapes and must not share blocks
# across groups. This matches the validated production deployment contract.
export VLLM_KV_DISABLE_CROSS_GROUP_SHARE="${VLLM_KV_DISABLE_CROSS_GROUP_SHARE:-1}"
# CoreX TP startup must avoid NCCL cuMem/VMM shareable-handle import.
export NCCL_CUMEM_ENABLE="${NCCL_CUMEM_ENABLE:-0}"
export VLLM_SKIP_P2P_CHECK="${VLLM_SKIP_P2P_CHECK:-1}"

export MODEL_PATH="${MODEL_PATH:-/data/nlp/Qwen3.5-27B-W8A8-fixed}"
export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen3.5-27B-W8A8-fixed}"
export TENSOR_PARALLEL="${TENSOR_PARALLEL:-2}"
export GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"
# Qwen3.5 hybrid align reports logical block size N=784. The validated
# scheduler constraint is N <= max-num-batched-tokens < 2N, i.e. [784, 1568).
# Keep 1024 to match the validated qwen35-lmcache-mp-test36 deployment.
export MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-1024}"
export SERVER_HOST="${SERVER_HOST:-127.0.0.1}"
export SERVER_PORT="${SERVER_PORT:-18030}"

export LMCACHE_MP_PORT="${LMCACHE_MP_PORT:-6575}"
export LMCACHE_HTTP_PORT="${LMCACHE_HTTP_PORT:-8090}"
export LMCACHE_URL="${LMCACHE_URL:-http://127.0.0.1:${LMCACHE_HTTP_PORT}}"
export LMCACHE_MP_SERVER_URL="${LMCACHE_MP_SERVER_URL:-tcp://127.0.0.1:${LMCACHE_MP_PORT}}"
export LMCACHE_CHUNK_SIZE="${LMCACHE_CHUNK_SIZE:-784}"
export LMCACHE_L1_SIZE_GB="${LMCACHE_L1_SIZE_GB:-8}"
export LMCACHE_L1_INIT_SIZE_GB="${LMCACHE_L1_INIT_SIZE_GB:-1}"
export LMCACHE_MAX_WORKERS="${LMCACHE_MAX_WORKERS:-$((2 * TENSOR_PARALLEL))}"
export LMCACHE_L2_PREFETCH_MAX_IN_FLIGHT="${LMCACHE_L2_PREFETCH_MAX_IN_FLIGHT:-8}"
export LMCACHE_L2_STORE_POLICY="${LMCACHE_L2_STORE_POLICY:-default}"

# VALKEY_STARTUP_NODES intentionally has no default. The test must never write
# to an arbitrary cluster just because one happens to be reachable.
export VALKEY_CLUSTER_MODE="${VALKEY_CLUSTER_MODE:-true}"
export VALKEY_NUM_WORKERS="${VALKEY_NUM_WORKERS:-8}"
export VALKEY_REQUEST_TIMEOUT="${VALKEY_REQUEST_TIMEOUT:-5}"
export VALKEY_CONNECTION_TIMEOUT="${VALKEY_CONNECTION_TIMEOUT:-10}"
export VALKEY_KEY_PREFIX="${VALKEY_KEY_PREFIX:-lmcache-iluvatar-valkey-cluster-smoke}"
export VALKEY_MAX_CAPACITY_GB="${VALKEY_MAX_CAPACITY_GB:-0}"

export EXPECTED_ANSWER="${EXPECTED_ANSWER:-7319}"
export PROMPT_REPEAT="${PROMPT_REPEAT:-160}"
export MIN_PROMPT_TOKENS="${MIN_PROMPT_TOKENS:-$((4 * LMCACHE_CHUNK_SIZE))}"
export REQUEST_MAX_TOKENS="${REQUEST_MAX_TOKENS:-32}"
