#!/usr/bin/env bash
# Copyright (c) 2026, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

# Shared defaults for 1P1D local-tiered offload example (LMCache only, no Mooncake).
#
# Consumed by start-*.sh, ci/run.sh, ci/smoke_test.sh, and benchmark scripts.
# Override inline before invoking a script, e.g.:
#   MODEL_PATH=/data/nlp/Qwen3-8B/ CUDA_VISIBLE_DEVICES=0,1 bash start-prefiller.sh

OFFLOAD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${OFFLOAD_DIR}/../../.." && pwd)"

# Loopback curl must bypass HTTP(S)_PROXY (e.g. corporate Squid).
export NO_PROXY="127.0.0.1,localhost,::1${NO_PROXY:+,${NO_PROXY}}"
export no_proxy="127.0.0.1,localhost,::1${no_proxy:+,${no_proxy}}"

export PYTHONHASHSEED="${PYTHONHASHSEED:-0}"
# Iluvatar CoreX flash-attn is physically HND; leaving this unset makes vLLM
# default to NHD and LMCache will pick the wrong EngineKVFormat → garbled decode.
export VLLM_KV_CACHE_LAYOUT="${VLLM_KV_CACHE_LAYOUT:-HND}"

export MODEL_PATH="${MODEL_PATH:-/data/nlp/Qwen3-32B-W8A8/}"
export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen3-32B-W8A8}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-40960}"
export GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.8}"

export PREFILLER_CONFIG="${PREFILLER_CONFIG:-${OFFLOAD_DIR}/configs/lmcache-local-tiered/prefiller.yaml}"
export DECODER_CONFIG="${DECODER_CONFIG:-${OFFLOAD_DIR}/configs/lmcache-local-tiered/decoder.yaml}"

export PREFILLER_PORT="${PREFILLER_PORT:-17100}"
export DECODER_PORT="${DECODER_PORT:-17200}"
export PUBLIC_PORT="${PUBLIC_PORT:-19100}"
export PROXY_ZMQ_PORT="${PROXY_ZMQ_PORT:-17500}"
export DECODER_INIT_PORTS="${DECODER_INIT_PORTS:-17300,17301}"
export DECODER_ALLOC_PORTS="${DECODER_ALLOC_PORTS:-17400,17401}"

export PREFILLER_RPC_PORT="${PREFILLER_RPC_PORT:-producer1}"
export DECODER_RPC_PORT="${DECODER_RPC_PORT:-consumer1}"
export SKIP_LAST_N_TOKENS="${SKIP_LAST_N_TOKENS:-1}"

# Must match configs/lmcache-local-tiered/*.yaml (used by disagg_proxy_server.py).
export LMCACHE_CHUNK_SIZE="${LMCACHE_CHUNK_SIZE:-256}"
export PD_BUFFER_SIZE="${PD_BUFFER_SIZE:-5368709120}"

export SMOKE_TIMEOUT="${SMOKE_TIMEOUT:-360}"
export PROMPT_REPEAT="${PROMPT_REPEAT:-120}"
