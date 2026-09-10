#!/usr/bin/env bash
# Copyright (c) 2026, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

# Shared defaults for 2P2D PDBackend local-tiered offload (no shared MP).
#
# Comparison counterpart to:
#   examples/disagg_prefill_mp/p-cache-nixl-transfer-2p2d/
# Proxy uses synchronized RR (P_i↔D_i), not nested P×D product RR.

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
export MODEL_NAME="${MODEL_NAME:-${SERVED_MODEL_NAME}}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-40960}"
export GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.8}"
export TENSOR_PARALLEL="${TENSOR_PARALLEL:-2}"

# GPU layout (8 cards). start_*.sh set CUDA_VISIBLE_DEVICES from these.
export PREFILLER1_GPU="${PREFILLER1_GPU:-0,1}"
export PREFILLER2_GPU="${PREFILLER2_GPU:-2,3}"
export DECODER1_GPU="${DECODER1_GPU:-4,5}"
export DECODER2_GPU="${DECODER2_GPU:-6,7}"

# Per-instance LMCache configs (distinct local_disk paths).
export PREFILLER1_CONFIG="${PREFILLER1_CONFIG:-${OFFLOAD_DIR}/configs/lmcache-local-tiered/prefiller1.yaml}"
export PREFILLER2_CONFIG="${PREFILLER2_CONFIG:-${OFFLOAD_DIR}/configs/lmcache-local-tiered/prefiller2.yaml}"
export DECODER1_CONFIG="${DECODER1_CONFIG:-${OFFLOAD_DIR}/configs/lmcache-local-tiered/decoder1.yaml}"
export DECODER2_CONFIG="${DECODER2_CONFIG:-${OFFLOAD_DIR}/configs/lmcache-local-tiered/decoder2.yaml}"

# Per-instance disk cache roots (start_*.sh only clears its own dir).
export PREFILLER1_CACHE_DIR="${PREFILLER1_CACHE_DIR:-/tmp/.cache/pd-2p2d-p1}"
export PREFILLER2_CACHE_DIR="${PREFILLER2_CACHE_DIR:-/tmp/.cache/pd-2p2d-p2}"
export DECODER1_CACHE_DIR="${DECODER1_CACHE_DIR:-/tmp/.cache/pd-2p2d-d1}"
export DECODER2_CACHE_DIR="${DECODER2_CACHE_DIR:-/tmp/.cache/pd-2p2d-d2}"

# HTTP ports (17xxx/19xxx to avoid clash with nixl-2p2d 8100/9110).
# Soft defaults only. If this shell already sourced disagg_prefill_mp/env.defaults.sh
# (PREFILLER1_PORT=8100 etc.), unset those vars or override them before source.
export PREFILLER1_PORT="${PREFILLER1_PORT:-17100}"
export PREFILLER2_PORT="${PREFILLER2_PORT:-17101}"
export DECODER1_PORT="${DECODER1_PORT:-17200}"
export DECODER2_PORT="${DECODER2_PORT:-17201}"
export PUBLIC_PORT="${PUBLIC_PORT:-19100}"
export PROXY_ZMQ_PORT="${PROXY_ZMQ_PORT:-17500}"
export DECODER_INIT_PORTS="${DECODER_INIT_PORTS:-17300,17301}"
export DECODER_ALLOC_PORTS="${DECODER_ALLOC_PORTS:-17400,17401}"
# Alias for bench_lib / smoke (bench_wait_for_2p2d_stack).
export PROXY_PORT="${PUBLIC_PORT}"

# Unique LMCache RPC names per instance.
export PREFILLER1_RPC_PORT="${PREFILLER1_RPC_PORT:-producer1}"
export PREFILLER2_RPC_PORT="${PREFILLER2_RPC_PORT:-producer2}"
export DECODER1_RPC_PORT="${DECODER1_RPC_PORT:-consumer1}"
export DECODER2_RPC_PORT="${DECODER2_RPC_PORT:-consumer2}"
export SKIP_LAST_N_TOKENS="${SKIP_LAST_N_TOKENS:-1}"

# Must match configs/lmcache-local-tiered/*.yaml (used by disagg_proxy_server.py).
export LMCACHE_CHUNK_SIZE="${LMCACHE_CHUNK_SIZE:-256}"
export PD_BUFFER_SIZE="${PD_BUFFER_SIZE:-5368709120}"

export SMOKE_TIMEOUT="${SMOKE_TIMEOUT:-360}"
export PROMPT_REPEAT="${PROMPT_REPEAT:-120}"
export UCX_TLS="${UCX_TLS:-cuda_ipc,cuda_copy,tcp}"
export UCX_MEM_MMAP_HOOK_MODE="${UCX_MEM_MMAP_HOOK_MODE:-none}"
