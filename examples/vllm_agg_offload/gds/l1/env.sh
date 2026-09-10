#!/usr/bin/env bash
# Copyright (c) 2026, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

# Shared defaults for aggregated vLLM + LMCache MP with GDS L1 slab.
#
# Topology: lmcache server (--gds-l1-path) + one vLLM process (TP=2). No P/D.
# GDS_L1_PATH is required (no default): directory for lmcache_gds_slab.bin.
#
# Override examples:
#   GDS_L1_PATH=/mnt/nvme/lmcache-gds-l1 CUDA_VISIBLE_DEVICES=0,1 bash start_server.sh

# Loopback curl must bypass HTTP(S)_PROXY (e.g. corporate Squid).
export NO_PROXY="127.0.0.1,localhost,::1${NO_PROXY:+,${NO_PROXY}}"
export no_proxy="127.0.0.1,localhost,::1${no_proxy:+,${no_proxy}}"

# Required for stable cache keys and correct H2D layout on Iluvatar.
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
export LMCACHE_MP_SERVER_URL="${LMCACHE_MP_SERVER_URL:-tcp://127.0.0.1:${LMCACHE_MP_PORT}}"

# GDS L1: required. Sizes the slab via LMCACHE_L1_SIZE_GB; file is
# ${GDS_L1_PATH}/lmcache_gds_slab.bin.
if [[ -z "${GDS_L1_PATH:-}" ]]; then
  echo "GDS_L1_PATH is required (NVMe directory for the GDS L1 slab)." >&2
  echo "Example: GDS_L1_PATH=/mnt/nvme/lmcache-gds-l1 bash start_lmcache_mp.sh" >&2
  exit 2
fi
export GDS_L1_PATH
# Default matches upstream --gds-l1-use-direct-io (True). Set 0/false to pass
# --no-gds-l1-use-direct-io.
export GDS_L1_USE_DIRECT_IO="${GDS_L1_USE_DIRECT_IO:-1}"
