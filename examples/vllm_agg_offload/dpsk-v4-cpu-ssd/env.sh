#!/usr/bin/env bash
# Copyright (c) 2026, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

# Aggregated DeepSeek-V4 + LMCache MP Server: CPU-L1 vs SSD-L2 comparison.
#
# Topology: one lmcache server (ZMQ/IPC) + one vLLM process (PP=16, kv_both).
# Two MP profiles (mutually exclusive — do not run both servers at once):
#   start_lmcache_mp_cpu.sh  — large L1 only
#   start_lmcache_mp_ssd.sh  — small L1 + filesystem L2 on SSD
#
# Override before sourcing / invoking, e.g.:
#   DOCUMENT_LENGTH=1000000 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 bash run-longQA-benchmark.sh

# Loopback curl must bypass HTTP(S)_PROXY (e.g. corporate Squid).
export NO_PROXY="127.0.0.1,localhost,::1${NO_PROXY:+,${NO_PROXY}}"
export no_proxy="127.0.0.1,localhost,::1${no_proxy:+,${no_proxy}}"

# Required for stable cache keys and correct H2D layout on Iluvatar.
export PYTHONHASHSEED="${PYTHONHASHSEED:-0}"
export VLLM_KV_CACHE_LAYOUT="${VLLM_KV_CACHE_LAYOUT:-HND}"

export MODEL_PATH="${MODEL_PATH:-/data/nlp/DeepSeek-V4-Flash-w4a8-v2-TN/}"
# Must match the model — do not reuse Qwen served-name (cache-key pollution).
export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-DeepSeek-V4-Flash-w4a8-v2-TN}"
export MODEL_NAME="${MODEL_NAME:-${SERVED_MODEL_NAME}}"

# Align with known-good DeepSeek-V4 BI serve (see p-cache-nixl-transfer-1p1d-dpsk-v4).
export GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.85}"
# Cover LongQA DOCUMENT_LENGTH=1e6 / 2e5; override if the checkpoint rejects this.
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-1048576}"
# Chunked prefill budget requested for the 200k / 1M LongQA workloads.
export MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-2048}"
export TENSOR_PARALLEL="${TENSOR_PARALLEL:-1}"
export PIPELINE_PARALLEL="${PIPELINE_PARALLEL:-16}"
export SERVER_HOST="${SERVER_HOST:-127.0.0.1}"
export SERVER_PORT="${SERVER_PORT:-18010}"

export IXFORMER_DS4_MHC_PRE_V2=2
export VLLM_ILU_DS4_PREFILL_AS_DECODE=1

# LMCache MP server (control plane ZMQ + HTTP metrics).
export LMCACHE_MP_PORT="${LMCACHE_MP_PORT:-6555}"
export LMCACHE_HTTP_PORT="${LMCACHE_HTTP_PORT:-8080}"
export LMCACHE_URL="${LMCACHE_URL:-http://localhost:${LMCACHE_HTTP_PORT}}"
export LMCACHE_CHUNK_SIZE="${LMCACHE_CHUNK_SIZE:-2048}"
export LMCACHE_MAX_WORKERS="${LMCACHE_MAX_WORKERS:-$((2 * TENSOR_PARALLEL * PIPELINE_PARALLEL))}"
export LMCACHE_MP_SERVER_URL="${LMCACHE_MP_SERVER_URL:-tcp://127.0.0.1:${LMCACHE_MP_PORT}}"
# PP16 STORE flood can starve PINGs; keep reap >= 3x heartbeat.
export LMCACHE_MP_HEARTBEAT_INTERVAL="${LMCACHE_MP_HEARTBEAT_INTERVAL:-300}"
export LMCACHE_WORKER_REAP_TIMEOUT_SECONDS="${LMCACHE_WORKER_REAP_TIMEOUT_SECONDS:-900}"

# LongQA drained-store baseline. Wait after the cold request returns so queued
# STORE work can drain before the warm query. Set 0 for immediate-query HOL.
export WARMUP_QUERY_WAIT_SECONDS="${WARMUP_QUERY_WAIT_SECONDS:-600}"

# CPU-only profile: keep the working set in L1 DRAM.
export LMCACHE_CPU_L1_SIZE_GB="${LMCACHE_CPU_L1_SIZE_GB:-200}"

# SSD profile: small L1 staging (GPU→CPU→Disk) + FS L2 on disk.
# Principle: warm hits should come from SSD; L1 is only a staging buffer.
# Server knobs (wired by start_lmcache_mp_ssd.sh):
#   --l1-size-gb              LMCACHE_SSD_L1_SIZE_GB
#   --l2-adapter              fs_native + LMCACHE_L2_BASE_PATH / NUM_WORKERS / MAX_CAPACITY_GB
#   --l2-store-policy         LMCACHE_L2_STORE_POLICY (skip_l1 = free L1 after L2 store)
export LMCACHE_SSD_L1_SIZE_GB="${LMCACHE_SSD_L1_SIZE_GB:-16}"
export LMCACHE_L2_BASE_PATH="${LMCACHE_L2_BASE_PATH:-/data/tmp/lmcache-l2-dpsk-v4}"
export LMCACHE_L2_NUM_WORKERS="${LMCACHE_L2_NUM_WORKERS:-8}"
# fs_native: LMCache-driven L2 eviction budget; 0 = unlimited / no usage cap.
export LMCACHE_L2_MAX_CAPACITY_GB="${LMCACHE_L2_MAX_CAPACITY_GB:-500}"
# default = keep L1 after L2 store; skip_l1 = delete L1 (true staging → warm from SSD).
export LMCACHE_L2_STORE_POLICY="${LMCACHE_L2_STORE_POLICY:-skip_l1}"

# Back-compat alias used by start scripts that only need "current" L1 size.
# Prefer setting LMCACHE_CPU_L1_SIZE_GB / LMCACHE_SSD_L1_SIZE_GB; start_*_mp_*.sh
# sets LMCACHE_L1_SIZE_GB explicitly per profile.
export LMCACHE_L1_SIZE_GB="${LMCACHE_L1_SIZE_GB:-${LMCACHE_CPU_L1_SIZE_GB}}"
