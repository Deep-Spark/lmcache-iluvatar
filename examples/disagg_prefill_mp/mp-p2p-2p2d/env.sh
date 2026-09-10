#!/usr/bin/env bash
# Copyright (c) 2026, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

# 2P2D MP P2P profile: dual lmcache servers + coordinator + peer L1 RDMA read.
# Pair-1 (P1+D1) → Server A; Pair-2 (P2+D2) → Server B. No NIXL.
#
# Topology (connect / advertise addresses — never 0.0.0.0):
#   IP_A — primary: coordinator + proxy + Server A + P1 + D1
#   IP_B — secondary: Server B + P2 + D2
# Single-node (default): IP_A=IP_B=127.0.0.1
# Dual-node:             IP_A=<hostA> IP_B=<hostB>
# BIND_HOST is listen-only (coordinator / HTTP / P2P listen).

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
# longQA needs large context (align with p-cache-nixl-transfer-2p2d).
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-40960}"
export TENSOR_PARALLEL="${TENSOR_PARALLEL:-2}"

# Each server hosts 1P1D → 2×TP workers (not 4×TP like a shared single server).
export LMCACHE_MAX_WORKERS="${LMCACHE_MAX_WORKERS:-$((2 * TENSOR_PARALLEL))}"
# Per-server L1. 40-doc longQA needs headroom; default 200GB each (override if host RAM tight).
export LMCACHE_L1_SIZE_GB="${LMCACHE_L1_SIZE_GB:-200}"
# P2P tip: ≥64KB alignment for larger RDMA reads (upstream p2p.rst).
export LMCACHE_L1_ALIGN_BYTES="${LMCACHE_L1_ALIGN_BYTES:-65536}"

export PYTHONHASHSEED="${PYTHONHASHSEED:-0}"
export VLLM_KV_CACHE_LAYOUT="${VLLM_KV_CACHE_LAYOUT:-HND}"

# --- Host topology -----------------------------------------------------------
export IP_A="${IP_A:-127.0.0.1}"
export IP_B="${IP_B:-127.0.0.1}"
# Listen/bind only. Do not use as advertise or client connect target.
export BIND_HOST="${BIND_HOST:-0.0.0.0}"
# ZMQ MQ bind. Must be reachable by the peer server for P2P lookup RPC
# (dual-node). P/D still connect via localhost (works when bound to 0.0.0.0).
# Override to 127.0.0.1 only for strict single-node isolation.
export LMCACHE_ZMQ_HOST="${LMCACHE_ZMQ_HOST:-${BIND_HOST}}"

# GPU layout. Single-node 8-card default; dual-node override per host
# (e.g. on A: P1=0,1 D1=2,3; on B: P2=0,1 D2=2,3).
export PREFILLER1_GPU="${PREFILLER1_GPU:-0,1}"
export PREFILLER2_GPU="${PREFILLER2_GPU:-2,3}"
export DECODER1_GPU="${DECODER1_GPU:-4,5}"
export DECODER2_GPU="${DECODER2_GPU:-6,7}"

# Coordinator (membership only) — process on IP_A.
export LMCACHE_COORDINATOR_HOST="${LMCACHE_COORDINATOR_HOST:-${BIND_HOST}}"
export LMCACHE_COORDINATOR_PORT="${LMCACHE_COORDINATOR_PORT:-9300}"
export LMCACHE_COORDINATOR_URL="${LMCACHE_COORDINATOR_URL:-http://${IP_A}:${LMCACHE_COORDINATOR_PORT}}"

# Server A (pair-1: P1+D1).
export LMCACHE_MP_PORT_A="${LMCACHE_MP_PORT_A:-6555}"
export LMCACHE_HTTP_PORT_A="${LMCACHE_HTTP_PORT_A:-8080}"
export LMCACHE_HTTP_HOST_A="${LMCACHE_HTTP_HOST_A:-${BIND_HOST}}"
export LMCACHE_P2P_PORT_A="${LMCACHE_P2P_PORT_A:-8555}"
export LMCACHE_P2P_ADVERTISE_URL_A="${LMCACHE_P2P_ADVERTISE_URL_A:-${IP_A}:${LMCACHE_P2P_PORT_A}}"
export LMCACHE_P2P_LISTEN_URL_A="${LMCACHE_P2P_LISTEN_URL_A:-${BIND_HOST}:${LMCACHE_P2P_PORT_A}}"
export LMCACHE_COORDINATOR_ADVERTISE_IP_A="${LMCACHE_COORDINATOR_ADVERTISE_IP_A:-${IP_A}}"
export LMCACHE_INSTANCE_ID_A="${LMCACHE_INSTANCE_ID_A:-mp-p2p-node-a}"
export LMCACHE_URL_A="${LMCACHE_URL_A:-http://${IP_A}:${LMCACHE_HTTP_PORT_A}}"

# Server B (pair-2: P2+D2).
export LMCACHE_MP_PORT_B="${LMCACHE_MP_PORT_B:-6556}"
export LMCACHE_HTTP_PORT_B="${LMCACHE_HTTP_PORT_B:-8081}"
export LMCACHE_HTTP_HOST_B="${LMCACHE_HTTP_HOST_B:-${BIND_HOST}}"
export LMCACHE_P2P_PORT_B="${LMCACHE_P2P_PORT_B:-8556}"
export LMCACHE_P2P_ADVERTISE_URL_B="${LMCACHE_P2P_ADVERTISE_URL_B:-${IP_B}:${LMCACHE_P2P_PORT_B}}"
export LMCACHE_P2P_LISTEN_URL_B="${LMCACHE_P2P_LISTEN_URL_B:-${BIND_HOST}:${LMCACHE_P2P_PORT_B}}"
export LMCACHE_COORDINATOR_ADVERTISE_IP_B="${LMCACHE_COORDINATOR_ADVERTISE_IP_B:-${IP_B}}"
export LMCACHE_INSTANCE_ID_B="${LMCACHE_INSTANCE_ID_B:-mp-p2p-node-b}"
export LMCACHE_URL_B="${LMCACHE_URL_B:-http://${IP_B}:${LMCACHE_HTTP_PORT_B}}"

# Proxy reaches P/D by host (same-idx RR: P1↔D1 on IP_A, P2↔D2 on IP_B).
export PREFILLER_HOSTS="${PREFILLER_HOSTS:-${IP_A},${IP_B}}"
export DECODER_HOSTS="${DECODER_HOSTS:-${IP_A},${IP_B}}"
# Telemetry lives with proxy on IP_A.
export LMCACHE_REQUEST_TELEMETRY_ENDPOINT="${LMCACHE_REQUEST_TELEMETRY_ENDPOINT:-http://${IP_A}:${TELEMETRY_PORT}/api/v1/telemetry}"

# Keep defaults aliases pointing at A for any shared tooling that reads LMCACHE_MP_PORT.
export LMCACHE_MP_PORT="${LMCACHE_MP_PORT_A}"
export LMCACHE_HTTP_PORT="${LMCACHE_HTTP_PORT_A}"
export LMCACHE_URL="${LMCACHE_URL_A}"
# Back-compat alias used by older notes / smoke overrides.
export LMCACHE_COORDINATOR_ADVERTISE_IP="${LMCACHE_COORDINATOR_ADVERTISE_IP:-${LMCACHE_COORDINATOR_ADVERTISE_IP_A}}"
