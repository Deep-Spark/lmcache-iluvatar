#!/usr/bin/env bash
# Copyright (c) 2026, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

# Shared MP disagg stack defaults (ports, chunk size). Profile env.sh sources this.

export NO_PROXY="127.0.0.1,localhost,::1${NO_PROXY:+,${NO_PROXY}}"
export no_proxy="127.0.0.1,localhost,::1${no_proxy:+,${no_proxy}}"

export VLLM_BLOCK_SIZE="${VLLM_BLOCK_SIZE:-16}"
export LMCACHE_CHUNK_SIZE="${LMCACHE_CHUNK_SIZE:-256}"

export LMCACHE_MP_PORT="${LMCACHE_MP_PORT:-6555}"
export LMCACHE_HTTP_PORT="${LMCACHE_HTTP_PORT:-8080}"
export LMCACHE_URL="${LMCACHE_URL:-http://localhost:${LMCACHE_HTTP_PORT}}"
export PROXY_PORT="${PROXY_PORT:-9110}"
export TELEMETRY_PORT="${TELEMETRY_PORT:-5768}"
export PREFILLER1_PORT="${PREFILLER1_PORT:-8100}"
export PREFILLER2_PORT="${PREFILLER2_PORT:-8101}"
export DECODER1_PORT="${DECODER1_PORT:-8200}"
export DECODER2_PORT="${DECODER2_PORT:-8201}"
