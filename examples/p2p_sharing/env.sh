#!/usr/bin/env bash
# Copyright (c) 2026, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

# Shared defaults for the P2P sharing example.
#
# Source this file from the example scripts. Override values before running a
# script, for example:
#   MODEL_PATH=/data/nlp/Qwen3-8B/ CUDA_VISIBLE_DEVICES=0 bash start_instance1.sh

P2P_EXAMPLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Loopback curl must bypass HTTP(S)_PROXY (e.g. corporate Squid).
export NO_PROXY="127.0.0.1,localhost,::1${NO_PROXY:+,${NO_PROXY}}"
export no_proxy="127.0.0.1,localhost,::1${no_proxy:+,${no_proxy}}"

export PYTHONHASHSEED="${PYTHONHASHSEED:-123}"
export MODEL_PATH="${MODEL_PATH:-/data/nlp/Qwen3-8B/}"
export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen3-8B}"
export GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.8}"

export CONTROLLER_HOST="${CONTROLLER_HOST:-localhost}"
export CONTROLLER_API_PORT="${CONTROLLER_API_PORT:-9000}"
export CONTROLLER_PULL_PORT="${CONTROLLER_PULL_PORT:-8300}"
export CONTROLLER_REPLY_PORT="${CONTROLLER_REPLY_PORT:-8400}"

export INSTANCE1_PORT="${INSTANCE1_PORT:-8010}"
export INSTANCE2_PORT="${INSTANCE2_PORT:-8011}"
export INSTANCE1_CONFIG_FILE="${INSTANCE1_CONFIG_FILE:-${P2P_EXAMPLE_DIR}/instance1.yaml}"
export INSTANCE2_CONFIG_FILE="${INSTANCE2_CONFIG_FILE:-${P2P_EXAMPLE_DIR}/instance2.yaml}"
