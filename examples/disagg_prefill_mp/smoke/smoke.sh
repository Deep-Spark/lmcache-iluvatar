#!/usr/bin/env bash
# Copyright (c) 2026, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

# Thin wrapper around examples/common/smoke_long_prompt_cache.py
#
# Usage:
#   bash smoke.sh
#   PROMPT_REPEAT=200 NUM_RUNS=4 bash smoke.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMMON_DIR="$(cd "${SCRIPT_DIR}/../../common" && pwd)"

# shellcheck source=env.sh
source "${SCRIPT_DIR}/env.sh"
# shellcheck source=../../common/bench_lib.sh
source "${COMMON_DIR}/bench_lib.sh"

export BENCH_LOG_PREFIX="${BENCH_LOG_PREFIX:-[smoke]}"

PROXY_HOST="${PROXY_HOST:-127.0.0.1}"
SERVER_PORT="${SERVER_PORT:-${PROXY_PORT}}"

if [[ "${SKIP_STACK_WAIT:-0}" != "1" ]]; then
  bench_wait_for_2p2d_stack "${PROXY_HOST}" "${SERVER_PORT}"
fi

exec python3 "${COMMON_DIR}/smoke_long_prompt_cache.py" "$@"
