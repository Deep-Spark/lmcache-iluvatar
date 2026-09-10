#!/usr/bin/env bash
# Copyright (c) 2026, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

# Start the full MP 2P2D stack for smoke CI.
# Requires: profile env sourced; PROFILE_DIR set; ci/lib.sh loaded.

set -euo pipefail

: "${PROFILE_DIR:?PROFILE_DIR is not set}"

mp_run_stack_script() {
  local name="$1"
  local script="$2"
  ci_start_bg "${name}" bash "${PROFILE_DIR}/${script}"
}

mp_ci_start_lmcache_server() {
  ci_start_bg lmcache_server bash "${PROFILE_DIR}/start_lmcache_mp.sh"
  ci_wait_for_port "127.0.0.1" "${LMCACHE_MP_PORT}"
  ci_wait_for_http "${LMCACHE_URL}/metrics"
}

mp_ci_start_proxy_and_workers() {
  mp_run_stack_script proxy start_proxy.sh
  ci_wait_for_port "127.0.0.1" "${TELEMETRY_PORT}"

  mp_run_stack_script decoder1 start_d1.sh
  ci_wait_for_http "http://127.0.0.1:${DECODER1_PORT}/health"

  mp_run_stack_script decoder2 start_d2.sh
  ci_wait_for_http "http://127.0.0.1:${DECODER2_PORT}/health"

  mp_run_stack_script prefiller1 start_p1.sh
  ci_wait_for_http "http://127.0.0.1:${PREFILLER1_PORT}/health"

  mp_run_stack_script prefiller2 start_p2.sh
  ci_wait_for_http "http://127.0.0.1:${PREFILLER2_PORT}/health"

  ci_wait_for_http "http://127.0.0.1:${PROXY_PORT}/v1/models"
}
