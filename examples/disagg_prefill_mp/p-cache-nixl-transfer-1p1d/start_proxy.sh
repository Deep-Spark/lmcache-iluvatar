#!/usr/bin/env bash
# Copyright (c) 2026, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

# Start the 1P1D disaggregated proxy.
#
# Keeps LMCache telemetry wait (MP store finished) AND injects NixlPush
# kv_transfer_params so D actually receives P→D KV (not recompute prefill).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "${SCRIPT_DIR}/env.sh"

PROXY_ARGS=(
  --port "${PROXY_PORT}"
  --prefiller-host localhost --prefiller-port "${PREFILLER1_PORT}"
  --num-prefillers 1
  --decoder-host localhost --decoder-port "${DECODER1_PORT}"
  --num-decoders 1
  --telemetry-port "${TELEMETRY_PORT}"
  --nixl-push
  --nixl-prefill-engine-id "${NIXL_PREFILL_ENGINE_ID}"
  --nixl-decode-engine-id "${NIXL_DECODE_ENGINE_ID}"
  --nixl-prefill-side-channel-host "${NIXL_SIDE_CHANNEL_HOST}"
  --nixl-prefill-side-channel-port "${NIXL_PREFILL_SIDE_CHANNEL_PORT}"
  --nixl-tp-size "${TENSOR_PARALLEL}"
)

python3 "${MP_ROOT}/disagg_proxy_server_nixl_push.py" "${PROXY_ARGS[@]}"
