#!/usr/bin/env bash
# Copyright (c) 2026, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

# Start the 1P1D disaggregated proxy with NixlPush kv_transfer_params.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "${SCRIPT_DIR}/env.sh"

# NixlPushConnector only activates when the proxy injects kv_transfer_params
# (do_remote_decode / do_remote_prefill) and shares X-Request-Id across P/D.
# Without that, D recomputes full prefill and nixl_bytes_transferred stays 0.
PROXY_ARGS=(
  --port "${PROXY_PORT}"
  --prefiller-host localhost --prefiller-port "${PREFILLER1_PORT}"
  --num-prefillers 1
  --decoder-host localhost --decoder-port "${DECODER1_PORT}"
  --num-decoders 1
  --telemetry-port "${TELEMETRY_PORT}"
  --skip-kv-notify-wait
  --nixl-push
  --nixl-prefill-engine-id "${NIXL_PREFILL_ENGINE_ID}"
  --nixl-decode-engine-id "${NIXL_DECODE_ENGINE_ID}"
  --nixl-prefill-side-channel-host "${NIXL_SIDE_CHANNEL_HOST}"
  --nixl-prefill-side-channel-port "${NIXL_PREFILL_SIDE_CHANNEL_PORT}"
  --nixl-tp-size "${TENSOR_PARALLEL}"
)

python3 "${MP_ROOT}/disagg_proxy_server_nixl_push.py" "${PROXY_ARGS[@]}"
