#!/usr/bin/env bash
# Copyright (c) 2026, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

# Install runtime deps (incl. upstream LMCache from third_party/LMCache), then
# build lmcache-iluvatar wheel into dist/.
# Does not clean or install the plugin wheel; pair with install_lmcache.sh,
# or run clean_lmcache.sh first for a full rebuild.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# Install deps first without LOCAL_VERSION_IDENTIFIER so the upstream lmcache
# wheel keeps the pinned tag version (see upstream_pin.json).
bash "${ROOT}/scripts/install_dependencies.sh"

COREX_VERSION="${COREX_VERSION:-latest}"
MAX_JOBS="${MAX_JOBS:-$(nproc --all)}"
export MAX_JOBS

if [[ "${COREX_VERSION}" == "latest" ]]; then
  COREX_VERSION="$(date --utc +%Y%m%d%H%M%S)"
fi
# Local suffix applies only to the lmcache-iluvatar plugin wheel.
export LOCAL_VERSION_IDENTIFIER="corex.${COREX_VERSION}"

echo "==> build wheel (version suffix: ${LOCAL_VERSION_IDENTIFIER})"
"${ROOT}/scripts/build_wheel.sh"

echo "==> build complete"
