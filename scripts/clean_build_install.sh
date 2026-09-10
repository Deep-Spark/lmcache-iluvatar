#!/usr/bin/env bash
# Copyright (c) 2026, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

# Full rebuild: clean -> build (deps + wheel) -> install plugin wheel.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

bash "${ROOT}/scripts/clean_lmcache.sh"
bash "${ROOT}/scripts/build_lmcache.sh"
bash "${ROOT}/scripts/install_lmcache.sh"

echo "==> clean build install complete"
