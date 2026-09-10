#!/usr/bin/env bash
# Copyright (c) 2026, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

# Install the latest lmcache-iluvatar wheel from dist/ into the active environment.
# Runtime deps (upstream LMCache from third_party/LMCache, torch, etc.) must
# already be installed — e.g. via scripts/install_dependencies.sh. The wheel is
# installed with --no-deps so pip does not re-resolve dependencies (avoids
# pulling PyPI torch over CoreX torch).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

TARGET_DIR="${TARGET_DIR:-}"
PYTHON_PATH="$(command -v python3)"
PKG_DIR="dist"
PKG_NAME="lmcache_iluvatar"

if [[ ! -d "${PKG_DIR}" ]]; then
  echo "ERROR: package directory ${PKG_DIR} does not exist; run scripts/build_lmcache.sh first" >&2
  exit 1
fi

latest_pkg="$(ls -t "${PKG_DIR}"/${PKG_NAME}-*.whl 2>/dev/null | head -1 || true)"
if [[ -z "${latest_pkg}" ]]; then
  echo "ERROR: cannot find ${PKG_NAME} wheel under ${PKG_DIR}" >&2
  exit 1
fi

echo "==> install wheel: ${latest_pkg}"

if [[ -n "${TARGET_DIR}" ]]; then
  PYTHON_DIST_PATH="${TARGET_DIR}/lib/python3/dist-packages"
  mkdir -p "${PYTHON_DIST_PATH}"
  "${PYTHON_PATH}" -m pip install --upgrade --no-deps -t "${PYTHON_DIST_PATH}" "${latest_pkg}"
  echo "lmcache-iluvatar installed in ${PYTHON_DIST_PATH}; add it to PYTHONPATH if needed."
  echo "NOTE: pip install -t targets are not site dirs, so lmcache_iluvatar.pth is" \
       "not processed automatically; call site.addsitedir('${PYTHON_DIST_PATH}')" \
       "to activate the plugin outside vLLM."
else
  "${PYTHON_PATH}" -m pip uninstall -y lmcache-iluvatar 2>/dev/null || true
  "${PYTHON_PATH}" -m pip install --no-deps "${latest_pkg}"
fi

echo "==> install complete"
