#!/usr/bin/env bash
# Copyright (c) 2026, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

# Install Iluvatar runtime dependencies from requirements.txt
# (keep in sync with pyproject.toml [project.dependencies]), then build and
# install upstream LMCache from the third_party/LMCache submodule as a
# pure-Python wheel (NO_NATIVE_EXT=1, --no-deps).
#
# --no-deps is required: upstream install_requires may pull cupy-cuda13x,
# which conflicts with CoreX-adapted CuPy. Native C++ extensions come from
# lmcache-iluvatar, not this upstream wheel.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LMCACHE_ROOT="${ROOT}/third_party/LMCache"
cd "$ROOT"

echo "==> upgrade pip tooling"
python3 -m pip install --upgrade --ignore-installed pip wheel

echo "==> install runtime deps (synced with pyproject.toml [project.dependencies])"
python3 -m pip install -r "${ROOT}/requirements.txt"

if [[ ! -f "${LMCACHE_ROOT}/pyproject.toml" ]]; then
  echo "ERROR: ${LMCACHE_ROOT} submodule is missing; run:" >&2
  echo "  git submodule update --init --recursive third_party/LMCache" >&2
  exit 1
fi

echo "==> install LMCache build tooling"
python3 -m pip install -r "${LMCACHE_ROOT}/requirements/build.txt"

# Do not forward LOCAL_VERSION_IDENTIFIER into the upstream lmcache wheel:
# plugin VersionRange pins expect base version == upstream_pin.json
# (e.g. 0.5.3), and setuptools_scm local suffixes would break that match.
echo "==> build upstream lmcache wheel (pure Python, NO_NATIVE_EXT=1)"
mkdir -p "${LMCACHE_ROOT}/dist"
(
  cd "${LMCACHE_ROOT}"
  # Drop any inherited local version env so the wheel stays on the tag version.
  env -u LOCAL_VERSION_IDENTIFIER \
    NO_NATIVE_EXT=1 python3 -m pip wheel . --no-build-isolation --no-deps -w dist/
)

latest_lmcache_wheel="$(ls -t "${LMCACHE_ROOT}"/dist/lmcache-*.whl 2>/dev/null | head -1 || true)"
if [[ -z "${latest_lmcache_wheel}" ]]; then
  echo "ERROR: no lmcache wheel found under ${LMCACHE_ROOT}/dist" >&2
  exit 1
fi

echo "==> install upstream lmcache wheel: ${latest_lmcache_wheel}"
python3 -m pip uninstall -y lmcache 2>/dev/null || true
python3 -m pip install --no-deps "${latest_lmcache_wheel}"

echo "==> install dependencies complete"
