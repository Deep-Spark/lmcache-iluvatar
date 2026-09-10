#!/usr/bin/env bash
# Copyright (c) 2026, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

# Remove lmcache-iluvatar build artifacts and uninstall the package from the active env.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_PATH="$(command -v python3)"

echo "==> clean build directories"
rm -rf build/ dist/ lmcache_iluvatar.egg-info

echo "==> remove stale native extensions from source tree"
# Editable installs and PYTHONPATH=src layouts load *.so next to package code.
# Leftover artifacts shadow the wheel/site-packages build and break ABI checks.
rm -f lmcache_iluvatar/*.so lmcache_iluvatar/_c_ops*.so

if [[ -f setup.py ]]; then
  echo "==> setup.py clean"
  "${PYTHON_PATH}" setup.py clean 2>/dev/null || true
fi

echo "==> uninstall lmcache-iluvatar"
"${PYTHON_PATH}" -m pip uninstall -y lmcache-iluvatar 2>/dev/null || true

echo "==> remove orphan .pth import hook from site-packages"
# Wheels built before the .pth moved into the wheel RECORD wrote these straight
# to site-packages as a build side effect, so pip uninstall never removed them.
# A leftover .pth pointing at an uninstalled package makes every Python process
# print an import traceback at startup.
while IFS= read -r sp_dir; do
  [[ -z "${sp_dir}" ]] && continue
  for name in lmcache_iluvatar.pth _lmcache_iluvatar_hook.py; do
    path="${sp_dir}/${name}"
    if [[ -f "${path}" ]]; then
      if rm -f "${path}"; then
        echo "Removed ${path}"
      else
        echo "Warning: could not remove ${path}"
      fi
    fi
  done
done < <("${PYTHON_PATH}" -c 'import site; print("\n".join(site.getsitepackages()))')

echo "==> clean complete"
