#!/usr/bin/env bash
# Copyright (c) 2026, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

# Build lmcache-iluvatar wheel with Iluvatar native extensions.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

mkdir -p dist

# Host .cpp (e.g. pybind.cpp) pulls in torch headers and must satisfy
# c10/util/C++17.h (GCC 9+). CentOS default c++ is often too old; .cu files
# already use CoreX clang++ via torch cpp_extension, but plain .cpp does not.
_cuda_home="${CUDA_HOME:-/usr/local/corex}"
if [[ -z "${CXX:-}" && -x "${_cuda_home}/bin/clang++" ]]; then
  export CC="${CC:-${_cuda_home}/bin/clang}"
  export CXX="${_cuda_home}/bin/clang++"
  echo "==> host C++ compiler: CXX=${CXX}"
fi

echo "==> build wheel (Iluvatar native extensions)"
env -u NO_CUDA_EXT python3 -m pip wheel . --no-build-isolation --no-deps -w dist/

echo "==> wheel written to dist/"
ls -1 dist/lmcache_iluvatar-*.whl
