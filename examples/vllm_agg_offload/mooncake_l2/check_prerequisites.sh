#!/usr/bin/env bash
# Copyright (c) 2026, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

# Read-only preflight for the Mooncake L2 example.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "${SCRIPT_DIR}/env.sh"

case "${MOONCAKE_PROTOCOL}" in
  tcp | rdma) ;;
  *)
    echo "MOONCAKE_PROTOCOL must be tcp or rdma, got: ${MOONCAKE_PROTOCOL}" >&2
    exit 2
    ;;
esac

for command_name in mooncake_master mooncake_client lmcache python3; do
  command -v "${command_name}" >/dev/null || {
    echo "missing command: ${command_name}" >&2
    exit 1
  }
  echo "command ${command_name}: OK"
done

python3 - <<'PY'
import importlib
import importlib.metadata

print("lmcache=", importlib.metadata.version("lmcache"), sep="")
print("lmcache-iluvatar=", importlib.metadata.version("lmcache-iluvatar"), sep="")
importlib.import_module("openai")
importlib.import_module("pandas")
import lmcache_iluvatar  # noqa: F401

plugin = importlib.import_module("lmcache_iluvatar.lmcache_mooncake")
redirected = importlib.import_module("lmcache.lmcache_mooncake")
adapter = importlib.import_module(
    "lmcache.v1.distributed.l2_adapters.mooncake_store_l2_adapter"
)
assert redirected is plugin
assert hasattr(plugin, "LMCacheMooncakeClient")
assert hasattr(adapter, "MooncakeStoreL2AdapterConfig")
print("LongQA dependencies: OK")
print("Mooncake native redirect and MP L2 adapter: OK")
PY

native_so="$(python3 - <<'PY'
from importlib.machinery import EXTENSION_SUFFIXES
from importlib.metadata import distribution

dist = distribution("lmcache-iluvatar")
matches = [
    dist.locate_file(entry)
    for entry in (dist.files or ())
    if entry.name.startswith("lmcache_mooncake")
    and any(entry.name.endswith(suffix) for suffix in EXTENSION_SUFFIXES)
]
print(matches[0] if len(matches) == 1 else "")
PY
)"
if [[ -z "${native_so}" || ! -f "${native_so}" ]]; then
  echo "cannot locate lmcache_iluvatar.lmcache_mooncake native extension" >&2
  exit 1
fi
if ldd "${native_so}" | grep -q 'not found'; then
  echo "native extension has unresolved libraries: ${native_so}" >&2
  ldd "${native_so}" >&2
  exit 1
fi
echo "native extension dependencies: OK"

if [[ "${MOONCAKE_PROTOCOL}" == "rdma" ]]; then
  if [[ -z "${MOONCAKE_RDMA_DEVICES}" ]]; then
    echo "MOONCAKE_RDMA_DEVICES is required for RDMA" >&2
    exit 2
  fi
  command -v ibv_devinfo >/dev/null || {
    echo "ibv_devinfo is required for RDMA preflight" >&2
    exit 1
  }
  IFS=',' read -ra devices <<<"${MOONCAKE_RDMA_DEVICES}"
  for device in "${devices[@]}"; do
    device="${device//[[:space:]]/}"
    [[ -n "${device}" ]] || continue
    if [[ ! -e "/sys/class/infiniband/${device}" ]]; then
      echo "RDMA device is not visible: ${device}" >&2
      exit 1
    fi
    device_info="$(ibv_devinfo -d "${device}")"
    if ! grep -Eq 'state:[[:space:]]+PORT_ACTIVE' <<<"${device_info}"; then
      echo "RDMA device has no ACTIVE port: ${device}" >&2
      exit 1
    fi
    echo "RDMA device ${device}: visible with ACTIVE port"
  done
  if [[ ! -e /dev/infiniband/rdma_cm ]]; then
    echo "/dev/infiniband/rdma_cm is not mounted" >&2
    exit 1
  fi
  if ! compgen -G '/dev/infiniband/uverbs*' >/dev/null; then
    echo "no /dev/infiniband/uverbs* device is mounted" >&2
    exit 1
  fi
  echo "RDMA character devices: OK"
  memlock_limit="$(ulimit -l)"
  if [[ "${memlock_limit}" != "unlimited" ]]; then
    echo "WARNING: memlock is ${memlock_limit}; verify it covers registered Mooncake memory" >&2
  else
    echo "memlock: unlimited"
  fi
fi

echo "PREFLIGHT_OK protocol=${MOONCAKE_PROTOCOL}"
