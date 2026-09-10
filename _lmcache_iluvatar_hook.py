# Copyright (c) 2026, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Meta-path hook installed by lmcache_iluvatar.pth.

This module must be importable without triggering ``lmcache_iluvatar/__init__.py``.
It lives at site-packages level (not inside the package) so it can be imported
by the .pth file before any user code runs.
"""

from __future__ import annotations

import sys
from importlib import machinery

_MULTI_CONNECTOR = (
    "vllm.distributed.kv_transfer.kv_connector.v1.multi_connector"
)
_LMCACHE_MP_CONNECTOR = (
    "vllm.distributed.kv_transfer.kv_connector.v1.lmcache_mp_connector"
)


class _LMCacheIluvatarHook:
    _lmcache_done: bool = False
    _worker_done: bool = False
    _multi_connector_done: bool = False
    _lmcache_mp_connector_done: bool = False

    def find_spec(self, fullname, path, target=None):
        if fullname == "lmcache" and not _LMCacheIluvatarHook._lmcache_done:
            _LMCacheIluvatarHook._lmcache_done = True
            import lmcache_iluvatar  # noqa: F401
        if (
            fullname == "vllm.v1.worker.gpu_worker"
            and not _LMCacheIluvatarHook._worker_done
        ):
            return self._wrap_spec(fullname, path, _post_vllm_import)
        if (
            fullname == _MULTI_CONNECTOR
            and not _LMCacheIluvatarHook._multi_connector_done
        ):
            return self._wrap_spec(fullname, path, _post_vllm_import)
        if (
            fullname == _LMCACHE_MP_CONNECTOR
            and not _LMCacheIluvatarHook._lmcache_mp_connector_done
        ):
            return self._wrap_spec(fullname, path, _post_vllm_import)
        return None

    def _wrap_spec(self, fullname, path, post_exec):
        spec = machinery.PathFinder.find_spec(fullname, path)
        if spec is None or spec.loader is None:
            return None
        exec_module = getattr(spec.loader, "exec_module", None)
        if exec_module is None:
            return spec
        spec.loader = _PostImportLoader(spec.loader, post_exec)
        return spec


class _PostImportLoader:
    def __init__(self, wrapped, post_exec):
        self._wrapped = wrapped
        self._post_exec = post_exec

    def create_module(self, spec):
        create_module = getattr(self._wrapped, "create_module", None)
        if create_module is None:
            return None
        return create_module(spec)

    def exec_module(self, module):
        self._wrapped.exec_module(module)
        self._post_exec(module)


def _post_vllm_import(module):
    """Ensure plugin patches run after key vLLM modules finish importing.

    Covers CacheBlend ``Worker`` and MultiConnector save-to-all blocks. Idempotent:
    ``activate_patches`` only applies once per process.
    LMCacheMPConnector also uses the runtime registry retry after class definition.
    """
    name = getattr(module, "__name__", "")
    if name == "vllm.v1.worker.gpu_worker":
        _LMCacheIluvatarHook._worker_done = True
    if name == _MULTI_CONNECTOR:
        _LMCacheIluvatarHook._multi_connector_done = True
    if name == _LMCACHE_MP_CONNECTOR:
        _LMCacheIluvatarHook._lmcache_mp_connector_done = True
    _LMCacheIluvatarHook._lmcache_done = True
    import lmcache_iluvatar

    lmcache_iluvatar.activate_post_import_patches(name)


sys.meta_path.insert(0, _LMCacheIluvatarHook())
