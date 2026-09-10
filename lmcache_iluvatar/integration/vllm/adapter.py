# Copyright (c) 2026, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Iluvatar wrapper for LMCache's vLLM V1 adapter."""

from __future__ import annotations

import logging
from typing import Any, Optional, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=type)


def get_req_id(request: Any) -> str:
    """Return the LMCache cleanup id for a vLLM request."""

    disagg_spec = getattr(request, "disagg_spec", None)
    if disagg_spec is not None:
        return disagg_spec.req_id
    return request.req_id


def build_iluvatar_v1_impl(base_cls: T) -> T:
    """Create an Iluvatar adapter subclass from upstream LMCache's implementation.

    The initial plugin keeps scheduler/worker semantics in the upstream adapter
    and relies on patched factories for Iluvatar-specific extension points.
    Wrapping the class gives diagnostics and a stable hook point for future
    vLLM-only behavior without forking the upstream adapter.
    """

    if getattr(base_cls, "__lmcache_iluvatar_adapter__", False):
        return base_cls

    class LMCacheIluvatarConnectorV1Impl(base_cls):  # type: ignore[misc, valid-type]
        __lmcache_iluvatar_adapter__ = True

        def __init__(self, *args, **kwargs):
            logger.info("Initializing LMCache vLLM V1 adapter with Iluvatar patches.")
            super().__init__(*args, **kwargs)

        def _init_connector_state(self, role, vllm_config, config):
            if getattr(config, "enable_blending", False) and getattr(
                role, "name", None
            ) != "SCHEDULER":
                config.enable_blending = False
                try:
                    super()._init_connector_state(role, vllm_config, config)
                finally:
                    config.enable_blending = True

                self.enable_blending = True
                assert self.lmcache_engine is not None
                assert self.lmcache_engine.gpu_connector is not None, (
                    "GPU connector must be available for blending"
                )
                self.blender = _LazyCacheBlendBuilder(
                    self.lmcache_engine,
                    self.lmcache_engine.gpu_connector,
                    config,
                    vllm_config,
                )
                return

            return super()._init_connector_state(role, vllm_config, config)

        def start_load_kv(self, *args, **kwargs):
            _register_disagg_request_ids(self)
            return super().start_load_kv(*args, **kwargs)

        def wait_for_save(self, *args, **kwargs):
            _register_disagg_request_ids(self)
            _restore_pd_transfer_progress(self)
            _submit_zero_chunk_disagg_saves(self)
            result = super().wait_for_save(*args, **kwargs)
            _record_pd_transfer_progress(self)
            return result

        def get_finished(
            self, finished_req_ids: set[str]
        ) -> tuple[Optional[set[str]], Optional[set[str]]]:
            pd_transferred = getattr(self, "_pd_transferred_tokens", None)
            if pd_transferred is not None:
                for req_id in finished_req_ids:
                    pd_transferred.pop(req_id, None)
            return super().get_finished(finished_req_ids)

    LMCacheIluvatarConnectorV1Impl.__name__ = "LMCacheIluvatarConnectorV1Impl"
    LMCacheIluvatarConnectorV1Impl.__qualname__ = "LMCacheIluvatarConnectorV1Impl"
    LMCacheIluvatarConnectorV1Impl.__module__ = __name__
    return LMCacheIluvatarConnectorV1Impl  # type: ignore[return-value]


class _LazyCacheBlendBuilder:
    """Delay CacheBlend construction until vLLM has registered the model."""

    def __init__(
        self,
        cache_engine: Any,
        gpu_connector: Any,
        config: Any,
        vllm_config: Any,
    ) -> None:
        self._cache_engine = cache_engine
        self._gpu_connector = gpu_connector
        self._config = config
        self._vllm_config = vllm_config
        self._blender = None

    def _get_blender(self) -> Any:
        if self._blender is None:
            from vllm.config import set_current_vllm_config
            from lmcache.integration.vllm.utils import ENGINE_NAME
            from lmcache.v1.compute.blend import LMCBlenderBuilder

            with set_current_vllm_config(self._vllm_config):
                self._blender = LMCBlenderBuilder.get_or_create(
                    ENGINE_NAME,
                    self._cache_engine,
                    self._gpu_connector,
                    self._config,
                )
        return self._blender

    def blend(self, *args, **kwargs):  # noqa: ANN002, ANN003
        return self._get_blender().blend(*args, **kwargs)


def _register_disagg_request_ids(adapter: Any) -> None:
    """Map vLLM request ids to disagg request ids for later cleanup calls."""

    engine = getattr(adapter, "lmcache_engine", None)
    parent = getattr(adapter, "_parent", None)
    get_metadata = getattr(parent, "_get_connector_metadata", None)
    if engine is None or get_metadata is None:
        return

    try:
        metadata = get_metadata()
    except Exception:
        return

    register = getattr(engine, "register_iluvatar_req_id_alias", None)
    if register is None:
        return

    for request in getattr(metadata, "requests", ()):
        req_id = getattr(request, "req_id", None)
        cleanup_id = get_req_id(request)
        if req_id is not None and cleanup_id != req_id:
            register(req_id, cleanup_id)


def _pd_transferred_tokens(adapter: Any) -> dict[str, int]:
    """Worker-local PD watermark: vLLM req_id → tokens transferred to peer."""

    tracker = getattr(adapter, "_pd_transferred_tokens", None)
    if tracker is None:
        tracker = {}
        adapter._pd_transferred_tokens = tracker
    return tracker


def _iter_connector_requests(adapter: Any):
    parent = getattr(adapter, "_parent", None)
    get_metadata = getattr(parent, "_get_connector_metadata", None)
    if get_metadata is None:
        return

    try:
        metadata = get_metadata()
    except Exception:
        return

    yield from getattr(metadata, "requests", ())


def _restore_pd_transfer_progress(adapter: Any) -> None:
    """Restore PD progress lost when scheduler rebuilds DisaggSpec each chunk.

    Mirrors upstream worker-local watermark: never seed from
    ``SaveSpec.skip_leading_tokens`` (local cache ≠ peer PD progress).
    """

    tracker = _pd_transferred_tokens(adapter)
    for request in _iter_connector_requests(adapter):
        disagg_spec = getattr(request, "disagg_spec", None)
        if disagg_spec is None:
            continue
        remembered = tracker.get(request.req_id, 0)
        transferred = int(getattr(disagg_spec, "num_transferred_tokens", 0) or 0)
        if remembered > transferred:
            disagg_spec.num_transferred_tokens = remembered


def _record_pd_transfer_progress(adapter: Any) -> None:
    """Record PD progress after upstream ``wait_for_save`` advances it."""

    tracker = _pd_transferred_tokens(adapter)
    for request in _iter_connector_requests(adapter):
        disagg_spec = getattr(request, "disagg_spec", None)
        if disagg_spec is None:
            continue
        transferred = int(getattr(disagg_spec, "num_transferred_tokens", 0) or 0)
        if transferred > tracker.get(request.req_id, 0):
            tracker[request.req_id] = transferred


def _submit_zero_chunk_disagg_saves(adapter: Any) -> None:
    """Submit an empty final prefill save when no PD chunks remain to transfer.

    The proxy waits for one completion notification from each TP rank after the
    prefill HTTP request returns. When PD has already transferred every token
    (``num_transferred_tokens >= token_len``), upstream ``wait_for_save`` skips
    the store path, so we submit an empty ``batched_put`` for the completion
    signal.

    LocalCPU ``skip_leading_tokens`` alone must not trigger this: a full local
    hit still needs a real P→D transfer.
    """

    if getattr(adapter, "kv_role", None) != "kv_producer":
        return

    storage_manager = _get_storage_manager(adapter)
    if storage_manager is None:
        return

    notified = getattr(adapter, "_iluvatar_zero_chunk_notified", None)
    if notified is None:
        notified = set()
        adapter._iluvatar_zero_chunk_notified = notified

    for request in _iter_connector_requests(adapter):
        disagg_spec = getattr(request, "disagg_spec", None)
        save_spec = getattr(request, "save_spec", None)
        if disagg_spec is None or save_spec is None:
            continue

        req_id = getattr(disagg_spec, "req_id", None)
        if not req_id or req_id in notified:
            continue

        token_ids = getattr(request, "token_ids", ())
        token_len = len(token_ids) if token_ids is not None else 0
        transferred = int(getattr(disagg_spec, "num_transferred_tokens", 0) or 0)
        is_last_prefill = bool(getattr(request, "is_last_prefill", False))
        can_save = bool(getattr(save_spec, "can_save", False))
        if not (can_save and is_last_prefill and token_len > 0):
            continue
        # Match upstream producer skip: min(local_skip, pd_transferred) >= len.
        skip_leading_tokens = int(getattr(save_spec, "skip_leading_tokens", 0) or 0)
        effective_skip = min(skip_leading_tokens, transferred)
        if effective_skip < token_len:
            continue

        disagg_spec.is_last_prefill = True
        storage_manager.batched_put([], [], transfer_spec=disagg_spec)
        notified.add(req_id)


def _get_storage_manager(adapter: Any) -> Any | None:
    """Return the adapter's LMCache storage manager when it is available."""

    engine = getattr(adapter, "lmcache_engine", None)
    return getattr(engine, "storage_manager", None)
