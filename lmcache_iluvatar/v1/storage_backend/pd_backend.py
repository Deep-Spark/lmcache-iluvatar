# Copyright (c) 2026, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""PDBackend compatibility hooks for request-scoped reused-key cleanup."""

from __future__ import annotations

import inspect
import threading
from functools import reduce
from importlib import import_module
from typing import Any, TypeVar, Union

import msgspec

T = TypeVar("T", bound=type)


def build_iluvatar_pd_backend(base_cls: T) -> T:
    """Create a sync PDBackend subclass with Iluvatar request-id fixes."""

    if getattr(base_cls, "__lmcache_iluvatar_pd_backend__", False):
        return base_cls

    module = import_module(base_cls.__module__)
    _patch_alloc_request(module)

    class IluvatarPDBackend(_PDReuseMixin, base_cls):  # type: ignore[misc, valid-type]
        __lmcache_iluvatar_pd_backend__ = True

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            _ensure_reuse_state(self)

        def batched_submit_put_task(self, keys, memory_objs, transfer_spec=None, *args, **kwargs):  # noqa: ANN001, E501
            """Preserve the current disagg req id and handle empty final saves."""

            previous = getattr(self, "_iluvatar_current_req_id", None)
            req_id = getattr(transfer_spec, "req_id", "") if transfer_spec is not None else ""
            self._iluvatar_current_req_id = req_id
            try:
                if (not keys or not memory_objs) and getattr(
                    transfer_spec, "is_last_prefill", False
                ):
                    self._notify_proxy_transfer_done(req_id)
                    return None
                return super().batched_submit_put_task(
                    keys, memory_objs, transfer_spec, *args, **kwargs
                )
            finally:
                self._iluvatar_current_req_id = previous

        def _notify_proxy_transfer_done(self, req_id: str) -> None:
            """Send the proxy completion signal for a request with no KV payload."""

            super_method = getattr(super(), "_notify_proxy_transfer_done", None)
            if super_method is not None:
                super_method(req_id)
                return

            proxy_notif_cls = getattr(module, "ProxyNotif", None)
            proxy_side_channel = getattr(self, "proxy_side_channel", None)
            if not req_id or proxy_notif_cls is None or proxy_side_channel is None:
                raise RuntimeError("PDBackend cannot send proxy transfer notification")

            notif_msg = proxy_notif_cls(req_id=req_id)
            proxy_side_channel.send(msgspec.msgpack.encode(notif_msg))

        def _get_remote_alloc_request(self, *args, **kwargs):
            """Build a remote allocation request and attach the disagg req id."""

            req_id, keys, mem_objs, passthrough_kwargs = _parse_alloc_request_args(
                self, args, kwargs
            )
            if passthrough_kwargs:
                unexpected = ", ".join(sorted(passthrough_kwargs))
                raise TypeError(
                    f"Unexpected PDBackend alloc request keyword(s): {unexpected}"
                )
            super_method = super()._get_remote_alloc_request
            parameters = inspect.signature(super_method).parameters
            if "req_id" in parameters:
                alloc_request = super_method(req_id, keys, mem_objs)
            else:
                alloc_request = super_method(keys, mem_objs)
            return _set_alloc_request_req_id(alloc_request, req_id)

    IluvatarPDBackend.__name__ = "IluvatarPDBackend"
    IluvatarPDBackend.__qualname__ = "IluvatarPDBackend"
    IluvatarPDBackend.__module__ = __name__
    return IluvatarPDBackend  # type: ignore[return-value]


class _PDReuseMixin:
    """Track receiver-side reused keys by disagg request id."""

    def _allocate_and_put(self, alloc_request):
        """Expose the allocation request id while checking already-present keys."""

        previous = getattr(self, "_iluvatar_allocating_req_id", None)
        self._iluvatar_allocating_req_id = getattr(alloc_request, "req_id", "")
        try:
            return super()._allocate_and_put(alloc_request)
        finally:
            self._iluvatar_allocating_req_id = previous

    def contains(self, key, pin=False):  # noqa: ANN001
        """Acquire reused keys during receiver allocation without pinning them."""

        req_id = getattr(self, "_iluvatar_allocating_req_id", "")
        if req_id and not pin:
            return self.acquire_reused_key(req_id, key)
        return super().contains(key, pin)

    def acquire_reused_key(self, req_id: str, key) -> bool:  # noqa: ANN001
        """Record an existing key as reused for this disagg request."""

        _ensure_reuse_state(self)
        with self.data_lock:
            mem_obj = self.data.get(key)
            if mem_obj is None:
                return False
            mem_obj.ref_count_up()

        self._track_reused_key(req_id, key)
        return True

    def release_reused_keys(self, req_id: str) -> None:
        """Release all receiver-side reused-key refs for a disagg request."""

        _ensure_reuse_state(self)
        for key in self._pop_reused_keys(req_id):
            with self.data_lock:
                mem_obj = self.data.get(key)
                if mem_obj is None:
                    continue
                remove_from_data = mem_obj.get_ref_count() == 1
                if remove_from_data:
                    del self.data[key]

            mem_obj.ref_count_down()

    def _track_reused_key(self, req_id: str, key) -> None:  # noqa: ANN001
        _ensure_reuse_state(self)
        with self.reused_keys_lock:
            self.reused_keys_by_req.setdefault(req_id, []).append(key)

    def _pop_reused_keys(self, req_id: str) -> list[Any]:
        _ensure_reuse_state(self)
        with self.reused_keys_lock:
            return self.reused_keys_by_req.pop(req_id, [])


def _ensure_reuse_state(backend: Any) -> None:
    """Create reused-key bookkeeping fields on wrapped backends."""

    if not hasattr(backend, "reused_keys_by_req"):
        backend.reused_keys_by_req = {}
    if not hasattr(backend, "reused_keys_lock"):
        backend.reused_keys_lock = threading.Lock()


def _parse_alloc_request_args(
    backend: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> tuple[str, Any, Any, dict[str, Any]]:
    """Normalize sync/async allocation helper call shapes into named fields."""

    req_id = kwargs.pop("req_id", None)
    keys = kwargs.pop("keys", None)
    mem_objs = kwargs.pop("mem_objs", kwargs.pop("memory_objs", None))

    if len(args) == 3:
        req_id, keys, mem_objs = args
    elif len(args) == 2:
        keys, mem_objs = args
    elif args:
        raise TypeError(f"Unexpected _get_remote_alloc_request arguments: {args!r}")

    if req_id is None:
        req_id = getattr(backend, "_iluvatar_current_req_id", "")
    if keys is None or mem_objs is None:
        raise TypeError("_get_remote_alloc_request requires keys and mem_objs")
    return str(req_id or ""), keys, mem_objs, kwargs


def _set_alloc_request_req_id(alloc_request: Any, req_id: str) -> Any:
    """Attach ``req_id`` to allocation messages when upstream lacks the field."""

    if getattr(alloc_request, "req_id", None) == req_id:
        return alloc_request
    try:
        alloc_request.req_id = req_id
        return alloc_request
    except (AttributeError, TypeError):
        return alloc_request


def _patch_alloc_request(module: Any) -> None:
    """Patch upstream PD msgpack union so allocation requests can carry req ids."""

    alloc_request_cls = getattr(module, "AllocRequest", None)
    if alloc_request_cls is None or "req_id" in getattr(alloc_request_cls, "__annotations__", {}):
        return

    class IluvatarAllocRequest(alloc_request_cls):  # type: ignore[misc, valid-type]
        req_id: str = ""

    IluvatarAllocRequest.__name__ = "AllocRequest"
    IluvatarAllocRequest.__qualname__ = "AllocRequest"
    IluvatarAllocRequest.__module__ = module.__name__
    module.AllocRequest = IluvatarAllocRequest

    msg_types = [
        cls
        for cls in (
            IluvatarAllocRequest,
            getattr(module, "AllocResponse", None),
            getattr(module, "ProxyNotif", None),
            getattr(module, "CacheQueryRequest", None),
            getattr(module, "CacheQueryResponse", None),
        )
        if cls is not None
    ]
    if msg_types:
        module.PDMsg = reduce(lambda left, right: Union[left, right], msg_types)


__all__ = ["build_iluvatar_pd_backend"]
