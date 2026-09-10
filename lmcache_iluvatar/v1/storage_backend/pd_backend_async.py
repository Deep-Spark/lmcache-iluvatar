# Copyright (c) 2026, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""PDBackendAsync compatibility hooks for Iluvatar disaggregated transfer."""

from __future__ import annotations

import asyncio
import fcntl
import os
import time
from importlib import import_module
from typing import TypeVar

_PEER_INIT_LOCK_PATH = os.environ.get(
    "LMCACHE_ILUVATAR_PD_PEER_INIT_LOCK",
    "/tmp/lmcache_iluvatar_pd_peer_init.lock",
)
_PEER_INIT_TIMEOUT_SEC = float(
    os.environ.get("LMCACHE_ILUVATAR_PD_PEER_INIT_TIMEOUT", "120")
)
_PEER_INIT_RETRIES = int(os.environ.get("LMCACHE_ILUVATAR_PD_PEER_INIT_RETRIES", "3"))

from lmcache_iluvatar.v1.storage_backend.pd_backend import (
    _PDReuseMixin,
    _ensure_reuse_state,
    _patch_alloc_request,
)

T = TypeVar("T", bound=type)


def build_iluvatar_pd_backend_async(base_cls: T) -> T:
    """Create an async PDBackend subclass with Iluvatar request-id fixes."""

    if getattr(base_cls, "__lmcache_iluvatar_pd_backend_async__", False):
        return base_cls

    module = import_module(base_cls.__module__)
    _patch_alloc_request(module)

    class IluvatarPDBackendAsync(_PDReuseMixin, base_cls):  # type: ignore[misc, valid-type]
        __lmcache_iluvatar_pd_backend_async__ = True

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            _ensure_reuse_state(self)

        def batched_submit_put_task(
            self,
            keys,
            memory_objs,
            transfer_spec=None,
            *args,
            **kwargs,
        ):  # noqa: ANN001
            """Preserve req id context and translate empty final saves to notify."""

            previous = getattr(self, "_iluvatar_current_req_id", None)
            req_id = getattr(transfer_spec, "req_id", "") if transfer_spec is not None else ""
            self._iluvatar_current_req_id = req_id
            try:
                if (not keys or not memory_objs) and getattr(
                    transfer_spec, "is_last_prefill", False
                ):
                    self._notify_proxy_transfer_done(req_id, transfer_spec)
                    return None
                return super().batched_submit_put_task(
                    keys, memory_objs, transfer_spec, *args, **kwargs
                )
            finally:
                self._iluvatar_current_req_id = previous

        def _ensure_peer_connection(
            self,
            receiver_id: str,
            receiver_host: str,
            receiver_init_port: int,
            receiver_alloc_port: int,
        ) -> None:
            """Serialize NIXL/UCX peer init across TP ranks with timeout retries.

            Iluvatar can hang inside ``async_lazy_init_peer_connection`` when TP
            workers connect concurrently (decoder TP1 never reaches mem register).
            """

            if receiver_id in self.initialized_peers:
                return

            with open(_PEER_INIT_LOCK_PATH, "a", encoding="utf-8") as lock_file:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
                try:
                    with self._peer_connection_lock:
                        if receiver_id in self.initialized_peers:
                            return

                        receiver_init_url = f"{receiver_host}:{receiver_init_port}"
                        receiver_mem_alloc_url = f"{receiver_host}:{receiver_alloc_port}"
                        last_err: BaseException | None = None

                        for attempt in range(_PEER_INIT_RETRIES):
                            try:
                                init_future = asyncio.run_coroutine_threadsafe(
                                    self.transfer_channel.async_lazy_init_peer_connection(
                                        local_id=self.local_id,
                                        peer_id=receiver_id,
                                        peer_init_url=receiver_init_url,
                                    ),
                                    self._sender_loop,
                                )
                                init_future.result(timeout=_PEER_INIT_TIMEOUT_SEC)

                                socket_future = asyncio.run_coroutine_threadsafe(
                                    self._async_create_alloc_socket(
                                        receiver_id, receiver_mem_alloc_url
                                    ),
                                    self._sender_loop,
                                )
                                socket_future.result(timeout=10)

                                self.initialized_peers.add(receiver_id)
                                return
                            except BaseException as exc:
                                last_err = exc
                                time.sleep(0.5 * (attempt + 1))

                        assert last_err is not None
                        raise last_err
                finally:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

        def _notify_proxy_transfer_done(self, req_id: str, transfer_spec) -> None:
            """Schedule async proxy completion on the backend's sender loop."""

            send_proxy_notif = getattr(self, "_send_proxy_notif", None)
            sender_loop = getattr(self, "_sender_loop", None)
            if not req_id or send_proxy_notif is None or sender_loop is None:
                raise RuntimeError(
                    "PDBackendAsync cannot send proxy transfer notification"
                )

            future = asyncio.run_coroutine_threadsafe(
                send_proxy_notif(transfer_spec),
                sender_loop,
            )
            future.result(timeout=1)

    IluvatarPDBackendAsync.__name__ = "IluvatarPDBackendAsync"
    IluvatarPDBackendAsync.__qualname__ = "IluvatarPDBackendAsync"
    IluvatarPDBackendAsync.__module__ = __name__
    return IluvatarPDBackendAsync  # type: ignore[return-value]


__all__ = ["build_iluvatar_pd_backend_async"]
