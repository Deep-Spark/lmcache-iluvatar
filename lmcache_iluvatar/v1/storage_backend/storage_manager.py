# Copyright (c) 2026, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""StorageManager compatibility hooks for PD reuse and async loading cleanup."""

from __future__ import annotations

import asyncio
from typing import TypeVar

from lmcache_iluvatar.v1.storage_backend.async_prefetch_cleanup import (
    prefetch_all_done_callback as iluvatar_prefetch_all_done_callback,
)

T = TypeVar("T", bound=type)


def build_iluvatar_storage_manager(base_cls: T) -> T:
    """Create a StorageManager subclass with PD reuse and async prefetch cleanup."""

    if getattr(base_cls, "__lmcache_iluvatar_storage_manager__", False):
        return base_cls

    class IluvatarStorageManager(base_cls):  # type: ignore[misc, valid-type]
        __lmcache_iluvatar_storage_manager__ = True

        def release_pd_reuse(self, req_id: str) -> None:
            """Forward request-scoped reused-key cleanup to the PD backend."""

            backend = getattr(self, "storage_backends", {}).get("PDBackend")
            release = getattr(backend, "release_reused_keys", None)
            if release is not None:
                release(req_id)

        def prefetch_all_done_callback(
            self,
            task: asyncio.Future,
            lookup_id: str,
            cum_chunk_lengths_total: list[int],
            tier_expected_chunks: list[int],
            keys_per_chunk: int = 1,
            loading_tasks: list[asyncio.Task] | None = None,
        ) -> None:
            iluvatar_prefetch_all_done_callback(
                self,
                task,
                lookup_id,
                cum_chunk_lengths_total,
                tier_expected_chunks,
                keys_per_chunk=keys_per_chunk,
                loading_tasks=loading_tasks,
            )

    IluvatarStorageManager.__name__ = "IluvatarStorageManager"
    IluvatarStorageManager.__qualname__ = "IluvatarStorageManager"
    IluvatarStorageManager.__module__ = __name__
    return IluvatarStorageManager  # type: ignore[return-value]


__all__ = ["build_iluvatar_storage_manager"]
