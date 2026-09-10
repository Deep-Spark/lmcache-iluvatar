# Copyright (c) 2026, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Iluvatar compatibility wrapper for LMCacheEngine."""

from __future__ import annotations

import logging
from typing import Any, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=type)


def build_iluvatar_cache_engine(base_cls: T) -> T:
    """Create an LMCacheEngine subclass with PD reuse and async loading cleanup."""

    if getattr(base_cls, "__lmcache_iluvatar_cache_engine__", False):
        return base_cls

    class LMCacheIluvatarEngine(base_cls):  # type: ignore[misc, valid-type]
        __lmcache_iluvatar_cache_engine__ = True

        def register_iluvatar_req_id_alias(self, req_id: str, cleanup_id: str) -> None:
            aliases = getattr(self, "_iluvatar_req_id_aliases", None)
            if aliases is None:
                aliases = {}
                setattr(self, "_iluvatar_req_id_aliases", aliases)
            aliases[req_id] = cleanup_id

        def retrieve(self, tokens, mask=None, **kwargs):  # noqa: ANN001
            req_id = kwargs.get("req_id")
            cleanup_id = self._iluvatar_cleanup_req_id(req_id)
            if cleanup_id is not None:
                kwargs["req_id"] = cleanup_id

            ret_mask = super().retrieve(tokens, mask, **kwargs)

            if (
                cleanup_id is not None
                and getattr(self, "remove_after_retrieve", False)
                and not self._is_passive()
                and getattr(self, "storage_manager", None) is not None
            ):
                self.storage_manager.release_pd_reuse(cleanup_id)
            return ret_mask

        def lookup_unpin(self, lookup_id: str) -> None:
            cleanup_id = self._iluvatar_cleanup_req_id(lookup_id) or lookup_id
            super().lookup_unpin(cleanup_id)

            storage_manager = getattr(self, "storage_manager", None)
            if storage_manager is not None:
                storage_manager.release_pd_reuse(cleanup_id)

        def cleanup_memory_objs(self, lookup_id: str) -> None:
            from lmcache.v1.event_manager import EventStatus, EventType

            try:
                event_status = self.event_manager.get_event_status(
                    EventType.LOADING, lookup_id
                )
                if event_status == EventStatus.NOT_FOUND:
                    logger.debug(
                        "No event found for lookup_id=%s to clean up.", lookup_id
                    )
                    return

                future, popped_status = self.event_manager.pop_event_any_status(
                    EventType.LOADING, lookup_id
                )
                if popped_status == EventStatus.ONGOING:
                    storage_manager = getattr(self, "storage_manager", None)
                    if storage_manager is not None:
                        storage_manager.loop.call_soon_threadsafe(future.cancel)
                    else:
                        future.cancel()
                    logger.debug(
                        "Cancelled ongoing async prefetch event for lookup_id=%s",
                        lookup_id,
                    )
                    return

                if future.cancelled():
                    logger.debug(
                        "Async prefetch event already cancelled for lookup_id=%s",
                        lookup_id,
                    )
                    return

                memory_objs = future.result()
                memory_objs_flat = [
                    item[1] if isinstance(item, tuple) else item
                    for tier_result in memory_objs
                    for item in tier_result
                ]

                for memory_obj in memory_objs_flat:
                    try:
                        logger.debug(
                            "Releasing memory object for lookup_id=%s", lookup_id
                        )
                        if getattr(memory_obj, "is_pinned", True):
                            memory_obj.unpin()
                        memory_obj.ref_count_down()
                    except Exception as exc:
                        logger.error("Error releasing memory object: %s", exc)
            except KeyError:
                logger.debug(
                    "No event found for lookup_id=%s to clean up.", lookup_id
                )
            except Exception as exc:
                logger.error(
                    "Error during cleanup_memory_objs for lookup_id=%s: %s",
                    lookup_id,
                    exc,
                )

        def _broadcast_or_receive_memory_objs(self, reordered_chunks, ret_mask):  # noqa: ANN001
            from lmcache_iluvatar.v1.pp_broadcast import broadcast_or_receive_memory_objs

            broadcast_or_receive_memory_objs(self, reordered_chunks, ret_mask)

        def _iluvatar_cleanup_req_id(self, req_id: Any) -> str | None:
            if req_id is None:
                return None
            aliases = getattr(self, "_iluvatar_req_id_aliases", {})
            return aliases.pop(req_id, req_id)

    LMCacheIluvatarEngine.__name__ = "LMCacheIluvatarEngine"
    LMCacheIluvatarEngine.__qualname__ = "LMCacheIluvatarEngine"
    LMCacheIluvatarEngine.__module__ = __name__
    return LMCacheIluvatarEngine  # type: ignore[return-value]


__all__ = ["build_iluvatar_cache_engine"]
