# Copyright (c) 2026, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""LMCacheAsyncLookupClient compatibility hooks for async loading cleanup."""

from __future__ import annotations

import logging
import time
from typing import TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=type)


def build_iluvatar_async_lookup_client(base_cls: T) -> T:
    """Ensure scheduler-side async lookup cleanup is sent promptly."""

    if getattr(base_cls, "__lmcache_iluvatar_async_lookup_client__", False):
        return base_cls

    class IluvatarAsyncLookupClient(base_cls):  # type: ignore[misc, valid-type]
        __lmcache_iluvatar_async_lookup_client__ = True

        def __init__(self, *args, **kwargs):  # noqa: ANN002, ANN003
            super().__init__(*args, **kwargs)
            self.cleanup_requested_lookups: set[str] = set()

        def lookup_cache(self, lookup_id: str):  # noqa: ANN001
            self._cleanup_finished_aborted_lookups()

            with self.lock:
                if (req_status := self.reqs_status.get(lookup_id, -1)) == -1:
                    self.reqs_status[lookup_id] = None
                    self.first_lookup_time[lookup_id] = time.time()
                elif req_status is None:
                    time.sleep(self.lookup_backoff_time)
                    if (
                        time.time() - self.first_lookup_time[lookup_id]
                    ) * 1000 > self.config.lookup_timeout_ms:
                        logger.warning(
                            (
                                "Request %s is still waiting for async lookup "
                                "after %d seconds, returning 0 lmcache cached tokens "
                                "so vllm can recompute"
                            ),
                            lookup_id,
                            self.config.lookup_timeout_ms // 1000,
                        )
                        self.first_lookup_time.pop(lookup_id, None)
                        self.reqs_status[lookup_id] = 0
                        self._request_cleanup(lookup_id, "timeout")
                        return 0

                return req_status

        def clear_lookup_status(self, lookup_id: str) -> None:
            with self.lock:
                self.reqs_status.pop(lookup_id, None)
                self.first_lookup_time.pop(lookup_id, None)
                self.res_for_each_worker.pop(lookup_id, None)
                self.aborted_lookups.discard(lookup_id)
                self.cleanup_requested_lookups.discard(lookup_id)

        def cancel_lookup(self, lookup_id: str) -> None:
            self.aborted_lookups.add(lookup_id)
            logger.debug("Marked async lookup as aborted: lookup_id=%s", lookup_id)
            self._request_cleanup(lookup_id, "abort")

        def _cleanup_finished_aborted_lookups(self) -> None:
            finished_lookups = [
                lookup_id
                for lookup_id in self.aborted_lookups
                if self.reqs_status.get(lookup_id) is not None
            ]
            if finished_lookups:
                self.aborted_lookups.difference_update(finished_lookups)

            for lookup_id in finished_lookups:
                self._request_cleanup(lookup_id, "aborted-finished")
                self.clear_lookup_status(lookup_id)

        def _request_cleanup(self, lookup_id: str, reason: str) -> None:
            if lookup_id in self.cleanup_requested_lookups:
                logger.debug(
                    "Cleanup already requested for lookup_id=%s, reason=%s",
                    lookup_id,
                    reason,
                )
                return
            self.cleanup_requested_lookups.add(lookup_id)
            logger.debug(
                "Requesting async lookup cleanup: lookup_id=%s, reason=%s",
                lookup_id,
                reason,
            )
            self._send_cleanup_message(lookup_id)

    IluvatarAsyncLookupClient.__name__ = "IluvatarAsyncLookupClient"
    IluvatarAsyncLookupClient.__qualname__ = "IluvatarAsyncLookupClient"
    IluvatarAsyncLookupClient.__module__ = __name__
    return IluvatarAsyncLookupClient  # type: ignore[return-value]


__all__ = ["build_iluvatar_async_lookup_client"]
