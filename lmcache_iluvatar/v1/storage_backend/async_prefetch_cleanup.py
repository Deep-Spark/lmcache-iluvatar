# Copyright (c) 2026, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Helpers for async prefetch memory cleanup (CSUPPORT-7108)."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any, Union

if TYPE_CHECKING:
    from lmcache.utils import CacheEngineKey
    from lmcache.v1.memory_management import MemoryObj

logger = logging.getLogger(__name__)


def release_prefetch_results(
    prefetch_results: list[list[Union["MemoryObj", tuple["CacheEngineKey", "MemoryObj"]]]],
) -> int:
    """Release memory objects returned by async prefetch results."""
    released_count = 0
    for tier_result in prefetch_results:
        for item in tier_result:
            memory_obj = item[1] if isinstance(item, tuple) else item
            memory_obj.unpin()
            memory_obj.ref_count_down()
            released_count += 1
    return released_count


def release_done_prefetch_tasks(loading_tasks: list[asyncio.Task]) -> int:
    """Release results from completed backend prefetch tasks."""
    released_count = 0
    for loading_task in loading_tasks:
        if not loading_task.done() or loading_task.cancelled():
            continue
        try:
            tier_result = loading_task.result()
        except Exception:
            continue
        released_count += release_prefetch_results([tier_result])
    return released_count


def prefetch_all_done_callback(
    storage_manager: Any,
    task: asyncio.Future,
    lookup_id: str,
    cum_chunk_lengths_total: list[int],
    tier_expected_chunks: list[int],
    keys_per_chunk: int = 1,
    loading_tasks: list[asyncio.Task] | None = None,
) -> None:
    """Callback when all prefetch tasks for a lookup are done."""
    from lmcache.v1.event_manager import EventStatus, EventType

    assert storage_manager.async_lookup_server is not None
    if keys_per_chunk < 1:
        raise ValueError(f"keys_per_chunk must be >= 1, got {keys_per_chunk}")
    if task.cancelled():
        released_count = (
            release_done_prefetch_tasks(loading_tasks)
            if loading_tasks is not None
            else 0
        )
        logger.debug(
            "Async prefetch task cancelled for lookup_id=%s; "
            "released completed results=%d",
            lookup_id,
            released_count,
        )
        return
    try:
        res = task.result()
    except asyncio.CancelledError:
        released_count = (
            release_done_prefetch_tasks(loading_tasks)
            if loading_tasks is not None
            else 0
        )
        logger.debug(
            "Async prefetch task cancelled while getting result: "
            "lookup_id=%s, released completed results=%d",
            lookup_id,
            released_count,
        )
        return
    except Exception as exc:
        logger.warning(
            "Async prefetch task failed for lookup_id=%s: %s", lookup_id, exc
        )
        try:
            storage_manager.event_manager.update_event_status(
                EventType.LOADING, lookup_id, status=EventStatus.DONE
            )
        except KeyError:
            logger.debug(
                "Async prefetch event already purged for failed lookup_id=%s",
                lookup_id,
            )
            return
        storage_manager.async_lookup_server.send_response_to_scheduler(lookup_id, 0)
        return

    try:
        storage_manager.event_manager.update_event_status(
            EventType.LOADING, lookup_id, status=EventStatus.DONE
        )
    except KeyError:
        released_count = release_prefetch_results(res)
        logger.debug(
            "Async prefetch event already purged for lookup_id=%s; "
            "released late results=%d",
            lookup_id,
            released_count,
        )
        return

    total_retrieved_chunks = 0
    for tier_idx, tier_result in enumerate(res):
        actual_chunks = len(tier_result) // keys_per_chunk
        expected_chunks = tier_expected_chunks[tier_idx]
        total_retrieved_chunks += actual_chunks

        tail_start = actual_chunks * keys_per_chunk
        release_prefetch_results([tier_result[tail_start:]])

        if actual_chunks < expected_chunks:
            for subsequent_tier in res[tier_idx + 1 :]:
                release_prefetch_results([subsequent_tier])
            break

    retrieved_length = cum_chunk_lengths_total[total_retrieved_chunks]
    logger.info(
        "Responding to scheduler for lookup id %s with retrieved length %s",
        lookup_id,
        retrieved_length,
    )
    storage_manager.async_lookup_server.send_response_to_scheduler(
        lookup_id, retrieved_length
    )


__all__ = [
    "prefetch_all_done_callback",
    "release_done_prefetch_tasks",
    "release_prefetch_results",
]
