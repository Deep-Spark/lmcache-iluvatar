# Copyright (c) 2026, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""EventManager compatibility hooks for async loading cleanup."""

from __future__ import annotations

import asyncio
from typing import TypeVar

T = TypeVar("T", bound=type)


def build_iluvatar_event_manager(base_cls: T) -> T:
    """Add ``pop_event_any_status`` while upstream LMCache v0.5.3 lacks it."""

    if getattr(base_cls, "__lmcache_iluvatar_event_manager__", False):
        return base_cls

    if hasattr(base_cls, "pop_event_any_status"):
        setattr(base_cls, "__lmcache_iluvatar_event_manager__", True)
        return base_cls

    def pop_event_any_status(
        self,
        event_type,
        event_id: str,
    ) -> tuple[asyncio.Future, object]:
        """Pop and return an event with the given type and id from any status."""
        from lmcache.v1.event_manager import EventStatus

        with self.lock:
            status_dict = self.events.get(event_type, None)
            assert status_dict is not None, (
                f"Invalid event type {event_type} in EventManager."
            )
            for status in EventStatus:
                if event_id in status_dict[status]:
                    return status_dict[status].pop(event_id), status
            raise KeyError(f"Event {event_id} of type {event_type} not found.")

    base_cls.pop_event_any_status = pop_event_any_status  # type: ignore[attr-defined]
    setattr(base_cls, "__lmcache_iluvatar_event_manager__", True)
    return base_cls


__all__ = ["build_iluvatar_event_manager"]
