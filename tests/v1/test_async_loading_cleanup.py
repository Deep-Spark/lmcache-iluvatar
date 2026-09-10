# Copyright (c) 2026, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Tests for CSUPPORT-7108 async loading cleanup patches."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock
import asyncio
import importlib
import threading
import time

import msgspec
import pytest

from fake_lmcache import install_fake_lmcache, purge_modules


class _MemoryObj:
    def __init__(self, *, is_pinned: bool = True) -> None:
        self.is_pinned = is_pinned
        self.unpin_count = 0
        self.ref_count_down_count = 0

    def unpin(self) -> None:
        self.unpin_count += 1

    def ref_count_down(self) -> None:
        self.ref_count_down_count += 1


class _LoopProxy:
    def __init__(self) -> None:
        self.call_count = 0

    def call_soon_threadsafe(self, callback) -> None:
        self.call_count += 1
        callback()


class _Socket:
    def __init__(self) -> None:
        self.messages: list[bytes] = []

    def send(self, msg_buf: bytes, copy: bool = False) -> None:
        self.messages.append(bytes(msg_buf))


@pytest.fixture
def patched_lmcache(monkeypatch):
    install_fake_lmcache(monkeypatch)
    plugin = importlib.import_module("lmcache_iluvatar")
    plugin.activate_patches()
    event_manager_mod = importlib.import_module("lmcache.v1.event_manager")
    cache_engine_mod = importlib.import_module("lmcache.v1.cache_engine")
    async_client_mod = importlib.import_module(
        "lmcache.v1.lookup_client.lmcache_async_lookup_client"
    )
    return SimpleNamespace(
        EventManager=event_manager_mod.EventManager,
        EventStatus=event_manager_mod.EventStatus,
        EventType=event_manager_mod.EventType,
        LMCacheEngine=cache_engine_mod.LMCacheEngine,
        LMCacheAsyncLookupClient=async_client_mod.LMCacheAsyncLookupClient,
        LookupCleanupMsg=async_client_mod.LookupCleanupMsg,
    )


def _make_engine_for_cleanup(lmcache) -> object:
    engine = lmcache.LMCacheEngine.__new__(lmcache.LMCacheEngine)
    engine.event_manager = lmcache.EventManager()
    engine.storage_manager = MagicMock()
    engine.storage_manager.loop = _LoopProxy()
    return engine


def test_event_manager_pop_event_any_status(patched_lmcache) -> None:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    event_manager = patched_lmcache.EventManager()
    future = loop.create_future()
    event_manager.add_event(patched_lmcache.EventType.LOADING, "lookup-1", future)
    popped_future, status = event_manager.pop_event_any_status(
        patched_lmcache.EventType.LOADING, "lookup-1"
    )
    loop.close()

    assert popped_future is future
    assert status == patched_lmcache.EventStatus.ONGOING
    assert (
        event_manager.get_event_status(
            patched_lmcache.EventType.LOADING, "lookup-1"
        )
        == patched_lmcache.EventStatus.NOT_FOUND
    )


def test_cleanup_memory_objs_cancels_ongoing_prefetch(patched_lmcache) -> None:
    engine = _make_engine_for_cleanup(patched_lmcache)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    future = loop.create_future()
    engine.event_manager.add_event(
        patched_lmcache.EventType.LOADING, "lookup-1", future
    )

    engine.cleanup_memory_objs("lookup-1")
    loop.close()

    assert future.cancelled()
    assert engine.storage_manager.loop.call_count == 1
    assert (
        engine.event_manager.get_event_status(
            patched_lmcache.EventType.LOADING, "lookup-1"
        )
        == patched_lmcache.EventStatus.NOT_FOUND
    )


@pytest.mark.parametrize(
    "prefetch_results",
    [
        [[_MemoryObj(), _MemoryObj()]],
        [[("key0", _MemoryObj()), ("key1", _MemoryObj())]],
    ],
)
def test_cleanup_memory_objs_releases_done_prefetch_results(
    patched_lmcache, prefetch_results
) -> None:
    engine = _make_engine_for_cleanup(patched_lmcache)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    future = loop.create_future()
    future.set_result(prefetch_results)
    engine.event_manager.add_event(
        patched_lmcache.EventType.LOADING, "lookup-1", future
    )
    engine.event_manager.update_event_status(
        patched_lmcache.EventType.LOADING, "lookup-1", patched_lmcache.EventStatus.DONE
    )

    engine.cleanup_memory_objs("lookup-1")
    loop.close()

    memory_objs = [
        item[1] if isinstance(item, tuple) else item
        for tier_result in prefetch_results
        for item in tier_result
    ]
    assert all(memory_obj.unpin_count == 1 for memory_obj in memory_objs)
    assert all(memory_obj.ref_count_down_count == 1 for memory_obj in memory_objs)


def test_cleanup_memory_objs_does_not_unpin_unpinned_results(
    patched_lmcache,
) -> None:
    engine = _make_engine_for_cleanup(patched_lmcache)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    future = loop.create_future()
    memory_obj = _MemoryObj(is_pinned=False)
    future.set_result([[("key0", memory_obj)]])
    engine.event_manager.add_event(
        patched_lmcache.EventType.LOADING, "lookup-1", future
    )
    engine.event_manager.update_event_status(
        patched_lmcache.EventType.LOADING, "lookup-1", patched_lmcache.EventStatus.DONE
    )

    engine.cleanup_memory_objs("lookup-1")
    loop.close()

    assert memory_obj.unpin_count == 0
    assert memory_obj.ref_count_down_count == 1


def _make_async_lookup_client(lmcache, world_size: int = 2) -> object:
    client = lmcache.LMCacheAsyncLookupClient.__new__(
        lmcache.LMCacheAsyncLookupClient
    )
    client.world_size = world_size
    client.push_sockets = [_Socket() for _ in range(world_size)]
    client.reqs_status = {}
    client.first_lookup_time = {}
    client.res_for_each_worker = {}
    client.aborted_lookups = set()
    client.cleanup_requested_lookups = set()
    client.lock = threading.Lock()
    client.lookup_backoff_time = 0
    client.config = SimpleNamespace(lookup_timeout_ms=1)
    return client


def _decode_cleanup_messages(client, lmcache) -> list[str]:
    return [
        msgspec.msgpack.decode(message, type=lmcache.LookupCleanupMsg).lookup_id
        for socket in client.push_sockets
        for message in socket.messages
    ]


def test_cancel_lookup_sends_cleanup_once(patched_lmcache) -> None:
    client = _make_async_lookup_client(patched_lmcache, world_size=2)
    client._send_cleanup_message = MagicMock()

    client.cancel_lookup("lookup-1")
    client.cancel_lookup("lookup-1")

    assert client.aborted_lookups == {"lookup-1"}
    assert client.cleanup_requested_lookups == {"lookup-1"}
    assert client._send_cleanup_message.call_count == 1


def test_lookup_cache_timeout_sends_cleanup_and_returns_zero(patched_lmcache) -> None:
    client = _make_async_lookup_client(patched_lmcache, world_size=1)
    client._send_cleanup_message = MagicMock()
    client.reqs_status["lookup-1"] = None
    client.first_lookup_time["lookup-1"] = time.time() - 1

    result = client.lookup_cache("lookup-1")

    assert result == 0
    assert client.reqs_status["lookup-1"] == 0
    assert "lookup-1" not in client.first_lookup_time
    assert client._send_cleanup_message.call_count == 1


def test_clear_lookup_status_clears_cleanup_state(patched_lmcache) -> None:
    client = _make_async_lookup_client(patched_lmcache, world_size=1)
    client.reqs_status["lookup-1"] = 0
    client.first_lookup_time["lookup-1"] = time.time()
    client.res_for_each_worker["lookup-1"] = [1]
    client.aborted_lookups.add("lookup-1")
    client.cleanup_requested_lookups.add("lookup-1")

    client.clear_lookup_status("lookup-1")

    assert "lookup-1" not in client.reqs_status
    assert "lookup-1" not in client.first_lookup_time
    assert "lookup-1" not in client.res_for_each_worker
    assert "lookup-1" not in client.aborted_lookups
    assert "lookup-1" not in client.cleanup_requested_lookups

