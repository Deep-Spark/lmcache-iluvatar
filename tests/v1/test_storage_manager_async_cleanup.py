# Copyright (c) 2026, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""StorageManager async prefetch cleanup tests (CSUPPORT-7108)."""

from __future__ import annotations

import asyncio
import importlib

import pytest

from fake_lmcache import install_fake_lmcache


class MockMemoryObj:
    def __init__(self, obj_id: int) -> None:
        self.obj_id = obj_id
        self.ref_count_down_called = False
        self.unpin_called = False

    def ref_count_down(self) -> None:
        self.ref_count_down_called = True

    def unpin(self) -> None:
        self.unpin_called = True


class MockAsyncLookupServer:
    def __init__(self) -> None:
        self.responses: list[tuple[str, int]] = []

    def send_response_to_scheduler(self, lookup_id: str, length: int) -> None:
        self.responses.append((lookup_id, length))


@pytest.fixture
def patched_storage_manager(monkeypatch):
    modules = install_fake_lmcache(monkeypatch)
    importlib.import_module("lmcache_iluvatar").activate_patches()

    event_manager_mod = importlib.import_module("lmcache.v1.event_manager")
    storage_manager_cls = modules[
        "lmcache.v1.storage_backend.storage_manager"
    ].StorageManager

    manager = storage_manager_cls.__new__(storage_manager_cls)
    manager.event_manager = event_manager_mod.EventManager()
    manager.async_lookup_server = MockAsyncLookupServer()
    return manager, event_manager_mod


def test_late_prefetch_results_are_released_when_event_was_purged(
    patched_storage_manager,
) -> None:
    manager, event_manager_mod = patched_storage_manager
    cum_chunk_lengths_total = [0, 256, 512]
    tier_expected_chunks = [2]
    obj0 = MockMemoryObj(0)
    obj1 = MockMemoryObj(1)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    future = loop.create_future()
    future.set_result([[("key0", obj0), ("key1", obj1)]])

    manager.event_manager.add_event(
        event_manager_mod.EventType.LOADING, "late_lookup", future
    )
    manager.event_manager.pop_event_any_status(
        event_manager_mod.EventType.LOADING, "late_lookup"
    )

    manager.prefetch_all_done_callback(
        future, "late_lookup", cum_chunk_lengths_total, tier_expected_chunks
    )
    loop.close()

    assert manager.async_lookup_server.responses == []
    assert obj0.ref_count_down_called
    assert obj1.ref_count_down_called
    assert obj0.unpin_called
    assert obj1.unpin_called


def test_cancelled_prefetch_releases_completed_backend_task_results(
    patched_storage_manager,
) -> None:
    manager, _event_manager_mod = patched_storage_manager
    cum_chunk_lengths_total = [0, 256, 512]
    tier_expected_chunks = [2]
    obj0 = MockMemoryObj(0)
    obj1 = MockMemoryObj(1)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    all_done = loop.create_future()
    all_done.cancel()
    loading_task = loop.create_future()
    loading_task.set_result([("key0", obj0), ("key1", obj1)])

    manager.prefetch_all_done_callback(
        all_done,
        "cancelled_lookup",
        cum_chunk_lengths_total,
        tier_expected_chunks,
        loading_tasks=[loading_task],
    )
    loop.close()

    assert manager.async_lookup_server.responses == []
    assert obj0.ref_count_down_called
    assert obj1.ref_count_down_called
    assert obj0.unpin_called
    assert obj1.unpin_called


def test_layerwise_prefetch_counts_whole_chunks_and_releases_tail(
    patched_storage_manager,
) -> None:
    manager, event_manager_mod = patched_storage_manager
    cum_chunk_lengths_total = [0, 256, 512]
    tier_expected_chunks = [2]
    objs = [MockMemoryObj(idx) for idx in range(3)]

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    future = loop.create_future()
    future.set_result(
        [[("layer0_chunk0", objs[0]), ("layer1_chunk0", objs[1]), ("layer0_tail", objs[2])]]
    )

    manager.event_manager.add_event(
        event_manager_mod.EventType.LOADING, "layerwise_lookup", future
    )

    manager.prefetch_all_done_callback(
        future,
        "layerwise_lookup",
        cum_chunk_lengths_total,
        tier_expected_chunks,
        keys_per_chunk=2,
    )
    loop.close()

    assert manager.async_lookup_server.responses == [("layerwise_lookup", 256)]
    assert not objs[0].ref_count_down_called
    assert not objs[1].ref_count_down_called
    assert objs[2].ref_count_down_called
    assert objs[2].unpin_called
