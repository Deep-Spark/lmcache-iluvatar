# Copyright (c) 2026, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
import importlib
from types import SimpleNamespace

import pytest

from fake_lmcache import install_fake_lmcache


@dataclass
class AllocRequest:
    keys: list[str]
    req_id: str = ""


@dataclass
class ProxyNotif:
    req_id: str


class FakeMemoryObj:
    def __init__(self, ref_count: int = 1):
        self.ref_count = ref_count

    def ref_count_up(self):
        self.ref_count += 1

    def ref_count_down(self):
        self.ref_count -= 1

    def get_ref_count(self):
        return self.ref_count


@pytest.fixture
def plugin_modules(monkeypatch):
    install_fake_lmcache(monkeypatch)
    return {
        "adapter": importlib.import_module("lmcache_iluvatar.integration.vllm.adapter"),
        "cache_engine": importlib.import_module("lmcache_iluvatar.v1.cache_engine"),
        "pd_backend": importlib.import_module(
            "lmcache_iluvatar.v1.storage_backend.pd_backend"
        ),
        "pd_backend_async": importlib.import_module(
            "lmcache_iluvatar.v1.storage_backend.pd_backend_async"
        ),
        "storage_manager": importlib.import_module(
            "lmcache_iluvatar.v1.storage_backend.storage_manager"
        ),
    }


def test_get_req_id_prefers_disagg_req_id(plugin_modules):
    get_req_id = plugin_modules["adapter"].get_req_id
    request = SimpleNamespace(
        req_id="vllm-request",
        disagg_spec=SimpleNamespace(req_id="disagg-request"),
    )

    assert get_req_id(request) == "disagg-request"
    assert get_req_id(SimpleNamespace(req_id="vllm-request")) == "vllm-request"


def test_wait_for_save_registers_disagg_id_for_lookup_unpin(plugin_modules):
    build_iluvatar_v1_impl = plugin_modules["adapter"].build_iluvatar_v1_impl
    build_iluvatar_cache_engine = plugin_modules[
        "cache_engine"
    ].build_iluvatar_cache_engine

    class BaseAdapter:
        def wait_for_save(self):
            for request in self._parent._get_connector_metadata().requests:
                self.lmcache_engine.lookup_unpin(request.req_id)

    Adapter = build_iluvatar_v1_impl(BaseAdapter)
    Engine = build_iluvatar_cache_engine(_RecordingEngine)
    engine = Engine()
    request = SimpleNamespace(
        req_id="vllm-request",
        disagg_spec=SimpleNamespace(req_id="disagg-request"),
    )
    adapter = Adapter()
    adapter.lmcache_engine = engine
    adapter._parent = SimpleNamespace(
        _get_connector_metadata=lambda: SimpleNamespace(requests=[request])
    )

    adapter.wait_for_save()

    assert engine.unpinned == ["disagg-request"]


def test_wait_for_save_does_not_treat_local_skip_as_pd_progress(plugin_modules):
    """LocalCPU skip_leading must not inflate DisaggSpec.num_transferred_tokens."""

    build_iluvatar_v1_impl = plugin_modules["adapter"].build_iluvatar_v1_impl

    class BaseAdapter:
        def wait_for_save(self):
            request = self._parent._get_connector_metadata().requests[0]
            self.observed_transferred = request.disagg_spec.num_transferred_tokens

    Adapter = build_iluvatar_v1_impl(BaseAdapter)
    request = SimpleNamespace(
        req_id="vllm-request",
        token_ids=list(range(10001)),
        is_last_prefill=True,
        save_spec=SimpleNamespace(skip_leading_tokens=10001, can_save=True),
        disagg_spec=SimpleNamespace(
            req_id="disagg-request",
            num_transferred_tokens=0,
            total_chunks=40,
        ),
    )
    adapter = Adapter()
    adapter.kv_role = "kv_producer"
    adapter.lmcache_engine = SimpleNamespace(storage_manager=SimpleNamespace())
    adapter._parent = SimpleNamespace(
        _get_connector_metadata=lambda: SimpleNamespace(requests=[request])
    )

    adapter.wait_for_save()

    assert adapter.observed_transferred == 0
    assert request.disagg_spec.num_transferred_tokens == 0
    assert request.disagg_spec.total_chunks == 40


def test_wait_for_save_restores_pd_progress_after_vllm_reset(plugin_modules):
    """Restore PD watermark when vLLM resets num_transferred between chunks."""

    build_iluvatar_v1_impl = plugin_modules["adapter"].build_iluvatar_v1_impl
    restore = plugin_modules["adapter"]._restore_pd_transfer_progress
    record = plugin_modules["adapter"]._record_pd_transfer_progress

    class BaseAdapter:
        def wait_for_save(self):
            pass

    Adapter = build_iluvatar_v1_impl(BaseAdapter)
    request = SimpleNamespace(
        req_id="vllm-request",
        token_ids=list(range(16384)),
        is_last_prefill=False,
        save_spec=SimpleNamespace(skip_leading_tokens=8192, can_save=True),
        disagg_spec=SimpleNamespace(
            req_id="disagg-request",
            num_transferred_tokens=8192,
            total_chunks=64,
        ),
    )
    adapter = Adapter()
    adapter._parent = SimpleNamespace(
        _get_connector_metadata=lambda: SimpleNamespace(requests=[request])
    )

    # After a real PD store, remember the watermark (keyed by vLLM req_id).
    record(adapter)
    assert adapter._pd_transferred_tokens["vllm-request"] == 8192

    # vLLM resets DisaggSpec progress; LocalCPU skip may already cover more.
    request.disagg_spec.num_transferred_tokens = 0
    request.save_spec.skip_leading_tokens = 16384

    restore(adapter)

    assert request.disagg_spec.num_transferred_tokens == 8192
    # LocalCPU skip must not have been copied into PD progress.
    assert request.disagg_spec.num_transferred_tokens != 16384
    assert request.disagg_spec.total_chunks == 64


def test_get_finished_clears_pd_watermark(plugin_modules):
    build_iluvatar_v1_impl = plugin_modules["adapter"].build_iluvatar_v1_impl

    class BaseAdapter:
        def get_finished(self, finished_req_ids):
            return None, None

    Adapter = build_iluvatar_v1_impl(BaseAdapter)
    adapter = Adapter()
    adapter._pd_transferred_tokens = {"req-done": 1024, "req-keep": 256}

    adapter.get_finished({"req-done"})

    assert "req-done" not in adapter._pd_transferred_tokens
    assert adapter._pd_transferred_tokens["req-keep"] == 256


def test_wait_for_save_notifies_zero_chunk_only_when_pd_done(
    plugin_modules,
):
    build_iluvatar_v1_impl = plugin_modules["adapter"].build_iluvatar_v1_impl

    class BaseAdapter:
        def wait_for_save(self):
            self.wait_calls = getattr(self, "wait_calls", 0) + 1

    batched_put_calls = []
    Adapter = build_iluvatar_v1_impl(BaseAdapter)
    storage_manager = SimpleNamespace(
        batched_put=lambda keys, objs, transfer_spec=None: batched_put_calls.append(
            (keys, objs, transfer_spec.req_id, transfer_spec.is_last_prefill)
        )
    )

    # Full LocalCPU hit but PD not done → must NOT emit empty put.
    request = SimpleNamespace(
        req_id="vllm-request",
        token_ids=list(range(40611)),
        is_last_prefill=True,
        save_spec=SimpleNamespace(skip_leading_tokens=40611, can_save=True),
        disagg_spec=SimpleNamespace(
            req_id="disagg-request",
            num_transferred_tokens=0,
            total_chunks=159,
        ),
    )
    adapter = Adapter()
    adapter.kv_role = "kv_producer"
    adapter.lmcache_engine = SimpleNamespace(storage_manager=storage_manager)
    adapter._parent = SimpleNamespace(
        _get_connector_metadata=lambda: SimpleNamespace(requests=[request])
    )

    adapter.wait_for_save()
    assert batched_put_calls == []
    assert request.disagg_spec.num_transferred_tokens == 0

    # After PD has transferred everything, empty final put is allowed once.
    request.disagg_spec.num_transferred_tokens = 40611
    adapter.wait_for_save()
    adapter.wait_for_save()

    assert batched_put_calls == [([], [], "disagg-request", True)]
    assert adapter.wait_calls == 3
    assert request.disagg_spec.num_transferred_tokens == 40611


def test_start_load_kv_registers_disagg_id_for_retrieve(plugin_modules):
    build_iluvatar_v1_impl = plugin_modules["adapter"].build_iluvatar_v1_impl
    build_iluvatar_cache_engine = plugin_modules[
        "cache_engine"
    ].build_iluvatar_cache_engine

    class BaseAdapter:
        def start_load_kv(self):
            request = self._parent._get_connector_metadata().requests[0]
            self.lmcache_engine.retrieve([], req_id=request.req_id)

    Adapter = build_iluvatar_v1_impl(BaseAdapter)
    Engine = build_iluvatar_cache_engine(_RecordingEngine)
    engine = Engine()
    request = SimpleNamespace(
        req_id="vllm-request",
        disagg_spec=SimpleNamespace(req_id="disagg-request"),
    )
    adapter = Adapter()
    adapter.lmcache_engine = engine
    adapter._parent = SimpleNamespace(
        _get_connector_metadata=lambda: SimpleNamespace(requests=[request])
    )

    adapter.start_load_kv()

    assert engine.retrieved_kwargs == [{"req_id": "disagg-request"}]


def test_storage_manager_release_pd_reuse_noops_and_forwards(plugin_modules):
    build_iluvatar_storage_manager = plugin_modules[
        "storage_manager"
    ].build_iluvatar_storage_manager
    StorageManager = build_iluvatar_storage_manager(_BaseStorageManager)

    StorageManager({}).release_pd_reuse("req")
    StorageManager({"PDBackend": object()}).release_pd_reuse("req")

    backend = SimpleNamespace(
        calls=[],
        release_reused_keys=lambda req_id: backend.calls.append(req_id),
    )
    StorageManager({"PDBackend": backend}).release_pd_reuse("req")

    assert backend.calls == ["req"]


def test_pd_backend_reused_key_acquire_and_release_refcount(plugin_modules):
    build_iluvatar_pd_backend = plugin_modules["pd_backend"].build_iluvatar_pd_backend
    PDBackend = build_iluvatar_pd_backend(_BasePDBackend)
    backend = PDBackend()
    key = "key"
    mem_obj = FakeMemoryObj(ref_count=1)
    backend.data[key] = mem_obj

    assert backend.acquire_reused_key("req", key)
    assert mem_obj.ref_count == 2
    assert backend.reused_keys_by_req == {"req": [key]}

    backend.release_reused_keys("req")

    assert mem_obj.ref_count == 1
    assert key in backend.data
    assert backend.reused_keys_by_req == {}


def test_pd_backend_release_removes_last_reference(plugin_modules):
    build_iluvatar_pd_backend = plugin_modules["pd_backend"].build_iluvatar_pd_backend
    PDBackend = build_iluvatar_pd_backend(_BasePDBackend)
    backend = PDBackend()
    key = "key"
    mem_obj = FakeMemoryObj(ref_count=1)
    backend.data[key] = mem_obj
    backend._track_reused_key("req", key)

    backend.release_reused_keys("req")

    assert mem_obj.ref_count == 0
    assert key not in backend.data


def test_pd_backend_remote_alloc_request_carries_req_id(plugin_modules):
    build_iluvatar_pd_backend = plugin_modules["pd_backend"].build_iluvatar_pd_backend
    PDBackend = build_iluvatar_pd_backend(_BasePDBackend)
    backend = PDBackend()
    request = backend._get_remote_alloc_request(
        ["key"],
        ["mem"],
        req_id="disagg-request",
    )

    assert request.req_id == "disagg-request"


def test_pd_backend_empty_last_prefill_notifies_proxy(plugin_modules):
    build_iluvatar_pd_backend = plugin_modules["pd_backend"].build_iluvatar_pd_backend
    PDBackend = build_iluvatar_pd_backend(_BasePDBackend)
    backend = PDBackend()
    backend.proxy_side_channel = SimpleNamespace(sent=[])
    backend.proxy_side_channel.send = backend.proxy_side_channel.sent.append

    backend.batched_submit_put_task(
        [],
        [],
        transfer_spec=SimpleNamespace(req_id="disagg-request", is_last_prefill=True),
    )

    assert len(backend.proxy_side_channel.sent) == 1


def test_pd_backend_remote_alloc_request_preserves_async_signature(plugin_modules):
    build_iluvatar_pd_backend_async = plugin_modules[
        "pd_backend_async"
    ].build_iluvatar_pd_backend_async
    PDBackend = build_iluvatar_pd_backend_async(_BaseAsyncPDBackend)
    backend = PDBackend()

    request = backend._get_remote_alloc_request(
        ["key"],
        ["mem"],
        req_id="disagg-request",
        is_last_batch=True,
        total_chunks=3,
    )

    assert request.req_id == "disagg-request"
    assert request.keys == ["key"]
    assert request.mem_objs == ["mem"]
    assert request.is_last_batch is True
    assert request.total_chunks == 3


def test_pd_backend_async_empty_last_prefill_notifies_proxy(plugin_modules):
    build_iluvatar_pd_backend_async = plugin_modules[
        "pd_backend_async"
    ].build_iluvatar_pd_backend_async
    PDBackend = build_iluvatar_pd_backend_async(_BaseAsyncPDBackend)
    backend = PDBackend()
    sent = []
    backend._notify_proxy_transfer_done = (
        lambda req_id, transfer_spec: sent.append((req_id, transfer_spec))
    )
    transfer_spec = SimpleNamespace(req_id="disagg-request", is_last_prefill=True)

    backend.batched_submit_put_task([], [], transfer_spec=transfer_spec)

    assert sent == [("disagg-request", transfer_spec)]


def test_pd_backend_allocate_and_put_tracks_existing_key(plugin_modules):
    build_iluvatar_pd_backend = plugin_modules["pd_backend"].build_iluvatar_pd_backend
    PDBackend = build_iluvatar_pd_backend(_BasePDBackend)
    backend = PDBackend()
    backend.data["key"] = FakeMemoryObj(ref_count=1)

    response = backend._allocate_and_put(AllocRequest(keys=["key"], req_id="req"))

    assert response.already_sent_indexes == [0]
    assert backend.data["key"].ref_count == 2
    assert backend.reused_keys_by_req == {"req": ["key"]}


class _RecordingEngine:
    def __init__(self):
        self.storage_manager = SimpleNamespace(calls=[], release_pd_reuse=self._release)
        self.remove_after_retrieve = False
        self.unpinned = []
        self.retrieved_kwargs = []

    def _is_passive(self):
        return False

    def retrieve(self, tokens, mask=None, **kwargs):
        self.retrieved_kwargs.append(kwargs)
        return kwargs

    def lookup_unpin(self, lookup_id: str):
        self.unpinned.append(lookup_id)

    def _release(self, req_id: str):
        self.storage_manager.calls.append(req_id)


class _BaseStorageManager:
    def __init__(self, storage_backends):
        self.storage_backends = storage_backends


class _BasePDBackend:
    def __init__(self):
        self.data = {}
        from threading import Lock

        self.data_lock = Lock()

    def _get_remote_alloc_request(self, keys, mem_objs):
        return AllocRequest(keys=list(keys))

    def _allocate_and_put(self, alloc_request):
        already_sent_indexes = []
        for idx, key in enumerate(alloc_request.keys):
            if self.contains(key, pin=False):
                already_sent_indexes.append(idx)
        return SimpleNamespace(already_sent_indexes=already_sent_indexes)

    def contains(self, key, pin=False):
        if key not in self.data:
            return False
        if pin:
            self.data[key].ref_count_up()
        return True

    def batched_submit_put_task(self, keys, memory_objs, transfer_spec=None):
        return self._get_remote_alloc_request(keys, memory_objs)


class _BaseAsyncPDBackend:
    def __init__(self):
        self.pd_config = SimpleNamespace(role="sender")

    def _get_remote_alloc_request(
        self,
        keys,
        mem_objs,
        req_id="",
        is_last_batch=False,
        total_chunks=0,
    ):
        return SimpleNamespace(
            keys=keys,
            mem_objs=mem_objs,
            req_id=req_id,
            is_last_batch=is_last_batch,
            total_chunks=total_chunks,
        )
