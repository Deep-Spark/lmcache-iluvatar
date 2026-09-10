# Copyright (c) 2026, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
import importlib
import importlib.util
import inspect
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import threading

import pytest


def _load_patch_module(monkeypatch):
    for name in ("lmcache", "lmcache.v1", "lmcache.v1.distributed"):
        module = ModuleType(name)
        module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)
    api = ModuleType("lmcache.v1.distributed.api")

    class ObjectKey:
        @staticmethod
        def ComputeKVRank(world_size, global_rank, local_world_size, local_rank):
            return (
                world_size << 28
                | global_rank << 16
                | local_world_size << 12
                | local_rank
            )

    api.ObjectKey = ObjectKey
    api.DEFAULT_ATTN_WINDOW_DESC = object()
    distributed = ModuleType("lmcache.v1.distributed")
    distributed.__path__ = []
    monkeypatch.setitem(sys.modules, "lmcache.v1.distributed", distributed)
    monkeypatch.setitem(sys.modules, "lmcache.v1.distributed.api", api)
    path = (
        Path(__file__).parents[2]
        / "lmcache_iluvatar"
        / "v1"
        / "mp_rank_layout.py"
    )
    spec = importlib.util.spec_from_file_location("_test_mp_rank_layout", path)
    assert spec is not None and spec.loader is not None
    patch = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(patch)
    return patch, ObjectKey


def test_registry_merges_and_unregisters_rank_layouts(monkeypatch):
    patch, object_key = _load_patch_module(monkeypatch)

    @dataclass
    class Entry:
        layout_desc: object
        ref_count: int
        group_layout_descs: dict[int, object]

    class Registry:
        def __init__(self):
            self._registry = {}
            self._lock = threading.Lock()

        def register(
            self,
            model_name,
            world_size,
            layout_desc,
            attn_desc,
            group_layout_descs,
        ):
            key = (model_name, world_size)
            with self._lock:
                entry = self._registry.get(key)
                if entry is None:
                    self._registry[key] = Entry(
                        layout_desc, 1, dict(group_layout_descs)
                    )
                else:
                    entry.layout_desc = layout_desc
                    entry.group_layout_descs = dict(group_layout_descs)
                    entry.ref_count += 1

        def unregister(self, model_name, world_size):
            key = (model_name, world_size)
            with self._lock:
                entry = self._registry[key]
                if entry.ref_count == 1:
                    self._registry.pop(key)
                else:
                    entry.ref_count -= 1

    registry_cls = patch.build_rank_aware_layout_registry(Registry)
    registry = registry_cls()
    registry.register("model", 2, "flat-0", object(), {0: "rank-0"}, worker_id=0)
    registry.register("model", 2, "flat-1", object(), {0: "rank-1"}, worker_id=1)

    rank0 = object_key.ComputeKVRank(2, 0, 2, 0)
    rank1 = object_key.ComputeKVRank(2, 1, 2, 1)
    assert registry.find_rank_group_layout_descs("model", 2) == {
        rank0: {0: "rank-0"},
        rank1: {0: "rank-1"},
    }

    registry.unregister("model", 2, kv_rank=rank0)
    assert registry.find_rank_group_layout_descs("model", 2) == {
        rank1: {0: "rank-1"}
    }


def test_server_registration_binds_context_and_registry_rank(monkeypatch):
    patch, object_key = _load_patch_module(monkeypatch)

    @dataclass
    class Entry:
        layout_desc: object
        ref_count: int
        group_layout_descs: dict[int, object]

    class Registry:
        def __init__(self):
            self._registry = {}
            self._lock = threading.Lock()

        def register(
            self,
            model_name,
            world_size,
            layout_desc,
            attn_desc,
            group_layout_descs,
        ):
            with self._lock:
                self._registry[(model_name, world_size)] = Entry(
                    layout_desc, 1, dict(group_layout_descs)
                )

        def unregister(self, model_name, world_size):
            with self._lock:
                self._registry.pop((model_name, world_size), None)

    registry = patch.build_rank_aware_layout_registry(Registry)()

    class TransferModule:
        def __init__(self):
            self._lock = threading.Lock()
            self._cache_contexts = {}
            self._ctx = SimpleNamespace(layout_desc_registry=registry)

        def register_kv_cache(
            self,
            instance_id,
            kv_caches,
            model_name,
            world_size,
            engine_type,
            layout_hints,
            engine_group_infos,
        ):
            registry.register(
                model_name,
                world_size,
                "flat",
                object(),
                {0: f"layout-{instance_id}"},
            )
            with self._lock:
                self._cache_contexts[instance_id] = SimpleNamespace(
                    model_name=model_name, world_size=world_size
                )

        def _release_entries(self, entries):
            for entry in entries:
                registry.unregister(entry.model_name, entry.world_size)

    module_cls = patch.build_rank_aware_transfer_module(TransferModule)
    sig = inspect.signature(module_cls.register_kv_cache)
    assert list(sig.parameters) == [
        "self",
        "instance_id",
        "kv_caches",
        "model_name",
        "world_size",
        "engine_type",
        "layout_hints",
        "engine_group_infos",
        "worker_id",
    ]
    assert sig.parameters["worker_id"].annotation in (int, "int")

    module = module_cls()
    module.register_kv_cache(7, {}, "model", 1, object(), None, [], 0)

    rank = object_key.ComputeKVRank(1, 0, 1, 0)
    entry = module._cache_contexts[7]
    assert entry.kv_rank == rank
    assert registry.find_rank_group_layout_descs("model", 1) == {
        rank: {0: "layout-7"}
    }

    module._release_entries([entry])
    assert registry.find_rank_group_layout_descs("model", 1) is None


def test_prefetch_reserve_uses_rank_and_object_group_layout(monkeypatch):
    patch, _ = _load_patch_module(monkeypatch)

    @dataclass(frozen=True)
    class Key:
        kv_rank: int
        object_group_id: int

    @dataclass(frozen=True)
    class Layout:
        size_bytes: int

    @dataclass(frozen=True)
    class MemoryObj:
        size_bytes: int

    class Manager:
        def __init__(self):
            self.calls = []

        def reserve_write(self, *, keys, is_temporary, layout_desc, mode):
            self.calls.append((tuple(keys), layout_desc.size_bytes))
            return {
                key: ("ok", MemoryObj(layout_desc.size_bytes)) for key in keys
            }

    class Controller:
        def __init__(self):
            self._l1_manager = Manager()

        def _start_lookup_phase(self, request_id, spec):
            return None

        def _reserve_load_buffers(self, request, keys_to_reserve):
            return self._l1_manager.reserve_write(
                keys=keys_to_reserve,
                is_temporary=[False] * len(keys_to_reserve),
                layout_desc=request.group_layout_descs[0],
                mode="new",
            )

    key0 = Key(kv_rank=10, object_group_id=0)
    key1 = Key(kv_rank=11, object_group_id=0)
    request = SimpleNamespace(
        group_layout_descs={0: Layout(999)},
        rank_group_layout_descs={10: {0: Layout(128)}, 11: {0: Layout(256)}},
    )
    controller_cls = patch.build_rank_aware_prefetch_controller(Controller)
    controller = controller_cls()
    results = controller._reserve_load_buffers(request, [key0, key1])

    assert controller._l1_manager.calls == [
        ((key0,), 128),
        ((key1,), 256),
    ]
    assert results[key0][1].size_bytes == 128
    assert results[key1][1].size_bytes == 256


def test_prefetch_reserve_keeps_flat_path_without_rank_layouts(monkeypatch):
    patch, _ = _load_patch_module(monkeypatch)

    class Controller:
        def __init__(self):
            self._l1_manager = object()
            self.flat_calls = 0

        def _start_lookup_phase(self, request_id, spec):
            return None

        def _reserve_load_buffers(self, request, keys_to_reserve):
            self.flat_calls += 1
            return {"flat"}

    controller_cls = patch.build_rank_aware_prefetch_controller(Controller)
    controller = controller_cls()

    assert controller._reserve_load_buffers(
        SimpleNamespace(rank_group_layout_descs=None), []
    ) == {"flat"}
    assert controller.flat_calls == 1


def test_lookup_fails_when_expanded_rank_layout_is_missing(monkeypatch):
    patch, object_key = _load_patch_module(monkeypatch)
    rank0 = object_key.ComputeKVRank(2, 0, 2, 0)

    class Registry:
        def find(self, model_name, world_size):
            return object()

        def find_rank_group_layout_descs(self, model_name, world_size):
            return {rank0: {0: "layout"}}

    class Lookup:
        def __init__(self):
            self._ctx = SimpleNamespace(layout_desc_registry=Registry())

        def lookup(self, key, tp_size):
            pytest.fail("upstream lookup must not run with a missing rank layout")

    lookup_cls = patch.build_rank_aware_lookup_module(Lookup)
    lookup = lookup_cls()
    key = SimpleNamespace(model_name="model", world_size=2)

    with pytest.raises(RuntimeError, match="Missing per-kv_rank layout descriptors"):
        lookup.lookup(key, 1)


def test_register_wire_appends_worker_id(monkeypatch):
    patch, _ = _load_patch_module(monkeypatch)
    sent = []

    class RequestType:
        name = "REGISTER_KV_CACHE"

    class Transfer:
        def register(self, *args, send_request, **kwargs):
            send_request("client", RequestType(), ["existing"])

    class Adapter:
        worker_id = 7

        def __init__(self):
            self.transfer = patch.build_rank_aware_worker_transfer(Transfer)()

        def _send_register_kv_caches_request(self, kv_caches):
            self.transfer.register(
                1,
                kv_caches,
                "model",
                8,
                1,
                "client",
                1.0,
                send_request=lambda client, request, payload: sent.append(payload),
            )

    adapter_cls = patch.build_rank_aware_worker_adapter(Adapter)
    adapter_cls()._send_register_kv_caches_request({})

    assert sent == [["existing", 7]]


def test_protocol_payload_schema_includes_worker_id(monkeypatch):
    patch, _ = _load_patch_module(monkeypatch)

    class RequestType:
        def __init__(self, name):
            self.name = name

    def get_payload_classes(request_type):
        if request_type.name == "REGISTER_KV_CACHE":
            return [int, object, str, int, object, object, list]
        return [str]

    wrapped = patch.build_rank_aware_get_payload_classes(get_payload_classes)

    assert wrapped(RequestType("REGISTER_KV_CACHE")) == [
        int,
        object,
        str,
        int,
        object,
        object,
        list,
        int,
    ]
    assert wrapped(RequestType("LOOKUP")) == [str]
