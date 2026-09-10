# Copyright (c) 2026, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Tests for CSUPPORT-7101 EIC connector patches."""

from __future__ import annotations

import asyncio
import importlib
import sys
from types import SimpleNamespace

from lmcache_iluvatar.integration.patch.runtime import _PATCH_STATE


class _StatusCode:
    SUCCESS = 0
    KEY_NOT_EXIST = 1


class _StringVector(list):
    def append(self, value):
        super().append(value)


class _ExistOption:
    def __init__(self):
        self.ns = ""


def _load_patched_eic_connector(monkeypatch):
    fake_eic = SimpleNamespace(
        StatusCode=_StatusCode,
        StringVector=_StringVector,
        ExistOption=_ExistOption,
        TransportType=SimpleNamespace(TRANSPORT_GDR=1),
        GetOption=_ExistOption,
        IOBuffers=list,
    )
    monkeypatch.setitem(sys.modules, "eic", fake_eic)

    module_name = "lmcache.v1.storage_backend.connector.eic_connector"
    for name in list(sys.modules):
        if name == "lmcache_iluvatar" or name.startswith("lmcache_iluvatar."):
            monkeypatch.delitem(sys.modules, name, raising=False)
        if name == module_name:
            monkeypatch.delitem(sys.modules, name, raising=False)

    _PATCH_STATE.applied = False
    _PATCH_STATE.results = []

    importlib.import_module(module_name)
    importlib.import_module("lmcache_iluvatar")
    return importlib.import_module(module_name)


def test_exists_sync_uses_configured_namespace(monkeypatch):
    eic_connector = _load_patched_eic_connector(monkeypatch)
    connector = eic_connector.EICConnector.__new__(eic_connector.EICConnector)
    connector.eic_kv_ns = "tenant-a"

    captured = {}

    class _Connection:
        def mexist(self, keys, option):
            captured["keys"] = list(keys)
            captured["ns"] = option.ns
            return _StatusCode.SUCCESS, SimpleNamespace(status_codes=[_StatusCode.SUCCESS])

    connector.connection = _Connection()

    assert connector._exists_sync("key-1")
    assert captured == {"keys": ["key-1"], "ns": "tenant-a"}


def test_batched_async_contains_uses_configured_namespace(monkeypatch):
    eic_connector = _load_patched_eic_connector(monkeypatch)
    connector = eic_connector.EICConnector.__new__(eic_connector.EICConnector)
    connector.eic_kv_ns = "tenant-a"
    captured = {}

    class _Key:
        def __init__(self, value):
            self.value = value

        def to_string(self):
            return self.value

    class _Connection:
        def mexist(self, keys, option):
            captured["keys"] = list(keys)
            captured["ns"] = option.ns
            return _StatusCode.SUCCESS, SimpleNamespace(
                status_codes=[_StatusCode.SUCCESS, _StatusCode.KEY_NOT_EXIST]
            )

    connector.connection = _Connection()

    hits = asyncio.run(
        connector._batched_async_contains(
            "lookup-1",
            [_Key("key-1"), _Key("key-2")],
        )
    )

    assert hits == 1
    assert captured == {"keys": ["key-1", "key-2"], "ns": "tenant-a"}


def test_batched_get_non_blocking_tolerates_partial_failures(monkeypatch):
    eic_connector = _load_patched_eic_connector(monkeypatch)
    connector = eic_connector.EICConnector.__new__(eic_connector.EICConnector)
    good_result = object()
    second_good_result = object()

    class _Key:
        def __init__(self, value):
            self.value = value

        def to_string(self):
            return self.value

    async def _get(key):
        if key.value == "bad":
            raise RuntimeError("boom")
        if key.value == "good-1":
            return good_result
        return second_good_result

    connector._get = _get

    results = asyncio.run(
        connector._batched_get_non_blocking(
            "lookup-1",
            [_Key("good-1"), _Key("bad"), _Key("good-2")],
        )
    )

    assert results == [good_result, second_good_result]


def test_get_data_uses_configured_busy_loop_for_allocation(monkeypatch):
    eic_connector = _load_patched_eic_connector(monkeypatch)
    connector = eic_connector.EICConnector.__new__(eic_connector.EICConnector)
    connector.eic_get_data_busy_loop = True
    connector.eic_kv_ns = "tenant-a"
    connector.trans_type = 0
    connector.connection = SimpleNamespace()

    captured = {}

    def _allocate(shapes, dtypes, fmt, busy_loop=False):
        captured["shapes"] = shapes
        captured["dtypes"] = dtypes
        captured["fmt"] = fmt
        captured["busy_loop"] = busy_loop
        return None

    connector.memory_allocator = SimpleNamespace(allocate=_allocate)
    meta = SimpleNamespace(
        shapes=["shape"],
        dtypes=["dtype"],
        fmt="fmt",
        length=123,
    )

    result = asyncio.run(connector.get_data("key", meta))

    assert result is None
    assert captured == {
        "shapes": ["shape"],
        "dtypes": ["dtype"],
        "fmt": "fmt",
        "busy_loop": True,
    }
