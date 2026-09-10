# Copyright (c) 2026, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Tests for CSUPPORT-7101 PP cache broadcast patches."""

from __future__ import annotations

import importlib
from types import SimpleNamespace

import pytest
import torch

from lmcache_iluvatar.v1.pp_broadcast import (
    BROADCAST_DTYPE,
    allocate_uint8_broadcast_receive_tensors,
    broadcast_or_receive_memory_objs,
    pack_uint8_tensor_for_broadcast,
    resolve_broadcast_src_rank,
    unpack_broadcast_tensor_to_uint8,
)
from fake_lmcache import install_fake_lmcache, purge_modules


def _metadata(*, worker_id: int = 0, first_rank: int = 0, broadcast_src_rank: int = 0):
    return SimpleNamespace(
        worker_id=worker_id,
        first_rank=first_rank,
        broadcast_src_rank=broadcast_src_rank,
        is_first_rank=lambda: worker_id == first_rank,
    )


def test_resolve_broadcast_src_rank_prefers_broadcast_field() -> None:
    metadata = _metadata(worker_id=2, first_rank=2, broadcast_src_rank=0)
    assert resolve_broadcast_src_rank(metadata) == 0


def test_resolve_broadcast_src_rank_falls_back_to_first_rank() -> None:
    metadata = SimpleNamespace(worker_id=2, first_rank=2)
    assert resolve_broadcast_src_rank(metadata) == 2


def test_protocol_serializes_torch_int8_dtype(monkeypatch) -> None:
    install_fake_lmcache(monkeypatch)
    purge_modules("lmcache_iluvatar")
    plugin = importlib.import_module("lmcache_iluvatar")
    plugin.activate_patches()

    protocol_module = importlib.import_module("lmcache.v1.protocol")
    assert protocol_module.DTYPE_TO_INT[torch.int8] == 9
    assert protocol_module.INT_TO_DTYPE[9] is torch.int8


def test_pack_uint8_tensor_for_broadcast_round_trips_padded_size() -> None:
    raw = torch.arange(5, dtype=torch.uint8)
    packed = pack_uint8_tensor_for_broadcast(raw)

    assert packed.dtype == BROADCAST_DTYPE
    assert packed.numel() == 2

    received_raw, receive_tensor = allocate_uint8_broadcast_receive_tensors(
        raw.numel(), raw.device
    )
    receive_tensor.copy_(packed)
    unpack_broadcast_tensor_to_uint8(received_raw, receive_tensor)

    assert torch.equal(received_raw, raw)


def test_broadcast_uses_broadcast_source_rank_and_packed_uint8_tensor() -> None:
    import importlib

    from lmcache.v1.memory_management import MemoryObjMetadata

    purge_modules("lmcache_iluvatar")
    importlib.import_module("lmcache_iluvatar")
    cache_engine_mod = importlib.import_module("lmcache.v1.cache_engine")

    metadata = _metadata(worker_id=2, first_rank=2, broadcast_src_rank=0)
    raw_tensor = torch.arange(5, dtype=torch.uint8)
    obj_metadata = MemoryObjMetadata(
        shape=torch.Size([5]),
        dtype=torch.uint8,
        address=0,
        phy_size=0,
        ref_count=1,
    )
    memory_obj = SimpleNamespace(metadata=obj_metadata, raw_tensor=raw_tensor)
    engine = cache_engine_mod.LMCacheEngine.__new__(cache_engine_mod.LMCacheEngine)
    engine.metadata = metadata

    object_broadcasts = []
    tensor_broadcasts = []

    def broadcast_object_fn(value, src_rank):
        object_broadcasts.append((value, src_rank))
        return value

    def broadcast_fn(tensor, src_rank):
        tensor_broadcasts.append((tensor.clone(), src_rank))

    engine.broadcast_object_fn = broadcast_object_fn
    engine.broadcast_fn = broadcast_fn

    broadcast_or_receive_memory_objs(
        engine,
        [(None, memory_obj, 0, 5)],
        torch.zeros(5, dtype=torch.bool),
    )

    assert [src_rank for _, src_rank in object_broadcasts] == [0, 0]
    assert len(tensor_broadcasts) == 1
    tensor, src_rank = tensor_broadcasts[0]
    assert src_rank == 0
    assert tensor.dtype == torch.int32
    assert tensor.numel() == 2


def test_patched_cache_engine_delegates_to_pp_broadcast_helper(monkeypatch) -> None:
    install_fake_lmcache(monkeypatch)
    purge_modules("lmcache_iluvatar")
    plugin = importlib.import_module("lmcache_iluvatar")
    plugin.activate_patches()
    cache_engine_mod = importlib.import_module("lmcache.v1.cache_engine")

    engine = cache_engine_mod.LMCacheEngine()
    calls = []

    def _record(*args, **kwargs):
        calls.append((args, kwargs))

    monkeypatch.setattr(
        "lmcache_iluvatar.v1.pp_broadcast.broadcast_or_receive_memory_objs",
        _record,
    )

    reordered_chunks = []
    ret_mask = torch.zeros(3, dtype=torch.bool)
    engine._broadcast_or_receive_memory_objs(reordered_chunks, ret_mask)

    assert len(calls) == 1
    assert calls[0][0][0] is engine
    assert calls[0][0][1] is reordered_chunks
    assert calls[0][0][2] is ret_mask
