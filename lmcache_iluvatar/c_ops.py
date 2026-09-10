# Copyright (c) 2026, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Fallback for the LMCache v0.5.3 ``lmcache.c_ops`` ABI.

The active native build target is ``lmcache_iluvatar.c_ops``. When that compiled
extension is not present, this source module exposes the same ABI names and
raises diagnostic errors for native operations instead of pretending they
succeeded.
"""

from __future__ import annotations

from enum import IntEnum
from typing import Any


class UnsupportedIluvatarNativeOpError(RuntimeError):
    """Raised when the Iluvatar native c_ops extension is unavailable."""


def is_native_available() -> bool:
    """Return whether the compiled ``lmcache_iluvatar.c_ops`` was imported."""

    return False


def _unsupported(op_name: str):
    def _raise(*args: Any, **kwargs: Any) -> Any:
        detail = (
            f"Iluvatar native c_ops op {op_name!r} is unavailable because "
            "the compiled lmcache_iluvatar.c_ops extension is not built or "
            "could not be imported. "
            "Build the plugin with the active setup.py entry."
        )
        raise UnsupportedIluvatarNativeOpError(detail)

    _raise.__name__ = op_name
    return _raise


class TransferDirection(IntEnum):
    H2D = 0
    D2H = 1


class EngineKVFormat(IntEnum):
    NB_NL_TWO_BS_NH_HS = 0
    NL_X_TWO_NB_BS_NH_HS = 1
    NL_X_NB_TWO_BS_NH_HS = 2
    NL_X_NB_BS_HS = 3
    TWO_X_NL_X_NBBS_NH_HS = 4
    NL_X_NBBS_ONE_HS = 5
    NL_X_TWO_NB_NH_BS_HS = 6
    NL_X_NB_TWO_NH_BS_HS = 7
    NB_NL_TWO_NH_BS_HS = 8
    TWO_X_NL_X_NB_BS_NH_HS = 9
    NL_X_NB_NH_BS_TWO_HS = 10
    NL_X_NB_BS_NH_TWO_HS = 11
    NL_X_NB_NH_BS_CS = 12
    NL_X_NB_BS_NH_CS = 13
    NL_X_NB_BSV_BSS = 14


GPUKVFormat = EngineKVFormat


class PageBufferShapeDesc:
    def __init__(self) -> None:
        self.kv_size = 0
        self.nl = 0
        self.nb = 0
        self.bs = 0
        self.nh = 0
        self.hs = 0
        self.element_size = 0
        self.block_stride_elems = 0


def is_cross_layer(engine_kv_format: EngineKVFormat) -> bool:
    return engine_kv_format in {
        EngineKVFormat.NB_NL_TWO_BS_NH_HS,
        EngineKVFormat.NB_NL_TWO_NH_BS_HS,
    }


def is_kv_list(engine_kv_format: EngineKVFormat) -> bool:
    return engine_kv_format in {
        EngineKVFormat.TWO_X_NL_X_NBBS_NH_HS,
        EngineKVFormat.TWO_X_NL_X_NB_BS_NH_HS,
    }


def is_layer_list(engine_kv_format: EngineKVFormat) -> bool:
    return engine_kv_format in {
        EngineKVFormat.NL_X_TWO_NB_BS_NH_HS,
        EngineKVFormat.NL_X_NB_TWO_BS_NH_HS,
        EngineKVFormat.NL_X_NB_BS_HS,
        EngineKVFormat.NL_X_NBBS_ONE_HS,
        EngineKVFormat.NL_X_TWO_NB_NH_BS_HS,
        EngineKVFormat.NL_X_NB_TWO_NH_BS_HS,
        EngineKVFormat.NL_X_NB_NH_BS_TWO_HS,
        EngineKVFormat.NL_X_NB_BS_NH_TWO_HS,
        EngineKVFormat.NL_X_NB_NH_BS_CS,
        EngineKVFormat.NL_X_NB_BS_NH_CS,
        EngineKVFormat.NL_X_NB_BSV_BSS,
    }


def is_mla(engine_kv_format: EngineKVFormat) -> bool:
    return engine_kv_format in {
        EngineKVFormat.NL_X_NB_BS_HS,
        EngineKVFormat.NL_X_NBBS_ONE_HS,
        EngineKVFormat.NL_X_NB_BSV_BSS,
    }


multi_layer_kv_transfer = _unsupported("multi_layer_kv_transfer")
multi_layer_kv_transfer_unilateral = _unsupported("multi_layer_kv_transfer_unilateral")
single_layer_kv_transfer = _unsupported("single_layer_kv_transfer")
single_layer_kv_transfer_sgl = _unsupported("single_layer_kv_transfer_sgl")
load_and_reshape_flash = _unsupported("load_and_reshape_flash")
reshape_and_cache_back_flash = _unsupported("reshape_and_cache_back_flash")
lmcache_memcpy_async = _unsupported("lmcache_memcpy_async")
encode_fast_new = _unsupported("encode_fast_new")
decode_fast_new = _unsupported("decode_fast_new")
decode_fast_prefsum = _unsupported("decode_fast_prefsum")
calculate_cdf = _unsupported("calculate_cdf")
rotary_embedding_k_fused = _unsupported("rotary_embedding_k_fused")
rotary_embedding_k_fused_strided = _unsupported("rotary_embedding_k_fused_strided")
alloc_pinned_ptr = _unsupported("alloc_pinned_ptr")
free_pinned_ptr = _unsupported("free_pinned_ptr")
alloc_hugepage_pinned_ptr = _unsupported("alloc_hugepage_pinned_ptr")
free_hugepage_pinned_ptr = _unsupported("free_hugepage_pinned_ptr")
alloc_pinned_numa_ptr = _unsupported("alloc_pinned_numa_ptr")
free_pinned_numa_ptr = _unsupported("free_pinned_numa_ptr")
alloc_hugepage_pinned_numa_ptr = _unsupported("alloc_hugepage_pinned_numa_ptr")
free_hugepage_pinned_numa_ptr = _unsupported("free_hugepage_pinned_numa_ptr")
alloc_numa_ptr = _unsupported("alloc_numa_ptr")
free_numa_ptr = _unsupported("free_numa_ptr")
alloc_shm_pinned_ptr = _unsupported("alloc_shm_pinned_ptr")
free_shm_pinned_ptr = _unsupported("free_shm_pinned_ptr")
batched_memcpy = _unsupported("batched_memcpy")
get_gpu_pci_bus_id = _unsupported("get_gpu_pci_bus_id")
multi_layer_block_kv_transfer = _unsupported("multi_layer_block_kv_transfer")
execute_object_group_transfer = _unsupported("execute_object_group_transfer")
execute_cb_retrieve_plan_flat = _unsupported("execute_cb_retrieve_plan_flat")
record_event_on_stream = _unsupported("record_event_on_stream")
drain_recorded_events = _unsupported("drain_recorded_events")
record_completion_on_stream = _unsupported("record_completion_on_stream")
drain_recorded_completions = _unsupported("drain_recorded_completions")


__all__ = [
    "EngineKVFormat",
    "GPUKVFormat",
    "PageBufferShapeDesc",
    "TransferDirection",
    "UnsupportedIluvatarNativeOpError",
    "alloc_hugepage_pinned_numa_ptr",
    "alloc_hugepage_pinned_ptr",
    "alloc_numa_ptr",
    "alloc_pinned_numa_ptr",
    "alloc_pinned_ptr",
    "alloc_shm_pinned_ptr",
    "batched_memcpy",
    "calculate_cdf",
    "decode_fast_new",
    "decode_fast_prefsum",
    "drain_recorded_completions",
    "drain_recorded_events",
    "encode_fast_new",
    "execute_cb_retrieve_plan_flat",
    "execute_object_group_transfer",
    "free_hugepage_pinned_numa_ptr",
    "free_hugepage_pinned_ptr",
    "free_numa_ptr",
    "free_pinned_numa_ptr",
    "free_pinned_ptr",
    "free_shm_pinned_ptr",
    "get_gpu_pci_bus_id",
    "is_cross_layer",
    "is_kv_list",
    "is_layer_list",
    "is_mla",
    "is_native_available",
    "lmcache_memcpy_async",
    "load_and_reshape_flash",
    "multi_layer_block_kv_transfer",
    "multi_layer_kv_transfer",
    "multi_layer_kv_transfer_unilateral",
    "record_completion_on_stream",
    "record_event_on_stream",
    "reshape_and_cache_back_flash",
    "rotary_embedding_k_fused",
    "rotary_embedding_k_fused_strided",
    "single_layer_kv_transfer",
    "single_layer_kv_transfer_sgl",
]
