# Copyright (c) 2026, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Correctness tests for ``multi_layer_block_kv_transfer`` on Qwen3.5 HND.

The native block API uses engine KV format 7 for this layout.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from kv_transfer.utils import import_torch_and_c_ops, make_kv_cache_pointers


@pytest.fixture
def kv_transfer_env():
    try:
        torch, c_ops = import_torch_and_c_ops()
    except (ImportError, RuntimeError) as exc:
        pytest.skip(str(exc))
    return torch, c_ops, torch.device("cuda:0")


def _shape_desc(c_ops, logical_pages):
    first = logical_pages[0]
    desc = c_ops.PageBufferShapeDesc()
    desc.kv_size = int(first.shape[1])
    desc.nl = len(logical_pages)
    desc.nb = int(first.shape[0])
    desc.nh = int(first.shape[2])
    desc.bs = int(first.shape[3])
    desc.hs = int(first.shape[4])
    desc.element_size = first.element_size()
    return desc


def test_qwen35_format7_block_d2h_h2d_round_trip(kv_transfer_env, monkeypatch):
    """Group-edited format-7 pages must survive the native MP block path."""
    torch, c_ops, device = kv_transfer_env
    from lmcache_iluvatar.integration.vllm.kv_cache_group_edits import (
        apply_hnd_subpaged_attention_view,
    )

    monkeypatch.setenv("VLLM_KV_CACHE_LAYOUT", "HND")

    num_layers = 2
    num_logical_pages = 3
    logical_block_size = 784
    kernel_block_size = 16
    kernel_pages_per_logical_page = logical_block_size // kernel_block_size
    num_kernel_pages = num_logical_pages * kernel_pages_per_logical_page
    dtype = torch.bfloat16
    spec = SimpleNamespace(
        block_size=logical_block_size,
        page_size_bytes=2 * logical_block_size * 512 * torch.empty(
            (), dtype=dtype
        ).element_size(),
    )

    physical_shape = (num_kernel_pages, 2, 2, kernel_block_size, 256)
    physical_numel = num_kernel_pages * 2 * 2 * kernel_block_size * 256
    # CoreX does not support every random-distribution/dtype combination on
    # device. Build a non-uniform deterministic pattern on CPU, then upload it
    # so axis/order mistakes remain observable without depending on GPU RNG.
    physical_layers = [
        (
            torch.arange(physical_numel, dtype=torch.int32)
            .remainder(251)
            .add_(layer_idx)
            .to(dtype=dtype)
            .reshape(physical_shape)
            .to(device=device)
        )
        for layer_idx in range(num_layers)
    ]
    restored_physical_layers = [
        torch.zeros_like(layer) for layer in physical_layers
    ]
    exposed_layers = [layer.permute(1, 0, 2, 3, 4) for layer in physical_layers]
    restored_exposed_layers = [
        layer.permute(1, 0, 2, 3, 4) for layer in restored_physical_layers
    ]
    assert all(not layer.is_contiguous() for layer in exposed_layers)
    assert all(not layer.is_contiguous() for layer in restored_exposed_layers)

    logical_pages = [
        apply_hnd_subpaged_attention_view(spec, exposed)
        for exposed in exposed_layers
    ]
    restored_pages = [
        apply_hnd_subpaged_attention_view(spec, exposed)
        for exposed in restored_exposed_layers
    ]
    for physical, logical in zip(physical_layers, logical_pages):
        assert logical.shape == (num_logical_pages, 2, 1, 784, 512)
        assert logical.data_ptr() == physical.data_ptr()
    for physical, restored in zip(restored_physical_layers, restored_pages):
        assert restored.shape == (num_logical_pages, 2, 1, 784, 512)
        assert restored.data_ptr() == physical.data_ptr()

    block_ids_list = [0, 2]
    block_ids = torch.tensor(block_ids_list, dtype=torch.int64, device=device)
    source_ptrs = make_kv_cache_pointers(torch, logical_pages, device)
    restored_ptrs = make_kv_cache_pointers(torch, restored_pages, device)
    lmcache_objects = [
        torch.empty(
            (2, num_layers, logical_block_size, 512),
            dtype=dtype,
            device="cpu",
            pin_memory=True,
        )
        for _ in block_ids_list
    ]
    expected = {
        (layer_idx, block_id): page[block_id].clone()
        for layer_idx, page in enumerate(logical_pages)
        for block_id in block_ids_list
    }

    desc = _shape_desc(c_ops, logical_pages)
    fmt = c_ops.EngineKVFormat.NL_X_NB_TWO_NH_BS_HS
    assert int(fmt) == 7
    c_ops.multi_layer_block_kv_transfer(
        source_ptrs,
        [obj.data_ptr() for obj in lmcache_objects],
        block_ids,
        device,
        c_ops.TransferDirection.D2H,
        desc,
        logical_block_size,
        fmt,
        0,
    )
    torch.cuda.synchronize()

    c_ops.multi_layer_block_kv_transfer(
        restored_ptrs,
        [obj.data_ptr() for obj in lmcache_objects],
        block_ids,
        device,
        c_ops.TransferDirection.H2D,
        desc,
        logical_block_size,
        fmt,
        0,
    )
    torch.cuda.synchronize()

    for layer_idx, page in enumerate(restored_pages):
        for block_id in block_ids_list:
            assert torch.equal(page[block_id], expected[(layer_idx, block_id)])
