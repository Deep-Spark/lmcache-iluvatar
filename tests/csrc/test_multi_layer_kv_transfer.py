# Copyright (c) 2026, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Correctness tests for ``multi_layer_kv_transfer`` on Iluvatar HND (format 6)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from kv_transfer.utils import (
    FAST_CONFIG,
    KvTransferConfig,
    engine_kv_format,
    import_torch_and_c_ops,
    make_cleared_paged_kv_caches,
    make_key_value_buffer,
    make_kv_cache_pointers,
    make_paged_kv_caches,
    make_slot_mapping,
    reference_fill_key_value_from_paged,
    run_multi_layer_kv_transfer,
)


@pytest.fixture
def kv_transfer_env():
    try:
        torch, c_ops = import_torch_and_c_ops()
    except (ImportError, RuntimeError) as exc:
        pytest.skip(str(exc))
    return torch, c_ops, torch.device("cuda:0")


def _make_v2_memory_obj(tensor, fmt):
    return SimpleNamespace(tensor=tensor, metadata=SimpleNamespace(fmt=fmt))


def _make_v2_staging_connector(torch, cfg, device):
    from lmcache.v1.gpu_connector.gpu_connectors import VLLMPagedMemGPUConnectorV2

    return VLLMPagedMemGPUConnectorV2(
        cfg.hidden_dim,
        cfg.num_layers,
        use_gpu=True,
        chunk_size=cfg.num_tokens,
        dtype=getattr(torch, cfg.dtype_name),
        device=device,
        layout_hints={"kv_layout": "HND"},
    )


def test_kernel_transfer_roundtrip_hnd(kv_transfer_env):
    """D2H then H2D must restore contiguous HND paged KV contents."""
    torch, c_ops, device = kv_transfer_env
    cfg = FAST_CONFIG
    fmt = engine_kv_format(c_ops)
    original = make_paged_kv_caches(torch, cfg, device, seed=37)
    assert original[0].is_contiguous()

    restored = make_cleared_paged_kv_caches(torch, original)
    key_value = make_key_value_buffer(torch, cfg)
    slot_mapping = make_slot_mapping(torch, cfg, device)

    run_multi_layer_kv_transfer(
        torch,
        c_ops,
        key_value=key_value,
        kv_caches=original,
        slot_mapping=slot_mapping,
        direction=c_ops.TransferDirection.D2H,
        engine_kv_format=fmt,
        cfg=cfg,
    )
    run_multi_layer_kv_transfer(
        torch,
        c_ops,
        key_value=key_value,
        kv_caches=restored,
        slot_mapping=slot_mapping,
        direction=c_ops.TransferDirection.H2D,
        engine_kv_format=fmt,
        cfg=cfg,
    )

    for before, after in zip(original, restored):
        assert torch.allclose(before, after, rtol=0, atol=0)


def test_kernel_h2d_skip_prefix_uses_chunk_relative_slot_mapping(kv_transfer_env):
    """A sliced slot_mapping remains valid when prefix-cache skips part of a chunk."""
    torch, c_ops, device = kv_transfer_env
    full_cfg = FAST_CONFIG
    chunk_cfg = KvTransferConfig(
        num_layers=full_cfg.num_layers,
        num_tokens=32,
        num_heads=full_cfg.num_heads,
        head_size=full_cfg.head_size,
        block_size=full_cfg.block_size,
        dtype_name=full_cfg.dtype_name,
    )
    start = 16
    end = start + chunk_cfg.num_tokens
    vllm_cached_tokens = start + 8
    skip_prefix_n_tokens = vllm_cached_tokens - start

    source = make_paged_kv_caches(torch, full_cfg, device, seed=51)
    restored = make_cleared_paged_kv_caches(torch, source)
    slot_mapping = torch.arange(full_cfg.num_tokens, dtype=torch.int64, device=device)
    chunk_slot_mapping = slot_mapping[start:end]
    key_value = reference_fill_key_value_from_paged(
        torch,
        source,
        chunk_slot_mapping,
        chunk_cfg,
    ).to(device=device, non_blocking=True)

    c_ops.multi_layer_kv_transfer(
        key_value,
        make_kv_cache_pointers(torch, restored, device),
        chunk_slot_mapping,
        device,
        full_cfg.page_buffer_size,
        c_ops.TransferDirection.H2D,
        engine_kv_format(c_ops),
        block_size=full_cfg.block_size,
        head_size=full_cfg.head_size,
        skip_prefix_n_tokens=skip_prefix_n_tokens,
    )
    torch.cuda.synchronize()

    expected = make_cleared_paged_kv_caches(torch, source)
    for layer_idx, paged_layer in enumerate(source):
        for slot in range(vllm_cached_tokens, end):
            block_idx = slot // full_cfg.block_size
            block_offset = slot % full_cfg.block_size
            expected[layer_idx][:, block_idx, :, block_offset, :] = paged_layer[
                :, block_idx, :, block_offset, :
            ]

    for actual, expected_layer in zip(restored, expected):
        assert torch.allclose(actual, expected_layer, rtol=0, atol=0)


def test_v2_connector_h2d_staging_roundtrip(kv_transfer_env):
    from lmcache.v1.memory_management import MemoryFormat

    torch, _c_ops, device = kv_transfer_env
    cfg = FAST_CONFIG
    source = make_paged_kv_caches(torch, cfg, device, seed=71)
    restored = make_cleared_paged_kv_caches(torch, source)
    key_value = reference_fill_key_value_from_paged(
        torch,
        source,
        make_slot_mapping(torch, cfg, device),
        cfg,
    )
    connector = _make_v2_staging_connector(torch, cfg, device)
    assert connector.__lmcache_iluvatar_h2d_staging__ is True

    connector.to_gpu(
        _make_v2_memory_obj(key_value, MemoryFormat.KV_2LTD),
        0,
        cfg.num_tokens,
        kvcaches=restored,
        slot_mapping=make_slot_mapping(torch, cfg, device),
    )
    torch.cuda.synchronize()

    for actual, expected in zip(restored, source):
        assert torch.allclose(actual, expected, rtol=0, atol=0)


def test_v2_connector_h2d_staging_uses_patch(kv_transfer_env, monkeypatch):
    """Full-chunk HND loads must take the patched staging path, not super().to_gpu."""
    torch, _c_ops, device = kv_transfer_env
    from lmcache.v1.gpu_connector.gpu_connectors import VLLMPagedMemGPUConnectorV2
    from lmcache.v1.memory_management import MemoryFormat

    cfg = FAST_CONFIG
    source = make_paged_kv_caches(torch, cfg, device, seed=91)
    restored = make_cleared_paged_kv_caches(torch, source)
    key_value = reference_fill_key_value_from_paged(
        torch,
        source,
        make_slot_mapping(torch, cfg, device),
        cfg,
    )
    connector = _make_v2_staging_connector(torch, cfg, device)

    original_cls = getattr(
        VLLMPagedMemGPUConnectorV2, "__lmcache_iluvatar_original__", None
    )
    assert original_cls is not None

    super_called = {"value": False}

    def _failing_super_to_gpu(self, *args, **kwargs):
        super_called["value"] = True
        raise AssertionError("staging path should not call super().to_gpu")

    monkeypatch.setattr(original_cls, "to_gpu", _failing_super_to_gpu)

    staging_buffer = connector.gpu_buffer
    assert staging_buffer is not None

    connector.to_gpu(
        _make_v2_memory_obj(key_value, MemoryFormat.KV_2LTD),
        0,
        cfg.num_tokens,
        kvcaches=restored,
        slot_mapping=make_slot_mapping(torch, cfg, device),
    )
    torch.cuda.synchronize()

    assert not super_called["value"]
    staged_slice = staging_buffer[:, :, : cfg.num_tokens, :].detach().cpu()
    assert torch.allclose(staged_slice, key_value, rtol=0, atol=0)
    for actual, expected in zip(restored, source):
        assert torch.allclose(actual, expected, rtol=0, atol=0)
