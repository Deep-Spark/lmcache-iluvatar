# Copyright (c) 2026, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers and benchmarks for Iluvatar HND KV transfer (format 6)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import time

# Iluvatar / CoreX: physical GPU KV is contiguous HND [2, NB, NH, BS, HS].


@dataclass(frozen=True)
class KvTransferConfig:
    num_layers: int
    num_tokens: int
    num_heads: int
    head_size: int
    block_size: int
    dtype_name: str = "bfloat16"

    @property
    def hidden_dim(self) -> int:
        return self.num_heads * self.head_size

    @property
    def num_blocks(self) -> int:
        return (self.num_tokens + self.block_size - 1) // self.block_size

    @property
    def page_buffer_size(self) -> int:
        return self.num_blocks * self.block_size


FAST_CONFIG = KvTransferConfig(
    num_layers=4,
    num_tokens=64,
    num_heads=8,
    head_size=128,
    block_size=16,
)

QWEN3_8B_CHUNK_CONFIG = KvTransferConfig(
    num_layers=36,
    num_tokens=256,
    num_heads=8,
    head_size=128,
    block_size=16,
)

TransferDirection = frozenset({"d2h", "h2d", "both"})


def import_torch_and_c_ops():
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")

    try:
        from lmcache_iluvatar import c_ops
    except ImportError as exc:
        raise RuntimeError("lmcache_iluvatar.c_ops is not importable") from exc

    if not c_ops.is_native_available():
        raise RuntimeError("lmcache_iluvatar.c_ops native extension is not built")

    return torch, c_ops


def engine_kv_format(c_ops):
    return c_ops.EngineKVFormat.NL_X_TWO_NB_NH_BS_HS

def _paged_layer_shape(cfg: KvTransferConfig) -> tuple[int, ...]:
    return (2, cfg.num_blocks, cfg.num_heads, cfg.block_size, cfg.head_size)


def make_paged_kv_caches(torch, cfg: KvTransferConfig, device, seed: int):
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    shape = _paged_layer_shape(cfg)
    dtype = getattr(torch, cfg.dtype_name)
    return [
        torch.randn(shape, dtype=dtype, device=device, generator=generator)
        for _ in range(cfg.num_layers)
    ]


def make_key_value_buffer(torch, cfg: KvTransferConfig):
    return torch.zeros(
        (2, cfg.num_layers, cfg.num_tokens, cfg.hidden_dim),
        dtype=getattr(torch, cfg.dtype_name),
        device="cpu",
        pin_memory=True,
    )


def make_slot_mapping(torch, cfg: KvTransferConfig, device):
    return torch.arange(cfg.num_tokens, dtype=torch.int64, device=device)


def make_kv_cache_pointers(torch, kv_caches, device):
    return torch.tensor(
        [tensor.data_ptr() for tensor in kv_caches],
        dtype=torch.int64,
        device=device,
    )


def _paged_value(
    paged_layer, k_or_v: int, slot: int, flat_idx: int, cfg: KvTransferConfig
):
    block_idx = slot // cfg.block_size
    block_offset = slot % cfg.block_size
    head_idx = flat_idx // cfg.head_size
    head_offset = flat_idx % cfg.head_size
    return paged_layer[k_or_v, block_idx, head_idx, block_offset, head_offset]


def reference_fill_key_value_from_paged(
    torch, kv_caches, slot_mapping, cfg: KvTransferConfig
):
    expected = make_key_value_buffer(torch, cfg)
    slot_mapping_cpu = slot_mapping.detach().cpu()
    for layer_idx, paged_layer in enumerate(kv_caches):
        paged_cpu = paged_layer.detach().cpu()
        for token_idx in range(cfg.num_tokens):
            slot = int(slot_mapping_cpu[token_idx].item())
            for k_or_v in range(2):
                for flat_idx in range(cfg.hidden_dim):
                    expected[k_or_v, layer_idx, token_idx, flat_idx] = _paged_value(
                        paged_cpu, k_or_v, slot, flat_idx, cfg
                    )
    return expected


def make_cleared_paged_kv_caches(torch, kv_caches):
    return [torch.zeros_like(tensor) for tensor in kv_caches]


def run_multi_layer_kv_transfer(
    torch,
    c_ops,
    *,
    key_value,
    kv_caches,
    slot_mapping,
    direction,
    engine_kv_format,
    cfg: KvTransferConfig,
):
    device = kv_caches[0].device
    ptrs = make_kv_cache_pointers(torch, kv_caches, device)
    c_ops.multi_layer_kv_transfer(
        key_value,
        ptrs,
        slot_mapping,
        device,
        cfg.page_buffer_size,
        direction,
        engine_kv_format,
        block_size=cfg.block_size,
        head_size=cfg.head_size,
        skip_prefix_n_tokens=0,
    )
    torch.cuda.synchronize()


def time_cuda_call(torch, fn: Callable[[], None], *, warmup_iters: int, bench_iters: int) -> float:
    for _ in range(warmup_iters):
        fn()
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(bench_iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / bench_iters


def transfer_bytes(cfg: KvTransferConfig) -> int:
    return 2 * cfg.num_layers * cfg.num_tokens * cfg.hidden_dim * 2  # bf16 K+V


def _normalize_direction(direction: str) -> str:
    normalized = direction.lower()
    if normalized not in TransferDirection:
        raise ValueError(f"direction must be one of {sorted(TransferDirection)}, got {direction!r}")
    return normalized


def _time_transfer_direction(
    torch,
    c_ops,
    *,
    key_value,
    kv_caches,
    slot_mapping,
    transfer_direction,
    engine_kv_format,
    cfg: KvTransferConfig,
    warmup_iters: int,
    bench_iters: int,
) -> float:
    def run_once() -> None:
        run_multi_layer_kv_transfer(
            torch,
            c_ops,
            key_value=key_value,
            kv_caches=kv_caches,
            slot_mapping=slot_mapping,
            direction=transfer_direction,
            engine_kv_format=engine_kv_format,
            cfg=cfg,
        )

    for _ in range(warmup_iters):
        run_once()
    start = time.perf_counter()
    for _ in range(bench_iters):
        run_once()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / bench_iters


def benchmark_multi_layer_kv_transfer(
    cfg: KvTransferConfig = QWEN3_8B_CHUNK_CONFIG,
    *,
    direction: str = "both",
    device_index: int = 0,
    warmup_iters: int = 3,
    bench_iters: int = 10,
) -> dict[str, float | str]:
    """Time ``multi_layer_kv_transfer`` for Iluvatar contiguous HND (format 6)."""
    direction = _normalize_direction(direction)
    torch, c_ops = import_torch_and_c_ops()
    device = torch.device(f"cuda:{device_index}")
    fmt = engine_kv_format(c_ops)

    kv_caches = make_paged_kv_caches(torch, cfg, device, seed=7)
    key_value = make_key_value_buffer(torch, cfg)
    slot_mapping = make_slot_mapping(torch, cfg, device)
    nbytes = transfer_bytes(cfg)

    results: dict[str, float | str] = {
        "direction": direction,
        "transfer_bytes": float(nbytes),
        "num_layers": float(cfg.num_layers),
        "num_tokens": float(cfg.num_tokens),
        "paged_kv_contiguous": float(kv_caches[0].is_contiguous()),
    }

    if direction in ("d2h", "both"):
        d2h_seconds = _time_transfer_direction(
            torch,
            c_ops,
            key_value=key_value,
            kv_caches=kv_caches,
            slot_mapping=slot_mapping,
            transfer_direction=c_ops.TransferDirection.D2H,
            engine_kv_format=fmt,
            cfg=cfg,
            warmup_iters=warmup_iters,
            bench_iters=bench_iters,
        )
        results["d2h_gbps"] = nbytes / d2h_seconds / 1e9
        results["d2h_ms"] = d2h_seconds * 1e3

    if direction in ("h2d", "both"):
        h2d_seconds = _time_transfer_direction(
            torch,
            c_ops,
            key_value=key_value,
            kv_caches=kv_caches,
            slot_mapping=slot_mapping,
            transfer_direction=c_ops.TransferDirection.H2D,
            engine_kv_format=fmt,
            cfg=cfg,
            warmup_iters=warmup_iters,
            bench_iters=bench_iters,
        )
        results["h2d_gbps"] = nbytes / h2d_seconds / 1e9
        results["h2d_ms"] = h2d_seconds * 1e3

    return results


def benchmark_multi_layer_h2d_staging(
    cfg: KvTransferConfig = QWEN3_8B_CHUNK_CONFIG,
    *,
    device_index: int = 0,
    warmup_iters: int = 3,
    bench_iters: int = 10,
) -> dict[str, float]:
    """Compare direct pinned H2D against copy-to-GPU-buffer then kernel reshape."""
    torch, c_ops = import_torch_and_c_ops()
    device = torch.device(f"cuda:{device_index}")
    fmt = engine_kv_format(c_ops)

    source_kv_caches = make_paged_kv_caches(torch, cfg, device, seed=53)
    slot_mapping = make_slot_mapping(torch, cfg, device)
    pinned_key_value = make_key_value_buffer(torch, cfg)
    staging_key_value = torch.empty_like(pinned_key_value, device=device)
    nbytes = transfer_bytes(cfg)

    run_multi_layer_kv_transfer(
        torch,
        c_ops,
        key_value=pinned_key_value,
        kv_caches=source_kv_caches,
        slot_mapping=slot_mapping,
        direction=c_ops.TransferDirection.D2H,
        engine_kv_format=fmt,
        cfg=cfg,
    )
    staging_key_value.copy_(pinned_key_value, non_blocking=True)
    torch.cuda.synchronize()

    direct_restored = make_cleared_paged_kv_caches(torch, source_kv_caches)
    staged_restored = make_cleared_paged_kv_caches(torch, source_kv_caches)

    def direct_h2d() -> None:
        run_multi_layer_kv_transfer(
            torch,
            c_ops,
            key_value=pinned_key_value,
            kv_caches=direct_restored,
            slot_mapping=slot_mapping,
            direction=c_ops.TransferDirection.H2D,
            engine_kv_format=fmt,
            cfg=cfg,
        )

    def staging_copy() -> None:
        staging_key_value.copy_(pinned_key_value, non_blocking=True)

    def staging_reshape() -> None:
        run_multi_layer_kv_transfer(
            torch,
            c_ops,
            key_value=staging_key_value,
            kv_caches=staged_restored,
            slot_mapping=slot_mapping,
            direction=c_ops.TransferDirection.H2D,
            engine_kv_format=fmt,
            cfg=cfg,
        )

    def staged_total() -> None:
        staging_key_value.copy_(pinned_key_value, non_blocking=True)
        staging_reshape()

    direct_h2d()
    staged_total()
    for direct_layer, staged_layer, expected_layer in zip(
        direct_restored, staged_restored, source_kv_caches
    ):
        if not torch.allclose(direct_layer, expected_layer, rtol=0, atol=0):
            raise RuntimeError("direct H2D staging benchmark correctness check failed")
        if not torch.allclose(staged_layer, expected_layer, rtol=0, atol=0):
            raise RuntimeError("staged H2D benchmark correctness check failed")

    direct_seconds = time_cuda_call(
        torch, direct_h2d, warmup_iters=warmup_iters, bench_iters=bench_iters
    )
    copy_seconds = time_cuda_call(
        torch, staging_copy, warmup_iters=warmup_iters, bench_iters=bench_iters
    )
    reshape_seconds = time_cuda_call(
        torch, staging_reshape, warmup_iters=warmup_iters, bench_iters=bench_iters
    )
    total_seconds = time_cuda_call(
        torch, staged_total, warmup_iters=warmup_iters, bench_iters=bench_iters
    )

    return {
        "direct_h2d_ms": direct_seconds * 1e3,
        "staging_copy_ms": copy_seconds * 1e3,
        "staging_reshape_ms": reshape_seconds * 1e3,
        "staged_total_ms": total_seconds * 1e3,
        "direct_h2d_gbps": nbytes / direct_seconds / 1e9,
        "staging_copy_gbps": nbytes / copy_seconds / 1e9,
        "staging_reshape_effective_gbps": nbytes / reshape_seconds / 1e9,
        "staged_total_gbps": nbytes / total_seconds / 1e9,
        "staged_vs_direct_speedup": direct_seconds / total_seconds,
        "transfer_bytes": float(nbytes),
        "num_layers": float(cfg.num_layers),
        "num_tokens": float(cfg.num_tokens),
        "num_heads": float(cfg.num_heads),
        "head_size": float(cfg.head_size),
        "block_size": float(cfg.block_size),
    }
