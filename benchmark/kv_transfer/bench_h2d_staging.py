#!/usr/bin/env python3
# Copyright (c) 2026, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Benchmark direct pinned H2D vs GPU staging on Iluvatar HND (format 6)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "benchmark") not in sys.path:
    sys.path.insert(0, str(REPO / "benchmark"))

from kv_transfer.utils import (  # noqa: E402
    KvTransferConfig,
    QWEN3_8B_CHUNK_CONFIG,
    benchmark_multi_layer_h2d_staging,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark H2D staging (Iluvatar HND contiguous paged KV)."
    )
    parser.add_argument("--num-layers", type=int, default=QWEN3_8B_CHUNK_CONFIG.num_layers)
    parser.add_argument("--num-tokens", type=int, default=QWEN3_8B_CHUNK_CONFIG.num_tokens)
    parser.add_argument("--num-heads", type=int, default=QWEN3_8B_CHUNK_CONFIG.num_heads)
    parser.add_argument("--head-size", type=int, default=QWEN3_8B_CHUNK_CONFIG.head_size)
    parser.add_argument("--block-size", type=int, default=QWEN3_8B_CHUNK_CONFIG.block_size)
    parser.add_argument("--dtype", default=QWEN3_8B_CHUNK_CONFIG.dtype_name)
    parser.add_argument("--warmup-iters", type=int, default=5)
    parser.add_argument("--bench-iters", type=int, default=30)
    parser.add_argument("--device-index", type=int, default=0)
    args = parser.parse_args()

    cfg = KvTransferConfig(
        num_layers=args.num_layers,
        num_tokens=args.num_tokens,
        num_heads=args.num_heads,
        head_size=args.head_size,
        block_size=args.block_size,
        dtype_name=args.dtype,
    )
    result = benchmark_multi_layer_h2d_staging(
        cfg,
        device_index=args.device_index,
        warmup_iters=args.warmup_iters,
        bench_iters=args.bench_iters,
    )
    print(
        json.dumps(
            {
                "layout": "HND",
                "gpu_kv_format": 6,
                **result,
                "dtype": cfg.dtype_name,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
