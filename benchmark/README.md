# KV transfer benchmarks

GPU throughput benchmarks for `lmcache_iluvatar.c_ops` on **Iluvatar contiguous HND** (format 6).

Correctness tests: `tests/csrc/test_multi_layer_kv_transfer.py`.

`multi_layer_block_kv_transfer`（block API）的 Qwen3.5 HND format 7
正确性由 `tests/csrc/test_multi_layer_block_kv_transfer_hnd.py` 覆盖；本目录的性能脚本仍仅测试
format 6 token/slot API（`multi_layer_kv_transfer`）。

## Prerequisites

- CUDA device
- Built `lmcache_iluvatar.c_ops`

## Scripts

| Script | Purpose |
|--------|---------|
| `kv_transfer/bench_d2h_h2d.py` | `multi_layer_kv_transfer` D2H and/or H2D |
| `kv_transfer/bench_h2d_staging.py` | Direct pinned H2D vs GPU staging |

Shared helpers: `kv_transfer/utils.py`.

## Examples

```bash
cd /path/to/lmcache-iluvatar

# D2H + H2D (default Qwen3-8B TP=1 chunk)
python3 benchmark/kv_transfer/bench_d2h_h2d.py

# H2D only
python3 benchmark/kv_transfer/bench_d2h_h2d.py --direction h2d

# H2D staging A/B (aligns with E2E TP=1 staging optimization)
python3 benchmark/kv_transfer/bench_h2d_staging.py

# TP=2 single-rank geometry
python3 benchmark/kv_transfer/bench_h2d_staging.py --num-heads 4
```

Select GPU with `CUDA_VISIBLE_DEVICES` or `--device-index`.

## Default geometry

`QWEN3_8B_CHUNK_CONFIG`: 36 layers, 256 tokens, 8 heads, 128 head size, block size 16 (~37 MB per transfer).
