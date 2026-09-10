# GDS 验证（聚合 MP）

在 `vllm_agg_offload` 拓扑下验证 GPUDirect Storage 相关路径。子目录按层级拆分：

| 目录 | 测什么 | 状态 |
|------|--------|------|
| [`l1/`](l1/) | MP `--gds-l1-path`：L1 介质为 NVMe slab（cuFile DMA） | 可用 |
| `l2/` | NIXL `GDS` / `GDS_MT` 作 L2（规划中） | 待添加 |

```text
[lmcache server + GDS L1 slab]
        ↑ ZMQ / CUDA IPC
[vLLM TP=2 kv_both LMCacheMPConnector]
```

**注意：** GDS L1 与依赖 L1 DRAM buffer 的 L2（如 `nixl_store`）以及 P2P 互斥；L2 GDS 用例会单独开 DRAM L1。
