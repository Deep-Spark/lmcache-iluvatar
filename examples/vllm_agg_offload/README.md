# vLLM 聚合 + LMCache MP Server

单实例 `kv_both` + 独立 `lmcache server`，非 P/D / P2P。各用例自带 `env.sh` 与 `start_*.sh`。

```text
[lmcache server :6555 ZMQ + :8080 metrics]
        ↑ ZMQ / CUDA IPC
[vLLM kv_both LMCacheMPConnector]
```

## 用例一览

| 目录 | 拓扑 | 测什么 | CI |
|------|------|--------|-----|
| [`smoke/`](smoke/) | Qwen3-8B TP=2 kv_both + MP server | MP shared storage 上的 external hit；LongQA | ✅ |
| [`gds/l1/`](gds/l1/) | Qwen3-8B TP=2 kv_both + MP `--gds-l1-path` | GDS L1 slab 路径举证 + external hit | ❌ 手动（需 GDS） |
| [`valkey_cluster/`](valkey_cluster/) | Qwen3.5 TP=2 kv_both + MP server；外部 Valkey Cluster 作为 L2 | Valkey L2 写入/回读；命中后输出正确性 | ❌ 手动 |
| [`mooncake_l2/`](mooncake_l2/) | 两个 aggregated vLLM + 两个 MP；共享 Mooncake Store | TCP/RDMA 跨实例 L2 写入、空 L1 回读与输出正确性 | ❌ 手动 |
| [`dpsk-v4-cpu-ssd/`](dpsk-v4-cpu-ssd/) | DeepSeek-V4 PP=16 TP=1 | CPU L1 vs SSD L2 对比；LongQA 前缀命中 | ❌ 手动 |

## 怎么选

- **回归 / 冒烟** → `smoke/`（Qwen3-8B，2 GPU）
- **GDS L1（cuFile slab）** → `gds/l1/`（需目标机 GDS；`GDS_L1_PATH` 必填）
- **Valkey Cluster L2 验收** → `valkey_cluster/`（Qwen3.5，2 GPU，需外部 Valkey seed）
- **Mooncake 分布式 L2 验收** → `mooncake_l2/`（多机手工，TCP/RDMA）
- **DeepSeek-V4 CPU vs SSD offload** → `dpsk-v4-cpu-ssd/`（16 GPU）

各用例的启动命令、profile 与判读标准见子目录 README。
