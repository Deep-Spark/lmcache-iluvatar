# Aggregated DeepSeek-V4 — CPU L1 vs SSD L2

单实例 `kv_both` + 独立 `lmcache server`，对比两种 MP 存储配置：

| Profile | 启动脚本 | L1 | L2 |
| ------- | -------- | -- | -- |
| CPU | `start_lmcache_mp_cpu.sh` | 大（默认 200 GB） | 无 |
| SSD | `start_lmcache_mp_ssd.sh` | 小（默认 16 GB，仅 staging） | `fs` / `fs_native` → `LMCACHE_L2_BASE_PATH` |

```text
[lmcache server :6555 ZMQ + :8080 metrics]
        ↑ ZMQ / CUDA IPC
[vLLM PP=16, TP=1, kv_both LMCacheMPConnector :18010]
  model: DeepSeek-V4-Flash-w4a8-v2-TN
```

两组实验**互斥**：同一组端口，不要同时起两个 MP server。

## Profile（`env.sh`）

| 项 | 默认值 |
| -- | ------ |
| 模型 | `/data/nlp/DeepSeek-V4-Flash-w4a8-v2-TN/` |
| `SERVED_MODEL_NAME` | `DeepSeek-V4-Flash-w4a8-v2-TN` |
| GPU | `0..15`（`PIPELINE_PARALLEL=16`，`TENSOR_PARALLEL=1`） |
| `MAX_MODEL_LEN` | `1048576`（覆盖 LongQA 1M / 200k） |
| Chunked prefill | 启用，`MAX_NUM_BATCHED_TOKENS=2048` |
| HTTP (vLLM) | `127.0.0.1:18010` |
| `LMCACHE_MP_PORT` / `LMCACHE_HTTP_PORT` | `6555` / `8080` |
| `LMCACHE_CHUNK_SIZE` | `2048`（client/server 必须一致） |
| Warmup→query wait | `WARMUP_QUERY_WAIT_SECONDS=600`（设 `0` 复现 immediate-query HOL） |
| CPU L1 | `LMCACHE_CPU_L1_SIZE_GB=200` |
| SSD L1 | `LMCACHE_SSD_L1_SIZE_GB=16` |
| SSD L2 path | `LMCACHE_L2_BASE_PATH=/data/tmp/lmcache-l2-dpsk-v4` |
| L2 adapter | `fs_native`（`LMCACHE_L2_BASE_PATH` / `LMCACHE_L2_NUM_WORKERS` / `LMCACHE_L2_MAX_CAPACITY_GB`） |
| `PYTHONHASHSEED` | `0` |
| `VLLM_KV_CACHE_LAYOUT` | `HND` |
| MLA | `IXFORMER_DS4_FLASH_MLA_WITH_KVCACHE_VERSION=2` |
| ixccl | `NCCL_IB_DISABLE=1`（见下方说明，本机 IB/RoCE 路径会 segfault） |

> **`NCCL_IB_DISABLE=1` 的原因**：CoreX 4.5.0 的 ixccl 在本机启用 IB/RoCE 时，
> 会在 `ixcclNetPluginInit`（`plugin/net.cc:216`）里 segfault，所有 worker 在
> communicator 初始化阶段被打死，vLLM 报
> `WorkerProc initialization failed`。实测 2/8/16 rank 都会崩，关掉 IB 后
> 16 rank 正常；`NCCL_NET=Socket` 同样可用。单机部署不需要 RDMA，因此默认关闭。

## 启动顺序

**必须先启动 MP Server，再启动 vLLM。**

### A) CPU-only

```bash
cd examples/vllm_agg_offload/dpsk-v4-cpu-ssd
source env.sh

# Terminal 1
bash start_lmcache_mp_cpu.sh

# Terminal 2
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 bash start_server.sh
```

### B) SSD L2

```bash
cd examples/vllm_agg_offload/dpsk-v4-cpu-ssd
source env.sh

# Optional: wipe previous L2 files for a clean cold start
# CLEAR_L2=1 bash start_lmcache_mp_ssd.sh

# Terminal 1 — point base_path at a real SSD mount if needed
LMCACHE_L2_BASE_PATH=/mnt/nvme0/lmcache-dpsk bash start_lmcache_mp_ssd.sh

# Terminal 2
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 bash start_server.sh
```

健康检查：

```bash
curl -sf http://127.0.0.1:8080/metrics >/dev/null && echo "mp metrics ok"
curl -sf http://127.0.0.1:18010/health && echo "vllm ok"
```

短请求 smoke：

```bash
bash curl.sh
WARM=1 bash curl.sh
```

PP16 SSD L2 layout 回归验收：

```bash
# 1. CLEAR_L2=1 启动 SSD profile，再启动 vLLM
# 2. 发 cold 请求，等待 STORE drain；随后以相同前缀发 warm 请求
bash curl.sh
WARM=1 bash curl.sh

# warm 必须 HTTP 200，且不能出现错误尺寸读取
! grep -R "incomplete read" logs/

# 同时检查 :8080/metrics 或 mp.log，确认发生 L2 load/hit，而非仅 L1 命中
curl -sf http://127.0.0.1:8080/metrics | grep -E "l2|prefetch"
```

该路径依赖 `lmcache-iluvatar` 对 LMCache 0.5.3 的 per-`kv_rank` layout
patch；patch 仅匹配 `lmcache>=0.5.3,<0.5.4`。若任一 PP rank 尚未完成
`REGISTER_KV_CACHE`，LOOKUP 会显式失败，不会用最后注册的 layout 分配错误
buffer。

## LongQA（前缀 offload 命中率）

本地 `long_doc_qa.py`：cold 只发 `DOCUMENT_LENGTH * TARGET_HIT_RATE` 前缀落 cache，
query 发完整 `DOCUMENT_LENGTH`（同 doc id + 同前缀），期望命中率 ≈ `TARGET_HIT_RATE`
（warmup 长度会向下对齐到 `LMCACHE_CHUNK_SIZE`，默认 2048）。

默认 drained-store baseline 在 cold/warmup request 返回后固定等待 600 秒，再发
warm query，让排队中的 STORE 有时间完成。该值通过
`WARMUP_QUERY_WAIT_SECONDS` 调整，benchmark 启动日志和
`longqa-bench.txt` 都会打印最终生效值。固定等待只建立可重复的 drained-store
baseline；仍需通过 MP metrics 和 `mp.log` 确认 STORE 已完成。

```bash
# 默认：1M 输入，目标命中 80%，1 条文档
bash run-longQA-benchmark.sh

# 显式覆盖
DOCUMENT_LENGTH=1000000 TARGET_HIT_RATE=0.8 NUM_DOCUMENTS=1 bash run-longQA-benchmark.sh
DOCUMENT_LENGTH=262144 TARGET_HIT_RATE=0.8 NUM_DOCUMENTS=4 bash run-longQA-benchmark.sh

# 保留原 immediate-query HOL 复现：cold 返回后立即发 warm query
WARMUP_QUERY_WAIT_SECONDS=0 bash run-longQA-benchmark.sh

# 覆盖固定等待（秒）
WARMUP_QUERY_WAIT_SECONDS=900 bash run-longQA-benchmark.sh
```

结果：`results/longqa-<timestamp>/{warmup,query}_round.csv`（含 OpenAI
`request_id`），日志：`logs/longqa-<timestamp>/longqa-bench.txt`（含固定等待
生效值）。

验收：query 后 `lmcache_mp_lookup_hit_tokens_total / lookup_requested_tokens_total ≈ TARGET_HIT_RATE`，
且 `mp.log` 出现 `Retrieved`。L1/L2 须装得下 warmup 前缀，否则仍会 0 hit。

## 清理

```bash
pkill -f "vllm serve ${MODEL_PATH:-/data/nlp/DeepSeek-V4-Flash-w4a8-v2-TN}" 2>/dev/null || true
pkill -f "lmcache server --port ${LMCACHE_MP_PORT:-6555}" 2>/dev/null || true
```

切换 CPU ↔ SSD 时务必先停掉旧 MP server（以及通常重启 vLLM），SSD 侧若要干净 cold 可用 `CLEAR_L2=1`。
