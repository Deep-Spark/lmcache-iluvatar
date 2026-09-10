# Aggregated MP Smoke（CI）

单实例 `kv_both` + 独立 `lmcache server`，非 P/D / P2P。验证 **MP shared storage** 上的 external hit，以及 LongQA 长文档复用。

```text
[lmcache server :6555 ZMQ + :8080 metrics]
        ↑ ZMQ / CUDA IPC
[vLLM TP=2 kv_both LMCacheMPConnector :18010]
```

## Profile（`env.sh`）


| 项                       | 默认值                  |
| ------------------------ | ---------------------- |
| 模型                     | `/data/nlp/Qwen3-8B`   |
| `SERVED_MODEL_NAME`      | `Qwen3-8B`             |
| GPU                      | `0,1`（`TENSOR_PARALLEL=2`，共 2 卡） |
| `TENSOR_PARALLEL`        | `2`                    |
| `MAX_MODEL_LEN`          | `40960`                |
| HTTP (vLLM)              | `127.0.0.1:18010`      |
| `LMCACHE_MP_PORT`        | `6555`                 |
| `LMCACHE_HTTP_PORT`      | `8080`（`/metrics`）   |
| `LMCACHE_L1_SIZE_GB`     | `100`                  |
| `LMCACHE_CHUNK_SIZE`     | `256`                  |
| `LMCACHE_MAX_WORKERS`    | `2 * TENSOR_PARALLEL`  |
| `PYTHONHASHSEED`         | `0`                    |
| `VLLM_KV_CACHE_LAYOUT`   | `HND`（Iluvatar H2D）  |


`start_server.sh` **不**设置 `CUDA_VISIBLE_DEVICES`，启动 vLLM 前自行 export 或前缀赋值。

## 前置条件

- `lmcache`、`vllm`、`lmcache-iluvatar` 已安装
- 至少 2 块空闲 GPU；`SERVER_PORT` / `LMCACHE_MP_PORT` / `LMCACHE_HTTP_PORT` 未被占用

## 启动顺序

**必须先启动 MP Server，再启动 vLLM。**

```bash
cd examples/vllm_agg_offload/smoke
source env.sh

# Terminal 1 — MP server
bash start_lmcache_mp.sh

# Terminal 2 — vLLM (after MP is up)
CUDA_VISIBLE_DEVICES=0,1 bash start_server.sh
```

健康检查：

```bash
curl -sf http://127.0.0.1:8080/metrics >/dev/null && echo "mp metrics ok"
curl -sf http://127.0.0.1:18010/health && echo "vllm ok"
```

## Smoke（MP store / lookup-hit）

```bash
# Default LMCACHE_URL (from env.sh) asserts lookup-hit metrics.
bash test_external_hit.sh
# Also assert MP server log Store evidence (CI exports both):
LMCACHE_SERVER_LOG=/path/to/lmcache.log \
  LMCACHE_URL=http://localhost:8080 \
  bash test_external_hit.sh
```

**通过标准**（仅 HTTP 200 不够）：

1. MP server 日志出现 `Stored`（COLD 后）
2. WARM 后 `lmcache_mp_lookup_hit_tokens_total > 0`（`LMCACHE_URL/metrics`）

可选：vLLM 日志中的 `LMCache hit tokens > 0` / `Retrieved`。

## LongQA benchmark

服务已启动且 `/health` 正常后（直接打 vLLM，无 proxy、无 `--pd-disagg-ttft`）：

```bash
bash run-longQA-benchmark.sh --quick
bash run-longQA-benchmark.sh
```

结果：`results/longqa-<timestamp>/{warmup,query}_round.csv`，日志：`logs/longqa-<timestamp>/longqa-bench.txt`。

## 清理

本示例保持两个独立启动入口，不提供一键 stack 管理。停止时：

```bash
# 停 vLLM（按端口或进程）
pkill -f "vllm.entrypoints.openai.api_server.*--port ${SERVER_PORT:-18010}" || true
# 停 MP server
pkill -f "lmcache server --port ${LMCACHE_MP_PORT:-6555}" || true
```

CI 使用 `examples/ci/lib.sh` 的 `trap ci_cleanup EXIT` 按已注册端口回收进程。
