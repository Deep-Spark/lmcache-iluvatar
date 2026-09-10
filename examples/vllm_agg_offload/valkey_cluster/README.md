# 聚合式 vLLM + LMCache MP + Valkey Cluster L2

单实例 `kv_both` + 独立 `lmcache server`，使用外部 Valkey Cluster 作为 L2，
验证 KV cache 的 L2 写入、跨进程重启回读，以及 L2 命中后的输出正确性。

```text
[OpenAI API 客户端]
          │ HTTP :18030
          ▼
[vLLM TP=2, kv_both, LMCacheMPConnector]
          │ ZMQ :6575（控制/请求）+ CUDA IPC（本机 KV 传输）
          ▼
[LMCache MP Server]
    ├── CPU L1
    ├── HTTP :8090（metrics/status，仅观测接口）
    └── glide_sync ↔ Valkey 协议/TCP ↔ [外部 Valkey Cluster L2]
```

`:8090` 的 HTTP 服务只提供 metrics、health 和 status，不承载 vLLM 与 LMCache
之间的 KV 传输。LMCache MP 到 Valkey Cluster 也不走 HTTP，而是通过
`glide_sync` 使用 Valkey 协议连接各个 cluster node。

## Profile（`env.sh`）

| 项 | 默认值 |
| -- | ------ |
| 模型 | `/data/nlp/Qwen3.5-27B-W8A8-fixed` |
| `SERVED_MODEL_NAME` | `Qwen3.5-27B-W8A8-fixed` |
| GPU | 需显式指定，例如 `CUDA_VISIBLE_DEVICES=0,1`（`TENSOR_PARALLEL=2`） |
| `MAX_MODEL_LEN` | `131072` |
| HTTP（vLLM） | `127.0.0.1:18030` |
| `LMCACHE_MP_PORT` / `LMCACHE_HTTP_PORT` | `6575` / `8090` |
| `LMCACHE_CHUNK_SIZE` | `784`（client/server 必须一致） |
| `LMCACHE_L1_SIZE_GB` | `8` |
| `LMCACHE_L1_INIT_SIZE_GB` | `1`（lazy allocation 初始值） |
| `LMCACHE_L2_STORE_POLICY` | `default` |
| `LMCACHE_L2_PREFETCH_MAX_IN_FLIGHT` | `8` |
| `VALKEY_CLUSTER_MODE` | `true`（使用 `GlideClusterClient`） |
| `VALKEY_NUM_WORKERS` | `8` |
| `VALKEY_REQUEST_TIMEOUT` | `5` 秒 |
| `VALKEY_CONNECTION_TIMEOUT` | `10` 秒 |
| `VALKEY_MAX_CAPACITY_GB` | `0`（不启用 LMCache 侧容量驱逐，容量与淘汰由 Valkey 配置决定） |
| `PYTHONHASHSEED` | `42` |
| `VLLM_KV_CACHE_LAYOUT` | `HND` |
| `VLLM_KV_DISABLE_CROSS_GROUP_SHARE` | `1`（hybrid group 不跨组共享 block） |
| `NCCL_CUMEM_ENABLE` | `0`（CoreX TP 初始化兼容） |
| `VLLM_SKIP_P2P_CHECK` | `1` |

`VALKEY_STARTUP_NODES` 没有默认值，必须显式指向本次测试允许写入的 cluster。
`VALKEY_KEY_PREFIX` 默认以 `lmcache-iluvatar-valkey-cluster-smoke` 为基础；
`run_test.sh` 会追加时间戳和 PID，避免不同测试运行共用相同 key namespace。
调用方显式指定前缀时则保持该值不变。

## 前置条件

- 已安装 `lmcache`、`vllm`、`lmcache-iluvatar`。
- LMCache 版本为 `0.5.3`，并包含 `valkey` L2 adapter。
- 已安装 `valkey-glide-sync>=2.3`，Python import 名为 `glide_sync`；普通
  `valkey-glide` 的异步客户端不能替代该 adapter 使用的 sync API。
- 具备 2 块空闲 GPU，且 `18030`、`6575`、`8090` 端口未被占用。
- 运行环境能够解析并访问 Valkey Cluster seed 地址。

Kubernetes 集群内 seed 地址示例：

```bash
export VALKEY_STARTUP_NODES='valkey-0.valkey-headless.lmcache-system.svc.cluster.local:6379,valkey-1.valkey-headless.lmcache-system.svc.cluster.local:6379,valkey-2.valkey-headless.lmcache-system.svc.cluster.local:6379'
```

这些 Service DNS 只能在 Kubernetes 集群内解析。从集群外的节点或物理机运行时，
需要替换成该环境实际可路由的 seed 地址。cluster 模式下 seed 用于初始拓扑发现，
后续 slot 路由及 `MOVED`/`ASK` 重定向由 `GlideClusterClient` 处理。

## 手工 L2 重启回读与输出验收

使用三个终端：Terminal 1 运行 LMCache MP，Terminal 2 运行 vLLM，Terminal 3
发送请求和检查日志。下面的公共环境变量需要在三个终端中保持一致，尤其是
`VALKEY_KEY_PREFIX`；cold 和 warm 使用不同前缀会导致 L2 miss。

```bash
cd examples/vllm_agg_offload/valkey_cluster
source env.sh

export VALKEY_STARTUP_NODES='host1:6379,host2:6379,host3:6379'
export VALKEY_KEY_PREFIX='lmcache-valkey-manual-001'
export MODEL_PATH=/models/qwen3-5-27b-w8a8-fixed
export SERVED_MODEL_NAME=Qwen3.5-27B-W8A8-fixed
export CUDA_VISIBLE_DEVICES=0,1
export MANUAL_LOG_DIR="$PWD/logs/manual-001"
mkdir -p "$MANUAL_LOG_DIR"
```

### 1. 启动 cold LMCache MP 和 vLLM

Terminal 1：

```bash
bash start_lmcache_mp.sh 2>&1 | tee "$MANUAL_LOG_DIR/lmcache_cold.log"
```

等待日志出现以下内容，确认连接的是 Valkey Cluster：

```text
ValkeyL2Adapter ready: ... cluster_mode=True
```

Terminal 2：

```bash
bash start_server.sh 2>&1 | tee "$MANUAL_LOG_DIR/vllm_cold.log"
```

Terminal 3 检查服务和 TP session：

```bash
curl -sf http://127.0.0.1:8090/healthcheck
curl -sf http://127.0.0.1:18030/health
curl -sf http://127.0.0.1:8090/status | \
  python3 -c 'import json,sys; d=json.load(sys.stdin); n=d["active_sessions"]; print("active_sessions=", n); assert n >= 2'
```

三个命令都成功后再发送 cold 请求。

### 2. 发送 cold 长请求并确认 L2 store

Terminal 3：

```bash
python3 semantic_l2_request.py \
  --base-url http://127.0.0.1:18030 \
  --model "$SERVED_MODEL_NAME" \
  --label cold \
  --output "$MANUAL_LOG_DIR/cold.json" \
  --expected 7319 \
  --prompt-repeat 160 \
  --min-prompt-tokens "$((4 * LMCACHE_CHUNK_SIZE))" \
  --max-tokens 32 \
  --timeout 600
```

等待 LMCache MP 完成异步 store，并确认 cold 日志出现 `Stored`：

```bash
timeout 600 bash -c \
  'until grep -Eq "Stored" "$MANUAL_LOG_DIR/lmcache_cold.log"; do sleep 2; done'
grep -E "Stored" "$MANUAL_LOG_DIR/lmcache_cold.log" | tail
```

`semantic_l2_request.py` 返回成功表示回答为 `7319`、响应可严格按 UTF-8 解码，
且 prompt 覆盖至少 4 个 LMCache chunks。

### 3. 停止并重新启动 LMCache MP 和 vLLM

在 Terminal 2 按 `Ctrl-C` 停止 vLLM，在 Terminal 1 按 `Ctrl-C` 停止 LMCache
MP，确认两个进程都已退出。Valkey Cluster 保持运行，且继续使用相同的
`VALKEY_KEY_PREFIX`。

Terminal 1 重新启动 LMCache MP：

```bash
bash start_lmcache_mp.sh 2>&1 | tee "$MANUAL_LOG_DIR/lmcache_warm.log"
```

Terminal 2 重新启动 vLLM：

```bash
bash start_server.sh 2>&1 | tee "$MANUAL_LOG_DIR/vllm_warm.log"
```

### 4. 确认重连完成

Terminal 3：

```bash
curl -sf http://127.0.0.1:8090/healthcheck
curl -sf http://127.0.0.1:18030/health
curl -sf http://127.0.0.1:8090/status | \
  python3 -c 'import json,sys; d=json.load(sys.stdin); n=d["active_sessions"]; print("active_sessions=", n); assert n >= 2'
```

同时确认 warm LMCache 日志再次出现
`ValkeyL2Adapter ready: ... cluster_mode=True`。只有 health 正常且
`active_sessions >= 2` 后，才发送 warm 请求。

```bash
grep -E "ValkeyL2Adapter ready:.*cluster_mode=True" \
  "$MANUAL_LOG_DIR/lmcache_warm.log"
```

### 5. 发送 warm 请求并验证 L2 hit

Terminal 3：

```bash
python3 semantic_l2_request.py \
  --base-url http://127.0.0.1:18030 \
  --model "$SERVED_MODEL_NAME" \
  --label warm \
  --output "$MANUAL_LOG_DIR/warm.json" \
  --expected 7319 \
  --prompt-repeat 160 \
  --min-prompt-tokens "$((4 * LMCACHE_CHUNK_SIZE))" \
  --max-tokens 32 \
  --timeout 600
```

确认 LMCache MP 从 Valkey L2 prefetch 到正数 chunks：

```bash
timeout 600 bash -c \
  'until grep -Eq "Prefetch request completed \(L1\+L2\).*\([0-9]+ L1, [1-9][0-9]* L2\)" "$MANUAL_LOG_DIR/lmcache_warm.log"; do sleep 2; done'
grep -E "Prefetch request completed \(L1\+L2\)" \
  "$MANUAL_LOG_DIR/lmcache_warm.log" | tail
```

确认 TP0、TP1 都有正数 `Retrieved`：

```bash
grep -E "Worker_TP0.*Retrieved [1-9][0-9]*" "$MANUAL_LOG_DIR/vllm_warm.log"
grep -E "Worker_TP1.*Retrieved [1-9][0-9]*" "$MANUAL_LOG_DIR/vllm_warm.log"
```

最终应同时满足：warm 请求返回 `7319` 且无乱码、warm LMCache 日志中 L2 数量
大于 0、TP0 和 TP1 均有正数 `Retrieved`。`cold.json` 和 `warm.json` 保存两次
请求的回答、token 数与 UTF-8 检查结果。

## 一键执行相同验收流程

`run_test.sh` 自动执行上述启动、cold store、重启、重连等待、warm L2 hit 和输出
检查。运行前不要手工启动 LMCache MP 或 vLLM，以免端口冲突。

```bash
cd examples/vllm_agg_offload/valkey_cluster
VALKEY_STARTUP_NODES='host1:6379,host2:6379,host3:6379' \
MODEL_PATH=/models/qwen3-5-27b-w8a8-fixed \
SERVED_MODEL_NAME=Qwen3.5-27B-W8A8-fixed \
CUDA_VISIBLE_DEVICES=0,1 \
bash run_test.sh
```

## 结果目录与参数覆盖

一键 L2 验证默认写入 `logs/valkey-cluster-<timestamp>/`：

- `cold.json` / `warm.json`：回答、token 数、耗时和 UTF-8 检查结果。
- `lmcache_cold.log` / `lmcache_warm.log`：adapter 启动、L2 store 和 prefetch 证据。
- `vllm_cold.log` / `vllm_warm.log`：各 TP worker 的 retrieve 证据。

可用以下变量覆盖行为：

| 变量 | 作用 |
| ---- | ---- |
| `TEST_LOG_DIR` | 修改一键验证的证据目录 |
| `STARTUP_TIMEOUT` | LMCache、vLLM 和 MP session 就绪超时 |
| `REQUEST_TIMEOUT` | 单次语义请求超时 |
| `CLEANUP_TIMEOUT` | 本测试进程组退出超时 |
| `VALKEY_KEY_PREFIX` | 显式指定 Valkey key namespace；省略时自动生成唯一前缀 |

## 清理

`run_test.sh` 会记录并停止自己启动的进程组。若所需端口已被其他服务占用，脚本
直接失败，不会杀死端口监听者，也不会按进程名清理其他 vLLM 服务。

手工双终端启动时，在各自终端按 `Ctrl-C` 停止 vLLM 和 LMCache MP。不要使用
不带精确 PID/端口范围的全局 `pkill`。本测试不会扫描或删除 Valkey 中其他前缀的
数据；测试 key 可由 Valkey 服务端策略回收，或由管理员按本次
`VALKEY_KEY_PREFIX` 定向清理。
