# 2P2D PDBackend + local-tiered offload（对照 MP+NIXL）

相对 [`../offload/`](../offload/)（1P1D PDBackend）扩到 **2P2D**；相对
[`../../disagg_prefill_mp/p-cache-nixl-transfer-2p2d/`](../../disagg_prefill_mp/p-cache-nixl-transfer-2p2d/)
（共享 MP + NixlPush）提供 **PDBackend + 每实例 local CPU/disk** 对照。

```text
Client → PD proxy (sync RR: P_i ↔ D_i)
           ├─ P_i: LMCacheConnectorV1Dynamic + PDBackend sender
           │         local_cpu + local_disk（每实例独立目录，无共享 MP）
           │         store 热路径 batched_write → D_i pd_buffer（与 P 重叠）
           └─ D_i: LMCacheConnectorV1Dynamic + PDBackend receiver
```

**不进 CI**，仅手动验证（默认 8 GPU）。

## 与 MP+NIXL 2P2D 的差异

| | 本目录（PDBackend） | `p-cache-nixl-transfer-2p2d`（MP+NixlPush） |
|--|---------------------|---------------------------------------------|
| P 侧缓存 | 每实例 local-tiered（CPU+disk） | 共享 `lmcache server`（跨 P 复用） |
| P→D 传输 | PDBackend `pd_buffer` + alloc 旁路（与 P 重叠） | NixlPush：`PUSH_REG` 后再 WRITE |
| Proxy RR | **同步** `P_i↔D_i`（同 idx） | **嵌套**独立 RR（任意 P×D） |
| 跨 P 命中 | RR 下某 doc 只在温过它的 P 上命中；跨 P miss 是预期 | 任意 P 可 retrieve 同一 MP 前缀 |

对照目的：解释 MP+NIXL 相对 PDBackend 的 warm TTFT 差距时，用同拓扑（2P2D）隔离「协议串行 gap」与「共享 MP 复用」；见
`.trellis/spec/backend/lmcache-pdbackend-vs-nixlpush-ttft.md`。

## Profile（`env.sh`）

| 项 | 默认值 |
|----|--------|
| 模型 | `/data/nlp/Qwen3-32B-W8A8/` |
| TP | 2 |
| GPU | P1:`0,1` P2:`2,3` D1:`4,5` D2:`6,7` |
| HTTP | P:`17100/17101` D:`17200/17201` proxy:`19100` |
| ZMQ | `17500` |
| D peer init/alloc | D1:`17300,17301` / `17400,17401`；D2（stride）:`17302,17303` / `17402,17403` |
| local cache | CPU **100 GB** / disk **0 GB**；路径 `/tmp/.cache/pd-2p2d-{p1,p2,d1,d2}`（互不覆盖） |
| `PYTHONHASHSEED` / `VLLM_KV_CACHE_LAYOUT` | `0` / `HND` |

> **端口注意**：端口是 soft default（`${VAR:-…}`）。若同 shell 已 `source` 过 `disagg_prefill_mp/env.defaults.sh`（8100/9110），请先 `unset PREFILLER1_PORT PREFILLER2_PORT DECODER1_PORT DECODER2_PORT PROXY_PORT`（或显式 export 本 profile 端口）再 `source env.sh`，否则 proxy 会连到错误后端。

## Proxy stride 修复

`disagg_proxy_server.py` 在 `incremental_mode`（单 host + 单 decoder HTTP base + `num_decoders>1`）下，对 `decoder_init_port` / `decoder_alloc_port` **按列表长度步进**：

```text
D_i.init  = base_init  + i * len(base_init)
D_i.alloc = base_alloc + i * len(base_alloc)
```

否则 TP=2 时 D2 会与 D1 端口重叠（旧逻辑 `p + i`）。`num_decoders=1`（1P1D `offload/`）行为不变。

## 启动栈

每步一个终端；`start_p*.sh` / `start_d*.sh` 从 `env.sh` 设置 `CUDA_VISIBLE_DEVICES`，且**只清空本实例** cache 目录：

```bash
cd examples/disagg_prefill/offload-2p2d
source env.sh

# 终端 1 — proxy（须先于 workers，PD handshake）
bash start_proxy.sh

# 终端 2–3 — decoders
bash start_d1.sh
bash start_d2.sh

# 终端 4–5 — prefiller
bash start_p1.sh
bash start_p2.sh
```

健康检查：

```bash
curl -sf http://127.0.0.1:17100/health
curl -sf http://127.0.0.1:17101/health
curl -sf http://127.0.0.1:17200/health
curl -sf http://127.0.0.1:17201/health
curl -sf http://127.0.0.1:19100/v1/models
```

## Smoke

```bash
bash smoke.sh
```

期望：

- ≥4 次 `/v1/completions` 均 HTTP 200
- sync RR 后 P1/P2 与 D1/D2 `/metrics` 均有请求流量

## Benchmark

```bash
bash run-longQA-benchmark.sh --quick
bash run-longQA-benchmark.sh
SKIP_PD_WARMUP=1 bash run-longQA-benchmark.sh
```

结果在 `results/longqa-<timestamp>/`（`warmup_round.csv` / `query_round.csv`）。

默认与 nixl-2p2d 对齐，便于对照 TTFT：

| 变量 | 默认 | 说明 |
|------|------|------|
| `DOCUMENT_LENGTH` | `10000` | 与 nixl-2p2d 相同 |
| `NUM_DOCUMENTS` | **`40`** | 2P2D 默认；1P1D 示例用 `20` |
| `REPEAT_MODE` | **`random`** | **必须**用 random 才能测跨 P miss（见下） |
| `SHUFFLE_SEED` | `0` | `random` 可复现 |

### 为何默认 `REPEAT_MODE=random`

Proxy 按**请求序号**做 sync RR（`P_i↔D_i`），不是按 doc 内容路由。

| `REPEAT_MODE` | 与 2P sync RR 的效果 |
|---------------|----------------------|
| **`random`（默认）** | query 打乱顺序 → 约一半 doc 落到「未温过它的 P」→ local miss；MP 对照仍可跨 P 命中 |
| `tile` + 偶数 N | warmup/query 同序 → **全 sticky**，local 也会「像全命中」，测不出跨 P 劣势 |
| `tile` + 奇数 N | 相对翻面 → **全 cross-P**，warmup 缓存整轮用不上 |

覆盖示例：`REPEAT_MODE=tile bash run-longQA-benchmark.sh`（仅用于 sticky 基线，不作跨 P 对照）。

## 说明

- 无共享 MP：每个 P 只保留自己温过的 doc；在 **`REPEAT_MODE=random`** 下 query 落到「未温该 doc 的 P」会出现 local miss——这是对照 MP 跨进程复用的点。
- PDBackend 落点是独立 `pd_buffer`，不依赖 decode HTTP 上的 `PUSH_REG`。
- 端口选用 17xxx/19xxx，避免与正在跑的 nixl-2p2d（8100/9110）冲突。
