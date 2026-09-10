# 2P2D MP P2P — dual server + coordinator TTFT (longQA)

相对 `[../bench-shared-system/](../bench-shared-system/)`（单 server 共享 L1）：本用例跑 **上游 MP P2P**——两台 `lmcache server` 各持本地 L1，跨 pair miss 时经 coordinator 发现 peer，再 **RDMA read** 对端 L1。同侧 P→D 仍走本 server MP store/retrieve（**无 NIXL**）。

```text
Client → disagg_proxy_server.py  (同 idx RR: i%2 → P1+D1 | P2+D2)
           │
Coordinator (:9300)  — membership only          ← IP_A (primary)
    ├─ Server A (:6555 / :8080 / P2P :8555) ← P1 (:8100), D1 (:8200)  ← IP_A
    └─ Server B (:6556 / :8081 / P2P :8556) ← P2 (:8101), D2 (:8201)  ← IP_B
         A ↔ B  peer L1 RDMA read
```

**不进 CI**，仅手动验证。单机 loopback 上的 P2P 多为功能验证；真实 RDMA 性能需跨机 IB/RoCE。

## Profile（`env.sh`）


| 项 | 默认值 |
| --- | --- |
| `IP_A` / `IP_B` | `127.0.0.1` / `127.0.0.1`（单机）；两机写成两台可达 IP |
| `BIND_HOST` | `0.0.0.0`（仅 listen；勿用作 advertise） |
| 模型 | `/data/nlp/Qwen3-32B-W8A8/` |
| TP | 2 |
| GPU（单机 8 卡） | P1:`0,1` P2:`2,3` D1:`4,5` D2:`6,7` |
| 连接 | P1+D1 → Server A；P2+D2 → Server B |
| 每侧 L1 | `LMCACHE_L1_SIZE_GB=200`（40-doc longQA；RAM 紧可下调） |
| L1 align | `--l1-align-bytes 65536`（P2P 建议） |
| Proxy | `[../disagg_proxy_server.py](../disagg_proxy_server.py)`（无 `--nixl-*`） |
| Workload | **仅 longQA** |


端口：Coordinator `9300`；Server A `6555/8080/8555`；Server B `6556/8081/8556`；P/D/proxy 见 `[../env.defaults.sh](../env.defaults.sh)`。

**先停掉其它** `disagg_prefill_mp` **栈**，避免端口与 GPU 冲突。

## 单机 vs 两机

```bash
# 单机（默认，可不 export）
export IP_A=127.0.0.1 IP_B=127.0.0.1

# 两机：IP_A=主节点（coord + proxy + ServerA + P1/D1）；IP_B=另一节点（ServerB + P2/D2）
export IP_A=10.112.2.45
export IP_B=<hostB>
# 每机 4 卡时在对应节点 override，例如：
#   A: PREFILLER1_GPU=0,1 DECODER1_GPU=2,3
#   B: PREFILLER2_GPU=0,1 DECODER2_GPU=2,3
```

`IP_*` 用于 **连接 / advertise**；`BIND_HOST` 仅用于 listen。不要把 `IP_A`/`IP_B` 写成 `0.0.0.0`。

## 启动栈

每步一个终端；`start_p*.sh` / `start_d*.sh` 会从 `env.sh` 设置 `CUDA_VISIBLE_DEVICES`。**Decoder 先于 Prefiller**。

### 单机

```bash
cd examples/disagg_prefill_mp/mp-p2p-2p2d
source env.sh

bash start_coordinator.sh
bash start_lmcache_mp_a.sh
bash start_lmcache_mp_b.sh
bash start_proxy.sh

bash start_d1.sh
bash start_d2.sh
bash start_p1.sh
bash start_p2.sh
```

### 两机

两边都 `source env.sh`（同一套 `IP_A`/`IP_B`）：

| IP_A（主） | IP_B |
| --- | --- |
| `start_coordinator.sh` | — |
| `start_lmcache_mp_a.sh` | `start_lmcache_mp_b.sh` |
| `start_proxy.sh` | — |
| `start_d1.sh` → `start_p1.sh` | `start_d2.sh` → `start_p2.sh` |

确认 P2P 已注册（每侧 `p2p_peer_count=1`）：

```bash
curl -s "http://${IP_A}:9300/instances" | python3 -m json.tool
curl -s "http://${IP_A}:8080/status" | python3 -m json.tool   # Server A
curl -s "http://${IP_B}:8081/status" | python3 -m json.tool   # Server B
```

期望：`p2p_state=registered`，`p2p_peer_count=1`；coordinator 列出两实例且带 `p2p_advertised_url`。

## Smoke

```bash
# 两机时先 export IP_A/IP_B，在能访问 proxy（IP_A:9110）的机器上跑
bash smoke.sh
bash run-longQA-benchmark.sh --quick
```

`run-longQA-benchmark.sh` / `smoke.sh` 会按 `IP_A` 探 P1/D1、按 `IP_B` 探 P2/D2，再等 proxy。

期望：

- proxy 返回 200
- coordinator 两实例 + 两侧 `p2p_state=registered`
- `NUM_RUNS=4` 后 P1/P2 与 D1/D2 均有请求流量



## Benchmark（仅 longQA）

```bash
bash run-longQA-benchmark.sh --quick
bash run-longQA-benchmark.sh
```

结果在 `results/longqa-<ts>/`（`warmup_round.csv` / `query_round.csv` 等）。

默认 `NUM_DOCUMENTS=40`、`REPEAT_MODE=random`：同一 document 的 warmup 与 query 可能落到不同 pair，从而触发 **跨 server P2P hit**（相对 cold TTFT 应明显下降）。若仍见 lmcache 日志 `triggering eviction`，加大 `LMCACHE_L1_SIZE_GB`。
