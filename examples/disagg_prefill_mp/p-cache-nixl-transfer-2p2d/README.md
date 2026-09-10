# 2P2D P-cache 复用 + NIXL P/D 传输（POC）

相对 [`../p-cache-nixl-transfer-1p1d/`](../p-cache-nixl-transfer-1p1d/)：拓扑扩到 **2P2D**，proxy（`disagg_proxy_server_nixl_push.py`）按独立 round-robin 选中的 `P_i` / `D_j` 注入 NIXL `kv_transfer_params`（每实例独立 `engine_id` + side-channel）。

```text
Client → proxy (RR P_i, RR D_j)
           ├─ P_i: MultiConnector(LMCacheMPConnector + NixlPushConnector)
           │         ↕ shared lmcache MP server
           │         → NIXL push KV to D_j
           └─ D_j: NixlPushConnector only（不连 MP）
```

**不进 CI**，仅手动验证（默认 8 GPU）。

## Profile（`env.sh`）

| 项 | 默认值 |
|----|--------|
| 模型 | `/data/nlp/Qwen3-32B-W8A8/` |
| TP | 2 |
| GPU | P1:`0,1` P2:`2,3` D1:`4,5` D2:`6,7` |
| MP L1 | **`LMCACHE_L1_SIZE_GB=200`**（40-doc longQA 在 100GB 下会 eviction，Query 变冷） |
| NIXL | `NixlPushConnector` |
| engine_id | P1/P2:`p-cache-nixl-p1/p2`；D1/D2:`p-cache-nixl-d1/d2` |
| side-channel | P1:`5600` D1:`5601` P2:`5602` D2:`5603` |
| NIXL kv_port | 每实例独立（P1:`14579` P2:`14581` D1:`14580` D2:`14582`） |

HTTP 端口见 [`../env.defaults.sh`](../env.defaults.sh)（P:`8100/8101` D:`8200/8201` proxy:`9110`）。

## 与 1P1D 的差异

| | 1P1D | 2P2D（本目录） |
|--|------|----------------|
| 实例数 | 1P+1D | 2P+2D |
| Proxy | 同文件 `disagg_proxy_server_nixl_push.py`；单值 `--nixl-*` | 同文件；CSV，长度对齐 `num_prefillers` / `num_decoders` |
| 选路 | 固定一对 | 嵌套 RR 覆盖 P×D；参数只来自本次选中的 ClientInfo |
| GPU | 4 | 8 |

## 启动栈

每步一个终端；`start_p*.sh` / `start_d*.sh` 会从 `env.sh` 设置 `CUDA_VISIBLE_DEVICES`：

```bash
cd examples/disagg_prefill_mp/p-cache-nixl-transfer-2p2d
source env.sh

# L1 默认 200GB（env.sh）；勿再用 100，否则 40-doc longQA 会 L1 eviction
bash start_lmcache_mp.sh
bash start_proxy.sh

# Decoder 先于 Prefiller（CUDA_VISIBLE_DEVICES 由 start_*.sh 从 *_GPU 设置）
bash start_d1.sh
bash start_d2.sh
bash start_p1.sh
bash start_p2.sh
```

确认 proxy 带 `--nixl-push` 且 CSV 为：

- `--nixl-prefill-engine-id p-cache-nixl-p1,p-cache-nixl-p2`
- `--nixl-decode-engine-id p-cache-nixl-d1,p-cache-nixl-d2`
- `--nixl-prefill-side-channel-port 5600,5602`

没有正确 `kv_transfer_params` 时 D 会全量重算 prefill（`nixl_bytes_transferred_count` 为 0）。

## Smoke

```bash
bash smoke.sh
```

期望（AC1–AC3）：

- proxy 返回 200
- `lmcache server` 日志：`Stored`（冷）/ `Retrieved`（热）
- 至少一个 P 的 `/metrics`：`nixl_bytes_transferred_count > 0`
- 默认 `NUM_RUNS=4` 后 P1/P2 与 D1/D2 均有请求流量

传输扫表：

```bash
bash check_nixl_transfer.sh
```

## Benchmark

```bash
bash run-longQA-benchmark.sh --quick
bash run-longQA-benchmark.sh
bash run_mooncake_trace_benchmark.sh --quick
```

结果分别在 `results/longqa-*`、`results/mooncake-*`。

longQA 默认与 [`offload-2p2d`](../../disagg_prefill/offload-2p2d/) 对齐：`NUM_DOCUMENTS=40`、`REPEAT_MODE=random`（便于对照跨 P 复用；MP 在 random 下仍可跨 P 命中）。**依赖 L1≥200GB**；若仍见 lmcache 日志 `triggering eviction`，继续加大 `LMCACHE_L1_SIZE_GB`。

## 说明

- `MultiConnector` 顺序 intentional：先 MP 复用，再 NIXL 传输。
- proxy 等 LMCache `request_store_finished`，再按选中 peer 注入 NixlPush 参数并共享 `X-Request-Id`。
- 1P1D 的 `start_proxy.sh` 仍用单值 CLI，与本目录 CSV 路径兼容（AC5）。
