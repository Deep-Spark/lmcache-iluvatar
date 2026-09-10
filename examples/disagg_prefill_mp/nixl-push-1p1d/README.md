# 1P1D 纯 NixlPush（无 MP）

隔离对照：P→D 只走 **vLLM `NixlPushConnector`**，不启 `lmcache server` / `LMCacheMPConnector`。

用来回答：相对 PDBackend 的 TTFT 差距，有多少来自 **NixlPush 传输栈本身**（而不是 MultiConnector / MP 恢复）。

```text
P：NixlPushConnector（producer, engine_id=nixl-push-p1）
D：NixlPushConnector（consumer, engine_id=nixl-push-d1）
proxy：`disagg_proxy_server_nixl_push.py`（`--nixl-push` 注入 kv_transfer_params + 共享 X-Request-Id）
```

## 启动

先停掉占用同端口的其它栈（默认 P `8100` / D `8200` / proxy `9110`）。**不需要** `start_lmcache_mp.sh`。

**P / D / proxy 必须用同一份 `env.sh`**。P 与 D 的 `engine_id` **必须不同**（`NIXL_PREFILL_ENGINE_ID` vs `NIXL_DECODE_ENGINE_ID`）。

```bash
cd examples/disagg_prefill_mp/nixl-push-1p1d
source env.sh

bash start_proxy.sh
CUDA_VISIBLE_DEVICES=2,3 bash start_d1.sh
CUDA_VISIBLE_DEVICES=0,1 bash start_p1.sh
```

启动后确认：

- P cmdline / 日志：`engine_id=nixl-push-p1`
- D cmdline / 日志：`engine_id=nixl-push-d1`
- proxy：带 `--nixl-push`

> 仅 `--skip-kv-notify-wait` **不够**：没有 `kv_transfer_params` 时 NixlPush 不会传 KV，D 会全量重算 prefill（`nixl_bytes_transferred_count` 一直为 0）。

## Benchmark

不要直接复用 p-cache 的 longQA warmup（它依赖 LMCache telemetry）。用：

```bash
SKIP_PD_WARMUP=1 BASE_URL=http://127.0.0.1:9110/v1 \
  bash ../p-cache-nixl-transfer-1p1d/run-longQA-benchmark.sh
```

或先手动发一条：

```bash
curl -s http://127.0.0.1:9110/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen3-32B-W8A8","prompt":"hello","max_tokens":8,"temperature":0}'
```

## 与 sibling 的差异

| | `p-cache-nixl-transfer-1p1d` | 本目录 |
|--|------------------------------|--------|
| P connector | MultiConnector(MP + NixlPush) | 仅 NixlPush |
| lmcache server | 需要 | 不需要 |
| proxy KV wait | 等 LMCache telemetry | `--skip-kv-notify-wait`（无 MP telemetry） |
| proxy NixlPush | `--nixl-push` 注入 params | `--nixl-push` 注入 params |
| 目的 | MP 复用 + NIXL 传输 | 隔离 NIXL 传输延迟 |

## 传输是否真正发生（必查）

发几条请求后：

```bash
# 应看到 nixl_bytes_transferred / nixl_xfer_time count > 0
curl -s http://127.0.0.1:8100/metrics | grep -E 'nixl_bytes_transferred_count|nixl_xfer_time_seconds_count|nixl_num_failed'
curl -s http://127.0.0.1:8200/metrics | grep -E 'nixl_bytes_transferred_count|request_prefill_time_seconds_sum'
```

健康信号：

- P/D：`vllm:nixl_bytes_transferred_count` **> 0**
- D：`request_prefill_time` 在 warm/命中传输后应接近 0（不应再出现 ~3s 全量 prefill）

proxy 日志也可拆：

- `prefill request duration` ≈ P 侧
- `latency between prefill first response and decode first response` ≈ D 调度 + PUSH_REG + WRITE + 首 token

若后者 ≈ P 的全量 prefill 时长，且 nixl bytes=0，则 **D 在重算，传输未发生**。常见原因：

1. proxy **未**加 `--nixl-push`（缺 `kv_transfer_params`）
2. P/D `engine_id` 相同或与 proxy 传入的不一致
3. D 请求未带与 P 相同的 `X-Request-Id`（PUSH_REG 无法匹配）

确认进程：

```bash
tr '\0' '\n' < /proc/$(pgrep -f 'port 8100' | head -1)/environ | grep NIXL_PREFILL_ENGINE_ID
tr '\0' '\n' < /proc/$(pgrep -f 'port 8200' | head -1)/environ | grep NIXL_DECODE_ENGINE_ID
ps aux | grep disagg_proxy_server | grep -oE '\-\-nixl[^ ]*( [^ ]*)?' | head
```
