# 1P1D P-cache 复用 + NIXL P/D 传输（POC）

P 侧用 **LMCache MP** 做 prefix 缓存复用，P→D 实时 KV 走 **vLLM NIXL**；D 不连 MP。

```text
P：LMCacheMPConnector（查/存 prefix）+ NIXL（推 KV 给 D）
D：NIXL（收 KV 后 decode）
proxy：`disagg_proxy_server_nixl_push.py`（等 LMCache telemetry + `--nixl-push` 注入 kv_transfer_params）
```

重复长 prefix 请求时：P 从 MP 恢复 prefix → 算 suffix → NIXL 发给 D → D decode。

**不进 CI**，仅手动验证。

## Profile（`env.sh`）


| 项          | 默认值                           |
| ---------- | ----------------------------- |
| 模型         | `/data/nlp/Qwen3-32B-W8A8/`   |
| TP         | 2                             |
| GPU        | P：`0,1`；D：`2,3`（共 4 卡）        |
| NIXL       | `NixlPushConnector`（P 主动推 KV） |
| engine_id  | P：`p-cache-nixl-p1`；D：`p-cache-nixl-d1`（必须不同） |
| MP workers | `2 × TP`                      |


端口见 `[../env.defaults.sh](../env.defaults.sh)`。若 shell 里已有 `NIXL_CONNECTOR`，先 `unset NIXL_CONNECTOR` 再 `source env.sh`。

## 启动栈

每步一个终端；`start_*.sh` 会 `source env.sh`，**GPU 需在启动 vLLM 前自行设置**：

```bash
cd examples/disagg_prefill_mp/p-cache-nixl-transfer-1p1d
source env.sh

bash start_lmcache_mp.sh
bash start_proxy.sh

# Decoder 先于 Prefiller
CUDA_VISIBLE_DEVICES=2,3 bash start_d1.sh
CUDA_VISIBLE_DEVICES=0,1 bash start_p1.sh
```

确认 proxy 带 `--nixl-push`，且 P/D `engine_id` 分别为 `p-cache-nixl-p1` / `p-cache-nixl-d1`。没有 `kv_transfer_params` 时 D 会全量重算 prefill（`nixl_bytes_transferred_count` 为 0）。


## Smoke

```bash
bash smoke.sh
```

期望：

- proxy 返回 200
- `lmcache server` 日志出现 `Stored`（首次）/ `Retrieved`（复用）
- D 日志有 NIXL 初始化与 KV 收发（非 MP retrieve）

`/metrics` 的 `lookup_hit` / L1 在默认短 prompt smoke（`PROMPT_REPEAT=100`）下**可能仍为 0**，不代表失败；可加大 repeat 或改看 server 日志 / 跑 longQA：

```bash
PROMPT_REPEAT=200 bash smoke.sh
bash run-longQA-benchmark.sh --quick
```

逻辑在 `[examples/common/smoke_long_prompt_cache.py](../../common/smoke_long_prompt_cache.py)`。

## Benchmark

**Mooncake trace（TTFT）**

```bash
bash run_mooncake_trace_benchmark.sh --quick
bash run_mooncake_trace_benchmark.sh
```

结果：`results/mooncake-<timestamp>/*.jsonl`，`logs/mooncake-<timestamp>/*.txt`。

**Long-doc QA（P 侧 MP 复用）**

```bash
bash run-longQA-benchmark.sh --quick
SKIP_PD_WARMUP=1 bash run-longQA-benchmark.sh   # 栈已预热时
```

默认会先跑一轮 `smoke.sh` 预热 NIXL 路径。结果：`results/longqa-<timestamp>/*.csv`。

## 说明

- `start_p1.sh` 中 `MultiConnector` 顺序 intentional：先 MP 复用，再 NIXL 传输。
- proxy 同时：等 LMCache `request_store_finished`（MP 落盘）+ 注入 NixlPush `kv_transfer_params`（P→D 真正传 KV）。
- **没有 `kv_transfer_params` 时** NixlPush 不会传 KV，D 会全量重算 prefill（`nixl_bytes_transferred_count` 为 0）。`start_proxy.sh` 已默认 `--nixl-push`。
- 纯 NixlPush 对照 / 传输健康检查：[`../nixl-push-1p1d/`](../nixl-push-1p1d/)（`check_nixl_transfer.sh`）。
- MultiConnector **save-to-all blocks** 补丁由已安装的 `lmcache_iluvatar.pth` hook 在 import `lmcache` / `vllm...multi_connector` 时自动激活（修冷启动少存约一个 chunked-prefill 窗口的问题）。
