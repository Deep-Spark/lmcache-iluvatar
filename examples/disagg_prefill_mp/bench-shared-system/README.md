# MP 2P2D — Shared System, Multi-User Concurrent Benchmark

2 prefiller + 2 decoder，共享 `lmcache server`，proxy round-robin。测 **相同 system prompt、多 user 并发** 时的 L1 复用与 TTFT。

固定 profile：**Qwen3-32B-W8A8，TP=2，8 GPU**（见 `env.sh`）。**不进 CI**，仅手动性能测试。

部署脚本在本目录（`start_*.sh`）。

## 拓扑

```text
lmcache server (6555)
  → disagg proxy (9110) + telemetry (5768)
  → D1 (8200), D2 (8201)
  → P1 (8100), P2 (8101)
```


| 实例  | GPU | 环境变量             |
| --- | --- | ---------------- |
| P1  | 0,1 | `PREFILLER1_GPU` |
| P2  | 2,3 | `PREFILLER2_GPU` |
| D1  | 4,5 | `DECODER1_GPU`   |
| D2  | 6,7 | `DECODER2_GPU`   |


## 手动运行

### 1. 启动栈

每步一个终端；各 `start_*.sh` 会自动 `source env.sh`。**Decoder 须在 Prefiller 之前启动**（与 `smoke/` 相同）。

```bash
cd examples/disagg_prefill_mp/bench-shared-system

# 1. MP server
bash start_lmcache_mp.sh

# 2–6. proxy + decoders + prefillers
bash start_proxy.sh
bash start_d1.sh
bash start_d2.sh
bash start_p1.sh
bash start_p2.sh
```



### 2. Benchmark

```bash
bash run_quick.sh          # 小规模 + hit 断言
bash run_benchmark.sh      # 全量
NUM_USERS=16 BENCH_ROUNDS=2 bash run_benchmark.sh
```

结果：`results/<timestamp>/requests.csv`、`summary.json`。

## Workload

```text
system: <SHARED_SYSTEM>   # 所有 user 相同（默认 ~1500 tokens）
user:   <user-{id} 后缀> + 固定短问题
```

1. warmup：1 user 写入共享 system KV
2. bench：`NUM_USERS` 并发，proxy round-robin 到 P1/P2



## 判读

- `summary.json` → `lookup_hit_tokens_delta` > 0
- `lmcache_mp_lookup_hit_tokens_total` on metrics endpoint



## 限制

- 仅 MP shared L1；无 `enable_pd`
- 需 8 GPU（4 × TP=2）
- proxy round-robin 不绑 session（刻意设计以测跨 P hit）

