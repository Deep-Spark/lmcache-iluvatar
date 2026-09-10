# MP 2P2D Smoke (CI)

2 prefiller + 2 decoder，共享 `lmcache server`，proxy + telemetry。固定 profile：**Qwen3-8B，TP=1，4 GPU**。

本目录包含 `env.sh`、部署脚本、测试脚本与 CI。

## 启动栈

每步一个终端；各 `start_*.sh` 会自动 `source env.sh`：

```bash
cd examples/disagg_prefill_mp/smoke

# 1. MP server
bash start_lmcache_mp.sh

# 2–6. proxy + decoders + prefillers（Decoder 须在 Prefiller 之前）
bash start_proxy.sh
bash start_d1.sh
bash start_d2.sh
bash start_p1.sh
bash start_p2.sh
```

## 测试

栈就绪后运行长 prefix smoke（逻辑在 `examples/common/smoke_long_prompt_cache.py`）：

```bash
bash smoke.sh
```



## Profile（`env.sh`）


| 项   | 值                      |
| --- | ---------------------- |
| 模型  | `/data/nlp/Qwen3-8B/`  |
| TP  | 1                      |
| GPU | P1=0, P2=1, D1=2, D2=3 |


端口见 `[../env.defaults.sh](../env.defaults.sh)`。

## CI

```bash
cd examples
python3 run_all_lmcache_iluvatar_tests.py \
  --cases disagg_prefill_mp_smoke \
  --base-model-path /data/nlp/Qwen3-8B \
  --gpus 0,1,2,3
```

