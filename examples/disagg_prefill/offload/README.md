# Disaggregated Prefill Offload (local-tiered, 1P1D)

TP2 prefiller + TP2 decoder，LMCache local-tiered 分层 offload（CPU + disk）

默认模型 Qwen3-8B，需 **4 块 GPU**（prefiller `0,1` + decoder `2,3`）。所有默认值在 `env.sh`；各 `start-*.sh` 与 benchmark 脚本会在内部 `source env.sh`。**GPU 需在启动 vLLM 前自行设置** `CUDA_VISIBLE_DEVICES`（与 `disagg_prefill_mp` 一致）。

## 前置条件

- `lmcache`、`vllm`、`lmcache-iluvatar` 已安装
- 模型默认 `/data/nlp/Qwen3-8B/`（可通过 `MODEL_PATH` 覆盖）
- 下方端口未被占用
- 如果测mooncake config，需要先安装mooncake

## 配置（env.sh）


| 变量                                           | 默认                                    | 说明                                                                    |
| -------------------------------------------- | ------------------------------------- | --------------------------------------------------------------------- |
| `MODEL_PATH`                                 | `/data/nlp/Qwen3-32B-W8A8/`           | vLLM 权重路径                                                             |
| `SERVED_MODEL_NAME`                          | `Qwen3-32B-W8A8`                      | API 请求里的 `model` 字段（**不要用路径**）                                        |
| `MAX_MODEL_LEN`                              | `40960`                               | vLLM `--max-model-len`（长上下文 benchmark 可 `export MAX_MODEL_LEN=40960`） |
| `GPU_MEM_UTIL`                               | `0.8`                                 | vLLM `--gpu-memory-utilization`                                       |
| `PREFILLER_CONFIG` / `DECODER_CONFIG`        | `configs/lmcache-local-tiered/*.yaml` | LMCache 配置                                                            |
| `PREFILLER_PORT` / `DECODER_PORT`            | `17100` / `17200`                     | prefiller / decoder HTTP                                              |
| `PUBLIC_PORT`                                | `19100`                               | disagg proxy 对外 API                                                   |
| `PROXY_ZMQ_PORT`                             | `17500`                               | proxy PD 元数据 ZMQ                                                      |
| `DECODER_INIT_PORTS` / `DECODER_ALLOC_PORTS` | `17300,17301` / `17400,17401`         | decoder KV 传输端口                                                       |
| `PREFILLER_RPC_PORT` / `DECODER_RPC_PORT`    | `producer1` / `consumer1`             | LMCache RPC 端口名                                                       |
| `VLLM_KV_CACHE_LAYOUT`                       | `HND`                                 | Iluvatar 物理布局；不设则 vLLM 默认 NHD，LMCache 会选错 EngineKVFormat 导致乱码         |
| `PYTHONHASHSEED`                             | `0`                                   | 与 MP 示例一致，保证 hash 稳定                                                  |


GPU：P `0,1`；D `2,3`（TP=2，共 4 卡）。`start-prefiller.sh` / `start-decoder.sh` **不**设置 `CUDA_VISIBLE_DEVICES`，启动前自行 export 或前缀赋值。

覆盖示例：

```bash
export MODEL_PATH=/data/nlp/Qwen3-8B/
```



## 手动测试（4 终端）

**启动顺序与 CI 一致**（proxy 需先于 prefiller/decoder，以便 PD handshake）：

```bash
cd examples/disagg_prefill/offload
source env.sh

# 终端 1 — proxy
bash start-proxy.sh

# 终端 2 — decoder（等 proxy 就绪）
CUDA_VISIBLE_DEVICES=2,3 bash start-decoder.sh

# 终端 3 — prefiller
CUDA_VISIBLE_DEVICES=0,1 bash start-prefiller.sh
```

健康检查：

```bash
curl -sf http://127.0.0.1:17100/health      # prefiller (vLLM)
curl -sf http://127.0.0.1:17200/health      # decoder (vLLM)
curl -sf http://127.0.0.1:19100/v1/models   # proxy（转发 prefiller，与 bench_wait_for_proxy 一致）
```

**终端 4 — smoke test**（一条 `/v1/completions` 经 proxy 走完整 1P1D 路径）：

```bash
bash ci/smoke_test.sh
```



## Benchmark（可选）

`run-mooncake-benchmark.sh` 使用 **Mooncake FAST25 trace 格式**回放请求（默认 `synthetic`），流量经 **disagg proxy** 走完整 1P1D。这与下方 Mooncake master **无关**——local-tiered CI 不需要启动 Mooncake 服务。

```bash
bash ci/smoke_test.sh
bash run-mooncake-benchmark.sh              # trace 回放，并发 c1/c4/c8
bash run-mooncake-benchmark.sh --quick      # 快速子集（20 prompts，c1/c4）
bash run-longQA-benchmark.sh                # 长文档 QA（默认先 smoke 预热 PD）
SKIP_PD_WARMUP=1 bash run-longQA-benchmark.sh  # 跳过 smoke（需已预热）
```

Benchmark 使用 `SERVED_MODEL_NAME`（默认 `Qwen3-8B`）和 `PUBLIC_PORT`（默认 `19100`）。默认 `NUM_WARMUP=1`；脚本会等待 prefiller / decoder 的 `/health`，proxy 用 `bench_wait_for_proxy`（`GET /v1/models`）。

## Mooncake master（可选，非 CI）

仅在使用 `configs/mooncake-single-instance/` 远程存储时需要；**local-tiered trace benchmark 不需要**：

```bash
bash start_mooncake_master.sh
```



## CI

`ci/run.sh` 会 `source env.sh` 并注入 `LMCACHE_CI_*`，按 **proxy → decoder → prefiller** 顺序起服务后跑 `ci/smoke_test.sh`。

仓库根目录编排：

```bash
cd examples
python3 run_all_lmcache_iluvatar_tests.py \
  --cases disagg_prefill_offload_local_tiered \
  --base-model-path /data/nlp/Qwen3-8B/ \
  --gpus 0,1,2,3
```

