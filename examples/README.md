# lmcache-iluvatar 集成测试编排

## 架构（manifest + 各 example 脚本）

GPU 集成测试：

1. 参与 CI 的 example 各自提供 `**ci/run.sh**`（完整生命周期：起服务 → 跑测试 → 断言 → cleanup）。
2. 启用的 case 登记在 `**ci/manifest.json**`。
3. `**run_all_lmcache_iluvatar_tests.py**` 是薄编排器，读取 manifest 并为每个脚本注入统一环境变量。

Layer 1 单元测试见 `ci/README.md`（`clean_build_install.sh` + `pytest`）。

### 新增或移除 CI case

1. 添加 `examples/<your_example>/ci/run.sh`（参考 `p2p_sharing/ci/run.sh` 与 `ci/lib.sh`）。
2. 在 `examples/ci/manifest.json` 追加条目：

```json
{
  "name": "my_example",
  "script": "my_example/ci/run.sh",
  "level": "L1",
  "gpus": 2,
  "requires_model": true,
  "enabled": true,
  "description": "给人看的简短说明"
}
```

1. 本地验证：

```bash
cd examples
python3 run_all_lmcache_iluvatar_tests.py \
  --cases my_example \
  --base-model-path /data/nlp/Qwen3-8B/ \
  --gpus 0,1
```

将 `"enabled": false` 可保留注册但在 `--cases all` 时跳过。

### 传给各 `ci/run.sh` 的环境变量


| 变量                             | 含义                            |
| ------------------------------ | ----------------------------- |
| `LMCACHE_CI_MODEL_PATH`        | 模型权重目录                        |
| `LMCACHE_CI_SERVED_MODEL_NAME` | 可选，vLLM served model 名称       |
| `LMCACHE_CI_GPUS`              | 本 case 使用的 GPU id，逗号分隔        |
| `LMCACHE_CI_LOG_DIR`           | 本 case 日志目录（位于 `--log-dir` 下） |
| `LMCACHE_CI_REPO_ROOT`         | 仓库根目录                         |
| `LMCACHE_CI_STARTUP_TIMEOUT`   | 等待端口/health 的超时（秒）            |
| `LMCACHE_CI_REQUEST_TIMEOUT`   | 测试脚本请求超时提示（秒）                 |
| `LMCACHE_CI_CLEANUP_TIMEOUT`   | cleanup 后等待端口释放（秒，默认 120）     |


共享 shell 辅助函数：`examples/ci/lib.sh`（`ci_wait_for_http`、`ci_start_bg`、`ci_assert_log_keywords` 等）。

每个 `ci/run.sh` 应 `trap 'ci_cleanup' EXIT`，用 `ci_register_ports` 登记 TCP 端口，用 `ci_start_bg`（`setsid`）启动长驻服务。日志路径用 `ci_log_path <name>` — **不要**把 `ci_start_bg` 包在 `$(...)` 里，子 shell 会丢失已跟踪的 PID。

### 默认 case（`--cases all`）


| Case                                  | 目录                        | GPU | 说明                                                                                               |
| ------------------------------------- | ------------------------- | --- | ------------------------------------------------------------------------------------------------ |
| `p2p_sharing`                         | `p2p_sharing/`            | 2   | GPU 不足 2 时在 `all` 中 skip                                                                         |
| `disagg_prefill_mp_smoke`             | `disagg_prefill_mp/smoke/` | 4   | MP 2P2D smoke（Qwen3-8B TP=1）；GPU 不足 4 时在 `all` 中 skip |
| `disagg_prefill_offload_local_tiered` | `disagg_prefill/offload/` | 4   | TP2 prefiller + TP2 decoder、Qwen3-8B；仅 LMCache local-tiered（无 Mooncake）；GPU 不足 4 时在 `all` 中 skip |
| `vllm_agg_offload`                    | `vllm_agg_offload/smoke/` | 2   | 单实例 agg + MP Server external hit（TP=2）；GPU 不足 2 时在 `all` 中 skip                              |
| `cacheblend`                          | `cacheblend/`             | 1   | CacheBlend shuffle-doc QA；通过 Iluvatar CacheBlend attention adapter 支持 `IluFlashAttentionImpl`；GPU 不足 1 时在 `all` 中 skip |


仅跑 agg + MP external-hit（2 GPU）：

```bash
python3 run_all_lmcache_iluvatar_tests.py \
  --cases vllm_agg_offload \
  --base-model-path /data/nlp/Qwen3-8B/ \
  --gpus 0,1
```

仅跑 offload case（TP2+TP2，如 Qwen3-8B）：

```bash
python3 run_all_lmcache_iluvatar_tests.py \
  --cases disagg_prefill_offload_local_tiered \
  --base-model-path /data/nlp/Qwen3-8B/ \
  --gpus 0,1,2,3
```

仅跑 MP 2P2D smoke：

```bash
python3 run_all_lmcache_iluvatar_tests.py \
  --cases disagg_prefill_mp_smoke \
  --base-model-path /data/nlp/Qwen3-8B/ \
  --gpus 0,1,2,3
```

MP 共享 system 多 user 并发 benchmark（**不进 CI**，手动）见 `disagg_prefill_mp/bench-shared-system/README.md`。

外部共享 L2 的手工用例见 `vllm_agg_offload/valkey_cluster/README.md`（Valkey
Cluster）和 `vllm_agg_offload/mooncake_l2/README.md`（Mooncake TCP/RDMA）。后者
依赖多机网络、Mooncake SDK/Store 和可选 RDMA 设备，因此不进入默认 CI。

共享 benchmark 客户端与 shell 工具见 `common/README.md`（`bench_serving.py`、`long_doc_qa.py`、`bench_lib.sh`）。各 example 的 `run-*-benchmark.sh` 为薄 wrapper，代理与栈启动脚本仍保留在各用例目录。

### 前置条件

需要已安装 `vllm`、`lmcache_controller` 与本插件；NIXL（CoreX 运行时，非本仓库子模块）/UCX 环境见 `p2p_sharing/README.md`。

### 常用运行命令

```bash
cd examples
python3 run_all_lmcache_iluvatar_tests.py \
  --base-model-path /data/nlp/Qwen3-8B/ \
  --gpus 0,1,2,3 \
  --keep-going \
  --log-dir ../runtime_result
```


| 参数                    | 含义                    |
| --------------------- | --------------------- |
| `--cases`             | `all` 或逗号分隔的 case 名   |
| `--base-model-path`   | 模型目录（多数 case 必填）      |
| `--served-model-name` | 覆盖 vLLM served name   |
| `--gpus`              | GPU id 列表，如 `0,1,2,3` |
| `--log-dir`           | 日志根目录                 |
| `--keep-going`        | 某个 case 失败后继续后续 case  |


各 example 的手动启动与端口说明见对应子目录下的 `README.md`（如 `p2p_sharing/README.md`）。