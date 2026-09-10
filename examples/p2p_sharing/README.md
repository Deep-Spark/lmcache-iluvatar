# P2P KV Cache 共享（Iluvatar）

两个 vLLM 实例通过 **P2PBackend + controller** 共享 KV cache：实例 1 计算并存储 KV，实例 2 经 NIXL/RDMA 直接从实例 1 拉取。本路径**没有** `lmcache server`。

```text
lmcache_controller (:9000 API, :8300 pull, :8400 reply)
       | ZMQ 元数据（谁持有哪个 hash？）
vLLM 实例 1 (GPU 0, :8010)       vLLM 实例 2 (GPU 1, :8011)
  ├─ LMCacheWorker (:8500)            ├─ LMCacheWorker (:8501)
  ├─ P2PBackend (:8200/8201)          ├─ P2PBackend (:8202/8203)
  └─ LocalCPUBackend (5 GB)           └─ LocalCPUBackend (5 GB)
         | NIXL/RDMA 点对点 KV 传输
```

跨请求复用：实例 1 的 LocalCPUBackend 持久化 KV；controller 维护 chunk hash → peer 映射。实例 2 遇到相同 prefix 时，controller 指向实例 1，NIXL 直接读取。

与 `lmcache server` MP 模式对比：无中心化 cache server，数据经 RDMA 在 peer 间直连；controller 只负责元数据。

## Profile（`env.sh`）

| 项 | 默认值 |
| --- | --- |
| 模型 | `/data/nlp/Qwen3-8B/` |
| `SERVED_MODEL_NAME` | `Qwen3-8B` |
| GPU | 实例 1：`0`；实例 2：`1`（共 2 卡，TP=1） |
| `PYTHONHASHSEED` | `123`（controller 与两个 vLLM **必须一致**） |
| 实例 HTTP | `8010` / `8011` |
| Controller | API `9000`；pull `8300`；reply `8400` |

`start_instance*.sh` **不**设置 `CUDA_VISIBLE_DEVICES`，启动 vLLM 前自行 export 或前缀赋值。

P2P / worker 端口在 `instance1.yaml`、`instance2.yaml`。若改 P2P 或 controller pull/reply 端口，复制 YAML 编辑后通过 `INSTANCE1_CONFIG_FILE` / `INSTANCE2_CONFIG_FILE` 指向。

## 前置条件

- `lmcache`、`vllm`、`lmcache-iluvatar` 已安装（`.pth` import hook 生效）
- `lmcache_controller`、`vllm` 在 `PATH` 上
- NIXL 来自 CoreX 运行时：`python -c "import nixl"`
- 至少 2 块空闲 Iluvatar GPU；下方端口未被占用

可选自检：

```bash
python -c "import nixl; print('nixl ok')"
python -c "import lmcache_iluvatar; print(lmcache_iluvatar.get_patch_state().results)"
```

## 启动栈

每步一个终端；`start_*.sh` 会 `source env.sh`，**GPU 需在启动 vLLM 前自行设置**：

```bash
cd examples/p2p_sharing
source env.sh

# 终端 1 — controller
bash start_controller.sh

# 终端 2 — 实例 1
CUDA_VISIBLE_DEVICES=0 bash start_instance1.sh

# 终端 3 — 实例 2
CUDA_VISIBLE_DEVICES=1 bash start_instance2.sh
```

健康检查：

```bash
curl -sf http://127.0.0.1:8010/health
curl -sf http://127.0.0.1:8011/health
```

## Smoke

```bash
bash test_cache_reuse.sh
```

流程：

1. **冷请求** → 实例 1：计算 KV，写入 LocalCPUBackend
2. **热请求** → 实例 2：应经 P2P 从实例 1 取 KV
3. **可选** → 再请求实例 1（默认开启，`RUN_WARM_REPEAT=0` 可跳过）

HTTP 200 只表示服务可达；P2P 命中须看日志。实例 2 关键日志示例：

```text
LMCache hit tokens: 512
need to load: 512
Retrieved 512 out of 512 required tokens
```

常用覆盖：

```bash
PROMPT_REPEAT=120 MAX_TOKENS=1 bash test_cache_reuse.sh
RUN_WARM_REPEAT=0 bash test_cache_reuse.sh
```

## CI

`ci/run.sh` 会 `source env.sh` 并注入 `LMCACHE_CI_*`，按 **controller → instance1 → instance2** 起服务后跑 `test_cache_reuse.sh`。

仓库根目录编排：

```bash
cd examples
python3 run_all_lmcache_iluvatar_tests.py \
  --cases p2p_sharing \
  --base-model-path /data/nlp/Qwen3-8B/ \
  --gpus 0,1
```

## 端口

| 服务 | 端口 |
| --- | --- |
| Controller API | 9000 |
| Controller pull / reply | 8300 / 8400 |
| 实例 1 vLLM | 8010 |
| 实例 2 vLLM | 8011 |
| 实例 1 P2P init / lookup | 8200 / 8201 |
| 实例 2 P2P init / lookup | 8202 / 8203 |
| 实例 1 / 2 Worker | 8500 / 8501 |

## 说明

- `PYTHONHASHSEED` 不一致会导致 chunk hash 对不上，P2P lookup 全 miss。
- Iluvatar 插件经 `.pth` import hook 自动激活；NIXL transfer channel 走上游实现。
- 非 disagg prefill：两实例独立 serving（`kv_both`），P2P mesh 在实例间共享 KV。
