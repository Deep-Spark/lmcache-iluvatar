# CI 测试说明

本仓库有两层 CI，分别覆盖 **插件单元测试** 与 **GPU 集成示例**。二者独立运行，前置依赖不同。

上游 LMCache 与插件版本配对见仓库根目录 [`upstream_pin.json`](../upstream_pin.json)（`lmcache.pypi_version`）；源码在 `third_party/LMCache` submodule。

| 层级 | 入口 | 是否需要 GPU | 是否需要 vLLM serving |
| --- | --- | --- | --- |
| Layer 1 | 自临时目录对 `$REPO_ROOT/tests` 跑 `pytest -q` | 是（环境需预装 GPU torch） | 否 |
| Layer 2 | `examples/run_all_lmcache_iluvatar_tests.py` | 是 | 是 |

更细的 example 参数与 manifest 维护方式见 `examples/README.md`。

---

## Jenkins 流水线

| 脚本 | 用途 |
| --- | --- |
| `ci/lmcache_test_pr.groovy` | PR：format → `clean_build_install`（含 `install_dependencies`）→ pytest → 集成测 |
| `ci/lmcache_test.groovy` | Framework/SDK：`install_dependencies.sh`（requirements + submodule LMCache，不覆盖插件 wheel）→ pytest → 集成测 |
| `ci/oneclick.groovy` | 触发下游 `T_func_vLLM` one-click job |

`install` 阶段的 `clean_build_install.sh` 会调用 `install_dependencies.sh` 安装运行时依赖与上游 LMCache。

---

## 仓库检出（本地与 CI 共用）

```bash
git clone --recurse-submodules ssh://git@bitbucket.iluvatar.ai:7999/swapp/lmcache-iluvatar.git
# or: git submodule update --init --recursive third_party/LMCache
```

---

## 构建与安装

CI 与发布环境使用 wheel 安装，不采用 editable 模式。统一入口：

```bash
bash scripts/clean_build_install.sh
```

该脚本依次：清理旧产物 → 调用 `install_dependencies.sh`（`requirements.txt` + 从
`third_party/LMCache` 以 `NO_NATIVE_EXT=1 --no-deps` 安装上游 LMCache）→
`pip wheel --no-deps` 编译插件 wheel → `pip install --no-deps` 安装插件 wheel。

仅安装运行时依赖（不打插件 wheel）：

```bash
bash scripts/install_dependencies.sh
```

仅编译产出 wheel（不安装）：

```bash
bash scripts/build_wheel.sh
```

LMCache 运行时版本见 `upstream_pin.json` 的 `lmcache.pypi_version`；源码来自
`third_party/LMCache` submodule

本地开发若需改代码即生效，见根目录 `README.md` 中的 **Development** 小节（`pip install -e .`，仅开发人员使用）。

---

## Layer 1：单元测试

### 范围

- **`pytest`**：运行 `tests/` 下全部单元测试（patch、packaging、docs、`v1/` 存储与 op 等），不启动 vLLM、不占用 GPU serving 端口。

### 前置条件

- **PR / 本地全量**：`bash scripts/clean_build_install.sh`
- **Framework job**：SDK 已安装 `lmcache_iluvatar`；job 内 `bash scripts/install_dependencies.sh`（补 pip 依赖）
- Python 3、pip；GPU 版 torch 由运行环境提供

### 运行

PR 或本地与 CI 对齐：

```bash
bash scripts/clean_build_install.sh
REPO_ROOT="$(pwd)"
tmpdir="$(mktemp -d)"
cd "$tmpdir"
python3 -m pytest -q --import-mode=append "$REPO_ROOT/tests"
```

Framework job（`ci/lmcache_test.groovy`）等价 shell：

```bash
bash scripts/install_dependencies.sh
REPO_ROOT="$(pwd)"
tmpdir="$(mktemp -d)"
cd "$tmpdir"
python3 -m pytest -q --import-mode=append "$REPO_ROOT/tests"
```

---

## Layer 2：`examples/run_all_lmcache_iluvatar_tests.py`

### 范围

薄编排器：读取 `examples/ci/manifest.json`，按顺序调用各 example 的 **`ci/run.sh`**。每个 `ci/run.sh` 负责完整生命周期（起服务 → 发请求/跑脚本 → 日志断言 → `ci_cleanup` 释放端口与进程）。

当前注册的 case（`--cases all` 时按 GPU 数量自动 skip 不足的项）：

| Case | 级别 | 目录 | GPU | 验证内容 |
| --- | --- | --- | --- | --- |
| `p2p_sharing` | L0 | `p2p_sharing/` | 2 | Controller + 双 vLLM P2P 复用；`test_cache_reuse.sh` + Stored/Retrieved 日志 |
| `disagg_prefill_mp_smoke` | L0 | `disagg_prefill_mp/smoke/` | 4 | Shared-storage MP 2P2D smoke；`smoke_long_prompt_cache.py` + server 日志（Stored、prefix hits） |
| `disagg_prefill_offload_local_tiered` | L1 | `disagg_prefill/offload/` | 4 | 1P1D local-tiered LMCache offload（TP2+TP2，无 Mooncake）；proxy/completions smoke + PD 日志（PDBackend） |
| `vllm_agg_offload` | L1 | `vllm_agg_offload/smoke/` | 2 | 单实例 agg + MP Server external hit（TP=2）；断言 MP `Stored` + lookup-hit metrics |
| `cacheblend` | L2 | `cacheblend/` | 1 | CacheBlend shuffle-doc QA；通过 Iluvatar CacheBlend attention adapter 支持 `IluFlashAttentionImpl` |

### 前置条件

- 已按 Layer 1 对应路径安装插件与依赖。
- `vllm`、`lmcache_controller` 在 PATH；NIXL（CoreX 运行时）/UCX 环境见 `examples/p2p_sharing/README.md`（部分 case 需要）。
- 模型权重：Jenkins 集成测在 job 内下载 Qwen3-8B 到仓库 `weights/`（替代
  `/data/nlp/Qwen3-8B/`）；本地可继续用已有路径或按下方命令自行下载。
- `--gpus` 提供的 GPU 数量须满足所选 case（显式 `--cases <name>` 时不足会直接失败；`--cases all` 时不足则 `[skipped]`）。

### 运行

```bash
bash scripts/clean_build_install.sh

cd examples
python3 run_all_lmcache_iluvatar_tests.py \
  --base-model-path weights/Qwen3-8B \
  --gpus 0,1,2,3 \
  --log-dir ../runtime_result \
  --keep-going
```

常用参数：

| 参数 | 含义 |
| --- | --- |
| `--cases` | `all` 或逗号分隔 case 名 |
| `--base-model-path` | 模型权重目录（多数 case 必填） |
| `--served-model-name` | 覆盖 vLLM served name |
| `--gpus` | 逗号分隔 GPU id，如 `0,1,2,3` |
| `--log-dir` | 各 case 日志根目录（默认 `examples/logs/<timestamp>`） |
| `--result-json` | 写出每用例 `name`/`level`/`pass` 的 JSON（默认 `<log-dir>/results.json`）；`pass` 为 `true` / `false` / `"skipped"` |
| `--keep-going` | 某个 case 失败后继续跑后续 case |
| `--startup-timeout` / `--request-timeout` / `--cleanup-timeout` | 传给各 `ci/run.sh` 的超时（秒） |

只跑单个 case 示例：

```bash
python3 run_all_lmcache_iluvatar_tests.py \
  --cases disagg_prefill_offload_local_tiered \
  --base-model-path weights/Qwen3-8B \
  --gpus 0,1,2,3 \
  --log-dir ../runtime_result
```

编排器向每个 `ci/run.sh` 注入 `LMCACHE_CI_*` 环境变量；共享 shell 辅助函数在 `examples/ci/lib.sh`。
