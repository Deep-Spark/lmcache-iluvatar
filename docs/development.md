# 开发指南

`lmcache-iluvatar` 是面向 Iluvatar 的 LMCache **无侵入插件**：独立 wheel 安装，import 时打运行时 patch，不修改上游 `lmcache` 源码。当前覆盖 **vLLM V1 dynamic connector** 与 **V1 MP connector** 入口。

兼容矩阵见 [`docs/compatibility.md`](compatibility.md)。

## 架构与入口

vLLM 通过 `kv_connector_module_path` 加载：

```text
lmcache_iluvatar.integration.vllm.lmcache_iluvatar_connector_v1
```

导入顺序（幂等）：

1. `import lmcache_iluvatar` → `ensure_lmcache_install()` 校验上游 LMCache / vLLM 接口 → `activate_patches(strict=True)`
2. 将 `lmcache.c_ops` 重定向到 `lmcache_iluvatar.c_ops`
3. connector 模块再次调用 `activate_patches()` 后委托上游 `LMCacheConnectorV1Dynamic`

模块同时导出 `LMCacheIluvatarConnectorV1Dynamic` 与 `LMCacheConnectorV1Dynamic`，以兼容不同 vLLM 版本的类名查找。

## Patch 注册表

所有 monkey patch 集中在 `lmcache_iluvatar/integration/patch/runtime.py`。

- `PatchSpec`：目标模块/属性、版本范围、`required`、可选 `patch_cached_refs`
- `replacement_factory(current)`：读取上游对象后生成 wrapper，默认配置仍委托上游
- `activate_patches()` 带锁且幂等；已 patch 对象标记 `__lmcache_iluvatar_patch__`
- `.pth` hook 只报告已完成加载的 vLLM 模块；
  `activate_post_import_patches()` 通过同一注册表重试并更新诊断状态

当前 patch 目标：

| 类别 | 上游符号 |
| --- | --- |
| Native | `lmcache.c_ops`、`lmcache.native_storage_ops`、`lmcache.lmcache_redis`、`lmcache.lmcache_fs`、可选 `lmcache.lmcache_mooncake` → `lmcache_iluvatar.*` |
| MP IPC event guard | `multiprocess.server.torch_dev`、`modules.lmcache_driven_transfer/blend/blend_v3.torch_dev`、`lmcache_mp_connector.torch_dev`（均 `required=False`）；另 patch `DefaultEventIPCBackend.create_event`（v0.5.3 lmcache_driven 完成 Event） |
| 工厂 / GPU connector | `CreateGPUConnector`、`VLLMPagedMemGPUConnectorV2`、`CreateStorageBackends`、`CreateTransferChannel`（及 cached refs） |
| 引擎 / 存储 | `LMCacheEngine`、`StorageManager`、`PDBackend`、`PDBackendAsync` |
| Hybrid KV group edits | `kv_cache_group_edits._EDITS`、`apply_kv_cache_group_edits`（pinned `lmcache==0.5.3` 的 required targets；显式 `layout_hints`；存在 `lmcache_mp_connector` cached ref 时同步更新） |
| Async lookup | `EventManager`、`LMCacheAsyncLookupClient` |
| vLLM | `LMCacheConnectorV1Impl`（及 connector 模块 cached refs） |
| KV layout | LMCache `LMCacheConnectorV1Dynamic`、vLLM `LMCacheMPConnector` → 非 MLA 声明 Iluvatar HND 并拒绝显式 NHD；MLA 交回 vLLM 默认处理 |
| EIC remote | `EICConnector`（`required=False`，无 `eic` 依赖时 skip） |

**不 patch**：vLLM device helper、LMCache transfer device helper（当前走 CUDA 兼容 PyTorch 路径）。

## 默认行为与占位实现

默认配置一律委托上游 LMCache；仅显式 Iluvatar 选项进入占位实现并抛出 `UnsupportedIluvatarFeatureError`：

- **GPU connector**（`lmcache_iluvatar/v1/gpu_connector/`）：显式开关才用 `VLLMIluvatarGPUConnector`
- **Storage**（`v1/storage_backend/`）：不解析 vLLM request，数据经 engine / `kwargs` 边界传递
- **Transfer channel**（`v1/transfer_channel/`）：显式 `iluvatar*` channel 名才走占位
- **c_ops**：v0.5.3 ABI；未编译 native 时占位函数显式报错
- **ops.py**：预留语义层封装，尚未实现

已落地的兼容性修复（相对 LMCache `v0.5.3` pin）包括 native redirect、V2 H2D staging、PD reused-key cleanup、async loading cleanup（`enable_async_loading`）、PP cache broadcast（`save_only_first_rank` + `broadcast_src_rank` + uint8 pack）、MP IPC event guard（含 `DefaultEventIPCBackend.create_event` FIFO）、EIC connector namespace/busy_loop/partial-failure修复、protocol `torch.int8` dtype 映射，以及 Qwen3.5 hybrid 注册路径上的 Iluvatar HND subpaged group-edit。

## Native 构建

- 源码：`csrc/**` 来自 LMCache `v0.5.3`；`setup_upstream.py` 仅作参考，勿用于本包构建
- 构建入口：仓库根目录 `setup.py` → `lmcache_iluvatar.c_ops`
- 细节：`csrc/UPSTREAM.md`、`csrc/ILUVATAR_PATCHES.md`

安装示例（CoreX 环境需 `--no-build-isolation`，见 README）：

```bash
bash scripts/install_dependencies.sh
pip install -e . --no-build-isolation
```

## 扩展插件

### 示例：为 `EventManager` 补 `pop_event_any_status`

> **背景**：pinned LMCache `v0.5.3` 的 `EventManager` 只有 `pop_event`（仅 DONE）。`enable_async_loading` 清理需要从 **ONGOING** 状态 pop event 并 cancel。插件在 baseline 上补 `pop_event_any_status`；完整实现见 [`lmcache_iluvatar/v1/event_manager.py`](../lmcache_iluvatar/v1/event_manager.py)。

#### 第 1 步：确认 patch 目标

```bash
python3 -c "
from lmcache.v1.event_manager import EventManager
print(hasattr(EventManager, 'pop_event_any_status'))
"

# 搜索引用方，判断是否需要 patch_cached_refs
rg 'from lmcache\.v1\.event_manager import|EventManager' /path/to/lmcache/lmcache
```

本例只需 patch `lmcache.v1.event_manager.EventManager`；`LMCacheEngine` 内 `EventManager()` 会使用已 patch 的类。

#### 第 2 步：写 builder

文件：[`lmcache_iluvatar/v1/event_manager.py`](../lmcache_iluvatar/v1/event_manager.py)

```python
def build_iluvatar_event_manager(base_cls):
    if getattr(base_cls, "__lmcache_iluvatar_event_manager__", False):
        return base_cls
    if hasattr(base_cls, "pop_event_any_status"):  # 上游已有则跳过
        setattr(base_cls, "__lmcache_iluvatar_event_manager__", True)
        return base_cls

    def pop_event_any_status(self, event_type, event_id):
        ...  # 与上游 341a64d 逻辑一致

    base_cls.pop_event_any_status = pop_event_any_status
    setattr(base_cls, "__lmcache_iluvatar_event_manager__", True)
    return base_cls
```

要点：

- 用 `__lmcache_iluvatar_*__` 标记幂等
- 上游已有同名 API 时直接返回，避免重复 patch

#### 第 3 步：注册到 patch 注册表

文件：[`lmcache_iluvatar/integration/patch/runtime.py`](../lmcache_iluvatar/integration/patch/runtime.py)

```python
@replacement_factory
def _build_event_manager(current):
    from lmcache_iluvatar.v1.event_manager import build_iluvatar_event_manager
    return build_iluvatar_event_manager(current)

# 在 _build_patch_specs() 的 return 元组中追加：
PatchSpec(
    "lmcache.v1.event_manager",
    "EventManager",
    _build_event_manager,
    "lmcache.v1.event_manager.EventManager",
    version_range=VersionRange("lmcache"),
),
```

#### 第 4 步：补测试

| 文件 | 作用 |
| --- | --- |
| `tests/fake_lmcache.py` | fake 模块提供最小 `EventManager` |
| `tests/test_runtime_patch.py` | 断言 patch 后存在 `pop_event_any_status` |
| `tests/v1/test_async_loading_cleanup.py` | 断言 ONGOING pop 等行为 |

```bash
python3 -m pytest tests/test_runtime_patch.py tests/v1/test_async_loading_cleanup.py -q
```

#### 第 5 步：确认运行时生效

```bash
python3 -c "
import lmcache_iluvatar
from lmcache.v1.event_manager import EventManager
print(hasattr(EventManager, 'pop_event_any_status'))
"
```

#### 补充：何时用 subclass override 实例方法

上文 `EventManager` 示例是 **往上游类上挂新方法**（`base_cls.pop_event_any_status = ...`），适合「补缺失 API、不改现有方法体」。

若要 **改掉已有实例方法的行为**（例如 `LMCacheEngine.cleanup_memory_objs`、`retrieve`、`lookup_unpin`），应使用 **subclass + 替换类符号**：

| 方式 | 适用场景 | 本仓库示例 |
| --- | --- | --- |
| 往 `base_cls` 挂方法/属性 | 上游缺 API，实例逻辑不变 | `build_iluvatar_event_manager` |
| `class X(base_cls)` + override | 要包装/替换已有方法，且需 `super()` 调原实现 | `build_iluvatar_cache_engine` |

**运行时发生了什么**

```text
activate_patches()
  → 读取 lmcache.v1.cache_engine.LMCacheEngine（上游类）
  → build_iluvatar_cache_engine(上游类) 得到 LMCacheIluvatarEngine
  → 把模块里的 LMCacheEngine 符号替换成 LMCacheIluvatarEngine
  → vLLM / LMCacheManager 后续 LMCacheEngine() 实际构造的是子类实例
```

因此 patch 的是 **类对象**，不是某一个已创建好的 engine 实例；必须在 manager 创建 engine **之前** 完成 `import lmcache_iluvatar`。

**`build_iluvatar_xxx(base_cls)` 模板**

文件：[`lmcache_iluvatar/v1/cache_engine.py`](../lmcache_iluvatar/v1/cache_engine.py)

```python
def build_iluvatar_cache_engine(base_cls):
    if getattr(base_cls, "__lmcache_iluvatar_cache_engine__", False):
        return base_cls  # 已 patch 过，避免嵌套子类

    class LMCacheIluvatarEngine(base_cls):
        __lmcache_iluvatar_cache_engine__ = True

        def cleanup_memory_objs(self, lookup_id: str) -> None:
            # 1. 插件侧新逻辑（如 pop_event_any_status + 释放 MemoryObj）
            ...
            # 2. 若仍需上游其余行为，可显式 super()；本方法若完全重写则可不调用

        def retrieve(self, tokens, mask=None, **kwargs):
            cleanup_id = self._iluvatar_cleanup_req_id(kwargs.get("req_id"))
            ret = super().retrieve(tokens, mask, **kwargs)  # 委托上游
            ...  # retrieve 结束后再 release_pd_reuse 等
            return ret

    return LMCacheIluvatarEngine
```

要点：

- **`base_cls` 是 patch 当下模块里的上游类**（`replacement_factory(current)` 传入），子类继承它才能保留未 override 的方法。
- **优先 `super().方法(...)`**：只改差异部分，降低与上游 v0.5.3 漂移的风险。
- **幂等标记** `__lmcache_iluvatar_*__`：防止对已经是子类的类再包一层。

**注册方式**（与 EventManager 相同入口，替换的是 class 而非方法）：

```python
PatchSpec(
    "lmcache.v1.cache_engine",
    "LMCacheEngine",
    _build_cache_engine,
    "lmcache.v1.cache_engine.LMCacheEngine",
    patch_cached_refs=(
        ("lmcache.integration.vllm.vllm_v1_adapter", "LMCacheEngine"),
        ("lmcache.v1.manager", "LMCacheEngine"),
    ),
),
```

`patch_cached_refs` 必须补全：若某模块在 patch 前已 `from lmcache.v1.cache_engine import LMCacheEngine`，只改 canonical 模块会导致那里仍指向**未 patch 的上游类**，实例不会走子类逻辑。

**与 `cleanup_memory_objs` 的依赖关系**：该方法在子类里会调用 `self.event_manager.pop_event_any_status`，因此通常 **先** patch `EventManager`（补 API），**再** patch `LMCacheEngine`（override 清理流程）。`StorageManager` 若也要改 prefetch 回调，同样用 subclass，见 [`storage_manager.py`](../lmcache_iluvatar/v1/storage_backend/storage_manager.py)。

### 新增 patch（通用 checklist）

1. 优先用上游扩展点；必要时在 `_build_patch_specs()` 增加 `PatchSpec`
2. 需要原对象时用 `@replacement_factory`，并搜索上游 **cached import** 补全 `patch_cached_refs`
3. 补单测：`tests/test_runtime_patch.py` + 行为测试；跨版本组合更新 `docs/compatibility.md`

### 实现数据面（GPU / Storage / Transfer / ops）

- 保持 factory 默认委托；仅显式 Iluvatar 配置进入新实现
- Storage 层不解析 vLLM request；先定义 memory object 生命周期与 cleanup 规则
- GPU 传输遵守 `slot_mapping` 契约，不根据 token offset 推断 block
- 高层逻辑放 `v1/ops.py`，底层 ABI 仍走 `lmcache_iluvatar.c_ops`

跨层语义变更时同步 `.trellis/spec/backend/` 相关契约。

## 测试

开发中：

```bash
python3 -m pytest tests/test_runtime_patch.py tests/v1/test_pd_reuse_cleanup.py
python3 -m pytest tests/v1/test_async_loading_cleanup.py tests/v1/test_storage_manager_async_cleanup.py
python3 -m pytest tests/v1/test_gpu_connector.py tests/v1/test_storage_transfer.py
```

提交前（与 CI 一致：先装依赖与 wheel，再跑全量 pytest）：

```bash
bash scripts/clean_build_install.sh
python3 -m pytest -q
```

GPU 集成（Jenkins `T_unit_lmcache_iluvatar` 同款，需 2 卡 + vLLM）：

```bash
bash scripts/clean_build_install.sh
cd examples/
python3 run_all_lmcache_iluvatar_tests.py --gpus 0,1 --base-model-path <model> --keep-going
```

布局说明见 [`ci/README.md`](../ci/README.md) 与 [`examples/README.md`](../examples/README.md)。

诊断 patch 状态：

```bash
python3 -c "import lmcache_iluvatar; print(lmcache_iluvatar.get_patch_state().results)"
```
