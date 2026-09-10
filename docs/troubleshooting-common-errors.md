# LMCache 常见错误与处理

`lmcache-iluvatar` 使用 LMCache 时的故障索引，按条目补充。

---

## 外部前缀缓存始终不命中

**问题**：相同 prompt 重复请求，`External prefix cache hit rate` 长期为 0；可能已有 store，但 lookup 仍像冷启动。

**建议处理**：设置 `PYTHONHASHSEED=0`，并**重启**全部推理进程（启动后改环境变量无效）。

---

## 前缀命中但生成乱码

**问题**：外部 cache 已命中，但再次请求的生成结果乱码或语义错误；多为 LMCache 回填 GPU 时 KV 物理排布（layout）与引擎实际不一致。

**建议处理**：设置 `VLLM_KV_CACHE_LAYOUT=HND`，并**重启**全部推理进程；同时保留 `PYTHONHASHSEED=0`。

---

## `need to load` 比命中 token 数少 1

**问题**：命中 token 数比 `need to load` 多 1，怀疑少加载。

**建议处理**：无需处理；整段 prompt 全命中时，引擎会重算最后一个 prompt token，属正常行为。

---

## 相关文档

- `[docs/compatibility.md](compatibility.md)` — 版本与依赖
- `[docs/development.md](development.md)` — 插件架构

