# Disaggregated Prefill — LMCache MP Examples

本目录包含多个独立用例，覆盖 **2P2D 标准 MP 路径**、**双 server MP P2P**、**1P1D / 2P2D MP+NIXL** 与 **纯 NixlPush 对照**。各用例自带 `env.sh` 与 `start_*.sh`；共享端口默认值在 [`env.defaults.sh`](env.defaults.sh)。

## Proxy 文件

| 文件 | 用途 |
|------|------|
| [`disagg_proxy_server.py`](disagg_proxy_server.py) | 纯 MP（telemetry 等 `store_finished`）；对齐上游 LMCache 同名文件。`smoke/` / `bench-shared-system/` / `mp-p2p-2p2d/` 使用 |
| [`disagg_proxy_server_nixl_push.py`](disagg_proxy_server_nixl_push.py) | 在 MP 骨架上注入 NixlPush `kv_transfer_params`（可选跳过 telemetry）。`nixl-push-1p1d/` 与 `p-cache-nixl-transfer-*` 使用。D 侧 params 形状对齐 vLLM `disagg_proxy_pushconnector_demo.py`；CSV 多实例 / 嵌套 RR / MP wait 为本地扩展 |

## 用例一览

| 目录 | 拓扑 | 测什么 | CI |
|------|------|--------|-----|
| [`smoke/`](smoke/) | 2P2D，P/D 均走 `LMCacheMPConnector` | 栈能否跑通；长 prefix 重复请求下 MP L1 是否命中 | ✅ |
| [`bench-shared-system/`](bench-shared-system/) | 2P2D，同上 | 多用户共享 system prompt 并发时，跨 P round-robin 的 L1 复用与 TTFT | ❌ 手动 |
| [`mp-p2p-2p2d/`](mp-p2p-2p2d/) | 2P2D，双 `lmcache server` + coordinator（MP P2P） | 跨 pair peer L1 RDMA read 下的 longQA TTFT（非单 server 共享、非 NIXL） | ❌ 手动 |
| [`p-cache-nixl-transfer-1p1d/`](p-cache-nixl-transfer-1p1d/) | 1P1D，P 侧 MP 复用 + NIXL P/D 传输 | MP 只做 prefix 缓存，实时 P→D KV 走 NIXL（D 不连 MP） | ❌ POC |
| [`p-cache-nixl-transfer-2p2d/`](p-cache-nixl-transfer-2p2d/) | 2P2D，同上 + 独立 RR | 多 P/D 下 per-instance NIXL 元数据注入；MP 命中 + 跨实例传输 | ❌ POC |
| [`nixl-push-1p1d/`](nixl-push-1p1d/) | 1P1D，仅 `NixlPushConnector` | 隔离 vLLM NIXL 传输延迟（无 MP） | ❌ 对照 |

## 怎么选

- **回归 / 冒烟** → `smoke/`（Qwen3-8B，4 GPU）
- **2P2D 多用户缓存性能** → `bench-shared-system/`（Qwen3-32B-W8A8，8 GPU）
- **双 server MP P2P（跨 pair peer read）TTFT** → `mp-p2p-2p2d/`
- **验证 NIXL P/D 传输 + P 侧 MP 复用（1P1D）** → `p-cache-nixl-transfer-1p1d/`
- **同上扩到 2P2D 独立 RR** → `p-cache-nixl-transfer-2p2d/`
- **对照纯 NixlPush vs PDBackend** → `nixl-push-1p1d/`

各用例的启动命令、profile 与判读标准见子目录 README。
