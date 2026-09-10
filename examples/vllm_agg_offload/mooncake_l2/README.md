# Aggregated vLLM + LMCache MP + Mooncake L2

两个 aggregated `kv_both` vLLM 实例分别连接本机 `lmcache server`，使用外部
Mooncake Store 作为共享 L2，验证 TCP/RDMA 下 KV Cache 的跨实例写入、空 L1
回读和 LongQA 请求成功率。

```text
Writer vLLM (TP=2)                    Reader vLLM (TP=2，首次收到 LongQA 文档)
     | LMCacheMPConnector :6585             | LMCacheMPConnector :6585
     | ZMQ + CUDA IPC                       | ZMQ + CUDA IPC
     v                                      v
Writer LMCache MP (requester)          Reader LMCache MP (requester)
     | metadata/control                     | metadata/control
     +-------------------+------------------+
                         v
              Mooncake Master :50051
              tenant=MOONCAKE_TENANT_ID

Writer MP / Reader MP -- TE TCP 或 RDMA Get/Put --> Store-0 + Store-1
                                                    shared DRAM L2 pool
```

1. Writer LongQA warmup 将 20 篇冷文档的 KV Cache 写入 Mooncake L2。
2. Reader 使用相同 tenant、模型、chunk/layout，在本地 L1 为空时运行同一批文档。
3. Reader 日志必须出现 `N/N retained keys (0 L1, N L2)`，且 `N > 0`。
4. Writer/Reader LongQA 各轮请求均须全部成功；只返回 HTTP 200 不算通过。

TCP/RDMA 只描述 **LMCache MP ↔ Mooncake Store** 的 L2 数据面。vLLM 与本机 MP
仍使用 `LMCacheMPConnector`。当前 LMCache P→D 传输通道使用 NIXL，Mooncake 在
本组合中只承担共享 L2 Store，因此本用例不增加 P/D 分离测试。

## Profile（`env.sh`）

| 项 | 默认值 |
| --- | --- |
| 模型 | `/data/nlp/Qwen3.5-27B-W8A8-fixed` |
| `SERVED_MODEL_NAME` | `Qwen3.5-27B-W8A8-fixed` |
| GPU | 需在 Writer/Reader 分别显式指定（`TENSOR_PARALLEL=2`） |
| `GPU_MEM_UTIL` / `MAX_MODEL_LEN` | `0.90` / `131072` |
| `MAX_NUM_BATCHED_TOKENS` | `1024` |
| HTTP（vLLM） | `0.0.0.0:18040` |
| `LMCACHE_MP_PORT` / `LMCACHE_HTTP_PORT` | `6585` / `8100` |
| `LMCACHE_CHUNK_SIZE` | `784` |
| `LMCACHE_L1_SIZE_GB` | `16`；不覆盖 LMCache 的 lazy 初始分配参数 |
| `MOONCAKE_PROTOCOL` | `tcp`，RDMA 测试时改为 `rdma` |
| `MOONCAKE_MASTER_ADDR` | `127.0.0.1:50051`，跨节点测试必须覆盖 |
| `MOONCAKE_METADATA_SERVER` | `P2PHANDSHAKE` |
| `MOONCAKE_STORE_SIZE` | `32GB`（每个 Store） |
| `MOONCAKE_TENANT_ID` | `lmcache-iluvatar-mooncake-l2-example` |
| `PYTHONHASHSEED` | `42` |
| `VLLM_KV_CACHE_LAYOUT` | `HND` |
| `VLLM_KV_DISABLE_CROSS_GROUP_SHARE` | `1` |
| LongQA | 20 篇 × 10000 tokens，`tile`，输出 2 tokens，并发 1 |

## 前置条件

### 1. vLLM、LMCache 与插件

- Writer/Reader 使用相同的 vLLM、LMCache、`lmcache-iluvatar`、模型文件和 TP。
- 本仓库固定 LMCache `0.5.3`；先按仓库根目录说明安装插件。
- 两侧必须保持相同的 `PYTHONHASHSEED`、`LMCACHE_CHUNK_SIZE` 和
  `VLLM_KV_CACHE_LAYOUT`，否则相同 prompt 也无法生成相同 Mooncake key。
- vLLM 与 LMCache MP 必须同节点部署。因为在独立 Pod 中，两者需设置
  `hostIPC: true` 并共享宿主机 `/dev/shm`，供 `LMCacheMPConnector` 通过 CUDA IPC
  传递 KV Cache。

### 2. Mooncake SDK 构建

Mooncake Master、Store 和每个 LMCache MP 运行环境都需要 Mooncake C++ SDK。

本用例基线为 LMCache `0.5.3`、Mooncake `v0.3.13`
（`e5598b0992cc258b06d22f24d875480e8931a28e`）。更换版本时必须重新验证 adapter
配置字段、ABI 和启动参数，不能只替换源码 tag。

```bash
cd /path/to/Mooncake
sudo bash dependencies.sh -y
sudo apt-get install -y ninja-build libibverbs-dev libnl-3-dev libnl-route-3-dev

cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=/opt/mooncake \
  -DBUILD_SHARED_LIBS=ON \
  -DBUILD_UNIT_TESTS=OFF \
  -DBUILD_EXAMPLES=OFF \
  -DBUILD_BENCHMARK=OFF \
  -DWITH_STORE_RUST=OFF \
  -DWITH_EP=OFF \
  -DUSE_COREX=ON \
  -DCOREX_ROOT=/path/to/corex \
  -DCOREX_INCLUDE_DIR=/path/to/corex/include \
  -DCOREX_LIB_DIR=/path/to/corex/lib64 \
  -DCMAKE_EXE_LINKER_FLAGS='-Wl,--no-as-needed -L/usr/lib/x86_64-linux-gnu' \
  -DCMAKE_SHARED_LINKER_FLAGS='-Wl,--no-as-needed -L/usr/lib/x86_64-linux-gnu' \
  -DCMAKE_CXX_STANDARD_LIBRARIES='-lnl-3 -lnl-route-3' \
  -DUSE_HTTP=ON \
  -DUSE_ETCD=OFF \
  -DSTORE_USE_ETCD=OFF \
  -DSTORE_USE_REDIS=OFF

cmake --build build --parallel 8
sudo cmake --install build
sudo ldconfig
```

RDMA 构建环境还需准备 `libibverbs`、`libnl-3`、`libnl-route-3` 及对应开发头文件。
如果链接阶段找不到 `nl-3` / `nl-route-3`，应先修复 SDK 的 CMake 链接参数。

### 3. 构建 `lmcache_mooncake` 扩展

Mooncake SDK 安装完成后，在 `lmcache-iluvatar` 根目录启用可选扩展：

```bash
export LMCACHE_ILUVATAR_ENABLE_MOONCAKE=1
export MOONCAKE_USE_PKG_CONFIG=0
export MOONCAKE_ENABLE_CXX11_ABI=1
export MOONCAKE_LIB_DIR=/opt/mooncake/lib
export LD_LIBRARY_PATH="/opt/mooncake/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

# 使用分号分隔。除公开头文件外，还要包含当前 Mooncake 构建生成的
# build/_deps/*-src/include、ylt/thirdparty 和 cachelib include。
export MOONCAKE_INCLUDE_DIR='<semicolon-separated Mooncake include directories>'

python3 -m pip install --force-reinstall --no-deps --no-build-isolation .
```

不要假设 yalantinglibs/cachelib 一定位于固定 `extern/` 目录；按所用 Mooncake
版本的 CMake `build/_deps` 实际路径收集 include。构建后执行：

```bash
MOONCAKE_PROTOCOL=tcp bash check_prerequisites.sh
```

必须同时满足：

- `mooncake_master`、`mooncake_client`、`lmcache` 在 `PATH` 中；
- `lmcache_iluvatar.lmcache_mooncake` 可导入；
- `lmcache.lmcache_mooncake` 已重定向到 Iluvatar 扩展；
- Mooncake MP L2 adapter 可导入；
- LongQA 所需的 `openai`、`pandas` 可导入；
- native `.so` 的 `ldd` 没有 `not found`。

### 4. TCP 网络

- 所有 Store 和 requester 都能访问 `MOONCAKE_MASTER_ADDR`。
- Store 的 `MOONCAKE_STORE_HOST:MOONCAKE_STORE_PORT` 必须可被其他节点访问。
- `MOONCAKE_STORE_HOST` / `MOONCAKE_REQUESTER_HOST` 应填写可路由地址，不能在
  跨节点场景使用 `127.0.0.1`。
- 防火墙和 NetworkPolicy 放行 Master、Store、vLLM API 以及环境实际需要的
  Transfer Engine 连接。

### 5. RDMA/RoCE 网络

RDMA 模式除满足 TCP 控制面要求外，还必须满足：

- Store 和 LMCache MP 所在节点具有处于 ACTIVE 状态的 RDMA HCA、有效 GID 和
  可互通的 RoCE 网络；先用环境标准的 RDMA 工具完成链路连通性测试。
- `MOONCAKE_RDMA_DEVICES` 使用容器内可见的 HCA 名称，例如 `mlx5_0`；所有
  Store/MP 都设置 `MOONCAKE_PROTOCOL=rdma`，不混用 TCP/RDMA。
- 容器能够访问 `/dev/infiniband/rdma_cm` 与相应 `uverbs*`，memlock 足够，进程
  具有 `IPC_LOCK`；部分环境还需要 `SYS_RESOURCE`。
- Kubernetes 推荐使用 RDMA device plugin 分配 HCA，并用 Multus 提供独立 RoCE
  网络。资源名和 NetworkAttachmentDefinition 名称由部署环境提供，例如：

```yaml
metadata:
  annotations:
    k8s.v1.cni.cncf.io/networks: >-
      [{"name":"<roce-network>","interface":"net1"}]
spec:
  containers:
    - name: mooncake
      resources:
        requests:
          <rdma-device-plugin-resource>: "1"
        limits:
          <rdma-device-plugin-resource>: "1"
      securityContext:
        capabilities:
          add: [IPC_LOCK, SYS_RESOURCE]
```

如果没有 device plugin，可由平台显式挂载 `/dev/infiniband`；不要同时采用两套
设备分配方案。`hostNetwork` 不是 RDMA 的必选项，只要 Multus `net1` 可路由即可。

## 手工测试公共配置

在每台机器上进入本目录并加载默认值：

```bash
cd examples/vllm_agg_offload/mooncake_l2
source env.sh

export MOONCAKE_MASTER_ADDR='<master-host>:50051'
export MOONCAKE_TENANT_ID='one-unique-test-tenant'
export MODEL_PATH='/path/to/model'
export SERVED_MODEL_NAME='model-name'
```

Writer 与 Reader 必须使用同一 tenant。Store 重启会丢失其 DRAM segment；切换
TCP/RDMA 时应停止测试流量，重启所有 Store 和 MP，并重新执行 cold 写入。

## TCP 测试

### 1. 启动 Master

```bash
MOONCAKE_MASTER_PORT=50051 bash start_mooncake_master.sh 2>&1 | tee master.log
```

### 2. 在一个或多个存储节点启动 Store

每个 Store 使用自己的可路由地址；不同机器可以复用相同端口：

```bash
MOONCAKE_PROTOCOL=tcp \
MOONCAKE_STORE_HOST='<this-store-host>' \
MOONCAKE_STORE_SIZE=32GB \
bash start_mooncake_store.sh 2>&1 | tee store.log
```

### 3. 在 Writer 和 Reader 分别启动 LMCache MP 与 vLLM

Writer：

```bash
MOONCAKE_PROTOCOL=tcp \
MOONCAKE_REQUESTER_HOST='<writer-host>' \
bash start_lmcache_mp.sh 2>&1 | tee writer-mp.log

CUDA_VISIBLE_DEVICES='<writer-gpus>' \
bash start_server.sh 2>&1 | tee writer-vllm.log
```

Reader 在另一台计算节点执行同样命令，只替换本机地址、GPU 和日志文件名。Reader
在正式测试前不能收到这批 LongQA 文档，否则无法证明本地 L1 为空。

### 4. 发送跨实例请求

```bash
MOONCAKE_PROTOCOL=tcp \
WRITER_BASE_URL='http://<writer-host>:18040' \
READER_BASE_URL='http://<reader-host>:18040' \
bash run-longQA-benchmark.sh
```

### 5. 在 Reader 检查 L2 证据

```bash
READER_MP_LOG=reader-mp.log \
READER_VLLM_LOG=reader-vllm.log \
bash verify_reader_logs.sh
```

## RDMA 测试

先停止 TCP Store/MP，保持 Master 或重新启动 Master；DRAM L2 清空后重新 cold 写入。
启动顺序与 TCP 相同，每个 Store 和 MP 增加：

```bash
export MOONCAKE_PROTOCOL=rdma
export MOONCAKE_RDMA_DEVICES='<rdma-device>'
bash check_prerequisites.sh
```

然后使用相同的 `start_mooncake_store.sh`、`start_lmcache_mp.sh`、
`start_server.sh` 和 `run-longQA-benchmark.sh`。RDMA 模式还必须检查：

- Store/MP 日志显示选择指定 HCA、有效 GID，并完成 RDMA ready handshake；
- 没有 `fallback to TCP`、`no usable GID` 或 transport 初始化失败；
- Reader 仍满足 `0 L1, N L2`，而不是只凭 RDMA 初始化日志判定通过。

## 通过标准

| 检查项 | 必须满足 |
| --- | --- |
| Writer cold store | Writer MP 日志出现正数 `Stored`，Master keys/容量上升 |
| Reader L2 lookup | `N/N retained keys (0 L1, N L2)`，`N > 0` |
| KV retrieve | Reader MP 日志出现正数 `Retrieved ... tokens` |
| vLLM 指标 | GPU `Prefix cache hit rate: 0.0%`，External prefix cache hit rate 大于 0 |
| LongQA | Writer/Reader 的 warmup 和 query 均全部成功 |
| RDMA 专项 | HCA/GID/ready handshake 成功，无 TCP fallback |

`run-longQA-benchmark.sh` 校验 LongQA CSV 中的请求成功率；必须再运行
`verify_reader_logs.sh`，测试才算闭环。

## 结果目录与参数覆盖

结果默认写入 `results/<RUN_ID>/{writer,reader}/`：

- `warmup_round.csv`：Writer 冷写或 Reader 跨实例 L2 回读；
- `query_round.csv`：同一实例的第二轮请求，不作为跨节点结果；
- `responses.txt`：LongQA 返回内容；
- `longqa.log`：客户端汇总与性能数据。

可用以下变量覆盖测试行为：

| 变量 | 作用 |
| --- | --- |
| `RUN_ID` | 结果目录名；省略时自动生成 |
| `RESULT_DIR` | 修改结果目录 |
| `LONGQA_DOCUMENT_LENGTH` / `LONGQA_NUM_DOCUMENTS` | 默认 `10000` / `20` |
| `LONGQA_OUTPUT_LEN` / `LONGQA_MAX_INFLIGHT` | 默认 `2` / `1` |
| `LONGQA_REPEAT_MODE` / `LONGQA_REPEAT_COUNT` | 默认 `tile` / `1` |
| `LONGQA_SLEEP_AFTER_WARMUP` | 每个实例 warmup 后等待时间，默认 `15` 秒 |
| `STORE_SETTLE_SECONDS` | Writer 完成后额外等待 L2 store，默认 `15` 秒 |

默认参数沿用已验证 LongQA workload：`document-length=10000`、
`num-documents=20`、`repeat-mode=tile`、`repeat-count=1`、`output-len=2`、
`max-inflight-requests=1`、`sleep-time-after-warmup=15` 和 `--completions`。
跨节点性能只比较 **Writer warmup（冷）** 与 **Reader warmup（L2）**；两侧 query
都是各自实例的第二轮请求。LongQA 文档内容固定，复跑前必须重启测试用 vLLM 和 MP
清空本地缓存，或者更换文档内容。

## 清理

各 `start_*.sh` 都以前台进程运行。按启动顺序的反向，在对应终端按 `Ctrl-C` 停止：

1. Writer/Reader vLLM；
2. Writer/Reader LMCache MP；
3. 所有 Mooncake Store；
4. Mooncake Master。

不要使用不带精确 PID 或端口范围的全局 `pkill`。Store 使用 DRAM 保存缓存，停止
Store 后本次 L2 数据即失效，无需执行数据恢复或额外清理。
