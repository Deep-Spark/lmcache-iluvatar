# Aggregated MP — GDS L1

单实例 `kv_both` + 独立 `lmcache server --gds-l1-path`，验证 **GDS L1 slab**（cuFile DMA）上的 store / lookup-hit。

```text
[lmcache server :6555 ZMQ + :8080 metrics]
   L1 = ${GDS_L1_PATH}/lmcache_gds_slab.bin
        ↑ ZMQ / CUDA IPC
[vLLM TP=2 kv_both LMCacheMPConnector :18010]
```

## Profile（`env.sh`）

| 项 | 默认值 |
| ---- | ------ |
| 模型 | `/data/nlp/Qwen3-8B` |
| `SERVED_MODEL_NAME` | `Qwen3-8B` |
| GPU | `0,1`（`TENSOR_PARALLEL=2`） |
| `MAX_MODEL_LEN` | `40960` |
| HTTP (vLLM) | `127.0.0.1:18010` |
| `LMCACHE_MP_PORT` | `6555` |
| `LMCACHE_HTTP_PORT` | `8080`（`/metrics`） |
| `LMCACHE_L1_SIZE_GB` | `100`（slab 容量） |
| **`GDS_L1_PATH`** | **无默认；未设置则退出码 2** |
| `GDS_L1_USE_DIRECT_IO` | `1`（`--gds-l1-use-direct-io`） |
| `PYTHONHASHSEED` | `0` |
| `VLLM_KV_CACHE_LAYOUT` | `HND` |

`start_server.sh` **不**设置 `CUDA_VISIBLE_DEVICES`，启动 vLLM 前自行 export。

## 前置条件

- 目标机已装可用的 cuFile / GDS（`cuFileReadAsync` / `cuFileWriteAsync` 等）
- `GDS_L1_PATH` 指向 **GDS-capable** 文件系统目录（建议 NVMe 上的 ext4/xfs）；可用 `gdscheck -p` 自检
- 目录有足够空间（默认 slab ≈ `LMCACHE_L1_SIZE_GB` GiB）
- `lmcache`、`vllm`、`lmcache-iluvatar` 已安装；至少 2 块空闲 GPU
- **不要**同时开 P2P 或依赖 L1 DRAM 的 L2（如 `nixl_store`）

## 启动顺序

**必须先启动 MP Server，再启动 vLLM。** 建议把 MP 日志落到文件，供路径举证。

```bash
cd examples/vllm_agg_offload/gds/l1
export GDS_L1_PATH=/mnt/nvme/lmcache-gds-l1   # 必填，按目标机修改
mkdir -p logs

# Terminal 1 — MP server
bash start_lmcache_mp.sh 2>&1 | tee logs/lmcache.log

# Terminal 2 — vLLM (after MP is up)
CUDA_VISIBLE_DEVICES=0,1 bash start_server.sh 2>&1 | tee logs/vllm.log
```

健康检查：

```bash
curl -sf --noproxy '*' http://127.0.0.1:8080/metrics >/dev/null && echo "mp metrics ok"
curl -sf http://127.0.0.1:18010/health && echo "vllm ok"
ls -lh "${GDS_L1_PATH}/lmcache_gds_slab.bin"
```

## 验证（路径举证 + external hit）

```bash
LMCACHE_SERVER_LOG="$(pwd)/logs/lmcache.log" \
  LMCACHE_URL=http://localhost:8080 \
  bash test_external_hit.sh
```

**通过标准：**

1. MP 日志含 `GDS L1 tier enabled`（含 CPU pinned-DRAM L1 disabled）
2. MP 日志含 `GDSContext: slab created`
3. `${GDS_L1_PATH}/lmcache_gds_slab.bin` 存在
4. COLD 后日志出现 `Stored`
5. WARM 后 `lmcache_mp_lookup_hit_tokens_total > 0`

未设置 `LMCACHE_SERVER_LOG` 或日志文件不存在时，脚本直接失败（路径举证为硬性要求）。

可选：

```bash
GDS_L1_PATH=... WARM=1 bash curl.sh
GDS_L1_PATH=... bash run-longQA-benchmark.sh --quick
```

## 清理

```bash
pkill -f "vllm.entrypoints.openai.api_server.*--port ${SERVER_PORT:-18010}" || true
pkill -f "lmcache server --port ${LMCACHE_MP_PORT:-6555}" || true
# 可选：删除 slab（重启 MP 本就会 truncate 重建）
# rm -f "${GDS_L1_PATH}/lmcache_gds_slab.bin"
```

本用例不挂 CI（依赖目标机 GDS 硬件 / 挂载）；人工在已配好 cuFile 的机器上跑。
