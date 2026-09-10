# CacheBlend Shuffle-Doc QA 示例

这个目录用于验证 LMCache CacheBlend 在 Iluvatar vLLM 镜像里的集成效果。
当前 v0.5.0 validation 通过 Iluvatar CacheBlend attention adapter 支持
Iluvatar vLLM 的 `IluFlashAttentionImpl`。
示例分成两个脚本：

- `start-server.sh`：只负责启动 vLLM server。
- `run_benchmark.sh`：只负责向已启动的 server 发送 warmup/query 请求。

## 运行依赖

CacheBlend 依赖一些上游 vLLM / attention 模块。它们应该由镜像环境提供，
不应该由 `lmcache-iluvatar` 在运行时伪造 fake module。

需要确认：

```bash
python3 - <<'PY'
import flashinfer
from vllm.attention import Attention
from vllm.attention.backends.abstract import AttentionMetadata
from vllm.vllm_flash_attn import flash_attn_varlen_func, get_scheduler_metadata

print("flashinfer:", getattr(flashinfer, "__version__", "unknown"))
print("Attention:", Attention)
print("AttentionMetadata:", AttentionMetadata)
print("flash_attn_varlen_func:", callable(flash_attn_varlen_func))
print("get_scheduler_metadata:", callable(get_scheduler_metadata))
PY
```

依赖归属：

- `vllm.vllm_flash_attn`：上游 LMCache 旧 import 路径；Iluvatar 镜像构建阶段会补兼容入口，转发到 ixformer / vLLM `fa_utils` 中的真实 FlashAttention provider。
- `flashinfer`：由 `vllm-image` 构建阶段安装真实 Iluvatar/FlashInfer 包。
- `vllm.attention`：上游 LMCache 旧 import 路径；vLLM 0.17 中真实符号在新路径下，镜像构建阶段会补兼容入口。
- `lmcache-iluvatar`：只负责 LMCache runtime patch、connector、native redirect、
  CacheBlend defaults 和 vLLM Worker model tracker，不再 fake 上游模块。

## 启动 Server

在包含 vLLM、LMCache、`lmcache-iluvatar` 和上述 CacheBlend 依赖的环境中运行：

```bash
MODEL_PATH=/data/models/Qwen3-8B \
SERVED_MODEL_NAME=Qwen3-8B \
GPU_DEVICE=0 \
MAX_MODEL_LEN=2048 \
GPU_MEM_UTIL=0.70 \
BENCH_PORT=8000 \
bash start-server.sh
```

如果在 Docker 里运行，调用方需要负责暴露 GPU 设备、模型目录和网络。

重要环境变量：

- `MODEL_PATH`：vLLM 加载的本地模型权重路径。
- `SERVED_MODEL_NAME`：OpenAI API 暴露的模型名，默认取 `MODEL_PATH` basename。
- `LMCACHE_CONFIG_FILE`：LMCache 配置文件，默认是本目录的 `lmcache_blend.yaml`。
- `BENCH_PORT`：server 端口。
- `GPU_DEVICE`：写入 `CUDA_VISIBLE_DEVICES`。

## 运行 Benchmark

server 启动成功后，在另一个终端运行：

```bash
OUT_DIR=/tmp/cacheblend-run \
SERVED_MODEL_NAME=Qwen3-8B \
TOKENIZER_PATH=/data/models/Qwen3-8B \
BENCH_PORT=8000 \
NUM_DOCUMENTS=4 \
DOCUMENT_LENGTH=128 \
NUM_REQUESTS=2 \
OUTPUT_LEN=1 \
bash run_benchmark.sh
```

`SERVED_MODEL_NAME` 是 API model name，必须和 server 的 `--served-model-name`
一致。`TOKENIZER_PATH` 是 `transformers.AutoTokenizer.from_pretrained()` 使用的
本地 tokenizer 路径；当 API model name 不是本地路径时必须显式设置。

## 输出日志

`OUT_DIR` 下会生成：

- `responses.log`：每个 warmup/query 请求、响应内容、TTFT 和最终汇总。
- `client.log`：benchmark 客户端 stdout/stderr。
- `lmcache_stats.txt`：提取出的 derangement 和 TTFT 行。

server 日志在启动 `start-server.sh` 的终端里。

## 如何判断 CacheBlend 生效

这个 benchmark 会先 warm up 每个 document segment，再用打乱顺序的 document
组合发 query。CacheBlend 生效时，query 阶段即使文档顺序变化，也可以复用已
缓存 segment 的 KV，并通过局部重算修正上下文差异。

重点看：

- `responses.log` 里 query round 的 `TTFT`。
- server 日志中的 LMCache retrieve / blending 相关输出。
- `lmcache_stats.txt` 中 derangement 请求和 TTFT 摘要。

