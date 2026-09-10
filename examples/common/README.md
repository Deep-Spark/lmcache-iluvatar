# Shared example benchmarks

Reusable benchmark clients and shell helpers for example wrappers.

| File | Role |
|------|------|
| `bench_serving.py` | Mooncake trace / random workload client (vLLM OpenAI API) |
| `long_doc_qa.py` | Synthetic long-document QA warmup + query benchmark |
| `bench_lib.sh` | Shell helpers: readiness checks, trace download, TTFT summary |
| `bench_summarize_ttft.py` | Print TTFT table from `bench_serving` JSONL results |
| `smoke_long_prompt_cache.py` | Long-prefix MP cache smoke via disagg proxy (request + metrics) |

Example wrappers under `disagg_prefill/`, `disagg_prefill_mp/`, and `vllm_agg_offload/`
source `bench_lib.sh` and pass deployment-specific flags (e.g. `--pd-disagg-ttft` for PD
disagg proxies; aggregated `vllm_agg_offload/smoke` LongQA hits vLLM directly and does not use
that flag). Proxy and stack startup scripts stay in each example directory.
