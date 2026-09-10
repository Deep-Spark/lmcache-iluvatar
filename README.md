# lmcache-iluvatar

`lmcache-iluvatar` is the Iluvatar plugin for running LMCache on CoreX with vLLM: native ops (`c_ops`, storage backends), import-time patches on pinned LMCache `0.5.3` (`upstream_pin.json`), and GPU examples for CPU/SSD offload, disaggregated prefill, P2P, and CacheBlend.

It is not a fork. Upstream LMCache stays in `third_party/LMCache` and is not edited. vLLM only (no SGLang / MindSpore). Layout, hybrid KV, and IPC fixes live in the patch registry (`docs/development.md`).

## Install

```bash
git clone --recurse-submodules ssh://git@bitbucket.iluvatar.ai:7999/swapp/lmcache-iluvatar.git
cd lmcache-iluvatar
# git submodule update --init --recursive third_party/LMCache
bash scripts/clean_build_install.sh
```

```bash
python -c "import lmcache_iluvatar; print(lmcache_iluvatar.get_patch_state().results)"
```

`lmcache` CLI loads this plugin via a `.pth` hook on first `import lmcache`.

## Examples

Runnable stacks live in `examples/`. Shared clients: `examples/common/`.


| Path                                                 | What                             | CI     |
| ---------------------------------------------------- | -------------------------------- | ------ |
| `examples/disagg_prefill_mp/smoke/`                  | MP 2P2D smoke                    | L0     |
| `examples/disagg_prefill_mp/bench-shared-system/`    | Multi-user shared system         | manual |
| `examples/disagg_prefill_mp/mp-p2p-2p2d/`            | Dual-server MP P2P               | manual |
| `examples/disagg_prefill_mp/p-cache-nixl-transfer-*` | P-side MP + NIXL P→D             | POC    |
| `examples/disagg_prefill_mp/nixl-push-1p1d/`         | NixlPush-only baseline           | manual |
| `examples/disagg_prefill/offload/`                   | PD dynamic + local-tiered 1P1D   | L1     |
| `examples/disagg_prefill/offload-2p2d/`              | Same, 2P2D                       | manual |
| `examples/vllm_agg_offload/smoke/`                   | Agg `kv_both` + MP server        | L1     |
| `examples/vllm_agg_offload/mooncake_l2/`             | Agg MP + Mooncake TCP/RDMA L2    | manual |
| `examples/vllm_agg_offload/dpsk-v4-cpu-ssd/`         | DeepSeek-V4 CPU L1 / SSD L2      | manual |
| `examples/p2p_sharing/`                              | Instance P2P (controller + NIXL) | L0     |
| `examples/cacheblend/`                               | CacheBlend                       | L2     |


```bash
cd examples
python3 run_all_lmcache_iluvatar_tests.py \
  --base-model-path /data/nlp/Qwen3-8B/ \
  --gpus 0,1,2,3 --keep-going --log-dir ../runtime_result
```

See `examples/README.md`.

## Tests and docs

- Unit: `pytest` on `tests/` (`ci/README.md`). Needs GPU torch, not serving.
- GPU integration: the runner above. Needs vLLM + a model.
- Jenkins: `ci/lmcache_test_pr.groovy`, `ci/lmcache_test.groovy`.


| Path                                    | Topic                                 |
| --------------------------------------- | ------------------------------------- |
| `docs/compatibility.md`                 | Pin, connectors, native redirects     |
| `docs/development.md`                   | Architecture and patch registry       |
| `docs/troubleshooting-common-errors.md` | Misses, garbage output, layout / hash |
| `csrc/ILUVATAR_PATCHES.md`              | Native diffs vs upstream              |

## License

This project is licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE).

Native sources under `csrc/` are a snapshot of [LMCache](https://github.com/LMCache/LMCache) v0.5.3 (Apache-2.0), with Iluvatar CoreX adaptations listed in `csrc/ILUVATAR_PATCHES.md`. Some examples and headers are adapted from [vLLM](https://github.com/vllm-project/vllm) and PyTorch; see the copyright lines in those files.
