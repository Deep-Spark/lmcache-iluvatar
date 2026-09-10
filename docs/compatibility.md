# Compatibility Matrix

`lmcache-iluvatar` is an independent plugin wheel. It should be validated as a
set with LMCache, vLLM, Iluvatar runtime, and PyTorch; do not modify upstream
LMCache requirements to install Iluvatar dependencies.

Developer extension steps and patch-flow details are documented in
`docs/development.md`.

## Upstream Pinning

Single source of truth: repository-root [`upstream_pin.json`](../upstream_pin.json).
`lmcache-iluvatar` patches exactly one upstream LMCache PyPI version; runtime
patch specs reject other installed versions.

| Component | Source |
|-----------|--------|
| LMCache Python | `third_party/LMCache` submodule @ tag matching `upstream_pin.json` → `lmcache.pypi_version` (currently `0.5.3`); installed by `scripts/install_dependencies.sh` with `NO_NATIVE_EXT=1 --no-deps` (do not `pip install lmcache` from PyPI — it pulls `cupy-cuda13x`) |
| UCX / NIXL | CoreX runtime / environment (`import nixl`); not vendored in this repo |
| vLLM | V1 connector API (dynamic + MP) |
| Iluvatar runtime | PyTorch CUDA-compatible extension build |

## Compatibility Status

| LMCache | vLLM | Iluvatar runtime | Status | Notes |
| --- | --- | --- | --- | --- |
| Pinned release (above) | V1 dynamic connector | CUDA-compatible PyTorch | Supported | `c_ops` redirect + factory patches + V2 H2D staging; non-MLA declares HND and rejects explicit NHD, while MLA defers layout to vLLM |
| Pinned release (above) | V1 MP connector (xPyD, `lmcache_driven`) | CUDA-compatible PyTorch | Supported | MP thin wrapper + native redirects + IPC event FIFO on `DefaultEventIPCBackend.create_event` plus module `torch_dev` guards (blend / blend_v3 / mp_connector); post-import runtime patch applies the same non-MLA HND policy; optional Mooncake redirect when the plugin extension is importable |
| Source-tree LMCache V1 connector API | V1 dynamic connector | CUDA-compatible PyTorch path | Framework supported | Tests cover import, idempotent patching, default factory delegation. |
| Unvalidated LMCache release | V1 dynamic connector | Iluvatar runtime installed | Requires smoke test | Check `lmcache_iluvatar.get_patch_state()`. |
| LMCache missing or without V1 adapter targets | Any | Any | Unsupported | `import lmcache_iluvatar` fails fast. |
| SGLang / MindSpore paths | Any | Any | Out of scope | vLLM-only plugin. |

## Native Extension Mapping

| Upstream Module | Plugin Extension | Plugin Redirect |
|-----------------|-----------------|-----------------|
| `lmcache.c_ops` | `lmcache_iluvatar.c_ops` | `lmcache_iluvatar.c_ops` |
| `lmcache.native_storage_ops` | `lmcache_iluvatar.native_storage_ops` | `lmcache_iluvatar.native_storage_ops` |
| `lmcache.lmcache_redis` | `lmcache_iluvatar.lmcache_redis` | `lmcache_iluvatar.lmcache_redis` |
| `lmcache.lmcache_fs` | `lmcache_iluvatar.lmcache_fs` | `lmcache_iluvatar.lmcache_fs` |
| `lmcache.lmcache_mooncake` | `lmcache_iluvatar.lmcache_mooncake` (optional) | `lmcache_iluvatar.lmcache_mooncake` when importable; otherwise skipped as optional |

## Entry Points

| Entry | Injection Mechanism |
|-------|--------------------|
| `lmcache server` / `lmcache` CLI | `.pth` meta-path hook → first `import lmcache` triggers `import lmcache_iluvatar` → redirects |
| `vllm serve` (V1 dynamic) | `kv_connector_module_path: lmcache_iluvatar.integration.vllm.lmcache_iluvatar_connector_v1` |
| `vllm serve` (MP xPyD) | vLLM `0.23.0` built-in `LMCacheMPConnector`; no Iluvatar module path |

## Minimum Smoke Checks

1. Import the plugin and verify patch state:

   ```bash
   python -c "import lmcache_iluvatar; print(lmcache_iluvatar.get_patch_state().results)"
   ```

2. Start vLLM with V1 dynamic connector:

   ```bash
   --kv-transfer-config '{"kv_connector_module_path":"lmcache_iluvatar.integration.vllm.lmcache_iluvatar_connector_v1"}'
   ```

3. Start vLLM with MP connector (xPyD):

   ```bash
   --kv-transfer-config '{"kv_connector":"LMCacheMPConnector","kv_role":"kv_both","kv_connector_extra_config":{"lmcache.mp.port":6555}}'
   ```

4. Start lmcache server via CLI override:

   ```bash
   lmcache server --port 6555 --l1-size-gb 100 --max-gpu-workers 4 --eviction-policy LRU
   ```

5. Confirm all redirected modules:

   ```bash
   python -c "
   import lmcache_iluvatar
   import sys
   for m in ['lmcache.c_ops', 'lmcache.native_storage_ops', 'lmcache.lmcache_redis', 'lmcache.lmcache_fs']:
       print(f'{m} → {sys.modules.get(m, \"NOT REDIRECTED\")}')
   "
   ```

   The optional Mooncake redirect is included automatically when
   `lmcache_iluvatar.lmcache_mooncake` is importable.

6. Confirm native availability:

   ```python
   from lmcache_iluvatar.c_ops import is_native_available
   import lmcache_iluvatar.native_storage_ops as native_storage_ops

   print("c_ops:", is_native_available())
   print("native_storage_ops:", hasattr(native_storage_ops, "TTLLock"))
   ```

7. On a compute-capable Iluvatar GPU, verify native HND D2H/H2D transfer for
   both the format 6 token/slot API and Qwen3.5 format 7 block API:

   ```bash
   python3 -m pytest -q \
     tests/csrc/test_multi_layer_kv_transfer.py \
     tests/csrc/test_multi_layer_block_kv_transfer_hnd.py
   ```

8. For MP server paths, confirm optional IPC event guard targets are either
   patched or reported missing according to the installed LMCache layout:

   ```bash
   python -c "
   import lmcache_iluvatar
   for r in lmcache_iluvatar.get_patch_state().results:
       if 'multiprocess' in r.target and r.target.endswith('.torch_dev'):
           print(r)
   "
   ```
