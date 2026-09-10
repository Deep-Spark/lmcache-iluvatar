# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- Install upstream LMCache from `third_party/LMCache` submodule
  (`NO_NATIVE_EXT=1 --no-deps`) instead of PyPI `lmcache==…`, so CoreX CuPy is
  not replaced by `cupy-cuda13x`.
- Bump upstream pin to `lmcache==0.5.3` (`upstream_pin.json`); resync `csrc/**`
  from LMCache `v0.5.3`.
- HND `kv_cache_group_edits` / `apply_kv_cache_group_edits` take explicit
  `layout_hints` (LMCache v0.5.3 API).
- MP IPC: FIFO-retain via `DefaultEventIPCBackend.create_event` (lmcache_driven
  completion Events); remove dead `blend_server_v2` PatchSpec.

## [0.1.0] - 2026-07-28

Initial release of `lmcache-iluvatar`: a non-invasive LMCache vLLM plugin for
Iluvatar CoreX. Installed as a separate package; patches upstream LMCache at
runtime without modifying upstream source.

Pinned upstream at release: `lmcache==0.5.0` (`upstream_pin.json`). Current pin
is listed under `[Unreleased]` / `upstream_pin.json`.

### Added

- Runtime patch framework (`integration/patch/runtime.py`) with version-gated  
`PatchSpec`, idempotent activation, and `get_patch_state()` diagnostics.
- Support for vLLM V1 MP connector (`LMCacheMPConnector`) on Iluvatar.
- Native extensions built under the plugin package and redirected at import:
`c_ops`, `native_storage_ops`, `lmcache_redis`, `lmcache_fs`, and optional
`lmcache_mooncake`.
- `.pth` meta-path hook so `lmcache` CLI / server paths pick up redirects on
first `import lmcache`.
- Iluvatar adaptations of LMCache `v0.5.0` `csrc/**` (FP8 headers, generic
load/store on MP kernels, `is_native_available()`).
- Compatibility fixes relative to pinned LMCache `v0.5.0`:
  - V2 H2D staging for GPU connector
  - PD reused-key cleanup
  - Async loading cleanup (`enable_async_loading`, `EventManager.pop_event_any_status`)
  - PP cache broadcast (`save_only_first_rank` + `broadcast_src_rank` + uint8 pack)
  - MP IPC event guard (legacy and v0.5.0 module paths)
  - EIC connector namespace / busy-loop / partial-failure handling
  - Protocol `torch.int8` dtype mapping
  - Qwen3.5 hybrid KV registration with Iluvatar HND subpaged group edits
- CacheBlend attention adapter for `IluFlashAttentionImpl`.
- Install scripts: `scripts/install_dependencies.sh`,
`scripts/clean_build_install.sh` (CoreX-safe; optional Mooncake via
`LMCACHE_ILUVATAR_ENABLE_MOONCAKE=1`).
- Docs: `README.md`, `docs/compatibility.md`, `docs/development.md`,
`docs/troubleshooting-common-errors.md`.
- CI and GPU integration examples (manifest + per-example `ci/run.sh`):
P2P sharing, MP disagg prefill smoke, local-tiered PD offload, agg
external-hit offload, CacheBlend; plus additional MP/NIXL and bench profiles
under `examples/`.



### Notes

- Default configuration delegates to upstream LMCache; explicit Iluvatar-only
GPU / storage / transfer options raise `UnsupportedIluvatarFeatureError`
until implemented.
- SGLang / MindSpore paths are out of scope (vLLM-only).

