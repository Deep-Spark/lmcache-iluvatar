# Iluvatar Native Build Adaptations

This file records plugin-local adaptations on top of the imported LMCache
`v0.5.3` `csrc/**` snapshot. Modified kernel/pybind files carry a
`Modified for Iluvatar CoreX` header in addition to LMCache and Iluvatar
copyright lines.

## Iluvatar Compiler Notes

- `csrc/mem_kernels.cu` avoids CUDA `cuda_fp8.h` on Iluvatar builds because
  the Iluvatar runtime provides its own FP8 definitions
  (`#elif !defined(__ILUVATAR__)`).
- `csrc/mp_mem_kernels.cu` avoids CUDA inline asm load/store paths on Iluvatar
  builds and uses the generic `uint4` load/store path instead
  (`#if defined(__CUDA_ARCH__) && !defined(__ILUVATAR__)`).

## Plugin pybind deltas (on top of upstream v0.5.3 `csrc/pybind.cpp`)

- `PYBIND11_MODULE(TORCH_EXTENSION_NAME, ...)` for `lmcache_iluvatar.c_ops`
- `is_native_available()` export

Upstream v0.5.3 continues to expose `GPUKVFormat` as an alias of `EngineKVFormat`
and adds format helpers (`is_cross_layer` / `is_kv_list` / `is_layer_list` /
`is_mla`), fused/content-size layouts, object-group transfer, and CB retrieve
plan bindings. The plugin keeps those upstream bindings; only the module name
and `is_native_available` are Iluvatar deltas.

## Build Target

- Repository-root `setup.py` builds plugin-scoped native extensions from this
  snapshot.
- The active `c_ops` extension target is `lmcache_iluvatar.c_ops`; the plugin
  redirects upstream `lmcache.c_ops` imports to that package-scoped module
  instead of building an upstream-owned `lmcache.c_ops` module.
- Extension targets use the same last-component name as upstream pybind modules
  (no `_` prefix needed), e.g. `lmcache_iluvatar.native_storage_ops` maps to
  `PYBIND11_MODULE(native_storage_ops, m)`.  No C++ source changes are required
  for the storage-backend extensions.
- `csrc/pybind.cpp` uses `TORCH_EXTENSION_NAME` (set by `CUDAExtension` to
  `c_ops`).
- `setup.py` `C_OPS_SOURCES` includes `csrc/blend_kernels.cu` (new in v0.5.3).

## v0.5.3 csrc additions (vs v0.5.0)

- `csrc/blend_kernels.cu` / `blend_kernels.cuh` — CacheBlend retrieve plan
- `csrc/engine_kv_format.h` — shared format classification predicates
- Additional `EngineKVFormat` values (`NL_X_NB_BS_NH_TWO_HS`, `*_CS`,
  `NL_X_NB_BSV_BSS`) and object-group / CB retrieve pybind surface
- Not vendored: `csrc/sycl/**`, `csrc/storage_backends/aerospike/**`
