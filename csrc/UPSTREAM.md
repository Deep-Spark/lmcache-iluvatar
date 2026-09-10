# Upstream Source Provenance

This directory vendors the LMCache native source snapshot used by the
`lmcache-iluvatar` compatibility build.

## Source

Canonical pairing is repository-root `upstream_pin.json`:

- Upstream project: [LMCache/LMCache](https://github.com/LMCache/LMCache)
- Runtime version pin: see `lmcache.pypi_version` (currently `0.5.3`)
- Source checkout: git submodule `third_party/LMCache` (tag `v0.5.3`)

## Imported Paths

- `csrc/**` was copied from upstream LMCache `v0.5.3` (excluding `csrc/sycl/**`
  and `csrc/storage_backends/aerospike/**`).
- Upstream `setup.py` was copied to repository root as `setup_upstream.py`
  (v0.5.3 uses `setup_extensions/` profiles; that file is reference-only).

When syncing to a new LMCache tag, also update the `third_party/LMCache`
submodule gitlink and reinstall via `scripts/install_dependencies.sh`.

## Build Boundary

`setup_upstream.py` is reference-only. Do not execute it for
`lmcache-iluvatar` plugin builds, and do not copy its upstream package metadata
into the plugin package.

The active build entry is repository-root `setup.py`. It uses the upstream
`lmcache.c_ops` source list and compile settings as a reference, but builds the
package-scoped extension target:

```text
lmcache_iluvatar.c_ops
```

## Built Extensions

The active plugin build produces these package-scoped extension targets:

| Extension | Module | Upstream Equivalent |
|-----------|--------|---------------------|
| `lmcache_iluvatar.c_ops` | GPU KV copy kernels | `lmcache.c_ops` |
| `lmcache_iluvatar.native_storage_ops` | Storage manager (Bitmap, TTLLock) | `lmcache.native_storage_ops` |
| `lmcache_iluvatar.lmcache_redis` | Redis storage backend | `lmcache.lmcache_redis` |
| `lmcache_iluvatar.lmcache_fs` | Filesystem storage backend | `lmcache.lmcache_fs` |
| `lmcache_iluvatar.lmcache_mooncake` | Mooncake storage backend (optional) | `lmcache.lmcache_mooncake` |

Runtime compatibility redirects these package-scoped modules into upstream
``sys.modules`` names during plugin activation.

## Intentionally Excluded

The active plugin build does not build:

- ROCm/HIP build outputs such as ``csrc_hip/``
- Upstream ``csrc/sycl/**``
- Upstream ``csrc/storage_backends/aerospike/**``
- Upstream ``requirements/**``
- Upstream Python package metadata or scripts

## Source List Sync Policy

When syncing to a new LMCache tag:

1. Replace `csrc/**` from the upstream tag (exclude sycl / aerospike as above).
2. Replace `setup_upstream.py` with the matching upstream `setup.py`.
3. Compare the upstream `lmcache.c_ops` extension source list
   (`setup_extensions/build_profiles/cuda.py`) against repository-root `setup.py`.
4. Keep the active extension target package-scoped as `lmcache_iluvatar.c_ops`.
5. Reapply and document Iluvatar build adaptations in `csrc/ILUVATAR_PATCHES.md`.
6. Run packaging tests and the native build/import check in the target Iluvatar
   environment.
