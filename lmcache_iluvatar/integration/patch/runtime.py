# Copyright (c) 2026, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Version-aware runtime patch registry for upstream LMCache symbols."""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib import import_module
from importlib import metadata as importlib_metadata
import logging
import re
import sys
from threading import RLock
from types import ModuleType
from typing import Any, Callable

logger = logging.getLogger(__name__)

ILUVATAR_PATCH_MARKER = "__lmcache_iluvatar_patch__"


@dataclass(frozen=True)
class PatchResult:
    """Result for one attempted patch target."""

    target: str
    status: str
    detail: str = ""


@dataclass(frozen=True)
class VersionRange:
    """Simple inclusive/exclusive version range for patch applicability."""

    package: str
    min_version: str | None = None
    max_version: str | None = None
    include_min: bool = True
    include_max: bool = False

    def contains(self, version: str) -> bool:
        parsed = _parse_version(version)
        if self.min_version is not None:
            minimum = _parse_version(self.min_version)
            if parsed < minimum or (parsed == minimum and not self.include_min):
                return False
        if self.max_version is not None:
            maximum = _parse_version(self.max_version)
            if parsed > maximum or (parsed == maximum and not self.include_max):
                return False
        return True


ReplacementFactory = Callable[[Any | None], Any]


@dataclass(frozen=True)
class PatchSpec:
    """Version-aware description of a monkey patch target."""

    target_module: str
    target_attr: str
    replacement: Any | ReplacementFactory
    target: str
    required: bool = True
    version_range: VersionRange | None = None
    patch_cached_refs: tuple[tuple[str, str], ...] = ()

    def build_replacement(self, current: Any | None) -> Any:
        if callable(self.replacement) and getattr(
            self.replacement, "__lmcache_iluvatar_replacement_factory__", False
        ):
            return self.replacement(current)
        return self.replacement


@dataclass
class PatchState:
    """Global patch state, kept stable across repeated imports."""

    applied: bool = False
    results: list[PatchResult] = field(default_factory=list)


_PATCH_STATE = PatchState()
_PATCH_LOCK = RLock()


def get_patch_state() -> PatchState:
    """Return the current patch state for tests and diagnostics."""

    return _PATCH_STATE


def activate_c_ops_redirect(*, strict: bool = False) -> PatchResult:
    """Preinstall the ``lmcache.c_ops`` redirect before upstream LMCache import."""

    with _PATCH_LOCK:
        return _redirect_lmcache_c_ops(strict=strict)


def activate_patches(*, strict: bool = False) -> list[PatchResult]:
    """Patch upstream LMCache targets once.

    Args:
        strict: When true, raise if a required target cannot be imported or
            patched. Importing ``lmcache_iluvatar`` calls this in strict mode
            after validating the required upstream LMCache vLLM APIs.
    """

    with _PATCH_LOCK:
        if _PATCH_STATE.applied:
            logger.debug("lmcache-iluvatar patches already applied")
            return list(_PATCH_STATE.results)

        results: list[PatchResult] = []
        results.append(_redirect_lmcache_c_ops(strict=strict))
        results.append(_redirect_lmcache_native_storage_ops(strict=strict))
        results.append(_redirect_lmcache_redis(strict=False))
        results.append(_redirect_lmcache_fs(strict=False))
        results.append(_redirect_lmcache_mooncake(strict=False))
        results.append(_patch_protocol_dtypes(strict=strict))
        # Install HND edit into upstream _EDITS under the same lock, before
        # the public apply_kv_cache_group_edits wrapper is swapped in.
        results.append(_install_hnd_kv_cache_group_edits(strict=strict))
        for spec in _build_patch_specs():
            results.extend(_apply_patch_spec(spec, strict=strict))

        _PATCH_STATE.results = results
        _PATCH_STATE.applied = True

        patched = [result.target for result in results if result.status == "patched"]
        skipped = [
            result.target
            for result in results
            if result.status in {"missing", "unsupported_version"}
        ]
        logger.info(
            "Activated lmcache-iluvatar patches: patched=%s skipped=%s",
            patched,
            skipped,
        )
        return list(results)


def activate_post_import_patches(
    module_name: str,
    *,
    strict: bool = False,
) -> list[PatchResult]:
    """Retry registered patches after a watched module finishes importing.

    The import hook owns only module-load timing. Patch selection, locking,
    replacement, and diagnostic state remain centralized in this registry.
    """

    with _PATCH_LOCK:
        specs = tuple(
            spec for spec in _build_patch_specs() if spec.target_module == module_name
        )
        results: list[PatchResult] = []
        for spec in specs:
            results.extend(_apply_patch_spec(spec, strict=strict))

        if _PATCH_STATE.applied:
            _merge_patch_results(results)
        return results


def _merge_patch_results(results: list[PatchResult]) -> None:
    """Merge retry results without downgrading a confirmed patched target."""

    positions = {
        result.target: index for index, result in enumerate(_PATCH_STATE.results)
    }
    for result in results:
        index = positions.get(result.target)
        if index is None:
            positions[result.target] = len(_PATCH_STATE.results)
            _PATCH_STATE.results.append(result)
            continue
        previous = _PATCH_STATE.results[index]
        if previous.status == "patched" and result.status == "already_patched":
            continue
        _PATCH_STATE.results[index] = result


def _import_optional(module_name: str, *, strict: bool) -> ModuleType | None:
    try:
        return import_module(module_name)
    except ImportError as exc:
        if strict:
            raise RuntimeError(
                f"Unable to import patch target module {module_name!r}: {exc}"
            ) from exc
        return None


def _is_torch_shared_library_error(exc: ImportError) -> bool:
    message = str(exc)
    return "libc10.so" in message or "libtorch" in message


def _import_native_module(module_name: str) -> ModuleType:
    try:
        return import_module(module_name)
    except ImportError as exc:
        if not _is_torch_shared_library_error(exc):
            raise
        # PyTorch extensions may need torch imported first so its shared
        # libraries are visible to the dynamic loader.
        import torch  # noqa: F401

        return import_module(module_name)


def _require_c_ops_v053_abi(c_ops_module: ModuleType) -> None:
    """Fail fast when the installed native extension predates v0.5.3 ABI."""

    required_symbols = (
        "EngineKVFormat",
        "execute_object_group_transfer",
        "execute_cb_retrieve_plan_flat",
        "is_cross_layer",
        "is_kv_list",
        "is_layer_list",
        "is_mla",
    )
    missing = [name for name in required_symbols if not hasattr(c_ops_module, name)]
    engine_kv_format = getattr(c_ops_module, "EngineKVFormat", None)
    if engine_kv_format is not None and not hasattr(
        engine_kv_format, "NL_X_NB_BSV_BSS"
    ):
        missing.append("EngineKVFormat.NL_X_NB_BSV_BSS")
    if not missing:
        return
    raise RuntimeError(
        "lmcache_iluvatar.c_ops is missing LMCache v0.5.3 ABI symbols "
        f"{missing!r} (stale native build). A pre-v0.5.3 *.so in "
        "lmcache_iluvatar/ can shadow the installed wheel. Rebuild after "
        "syncing csrc to LMCache v0.5.3: bash scripts/clean_build_install.sh"
    )


def _redirect_lmcache_c_ops(*, strict: bool) -> PatchResult:
    target = "lmcache.c_ops"
    try:
        iluvatar_c_ops = _import_native_module("lmcache_iluvatar.c_ops")
    except ImportError as exc:
        detail = f"unable to import c_ops redirect target: {exc}"
        if strict:
            raise RuntimeError(detail) from exc
        logger.error(detail)
        return PatchResult(target, "missing", detail)

    try:
        _require_c_ops_v053_abi(iluvatar_c_ops)
    except RuntimeError as exc:
        if strict:
            raise
        logger.error("%s", exc)
        return PatchResult(target, "missing", str(exc))

    sys.modules[target] = iluvatar_c_ops

    try:
        lmcache_module = import_module("lmcache")
    except ImportError as exc:
        detail = f"unable to import lmcache for c_ops redirect: {exc}"
        if strict:
            raise RuntimeError(detail) from exc
        logger.error(detail)
        return PatchResult(target, "missing", detail)

    # Re-set sys.modules after lmcache.__init__._install_c_ops_shim() has run.
    # v0.5.3 installs a DeviceOps PEP 562 shim that may overwrite
    # sys.modules["lmcache.c_ops"] during import. CudaDeviceOps.ensure_native
    # imports lmcache.c_ops (binding our redirect if we ran first); we then
    # write the Iluvatar module back so both the shim table and package attr
    # stay on lmcache_iluvatar.c_ops.
    sys.modules[target] = iluvatar_c_ops
    setattr(lmcache_module, "c_ops", iluvatar_c_ops)

    logger.info("Redirected LMCache target %s to lmcache_iluvatar.c_ops", target)
    return PatchResult(target, "patched", "redirected to lmcache_iluvatar.c_ops")


def _redirect_native_module(
    target: str,
    plugin_module: str,
    *,
    strict: bool,
) -> PatchResult:
    """Redirect ``sys.modules[target]`` to the plugin native extension (.so).

    The plugin extension target (e.g. ``lmcache_iluvatar.native_storage_ops``)
    matches the upstream pybind module name so no C++ source changes are needed.
    When the native .so is not built or cannot be loaded, the redirect is
    skipped and the upstream module path remains unchanged.
    """
    try:
        iluvatar_mod = _import_native_module(plugin_module)
    except ImportError as exc:
        detail = f"unable to import {plugin_module!r}: {exc}"
        if strict:
            raise RuntimeError(detail) from exc
        logger.warning("Skipping redirect %s: %s", target, detail)
        return PatchResult(target, "missing", detail)

    sys.modules[target] = iluvatar_mod
    logger.info("Redirected LMCache target %s to %s", target, plugin_module)
    return PatchResult(target, "patched", f"redirected to {plugin_module}")


def _redirect_lmcache_native_storage_ops(*, strict: bool) -> PatchResult:
    return _redirect_native_module(
        "lmcache.native_storage_ops",
        "lmcache_iluvatar.native_storage_ops",
        strict=strict,
    )


def _redirect_lmcache_redis(*, strict: bool) -> PatchResult:
    return _redirect_native_module(
        "lmcache.lmcache_redis",
        "lmcache_iluvatar.lmcache_redis",
        strict=strict,
    )


def _redirect_lmcache_fs(*, strict: bool) -> PatchResult:
    return _redirect_native_module(
        "lmcache.lmcache_fs",
        "lmcache_iluvatar.lmcache_fs",
        strict=strict,
    )


def _redirect_lmcache_mooncake(*, strict: bool) -> PatchResult:
    return _redirect_native_module(
        "lmcache.lmcache_mooncake",
        "lmcache_iluvatar.lmcache_mooncake",
        strict=strict,
    )


def _patch_protocol_dtypes(*, strict: bool) -> PatchResult:
    """Extend LMCache protocol dtype maps with torch.int8 (still missing in 0.5.3)."""

    target = "lmcache.v1.protocol"
    try:
        import torch

        protocol_module = import_module(target)
    except ImportError as exc:
        detail = f"unable to import protocol patch target: {exc}"
        if strict:
            raise RuntimeError(detail) from exc
        logger.error(detail)
        return PatchResult(f"{target}.DTYPE_TO_INT", "missing", detail)

    if protocol_module.DTYPE_TO_INT.get(torch.int8) == 9:
        return PatchResult(f"{target}.DTYPE_TO_INT", "already_patched")

    protocol_module.DTYPE_TO_INT[torch.int8] = 9
    protocol_module.INT_TO_DTYPE[9] = torch.int8
    logger.info("Patched LMCache protocol dtype maps for torch.int8")
    return PatchResult(f"{target}.DTYPE_TO_INT", "patched", "added torch.int8=9")


def _patch_attr(
    module: ModuleType,
    attr: str,
    replacement: Any,
    target: str,
    *,
    strict: bool,
) -> PatchResult:
    if not hasattr(module, attr):
        message = f"target attribute {target} is not present"
        if strict:
            raise RuntimeError(message)
        logger.warning(message)
        return PatchResult(target, "missing", message)

    current = getattr(module, attr)
    if getattr(current, ILUVATAR_PATCH_MARKER, False):
        return PatchResult(target, "already_patched")

    # In-place class mutation builders (EventManager, DefaultEventIPCBackend)
    # return the same object after mutating methods. Still count as patched.
    setattr(replacement, ILUVATAR_PATCH_MARKER, True)
    if current is not replacement:
        setattr(module, attr, replacement)
    logger.info("Patched LMCache target %s", target)
    return PatchResult(target, "patched")


def replacement_factory(func: ReplacementFactory) -> ReplacementFactory:
    """Mark a callable as a factory that receives the upstream target."""

    setattr(func, "__lmcache_iluvatar_replacement_factory__", True)
    return func


def _parse_version(version: str) -> tuple[int, ...]:
    parts = re.findall(r"\d+", version)
    return tuple(int(part) for part in parts[:4]) or (0,)


def _version_status(spec: PatchSpec) -> PatchResult | None:
    if spec.version_range is None:
        return None
    try:
        installed = importlib_metadata.version(spec.version_range.package)
    except importlib_metadata.PackageNotFoundError:
        # Source-tree and tests may import modules without installed package
        # metadata. Keep patching active in that case.
        return None
    if spec.version_range.contains(installed):
        return None
    detail = (
        f"installed {spec.version_range.package} version {installed!r} is outside "
        f"supported range min={spec.version_range.min_version!r} "
        f"max={spec.version_range.max_version!r}"
    )
    logger.warning("Skipping LMCache patch %s: %s", spec.target, detail)
    return PatchResult(spec.target, "unsupported_version", detail)


def _apply_patch_spec(spec: PatchSpec, *, strict: bool) -> list[PatchResult]:
    version_result = _version_status(spec)
    if version_result is not None:
        if strict and spec.required:
            raise RuntimeError(f"Unsupported version for required patch {spec.target}")
        return [version_result]

    module = _import_optional(spec.target_module, strict=strict and spec.required)
    if module is None:
        detail = f"module {spec.target_module} is not importable"
        if spec.required:
            logger.error("Required LMCache patch target missing: %s", spec.target)
        return [PatchResult(spec.target, "missing", detail)]

    if not hasattr(module, spec.target_attr):
        detail = f"target attribute {spec.target} is not present"
        if strict and spec.required:
            raise RuntimeError(detail)
        logger.error(detail) if spec.required else logger.warning(detail)
        return [PatchResult(spec.target, "missing", detail)]

    current = getattr(module, spec.target_attr)
    replacement = spec.build_replacement(current)
    results = [
        _patch_attr(
            module,
            spec.target_attr,
            replacement,
            spec.target,
            strict=strict and spec.required,
        )
    ]

    for cached_module_name, cached_attr in spec.patch_cached_refs:
        cached_module = _import_optional(cached_module_name, strict=False)
        cached_target = f"{cached_module_name}.{cached_attr}"
        if cached_module is None or not hasattr(cached_module, cached_attr):
            results.append(
                PatchResult(cached_target, "missing", "cached reference missing")
            )
            continue
        results.append(
            _patch_attr(
                cached_module,
                cached_attr,
                replacement,
                cached_target,
                strict=False,
            )
        )

    return results


@replacement_factory
def _build_gpu_connector_factory(current: Any | None) -> Any:
    from lmcache_iluvatar.v1.gpu_connector import build_iluvatar_gpu_connector_factory

    return build_iluvatar_gpu_connector_factory(current)


@replacement_factory
def _build_paged_mem_gpu_connector_v2(current: Any | None) -> Any:
    if current is None:
        raise RuntimeError("VLLMPagedMemGPUConnectorV2 is not present")
    from lmcache_iluvatar.v1.gpu_connector import (
        build_iluvatar_paged_mem_gpu_connector_v2,
    )

    return build_iluvatar_paged_mem_gpu_connector_v2(current)


@replacement_factory
def _build_storage_factory(current: Any | None) -> Any:
    from lmcache_iluvatar.v1.storage_backend import (
        build_iluvatar_storage_backend_factory,
    )

    return build_iluvatar_storage_backend_factory(current)


@replacement_factory
def _build_transfer_channel_factory(current: Any | None) -> Any:
    from lmcache_iluvatar.v1.transfer_channel import (
        build_iluvatar_transfer_channel_factory,
    )

    return build_iluvatar_transfer_channel_factory(current)


@replacement_factory
def _build_cache_engine(current: Any | None) -> Any:
    if current is None:
        raise RuntimeError("LMCacheEngine is not present")
    from lmcache_iluvatar.v1.cache_engine import build_iluvatar_cache_engine

    return build_iluvatar_cache_engine(current)


@replacement_factory
def _build_storage_manager(current: Any | None) -> Any:
    if current is None:
        raise RuntimeError("StorageManager is not present")
    from lmcache_iluvatar.v1.storage_backend.storage_manager import (
        build_iluvatar_storage_manager,
    )

    return build_iluvatar_storage_manager(current)


@replacement_factory
def _build_pd_backend(current: Any | None) -> Any:
    if current is None:
        raise RuntimeError("PDBackend is not present")
    from lmcache_iluvatar.v1.storage_backend.pd_backend import build_iluvatar_pd_backend

    return build_iluvatar_pd_backend(current)


@replacement_factory
def _build_pd_backend_async(current: Any | None) -> Any:
    """Build the Iluvatar wrapper for LMCache's async PD backend class."""

    if current is None:
        raise RuntimeError("PDBackendAsync is not present")
    from lmcache_iluvatar.v1.storage_backend.pd_backend_async import (
        build_iluvatar_pd_backend_async,
    )

    return build_iluvatar_pd_backend_async(current)


@replacement_factory
def _build_event_manager(current: Any | None) -> Any:
    if current is None:
        raise RuntimeError("EventManager is not present")
    from lmcache_iluvatar.v1.event_manager import build_iluvatar_event_manager

    return build_iluvatar_event_manager(current)


def _install_hnd_kv_cache_group_edits(*, strict: bool) -> PatchResult:
    """Install required Iluvatar HND hybrid edits into ``_EDITS`` (locked)."""

    target = "lmcache.integration.vllm.kv_cache_group_edits._EDITS"
    module = _import_optional(
        "lmcache.integration.vllm.kv_cache_group_edits",
        strict=strict,
    )
    if module is None:
        detail = (
            "module lmcache.integration.vllm.kv_cache_group_edits is not importable"
        )
        logger.error("Required HND kv_cache_group_edits target missing: %s", detail)
        return PatchResult(target, "missing", detail)

    from lmcache_iluvatar.integration.vllm.kv_cache_group_edits import (
        install_iluvatar_hnd_kv_cache_group_edits,
    )

    status, detail = install_iluvatar_hnd_kv_cache_group_edits(module)
    if status == "missing":
        if strict:
            raise RuntimeError(
                f"Required HND kv_cache_group_edits target missing: {detail}"
            )
        logger.error("Required HND kv_cache_group_edits target missing: %s", detail)
        return PatchResult(target, status, detail)
    return PatchResult(target, status, detail)


@replacement_factory
def _build_apply_kv_cache_group_edits(current: Any | None) -> Any:
    if current is None:
        raise RuntimeError("apply_kv_cache_group_edits is not present")
    from lmcache_iluvatar.integration.vllm.kv_cache_group_edits import (
        build_iluvatar_apply_kv_cache_group_edits,
    )

    return build_iluvatar_apply_kv_cache_group_edits(current)


@replacement_factory
def _build_async_lookup_client(current: Any | None) -> Any:
    if current is None:
        raise RuntimeError("LMCacheAsyncLookupClient is not present")
    from lmcache_iluvatar.v1.lookup_client import build_iluvatar_async_lookup_client

    return build_iluvatar_async_lookup_client(current)


@replacement_factory
def _build_vllm_adapter_impl(current: Any | None) -> Any:
    if current is None:
        raise RuntimeError("LMCacheConnectorV1Impl is not present in vLLM adapter")
    from lmcache_iluvatar.integration.vllm.adapter import build_iluvatar_v1_impl

    return build_iluvatar_v1_impl(current)


@replacement_factory
def _build_eic_connector(current: Any | None) -> Any:
    if current is None:
        raise RuntimeError("EICConnector is not present")
    from lmcache_iluvatar.v1.storage_backend.connector.eic_connector import (
        build_iluvatar_eic_connector,
    )

    return build_iluvatar_eic_connector(current)


@replacement_factory
def _build_ipc_event_torch_dev_guard(current: Any | None) -> Any:
    if current is None:
        return None
    from lmcache_iluvatar.v1.ipc_event_guard import build_guarded_torch_dev

    return build_guarded_torch_dev(current)


@replacement_factory
def _build_event_ipc_backend_create_event(current: Any | None) -> Any:
    """FIFO-retain events from DefaultEventIPCBackend.create_event (v0.5.3+)."""

    if current is None:
        return None
    from lmcache_iluvatar.v1.ipc_event_guard import build_guarded_event_ipc_backend

    return build_guarded_event_ipc_backend(current)


@replacement_factory
def _build_blocking_stage_block_ids(current: Any | None) -> Any:
    """Blocking Host→Device block-id staging (Iluvatar XID:24 / C2 fix)."""

    if current is None:
        return None
    from lmcache_iluvatar.v1.stage_block_ids_guard import build_blocking_stage_block_ids

    return build_blocking_stage_block_ids(current)


@replacement_factory
def _build_lmc_blender(current: Any | None) -> Any:
    if current is None:
        raise RuntimeError("LMCBlender is not present")
    from lmcache_iluvatar.v1.cacheblend.defaults import build_iluvatar_lmc_blender

    return build_iluvatar_lmc_blender(current)


@replacement_factory
def _build_attn_backend_infer(current: Any | None) -> Any:
    if current is None:
        raise RuntimeError("infer_attn_backend_from_vllm is not present")
    from lmcache_iluvatar.v1.cacheblend.attention import (
        build_iluvatar_attn_backend_infer,
    )

    return build_iluvatar_attn_backend_infer(current)


@replacement_factory
def _build_cacheblend_worker(current: Any | None) -> Any:
    if current is None:
        raise RuntimeError("vLLM Worker is not present")
    from lmcache_iluvatar.integration.vllm.model_tracker import (
        build_cacheblend_worker,
    )

    return build_cacheblend_worker(current)


@replacement_factory
def _build_multiconnector_save_blocks(current: Any | None) -> Any:
    if current is None:
        raise RuntimeError("vLLM MultiConnector is not present")
    from lmcache_iluvatar.integration.vllm.multi_connector import (
        build_multiconnector_save_blocks,
    )

    return build_multiconnector_save_blocks(current)


@replacement_factory
def _build_lmcache_mp_connector(current: Any | None) -> Any:
    if current is None:
        raise RuntimeError("vLLM LMCacheMPConnector is not present")
    from lmcache_iluvatar.integration.vllm.kv_cache_layout import (
        build_iluvatar_kv_layout_connector,
    )

    return build_iluvatar_kv_layout_connector(current)


@replacement_factory
def _build_lmcache_dynamic_connector(current: Any | None) -> Any:
    if current is None:
        raise RuntimeError("LMCacheConnectorV1Dynamic is not present")
    from lmcache_iluvatar.integration.vllm.kv_cache_layout import (
        build_iluvatar_kv_layout_connector,
    )

    return build_iluvatar_kv_layout_connector(current)


@replacement_factory
def _build_rank_aware_get_payload_classes(current: Any | None) -> Any:
    from lmcache_iluvatar.v1.mp_rank_layout import (
        build_rank_aware_get_payload_classes,
    )

    return build_rank_aware_get_payload_classes(current)


@replacement_factory
def _build_rank_aware_worker_adapter(current: Any | None) -> Any:
    from lmcache_iluvatar.v1.mp_rank_layout import build_rank_aware_worker_adapter

    return build_rank_aware_worker_adapter(current)


@replacement_factory
def _build_rank_aware_worker_transfer(current: Any | None) -> Any:
    from lmcache_iluvatar.v1.mp_rank_layout import build_rank_aware_worker_transfer

    return build_rank_aware_worker_transfer(current)


@replacement_factory
def _build_rank_aware_layout_registry(current: Any | None) -> Any:
    from lmcache_iluvatar.v1.mp_rank_layout import build_rank_aware_layout_registry

    return build_rank_aware_layout_registry(current)


@replacement_factory
def _build_rank_aware_transfer_module(current: Any | None) -> Any:
    from lmcache_iluvatar.v1.mp_rank_layout import build_rank_aware_transfer_module

    return build_rank_aware_transfer_module(current)


@replacement_factory
def _build_rank_aware_prefetch_spec(current: Any | None) -> Any:
    from lmcache_iluvatar.v1.mp_rank_layout import build_rank_aware_prefetch_spec

    return build_rank_aware_prefetch_spec(current)


@replacement_factory
def _build_rank_aware_lookup_module(current: Any | None) -> Any:
    from lmcache_iluvatar.v1.mp_rank_layout import build_rank_aware_lookup_module

    return build_rank_aware_lookup_module(current)


@replacement_factory
def _build_rank_aware_prefetch_controller(current: Any | None) -> Any:
    from lmcache_iluvatar.v1.mp_rank_layout import (
        build_rank_aware_prefetch_controller,
    )

    return build_rank_aware_prefetch_controller(current)


def _build_patch_specs() -> tuple[PatchSpec, ...]:
    from lmcache_iluvatar._upstream_pin import pinned_lmcache_pypi_version

    pin_version = pinned_lmcache_pypi_version()
    lmcache_versions = VersionRange(
        "lmcache",
        min_version=pin_version,
        max_version=pin_version,
        include_min=True,
        include_max=True,
    )
    rank_layout_versions = VersionRange(
        "lmcache",
        min_version="0.5.3",
        max_version="0.5.4",
        include_min=True,
        include_max=False,
    )
    return (
        PatchSpec(
            "lmcache.v1.multiprocess.protocol",
            "get_payload_classes",
            _build_rank_aware_get_payload_classes,
            "lmcache.v1.multiprocess.protocol.get_payload_classes",
            required=True,
            version_range=rank_layout_versions,
            patch_cached_refs=(
                ("lmcache.v1.multiprocess.mq", "get_payload_classes"),
                ("lmcache.v1.multiprocess.server", "get_payload_classes"),
            ),
        ),
        PatchSpec(
            "lmcache.integration.vllm.vllm_multi_process_adapter",
            "LMCacheMPWorkerAdapter",
            _build_rank_aware_worker_adapter,
            (
                "lmcache.integration.vllm.vllm_multi_process_adapter."
                "LMCacheMPWorkerAdapter"
            ),
            required=True,
            version_range=rank_layout_versions,
        ),
        PatchSpec(
            "lmcache.v1.multiprocess.transfer_context.worker_transfer",
            "LMCacheDrivenTransferContext",
            _build_rank_aware_worker_transfer,
            (
                "lmcache.v1.multiprocess.transfer_context.worker_transfer."
                "LMCacheDrivenTransferContext"
            ),
            required=True,
            version_range=rank_layout_versions,
        ),
        PatchSpec(
            "lmcache.v1.multiprocess.engine_context",
            "LayoutDescRegistry",
            _build_rank_aware_layout_registry,
            "lmcache.v1.multiprocess.engine_context.LayoutDescRegistry",
            required=True,
            version_range=rank_layout_versions,
        ),
        PatchSpec(
            "lmcache.v1.multiprocess.modules.lmcache_driven_transfer",
            "LMCacheDrivenTransferModule",
            _build_rank_aware_transfer_module,
            (
                "lmcache.v1.multiprocess.modules.lmcache_driven_transfer."
                "LMCacheDrivenTransferModule"
            ),
            required=True,
            version_range=rank_layout_versions,
        ),
        PatchSpec(
            "lmcache.v1.distributed.api",
            "PrefetchRequestSpec",
            _build_rank_aware_prefetch_spec,
            "lmcache.v1.distributed.api.PrefetchRequestSpec",
            required=True,
            version_range=rank_layout_versions,
            patch_cached_refs=(
                (
                    "lmcache.v1.multiprocess.modules.lookup",
                    "PrefetchRequestSpec",
                ),
                (
                    "lmcache.v1.distributed.storage_controllers."
                    "prefetch_controller",
                    "PrefetchRequestSpec",
                ),
                ("lmcache.v1.distributed.storage_manager", "PrefetchRequestSpec"),
            ),
        ),
        PatchSpec(
            "lmcache.v1.multiprocess.modules.lookup",
            "LookupModule",
            _build_rank_aware_lookup_module,
            "lmcache.v1.multiprocess.modules.lookup.LookupModule",
            required=True,
            version_range=rank_layout_versions,
        ),
        PatchSpec(
            (
                "lmcache.v1.distributed.storage_controllers."
                "prefetch_controller"
            ),
            "PrefetchController",
            _build_rank_aware_prefetch_controller,
            (
                "lmcache.v1.distributed.storage_controllers."
                "prefetch_controller.PrefetchController"
            ),
            required=True,
            version_range=rank_layout_versions,
        ),
        PatchSpec(
            "lmcache.v1.multiprocess.server",
            "torch_dev",
            _build_ipc_event_torch_dev_guard,
            "lmcache.v1.multiprocess.server.torch_dev",
            required=False,
        ),
        # Module-level torch_dev remains for device/stream helpers; completion
        # Events for lmcache_driven go through DefaultEventIPCBackend.create_event
        # (patched below) as of LMCache v0.5.3.
        PatchSpec(
            "lmcache.v1.multiprocess.modules.lmcache_driven_transfer",
            "torch_dev",
            _build_ipc_event_torch_dev_guard,
            "lmcache.v1.multiprocess.modules.lmcache_driven_transfer.torch_dev",
            required=False,
        ),
        PatchSpec(
            "lmcache.v1.platform.base.event_ipc",
            "DefaultEventIPCBackend",
            _build_event_ipc_backend_create_event,
            "lmcache.v1.platform.base.event_ipc.DefaultEventIPCBackend",
            required=False,
            version_range=lmcache_versions,
        ),
        # Iluvatar: upstream non_blocking stage_block_ids races transient
        # frombuffer host memory → dirty block ids → XID:24 on gather.
        # Keep upstream LMCache pristine; fix only via this plugin patch.
        # Orthogonal to rank-aware MP L2 layout patches above.
        PatchSpec(
            "lmcache.v1.platform.base.cache_context",
            "BaseCacheContext",
            _build_blocking_stage_block_ids,
            "lmcache.v1.platform.base.cache_context.BaseCacheContext",
            required=True,
            version_range=lmcache_versions,
        ),
        PatchSpec(
            "lmcache.v1.multiprocess.modules.blend",
            "torch_dev",
            _build_ipc_event_torch_dev_guard,
            "lmcache.v1.multiprocess.modules.blend.torch_dev",
            required=False,
        ),
        PatchSpec(
            "lmcache.v1.multiprocess.modules.blend_v3",
            "torch_dev",
            _build_ipc_event_torch_dev_guard,
            "lmcache.v1.multiprocess.modules.blend_v3.torch_dev",
            required=False,
        ),
        PatchSpec(
            "lmcache.integration.vllm.lmcache_mp_connector",
            "torch_dev",
            _build_ipc_event_torch_dev_guard,
            "lmcache.integration.vllm.lmcache_mp_connector.torch_dev",
            required=False,
            version_range=lmcache_versions,
        ),
        PatchSpec(
            "lmcache.v1.event_manager",
            "EventManager",
            _build_event_manager,
            "lmcache.v1.event_manager.EventManager",
            version_range=lmcache_versions,
        ),
        PatchSpec(
            "lmcache.v1.compute.blend.blender",
            "LMCBlender",
            _build_lmc_blender,
            "lmcache.v1.compute.blend.blender.LMCBlender",
            required=False,
            version_range=lmcache_versions,
            patch_cached_refs=(
                ("lmcache.v1.compute.blend", "LMCBlender"),
                ("lmcache.v1.compute.blend.utils", "LMCBlender"),
            ),
        ),
        PatchSpec(
            "lmcache.v1.compute.attention.utils",
            "infer_attn_backend_from_vllm",
            _build_attn_backend_infer,
            "lmcache.v1.compute.attention.utils.infer_attn_backend_from_vllm",
            required=False,
            version_range=lmcache_versions,
            patch_cached_refs=(
                ("lmcache.v1.compute.models.base", "infer_attn_backend_from_vllm"),
            ),
        ),
        PatchSpec(
            "vllm.v1.worker.gpu_worker",
            "Worker",
            _build_cacheblend_worker,
            "vllm.v1.worker.gpu_worker.Worker",
            required=False,
        ),
        PatchSpec(
            "vllm.distributed.kv_transfer.kv_connector.v1.multi_connector",
            "MultiConnector",
            _build_multiconnector_save_blocks,
            "vllm.distributed.kv_transfer.kv_connector.v1.multi_connector.MultiConnector",
            required=False,
        ),
        PatchSpec(
            "vllm.distributed.kv_transfer.kv_connector.v1.lmcache_mp_connector",
            "LMCacheMPConnector",
            _build_lmcache_mp_connector,
            (
                "vllm.distributed.kv_transfer.kv_connector.v1."
                "lmcache_mp_connector.LMCacheMPConnector"
            ),
            required=False,
        ),
        PatchSpec(
            "lmcache.integration.vllm.lmcache_connector_v1",
            "LMCacheConnectorV1Dynamic",
            _build_lmcache_dynamic_connector,
            ("lmcache.integration.vllm.lmcache_connector_v1.LMCacheConnectorV1Dynamic"),
            required=False,
            version_range=lmcache_versions,
        ),
        PatchSpec(
            "lmcache.v1.storage_backend.storage_manager",
            "StorageManager",
            _build_storage_manager,
            "lmcache.v1.storage_backend.storage_manager.StorageManager",
            version_range=lmcache_versions,
            patch_cached_refs=(("lmcache.v1.cache_engine", "StorageManager"),),
        ),
        PatchSpec(
            "lmcache.v1.cache_engine",
            "LMCacheEngine",
            _build_cache_engine,
            "lmcache.v1.cache_engine.LMCacheEngine",
            version_range=lmcache_versions,
            patch_cached_refs=(
                ("lmcache.integration.vllm.vllm_v1_adapter", "LMCacheEngine"),
                ("lmcache.v1.manager", "LMCacheEngine"),
            ),
        ),
        PatchSpec(
            "lmcache.v1.storage_backend.pd_backend",
            "PDBackend",
            _build_pd_backend,
            "lmcache.v1.storage_backend.pd_backend.PDBackend",
            required=False,
            version_range=lmcache_versions,
            patch_cached_refs=(("lmcache.v1.storage_backend", "PDBackend"),),
        ),
        PatchSpec(
            "lmcache.v1.storage_backend.pd_backend_async",
            "PDBackendAsync",
            _build_pd_backend_async,
            "lmcache.v1.storage_backend.pd_backend_async.PDBackendAsync",
            required=False,
            version_range=lmcache_versions,
            patch_cached_refs=(("lmcache.v1.storage_backend", "PDBackendAsync"),),
        ),
        PatchSpec(
            "lmcache.v1.gpu_connector",
            "CreateGPUConnector",
            _build_gpu_connector_factory,
            "lmcache.v1.gpu_connector.CreateGPUConnector",
            version_range=lmcache_versions,
            patch_cached_refs=(("lmcache.v1.manager", "CreateGPUConnector"),),
        ),
        PatchSpec(
            "lmcache.v1.gpu_connector.gpu_connectors",
            "VLLMPagedMemGPUConnectorV2",
            _build_paged_mem_gpu_connector_v2,
            "lmcache.v1.gpu_connector.gpu_connectors.VLLMPagedMemGPUConnectorV2",
            version_range=lmcache_versions,
        ),
        PatchSpec(
            "lmcache.v1.storage_backend",
            "CreateStorageBackends",
            _build_storage_factory,
            "lmcache.v1.storage_backend.CreateStorageBackends",
            version_range=lmcache_versions,
            patch_cached_refs=(
                ("lmcache.v1.storage_backend.storage_manager", "CreateStorageBackends"),
            ),
        ),
        PatchSpec(
            "lmcache.v1.transfer_channel",
            "CreateTransferChannel",
            _build_transfer_channel_factory,
            "lmcache.v1.transfer_channel.CreateTransferChannel",
            required=False,
            version_range=lmcache_versions,
            patch_cached_refs=(
                ("lmcache.v1.storage_backend.pd_backend", "CreateTransferChannel"),
                ("lmcache.v1.storage_backend.p2p_backend", "CreateTransferChannel"),
            ),
        ),
        PatchSpec(
            "lmcache.v1.lookup_client.lmcache_async_lookup_client",
            "LMCacheAsyncLookupClient",
            _build_async_lookup_client,
            "lmcache.v1.lookup_client.lmcache_async_lookup_client.LMCacheAsyncLookupClient",
            required=False,
            version_range=lmcache_versions,
            patch_cached_refs=(
                ("lmcache.v1.lookup_client.factory", "LMCacheAsyncLookupClient"),
            ),
        ),
        PatchSpec(
            "lmcache.integration.vllm.kv_cache_group_edits",
            "apply_kv_cache_group_edits",
            _build_apply_kv_cache_group_edits,
            "lmcache.integration.vllm.kv_cache_group_edits.apply_kv_cache_group_edits",
            required=True,
            version_range=lmcache_versions,
            patch_cached_refs=(
                (
                    "lmcache.integration.vllm.lmcache_mp_connector",
                    "apply_kv_cache_group_edits",
                ),
            ),
        ),
        PatchSpec(
            "lmcache.integration.vllm.vllm_v1_adapter",
            "LMCacheConnectorV1Impl",
            _build_vllm_adapter_impl,
            "lmcache.integration.vllm.vllm_v1_adapter.LMCacheConnectorV1Impl",
            version_range=lmcache_versions,
            patch_cached_refs=(
                (
                    "lmcache.integration.vllm.lmcache_connector_v1",
                    "LMCacheConnectorV1Impl",
                ),
                (
                    "lmcache.integration.vllm.lmcache_connector_v1_085",
                    "LMCacheConnectorV1Impl",
                ),
            ),
        ),
        PatchSpec(
            "lmcache.v1.storage_backend.connector.eic_connector",
            "EICConnector",
            _build_eic_connector,
            "lmcache.v1.storage_backend.connector.eic_connector.EICConnector",
            required=False,
            version_range=lmcache_versions,
        ),
    )
