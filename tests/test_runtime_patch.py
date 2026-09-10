# Copyright (c) 2026, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import gc
import importlib
import sys
import weakref
from types import ModuleType, SimpleNamespace

import pytest

from fake_lmcache import install_fake_lmcache, purge_modules


def test_activate_patches_idempotent_on_fake_lmcache(monkeypatch):
    install_fake_lmcache(monkeypatch)

    plugin = importlib.import_module("lmcache_iluvatar")
    first_results = plugin.get_patch_state().results
    second_results = plugin.activate_patches()

    assert sys.modules["lmcache.c_ops"].__name__ == "lmcache_iluvatar.c_ops"
    assert first_results == second_results


def test_activate_patches_wires_required_rank_layout_path(monkeypatch):
    modules = install_fake_lmcache(monkeypatch)
    plugin = importlib.import_module("lmcache_iluvatar")
    results = {result.target: result for result in plugin.get_patch_state().results}

    rank_targets = [
        target
        for target in results
        if target.endswith(
            (
                "get_payload_classes",
                "LMCacheMPWorkerAdapter",
                "LMCacheDrivenTransferContext",
                "LayoutDescRegistry",
                "LMCacheDrivenTransferModule",
                "PrefetchRequestSpec",
                "LookupModule",
                "PrefetchController",
            )
        )
    ]
    assert len(rank_targets) >= 8
    assert all(results[target].status == "patched" for target in rank_targets)
    protocol_module = modules["lmcache.v1.multiprocess.protocol"]
    request_type = modules[
        "lmcache.v1.multiprocess.protocols.base"
    ].RequestType.REGISTER_KV_CACHE
    assert len(protocol_module.get_payload_classes(request_type)) == 8
    assert (
        modules["lmcache.v1.multiprocess.mq"].get_payload_classes
        is protocol_module.get_payload_classes
    )
    assert (
        modules["lmcache.v1.multiprocess.server"].get_payload_classes
        is protocol_module.get_payload_classes
    )

    runtime = importlib.import_module("lmcache_iluvatar.integration.patch.runtime")
    rank_specs = [
        spec
        for spec in runtime._build_patch_specs()
        if spec.target in rank_targets
    ]
    assert all(spec.required for spec in rank_specs)
    assert all(spec.version_range.contains("0.5.3") for spec in rank_specs)
    assert not any(spec.version_range.contains("0.5.4") for spec in rank_specs)


def test_c_ops_redirect_rejects_pre_v053_native_abi(monkeypatch):
    install_fake_lmcache(monkeypatch)
    runtime = importlib.import_module("lmcache_iluvatar.integration.patch.runtime")
    stale_c_ops = ModuleType("stale_c_ops")
    stale_c_ops.EngineKVFormat = SimpleNamespace()

    with pytest.raises(
        RuntimeError,
        match=r"v0\.5\.3 ABI symbols.*execute_object_group_transfer",
    ):
        runtime._require_c_ops_v053_abi(stale_c_ops)


def test_post_import_retry_updates_runtime_patch_state(monkeypatch):
    modules = install_fake_lmcache(monkeypatch)
    target_module = "vllm.distributed.kv_transfer.kv_connector.v1.lmcache_mp_connector"
    mp_module = modules[target_module]
    original = mp_module.LMCacheMPConnector
    del mp_module.LMCacheMPConnector

    plugin = importlib.import_module("lmcache_iluvatar")
    initial = next(
        item
        for item in plugin.get_patch_state().results
        if item.target.endswith("lmcache_mp_connector.LMCacheMPConnector")
    )
    assert initial.status == "missing"

    mp_module.LMCacheMPConnector = original
    results = plugin.activate_post_import_patches(target_module)

    assert results[0].status == "patched"
    updated = next(
        item
        for item in plugin.get_patch_state().results
        if item.target.endswith("lmcache_mp_connector.LMCacheMPConnector")
    )
    assert updated.status == "patched"


def test_activate_patches_registers_required_hnd_group_edits(monkeypatch):
    modules = install_fake_lmcache(monkeypatch)
    original = modules[
        "lmcache.integration.vllm.kv_cache_group_edits"
    ].apply_kv_cache_group_edits

    plugin = importlib.import_module("lmcache_iluvatar")
    edits_mod = modules["lmcache.integration.vllm.kv_cache_group_edits"]
    patched = edits_mod.apply_kv_cache_group_edits
    cached = modules[
        "lmcache.integration.vllm.lmcache_mp_connector"
    ].apply_kv_cache_group_edits
    results = {item.target: item for item in plugin.get_patch_state().results}

    edits_result = results["lmcache.integration.vllm.kv_cache_group_edits._EDITS"]
    apply_result = results[
        "lmcache.integration.vllm.kv_cache_group_edits.apply_kv_cache_group_edits"
    ]
    assert edits_result.status == "patched"
    assert apply_result.status == "patched"
    assert patched is not original
    assert cached is patched
    assert getattr(patched, "__lmcache_iluvatar_kv_cache_group_edits__", False)
    names = [getattr(edit, "name", None) for edit in edits_mod._EDITS]
    assert names.index("iluvatar-hnd-mamba-page-view") < names.index("mamba-page-view")
    assert names.index("iluvatar-hnd-subpaged-attention-view") < names.index(
        "subpaged-attention-view"
    )


def test_activate_patches_fails_when_required_hybrid_registry_is_missing(monkeypatch):
    modules = install_fake_lmcache(monkeypatch)
    edits_mod = modules["lmcache.integration.vllm.kv_cache_group_edits"]
    del edits_mod._EDITS

    with pytest.raises(
        RuntimeError,
        match="Required HND kv_cache_group_edits target missing.*_EDITS registry",
    ):
        importlib.import_module("lmcache_iluvatar")


@pytest.mark.parametrize(
    "missing_anchor",
    ["mamba-page-view", "subpaged-attention-view"],
)
def test_activate_patches_fails_when_required_hybrid_anchor_is_missing(
    monkeypatch, missing_anchor
):
    modules = install_fake_lmcache(monkeypatch)
    edits_mod = modules["lmcache.integration.vllm.kv_cache_group_edits"]
    edits_mod._EDITS = tuple(
        edit for edit in edits_mod._EDITS if edit.name != missing_anchor
    )

    with pytest.raises(
        RuntimeError,
        match=f"Required HND kv_cache_group_edits target missing.*{missing_anchor}",
    ):
        importlib.import_module("lmcache_iluvatar")


def test_activate_patches_fails_when_required_hybrid_module_is_missing(monkeypatch):
    install_fake_lmcache(monkeypatch)
    module_name = "lmcache.integration.vllm.kv_cache_group_edits"
    monkeypatch.delitem(sys.modules, module_name)

    with pytest.raises(
        RuntimeError,
        match="Unable to import patch target module.*kv_cache_group_edits",
    ):
        importlib.import_module("lmcache_iluvatar")


def test_activate_patches_wraps_v2_gpu_connector(monkeypatch):
    modules = install_fake_lmcache(monkeypatch)
    original = modules[
        "lmcache.v1.gpu_connector.gpu_connectors"
    ].VLLMPagedMemGPUConnectorV2

    plugin = importlib.import_module("lmcache_iluvatar")

    patched = modules[
        "lmcache.v1.gpu_connector.gpu_connectors"
    ].VLLMPagedMemGPUConnectorV2
    result = next(
        item
        for item in plugin.get_patch_state().results
        if item.target
        == "lmcache.v1.gpu_connector.gpu_connectors.VLLMPagedMemGPUConnectorV2"
    )
    assert result.status == "patched"
    assert patched is not original
    assert issubclass(patched, original)
    assert patched.__lmcache_iluvatar_h2d_staging__ is True


def test_activate_patches_guards_mp_ipc_event_producers(monkeypatch):
    modules = install_fake_lmcache(monkeypatch)
    targets = (
        "lmcache.v1.multiprocess.modules.lmcache_driven_transfer",
        "lmcache.v1.multiprocess.modules.blend",
        "lmcache.v1.multiprocess.modules.blend_v3",
        "lmcache.integration.vllm.lmcache_mp_connector",
    )
    originals = {target: modules[target].torch_dev for target in targets}

    plugin = importlib.import_module("lmcache_iluvatar")

    for target in targets:
        patched = modules[target].torch_dev
        result = next(
            item
            for item in plugin.get_patch_state().results
            if item.target == f"{target}.torch_dev"
        )
        event = patched.Event(interprocess=True)
        assert result.status == "patched"
        assert patched is not originals[target]
        assert getattr(patched, "__lmcache_iluvatar_patch__") is True
        # Constructor must return the real upstream Event, not a Python proxy.
        assert type(event) is type(originals[target].Event())
        assert event.kwargs == {"interprocess": True}
        assert patched.Event.from_ipc_handle("device", b"event") == (
            "ipc-handle",
            ("device", b"event"),
            {},
        )

    # Reproduce chunked-prefill behavior: a newer event replaces the reference
    # held by request_id, but the exported real Event must remain alive for the
    # MP server to import its handle.
    client_torch_dev = modules[
        "lmcache.integration.vllm.lmcache_mp_connector"
    ].torch_dev
    first = client_torch_dev.Event(interprocess=True)
    first_ref = weakref.ref(first)
    client_torch_dev.Event(interprocess=True)
    del first
    gc.collect()
    assert first_ref() is not None


def test_activate_patches_retains_event_ipc_create_event(monkeypatch):
    """v0.5.3 lmcache_driven completion Events go through create_event FIFO."""

    modules = install_fake_lmcache(monkeypatch)
    backend_cls = modules["lmcache.v1.platform.base.event_ipc"].DefaultEventIPCBackend
    torch_dev = modules[
        "lmcache.v1.multiprocess.modules.lmcache_driven_transfer"
    ].torch_dev
    backend = backend_cls(event_module=torch_dev)

    plugin = importlib.import_module("lmcache_iluvatar")
    result = next(
        item
        for item in plugin.get_patch_state().results
        if item.target == "lmcache.v1.platform.base.event_ipc.DefaultEventIPCBackend"
    )
    assert result.status in {"patched", "already_patched"}
    assert getattr(backend_cls, "__lmcache_iluvatar_event_ipc_fifo__", False)

    event = backend.create_event("cuda:0")
    event_ref = weakref.ref(event)
    del event
    gc.collect()
    assert event_ref() is not None
    assert event_ref().kwargs == {"interprocess": True}


def test_activate_patches_blocking_stage_block_ids(monkeypatch):
    """Iluvatar forces blocking Host→Device block-id staging (XID:24 / C2)."""

    modules = install_fake_lmcache(monkeypatch)
    cache_cls = modules["lmcache.v1.platform.base.cache_context"].BaseCacheContext

    plugin = importlib.import_module("lmcache_iluvatar")
    result = next(
        item
        for item in plugin.get_patch_state().results
        if item.target == "lmcache.v1.platform.base.cache_context.BaseCacheContext"
    )
    assert result.status in {"patched", "already_patched"}
    assert getattr(cache_cls, "__lmcache_iluvatar_blocking_stage_block_ids__", False)

    # Rank-aware L2 layout patches must remain intact alongside this fix.
    rank_ok = [
        item
        for item in plugin.get_patch_state().results
        if item.target.endswith(
            (
                "LayoutDescRegistry",
                "PrefetchController",
                "LookupModule",
            )
        )
    ]
    assert rank_ok
    assert all(item.status == "patched" for item in rank_ok)

    ctx = cache_cls(buffer_len=8)
    views = ctx.stage_block_ids([[1, 2], [3]])
    assert len(views) == 2
    assert ctx.block_ids_buffer_.copy_kwargs
    assert ctx.block_ids_buffer_.copy_kwargs[0]["non_blocking"] is False


def test_blend_server_v2_is_not_a_patch_target(monkeypatch):
    install_fake_lmcache(monkeypatch)
    plugin = importlib.import_module("lmcache_iluvatar")
    targets = [item.target for item in plugin.get_patch_state().results]
    assert not any("blend_server_v2" in target for target in targets)


def test_mooncake_redirect_skips_when_extension_missing(monkeypatch):
    install_fake_lmcache(monkeypatch)
    real_import_module = importlib.import_module

    def block_mooncake_extension(name, package=None):
        if name == "lmcache_iluvatar.lmcache_mooncake":
            raise ImportError("blocked mooncake extension import")
        return real_import_module(name, package)

    monkeypatch.setattr(importlib, "import_module", block_mooncake_extension)

    plugin = importlib.import_module("lmcache_iluvatar")

    mooncake_result = next(
        result
        for result in plugin.get_patch_state().results
        if result.target == "lmcache.lmcache_mooncake"
    )
    assert mooncake_result.status == "missing"
    assert "lmcache_iluvatar.lmcache_mooncake" in mooncake_result.detail
    assert "lmcache.lmcache_mooncake" not in sys.modules


def test_mooncake_redirect_patches_when_extension_importable(monkeypatch):
    install_fake_lmcache(monkeypatch)
    mooncake_module = ModuleType("lmcache_iluvatar.lmcache_mooncake")
    monkeypatch.setitem(
        sys.modules,
        "lmcache_iluvatar.lmcache_mooncake",
        mooncake_module,
    )

    plugin = importlib.import_module("lmcache_iluvatar")

    mooncake_result = next(
        result
        for result in plugin.get_patch_state().results
        if result.target == "lmcache.lmcache_mooncake"
    )
    assert mooncake_result.status == "patched"
    assert sys.modules["lmcache.lmcache_mooncake"] is mooncake_module


def test_vllm_connector_module_import_triggers_patch(monkeypatch):
    modules = install_fake_lmcache(monkeypatch)

    connector_module = importlib.import_module(
        "lmcache_iluvatar.integration.vllm.lmcache_iluvatar_connector_v1"
    )
    connector = connector_module.LMCacheIluvatarConnectorV1Dynamic(
        "config", role="worker"
    )

    assert connector.args == ("config",)
    assert connector.kwargs == {"role": "worker"}
    assert (
        modules["lmcache.v1.gpu_connector"].CreateGPUConnector.__name__
        == "CreateIluvatarGPUConnector"
    )
    gpu_config = SimpleNamespace(extra_config={})
    assert modules["lmcache.v1.gpu_connector"].CreateGPUConnector(
        gpu_config, "metadata", "VLLM"
    ) == ((gpu_config, "metadata", "VLLM"), {})


def test_import_fails_when_lmcache_is_missing(monkeypatch):
    purge_modules("lmcache", "lmcache_iluvatar")
    real_import_module = importlib.import_module

    def block_lmcache_imports(name, package=None):
        if name == "lmcache" or name.startswith("lmcache."):
            raise ImportError("blocked lmcache import")
        return real_import_module(name, package)

    monkeypatch.setattr(importlib, "import_module", block_lmcache_imports)

    with pytest.raises(RuntimeError, match="requires upstream LMCache"):
        importlib.import_module("lmcache_iluvatar")


def test_import_fails_when_vllm_dynamic_connector_is_missing(monkeypatch):
    install_fake_lmcache(monkeypatch, include_dynamic_connector=False)

    importlib.import_module("lmcache_iluvatar")

    with pytest.raises(ImportError, match="LMCacheConnectorV1Dynamic"):
        importlib.import_module(
            "lmcache_iluvatar.integration.vllm.lmcache_iluvatar_connector_v1"
        )
