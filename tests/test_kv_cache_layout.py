# Copyright (c) 2026, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from fake_lmcache import install_fake_lmcache


def _vllm_config(*, use_mla: bool):
    return SimpleNamespace(model_config=SimpleNamespace(use_mla=use_mla))


@pytest.mark.parametrize("requested", [None, "HND", "hnd"])
def test_layout_policy_accepts_hnd_or_unset(monkeypatch, requested) -> None:
    install_fake_lmcache(monkeypatch)
    if requested is None:
        monkeypatch.delenv("VLLM_KV_CACHE_LAYOUT", raising=False)
    else:
        monkeypatch.setenv("VLLM_KV_CACHE_LAYOUT", requested)

    module = importlib.import_module(
        "lmcache_iluvatar.integration.vllm.kv_cache_layout"
    )

    module.validate_iluvatar_kv_cache_layout(_vllm_config(use_mla=False))


def test_layout_policy_rejects_nhd(monkeypatch) -> None:
    install_fake_lmcache(monkeypatch)
    monkeypatch.setenv("VLLM_KV_CACHE_LAYOUT", "NHD")

    module = importlib.import_module(
        "lmcache_iluvatar.integration.vllm.kv_cache_layout"
    )

    with pytest.raises(RuntimeError, match="supports only HND"):
        module.validate_iluvatar_kv_cache_layout(_vllm_config(use_mla=False))


def test_layout_policy_defers_for_mla_or_missing_model_config(monkeypatch) -> None:
    install_fake_lmcache(monkeypatch)
    monkeypatch.setenv("VLLM_KV_CACHE_LAYOUT", "NHD")
    module = importlib.import_module(
        "lmcache_iluvatar.integration.vllm.kv_cache_layout"
    )
    mla_config = _vllm_config(use_mla=True)
    missing_model_config = SimpleNamespace(model_config=None)

    assert module.get_required_iluvatar_kv_cache_layout(mla_config) is None
    assert (
        module.get_required_iluvatar_kv_cache_layout(missing_model_config) is None
    )
    module.validate_iluvatar_kv_cache_layout(mla_config)
    module.validate_iluvatar_kv_cache_layout(missing_model_config)


@pytest.mark.parametrize(
    ("env_layout", "resolved_layout", "expected_source"),
    [
        (None, "HND", 'layout_hints["kv_layout"]'),
        ("hnd", "HND", 'layout_hints["kv_layout"]'),
        ("NHD", None, "VLLM_KV_CACHE_LAYOUT"),
        (None, None, "shape/stride heuristic"),
    ],
)
def test_registration_layout_resolution(
    monkeypatch,
    env_layout,
    resolved_layout,
    expected_source,
) -> None:
    install_fake_lmcache(monkeypatch)
    if env_layout is None:
        monkeypatch.delenv("VLLM_KV_CACHE_LAYOUT", raising=False)
    else:
        monkeypatch.setenv("VLLM_KV_CACHE_LAYOUT", env_layout)
    module = importlib.import_module(
        "lmcache_iluvatar.integration.vllm.kv_cache_layout"
    )
    hints = {} if resolved_layout is None else {"kv_layout": resolved_layout}

    resolution = module.resolve_iluvatar_kv_cache_layout(hints)

    expected_layout = resolved_layout
    if expected_layout is None and env_layout is not None:
        expected_layout = env_layout.upper()
    assert resolution.layout == expected_layout
    assert resolution.source == expected_source


@pytest.mark.parametrize(
    ("env_layout", "resolved_layout"),
    [("NHD", "HND"), ("HND", "NHD")],
)
def test_registration_layout_resolution_rejects_conflict(
    monkeypatch,
    env_layout,
    resolved_layout,
) -> None:
    install_fake_lmcache(monkeypatch)
    monkeypatch.setenv("VLLM_KV_CACHE_LAYOUT", env_layout)
    module = importlib.import_module(
        "lmcache_iluvatar.integration.vllm.kv_cache_layout"
    )

    with pytest.raises(RuntimeError, match="KV cache layout conflict"):
        module.resolve_iluvatar_kv_cache_layout({"kv_layout": resolved_layout})


def test_dynamic_connector_declares_hnd_and_rejects_nhd(monkeypatch) -> None:
    install_fake_lmcache(monkeypatch)
    module = importlib.import_module(
        "lmcache_iluvatar.integration.vllm.lmcache_iluvatar_connector_v1"
    )

    # The module-path class remains a thin shell; the inherited upstream class
    # receives the layout behavior through the runtime patch registry.
    assert "get_required_kvcache_layout" not in (
        module.LMCacheIluvatarConnectorV1Dynamic.__dict__
    )
    config = _vllm_config(use_mla=False)
    assert module.LMCacheConnectorV1Dynamic.get_required_kvcache_layout(config) == "HND"
    monkeypatch.delenv("VLLM_KV_CACHE_LAYOUT", raising=False)
    module.LMCacheConnectorV1Dynamic(config, role="worker")

    monkeypatch.setenv("VLLM_KV_CACHE_LAYOUT", "HND")
    module.LMCacheConnectorV1Dynamic(config, role="worker")

    monkeypatch.setenv("VLLM_KV_CACHE_LAYOUT", "NHD")
    with pytest.raises(RuntimeError, match="supports only HND"):
        module.LMCacheConnectorV1Dynamic(config, role="worker")

    mla_config = _vllm_config(use_mla=True)
    assert (
        module.LMCacheConnectorV1Dynamic.get_required_kvcache_layout(mla_config)
        is None
    )
    module.LMCacheConnectorV1Dynamic(mla_config, role="worker")


def test_runtime_patch_wraps_mp_connector_with_hnd_policy(monkeypatch) -> None:
    modules = install_fake_lmcache(monkeypatch)
    original = modules[
        "vllm.distributed.kv_transfer.kv_connector.v1.lmcache_mp_connector"
    ].LMCacheMPConnector

    importlib.import_module("lmcache_iluvatar")
    patched = sys.modules[
        "vllm.distributed.kv_transfer.kv_connector.v1.lmcache_mp_connector"
    ].LMCacheMPConnector

    assert patched is not original
    config = _vllm_config(use_mla=False)
    assert patched.get_required_kvcache_layout(config) == "HND"
    monkeypatch.delenv("VLLM_KV_CACHE_LAYOUT", raising=False)
    patched(config, role="worker")

    monkeypatch.setenv("VLLM_KV_CACHE_LAYOUT", "HND")
    patched(config, role="worker")

    monkeypatch.setenv("VLLM_KV_CACHE_LAYOUT", "NHD")
    with pytest.raises(RuntimeError, match="supports only HND"):
        patched(config, role="worker")

    mla_config = _vllm_config(use_mla=True)
    assert patched.get_required_kvcache_layout(mla_config) is None
    patched(mla_config, role="worker")


def test_post_import_hook_wraps_mp_connector_after_class_definition(
    monkeypatch,
) -> None:
    modules = install_fake_lmcache(monkeypatch)
    mp_module = modules[
        "vllm.distributed.kv_transfer.kv_connector.v1.lmcache_mp_connector"
    ]
    original = mp_module.LMCacheMPConnector
    del mp_module.LMCacheMPConnector
    importlib.import_module("lmcache_iluvatar")

    # Recreate the important import-order case: initial activation observed a
    # partially initialized MP module, then its class appeared later.
    hook_path = Path(__file__).resolve().parents[1] / "_lmcache_iluvatar_hook.py"
    spec = importlib.util.spec_from_file_location("test_lmcache_iluvatar_hook", hook_path)
    assert spec is not None and spec.loader is not None
    hook = importlib.util.module_from_spec(spec)

    class _MPModuleLoader:
        def exec_module(self, module):
            module.LMCacheMPConnector = original

    mp_spec = importlib.machinery.ModuleSpec(
        mp_module.__name__,
        _MPModuleLoader(),
    )
    monkeypatch.setattr(
        importlib.machinery.PathFinder,
        "find_spec",
        lambda fullname, path: mp_spec,
    )
    original_meta_path = list(sys.meta_path)
    try:
        spec.loader.exec_module(hook)
        wrapped_spec = hook._LMCacheIluvatarHook().find_spec(
            mp_module.__name__,
            None,
        )
        assert wrapped_spec is mp_spec
        wrapped_spec.loader.exec_module(mp_module)
    finally:
        sys.meta_path[:] = original_meta_path

    assert mp_module.LMCacheMPConnector is not original
    assert (
        mp_module.LMCacheMPConnector.get_required_kvcache_layout(
            _vllm_config(use_mla=False)
        )
        == "HND"
    )
    result = next(
        item
        for item in importlib.import_module("lmcache_iluvatar").get_patch_state().results
        if item.target.endswith("lmcache_mp_connector.LMCacheMPConnector")
    )
    assert result.status == "patched"

    patched = mp_module.LMCacheMPConnector
    hook._post_vllm_import(mp_module)
    assert mp_module.LMCacheMPConnector is patched
