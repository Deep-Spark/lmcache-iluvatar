# Copyright (c) 2026, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import pytest

from fake_lmcache import install_fake_lmcache


def test_gpu_connector_requires_slot_mapping(monkeypatch):
    install_fake_lmcache(monkeypatch)
    from lmcache_iluvatar.v1.gpu_connector import VLLMIluvatarGPUConnector

    connector = VLLMIluvatarGPUConnector(SimpleNamespace(kv_shape=(1, 2, 16, 4, 8)))
    connector.initialize_kvcaches_ptr(kvcaches=["kv"])

    with pytest.raises(ValueError, match="slot_mapping"):
        connector.batched_to_gpu(["memory"], [0], [1])


def test_gpu_connector_reports_transfer_unsupported_after_validation(monkeypatch):
    install_fake_lmcache(monkeypatch)
    from lmcache_iluvatar.v1.gpu_connector import VLLMIluvatarGPUConnector
    from lmcache_iluvatar.v1.storage_backend import UnsupportedIluvatarFeatureError

    connector = VLLMIluvatarGPUConnector(SimpleNamespace(kv_shape=(1, 2, 16, 4, 8)))
    connector.initialize_kvcaches_ptr(kvcaches=["kv"])

    with pytest.raises(UnsupportedIluvatarFeatureError, match="KV H2D transfer"):
        connector.batched_to_gpu(["memory"], [1], [3], slot_mapping=[10, 11, 12, 13])


def test_gpu_factory_wrapper_delegates_default_config(monkeypatch):
    install_fake_lmcache(monkeypatch)
    from lmcache_iluvatar.v1.gpu_connector import build_iluvatar_gpu_connector_factory

    calls = []

    def upstream_factory(*args, **kwargs):
        calls.append((args, kwargs))
        return "upstream"

    factory = build_iluvatar_gpu_connector_factory(upstream_factory)
    result = factory(SimpleNamespace(extra_config={}), "metadata", "VLLM")

    assert result == "upstream"
    assert calls


def test_gpu_factory_wrapper_uses_iluvatar_only_when_explicit(monkeypatch):
    install_fake_lmcache(monkeypatch)
    from lmcache_iluvatar.v1.gpu_connector import (
        VLLMIluvatarGPUConnector,
        build_iluvatar_gpu_connector_factory,
    )
    from lmcache_iluvatar.v1.storage_backend import UnsupportedIluvatarFeatureError

    factory = build_iluvatar_gpu_connector_factory(lambda *args, **kwargs: "upstream")
    connector = factory(
        SimpleNamespace(extra_config={"enable_iluvatar_gpu_connector": True}),
        SimpleNamespace(kv_shape=(1, 2, 16, 4, 8), kv_dtype="float16"),
        "VLLM",
    )

    assert isinstance(connector, VLLMIluvatarGPUConnector)
    connector.initialize_kvcaches_ptr(kvcaches=["kv"])
    with pytest.raises(UnsupportedIluvatarFeatureError, match="KV H2D transfer"):
        connector.batched_to_gpu(["memory"], [1], [3], slot_mapping=[10, 11])

