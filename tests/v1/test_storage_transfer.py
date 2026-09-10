# Copyright (c) 2026, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import pytest

from fake_lmcache import install_fake_lmcache


def test_storage_factory_delegates_non_iluvatar_config(monkeypatch):
    install_fake_lmcache(monkeypatch)
    from lmcache_iluvatar.v1.storage_backend import (
        build_iluvatar_storage_backend_factory,
    )

    calls = []

    def upstream_factory(*args, **kwargs):
        calls.append((args, kwargs))
        return "upstream"

    factory = build_iluvatar_storage_backend_factory(upstream_factory)
    result = factory(SimpleNamespace(extra_config={}), dst_device="cuda")

    assert result == "upstream"
    assert calls


def test_storage_factory_fails_for_explicit_iluvatar_request(monkeypatch):
    install_fake_lmcache(monkeypatch)
    from lmcache_iluvatar.v1.storage_backend import (
        UnsupportedIluvatarFeatureError,
        build_iluvatar_storage_backend_factory,
    )

    factory = build_iluvatar_storage_backend_factory(lambda *args, **kwargs: "upstream")

    with pytest.raises(UnsupportedIluvatarFeatureError, match="Iluvatar storage backend"):
        factory(SimpleNamespace(extra_config={"enable_iluvatar_storage": True}))


def test_transfer_factory_delegates_non_iluvatar_channel(monkeypatch):
    install_fake_lmcache(monkeypatch)
    from lmcache_iluvatar.v1.transfer_channel import (
        build_iluvatar_transfer_channel_factory,
    )

    calls = []

    def upstream_factory(*args, **kwargs):
        calls.append((args, kwargs))
        return "upstream"

    factory = build_iluvatar_transfer_channel_factory(upstream_factory)
    result = factory("mock_memory", False, "sender", 0, 1024, 64, 0, "tcp://peer")

    assert result == "upstream"
    assert calls


def test_transfer_factory_fails_for_iluvatar_channel(monkeypatch):
    install_fake_lmcache(monkeypatch)
    from lmcache_iluvatar.v1.storage_backend import UnsupportedIluvatarFeatureError
    from lmcache_iluvatar.v1.transfer_channel import (
        build_iluvatar_transfer_channel_factory,
    )

    factory = build_iluvatar_transfer_channel_factory(lambda *args, **kwargs: "upstream")

    with pytest.raises(UnsupportedIluvatarFeatureError, match="Iluvatar transfer channel"):
        factory("iluvatar", False, "sender", 0, 1024, 64, 0, "tcp://peer")


def test_transfer_factory_fails_for_explicit_iluvatar_config(monkeypatch):
    install_fake_lmcache(monkeypatch)
    from lmcache_iluvatar.v1.storage_backend import UnsupportedIluvatarFeatureError
    from lmcache_iluvatar.v1.transfer_channel import (
        build_iluvatar_transfer_channel_factory,
    )

    factory = build_iluvatar_transfer_channel_factory(lambda *args, **kwargs: "upstream")

    with pytest.raises(UnsupportedIluvatarFeatureError, match="Iluvatar transfer channel"):
        factory(
            "mock_memory",
            False,
            "sender",
            0,
            1024,
            64,
            0,
            "tcp://peer",
            config=SimpleNamespace(
                extra_config={"enable_iluvatar_transfer_channel": True}
            ),
        )

