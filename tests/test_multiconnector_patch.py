# Copyright (c) 2026, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib
from types import SimpleNamespace

from fake_lmcache import install_fake_lmcache


def test_multiconnector_save_blocks_patched_via_runtime(monkeypatch) -> None:
    modules = install_fake_lmcache(monkeypatch)

    plugin = importlib.import_module("lmcache_iluvatar")

    multi_mod = modules[
        "vllm.distributed.kv_transfer.kv_connector.v1.multi_connector"
    ]
    multi_cls = multi_mod.MultiConnector
    result = next(
        item
        for item in plugin.get_patch_state().results
        if item.target.endswith("multi_connector.MultiConnector")
    )

    # In-place class mutation returns the same object, so the registry may
    # report already_patched (same pattern as the CacheBlend Worker patch).
    assert result.status in {"patched", "already_patched"}
    assert multi_cls.__lmcache_iluvatar_multiconnector_save_blocks__ is True

    connector = multi_cls()
    # Cold miss: nobody chosen for load.
    assert connector._requests_to_connector.get("req-cold", -1) == -1

    blocks = multi_mod._RealBlocks(((10, 11, 12),))
    request = SimpleNamespace(request_id="req-cold")
    connector.update_state_after_alloc(request, blocks, num_external_tokens=0)

    # Both children must see real blocks (save-to-all), not empty placeholders.
    for child in connector._connectors:
        assert len(child.calls) == 1
        assert child.calls[0]["empty"] is False
        assert child.calls[0]["block_ids"] == ((10, 11, 12),)
        assert child.calls[0]["num_external_tokens"] == 0


def test_multiconnector_save_blocks_keeps_load_assignment(monkeypatch) -> None:
    modules = install_fake_lmcache(monkeypatch)

    importlib.import_module("lmcache_iluvatar")

    multi_mod = modules[
        "vllm.distributed.kv_transfer.kv_connector.v1.multi_connector"
    ]
    connector = multi_mod.MultiConnector()
    connector._requests_to_connector["req-hit"] = 0

    blocks = multi_mod._RealBlocks(((1, 2),))
    request = SimpleNamespace(request_id="req-hit")
    connector.update_state_after_alloc(request, blocks, num_external_tokens=128)

    chosen, other = connector._connectors
    assert chosen.calls[0]["num_external_tokens"] == 128
    assert chosen.calls[0]["empty"] is False
    assert other.calls[0]["num_external_tokens"] == 0
    assert other.calls[0]["empty"] is False
    assert other.calls[0]["block_ids"] == ((1, 2),)
