# Copyright (c) 2026, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib
from types import SimpleNamespace

from fake_lmcache import install_fake_lmcache, purge_modules


def test_cacheblend_defaults_are_patched_via_runtime(monkeypatch) -> None:
    modules = install_fake_lmcache(monkeypatch)

    importlib.import_module("lmcache_iluvatar")

    blender_cls = modules["lmcache.v1.compute.blend.blender"].LMCBlender
    config = SimpleNamespace(
        enable_blending=True,
        blend_check_layers=None,
        blend_recompute_ratios=None,
    )
    blender = blender_cls(None, None, None, config)

    assert blender.config.blend_check_layers == [1]
    assert blender.config.blend_recompute_ratios == [0.15]
    assert blender_cls.__name__ == "IluvatarLMCBlender"


def test_worker_model_tracker_is_patched_via_runtime(monkeypatch) -> None:
    modules = install_fake_lmcache(monkeypatch)

    importlib.import_module("lmcache_iluvatar")

    worker_cls = modules["vllm.v1.worker.gpu_worker"].Worker
    tracker = modules["lmcache.v1.compute.models.utils"].VLLMModelTracker
    worker = worker_cls()

    assert worker.load_model() == "loaded"

    model = worker.model_runner.model
    assert tracker.registered["vllm-instance"] is model
    assert model.get_input_embeddings("ids") == ("embedded", "ids")
    assert worker_cls.__lmcache_iluvatar_cacheblend_worker__ is True


def test_iluvatar_attention_infer_is_patched_via_runtime(monkeypatch) -> None:
    modules = install_fake_lmcache(monkeypatch)

    plugin = importlib.import_module("lmcache_iluvatar")

    infer = modules["lmcache.v1.compute.attention.utils"].infer_attn_backend_from_vllm
    cached_infer = modules["lmcache.v1.compute.models.base"].infer_attn_backend_from_vllm
    result = next(
        item
        for item in plugin.get_patch_state().results
        if item.target
        == "lmcache.v1.compute.attention.utils.infer_attn_backend_from_vllm"
    )

    assert result.status == "patched"
    assert cached_infer is infer

    class IluFlashAttentionImpl:
        scale = 1.0
        alibi_slopes = None
        sliding_window = (-1, -1)
        logits_soft_cap = 0.0

    iluvatar_attn = SimpleNamespace(impl=IluFlashAttentionImpl())
    backend = infer(iluvatar_attn, enable_sparse=False)

    assert type(backend).__name__ == "LMCIluFlashAttnBackend"
    assert backend.vllm_attn is iluvatar_attn
    assert backend.vllm_attn_impl is iluvatar_attn.impl


def test_iluvatar_attention_infer_preserves_upstream_fallback(monkeypatch) -> None:
    modules = install_fake_lmcache(monkeypatch)

    importlib.import_module("lmcache_iluvatar")

    infer = modules["lmcache.v1.compute.attention.utils"].infer_attn_backend_from_vllm

    class FlashAttentionImpl:
        pass

    vllm_attn = SimpleNamespace(impl=FlashAttentionImpl())

    assert infer(vllm_attn, enable_sparse=False) == {
        "vllm_attn": vllm_attn,
        "enable_sparse": False,
        "source": "upstream",
    }
    assert infer(vllm_attn, enable_sparse=True) == {
        "vllm_attn": vllm_attn,
        "enable_sparse": True,
        "source": "upstream",
    }
