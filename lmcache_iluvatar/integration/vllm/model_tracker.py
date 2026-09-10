# Copyright (c) 2026, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""vLLM worker wrapper that registers models for LMCache CacheBlend."""

from __future__ import annotations

import logging
from typing import TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=type)


def build_cacheblend_worker(base_cls: T) -> T:
    """Register the loaded vLLM model in ``VLLMModelTracker`` after load."""

    if getattr(base_cls, "__lmcache_iluvatar_cacheblend_worker__", False):
        return base_cls

    original_load_model = base_cls.load_model

    def load_model(self, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003
        result = original_load_model(self, *args, **kwargs)
        model_runner = getattr(self, "model_runner", None)
        model = getattr(model_runner, "model", None)
        if model is not None:
            _ensure_get_input_embeddings(model)
            _register_model(model)
        else:
            logger.warning(
                "Worker.load_model completed but self.model_runner.model is "
                "missing; CacheBlend may fail to find the vLLM model"
            )
        return result

    setattr(load_model, "__wrapped__", original_load_model)
    setattr(base_cls, "load_model", load_model)
    setattr(base_cls, "__lmcache_iluvatar_cacheblend_worker__", True)
    return base_cls


def _register_model(model) -> None:  # noqa: ANN001
    from lmcache.integration.vllm.utils import ENGINE_NAME
    from lmcache.v1.compute.models.utils import VLLMModelTracker

    VLLMModelTracker.register_model(ENGINE_NAME, model)


def _ensure_get_input_embeddings(model) -> None:  # noqa: ANN001
    if callable(getattr(model, "get_input_embeddings", None)):
        return

    embedder = _find_input_embedder(model)
    if embedder is None:
        logger.warning(
            "Model %s has no get_input_embeddings/embed_input_ids adapter source; "
            "CacheBlend recompute may fail",
            type(model).__name__,
        )
        return

    def get_input_embeddings(input_ids):  # noqa: ANN001
        return embedder(input_ids)

    setattr(model, "get_input_embeddings", get_input_embeddings)
    setattr(model, "_lmcache_iluvatar_input_embeddings_adapter", True)
    logger.debug("Added get_input_embeddings adapter for %s", type(model).__name__)


def _find_input_embedder(model):  # noqa: ANN001
    candidates = [getattr(model, "embed_input_ids", None)]

    inner = getattr(model, "model", None)
    if inner is not None:
        candidates.extend(
            [
                getattr(inner, "embed_input_ids", None),
                getattr(inner, "embed_tokens", None),
            ]
        )

    candidates.append(getattr(model, "embed_tokens", None))
    for candidate in candidates:
        if callable(candidate):
            return candidate
    return None


__all__ = ["build_cacheblend_worker"]
