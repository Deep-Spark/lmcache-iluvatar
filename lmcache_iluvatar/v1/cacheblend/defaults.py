# Copyright (c) 2026, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""CacheBlend default-value compatibility wrapper."""

from __future__ import annotations

import logging
from typing import TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=type)

_DEFAULT_CHECK_LAYERS = [1]
_DEFAULT_RECOMPUTE_RATIOS = [0.15]


def build_iluvatar_lmc_blender(base_cls: T) -> T:
    """Fill documented CacheBlend defaults before upstream blender init."""

    if getattr(base_cls, "__lmcache_iluvatar_cacheblend_defaults__", False):
        return base_cls

    class IluvatarLMCBlender(base_cls):  # type: ignore[misc, valid-type]
        __lmcache_iluvatar_cacheblend_defaults__ = True

        def __init__(self, cache_engine, gpu_connector, vllm_model, config):
            if getattr(config, "enable_blending", False):
                if getattr(config, "blend_check_layers", None) is None:
                    config.blend_check_layers = list(_DEFAULT_CHECK_LAYERS)
                if getattr(config, "blend_recompute_ratios", None) is None:
                    config.blend_recompute_ratios = list(_DEFAULT_RECOMPUTE_RATIOS)
            super().__init__(cache_engine, gpu_connector, vllm_model, config)

    IluvatarLMCBlender.__name__ = "IluvatarLMCBlender"
    IluvatarLMCBlender.__qualname__ = "IluvatarLMCBlender"
    IluvatarLMCBlender.__module__ = __name__
    logger.debug("Built CacheBlend LMCBlender defaults wrapper")
    return IluvatarLMCBlender  # type: ignore[return-value]


__all__ = ["build_iluvatar_lmc_blender"]
