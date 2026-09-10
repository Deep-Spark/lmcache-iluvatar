# Copyright (c) 2026, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""vLLM V1 connector module path for LMCache Iluvatar."""

from __future__ import annotations

from typing import Any

from lmcache_iluvatar import activate_patches, ensure_lmcache_install

ensure_lmcache_install()
activate_patches(strict=True)

from lmcache.integration.vllm.lmcache_connector_v1 import (  # noqa: E402  # pyright: ignore[reportMissingImports]
    LMCacheConnectorV1Dynamic as _LMCacheConnectorV1Dynamic,
)


class LMCacheIluvatarConnectorV1Dynamic(_LMCacheConnectorV1Dynamic):  # type: ignore[misc, valid-type]
    """Iluvatar vLLM connector that delegates behavior to upstream LMCache."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        activate_patches()
        super().__init__(*args, **kwargs)


# Some vLLM versions look up the standard LMCache class name after importing the
# module path. Export both names to keep the module path stable.
LMCacheConnectorV1Dynamic = LMCacheIluvatarConnectorV1Dynamic

__all__ = ["LMCacheConnectorV1Dynamic", "LMCacheIluvatarConnectorV1Dynamic"]
