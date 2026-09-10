# Copyright (c) 2026, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Iluvatar storage backend entry points for lmcache-iluvatar.

The first plugin release exposes the factory and patch hook, but does not claim
to implement P2P/PD/remote Iluvatar storage. Normal upstream LMCache storage paths
delegate to the original factory; explicit Iluvatar storage requests fail with a
diagnostic error instead of silently falling back.
"""

from __future__ import annotations

from collections.abc import Callable
import logging
from typing import Any

logger = logging.getLogger(__name__)

ILUVATAR_STORAGE_KEYS = {
    "enable_iluvatar_storage",
    "iluvatar_storage_backend",
    "iluvatar_storage_channel",
}


class UnsupportedIluvatarFeatureError(RuntimeError):
    """Raised when an Iluvatar data path is requested but not implemented yet."""


def _is_truthy(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on", "iluvatar"}
    return bool(value)


def is_iluvatar_storage_requested(config: Any) -> bool:
    """Return true when config explicitly requests an Iluvatar storage backend."""

    extra_config = getattr(config, "extra_config", None) or {}
    for key in ILUVATAR_STORAGE_KEYS:
        if _is_truthy(getattr(config, key, False)):
            return True
        if isinstance(extra_config, dict) and _is_truthy(extra_config.get(key)):
            return True

    storage_plugins = getattr(config, "storage_plugins", None) or []
    return any(str(plugin).lower().startswith("iluvatar") for plugin in storage_plugins)


def create_unsupported_iluvatar_storage_backends(*args: Any, **kwargs: Any) -> Any:
    """Explicit factory for currently unsupported Iluvatar storage data paths."""

    config = args[0] if args else kwargs.get("config")
    extra_config = getattr(config, "extra_config", None)
    raise UnsupportedIluvatarFeatureError(
        "Iluvatar storage backend is not implemented in this lmcache-iluvatar "
        "release. Use upstream LMCache storage tiers, or remove Iluvatar storage "
        f"options from config.extra_config={extra_config!r}."
    )


def build_iluvatar_storage_backend_factory(
    upstream_factory: Callable[..., Any] | None,
) -> Callable[..., Any]:
    """Wrap LMCache's storage factory with explicit Iluvatar unsupported handling."""

    def CreateIluvatarStorageBackends(*args: Any, **kwargs: Any) -> Any:
        config = args[0] if args else kwargs.get("config")
        if is_iluvatar_storage_requested(config):
            logger.error("Iluvatar storage backend was requested but is not implemented.")
            return create_unsupported_iluvatar_storage_backends(*args, **kwargs)
        if upstream_factory is None:
            raise UnsupportedIluvatarFeatureError(
                "LMCache CreateStorageBackends is unavailable; cannot delegate "
                "non-Iluvatar storage factory call."
            )
        return upstream_factory(*args, **kwargs)

    CreateIluvatarStorageBackends.__name__ = "CreateIluvatarStorageBackends"
    CreateIluvatarStorageBackends.__qualname__ = "CreateIluvatarStorageBackends"
    CreateIluvatarStorageBackends.__module__ = __name__
    setattr(CreateIluvatarStorageBackends, "__lmcache_iluvatar_original__", upstream_factory)
    return CreateIluvatarStorageBackends


__all__ = [
    "UnsupportedIluvatarFeatureError",
    "build_iluvatar_storage_backend_factory",
    "create_unsupported_iluvatar_storage_backends",
    "is_iluvatar_storage_requested",
]
