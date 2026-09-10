# Copyright (c) 2026, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Iluvatar transfer channel entry points for lmcache-iluvatar."""

from __future__ import annotations

from collections.abc import Callable
import logging
from typing import Any

from lmcache_iluvatar.v1.storage_backend import UnsupportedIluvatarFeatureError

logger = logging.getLogger(__name__)

ILUVATAR_CHANNEL_TYPES = {"iluvatar", "iluvatar_p2p"}
ILUVATAR_CHANNEL_CONFIG_KEYS = {
    "enable_iluvatar_transfer_channel",
    "use_iluvatar_transfer_channel",
    "iluvatar_transfer_channel",
}


def _is_truthy(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on", "iluvatar"}
    return bool(value)


def is_iluvatar_channel(channel_type: Any) -> bool:
    """Return true when a transfer channel name targets Iluvatar."""

    return str(channel_type or "").lower() in ILUVATAR_CHANNEL_TYPES


def is_iluvatar_transfer_channel_requested(config: Any) -> bool:
    """Return true when config explicitly opts into an Iluvatar channel."""

    extra_config = getattr(config, "extra_config", None) or {}
    for key in ILUVATAR_CHANNEL_CONFIG_KEYS:
        if _is_truthy(getattr(config, key, False)):
            return True
        if isinstance(extra_config, dict) and _is_truthy(extra_config.get(key)):
            return True
    return False


def create_unsupported_iluvatar_transfer_channel(*args: Any, **kwargs: Any) -> Any:
    """Explicit factory for currently unsupported Iluvatar transfer channels."""

    channel_type = args[0] if args else kwargs.get("channel_type")
    raise UnsupportedIluvatarFeatureError(
        "Iluvatar transfer channel is not implemented in this lmcache-iluvatar "
        f"release: channel_type={channel_type!r}."
    )


def build_iluvatar_transfer_channel_factory(
    upstream_factory: Callable[..., Any] | None,
) -> Callable[..., Any]:
    """Wrap LMCache's transfer factory with Iluvatar channel detection."""

    def CreateIluvatarTransferChannel(*args: Any, **kwargs: Any) -> Any:
        channel_type = kwargs.get("channel_type", args[0] if args else None)
        config = kwargs.get("config")
        if is_iluvatar_channel(channel_type) or is_iluvatar_transfer_channel_requested(
            config
        ):
            logger.error(
                "Iluvatar transfer channel was requested but is not implemented: %s",
                channel_type,
            )
            return create_unsupported_iluvatar_transfer_channel(*args, **kwargs)
        if upstream_factory is None:
            raise UnsupportedIluvatarFeatureError(
                "LMCache CreateTransferChannel is unavailable; cannot delegate "
                "non-Iluvatar transfer channel call."
            )
        return upstream_factory(*args, **kwargs)

    CreateIluvatarTransferChannel.__name__ = "CreateIluvatarTransferChannel"
    CreateIluvatarTransferChannel.__qualname__ = "CreateIluvatarTransferChannel"
    CreateIluvatarTransferChannel.__module__ = __name__
    setattr(CreateIluvatarTransferChannel, "__lmcache_iluvatar_original__", upstream_factory)
    return CreateIluvatarTransferChannel


__all__ = [
    "ILUVATAR_CHANNEL_TYPES",
    "build_iluvatar_transfer_channel_factory",
    "create_unsupported_iluvatar_transfer_channel",
    "is_iluvatar_channel",
    "is_iluvatar_transfer_channel_requested",
]
