# Copyright (c) 2026, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""KV cache layout policy for the Iluvatar vLLM integration."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import os
from typing import Any

REQUIRED_KV_CACHE_LAYOUT = "HND"
_KV_CACHE_LAYOUT_ENV = "VLLM_KV_CACHE_LAYOUT"
_PATCH_MARKER = "__lmcache_iluvatar_kv_cache_layout__"
_VALID_KV_CACHE_LAYOUTS = frozenset({"HND", "NHD"})


@dataclass(frozen=True)
class KVCacheLayoutResolution:
    """Resolved layout shared by all edits in one registration call."""

    layout: str | None
    source: str
    env_override: str | None


_ACTIVE_KV_CACHE_LAYOUT: ContextVar[KVCacheLayoutResolution | None] = ContextVar(
    "lmcache_iluvatar_active_kv_cache_layout",
    default=None,
)


def _normalize_kv_cache_layout(value: Any, *, source: str) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip().upper()
    if normalized not in _VALID_KV_CACHE_LAYOUTS:
        raise RuntimeError(
            f"Unsupported KV cache layout from {source}: {value!r}; "
            "expected HND or NHD."
        )
    return normalized


def resolve_iluvatar_kv_cache_layout(
    layout_hints: Mapping[str, Any] | None = None,
) -> KVCacheLayoutResolution:
    """Resolve registration layout metadata and validate the user override.

    ``layout_hints["kv_layout"]`` is produced from vLLM's final layout
    resolver. The environment variable is retained as an explicit user
    override, but it must agree with that resolved value when both exist.
    """

    env_layout = _normalize_kv_cache_layout(
        os.getenv(_KV_CACHE_LAYOUT_ENV),
        source=_KV_CACHE_LAYOUT_ENV,
    )
    resolved_layout = _normalize_kv_cache_layout(
        layout_hints.get("kv_layout") if layout_hints is not None else None,
        source='layout_hints["kv_layout"]',
    )
    if (
        env_layout is not None
        and resolved_layout is not None
        and env_layout != resolved_layout
    ):
        raise RuntimeError(
            "KV cache layout conflict: "
            f"{_KV_CACHE_LAYOUT_ENV}={env_layout} but vLLM resolved "
            f"{resolved_layout} in layout_hints; refusing hybrid KV cache "
            "registration."
        )
    if resolved_layout is not None:
        return KVCacheLayoutResolution(
            layout=resolved_layout,
            source='layout_hints["kv_layout"]',
            env_override=env_layout,
        )
    if env_layout is not None:
        return KVCacheLayoutResolution(
            layout=env_layout,
            source=_KV_CACHE_LAYOUT_ENV,
            env_override=env_layout,
        )
    return KVCacheLayoutResolution(
        layout=None,
        source="shape/stride heuristic",
        env_override=None,
    )


@contextmanager
def use_iluvatar_kv_cache_layout(
    layout_hints: Mapping[str, Any] | None = None,
) -> Iterator[KVCacheLayoutResolution]:
    """Expose one resolved layout to every edit in a registration call."""

    resolution = resolve_iluvatar_kv_cache_layout(layout_hints)
    token = _ACTIVE_KV_CACHE_LAYOUT.set(resolution)
    try:
        yield resolution
    finally:
        _ACTIVE_KV_CACHE_LAYOUT.reset(token)


def get_active_iluvatar_kv_cache_layout() -> KVCacheLayoutResolution:
    """Return the scoped registration layout, falling back to the env hint."""

    active = _ACTIVE_KV_CACHE_LAYOUT.get()
    if active is not None:
        return active
    return resolve_iluvatar_kv_cache_layout()


def get_required_iluvatar_kv_cache_layout(vllm_config: Any) -> str | None:
    """Return HND for non-MLA models and defer layout for MLA or missing config."""

    model_config = getattr(vllm_config, "model_config", None)
    if model_config is None or getattr(model_config, "use_mla", False):
        return None
    return REQUIRED_KV_CACHE_LAYOUT


def validate_iluvatar_kv_cache_layout(vllm_config: Any) -> None:
    """Reject layouts unsupported by the current Iluvatar attention backend."""

    if get_required_iluvatar_kv_cache_layout(vllm_config) is None:
        return
    requested = os.getenv(_KV_CACHE_LAYOUT_ENV)
    if requested is None or requested.upper() == REQUIRED_KV_CACHE_LAYOUT:
        return
    raise RuntimeError(
        "The current Iluvatar attention backend supports only HND KV cache "
        "layout. Unset VLLM_KV_CACHE_LAYOUT or set it to HND; "
        f"got {requested!r}."
    )


def build_iluvatar_kv_layout_connector(base_cls: type[Any]) -> type[Any]:
    """Wrap a connector class with the Iluvatar HND layout policy."""

    if getattr(base_cls, _PATCH_MARKER, False):
        return base_cls

    class IluvatarKVLayoutConnector(base_cls):  # type: ignore[misc, valid-type]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            vllm_config = kwargs.get("vllm_config")
            if vllm_config is None and args:
                vllm_config = args[0]
            validate_iluvatar_kv_cache_layout(vllm_config)
            super().__init__(*args, **kwargs)

        @classmethod
        def get_required_kvcache_layout(cls, vllm_config: Any) -> str | None:
            return get_required_iluvatar_kv_cache_layout(vllm_config)

    IluvatarKVLayoutConnector.__name__ = base_cls.__name__
    IluvatarKVLayoutConnector.__qualname__ = base_cls.__qualname__
    IluvatarKVLayoutConnector.__module__ = base_cls.__module__
    setattr(IluvatarKVLayoutConnector, _PATCH_MARKER, True)
    setattr(IluvatarKVLayoutConnector, "__lmcache_iluvatar_original__", base_cls)
    return IluvatarKVLayoutConnector


__all__ = [
    "KVCacheLayoutResolution",
    "REQUIRED_KV_CACHE_LAYOUT",
    "build_iluvatar_kv_layout_connector",
    "get_active_iluvatar_kv_cache_layout",
    "get_required_iluvatar_kv_cache_layout",
    "resolve_iluvatar_kv_cache_layout",
    "use_iluvatar_kv_cache_layout",
    "validate_iluvatar_kv_cache_layout",
]
