# Copyright (c) 2026, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""LMCache 0.5.3 patches for rank-aware MP L2 prefetch layouts."""

from __future__ import annotations

import inspect
from contextvars import ContextVar
from dataclasses import dataclass
from functools import wraps
from itertools import groupby
from typing import Any


_REGISTER_WORKER_ID: ContextVar[int | None] = ContextVar(
    "lmcache_iluvatar_register_worker_id", default=None
)
_REGISTER_KV_RANK: ContextVar[int | None] = ContextVar(
    "lmcache_iluvatar_register_kv_rank", default=None
)
_RELEASE_KV_RANKS: ContextVar[list[int] | None] = ContextVar(
    "lmcache_iluvatar_release_kv_ranks", default=None
)
_PREFETCH_RANK_LAYOUTS: ContextVar[dict[int, dict[int, Any]] | None] = ContextVar(
    "lmcache_iluvatar_prefetch_rank_layouts", default=None
)


def _kv_rank(world_size: int, worker_id: int) -> int:
    from lmcache.v1.distributed.api import ObjectKey

    return ObjectKey.ComputeKVRank(
        world_size, worker_id, world_size, worker_id
    )


def build_rank_aware_get_payload_classes(current: Any) -> Any:
    """Expose worker_id in the cached 0.5.3 REGISTER_KV_CACHE wire schema."""

    @wraps(current)
    def get_payload_classes(request_type):
        payload_classes = current(request_type)
        if request_type.name != "REGISTER_KV_CACHE":
            return payload_classes
        if len(payload_classes) == 7:
            return [*payload_classes, int]
        if len(payload_classes) != 8:
            raise RuntimeError(
                "Unexpected LMCache 0.5.3 REGISTER_KV_CACHE payload shape: "
                f"{len(payload_classes)} fields"
            )
        return payload_classes

    return get_payload_classes


def build_rank_aware_worker_adapter(current: Any) -> Any:
    """Expose the adapter's kv_worker_id while it registers its transfer context."""

    original = current._send_register_kv_caches_request
    if getattr(original, "__lmcache_iluvatar_rank_layout__", False):
        return current

    @wraps(original)
    def send_register(self, kv_caches):
        token = _REGISTER_WORKER_ID.set(self.worker_id)
        try:
            return original(self, kv_caches)
        finally:
            _REGISTER_WORKER_ID.reset(token)

    send_register.__lmcache_iluvatar_rank_layout__ = True
    current._send_register_kv_caches_request = send_register
    return current


def build_rank_aware_worker_transfer(current: Any) -> Any:
    """Add worker_id to REGISTER without copying the transfer implementation."""

    original = current.register
    if getattr(original, "__lmcache_iluvatar_rank_layout__", False):
        return current

    @wraps(original)
    def register(self, *args, **kwargs):
        worker_id = _REGISTER_WORKER_ID.get()
        if worker_id is None:
            raise RuntimeError(
                "LMCache REGISTER_KV_CACHE requires kv_worker_id on lmcache 0.5.3"
            )

        def add_worker_id(send_request):
            @wraps(send_request)
            def wrapped_send(mq_client, request_type, payload):
                if request_type.name == "REGISTER_KV_CACHE":
                    payload = [*payload, worker_id]
                return send_request(mq_client, request_type, payload)

            return wrapped_send

        if "send_request" in kwargs:
            kwargs["send_request"] = add_worker_id(kwargs["send_request"])
        elif len(args) > 7:
            args = (*args[:7], add_worker_id(args[7]), *args[8:])
        else:
            raise RuntimeError("REGISTER_KV_CACHE send_request argument is missing")
        return original(self, *args, **kwargs)

    register.__lmcache_iluvatar_rank_layout__ = True
    current.register = register
    return current


def build_rank_aware_layout_registry(current: Any) -> Any:
    """Merge group layouts by kv_rank while retaining the 0.5.3 flat API."""

    original_register = current.register
    original_unregister = current.unregister
    if getattr(original_register, "__lmcache_iluvatar_rank_layout__", False):
        return current

    @wraps(original_register)
    def register(
        self,
        model_name,
        world_size,
        layout_desc,
        attn_desc=None,
        group_layout_descs=None,
        *,
        worker_id=None,
    ):
        if attn_desc is None:
            from lmcache.v1.distributed.api import DEFAULT_ATTN_WINDOW_DESC

            attn_desc = DEFAULT_ATTN_WINDOW_DESC
        original_register(
            self,
            model_name,
            world_size,
            layout_desc,
            attn_desc,
            group_layout_descs,
        )
        rank = (
            _kv_rank(world_size, worker_id)
            if worker_id is not None
            else _REGISTER_KV_RANK.get()
        )
        if rank is None:
            return
        with self._lock:
            entry = self._registry[(model_name, world_size)]
            rank_layouts = getattr(entry, "rank_group_layout_descs", None)
            if rank_layouts is None:
                rank_layouts = {}
                entry.rank_group_layout_descs = rank_layouts
            rank_layouts[rank] = dict(entry.group_layout_descs)

    @wraps(original_unregister)
    def unregister(self, model_name, world_size, *, kv_rank=None):
        if kv_rank is None:
            ranks = _RELEASE_KV_RANKS.get()
            if ranks:
                kv_rank = ranks.pop(0)
        original_unregister(self, model_name, world_size)
        if kv_rank is None:
            return
        with self._lock:
            entry = self._registry.get((model_name, world_size))
            if entry is not None:
                getattr(entry, "rank_group_layout_descs", {}).pop(kv_rank, None)

    def find_rank_group_layout_descs(self, model_name, world_size):
        with self._lock:
            entry = self._registry.get((model_name, world_size))
            if entry is None:
                return None
            rank_layouts = getattr(entry, "rank_group_layout_descs", None)
            if rank_layouts is None:
                return None
            return {
                rank: dict(group_layouts)
                for rank, group_layouts in rank_layouts.items()
            }

    register.__lmcache_iluvatar_rank_layout__ = True
    current.register = register
    current.unregister = unregister
    current.find_rank_group_layout_descs = find_rank_group_layout_descs
    return current


def build_rank_aware_transfer_module(current: Any) -> Any:
    """Bind server registrations and releases to their computed kv_rank."""

    original_register = current.register_kv_cache
    original_release = current._release_entries
    if getattr(original_register, "__lmcache_iluvatar_rank_layout__", False):
        return current

    # Do not @wraps the original: MQ validates handlers via inspect.signature /
    # get_type_hints against get_payload_classes. Copying the 0.5.3 7-arg
    # signature would reject the worker_id wire field we append.
    def register_kv_cache(
        self,
        instance_id: int,
        kv_caches,
        model_name: str,
        world_size: int,
        engine_type,
        layout_hints,
        engine_group_infos,
        worker_id: int,
    ) -> None:
        rank = _kv_rank(world_size, worker_id)
        token = _REGISTER_KV_RANK.set(rank)
        try:
            result = original_register(
                self,
                instance_id,
                kv_caches,
                model_name,
                world_size,
                engine_type,
                layout_hints,
                engine_group_infos,
            )
        finally:
            _REGISTER_KV_RANK.reset(token)
        with self._lock:
            entry = self._cache_contexts.get(instance_id)
            if entry is not None:
                entry.kv_rank = rank
        return result

    # Preserve original annotations for the first 7 payload fields so MQ's
    # type check still matches ProtocolDefinition classes.
    orig_hints = getattr(original_register, "__annotations__", {})
    register_kv_cache.__annotations__ = {
        **orig_hints,
        "worker_id": int,
        "return": type(None),
    }
    orig_sig = inspect.signature(original_register)
    params = [
        p
        for name, p in orig_sig.parameters.items()
        if name != "self"
    ]
    worker_param = inspect.Parameter(
        "worker_id",
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        annotation=int,
    )
    self_param = next(iter(orig_sig.parameters.values()))
    register_kv_cache.__signature__ = orig_sig.replace(
        parameters=[self_param, *params, worker_param]
    )
    register_kv_cache.__name__ = original_register.__name__
    register_kv_cache.__qualname__ = original_register.__qualname__
    register_kv_cache.__doc__ = original_register.__doc__
    register_kv_cache.__module__ = original_register.__module__

    @wraps(original_release)
    def release_entries(self, entries):
        ranks = [getattr(entry, "kv_rank", None) for entry in entries]
        token = _RELEASE_KV_RANKS.set(ranks)
        try:
            return original_release(self, entries)
        finally:
            _RELEASE_KV_RANKS.reset(token)

    register_kv_cache.__lmcache_iluvatar_rank_layout__ = True
    current.register_kv_cache = register_kv_cache
    current._release_entries = release_entries
    return current


def build_rank_aware_prefetch_spec(current: Any) -> Any:
    """Extend PrefetchRequestSpec with optional per-rank group layouts."""

    @dataclass(frozen=True)
    class RankAwarePrefetchRequestSpec(current):
        rank_group_layout_descs: dict[int, dict[int, Any]] | None = None

        def __post_init__(self):
            super().__post_init__()
            rank_layouts = self.rank_group_layout_descs
            if rank_layouts is None:
                rank_layouts = _PREFETCH_RANK_LAYOUTS.get()
                if rank_layouts is not None:
                    object.__setattr__(self, "rank_group_layout_descs", rank_layouts)
            if rank_layouts is None:
                return
            expected_groups = set(self.group_layout_descs)
            for rank, group_layouts in rank_layouts.items():
                if set(group_layouts) != expected_groups:
                    raise ValueError(
                        "PrefetchRequestSpec: rank_group_layout_descs for kv_rank "
                        f"{rank} must map groups {sorted(expected_groups)}, got "
                        f"{sorted(group_layouts)}"
                    )

    RankAwarePrefetchRequestSpec.__name__ = current.__name__
    RankAwarePrefetchRequestSpec.__qualname__ = current.__qualname__
    return RankAwarePrefetchRequestSpec


def build_rank_aware_lookup_module(current: Any) -> Any:
    """Validate expanded ranks and attach heterogeneous layouts to prefetch."""

    original = current.lookup
    if getattr(original, "__lmcache_iluvatar_rank_layout__", False):
        return current

    @wraps(original)
    def lookup(self, key, tp_size):
        registry = self._ctx.layout_desc_registry
        if registry.find(key.model_name, key.world_size) is None:
            return original(self, key, tp_size)
        rank_layouts = registry.find_rank_group_layout_descs(
            key.model_name, key.world_size
        )
        expected = {
            _kv_rank(key.world_size, worker_id)
            for worker_id in range(key.world_size)
        }
        present = set(rank_layouts or {})
        missing = expected - present
        if missing:
            raise RuntimeError(
                "Missing per-kv_rank layout descriptors for expanded MP lookup: "
                f"model={key.model_name!r} world_size={key.world_size} "
                f"missing={sorted(missing)}"
            )
        layouts = list(rank_layouts.values())
        nested = None if all(item == layouts[0] for item in layouts[1:]) else rank_layouts
        token = _PREFETCH_RANK_LAYOUTS.set(nested)
        try:
            return original(self, key, tp_size)
        finally:
            _PREFETCH_RANK_LAYOUTS.reset(token)

    lookup.__lmcache_iluvatar_rank_layout__ = True
    current.lookup = lookup
    return current


class _RankAwareL1Manager:
    def __init__(self, manager: Any, rank_layouts: dict[int, dict[int, Any]]):
        self._manager = manager
        self._rank_layouts = rank_layouts

    def __getattr__(self, name: str) -> Any:
        return getattr(self._manager, name)

    def reserve_write(self, *, keys, is_temporary, layout_desc, mode):
        temporary = dict(zip(keys, is_temporary, strict=True))
        results = {}
        ordered = sorted(keys, key=lambda key: (key.kv_rank, key.object_group_id))
        for (rank, gid), grouped in groupby(
            ordered, key=lambda key: (key.kv_rank, key.object_group_id)
        ):
            group_keys = list(grouped)
            results.update(
                self._manager.reserve_write(
                    keys=group_keys,
                    is_temporary=[temporary[key] for key in group_keys],
                    layout_desc=self._rank_layouts[rank][gid],
                    mode=mode,
                )
            )
        return results


def build_rank_aware_prefetch_controller(current: Any) -> Any:
    """Reserve L1 load buffers by (kv_rank, object_group_id)."""

    original_start = current._start_lookup_phase
    original_reserve = current._reserve_load_buffers
    if getattr(original_reserve, "__lmcache_iluvatar_rank_layout__", False):
        return current

    @wraps(original_start)
    def start_lookup_phase(self, request_id, spec):
        result = original_start(self, request_id, spec)
        request = self._in_flight_requests.get(request_id)
        if request is not None:
            request.rank_group_layout_descs = getattr(
                spec, "rank_group_layout_descs", None
            )
        return result

    @wraps(original_reserve)
    def reserve_load_buffers(self, request, keys_to_reserve):
        rank_layouts = getattr(request, "rank_group_layout_descs", None)
        if rank_layouts is None:
            return original_reserve(self, request, keys_to_reserve)
        manager = self._l1_manager
        self._l1_manager = _RankAwareL1Manager(manager, rank_layouts)
        try:
            return original_reserve(self, request, keys_to_reserve)
        finally:
            self._l1_manager = manager

    reserve_load_buffers.__lmcache_iluvatar_rank_layout__ = True
    current._start_lookup_phase = start_lookup_phase
    current._reserve_load_buffers = reserve_load_buffers
    return current
