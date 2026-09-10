# Copyright (c) 2026, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""EICConnector compatibility hooks (CSUPPORT-7101)."""

from __future__ import annotations

import asyncio
import logging
import os
from typing import TYPE_CHECKING, Any, TypeVar

import yaml

if TYPE_CHECKING:
    from lmcache.utils import CacheEngineKey
    from lmcache.v1.memory_management import MemoryObj
    from lmcache.v1.protocol import RemoteMetadata

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=type)


def _parse_eic_busy_loop(value: Any) -> bool:
    if isinstance(value, str):
        return value.lower() in ("1", "true", "yes", "on")
    return bool(value)


def _read_eic_busy_loop_from_config() -> bool:
    config_file = os.getenv("LMCACHE_CONFIG_FILE")
    if not config_file:
        return False
    with open(config_file, "r", encoding="utf-8") as fin:
        config = yaml.safe_load(fin) or {}
    extra_config = config.get("extra_config") or {}
    return _parse_eic_busy_loop(extra_config.get("eic_get_data_busy_loop", False))


def build_iluvatar_eic_connector(base_cls: T) -> T:
    """Create an EICConnector subclass with namespace and prefetch fixes."""

    if getattr(base_cls, "__lmcache_iluvatar_eic_connector__", False):
        return base_cls

    import eic

    class IluvatarEICConnector(base_cls):  # type: ignore[misc, valid-type]
        __lmcache_iluvatar_eic_connector__ = True

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            if not hasattr(self, "eic_get_data_busy_loop"):
                self.eic_get_data_busy_loop = _read_eic_busy_loop_from_config()
                logger.info(
                    "eic get_data busy_loop: %s", self.eic_get_data_busy_loop
                )

        def _exists_sync(self, key_str: str) -> bool:
            keys = eic.StringVector()
            keys.append(key_str)
            exist_option = eic.ExistOption()
            exist_option.ns = self.eic_kv_ns
            status_code, exist_outcome = self.connection.mexist(keys, exist_option)
            if status_code != eic.StatusCode.SUCCESS:
                logger.debug(
                    "eic exists %s failed, status_code %s", key_str, status_code
                )
                return False

            err_code = exist_outcome.status_codes[0]
            success = err_code == eic.StatusCode.SUCCESS
            if success:
                logger.debug("eic exists %s success", key_str)
            else:
                logger.debug(
                    "eic exists %s failed, status_code %s err_code %s",
                    key_str,
                    status_code,
                    err_code,
                )
            return success

        async def _batched_async_contains(
            self,
            lookup_id: str,
            keys: list["CacheEngineKey"],
            pin: bool = False,
        ) -> int:
            if not keys:
                return 0

            key_strings = eic.StringVector()
            for key in keys:
                key_strings.append(key.to_string())

            exist_option = eic.ExistOption()
            exist_option.ns = self.eic_kv_ns
            status_code, exist_outcome = self.connection.mexist(key_strings, exist_option)

            if status_code != eic.StatusCode.SUCCESS:
                logger.error(
                    "eic batched_async_contains mexist failed, status_code %s",
                    status_code,
                )
                return 0

            num_hit_counts = 0
            for index, key in enumerate(keys):
                key_status = exist_outcome.status_codes[index]
                if key_status != eic.StatusCode.SUCCESS:
                    logger.debug(
                        "eic batched_async_contains %s miss, err_code %s",
                        key.to_string(),
                        key_status,
                    )
                    break
                num_hit_counts += 1
            return num_hit_counts

        async def get_data(
            self, key_str: str, meta: "RemoteMetadata"
        ) -> "MemoryObj | None":
            from importlib import import_module

            perf_timer_cls = import_module(base_cls.__module__).PerformanceTimer
            perf_timer = perf_timer_cls(key_str, "get_data")
            perf_timer.start("total_cost")
            perf_timer.start("alloc_obj")
            busy_loop = getattr(self, "eic_get_data_busy_loop", False)
            try:
                memory_obj = self.memory_allocator.allocate(
                    meta.shapes,
                    meta.dtypes,
                    meta.fmt,
                    busy_loop=busy_loop,
                )
            except TypeError:
                memory_obj = self.memory_allocator.allocate(
                    meta.shapes,
                    meta.dtypes,
                    meta.fmt,
                )
            if memory_obj is None:
                logger.warning(
                    "fail to allocate memory during remote receive key %s length %s; "
                    "busy_loop=%s",
                    key_str,
                    meta.length,
                    busy_loop,
                )
                return None
            perf_timer.stop("alloc_obj")

            perf_timer.start("alloc_mem")
            obj_size = memory_obj.get_size()
            data_ptr = memory_obj.tensor.data_ptr()
            data_keys = eic.StringVector()
            data_vals = eic.IOBuffers()
            data_keys.append(key_str)

            perf_timer.set_size(obj_size)
            perf_timer.stop("alloc_mem")

            try:
                if self.trans_type == eic.TransportType.TRANSPORT_GDR:
                    data_vals.append(data_ptr, obj_size, True)
                else:
                    data_vals.append(data_ptr, obj_size, False)

                perf_timer.start("eic_mget")
                get_option = eic.GetOption()
                get_option.ns = self.eic_kv_ns
                status_code, data_vals, get_outcome = self.connection.mget(
                    data_keys, get_option, data_vals
                )
                err_code = get_outcome.status_codes[0]
                if (
                    status_code != eic.StatusCode.SUCCESS
                    or err_code != eic.StatusCode.SUCCESS
                ):
                    logger.error(
                        "eic mget data %s failed, status_code %s err_code %s",
                        key_str,
                        status_code,
                        err_code,
                    )
                    memory_obj.ref_count_down()
                    return None
                logger.debug("eic mget data %s success", key_str)
            except Exception as exc:
                logger.error(
                    "eic mget data %s raised exception: %s",
                    key_str,
                    exc,
                    exc_info=True,
                )
                memory_obj.ref_count_down()
                return None

            perf_timer.stop("eic_mget")
            perf_timer.stop("total_cost")
            perf_timer.debug_all_elapsed_times()
            return memory_obj

        async def _batched_get_non_blocking(
            self,
            lookup_id: str,
            keys: list["CacheEngineKey"],
        ) -> list["MemoryObj"]:
            results = await asyncio.gather(
                *(self._get(key) for key in keys),
                return_exceptions=True,
            )
            memory_objs: list[MemoryObj] = []
            for key, result in zip(keys, results, strict=False):
                if isinstance(result, Exception):
                    logger.error(
                        "batched_get_non_blocking key failed: %s, exc=%r",
                        key.to_string(),
                        result,
                    )
                    continue
                if result is not None:
                    memory_objs.append(result)
            return memory_objs

    IluvatarEICConnector.__name__ = "IluvatarEICConnector"
    IluvatarEICConnector.__qualname__ = "IluvatarEICConnector"
    IluvatarEICConnector.__module__ = __name__
    return IluvatarEICConnector  # type: ignore[return-value]


__all__ = ["build_iluvatar_eic_connector"]
