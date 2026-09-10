# Copyright (c) 2026, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Iluvatar GPU connector factory and minimal vLLM connector implementation."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from lmcache_iluvatar.v1.storage_backend import UnsupportedIluvatarFeatureError

logger = logging.getLogger(__name__)

ILUVATAR_GPU_CONNECTOR_KEYS = {
    "enable_iluvatar_gpu_connector",
    "iluvatar_gpu_connector",
    "use_iluvatar_gpu_connector",
    "enable_iluvatar_kv_transfer",
    "use_iluvatar_kv_transfer",
}

ILUVATAR_GPU_CONNECTOR_VALUE_KEYS = {
    "gpu_connector",
    "gpu_connector_type",
}


from lmcache.v1.gpu_connector.gpu_connectors import GPUConnectorInterface  # pyright: ignore[reportMissingImports]


class VLLMIluvatarGPUConnector(GPUConnectorInterface):  # type: ignore[misc, valid-type]
    """vLLM paged KV connector placeholder for Iluvatar KV transfer adapters."""

    def __init__(self, metadata: Any, use_gpu: bool = False, device: Any = None):
        self.metadata = metadata
        self.use_gpu = use_gpu
        self.device = device
        self.kvcaches = None
        self.kv_shape = getattr(metadata, "kv_shape", None)
        self.kv_dtype = getattr(metadata, "kv_dtype", None)

    @classmethod
    def from_metadata(
        cls, metadata: Any, use_gpu: bool = False, device: Any = None
    ) -> "VLLMIluvatarGPUConnector":
        logger.info(
            "Creating Iluvatar vLLM GPU connector: kv_shape=%s dtype=%s device=%s",
            getattr(metadata, "kv_shape", None),
            getattr(metadata, "kv_dtype", None),
            device,
        )
        return cls(metadata=metadata, use_gpu=use_gpu, device=device)

    def initialize_kvcaches_ptr(self, **kwargs: Any) -> None:
        if "kvcaches" in kwargs:
            self.kvcaches = kwargs["kvcaches"]

    def to_gpu(self, memory_obj: Any, start: int, end: int, **kwargs: Any) -> Any:
        return self.batched_to_gpu([memory_obj], [start], [end], **kwargs)

    def from_gpu(self, memory_obj: Any, start: int, end: int, **kwargs: Any) -> Any:
        return self.batched_from_gpu([memory_obj], [start], [end], **kwargs)

    def batched_to_gpu(
        self,
        memory_objs: Any = None,
        starts: list[int] | None = None,
        ends: list[int] | None = None,
        **kwargs: Any,
    ) -> None:
        self._transfer_batch(
            memory_objs,
            starts,
            ends,
            direction="H2D",
            **kwargs,
        )

    def batched_from_gpu(
        self,
        memory_objs: Any,
        starts: list[int],
        ends: list[int],
        **kwargs: Any,
    ) -> None:
        self._transfer_batch(
            memory_objs,
            starts,
            ends,
            direction="D2H",
            **kwargs,
        )

    def get_shape(self, num_tokens: int) -> Any:
        if self.kv_shape is None:
            return (num_tokens,)
        num_layer, kv_count, _, num_kv_head, head_dim = self.kv_shape
        shape = (num_layer, kv_count, num_tokens, num_kv_head * head_dim)
        try:
            import torch  # pyright: ignore[reportMissingImports]

            return torch.Size(shape)
        except ImportError:
            return shape

    def _transfer_batch(
        self,
        memory_objs: Any,
        starts: list[int] | None,
        ends: list[int] | None,
        *,
        direction: str,
        **kwargs: Any,
    ) -> None:
        starts = starts or []
        ends = ends or []
        if len(starts) != len(ends):
            raise ValueError(
                f"starts/ends length mismatch for {direction}: "
                f"{len(starts)} != {len(ends)}"
            )
        if "slot_mapping" not in kwargs:
            raise ValueError(
                f"slot_mapping is required for Iluvatar KV {direction} transfer"
            )

        kvcaches = kwargs.get("kvcaches", self.kvcaches)
        if kvcaches is None:
            raise ValueError(
                f"kvcaches must be initialized before Iluvatar KV {direction} transfer"
            )

        if not starts:
            return

        logger.error(
            "Iluvatar KV %s transfer requested before an external op adapter "
            "is implemented: starts=%s ends=%s device=%s",
            direction,
            starts,
            ends,
            self.device,
        )
        raise UnsupportedIluvatarFeatureError(
            f"Iluvatar KV {direction} transfer is not implemented in this "
            "lmcache-iluvatar release. This repository does not ship local "
            "C++/kernel sources; add an external op adapter before enabling "
            "Iluvatar KV copy."
        )


def _is_truthy(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on", "iluvatar"}
    return bool(value)


def _is_iluvatar_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip().lower().startswith("iluvatar")
    return False


def is_iluvatar_gpu_connector_requested(config: Any) -> bool:
    """Return true when config explicitly opts into Iluvatar KV transfer."""

    extra_config = getattr(config, "extra_config", None) or {}
    for key in ILUVATAR_GPU_CONNECTOR_KEYS:
        if _is_truthy(getattr(config, key, False)):
            return True
        if isinstance(extra_config, dict) and _is_truthy(extra_config.get(key)):
            return True

    for key in ILUVATAR_GPU_CONNECTOR_VALUE_KEYS:
        if _is_iluvatar_value(getattr(config, key, None)):
            return True
        if isinstance(extra_config, dict) and _is_iluvatar_value(extra_config.get(key)):
            return True

    return False


def CreateIluvatarGPUConnector(config: Any, metadata: Any, engine: Any) -> Any:
    """Create an Iluvatar connector for LMCache's vLLM engine path."""

    engine_name = getattr(engine, "name", str(engine)).upper()
    if engine_name != "VLLM":
        raise RuntimeError(
            "lmcache-iluvatar currently supports only LMCache EngineType.VLLM; "
            f"got engine={engine!r}"
        )

    use_gpu = bool(
        getattr(config, "use_gpu", False) or getattr(config, "use_gpu_buffer", False)
    )
    return VLLMIluvatarGPUConnector.from_metadata(metadata, use_gpu=use_gpu)


def build_iluvatar_gpu_connector_factory(
    upstream_factory: Callable[..., Any] | None,
) -> Callable[..., Any]:
    """Wrap LMCache's GPU connector factory without changing default behavior."""

    def CreateIluvatarGPUConnectorFactory(*args: Any, **kwargs: Any) -> Any:
        config = args[0] if args else kwargs.get("config")
        if is_iluvatar_gpu_connector_requested(config):
            logger.error(
                "Iluvatar GPU connector was requested but KV transfer is not "
                "implemented in this release."
            )
            return CreateIluvatarGPUConnector(*args, **kwargs)
        if upstream_factory is None:
            raise UnsupportedIluvatarFeatureError(
                "LMCache CreateGPUConnector is unavailable; cannot delegate "
                "non-Iluvatar GPU connector factory call."
            )
        return upstream_factory(*args, **kwargs)

    CreateIluvatarGPUConnectorFactory.__name__ = "CreateIluvatarGPUConnector"
    CreateIluvatarGPUConnectorFactory.__qualname__ = "CreateIluvatarGPUConnector"
    CreateIluvatarGPUConnectorFactory.__module__ = __name__
    setattr(
        CreateIluvatarGPUConnectorFactory,
        "__lmcache_iluvatar_original__",
        upstream_factory,
    )
    return CreateIluvatarGPUConnectorFactory


def build_iluvatar_paged_mem_gpu_connector_v2(upstream_cls: type[Any]) -> type[Any]:
    """Add H2D staging to upstream VLLMPagedMemGPUConnectorV2."""

    if getattr(upstream_cls, "__lmcache_iluvatar_h2d_staging__", False):
        return upstream_cls

    class IluvatarPagedMemGPUConnectorV2(upstream_cls):  # type: ignore[misc, valid-type]
        __lmcache_iluvatar_h2d_staging__ = True

        def to_gpu(self, memory_obj: Any, start: int, end: int, **kwargs: Any) -> Any:
            assert memory_obj.tensor is not None

            self.initialize_kvcaches_ptr(**kwargs)
            assert self.kvcaches is not None, (
                "kvcaches should be provided in kwargs or initialized beforehand."
            )

            if "slot_mapping" not in kwargs:
                raise ValueError("'slot_mapping' should be provided in kwargs.")

            slot_mapping = kwargs["slot_mapping"]
            kv_cache_pointers = self._initialize_pointers(self.kvcaches)
            num_tokens = end - start
            will_stage = (
                self.gpu_buffer is not None
                and num_tokens == self.gpu_buffer.shape[2]
            )

            if not will_stage:
                return super().to_gpu(memory_obj, start, end, **kwargs)

            assert self.gpu_buffer.device == self.kvcaches[0].device
            tmp_gpu_buffer = self.gpu_buffer[:, :, :num_tokens, :]
            tmp_gpu_buffer.copy_(memory_obj.tensor, non_blocking=True)

            vllm_cached = kwargs.get("vllm_cached_tokens", 0)
            skip_prefix_n_tokens = min(num_tokens, max(0, vllm_cached - start))

            import lmcache.c_ops as lmc_ops  # pyright: ignore[reportMissingImports]

            # The kernel treats skip_prefix_n_tokens as chunk-relative, so the
            # sliced slot_mapping and staged key_value buffer share one origin.
            lmc_ops.multi_layer_kv_transfer(
                tmp_gpu_buffer,
                kv_cache_pointers,
                slot_mapping[start:end],
                self.device,
                self.page_buffer_size,
                lmc_ops.TransferDirection.H2D,
                self.engine_kv_format,
                block_size=self.block_size,
                head_size=self.head_size,
                skip_prefix_n_tokens=skip_prefix_n_tokens,
            )
            return None

    IluvatarPagedMemGPUConnectorV2.__name__ = upstream_cls.__name__
    IluvatarPagedMemGPUConnectorV2.__qualname__ = upstream_cls.__qualname__
    IluvatarPagedMemGPUConnectorV2.__module__ = upstream_cls.__module__
    setattr(IluvatarPagedMemGPUConnectorV2, "__lmcache_iluvatar_original__", upstream_cls)
    return IluvatarPagedMemGPUConnectorV2


__all__ = [
    "CreateIluvatarGPUConnector",
    "ILUVATAR_GPU_CONNECTOR_KEYS",
    "VLLMIluvatarGPUConnector",
    "build_iluvatar_gpu_connector_factory",
    "build_iluvatar_paged_mem_gpu_connector_v2",
    "is_iluvatar_gpu_connector_requested",
]

