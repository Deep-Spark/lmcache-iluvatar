# Copyright (c) 2026, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""PP cache broadcast helpers (CSUPPORT-7101)."""

from __future__ import annotations

import logging
from typing import Any, Union

import torch

logger = logging.getLogger(__name__)

BROADCAST_DTYPE = torch.int32
BROADCAST_DTYPE_SIZE_BYTES = 4


def resolve_broadcast_src_rank(metadata: Any) -> int:
    """Return collective broadcast src rank with v0.4.5 metadata fallback."""

    return getattr(
        metadata,
        "broadcast_src_rank",
        getattr(metadata, "first_rank", 0),
    )


def get_padded_broadcast_size(num_bytes: int) -> int:
    return (
        (num_bytes + BROADCAST_DTYPE_SIZE_BYTES - 1)
        // BROADCAST_DTYPE_SIZE_BYTES
        * BROADCAST_DTYPE_SIZE_BYTES
    )


def pack_uint8_tensor_for_broadcast(raw_tensor: torch.Tensor) -> torch.Tensor:
    if raw_tensor.dtype != torch.uint8:
        raise TypeError(
            "Expected uint8 raw tensor for LMCache broadcast, "
            f"got {raw_tensor.dtype}"
        )
    if not raw_tensor.is_contiguous():
        raw_tensor = raw_tensor.contiguous()
    elif raw_tensor.storage_offset() % BROADCAST_DTYPE_SIZE_BYTES != 0:
        raw_tensor = raw_tensor.clone()

    num_bytes = raw_tensor.numel()
    padded_num_bytes = get_padded_broadcast_size(num_bytes)
    if padded_num_bytes == num_bytes:
        return raw_tensor.view(BROADCAST_DTYPE)

    padded_tensor = torch.empty(
        torch.Size([padded_num_bytes]),
        dtype=torch.uint8,
        device=raw_tensor.device,
    )
    padded_tensor[:num_bytes].copy_(raw_tensor)
    padded_tensor[num_bytes:].zero_()
    return padded_tensor.view(BROADCAST_DTYPE)


def allocate_uint8_broadcast_receive_tensors(
    num_bytes: int,
    device: Union[str, torch.device],
) -> tuple[torch.Tensor, torch.Tensor]:
    raw_tensor = torch.empty(
        torch.Size([num_bytes]),
        dtype=torch.uint8,
        device=device,
    )
    padded_num_bytes = get_padded_broadcast_size(num_bytes)
    if padded_num_bytes == num_bytes:
        return raw_tensor, raw_tensor.view(BROADCAST_DTYPE)

    broadcast_tensor = torch.empty(
        torch.Size([padded_num_bytes]),
        dtype=torch.uint8,
        device=device,
    ).view(BROADCAST_DTYPE)
    return raw_tensor, broadcast_tensor


def unpack_broadcast_tensor_to_uint8(
    raw_tensor: torch.Tensor,
    broadcast_tensor: torch.Tensor,
) -> None:
    if broadcast_tensor.numel() * BROADCAST_DTYPE_SIZE_BYTES == raw_tensor.numel():
        return
    raw_tensor.copy_(broadcast_tensor.view(torch.uint8)[: raw_tensor.numel()])


def broadcast_or_receive_memory_objs(
    engine: Any,
    reordered_chunks: list,
    ret_mask: torch.Tensor,
) -> None:
    """Broadcast or receive memory objects using PP-safe src rank and packing."""

    from lmcache import torch_dev, torch_device_type
    from lmcache.v1.memory_management import MemoryObjMetadata, TensorMemoryObj

    metadata = engine.metadata
    broadcast_src_rank = resolve_broadcast_src_rank(metadata)

    if metadata.is_first_rank():
        chunk_count = len(reordered_chunks)
        engine.broadcast_object_fn(chunk_count, broadcast_src_rank)

        for _key, memory_obj, start, end in reordered_chunks:
            metadata_dict = memory_obj.metadata.to_dict()
            combined_metadata = (start, end, metadata_dict)
            engine.broadcast_object_fn(combined_metadata, broadcast_src_rank)

            raw_tensor = memory_obj.raw_tensor
            assert raw_tensor is not None
            tensor_to_broadcast = raw_tensor.to(
                f"{torch_device_type}:{metadata.worker_id}"
            )
            tensor_to_broadcast = pack_uint8_tensor_for_broadcast(tensor_to_broadcast)
            engine.broadcast_fn(tensor_to_broadcast, broadcast_src_rank)
        return

    chunk_count = engine.broadcast_object_fn(None, broadcast_src_rank)
    if chunk_count is None:
        logger.warning("rank=%s received None chunk_count", metadata.worker_id)
        return

    for _ in range(chunk_count):
        combined_metadata = engine.broadcast_object_fn(None, broadcast_src_rank)
        if combined_metadata is None:
            logger.warning(
                "rank=%s received None combined_metadata",
                metadata.worker_id,
            )
            break
        start, end, metadata_dict = combined_metadata
        ret_mask[start:end] = True

        obj_metadata = MemoryObjMetadata.from_dict(metadata_dict)
        local_rank = metadata.worker_id % torch_dev.device_count()
        raw_tensor, tensor_to_receive = allocate_uint8_broadcast_receive_tensors(
            obj_metadata.get_size(),
            f"{torch_device_type}:{local_rank}",
        )
        engine.broadcast_fn(tensor_to_receive, broadcast_src_rank)
        unpack_broadcast_tensor_to_uint8(raw_tensor, tensor_to_receive)

        memory_obj = TensorMemoryObj(
            raw_data=raw_tensor, metadata=obj_metadata, parent_allocator=None
        )
        reordered_chunks.append((None, memory_obj, start, end))


__all__ = [
    "BROADCAST_DTYPE",
    "allocate_uint8_broadcast_receive_tensors",
    "broadcast_or_receive_memory_objs",
    "get_padded_broadcast_size",
    "pack_uint8_tensor_for_broadcast",
    "resolve_broadcast_src_rank",
    "unpack_broadcast_tensor_to_uint8",
]
