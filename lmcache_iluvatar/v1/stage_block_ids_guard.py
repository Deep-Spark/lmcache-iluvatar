# Copyright (c) 2026, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Force blocking Host→Device staging of STORE block IDs on Iluvatar.

Upstream ``BaseCacheContext.stage_block_ids`` builds a transient
``array.array`` / ``torch.frombuffer`` host view and copies it into the
shared GPU ``block_ids_buffer_`` with ``non_blocking=True``. On stacks that
perform a true async DMA from unpinned host memory (observed on Iluvatar),
the host buffer can be freed/reused before the copy completes. The gather
kernel then indexes IPC-mapped KV with garbage block IDs and faults
(``XID:24 mmu page fault``). Exclusive prior-full A/B showed consecutive
same-device STOREs reproduce the fault with non-blocking staging and stay
clean with a blocking ``copy_``.

This module in-place wraps ``BaseCacheContext.stage_block_ids`` so the
plugin does not fork upstream LMCache. It does not touch the rank-aware
MP L2 prefetch layout patches.
"""

from __future__ import annotations

from functools import wraps
from typing import Any
import array
import logging

logger = logging.getLogger(__name__)

_MARKER = "__lmcache_iluvatar_blocking_stage_block_ids__"


def build_blocking_stage_block_ids(cache_context_cls: Any) -> Any:
    """Replace ``stage_block_ids`` with a blocking-copy implementation.

    Args:
        cache_context_cls: Upstream ``BaseCacheContext`` (or compatible) class.

    Returns:
        The same class after in-place method replacement (idempotent).
    """
    if cache_context_cls is None:
        return None
    if getattr(cache_context_cls, _MARKER, False):
        return cache_context_cls

    original = cache_context_cls.stage_block_ids

    @wraps(original)
    def stage_block_ids(
        self: Any, block_ids_per_group: list[list[int]]
    ) -> list[Any]:
        # Third Party
        import torch

        offsets = [0]
        flat: array.array = array.array("q")
        for view_block_ids in block_ids_per_group:
            flat.extend(view_block_ids)
            offsets.append(len(flat))

        total = offsets[-1]
        if total > self.block_ids_buffer_.shape[0]:
            raise ValueError(
                "block ID total %d exceeds the pre-allocated buffer "
                "size %d" % (total, self.block_ids_buffer_.shape[0])
            )
        if total:
            # Blocking copy: host ``flat`` must remain valid until H2D
            # completes. Do not restore non_blocking without pin+retain.
            cpu_tensor = torch.frombuffer(flat, dtype=torch.long)
            self.block_ids_buffer_[:total].copy_(cpu_tensor)

        return [
            self.block_ids_buffer_[offsets[i] : offsets[i + 1]]
            for i in range(len(block_ids_per_group))
        ]

    cache_context_cls.stage_block_ids = stage_block_ids
    setattr(cache_context_cls, _MARKER, True)
    logger.info(
        "Patched %s.stage_block_ids to use blocking Host→Device copy",
        getattr(cache_context_cls, "__name__", type(cache_context_cls).__name__),
    )
    return cache_context_cls
