# Copyright (c) 2026, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""vLLM MultiConnector patch: pass real blocks to all children for save tracking.

Upstream ``MultiConnector.update_state_after_alloc`` only forwards real
``blocks`` to the connector chosen for *load*. Non-chosen connectors (and the
cold-miss case ``chosen=-1``) receive empty blocks.

That breaks LMCache MP store under ``MultiConnector(MP + NixlPush)``: on cold
miss nobody is chosen, so LMCache never records the first chunked-prefill
window's block IDs and systematically under-stores ~``max_num_batched_tokens``
(default 2048). Warm lookups then PREFIX-stop at the hole.

MultiConnector's own contract is "load from the first hit connector, save to
all". This patch keeps load assignment (``num_external_tokens`` only on the
chosen connector) but always forwards real ``blocks`` so save-side tracking
works for every child.
"""

from __future__ import annotations

import logging
from typing import TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=type)

_MARKER = "__lmcache_iluvatar_multiconnector_save_blocks__"


def build_multiconnector_save_blocks(base_cls: T) -> T:
    """Patch ``MultiConnector.update_state_after_alloc`` for save-to-all blocks."""

    if getattr(base_cls, _MARKER, False):
        return base_cls

    original = base_cls.update_state_after_alloc

    def update_state_after_alloc(
        self,
        request,
        blocks,
        num_external_tokens: int,
    ):  # noqa: ANN001
        chosen_connector = self._requests_to_connector.get(request.request_id, -1)
        for i, connector in enumerate(self._connectors):
            if i == chosen_connector:
                connector.update_state_after_alloc(
                    request, blocks, num_external_tokens
                )
            else:
                # Save-to-all: real blocks for allocation tracking; no external load.
                connector.update_state_after_alloc(request, blocks, 0)

    setattr(update_state_after_alloc, "__wrapped__", original)
    setattr(base_cls, "update_state_after_alloc", update_state_after_alloc)
    setattr(base_cls, _MARKER, True)
    logger.info(
        "Patched vLLM MultiConnector.update_state_after_alloc for save-to-all "
        "block tracking (cold-miss LMCache store fix)"
    )
    return base_cls


__all__ = ["build_multiconnector_save_blocks"]
