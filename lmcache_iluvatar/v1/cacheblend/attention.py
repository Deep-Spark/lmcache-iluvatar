# Copyright (c) 2026, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Iluvatar attention backend support for LMCache CacheBlend."""

from __future__ import annotations

from functools import wraps
from typing import Any, Callable, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from lmcache.v1.compute.attention.metadata import LMCAttnMetadata


class LMCIluFlashAttnBackend:
    """CacheBlend attention adapter for vLLM's Iluvatar FlashAttention impl."""

    def __init__(self, vllm_attn: torch.nn.Module):
        self.vllm_attn = vllm_attn
        self.vllm_attn_impl = vllm_attn.impl
        self.aot_schedule = False

    def forward_contiguous(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        output: torch.Tensor,
        attn_metadata: "LMCAttnMetadata",
        **kwargs: Any,
    ) -> torch.Tensor:
        from lmcache.v1.compute.attention.metadata import LMCFlashAttnMetadata
        from vllm.v1.attention.backends.fa_utils import flash_attn_varlen_func

        assert isinstance(attn_metadata, LMCFlashAttnMetadata)

        sliding_window = getattr(self.vllm_attn_impl, "sliding_window", None)
        window_size = list(sliding_window) if sliding_window is not None else None

        flash_attn_varlen_func(
            q=query,
            k=key,
            v=value,
            out=output,
            cu_seqlens_q=attn_metadata.query_start_loc,
            cu_seqlens_k=attn_metadata.cu_seqlens_k,
            max_seqlen_q=attn_metadata.max_query_len,
            max_seqlen_k=attn_metadata.max_seq_len,
            softmax_scale=self.vllm_attn_impl.scale,
            causal=True,
            alibi_slopes=self.vllm_attn_impl.alibi_slopes,
            window_size=window_size,
            softcap=self.vllm_attn_impl.logits_soft_cap,
            sinks=getattr(self.vllm_attn_impl, "sinks", None),
        )

        return output

    def init_attn_metadata(
        self,
        input_ids: torch.Tensor,
        **kwargs: Any,
    ) -> "LMCAttnMetadata":
        from lmcache.v1.compute.attention.metadata import LMCFlashAttnMetadata

        seq_len = input_ids.shape[0]
        device = input_ids.device
        return LMCFlashAttnMetadata(
            query_start_loc=torch.tensor(
                [0, seq_len],
                dtype=torch.int32,
                device=device,
            ),
            seq_lens=torch.tensor([seq_len], device=device),
            cu_seqlens_k=torch.tensor(
                [0, seq_len],
                dtype=torch.int32,
                device=device,
            ),
            max_query_len=seq_len,
            max_seq_len=seq_len,
        )


def build_iluvatar_attn_backend_infer(
    upstream_infer: Callable[[torch.nn.Module, bool], Any],
) -> Callable[[torch.nn.Module, bool], Any]:
    """Recognize Iluvatar FlashAttention while preserving upstream behavior."""

    @wraps(upstream_infer)
    def infer_attn_backend_from_vllm(
        vllm_attn: torch.nn.Module,
        enable_sparse: bool = False,
    ) -> Any:
        attn_name = type(vllm_attn.impl).__name__
        if attn_name == "IluFlashAttentionImpl" and not enable_sparse:
            return LMCIluFlashAttnBackend(vllm_attn)
        return upstream_infer(vllm_attn, enable_sparse)

    return infer_attn_backend_from_vllm


__all__ = ["LMCIluFlashAttnBackend", "build_iluvatar_attn_backend_infer"]
