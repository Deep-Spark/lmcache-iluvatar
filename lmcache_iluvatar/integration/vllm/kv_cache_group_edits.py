# Copyright (c) 2026, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Iluvatar HND-aware vLLM hybrid KV cache group edits for LMCache registration.

Upstream ``_SubpagedAttentionViewEdit`` (LMCache PR #3613) only handles the
NHD kernel-paged layout ``(NB, 2, BS_k, NH, HS)``. Iluvatar flash-attn HND
uses ``(2, NB, NH, BS_k, HS)`` (format 6), which false-matches that rule and
raises:

    ValueError: expected a (num_blocks, 2, block_size, num_heads, head_size)
    attention KV tensor, got (2, ...)

This module:

1. Installs an HND subpaged edit into upstream ``_EDITS`` during
   ``activate_patches()`` (under the patch lock), ahead of the NHD rule.
2. Tightens upstream NHD ``matches`` so ``shape[1] == 2`` is required.
3. Recovers the contiguous block-first physical view behind vLLM's HND
   transpose, then re-views kernel pages as byte-opaque logical pages, using
   **only** same-storage metadata operations.
4. Scopes vLLM's resolved ``layout_hints["kv_layout"]`` across every hybrid
   group edit and rejects missing or conflicting registration layout metadata.

Invariants
----------
- ``apply`` must return a view on the same storage as the vLLM KV tensor
  (upstream KVCacheGroupEdit contract). Non-contiguous tensors that cannot
  be re-interpreted without a copy are rejected.
- Edited views are byte-opaque addressing (PR #3613): CacheGen / CacheBlend
  and cross-backend sharing are not valid for these groups.
- Kernel page size, vLLM logical block size, and LMCache chunk size are
  distinct; this edit only bridges kernel→logical page for registration.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from typing import Any, Protocol, runtime_checkable

import torch

from lmcache_iluvatar.integration.vllm.kv_cache_layout import (
    get_active_iluvatar_kv_cache_layout,
    use_iluvatar_kv_cache_layout,
)

logger = logging.getLogger(__name__)

_MARKER = "__lmcache_iluvatar_kv_cache_group_edits__"
_EDITS_INSTALLED = "__lmcache_iluvatar_hnd_edits_installed__"
_SYNTHETIC_NUM_HEADS = 1
_HND_EDIT_NAME = "iluvatar-hnd-subpaged-attention-view"
_HND_MAMBA_EDIT_NAME = "iluvatar-hnd-mamba-page-view"
_UPSTREAM_NHD_EDIT_NAME = "subpaged-attention-view"
_UPSTREAM_MAMBA_EDIT_NAME = "mamba-page-view"


@runtime_checkable
class SupportsSubpagedSpec(Protocol):
    """Minimal vLLM KV-cache spec fields required for subpaged re-view."""

    block_size: int
    page_size_bytes: int


def _synthetic_attention_shape(elems_per_page: int, block_size: int) -> tuple[int, int]:
    """Factor a page's element count into synthetic ``(num_heads, head_size)``.

    Args:
        elems_per_page: Number of scalar elements in one logical page.
        block_size: Logical tokens per page (``spec.block_size``).

    Returns:
        ``(num_heads, head_size)`` with synthetic ``num_heads == 1``.

    Raises:
        ValueError: If ``elems_per_page`` does not factor cleanly.
    """

    denom = 2 * block_size * _SYNTHETIC_NUM_HEADS
    if elems_per_page % denom != 0:
        raise ValueError(
            f"page ({elems_per_page} elems) does not factor into "
            f"(2, block_size={block_size}, num_heads={_SYNTHETIC_NUM_HEADS}, head_size)"
        )
    return _SYNTHETIC_NUM_HEADS, elems_per_page // denom


def _declares_slot_compression(spec: Any) -> bool:
    return (
        getattr(spec, "compress_ratio", 1) > 1 or getattr(spec, "tq_slot_size", 0) > 0
    )


def _kv_layout_hint() -> str | None:
    """Return the registration-scoped resolved KV layout when available."""

    return get_active_iluvatar_kv_cache_layout().layout


def is_iluvatar_hnd_flash_attn(kv_cache: Any, spec: Any | None = None) -> bool:
    """Return whether ``kv_cache`` is Iluvatar flash-attn HND ``(2, NB, NH, BS, HS)``.

    Args:
        kv_cache: Candidate attention KV tensor.
        spec: Optional KV cache spec; used only when ``shape[0] == shape[1] == 2``
            (ambiguous with NHD ``(NB=2, 2, ...)``).

    Returns:
        True when layout metadata / shape / stride indicate HND flash-attn pages.
    """

    if not isinstance(kv_cache, torch.Tensor) or kv_cache.ndim != 5:
        return False
    if int(kv_cache.shape[0]) != 2:
        return False

    layout = _kv_layout_hint()
    if layout == "NHD":
        return False
    if layout == "HND":
        return True

    # Classic NHD: K/V axis at dim 1, num_blocks at dim 0.
    if int(kv_cache.shape[1]) == 2 and int(kv_cache.shape[0]) != 2:
        return False

    if int(kv_cache.shape[1]) == 2 and int(kv_cache.shape[0]) == 2:
        # Ambiguous (2, 2, ...): disambiguate with logical block size when possible.
        if spec is None or not hasattr(spec, "block_size"):
            return False
        logical = int(spec.block_size)
        dim2 = int(kv_cache.shape[2])
        dim3 = int(kv_cache.shape[3])
        if dim3 != logical and logical % dim3 == 0:
            return True
        if dim2 != logical and logical % dim2 == 0:
            return False
        return False

    # Clear HND flash-attn: (2, NB!=2, NH, BS, HS). Contiguity is enforced
    # in apply() so registration never aliases a copied buffer.
    return True


def needs_hnd_subpaged_edit(spec: Any, kv_cache: Any) -> bool:
    """Return whether an Iluvatar HND tensor needs kernel→logical page re-view.

    Args:
        spec: vLLM group spec with ``block_size`` / ``page_size_bytes``.
        kv_cache: Layer KV tensor or opaque state.

    Returns:
        True when HND kernel block size differs from the logical ``spec.block_size``.
    """

    if _declares_slot_compression(spec) or not is_iluvatar_hnd_flash_attn(
        kv_cache, spec
    ):
        return False
    kernel_block_size = int(kv_cache.shape[3])
    return kernel_block_size != int(spec.block_size)


def _describe_kv_tensor(kv_cache: torch.Tensor) -> str:
    """Compact diagnostic string for registration-time layout debugging."""

    device = str(getattr(kv_cache, "device", "unknown"))
    dtype = str(getattr(kv_cache, "dtype", "unknown"))
    return (
        f"shape={tuple(int(x) for x in kv_cache.shape)} "
        f"stride={tuple(int(x) for x in kv_cache.stride())} "
        f"contiguous={kv_cache.is_contiguous()} "
        f"data_ptr={kv_cache.data_ptr()} "
        f"dtype={dtype} device={device}"
    )


def _describe_registered_kv_cache(kv_cache: Any) -> str:
    if isinstance(kv_cache, torch.Tensor):
        return _describe_kv_tensor(kv_cache)
    return f"type={type(kv_cache).__name__}"


def _log_hybrid_group_layouts(
    kv_cache_config: Any,
    kv_caches: Mapping[str, Any],
    *,
    resolved_layout: str | None,
    source: str,
    env_override: str | None,
) -> None:
    format_abi = {
        "HND": "[NB,2,NH,BS,HS]",
        "NHD": "[NB,2,BS,NH,HS]",
    }.get(resolved_layout, "heuristic")
    logger.info(
        "Hybrid KV cache layout contract: source=%s resolved_layout=%s "
        "env_override=%s format_abi=%s",
        source,
        resolved_layout,
        env_override,
        format_abi,
    )
    if not logger.isEnabledFor(logging.DEBUG):
        return
    for group_index, group in enumerate(
        getattr(kv_cache_config, "kv_cache_groups", ())
    ):
        for layer_name in getattr(group, "layer_names", ()):
            cache = kv_caches.get(layer_name)
            logger.debug(
                "Hybrid KV cache group=%s layer=%s resolved_layout=%s "
                "format_abi=%s %s",
                group_index,
                layer_name,
                resolved_layout,
                format_abi,
                _describe_registered_kv_cache(cache),
            )


def _restore_hnd_block_first_view(kv_cache: torch.Tensor) -> torch.Tensor:
    """Recover the contiguous block-first storage behind vLLM's HND view.

    Iluvatar vLLM allocates the kernel pages as ``(NB, 2, NH, BS, HS)`` and
    exposes them as ``(2, NB, NH, BS, HS)`` by swapping the first two axes.
    Undoing that transpose is metadata-only and restores the physical page
    order required for logical-page grouping.

    Args:
        kv_cache: Exposed HND tensor ``(2, NB, NH, BS, HS)``.

    Returns:
        Contiguous ``(NB, 2, NH, BS, HS)`` view aliasing ``kv_cache`` storage.

    Raises:
        ValueError: If the observed tensor is not the expected transpose of a
            contiguous block-first allocation.
    """
    block_first = kv_cache.permute(1, 0, 2, 3, 4)
    if not block_first.is_contiguous() or block_first.data_ptr() != kv_cache.data_ptr():
        raise ValueError(
            "kernel-paged HND attention KV tensor must be a zero-copy transpose "
            "of contiguous (num_blocks, 2, num_heads, block_size, head_size) "
            "storage; refusing to copy "
            f"({_describe_kv_tensor(kv_cache)} recovered_shape="
            f"{tuple(int(x) for x in block_first.shape)} recovered_stride="
            f"{tuple(int(x) for x in block_first.stride())} "
            f"recovered_contiguous={block_first.is_contiguous()} "
            f"layout_hint={_kv_layout_hint()})"
        )
    logger.info(
        "HND subpaged recovered block-first physical view: %s -> shape=%s stride=%s",
        _describe_kv_tensor(kv_cache),
        tuple(int(x) for x in block_first.shape),
        tuple(int(x) for x in block_first.stride()),
    )
    return block_first


def apply_hnd_subpaged_attention_view(
    spec: Any, kv_cache: torch.Tensor
) -> torch.Tensor:
    """Re-view kernel-paged HND attention KV as logical-block HND pages.

    Args:
        spec: Spec with ``block_size`` (logical tokens) and ``page_size_bytes``.
        kv_cache: Input ``(2, NB_k, NH, BS_k, HS)`` on vLLM storage.

    Returns:
        Byte-opaque HND output ``(NB_logical, 2, NH', logical_BS, HS')`` aliasing
        ``kv_cache``. As in upstream's subpaged edit, the dimensions are
        addressing metadata rather than semantic K/V slabs.

    Raises:
        ValueError: On layout / tiling mismatch, or when a same-storage view
            cannot be formed.
    """

    if not is_iluvatar_hnd_flash_attn(kv_cache, spec):
        raise ValueError(
            "expected a (2, num_blocks, num_heads, block_size, head_size) "
            f"Iluvatar HND attention KV tensor, got {tuple(kv_cache.shape)}"
        )

    logger.info(
        "HND subpaged apply: %s logical_bs=%s page_size_bytes=%s layout_hint=%s",
        _describe_kv_tensor(kv_cache),
        getattr(spec, "block_size", None),
        getattr(spec, "page_size_bytes", None),
        _kv_layout_hint(),
    )

    logical_block_size = int(spec.block_size)
    kernel_block_size = int(kv_cache.shape[3])
    if logical_block_size % kernel_block_size != 0:
        raise ValueError(
            f"logical block size {logical_block_size} is not a multiple of "
            f"kernel block size {kernel_block_size}"
        )
    ratio = logical_block_size // kernel_block_size

    num_kernel_pages = int(kv_cache.shape[1])
    if num_kernel_pages % ratio != 0:
        raise ValueError(
            f"kernel page count {num_kernel_pages} is not a multiple of "
            f"the logical/kernel block ratio {ratio}"
        )

    # One kernel page along the NB axis: (2, 1, NH, BS_k, HS).
    kernel_page_elems = (
        int(kv_cache.shape[0])
        * int(kv_cache.shape[2])
        * int(kv_cache.shape[3])
        * int(kv_cache.shape[4])
    )
    kernel_page_bytes = kernel_page_elems * kv_cache.element_size()
    if kernel_page_bytes * ratio != int(spec.page_size_bytes):
        raise ValueError(
            f"{ratio} kernel pages ({kernel_page_bytes * ratio} bytes) do "
            f"not tile the logical page ({spec.page_size_bytes} bytes)"
        )

    num_blocks = num_kernel_pages // ratio
    elems_per_page = int(spec.page_size_bytes) // kv_cache.element_size()
    num_heads, head_size = _synthetic_attention_shape(
        elems_per_page, logical_block_size
    )
    block_first = _restore_hnd_block_first_view(kv_cache)
    # The explicit HND layout hint makes LMCache detect this block-first view
    # as NL_X_NB_TWO_NH_BS_HS, whose ABI is [NB, 2, NH, BS, HS]. Do not copy
    # upstream's NHD subpaged order [NB, 2, BS, NH, HS]: the MP native kernel
    # reads shape[2] as nh (blockDim.y) and shape[3] as the token block size.
    target_shape = (num_blocks, 2, num_heads, logical_block_size, head_size)
    out = block_first.view(*target_shape)
    logger.info(
        "HND subpaged logical-page view: data_ptr=%s shape=%s stride=%s",
        out.data_ptr(),
        target_shape,
        tuple(int(x) for x in out.stride()),
    )
    return out


def _spec_kind_allows_hnd_subpaged(spec: Any) -> bool | None:
    """Return whether vLLM kind allows subpaged attention edits.

    Returns:
        ``True`` / ``False`` when vLLM kind helpers are importable.
        ``None`` when helpers are unavailable (unit-test fakes).
    """

    try:
        from vllm.v1.kv_cache_interface import (  # type: ignore[import-not-found]
            KVCacheSpecKind,
            get_kv_cache_spec_kind,
        )
    except ImportError:
        return None

    kind = get_kv_cache_spec_kind(spec)
    rejected = {
        KVCacheSpecKind.MAMBA,
        KVCacheSpecKind.CROSS_ATTENTION,
    }
    if kind in rejected:
        return False
    subpageable = {
        KVCacheSpecKind.FULL_ATTENTION,
        KVCacheSpecKind.SLIDING_WINDOW,
        KVCacheSpecKind.CHUNKED_LOCAL_ATTENTION,
        KVCacheSpecKind.SINK_FULL_ATTENTION,
        KVCacheSpecKind.UNKNOWN,
    }
    return kind in subpageable


class _IluvatarHndSubpagedAttentionViewEdit:
    """HND analogue of upstream ``_SubpagedAttentionViewEdit``."""

    name = _HND_EDIT_NAME

    def matches(self, spec: Any, kv_cache: Any) -> bool:
        if not needs_hnd_subpaged_edit(spec, kv_cache):
            return False
        allowed = _spec_kind_allows_hnd_subpaged(spec)
        if allowed is None:
            # No vLLM helpers (fake LMCache tests): require a clear HND shape
            # or an explicit HND layout hint — never accept lists / opaque state.
            if isinstance(kv_cache, list):
                return False
            if _kv_layout_hint() == "HND":
                return True
            return (
                isinstance(kv_cache, torch.Tensor)
                and kv_cache.ndim == 5
                and int(kv_cache.shape[0]) == 2
                and int(kv_cache.shape[1]) != 2
            )
        return allowed

    def apply(
        self, spec: Any, kv_cache: Any, layout_hints: Mapping[str, Any] | None = None
    ) -> torch.Tensor:
        # Matching already consumed the registration-scoped resolved layout;
        # keep the explicit argument only to preserve LMCache v0.5.3's ABI.
        del layout_hints
        if not isinstance(kv_cache, torch.Tensor):
            raise ValueError(
                "expected an Iluvatar HND attention KV tensor, "
                f"got {type(kv_cache).__name__}"
            )
        return apply_hnd_subpaged_attention_view(spec, kv_cache)


class _IluvatarHndMambaPageViewEdit:
    """Adapt upstream's byte-opaque Mamba page view to the HND ABI.

    Upstream emits ``[NB, 2, BS, NH, HS]`` because its page view follows the
    NHD transfer ABI. LMCache MP currently carries one engine KV format for
    all kernel groups, so an Iluvatar HND registration interprets every group
    as ``[NB, 2, NH, BS, HS]``. Re-view the already flattened, byte-opaque
    Mamba page as ``[NB, 2, 1, BS, HS]`` without moving storage.

    Does not wrap MLA or ``mamba-unified-view``; only the list-form
    ``mamba-page-view`` anchor.
    """

    name = _HND_MAMBA_EDIT_NAME

    def __init__(self, upstream_edit: Any) -> None:
        self._upstream_edit = upstream_edit

    def matches(self, spec: Any, kv_cache: Any) -> bool:
        return _kv_layout_hint() == "HND" and self._upstream_edit.matches(
            spec, kv_cache
        )

    def apply(
        self, spec: Any, kv_cache: Any, layout_hints: Mapping[str, Any] | None = None
    ) -> torch.Tensor:
        page_view = self._upstream_edit.apply(spec, kv_cache, layout_hints)
        if not isinstance(page_view, torch.Tensor) or page_view.ndim != 5:
            raise ValueError(
                "upstream Mamba page edit must return a five-dimensional "
                f"tensor, got {type(page_view).__name__}"
            )
        if int(page_view.shape[1]) != 2 or int(page_view.shape[3]) != 1:
            raise ValueError(
                "expected upstream Mamba NHD page view "
                "(num_blocks, 2, block_size, 1, head_size), got "
                f"{tuple(int(x) for x in page_view.shape)}"
            )
        target_shape = (
            int(page_view.shape[0]),
            2,
            1,
            int(page_view.shape[2]),
            int(page_view.shape[4]),
        )
        out = page_view.view(*target_shape)
        if out.data_ptr() != page_view.data_ptr():
            raise ValueError("HND Mamba page view must alias upstream storage")
        logger.info(
            "HND Mamba logical-page view: shape=%s -> %s stride=%s",
            tuple(int(x) for x in page_view.shape),
            target_shape,
            tuple(int(x) for x in out.stride()),
        )
        return out


class _NhdSubpagedMatchGuard:
    """Registry wrapper that prevents HND tensors matching the NHD edit.

    Keep the compatibility change local to the installed ``_EDITS`` tuple;
    never mutate the upstream edit class or its ``matches`` method globally.
    """

    name = _UPSTREAM_NHD_EDIT_NAME
    __lmcache_iluvatar_nhd_match_guard__ = True

    def __init__(self, upstream_edit: Any) -> None:
        self._upstream_edit = upstream_edit

    def matches(self, spec: Any, kv_cache: Any) -> bool:
        return (
            isinstance(kv_cache, torch.Tensor)
            and kv_cache.ndim == 5
            and int(kv_cache.shape[1]) == 2
            and self._upstream_edit.matches(spec, kv_cache)
        )

    def apply(
        self, spec: Any, kv_cache: Any, layout_hints: Mapping[str, Any] | None = None
    ) -> torch.Tensor:
        return self._upstream_edit.apply(spec, kv_cache, layout_hints)


def install_iluvatar_hnd_kv_cache_group_edits(edits_mod: Any) -> tuple[str, str]:
    """Install HND edits and an NHD match guard into upstream ``_EDITS``.

    Must be called from ``activate_patches()`` under ``_PATCH_LOCK``, not from
    request-time ``apply_kv_cache_group_edits``.

    Args:
        edits_mod: ``lmcache.integration.vllm.kv_cache_group_edits`` module.

    Returns:
        ``(status, detail)`` suitable for ``PatchResult``:
        ``patched``, ``already_patched``, or ``missing``.
    """

    edits = getattr(edits_mod, "_EDITS", None)
    if edits is None:
        return (
            "missing",
            "upstream kv_cache_group_edits._EDITS registry is not present",
        )

    installed_names = {getattr(edit, "name", None) for edit in edits}
    missing_anchors = {
        _UPSTREAM_NHD_EDIT_NAME,
        _UPSTREAM_MAMBA_EDIT_NAME,
    } - installed_names
    if missing_anchors:
        return (
            "missing",
            "upstream hybrid edit anchors are not present: "
            + ", ".join(sorted(missing_anchors)),
        )

    if getattr(edits_mod, _EDITS_INSTALLED, False):
        return ("already_patched", "HND hybrid edits already installed")

    if {_HND_EDIT_NAME, _HND_MAMBA_EDIT_NAME}.issubset(installed_names):
        setattr(edits_mod, _EDITS_INSTALLED, True)
        return ("already_patched", "HND hybrid edits already present in _EDITS")

    hnd_edit = _IluvatarHndSubpagedAttentionViewEdit()
    new_edits: list[Any] = []
    inserted = False
    for edit in edits:
        if getattr(edit, "name", None) == _UPSTREAM_MAMBA_EDIT_NAME:
            new_edits.append(_IluvatarHndMambaPageViewEdit(edit))
        if not inserted and getattr(edit, "name", None) == _UPSTREAM_NHD_EDIT_NAME:
            new_edits.append(hnd_edit)
            edit = _NhdSubpagedMatchGuard(edit)
            inserted = True
        new_edits.append(edit)

    edits_mod._EDITS = tuple(new_edits)
    setattr(edits_mod, _EDITS_INSTALLED, True)
    detail = (
        f"inserted {_HND_EDIT_NAME} before {_UPSTREAM_NHD_EDIT_NAME} and "
        f"{_HND_MAMBA_EDIT_NAME} before {_UPSTREAM_MAMBA_EDIT_NAME}; "
        "wrapped upstream NHD matcher"
    )
    logger.info("Installed Iluvatar HND kv_cache_group_edits: %s", detail)
    return ("patched", detail)


def build_iluvatar_apply_kv_cache_group_edits(
    upstream_fn: Callable[..., dict[str, Any]] | None,
) -> Callable[..., dict[str, Any]]:
    """Wrap upstream ``apply_kv_cache_group_edits`` (thin, idempotent marker).

    HND edit installation is performed separately by
    ``install_iluvatar_hnd_kv_cache_group_edits`` during ``activate_patches``.
    This wrapper resolves the registration layout once, scopes it across the
    upstream registry dispatch, and otherwise delegates unchanged so Mamba /
    validation / future edits keep running.

    Args:
        upstream_fn: Upstream ``apply_kv_cache_group_edits``.

    Returns:
        Patched callable with the same signature as ``upstream_fn``.

    Raises:
        RuntimeError: If ``upstream_fn`` is missing, the hybrid layout cannot
            be resolved, or the environment override conflicts with it.
    """

    if upstream_fn is None:
        raise RuntimeError("apply_kv_cache_group_edits is not present")
    if getattr(upstream_fn, _MARKER, False):
        return upstream_fn

    def apply_kv_cache_group_edits(
        kv_cache_config: Any,
        kv_caches: Mapping[str, Any],
        layout_hints: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        if kv_cache_config is None or not getattr(
            kv_cache_config, "has_mamba_layers", False
        ):
            return upstream_fn(kv_cache_config, kv_caches, layout_hints)

        with use_iluvatar_kv_cache_layout(layout_hints) as resolution:
            if resolution.layout is None:
                raise RuntimeError(
                    "Hybrid KV cache registration requires a resolved HND/NHD "
                    "layout from vLLM layout_hints or VLLM_KV_CACHE_LAYOUT; "
                    "refusing shape/stride-only layout inference."
                )
            _log_hybrid_group_layouts(
                kv_cache_config,
                kv_caches,
                resolved_layout=resolution.layout,
                source=resolution.source,
                env_override=resolution.env_override,
            )
            return upstream_fn(kv_cache_config, kv_caches, layout_hints)

    setattr(apply_kv_cache_group_edits, _MARKER, True)
    return apply_kv_cache_group_edits


__all__ = [
    "SupportsSubpagedSpec",
    "apply_hnd_subpaged_attention_view",
    "build_iluvatar_apply_kv_cache_group_edits",
    "install_iluvatar_hnd_kv_cache_group_edits",
    "is_iluvatar_hnd_flash_attn",
    "needs_hnd_subpaged_edit",
]
