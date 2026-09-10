# Copyright (c) 2026, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Tests for Iluvatar HND hybrid KV cache group-edit compatibility."""

from __future__ import annotations

import importlib
import sys
from types import ModuleType, SimpleNamespace

import pytest

from fake_lmcache import install_fake_lmcache


class _FakeShape(tuple):
    def numel(self) -> int:
        total = 1
        for dim in self:
            total *= int(dim)
        return total


class _FakeTensor:
    """Minimal tensor stand-in for hosts without a real torch install."""

    def __init__(
        self,
        shape,
        *,
        element_size: int = 4,
        storage_id: int | None = None,
        contiguous: bool = True,
        stride: tuple[int, ...] | None = None,
        data: list | None = None,
    ):
        self.shape = _FakeShape(int(dim) for dim in shape)
        self._element_size = element_size
        self._storage_id = id(self) if storage_id is None else storage_id
        self._contiguous = contiguous
        self._stride = stride
        self._data = data

    @property
    def ndim(self) -> int:
        return len(self.shape)

    def element_size(self) -> int:
        return self._element_size

    def is_contiguous(self) -> bool:
        return self._contiguous

    def data_ptr(self) -> int:
        return self._storage_id

    def stride(self) -> tuple[int, ...]:
        if self._stride is not None:
            return self._stride
        if self._contiguous:
            strides = [1] * len(self.shape)
            acc = 1
            for i in range(len(self.shape) - 1, -1, -1):
                strides[i] = acc
                acc *= int(self.shape[i])
            return tuple(strides)
        return tuple(range(len(self.shape), 0, -1))

    def permute(self, *dims) -> "_FakeTensor":
        shape = tuple(self.shape[i] for i in dims)
        stride = tuple(self.stride()[i] for i in dims)
        expected = []
        acc = 1
        for dim in reversed(shape):
            expected.append(acc)
            acc *= int(dim)
        contiguous = stride == tuple(reversed(expected))
        return _FakeTensor(
            shape,
            element_size=self._element_size,
            storage_id=self._storage_id,
            contiguous=contiguous,
            stride=stride,
            data=self._data,
        )

    def view(self, *shape) -> "_FakeTensor":
        if not self._contiguous:
            raise RuntimeError("view is not supported for non-contiguous fake tensors")
        new_shape = (
            shape[0] if len(shape) == 1 and isinstance(shape[0], tuple) else shape
        )
        if _FakeShape(new_shape).numel() != self.shape.numel():
            raise RuntimeError(
                f"shape {new_shape} is invalid for input of size {self.shape.numel()}"
            )
        return _FakeTensor(
            new_shape,
            element_size=self._element_size,
            storage_id=self._storage_id,
            contiguous=True,
            data=self._data,
        )

    def reshape(self, *shape) -> "_FakeTensor":
        new_shape = (
            shape[0] if len(shape) == 1 and isinstance(shape[0], tuple) else shape
        )
        if self._contiguous:
            return self.view(*new_shape)
        return _FakeTensor(
            new_shape,
            element_size=self._element_size,
            storage_id=None,
            contiguous=True,
            data=list(self._data) if self._data is not None else None,
        )

    def contiguous(self) -> "_FakeTensor":
        if self._contiguous:
            return self
        return _FakeTensor(
            self.shape,
            element_size=self._element_size,
            storage_id=None,
            contiguous=True,
            data=list(self._data) if self._data is not None else None,
        )


def _install_torch_for_hnd_tests(monkeypatch):
    """Ensure ``torch.Tensor`` / ``torch.zeros`` exist for HND shape tests."""

    try:
        import torch

        if hasattr(torch, "zeros") and hasattr(torch, "Tensor"):
            return torch
    except ImportError:
        pass

    torch = ModuleType("torch")
    torch.Tensor = _FakeTensor  # type: ignore[attr-defined]
    torch.float32 = "float32"  # type: ignore[attr-defined]
    torch.int8 = object()  # type: ignore[attr-defined]

    def zeros(*shape, dtype=None):
        del dtype
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        return _FakeTensor(shape)

    torch.zeros = zeros  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", torch)
    return torch


def _hnd_kernel_paged_tensor(
    torch,
    *,
    num_logical_blocks: int = 2,
    ratio: int = 49,
    num_heads: int = 2,
    kernel_block_size: int = 16,
    head_size: int = 256,
):
    num_kernel_pages = num_logical_blocks * ratio
    physical = torch.zeros(
        num_kernel_pages,
        2,
        num_heads,
        kernel_block_size,
        head_size,
        dtype=torch.float32,
    )
    return physical.permute(1, 0, 2, 3, 4)


def _hybrid_config(
    block_size: int,
    page_size_bytes: int,
    layer_names: list[str] | None = None,
    *,
    include_mamba: bool = False,
):
    groups = []
    attn_spec = SimpleNamespace(
        block_size=block_size,
        page_size_bytes=page_size_bytes,
        compress_ratio=1,
        tq_slot_size=0,
    )
    names = layer_names or ["layers.0"]
    groups.append(SimpleNamespace(kv_cache_spec=attn_spec, layer_names=list(names)))
    if include_mamba:
        mamba_spec = SimpleNamespace(
            block_size=block_size,
            page_size_bytes=page_size_bytes,
            compress_ratio=1,
            tq_slot_size=0,
        )
        groups.append(
            SimpleNamespace(kv_cache_spec=mamba_spec, layer_names=["layers.0.mamba"])
        )
    return SimpleNamespace(has_mamba_layers=True, kv_cache_groups=groups)


def _page_size_bytes(tensor, ratio: int) -> int:
    return (
        2
        * int(tensor.shape[2])
        * int(tensor.shape[3])
        * int(tensor.shape[4])
        * ratio
        * tensor.element_size()
    )


def _reset_to_fake_upstream_edits(edits_mod):
    """Undo import-time plugin installation for installer unit tests."""

    nhd = next(
        edit for edit in edits_mod._EDITS if edit.name == "subpaged-attention-view"
    )
    nhd = getattr(nhd, "_upstream_edit", nhd)
    mamba = next(edit for edit in edits_mod._EDITS if edit.name == "mamba-page-view")
    edits_mod._EDITS = (nhd, mamba)
    if hasattr(edits_mod, "__lmcache_iluvatar_hnd_edits_installed__"):
        delattr(edits_mod, "__lmcache_iluvatar_hnd_edits_installed__")


def test_apply_hnd_subpaged_attention_view_recovers_block_first_storage(monkeypatch):
    torch = _install_torch_for_hnd_tests(monkeypatch)
    install_fake_lmcache(monkeypatch)
    from lmcache_iluvatar.integration.vllm.kv_cache_group_edits import (
        apply_hnd_subpaged_attention_view,
    )

    ratio = 49
    logical_bs = 784
    tensor = _hnd_kernel_paged_tensor(torch, num_logical_blocks=4, ratio=ratio)
    assert tuple(tensor.stride()) == (8192, 16384, 4096, 256, 1)
    spec = SimpleNamespace(
        block_size=logical_bs,
        page_size_bytes=_page_size_bytes(tensor, ratio),
    )

    out = apply_hnd_subpaged_attention_view(spec, tensor)

    # HND block-first ABI is [NB, 2, NH, BS, HS].  Keep these semantic
    # assertions independent of the implementation's target_shape ordering:
    # the MP native kernel consumes NH as blockDim.y and loops over BS.
    assert tuple(out.shape) == (4, 2, 1, logical_bs, 512)
    assert out.shape[2] == 1
    assert out.shape[3] == logical_bs
    assert out.is_contiguous()
    assert out.data_ptr() == tensor.data_ptr()


def test_apply_hnd_subpaged_attention_view_rejects_non_contiguous_copy(monkeypatch):
    """Same-storage contract: refuse registration that would copy KV pages."""

    torch = _install_torch_for_hnd_tests(monkeypatch)
    install_fake_lmcache(monkeypatch)
    from lmcache_iluvatar.integration.vllm.kv_cache_group_edits import (
        apply_hnd_subpaged_attention_view,
    )

    ratio = 49
    logical_bs = 784
    if hasattr(torch, "Tensor") and torch.Tensor is not _FakeTensor:
        base = torch.zeros(2, 2, 4 * ratio, 16, 256, dtype=torch.float32)
        tensor = base.permute(0, 2, 1, 3, 4)
        assert tuple(tensor.shape) == (2, 4 * ratio, 2, 16, 256)
        assert not tensor.is_contiguous()
    else:
        tensor = _FakeTensor(
            (2, 4 * ratio, 2, 16, 256),
            contiguous=False,
        )

    spec = SimpleNamespace(
        block_size=logical_bs,
        page_size_bytes=_page_size_bytes(tensor, ratio),
    )
    with pytest.raises(ValueError, match="zero-copy transpose"):
        apply_hnd_subpaged_attention_view(spec, tensor)


def test_install_hnd_edit_then_upstream_dispatch(monkeypatch):
    torch = _install_torch_for_hnd_tests(monkeypatch)
    modules = install_fake_lmcache(monkeypatch)
    from lmcache_iluvatar.integration.vllm.kv_cache_group_edits import (
        build_iluvatar_apply_kv_cache_group_edits,
        install_iluvatar_hnd_kv_cache_group_edits,
    )

    edits_mod = modules["lmcache.integration.vllm.kv_cache_group_edits"]
    status, detail = install_iluvatar_hnd_kv_cache_group_edits(edits_mod)
    # Importing lmcache_iluvatar.v1.* may already have run activate_patches().
    assert status in {"patched", "already_patched"}
    assert any(
        getattr(edit, "name", None) == "iluvatar-hnd-subpaged-attention-view"
        for edit in edits_mod._EDITS
    )

    ratio = 49
    logical_bs = 784
    tensor = _hnd_kernel_paged_tensor(torch, num_logical_blocks=2, ratio=ratio)
    config = _hybrid_config(logical_bs, _page_size_bytes(tensor, ratio))
    caches = {"layers.0": tensor}

    nhd_edit = next(
        edit
        for edit in edits_mod._EDITS
        if getattr(edit, "name", None) == "subpaged-attention-view"
    )
    with pytest.raises(ValueError, match="expected a \\(num_blocks, 2"):
        # Upstream NHD apply still rejects raw HND when forced.
        nhd_edit.apply(
            config.kv_cache_groups[0].kv_cache_spec,
            tensor,
            {"kv_layout": "HND"},
        )

    patched = build_iluvatar_apply_kv_cache_group_edits(
        edits_mod.apply_kv_cache_group_edits
    )
    out = patched(config, caches, {"kv_layout": "HND"})
    assert tuple(out["layers.0"].shape) == (2, 2, 1, logical_bs, 512)
    assert out["layers.0"].data_ptr() == tensor.data_ptr()


def test_runtime_patch_installs_edits_under_activate(monkeypatch):
    torch = _install_torch_for_hnd_tests(monkeypatch)
    modules = install_fake_lmcache(monkeypatch)
    original = modules[
        "lmcache.integration.vllm.kv_cache_group_edits"
    ].apply_kv_cache_group_edits

    plugin = importlib.import_module("lmcache_iluvatar")
    patched = modules[
        "lmcache.integration.vllm.kv_cache_group_edits"
    ].apply_kv_cache_group_edits
    cached = modules[
        "lmcache.integration.vllm.lmcache_mp_connector"
    ].apply_kv_cache_group_edits
    edits_mod = modules["lmcache.integration.vllm.kv_cache_group_edits"]

    edits_result = next(
        item
        for item in plugin.get_patch_state().results
        if item.target == "lmcache.integration.vllm.kv_cache_group_edits._EDITS"
    )
    apply_result = next(
        item
        for item in plugin.get_patch_state().results
        if item.target
        == "lmcache.integration.vllm.kv_cache_group_edits.apply_kv_cache_group_edits"
    )
    assert edits_result.status == "patched"
    assert apply_result.status == "patched"
    assert patched is not original
    assert cached is patched
    assert getattr(patched, "__lmcache_iluvatar_kv_cache_group_edits__", False)
    assert any(
        getattr(edit, "name", None) == "iluvatar-hnd-subpaged-attention-view"
        for edit in edits_mod._EDITS
    )

    ratio = 49
    logical_bs = 784
    tensor = _hnd_kernel_paged_tensor(
        torch,
        # Match the Qwen3.5 smoke failure shape: (2, 15582, 2, 16, 256)
        num_logical_blocks=318,
        ratio=ratio,
        num_heads=2,
        kernel_block_size=16,
        head_size=256,
    )
    assert tuple(tensor.shape) == (2, 15582, 2, 16, 256)
    assert tuple(tensor.stride()) == (8192, 16384, 4096, 256, 1)
    page_size_bytes = _page_size_bytes(tensor, ratio)
    config = _hybrid_config(logical_bs, page_size_bytes)
    out = patched(config, {"layers.0": tensor}, {"kv_layout": "HND"})
    elems_per_page = page_size_bytes // tensor.element_size()
    head_size = elems_per_page // (2 * logical_bs)
    assert tuple(out["layers.0"].shape) == (318, 2, 1, logical_bs, head_size)
    assert out["layers.0"].shape[2] == 1
    assert (
        out["layers.0"].shape[3] == config.kv_cache_groups[0].kv_cache_spec.block_size
    )
    assert out["layers.0"].data_ptr() == tensor.data_ptr()


def test_hybrid_still_runs_mamba_edit(monkeypatch):
    torch = _install_torch_for_hnd_tests(monkeypatch)
    modules = install_fake_lmcache(monkeypatch)
    importlib.import_module("lmcache_iluvatar")
    patched = modules[
        "lmcache.integration.vllm.kv_cache_group_edits"
    ].apply_kv_cache_group_edits

    ratio = 49
    logical_bs = 784
    tensor = _hnd_kernel_paged_tensor(torch, num_logical_blocks=2, ratio=ratio)
    mamba_state = [object(), object()]
    config = _hybrid_config(
        logical_bs,
        _page_size_bytes(tensor, ratio),
        include_mamba=True,
    )
    out = patched(
        config,
        {"layers.0": tensor, "layers.0.mamba": mamba_state},
        {"kv_layout": "NHD"},
    )
    assert out["layers.0"] is tensor
    assert out["layers.0.mamba"] is mamba_state
    mamba_edit = next(
        edit
        for edit in modules["lmcache.integration.vllm.kv_cache_group_edits"]._EDITS
        if getattr(edit, "name", None) == "mamba-page-view"
    )
    assert type(mamba_edit).calls >= 1


def test_hybrid_consumes_resolved_hnd_without_environment(monkeypatch, caplog):
    """Attention and Mamba edits share vLLM's registration layout contract."""

    torch = _install_torch_for_hnd_tests(monkeypatch)
    modules = install_fake_lmcache(monkeypatch)
    monkeypatch.delenv("VLLM_KV_CACHE_LAYOUT", raising=False)
    importlib.import_module("lmcache_iluvatar")
    edits = modules["lmcache.integration.vllm.kv_cache_group_edits"]._EDITS
    hnd_mamba_edit = next(
        edit
        for edit in edits
        if getattr(edit, "name", None) == "iluvatar-hnd-mamba-page-view"
    )

    ratio = 49
    logical_bs = 784
    tensor = _hnd_kernel_paged_tensor(torch, num_logical_blocks=2, ratio=ratio)
    mamba_page = torch.zeros(2, 2, logical_bs, 1, 512)
    monkeypatch.setattr(
        hnd_mamba_edit._upstream_edit,
        "apply",
        lambda spec, kv_cache, layout_hints=None: mamba_page,
    )
    config = _hybrid_config(
        logical_bs,
        _page_size_bytes(tensor, ratio),
        include_mamba=True,
    )
    patched = modules[
        "lmcache.integration.vllm.kv_cache_group_edits"
    ].apply_kv_cache_group_edits

    with caplog.at_level(
        "DEBUG",
        logger="lmcache_iluvatar.integration.vllm.kv_cache_group_edits",
    ):
        out = patched(
            config,
            {"layers.0": tensor, "layers.0.mamba": [object(), object()]},
            {"kv_layout": "HND"},
        )

    assert tuple(out["layers.0"].shape) == (2, 2, 1, logical_bs, 512)
    assert tuple(out["layers.0.mamba"].shape) == (2, 2, 1, logical_bs, 512)
    assert "source=layout_hints[\"kv_layout\"] resolved_layout=HND" in caplog.text
    assert "group=0 layer=layers.0" in caplog.text
    assert "group=1 layer=layers.0.mamba" in caplog.text
    contract_record = next(
        record
        for record in caplog.records
        if "Hybrid KV cache layout contract" in record.message
    )
    group_records = [
        record
        for record in caplog.records
        if "Hybrid KV cache group=" in record.message
    ]
    assert contract_record.levelname == "INFO"
    assert group_records
    assert all(record.levelname == "DEBUG" for record in group_records)


@pytest.mark.parametrize(
    ("resolved_layout", "expected"),
    [("HND", True), ("NHD", False)],
)
def test_resolved_layout_disambiguates_nb_two_shape(
    monkeypatch,
    resolved_layout,
    expected,
):
    """Resolved metadata wins for the ambiguous (2, 2, ...) tensor shape."""

    torch = _install_torch_for_hnd_tests(monkeypatch)
    install_fake_lmcache(monkeypatch)
    monkeypatch.delenv("VLLM_KV_CACHE_LAYOUT", raising=False)
    from lmcache_iluvatar.integration.vllm.kv_cache_group_edits import (
        is_iluvatar_hnd_flash_attn,
    )
    from lmcache_iluvatar.integration.vllm.kv_cache_layout import (
        use_iluvatar_kv_cache_layout,
    )

    tensor = _hnd_kernel_paged_tensor(
        torch,
        num_logical_blocks=1,
        ratio=2,
    )
    assert tuple(tensor.shape[:2]) == (2, 2)
    spec = SimpleNamespace(block_size=16)

    with use_iluvatar_kv_cache_layout({"kv_layout": resolved_layout}):
        assert is_iluvatar_hnd_flash_attn(tensor, spec) is expected


@pytest.mark.parametrize(
    ("env_layout", "resolved_layout"),
    [("NHD", "HND"), ("HND", "NHD")],
)
def test_hybrid_registration_fails_before_edit_on_layout_conflict(
    monkeypatch,
    env_layout,
    resolved_layout,
):
    torch = _install_torch_for_hnd_tests(monkeypatch)
    modules = install_fake_lmcache(monkeypatch)
    monkeypatch.setenv("VLLM_KV_CACHE_LAYOUT", env_layout)
    importlib.import_module("lmcache_iluvatar")
    tensor = _hnd_kernel_paged_tensor(torch)
    config = _hybrid_config(784, _page_size_bytes(tensor, 49))
    patched = modules[
        "lmcache.integration.vllm.kv_cache_group_edits"
    ].apply_kv_cache_group_edits

    with pytest.raises(RuntimeError, match="KV cache layout conflict"):
        patched(config, {"layers.0": tensor}, {"kv_layout": resolved_layout})


def test_hybrid_registration_rejects_missing_resolved_layout(monkeypatch):
    torch = _install_torch_for_hnd_tests(monkeypatch)
    modules = install_fake_lmcache(monkeypatch)
    monkeypatch.delenv("VLLM_KV_CACHE_LAYOUT", raising=False)
    importlib.import_module("lmcache_iluvatar")
    tensor = _hnd_kernel_paged_tensor(torch)
    config = _hybrid_config(784, _page_size_bytes(tensor, 49))
    patched = modules[
        "lmcache.integration.vllm.kv_cache_group_edits"
    ].apply_kv_cache_group_edits

    with pytest.raises(RuntimeError, match="requires a resolved HND/NHD layout"):
        patched(config, {"layers.0": tensor})


def test_hybrid_registration_keeps_explicit_hnd_fallback(monkeypatch, caplog):
    torch = _install_torch_for_hnd_tests(monkeypatch)
    modules = install_fake_lmcache(monkeypatch)
    monkeypatch.setenv("VLLM_KV_CACHE_LAYOUT", "HND")
    importlib.import_module("lmcache_iluvatar")
    tensor = _hnd_kernel_paged_tensor(torch)
    config = _hybrid_config(784, _page_size_bytes(tensor, 49))
    patched = modules[
        "lmcache.integration.vllm.kv_cache_group_edits"
    ].apply_kv_cache_group_edits

    with caplog.at_level("INFO"):
        out = patched(config, {"layers.0": tensor})

    assert tuple(out["layers.0"].shape) == (2, 2, 1, 784, 512)
    assert "source=VLLM_KV_CACHE_LAYOUT resolved_layout=HND" in caplog.text
    assert "Hybrid KV cache group=" not in caplog.text


def test_hnd_mamba_page_view_uses_hnd_dimension_order(monkeypatch):
    """Mamba byte pages must follow the global HND ABI used by MP transfer."""

    torch = _install_torch_for_hnd_tests(monkeypatch)
    install_fake_lmcache(monkeypatch)
    monkeypatch.setenv("VLLM_KV_CACHE_LAYOUT", "HND")
    from lmcache_iluvatar.integration.vllm.kv_cache_group_edits import (
        _IluvatarHndMambaPageViewEdit,
    )

    num_blocks = 318
    logical_bs = 784
    head_size = 512
    nhd_page_view = torch.zeros(num_blocks, 2, logical_bs, 1, head_size)

    class _UpstreamMambaEdit:
        def matches(self, spec, kv_cache):
            return isinstance(kv_cache, list)

        def apply(self, spec, kv_cache, layout_hints=None):
            return nhd_page_view

    edit = _IluvatarHndMambaPageViewEdit(_UpstreamMambaEdit())
    assert edit.matches(SimpleNamespace(), [object(), object()])

    out = edit.apply(
        SimpleNamespace(),
        [object(), object()],
        {"kv_layout": "HND"},
    )

    assert tuple(out.shape) == (num_blocks, 2, 1, logical_bs, head_size)
    assert out.data_ptr() == nhd_page_view.data_ptr()


def test_install_places_hnd_mamba_edit_before_upstream(monkeypatch):
    _install_torch_for_hnd_tests(monkeypatch)
    modules = install_fake_lmcache(monkeypatch)
    from lmcache_iluvatar.integration.vllm.kv_cache_group_edits import (
        install_iluvatar_hnd_kv_cache_group_edits,
    )

    edits_mod = modules["lmcache.integration.vllm.kv_cache_group_edits"]
    install_iluvatar_hnd_kv_cache_group_edits(edits_mod)
    names = [getattr(edit, "name", None) for edit in edits_mod._EDITS]

    assert names.index("iluvatar-hnd-mamba-page-view") < names.index("mamba-page-view")


def test_unified_mamba_hnd_stays_on_upstream_edit(monkeypatch):
    """The vLLM 0.26 unified-Mamba path must receive hints without a wrapper."""

    torch = _install_torch_for_hnd_tests(monkeypatch)
    modules = install_fake_lmcache(monkeypatch)
    monkeypatch.setenv("VLLM_KV_CACHE_LAYOUT", "HND")
    from lmcache_iluvatar.integration.vllm.kv_cache_group_edits import (
        build_iluvatar_apply_kv_cache_group_edits,
        install_iluvatar_hnd_kv_cache_group_edits,
    )

    edits_mod = modules["lmcache.integration.vllm.kv_cache_group_edits"]
    _reset_to_fake_upstream_edits(edits_mod)
    unified_cache = torch.zeros(2, 1, 1, 64)
    unified_output = torch.zeros(2, 1, 16, 4)

    class _UpstreamMambaUnifiedEdit:
        name = "mamba-unified-view"

        def __init__(self):
            self.layout_hints = None

        def matches(self, spec, kv_cache):
            return kv_cache is unified_cache

        def apply(self, spec, kv_cache, layout_hints):
            self.layout_hints = layout_hints
            return unified_output

    unified_edit = _UpstreamMambaUnifiedEdit()
    edits_mod._EDITS = (unified_edit, *edits_mod._EDITS)
    status, _ = install_iluvatar_hnd_kv_cache_group_edits(edits_mod)
    patched = build_iluvatar_apply_kv_cache_group_edits(
        edits_mod.apply_kv_cache_group_edits
    )
    config = SimpleNamespace(
        has_mamba_layers=True,
        kv_cache_groups=[
            SimpleNamespace(
                kv_cache_spec=SimpleNamespace(block_size=16),
                layer_names=["layers.0.mamba"],
            )
        ],
    )
    hints = {"kv_layout": "HND"}

    out = patched(config, {"layers.0.mamba": unified_cache}, hints)

    assert status == "patched"
    assert (
        next(edit for edit in edits_mod._EDITS if edit.name == "mamba-unified-view")
        is unified_edit
    )
    assert unified_edit.layout_hints is hints
    assert out["layers.0.mamba"] is unified_output


@pytest.mark.parametrize("missing_name", ["mamba-page-view", "subpaged-attention-view"])
def test_install_reports_missing_upstream_anchor(monkeypatch, missing_name):
    _install_torch_for_hnd_tests(monkeypatch)
    modules = install_fake_lmcache(monkeypatch)
    from lmcache_iluvatar.integration.vllm.kv_cache_group_edits import (
        install_iluvatar_hnd_kv_cache_group_edits,
    )

    edits_mod = modules["lmcache.integration.vllm.kv_cache_group_edits"]
    _reset_to_fake_upstream_edits(edits_mod)
    edits_mod._EDITS = tuple(
        edit for edit in edits_mod._EDITS if getattr(edit, "name", None) != missing_name
    )
    status, detail = install_iluvatar_hnd_kv_cache_group_edits(edits_mod)

    assert status == "missing"
    assert missing_name in detail
    assert not getattr(edits_mod, "__lmcache_iluvatar_hnd_edits_installed__", False)


def test_install_is_idempotent_and_wraps_only_registry_instance(monkeypatch):
    torch = _install_torch_for_hnd_tests(monkeypatch)
    modules = install_fake_lmcache(monkeypatch)
    from lmcache_iluvatar.integration.vllm.kv_cache_group_edits import (
        install_iluvatar_hnd_kv_cache_group_edits,
    )

    edits_mod = modules["lmcache.integration.vllm.kv_cache_group_edits"]
    _reset_to_fake_upstream_edits(edits_mod)
    original_nhd = next(
        edit for edit in edits_mod._EDITS if edit.name == "subpaged-attention-view"
    )
    original_matches = type(original_nhd).matches

    first_status, _ = install_iluvatar_hnd_kv_cache_group_edits(edits_mod)
    installed_once = edits_mod._EDITS
    second_status, _ = install_iluvatar_hnd_kv_cache_group_edits(edits_mod)
    guarded_nhd = next(
        edit for edit in edits_mod._EDITS if edit.name == "subpaged-attention-view"
    )

    assert first_status == "patched"
    assert second_status == "already_patched"
    assert edits_mod._EDITS is installed_once
    assert type(original_nhd).matches is original_matches
    assert getattr(guarded_nhd, "__lmcache_iluvatar_nhd_match_guard__", False)
    hnd = _hnd_kernel_paged_tensor(torch)
    spec = SimpleNamespace(block_size=784, page_size_bytes=_page_size_bytes(hnd, 49))
    assert guarded_nhd.matches(spec, hnd) is False


def test_install_revalidates_required_anchors_after_marker_is_set(monkeypatch):
    _install_torch_for_hnd_tests(monkeypatch)
    modules = install_fake_lmcache(monkeypatch)
    from lmcache_iluvatar.integration.vllm.kv_cache_group_edits import (
        install_iluvatar_hnd_kv_cache_group_edits,
    )

    edits_mod = modules["lmcache.integration.vllm.kv_cache_group_edits"]
    _reset_to_fake_upstream_edits(edits_mod)
    first_status, _ = install_iluvatar_hnd_kv_cache_group_edits(edits_mod)
    edits_mod._EDITS = tuple(
        edit for edit in edits_mod._EDITS if edit.name != "mamba-page-view"
    )

    second_status, detail = install_iluvatar_hnd_kv_cache_group_edits(edits_mod)

    assert first_status == "patched"
    assert second_status == "missing"
    assert "mamba-page-view" in detail


def test_non_hybrid_delegates_to_upstream(monkeypatch):
    torch = _install_torch_for_hnd_tests(monkeypatch)
    modules = install_fake_lmcache(monkeypatch)
    importlib.import_module("lmcache_iluvatar")
    patched = modules[
        "lmcache.integration.vllm.kv_cache_group_edits"
    ].apply_kv_cache_group_edits

    tensor = _hnd_kernel_paged_tensor(torch)
    config = SimpleNamespace(has_mamba_layers=False, kv_cache_groups=[])
    out = patched(config, {"layers.0": tensor})
    assert out["layers.0"] is tensor


def test_byte_mapping_round_trip_on_real_torch(monkeypatch):
    """Element order of logical pages matches kernel-page packing (real torch)."""

    try:
        import torch
    except ImportError:
        pytest.skip("torch not installed")
    if not hasattr(torch, "arange"):
        pytest.skip("torch.arange unavailable")

    install_fake_lmcache(monkeypatch)
    from lmcache_iluvatar.integration.vllm.kv_cache_group_edits import (
        apply_hnd_subpaged_attention_view,
    )

    ratio = 4
    logical_bs = 64
    kernel_bs = 16
    num_heads = 2
    head_size = 8
    num_logical = 2
    num_kernel = num_logical * ratio
    numel = 2 * num_kernel * num_heads * kernel_bs * head_size
    flat = torch.arange(numel, dtype=torch.float32)
    physical = flat.view(num_kernel, 2, num_heads, kernel_bs, head_size)
    tensor = physical.permute(1, 0, 2, 3, 4)
    spec = SimpleNamespace(
        block_size=logical_bs,
        page_size_bytes=(
            2 * num_heads * kernel_bs * head_size * ratio * tensor.element_size()
        ),
    )
    out = apply_hnd_subpaged_attention_view(spec, tensor)
    assert out.data_ptr() == tensor.data_ptr()
    assert tuple(out.shape) == (num_logical, 2, 1, logical_bs, 2 * head_size)
    assert torch.equal(out.reshape(-1), physical.reshape(-1))
